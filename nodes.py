import ast
import json
import os
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageColor, ImageDraw, ImageFont


_REF_DET_RE = re.compile(
    r"<\|ref\|>(?P<ref>.*?)<\|/ref\|>\s*"
    r"<\|det\|>(?P<det>.*?)<\|/det\|>"
    r"(?P<text>.*?)(?=(?:<\|ref\|>)|\Z)",
    re.S,
)
_DET_ONLY_RE = re.compile(r"<\|det\|>(?P<det>.*?)<\|/det\|>", re.S)
_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


_COMMON_CJK_FONTS = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyh.ttf",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
]


def _as_text(value: Any) -> str:
    """Normalize a ComfyUI STRING/list output to one plain string."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(_as_text(v) for v in value)
    return str(value)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_flat_number_list(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) > 0 and all(_is_number(v) for v in value)


def _is_point_list(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return False
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return False
        if not (_is_number(point[0]) and _is_number(point[1])):
            return False
    return True


def _polygon_from_flat_numbers(nums: Sequence[float]) -> Dict[str, Any]:
    points = [(float(nums[i]), float(nums[i + 1])) for i in range(0, len(nums) - 1, 2)]
    return {"type": "polygon", "points": points}


def _detections_from_obj(obj: Any) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []

    if _is_flat_number_list(obj):
        nums = [float(v) for v in obj]
        if len(nums) == 4:
            detections.append({"type": "rect", "box": nums})
        elif len(nums) >= 6 and len(nums) % 2 == 0:
            detections.append(_polygon_from_flat_numbers(nums))
        return detections

    if _is_point_list(obj):
        detections.append({"type": "polygon", "points": [(float(p[0]), float(p[1])) for p in obj]})
        return detections

    if isinstance(obj, (list, tuple)):
        for item in obj:
            detections.extend(_detections_from_obj(item))

    return detections


def _parse_det(det_text: str) -> List[Dict[str, Any]]:
    det_text = (det_text or "").strip()
    if not det_text:
        return []

    try:
        obj = ast.literal_eval(det_text)
        detections = _detections_from_obj(obj)
        if detections:
            return detections
    except Exception:
        pass

    nums = [float(v) for v in _NUMBER_RE.findall(det_text)]
    detections: List[Dict[str, Any]] = []
    for i in range(0, len(nums) - 3, 4):
        detections.append({"type": "rect", "box": nums[i : i + 4]})
    return detections


def parse_deepseek_ocr(ocr_result: Any) -> List[Dict[str, Any]]:
    text = _as_text(ocr_result)
    annotations: List[Dict[str, Any]] = []

    for match in _REF_DET_RE.finditer(text):
        ref = match.group("ref").strip()
        body = match.group("text").strip()
        for det in _parse_det(match.group("det")):
            det = dict(det)
            det["ref"] = ref
            det["text"] = body
            annotations.append(det)

    if not annotations:
        for match in _DET_ONLY_RE.finditer(text):
            for det in _parse_det(match.group("det")):
                det = dict(det)
                det["ref"] = ""
                det["text"] = ""
                annotations.append(det)

    return annotations


def _parse_color(color: Any, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
    text = _as_text(color).strip()
    if not text:
        return default
    try:
        rgb = ImageColor.getrgb(text)
        if isinstance(rgb, tuple):
            return tuple(int(v) for v in rgb[:3])
    except Exception:
        pass

    nums = [int(float(v)) for v in _NUMBER_RE.findall(text)]
    if len(nums) >= 3:
        return tuple(max(0, min(255, v)) for v in nums[:3])
    return default


def _load_font(font_path: str, font_size: int) -> ImageFont.ImageFont:
    font_size = max(1, int(font_size))
    candidates: List[str] = []
    if font_path and font_path.strip():
        candidates.append(font_path.strip().strip("\""))
    candidates.extend(_COMMON_CJK_FONTS)

    for path in candidates:
        try:
            if os.path.exists(path):
                return ImageFont.truetype(path, font_size)
        except Exception:
            continue

    return ImageFont.load_default()


def _clean_label_text(text: str) -> str:
    lines = [line.strip() for line in (text or "").replace("\r", "\n").split("\n") if line.strip()]
    if not lines:
        return ""
    label = lines[0]
    label = re.sub(r"^#+\s*", "", label)
    label = re.sub(r"^[·•\-\*]\s*", "", label)
    return label[:80]


def _annotation_label(annotation: Dict[str, Any], label_mode: str) -> str:
    mode = (label_mode or "none").lower().replace("_", " ")
    ref = _as_text(annotation.get("ref", "")).strip()
    body = _clean_label_text(_as_text(annotation.get("text", "")))

    if mode == "ref":
        return ref
    if mode == "text":
        return body
    if mode in ("ref + text", "ref+text", "both"):
        if ref and body:
            return f"{ref}: {body}"
        return ref or body
    return ""


def _text_bbox(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, font: ImageFont.ImageFont):
    try:
        return draw.textbbox(xy, text, font=font)
    except Exception:
        try:
            w, h = draw.textsize(text, font=font)
        except Exception:
            w, h = (len(text) * 8, 12)
        x, y = xy
        return (x, y, x + w, y + h)


def _scale_value(value: float, length: int, coord_base: int) -> int:
    if coord_base and coord_base > 0:
        scaled = int(round(float(value) * float(length) / float(coord_base)))
    else:
        scaled = int(round(float(value)))
    return max(0, min(max(0, length - 1), scaled))


def _scale_crop_start(value: float, length: int, coord_base: int) -> int:
    if length <= 0:
        return 0
    if coord_base and coord_base > 0:
        scaled = int(round(float(value) * float(length) / float(coord_base)))
    else:
        scaled = int(round(float(value)))
    return max(0, min(length - 1, scaled))


def _scale_crop_end(value: float, length: int, coord_base: int) -> int:
    if length <= 0:
        return 0
    if coord_base and coord_base > 0:
        scaled = int(round(float(value) * float(length) / float(coord_base)))
    else:
        scaled = int(round(float(value)))
    return max(0, min(length, scaled))


def _scaled_rect(box: Sequence[float], width: int, height: int, coord_base: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    px1 = _scale_value(x1, width, coord_base)
    py1 = _scale_value(y1, height, coord_base)
    px2 = _scale_value(x2, width, coord_base)
    py2 = _scale_value(y2, height, coord_base)
    if px1 > px2:
        px1, px2 = px2, px1
    if py1 > py2:
        py1, py2 = py2, py1
    return px1, py1, px2, py2


def _scaled_crop_rect_from_values(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
    coord_base: int,
) -> Tuple[int, int, int, int]:
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    px1 = _scale_crop_start(x1, width, coord_base)
    py1 = _scale_crop_start(y1, height, coord_base)
    px2 = _scale_crop_end(x2, width, coord_base)
    py2 = _scale_crop_end(y2, height, coord_base)

    if px2 <= px1:
        px2 = min(width, px1 + 1)
    if py2 <= py1:
        py2 = min(height, py1 + 1)

    return px1, py1, px2, py2


def _scaled_crop_rect(box: Sequence[float], width: int, height: int, coord_base: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    return _scaled_crop_rect_from_values(x1, y1, x2, y2, width, height, coord_base)


def _expand_crop_rect(
    rect: Tuple[int, int, int, int],
    width: int,
    height: int,
    expand_px: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = rect
    expand_px = max(0, int(expand_px))
    if expand_px <= 0:
        return x1, y1, x2, y2

    x1 = max(0, x1 - expand_px)
    y1 = max(0, y1 - expand_px)
    x2 = min(width, x2 + expand_px)
    y2 = min(height, y2 + expand_px)

    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)

    return x1, y1, x2, y2


def _scaled_points(points: Iterable[Tuple[float, float]], width: int, height: int, coord_base: int) -> List[Tuple[int, int]]:
    return [(_scale_value(x, width, coord_base), _scale_value(y, height, coord_base)) for x, y in points]


def _annotation_crop_rect(
    annotation: Dict[str, Any],
    width: int,
    height: int,
    coord_base: int,
) -> Tuple[int, int, int, int]:
    if annotation.get("type") == "rect":
        box = annotation.get("box", [0, 0, 0, 0])
        return _scaled_crop_rect(box, width, height, coord_base)

    points = annotation.get("points", [])
    if not points:
        return (0, 0, 1, 1)
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return _scaled_crop_rect_from_values(min(xs), min(ys), max(xs), max(ys), width, height, coord_base)


def _draw_label(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    text_color: Tuple[int, int, int],
    background_color: Tuple[int, int, int],
    image_width: int,
    image_height: int,
) -> None:
    if not text:
        return

    x, y = xy
    pad_x, pad_y = 3, 2

    try:
        bbox = _text_bbox(draw, (0, 0), text, font)
    except Exception:
        safe = text.encode("latin-1", "ignore").decode("latin-1", "ignore")
        if not safe:
            return
        text = safe
        bbox = _text_bbox(draw, (0, 0), text, font)

    text_w = max(1, bbox[2] - bbox[0])
    text_h = max(1, bbox[3] - bbox[1])

    label_x = max(0, min(x, max(0, image_width - text_w - 2 * pad_x)))
    label_y = y - text_h - 2 * pad_y
    if label_y < 0:
        label_y = min(max(0, y + 1), max(0, image_height - text_h - 2 * pad_y))

    rect = [
        label_x,
        label_y,
        min(image_width - 1, label_x + text_w + 2 * pad_x),
        min(image_height - 1, label_y + text_h + 2 * pad_y),
    ]
    draw.rectangle(rect, fill=background_color)

    try:
        draw.text((label_x + pad_x, label_y + pad_y), text, fill=text_color, font=font)
    except Exception:
        safe = text.encode("latin-1", "ignore").decode("latin-1", "ignore")
        if safe:
            draw.text((label_x + pad_x, label_y + pad_y), safe, fill=text_color, font=font)


def _tensor_image_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    arr = image_tensor.detach().cpu().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L").convert("RGB")
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return Image.fromarray(arr[..., 0], mode="L").convert("RGB")
    if arr.ndim == 3 and arr.shape[-1] >= 4:
        return Image.fromarray(arr[..., :4], mode="RGBA").convert("RGB")
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        return Image.fromarray(arr[..., :3], mode="RGB")
    raise ValueError(f"Unsupported IMAGE tensor shape: {tuple(image_tensor.shape)}")


def _pil_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).to(device=device)


def _normalize_image_batch(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 3:
        return image.unsqueeze(0)
    if image.ndim == 4:
        return image
    raise ValueError(f"Unsupported IMAGE tensor shape: {tuple(image.shape)}")


def _empty_crop_batch(device: torch.device) -> torch.Tensor:
    return torch.zeros((1, 1, 1, 3), dtype=torch.float32, device=device)


def _crops_to_batch(
    crops: List[Image.Image],
    crop_infos: List[Dict[str, Any]],
    device: torch.device,
    padding_color: Tuple[int, int, int] = (0, 0, 0),
) -> torch.Tensor:
    if not crops:
        return _empty_crop_batch(device)

    max_w = max(max(1, crop.width) for crop in crops)
    max_h = max(max(1, crop.height) for crop in crops)

    tensors: List[torch.Tensor] = []
    for crop, info in zip(crops, crop_infos):
        info["canvas_size"] = [max_w, max_h]
        canvas = Image.new("RGB", (max_w, max_h), padding_color)
        canvas.paste(crop.convert("RGB"), (0, 0))
        tensors.append(_pil_to_tensor(canvas, device))

    return torch.stack(tensors, dim=0)


def _parse_crop_info(crop_info: Any) -> List[Dict[str, Any]]:
    if isinstance(crop_info, list):
        return [dict(item) for item in crop_info if isinstance(item, dict)]
    if isinstance(crop_info, dict):
        return [crop_info]

    text = _as_text(crop_info).strip()
    if not text:
        return []

    try:
        data = json.loads(text)
    except Exception:
        try:
            data = ast.literal_eval(text)
        except Exception:
            return []

    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [dict(item) for item in data if isinstance(item, dict)]
    return []


def _resize_image(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
    return image.resize(size, resampling)


def _strip_padding_from_crop(crop: Image.Image, info: Dict[str, Any]) -> Image.Image:
    crop_size = info.get("crop_size") or [crop.width, crop.height]
    canvas_size = info.get("canvas_size") or [crop.width, crop.height]

    try:
        original_crop_w = max(1, int(round(float(crop_size[0]))))
        original_crop_h = max(1, int(round(float(crop_size[1]))))
        original_canvas_w = max(1, int(round(float(canvas_size[0]))))
        original_canvas_h = max(1, int(round(float(canvas_size[1]))))
    except Exception:
        return crop

    content_w = int(round(crop.width * original_crop_w / original_canvas_w))
    content_h = int(round(crop.height * original_crop_h / original_canvas_h))
    content_w = max(1, min(crop.width, content_w))
    content_h = max(1, min(crop.height, content_h))
    return crop.crop((0, 0, content_w, content_h))


def _target_box_from_info(info: Dict[str, Any], image_width: int, image_height: int) -> Tuple[int, int, int, int]:
    box = info.get("box") or [0, 0, 1, 1]
    try:
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
    except Exception:
        return (0, 0, 1, 1)

    source_size = info.get("source_size") or [image_width, image_height]
    try:
        source_w = float(source_size[0])
        source_h = float(source_size[1])
    except Exception:
        source_w = float(image_width)
        source_h = float(image_height)

    if source_w > 0 and source_h > 0:
        x1 = x1 * image_width / source_w
        x2 = x2 * image_width / source_w
        y1 = y1 * image_height / source_h
        y2 = y2 * image_height / source_h

    ix1 = int(round(min(x1, x2)))
    iy1 = int(round(min(y1, y2)))
    ix2 = int(round(max(x1, x2)))
    iy2 = int(round(max(y1, y2)))

    ix1 = max(0, min(max(0, image_width - 1), ix1))
    iy1 = max(0, min(max(0, image_height - 1), iy1))
    ix2 = max(0, min(image_width, ix2))
    iy2 = max(0, min(image_height, iy2))

    if ix2 <= ix1:
        ix2 = min(image_width, ix1 + 1)
    if iy2 <= iy1:
        iy2 = min(image_height, iy1 + 1)

    return ix1, iy1, ix2, iy2


def _paste_clipped(base: Image.Image, patch: Image.Image, x: int, y: int) -> None:
    left = max(0, -x)
    top = max(0, -y)
    right = min(patch.width, base.width - x)
    bottom = min(patch.height, base.height - y)

    if right <= left or bottom <= top:
        return

    if left != 0 or top != 0 or right != patch.width or bottom != patch.height:
        patch = patch.crop((left, top, right, bottom))

    base.paste(patch.convert("RGB"), (x + left, y + top))


class DeepSeekOCRDrawBBox:
    """Draw DeepSeek OCR bounding boxes and crop the bbox regions."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "ocr_result": ("STRING", {"forceInput": True}),
                "coord_base": ("INT", {"default": 1000, "min": 0, "max": 100000, "step": 1}),
                "crop_expand": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "box_color": ("STRING", {"default": "#ff0000"}),
                "box_width": ("INT", {"default": 3, "min": 1, "max": 100, "step": 1}),
                "label": (["none", "ref", "text", "ref + text"], {"default": "none"}),
                "label_color": ("STRING", {"default": "#ffffff"}),
                "label_font_size": ("INT", {"default": 20, "min": 6, "max": 200, "step": 1}),
                "font_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("image", "crop_images", "crop_info")
    FUNCTION = "draw_bbox"
    CATEGORY = "DeepSeek OCR"

    def draw_bbox(
        self,
        image: torch.Tensor,
        ocr_result: Any,
        coord_base: int = 1000,
        crop_expand: int = 0,
        box_color: str = "#ff0000",
        box_width: int = 3,
        label: str = "none",
        label_color: str = "#ffffff",
        label_font_size: int = 20,
        font_path: str = "",
    ):
        annotations = parse_deepseek_ocr(ocr_result)
        outline = _parse_color(box_color, (255, 0, 0))
        label_fg = _parse_color(label_color, (255, 255, 255))
        width_px = max(1, int(box_width))
        coord_base = int(coord_base) if coord_base is not None else 1000
        crop_expand_px = max(0, int(crop_expand or 0))
        font = _load_font(font_path, int(label_font_size)) if label and label != "none" else ImageFont.load_default()

        batch = _normalize_image_batch(image)

        output_images: List[torch.Tensor] = []
        crop_images: List[Image.Image] = []
        crop_infos: List[Dict[str, Any]] = []

        for image_index, img in enumerate(batch):
            source_pil = _tensor_image_to_pil(img)
            pil_image = source_pil.copy()
            draw = ImageDraw.Draw(pil_image)
            img_w, img_h = pil_image.size

            for ann_index, ann in enumerate(annotations):
                bbox_x1, bbox_y1, bbox_x2, bbox_y2 = _annotation_crop_rect(ann, img_w, img_h, coord_base)
                x1, y1, x2, y2 = _expand_crop_rect(
                    (bbox_x1, bbox_y1, bbox_x2, bbox_y2), img_w, img_h, crop_expand_px
                )
                crop = source_pil.crop((x1, y1, x2, y2)).convert("RGB")
                crop_images.append(crop)
                crop_infos.append(
                    {
                        "image_index": image_index,
                        "bbox_index": ann_index,
                        "ref": _as_text(ann.get("ref", "")),
                        "text": _as_text(ann.get("text", "")),
                        "box": [x1, y1, x2, y2],
                        "original_box": [bbox_x1, bbox_y1, bbox_x2, bbox_y2],
                        "crop_size": [max(1, x2 - x1), max(1, y2 - y1)],
                        "source_size": [img_w, img_h],
                        "coord_base": coord_base,
                        "crop_expand": crop_expand_px,
                    }
                )

                if ann.get("type") == "polygon":
                    points = _scaled_points(ann.get("points", []), img_w, img_h, coord_base)
                    if len(points) >= 2:
                        draw.line(points + [points[0]], fill=outline, width=width_px, joint="curve")
                else:
                    draw.rectangle(
                        [bbox_x1, bbox_y1, max(bbox_x1, bbox_x2 - 1), max(bbox_y1, bbox_y2 - 1)],
                        outline=outline,
                        width=width_px,
                    )

                label_text = _annotation_label(ann, label)
                _draw_label(draw, (bbox_x1, bbox_y1), label_text, font, label_fg, outline, img_w, img_h)

            output_images.append(_pil_to_tensor(pil_image, image.device))

        crop_batch = _crops_to_batch(crop_images, crop_infos, image.device)
        crop_info_json = json.dumps(crop_infos, ensure_ascii=False)

        return (torch.stack(output_images, dim=0), crop_batch, crop_info_json)


class DeepSeekOCRDrawBBoxPasteText(DeepSeekOCRDrawBBox):
    """Same as DeepSeekOCRDrawBBox, but OCR result is a multiline paste textbox."""

    @classmethod
    def INPUT_TYPES(cls):
        types = super().INPUT_TYPES()
        types = {"required": dict(types["required"])}
        types["required"]["ocr_result"] = (
            "STRING",
            {
                "multiline": True,
                "default": "<|ref|>title<|/ref|><|det|>[[12, 0, 386, 45]]<|/det|>\n# BENBO本博",
            },
        )
        return types


class DeepSeekOCRPasteBBoxCrops:
    """Paste bbox crops back onto the original image using crop_info from DeepSeekOCRDrawBBox."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "crop_images": ("IMAGE",),
                "crop_info": ("STRING", {"forceInput": True}),
                "strip_padding": ("BOOLEAN", {"default": True}),
                "resize_to_bbox": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "paste_crops"
    CATEGORY = "DeepSeek OCR"

    def paste_crops(
        self,
        original_image: torch.Tensor,
        crop_images: torch.Tensor,
        crop_info: Any,
        strip_padding: bool = True,
        resize_to_bbox: bool = True,
    ):
        original_batch = _normalize_image_batch(original_image)
        crop_batch = _normalize_image_batch(crop_images)
        infos = _parse_crop_info(crop_info)

        output_pils = [_tensor_image_to_pil(img).copy() for img in original_batch]
        if not infos:
            return (torch.stack([_pil_to_tensor(img, original_image.device) for img in output_pils], dim=0),)

        should_strip_padding = _to_bool(strip_padding)
        should_resize = _to_bool(resize_to_bbox)
        count = min(len(infos), int(crop_batch.shape[0]))

        for crop_index in range(count):
            info = infos[crop_index]
            try:
                image_index = int(info.get("image_index", 0))
            except Exception:
                image_index = 0
            if image_index < 0 or image_index >= len(output_pils):
                continue

            base = output_pils[image_index]
            crop = _tensor_image_to_pil(crop_batch[crop_index]).convert("RGB")
            if should_strip_padding:
                crop = _strip_padding_from_crop(crop, info)

            x1, y1, x2, y2 = _target_box_from_info(info, base.width, base.height)
            target_w = max(1, x2 - x1)
            target_h = max(1, y2 - y1)
            if should_resize and crop.size != (target_w, target_h):
                crop = _resize_image(crop, (target_w, target_h))

            _paste_clipped(base, crop, x1, y1)

        return (torch.stack([_pil_to_tensor(img, original_image.device) for img in output_pils], dim=0),)


NODE_CLASS_MAPPINGS = {
    "DeepSeekOCRDrawBBox": DeepSeekOCRDrawBBox,
    "DeepSeekOCRDrawBBoxPasteText": DeepSeekOCRDrawBBoxPasteText,
    "DeepSeekOCRPasteBBoxCrops": DeepSeekOCRPasteBBoxCrops,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeepSeekOCRDrawBBox": "DeepSeek OCR Draw BBox",
    "DeepSeekOCRDrawBBoxPasteText": "DeepSeek OCR Draw BBox (Paste Text)",
    "DeepSeekOCRPasteBBoxCrops": "DeepSeek OCR Paste BBox Crops",
}
