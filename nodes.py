import ast
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


def _scaled_points(points: Iterable[Tuple[float, float]], width: int, height: int, coord_base: int) -> List[Tuple[int, int]]:
    return [(_scale_value(x, width, coord_base), _scale_value(y, height, coord_base)) for x, y in points]


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


class DeepSeekOCRDrawBBox:
    """Draw DeepSeek OCR bounding boxes on a ComfyUI IMAGE tensor."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "ocr_result": ("STRING", {"forceInput": True}),
                "coord_base": ("INT", {"default": 1000, "min": 0, "max": 100000, "step": 1}),
                "box_color": ("STRING", {"default": "#ff0000"}),
                "box_width": ("INT", {"default": 3, "min": 1, "max": 100, "step": 1}),
                "label": (["none", "ref", "text", "ref + text"], {"default": "none"}),
                "label_color": ("STRING", {"default": "#ffffff"}),
                "label_font_size": ("INT", {"default": 20, "min": 6, "max": 200, "step": 1}),
                "font_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "draw_bbox"
    CATEGORY = "DeepSeek OCR"

    def draw_bbox(
        self,
        image: torch.Tensor,
        ocr_result: Any,
        coord_base: int = 1000,
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
        font = _load_font(font_path, int(label_font_size)) if label and label != "none" else ImageFont.load_default()

        if image.ndim == 3:
            batch = image.unsqueeze(0)
        elif image.ndim == 4:
            batch = image
        else:
            raise ValueError(f"Unsupported IMAGE tensor shape: {tuple(image.shape)}")

        output_images: List[torch.Tensor] = []
        for img in batch:
            pil_image = _tensor_image_to_pil(img)
            draw = ImageDraw.Draw(pil_image)
            img_w, img_h = pil_image.size

            for ann in annotations:
                if ann.get("type") == "rect":
                    x1, y1, x2, y2 = _scaled_rect(ann.get("box", [0, 0, 0, 0]), img_w, img_h, coord_base)
                    draw.rectangle([x1, y1, x2, y2], outline=outline, width=width_px)
                    label_text = _annotation_label(ann, label)
                    _draw_label(draw, (x1, y1), label_text, font, label_fg, outline, img_w, img_h)
                elif ann.get("type") == "polygon":
                    points = _scaled_points(ann.get("points", []), img_w, img_h, coord_base)
                    if len(points) >= 2:
                        draw.line(points + [points[0]], fill=outline, width=width_px, joint="curve")
                        min_x = min(p[0] for p in points)
                        min_y = min(p[1] for p in points)
                        label_text = _annotation_label(ann, label)
                        _draw_label(draw, (min_x, min_y), label_text, font, label_fg, outline, img_w, img_h)

            output_images.append(_pil_to_tensor(pil_image, image.device))

        return (torch.stack(output_images, dim=0),)


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


NODE_CLASS_MAPPINGS = {
    "DeepSeekOCRDrawBBox": DeepSeekOCRDrawBBox,
    "DeepSeekOCRDrawBBoxPasteText": DeepSeekOCRDrawBBoxPasteText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeepSeekOCRDrawBBox": "DeepSeek OCR Draw BBox",
    "DeepSeekOCRDrawBBoxPasteText": "DeepSeek OCR Draw BBox (Paste Text)",
}
