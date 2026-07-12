"""RapidOCR text detection to a native ComfyUI MASK.

The OCR engine is loaded lazily so the rest of this node pack remains usable
when RapidOCR is not installed. Models are provided by rapidocr-onnxruntime.
"""

import json
import threading
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageOps


_ENGINE_CACHE: Dict[Tuple[str, int], Any] = {}
_ENGINE_CACHE_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()


def _available_providers() -> List[str]:
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except Exception:
        return []


def _preload_cuda_libraries() -> None:
    """Load CUDA/cuDNN libraries shipped by PyTorch/NVIDIA pip packages."""
    try:
        import onnxruntime as ort

        preload = getattr(ort, "preload_dlls", None)
        if callable(preload):
            preload()
    except Exception:
        # Session construction below provides the actionable error/fallback.
        pass


def _make_engine(provider: str, cpu_threads: int):
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise RuntimeError(
            "RapidOCR Text Mask requires rapidocr-onnxruntime. Install it in "
            "ComfyUI's Python environment with: pip install rapidocr-onnxruntime"
        ) from exc

    kwargs: Dict[str, Any] = {}
    if cpu_threads > 0:
        kwargs["intra_op_num_threads"] = cpu_threads
        kwargs["inter_op_num_threads"] = 1

    use_cuda = provider == "cuda"
    if use_cuda:
        _preload_cuda_libraries()
    kwargs.update(
        det_use_cuda=use_cuda,
        cls_use_cuda=use_cuda,
        rec_use_cuda=use_cuda,
    )
    return RapidOCR(**kwargs)


def _get_engine(accelerator: str, cpu_threads: int):
    requested = (accelerator or "auto").strip().lower()
    providers = _available_providers()
    cuda_available = "CUDAExecutionProvider" in providers

    if requested == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA was requested for RapidOCR, but ONNX Runtime does not expose "
            "CUDAExecutionProvider. Install a CUDA-compatible onnxruntime-gpu "
            "build in ComfyUI's Python environment, then restart ComfyUI. "
            f"Available providers: {providers or ['none']}"
        )

    provider = "cuda" if requested == "cuda" or (requested == "auto" and cuda_available) else "cpu"
    key = (provider, max(0, int(cpu_threads or 0)))

    with _ENGINE_CACHE_LOCK:
        if key in _ENGINE_CACHE:
            return _ENGINE_CACHE[key], provider
        try:
            engine = _make_engine(provider, key[1])
            if provider == "cuda" and not _engine_uses_cuda(engine):
                raise RuntimeError(
                    "RapidOCR initialized without CUDAExecutionProvider. Check the "
                    "onnxruntime-gpu/CUDA/cuDNN installation in ComfyUI's environment."
                )
        except Exception:
            # Auto remains usable if CUDA is advertised but cannot initialize.
            # Explicit CUDA still raises the initialization error.
            if requested != "auto" or provider != "cuda":
                raise
            provider = "cpu"
            key = (provider, max(0, int(cpu_threads or 0)))
            if key in _ENGINE_CACHE:
                return _ENGINE_CACHE[key], provider
            engine = _make_engine(provider, key[1])
        _ENGINE_CACHE[key] = engine
        return engine, provider


def _engine_uses_cuda(engine: Any) -> bool:
    """Verify all three RapidOCR ONNX sessions selected CUDA first."""
    for component_name in ("text_det", "text_cls", "text_rec"):
        component = getattr(engine, component_name, None)
        # RapidOCR 1.4 uses `infer` for det/cls and `session` for rec.
        infer = getattr(component, "infer", None) or getattr(component, "session", None)
        ort_session = getattr(infer, "session", None)
        if ort_session is None:
            return False
        try:
            if ort_session.get_providers()[0] != "CUDAExecutionProvider":
                return False
        except Exception:
            return False
    return True


def _image_tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = image.detach().cpu().numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        return Image.fromarray(array, mode="L").convert("RGB")
    if array.ndim == 3 and array.shape[-1] == 1:
        return Image.fromarray(array[..., 0], mode="L").convert("RGB")
    if array.ndim == 3 and array.shape[-1] >= 3:
        return Image.fromarray(array[..., :3], mode="RGB")
    raise ValueError(f"Unsupported ComfyUI IMAGE shape: {tuple(image.shape)}")


def _normalize_batch(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 3:
        return image.unsqueeze(0)
    if image.ndim == 4:
        return image
    raise ValueError(f"Unsupported ComfyUI IMAGE shape: {tuple(image.shape)}")


def _build_variants(image: Image.Image, speed_profile: str) -> List[Tuple[str, Image.Image]]:
    rgb = image.convert("RGB")
    if speed_profile == "fast":
        return [("rgb", rgb)]

    gray = ImageOps.grayscale(rgb)
    autocontrast = ImageOps.autocontrast(gray)
    high_contrast = ImageEnhance.Contrast(autocontrast).enhance(2.8)
    sharp_contrast = ImageEnhance.Sharpness(high_contrast).enhance(1.8)
    variants = [("rgb", rgb), ("gray_contrast", sharp_contrast.convert("RGB"))]
    if speed_profile == "thorough":
        inverted = ImageOps.invert(autocontrast)
        variants.append(
            ("inverted_gray", ImageEnhance.Contrast(inverted).enhance(2.2).convert("RGB"))
        )
    return variants


def _box_iou(a: Sequence[int], b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if intersection <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return intersection / float(area_a + area_b - intersection)


def _normalize_text(text: str) -> str:
    return "".join(str(text).lower().split())


def _dedupe(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    for item in sorted(items, key=lambda row: row.get("score", 0.0), reverse=True):
        text = _normalize_text(item.get("text", ""))
        if not text:
            continue
        duplicate = any(
            text == _normalize_text(existing.get("text", ""))
            and _box_iou(item["box"], existing["box"]) > 0.45
            for existing in deduped
        )
        if not duplicate:
            deduped.append(item)
    return sorted(deduped, key=lambda row: (row["box"][1], row["box"][0]))


def _run_once(
    engine: Any,
    image: Image.Image,
    scale: float,
    variant_name: str,
    minimum_confidence: float,
) -> List[Dict[str, Any]]:
    if scale == 1.0:
        ocr_image = image
    else:
        width, height = image.size
        resampling = getattr(Image, "Resampling", Image)
        ocr_image = image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            resampling.LANCZOS,
        )

    # RapidOCR mutates text_score when supplied, so serialize calls shared by
    # cached engines. This also keeps each ONNX session invocation thread-safe.
    with _INFERENCE_LOCK:
        result, _elapsed = engine(
            np.asarray(ocr_image),
            text_score=max(0.0, min(1.0, float(minimum_confidence))),
        )

    output: List[Dict[str, Any]] = []
    for row in result or []:
        if not row or len(row) < 3:
            continue
        points, text, score = row[:3]
        score = float(score)
        if score < minimum_confidence:
            continue
        polygon = [[float(point[0]) / scale, float(point[1]) / scale] for point in points]
        if len(polygon) < 3:
            continue
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        output.append(
            {
                "text": str(text),
                "score": score,
                "box": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                "polygon": [[round(x, 2), round(y, 2)] for x, y in polygon],
                "ocr_scale": scale,
                "ocr_variant": variant_name,
            }
        )
    return output


def _run_ocr(
    engine: Any,
    image: Image.Image,
    speed_profile: str,
    minimum_confidence: float,
) -> List[Dict[str, Any]]:
    width, height = image.size
    scales = [1.0]
    if speed_profile == "thorough" and max(width, height) <= 1400:
        scales.append(2.0)
    elif speed_profile == "balanced" and max(width, height) <= 1100:
        scales.append(2.0)

    items: List[Dict[str, Any]] = []
    for variant_name, variant_image in _build_variants(image, speed_profile):
        variant_scales = scales
        if speed_profile == "balanced" and variant_name != "rgb":
            variant_scales = [1.0]
        for scale in variant_scales:
            items.extend(
                _run_once(engine, variant_image, scale, variant_name, minimum_confidence)
            )
    return _dedupe(items)


def _expanded_polygon(
    polygon: Sequence[Sequence[float]], padding: int, width: int, height: int
) -> List[Tuple[int, int]]:
    points = np.asarray(polygon, dtype=np.float32)
    center = points.mean(axis=0)
    vectors = points - center
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    directions = vectors / np.maximum(lengths, 1.0)
    expanded = points + directions * float(max(0, padding))
    expanded[:, 0] = np.clip(expanded[:, 0], 0, max(0, width - 1))
    expanded[:, 1] = np.clip(expanded[:, 1], 0, max(0, height - 1))
    return [(int(round(x)), int(round(y))) for x, y in expanded]


def _items_to_mask(
    items: List[Dict[str, Any]],
    width: int,
    height: int,
    padding: int,
    mask_shape: str,
) -> np.ndarray:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    padding = max(0, int(padding))
    for item in items:
        if mask_shape == "polygon" and item.get("polygon"):
            draw.polygon(
                _expanded_polygon(item["polygon"], padding, width, height),
                fill=255,
            )
        else:
            x1, y1, x2, y2 = item["box"]
            box = (
                max(0, x1 - padding),
                max(0, y1 - padding),
                min(width - 1, x2 + padding),
                min(height - 1, y2 + padding),
            )
            draw.rectangle(box, fill=255)
    return np.asarray(mask, dtype=np.float32) / 255.0


class RapidOCRDetectText:
    """Run RapidOCR and return stable, pixel-coordinate detections for downstream logic."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "speed_profile": (["fast", "balanced", "thorough"], {"default": "balanced"}),
                "candidate_confidence": (
                    "FLOAT",
                    {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "accelerator": (["auto", "cpu", "cuda"], {"default": "auto"}),
                "cpu_threads": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("detections_json",)
    FUNCTION = "detect"
    CATEGORY = "DeepSeek OCR"
    DESCRIPTION = (
        "Detection-only RapidOCR node. Use a relatively permissive candidate "
        "threshold, then classify or review detections downstream."
    )

    def detect(
        self,
        image: torch.Tensor,
        speed_profile: str = "balanced",
        candidate_confidence: float = 0.50,
        accelerator: str = "auto",
        cpu_threads: int = 0,
    ):
        batch = _normalize_batch(image)
        engine, active_provider = _get_engine(accelerator, cpu_threads)
        batch_detections: List[Dict[str, Any]] = []
        threshold = max(0.0, min(1.0, float(candidate_confidence)))

        for image_index, tensor_image in enumerate(batch):
            pil_image = _image_tensor_to_pil(tensor_image)
            items = _run_ocr(engine, pil_image, speed_profile, threshold)
            for detection_index, item in enumerate(items):
                item["id"] = f"b{image_index}_d{detection_index}"
            batch_detections.append(
                {
                    "schema": "comfyui_rapidocr_detections/v1",
                    "image_index": image_index,
                    "width": pil_image.width,
                    "height": pil_image.height,
                    "provider": active_provider,
                    "candidate_confidence": threshold,
                    "count": len(items),
                    "detections": items,
                }
            )
        return (json.dumps(batch_detections, ensure_ascii=False),)


class RapidOCRTextMask:
    """Recognize all text with RapidOCR and return its regions as a MASK."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "speed_profile": (["fast", "balanced", "thorough"], {"default": "balanced"}),
                "minimum_confidence": (
                    "FLOAT",
                    {"default": 0.72, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "padding": ("INT", {"default": 8, "min": 0, "max": 256, "step": 1}),
                "mask_shape": (["polygon", "rectangle"], {"default": "polygon"}),
                "accelerator": (["auto", "cpu", "cuda"], {"default": "auto"}),
                "cpu_threads": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "invert_mask": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "detections_json")
    FUNCTION = "detect"
    CATEGORY = "DeepSeek OCR"
    DESCRIPTION = (
        "Runs PP-OCR models through RapidOCR ONNXRuntime and turns every text "
        "detection into a standard ComfyUI MASK. CUDA requires onnxruntime-gpu."
    )

    def detect(
        self,
        image: torch.Tensor,
        speed_profile: str = "balanced",
        minimum_confidence: float = 0.72,
        padding: int = 8,
        mask_shape: str = "polygon",
        accelerator: str = "auto",
        cpu_threads: int = 0,
        invert_mask: bool = False,
    ):
        batch = _normalize_batch(image)
        engine, active_provider = _get_engine(accelerator, cpu_threads)
        masks: List[torch.Tensor] = []
        batch_detections: List[Dict[str, Any]] = []

        for image_index, tensor_image in enumerate(batch):
            pil_image = _image_tensor_to_pil(tensor_image)
            items = _run_ocr(
                engine,
                pil_image,
                speed_profile,
                max(0.0, min(1.0, float(minimum_confidence))),
            )
            mask_array = _items_to_mask(
                items,
                pil_image.width,
                pil_image.height,
                padding,
                mask_shape,
            )
            if invert_mask:
                mask_array = 1.0 - mask_array
            masks.append(torch.from_numpy(np.array(mask_array, dtype=np.float32, copy=True)))
            batch_detections.append(
                {
                    "image_index": image_index,
                    "width": pil_image.width,
                    "height": pil_image.height,
                    "provider": active_provider,
                    "count": len(items),
                    "detections": items,
                }
            )

        # Native ComfyUI MASK: float32 [batch, height, width], where 1 is selected.
        mask_batch = torch.stack(masks, dim=0)
        return (mask_batch, json.dumps(batch_detections, ensure_ascii=False))
