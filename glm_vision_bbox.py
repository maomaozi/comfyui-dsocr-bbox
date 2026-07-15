"""ComfyUI nodes for GLM vision bbox extraction and JSON bbox masks.

The API node uses the OpenAI-compatible Zhipu chat completions endpoint and
returns a strict JSON list with pixel-coordinate bboxes.  The mask node consumes
that list and expands every rectangle by a fixed number of pixels.
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw


DEFAULT_ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
DEFAULT_MODEL = "glm-4.6v-flash"
DEFAULT_PROMPT = """输出图中所有banner，logo，品牌，平台，活动，质保，店铺相关描述的区域bbox，输出为json list，格式如下，不要输出任何其他内容/格式：
[
    {
        "desc": "官方旗舰店",
        "class": "店铺",
        "bbox": [x1,y1,x2,y2]
    }
]"""

_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _normalize_image_batch(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError("GLM Vision BBox requires a ComfyUI IMAGE tensor")
    if image.ndim == 3:
        return image.unsqueeze(0)
    if image.ndim == 4:
        return image
    raise ValueError(f"Unsupported ComfyUI IMAGE shape: {tuple(image.shape)}")


def _tensor_image_to_pil(image: torch.Tensor) -> Image.Image:
    array = image.detach().cpu().float().numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        return Image.fromarray(array, mode="L").convert("RGB")
    if array.ndim == 3 and array.shape[-1] == 1:
        return Image.fromarray(array[..., 0], mode="L").convert("RGB")
    if array.ndim == 3 and array.shape[-1] >= 3:
        return Image.fromarray(array[..., :3], mode="RGB")
    raise ValueError(f"Unsupported ComfyUI IMAGE item shape: {tuple(image.shape)}")


def _image_as_base64_png(image: torch.Tensor) -> Tuple[str, int, int]:
    pil_image = _tensor_image_to_pil(image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG", optimize=False)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return encoded, pil_image.width, pil_image.height


def _content_as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(value)


def _http_error_message(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        detail = body
        try:
            parsed = json.loads(body)
            error = parsed.get("error", parsed) if isinstance(parsed, dict) else parsed
            if isinstance(error, dict):
                code = error.get("code")
                message = error.get("message") or error.get("msg")
                detail = " - ".join(str(value) for value in (code, message) if value)
            else:
                detail = str(error)
        except (ValueError, TypeError):
            pass
        detail = detail.strip()[:2000]
        return f"GLM API HTTP {exc.code} {exc.reason}" + (f": {detail}" if detail else "")
    if isinstance(exc, urllib.error.URLError):
        return f"GLM API connection failed: {exc.reason}"
    if isinstance(exc, TimeoutError):
        return "GLM API request timed out"
    return str(exc) or exc.__class__.__name__


def _decoded_json_candidates(text: str):
    """Yield JSON values from a plain or fenced LLM response."""
    cleaned = (text or "").strip().lstrip("﻿")
    if not cleaned:
        return

    candidates = [cleaned]
    for fenced in re.findall(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.I | re.S):
        candidates.insert(0, fenced.strip())

    decoder = json.JSONDecoder()
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            yield json.loads(candidate)
            continue
        except json.JSONDecodeError:
            pass

        # Prefer a complete Python literal before scanning nested JSON fragments.
        # Otherwise a response like [{'bbox': [1,2,3,4]}] would incorrectly
        # expose only its inner, valid-JSON coordinate list.
        try:
            import ast

            literal = ast.literal_eval(candidate)
        except (ImportError, ValueError, SyntaxError):
            literal = None
        if isinstance(literal, (list, dict)):
            yield literal
            continue

        # Accept surrounding prose by trying every list/object start. raw_decode
        # correctly handles brackets and braces inside quoted strings.
        for index, char in enumerate(candidate):
            if char not in "[{":
                continue
            try:
                value, _end = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            yield value

        # Also allow a Python-style list/dict after a short prose prefix.
        try:
            import ast

            for index, char in enumerate(candidate):
                if index == 0 or char not in "[{":
                    continue
                try:
                    literal = ast.literal_eval(candidate[index:])
                except (ValueError, SyntaxError):
                    continue
                if isinstance(literal, (list, dict)):
                    yield literal
                    break
        except ImportError:
            pass


def _unwrap_result_list(value: Any) -> List[Any] | None:
    if isinstance(value, list):
        return value
    if not isinstance(value, dict):
        return None
    if any(key in value for key in ("bbox", "box", "rect")):
        return [value]
    for key in ("results", "items", "detections", "data", "regions"):
        nested = value.get(key)
        if isinstance(nested, list):
            return nested
    return None


def _bbox_values(value: Any) -> List[float] | None:
    if isinstance(value, str):
        numbers = _NUMBER_RE.findall(value)
        if len(numbers) >= 4:
            value = [float(number) for number in numbers[:4]]
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    if not all(isinstance(number, (int, float)) and not isinstance(number, bool) for number in value[:4]):
        return None
    return [float(number) for number in value[:4]]


def _normalize_bbox(
    bbox: Sequence[float],
    width: int | None = None,
    height: int | None = None,
    coord_base: int = 0,
) -> List[int] | None:
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    # GLM visual grounding natively uses a 0-1000 coordinate space.  The API
    # node requests that stable format, then converts it to source-image pixels
    # before exposing JSON to downstream ComfyUI nodes.
    base = max(0, int(coord_base or 0))
    if base > 0:
        if width is None or height is None or width <= 0 or height <= 0:
            raise ValueError("coord_base conversion requires positive image dimensions")
        x1 = x1 * float(width) / float(base)
        x2 = x2 * float(width) / float(base)
        y1 = y1 * float(height) / float(base)
        y2 = y2 * float(height) / float(base)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if width is not None and width > 0:
        x1 = min(max(0.0, x1), float(width))
        x2 = min(max(0.0, x2), float(width))
    if height is not None and height > 0:
        y1 = min(max(0.0, y1), float(height))
        y2 = min(max(0.0, y2), float(height))
    output = [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
    if output[2] <= output[0] or output[3] <= output[1]:
        return None
    return output


def parse_bbox_json(
    text_or_value: Any,
    width: int | None = None,
    height: int | None = None,
    coord_base: int = 0,
) -> List[Dict[str, Any]]:
    """Extract and normalize ``desc/class/bbox`` records.

    ``coord_base=0`` means input coordinates are pixels.  A positive value,
    notably GLM grounding's native 1000, is converted to source-image pixels.
    """
    if isinstance(text_or_value, str):
        raw_items = None
        for candidate in _decoded_json_candidates(text_or_value):
            raw_items = _unwrap_result_list(candidate)
            if raw_items is not None:
                break
        if raw_items is None:
            raise ValueError("GLM response does not contain a JSON list or bbox object")
    else:
        raw_items = _unwrap_result_list(text_or_value)
        if raw_items is None:
            raise ValueError("BBox JSON must be a list or an object containing a bbox")

    output: List[Dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        bbox = _bbox_values(item.get("bbox", item.get("box", item.get("rect"))))
        if bbox is None:
            continue
        normalized = _normalize_bbox(bbox, width, height, coord_base=coord_base)
        if normalized is None:
            continue
        desc = item.get("desc", item.get("description", item.get("text", item.get("label", ""))))
        class_name = item.get("class", item.get("category", item.get("type", "")))
        output.append(
            {
                "desc": "" if desc is None else str(desc),
                "class": "" if class_name is None else str(class_name),
                "bbox": normalized,
            }
        )
    return output


def _build_api_prompt(user_prompt: str) -> str:
    return f"""{user_prompt.strip()}

强制要求：
1. bbox 使用 GLM 视觉定位的 0-1000 归一化坐标，左上角为 (0,0)，右下角为 (1000,1000)。
2. 坐标必须满足 0 <= x1 < x2 <= 1000，0 <= y1 < y2 <= 1000。
3. 每个元素只保留 desc、class、bbox 三个字段。
4. 用户提示词中的 JSON/“官方旗舰店”仅是输出结构示例，不代表图中真实内容；禁止照抄示例，必须以图片为准。
5. 严格按照用户指定的目标范围检测；如果用户列出了类别，只输出这些类别，不要自行加入无关的产品、规格或功能类别。
6. 最终只输出合法 JSON list；不要 Markdown 代码块、解释、前后缀或其他格式。"""


def request_glm_bbox(
    image: torch.Tensor,
    prompt: str,
    endpoint: str,
    model: str,
    api_key: str,
    timeout: int = 300,
) -> str:
    """Call GLM and return a normalized, pretty-printed JSON list."""
    endpoint = str(endpoint or "").strip()
    model = str(model or "").strip()
    api_key = str(api_key or "").strip()
    prompt = str(prompt or "").strip()
    if not endpoint.startswith(("https://", "http://")):
        raise ValueError("endpoint must start with https:// or http://")
    if not model:
        raise ValueError("model cannot be empty")
    if not api_key:
        raise ValueError("api_key cannot be empty")
    if not prompt:
        raise ValueError("prompt cannot be empty")

    image_base64, width, height = _image_as_base64_png(image)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_base64}},
                    {"type": "text", "text": _build_api_prompt(prompt)},
                ],
            }
        ],
        "thinking": {"type": "disabled"},
        "stream": False,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "comfyui-dsocr-bbox-glm/1.0",
        },
    )

    response_data = None
    last_error: Exception | None = None
    # The free Flash endpoint may briefly return 429/code 1305 while capacity is
    # saturated. Retry a few times without requiring the user to requeue ComfyUI.
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=max(1, int(timeout))) as response:
                response_data = json.loads(response.read().decode("utf-8", errors="replace"))
            break
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code != 429 or attempt >= 2:
                raise RuntimeError(_http_error_message(exc)) from exc
            # Consume this response before retrying. A fresh Request is created
            # because some HTTP handlers attach state to the previous instance.
            try:
                exc.read()
            except Exception:
                pass
            time.sleep(2.0 * (attempt + 1))
            request = urllib.request.Request(
                endpoint,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                method="POST",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "comfyui-dsocr-bbox-glm/1.0",
                },
            )
        except Exception as exc:
            raise RuntimeError(_http_error_message(exc)) from exc

    if response_data is None:
        raise RuntimeError(_http_error_message(last_error or RuntimeError("empty API response")))

    try:
        message = response_data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "GLM API returned an unexpected response: "
            + json.dumps(response_data, ensure_ascii=False)[:2000]
        ) from exc

    raw_text = _content_as_text(message.get("content"))
    if not raw_text:
        raw_text = _content_as_text(message.get("reasoning_content"))
    records = parse_bbox_json(raw_text, width=width, height=height, coord_base=1000)
    return json.dumps(records, ensure_ascii=False, indent=2)


class GLMVisionBBoxExtractor:
    """Send one image and a directly editable prompt to a configurable GLM API."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": DEFAULT_PROMPT}),
                "endpoint": ("STRING", {"default": DEFAULT_ENDPOINT}),
                "model": ("STRING", {"default": DEFAULT_MODEL}),
                "api_key": ("STRING", {"default": "", "password": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("bbox_json",)
    FUNCTION = "extract"
    CATEGORY = "dsocr_bbox/GLM Vision BBox"
    DESCRIPTION = (
        "Calls a configurable GLM vision chat-completions API for one image and "
        "returns a strict JSON list containing desc, class, and pixel bbox."
    )

    def extract(
        self,
        image: torch.Tensor,
        prompt: str = DEFAULT_PROMPT,
        endpoint: str = DEFAULT_ENDPOINT,
        model: str = DEFAULT_MODEL,
        api_key: str = "",
    ):
        batch = _normalize_image_batch(image)
        if int(batch.shape[0]) != 1:
            raise ValueError(
                f"GLM Vision BBox Extract supports exactly one image, received batch size {batch.shape[0]}"
            )
        result = request_glm_bbox(batch[0], prompt, endpoint, model, api_key)
        return (result,)


class GLMBBoxJSONToMask:
    """Rasterize GLM bbox JSON and expand all sides by a fixed pixel value."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bbox_json": ("STRING", {"forceInput": True}),
                "mask_expand": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "coord_base": ("INT", {"default": 1000, "min": 0, "max": 100000, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "json_to_mask"
    CATEGORY = "dsocr_bbox/GLM Vision BBox"
    DESCRIPTION = (
        "Converts GLM desc/class/bbox JSON to a native ComfyUI MASK. Positive "
        "coord_base values are converted to pixels; 0 treats coordinates as pixels."
    )

    def json_to_mask(
        self,
        image: torch.Tensor,
        bbox_json: Any,
        mask_expand: int = 0,
        coord_base: int = 1000,
    ):
        batch = _normalize_image_batch(image)
        batch_size, height, width = int(batch.shape[0]), int(batch.shape[1]), int(batch.shape[2])
        records = parse_bbox_json(
            bbox_json, width=width, height=height, coord_base=max(0, int(coord_base or 0))
        )
        padding = max(0, int(mask_expand or 0))

        canvas = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(canvas)
        for record in records:
            x1, y1, x2, y2 = record["bbox"]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)
            if x2 > x1 and y2 > y1:
                # PIL's rectangle end is inclusive; bbox coordinates are half-open.
                draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=255)

        mask_array = np.asarray(canvas, dtype=np.float32) / 255.0
        mask = torch.from_numpy(mask_array.copy()).unsqueeze(0).repeat(batch_size, 1, 1)
        return (mask,)


__all__ = [
    "DEFAULT_ENDPOINT",
    "DEFAULT_MODEL",
    "DEFAULT_PROMPT",
    "GLMBBoxJSONToMask",
    "GLMVisionBBoxExtractor",
    "parse_bbox_json",
    "request_glm_bbox",
]
