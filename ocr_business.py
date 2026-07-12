"""Composable business classification, LLM review, and region expansion for OCR."""

import json
import re
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

try:
    from .rapidocr_mask import _expanded_polygon, _image_tensor_to_pil, _normalize_batch
except Exception:
    from rapidocr_mask import _expanded_polygon, _image_tensor_to_pil, _normalize_batch


BRAND_TERMS = "品牌,logo,trademark,ROTAI,荣泰,PISEN,品胜,BENBO,CCTV,Tosfu,CuteAngel"
RESTRICTED_TERMS = "官方,授权,官方授权,专卖,专营,旗舰店,正品,质保,保修,认证,CERTIFIED,APPROVED"
PROMOTION_TERMS = "价格,到手价,补贴,立省,优惠,满减,免费试用,退货,运费,咨询享,赠品,赠送,送好礼,¥,￥,price"
BUILTIN_PRESERVE_TERMS = "规格,型号,尺寸,容量,功率,材质,颜色,风力,香薰,高度,智能,机芯,小时,遥控,定时,功能,Type-C,数据线"
_ALLOWED_ACTIONS = {"remove", "preserve", "ignore", "review"}
_ALLOWED_POLICIES = {
    "text", "box", "top_banner", "bottom_banner", "group_box",
    "gift_object", "explicit_box", "none",
}


def _loads(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    text = str(value or "").strip()
    if not text:
        return []
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.I | re.S)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        starts = [p for p in (text.find("{"), text.find("[")) if p >= 0]
        for start in sorted(starts):
            for end_char in ("}", "]"):
                end = text.rfind(end_char)
                if end > start:
                    try:
                        return json.loads(text[start : end + 1])
                    except Exception:
                        pass
    raise ValueError("Expected valid JSON, optionally inside a Markdown JSON code block")


def _batches(value: Any) -> List[Dict[str, Any]]:
    data = _loads(value)
    if isinstance(data, dict):
        if isinstance(data.get("batches"), list):
            data = data["batches"]
        elif isinstance(data.get("detections"), list):
            data = [data]
        else:
            data = [{"detections": []}]
    if not isinstance(data, list):
        raise ValueError("OCR data must be a batch list or an object containing detections")
    if data and all(isinstance(item, dict) and "text" in item for item in data):
        data = [{"detections": data}]
    output = []
    for batch_index, batch in enumerate(data):
        if not isinstance(batch, dict):
            continue
        copied = dict(batch)
        copied["image_index"] = int(copied.get("image_index", batch_index))
        copied["detections"] = [dict(item) for item in copied.get("detections", []) if isinstance(item, dict)]
        output.append(copied)
    return output


def _terms(text: str) -> List[str]:
    return ["".join(x.lower().split()) for x in re.split(r"[,，\n]+", str(text or "")) if x.strip()]


def _norm(text: Any) -> str:
    return "".join(str(text or "").lower().split())


def _contains(text: str, terms: Sequence[str]) -> bool:
    return any(term and term in text for term in terms)


def _is_product_spec(text: str) -> bool:
    normalized = _norm(text)
    if _contains(normalized, _terms(BUILTIN_PRESERVE_TERMS)):
        return True
    patterns = (
        r"\d+(?:\.\d+)?\s*(?:ml|l|kg|g|mm|cm|m|w|v|mah|hz|gb|tb|寸|升|克|斤|瓦)",
        r"(?:型号|规格|尺寸|容量|功率|材质|颜色)[:：]?",
    )
    return any(re.search(pattern, normalized, re.I) for pattern in patterns)


def _classify(
    item: Dict[str, Any], width: int, height: int, minimum_confidence: float,
    bottom_start_ratio: float, preserve_left_features: bool,
    remove_terms: Sequence[str], preserve_terms: Sequence[str],
) -> Tuple[str, str, str, str]:
    text = _norm(item.get("text"))
    score = float(item.get("score", 0.0))
    box = item.get("box") or [0, 0, 0, 0]
    x1, y1, x2, y2 = [float(v) for v in box[:4]]

    if score < minimum_confidence:
        return "ignore", "low_confidence", "unknown", "none"
    if _is_product_spec(text) or _contains(text, preserve_terms):
        return "preserve", "preserve_keyword_or_spec", "functional", "none"
    if _contains(text, remove_terms):
        if _contains(text, _terms(BRAND_TERMS)):
            policy = "top_banner" if y2 <= height * 0.14 else "text"
            return "remove", "remove_brand_keyword", "brand", policy
        if _contains(text, _terms(PROMOTION_TERMS)):
            # A rule can identify promotional semantics, but cannot reliably
            # infer that every nearby visual block is removable. Let an LLM or
            # explicit downstream policy promote it to bottom_banner.
            policy = "bottom_banner" if y1 >= height * bottom_start_ratio else "text"
            return "remove", "remove_promotion_keyword", "promotion", policy
        return "remove", "remove_restricted_keyword", "restricted", "text"
    if preserve_left_features and x1 < width * 0.48 and height * 0.10 < y1 < height * 0.72:
        return "preserve", "left_feature_copy", "functional", "none"
    if y1 >= height * bottom_start_ratio:
        return "review", "bottom_unknown", "unknown", "text"
    if y2 <= height * 0.10 and x1 < width * 0.50:
        return "review", "top_logo_candidate", "brand_candidate", "top_banner"
    return "review", "unclassified", "unknown", "text"


def classify_batches(
    ocr_json: Any, minimum_confidence: float, bottom_start_ratio: float,
    preserve_left_features: bool, remove_keywords: str, preserve_keywords: str,
) -> List[Dict[str, Any]]:
    remove_terms = _terms(
        ",".join((BRAND_TERMS, RESTRICTED_TERMS, PROMOTION_TERMS, remove_keywords or ""))
    )
    preserve_terms = _terms(",".join((BUILTIN_PRESERVE_TERMS, preserve_keywords or "")))
    output = []
    for batch_index, batch in enumerate(_batches(ocr_json)):
        width, height = int(batch.get("width", 0)), int(batch.get("height", 0))
        detections = []
        for detection_index, source in enumerate(batch["detections"]):
            item = dict(source)
            item_id = str(item.get("id") or f"b{batch_index}_d{detection_index}")
            action, reason, category, policy = _classify(
                item, width, height, minimum_confidence, bottom_start_ratio,
                preserve_left_features, remove_terms, preserve_terms,
            )
            item.update(
                id=item_id,
                action=action,
                reason=reason,
                category=category,
                region_policy=policy,
                decision_source="rules",
            )
            detections.append(item)
        copied = dict(batch)
        copied.update(
            schema="comfyui_ocr_business/v1",
            width=width,
            height=height,
            count=len(detections),
            detections=detections,
        )
        output.append(copied)
    return output


def _decision_payload(value: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    data = _loads(value)
    additional: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        additional = data.get("additional_regions", [])
        data = data.get("decisions", data.get("detections", data.get("items", [])))
    if not isinstance(data, list):
        raise ValueError("LLM decision JSON must contain a decisions array")
    if not isinstance(additional, list):
        additional = []
    return (
        [item for item in data if isinstance(item, dict)],
        [item for item in additional if isinstance(item, dict)],
    )


def apply_decisions(classified_json: Any, decision_json: Any, unresolved: str) -> List[Dict[str, Any]]:
    batches = _batches(classified_json)
    decisions, additional_regions = _decision_payload(decision_json)
    by_id = {str(item.get("id")): item for item in decisions if item.get("id") is not None}
    for batch in batches:
        for item in batch["detections"]:
            decision = by_id.get(str(item.get("id")))
            if decision:
                action = str(decision.get("action", item.get("action", "review"))).lower()
                if action in _ALLOWED_ACTIONS:
                    item["action"] = action
                policy = str(decision.get("region_policy", item.get("region_policy", "text"))).lower()
                if policy in _ALLOWED_POLICIES:
                    item["region_policy"] = policy
                for key in ("category", "reason", "group", "region", "expand", "notes"):
                    if key in decision:
                        item[key] = decision[key]
                item["decision_source"] = "llm"
            elif item.get("action") == "review" and unresolved != "keep_rule":
                item["action"] = unresolved
                item["reason"] = f"unresolved_{unresolved}"
                item["decision_source"] = "fallback"

    # A multimodal LLM may see a logo, badge, gift, or banner with no usable OCR
    # anchor. Accept such geometry as an explicit synthetic removal decision.
    for region_index, region in enumerate(additional_regions):
        try:
            batch_index = int(region.get("image_index", 0))
        except Exception:
            batch_index = 0
        if batch_index < 0 or batch_index >= len(batches):
            continue
        box = region.get("region", region.get("box"))
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        policy = str(region.get("region_policy", "explicit_box")).lower()
        if policy not in _ALLOWED_POLICIES or policy in {"text", "none"}:
            policy = "explicit_box"
        synthetic = {
            "id": str(region.get("id") or f"llm_region_{batch_index}_{region_index}"),
            "text": str(region.get("text", "")),
            "score": 1.0,
            "box": list(box),
            "region": list(box),
            "action": "remove",
            "category": str(region.get("category", "visual_object")),
            "reason": str(region.get("reason", "llm_additional_region")),
            "region_policy": policy,
            "decision_source": "llm",
        }
        for key in ("group", "expand", "notes"):
            if key in region:
                synthetic[key] = region[key]
        batches[batch_index]["detections"].append(synthetic)
        batches[batch_index]["count"] = len(batches[batch_index]["detections"])
    return batches


def _clip_box(box: Sequence[Any], width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
    x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
    y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
    return [x1, y1, x2, y2]


def _pad_box(box: Sequence[Any], padding: int, width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = _clip_box(box, width, height)
    return [max(0, x1 - padding), max(0, y1 - padding), min(width, x2 + padding), min(height, y2 + padding)]


def _custom_expand(box: Sequence[Any], expand: Any, width: int, height: int) -> List[int]:
    if isinstance(expand, (int, float)):
        return _pad_box(box, int(expand), width, height)
    if isinstance(expand, dict):
        x1, y1, x2, y2 = _clip_box(box, width, height)
        return _clip_box([
            x1 - int(expand.get("left", 0)), y1 - int(expand.get("top", 0)),
            x2 + int(expand.get("right", 0)), y2 + int(expand.get("bottom", 0)),
        ], width, height)
    return _clip_box(box, width, height)


def _draw_item(draw: ImageDraw.ImageDraw, item: Dict[str, Any], padding: int, shape: str, width: int, height: int):
    if shape == "polygon" and item.get("polygon"):
        draw.polygon(_expanded_polygon(item["polygon"], padding, width, height), fill=255)
    else:
        draw.rectangle(tuple(_pad_box(item.get("box", [0, 0, 0, 0]), padding, width, height)), fill=255)


def build_regions(
    decisions: List[Dict[str, Any]], width: int, height: int, text_padding: int,
    region_padding: int, mask_shape: str, infer_gift_object: bool,
) -> Tuple[Image.Image, Image.Image, Image.Image, Image.Image, List[Dict[str, Any]]]:
    full = Image.new("L", (width, height), 0)
    large = Image.new("L", (width, height), 0)
    preserve = Image.new("L", (width, height), 0)
    full_draw, large_draw, preserve_draw = ImageDraw.Draw(full), ImageDraw.Draw(large), ImageDraw.Draw(preserve)
    records: List[Dict[str, Any]] = []

    remove_items = [x for x in decisions if x.get("action") == "remove"]
    for item in decisions:
        if item.get("action") == "preserve":
            _draw_item(preserve_draw, item, 2, mask_shape, width, height)

    top = [x for x in remove_items if x.get("region_policy") == "top_banner"]
    bottom = [x for x in remove_items if x.get("region_policy") == "bottom_banner"]
    handled = set()
    if top:
        boxes = [x.get("box") for x in top if len(x.get("box") or []) == 4]
        if boxes:
            union = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
            x1, y1, x2, y2 = _pad_box(union, region_padding, width, height)
            if x1 <= width * 0.15:
                x1 = 0
            if y1 <= height * 0.08:
                y1 = 0
            box = [x1, y1, x2, y2]
            full_draw.rectangle(tuple(box), fill=255)
            large_draw.rectangle(tuple(box), fill=255)
            records.append({"kind": "top_banner", "box": box, "source_ids": [x["id"] for x in top]})
            handled.update(x["id"] for x in top)
    if bottom:
        boxes = [x.get("box") for x in bottom if len(x.get("box") or []) == 4]
        if boxes:
            start = max(0, min(int(b[1]) for b in boxes) - text_padding - max(region_padding, int(height * 0.03)))
            box = [0, start, width, height]
            full_draw.rectangle(tuple(box), fill=255)
            large_draw.rectangle(tuple(box), fill=255)
            records.append({"kind": "bottom_banner", "box": box, "source_ids": [x["id"] for x in bottom]})
            handled.update(x["id"] for x in bottom)

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in remove_items:
        if item.get("region_policy") == "group_box":
            groups.setdefault(str(item.get("group") or item.get("category") or "default"), []).append(item)
    for group, items in groups.items():
        boxes = [x.get("box") for x in items if len(x.get("box") or []) == 4]
        if not boxes:
            continue
        union = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
        box = _pad_box(union, region_padding, width, height)
        full_draw.rectangle(tuple(box), fill=255)
        large_draw.rectangle(tuple(box), fill=255)
        records.append({"kind": "group_box", "group": group, "box": box, "source_ids": [x["id"] for x in items]})
        handled.update(x["id"] for x in items)

    for item in remove_items:
        item_id, policy = item.get("id"), item.get("region_policy", "text")
        if item_id in handled or policy == "none":
            continue
        if policy == "explicit_box" and len(item.get("region") or []) == 4:
            box = _custom_expand(item["region"], item.get("expand", region_padding), width, height)
            full_draw.rectangle(tuple(box), fill=255)
            large_draw.rectangle(tuple(box), fill=255)
            records.append({"kind": policy, "box": box, "source_ids": [item_id]})
        elif policy == "gift_object" and infer_gift_object:
            x1, y1, x2, y2 = _clip_box(item.get("box", [0, 0, 0, 0]), width, height)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            bw, bh = max(int(width * 0.20), 120), max(int(height * 0.16), 130)
            box = _clip_box([cx - bw * 0.45, cy - bh * 0.35, cx + bw * 0.55, cy + bh * 0.65], width, height)
            full_draw.rectangle(tuple(box), fill=255)
            large_draw.rectangle(tuple(box), fill=255)
            records.append({"kind": policy, "box": box, "source_ids": [item_id]})
        else:
            if policy == "box":
                box = _custom_expand(item.get("box", [0, 0, 0, 0]), item.get("expand", text_padding), width, height)
                full_draw.rectangle(tuple(box), fill=255)
            else:
                _draw_item(full_draw, item, text_padding, mask_shape, width, height)
                box = _pad_box(item.get("box", [0, 0, 0, 0]), text_padding, width, height)
            records.append({"kind": "text" if policy != "box" else policy, "box": box, "source_ids": [item_id]})

    full_array = np.asarray(full, dtype=np.uint8)
    large_array = np.asarray(large, dtype=np.uint8)
    detail = Image.fromarray(np.where(large_array > 0, 0, full_array).astype(np.uint8), mode="L")
    preserve_array = np.asarray(preserve, dtype=np.uint8).copy()
    preserve_array[full_array > 0] = 0
    return full, large, detail, Image.fromarray(preserve_array, mode="L"), records


def _mask_tensor(images: Sequence[Image.Image]) -> torch.Tensor:
    return torch.stack([torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0) for image in images], 0)


def _preview(image: Image.Image, mask: Image.Image) -> Image.Image:
    base = image.convert("RGB")
    overlay = Image.new("RGB", base.size, (255, 40, 40))
    alpha = mask.point(lambda value: int(value * 0.42))
    return Image.composite(overlay, base, alpha)


class OCRBusinessRuleClassifier:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "ocr_detections_json": ("STRING", {"forceInput": True}),
            "minimum_confidence": ("FLOAT", {"default": 0.72, "min": 0.0, "max": 1.0, "step": 0.01}),
            "bottom_start_ratio": ("FLOAT", {"default": 0.83, "min": 0.5, "max": 0.98, "step": 0.01}),
            "preserve_left_features": ("BOOLEAN", {"default": True}),
            "remove_keywords": ("STRING", {"multiline": True, "default": ""}),
            "preserve_keywords": ("STRING", {"multiline": True, "default": ""}),
        }}
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("classified_json", "review_items_json")
    FUNCTION = "classify"
    CATEGORY = "OCR/Business"

    def classify(self, ocr_detections_json, minimum_confidence=0.72, bottom_start_ratio=0.83,
                 preserve_left_features=True, remove_keywords="", preserve_keywords=""):
        batches = classify_batches(ocr_detections_json, float(minimum_confidence), float(bottom_start_ratio),
                                   bool(preserve_left_features), remove_keywords, preserve_keywords)
        review = [{"id": x["id"], "text": x.get("text", ""), "score": x.get("score", 0),
                   "box": x.get("box"), "proposed_action": x["action"], "category": x["category"],
                   "region_policy": x["region_policy"], "reason": x["reason"]}
                  for batch in batches for x in batch["detections"] if x["action"] == "review"]
        return (json.dumps(batches, ensure_ascii=False), json.dumps({"review_items": review}, ensure_ascii=False))


class OCRBusinessReviewPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "classified_json": ("STRING", {"forceInput": True}),
            "business_goal": ("STRING", {"multiline": True, "default": "Remove brand, authorization, promotional and gift content; preserve product specifications and functional copy."}),
        }}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("llm_prompt",)
    FUNCTION = "build"
    CATEGORY = "OCR/Business"

    def build(self, classified_json, business_goal):
        batches = _batches(classified_json)
        items = [{"id": x.get("id"), "text": x.get("text"), "score": x.get("score"), "box": x.get("box"),
                  "proposed_action": x.get("action"), "category": x.get("category"),
                  "region_policy": x.get("region_policy"), "reason": x.get("reason")}
                 for batch in batches for x in batch["detections"]]
        schema = {
            "decisions": [{"id": "b0_d0", "action": "remove|preserve|ignore",
                "category": "brand|promotion|restricted|gift|functional|unknown",
                "region_policy": "text|box|top_banner|bottom_banner|group_box|gift_object|explicit_box|none",
                "region": [0, 0, 100, 100], "group": "optional", "notes": "short reason"}],
            "additional_regions": [{"image_index": 0, "id": "visual_0", "action": "remove",
                "category": "logo|badge|gift|banner|visual_object", "region_policy": "explicit_box",
                "region": [0, 0, 100, 100], "notes": "visible target without an OCR anchor"}],
        }
        prompt = (
            "You review OCR detections for an image-editing mask. Business goal:\n" + str(business_goal) +
            "\nReturn JSON only. Keep every detection id exactly. Choose an action for every item. "
            "Use explicit_box with pixel region when text implies a nearby non-text object/background must also be removed. "
            "If you can see a target with no OCR detection, add it to additional_regions using pixel coordinates. "
            "Use top_banner/bottom_banner only when the whole visual block should be reconstructed. Schema:\n" +
            json.dumps(schema, ensure_ascii=False) + "\nDetections:\n" + json.dumps(items, ensure_ascii=False)
        )
        return (prompt,)


class OCRApplyBusinessDecisions:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "classified_json": ("STRING", {"forceInput": True}),
            "llm_decisions": ("STRING", {"forceInput": True}),
            "unresolved_action": (["keep_rule", "preserve", "ignore", "remove"], {"default": "keep_rule"}),
        }}
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("decisions_json", "remove_items_json", "preserve_items_json")
    FUNCTION = "apply"
    CATEGORY = "OCR/Business"

    def apply(self, classified_json, llm_decisions, unresolved_action="keep_rule"):
        batches = apply_decisions(classified_json, llm_decisions, unresolved_action)
        remove = [x for b in batches for x in b["detections"] if x.get("action") == "remove"]
        preserve = [x for b in batches for x in b["detections"] if x.get("action") == "preserve"]
        return tuple(json.dumps(x, ensure_ascii=False) for x in (batches, remove, preserve))


class OCRBusinessRegionsToMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "decisions_json": ("STRING", {"forceInput": True}),
            "text_padding": ("INT", {"default": 8, "min": 0, "max": 256}),
            "region_padding": ("INT", {"default": 24, "min": 0, "max": 512}),
            "mask_shape": (["polygon", "rectangle"], {"default": "polygon"}),
            "infer_gift_object": ("BOOLEAN", {"default": True}),
        }}
    RETURN_TYPES = ("MASK", "MASK", "MASK", "MASK", "STRING", "IMAGE")
    RETURN_NAMES = ("mask", "large_block_mask", "detail_mask", "preserve_mask", "regions_json", "preview")
    FUNCTION = "render"
    CATEGORY = "OCR/Business"

    def render(self, image, decisions_json, text_padding=8, region_padding=24,
               mask_shape="polygon", infer_gift_object=True):
        image_batch, batches = _normalize_batch(image), _batches(decisions_json)
        fulls, larges, details, preserves, previews, metadata = [], [], [], [], [], []
        for index, tensor_image in enumerate(image_batch):
            pil = _image_tensor_to_pil(tensor_image)
            batch = batches[min(index, len(batches) - 1)] if batches else {"detections": []}
            result = build_regions(batch.get("detections", []), pil.width, pil.height, int(text_padding),
                                   int(region_padding), mask_shape, bool(infer_gift_object))
            full, large, detail, preserve, records = result
            fulls.append(full)
            larges.append(large)
            details.append(detail)
            preserves.append(preserve)
            previews.append(torch.from_numpy(np.asarray(_preview(pil, full), dtype=np.float32) / 255.0))
            metadata.append({"image_index": index, "width": pil.width, "height": pil.height, "regions": records})
        return (_mask_tensor(fulls), _mask_tensor(larges), _mask_tensor(details), _mask_tensor(preserves),
                json.dumps(metadata, ensure_ascii=False), torch.stack(previews, 0))
