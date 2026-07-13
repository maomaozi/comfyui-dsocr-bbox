"""Utilities for expanding RapidOCR pixel-coordinate polygon JSON.

The JSON shape is preserved. Polygon records may appear directly in a list or
inside wrappers such as ``detections``, ``items``, ``results``, or batch items.
Expansion uses each polygon's enclosing rectangle for the same edge-wise rules
as the DeepSeek bbox nodes, then maps the original polygon into that expanded
rectangle so rotated/skewed polygon masks remain polygons.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from .bbox_expand import expand_subset_bboxes
except Exception:
    from bbox_expand import expand_subset_bboxes


Point = List[float]
Box = List[float]
PolygonRecord = Tuple[Dict[str, Any], str, List[Point]]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_polygon(value: Any) -> Optional[List[Point]]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    points: List[Point] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return None
        if not (_is_number(point[0]) and _is_number(point[1])):
            return None
        points.append([float(point[0]), float(point[1])])
    return points


def parse_json_value(value: Any) -> Any:
    """Parse strict JSON text or deep-copy an already decoded JSON value."""
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"RapidOCR polygon node received invalid JSON: {exc}") from exc
    return copy.deepcopy(value)


def collect_polygon_records(value: Any) -> List[PolygonRecord]:
    """Return mutable polygon records in stable document order."""
    records: List[PolygonRecord] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            # RapidOCR emits `polygon`; `points` is accepted for compatible JSON.
            for key in ("polygon", "points"):
                polygon = _as_polygon(item.get(key))
                if polygon is not None:
                    records.append((item, key, polygon))
                    return
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return records


def polygon_box(points: Sequence[Sequence[float]]) -> Box:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def infer_image_size(value: Any) -> Optional[Tuple[int, int]]:
    """Find the first positive width/height pair in a RapidOCR JSON wrapper."""
    if isinstance(value, dict):
        width = value.get("width")
        height = value.get("height")
        if _is_number(width) and _is_number(height) and width > 0 and height > 0:
            return int(round(float(width))), int(round(float(height)))
        for child in value.values():
            size = infer_image_size(child)
            if size is not None:
                return size
    elif isinstance(value, (list, tuple)):
        for child in value:
            size = infer_image_size(child)
            if size is not None:
                return size
    return None


def _clean_number(value: float) -> Any:
    rounded = round(float(value), 4)
    if abs(rounded) < 0.00005:
        return 0
    nearest_integer = round(rounded)
    if abs(rounded - nearest_integer) < 0.00005:
        return int(nearest_integer)
    return rounded


def _remap_polygon(
    points: Sequence[Sequence[float]],
    source_box: Sequence[float],
    target_box: Sequence[float],
) -> List[Point]:
    sx1, sy1, sx2, sy2 = [float(value) for value in source_box[:4]]
    tx1, ty1, tx2, ty2 = [float(value) for value in target_box[:4]]
    source_width = sx2 - sx1
    source_height = sy2 - sy1
    target_width = tx2 - tx1
    target_height = ty2 - ty1

    output: List[Point] = []
    for point in points:
        x, y = float(point[0]), float(point[1])
        if abs(source_width) > 1e-9:
            mapped_x = tx1 + (x - sx1) * target_width / source_width
        else:
            mapped_x = x + ((tx1 + tx2) - (sx1 + sx2)) / 2.0
        if abs(source_height) > 1e-9:
            mapped_y = ty1 + (y - sy1) * target_height / source_height
        else:
            mapped_y = y + ((ty1 + ty2) - (sy1 + sy2)) / 2.0
        output.append([_clean_number(mapped_x), _clean_number(mapped_y)])
    return output


def _updated_box_value(original: Any, box: Sequence[float]) -> List[Any]:
    values = [_clean_number(value) for value in box[:4]]
    if isinstance(original, (list, tuple)) and len(original) >= 4:
        if all(isinstance(value, int) and not isinstance(value, bool) for value in original[:4]):
            return [int(round(value)) for value in values]
    return values


def _apply_expanded_boxes(records: Sequence[PolygonRecord], boxes: Sequence[Sequence[float]]) -> None:
    if len(records) != len(boxes):
        raise ValueError(
            f"Polygon expansion produced {len(boxes)} boxes for {len(records)} polygons"
        )

    for (record, polygon_key, points), target_box in zip(records, boxes):
        source_box = polygon_box(points)
        expanded_polygon = _remap_polygon(points, source_box, target_box)
        record[polygon_key] = expanded_polygon
        enclosing_box = polygon_box(expanded_polygon)

        # RapidOCR detection JSON normally contains `box` as well. Keep any
        # existing rectangle metadata synchronized with the expanded polygon.
        for key in ("box", "bbox", "rect"):
            if key in record:
                record[key] = _updated_box_value(record[key], enclosing_box)


def _effective_image_size(
    value: Any,
    image_size: Optional[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    if image_size is not None:
        width, height = image_size
        if int(width) > 0 and int(height) > 0:
            return int(width), int(height)
    return infer_image_size(value)


def extend_polygon_json(
    value: Any,
    *,
    expand: float = 8,
    image_size: Optional[Tuple[int, int]] = None,
    coord_base: int = 0,
    clip_to_image: bool = True,
) -> Any:
    """Expand every polygon independently while preserving the JSON shape."""
    output = parse_json_value(value)
    records = collect_polygon_records(output)
    if not records:
        return output

    boxes = [polygon_box(points) for _record, _key, points in records]
    effective_size = _effective_image_size(output, image_size)
    expanded_boxes = expand_subset_bboxes(
        boxes,
        boxes,
        image_size=effective_size,
        max_expand=max(0.0, float(expand)),
        safety_margin=0,
        coord_base=max(0, int(coord_base or 0)),
        round_output=False,
        clip_to_image=bool(clip_to_image),
        ignore_empty_label_in_a=False,
    )
    _apply_expanded_boxes(records, expanded_boxes)
    return output


def extend_polygon_subset_json(
    value_a: Any,
    value_b: Any,
    *,
    max_expand: float = 100,
    safety_margin: float = 0,
    image_size: Optional[Tuple[int, int]] = None,
    coord_base: int = 0,
    clip_to_image: bool = True,
    ignore_empty_text_in_a: bool = True,
) -> Any:
    """Expand B polygons while treating polygons in A - B as obstacles."""
    parsed_a = parse_json_value(value_a)
    output_b = parse_json_value(value_b)
    records_a = collect_polygon_records(parsed_a)
    records_b = collect_polygon_records(output_b)
    if not records_b:
        return output_b

    if ignore_empty_text_in_a:
        records_a = [
            record
            for record in records_a
            if str(record[0].get("text", "")).strip()
        ]

    boxes_a = [polygon_box(points) for _record, _key, points in records_a]
    boxes_b = [polygon_box(points) for _record, _key, points in records_b]
    effective_size = _effective_image_size(parsed_a, image_size)
    if effective_size is None:
        effective_size = _effective_image_size(output_b, image_size)

    expanded_boxes = expand_subset_bboxes(
        boxes_a,
        boxes_b,
        image_size=effective_size,
        max_expand=max(0.0, float(max_expand)),
        safety_margin=max(0.0, float(safety_margin)),
        coord_base=max(0, int(coord_base or 0)),
        round_output=False,
        clip_to_image=bool(clip_to_image),
        ignore_empty_label_in_a=False,
    )
    _apply_expanded_boxes(records_b, expanded_boxes)
    return output_b


def dumps_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


__all__ = [
    "collect_polygon_records",
    "dumps_json",
    "extend_polygon_json",
    "extend_polygon_subset_json",
    "infer_image_size",
    "parse_json_value",
    "polygon_box",
]
