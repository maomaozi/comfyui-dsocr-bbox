"""Expand selected OCR bounding boxes without entering unselected OCR regions.

The main entry point is :func:`expand_subset_bboxes`.

Problem model
-------------
Given two bbox sets ``A`` and ``B`` where ``B`` is a subset of ``A``, every
box in ``B`` is expanded as much as possible, up to ``max_expand`` pixels per
side.  Boxes in ``A - B`` are treated as protected obstacles.  A configurable
``safety_margin`` is applied by inflating those obstacle boxes before expansion.

Rectangles are treated as half-open pixel boxes: ``[x1, y1, x2, y2)``.  Touching
an obstacle edge is allowed; overlapping it is not.

The algorithm maximizes the area of the expanded rectangle for each selected
box independently.  Other selected ``B`` boxes are *not* obstacles, because the
requirement only forbids entering ``A - B``.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

Number = Union[int, float]
Box = List[Number]

_DET_RE = re.compile(r"<\|det\|>(?P<det>.*?)<\|/det\|>", re.S)
_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_EPS = 1e-9


@dataclass(frozen=True)
class Rect:
    """A half-open axis-aligned rectangle: [x1, y1, x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_box(cls, box: Sequence[Number]) -> "Rect":
        if len(box) < 4:
            raise ValueError(f"bbox needs at least 4 numbers, got {box!r}")
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        return cls(x1, y1, x2, y2)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def has_positive_area(self) -> bool:
        return self.x2 > self.x1 + _EPS and self.y2 > self.y1 + _EPS

    def expanded(self, amount: float) -> "Rect":
        amount = max(0.0, float(amount))
        if amount <= 0:
            return self
        return Rect(self.x1 - amount, self.y1 - amount, self.x2 + amount, self.y2 + amount)

    def clipped(self, bounds: Optional["Rect"]) -> "Rect":
        if bounds is None:
            return self
        x1 = min(max(self.x1, bounds.x1), bounds.x2)
        y1 = min(max(self.y1, bounds.y1), bounds.y2)
        x2 = min(max(self.x2, bounds.x1), bounds.x2)
        y2 = min(max(self.y2, bounds.y1), bounds.y2)
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        return Rect(x1, y1, x2, y2)

    def ensure_non_empty(self, bounds: Optional["Rect"]) -> "Rect":
        """Make a degenerate rect at least one pixel/unit wide when possible."""
        if self.width > _EPS and self.height > _EPS:
            return self
        x1, y1, x2, y2 = self.x1, self.y1, self.x2, self.y2

        if self.width <= _EPS:
            if bounds is not None and bounds.width > 0:
                x1 = min(max(x1, bounds.x1), bounds.x2)
                if x1 >= bounds.x2:
                    x1 = max(bounds.x1, bounds.x2 - 1)
                x2 = min(bounds.x2, x1 + 1)
            else:
                x2 = x1 + 1

        if self.height <= _EPS:
            if bounds is not None and bounds.height > 0:
                y1 = min(max(y1, bounds.y1), bounds.y2)
                if y1 >= bounds.y2:
                    y1 = max(bounds.y1, bounds.y2 - 1)
                y2 = min(bounds.y2, y1 + 1)
            else:
                y2 = y1 + 1

        return Rect(x1, y1, x2, y2)

    def intersects(self, other: "Rect") -> bool:
        return (
            self.x1 < other.x2 - _EPS
            and self.x2 > other.x1 + _EPS
            and self.y1 < other.y2 - _EPS
            and self.y2 > other.y1 + _EPS
        )

    def contains(self, other: "Rect") -> bool:
        return (
            self.x1 <= other.x1 + _EPS
            and self.y1 <= other.y1 + _EPS
            and self.x2 >= other.x2 - _EPS
            and self.y2 >= other.y2 - _EPS
        )

    def to_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_flat_number_list(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) >= 4 and all(_is_number(v) for v in value[:4])


def _is_point_list(value: Any) -> bool:
    # Require coordinate pairs so [[x1, y1, x2, y2], ...] is not mistaken for
    # a polygon when the caller passes a list of boxes.
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return False
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return False
        if not (_is_number(point[0]) and _is_number(point[1])):
            return False
    return True


def _rect_from_points(points: Iterable[Sequence[Number]]) -> List[float]:
    pts = [(float(p[0]), float(p[1])) for p in points]
    if not pts:
        return []
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def _boxes_from_obj(obj: Any) -> List[List[float]]:
    """Normalize Python objects into a list of [x1, y1, x2, y2] boxes."""
    boxes: List[List[float]] = []

    if obj is None:
        return boxes

    if isinstance(obj, dict):
        for key in ("box", "bbox", "rect"):
            if key in obj:
                return _boxes_from_obj(obj[key])
        if "points" in obj:
            rect = _rect_from_points(obj["points"])
            return [rect] if rect else []
        if "det" in obj:
            return _parse_det_text(str(obj["det"]))
        return boxes

    if _is_flat_number_list(obj):
        return [[float(obj[0]), float(obj[1]), float(obj[2]), float(obj[3])]]

    if _is_point_list(obj):
        rect = _rect_from_points(obj)
        return [rect] if rect else []

    if isinstance(obj, (list, tuple)):
        for item in obj:
            boxes.extend(_boxes_from_obj(item))

    return boxes


def _parse_det_text(det_text: str) -> List[List[float]]:
    det_text = (det_text or "").strip()
    if not det_text:
        return []

    try:
        parsed = ast.literal_eval(det_text)
        boxes = _boxes_from_obj(parsed)
        if boxes:
            return boxes
    except Exception:
        pass

    nums = [float(v) for v in _NUMBER_RE.findall(det_text)]
    return [[nums[i], nums[i + 1], nums[i + 2], nums[i + 3]] for i in range(0, len(nums) - 3, 4)]


def parse_ocr_bboxes(value: Any) -> List[List[float]]:
    """Extract OCR bboxes from DeepSeek OCR text or normalize list/dict inputs.

    Supported inputs:
    - DeepSeek OCR text containing ``<|det|>[[x1, y1, x2, y2]]<|/det|>`` tags
    - ``[[x1, y1, x2, y2], ...]``
    - dictionaries containing ``box``/``bbox``/``rect``/``points``
    - polygon point lists, converted to their enclosing rectangle
    """
    if isinstance(value, str):
        matches = list(_DET_RE.finditer(value))
        if matches:
            boxes: List[List[float]] = []
            for match in matches:
                boxes.extend(_parse_det_text(match.group("det")))
            return boxes
        return _parse_det_text(value)

    return _boxes_from_obj(value)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(_as_text(v) for v in value)
    return str(value)


def _format_number(value: Number) -> str:
    number = float(value)
    rounded = round(number)
    if abs(number - rounded) <= 1e-9:
        return str(int(rounded))
    return f"{number:.6f}".rstrip("0").rstrip(".")


def format_det_box(box: Sequence[Number]) -> str:
    """Format one bbox as DeepSeek OCR det text: ``[[x1, y1, x2, y2]]``."""
    if len(box) < 4:
        raise ValueError(f"bbox needs at least 4 numbers, got {box!r}")
    return "[[" + ", ".join(_format_number(v) for v in box[:4]) + "]]"


def format_box_list(boxes: Sequence[Sequence[Number]]) -> str:
    """Format boxes as a plain Python/JSON-like list string."""
    return "[" + ", ".join("[" + ", ".join(_format_number(v) for v in box[:4]) + "]" for box in boxes) + "]"


def replace_ocr_bboxes(ocr_result: Any, boxes: Sequence[Sequence[Number]]) -> str:
    """Replace det bboxes in an OCR result while preserving refs and text.

    Replacement is positional: the first ``<|det|>...<|/det|>`` in
    ``ocr_result`` receives ``boxes[0]``, the second receives ``boxes[1]``, and
    so on.  If the input has no det tags, a plain box-list string is returned.
    """
    text = _as_text(ocr_result)
    idx = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal idx
        if idx >= len(boxes):
            return match.group(0)
        det_text = format_det_box(boxes[idx])
        idx += 1
        return f"<|det|>{det_text}<|/det|>"

    replaced = _DET_RE.sub(repl, text)
    if idx == 0:
        return format_box_list(boxes)
    return replaced


def _make_bounds(image_size: Optional[Tuple[int, int]], coord_base: int) -> Optional[Rect]:
    if image_size is not None:
        width, height = image_size
        width = max(0, int(width))
        height = max(0, int(height))
        return Rect(0.0, 0.0, float(width), float(height))
    if coord_base and coord_base > 0:
        base = float(coord_base)
        return Rect(0.0, 0.0, base, base)
    return None


def _to_internal_rect(
    box: Sequence[Number],
    image_size: Optional[Tuple[int, int]],
    coord_base: int,
    bounds: Optional[Rect],
) -> Rect:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    if coord_base and coord_base > 0 and image_size is not None:
        width, height = image_size
        scale_x = float(width) / float(coord_base) if coord_base else 1.0
        scale_y = float(height) / float(coord_base) if coord_base else 1.0
        x1, x2 = x1 * scale_x, x2 * scale_x
        y1, y2 = y1 * scale_y, y2 * scale_y
    rect = Rect.from_box([x1, y1, x2, y2])
    rect = rect.clipped(bounds).ensure_non_empty(bounds)
    return rect


def _from_internal_rect(
    rect: Rect,
    image_size: Optional[Tuple[int, int]],
    coord_base: int,
    output_coord_base: Optional[int],
    round_output: bool,
) -> Box:
    # Default: return in the same coordinate system as input.
    if output_coord_base is None:
        output_coord_base = coord_base

    x1, y1, x2, y2 = rect.to_list()

    if output_coord_base and output_coord_base > 0 and image_size is not None:
        width, height = image_size
        if width > 0:
            x1 = x1 * float(output_coord_base) / float(width)
            x2 = x2 * float(output_coord_base) / float(width)
        if height > 0:
            y1 = y1 * float(output_coord_base) / float(height)
            y2 = y2 * float(output_coord_base) / float(height)

    if not round_output:
        return [x1, y1, x2, y2]

    # OCR bboxes are usually integer coordinates.  Round to the nearest integer
    # for predictable output.  Internal computations are still done in floats.
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def _rect_close(a: Rect, b: Rect, tolerance: float) -> bool:
    tolerance = max(0.0, float(tolerance))
    return (
        abs(a.x1 - b.x1) <= tolerance
        and abs(a.y1 - b.y1) <= tolerance
        and abs(a.x2 - b.x2) <= tolerance
        and abs(a.y2 - b.y2) <= tolerance
    )


def _subtract_b_from_a(a_rects: Sequence[Rect], b_rects: Sequence[Rect], tolerance: float) -> List[Rect]:
    """Return A - B using bbox equality as a multiset match."""
    used_b = [False] * len(b_rects)
    obstacles: List[Rect] = []

    for a in a_rects:
        matched = False
        for idx, b in enumerate(b_rects):
            if not used_b[idx] and _rect_close(a, b, tolerance):
                used_b[idx] = True
                matched = True
                break
        if not matched:
            obstacles.append(a)

    return obstacles


def _overlap_1d(a1: float, a2: float, b1: float, b2: float) -> bool:
    return a1 < b2 - _EPS and a2 > b1 + _EPS


def _candidate_extra_area_intersects(candidate: Rect, base: Rect, obstacle: Rect) -> bool:
    """Return True if (candidate - base) intersects obstacle.

    Intersections inside ``base`` are ignored because the original bbox is
    mandatory and cannot be shrunk.  This makes the function robust when OCR
    boxes overlap slightly or when a large safety margin reaches the original
    selected bbox.
    """
    if not candidate.intersects(obstacle):
        return False

    strips: List[Rect] = []
    if candidate.x1 < base.x1 - _EPS:
        strips.append(Rect(candidate.x1, candidate.y1, base.x1, candidate.y2))
    if candidate.x2 > base.x2 + _EPS:
        strips.append(Rect(base.x2, candidate.y1, candidate.x2, candidate.y2))
    if candidate.y1 < base.y1 - _EPS:
        strips.append(Rect(base.x1, candidate.y1, base.x2, base.y1))
    if candidate.y2 > base.y2 + _EPS:
        strips.append(Rect(base.x1, base.y2, base.x2, candidate.y2))

    return any(strip.has_positive_area() and strip.intersects(obstacle) for strip in strips)


def _candidate_is_safe(candidate: Rect, base: Rect, obstacles: Sequence[Rect]) -> bool:
    return not any(_candidate_extra_area_intersects(candidate, base, obstacle) for obstacle in obstacles)


def _better_candidate(candidate: Rect, best: Rect, base: Rect) -> bool:
    """Compare candidates by area, then total expansion, then balance."""
    area_delta = candidate.area - best.area
    if area_delta > _EPS:
        return True
    if area_delta < -_EPS:
        return False

    cand_expansions = (
        base.x1 - candidate.x1,
        candidate.x2 - base.x2,
        base.y1 - candidate.y1,
        candidate.y2 - base.y2,
    )
    best_expansions = (
        base.x1 - best.x1,
        best.x2 - base.x2,
        base.y1 - best.y1,
        best.y2 - base.y2,
    )
    cand_sum = sum(cand_expansions)
    best_sum = sum(best_expansions)
    if cand_sum > best_sum + _EPS:
        return True
    if cand_sum < best_sum - _EPS:
        return False

    # Prefer a more balanced expansion if area and total expansion tie.
    cand_min = min(cand_expansions)
    best_min = min(best_expansions)
    if cand_min > best_min + _EPS:
        return True
    if cand_min < best_min - _EPS:
        return False

    # Stable deterministic final tie-breaker: top-left-most, then larger right/bottom.
    return (candidate.y1, candidate.x1, -candidate.y2, -candidate.x2) < (
        best.y1,
        best.x1,
        -best.y2,
        -best.x2,
    )


def _inflate_obstacles(
    obstacles: Sequence[Rect],
    safety_margin: float,
    bounds: Optional[Rect],
    limit: Rect,
) -> List[Rect]:
    inflated: List[Rect] = []
    margin = max(0.0, float(safety_margin))
    for obstacle in obstacles:
        obs = obstacle.expanded(margin).clipped(bounds)
        if obs.has_positive_area() and obs.intersects(limit):
            inflated.append(obs)
    return inflated


def _max_area_expansion_for_one(
    base: Rect,
    obstacles: Sequence[Rect],
    *,
    bounds: Optional[Rect],
    max_expand: float,
    safety_margin: float,
) -> Rect:
    """Find the maximum-area safe rectangle for one selected box.

    Kept for callers that need strict rectangle-obstacle avoidance.  The public
    default below uses edge-wise expansion, because OCR crop expansion usually
    expects all four sides to move outward independently.
    """
    max_expand = max(0.0, float(max_expand))
    base = base.clipped(bounds).ensure_non_empty(bounds)
    limit = base.expanded(max_expand).clipped(bounds)
    limit = Rect(
        min(limit.x1, base.x1),
        min(limit.y1, base.y1),
        max(limit.x2, base.x2),
        max(limit.y2, base.y2),
    ).clipped(bounds)

    protected = _inflate_obstacles(obstacles, safety_margin, bounds, limit)

    top_values = {limit.y1, base.y1}
    bottom_values = {base.y2, limit.y2}
    for obs in protected:
        if limit.y1 - _EPS <= obs.y2 <= base.y1 + _EPS:
            top_values.add(min(max(obs.y2, limit.y1), base.y1))
        if base.y2 - _EPS <= obs.y1 <= limit.y2 + _EPS:
            bottom_values.add(min(max(obs.y1, base.y2), limit.y2))

    tops = sorted(top_values)
    bottoms = sorted(bottom_values)
    best = base

    for top in tops:
        if top > base.y1 + _EPS:
            continue
        for bottom in bottoms:
            if bottom < base.y2 - _EPS or bottom <= top + _EPS:
                continue

            left = limit.x1
            right = limit.x2
            invalid = False

            for obs in protected:
                if not _overlap_1d(top, bottom, obs.y1, obs.y2):
                    continue

                if _overlap_1d(base.x1, base.x2, obs.x1, obs.x2):
                    if top < base.y1 - _EPS and _overlap_1d(top, base.y1, obs.y1, obs.y2):
                        invalid = True
                        break
                    if bottom > base.y2 + _EPS and _overlap_1d(base.y2, bottom, obs.y1, obs.y2):
                        invalid = True
                        break

                if obs.x1 < base.x1 - _EPS:
                    left = max(left, min(obs.x2, base.x1))
                if obs.x2 > base.x2 + _EPS:
                    right = min(right, max(obs.x1, base.x2))

                if left > base.x1 + _EPS or right < base.x2 - _EPS:
                    invalid = True
                    break

            if invalid:
                continue

            candidate = Rect(left, top, right, bottom)
            if not limit.contains(candidate) or not candidate.contains(base):
                continue
            if not _candidate_is_safe(candidate, base, protected):
                continue
            if _better_candidate(candidate, best, base):
                best = candidate

    return best


def _edge_wise_expansion_for_one(
    base: Rect,
    obstacles: Sequence[Rect],
    *,
    bounds: Optional[Rect],
    max_expand: float,
    safety_margin: float,
) -> Rect:
    """Expand the four bbox sides outward independently.

    Only boxes in ``A - B`` constrain the matching side:
    - left/right are constrained by protected boxes that vertically overlap the
      original B box and are to its left/right;
    - top/bottom are constrained by protected boxes that horizontally overlap
      the original B box and are above/below it.

    This intentionally does not use B boxes as obstacles and does not clip to
    image bounds unless ``bounds`` is explicitly supplied by the caller.
    """
    max_expand = max(0.0, float(max_expand))
    base = base.clipped(bounds).ensure_non_empty(bounds) if bounds is not None else base.ensure_non_empty(None)

    left_expand = max_expand
    right_expand = max_expand
    top_expand = max_expand
    bottom_expand = max_expand
    margin = max(0.0, float(safety_margin))

    for obstacle in obstacles:
        original = obstacle.clipped(bounds) if bounds is not None else obstacle
        if not original.has_positive_area():
            continue
        protected = original.expanded(margin)
        if bounds is not None:
            protected = protected.clipped(bounds)
        if not protected.has_positive_area():
            continue

        # Left / right constraints use the original obstacle's vertical overlap
        # and side relation, while the inflated protected box provides the safe
        # stopping edge.
        if _overlap_1d(base.y1, base.y2, original.y1, original.y2):
            if original.x2 <= base.x1 + _EPS:
                left_expand = min(left_expand, max(0.0, base.x1 - protected.x2))
            elif original.x1 < base.x1 < original.x2:
                left_expand = 0.0

            if original.x1 >= base.x2 - _EPS:
                right_expand = min(right_expand, max(0.0, protected.x1 - base.x2))
            elif original.x1 < base.x2 < original.x2:
                right_expand = 0.0

        # Top / bottom constraints use the original obstacle's horizontal
        # overlap and side relation.  A side obstacle does not suppress vertical
        # expansion just because its safety margin reaches a corner.
        if _overlap_1d(base.x1, base.x2, original.x1, original.x2):
            if original.y2 <= base.y1 + _EPS:
                top_expand = min(top_expand, max(0.0, base.y1 - protected.y2))
            elif original.y1 < base.y1 < original.y2:
                top_expand = 0.0

            if original.y1 >= base.y2 - _EPS:
                bottom_expand = min(bottom_expand, max(0.0, protected.y1 - base.y2))
            elif original.y1 < base.y2 < original.y2:
                bottom_expand = 0.0

    candidate = Rect(
        base.x1 - left_expand,
        base.y1 - top_expand,
        base.x2 + right_expand,
        base.y2 + bottom_expand,
    )
    return candidate.clipped(bounds) if bounds is not None else candidate


def expand_subset_bboxes(
    a_boxes: Any,
    b_boxes: Any,
    *,
    image_size: Optional[Tuple[int, int]] = None,
    max_expand: float = 100,
    safety_margin: float = 0,
    coord_base: int = 0,
    output_coord_base: Optional[int] = None,
    match_tolerance: float = 1e-6,
    round_output: bool = True,
    clip_to_image: bool = True,
) -> List[Box]:
    """Expand selected OCR bboxes while avoiding unselected OCR bboxes.

    Parameters
    ----------
    a_boxes:
        The full OCR bbox set A.  Can be a DeepSeek OCR text string, a list of
        boxes, or dictionaries accepted by :func:`parse_ocr_bboxes`.
    b_boxes:
        The selected OCR bbox set B.  B is normally a subset of A.  The output
        order follows this input order.
    image_size:
        Optional ``(width, height)`` in pixels.  When supplied, bbox expansion is
        clipped to ``[0, 0, width, height]`` by default, so boxes can expand up
        to the image border but not outside it.  With ``coord_base > 0``, this
        is also used to convert normalized OCR coordinates to pixels before
        applying ``max_expand`` and ``safety_margin``.
    max_expand:
        Maximum outward expansion in pixels per side.  If ``image_size`` is not
        supplied and ``coord_base`` is used, this value is in coordinate units.
    safety_margin:
        Minimum distance from each unselected ``A - B`` box, implemented by
        inflating protected boxes before expansion.
    coord_base:
        Set to ``1000`` for DeepSeek OCR normalized coordinates.  Set to ``0``
        when coordinates are already pixels.
    output_coord_base:
        ``None`` returns in the same coordinate system as input.  Set ``0`` to
        force pixel output, or set another positive base to return normalized
        coordinates using that base.
    match_tolerance:
        Tolerance used when subtracting B from A by bbox equality.
    round_output:
        Return integer coordinates when true; return floats when false.
    clip_to_image:
        Clip input/output boxes to image bounds when true.  The default is true,
        matching crop-safe behavior: expansion may touch image borders but not
        cross them.  Set false only if out-of-image coordinates are desired.

    Returns
    -------
    list[list[int | float]]
        Expanded B boxes in the same order as ``b_boxes``.
    """
    coord_base = int(coord_base or 0)
    raw_a = parse_ocr_bboxes(a_boxes)
    raw_b = parse_ocr_bboxes(b_boxes)

    bounds = _make_bounds(image_size, coord_base) if clip_to_image else None
    a_rects = [_to_internal_rect(box, image_size, coord_base, bounds) for box in raw_a]
    b_rects = [_to_internal_rect(box, image_size, coord_base, bounds) for box in raw_b]
    obstacles = _subtract_b_from_a(a_rects, b_rects, match_tolerance)

    expanded: List[Box] = []
    for base in b_rects:
        rect = _edge_wise_expansion_for_one(
            base,
            obstacles,
            bounds=bounds,
            max_expand=max_expand,
            safety_margin=safety_margin,
        )
        expanded.append(_from_internal_rect(rect, image_size, coord_base, output_coord_base, round_output))

    return expanded


# Backward-friendly alias with a more descriptive name.
expand_ocr_subset_bboxes = expand_subset_bboxes


__all__ = [
    "Box",
    "Rect",
    "parse_ocr_bboxes",
    "format_det_box",
    "format_box_list",
    "replace_ocr_bboxes",
    "expand_subset_bboxes",
    "expand_ocr_subset_bboxes",
]
