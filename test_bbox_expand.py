import random
import unittest

from bbox_expand import expand_subset_bboxes, parse_ocr_bboxes, replace_ocr_bboxes


A_BOXES = [
    [30, 12, 515, 50],
    [36, 123, 460, 260],
    [48, 270, 88, 300],
    [92, 280, 309, 304],
    [33, 401, 204, 439],
    [33, 440, 201, 530],
    [33, 540, 201, 625],
    [78, 686, 193, 705],
    [78, 704, 205, 752],
    [52, 773, 220, 792],
    [50, 790, 427, 853],
    [440, 780, 693, 856],
    [732, 820, 900, 847],
    [33, 870, 377, 968],
    [444, 864, 625, 900],
    [437, 899, 970, 968],
    [787, 860, 830, 896],
    [810, 880, 917, 897],
]

B_BOXES = [
    [36, 123, 460, 260],
    [92, 280, 309, 304],
]

A_TEXT = """--- Page 1 ---
<|ref|>text<|/ref|><|det|>[[30, 12, 515, 50]]<|/det|>
ROTAI荣泰 超级88

<|ref|>text<|/ref|><|det|>[[36, 123, 460, 260]]<|/det|>
“按得好
睡得香”

<|ref|>image<|/ref|><|det|>[[48, 270, 88, 300]]<|/det|>

<|ref|>text<|/ref|><|det|>[[92, 280, 309, 304]]<|/det|>
灵犀4D畅感³⁰机芯

<|ref|>text<|/ref|><|det|>[[33, 401, 204, 439]]<|/det|>
整机质保

<|ref|>text<|/ref|><|det|>[[33, 440, 201, 530]]<|/det|>
5年

<|ref|>image<|/ref|><|det|>[[33, 540, 201, 625]]<|/det|>

<|ref|>text<|/ref|><|det|>[[78, 686, 193, 705]]<|/det|>
满5000使用

<|ref|>text<|/ref|><|det|>[[78, 704, 205, 752]]<|/det|>
500

<|ref|>text<|/ref|><|det|>[[52, 773, 220, 792]]<|/det|>
可叠加国补使用

<|ref|>text<|/ref|><|det|>[[50, 790, 427, 853]]<|/det|>
免费用7天 退货“0”运费

<|ref|>text<|/ref|><|det|>[[440, 780, 693, 856]]<|/det|>
免费试用

<|ref|>text<|/ref|><|det|>[[732, 820, 900, 847]]<|/det|>
咨询享立减

<|ref|>text<|/ref|><|det|>[[33, 870, 377, 968]]<|/det|>
前20名补贴
立省2200元

<|ref|>text<|/ref|><|det|>[[444, 864, 625, 900]]<|/det|>
补贴到手价

<|ref|>text<|/ref|><|det|>[[437, 899, 970, 968]]<|/det|>
9499  Vs 11699

<|ref|>image<|/ref|><|det|>[[787, 860, 830, 896]]<|/det|>

<|ref|>text<|/ref|><|det|>[[810, 880, 917, 897]]<|/det|>
日常到手价
"""

B_TEXT = """<|ref|>text<|/ref|><|det|>[[36, 123, 460, 260]]<|/det|>
按得好
睡得香

<|ref|>text<|/ref|><|det|>[[92, 280, 309, 304]]<|/det|>
灵犀4D畅感³⁰机芯
"""


def intersects(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def contains(a, b):
    return a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]


def extra_area_intersects(candidate, base, obstacle):
    strips = []
    if candidate[0] < base[0]:
        strips.append([candidate[0], candidate[1], base[0], candidate[3]])
    if candidate[2] > base[2]:
        strips.append([base[2], candidate[1], candidate[2], candidate[3]])
    if candidate[1] < base[1]:
        strips.append([base[0], candidate[1], base[2], base[1]])
    if candidate[3] > base[3]:
        strips.append([base[0], base[3], base[2], candidate[3]])
    return any(s[0] < s[2] and s[1] < s[3] and intersects(s, obstacle) for s in strips)


def inflate(box, margin, width, height):
    return [
        max(0, box[0] - margin),
        max(0, box[1] - margin),
        min(width, box[2] + margin),
        min(height, box[3] + margin),
    ]


def brute_best_area(base, obstacles, width, height, max_expand, safety_margin=0):
    protected = [inflate(o, safety_margin, width, height) for o in obstacles]
    min_x = max(0, base[0] - max_expand)
    min_y = max(0, base[1] - max_expand)
    max_x = min(width, base[2] + max_expand)
    max_y = min(height, base[3] + max_expand)
    best_area = -1
    for x1 in range(min_x, base[0] + 1):
        for y1 in range(min_y, base[1] + 1):
            for x2 in range(base[2], max_x + 1):
                for y2 in range(base[3], max_y + 1):
                    candidate = [x1, y1, x2, y2]
                    if any(extra_area_intersects(candidate, base, obs) for obs in protected):
                        continue
                    area = (x2 - x1) * (y2 - y1)
                    best_area = max(best_area, area)
    return best_area


class TestBBoxExpand(unittest.TestCase):
    def test_parse_deepseek_text(self):
        self.assertEqual(parse_ocr_bboxes(A_TEXT), [[float(v) for v in box] for box in A_BOXES])
        self.assertEqual(parse_ocr_bboxes(B_TEXT), [[float(v) for v in box] for box in B_BOXES])

    def test_provided_sample_without_safety_margin(self):
        expanded = expand_subset_bboxes(
            A_TEXT,
            B_TEXT,
            image_size=(1000, 1000),
            max_expand=100,
            safety_margin=0,
            coord_base=0,
        )
        self.assertEqual(expanded, [[0, 50, 560, 270], [88, 180, 409, 401]])

        obstacles = [box for box in A_BOXES if box not in B_BOXES]
        for base, candidate in zip(B_BOXES, expanded):
            self.assertTrue(contains(candidate, base))
            for obstacle in obstacles:
                self.assertFalse(extra_area_intersects(candidate, base, obstacle))

    def test_provided_sample_with_safety_margin(self):
        expanded = expand_subset_bboxes(
            A_BOXES,
            B_BOXES,
            image_size=(1000, 1000),
            max_expand=100,
            safety_margin=5,
            coord_base=0,
        )
        self.assertEqual(expanded, [[0, 55, 560, 265], [92, 280, 409, 304]])

    def test_replace_ocr_bboxes_preserves_b_text(self):
        expanded_text = replace_ocr_bboxes(B_TEXT, [[0, 50, 560, 270], [88, 180, 409, 401]])
        self.assertIn("<|det|>[[0, 50, 560, 270]]<|/det|>", expanded_text)
        self.assertIn("<|det|>[[88, 180, 409, 401]]<|/det|>", expanded_text)
        self.assertIn("按得好", expanded_text)
        self.assertIn("灵犀4D", expanded_text)
        self.assertEqual(parse_ocr_bboxes(expanded_text), [[0.0, 50.0, 560.0, 270.0], [88.0, 180.0, 409.0, 401.0]])

    def test_no_obstacles_clips_to_image_bounds(self):
        expanded = expand_subset_bboxes(
            [[10, 10, 30, 30]],
            [[10, 10, 30, 30]],
            image_size=(50, 50),
            max_expand=100,
        )
        self.assertEqual(expanded, [[0, 0, 50, 50]])

    def test_obstacle_and_safety_margin_block_one_side(self):
        expanded = expand_subset_bboxes(
            [[20, 20, 40, 40], [45, 20, 60, 40]],
            [[20, 20, 40, 40]],
            image_size=(100, 100),
            max_expand=10,
            safety_margin=5,
        )
        self.assertEqual(expanded, [[10, 10, 40, 50]])

    def test_selected_b_boxes_do_not_block_each_other(self):
        expanded = expand_subset_bboxes(
            [[100, 100, 120, 120], [125, 100, 145, 120]],
            [[100, 100, 120, 120], [125, 100, 145, 120]],
            image_size=(300, 300),
            max_expand=10,
        )
        self.assertEqual(expanded, [[90, 90, 130, 130], [115, 90, 155, 130]])

    def test_normalized_input_returns_same_coordinate_base_by_default(self):
        expanded = expand_subset_bboxes(
            [[100, 100, 200, 200]],
            [[100, 100, 200, 200]],
            image_size=(200, 100),
            coord_base=1000,
            max_expand=10,
        )
        self.assertEqual(expanded, [[50, 0, 250, 300]])

    def test_normalized_input_can_force_pixel_output(self):
        expanded = expand_subset_bboxes(
            [[100, 100, 200, 200]],
            [[100, 100, 200, 200]],
            image_size=(200, 100),
            coord_base=1000,
            output_coord_base=0,
            max_expand=10,
        )
        self.assertEqual(expanded, [[10, 0, 50, 30]])

    def test_reversed_and_out_of_bounds_input_is_normalized_and_clipped(self):
        expanded = expand_subset_bboxes(
            [[40, 40, 10, 10]],
            [[40, 40, 10, 10]],
            image_size=(30, 30),
            max_expand=5,
        )
        self.assertEqual(expanded, [[5, 5, 30, 30]])

    def test_polygon_points_are_converted_to_enclosing_rect(self):
        polygon = [[(10, 10), (20, 10), (20, 30), (10, 30)]]
        expanded = expand_subset_bboxes(polygon, polygon, image_size=(100, 100), max_expand=5)
        self.assertEqual(expanded, [[5, 5, 25, 35]])

    def test_random_small_cases_match_bruteforce_max_area(self):
        rng = random.Random(7)
        width = height = 9
        for _ in range(50):
            x1 = rng.randint(1, 5)
            y1 = rng.randint(1, 5)
            base = [x1, y1, x1 + rng.randint(1, 2), y1 + rng.randint(1, 2)]
            obstacles = []
            for _ in range(rng.randint(0, 3)):
                ox1 = rng.randint(0, 7)
                oy1 = rng.randint(0, 7)
                obs = [ox1, oy1, min(width, ox1 + rng.randint(1, 2)), min(height, oy1 + rng.randint(1, 2))]
                # Keep the original B feasible; obstacles may touch but do not overlap it.
                if intersects(obs, base):
                    continue
                obstacles.append(obs)

            max_expand = rng.randint(0, 3)
            safety_margin = rng.randint(0, 1)
            expanded = expand_subset_bboxes(
                [base] + obstacles,
                [base],
                image_size=(width, height),
                max_expand=max_expand,
                safety_margin=safety_margin,
            )[0]
            expected_area = brute_best_area(base, obstacles, width, height, max_expand, safety_margin)
            actual_area = (expanded[2] - expanded[0]) * (expanded[3] - expanded[1])
            self.assertEqual(actual_area, expected_area, (base, obstacles, max_expand, safety_margin, expanded))


if __name__ == "__main__":
    unittest.main()
