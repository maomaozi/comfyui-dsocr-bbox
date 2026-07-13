import json
import unittest

import torch

from nodes import (
    RapidOCRJSONPolygonABExtend,
    RapidOCRJSONPolygonExtend,
    RapidOCRJSONPolygonToMask,
    parse_deepseek_ocr,
)


class TestJSONPolygonToMask(unittest.TestCase):
    def test_sample_json_rasterizes_multiple_polygons(self):
        payload = [
            {
                "text": "first",
                "polygon": [[32.0, 82.0], [198.0, 82.0], [198.0, 123.0], [32.0, 123.0]],
            },
            {
                "text": "second",
                "polygon": [[24.0, 128.0], [216.0, 128.0], [216.0, 178.0], [24.0, 178.0]],
            },
        ]

        (mask,) = RapidOCRJSONPolygonToMask().json_to_mask(
            json.dumps(payload), image_width=240, image_height=200
        )

        self.assertEqual(mask.dtype, torch.float32)
        self.assertEqual(tuple(mask.shape), (1, 200, 240))
        self.assertEqual(mask[0, 100, 100].item(), 1.0)
        self.assertEqual(mask[0, 150, 100].item(), 1.0)
        self.assertEqual(mask[0, 125, 100].item(), 0.0)
        self.assertEqual(mask[0, 20, 20].item(), 0.0)

    def test_polygon_is_preferred_over_enclosing_bbox(self):
        payload = [{
            "bbox": [10, 10, 90, 90],
            "polygon": [[50, 10], [90, 50], [50, 90], [10, 50]],
        }]
        (mask,) = RapidOCRJSONPolygonToMask().json_to_mask(
            json.dumps(payload), image_width=100, image_height=100
        )

        self.assertEqual(mask[0, 50, 50].item(), 1.0)
        self.assertEqual(mask[0, 15, 15].item(), 0.0)

    def test_image_supplies_dimensions_and_batch_size(self):
        image = torch.zeros((2, 60, 80, 3), dtype=torch.float32)
        payload = {"detections": [{"polygon": [[1, 1], [20, 1], [20, 20], [1, 20]]}]}
        (mask,) = RapidOCRJSONPolygonToMask().json_to_mask(
            json.dumps(payload), image=image, invert_mask=True
        )

        self.assertEqual(tuple(mask.shape), (2, 60, 80))
        self.assertEqual(mask[0, 10, 10].item(), 0.0)
        self.assertEqual(mask[1, 40, 40].item(), 1.0)
        self.assertTrue(torch.equal(mask[0], mask[1]))

    def test_mask_infers_dimensions_from_rapidocr_wrapper(self):
        payload = [{
            "width": 80,
            "height": 60,
            "detections": [{"polygon": [[1, 1], [20, 1], [20, 20], [1, 20]]}],
        }]
        (mask,) = RapidOCRJSONPolygonToMask().json_to_mask(json.dumps(payload))
        self.assertEqual(tuple(mask.shape), (1, 60, 80))
        self.assertEqual(mask[0, 10, 10].item(), 1.0)

    def test_general_parser_accepts_polygon_field(self):
        annotations = parse_deepseek_ocr(
            json.dumps([{"text": "hello", "polygon": [[1, 2], [3, 2], [3, 4], [1, 4]]}])
        )
        self.assertEqual(len(annotations), 1)
        self.assertEqual(annotations[0]["type"], "polygon")
        self.assertEqual(
            annotations[0]["points"],
            [(1.0, 2.0), (3.0, 2.0), (3.0, 4.0), (1.0, 4.0)],
        )

    def test_invalid_json_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            RapidOCRJSONPolygonToMask().json_to_mask(
                "not json", image_width=10, image_height=10
            )

    def test_independent_extend_preserves_metadata_and_polygon_shape(self):
        payload = [{
            "id": "b0_d0",
            "text": "hello",
            "score": 0.9,
            "box": [10, 10, 30, 30],
            "polygon": [[20, 10], [30, 20], [20, 30], [10, 20]],
        }]
        (output_json,) = RapidOCRJSONPolygonExtend().extend_polygons(
            json.dumps(payload), expand=5, image_width=100, image_height=100
        )
        output = json.loads(output_json)

        self.assertEqual(output[0]["id"], "b0_d0")
        self.assertEqual(output[0]["text"], "hello")
        self.assertEqual(output[0]["score"], 0.9)
        self.assertEqual(output[0]["box"], [5, 5, 35, 35])
        self.assertEqual(
            output[0]["polygon"],
            [[20, 5], [35, 20], [20, 35], [5, 20]],
        )

    def test_independent_extend_infers_wrapper_dimensions(self):
        payload = [{
            "width": 40,
            "height": 30,
            "detections": [{
                "text": "edge",
                "polygon": [[2, 3], [12, 3], [12, 13], [2, 13]],
            }],
        }]
        (output_json,) = RapidOCRJSONPolygonExtend().extend_polygons(
            json.dumps(payload), expand=10
        )
        polygon = json.loads(output_json)[0]["detections"][0]["polygon"]
        self.assertEqual(polygon, [[0, 0], [22, 0], [22, 23], [0, 23]])

    def test_ab_extend_avoids_a_minus_b_and_preserves_b_json(self):
        selected = {
            "id": "selected",
            "text": "selected",
            "box": [20, 20, 40, 40],
            "polygon": [[20, 20], [40, 20], [40, 40], [20, 40]],
        }
        obstacle = {
            "id": "obstacle",
            "text": "protected",
            "box": [45, 20, 60, 40],
            "polygon": [[45, 20], [60, 20], [60, 40], [45, 40]],
        }
        a_payload = [{"width": 100, "height": 100, "detections": [selected, obstacle]}]
        b_payload = [selected]

        (output_json,) = RapidOCRJSONPolygonABExtend().extend_subset_polygons(
            json.dumps(a_payload),
            json.dumps(b_payload),
            max_expand=10,
            safety_margin=5,
        )
        output = json.loads(output_json)

        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]["id"], "selected")
        self.assertEqual(output[0]["box"], [10, 10, 40, 50])
        self.assertEqual(
            output[0]["polygon"],
            [[10, 10], [40, 10], [40, 50], [10, 50]],
        )

    def test_ab_extend_can_ignore_empty_text_obstacle(self):
        selected = {
            "text": "selected",
            "polygon": [[20, 20], [40, 20], [40, 40], [20, 40]],
        }
        blank = {
            "text": "  ",
            "polygon": [[45, 20], [60, 20], [60, 40], [45, 40]],
        }
        (ignored_json,) = RapidOCRJSONPolygonABExtend().extend_subset_polygons(
            json.dumps([selected, blank]), json.dumps([selected]), max_expand=10
        )
        (kept_json,) = RapidOCRJSONPolygonABExtend().extend_subset_polygons(
            json.dumps([selected, blank]),
            json.dumps([selected]),
            max_expand=10,
            ignore_empty_text_in_a=False,
        )
        ignored_polygon = json.loads(ignored_json)[0]["polygon"]
        kept_polygon = json.loads(kept_json)[0]["polygon"]
        self.assertEqual(ignored_polygon[1][0], 50)
        self.assertEqual(kept_polygon[1][0], 45)


if __name__ == "__main__":
    unittest.main()
