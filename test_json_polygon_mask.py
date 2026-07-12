import json
import unittest

import torch

from nodes import DeepSeekOCRJSONPolygonToMask, parse_deepseek_ocr


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

        (mask,) = DeepSeekOCRJSONPolygonToMask().json_to_mask(
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
        (mask,) = DeepSeekOCRJSONPolygonToMask().json_to_mask(
            json.dumps(payload), image_width=100, image_height=100
        )

        self.assertEqual(mask[0, 50, 50].item(), 1.0)
        self.assertEqual(mask[0, 15, 15].item(), 0.0)

    def test_image_supplies_dimensions_and_batch_size(self):
        image = torch.zeros((2, 60, 80, 3), dtype=torch.float32)
        payload = {"detections": [{"polygon": [[1, 1], [20, 1], [20, 20], [1, 20]]}]}
        (mask,) = DeepSeekOCRJSONPolygonToMask().json_to_mask(
            json.dumps(payload), image=image, invert_mask=True
        )

        self.assertEqual(tuple(mask.shape), (2, 60, 80))
        self.assertEqual(mask[0, 10, 10].item(), 0.0)
        self.assertEqual(mask[1, 40, 40].item(), 1.0)
        self.assertTrue(torch.equal(mask[0], mask[1]))

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
            DeepSeekOCRJSONPolygonToMask().json_to_mask(
                "not json", image_width=10, image_height=10
            )


if __name__ == "__main__":
    unittest.main()
