import json
import unittest

import torch

from glm_vision_bbox import GLMBBoxJSONToMask, parse_bbox_json


class TestGLMVisionBBox(unittest.TestCase):
    def test_parse_plain_json_and_clamp(self):
        source = json.dumps(
            [
                {"desc": "官方旗舰店", "class": "店铺", "bbox": [-2, 3, 110, 90]},
                {"description": "品牌标志", "category": "品牌", "box": [20, 20, 40, 45]},
            ],
            ensure_ascii=False,
        )
        result = parse_bbox_json(source, width=100, height=80)
        self.assertEqual(
            result,
            [
                {"desc": "官方旗舰店", "class": "店铺", "bbox": [0, 3, 100, 80]},
                {"desc": "品牌标志", "class": "品牌", "bbox": [20, 20, 40, 45]},
            ],
        )

    def test_parse_fenced_json_with_surrounding_text(self):
        source = "result:\n```json\n[{\"desc\":\"logo\",\"class\":\"logo\",\"bbox\":[1,2,9,10]}]\n```"
        result = parse_bbox_json(source, width=20, height=20)
        self.assertEqual(result[0]["bbox"], [1, 2, 9, 10])

    def test_parse_python_literal_fallback(self):
        source = "[{'desc':'logo','class':'logo','bbox':[1,2,9,10]}]"
        result = parse_bbox_json(source, width=20, height=20)
        self.assertEqual(result[0], {"desc": "logo", "class": "logo", "bbox": [1, 2, 9, 10]})

    def test_parse_glm_1000_coordinates_to_pixels(self):
        source = '[{"desc":"店铺","class":"店铺","bbox":[802,696,938,811]}]'
        result = parse_bbox_json(source, width=472, height=466, coord_base=1000)
        self.assertEqual(result[0]["bbox"], [379, 324, 443, 378])

    def test_mask_coord_base_defaults_to_1000(self):
        coord_input = GLMBBoxJSONToMask.INPUT_TYPES()["required"]["coord_base"]
        self.assertEqual(coord_input[0], "INT")
        self.assertEqual(coord_input[1]["default"], 1000)
        self.assertEqual(coord_input[1]["min"], 0)

    def test_mask_converts_1000_coordinates_to_pixels(self):
        image = torch.zeros((1, 80, 100, 3), dtype=torch.float32)
        bbox_json = '[{"desc":"店铺","class":"店铺","bbox":[100,250,300,500]}]'
        (mask,) = GLMBBoxJSONToMask().json_to_mask(image, bbox_json, mask_expand=0)
        self.assertEqual(mask[0, 20, 10].item(), 1.0)
        self.assertEqual(mask[0, 39, 29].item(), 1.0)
        self.assertEqual(mask[0, 19, 10].item(), 0.0)
        self.assertEqual(mask[0, 40, 29].item(), 0.0)

    def test_mask_uses_pixel_coordinates_and_fixed_expansion(self):
        image = torch.zeros((1, 30, 40, 3), dtype=torch.float32)
        bbox_json = '[{"desc":"店铺","class":"店铺","bbox":[10,10,20,20]}]'
        (mask,) = GLMBBoxJSONToMask().json_to_mask(
            image, bbox_json, mask_expand=3, coord_base=0
        )
        self.assertEqual(tuple(mask.shape), (1, 30, 40))
        self.assertEqual(mask.dtype, torch.float32)
        self.assertEqual(mask[0, 7, 7].item(), 1.0)
        self.assertEqual(mask[0, 22, 22].item(), 1.0)
        self.assertEqual(mask[0, 6, 7].item(), 0.0)
        self.assertEqual(mask[0, 23, 22].item(), 0.0)

    def test_mask_repeats_for_image_batch(self):
        image = torch.zeros((2, 10, 12, 3), dtype=torch.float32)
        (mask,) = GLMBBoxJSONToMask().json_to_mask(image, '[]', mask_expand=0)
        self.assertEqual(tuple(mask.shape), (2, 10, 12))
        self.assertTrue(torch.equal(mask[0], mask[1]))


if __name__ == "__main__":
    unittest.main()
