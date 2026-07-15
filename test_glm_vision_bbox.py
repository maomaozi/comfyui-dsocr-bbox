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

    def test_mask_inputs_use_coord_base_and_percentage_expansion(self):
        required = GLMBBoxJSONToMask.INPUT_TYPES()["required"]
        coord_input = required["coord_base"]
        self.assertEqual(coord_input[0], "INT")
        self.assertEqual(coord_input[1]["default"], 1000)
        self.assertEqual(coord_input[1]["min"], 0)

        expand_input = required["mask_expand"]
        self.assertEqual(expand_input[0], "FLOAT")
        self.assertEqual(expand_input[1]["default"], 0.0)
        self.assertEqual(expand_input[1]["step"], 0.1)

    def test_mask_converts_1000_coordinates_to_pixels(self):
        image = torch.zeros((1, 80, 100, 3), dtype=torch.float32)
        bbox_json = '[{"desc":"店铺","class":"店铺","bbox":[100,250,300,500]}]'
        (mask,) = GLMBBoxJSONToMask().json_to_mask(image, bbox_json, mask_expand=0)
        self.assertEqual(mask[0, 20, 10].item(), 1.0)
        self.assertEqual(mask[0, 39, 29].item(), 1.0)
        self.assertEqual(mask[0, 19, 10].item(), 0.0)
        self.assertEqual(mask[0, 40, 29].item(), 0.0)

    def test_mask_expansion_uses_percentage_of_image_dimensions(self):
        bbox_json = '[{"desc":"店铺","class":"店铺","bbox":[200,300,400,500]}]'

        small_image = torch.zeros((1, 80, 100, 3), dtype=torch.float32)
        (small_mask,) = GLMBBoxJSONToMask().json_to_mask(
            small_image, bbox_json, mask_expand=10.0, coord_base=1000
        )
        self.assertEqual(small_mask[0, 16, 10].item(), 1.0)
        self.assertEqual(small_mask[0, 47, 49].item(), 1.0)
        self.assertEqual(small_mask[0, 15, 10].item(), 0.0)
        self.assertEqual(small_mask[0, 48, 49].item(), 0.0)

        large_image = torch.zeros((1, 160, 200, 3), dtype=torch.float32)
        (large_mask,) = GLMBBoxJSONToMask().json_to_mask(
            large_image, bbox_json, mask_expand=10.0, coord_base=1000
        )
        self.assertEqual(large_mask[0, 32, 20].item(), 1.0)
        self.assertEqual(large_mask[0, 95, 99].item(), 1.0)
        self.assertEqual(large_mask[0, 31, 20].item(), 0.0)
        self.assertEqual(large_mask[0, 96, 99].item(), 0.0)

    def test_mask_percentage_expansion_is_independent_of_coord_base(self):
        image = torch.zeros((1, 30, 40, 3), dtype=torch.float32)
        bbox_json = '[{"desc":"店铺","class":"店铺","bbox":[10,10,20,20]}]'
        (mask,) = GLMBBoxJSONToMask().json_to_mask(
            image, bbox_json, mask_expand=10.0, coord_base=0
        )
        self.assertEqual(tuple(mask.shape), (1, 30, 40))
        self.assertEqual(mask.dtype, torch.float32)
        self.assertEqual(mask[0, 7, 6].item(), 1.0)
        self.assertEqual(mask[0, 22, 23].item(), 1.0)
        self.assertEqual(mask[0, 6, 6].item(), 0.0)
        self.assertEqual(mask[0, 23, 23].item(), 0.0)

    def test_mask_expansion_accepts_fractional_percentages(self):
        image = torch.zeros((1, 200, 400, 3), dtype=torch.float32)
        bbox_json = '[{"bbox":[100,50,200,100]}]'
        (mask,) = GLMBBoxJSONToMask().json_to_mask(
            image, bbox_json, mask_expand=0.5, coord_base=0
        )
        self.assertEqual(mask[0, 49, 98].item(), 1.0)
        self.assertEqual(mask[0, 100, 201].item(), 1.0)
        self.assertEqual(mask[0, 48, 98].item(), 0.0)
        self.assertEqual(mask[0, 101, 201].item(), 0.0)

    def test_mask_repeats_for_image_batch(self):
        image = torch.zeros((2, 10, 12, 3), dtype=torch.float32)
        (mask,) = GLMBBoxJSONToMask().json_to_mask(image, '[]', mask_expand=0)
        self.assertEqual(tuple(mask.shape), (2, 10, 12))
        self.assertTrue(torch.equal(mask[0], mask[1]))


if __name__ == "__main__":
    unittest.main()
