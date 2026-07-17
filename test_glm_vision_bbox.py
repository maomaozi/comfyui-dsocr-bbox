import json
import threading
import unittest
from unittest.mock import patch

import torch

from glm_vision_bbox import (
    GLMBBoxJSONProtectedExpand,
    GLMBBoxJSONToMask,
    GLMVisionBBoxDualExtractor,
    parse_bbox_json,
)


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

    def test_dual_extractor_schema(self):
        required = GLMVisionBBoxDualExtractor.INPUT_TYPES()["required"]
        self.assertEqual(
            tuple(required),
            ("image_1", "prompt_1", "image_2", "prompt_2", "endpoint", "model", "api_key"),
        )
        self.assertEqual(required["image_1"], ("IMAGE",))
        self.assertEqual(required["image_2"], ("IMAGE",))
        self.assertTrue(required["prompt_1"][1]["multiline"])
        self.assertTrue(required["prompt_2"][1]["multiline"])
        self.assertEqual(GLMVisionBBoxDualExtractor.RETURN_TYPES, ("STRING", "STRING"))
        self.assertEqual(GLMVisionBBoxDualExtractor.RETURN_NAMES, ("bbox_json_1", "bbox_json_2"))

    def test_dual_extractor_runs_concurrently_and_preserves_output_order(self):
        image_1 = torch.full((1, 2, 3, 3), 0.1, dtype=torch.float32)
        image_2 = torch.full((1, 4, 5, 3), 0.2, dtype=torch.float32)
        barrier = threading.Barrier(2, timeout=2.0)
        release_first = threading.Event()
        calls = {}
        calls_lock = threading.Lock()

        def fake_request(image, prompt, endpoint, model, api_key):
            with calls_lock:
                calls[prompt] = (tuple(image.shape), float(image[0, 0, 0].item()), endpoint, model, api_key)
            barrier.wait()
            if prompt == "first prompt":
                self.assertTrue(release_first.wait(timeout=2.0))
                return "json-one"
            release_first.set()
            return "json-two"

        with patch("glm_vision_bbox.request_glm_bbox", side_effect=fake_request) as request_mock:
            result = GLMVisionBBoxDualExtractor().extract_dual(
                image_1,
                "first prompt",
                image_2,
                "second prompt",
                endpoint="https://example.test/v1",
                model="test-model",
                api_key="secret",
            )

        self.assertEqual(result, ("json-one", "json-two"))
        self.assertEqual(request_mock.call_count, 2)
        self.assertEqual(
            calls["first prompt"],
            ((2, 3, 3), image_1[0, 0, 0, 0].item(), "https://example.test/v1", "test-model", "secret"),
        )
        self.assertEqual(
            calls["second prompt"],
            ((4, 5, 3), image_2[0, 0, 0, 0].item(), "https://example.test/v1", "test-model", "secret"),
        )

    def test_dual_extractor_validates_both_batches_before_requests(self):
        valid = torch.zeros((1, 2, 3, 3), dtype=torch.float32)
        invalid = torch.zeros((2, 2, 3, 3), dtype=torch.float32)
        extractor = GLMVisionBBoxDualExtractor()

        for image_1, image_2, input_name in (
            (invalid, valid, "image_1"),
            (valid, invalid, "image_2"),
        ):
            with self.subTest(input_name=input_name):
                with patch("glm_vision_bbox.request_glm_bbox") as request_mock:
                    with self.assertRaisesRegex(ValueError, input_name):
                        extractor.extract_dual(image_1, "one", image_2, "two")
                    request_mock.assert_not_called()

    def test_dual_extractor_propagates_request_error_without_partial_output(self):
        image = torch.zeros((1, 2, 3, 3), dtype=torch.float32)
        barrier = threading.Barrier(2, timeout=2.0)

        def fake_request(_image, prompt, _endpoint, _model, _api_key):
            barrier.wait()
            if prompt == "one":
                raise RuntimeError("first request failed")
            return "json-two"

        with patch("glm_vision_bbox.request_glm_bbox", side_effect=fake_request) as request_mock:
            with self.assertRaisesRegex(RuntimeError, "first request failed"):
                GLMVisionBBoxDualExtractor().extract_dual(image, "one", image, "two")
        self.assertEqual(request_mock.call_count, 2)

    def test_dual_extractor_is_registered(self):
        from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

        self.assertIs(NODE_CLASS_MAPPINGS["GLMVisionBBoxDualExtractor"], GLMVisionBBoxDualExtractor)
        self.assertEqual(
            NODE_DISPLAY_NAME_MAPPINGS["GLMVisionBBoxDualExtractor"],
            "GLM Vision BBox Dual Extractor",
        )

    def test_protected_expand_schema(self):
        required = GLMBBoxJSONProtectedExpand.INPUT_TYPES()["required"]
        self.assertEqual(
            tuple(required),
            (
                "image",
                "bbox_json_a",
                "bbox_json_b",
                "horizontal_expand",
                "vertical_expand",
                "safety_margin",
                "coord_base",
            ),
        )
        self.assertEqual(required["coord_base"][1]["default"], 0)
        self.assertEqual(GLMBBoxJSONProtectedExpand.RETURN_TYPES, ("STRING",))
        self.assertEqual(GLMBBoxJSONProtectedExpand.RETURN_NAMES, ("bbox_json",))

    def test_protected_expand_uses_percentages_and_splits_with_metadata(self):
        image = torch.zeros((1, 100, 200, 3), dtype=torch.float32)
        protected = '[{"desc":"keep","class":"protected","bbox":[95,0,105,100]}]'
        targets = '[{"desc":"banner","class":"remove","bbox":[80,40,120,60]}]'
        (result_json,) = GLMBBoxJSONProtectedExpand().expand_protected(
            image,
            protected,
            targets,
            horizontal_expand=10.0,
            vertical_expand=10.0,
        )
        result = json.loads(result_json)
        self.assertEqual(
            result,
            [
                {"desc": "banner", "class": "remove", "bbox": [60, 30, 95, 70]},
                {"desc": "banner", "class": "remove", "bbox": [105, 30, 140, 70]},
            ],
        )

    def test_protected_expand_removes_existing_overlap_and_covered_target(self):
        image = torch.zeros((1, 100, 100, 3), dtype=torch.float32)
        node = GLMBBoxJSONProtectedExpand()
        protected = '[{"bbox":[40,20,60,80]}]'
        targets = '[{"desc":"b","class":"target","bbox":[30,30,70,70]}]'
        (partial_json,) = node.expand_protected(image, protected, targets)
        self.assertEqual(
            json.loads(partial_json),
            [
                {"desc": "b", "class": "target", "bbox": [30, 30, 40, 70]},
                {"desc": "b", "class": "target", "bbox": [60, 30, 70, 70]},
            ],
        )
        (covered_json,) = node.expand_protected(image, '[{"bbox":[0,0,100,100]}]', targets)
        self.assertEqual(json.loads(covered_json), [])

    def test_protected_expand_supports_glm_1000_coordinates_and_aliases(self):
        image = torch.zeros((1, 100, 200, 3), dtype=torch.float32)
        protected = 'result:\n```json\n[{"description":"a","category":"keep","box":[450,0,550,1000]}]\n```'
        targets = '{"results":[{"text":"b","type":"target","rect":[400,400,600,600]}]}'
        (result_json,) = GLMBBoxJSONProtectedExpand().expand_protected(
            image, protected, targets, coord_base=1000
        )
        self.assertEqual(
            json.loads(result_json),
            [
                {"desc": "b", "class": "target", "bbox": [80, 40, 90, 60]},
                {"desc": "b", "class": "target", "bbox": [110, 40, 120, 60]},
            ],
        )

    def test_protected_expand_handles_empty_inputs_and_rejects_image_batch(self):
        node = GLMBBoxJSONProtectedExpand()
        image = torch.zeros((1, 20, 30, 3), dtype=torch.float32)
        (empty_b,) = node.expand_protected(image, '[]', '[]', horizontal_expand=10)
        self.assertEqual(json.loads(empty_b), [])
        (empty_a,) = node.expand_protected(
            image,
            '[]',
            '[{"desc":"b","class":"target","bbox":[10,5,20,15]}]',
            horizontal_expand=10,
            vertical_expand=10,
        )
        self.assertEqual(
            json.loads(empty_a),
            [{"desc": "b", "class": "target", "bbox": [7, 3, 23, 17]}],
        )
        with self.assertRaisesRegex(ValueError, "exactly one image"):
            node.expand_protected(torch.zeros((2, 20, 30, 3)), '[]', '[]')

    def test_protected_expand_is_registered(self):
        from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

        self.assertIs(NODE_CLASS_MAPPINGS["GLMBBoxJSONProtectedExpand"], GLMBBoxJSONProtectedExpand)
        self.assertEqual(
            NODE_DISPLAY_NAME_MAPPINGS["GLMBBoxJSONProtectedExpand"],
            "GLM BBox JSON Protected Expand",
        )

    def test_mask_inputs_use_coord_base_and_separate_percentage_expansion(self):
        required = GLMBBoxJSONToMask.INPUT_TYPES()["required"]
        coord_input = required["coord_base"]
        self.assertEqual(coord_input[0], "INT")
        self.assertEqual(coord_input[1]["default"], 1000)
        self.assertEqual(coord_input[1]["min"], 0)

        for name in ("horizontal_expand", "vertical_expand"):
            expand_input = required[name]
            self.assertEqual(expand_input[0], "FLOAT")
            self.assertEqual(expand_input[1]["default"], 0.0)
            self.assertEqual(expand_input[1]["step"], 0.1)

    def test_mask_converts_1000_coordinates_to_pixels(self):
        image = torch.zeros((1, 80, 100, 3), dtype=torch.float32)
        bbox_json = '[{"desc":"店铺","class":"店铺","bbox":[100,250,300,500]}]'
        (mask,) = GLMBBoxJSONToMask().json_to_mask(image, bbox_json, horizontal_expand=0, vertical_expand=0)
        self.assertEqual(mask[0, 20, 10].item(), 1.0)
        self.assertEqual(mask[0, 39, 29].item(), 1.0)
        self.assertEqual(mask[0, 19, 10].item(), 0.0)
        self.assertEqual(mask[0, 40, 29].item(), 0.0)

    def test_mask_expansion_uses_percentage_of_image_dimensions(self):
        bbox_json = '[{"desc":"店铺","class":"店铺","bbox":[200,300,400,500]}]'

        small_image = torch.zeros((1, 80, 100, 3), dtype=torch.float32)
        (small_mask,) = GLMBBoxJSONToMask().json_to_mask(
            small_image, bbox_json, horizontal_expand=10.0, vertical_expand=10.0, coord_base=1000
        )
        self.assertEqual(small_mask[0, 16, 10].item(), 1.0)
        self.assertEqual(small_mask[0, 47, 49].item(), 1.0)
        self.assertEqual(small_mask[0, 15, 10].item(), 0.0)
        self.assertEqual(small_mask[0, 48, 49].item(), 0.0)

        large_image = torch.zeros((1, 160, 200, 3), dtype=torch.float32)
        (large_mask,) = GLMBBoxJSONToMask().json_to_mask(
            large_image, bbox_json, horizontal_expand=10.0, vertical_expand=10.0, coord_base=1000
        )
        self.assertEqual(large_mask[0, 32, 20].item(), 1.0)
        self.assertEqual(large_mask[0, 95, 99].item(), 1.0)
        self.assertEqual(large_mask[0, 31, 20].item(), 0.0)
        self.assertEqual(large_mask[0, 96, 99].item(), 0.0)

    def test_mask_percentage_expansion_is_independent_of_coord_base(self):
        image = torch.zeros((1, 30, 40, 3), dtype=torch.float32)
        bbox_json = '[{"desc":"店铺","class":"店铺","bbox":[10,10,20,20]}]'
        (mask,) = GLMBBoxJSONToMask().json_to_mask(
            image, bbox_json, horizontal_expand=10.0, vertical_expand=10.0, coord_base=0
        )
        self.assertEqual(tuple(mask.shape), (1, 30, 40))
        self.assertEqual(mask.dtype, torch.float32)
        self.assertEqual(mask[0, 7, 6].item(), 1.0)
        self.assertEqual(mask[0, 22, 23].item(), 1.0)
        self.assertEqual(mask[0, 6, 6].item(), 0.0)
        self.assertEqual(mask[0, 23, 23].item(), 0.0)

    def test_horizontal_and_vertical_expansion_are_independent(self):
        image = torch.zeros((1, 100, 200, 3), dtype=torch.float32)
        bbox_json = '[{"bbox":[50,30,100,60]}]'
        (mask,) = GLMBBoxJSONToMask().json_to_mask(
            image, bbox_json, horizontal_expand=10.0, vertical_expand=5.0, coord_base=0
        )
        self.assertEqual(mask[0, 25, 30].item(), 1.0)
        self.assertEqual(mask[0, 64, 119].item(), 1.0)
        self.assertEqual(mask[0, 24, 30].item(), 0.0)
        self.assertEqual(mask[0, 25, 29].item(), 0.0)
        self.assertEqual(mask[0, 65, 119].item(), 0.0)
        self.assertEqual(mask[0, 64, 120].item(), 0.0)

    def test_mask_expansion_accepts_fractional_percentages(self):
        image = torch.zeros((1, 200, 400, 3), dtype=torch.float32)
        bbox_json = '[{"bbox":[100,50,200,100]}]'
        (mask,) = GLMBBoxJSONToMask().json_to_mask(
            image, bbox_json, horizontal_expand=0.5, vertical_expand=0.5, coord_base=0
        )
        self.assertEqual(mask[0, 49, 98].item(), 1.0)
        self.assertEqual(mask[0, 100, 201].item(), 1.0)
        self.assertEqual(mask[0, 48, 98].item(), 0.0)
        self.assertEqual(mask[0, 101, 201].item(), 0.0)

    def test_mask_repeats_for_image_batch(self):
        image = torch.zeros((2, 10, 12, 3), dtype=torch.float32)
        (mask,) = GLMBBoxJSONToMask().json_to_mask(image, '[]', horizontal_expand=0, vertical_expand=0)
        self.assertEqual(tuple(mask.shape), (2, 10, 12))
        self.assertTrue(torch.equal(mask[0], mask[1]))


if __name__ == "__main__":
    unittest.main()
