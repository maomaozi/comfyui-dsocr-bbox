import json
import unittest

from PIL import Image

try:
    from .ocr_business import apply_decisions, build_regions, classify_batches
except Exception:
    from ocr_business import apply_decisions, build_regions, classify_batches


class TestOCRBusinessPipeline(unittest.TestCase):
    def sample(self):
        texts = [
            ("ROTAI荣泰", 0.99, [10, 5, 170, 35]),
            ("智能机芯", 0.98, [20, 150, 130, 180]),
            ("补贴到手价", 0.97, [30, 850, 170, 890]),
            ("送", 0.96, [600, 700, 630, 730]),
            ("普通文案", 0.95, [400, 400, 500, 430]),
        ]
        return json.dumps([{
            "width": 750,
            "height": 1000,
            "detections": [
                {"text": text, "score": score, "box": box,
                 "polygon": [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]}
                for text, score, box in texts
            ],
        }], ensure_ascii=False)

    def test_rule_classification_has_stable_ids(self):
        batches = classify_batches(self.sample(), 0.72, 0.83, True, "", "")
        items = batches[0]["detections"]
        self.assertEqual([x["id"] for x in items], [f"b0_d{i}" for i in range(5)])
        self.assertEqual(items[0]["action"], "remove")
        self.assertEqual(items[0]["region_policy"], "top_banner")
        self.assertEqual(items[1]["action"], "preserve")
        self.assertEqual(items[2]["region_policy"], "bottom_banner")
        self.assertEqual(items[4]["action"], "review")

    def test_llm_override_and_region_expansion(self):
        batches = classify_batches(self.sample(), 0.72, 0.83, True, "", "")
        decisions = {
            "decisions": [
                {"id": "b0_d3", "action": "remove", "category": "gift", "region_policy": "gift_object"},
                {"id": "b0_d4", "action": "remove", "region_policy": "explicit_box", "region": [350, 350, 600, 500]},
            ],
            "additional_regions": [
                {"image_index": 0, "id": "visual_badge", "category": "badge",
                 "region_policy": "explicit_box", "region": [650, 100, 730, 180]}
            ],
        }
        merged = apply_decisions(batches, decisions, "preserve")
        by_id = {x["id"]: x for x in merged[0]["detections"]}
        self.assertEqual(by_id["b0_d3"]["decision_source"], "llm")
        self.assertEqual(by_id["b0_d4"]["region_policy"], "explicit_box")
        self.assertEqual(by_id["visual_badge"]["category"], "badge")

        full, large, detail, preserve, regions = build_regions(
            merged[0]["detections"], 750, 1000, 8, 24, "polygon", True
        )
        self.assertEqual(full.size, (750, 1000))
        self.assertGreater(sum(full.getdata()), 0)
        self.assertGreater(sum(large.getdata()), 0)
        self.assertTrue(any(x["kind"] == "top_banner" for x in regions))
        self.assertTrue(any(x["kind"] == "bottom_banner" for x in regions))
        self.assertTrue(any(x["kind"] == "gift_object" for x in regions))
        self.assertTrue(any(x["kind"] == "explicit_box" for x in regions))
        self.assertLessEqual(sum(detail.getdata()), sum(full.getdata()))
        self.assertGreater(sum(preserve.getdata()), 0)

    def test_markdown_llm_json_is_accepted(self):
        batches = classify_batches(self.sample(), 0.72, 0.83, True, "", "")
        answer = 'analysis\n```json\n{"decisions":[{"id":"b0_d4","action":"preserve","region_policy":"none"}]}\n```'
        merged = apply_decisions(batches, answer, "ignore")
        by_id = {x["id"]: x for x in merged[0]["detections"]}
        self.assertEqual(by_id["b0_d4"]["action"], "preserve")


if __name__ == "__main__":
    unittest.main()
