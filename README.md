# ComfyUI DeepSeek OCR BBox

A small ComfyUI custom node that draws bounding boxes from DeepSeek OCR output, crops each bbox region, and can paste processed crops back to the original coordinates.

DeepSeek OCR coordinates are usually normalized to `0-1000`; keep `coord_base=1000`.
Set `coord_base=0` only when your OCR coordinates are already pixel coordinates.

## GLM Vision BBox nodes

Both new nodes are grouped under the ComfyUI category **`dsocr_bbox/GLM Vision BBox`**.
They use only Python's standard HTTP library, so this API integration adds no package dependency.

### GLM Vision BBox Extractor

Sends exactly one ComfyUI `IMAGE` and an editable multiline prompt to a configurable
OpenAI-compatible vision chat-completions API. Defaults:

- `endpoint`: `https://open.bigmodel.cn/api/paas/v4/chat/completions`
- `model`: `glm-4.6v-flash`
- `api_key`: entered on the node; no key is stored in this repository

The prompt can be entered directly on the node. The node asks GLM for its native
`0-1000` visual-grounding coordinates, validates the response, converts coordinates
to source-image pixels, clamps them to the image, and outputs only normalized JSON:

```json
[
  {
    "desc": "official flagship store",
    "class": "store",
    "bbox": [x1, y1, x2, y2]
  }
]
```

Markdown fences or surrounding prose accidentally emitted by a model are stripped.
Malformed items and invalid/zero-area boxes are omitted. The node rejects image batches
larger than one because its API contract is one image per call.

### GLM BBox JSON To Mask

Consumes the extractor's pixel-coordinate `bbox_json` plus the source `image`, and
outputs a native ComfyUI `MASK` (`float32`, `[batch,height,width]`). `mask_expand` is a
fixed pixel value added independently to all four sides of every bbox and clipped to
the image boundary. Use `0` for no expansion.

Typical workflow:

```text
Load Image -> GLM Vision BBox Extractor -> GLM BBox JSON To Mask
     |                                      ^
     +--------------------------------------+
```

## Modular OCR business-mask pipeline

The business filtering and region expansion are split into independent nodes, so a text or multimodal LLM can be inserted without rerunning OCR:

```text
IMAGE
  -> RapidOCR Detect Text
  -> OCR Business Rule Classifier
  -> OCR Business LLM Review Prompt
  -> your LLM node (optionally also give it IMAGE)
  -> OCR Apply Business Decisions
  -> OCR Business Regions To Mask
```

For deterministic rules only, skip the prompt/apply nodes and connect `classified_json` directly to `OCR Business Regions To Mask`.

The intermediate JSON uses stable detection IDs (`b0_d0`, `b0_d1`, ...). A decision has two independent dimensions:

- `action`: `remove`, `preserve`, `ignore`, or `review`
- `region_policy`: `text`, `box`, `top_banner`, `bottom_banner`, `group_box`, `gift_object`, `explicit_box`, or `none`

This separation is intentional: deciding that text is promotional is a semantic decision, while deciding to reconstruct an entire banner or nearby gift object is a geometric decision.

### RapidOCR Detect Text (PP-OCR)

Detection-only node. It outputs pixel-coordinate OCR JSON and does not make business or mask decisions. The default candidate confidence is `0.50`, intentionally lower than the classifier threshold so a downstream rule/LLM stage can inspect faint text and watermarks. Every detection includes a stable ID, text, score, bbox, polygon, OCR scale, and preprocessing variant.

### OCR Business Rule Classifier

Adds deterministic initial fields to every OCR detection:

- `action`
- `category`
- `reason`
- `region_policy`
- `decision_source=rules`

Known brand/restricted/promotion terms are removed, product specifications and configured functional terms are preserved, and ambiguous content is marked `review`. It outputs both the complete `classified_json` and a smaller `review_items_json`.

### OCR Business LLM Review Prompt

Builds a strict JSON-only prompt from classified detections and a configurable business goal. Connect its output to any LLM node. For better decisions about banners, logos, badges, or nearby gift objects, use a multimodal LLM and provide the original image to that LLM as well.

### OCR Apply Business Decisions

`llm_decisions` is optional. With no LLM connected, the node passes the rule decisions through and still provides the separate remove/preserve JSON outputs; `unresolved_action` controls how `review` items fall back. When connected, it merges LLM JSON back by stable detection ID. The LLM may override actions and region policies, group multiple detections, or provide an exact pixel region:

```json
{
  "decisions": [
    {
      "id": "b0_d12",
      "action": "remove",
      "category": "gift",
      "region_policy": "explicit_box",
      "region": [557, 656, 707, 816],
      "notes": "Remove the gift object next to the gift marker"
    }
  ]
}
```

Markdown JSON fences and surrounding LLM prose are accepted. Items omitted by the LLM can retain their rule result or use a configurable fallback. A multimodal LLM may also return `additional_regions` for visible logos, badges, gifts, or banners that have no OCR anchor; these become synthetic removal decisions with explicit pixel boxes.

### OCR Business Regions To Mask

Turns final decisions into four native ComfyUI masks:

- `mask`: complete removal mask
- `large_block_mask`: whole banners, grouped regions, gifts, and explicit regions
- `detail_mask`: `mask - large_block_mask`, suitable for a second local inpaint pass
- `preserve_mask`: approved OCR text regions, excluding anything covered by removal

It also outputs `regions_json` for auditing and an overlay preview. Region expansion is policy-based rather than hard-coded into semantic classification:

- `text`: padded OCR polygon/rectangle
- `box`: independently expanded bbox; optional LLM `expand` can be a number or `{left,top,right,bottom}`
- `top_banner`: union matching top items, expand, and snap to top/left edges when close
- `bottom_banner`: expand from the earliest matching item to the bottom across full width
- `group_box`: union items sharing `group`, then expand
- `gift_object`: infer a nearby object region from a gift marker
- `explicit_box`: use LLM-supplied pixel `region`
- `none`: do not draw a removal region

### RapidOCR Text Mask (PP-OCR)

Runs local RapidOCR/PP-OCR directly on a ComfyUI `IMAGE` and outputs every detected text region as a native ComfyUI `MASK` (`float32`, shape `[batch, height, width]`). It also returns `detections_json` with recognized text, confidence, polygon, bbox, OCR variant, and active provider.

Inputs:

- `image`: source IMAGE, including image batches
- `speed_profile`: `fast` runs original RGB once; `balanced` adds enhanced grayscale and optional 2x OCR; `thorough` additionally checks inverted grayscale
- `minimum_confidence`: recognition confidence threshold, default `0.72`
- `padding`: outward mask padding in pixels, default `8`
- `mask_shape`: `polygon` preserves rotated OCR boxes; `rectangle` uses enclosing boxes
- `accelerator`: `auto`, `cpu`, or `cuda`; `auto` chooses CUDA only when ONNX Runtime exposes `CUDAExecutionProvider`
- `cpu_threads`: `0` keeps ONNX Runtime defaults; a positive value sets CPU intra-op threads
- `invert_mask`: reverses selected and unselected areas

Outputs:

- `mask`: native ComfyUI MASK; detected text is `1` (white/selected)
- `detections_json`: OCR metadata and the provider actually used

Install dependencies in ComfyUI's Python environment:

```bash
pip install -r custom_nodes/comfyui-dsocr-bbox/requirements.txt
```

For NVIDIA GPU inference, replace the CPU ONNX Runtime package with a CUDA-compatible `onnxruntime-gpu` build. Verify it before selecting `cuda`:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

The output must include `CUDAExecutionProvider`. GPU acceleration applies to the PP-OCR detection, classification, and recognition ONNX models; PIL preprocessing and mask rasterization remain on CPU. For small single images, transfer/session overhead can make CPU as fast as or faster than GPU.

### DeepSeek OCR Draw BBox

Inputs:

- `image`: source IMAGE
- `ocr_result`: DeepSeek OCR text output, STRING socket
- `coord_base`: coordinate base, default `1000`
- `crop_expand`: expand crop area outward by this many pixels on each side, default `0` (clipped to image bounds)

Outputs:

- `image`: source image with bbox drawn
- `crop_images`: all bbox crops as an IMAGE batch
- `crop_info`: JSON metadata containing crop pixel coordinates for every crop; when `crop_expand > 0`, `box` is the expanded crop area and `original_box` keeps the unexpanded bbox

### DeepSeek OCR Draw BBox (Paste Text)

Same as **DeepSeek OCR Draw BBox**, but `ocr_result` is a multiline textbox for manual paste/testing.

### DeepSeek OCR Expand Subset BBox

Expand bboxes from OCR result `B` while treating boxes in OCR result `A - B` as protected regions. Expansion is side-wise: left/right/top/bottom all try to move outward up to `max_expand`, and only protected boxes on the corresponding side limit that side. Boxes inside `B` do not avoid each other. Image borders are allowed as final stopping edges: expanded boxes can touch the border but are clipped inside the image when image size is provided. The output keeps B's original OCR text/ref format and only replaces each `<|det|>...</|det|>` bbox with the expanded bbox.

Inputs:

- `ocr_result_a`: full OCR result A, STRING socket
- `ocr_result_b`: subset OCR result B, STRING socket
- `image_width`, `image_height`: source image size; used to clip the expanded bbox to image bounds and for `coord_base=1000` normalized-coordinate conversion. If both are `0` and optional `image` is connected, the node auto-reads the image size.
- `coord_base`: coordinate base, default `1000`; set to `0` when OCR coordinates are already pixels
- `max_expand`: maximum outward expansion in pixels, default `100`
- `safety_margin`: protected margin around boxes in `A - B`, default `0`
- `ignore_empty_label`: default `true`; removes unlabeled/empty OCR blocks from A before computing protected boxes, including blocks whose label text only contains whitespace/invisible Unicode characters, so blank image detections do not block expansion
- `output_coord_base`: `-1` keeps the same coordinate base as input; `0` forces pixel output; positive values output normalized coordinates with that base

Output:

- `ocr_result`: B OCR result with expanded bbox coordinates

### DeepSeek OCR BBox To Mask

Convert bbox information into a native ComfyUI `MASK` (`float32`, shape `[batch, height, width]`), compatible with the mask output of ComfyUI's built-in **Load Image** node and usable directly by inpainting workflows. The bbox interior is `1` (white / inpaint area), and all other pixels are `0` (black).

Inputs:

- `bbox_info`: bbox/OCR information from a STRING socket. It accepts full DeepSeek OCR text, `<|det|>` blocks, plain lists such as `[[x1, y1, x2, y2], ...]`, dictionaries such as `{"bbox": [x1, y1, x2, y2]}`, and polygons
- `image_width`, `image_height`: output mask dimensions. Leave both at `0` when optional `image` is connected to read its dimensions automatically
- `coord_base`: coordinate base, default `1000`; set to `0` for pixel coordinates
- `invert_mask`: default `false`; when enabled, reverses the mask so bbox regions are `0` (black) and regions outside all bboxes are `1` (white / inpaint area)
- optional `image`: supplies the mask dimensions and batch size; image content is not changed or returned

Output:

- `mask`: standard ComfyUI MASK; white bbox regions are selected for inpainting

## RapidOCR polygon JSON nodes

The polygon JSON workflow is separate from the DeepSeek OCR tag/bbox workflow. All nodes in this set use the `RapidOCR` name and the **RapidOCR** category. They accept a top-level detection list as well as nested RapidOCR output containing `detections` and batch metadata such as `width` and `height`.

Example input:

```json
[
  {"text": "智能四轮", "polygon": [[32.0, 82.0], [198.0, 82.0], [198.0, 123.0], [32.0, 123.0]]},
  {"text": "2025新款", "polygon": [[24.0, 128.0], [216.0, 128.0], [216.0, 178.0], [24.0, 178.0]]}
]
```

### RapidOCR JSON Polygon To Mask

Convert RapidOCR pixel-coordinate polygons into a native ComfyUI `MASK`.

Inputs:

- `json_data`: strict JSON from a STRING socket; each detection needs `polygon` (compatible `points` is also accepted)
- `image_width`, `image_height`: output dimensions; leave both at `0` when optional `image` is connected or the RapidOCR JSON contains `width`/`height` metadata
- `coord_base`: default `0` for RapidOCR pixel coordinates
- `invert_mask`: invert selected and unselected areas
- optional `image`: supplies dimensions and output batch size

Output:

- `mask`: polygon interiors are `1` (white/selected), other pixels are `0`

### RapidOCR JSON Polygon Extend

Expand every polygon independently by up to `expand` pixels on each side. The original polygon is mapped into its expanded enclosing rectangle, preserving rotated or skewed polygon geometry rather than replacing it with a rectangle. Existing `box`/`bbox` metadata is synchronized, while text, IDs, confidence scores, wrappers, and all other JSON fields are preserved.

Image bounds come from `image_width`/`image_height`, optional `image`, or RapidOCR wrapper `width`/`height`. Set `clip_to_image=false` to allow output coordinates outside the image. `image_edge_margin` (default `0`) limits newly expanded polygon edges to remain this many pixels away from the image border. If an original polygon is already closer than the requested margin, it is not shrunk; that side simply remains unchanged. A positive margin requires known image dimensions.

### RapidOCR JSON Polygon A-B Based Extend

Expand polygons in JSON B while treating polygons in `A - B` as protected obstacles. It follows the same independent, side-wise behavior as **DeepSeek OCR Expand Subset BBox**, but consumes and returns RapidOCR polygon JSON:

- B polygons do not block each other
- left/right/top/bottom expand independently up to `max_expand`
- `safety_margin` inflates protected A-B regions
- `image_edge_margin` keeps newly expanded edges the requested pixel distance from image borders without shrinking an original polygon that is already closer
- `ignore_empty_text_in_a` excludes A detections with blank `text` from obstacles
- B's JSON structure and metadata are preserved
- image dimensions can be supplied manually, by optional `image`, or by `width`/`height` in the JSON wrapper

### DeepSeek OCR Expand Subset BBox (Paste Text)

Same as **DeepSeek OCR Expand Subset BBox**, but both OCR inputs are multiline textboxes for manual paste/testing.

### DeepSeek OCR Paste BBox Crops

Paste cropped/processed bbox images back onto the original image using `crop_info`.

Inputs:

- `original_image`: original/source IMAGE
- `crop_images`: crops IMAGE batch, usually from `crop_images` output of Draw BBox or processed by other nodes
- `crop_info`: JSON metadata from Draw BBox `crop_info` output
- `strip_padding`: default `true`; removes padding added to make ComfyUI image batches same size
- `resize_to_bbox`: default `true`; resizes each crop to its original bbox size before pasting
- `feather_radius`: default `0`; feather radius in pixels around the pasted crop edge, `0` disables feathering
- `feather_strength`: default `1.0`; feather blend strength from `0.0` to `1.0`; higher values make the crop edge more transparent and softer

Output:

- `image`: image after pasting crops to original bbox coordinates

## Expected OCR format

```text
<|ref|>已售6万+健腹轮<|/ref|><|det|>[[59, 72, 485, 124]]<|/det|>
<|ref|>更懂你的需求<|/ref|><|det|>[[59, 145, 415, 200]]<|/det|>
<|ref|>数据来源店铺健腹轮累计销量！<|/ref|><|det|>[[54, 217, 329, 240]]<|/det|>
```

The parser also remains compatible with the older layout where `<|ref|>` is a generic type such as `text`/`title` and the recognized text is placed on the following lines.

Supported detection shapes include rectangles like `[[x1, y1, x2, y2]]` and polygon points like
`[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]`.

For polygon detections, cropping uses the polygon's enclosing rectangle, while bbox drawing keeps the polygon outline.
