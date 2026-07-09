# ComfyUI DeepSeek OCR BBox

A small ComfyUI custom node that draws bounding boxes from DeepSeek OCR output, crops each bbox region, and can paste processed crops back to the original coordinates.

DeepSeek OCR coordinates are usually normalized to `0-1000`; keep `coord_base=1000`.
Set `coord_base=0` only when your OCR coordinates are already pixel coordinates.

## Nodes

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

Output:

- `image`: image after pasting crops to original bbox coordinates

## Expected OCR format

```text
<|ref|>title<|/ref|><|det|>[[12, 0, 386, 45]]<|/det|>
# BENBO本博
```

Supported detection shapes include rectangles like `[[x1, y1, x2, y2]]` and polygon points like
`[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]`.

For polygon detections, cropping uses the polygon's enclosing rectangle, while bbox drawing keeps the polygon outline.
