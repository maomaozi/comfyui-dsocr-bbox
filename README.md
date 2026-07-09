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
