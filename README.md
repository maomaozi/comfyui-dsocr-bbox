# ComfyUI DeepSeek OCR BBox

A small ComfyUI custom node that draws bounding boxes from DeepSeek OCR output.

DeepSeek OCR coordinates are usually normalized to `0-1000`; keep `coord_base=1000`.
Set `coord_base=0` only when your OCR coordinates are already pixel coordinates.

## Nodes

- **DeepSeek OCR Draw BBox**: `ocr_result` is a STRING input socket, suitable for connecting an OCR node.
- **DeepSeek OCR Draw BBox (Paste Text)**: `ocr_result` is a multiline textbox, suitable for manual paste/testing.

## Expected OCR format

```text
<|ref|>title<|/ref|><|det|>[[12, 0, 386, 45]]<|/det|>
# BENBO本博
```

Supported detection shapes include rectangles like `[[x1, y1, x2, y2]]` and polygon points like
`[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]`.
