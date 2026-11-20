# `09-detections`

A detector gives boxes as centre and size in fractions of its input, each
with a class and a score. This example turns twelve such outputs into
COCO's detections: the best eight, in the photograph's pixels, one box
per object and class.

```bash
dune exec ./main.exe
```

- The outputs are written in the program, around the cat's eyes and
  nose, with the near-duplicates a detector gives, a shifted eye box and
  an eye box over the nose.
- `Nx.top_k` keeps the eight best scores, best first.
- `Mynd.Box.of_cxcywh` gives corners, and a multiply by `[w h w h]` the
  photograph's pixels, since a fraction of the input's width is the same
  fraction of the photograph's.
- `Mynd.Box.iou` gives each pair's intersection over union.
- `Mynd.Box.nms ~classes` keeps, by descending score, each box that no
  kept box of its class overlaps by more than an IoU of 0.5. The shifted
  eye box overlaps the eye by 0.39 and stays; the eye box over the nose
  is of another class than the nose's and stays.
- `Mynd.Box.to_xywh` gives COCO's `bbox`, `[x, y, w, h]`.

It prints the indices of the best eight, the IoUs of the eye boxes, the
boxes kept, and one COCO detection per kept box. `detections.png` draws
the kept boxes over the photograph.

Margins: every IoU between boxes of one class lies at least `0.030` from
the threshold 0.5 and `1e-3` from a change of its second decimal; the
eighth and ninth scores are 0.35 and 0.21; every `bbox` value lies at
least `0.021` from a change of its last digit.
