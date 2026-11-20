# `03-model-input`

Prepare the photograph as two models take their input, and map a box a
detector returns back onto the photograph.

```bash
dune exec ./main.exe
```

- `Mynd.Sample.resize` with `` `Triangle `` is bilinear resizing. SAM 2
  squares the image to 1024 × 1024 and RT-DETR to 640 × 640, neither
  keeping the aspect ratio. Both enlarge the photograph, so each new pixel
  reads the photograph at its centre, and past the border the border
  pixels repeat (`` `Replicate ``).
- SAM 2 then subtracts ImageNet's mean and divides by its deviation, per
  channel, with `Nx.sub` and `Nx.div`; RT-DETR keeps the fractions
  `Mynd.convert` gives.
- `Nx.moveaxis` puts the channels first, as a model of 2-d convolutions
  reads them.
- `Mynd.Box.of_cxcywh` turns a detector's centre and size into corners.
  Scaling an image scales its coordinates (the Coordinates paragraph of
  [`mynd.mli`](../../lib/mynd.mli)), so a fraction of the input's width
  is the same fraction of the photograph's: one multiply by `[w h w h]`
  maps the box to the photograph, and one by 1024 to SAM 2's input.

It prints each input's shape with its per-channel mean and deviation,
and the box in both coordinate systems. `input.png` shows the photograph
with the box, and RT-DETR's input.

RT-DETR's reference resizes the 8-bit image in two rounded passes; this
example resizes the fractions, which differ from those passes by
rounding.

Every function here takes batch axes, so the same code prepares a batch
`[n; h; w; 3]` of images of one size. In a training loop, kaun's
`Data.batches2` cuts a dataset of such images and their targets into
batches, and `Seq.map` applies the preparation to each.

Margins: the box's numbers are exact. The means lie at least `1.2e-4`
from a change of their third decimal and the deviations `1.1e-4` from a
change of their second.
