# `08-masks`

A segmentation model predicts each mask as logits on a coarse grid that
covers the image. This example turns three such masks into what a user
gets: masks at the image's size without specks or small holes, their
boxes, areas and COCO run-length encodings, drawn over the photograph.

```bash
dune exec ./main.exe
```

- The logits are written in the program, as SAM 2's decoder gives them:
  `[3; 256; 256; 1]`, one mask per object in one batch, positive inside
  the object. The left eye's has a speck outside it and a hole inside.
- SAM 2 cleans the logits before it upsamples them. On the grid, the
  8-connected pieces of background of at most 50 cells become
  foreground, and the pieces of foreground of at most 50 cells, counted
  on the same logits, become background, by moving their logits to 10
  and -10. Since both are decided from the same mask, a ring of 8 cells
  around a hole of 1 cell, under a limit of 8, leaves the hole's cell
  alone in the foreground; filling first would make a piece of 9 cells
  that stays. SAM 2's predictor runs this step only when given the
  limits, and skips it without a GPU.
  `Mynd.Mask.components` labels the pieces, and counting each label's
  cells gives their areas (Areas and small objects in
  [`doc/morphology.md`](../../doc/morphology.md)).
- `Mynd.Sample.resize` upsamples the logits to the photograph,
  `[3; 300; 451; 1]`, bilinearly: each pixel reads the grid at its
  centre. A pixel is in the mask where its logit is above 0.
- `Mynd.Box.of_mask` gives each mask's box, `[x₀ y₀ x₁ y₁]` in pixel
  edges, which COCO writes as `[x, y, w, h]`.
- `Mynd.Mask.iou` compares each cleaned mask with the one upsampled from
  the raw logits.
- `Mynd.Rle.encode` and `Mynd.Rle.to_string` give COCO's compressed
  counts, and `Mynd.Rle.decode` gives the mask back.

It prints, for each mask, its pieces and holes before and after
cleaning, its IoU with the raw mask, its box and area, the start and
length of its counts, and whether they decode back to the mask.
`masks.png` draws each mask in a colour over the photograph, with its
box. A figure's image covers `[0, w] × [0, h]` with row 0 at the top,
which are mynd's coordinates, so the boxes need no conversion.

Margins: every logit on the grid lies at least `3.3e-3` from 0, and
every upsampled logit `1.5e-4`, about 300 float32 ulps at the logits'
scale. No piece of the grid has between 41 and 60 cells, so none is near
the limit of 50. The rest are counts, and an IoU of counts rounded once,
`8e-6` from a change of its fourth decimal.
