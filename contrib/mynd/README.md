# Mynd

Images for OCaml, built on [Nx](../../packages/nx/). An image is an Nx
array `[…; h; w; c]`. Mynd defines what an image's numbers mean and
converts them between dtypes, transfers colour between sRGB and linear
light, handles alpha, computes luma, luminance, HSV and CIE Lab, samples
images at positions under reconstruction kernels, warps them by
transforms, resizes them with band-limiting, filters them by
correlation, Gaussian and median blurs, gradients and morphology, and
carries boxes, masks, COCO's run-length encoding, connected components
and non-maximum suppression. Rune differentiates, batches and compiles
its functions on arrays, with the exceptions the preamble of
[`lib/mynd.mli`](lib/mynd.mli) names.

mynd is a contrib package. It builds and tests against raven's main,
releases on its own schedule and carries no API stability guarantee.
`opam install mynd` installs it, and `opam install raven` does not.

- [`lib/mynd.mli`](lib/mynd.mli) is the API and states every rule.
- [`doc/index.md`](doc/index.md) is the quick start. The other pages
  under `doc/` compose the API into colour adjustments, filters, edge
  detectors and morphological operators.
- `test/` holds the laws and the cases against reference implementations
  (torch, torchvision, Pillow, pycocotools, scipy, scikit-image,
  OpenCV).
