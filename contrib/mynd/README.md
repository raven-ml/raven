# Mynd

Images for OCaml, built on [Nx](../../packages/nx/). An image is an Nx
array `[…; h; w; c]`. Mynd gives its numbers a meaning and converts them
between dtypes, transfers colour between sRGB and linear light, handles
alpha, samples images at positions under reconstruction kernels, warps them
by transforms, resizes them with band-limiting, and carries boxes, masks,
COCO's run-length encoding, connected components and non-maximum
suppression. Every operation is a composition of nx operations, so rune
differentiates, batches and compiles it.

mynd is a contrib package: it builds and tests against raven's main,
releases on its own schedule and carries no API stability guarantee.
`opam install mynd` installs it; `opam install raven` does not.

- `lib/mynd.mli` is the API and states every rule.
- `test/` holds the laws and the cases against reference implementations
  (torch, torchvision, Pillow, pycocotools, scipy).
