# nx

n-dimensional arrays on rig's devices: a dtype, a view over a buffer, and
eager operations computed by a device's kernels. Everything else in raven
exchanges `Nx.t` values and `Nx.Ptree` structures.

## Boundaries

- nx knows nothing of differentiation or compilation. No derivative, trace
  or jit vocabulary in nx, its effects or its docs. rune answers nx from
  outside.
- nx never moves data between devices on its own. An operation a device has
  no kernel for raises an error that names the move to make. It never
  computes on the host instead.
- Device storage and kernels are separate libraries. rig holds bytes; a
  kernel library computes. A dependency edge that adds link weight to every
  user of a device blocks a design.
- tolk is the only compiler. nx's kernel libraries never generate code at
  run time: they interpret programs over blocks or run hand-written loops.
- Kernel libraries implement operation families. No kernel for a named
  formula (RoPE, AdamW, SwiGLU, a norm's tail); those compose from
  operations, and fusion is tolk's job under `Rune.jit`.

## Semantics

- Narrow floats (`float16`, `bfloat16`, 8-bit floats) compute at `float32`
  and round once.
- Integer arithmetic is modular (`-fwrapv`). Floats keep IEEE NaN and
  infinity behaviour. No `-ffast-math`, and no contraction of a product and
  a sum unless the `.mli` states it.
- One Threefry generator, `Nx.Rng`. Its key is an `int32` tensor, so it can
  be a structure's leaf, a jit input or a vmap axis.
- A fix that changes results or random draws takes the correct version.
  The commit body states the change.
- The surface stays familiar to array users. Departing from a familiar
  name or behaviour needs a clear win. Linear algebra and FFT live at
  `Nx`'s top level.

## Kernels and performance

- Contraction (matmul, einsum, GEMV, dot) runs at vendor parity (ratio at
  most 1.02) on every backend; every other kernel within 1.5× of the vendor.
- Vendor libraries are measured outside the repo, never linked.
- Close a gap by studying the best permissive reference (BLIS, libxsmm,
  SLEEF, CUTLASS, CK) and writing our own code. Check its licence first;
  never GPL. Never read the instruction stream of a proprietary library.
- A change to GPU kernels times a smoke set before landing: a large `bfloat16`
  square, a decode row and a small square. Tests alone missed a 188×
  slowdown once.

## Tests

- API suites mirror the libraries: `test/test_*.ml` for `nx`,
  `test/<library>/` for each sub-library.
- `test/gen/` makes the goldens under `test/golden/`, through `uv run`.
