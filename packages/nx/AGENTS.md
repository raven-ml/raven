# nx

n-dimensional arrays as values on device sets: a dtype, a shape and a
placement over the set the value's brand names, and eager operations that a
kernel library computes where their operands lie. Everything else in raven
exchanges `Nx.t` values and `Nx.Ptree` structures.

## Boundaries

- nx knows nothing of differentiation or compilation. No derivative or jit
  vocabulary in nx or its docs: `Nx.Prim` gives interpretations, and rune
  answers nx from outside through them.
- nx moves data between sets only at `Nx.place`, and within a set only as
  a placement and the routing rules state. A set with no kernel library
  computes nothing eagerly: an operation there raises naming the remedy. It
  never computes on the host instead.
- Device storage and kernels are separate libraries. rig holds bytes; a
  kernel library implements `Nx_kernel.S` and computes; nx.cpu is the only
  one. A dependency edge that adds link weight to every user of a
  device blocks a design.
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
- A value made from numbers is a host value; an `_at` form makes it at a
  placement. Layout is a cost, never a meaning.
- A fix that changes results or random draws takes the correct version.
  The commit body states the change.
- The surface stays familiar to array users. Departing from a familiar
  name or behaviour needs a clear win. Linear algebra and FFT live at
  `Nx`'s top level.

## Kernels and performance

- Contraction (matmul, einsum, GEMV, dot) runs at vendor parity (ratio at
  most 1.02) in every kernel library; every other kernel within 1.5× of the
  vendor.
- Vendor libraries are measured outside the repo, never linked.
- Close a gap by studying the best permissive reference (BLIS, libxsmm,
  SLEEF, CUTLASS, CK) and writing our own code. Check its licence first;
  never GPL. Never read the instruction stream of a proprietary library.

## Tests

- API suites mirror the libraries: `test/test_*.ml` for `nx`,
  `test/<library>/` for each sub-library.
- `test/gen/` makes the goldens under `test/golden/`, through `uv run`.
