# nx.cuda's cubins

`src/` holds nx.cuda's device code. `gen.py` compiles it, and the CUDA
suite's harness (`test/cuda/support/harness.cu`), to the sm_89 cubins the
libraries embed, and writes `../kernels.ml`, the facts of `../kernels.h`
the host plans with. The cubins and `kernels.ml` are committed, so a
checkout without CUDA builds nx.cuda.

Regenerate them after changing the device code, the headers it reads or
`gen.py`, and check them, from the repository root:

```
uv run packages/nx/lib/cuda/kernels/gen.py
uv run packages/nx/lib/cuda/kernels/gen.py --check
```

uv installs the pinned toolchain from PyPI: `nvidia-cuda-nvcc`,
`nvidia-cuda-crt`, `nvidia-cuda-runtime` and `nvidia-nvvm`, all 13.4.92.
They are one release: ptxas fails on a mixed one. nvcc also needs the
host's C++ compiler, with which it preprocesses.

`kernels.ml` comes from clang on the 64-bit host, whose layout of the
structs nvcc gives the device: its preprocessor expands the X-macros (`NX_CUDA_KERNELS`, `NX_CUDA_TILES`)
and lists the macros, its syntax tree names the enumerations' constants
and the structs' fields, and a program it compiles prints their values,
sizes and offsets. A name follows the header's: `NX_LOOP_WORDS` is
`loop_words`, struct `pack_params` is module `Pack_params`. `FAMILIES` in
`gen.py` types the arguments of the kernels' rows the host reads.

After nvcc, `gen.py` empties `.debug_frame` and `.rel.debug_frame`,
call-frame records for a debugger. The CUDA driver and rig's NV loader read
the stripped cubin as they read the original.

nvcc gives the same bytes on every run and from any directory, so `--check`
compiles both cubins and writes `kernels.ml` in memory, and fails if any
differs from the committed file. The dune build checks something weaker of
nx.cuda's cubin in every tree: it holds an empty kernel named after its
sources' digest, and `fresh.ml` refuses a cubin whose name does not match
today's sources. Its sources include `kernels.ml`'s, so a change to them
fails the build until `gen.py` writes both.

Git stores a new cubin as a delta against the last, about the size of the
kernels that changed. A compressed cubin would cost its whole size each
time, so the cubins are committed as nvcc writes them.
