# nx.amd's code object

`src/` holds nx.amd's device code, HIP compiled for the device alone.
`gen.py` compiles it, and the AMD suite's harness
(`test/amd/support/harness.hip`, and `staged.hip`, which tests
`src/device.h`), to the gfx12-generic code objects
(code object version 6) the libraries embed, and writes `../kernels.ml`,
the facts of `src/kernels.h` the host plans with. The objects and
`kernels.ml` are committed, so a checkout without the toolchain builds
nx.amd.

Regenerate them after changing the device code, the headers it reads or
`gen.py`, and check them, from the repository root:

```
uv run packages/nx/lib/amd/kernels/gen.py
uv run packages/nx/lib/amd/kernels/gen.py --check
```

The toolchain is LLVM 22.1.8's release archive for the host (macOS arm64
or Linux x86-64), which `gen.py` fetches once into
`$XDG_CACHE_HOME/raven/llvm` (else `~/.cache/raven/llvm`), checks against
its pinned SHA-256, and refuses on a mismatch. It unpacks only clang,
lld, llvm-objcopy and clang's own headers. No ROCm and no device library:
`libc/` holds the few C declarations the device code uses, as clang
builtins.

Each unit compiles with `-fgpu-rdc` to bitcode under a cuid of its own,
and ld.lld links the units' bitcode into one shared object, compiling it
as one module. Before stripping it, `gen.py` refuses an object with a
private segment, a VGPR spill, a dynamic stack, an implicit argument, a
system value rig gives no kernel, a function that is no kernel, waves of
other than 32 lanes, or a kernel whose group size or argument bytes
differ from its row in `kernels.h`. An SGPR spill reaches memory only
through a spilled VGPR, so the private segment refuses it; a spill into a
VGPR's lane is allowed. It prints each kernel's registers, SGPR lane
spills, LDS and bytes. The strip keeps each `name.kd` in `.dynsym`, by
which rig resolves a kernel.

`kernels.ml` comes from the pinned clang: its preprocessor expands the
kernel table (`NX_AMD_KERNELS`) and the enumerations, and the values of
their constants come from an array of them it compiles to IR. A name
follows the header's: `NX_LOOP_USED` is `loop_used`. No struct crosses
from the host to the device: a kernel's parameters are 64-bit words at
the indices the enumerations name. Each kernel's static LDS bytes
(`Kernels.lds`) and VGPRs (`Kernels.vgprs`) come from the library
object's metadata; a contraction tile's dynamic LDS bytes (`lds` of
`Kernels.tile` and `Kernels.wtile`) from `kernels.h`'s formula, which
the device code checks its tile's layout against.

The toolchain gives the same bytes on every run and from any directory,
so `--check` builds both objects and `kernels.ml` in memory and fails if
any differs from the committed file. The dune build checks something
weaker in every tree: each object holds a symbol named after its
sources' digest, and `fresh.ml` refuses an object whose symbol does not
match today's sources.

The library's object is 5,679,472 bytes; the harness's, 38,688.
Lane spills of the loops: loop_reduce 61, loop_plain 34, loop_scan 28,
loop_network 24; of the factorisations' panels: la_qr_panel 200 to 209,
la_lu_panel 31 to 32.
