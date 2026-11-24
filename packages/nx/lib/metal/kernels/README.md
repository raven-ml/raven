# nx.metal's metallib

`src/` holds nx.metal's device code, in the Metal Shading Language.
`gen.py` compiles it, and the Metal suite's harness
(`test/metal/support/harness.metal`), to the metallibs the library and
the suite embed, and writes `../kernels.ml`, the facts of `src/kernels.h`
the host plans with. The metallibs and `kernels.ml` are committed, so a
checkout without the toolchain builds nx.metal, and the library has no
C: rig makes every Metal call.

Regenerate them after changing the device code, the headers it reads or
`gen.py`, and check them, from the repository root:

```
uv run packages/nx/lib/metal/kernels/gen.py
uv run packages/nx/lib/metal/kernels/gen.py --check
```

The toolchain is Xcode's Metal toolchain at version 32023.864, through
`xcrun -sdk macosx`. `gen.py` refuses a `metal` or a `metallib` of
another version, since another compiler gives other bytes and may give
other bits. Every unit compiles as MSL 3.2 for macOS 15, rig's floor,
with `-fno-fast-math` and `-ffp-contract=off`, warnings as errors, and
`src/device.h` adds `#pragma METAL fp contract(off)`: no product fuses
into a sum the source does not write as an `fma`. `metallib` links a
library's units into one metallib.

A kernel reads its parameters as a struct of 64-bit fields at
`[[buffer(0)]]`, whose size Metal states, so rig checks each launch's
parameters against it. A field that addresses a buffer is a pointer into
the device or the constant address space: the struct is then an argument
buffer, whose pointers rig writes as GPU addresses. The Metal Shading
Language defines that (§2.13, "argument buffers extend the basic buffer
types to include pointers (buffers)"; §5.2 lists a constant buffer as "a
pointer or reference to any data type in the constant address space"),
and it defines no cast of an integer to a pointer. The address rig writes
is the buffer's `MTLBuffer.gpuAddress` plus the ref's offset, which
Metal's API defines. A table that changes from call to call, as a copy's
axes, lives in the struct, within rig's 256 bytes of parameters a
launch; past them, and for a table of the program, it is a constant
buffer the struct points to.
`NX_FIELD` checks each field's offset against its word's index.

Before it writes a metallib, `gen.py` reads each unit's IR and refuses
static threadgroup memory (a group's threadgroup memory is its one
dynamic binding, of the bytes its launch states), and, for the library,
a kernel whose parameter bytes differ from its row of `kernels.h`, whose
`max_total_threads_per_threadgroup`, where it states one, differs from
its row's threads, or a kernel the table lacks or a row the metallib
lacks. A kernel that states no bound lets its registers set its
pipeline's limit, which `nx_metal.ml` reads when it resolves the kernel
and fails on where it is under the row's threads. It prints each
kernel's bytes in the metallib and each metallib's total.

`kernels.ml` comes from the pinned `metal`: its preprocessor expands the
kernel table (`NX_METAL_KERNELS`) and the enumerations, and the values of
their constants come from a constant array it compiles to IR. A name
follows the header's: `NX_COPY_DST` is `copy_dst`.

`metal` and `metallib` give the same bytes on every run and from any
directory: `gen.py` builds each metallib twice, the second time from a
copy of its sources under another directory, and refuses one whose
builds differ, and `--check` builds every metallib and `kernels.ml` in
memory and fails if any differs from the committed file. The dune build
checks something weaker in every tree: each metallib holds an empty
kernel named after its sources' digest, and `fresh.ml` refuses a
metallib whose kernel does not match today's sources. The sources a
stamp covers are listed by hand in `gen.py` and the dune rules, so
`gen.py` refuses a unit that includes, outside the SDK, a file its stamp
does not cover, from the dependencies `metal -MD` reports.

The library's metallib is 9,467,499 bytes (copy 4,736, copy_8 to
copy_64 4,448 to 4,464 each, copy_run 4,544, copy_rows 8,320,
copy_rows_wide 8,192, zero 3,552, widen 9,728; the casts' 50 kernels
7,536 to 18,464 each, a byte source class's with its tables; the
contraction's 354 kernels 4.0
MB, the spectral routines' 77 0.53 MB, attention's 27 0.64 MB, the
transforms' 35 0.26 MB, the sorts' 25 0.27 MB, the matrix routines' 15
0.22 MB, the gathers, assemblies and folds' 11 (fold 6, runs 3, move 2)
0.1 MB, the loops the rest, of which the flat direct maps of 32 bits
take 12,768 and 19,520); the harness's, 16,555.
