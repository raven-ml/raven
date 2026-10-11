# ELF fixtures

`gen.sh` makes each file from its source.

- `host_x86_64.o`, a host object with RELA calls and addends.
- `host_aarch64.o`, the same for aarch64, and the bench's host object.
- `host_unwind_x86_64.o`, `host_x86_64.o` with unwind tables, whose
  `.eh_frame` is of type `SHT_X86_64_UNWIND`.
- `amd_gfx1100.o`, a relocatable GPU object whose descriptor relocates to its
  kernel.
- `amd_gfx1100.hsaco`, the code object linked from it, with sections at
  addresses.
- `amd_<k>_gfx1100.hsaco` for `<k>` of 16 and 128, `<k>` kernels without a
  symbol table, for the bench.
- `global_sm89.cubin`, a cubin whose kernel uses an uninitialised
  `__device__` global, which NVRTC puts in `.nv.global` (SHT_NOBITS),
  relocated to from `.nv.constant4`.
