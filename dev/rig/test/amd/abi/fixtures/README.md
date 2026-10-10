# AMD ABI fixtures

Made in this directory from `kernels.s` with Homebrew clang 22.1.7
(`clang`) and Homebrew LLD 21.1.8 (`ld.lld`):

- `kernels_gfx1030.o`, a relocatable code object whose descriptors reach
  their kernels through `R_AMDGPU_REL64` relocations:
  `clang -c -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1030 -mcode-object-version=5 kernels.s -o kernels_gfx1030.o`
- `kernels_gfx1030.hsaco`, the code object linked from it, its sections at
  addresses and no relocations:
  `ld.lld -shared kernels_gfx1030.o -o kernels_gfx1030.hsaco`

Made in this directory from `many.cl` with the same tools:

- `many_gfx1201.hsaco`, 128 kernels for gfx1201 without a symbol table,
  each taking 260 bytes of scratch per work-item, for the bench and the
  lookup of many kernels:
  `clang -c -x cl -cl-std=CL2.0 -target amdgcn-amd-amdhsa -mcpu=gfx1201 -mcode-object-version=5 -nogpulib -O2 many.cl -o many.o && ld.lld -shared --strip-all many.o -o many_gfx1201.hsaco && rm many.o`

Made in this directory from `hidden.s` with the same tools:

- `hidden_gfx1201.hsaco`, two kernels for gfx1201 and their metadata:
  `every` names each implicit argument of code object version 5, and
  bounds its workgroups at 128 work-items; `plain` names none:
  `clang -c -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -mcode-object-version=5 hidden.s -o hidden.o && ld.lld -shared hidden.o -o hidden_gfx1201.hsaco && rm hidden.o`

Captured on a Radeon AI PRO R9700 (gfx1201) under Linux's amdgpu driver:

- `ttracedata_gfx1201.sqtt`, shader engine 0's thread trace of `k_ttrace`
  from `ttracedata.cl`, compiled as `many.cl` is, launched over 64
  workgroups of 64 work-items between `Thread_trace.start` and
  `Thread_trace.stop`, its bytes up to the end `Thread_trace.length`
  gives. Each wave issues eight `s_ttracedata` (0x06 packets, 14 nibbles)
  and eight `s_ttracedata_imm` (0x46 packets, 8 nibbles).
