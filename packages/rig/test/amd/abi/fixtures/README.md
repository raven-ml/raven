# AMD ABI fixtures

`gen.sh` makes each code object from its source.

- `kernels_gfx1030.o`, of `kernels.s`, is a relocatable code object whose
  descriptors reach their kernels through `R_AMDGPU_REL64` relocations.
- `kernels_gfx1030.hsaco` is the code object linked from it. Its sections
  sit at addresses, and it holds no relocations.
- `many_gfx1201.hsaco`, of `many.cl`, holds 128 kernels for gfx1201, each
  taking 260 bytes of scratch per work-item, without a symbol table. The
  bench and the lookup of many kernels read it.

The suite also reads `hidden_gfx1201.hsaco` of `../../fixtures`.

One file was captured on a Radeon AI PRO R9700 (gfx1201) under Linux's
amdgpu driver.

- `ttracedata_gfx1201.sqtt` is shader engine 0's thread trace of `k_ttrace`
  from `ttracedata.cl`, compiled as `gen.sh` compiles `many.cl`, launched
  over 64 workgroups of 64 work-items between `Thread_trace.start` and
  `Thread_trace.stop`, its bytes up to the end `Thread_trace.length`
  gives. Each wave issues eight `s_ttracedata` (0x06 packets, 14 nibbles)
  and eight `s_ttracedata_imm` (0x46 packets, 8 nibbles).
