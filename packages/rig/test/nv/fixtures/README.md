# Fixtures

Cubins for sm_89, each made from its source by `gen.sh`.

- `kernels_sm89.cubin` holds the kernels of `kernels.cu`.
- `launch_sm89.cubin` holds the kernels of `launch.cu`, which the
  conformance and NV suites launch with their parameters in a run's block.
- `twice_sm89.cubin` and `thrice_sm89.cubin` each hold one kernel `index`,
  of one size, which writes `2i` and `3i`.
- `grid_params_sm89.cubin` is made by NVCC, the others by NVRTC. Its kernel
  reads its `__grid_constant__` parameter through a generic pointer.
  Disassembly reads bank 0's base from `c[0x0][0x40]` and adds `0x160`.
  Compiling the same source for `sm_120` reads the parameter address
  directly from `c[0x0][0x348]`. The ABI suite checks that layout without
  Blackwell hardware.
