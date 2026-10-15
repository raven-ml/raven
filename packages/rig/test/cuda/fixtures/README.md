# CUDA fixtures

`kernels.ptx`, `global.ptx`, `launch.ptx` and `wide.ptx` are written by
hand. `wide.ptx` holds the conformance suite's `wide`, apart because its
8,192 bytes of parameters need PTX 8.1 for sm_70. `gen.sh` makes
`kernels.cubin` from `kernels.ptx`, and `many.cubin` for the bench's
image row. `many.cubin` holds 140 kernels, `k0` to `k139`, the size of a
kernel library's image. Kernel `i` stores `i` into the 32-bit word at its
one parameter.
