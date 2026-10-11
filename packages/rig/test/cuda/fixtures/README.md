# CUDA fixtures

`kernels.ptx`, `global.ptx` and `launch.ptx` are written by hand. `gen.sh`
makes `kernels.cubin` from `kernels.ptx`, and `many.cubin` for the bench's
image row. `many.cubin` holds 140 kernels, `k0` to `k139`, the size of a
kernel library's image. Kernel `i` stores `i` into the 32-bit word at its
one parameter.
