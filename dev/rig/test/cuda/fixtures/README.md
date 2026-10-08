# CUDA fixtures

`kernels.ptx` and `global.ptx` are written by hand. `kernels.cubin` is made from it in this
directory with ptxas from CUDA 13.4 (V13.4.92):

    ptxas -arch=sm_89 -o kernels.cubin kernels.ptx
