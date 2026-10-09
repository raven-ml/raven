# CUDA fixtures

`kernels.ptx`, `global.ptx` and `launch.ptx` are written by hand.
`kernels.cubin` is made from `kernels.ptx` in this directory with ptxas
from CUDA 13.4 (V13.4.92):

    ptxas -arch=sm_89 -o kernels.cubin kernels.ptx

`many.cubin` holds 140 kernels, `k0` to `k139`, the size of a kernel
library's image, for the bench's image row: kernel `i` stores `i` into the
32-bit word at its one parameter. It is made in this directory with the
same ptxas from the PTX this script writes:

    {
      printf '.version 7.0\n.target sm_50\n.address_size 64\n'
      for i in $(seq 0 139); do
        printf '.visible .entry k%d(.param .u64 out)\n{\n  .reg .b64 %%rd<2>;\n  ld.param.u64 %%rd1, [out];\n  cvta.to.global.u64 %%rd1, %%rd1;\n  st.global.u32 [%%rd1], %d;\n  ret;\n}\n' $i $i
      done
    } > many.ptx
    ptxas -arch=sm_89 -o many.cubin many.ptx
