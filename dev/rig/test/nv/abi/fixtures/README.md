# NV ABI fixtures

- `simple_add_sm89.cubin`, the cubin of `simple_add.cu` for sm_89, made in
  this directory by `../../../gen/nvrtc.c` with NVRTC 12.8.93 (CUDA 12.8,
  the PyPI package `nvidia-cuda-nvrtc-cu12==12.8.93`), whose files are
  under `$NVRTC`:
  `cc ../../../gen/nvrtc.c -I$NVRTC/include -L$NVRTC/lib -l:libnvrtc.so.12 -Wl,-rpath,$NVRTC/lib -o nvrtc && ./nvrtc simple_add.cu sm_89 simple_add_sm89.cubin`.
  Another release of NVRTC writes its own version into the file.

- `globals_sm89.cubin`, the cubin of `globals.cu` for sm_89, made the same
  way on kimchi (Debian 13, x86_64), which rebuilds the committed bytes (md5
  8fa197ddee692082de0ae3c46916b770):
  `./nvrtc globals.cu sm_89 globals_sm89.cubin`. `llvm-readelf -S -r -s`
  reads its uninitialised `scale` in `.nv.global` (NOBITS, allocated, 4
  bytes) and its initialised `bias` in `.nv.global.init` (4 bytes), each
  address written into `.nv.constant4` by a `R_CUDA_64` relocation.

What the suite expects of `simple_add_sm89.cubin` was read with Homebrew LLVM 22.1.7's
`llvm-readelf -S -r -s -x .nv.info -x .nv.info.simple_add simple_add_sm89.cubin`:

- two allocated sections, `.nv.constant0.simple_add` (0x17c bytes,
  alignment 4) then `.text.simple_add` (0x200 bytes, alignment 128): laid
  out at 128 bytes, the code starts at 0x180 and the image ends at 0x380;
- `.nv.info` holds `04 2f 0800 06000000 0c000000`, EIATTR_REGCOUNT (0x2f)
  of symbol 6, `simple_add`: 12 registers; and `04 12 0800 06000000
  00000000`, EIATTR_MIN_STACK_SIZE (0x12): no stack;
- `.nv.info.simple_add` holds `04 0a 0800 02000000 6001 1c00`,
  EIATTR_PARAM_CBANK (0xa): parameters at 0x160 of bank 0, 0x1c bytes;
- its one relocation, in `.rel.debug_frame`, patches a section the image
  does not hold.

Made in this directory from `many.cu` with nvcc 13.4 (V13.4.92, CUDA 13.4, `/usr/local/cuda/bin/nvcc`) on kimchi (Debian 13, x86_64), which rebuilds the committed bytes (md5 ee5e154c96c42e3e1c0c653e8dff2d4e):

- `many_sm89.cubin`, 128 kernels for sm_89, each with an initialised global
  of its own (in `.nv.global.init`) whose address a `R_CUDA_64` relocation
  writes into bank 4:
  `nvcc -cubin -arch=sm_89 -O2 many.cu -o many_sm89.cubin`
