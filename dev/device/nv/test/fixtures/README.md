# Fixtures

- `simple_add_sm89.cubin`, the cubin of `simple_add.cu` for sm_89, made in
  this directory by `nvrtc.c` with NVRTC 12.8.93 (CUDA 12.8, the PyPI
  package `nvidia-cuda-nvrtc-cu12==12.8.93`), whose files are under
  `$NVRTC`:
  `cc nvrtc.c -I$NVRTC/include -L$NVRTC/lib -l:libnvrtc.so.12 -Wl,-rpath,$NVRTC/lib -o nvrtc && ./nvrtc simple_add.cu sm_89 simple_add_sm89.cubin`.
  Another release of NVRTC writes its own version into the file.
