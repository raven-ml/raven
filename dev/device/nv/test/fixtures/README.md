# Fixtures

Cubins for sm_89, each made in this directory from its source by `nvrtc.c`
with NVRTC 12.8.93 (CUDA 12.8, the PyPI package
`nvidia-cuda-nvrtc-cu12==12.8.93`), whose files are under `$NVRTC`:

```
cc nvrtc.c -I$NVRTC/include -L$NVRTC/lib -l:libnvrtc.so.12 -Wl,-rpath,$NVRTC/lib -o nvrtc
./nvrtc kernels.cu sm_89 kernels_sm89.cubin
./nvrtc twice.cu sm_89 twice_sm89.cubin
./nvrtc thrice.cu sm_89 thrice_sm89.cubin
```

Made on kimchi (Debian 13, x86_64), which rebuilds the committed bytes
(md5 `02b28969a8cbb3534d116002635cea0a`, `b6f45ba85a182b94a1beccd6ccadc29f`,
`7427ea577b0ea526630d431680fce825`). Another release of NVRTC writes its
own version into the files.

- `kernels_sm89.cubin`: the kernels of `kernels.cu`.
- `twice_sm89.cubin`, `thrice_sm89.cubin`: one kernel `index` each, of one
  size, writing `2i` and `3i`.
