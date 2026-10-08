# Bench fixtures

The kernels the bench launches, one binary per GPU vendor, each made in
this directory from its source. The bench runs the smallest kernel of
each: `step` for Metal (one thread adding 1 to a word), `empty` for the
others.

- `fill.metallib`, the kernels of `fill.metal`, made on macOS 26.3.1 with
  Xcode 26.3's Metal toolchain (`metal` 32023.864):

  ```
  xcrun -sdk macosx metal -c fill.metal -o fill.air
  xcrun -sdk macosx metallib fill.air -o fill.metallib
  rm fill.air
  ```

- `kernels.ptx`, written by hand: CUDA loads it as PTX.

- `kernels_gfx1201.hsaco`, the kernels of `kernels.cl` for gfx1201, made
  with Homebrew clang 22.1.7 and Homebrew LLD 21.1.8:

  ```
  clang -c -x cl -cl-std=CL2.0 -target amdgcn-amd-amdhsa -mcpu=gfx1201 -mcode-object-version=5 -nogpulib -O2 kernels.cl -o kernels.o
  ld.lld -shared kernels.o -o kernels_gfx1201.hsaco
  rm kernels.o
  ```

- `kernels_sm89.cubin`, the kernels of `kernels.cu` for sm_89, made on
  Debian 13 (x86_64) by `nvrtc.c` with NVRTC 12.8.93 (CUDA 12.8, the PyPI
  package `nvidia-cuda-nvrtc-cu12==12.8.93`), whose files are under
  `$NVRTC`; it rebuilds the committed bytes (md5
  `02b28969a8cbb3534d116002635cea0a`):

  ```
  cc nvrtc.c -I$NVRTC/include -L$NVRTC/lib -l:libnvrtc.so.12 -Wl,-rpath,$NVRTC/lib -o nvrtc
  ./nvrtc kernels.cu sm_89 kernels_sm89.cubin
  ```
