# AMD driver fixtures

Made in this directory from `kernels.cl` with Homebrew clang 22.1.7
(`clang`) and Homebrew LLD 21.1.8 (`ld.lld`):

- `kernels_gfx1201.hsaco`, the kernels `empty`, `double_index`, `spin` and
  `wild` for gfx1201:
  `clang -c -x cl -cl-std=CL2.0 -target amdgcn-amd-amdhsa -mcpu=gfx1201 -mcode-object-version=5 -nogpulib -O2 kernels.cl -o kernels.o && ld.lld -shared kernels.o -o kernels_gfx1201.hsaco && rm kernels.o`

Made in this directory from `other.cl` with the same tools:

- `other_gfx1201.hsaco`, `kernels_gfx1201.hsaco`'s kernels but for
  `double_index`, which computes `3 * i`: an object of the same size, for
  code placed where another object's ran:
  `clang -c -x cl -cl-std=CL2.0 -target amdgcn-amd-amdhsa -mcpu=gfx1201 -mcode-object-version=5 -nogpulib -O2 other.cl -o other.o && ld.lld -shared other.o -o other_gfx1201.hsaco && rm other.o`

Made in this directory from `work.cl` with the same tools:

- `work_gfx1201.hsaco`, the kernels `copy`, which copies words after a
  delay, `shared`, which takes 256 bytes of local data share, and `inc`,
  which adds 1 to each word:
  `clang -c -x cl -cl-std=CL2.0 -target amdgcn-amd-amdhsa -mcpu=gfx1201 -mcode-object-version=5 -nogpulib -O2 work.cl -o work.o && ld.lld -shared work.o -o work_gfx1201.hsaco && rm work.o`
