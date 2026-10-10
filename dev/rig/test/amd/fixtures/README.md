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

Made in this directory from `launch.cl` with the same tools:

- `launch_gfx1201.hsaco`, the kernels the suites launch: `ids` and `twice`,
  the conformance laws' (`Rig_gpu_support.Conformance.launch_binary`),
  which learn their grid from the implicit arguments; `lds`, which takes
  dynamic LDS after 256 bytes of its own; `scratch`, which takes 4100
  bytes of scratch per work-item; and `packet`, which reads its dispatch
  packet:
  `clang -c -x cl -cl-std=CL2.0 -target amdgcn-amd-amdhsa -mcpu=gfx1201 -mcode-object-version=5 -nogpulib -O2 launch.cl -o launch.o && ld.lld -shared launch.o -o launch_gfx1201.hsaco && rm launch.o`

Made in this directory from `launch.cl` with the same tools, for gfx942:

- `launch_gfx942.hsaco`, the same kernels for a GPU of several dies, whose
  queue reads AQL packets:
  `clang -c -x cl -cl-std=CL2.0 -target amdgcn-amd-amdhsa -mcpu=gfx942 -mcode-object-version=5 -nogpulib -O2 launch.cl -o launch.o && ld.lld -shared launch.o -o launch_gfx942.hsaco && rm launch.o`
