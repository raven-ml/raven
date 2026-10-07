# AMD ABI fixtures

Made in this directory from `kernels.s` with Homebrew clang 22.1.7
(`clang`) and Homebrew LLD 21.1.8 (`ld.lld`):

- `kernels_gfx1030.o`, a relocatable code object whose descriptors reach
  their kernels through `R_AMDGPU_REL64` relocations:
  `clang -c -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1030 -mcode-object-version=5 kernels.s -o kernels_gfx1030.o`
- `kernels_gfx1030.hsaco`, the code object linked from it, its sections at
  addresses and no relocations:
  `ld.lld -shared kernels_gfx1030.o -o kernels_gfx1030.hsaco`
