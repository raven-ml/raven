# ELF fixtures

Real objects read by the ELF suite and its bench. No test compiles one:
each is committed with where it came from.

## Host objects

`host/host_x86_64.o` and `host/host_aarch64.o` are `host/host.c`, and
`kernels/host_aarch64.o` is `kernels/host.c`, compiled by Homebrew clang
22.1.7. Each source's header holds its command. The commands reproduce the
objects byte for byte.

## NVIDIA

`simple_add_sm89.cubin` is `simple_add.cu` compiled for sm_89 by NVRTC 12.8
with `--gpu-architecture=sm_89 --minimal` (and the include directories
`/usr/local/cuda/include`, `/usr/include`, `/opt/cuda/include`), through
tinygrad's `NVRTCCompiler("sm_89", ptx=False)` at tinygrad
471a3aeb6924257d5e9bf321f5ff0a519163f18e, in the container
`ghcr.io/tinygrad/cuda-arm64@sha256:350801c5bc2fd3cf7a0c59e88e10483419a97a039e39bcb304cb9598d995349a`.
A copy of `packages/tolk/test/runtime/ops_nv/simple_add_sm89.cubin`.

- `simple_add.cu`: sha256 aeb42772fb0dd43a92162bd8827d0ade73466b8207dcfd6eb79ff077bdea2ea7
- `simple_add_sm89.cubin`: sha256 4f80310e18666d28a1e7ce79c7c16c8cc23b5bda349c103b0ad3c5e44a092a90

## AMD

`simple_add_gfx1100.hsaco` is `simple_add.cpp` compiled by Homebrew clang
22.1.7 and linked by Ubuntu LLD 18.1.3:

```
clang -x hip --cuda-device-only --offload-arch=gfx1100 -nogpuinc -nogpulib \
  -O3 -mcumode -mcode-object-version=5 -cuid=tolk_fixture -fgpu-rdc -c \
  simple_add.cpp -o simple_add.bc
clang -target amdgcn-amd-amdhsa -mcpu=gfx1100 -nogpulib -O3 \
  -mcode-object-version=5 -c simple_add.bc -o simple_add.o
ld.lld -shared simple_add.o -o simple_add_gfx1100.hsaco
```

A copy of `packages/tolk/test/gen/runtime/ops_amd_fixtures/simple_add_gfx1100.hsaco`.

- `simple_add.cpp`: sha256 09145b8f0798ed1c4ce26527097da4fb699c909fec17c725eff1306a9d46b68d
- `simple_add_gfx1100.hsaco`: sha256 4351fb5640b1348c725268f809e5d023c16ef813607e71450056c04eb34640a5

`lds_gfx1100.o` is `lds_gfx1100.ll` compiled by LLVM 21's AMDGPU backend
through tinygrad's `AMDLLVMCompiler("gfx1100")` at tinygrad
471a3aeb6924257d5e9bf321f5ff0a519163f18e: the passes `default<O2>`, then code
generation at the default level, position independent, for `gfx1100` with
`+cumode`. With LLVM's tools that is:

```
opt -passes='default<O2>' lds_gfx1100.ll -o lds_gfx1100.bc
llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -mattr=+cumode \
  -relocation-model=pic -O2 -filetype=obj lds_gfx1100.bc -o lds_gfx1100.o
```

LLVM 22 makes a different object. `lds_gfx1100.ll` is the `lds` module of
`packages/tolk/test/gen/runtime/ops_amd_fixtures/kernels.ll` with its target
triple, and the object a copy of that directory's `lds_gfx1100.o`.

- `lds_gfx1100.o`: sha256 40fa1729ff64d52239115272c84994945dc2391b40c0ded7ff60775bb6532e42

`unfold.8.co` and `unary.float16.co` are copies of nx's AMD kernels in
`packages/nx/lib/amd/kernels/gfx12-generic/`, which this command makes from
that directory's `src/` with the comgr it pins in `pins.json`:

```
uv run packages/nx/lib/amd/kernels/gen.py
```

- `unfold.8.co`: sha256 5c20b7f5b033e393f58f25e25455e7e9a60bd08ff0110aedc59b9c42d51632ad
- `unary.float16.co`: sha256 5f27866b37c61784060308ad7c774885bd00e78df67b9e78a6e0c099b90ca2a5

They stay copies, so the suite and the bench read the same bytes when nx
regenerates its kernels.
