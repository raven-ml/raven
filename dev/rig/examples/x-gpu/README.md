# `x-gpu`

**Needs a GPU:** the Mac's through Metal (macOS 15 or later); on Linux, an
NVIDIA GPU through CUDA or NVIDIA's kernel driver, or an AMD GPU through
`amdgpu`. Without one it prints a line and exits.

A GPU is opened by its driver, the library that runs it, and a path, the library
that reached it, passed together to `Rig.open_` under one name. After that it is
a device like the others. This example opens GPU 0, prints its facts, copies
256 MiB to it and back, and shows a submission still running when `submit`
returns.

```bash
dune exec dev/rig/examples/x-gpu/main.exe [metal|cuda|nv|amd]
```

Without an argument it opens the first path that sees a GPU.

## What You'll Learn

- How each path opens a GPU:

  | Path    | Driver      | Opener                     |
  | ------- | ----------- | -------------------------- |
  | `metal` | `Rig_metal` | `Rig_metal.open_`          |
  | `cuda`  | `Rig_cuda`  | `Rig_cuda.open_`           |
  | `nv`    | `Rig_nv`    | `Rig_nv_nvidia.open_`      |
  | `amd`   | `Rig_amd`   | `Rig_amd_amdgpu.open_`     |

- A GPU's facts: `arch`, `budget`, `shares_host_memory`, `reaches`
- Copies between the host and a GPU, timed by a profile's copy events
- Work in flight: `signaled` behind `submitted` until `Point.wait` returns

## Key Functions

| Function                                | Purpose                                   |
| --------------------------------------- | ----------------------------------------- |
| `Rig.open_ (module D) ~name make`       | The device `make` opens, driven by `D`     |
| `Buffer.copy ~src ~dst`                 | Copy between the host and the GPU          |
| `submit`, `signaled`, `Point.wait`      | Hand work over, see it complete            |

## Next Steps

Continue to [x-metal-kernel](../x-metal-kernel/) on a Mac, or
[x-cuda-kernel](../x-cuda-kernel/), [x-nv-kernel](../x-nv-kernel/) and
[x-amd-kernel](../x-amd-kernel/) on Linux.
