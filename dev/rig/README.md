# rig

rig drives a machine's devices from OCaml: the host's CPU, its GPUs and its
stores of bytes, their memory, and the order of work across them. It is OCaml
and C and depends on nothing outside the OCaml distribution. Building it needs
no CUDA, ROCm or vendor SDK. On Linux it is built to boot AMD and NVIDIA
GPUs itself, over PCI, with no kernel driver at all; [What has run on
hardware](#what-has-run-on-hardware) says how far that has gone.

## Driver-less GPUs

A GPU usually comes with three layers of vendor software: a kernel driver, a
user-space runtime, and an SDK to build against. A program that uses the GPU
inherits all three, at versions that must agree with each other and with the
kernel.

`rig.amd.pci` and `rig.nv.pci` do the kernel driver's work from the process.
The process takes the GPU's PCI function, loads the GPU's firmware from a
directory it names, brings up the GPU's blocks, writes its page tables and
reads its interrupts. A program built on rig is then one executable and a
directory of firmware files. On a Linux machine with the GPU it needs nothing
else: no `amdgpu` or `nvidia` module, and no ROCm or CUDA. Both paths are
young; [What has run on hardware](#what-has-run-on-hardware) says how far each
has gone.

```ocaml
let gpu =
  Rig.open_ (module Rig_amd) ~name:(Rig_amd_pci.device_name 0) (fun () ->
      Rig_amd_pci.open_ ~firmware:[ "firmware" ] 0)
  |> Result.get_ok

let () =
  Printf.printf "%s: %s, %d MiB\n" (Rig.name gpu) (Rig.arch gpu)
    (Rig.budget gpu lsr 20);
  let file = Result.get_ok (Rig_disk.of_file "weights.bin") in
  let weights = Rig.Buffer.create gpu (Rig.Buffer.length file) in
  Rig.Buffer.copy ~src:file ~dst:weights
```

```
(executable
 (name main)
 (libraries rig rig.amd rig.amd.pci rig.disk))
```

The opener is the only line that names a path. Through `amdgpu`, the same
program opens `Rig_amd_amdgpu.open_ 0` under `Rig_amd_amdgpu.device_name 0`.
After the open, a GPU is a device like any other: buffers, copies and
submissions are the same calls on every path.

### What a driver-less open needs

- **Linux, and the GPU's PCI function**, taken one of two ways. Behind an
  IOMMU, through VFIO: an administrator binds the function to `vfio-pci` and
  grants the user its group's `/dev/vfio/N`, and the process needs no root.
  Without an IOMMU, physically: write access to the function's files under
  `/sys/bus/pci`, which root has, on a kernel that is not locked down.
- **The GPU detached from its kernel driver.** `Rig_amd_pci.detach i` and
  `Rig_nv_pci.detach i` unbind it and size its memory BAR; `attach i` gives it
  back. Both need `CAP_SYS_ADMIN` and persist after the process.
- **The firmware.** Each image is pinned by its BLAKE2b-256 digest to
  linux-firmware commit `0a6871b1`. rig downloads nothing; two scripts fill a
  directory with the pinned images:

  ```bash
  uv run dev/rig/lib/amd/pci/gen/fetch.py DIR   # for rig.amd.pci
  uv run dev/rig/lib/nv/pci/gen/fetch.py DIR    # for rig.nv.pci
  ```

  The R9700, for one, boots with `psp_14_0_3_sos.bin`, `smu_14_0_3.bin`,
  `sdma_7_0_1.bin` and `gc_12_0_1_{pfp,me,mec,imu,rlc}.bin`. NVIDIA GPUs boot
  with GSP firmware 570.144.
- **Host memory the GPU may write.** Without an IOMMU the GPU writes physical
  addresses and nothing confines it. rig then gives it only memory from 2 MiB
  huge pages under `/dev/hugepages`, which the system must reserve
  (`vm.nr_hugepages`), and maps none of the program's own memory for it. An
  IOMMU is the safer setup: the GPU reaches only what the process maps. An NV
  boot also takes about 64 MiB for the GSP, which behind an IOMMU counts
  against `RLIMIT_MEMLOCK`.

The process owns the GPU until it stops the device. An AMD GPU stopped
cleanly opens again with a partial boot of its compute and copy blocks. One
a dead process left running is reset by the next open. One booted by its
kernel driver opens after `Rig_amd_pci.reset i`. An NVIDIA GPU opens again after `Rig_nv_pci.reset i`.

### What has run on hardware

- **`rig.amd.pci` on a Radeon AI PRO R9700** (GC 12.0.1, taken physically, no
  IOMMU): detach, reset and attach; a boot through the security processor,
  every firmware image, the power manager, the memory hubs, the interrupt
  rings, the compute and copy engines, clocks and gating, all with bus
  mastering off. The GPU has not yet run a submission on this path. Its other
  GPUs (GC 9.4.3 and 9.5.0, the MI300 and MI350 series; GC 11.0.0, 11.0.2 and
  12.0.0) have not booted here.
- **`rig.nv.pci`** is written for Ampere, Ada and Blackwell chips and has
  never booted a GPU: no NVIDIA machine here gives root.
- **Virtual functions and XGMI fabrics** on AMD Instinct GPUs are written and
  never run on hardware.

## Kernel-driver paths

Where a kernel driver holds the GPU, rig talks to it directly. The AMD and
NVIDIA paths still need no user-space library from the vendor.

| Path | Library | Needs | Use it when |
|---|---|---|---|
| AMD through `amdgpu` | `rig.amd.amdgpu` | `/dev/kfd` and the GPU's render node (the `render` group) | the GPU is shared, drives a display, or root is out of reach |
| NVIDIA through `nvidia` | `rig.nv.nvidia` | the kernel driver, releases 570, 580, 610 or 615 | the GPU is shared or root is out of reach, on Linux |
| CUDA | `rig.cuda` | `libcuda`, loaded at the first call | Windows, or a driver release `rig.nv.nvidia` refuses |
| Metal | `rig.metal` | macOS 15 or later | the Mac's GPU |

## Measured

Each table times the same work on one machine through rig and through the
vendor's runtime and the frameworks installed there. Kernels are the smallest
each runtime launches, one thread adding one element or doing nothing; every
row ends by waiting for the work.

### NVIDIA RTX 5000 Ada

Intel Core Ultra 5 235, driver 615.71.09; PyTorch 2.14.1+cu130, JAX 0.11.2.
"CUDA from C" is the CUDA driver API called from C.

| Work | rig CUDA | rig NV | CUDA from C | PyTorch | JAX |
|---|---|---|---|---|---|
| One kernel, launch and wait | 4.89 µs | 4.77 µs | 4.77 µs | 5.62 µs | 16.3 µs |
| A step of 64 dependent kernels, replay and wait | 82.7 µs; 35.7 µs as a CUDA graph | 42.9 µs | 35.6 µs (`cuGraphLaunch`) | 53.1 µs (CUDA graph) | 72.4 µs (jit) |
| Each further kernel in a step | 1.24 µs | 0.61 µs | | 0.57-0.77 µs (CUDA graph) | 0.77-0.91 µs |
| A one-kernel step, 100 in flight, per step | 3.01 µs (a) | 2.54 µs (a) | | 1.97 µs (CUDA graph) | |
| 256 MiB host to GPU / back, pinned | 25.3 / 19.1 GB/s | 25.2 / 19.0 GB/s | | 25.2 / 19.0 GB/s | 22.4 / 17.2 GB/s |
| 256 MiB host to GPU / back, pageable | 19.1 / 15.5 GB/s | 18.8 / 15.4 GB/s | | 18.4 / 12.7 GB/s | 11.8 GB/s / no API |
| 256 MiB GPU to GPU | 1.11 ms | 4.37 ms (copy engine) | | 1.11 ms | 1.71 ms |
| Allocate and free 64 KiB / 64 MiB | 57.9 µs / 1.14 ms | 45.6 µs / 1.25 ms | | 1.03 µs (cached) / 1.13 ms | |

(a) Each step waited for the step before last. With three steps in flight,
rig's bench reads 2.92 µs (CUDA) and 1.71 µs (NV) a step; PyTorch was not
measured that way.

### AMD Radeon AI PRO R9700

Intel i9-9900K, PCIe 3.0 x16; rig through `amdgpu`; PyTorch 2.14.1+rocm7.2,
and HIP called from C through that wheel's `libamdhip64`. JAX's ROCm plugin
did not load, so it has no column.

| Work | rig AMD | HIP | PyTorch |
|---|---|---|---|
| One kernel, launch and wait | 16.1 µs | 30.0 µs | 42.2 µs |
| A step of 64 dependent kernels, replay and wait | 48.6 µs | 191 µs (graph) | 245 µs (graph) |
| Each further kernel in a step | 0.52 µs | 2.57 µs (graph) | 3.17 µs (graph) |
| A one-kernel step, 100 in flight, per step | 15.1 µs (a) | 11.3 µs (graph) | 11.9 µs (graph) |
| 256 MiB host to GPU / back, pinned | 13.7 / 13.9 GB/s | 13.9 / 14.0 GB/s | 13.8 / 13.9 GB/s |
| 256 MiB host to GPU / back, pageable | 11.0 / 11.1 GB/s | 9.9 / 8.6 GB/s | 10.6 / 10.1 GB/s |
| 256 MiB GPU to GPU | 2.70 ms (copy engine) | 0.94 ms | 0.95 ms |
| Allocate and free 64 KiB / 64 MiB | 413 / 373 µs | 0.94 (cached) / 505 µs | 2.1 µs (cached) / 497 µs |

(a) Each step waited for the step before last, which lets this GPU's queue
run dry and pay its wake-up on every step. With three steps in flight, rig's
bench reads 2.15 µs a step; HIP and PyTorch were not measured that way.

### Apple M1 Max

PyTorch 2.14.1 on MPS; JAX 0.4.34 with jax-metal 0.1.1, the newest pair
that runs together. Measured under load; the GPU's wake-up takes most of a launch and wait.

| Work | rig Metal | PyTorch | JAX |
|---|---|---|---|
| One kernel, launch and wait | 224 µs | 225 µs | 267 µs |
| A step of 64 kernels, replay and wait | 607 µs | | 595 µs |
| Each further kernel in a step | 6.1 µs | | 4.7 µs |
| One-kernel submissions in flight, per submission | 98 µs | | |
| 100 kernels launched one by one, per kernel | | 6.9 µs | 54 µs |
| Allocate and free 64 KiB / 64 MiB | 4.4 / 9.7 µs | 1.2 / 40 µs | |

rig is slower where these tables show it: copies between two buffers of one
GPU run on its copy engines; every allocation goes to the driver, with no
cache in front; with two steps in flight, a one-kernel step costs more than a
recorded graph's replay; and Metal pays about 100 µs for each submission in flight.

Two measurements without another runtime beside them: a 256 MiB file loads
into GPU memory in 33.3 ms on the RTX 5000 Ada and 33.2 ms on the R9700, about
the time to read it into host memory (32.8 and 28.8 ms); opening and stopping
a GPU through its kernel driver takes 12.1 ms (NV) and 18.1 ms (AMD).

None of these ran on a driver-less path. That path writes the same rings and
packets as its vendor's kernel-driver path, so per-kernel costs should carry
over; a submission's round trip and allocation could differ.

## Libraries

| Library | Role |
|---|---|
| `rig` | Devices, buffers, submissions, timelines and profiles |
| `rig.host` | ELF programs linked and called on the host's cores |
| `rig.elf` | ELF objects read into their images |
| `rig.pool` | The host's worker threads, in C |
| `rig.disk` | Files as a device's memory |
| `rig.metal`, `rig.cuda` | The Mac's GPU, and NVIDIA GPUs through CUDA |
| `rig.amd` | AMD GPUs, opened by `rig.amd.amdgpu` or `rig.amd.pci` |
| `rig.nv` | NVIDIA GPUs, opened by `rig.nv.nvidia` or `rig.nv.pci` |
| `rig.*.abi` | Each vendor's formats: code objects, packets, launch descriptors |
| `rig.pci` | A machine's PCI functions: taking one, its BARs, DMA memory, page tables |
| `rig.edge` | The C interface, `rig_edge.h`, through which work reaches a driver |
| `rig.remote` | Another machine over TCP: interface only, not implemented |

A program links only the drivers it uses; `rig` links none.

Each `.mli` documents its library, and `rig.mli` opens with the model:
devices, timelines, points and how memory returns. `doc/hardware-notes.md`
holds what the GPUs do that their documentation does not say, and
`doc/testing.md` how the suites are built. `examples/` goes from host buffers
to kernels on each GPU; the `x-` examples need hardware.

## Tests and benches

```bash
dune build @dev/rig/runtest   # every suite; GPU suites skip without their GPU
dune build @dev/rig/bench     # every bench, against its machine's baseline
```

Baselines are per machine, in `bench/**/*.thumper`.
