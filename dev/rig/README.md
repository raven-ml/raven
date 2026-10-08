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
kernel driver opens after `Rig_amd_pci.reset i`. An NVIDIA GPU opens again
after `Rig_nv_pci.reset i`.

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

### Against the vendor's own calls

Each row times what a program calls, beside a floor measured on the same
machine: the vendor's own call made from C, or the same work handed to the
driver with rig's core left out. Kernels are empty and run on one thread;
every row waits for its work.

| Work | rig | Floor | Machine |
|---|---|---|---|
| One kernel, launch and wait, CUDA | 5.00 µs | `cuLaunchKernel`: 4.77 µs | RTX 5000 Ada |
| One kernel, launch and wait, `rig.nv.nvidia` | 4.75 µs | `cuLaunchKernel`: 4.77 µs | RTX 5000 Ada |
| A CUDA graph of 64 kernels, launch and wait | 35.7 µs | `cuGraphLaunch`: 35.6 µs | RTX 5000 Ada |
| One kernel, launch and wait, `rig.amd.amdgpu` | 15.8 µs | the same packets on a KFD queue: 15.5 µs | R9700 |
| 100 one-kernel steps, three in flight, per step: CUDA / NV / AMD | 2.92 / 1.71 / 2.15 µs | the driver's C entries alone: 2.80 / 1.72 / 2.19 µs | both |
| 256 MiB host to GPU / back | 25.1 / 19.0 GB/s | `cuMemcpy` from pinned memory: 25.2 / 18.9 GB/s | RTX 5000 Ada |
| 256 MiB host to GPU / back | 13.9 / 14.0 GB/s | the driver's DMA from pinned memory: 13.9 / 14.0 GB/s | R9700 |
| A 256 MiB file into GPU memory, page cache warm | 33.3 ms (NV), 33.2 ms (AMD) | the file read into host memory: 32.8, 28.8 ms | both |
| Open and stop a GPU through its kernel driver | 12.1 ms (NV), 18.1 ms (AMD) | | both |
| One kernel, launch and wait, Metal | 254 µs | Metal's own indirect command buffer: 275 µs | M1 Max |

The RTX 5000 Ada sits in an Intel Core Ultra 5 235 with driver 615.71.09; the
R9700 in an Intel i9-9900K on PCIe 3.0 x16; the Mac is an M1 Max, measured
under load, where the GPU's wake-up takes most of a launch.

### Against PyTorch and JAX

The same work through rig and through the frameworks, on the same machines,
one session per machine. PyTorch 2.14.1 (cu130, rocm7.2, MPS), JAX 0.11.2 on
CUDA, and JAX 0.4.34 with jax-metal 0.1.1 on the Mac, the newest pair that
runs together. JAX's ROCm plugin did not load. HIP is called from C through
the `libamdhip64` of PyTorch's ROCm wheel. Kernels are the smallest each
runtime launches.

| Work | rig | HIP | PyTorch | JAX | Machine |
|---|---|---|---|---|---|
| One kernel, launch and wait | CUDA 4.89 µs, NV 4.77 µs | | 5.62 µs | 16.3 µs | RTX 5000 Ada |
| A step of 64 dependent kernels, replay and wait | CUDA 82.7 µs, NV 42.9 µs | | 53.1 µs (CUDA graph) | 72.4 µs (jit) | RTX 5000 Ada |
| A one-kernel step, 100 in flight, per step (a) | CUDA 3.01 µs, NV 2.54 µs | | 1.97 µs (CUDA graph) | | RTX 5000 Ada |
| 256 MiB host to GPU / back, pageable memory (b) | CUDA 19.3 / 15.5 GB/s, NV 19.6 / 15.4 GB/s | | 18.4 / 12.7 GB/s | 11.8 GB/s / no API | RTX 5000 Ada |
| 256 MiB GPU to GPU | CUDA 1.11 ms, NV 4.37 ms (copy engine) | | 1.11 ms | 1.71 ms | RTX 5000 Ada |
| Allocate and free 64 KiB / 64 MiB | CUDA 57.9 µs / 1.14 ms, NV 45.6 µs / 1.25 ms | | 1.03 µs (cached) / 1.13 ms | | RTX 5000 Ada |
| One kernel, launch and wait | 16.1 µs | 30.0 µs | 42.2 µs | | R9700 |
| Each further dependent kernel | 0.52 µs | 2.57 µs (graph) | 3.17 µs (graph) | | R9700 |
| A step of 64 dependent kernels, replay and wait | 48.6 µs | 191 µs (graph) | 245 µs (graph) | | R9700 |
| A one-kernel step, 100 in flight, per step (a) | 15.1 µs | 11.3 µs (graph) | 11.9 µs (graph) | | R9700 |
| 256 MiB GPU to GPU | 2.70 ms (copy engine) | 0.94 ms | 0.95 ms | | R9700 |
| Allocate and free 64 KiB / 64 MiB | 413 / 373 µs | 0.94 µs (cached) / 505 µs | 2.1 µs (cached) / 497 µs | | R9700 |
| One kernel, launch and wait (b) | 254 µs | | 225 µs (MPS) | 267 µs | M1 Max |

(a) Each step waited for the step before last. rig's own bench, with three
steps in flight, reads 2.92 µs (CUDA), 1.71 µs (NV) and 2.15 µs (AMD) a step;
the others were not measured that way.
(b) rig's figure is its recorded baseline on the same machine; the
frameworks' come from the comparison session.

rig is slower where these rows show it: copies between two buffers of one
GPU run on its copy engines; every allocation goes to the driver, with no
cache in front; with two steps in flight, a one-kernel step costs more than a
recorded graph's replay; and Metal pays about 100 µs for each submission in
flight.

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
