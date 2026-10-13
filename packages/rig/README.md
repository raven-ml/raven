# rig

rig runs work on GPUs from OCaml, and depends on nothing outside the OCaml
distribution.

rig talks to NVIDIA's and AMD's kernel drivers itself, so a program needs
no CUDA or ROCm, and on Linux it opens AMD and NVIDIA GPUs with no kernel
driver. A program allocates memory on a device and submits work that uses
it. rig orders that work by the memory it reads and writes, so a program
never names an event or a fence. Another machine's GPUs open through
`rig agent` running there, and take the same calls as this machine's.

## A first program

This program copies five bytes to a *memory device*, a device whose memory
is the host's and whose work runs before the call that submits it returns,
and back again. It runs on any machine:

```ocaml
# open Rig;;
# let () =
    let d = memory_device "M" in
    let on_d = Buffer.create d 5 in
    Buffer.blit_from_string "hello" 0 on_d 0 5;
    let b = Bytes.create 5 in
    Buffer.blit_to_bytes on_d 0 b 0 5;
    print_endline (Bytes.to_string b);;
hello
```

The blit back reads `on_d`, so rig starts it after the first blit has
written it. Whatever reads memory follows every device's last write of
it, and whatever writes memory follows every use (the Order section of
[`rig.mli`](lib/rig.mli)).

On a GPU, only the line that opens the device changes. On a Mac:

<!-- $MDX skip -->
```ocaml
let d = Result.get_ok (Rig.open_ (Rig_metal.gpu 0))
```

with `rig.metal` added to the executable's libraries. The opener is the
only line that names a GPU and the path to it, and the table below lists
them. [The guide](doc/index.md) runs a kernel on a GPU.

## Why does rig exist?

Running a model on a GPU today means installing a vendor's stack: PyTorch
for CUDA alone is 5.5 GB, most of it NVIDIA's libraries, and a full ROCm
install is over 26 GB, at versions that must agree with each other and with
the kernel. None of it fits in one person's head, and all of it must be
installed wherever the program runs.

We want raven to own the whole path from a model to the GPU, and to keep
that path small enough to read. rig is its foundation. It does the work of
AMD's runtime, about 300,000 lines, and of NVIDIA's closed `libcuda`, in
about 37,000 lines of OCaml and C, and on Linux it boots their GPUs with no
kernel driver at all. Its interfaces are written to read like a book. A
reader who starts with `rig.mli` and follows one driver learns how a GPU
runs their work, from the call that submits it to the moment its memory
comes back.

## What does it run on?

rig has run on three machines: an M1 Max Mac, a Linux PC with an NVIDIA
RTX 5000 Ada (kernel driver 615), and a Linux PC with an AMD Radeon AI PRO
R9700 (gfx1201). No test machine runs Windows. The core, host, pool, file
and CUDA code has paths for Windows that have never run there. A GPU's opener
goes to `Rig.open_`.

| Device | Open with | Status |
|---|---|---|
| The host's cores | `Rig.host` | tested on all three machines |
| Files | `Rig_disk.of_file` | tested on all three machines |
| Apple GPUs, through Metal | `Rig_metal.gpu 0` | tested on the M1 Max |
| NVIDIA GPUs, through CUDA | `Rig_cuda.gpu 0` | tested on the RTX 5000 Ada |
| NVIDIA GPUs, through NVIDIA's kernel driver | `Rig_nv_nvidia.gpu 0` | tested on the RTX 5000 Ada with release 615; releases 570, 580 and 610 untested |
| NVIDIA GPUs, with no kernel driver | `Rig_nv_pci.gpu 0` | written, never run on hardware |
| AMD GPUs, through `amdgpu` | `Rig_amd_amdgpu.gpu 0` | tested on the R9700 |
| AMD GPUs, with no kernel driver | `Rig_amd_pci.gpu 0` | tested on the R9700 |
| AMD GPUs other than gfx1201 | either AMD opener | untested |
| AMD Instinct virtual functions and XGMI | `Rig_amd_pci.gpu` | written, never run on hardware |
| Two GPUs of one machine reaching each other's memory | the AMD, NVIDIA and CUDA openers | untested |
| Other machines | `Rig_remote.connect`, `rig run`, `rig agent` | tested between the RTX 5000 Ada and R9700 machines |
| ConnectX NICs | `Rig_mlx5_uverbs.open_` | run against Linux's Soft-RoCE only, never a real NIC |
| Broadcom NetXtreme NICs, with no kernel driver | `Rig_bnxt.open_` | written, never run on hardware |

## When should you not use rig?

- You want to write GPU kernels. rig loads code compiled for the device,
  by nvcc, hipcc, Metal's compiler or raven's compiler, tolk, and runs it.
- You need the GPU for other programs while yours runs. A driver-less open
  holds the GPU for its process alone.
- You want a GPU without its kernel driver on macOS or Windows. The
  driver-less paths run on Linux only.
- You wait on each small step of work on a Mac. Each wait pays Metal's
  round trip, about 100 µs from commit to the start of an idle GPU and about
  95 µs from the end of a command buffer to the call that tells the host.
  Work no one waits for is batched, at about 2.5 µs a submission.
- Your machines share a network with people you do not trust. A job across
  machines checks a shared key when it connects, and authenticates or
  encrypts nothing after that.

## Where next

- [The guide](doc/index.md) installs rig, explains each call of a first
  program, and runs a kernel on a GPU. Its pages trace the rules at work,
  such as one submission from `submit` to the hardware and back.
- The interfaces, in reading order: [`rig.mli`](lib/rig.mli), then
  [`rig_edge.mli`](lib/edge/rig_edge.mli), the contract a driver
  implements, then one driver, such as
  [`rig_metal.mli`](lib/metal/rig_metal.mli).
- [The examples](examples/README.md) teach one idea each. The numbered ones
  run on any machine.
- [The hardware notes](doc/hardware-notes.md) record what the GPUs do that
  their vendors' documentation does not say.
- [The tests](test/README.md) and [the benches](bench/README.md) are for
  those who change rig.

## Credits and licence

rig learned to drive AMD and NVIDIA GPUs and Broadcom NICs from tinygrad's
runtimes, driver-less AMD path included. Its register, packet and interface
layouts are excerpts that generators copy from pinned commits of AMD's
`amdgpu` and `amdkfd` kernel drivers, ROCm's runtime and profilers, AMD's
PAL, NVIDIA's open kernel modules and open-gpu-doc, Linux's nouveau, mlx5
and bnxt_re headers, rdma-core, DPDK's bnxt driver, and LLVM. A
driver-less open boots the GPU with firmware from linux-firmware. rig
ships no firmware, and `rig firmware` fetches the images it pins by
digest.

rig is distributed under the ISC licence, as raven is. Each generated file
names its sources and carries their licences. Code derived from tinygrad
keeps its MIT licence in [`LICENSE-tinygrad`](LICENSE-tinygrad), the
excerpts of LLVM keep theirs in [`LICENSE-llvm`](LICENSE-llvm), and the
copy kernels in `lib/amd/kernel` and `lib/nv/kernel` keep theirs beside
them.
