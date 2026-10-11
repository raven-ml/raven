# Benchmarks

This directory holds three thumper suites, `bench_rig`, `bench_gpu` and
`bench_profile`, and each directory under it except `compare/` holds more. A
suite's rows time calls a program makes. A run judges each row against its
baseline for this machine, in the suite's `*.thumper` file. `compare/` measures
rig against the vendor's toolkit and the frameworks, and `remote/lan.sh` times
a job between two machines.

## What a timing needs

A timing needs the GPU to itself and a quiet machine. Within one build, the
suites take turns on the GPU through dune's lock `%{workspace_root}/gpu`.
The lock orders the suites alone, so other programs on the GPU must stop
first.

Time release builds only, since the dev profile's `-opaque` removes
inlining across modules. A dune server running in the tree takes a build and
ignores its `--profile`, so time in a tree without one.

## Commands

From the repository root:

| What | Command |
|---|---|
| Every suite | `dune build --profile release @packages/rig/bench` |
| One directory's suites, such as `amd` or `amd/pci` | `dune build --profile release @@packages/rig/bench/DIR/bench` |
| The comparison with the vendor's toolkit, PyTorch, JAX and tinygrad | `uv run packages/rig/bench/compare/run.py SCENARIO` |
| Copies and a round trip between two machines | `packages/rig/bench/remote/lan.sh AGENT TREE` |

A GPU suite has no rows on a machine without its GPU. The `amd/pci` suite
parses an AMD GPU's firmware, which lives outside the repository. Fetch it
into rig's firmware directory with
`dune exec packages/rig/bin/main.exe -- firmware amd`. Without the images the
suite skips. With `RIG_AMD_PCI_FIRMWARE=DIR` set, DIR holding the images,
the `amd` suite opens GPU 0 with no kernel driver. That needs GPU 0
detached from amdgpu by an administrator, and root or VFIO access to it, as
[../doc/06-gpus-without-a-driver.md](../doc/06-gpus-without-a-driver.md#preparing-the-machine)
explains. On a machine whose GPU 0 is still attached, build `amd/pci` alone.
The `amd` suite records its driver-less rows under the machine's key with
`-pci` appended, as the head of `amd/bench_amd.ml` says.

[compare/README.md](compare/README.md) says how to run the comparison. The
comment at the head of `remote/lan.sh` says what a two-machine run needs.
lan.sh builds its executables in release on both machines.

## A machine's first baseline

On a machine without a baseline, a suite's rows are new and the run
compares nothing. Record them with the `bless` subcommand of the suite's
executable. Run it from the suite's source directory, since a suite reads
its fixtures by paths relative to the working directory. `bless` records the
rows in this machine's section of the suite's `*.thumper` file. It refuses
while the machine's load average is half a task per core or more.

```sh
cd packages/rig/bench/cuda
dune exec --profile release ./bench_cuda.exe -- bless
```
