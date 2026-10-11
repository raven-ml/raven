# Comparison with the vendor's toolkit and frameworks

`run.py` measures rig against the GPU vendor's toolkit and against PyTorch,
JAX and tinygrad on one machine's GPU, and writes the raw results to
`results/`. On a machine with a Metal, NVIDIA or AMD GPU, run from the
repository root:

```sh
uv run packages/rig/bench/compare/run.py SCENARIO \
  [--torch PYTHON] [--jax PYTHON] [--tinygrad PYTHON] [FILE...]
```

`SCENARIO` is `ship`, `start`, `load`, `fit` or `micro`. Each `PYTHON` is an
interpreter that has the framework installed, and a framework without one is
left out. Each scenario prints a table and writes its record into
`results/GPU-DATE.json`, where `GPU` is the GPU's name in lower case with its
words joined by hyphens, and `DATE` is the UTC date as `YYYYMMDD`. The record
replaces that scenario's record of the same day. In the printed table, `x rig`
is a row's median divided by the least median among rig's rows of the same
case.

## What a run needs

- The GPU to itself and a quiet machine, as every timing does
  ([../README.md](../README.md#what-a-timing-needs)).
- A tree without a running dune server. `run.py` builds `rig/rig.ml` with
  `--profile release`, and stops if a server would take the build.
- On NVIDIA, `nvidia-smi` on `PATH`, and `CUDA_HOME` (default
  `/usr/local/cuda`) holding `bin/ptxas`, `include/cuda.h` and
  `lib64/stubs/libcuda.so`.
- Each framework in an environment of its own, on a Python 3.12 that `uv`
  installed, since ship counts the environment with its Python and refuses
  the system's. The recorded runs used these versions:

  ```sh
  uv venv --python-preference only-managed -p 3.12 torch
  uv pip install -p torch torch==2.14.1
  uv venv --python-preference only-managed -p 3.12 jax
  uv pip install -p jax 'jax[cuda12]==0.11.2'
  uv venv --python-preference only-managed -p 3.12 tinygrad
  uv pip install -p tinygrad tinygrad==0.14.0
  ```

  On AMD, PyTorch comes from its ROCm index
  (`--index-url https://download.pytorch.org/whl/rocm7.2`). On the Mac, JAX
  is `jax==0.4.35 jaxlib==0.4.35 jax-metal==0.1.1`. Pass
  `--torch torch/bin/python`, and so on.
- On AMD, a gfx1201 GPU, since the AMD image of `add` is built for
  gfx1201 alone, and `lspci`, which names the GPU in the record. `ROCM_PATH`
  (default `/opt/rocm`) must hold `include/hip/hip_runtime_api.h` and
  `lib/libamdhip64.so`, with its soname link. Without `/opt/rocm`, a ROCm
  build of PyTorch carries HIP, and `ROCM_PATH` can name a directory of
  links into it, made after the PyTorch environment above:

  ```sh
  w=torch/lib/python3.12/site-packages
  mkdir -p rocm/lib
  ln -s "$PWD/$w/triton/backends/amd/include" rocm/include
  ln -s "$PWD/$w"/torch/lib/* rocm/lib/
  ln -s libamdhip64.so rocm/lib/libamdhip64.so.7
  export ROCM_PATH=$PWD/rocm
  ```
- `cc`, which on the Mac comes with the Xcode command-line tools.
- For ship, an unset `PYTHONPATH`.
- For load, the files `FILE...`, which may hold any bytes. The recorded
  runs load one file of 256 MiB, and of 100 MB on the Mac.
  `head -c 268435456 /dev/urandom > ckpt.bin` makes one of 256 MiB.

## Stacks

Each stack runs in a process of its own, the stack's *probe*. A probe that
fails stops the run before anything is written.

- **rig** runs `rig/rig.ml` through each kernel driver of the machine:
  `metal`, `cuda` and `nv`, or `amd`.
- **native** runs `native.c` through the vendor's toolkit: CUDA's driver
  API, HIP or Metal. `run.py` compiles it with `cc`.
- **torch**, **jax** and **tinygrad** run `frameworks.py` with each
  framework's interpreter.

rig and native run the same kernel, `add`, compiled ahead of time, from
the binaries of the [`01-add`](../../examples/01-add/kernels/) example. On NVIDIA
`run.py` compiles its PTX for the machine's GPU before the probes
start. The probes of rig
and native compile nothing and load no math library. The frameworks run
their own addition.

## Scenarios

Each scenario but ship runs 5 rounds, and each round runs every stack in
turn.

**ship** counts the bytes installed to run a program on the stack. These
are the stack's own files and, on Linux, the GPU vendor's libraries that it
loads from outside them during one checked addition. A stack's own files are
rig's or the native probe's executable, or a framework's environment with
its Python. A loaded library is the vendor's when its file name starts with
`libcuda`, `libnv`, `libamdhip`, `libhsa`, `libhiprtc`, `libamd_comgr` or
`libroc`. Other loaded libraries belong to the system and are not counted.
On AMD they include the copies of `libdrm_amdgpu`, `libelf` and `libnuma`
that a HIP from a PyTorch wheel loads. Every probe starts with empty
compiler caches, tinygrad's `CACHEDB` and CUDA's `CUDA_CACHE_PATH`, so a
stack that compiles its kernel loads its compiler, and the compiler's
libraries are counted. The Mac's system cache of Metal shaders is kept.
Metal is part of macOS, so on the Mac every stack's vendor bytes are 0.

The number is the size of a program's install, apart from the system's
libraries. PyTorch's and JAX's rows hold kernel compilers and math libraries,
and tinygrad's holds its compiler. HIP loads its compiler library,
`libamd_comgr`, so native's AMD row counts it. The number leaves out the
toolchain that builds the program and the model's weights.

**start** times a process from just before it is spawned to the completion
of its first addition of 64 floats. The probe checks the sum after the
timer stops. An untimed round first fills the compiler caches.

The number is the wait before a fresh process has its first GPU result, as for
a command-line tool started per request. The compiler caches are warm and the
kernel is one addition, so the number leaves out compiling a real model's
kernels and loading its weights.

**load** times copying each `FILE` into a device buffer of its own, from
opening the files to the completion of the last copy. The probe compares
every byte after the timer stops. On Linux each round takes a sample with
the files out of the page cache (cold) and one with them in it (warm). The
Mac takes warm samples only. Beside the stacks, the `read` row reads the
files into host memory through a 4 MiB buffer. On Linux, the `floor` row
times one copy of 256 MiB from pinned host memory to the device through the
vendor's toolkit. A copy from a file crosses the same link.

The number is the time to put a checkpoint's bytes on the GPU. It leaves
out parsing a format, converting dtypes and splitting a model across
devices.

**fit** reads the memory the system charges to the process once its first
addition has completed, while the process still runs. A runtime that sets
itself up at its first allocation or launch has done so by then. On NVIDIA
it is the process's GPU memory as `nvidia-smi` reports it, in whole MiB. On
AMD it is the process's VRAM as KFD counts it. The Mac's memory is unified,
so there it is the process's whole physical footprint
(`ri_phys_footprint`), the interpreter's included.

The number is the GPU memory a runtime holds before a model's weights
arrive. Weights and KV cache cannot use it. By default JAX reserves 75% of
an NVIDIA GPU's memory when it starts, and its NVIDIA row shows that
reservation. The number leaves out memory a runtime takes later, such as a
math library's workspace at its first matrix product or a caching
allocator's pool.

**micro** times 1, then 64, additions of 64 floats, each adding to the
previous sum, and a wait for the last. Each process takes 5 samples after 3
warmups, each after an idle of 20 ms.

The number is the time per launch of a chain of small dependent kernels, the
host's submission and the GPU's run together. A step of short kernels launched
one at a time pays about this much for each. A framework that captures a whole
step (PyTorch through CUDA graphs, TinyJit, a `jax.jit` of the step) submits it
once, so micro does not predict tokens per second. Every stack runs the same
small addition, so the number leaves out kernel speed. The frameworks run
eagerly, one dispatch per addition: PyTorch's operator, a `jax.jit` of one
addition, a tinygrad `realize`. rig replays two prepared submissions. The
host's share of a step is timed alone by the `step/remote-1900` row of the
`program` suite (`../program/bench_program.ml`), which makes 1,880 submissions
to a memory device, whose fills run on the host, and hands two runs to a
`rig agent` over loopback.

## Results

The table gives the medians of release builds run on 2026-10-11. NVIDIA is an
RTX 5000 Ada Generation, where rig and tinygrad each run through their own
driver (`nv`, `NV`) and through CUDA, in that order. AMD is a Radeon AI PRO
R9700, where PyTorch is the only framework run. A micro figure is the
64-addition case's time per call, and a load figure is the warm case of one
256 MiB file.

| | rig | native | torch | jax | tinygrad |
|---|---|---|---|---|---|
| ship, NVIDIA | 7.6 / 115 MiB | 107 MiB | 5,458 MiB | 5,774 MiB | 461 / 496 MiB |
| ship, AMD | 7.6 MiB | 187 MiB | 15,045 MiB | | |
| start, NVIDIA | 70 / 134 ms | 123 ms | 720 ms | 538 ms | 336 / 279 ms |
| start, AMD | 53 ms | 67 ms | 1,106 ms | | |
| fit, NVIDIA | 123 / 312 MiB | 312 MiB | 364 MiB | 24,546 MiB, reserved | 144 / 312 MiB |
| fit, AMD | 29 MiB | 142 MiB | 147 MiB | | |
| micro, NVIDIA | 2.1 / 3.2 µs | 2.6 µs | 7.7 µs | 28.9 µs | 511 / 447 µs |
| micro, AMD | 4.9 µs | 6.4 µs | 8.1 µs | | |
| load, NVIDIA | 46.0 / 49.3 ms | 27.0 ms | 32.9 ms | 169.4 ms | 62.0 / 99.9 ms |
| load, AMD | 38.9 ms | 62.1 ms | 215.9 ms | | |

The Mac (Apple M1 Max) has no records yet: its runs wait for a quiet
machine.

## Records

`results/GPU-DATE.json` holds the `date`, the `machine` and one record per
scenario run that day, under `scenarios`. A path under the home directory
is relative to the working directory, or starts with a framework's name
when it lies in that framework's environment.

| Field | Meaning |
|---|---|
| `machine.gpu` | the GPU's name, with its gfx target on AMD, or the chip's name on the Mac |
| `machine.driver` | the version of NVIDIA's driver, of amdgpu's module when it states one and of the Linux kernel otherwise, or of macOS |
| `machine.system` | the kernel's name, release and architecture |
| `utc` | when the scenario started |
| `commit` | the checkout's commit, with `+dirty` when `packages/rig` differs from it outside `results/` |
| `profile` | the dune profile of rig's probe, `release`, or `dev` in records made before `run.py` built it in release |
| `load_average` | the 1, 5 and 15 minute load averages when the scenario started |
| `rounds` | 1 for ship, 5 for the others |
| `unit` | the samples' unit, `ns` or `bytes` |
| `rows` | one per stack, device and case |

Each row holds these fields:

| Field | Meaning |
|---|---|
| `stack` | `rig`, `native`, `torch`, `jax`, `tinygrad`, `read` or `floor` |
| `device` | rig's driver, the native toolkit, the framework's name for the GPU, `host` for read, or `pcie` for the floor |
| `version` | rig's commit or the framework's version, and `null` for native, read and the floor |
| `samples` | the samples; a micro sample times all the calls of its case, and ship's one sample is the total |
| `case` | load's `cold` or `warm`, and `null` for the floor; micro's `add1` or `add64` |
| `bytes` | load's bytes moved |
| `installed`, `vendor` | ship's bytes of the stack's own files and of the vendor's libraries |
| `vendor_libraries` | ship's vendor libraries, each with its `path`, `bytes` and `sha256` |
| `os_libraries` | ship's paths of the other loaded libraries, which are not counted |
