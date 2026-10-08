# `x-nv-kernel`

**Needs an NVIDIA GPU of architecture sm_89, such as an RTX 5000 Ada, held by
NVIDIA's Linux kernel driver.** Elsewhere it prints a line and exits.

An NVIDIA GPU's channels run words that compiled code writes. This example loads
a cubin, builds the launch descriptor of its kernel `simple_add` and its
constant bank, writes a segment that schedules the descriptor, and places one
ring entry naming that segment on the compute channel, as a step (the launch's
memory in a hold, the arrays passed to each submit). It checks the result.

```bash
cd dev/rig/examples/x-nv-kernel
dune exec ./main.exe
```

## What You'll Learn

- Reading a cubin and setting a kernel up for the GPU: `Cubin.kernel`,
  `Launch.make`
- What compiled code finds in an NVIDIA device's capability:
  `Rig.capability g Rig_nv_abi.Gpu.key`, the GPU's classes, geometry and
  local memory
- A launch descriptor and constant bank 0: `Qmd.make`, `set_dim`,
  `set_program`, `set_bank`, `Qmd.parameters`, `Structure.encode`
- Channel words: `Method.schedule`, and the ring entry `Gpfifo.entry`
- Work as words: `Submission.Words`

## Key Functions

| Function                                | Purpose                                    |
| --------------------------------------- | ------------------------------------------ |
| `Launch.make gpu k`                     | A kernel set up for launch on a GPU        |
| `Qmd.make l` and its setters            | The launch descriptor                      |
| `Method.schedule addr`                  | Channel words that run a descriptor        |
| `Gpfifo.entry addr ~offset ~words`      | The ring entry of a segment of words       |

## The cubin

`simple_add_sm89.cubin` is `simple_add.cu` compiled by `nvrtc.c` with NVRTC
12.8.93 (CUDA 12.8, the PyPI package `nvidia-cuda-nvrtc-cu12==12.8.93`),
whose files are under `$NVRTC`, on Linux x86_64:

```bash
cc nvrtc.c -I$NVRTC/include -L$NVRTC/lib -l:libnvrtc.so.12 \
  -Wl,-rpath,$NVRTC/lib -o nvrtc
./nvrtc simple_add.cu sm_89 simple_add_sm89.cubin
```

Another release of NVRTC writes its own version into the file.
