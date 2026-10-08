# `x-cuda-kernel`

**Needs an NVIDIA GPU and the CUDA library (`libcuda`), which NVIDIA's driver
installs.** Elsewhere it prints a line and exits.

A kernel reaches a CUDA device through a fill, `run.c`, that calls
`cuLaunchKernel` on the stream the device hands it. The fill finds that function
through the device's capability, so nothing links a CUDA library or needs its
headers. This example loads a PTX kernel, adds two arrays of a million floats as
a step (its argument in a hold, its arrays passed to each submit) and checks the
result.

```bash
cd dev/rig/examples/x-cuda-kernel
dune exec ./main.exe
```

## What You'll Learn

- Loading PTX text, which CUDA compiles for the GPU it loads on:
  `Image.load`, `Image.entry`
- What compiled code finds in a CUDA device's capability:
  `Rig.capability g Rig_cuda_abi.key`, its `symbol`
- A fill's argument in pinned host memory: `Buffer.create ~memory:Pinned`
- A copy back that waits for the kernel's write

## Key Functions

| Function                          | Purpose                                    |
| --------------------------------- | ------------------------------------------ |
| `Image.load g ptx`                | The module of `ptx` on `g`                 |
| `cap.symbol "cuLaunchKernel"`     | A CUDA function's address                  |
| `Submission.Fill { fill; arg; _ }`| The launch, as C work on the stream        |

## The kernel

`add.ptx` is written by hand for the virtual architecture `sm_50` (PTX ISA 7.0).
CUDA compiles it for any GPU of compute capability 5.0 or later, with a driver
of CUDA 11.0 or later.
