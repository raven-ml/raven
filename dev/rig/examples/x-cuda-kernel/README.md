# `x-cuda-kernel`

**Needs an NVIDIA GPU and the CUDA library (`libcuda`), which NVIDIA's driver
installs.** Elsewhere it prints a line and exits.

A kernel reaches a CUDA device as a launch: a part that names the image's
function and, for each parameter that holds an address, the buffer it points
into. This example loads a PTX kernel, adds two arrays of a million floats as a
step (its arrays passed to each submit, its grid and parameters stored in a
run) and checks the result.

```bash
cd dev/rig/examples/x-cuda-kernel
dune exec ./main.exe
```

## What You'll Learn

- Loading PTX text, which CUDA compiles for the GPU it loads on: `Image.load`
- A launch whose parameters point into the buffers a submit passes:
  `Submission.Launch`, its `refs`
- A run's grid and parameters: `Submission.block`, `Submission.Run`
- A copy back that waits for the kernel's write

## Key Functions

| Function                                            | Purpose                                 |
| --------------------------------------------------- | --------------------------------------- |
| `Image.load g ptx`                                  | The module of `ptx` on `g`              |
| `Submission.Launch { image; kernel; params; refs }` | The kernel as a part                    |
| `Submission.Run.groups`, `threads`, `int32`         | A launch's grid and parameters in a run |

## The kernel

`add.ptx` is written by hand for the virtual architecture `sm_50` (PTX ISA 7.0).
CUDA compiles it for any GPU of compute capability 5.0 or later, with a driver
of CUDA 11.0 or later.
