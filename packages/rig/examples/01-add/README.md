# `01-add`

A kernel is a function that a device runs once for each thread of a grid.
This example adds two arrays of a million floats with one launch of a kernel
`add`. It opens the first GPU here for which it has a binary of `add`, and
without one the host runs the same launch on its cores.

```bash
cd packages/rig/examples/01-add
dune exec ./main.exe
```

It prints the sum on stdout, the same on every machine, and the device that
ran `add` on stderr. With an argument, the name of one of the paths of
[`Rig_gpus.openers`](../../lib/gpus/rig_gpus.mli), such as `CUDA` or `NV`,
it tries only that path's GPUs.

The examples after this one take the program apart: memory, timelines and
order, steps, the lives of buffers and devices, and host code.

## What You'll Learn

- Opening the first GPU that rig sees here: `Rig_gpus.openers`
- Choosing a binary by the device's architecture: `Rig.arch`
- Loading code: `Image.load`, on a GPU or on `Rig.host`
- Moving arrays to a device and back: `Buffer.create`, `Buffer.of_bigarray`,
  `Buffer.copy`
- A launch whose parameters hold refs to the buffers of a submit:
  `Submission.launch`, `Submission.read`, `Submission.write`
- A launch's grid and scalar parameters in a run: `Submission.block`,
  `Run.groups`, `Run.threads`, `Run.int64`
- A copy back that follows the launch's write, with no wait in the program

## Key Functions

| Function                                  | Purpose                                     |
| ----------------------------------------- | ------------------------------------------- |
| `Rig_gpus.openers ()`                     | An opener for each GPU here, in rig's order |
| `Image.load d binary`                     | `binary` loaded on `d`, or why not          |
| `Submission.launch ~refs image f ~params` | A part that runs `f` over a grid            |
| `Submission.Run.groups r b x y z`         | The grid of the launch whose block is `b`   |
| `submit s ~run ~buffers`                  | Run `s` once and return its point           |

## The kernel

Every binary of `add` takes 32 bytes of parameters: the addresses of `a`,
`b` and `out` at bytes 0, 8 and 16, then the 64-bit length `n` at 24. Each
thread `i` below `n` writes `out[i] = a[i] + b[i]`. Its sources and
binaries are in `kernels/`.

| Source      | Binary                        | Runs on                                                |
| ----------- | ----------------------------- | ------------------------------------------------------ |
| `add.metal` | `add.metallib`                | any Metal GPU, which compiles it as it loads           |
| `add.ptx`   | itself                        | CUDA on `sm_50` and later, which compiles it           |
| `add.ptx`   | `add_sm_89.cubin`             | an `sm_89` GPU, through NVIDIA's kernel driver or CUDA |
| `add.cl`    | `add_gfx1201.hsaco`           | an AMD `gfx1201` GPU                                   |
| `add.c`     | `add_arm64.o`, `add_x86_64.o` | the host's cores                                       |

`kernels/gen.sh` makes each binary from its source, and names the tools and
versions each needs. Dune never runs it, so building and running need only
rig. A new architecture takes one line in `gen.sh` and one case in the
program's `binary`.

## Next Steps

Continue to [02-buffers](../02-buffers/).
