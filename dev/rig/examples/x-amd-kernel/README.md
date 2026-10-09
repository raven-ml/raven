# `x-amd-kernel`

**Needs an AMD GPU of processor gfx1201, such as a Radeon AI PRO R9700, held by
Linux's `amdgpu` driver.** Elsewhere it prints a line and exits.

This example loads a code object and launches its kernel `add` over a million
floats: a submission of one `Launch` part, whose parameters are the addresses
of three arrays, each a ref to a buffer the submit passes, and a run that holds
the launch's grid, its groups and the offsets into the arrays.

```bash
cd dev/rig/examples/x-amd-kernel
dune exec ./main.exe
```

## What You'll Learn

- Loading code: `Image.load` places the code object's image in the GPU's memory
- Launching a kernel: `Submission.Launch`, its parameter bytes and its refs
- Storing a launch's grid, groups and parameters into a run:
  `Submission.block`, `Run.groups`, `Run.threads`, `Run.int64`
- Which devices launch: `queues` and the kinds of work each runs

## Key Functions

| Function                               | Purpose                                      |
| -------------------------------------- | -------------------------------------------- |
| `Image.load d binary`                  | Load a code object on a device               |
| `Launch { image; kernel; params; refs }` | A part that runs a kernel once            |
| `Submission.block s i`                 | Where part [i]'s grid and parameters lie in a run |
| `Run.int64 run b at v`                 | Store a parameter, a ref's offset at a ref's [at] |

## The code object

`add_gfx1201.hsaco` is `add.cl` compiled in this directory with Homebrew clang
22.1.7 (`clang`) and Homebrew LLD 21.1.8 (`ld.lld`):

```bash
clang -c -x cl -cl-std=CL2.0 -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
  -mcode-object-version=5 -nogpulib -O2 add.cl -o add.o
ld.lld -shared add.o -o add_gfx1201.hsaco
rm add.o
```
