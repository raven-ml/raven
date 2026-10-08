# `x-amd-kernel`

**Needs an AMD GPU of processor gfx1201, such as a Radeon AI PRO R9700, held by
Linux's `amdgpu` driver, with a compute queue that reads PM4 packets.**
Elsewhere it prints a line and exits.

An AMD GPU's queues read packets that compiled code writes. This example loads a
code object, writes the dispatch of its kernel `add` as PM4 words from the
kernel's descriptor, places them on the compute queue as a step (arguments in a
hold, arrays passed to each submit), and checks the result.

```bash
cd dev/rig/examples/x-amd-kernel
dune exec ./main.exe
```

## What You'll Learn

- Reading a code object: `Rig_amd_abi.Code_object.of_string`, `kernel`
- Loading it: the device places its image in the GPU's memory, and
  `Image.entry` names the kernel's descriptor
- What compiled code finds in an AMD device's capability:
  `Rig.capability g Rig_amd_abi.Capability.key`, its `gpu` and `compute`
- Writing a dispatch: `Pm4.dispatch` inside `Pm4.run`, encoded by
  `Rig_packet.encode`
- Work as words: `Submission.Words`

## Key Functions

| Function                                  | Purpose                                 |
| ----------------------------------------- | --------------------------------------- |
| `Code_object.kernel co name`              | A kernel as its descriptor describes it |
| `Pm4.dispatch gpu k ~program ~args ...`   | The packets that launch it              |
| `Pm4.run gpu p`                           | Order it after earlier work, and before later |
| `Rig_packet.encode Int64.of_int p`        | The packets' bytes                      |

## The code object

`add_gfx1201.hsaco` is `add.cl` compiled in this directory with Homebrew clang
22.1.7 (`clang`) and Homebrew LLD 21.1.8 (`ld.lld`):

```bash
clang -c -x cl -cl-std=CL2.0 -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
  -mcode-object-version=5 -nogpulib -O2 add.cl -o add.o
ld.lld -shared add.o -o add_gfx1201.hsaco
rm add.o
```
