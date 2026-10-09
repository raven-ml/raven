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

The example loads `simple_add_sm89.cubin`, `simple_add.cu` compiled for sm_89,
from rig's ELF fixtures in `dev/rig/test/elf/fixtures`, whose README says how
it is made.
