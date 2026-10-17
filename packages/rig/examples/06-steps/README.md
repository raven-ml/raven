# `06-steps`

Compiled work is made once and run many times. This example builds a step
from a C function, `scale.c`, that a device calls to run its part. The step
names its constants as fixed memory and takes its input and output at each
submit. It runs twice, and once it is unreachable and its work is done, it
no longer keeps its value.

```bash
dune exec packages/rig/examples/06-steps/main.exe
```

A memory device calls the fill on the submitting thread. A GPU's driver
calls it from its `submit` with the queue's context, and the argument and
the run's buffers work the same way. Fixed memory has an address known when
the step is made, so compiled code writes it once into parameters or
recorded commands, which every run replays as they are.

## What You'll Learn

- Fills, C work with per-run parameters and refs to the run's buffers, each
  stating how the work uses its buffer: `Submission.read`, `Submission.write`
- Fixed memory, which every run uses, named once when the step is made
- A fixed buffer's address, written once into the run: `Submission.address`
- A kept value, which lives until the step is unreachable and its work done
- The buffers that change from run to run, passed to each submit

## Key Functions

| Function                               | Purpose                                                |
| -------------------------------------- | ------------------------------------------------------ |
| `Submission.fill ~refs f ~params`      | Work that is a C function                              |
| `Submission.make ~keep ~fixed d parts` | A step's submission, its fixed memory and a kept value |
| `Submission.address s b`               | The address of fixed memory `b`                        |
| `Submission.Run.make s`                | Storage of the step's parameters                       |
| `submit r ~buffers`                    | Run it once with these buffers                         |

## Next Steps

Continue to [07-reclamation](../07-reclamation/).
