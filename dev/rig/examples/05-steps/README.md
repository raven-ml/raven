# `05-steps`

Compiled work is prepared once and run many times. This example builds a step
from a C function, `scale.c`, that a device calls to run its part: the step
keeps its constants and its argument in a hold, takes its input and output
through slots, runs twice, and is released once nothing reaches it.

```bash
dune exec dev/rig/examples/05-steps/main.exe
```

A memory device calls the fill in the submitting thread. A GPU's driver calls
it to encode work into its queue; the argument and slots work the same way.

## What You'll Learn

- Fills: C work in a submission, with an argument buffer that holds addresses
  (`Buffer.address`)
- Holds: fixed memory kept across submissions, with a release that runs once
  the step is unreachable and its work done
- Slots: the buffers a run reads and writes, set before each submit and
  cleared by it: `Submission.read`, `Submission.write`
- The host rewriting the argument only after the work that read it:
  `Buffer.wait arg Read_write`

## Key Functions

| Function                                     | Purpose                                 |
| -------------------------------------------- | --------------------------------------- |
| `Submission.Fill { fill; arg; _ }`           | Work that is a C function               |
| `Hold.make ~release bs`                      | Memory kept for a step's life           |
| `Submission.make ~hold ~reads ~writes ...`   | A step's submission, with its slots     |
| `Submission.read s i b`, `write s i b`       | Set a slot for the next submit          |

## Next Steps

Continue to [06-reclamation](../06-reclamation/).
