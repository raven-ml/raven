# `05-steps`

Compiled work is prepared once and run many times. This example builds a step
from a C function, `scale.c`, that a device calls to run its part: the step
names its constants as fixed memory, passes its input and output to each
submit, runs twice, and its hold is released once nothing reaches it.

```bash
dune exec dev/rig/examples/05-steps/main.exe
```

A memory device calls the fill in the submitting thread. A GPU's driver calls
it to encode work into its queue; the argument and the run's buffers work the
same way.

## What You'll Learn

- Fills: C work in a submission, with an argument buffer that holds addresses
  (`Buffer.address`)
- Fixed memory: memory every run uses, named once when the step is made
- Holds: a release that runs once the step is unreachable and its work done
- A run's buffers: what it reads and writes, passed to each submit with the
  storage it uses: `submit s ~run ~reads ~writes ~waits`
- The host rewriting the argument only after the work that read it:
  `Buffer.wait arg Read_write`

## Key Functions

| Function                                     | Purpose                                 |
| -------------------------------------------- | --------------------------------------- |
| `Submission.Fill { fill; arg; _ }`           | Work that is a C function               |
| `Hold.make ~release v`                       | What a step's release frees             |
| `Submission.make ~hold ~fixed ~reads ~writes ...` | A step's submission, its fixed memory and its run's arity |
| `Submission.Run.make ()`                     | Storage for one submit at a time         |
| `submit s ~run ~reads ~writes ~waits`        | Run it once with these buffers           |

## Next Steps

Continue to [06-reclamation](../06-reclamation/).
