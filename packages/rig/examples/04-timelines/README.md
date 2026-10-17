# `04-timelines`

A device numbers the work submitted to it 1, 2, 3, ..., the values of its
timeline. This example makes a submission of copies once, submits it several
times, and reads the point each submit returns.

```bash
dune exec packages/rig/examples/04-timelines/main.exe
```

It runs on a memory device, a device with a timeline of its own whose memory
lies in the host's memory. Its work is done when `submit` returns, so every
point is reached by then. A GPU's work runs after `submit` returns, with the
same calls, as in [01-add](../01-add/).

## What You'll Learn

- Memory devices: `Rig.memory_device`
- Submissions made once and submitted many times, each submit with a run:
  `Submission.make`, `Submission.Run.make`, `submit`
- Parts ordered by the memory they share
- Points and values: `Point.pp`, `Point.reached`, `submitted`, `signaled`
- An empty submission, whose point is reached once all earlier work has
  completed

## Key Functions

| Function                  | Purpose                                         |
| ------------------------- | ----------------------------------------------- |
| `memory_device name`      | A device with a timeline, in host memory        |
| `Submission.make d parts` | Work for `d`, made once                         |
| `Submission.Run.make s`   | Storage of `s`'s parameters, a submit at a time |
| `submit r ~buffers`       | Run `r`'s submission once and return its point  |
| `Point.reached p`         | Whether `p`'s device has reached it             |

## Next Steps

Continue to [05-stamps](../05-stamps/).
