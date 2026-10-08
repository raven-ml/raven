# `03-timelines`

A device's work is one sequence of submissions, numbered 1, 2, …: the values of
its timeline. This example makes a submission of copies once, submits it
several times, and reads the points each submit returns.

```bash
dune exec dev/rig/examples/03-timelines/main.exe
```

It runs on a memory device, a device with a timeline of its own whose memory is
the host's. Its work is done when `submit` returns, so every value is reached
at once. A GPU's work runs after `submit` returns, with the same calls; see
[x-gpu](../x-gpu/).

## What You'll Learn

- Memory devices: `Rig.memory_device`
- Submissions made once and submitted many times: `Submission.make`, `submit`
- Parts on a device's queues, ordered by `after`
- Points and values: `Point.pp`, `submitted`, `signaled`, `wait`

## Key Functions

| Function                                  | Purpose                                |
| ----------------------------------------- | -------------------------------------- |
| `memory_device name`                      | A device with a timeline on the host   |
| `Submission.make ~reads ~writes ~waits d` | Work for `d`, prepared once            |
| `submit s`                                | Hand `s` over; the point of its value  |
| `wait d v`                                | Return once `d` reached `v`            |

## Next Steps

Continue to [04-stamps](../04-stamps/).
