# `07-reclamation`

Nothing frees a buffer by hand. Once a buffer is unreachable, its memory goes
back to its device for reuse. This example sets a device's budget, allocates
far more than the budget in buffers it does not keep, and shows when an
allocation fails.

```bash
dune exec packages/rig/examples/07-reclamation/main.exe
```

## What You'll Learn

- A device's budget and what it holds: `budget`, `set_budget`, `allocated`
- An allocation larger than the budget raises `Out_of_memory`
- An allocation that the budget refuses frees the cache, collects unreachable
  buffers and tries again, so unreachable buffers make room
- Live buffers are never freed, and an allocation raises when they fill the
  budget

## Key Functions

| Function         | Purpose                                  |
| ---------------- | ---------------------------------------- |
| `budget d`       | The most bytes `d` holds at one time     |
| `set_budget d n` | Lower or raise it                        |
| `allocated d`    | The bytes `d` holds now                  |
| `Out_of_memory`  | Raised once reclaiming has found no room |

## Next Steps

Continue to [08-claims](../08-claims/).
