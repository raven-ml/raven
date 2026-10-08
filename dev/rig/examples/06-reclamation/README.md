# `06-reclamation`

Nothing frees a buffer by hand. Once a buffer is unreachable, its memory goes
back to its device for reuse. This example sets a device's budget, allocates far
more than it in buffers it drops, and shows when an allocation fails.

```bash
dune exec dev/rig/examples/06-reclamation/main.exe
```

## What You'll Learn

- A device's budget: `budget`, `set_budget`
- An allocation over the budget raises `Out_of_memory` at once
- An allocation the budget refuses releases the cache, collects unreachable
  buffers and tries again, so dropped buffers make room
- Live buffers are never released: when they fill the budget, the allocation
  raises
- `free_cache d` gives a device's cached memory back to its driver at any time

## Key Functions

| Function           | Purpose                                        |
| ------------------ | ---------------------------------------------- |
| `budget d`         | The most bytes `d` holds at once               |
| `set_budget d n`   | Lower or raise it                              |
| `free_cache d`     | Return `d`'s cached memory to its driver       |
| `Out_of_memory`    | Raised once reclaiming found no room           |

## Next Steps

Continue to [07-claims](../07-claims/).
