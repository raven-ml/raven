# `04-stamps`

Memory records the point of its last write and, per device, the point of its
last use. Work that reads memory waits for its last write; work that writes it
waits for every use. This example passes data from one device to another and
back to the host, ordered by the memory alone.

```bash
dune exec dev/rig/examples/04-stamps/main.exe
```

Two memory devices stand for two GPUs. Their work is done when `submit`
returns, so no wait blocks here; the order the program states is the one a GPU
would keep.

## What You'll Learn

- A device's work addresses its own memory, and another's once it borrows it:
  `reaches`, `Buffer.borrow`
- A borrow shares the stamps of the memory it maps
- Ordering by memory, and by a point the submit waits for: `submit ~waits`
- The host's side of the same rule: `Buffer.wait` with `Read` and
  `Read_write`, and `Buffer.copy`

## Key Functions

| Function                    | Purpose                                         |
| --------------------------- | ----------------------------------------------- |
| `reaches d d'`              | Whether `d`'s work can address `d'`'s memory    |
| `Buffer.borrow d b`         | `b`'s memory as a buffer of `d`, without a copy |
| `submit s ~waits:[\| p \|]`  | The submit of `s` waits for `p`                 |
| `Buffer.wait b access`      | Wait before the host reads or writes `b`        |

## Next Steps

Continue to [05-steps](../05-steps/).
