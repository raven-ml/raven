# `05-stamps`

Each allocation records the point of its last write and, for each device,
the point of its last use. These points are its stamps. Work that reads
memory follows its last write, and work that writes it follows every use.
This example passes data from one device to another and back to the host,
ordered by the memory alone.

```bash
dune exec packages/rig/examples/05-stamps/main.exe
```

Two memory devices stand for two GPUs. Their work is done when `submit`
returns, so no wait blocks here. A GPU keeps the same order.

## What You'll Learn

- A device's work addresses its own memory, and another's once it borrows
  it: `reaches`, `Buffer.borrow`
- A borrow shares the stamps of the memory it maps
- Work on one device follows another device's write by the memory it reads,
  with no point named
- The host's side of the same rule: `Buffer.read`, `Buffer.write` and
  `Buffer.copy`

## Key Functions

| Function             | Purpose                                                       |
| -------------------- | ------------------------------------------------------------- |
| `reaches d d'`       | Whether `d`'s work can address `d'`'s memory through a borrow |
| `Buffer.borrow d b`  | `b`'s bytes as a buffer on `d`, without a copy                |
| `Buffer.write b k f` | `f` over `b`'s bytes once every use is done                   |
| `Buffer.read b k f`  | `f` over `b`'s bytes once its last write is done              |

## Next Steps

Continue to [06-steps](../06-steps/).
