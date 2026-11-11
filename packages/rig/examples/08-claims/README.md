# `08-claims`

A function handed buffers may write over an input in place only if nothing
else reads that memory. This example writes an `add` that takes the input
it writes over when nothing claims it, and allocates otherwise.

```bash
dune exec packages/rig/examples/08-claims/main.exe
```

## What You'll Learn

- Read claims, held while the host reads or writes memory: `Buffer.read`,
  `Buffer.write`
- Taking: `Buffer.take` takes the memory of buffers that span it when
  nothing claims it, and makes every buffer made over it until then dead,
  so a use of one raises `Invalid_argument` with the reason. The write in
  place goes through the buffer `take` returns.
- Keeping: `Buffer.keep` claims memory until `Buffer.release`, so no take
  writes it meanwhile
- Buffers that are never taken: views of part of an allocation, buffers
  over a bigarray, and memory that `Buffer.export` exported
- Why `add` keeps `x` while it takes `y`: it reads `x`, and no take may
  write `x` while it does, also when `y` is over `x`'s memory.

## Key Functions

| Function                | Purpose                                         |
| ----------------------- | ----------------------------------------------- |
| `Buffer.read b k f`     | `f` over `b`'s bytes under a read claim         |
| `Buffer.take ~why bs`   | Live buffers over `bs`'s memory, `bs` dead      |
| `Buffer.keep bs`        | A claim on `bs`'s memory until it is released   |
| `Buffer.release k`      | The end of the keep `k`                         |
| `Buffer.export b k`     | `b`'s bytes as an array, outside the claims     |

## Next Steps

Continue to [09-loss](../09-loss/).
