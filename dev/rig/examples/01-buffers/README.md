# `01-buffers`

A buffer is bytes of one device's memory; what they mean is the caller's. This
example makes buffers on the host, reads them as integers, views part of one,
borrows an OCaml bigarray without a copy, and copies between buffers.

```bash
dune exec dev/rig/examples/01-buffers/main.exe
```

## What You'll Learn

- The host as a device: `Rig.host`, `name`, `computes`, `runs_on_host`
- Owned and borrowed buffers: `Buffer.create`, `Buffer.of_bigarray`
- Reading bytes as elements with `Buffer.bigarray`
- Views of the same memory: `Buffer.view`, `spans`, `overlaps`
- `Buffer.copy`, and the `Invalid_argument` misuse raises

## Key Functions

| Function                           | Purpose                              |
| ---------------------------------- | ------------------------------------ |
| `Buffer.create d n`                | `n` bytes of `d`'s memory            |
| `Buffer.of_bigarray ba`            | A host buffer over `ba`'s bytes      |
| `Buffer.bigarray k b`              | `b`'s bytes as elements of kind `k`  |
| `Buffer.view b ~first ~length`     | A range of `b`'s memory              |
| `Buffer.copy ~src ~dst`            | Copy bytes, once they are ready      |

## Next Steps

Continue to [02-files](../02-files/).
