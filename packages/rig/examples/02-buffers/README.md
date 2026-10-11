# `02-buffers`

A buffer is bytes of one device's memory, and the caller gives them their
meaning. This example makes buffers on the host, reads them as integers,
views part of one, wraps an OCaml bigarray without a copy, and copies
between buffers.

```bash
dune exec packages/rig/examples/02-buffers/main.exe
```

## What You'll Learn

- The host as a device: `Rig.host`, `name`, `computes`, `runs_on_host`
- Buffers of new memory and buffers over a bigarray: `Buffer.create`,
  `Buffer.of_bigarray`, `Buffer.is_borrowed`
- Reading and writing bytes as elements: `Buffer.read`, `Buffer.write`
- Views of the same memory: `Buffer.view`, `spans`, `overlaps`
- `Buffer.copy`, and the `Invalid_argument` that misuse raises

## Key Functions

| Function                       | Purpose                               |
| ------------------------------ | ------------------------------------- |
| `Buffer.create d n`            | `n` bytes of `d`'s memory             |
| `Buffer.of_bigarray a`         | A host buffer over `a`'s bytes        |
| `Buffer.read b k f`            | `f` over `b`'s bytes as kind `k`      |
| `Buffer.view b ~first ~length` | A range of `b`'s bytes, in its memory |
| `Buffer.copy ~src ~dst`        | Copy `src`'s bytes into `dst`         |

## Next Steps

Continue to [03-files](../03-files/).
