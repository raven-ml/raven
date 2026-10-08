# `02-files`

The disk is a device whose buffers are files. This example writes a file by
copying into it, reads part of it back through a view, and borrows its pages
to read them in place.

```bash
cd dev/rig/examples/02-files
dune exec ./main.exe
```

It writes `numbers.bin` in the directory it runs in.

## What You'll Learn

- The disk as a device that holds bytes and runs no work
- Writing a file with `Buffer.copy`, and ordering the writes with `barrier`
- Reading part of a file through `Buffer.view`
- Borrowing a file's pages with `Buffer.borrow Rig.host`: a file opened for
  reading is mapped copy-on-write
- Copy to read all of a file; borrow to touch part of it in place

## Key Functions

| Function                     | Purpose                                     |
| ---------------------------- | ------------------------------------------- |
| `Rig_disk.create_file p n`   | A new file of `n` bytes, as a buffer        |
| `Rig_disk.of_file p`         | An existing file's bytes, for reading       |
| `Rig_disk.barrier b`         | Order `b`'s writes before later changes     |
| `Buffer.borrow Rig.host b`   | `b`'s pages as host memory                  |

## Next Steps

Continue to [03-timelines](../03-timelines/).
