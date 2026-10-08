# `11-host-programs`

A host program is a C function compiled to an ELF object, linked into the
process and called. This example links `affine.c`, calls it on rig's host
buffers, then splits a call of a million iterations into blocks that the
host's cores run at once.

```bash
cd dev/rig/examples/11-host-programs
dune exec ./main.exe
```

## What You'll Learn

- Linking an object and naming its entry: `Rig_host.link ~entry`
- The calling convention, `void f(void **buffers, const int64_t *values)`,
  with buffers passed by `Buffer.address`
- Split calls: `Rig_host.split`, whose blocks run on up to
  `Rig_host.workers ()` threads, each given its range in two of the values
- Choosing the object for the host: `Rig.arch Rig.host`

## Key Functions

| Function                           | Purpose                                     |
| ---------------------------------- | ------------------------------------------- |
| `Rig_host.link ~entry obj`         | The program of `obj`, or why not            |
| `Rig_host.call ?split p bs vs`     | Call it on buffer addresses and values      |
| `Rig_host.workers ()`              | The most threads a split call runs on       |

## The objects

`affine_x86_64.o` and `affine_aarch64.o` are `affine.c` compiled in this
directory with Homebrew clang 22.1.7. With
`H = -c -x c -O2 -fPIC -ffreestanding -fno-math-errno -nostdlib -fno-ident`:

```bash
clang $H --target=x86_64-none-unknown-elf affine.c -o affine_x86_64.o
clang $H -ffixed-x18 --target=aarch64-none-unknown-elf affine.c -o affine_aarch64.o
```

`-ffixed-x18` keeps the code off the register macOS and Windows reserve.

## Next Steps

Continue to [12-pool](../12-pool/).
