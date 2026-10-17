# `12-host-programs`

Host code is a C function compiled to an ELF object, loaded as an image on
the host and launched as a GPU's kernel is. This example loads `affine.c`,
launches it on host buffers, then over a million threads in groups that the
host's cores run in parallel.

```bash
cd packages/rig/examples/12-host-programs
dune exec ./main.exe
```

## What You'll Learn

- Loading an object on the host: `Rig.Image.load Rig.host`
- The calling convention, `void f(const void *params, int64_t first,
  int64_t end)`, with the buffers' host addresses in the parameters
- Launches over many groups, which the host's cores run in parallel, each
  call of the function given a range of whole groups
- Choosing the object for the host: `Rig.arch Rig.host`

## Key Functions

| Function                      | Purpose                              |
| ----------------------------- | ------------------------------------ |
| `Image.load host obj`         | The image of `obj`, or why not       |
| `Submission.launch`, `submit` | Run one of its functions over a grid |
| `Image.resolve image f`       | Its function `f`, if it has one      |

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

Continue to [13-pool](../13-pool/).
