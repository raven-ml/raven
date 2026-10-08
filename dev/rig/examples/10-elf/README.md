# `10-elf`

Host programs, GPU code and firmware come as ELF objects. This example reads an
object, lists the sections its image holds, its symbols and its relocations in
image offsets, and writes the image as a loader would.

```bash
cd dev/rig/examples/10-elf
dune exec ./main.exe
```

## What You'll Learn

- Reading an object into its image: `Rig_elf.of_string`
- The fields of `Rig_elf.t`: sections with their image offsets, symbols and
  their places, relocations
- Relocation values from image offsets alone, wherever the image lies
- Finding an entry by name: `Rig_elf.symbol`
- Building the image's bytes from the sections it holds

## Key Functions

| Function                 | Purpose                                       |
| ------------------------ | --------------------------------------------- |
| `Rig_elf.of_string obj`  | The object laid out in its image, or why not  |
| `Rig_elf.symbol o name`  | The image offset of a symbol                  |
| `Rig_elf.allocated`      | Whether a section is code or data a loader holds |

## The object

`poly_x86_64.o` is `poly.c` compiled in this directory with Homebrew clang
22.1.7:

```bash
clang -c -x c -O2 -fPIC -ffreestanding -fno-math-errno -nostdlib -fno-ident \
  --target=x86_64-none-unknown-elf poly.c -o poly_x86_64.o
```

## Next Steps

Continue to [11-host-programs](../11-host-programs/).
