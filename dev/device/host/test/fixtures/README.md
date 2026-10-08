# Host program fixtures

Made in this directory with Homebrew clang 22.1.7 (`clang`). `H` is `-c -x c
-O2 -fPIC -ffreestanding -fno-math-errno -nostdlib -fno-ident`; `X` is
`--target=x86_64-none-unknown-elf`; `A` is
`-ffixed-x18 --target=aarch64-none-unknown-elf`.

For each `<f>` of `affine`, `waiting`, `blocks`, `nested`, `writable`,
`undefined`, `data_entry`, `empty`, `scale`, `loop` and `many`, the suite's and the
bench's programs, whose sources say what they do:

- `<f>_x86_64.o`: `clang $H $X <f>.c -o <f>_x86_64.o`
- `<f>_aarch64.o`: `clang $H $A <f>.c -o <f>_aarch64.o`

`linked` and `got`, whose call of an external function goes through a word
on x86_64 too:

- `linked_x86_64.o`: `clang $H $X linked.c -o linked_x86_64.o`
- `linked_aarch64.o`: `clang $H $A linked.c -o linked_aarch64.o`
- `got_x86_64.o`: `clang $H -fno-plt $X got.c -o got_x86_64.o`
- `got_aarch64.o`: `clang $H $A got.c -o got_aarch64.o`

For x86_64 Windows, whose process functions follow its own convention:

- `linked_x86_64_windows.o`: `clang $H -DWINDOWS $X linked.c -o linked_x86_64_windows.o`
