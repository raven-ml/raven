# Host code fixtures

Made in this directory with Homebrew clang 22.1.7 (`clang`), as
test/host/fixtures' are. `H` is `-c -x c -O2 -fPIC -ffreestanding
-fno-math-errno -nostdlib -fno-ident`; `X` is
`--target=x86_64-none-unknown-elf`; `A` is
`-ffixed-x18 --target=aarch64-none-unknown-elf`.

`ready`, the suites' host code that writes a rail's area and calls its
ready function, whose source says what it does:

- `ready_x86_64.o`: `clang $H $X ready.c -o ready_x86_64.o`
- `ready_aarch64.o`: `clang $H $A ready.c -o ready_aarch64.o`

`spin`, the bench's fill that counts to the word its argument points at,
as work that takes a device a known while:

- `spin_x86_64.o`: `clang $H $X spin.c -o spin_x86_64.o`
- `spin_aarch64.o`: `clang $H $A spin.c -o spin_aarch64.o`
