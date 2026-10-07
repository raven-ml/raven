/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A function as the host loader takes it: a call to a function another object
   defines, a static counter in .bss and a constant table in .rodata.
   host_x86_64.o and host_aarch64.o are this file compiled, in this
   directory, by Homebrew clang 22.1.7:

   clang -c -x c -O2 -fPIC -ffreestanding -fno-math-errno -nostdlib -fno-ident \
     --target=TARGET-none-unknown-elf host.c -o host_TARGET.o

   with TARGET x86_64 and aarch64. The object names its source file, so the
   file keeps its name. */

void ext(int);
static int counter;
static const int table[4] = {1, 2, 3, 4};
void f(int i) { ext(i); counter += table[i & 3]; }
