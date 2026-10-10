/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The instructions of Apple's matrix unit (AMX) that nx.cpu uses, the only
   place they are spelled.

   Apple documents none of them. The encodings and operand fields are those
   of the corsix/amx write-up (github.com/corsix/amx: Instructions.md,
   ldst.md, fma.md, extr_v.md, setclr.md), and each one used here was
   checked on an M1 Max: fma32 rounds once, keeps subnormals, gives the
   default NaN for a NaN operand, and a chain of fma32 into one accumulator
   equals C's fmaf chain bit for bit.

   The unit's state is X and Y, eight registers of 64 bytes each, and Z, 64
   rows of 64 bytes. An instruction is the word 0x201000 + (op << 5) + r,
   r the number of the general register holding its operand: here always
   x0. SET makes the thread's state live and zero, CLR ends it; the system
   saves live state across a context switch. */

#ifndef NX_CPU_AMX_H
#define NX_CPU_AMX_H

#include <stdint.h>

/* The instruction [op] on the operand [v], held in x0 (Instructions.md). */
#define NX_AMX_OP(op, v)                                                    \
  do {                                                                      \
    register uint64_t nx_amx_x0 __asm__("x0") = (uint64_t)(v);             \
    __asm__ volatile(".word %c0" ::"i"(0x201000 + ((op) << 5)),             \
                     "r"(nx_amx_x0)                                         \
                     : "memory");                                           \
  } while (0)

/* SET and CLR are op 17 with the immediates 0 and 1 in place of a register,
   after three nops (setclr.md). */
#define NX_AMX_SET()                                                        \
  __asm__ volatile("nop\nnop\nnop\n.word %c0" ::"i"(0x201000 + (17 << 5)) \
                   : "memory")
#define NX_AMX_CLR()                                                        \
  __asm__ volatile("nop\nnop\nnop\n.word %c0" ::"i"(0x201000 +            \
                                                     (17 << 5) + 1)         \
                   : "memory")

/* Loads and stores (ldst.md): the operand is the address in bits 0-55 and
   the register (X or Y: 0-7; Z: the row, 0-63) in bits 56-61. With bit 62,
   an instruction moves 128 bytes, the register and the next, or the Z row
   and the next; the address must then be 128-byte aligned, or it faults.
   Single loads and stores take any address, aligned ones moving faster. */
#define NX_AMX_PAIR (1ull << 62)
#define NX_AMX_LDX(p, r) NX_AMX_OP(0, (uint64_t)(p) | ((uint64_t)(r) << 56))
#define NX_AMX_STX(p, r) NX_AMX_OP(2, (uint64_t)(p) | ((uint64_t)(r) << 56))
#define NX_AMX_LDY(p, r) NX_AMX_OP(1, (uint64_t)(p) | ((uint64_t)(r) << 56))
#define NX_AMX_STY(p, r) NX_AMX_OP(3, (uint64_t)(p) | ((uint64_t)(r) << 56))
#define NX_AMX_LDZ(p, r) NX_AMX_OP(4, (uint64_t)(p) | ((uint64_t)(r) << 56))
#define NX_AMX_STZ(p, r) NX_AMX_OP(5, (uint64_t)(p) | ((uint64_t)(r) << 56))

/* Outer products (fma.md): the operand holds the byte offset into Y in
   bits 0-8, into X in bits 10-18, and the Z row in bits 20-25. fma32 adds
   x[i]·y[j] into z[4j + q][i] for 16 × 16 float32, q the Z row modulo 4:
   four accumulators. */
#define NX_AMX_FMA32(x, y, z)                                               \
  NX_AMX_OP(12, (uint64_t)(y) | ((uint64_t)(x) << 10) | ((uint64_t)(z) << 20))

/* A column of Z into Y (extr_v.md, bit 26 clear; bits 28-29 1 for
   float32): Y's 16 lanes at byte offset [y], bits 0-8, take Z's lane c / 4
   of the rows 4i + c mod 4, c in bits 20-25: y[i] = z[4i + c mod 4][c / 4].
   It moves bits and rounds nothing. */
#define NX_AMX_EXTRV32(y, c)                                                \
  NX_AMX_OP(9, (uint64_t)(y) | ((uint64_t)(c) << 20) | (1ull << 28))

#endif
