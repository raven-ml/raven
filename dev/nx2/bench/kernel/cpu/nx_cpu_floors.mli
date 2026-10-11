(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** nx.cpu's floors for the kernel bench: what the host does with a row's bytes,
    running no nx.cpu code, on one thread, on the performance cores and on every
    core. A copy's floor is memcpy of its bytes; a cast's moves its elements'
    bytes in and out with an integer truncation or extension per element. A cast
    whose conversion can cost more than that has a second floor, the fastest
    loop of its codec. floor-move-4-2-1M-all moves 1 Mi elements of 4 bytes into
    2 on every core, floor-copy-4M-1t copies 4 MiB on one thread,
    floor-codec-f32-bf16-1M-1t encodes 1 Mi bfloat16 on one thread. A
    contraction's floor is the host's peak of fused multiply-adds, measured
    over 1024 Mi flops, floor-fma-f32-1024M-performance on the performance
    cores, at which its flops would run; one that streams its operands, as a
    product of few rows does, also reads their bytes, floor-read-64M-all on
    every core. An elementwise kind's floor streams its operands' bytes with
    one integer operation per element, floor-stream-2x4-4-1M-all reading two
    arrays of 1 Mi 4-byte elements and writing one on every core; a where's
    floor-select-4-1M also reads its condition's bytes. *)

(** The type for the work of a kernel row. *)
type work =
  | Copy of int  (** [Copy n] copies [n] bytes. *)
  | Cast of Nx_array.Dtype.any * Nx_array.Dtype.any * int
      (** [Cast (s, d, n)] casts [n] elements of [s] into [d]. *)
  | Fma of Nx_array.Dtype.any * int
      (** [Fma (dt, n)] runs [n] flops in [dt], float32 or float64. *)
  | Read of int  (** [Read n] reads [n] bytes. *)
  | Stream of { ins : int; inb : int; outb : int; n : int }
      (** [Stream { ins; inb; outb; n }] reads [n] elements of [inb] bytes
          from each of [ins] arrays, at most 3, and writes [n] of [outb]
          bytes, as an elementwise kind does; [inb] and [outb] are [1] and
          [1], [4] and [1], [4] and [4], or [8] and [8]. *)
  | Select of { inb : int; n : int }
      (** [Select { inb; n }] reads [n] condition bytes and [n] elements of
          [4] bytes from each of two arrays, and writes [n] of [4], as a
          where does; [inb] is [4]. *)

val rows : work list -> Thumper.bench list
(** [rows ws] is the floor rows that bound [ws], each once, then
    block-transposed-512x512-1t: nx.cpu's transposing block copy alone on one
    thread over 512x512 float32 in cache, which the vendors' one-thread
    transposes bound, and block-transposed-f64-512x512-1t over float64. *)
