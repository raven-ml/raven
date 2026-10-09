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
    every core. *)

(** The type for the work of a kernel row. *)
type work =
  | Copy of int  (** [Copy n] copies [n] bytes. *)
  | Cast of Nx_array.Dtype.any * Nx_array.Dtype.any * int
      (** [Cast (s, d, n)] casts [n] elements of [s] into [d]. *)
  | Fma of Nx_array.Dtype.any * int
      (** [Fma (dt, n)] runs [n] flops in [dt], float32 or float64. *)
  | Read of int  (** [Read n] reads [n] bytes. *)

val rows : work list -> Thumper.bench list
(** [rows ws] is the floor rows that bound [ws], each once, then
    block-transposed-512x512-1t: nx.cpu's transposing block copy alone on one
    thread over 512x512 float32 in cache, which the vendors' one-thread
    transposes bound. *)
