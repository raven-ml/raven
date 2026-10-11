(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The GPU's copy engines (SDMA), of which this library runs one queue,
    queue 0 of engine 0.

    Calls for one GPU run one at a time. *)

type t
(** The type for a GPU's copy engines. *)

val make : Regs.t -> t
(** [make r] is the copy engines of the GPU of [r]. *)

val start : t -> unit
(** [start s] enables every engine's traps and routes the engines' doorbells.
    From SDMA 6 on it also sets each engine's hang watchdog and translation
    cache, and releases its micro-engine from halt and reset. *)

val queue : t -> ring:int -> bytes:int -> read:int -> write:int -> int
(** [queue s ~ring ~bytes ~read ~write] programs queue 0 of engine 0 and is
    its doorbell index. Its ring is the [bytes] bytes at the GPU address
    [ring], which is 256-byte aligned, and [bytes] is a power of two. The
    queue writes its read position back to [read] and polls its write
    position at [write]. *)

val halt : t -> unit
(** [halt s] halts engine 0's micro-engine, from SDMA 6 on. *)

val stop : t -> unit
(** [stop s] disables the queue {!queue} programs, whichever process
    programmed it. From SDMA 6 on it then halts engine 0 and soft-resets
    it. *)
