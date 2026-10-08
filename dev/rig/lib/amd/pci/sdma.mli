(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The GPU's copy engines (SDMA): their start, one ring, and their stop.

    The GPU's owner serializes calls. *)

type t
(** The type for a GPU's copy engines. *)

val make : Regs.t -> t
(** [make r] is the copy engines of the GPU of [r]. *)

val start : t -> unit
(** [start s] programs every engine (its micro-engine, its traps) and routes the
    engines' doorbells. *)

val queue : t -> ring:int -> bytes:int -> read:int -> write:int -> int
(** [queue s ~ring ~bytes ~read ~write] programs queue 0 of engine 0: its ring
    of [bytes] bytes, a power of two, at the GPU address [ring], its read
    position written back to [read], its write position polled at [write]. It is
    the queue's doorbell index. *)

val halt : t -> unit
(** [halt s] halts the engines' micro-engines, from SDMA 6. *)

val stop : t -> unit
(** [stop s] disables the queue {!queue} programs, whichever process programmed
    it, then soft-resets the engines from SDMA 6. *)
