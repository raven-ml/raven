(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The GPU's copy engines (SDMA), of which this library runs queues 0 and 1
    of engine 0, and on SDMA 4 its gfx queue alone.

    Calls for one GPU run one at a time. *)

type t
(** The type for a GPU's copy engines. *)

val make : Regs.t -> t
(** [make r] is the copy engines of the GPU of [r]. *)

val start : t -> unit
(** [start s] enables every engine's traps and routes the engines' doorbells.
    From SDMA 6 on it also sets each engine's hang watchdog and translation
    cache, and releases its micro-engine from halt and reset. *)

val queue : t -> ring:int -> bytes:int -> read:int -> write:int -> int option
(** [queue s ~ring ~bytes ~read ~write] programs the first queue of engine 0
    that runs none {!queue} made, and is its doorbell index, or [None] if
    every one does. Its ring is the [bytes] bytes at the GPU address [ring],
    which is 256-byte aligned, and [bytes] is a power of two. The queue
    writes its read position back to [read] and polls its write position at
    [write], both starting at 0. *)

val retire : t -> int -> unit
(** [retire s index] disables the queue {!queue} made whose doorbell index is
    [index], so that a later {!queue} may program it again.

    Raises [Invalid_argument] if {!queue} made no queue of that index. *)

val slots : t -> int
(** [slots s] is the number of queues {!queue} programs at once: 2, and 1 on
    SDMA 4. *)

val halt : t -> unit
(** [halt s] halts engine 0's micro-engine, from SDMA 6 on. *)

val stop : t -> unit
(** [stop s] disables the queues {!queue} programs, whichever process
    programmed them. From SDMA 6 on it then halts engine 0 and soft-resets
    it. *)
