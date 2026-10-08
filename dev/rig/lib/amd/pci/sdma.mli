(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The GPU's copy engines (SDMA): their start, one ring, and their stop.

   The GPU's owner serializes calls. *)

type t

val make : Regs.t -> t

(* [start s] programs every engine: its micro-engine, its traps and its doorbell
   range. *)
val start : t -> unit

(* [queue s ~ring ~bytes ~read ~write] programs queue 0 of engine 0: its ring of
   [bytes] bytes at the GPU address [ring], its read position written to [read],
   its write position polled at [write]. It is the queue's doorbell index. *)
val queue : t -> ring:int -> bytes:int -> read:int -> write:int -> int

(* [halt s] halts the engines' micro-engines. *)
val halt : t -> unit

(* [stop s] disables the rings [queue] programmed, then soft-resets the engines
   where their version allows it. *)
val stop : t -> unit
