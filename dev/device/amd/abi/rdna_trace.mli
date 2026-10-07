(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The decoder of RDNA3 and RDNA4 thread traces. *)

(** The type for what a trace records: a marker of the GPU's clock, and the
    start and end of a wave, at their shader times. *)
type event =
  | Marker of { time : int; realtime : int }
  | Wave_start of { time : int; cu : int; simd : int; slot : int }
  | Wave_end of { time : int; cu : int; simd : int; slot : int }

val iter : (event -> unit) -> string -> unit
(** [iter f trace] calls [f] on the events of [trace], one shader engine's
    bytes, in order. A packet whose 8 bytes from its start pass [trace]'s end is
    not read. *)
