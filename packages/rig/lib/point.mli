(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Points as ints, the representation behind {!Rig.Point.t}.

    A point is a device's index in bits 47 to 62 and a value on its timeline
    in bits 0 to 46, the 63 bits of an OCaml int, so storing one allocates
    nothing. The int [0] stands for the absence of a point, since index 0
    is the host, whose values start at 1. The host's work is never stamped:
    it is done when its submit returns. From index 32768 on a point is a
    negative int, so code that orders points orders their {!value}s.

    The C stubs encode points the same way ([RIG_POINT] in [rig_stubs.h]),
    and the two change together. *)

val make : int -> int -> int
(** [make index v] is the point of the device of index [index] at value [v].
    It checks nothing: [0 <= index <= max_index] and [0 <= v <= max_value]
    must hold. *)

val index : int -> int
(** [index p] is the index of [p]'s device ({!Dev.of_index}). *)

val value : int -> int
(** [value p] is [p]'s value on its device's timeline. *)

val max_value : int
(** [max_value] is [2]{^ 47}[ - 1], the largest value a point holds. *)

val max_index : int
(** [max_index] is [65_535], the largest device index, so the process has
    this many device slots ({!Rig.section-opening}). *)
