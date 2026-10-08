(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Points ({!Rig.Point}) as immediate ints.

    A point is the device's index in bits 47 to 62 and the value in bits 0 to
    46. The int [0] is no point. The C stubs use the same encoding ([RIG_POINT]
    in [rig_stubs.h]); the two change together. *)

val make : int -> int -> int
(** [make index v] is the point of the device [index] at [v]. Unchecked:
    [0 <= index <= max_index] and [0 <= v <= max_value]. *)

val index : int -> int
val value : int -> int

val max_value : int
(** [max_value] is [2]{^ 47}[ - 1]. *)

val max_index : int
(** [max_index] is [65_535], the most devices a process opens. *)
