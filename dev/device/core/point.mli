(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Points as ints: a device's index in bits 47 to 62, a value in bits 0 to 46.
   The int 0 is no point. *)

val make : int -> int -> int
(* [make index v] is the point of the device [index] at [v]. *)

val index : int -> int
val value : int -> int
val max_value : int
(* [max_value] is the largest value, [2]{^ 47}[ - 1]. *)

val max_index : int
(* [max_index] is the largest device index, [65_535]. *)
