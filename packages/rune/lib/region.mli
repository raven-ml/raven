(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Regions of an array, as a [Slice] by ranges selects them, and whether
    two may share an element. *)

type axis = { first : int; count : int; step : int; back : bool }
(** The type for a region along an axis, ascending: its first index, its
    count, its step, [1] where it holds one element, and whether the region
    runs backwards. *)

val axis : Nx_array.Move.range -> axis
(** [axis r] is the range [r] ascending. *)

val axes : Nx_array.Move.range iarray -> axis array
(** [axes rs] is {!axis} of each range of [rs]. *)

val extent : axis -> int
(** [extent a] is the number of indices from [a]'s first to its last. *)

val gcd : int -> int -> int
(** [gcd a b] is the greatest common divisor of [a] and [b], [|a|] where [b]
    is [0]. *)

val meet : axis array -> axis array -> bool
(** [meet b c] is [false] where the regions [b] and [c] share no element: one
    holds none, or along an axis their spans miss each other or their steps
    leave no common index. Else it is [true], though two strided regions may
    still share none. *)
