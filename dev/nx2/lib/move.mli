(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Movements: the implementation of {!Nx_array.Move}, and the shape facts
    layouts share. *)

val max_rank : int

type range = { start : int; count : int; step : int }
type window = { axis : int; size : int; step : int; dilation : int }

type t =
  | Reshape of int array
  | Broadcast of int array
  | Permute of int array
  | Slice of range array
  | Window of window array

val shape : t -> int array -> int array

val numel : string -> int array -> int
(** [numel fn s] is the number of elements of shape [s]. Raises
    [Invalid_argument] naming [fn] if an extent is negative or the product
    overflows. *)

val pp_ints : Format.formatter -> int array -> unit
(** [pp_ints] formats a shape or an index as [[2; 3]]. *)
