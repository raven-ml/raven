(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The implementation of {!Norn.Support}. *)

type t =
  | Real
  | Greater of float
  | Interval of float * float
  | Simplex of int
  | Ordered
  | Correlation_cholesky of int
  | Sum_to_zero
  | Integers_from of int
  | Integer_interval of int * int
  | Boolean

val equal : t -> t -> bool

val pp : Format.formatter -> t -> unit
