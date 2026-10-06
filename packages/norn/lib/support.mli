(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Sets of values a distribution gives positive density.

    A support names the set; it carries the bounds that do not depend on a
    tensor. Each continuous support has one bijector ({!Norn.Bij}) whose
    coordinates cover it. *)

(** The type for supports. *)
type t =
  | Real  (** The real numbers. *)
  | Greater of float  (** The reals above a bound, exclusive. *)
  | Interval of float * float  (** The reals between two bounds, exclusive. *)
  | Simplex of int  (** Vectors of [n] positive components that sum to one. *)
  | Ordered  (** Vectors whose components strictly increase. *)
  | Correlation_cholesky of int
      (** Lower-triangular [n × n] matrices with a positive diagonal and rows of
          unit norm: Cholesky factors of correlation matrices. *)
  | Sum_to_zero  (** Vectors whose components sum to zero. *)
  | Integers_from of int  (** The integers from a bound, inclusive. *)
  | Integer_interval of int * int
      (** The integers between two bounds, inclusive. *)
  | Boolean  (** [false] and [true]. *)

val equal : t -> t -> bool
(** [equal s s'] is [true] iff [s] and [s'] are the same constructor with equal
    arguments, floats compared with [Float.equal]. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf s] formats [s] as a set: [(-inf, inf)], [(0, inf)], [(0, 1)],
    [{0, 1, 2, ...}], [{0, 1, ..., 10}], [{false, true}], or a phrase such as
    [simplex of 3]. *)
