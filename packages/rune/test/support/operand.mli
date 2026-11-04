(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Operands drawn to break code: shapes, views, values, placements. *)

(** {1:values Values} *)

(** The type for an element as drawn. [Least] and [Greatest] are a dtype's ends,
    made at the dtype itself rather than through a float. *)
type value =
  | F of float  (** A float, stored by {!Nx.Dtype.of_float}. *)
  | Least
      (** The least finite value: an integer dtype's least, a float format's
          negated largest finite, [false]. *)
  | Greatest  (** The greatest finite value, as {!Least}. *)
  | C of float * float
      (** A complex element's real and imaginary parts, each rounded to the
          component's dtype as it is stored; at a dtype that is not complex, its
          real part. *)

val pp_value : Format.formatter -> value -> unit
(** [pp_value] formats a float as [%h], the ends as [least] and [greatest], a
    complex element as its parts [%h] joined by [+i]. *)

val value : ('v, 's) Nx.dtype -> value -> 'v
(** [value dt v] is [v] at [dt]. A complex end is its component's end as the
    real part. *)

val pool : value array
(** [pool] is [0], [-0], [±1], [0.5], [-2.5], [3], [7.25], [-0.125], [1e-3],
    [±1e30], [±inf], NaN, the subnormals [±0x1p-1074] and [0x1p-149],
    [±max_float], [±min_float], {!Least} and {!Greatest}. *)

(** {1:shapes Shapes} *)

val shape : int array Windtrap.Gen.t
(** [shape] is a rank from 0 to 3 with each extent 0 (1 in 7), 1 (2 in 7) or 2
    to 4; one shape in 20 is instead of rank 1 to 3 with 300 to 1100 elements,
    past one kernel block. *)

val numel : int array -> int
(** [numel s] is the number of elements of shape [s]. *)

val pp_shape : Format.formatter -> int array -> unit
(** [pp_shape] formats a shape as [[2; 3]]. *)

(** {1:operands Operands} *)

(** The type for the layout an operand is made in. *)
type view =
  | Plain  (** C-contiguous. *)
  | Flipped  (** Every axis reversed: negative strides. *)
  | Strided  (** Every second element of the last axis. *)
  | Broadcast  (** Axis 0 broadcast from one row: stride 0. *)
  | Offset of int
      (** C-contiguous, [k] elements into a larger array, [1 <= k <= 3]. *)
  | Permuted  (** The transpose of a C-contiguous array: axes reversed. *)

val pp_view : Format.formatter -> view -> unit
(** [pp_view] formats a view as its constructor's name in lower case. *)

type t = { view : view; values : value array }
(** The type for operands: a view and the elements its storage holds, in storage
    order. *)

val draw : int array -> t Windtrap.Gen.t
(** [draw s] is an operand of shape [s]: a view, then its elements, 3 in 5 from
    {!pool} and 2 in 5 uniform in [[-4, 4]]. *)

val make : ('v, 's) Nx.dtype -> int array -> t -> ('v, 's, Nx.host) Nx.t
(** [make dt s o] is [o] at [dt] and shape [s], each element made by {!value}. A
    0-d operand is plain but for {!Offset}. *)

(** {1:bits Random bits} *)

val random_bits :
  ('v, 's) Nx.dtype -> int array -> Random.State.t -> ('v, 's, Nx.host) Nx.t
(** [random_bits dt s st] is a value of [dt] and shape [s] whose storage holds
    random bytes. *)

val viewed :
  view -> offset:bool -> int array -> (int array -> ('v, 's, 'd) Nx.t) ->
  ('v, 's, 'd) Nx.t
(** [viewed v ~offset s make] is a value of shape [s] as the view [v] of one
    [make] gives: its last axis stepped by 2 where [Strided], its first
    broadcast where [Broadcast], its storage's first element skipped where
    [offset]. [make] is called once.

    Raises [Invalid_argument] for an {!Offset} view. *)

val raw : ('v, 's, 'd) Nx.t -> int array * string array
(** [raw x] is [x]'s shape and each element's bits, read through a bitcast to
    the unsigned integers of its width, a complex128's as two [uint64]s. *)

(** {1:placements Placements} *)

module Two : Nx.Devices
(** [Two] is a set of the memory devices 0 and 1, computed by
    {!Nx_support.Counting}. *)

(** The type for where an operand lies. *)
type _ where =
  | Host : Nx.host where
  | Device : int -> Two.d where  (** Whole on [Two]'s device [k] alone. *)
  | Cut : Two.d where  (** Cut along axis 0 across [Two]. *)

(** The type for a placement on any set, as a generator draws one. *)
type any_where = Where : 'd where -> any_where

val pp_where : Format.formatter -> 'd where -> unit
(** [pp_where] formats [host], [device k] or [cut along axis 0]. *)

val on : 'd where -> 'd Nx.Placement.t
(** [on w] is [w]'s placement. *)

val where : int array -> any_where Windtrap.Gen.t
(** [where s] is a placement of a value of shape [s]: the host, either device of
    {!Two}, or a cut along axis 0 where [s] has an axis. *)
