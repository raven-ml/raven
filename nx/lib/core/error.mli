(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Error constructors for [nx.core] operations.

    This module centralizes error message formatting for shape, dtype, indexing,
    and validation failures. All constructors in this module raise
    [Invalid_argument].

    Use [cannot], [invalid], and [failed] as generic constructors, or one of the
    specialized helpers below. *)

(** {1:errors Error values} *)

exception Shape_mismatch of string
(** Legacy exception constructor for shape mismatch messages.

    Prefer [shape_mismatch] for current call sites. *)

(** {1:constructors Constructors} *)

val cannot :
  op:string ->
  what:string ->
  from:string ->
  to_:string ->
  ?reason:string ->
  ?hint:string ->
  unit ->
  'a
(** [cannot ~op ~what ~from ~to_ ?reason ?hint ()] raises
    [Invalid_argument] with a message of the form:
    {[<op>: cannot <what> <from> to <to_>[(<reason>)][\nhint: <hint>]]}.
    *)

val invalid :
  op:string -> what:string -> ?reason:string -> ?hint:string -> unit -> 'a
(** [invalid ~op ~what ?reason ?hint ()] raises [Invalid_argument] with a
    message of the form:
    {[<op>: invalid <what>[(<reason>)][\nhint: <hint>]]}.
    *)

val failed :
  op:string -> what:string -> ?reason:string -> ?hint:string -> unit -> 'a
(** [failed ~op ~what ?reason ?hint ()] raises [Invalid_argument] with a
    message of the form:
    {[<op>: <what>[(<reason>)][\nhint: <hint>]]}.
    *)

(** {1:specialized Specialized constructors} *)

val shape_mismatch :
  op:string ->
  expected:int array ->
  actual:int array ->
  ?hint:string ->
  unit ->
  'a
(** [shape_mismatch ~op ~expected ~actual ?hint ()] raises [Invalid_argument]
    for incompatible shapes.

    The message includes both shapes and a derived reason:
    - Per-dimension mismatches if ranks are equal.
    - Element-count mismatch if ranks differ and total sizes differ.
    - Rank incompatibility otherwise. *)

val broadcast_incompatible :
  op:string ->
  shape1:int array ->
  shape2:int array ->
  ?hint:string ->
  unit ->
  'a
(** [broadcast_incompatible ~op ~shape1 ~shape2 ?hint ()] raises
    [Invalid_argument] for broadcast failures and reports incompatible
    dimensions.

    If [hint] is omitted, a default broadcasting rule hint is added. *)

val dtype_mismatch :
  op:string -> expected:string -> actual:string -> ?hint:string -> unit -> 'a
(** [dtype_mismatch ~op ~expected ~actual ?hint ()] raises [Invalid_argument]
    for incompatible dtypes.

    If [hint] is omitted, the default hint suggests casting to [expected]. *)

val axis_out_of_bounds :
  op:string -> axis:int -> ndim:int -> ?hint:string -> unit -> 'a
(** [axis_out_of_bounds ~op ~axis ~ndim ?hint ()] raises [Invalid_argument] for
    an axis outside the valid range for [ndim]. *)

val invalid_shape :
  op:string -> shape:int array -> reason:string -> ?hint:string -> unit -> 'a
(** [invalid_shape ~op ~shape ~reason ?hint ()] raises [Invalid_argument] for a
    shape rejected by validation logic. *)

val empty_input : op:string -> what:string -> 'a
(** [empty_input ~op ~what] raises [Invalid_argument] with reason
    ["cannot be empty"]. *)

(** {1:checks Checks} *)

val check_bounds :
  op:string -> name:string -> value:int -> ?min:int -> ?max:int -> unit -> unit
(** [check_bounds ~op ~name ~value ?min ?max ()] checks inclusive bounds.

    Raises [Invalid_argument] if [min] is specified and [value < min], or if
    [max] is specified and [value > max]. *)

val check_axes : op:string -> axes:int array -> ndim:int -> unit
(** [check_axes ~op ~axes ~ndim] validates axes against [ndim].

    An axis is valid iff [-ndim <= axis < ndim]. Raises [Invalid_argument] on
    the first invalid axis. *)

(** {1:multi_issue Multi-issue errors} *)

val multi_issue : op:string -> (string * string) list -> 'a
(** [multi_issue ~op issues] raises [Invalid_argument] with a multiline message
    that lists each [(issue, detail)] pair. *)
