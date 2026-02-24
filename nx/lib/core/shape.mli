(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Concrete tensor shapes.

    A shape is an array of non-negative dimension sizes in row-major order. *)

type t = int array
(** The type for concrete shapes. *)

(** {1:basic Basic operations} *)

val numel : t -> int
(** [numel shape] is the product of dimensions in [shape].

    [numel [||]] is [1]. *)

val equal : t -> t -> bool
(** [equal s0 s1] is [true] iff [s0] and [s1] are structurally equal. *)

(** {1:strides Strides} *)

val c_contiguous_strides : t -> int array
(** [c_contiguous_strides shape] is the row-major stride vector of [shape].

    For any zero-size dimension, strides to its left are propagated with zero
    according to the implementation's canonical rule. *)

(** {1:indexing Index conversion} *)

val ravel_index : int array -> int array -> int
(** [ravel_index indices strides] is the linear offset
    [sum_i (indices.(i) * strides.(i))].

    Raises [Invalid_argument] if the array lengths differ.

    {b Note.} This function does not perform bounds checks on [indices]. *)

val unravel_index : int -> t -> int array
(** [unravel_index k shape] is the multi-index of [k] in a C-contiguous layout
    of [shape].

    For [shape = [||]], [k] must be [0].

    For zero-size shapes, only [k = 0] is accepted and the result is an array of
    zeros with the same rank as [shape].

    Raises [Invalid_argument] if [k] is out of bounds for [shape]. *)

val unravel_index_into : int -> t -> int array -> unit
(** [unravel_index_into k shape dst] is like {!unravel_index} but writes indices
    into [dst].

    [dst] must have length [Array.length shape].

    Raises [Invalid_argument] if [k] is out of bounds for [shape].

    {b Warning.} If [dst] has the wrong length, array access may raise
    [Invalid_argument] via OCaml's bounds checks. *)

(** {1:transform Shape transformations} *)

val resolve_neg_one : t -> int array -> t
(** [resolve_neg_one current_shape new_spec] resolves a single [-1] entry in
    [new_spec] using [numel current_shape].

    Raises [Invalid_argument] if:
    - [new_spec] contains more than one [-1].
    - The inferred size is not integral with the specified dimensions.
    - The specification is incompatible with zero-size inference rules. *)

val broadcast : t -> t -> t
(** [broadcast a b] is the broadcasted shape of [a] and [b] using NumPy rules
    (right alignment; dimensions are compatible iff equal or one is [1]).

    Raises [Invalid_argument] if the shapes are not broadcast-compatible. *)

val broadcast_index : int array -> t -> int array
(** [broadcast_index target_idx source_shape] maps a target index to the
    corresponding index in [source_shape] under broadcasting.

    Dimensions of [source_shape] equal to [1] map to index [0]. *)

val broadcast_index_into : int array -> t -> int array -> unit
(** [broadcast_index_into target_idx source_shape dst] is like
    {!broadcast_index} but writes into [dst].

    [dst] must have length [Array.length source_shape]. *)

(** {1:format Formatting} *)

val pp : Format.formatter -> t -> unit
(** [pp] formats shapes with the same syntax as [to_string]. *)

val to_string : t -> string
(** [to_string shape] formats [shape] as a bracketed comma-separated list, for
    example [[2,3,4]]. *)
