(** Shape operations for multi-dimensional arrays.

    This module provides fundamental operations on array shapes including index
    conversions, broadcasting, and stride calculations. *)

type t = int array
(** Shape representation as array of dimension sizes. *)

(** {1 Basic Operations} *)

val numel : t -> int
(** [numel shape] computes total number of elements. *)

val equal : t -> t -> bool
(** [equal s1 s2] tests shape equality. *)

(** {1 Strides} *)

val c_contiguous_strides : t -> int array
(** [c_contiguous_strides shape] computes row-major strides.

    Handles zero-size dimensions correctly. *)

(** {1 Index Conversions} *)

val ravel_index : int array -> int array -> int
(** [ravel_index indices strides] computes linear offset.

    @raise Invalid_argument if array lengths differ *)

val unravel_index : int -> t -> int array
(** [unravel_index offset shape] converts to multi-dimensional indices.

    Assumes C-contiguous layout.

    @raise Invalid_argument if offset out of bounds *)

(** {1 Shape Manipulation} *)

val resolve_neg_one : t -> int array -> t
(** [resolve_neg_one current_shape new_spec] infers dimension marked with -1.

    @raise Invalid_argument if multiple -1 or size mismatch *)

val broadcast : t -> t -> t
(** [broadcast shape_a shape_b] computes broadcast result shape.

    Follows NumPy rules: dimensions match if equal or one is 1.

    @raise Invalid_argument if shapes incompatible *)

val broadcast_index : int array -> t -> int array
(** [broadcast_index target_indices source_shape] maps indices for broadcasting.

    Returns indices in source shape corresponding to target position. *)

(** {1 Pretty Printing} *)

val pp : Format.formatter -> t -> unit
(** [pp fmt shape] prints shape in [2x3x4] format. *)

val to_string : t -> string
(** [to_string shape] converts to string "[2x3x4]". *)
