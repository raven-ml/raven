(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Tensor shell over the unified IR.

    A {!t} is a thin, mutable wrapper around a tensor-stage {!Tolk_uop.Uop.t}.
    It carries no data of its own: the shape, dtype, and device of a tensor are
    read directly from its underlying node. Operations build new nodes and wrap
    them, so a tensor is a handle onto a lazily constructed computation graph.

    This module holds the primitives the operation modules
    ({!Movement}, {!Dtype_ops}, {!Creation}, {!Elementwise}, {!Reduce}) build
    on: the wrapper itself, shape and dtype accessors, axis normalisation,
    scalar constants, and the arithmetic-node constructors. *)

type t
(** A handle onto a tensor-stage IR node. *)

val of_uop : Tolk_uop.Uop.t -> t
(** [of_uop u] wraps [u]. No validation is performed; [u] must be a
    tensor-stage node. *)

val uop : t -> Tolk_uop.Uop.t
(** [uop t] is the node underlying [t]. *)

val set_uop : t -> Tolk_uop.Uop.t -> unit
(** [set_uop t u] repoints [t] at node [u]. Used by realization to rebind a
    tensor onto its computed buffer once its value is materialised. This is the
    same in-place reassignment of a tensor's node that the reference frontend
    performs when it replaces a lazy graph with its realized buffer. *)

val apply_map : (Tolk_uop.Uop.t * Tolk_uop.Uop.t) list -> unit
(** [apply_map mappings] repoints every live tensor whose graph contains the
    first component of a pair so that it refers to the second instead.
    Replacement values are final: a value may mention its own key without the
    rewrite recursing into it. Every tensor is tracked weakly from creation,
    so this reaches all reachable handles. Used by in-place assignment to
    embed a write into the graphs of the tensors that alias the written
    buffer, and by realization to rebind computed nodes onto their buffers. *)

(** {1 Shape and type} *)

val dtype : t -> Tolk_uop.Dtype.t
(** [dtype t] is the element type of [t]. *)

val val_dtype : t -> Tolk_uop.Dtype.Val.t
(** [val_dtype t] is [dtype t] as a value dtype.

    @raise Invalid_argument if [t] has a pointer dtype. *)

val device : t -> Tolk_uop.Uop.device option
(** [device t] is the device [t] is placed on, or [None] for an unplaced
    (constant) tensor. *)

val symbolic_shape : t -> Tolk_uop.Uop.t list
(** [symbolic_shape t] is the shape of [t] as a list of dimension nodes, one
    per axis. Dimensions may be symbolic. *)

val shape : t -> int list
(** [shape t] is the shape of [t] as concrete integers.

    @raise Invalid_argument if any dimension is symbolic. *)

val ndim : t -> int
(** [ndim t] is the number of axes of [t]. *)

val numel : t -> int
(** [numel t] is the total number of elements of [t].

    @raise Invalid_argument if any dimension is symbolic. *)

val resolve_dim : ?extra:bool -> t -> int -> int
(** [resolve_dim t d] normalises axis index [d] against [ndim t], mapping a
    negative [d] to [d + ndim t]. With [~extra:true] the valid range is
    widened by one, as needed when inserting a new axis.

    @raise Invalid_argument if [d] is out of range. *)

(** {1 Scalars} *)

(** A Python-style scalar literal, used as a fill or operand value. Its dtype,
    when not given explicitly, follows the variant: {!Sint} is the default
    integer type, {!Sfloat} the default float, {!Sbool} boolean. *)
type scalar = Sint of int | Sfloat of float | Sbool of bool

val scalar_const : Tolk_uop.Dtype.Val.t -> scalar -> Tolk_uop.Const.t
(** [scalar_const dt s] is the constant value [s] coerced to dtype [dt]. *)

val i : int -> t
(** [i n] is a scalar tensor holding integer [n] at the default integer
    dtype. *)

val f : float -> t
(** [f x] is a scalar tensor holding float [x] at the default float dtype. *)

val b : bool -> t
(** [b v] is a scalar tensor holding boolean [v]. *)

(** {1 Node constructors} *)

val shape_uop : int list -> Tolk_uop.Uop.t
(** [shape_uop dims] encodes an integer shape as a shape argument node. *)

val symbolic_shape_uop : Tolk_uop.Uop.t list -> Tolk_uop.Uop.t
(** [symbolic_shape_uop dims] encodes a shape of (possibly symbolic) dimension
    nodes as a shape argument node. *)

val alu_unary : Tolk_uop.Ops.t -> t -> t
(** [alu_unary op t] applies unary arithmetic [op] to [t]. *)

val alu_binary : Tolk_uop.Ops.t -> t -> t -> t
(** [alu_binary op a b] applies binary arithmetic [op] to same-shape,
    same-dtype operands [a] and [b]. Callers are responsible for prior
    broadcasting and promotion. *)

val alu_ternary : Tolk_uop.Ops.t -> t -> t -> t -> t
(** [alu_ternary op a b c] applies ternary arithmetic [op] to [a], [b],
    and [c]. *)

(** {1 Broadcasting} *)

val broadcast_shape : int list list -> int list
(** [broadcast_shape shapes] is the common shape all of [shapes] broadcast to,
    aligning axes from the last and stretching size-[1] axes. A zero along any
    axis makes the result zero there. This is the concrete-integer view of
    {!Tolk_uop.Uop.broadcast_shape}.

    @raise Invalid_argument if the shapes are incompatible. *)

val broadcasted : ?reverse:bool -> t -> t -> t * t
(** [broadcasted a b] broadcasts [a] and [b] to a common shape and promotes
    them to a common dtype. With [~reverse:true] the returned pair is swapped.

    The concrete implementation lives in {!Op} and is installed into
    {!broadcasted_hook} when that module is linked; calling this before then
    raises [Failure]. *)

val broadcasted_hook : (reverse:bool -> t -> t -> t * t) ref
(** Mutable hook backing {!broadcasted}. {!Op} assigns it on initialisation to
    break the dependency cycle between the element-wise operations, which need
    broadcasting, and the promotion logic, which is defined alongside the
    composed operations. *)
