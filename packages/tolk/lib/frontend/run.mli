(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Realization: turning a lazy tensor graph into computed values.

    A {!Tensor.t} is a handle onto a lazily built computation. The functions
    here schedule that computation, run it on the CPU backend, and read the
    result back to the host. Building tensors stays pure; nothing executes until
    a value is requested. *)

(** {1 Host data} *)

val of_float_array : shape:int list -> float array -> Tensor.t
(** [of_float_array ~shape data] is a [float32] tensor of shape [shape] holding
    [data] in row-major order. The element count of [shape] must equal the
    length of [data]. *)

val of_int_array : shape:int list -> int array -> Tensor.t
(** [of_int_array ~shape data] is an [int32] tensor of shape [shape] holding
    [data] in row-major order. The element count of [shape] must equal the
    length of [data]. *)

(** {1 Realization} *)

val realize : Tensor.t -> Tensor.t
(** [realize t] computes [t]'s value and rebinds [t] onto the resulting buffer,
    returning [t]. Subsequent reads reuse the computed buffer. *)

val realize_many : Tensor.t list -> unit
(** [realize_many ts] realizes [ts] together, sharing a single schedule so that
    work common to several of them is computed once. *)

(** {1 Reading values}

    Each reader realizes the tensor if needed, then copies its buffer to the
    host. The tensor must have a concrete (non-symbolic) shape. *)

val data : Tensor.t -> bytes
(** [data t] is the raw little-endian bytes of [t]'s buffer. *)

val to_float_array : Tensor.t -> float array
(** [to_float_array t] is [t]'s elements decoded as [float32], in row-major
    order. *)

val to_int_array : Tensor.t -> int array
(** [to_int_array t] is [t]'s elements decoded as [int32], in row-major
    order. *)

val item_float : Tensor.t -> float
(** [item_float t] is the single [float32] element of a one-element [t].

    @raise Invalid_argument if [t] does not have exactly one element. *)

val item_int : Tensor.t -> int
(** [item_int t] is the single [int32] element of a one-element [t].

    @raise Invalid_argument if [t] does not have exactly one element. *)

(** {1 Data-dependent selection}

    {!Op.masked_select} and {!Op.nonzero} require the output length as [~size]
    because a tensor graph must have a static shape. The variants here lift that
    requirement: with [size] omitted they realize the number of selected
    elements and use it as the length, so the result shrinks to exactly what was
    selected — at the cost of running a small computation while the graph is
    still being built, which a purely lazy or captured program cannot do. Pass
    [size] to stay fully lazy. *)

val masked_select :
  ?fill_value:Tensor.scalar -> ?size:int -> Tensor.t -> Tensor.t -> Tensor.t
(** [masked_select t mask] is the elements of [t] where [mask] is true, packed
    into a 1-D tensor. With [size] it behaves as {!Op.masked_select}; without,
    the length is the number of elements kept. *)

val nonzero : ?fill_value:Tensor.scalar -> ?size:int -> Tensor.t -> Tensor.t
(** [nonzero t] is the coordinates of the non-zero elements of [t], one row per
    element. With [size] it behaves as {!Op.nonzero}; without, the row count is
    the number of non-zero elements. *)
