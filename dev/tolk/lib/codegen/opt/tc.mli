(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** TensorCore helpers and hardware configuration tables.

    The record type lives in {!Renderer}; this module provides helper
    functions and hardware tables. *)

val get_reduce_axes : Renderer.tensor_core -> (int * int) list
(** [get_reduce_axes tc] returns [(i, 2)] pairs, one per power-of-2 factor of
    the K dimension. *)

val get_upcast_axes : Renderer.tensor_core -> string list
(** [get_upcast_axes tc] returns opts starting with ["u"]. *)

val get_local_axes : Renderer.tensor_core -> string list
(** [get_local_axes tc] returns opts starting with ["l"]. *)

val base_shape_str : Renderer.tensor_core -> string list
(** [base_shape_str tc] builds the shape string from opts (with independent
    counters for ["u"] and ["l"]) followed by reduce axis labels. *)

val base_upcast_axes : Renderer.tensor_core -> string list
(** [base_upcast_axes tc] returns reduce and upcast labels in reverse order. *)

val permutes_for_shape_str :
  Renderer.tensor_core -> string list -> int list * int list
(** [permutes_for_shape_str tc shape_str] computes the two permutation vectors
    from the swizzle remapping applied to [shape_str]. *)

val dtype_name : Tolk_ir.Dtype.scalar -> string
(** [dtype_name s] is the canonical scalar name for WMMA instructions
    (e.g. [Float16 -> "half"], [Float32 -> "float"]). *)

val to_string : Renderer.tensor_core -> string
(** [to_string tc] is the WMMA instruction name,
    e.g. ["WMMA_8_16_16_half_float"]. *)

val validate : Renderer.tensor_core -> unit
(** [validate tc] checks structural invariants of [tc]. Raises [Failure] on the
    first violation. All hardware tables are validated at module init. *)

(** {1 Hardware configuration tables} *)

val cuda_sm75 : Renderer.tensor_core list
val cuda_sm80 : Renderer.tensor_core list
val cuda_sm89 : Renderer.tensor_core list
val amd_rdna3 : Renderer.tensor_core list
val amd_rdna4 : Renderer.tensor_core list
val amd_cdna3 : Renderer.tensor_core list
val amd_cdna4 : Renderer.tensor_core list
val metal : Renderer.tensor_core list
val amx : Renderer.tensor_core list
val intel : Renderer.tensor_core list
