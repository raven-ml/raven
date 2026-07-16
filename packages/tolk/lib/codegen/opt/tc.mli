(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Tinygrad tensor core definitions and swizzle helpers.

    Defines tinygrad-backed hardware WMMA/MFMA configurations for NVIDIA, AMD,
    and Apple Metal, and provides the axis-remapping logic needed by
    {!Postrange} to lower matmuls into tensor core instructions.

    See also {!Renderer.tensor_core}, {!Postrange}. *)

(** {1:types Types} *)

(** The type for tensor core (WMMA/MFMA) configurations.

    Describes a hardware matrix-multiply-accumulate instruction
    [D = A * B + C] where A is (M x K), B is (K x N), and C/D are
    (M x N).  The configuration specifies tile geometry, thread mapping,
    dtype requirements, and the dimension swizzle needed to lay data out
    for the instruction. *)
type t = private {
  dims : int * int * int;
      (** [(n, m, k)] matrix-multiply tile dimensions. *)
  threads : int;
      (** Number of threads cooperating on one tile. *)
  elements_per_thread : int * int * int;
      (** [(a, b, c)] elements each thread contributes for operands A, B,
          and accumulator C. *)
  dtype_in : Tolk_uop.Dtype.t;
      (** Element type of the A and B input operands. *)
  dtype_out : Tolk_uop.Dtype.t;
      (** Element type of the C accumulator operand. *)
  opts : string list;
      (** Scheduling option strings (["u0"], ["l1"], …) applied when this
          tensor core is active.  Passed to the kernel optimiser to
          configure tiling and unrolling. *)
  swizzle :
    (string list * string list * string list)
    * (string list * string list * string list);
      (** Operand layout remapping as
          [((a_local, a_upcast, a_reduce), (b_local, b_upcast, b_reduce))].
          Each triple contains (local, upcast, reduce) dimension index
          strings describing the physical layout required by the hardware
          instruction. *)
}

(** [create ~dims ~threads ~elements_per_thread ~dtype_in ~dtype_out ~opts
     ~swizzle] is a validated tensor core configuration.

    Raises [Failure] if the configuration violates the tensor-core invariants.
    This mirrors tinygrad's dataclass construction boundary. *)
val create :
  dims:int * int * int ->
  threads:int ->
  elements_per_thread:int * int * int ->
  dtype_in:Tolk_uop.Dtype.t ->
  dtype_out:Tolk_uop.Dtype.t ->
  opts:string list ->
  swizzle:
    (string list * string list * string list)
    * (string list * string list * string list) ->
  t

(** {1:helpers Helpers} *)

(** These helpers are exposed for the tensor-core lowering pipeline. They are
    not a renderer-selection API; renderer setup owns target selection. *)

val get_reduce_axes : t -> (int * int) list
(** [get_reduce_axes tc] is the reduce axes for [tc]: one [(i, 2)] pair
    per power-of-two factor in the K dimension. *)

val base_shape_str : t -> string list
(** [base_shape_str tc] is the shape string before the reduce UNROLL:
    numbered local/upcast labels from [tc.opts], then reduce labels. *)

val base_upcast_axes : t -> string list
(** [base_upcast_axes tc] is the upcast + reduce axis names in reverse
    order, used to define the UNROLL axes after opts are applied. *)

val permutes_for_shape_str : t -> string list -> int list * int list
(** [permutes_for_shape_str tc shape_str] is the two permutation vectors
    (for operands A and B) that reorder [shape_str] according to the
    swizzle. *)

val to_string : t -> string
(** [to_string tc] is ["WMMA_N_M_K_in_out"]. *)

(** {1:definitions Definitions}

    Each list contains one {!t} per supported dtype pair.  All entries
    are validated when constructed. *)

val cuda_sm75 : t list
(** NVIDIA SM 7.5 (Turing). *)

val cuda_sm80 : t list
(** NVIDIA SM 8.0 (Ampere). *)

val cuda_sm89 : t list
(** NVIDIA SM 8.9 (Ada Lovelace). *)

val amd_rdna3 : t list
(** AMD RDNA 3 WMMA. *)

val amd_rdna4 : t list
(** AMD RDNA 4 WMMA. *)

val amd_cdna3 : t list
(** AMD CDNA 3 MFMA. *)

val amd_cdna4 : t list
(** AMD CDNA 4 MFMA. *)

val metal : t list
(** Apple Metal simdgroup_matrix. *)
