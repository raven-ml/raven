(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Tensor-core fragment layouts for hardware matrix multiplication. *)

type fragment = string list * string list
(** [(lanes, elements)] lists the tile-coordinate bits selected by lane and
    element indices, least significant bit first. Coordinates ["m0"], ["n1"]
    and ["k2"] identify bits of the M, N and K tile dimensions. An input lane
    bit from the other input's output dimension denotes a broadcast. *)

type t = private {
  dtype_in : Tolk_uop.Dtype.t; (** Type of A and B. *)
  dtype_out : Tolk_uop.Dtype.t; (** Type of the accumulator. *)
  frag_a : fragment; (** A's M-by-K fragment. *)
  frag_b : fragment; (** B's K-by-N fragment. *)
  frag_c : fragment; (** The accumulator's M-by-N fragment. *)
  dims : int * int * int; (** Derived [(n, m, k)] tile dimensions. *)
  threads : int; (** Derived cooperating lane count. *)
}
(** A validated hardware matrix-multiply-accumulate layout. Tile dimensions
    and lane counts are derived once from the fragment bits. *)

val create :
  dtype_in:Tolk_uop.Dtype.t -> dtype_out:Tolk_uop.Dtype.t ->
  frag_a:fragment -> frag_b:fragment -> frag_c:fragment -> t
(** [create ~dtype_in ~dtype_out ~frag_a ~frag_b ~frag_c] validates the three
    fragments and derives their tile geometry. Each fragment covers all its
    own bits exactly once; only lane bits may broadcast. A and B must relabel
    K identically. Raises [Invalid_argument] for an invalid layout. *)

val axis_coords : t -> string list
(** [axis_coords tc] lists tile bits in split order: N, M, then K, with each
    dimension ordered from its least significant bit. *)

val base_upcast_axes : t -> string list
(** [base_upcast_axes tc] lists accumulator element bits followed by K bits,
    most significant bit first within each group. *)

val relabel : t -> (string * string) list list
(** [relabel tc] maps tile coordinates to fragment-slot coordinates for A
    and B, using accumulator lanes and the leading element slots. *)

val dtype_name : Tolk_uop.Dtype.t -> string
(** [dtype_name dt] is the C spelling of [dt] used inside tensor-core
    primitive names — ["half"], ["char"], ["float8_e4m3"]. This is the sole
    owner of that mapping: the renderer derives the emitted primitive's name
    from it, so a second spelling anywhere would desynchronise a [#define]
    from its call site.

    @raise Invalid_argument if [dt] has no tensor-core spelling, rather than
    falling back to a generic one that would name a primitive that does not
    exist. *)

val to_string : t -> string
(** [to_string tc] is ["WMMA_N_M_K_in_out"], built from {!dtype_name}. *)

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
