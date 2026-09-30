(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Tensor cores.

    A tensor core computes [D = A * B + C] on small tiles, with [A] of [M] rows
    and [K] columns, [B] of [K] rows and [N] columns, and [C] and [D] of [M]
    rows and [N] columns. Every dimension is a power of two. A group of threads,
    a warp, computes one product together: each thread holds some elements of
    each operand, in registers.

    A {e fragment} says which thread holds which element. A thread is numbered
    by its lane in the group, and its registers by their element index; each bit
    of the lane number and of the element index is one bit of the tile
    coordinates. The tensor cores of each target are values of this module. *)

(** {1:fragments Fragments} *)

(** The type for the bits of tile coordinates: [M i] is bit [i] of the row,
    [N i] bit [i] of the column, [K i] bit [i] of the reduction index. *)
type bit = N of int | M of int | K of int

val equal_bit : bit -> bit -> bool
(** [equal_bit b0 b1] is [true] iff [b0] and [b1] are the same bit. *)

val pp_bit : Format.formatter -> bit -> unit
(** [pp_bit] formats a bit as its dimension in lower case and its index: [m0],
    [k3]. *)

type fragment = {
  lanes : bit list;
      (** The bit of the coordinates each bit of the lane number gives, least
          significant first. *)
  elements : bit list;
      (** The bit each bit of the element index gives, least significant first.
      *)
}
(** The type for fragments: where an operand's elements are held. A lane bit of
    a dimension the operand does not have (an [N] bit for [A], an [M] bit for
    [B]) broadcasts: lanes that differ only by it hold the same elements. *)

(** {1:cores Tensor cores} *)

type t = private {
  dtype_in : Dtype.t;  (** The type of [A] and [B]. *)
  dtype_out : Dtype.t;  (** The type of [C] and [D]. *)
  frag_a : fragment;  (** [A]'s fragment, of [M] and [K] bits. *)
  frag_b : fragment;  (** [B]'s fragment, of [K] and [N] bits. *)
  frag_c : fragment;
      (** [C]'s and [D]'s fragment, of [M] and [N] bits. Its lanes are the
          threads of the warp, and its elements are unrolled in each thread. *)
}
(** The type for tensor cores. *)

val v :
  dtype_in:Dtype.t ->
  dtype_out:Dtype.t ->
  frag_a:fragment ->
  frag_b:fragment ->
  frag_c:fragment ->
  t
(** [v ~dtype_in ~dtype_out ~frag_a ~frag_b ~frag_c] is the tensor core with
    these types and fragments. The bits of the tile are those {!axis_coords}
    derives.

    Raises [Invalid_argument] naming the fragment at fault unless:
    - the three fragments have as many lanes;
    - each fragment's bits are distinct, its elements are bits of its own
      dimensions, it has every bit of its own dimensions, and its other bits are
      [M] or [N] bits;
    - [A] and [B] hold the [K] bits in the same order, elements first. *)

val dims : t -> int * int * int
(** [dims tc] is [(n, m, k)], the tile's dimensions [N], [M] and [K]. *)

val threads : t -> int
(** [threads tc] is the number of threads of the warp: 2 to the number of lanes.
*)

val axis_coords : t -> bit list
(** [axis_coords tc] is every bit of the tile, the [N] bits, then the [M] bits,
    then the [K] bits, each from bit [0]. A dimension has the bits up to the
    highest that [A]'s or [C]'s fragment names. *)

val base_upcast_axes : t -> bit list
(** [base_upcast_axes tc] is the bits of [C]'s elements, then the [K] bits, most
    significant first. *)

val relabel : t -> (bit * bit) list * (bit * bit) list
(** [relabel tc] is, for [A] and for [B], each bit of its fragment paired with
    the bit it takes the place of when its elements are loaded as [C]'s: its
    lane bits pair with [C]'s lane bits in order, its element bits with the
    first of {!base_upcast_axes}, from the last. *)

val frag_coords :
  t ->
  (int * int) array array * (int * int) array array * (int * int) array array
(** [frag_coords tc] is, for [A], [B] and [C] in turn, the coordinates of the
    element that each lane holds at each element index: [(row, column)] for [A]
    and [C], [(k, column)] for [B]. *)

val equal : t -> t -> bool
(** [equal tc0 tc1] is [true] iff [tc0] and [tc1] have the same types and
    fragments. *)

val pp : Format.formatter -> t -> unit
(** [pp] formats a tensor core as its types and fragments:
    [TensorCore(dtype_in=dtypes.half, dtype_out=dtypes.float, frag_a=(('k1',
     'm0', ...), ('k0',)), frag_b=..., frag_c=...)]. *)

(** {1:targets Targets}

    The tensor cores of each target, one per tile shape and pair of types. *)

val cuda_sm75 : t list
(** [cuda_sm75] is NVIDIA's tensor cores from compute capability 7.5:
    [8 x 16 x 8] tiles of {!Dtype.Float16}, into {!Dtype.Float32} or
    {!Dtype.Float16}. *)

val cuda_sm80 : t list
(** [cuda_sm80] is those from compute capability 8.0: [cuda_sm75], [8 x 16 x 16]
    tiles of {!Dtype.Float16} and {!Dtype.Bfloat16}, and [8 x 16 x 8] tiles of
    {!Dtype.Float32}. *)

val cuda_sm89 : t list
(** [cuda_sm89] is those from compute capability 8.9: [cuda_sm80] and
    [8 x 16 x 32] tiles of the two 8-bit floats. *)

val cuda : string -> t list
(** [cuda arch] is the tensor cores of the NVIDIA architecture [arch], named
    [sm_] followed by its compute capability: {!cuda_sm89} from [89],
    {!cuda_sm80} from [80], {!cuda_sm75} from [75], and none below.

    Raises [Invalid_argument] if [arch] has no compute capability after its
    first three characters. *)

val amd_rdna3 : t list
(** [amd_rdna3] is RDNA 3's [16 x 16 x 16] tensor cores. *)

val amd_rdna4 : t list
(** [amd_rdna4] is RDNA 4's [16 x 16 x 16] tensor cores. *)

val amd_cdna3 : t list
(** [amd_cdna3] is CDNA 3's tensor cores, of [K] 32 for the 8-bit floats without
    infinities and 16 for the 16-bit floats. *)

val amd_cdna4 : t list
(** [amd_cdna4] is CDNA 4's tensor cores, of [K] 128 and 32 for the 8-bit
    floats, 32 and 16 for the 16-bit floats. *)

val amd : string -> t list
(** [amd arch] is the tensor cores of the AMD architecture [arch]: {!amd_cdna3}
    for [gfx942], {!amd_cdna4} for [gfx950], {!amd_rdna4} for [gfx1200] and
    [gfx1201], and {!amd_rdna3} for any other. *)

val metal : t list
(** [metal] is Apple's [8 x 8 x 8] tensor cores. *)
