(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Data type system for tolk.

    Scalar types with priority-based promotion, vectorization, and pointer types
    for GPU memory spaces. *)

(** {1:scalars Scalar Types} *)

(** Scalar data type identity. Ordered by promotion priority from [Bool]
    (lowest) to [Float64] (highest). [Void] and [Index] have priority [-1]. *)
type scalar =
  | Void
  | Bool
  | Int8
  | Int16
  | Int32
  | Int64
  | Uint8
  | Uint16
  | Uint32
  | Uint64
  | Float16
  | Bfloat16
  | Float32
  | Float64
  | Fp8e4m3  (** 8-bit float: 4-bit exponent, 3-bit mantissa. *)
  | Fp8e5m2  (** 8-bit float: 5-bit exponent, 2-bit mantissa. *)
  | Index
      (** Symbolic index dtype for loop counters and address arithmetic. Uses
          800 bits as a non-machine sentinel and does not participate in dtype
          promotion. *)

(** {1:addr_spaces Address Spaces} *)

(** GPU memory address space. *)
type addr_space =
  | Global  (** Global device memory. *)
  | Local  (** Shared/local memory (workgroup scope). *)
  | Reg  (** Register storage. *)

(** {1:types Data Types} *)

type t = {
  scalar : scalar;  (** The element type. *)
  count : int;  (** Vector width ([1] for scalar types). *)
}
(** Data type with optional vectorization. *)

type ptr = {
  base : t;  (** Pointed-to element type. *)
  addrspace : addr_space;  (** Memory address space. *)
  v : int;  (** Pointer vector width (distinct from [base.count]). *)
  size : int;  (** Element count, or [-1] for unbounded. *)
  image : (int * int) option;
      (** Optional 2D image shape [(height, width)] for image-backed buffers. *)
}
(** Pointer type. *)

(** {1:constructors Constructors}

    Scalar dtype constants. All have [count = 1]. *)

val void : t
val bool : t
val int8 : t
val int16 : t
val int32 : t
val int64 : t
val uint8 : t
val uint16 : t
val uint32 : t
val uint64 : t
val float16 : t
val bfloat16 : t
val float32 : t
val float64 : t
val fp8e4m3 : t
val fp8e5m2 : t
val index : t

val default_float : t
(** Default floating-point dtype for generic numeric promotion. Used by
    {!least_upper_float} and {!sum_acc_dtype}. *)

val default_int : t
(** Default integer dtype for generic numeric promotion. Used by
    {!sum_acc_dtype}. *)

(** {1:predicates Type Predicates} *)

val is_float : t -> bool
(** [is_float t] is [true] for float types including fp8 variants. *)

val is_int : t -> bool
(** [is_int t] is [true] for signed and unsigned integers. Includes [Index]. *)

val is_unsigned : t -> bool
(** [is_unsigned t] is [true] for unsigned integer types ([Uint8], [Uint16],
    [Uint32], [Uint64]). *)

val is_bool : t -> bool
(** [is_bool t] is [true] for boolean types, including vectorized bools. *)

val is_fp8 : t -> bool
(** [is_fp8 t] is [true] for 8-bit float types ([Fp8e4m3], [Fp8e5m2]). *)

(** {1:properties Type Properties} *)

val bitsize : t -> int
(** [bitsize t] is the total size in bits ([scalar_bitsize * count]). *)

val itemsize : t -> int
(** [itemsize t] is the size in bytes, rounded up. *)

val priority : t -> int
(** [priority t] is the promotion priority of the scalar component. Higher
    priority types absorb lower ones in {!least_upper_dtype}. *)

(** {1:operations Type Operations} *)

val vec : t -> int -> t
(** [vec t n] creates a vector type of [n] elements. Returns [t] unchanged if
    [n = 1] or [t] is [void]. [index.vec 0] is allowed to represent empty shape
    vectors (scalar tensors with zero dimensions).

    Raises [Invalid_argument] if [t.count <> 1] (already vectorized), [n < 0],
    or [n = 0] on a non-index dtype. *)

val scalar_of : t -> t
(** [scalar_of t] is [t] with [count = 1]. *)

type bound =
  [ `Bool of bool | `SInt of int64 | `UInt of int64 | `Float of float ]
(** Numeric bounds for dtypes. Returned by {!min} and {!max}.

    [`UInt] values are represented as raw 64-bit unsigned bit patterns in
    [int64] (for example, uint64 max is [`UInt Int64.minus_one]). *)

val min : t -> bound
(** [min t] is the lower bound representable by [t].

    Raises [Invalid_argument] if [t] is [void]. *)

val max : t -> bound
(** [max t] is the upper bound representable by [t].

    Raises [Invalid_argument] if [t] is [void]. *)

(** {1:promotion Type Promotion} *)

val least_upper_dtype : t list -> t
(** [least_upper_dtype ts] is the least upper bound in the promotion lattice.
    Promotion is total (always succeeds for valid numeric dtypes) at the cost of
    some lossy edges. Result always has [count = 1].

    Raises [Invalid_argument] if [ts] is empty or contains [Index]. *)

val least_upper_float : t -> t
(** [least_upper_float t] is [t] if [t] is floating-point, otherwise the
    promoted float type with {!default_float}. *)

val can_lossless_cast : t -> t -> bool
(** [can_lossless_cast src dst] is [true] iff casting [src] to [dst] preserves
    all representable values. *)

val sum_acc_dtype : t -> t
(** [sum_acc_dtype t] is the default accumulator dtype for sum-like reductions.

    Raises [Invalid_argument] if [t] is [Index]. *)

val finfo : t -> int * int
(** [finfo t] is [(exponent_bits, mantissa_bits)] for floating-point dtypes.

    Raises [Invalid_argument] if [t] is not floating-point. *)

(** {1:comparison Comparison} *)

val equal : t -> t -> bool
(** [equal a b] is [true] iff [a] and [b] have the same scalar and count. *)

val compare : t -> t -> int
(** [compare a b] orders by scalar promotion tuple and then by [count]. *)

(** {1:fmt Pretty Printing} *)

val scalar_to_string : scalar -> string
(** [scalar_to_string s] is the short name of scalar type [s] (e.g., ["f32"],
    ["i64"], ["bool"]). *)

val to_string : t -> string
(** [to_string t] is the short name of [t], including vector width if
    [count > 1] (e.g., ["f32"], ["f32×4"]). *)

val pp_scalar : Format.formatter -> scalar -> unit
(** [pp_scalar fmt s] prints [scalar_to_string s] to [fmt]. *)

val pp : Format.formatter -> t -> unit
(** [pp fmt t] prints [to_string t] to [fmt]. *)

val addr_space_to_string : addr_space -> string
(** [addr_space_to_string a] is ["global"], ["local"], or ["reg"]. *)

val pp_addr_space : Format.formatter -> addr_space -> unit
(** [pp_addr_space fmt a] formats [a] on [fmt]. *)

(** {1:ptr Pointer Operations} *)

(** Pointer type operations. *)
module Ptr : sig
  val create :
    t ->
    ?size:int ->
    ?addrspace:addr_space ->
    ?v:int ->
    ?image:int * int ->
    unit ->
    ptr
  (** [create t ?size ?addrspace ?v ?image ()] creates a pointer to [t].

      [size] defaults to [-1] (unbounded). [addrspace] defaults to [Global]. [v]
      defaults to [1].

      Raises [Invalid_argument] if [v < 1] or [image] is provided with
      non-[Global] addrspace. *)

  val vec : ptr -> int -> ptr
  (** [vec p n] is [p] with pointer vector width [n].

      Raises [Invalid_argument] if [n < 1]. *)

  val scalar : ptr -> ptr
  (** [scalar p] is [p] with pointer vector width [1]. *)

  val vcount : ptr -> int
  (** [vcount p] is the pointer vector width. *)

  val equal : ptr -> ptr -> bool
  (** [equal a b] is [true] iff all fields of [a] and [b] are equal (base,
      addrspace, v, size, image).

      {b Note.} IR validators may use partial field comparisons (e.g., Index
      validation ignores [size], Ptrcat validation ignores [v]). Those are
      intentionally different from [equal]. *)

  val compare : ptr -> ptr -> int
  (** [compare a b] total ordering over pointer types. *)

  val to_string : ptr -> string
  (** [to_string p] is a human-readable string for pointer type [p]. *)

  val pp : Format.formatter -> ptr -> unit
  (** [pp fmt p] formats [p] on [fmt]. *)
end

(** {1:cnames C Type Names} *)

val scalar_cname : scalar -> string
(** [scalar_cname s] is the C-language type name for [s]. Used by codegen
    renderers as the fallback type name (e.g., ["int"] for [Int32],
    ["signed char"] for [Int8], ["half"] for [Float16], ["__bf16"] for
    [Bfloat16]). *)

(** {1:fp_conv FP Conversion}

    Precision-truncation utilities for constant folding. *)

val float_to_fp16 : float -> float
(** [float_to_fp16 x] rounds [x] to IEEE 754 binary16 (half) precision.
    Overflows to infinity, underflows to zero/denormal, preserves NaN/Inf. *)

val float_to_bf16 : float -> float
(** [float_to_bf16 x] rounds [x] to bfloat16 precision. Non-finite values pass
    through unchanged. *)

val float_to_fp8 : scalar -> float -> int
(** [float_to_fp8 s x] converts [x] to an fp8 byte. [s] must be [Fp8e4m3] or
    [Fp8e5m2].

    Raises [Invalid_argument] if [s] is not an fp8 type. *)

val fp8_to_float : scalar -> int -> float
(** [fp8_to_float s x] converts fp8 byte [x] to a float. [s] must be [Fp8e4m3]
    or [Fp8e5m2].

    Raises [Invalid_argument] if [s] is not an fp8 type. *)

val truncate_float : t -> float -> float
(** [truncate_float t x] truncates [x] to the precision of floating-point dtype
    [t]. For float64 this is identity; for float32 it round-trips through
    [Int32.bits_of_float]; for narrower types it uses the appropriate
    conversion.

    Raises [Invalid_argument] if [t] is not floating-point. *)

val truncate_int : t -> int -> int
(** [truncate_int t x] truncates integer [x] to the range of integer dtype [t]
    using modular arithmetic with sign extension for signed types. Handles
    [Index] in the signed integer branch.

    Raises [Invalid_argument] if [t] is not integer, index, or bool. *)
