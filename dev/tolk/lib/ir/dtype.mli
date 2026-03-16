(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Data types for tensor computations.

    This module defines the {e scalar} types (booleans, integers, floats),
    {e vector} types (a scalar repeated [count] times), and {e pointer} types
    (references into GPU memory spaces) that form the type system for tolk's IR.

    {b Promotion.} Scalar types are organized in a promotion lattice based on
    {{:https://jax.readthedocs.io/en/latest/jep/9407-type-promotion.html} JAX
     JEP-9407}. Promotion is total: any pair of numeric types has a common
    supertype, at the cost of some lossy edges (e.g., [Uint64] promotes through
    fp8 to reach floats). See {!least_upper_dtype}.

    {b Vectorization.} A dtype [{scalar; count}] represents [count] lanes of
    [scalar]. Most operations work on the scalar component and ignore [count].
    Use {!vec} and {!scalar_of} to move between scalar and vector forms.

    {b Index.} The {!Index} scalar is a symbolic type for loop counters and
    address arithmetic. It does not participate in promotion and is lowered to
    [int32] or [int64] by backends. *)

(** {1:scalars Scalar types} *)

(** Scalar data type identity. Ordered by promotion priority from {!Bool}
    (lowest, priority [0]) to {!Float64} (highest, priority [14]). {!Void} and
    {!Index} have priority [-1]. *)
type scalar =
  | Void  (** Absence of a value. Has zero {!bitsize}. *)
  | Bool  (** Boolean. 1 bit. *)
  | Int8  (** Signed 8-bit integer. *)
  | Int16  (** Signed 16-bit integer. *)
  | Int32  (** Signed 32-bit integer. *)
  | Int64  (** Signed 64-bit integer. *)
  | Uint8  (** Unsigned 8-bit integer. *)
  | Uint16  (** Unsigned 16-bit integer. *)
  | Uint32  (** Unsigned 32-bit integer. *)
  | Uint64  (** Unsigned 64-bit integer. *)
  | Float16  (** IEEE 754 binary16 (half precision). *)
  | Bfloat16  (** Brain floating-point 16: 8-bit exponent, 7-bit mantissa. *)
  | Float32  (** IEEE 754 binary32 (single precision). *)
  | Float64  (** IEEE 754 binary64 (double precision). *)
  | Fp8e4m3  (** 8-bit float: 4-bit exponent, 3-bit mantissa. *)
  | Fp8e5m2  (** 8-bit float: 5-bit exponent, 2-bit mantissa. *)
  | Index
      (** Symbolic index type for loop counters and address arithmetic. Uses 800
          bits as a non-machine sentinel. Does not participate in dtype
          promotion. *)

(** {1:addr_spaces Address spaces} *)

(** GPU memory address space. *)
type addr_space =
  | Global  (** Global device memory. *)
  | Local  (** Shared/local memory (workgroup scope). *)
  | Reg  (** Register storage. *)

(** {1:types Data types} *)

type t = {
  scalar : scalar;  (** The element type. *)
  count : int;  (** Vector width. [1] for scalar types. *)
}
(** Data type with optional vectorization. A scalar dtype has [count = 1]. *)

type ptr = {
  base : t;  (** Pointed-to element type. *)
  addrspace : addr_space;  (** Memory address space. *)
  v : int;  (** Pointer vector width (distinct from [base.count]). *)
  size : int;  (** Element count, or [-1] for unbounded. *)
}
(** Pointer into a typed buffer in a GPU memory space. *)

(** {1:constructors Constructors}

    Scalar dtype constants. All have [count = 1]. *)

val void : t
(** [void] is [{scalar = Void; count = 1}]. *)

val bool : t
(** [bool] is [{scalar = Bool; count = 1}]. *)

val int8 : t
(** [int8] is [{scalar = Int8; count = 1}]. *)

val int16 : t
(** [int16] is [{scalar = Int16; count = 1}]. *)

val int32 : t
(** [int32] is [{scalar = Int32; count = 1}]. *)

val int64 : t
(** [int64] is [{scalar = Int64; count = 1}]. *)

val uint8 : t
(** [uint8] is [{scalar = Uint8; count = 1}]. *)

val uint16 : t
(** [uint16] is [{scalar = Uint16; count = 1}]. *)

val uint32 : t
(** [uint32] is [{scalar = Uint32; count = 1}]. *)

val uint64 : t
(** [uint64] is [{scalar = Uint64; count = 1}]. *)

val float16 : t
(** [float16] is [{scalar = Float16; count = 1}]. *)

val bfloat16 : t
(** [bfloat16] is [{scalar = Bfloat16; count = 1}]. *)

val float32 : t
(** [float32] is [{scalar = Float32; count = 1}]. *)

val float64 : t
(** [float64] is [{scalar = Float64; count = 1}]. *)

val fp8e4m3 : t
(** [fp8e4m3] is [{scalar = Fp8e4m3; count = 1}]. *)

val fp8e5m2 : t
(** [fp8e5m2] is [{scalar = Fp8e5m2; count = 1}]. *)

val index : t
(** [index] is [{scalar = Index; count = 1}]. *)

val default_float : t
(** [default_float] is {!float32}. Used by {!least_upper_float} and
    {!sum_acc_dtype} when promoting non-float types to a floating-point domain.
*)

val default_int : t
(** [default_int] is {!int32}. Used by {!sum_acc_dtype} to widen narrow integer
    accumulators. *)

(** {1:predicates Predicates} *)

val is_float : t -> bool
(** [is_float t] is [true] iff [t.scalar] is a floating-point type ({!Float16},
    {!Bfloat16}, {!Float32}, {!Float64}, {!Fp8e4m3}, or {!Fp8e5m2}). Ignores
    [t.count]. *)

val is_int : t -> bool
(** [is_int t] is [true] iff [t.scalar] is a signed or unsigned integer type,
    including {!Index}. Ignores [t.count]. *)

val is_unsigned : t -> bool
(** [is_unsigned t] is [true] iff [t.scalar] is one of {!Uint8}, {!Uint16},
    {!Uint32}, or {!Uint64}. *)

val is_bool : t -> bool
(** [is_bool t] is [true] iff [t.scalar] is {!Bool}. Includes vectorized bools
    (e.g., [count > 1]). *)

val is_fp8 : t -> bool
(** [is_fp8 t] is [true] iff [t.scalar] is {!Fp8e4m3} or {!Fp8e5m2}. *)

(** {1:properties Properties} *)

val bitsize : t -> int
(** [bitsize t] is the total size of [t] in bits, i.e., the scalar bit width
    multiplied by [count]. {!Void} has bitsize [0]. {!Index} has a sentinel
    bitsize of [800]. *)

val itemsize : t -> int
(** [itemsize t] is [{!bitsize} t] rounded up to the nearest byte (i.e.,
    [(bitsize t + 7) / 8]). *)

val priority : t -> int
(** [priority t] is the promotion priority of [t.scalar]. Higher priority types
    absorb lower ones in {!least_upper_dtype}. Ranges from [-1] ({!Void},
    {!Index}) through [0] ({!Bool}) to [14] ({!Float64}). *)

(** {1:operations Operations} *)

val vec : t -> int -> t
(** [vec t n] is a vector type with [n] lanes of [t.scalar].

    If [n = 1] or [t.scalar] is {!Void}, the result is [t] unchanged.
    [vec index 0] is permitted to represent empty shape vectors (scalar tensors
    with zero dimensions).

    Raises [Invalid_argument] if [t.count <> 1] (already vectorized), [n < 0],
    or [n = 0] on a non-{!Index} dtype.

    See also {!scalar_of}. *)

val scalar_of : t -> t
(** [scalar_of t] is [t] with [count = 1].

    See also {!vec}. *)

type bound =
  [ `Bool of bool | `SInt of int64 | `UInt of int64 | `Float of float ]
(** Numeric bounds for dtypes. Returned by {!min} and {!max}.

    - [`Bool b] for boolean bounds.
    - [`SInt n] for signed integer bounds (including {!Index}, which
      approximates with [Int64] bounds).
    - [`UInt n] for unsigned integer bounds. Values are raw 64-bit unsigned bit
      patterns in [int64] (e.g., uint64 max is [`UInt Int64.minus_one]).
    - [`Float f] for floating-point bounds ([-infinity] and [infinity]). *)

val min : t -> bound
(** [min t] is the smallest value representable by [t.scalar].

    Raises [Invalid_argument] if [t.scalar] is {!Void}.

    See also {!max}. *)

val max : t -> bound
(** [max t] is the largest value representable by [t.scalar].

    Raises [Invalid_argument] if [t.scalar] is {!Void}.

    See also {!min}. *)

(** {1:promotion Type promotion} *)

val least_upper_dtype : t list -> t
(** [least_upper_dtype ts] is the least upper bound of [ts] in the promotion
    lattice. The result always has [count = 1].

    Promotion is total for numeric types: any pair has a common supertype. Some
    edges are lossy (e.g., [Int64] to [Uint64] loses negative values, [Uint64]
    to fp8 loses most precision).

    Raises [Invalid_argument] if [ts] is empty or contains {!Index}.

    See also {!least_upper_float} and {!can_lossless_cast}. *)

val least_upper_float : t -> t
(** [least_upper_float t] is [scalar_of t] if [t] is floating-point, or
    [least_upper_dtype [scalar_of t; float32]] otherwise.

    See also {!least_upper_dtype}. *)

val can_lossless_cast : t -> t -> bool
(** [can_lossless_cast src dst] is [true] iff every value representable by
    [src.scalar] is exactly representable by [dst.scalar]. {!Bool} casts
    losslessly to any type. {!Index} accepts all integer types as lossless
    sources.

    This checks exact representability, not promotion: for example,
    [can_lossless_cast int32 float32] is [false] (float32 cannot represent all
    32-bit integers).

    See also {!least_upper_dtype}. *)

val sum_acc_dtype : t -> t
(** [sum_acc_dtype t] is the accumulator dtype for sum-like reductions over [t].
    The result always has [count = 1].

    Widening rules:
    - Unsigned integers promote to at least {!uint32}.
    - Signed integers and booleans promote to at least {!int32}.
    - Floats promote to at least {!float32}.

    Raises [Invalid_argument] if [t.scalar] is {!Index}. *)

val finfo : t -> int * int
(** [finfo t] is [(exponent_bits, mantissa_bits)] for the floating-point dtype
    [t]. For example, [finfo float32] is [(8, 23)] and [finfo float16] is
    [(5, 10)].

    Raises [Invalid_argument] if [t] is not floating-point.

    See also {!is_float}. *)

(** {1:comparison Comparison} *)

val equal : t -> t -> bool
(** [equal a b] is [true] iff [a] and [b] have the same [scalar] and [count]. *)

val compare : t -> t -> int
(** [compare a b] is a total order over dtypes. Orders first by scalar promotion
    priority and bit width, then by [count]. *)

(** {1:fmt Formatting} *)

val scalar_to_string : scalar -> string
(** [scalar_to_string s] is the short name of [s] (e.g., ["f32"], ["i64"],
    ["bool"], ["void"], ["index"]). *)

val to_string : t -> string
(** [to_string t] is the short name of [t]. For scalar types this is
    [scalar_to_string t.scalar]. For vector types it appends the count with a
    multiplication sign (e.g., ["f32×4"]). *)

val pp_scalar : Format.formatter -> scalar -> unit
(** [pp_scalar] formats a {!scalar} using {!scalar_to_string}. *)

val pp : Format.formatter -> t -> unit
(** [pp] formats a dtype using {!to_string}. *)

val addr_space_to_string : addr_space -> string
(** [addr_space_to_string a] is ["global"], ["local"], or ["reg"]. *)

val pp_addr_space : Format.formatter -> addr_space -> unit
(** [pp_addr_space] formats an {!addr_space} using {!addr_space_to_string}. *)

(** {1:ptr Pointer operations} *)

(** Operations on pointer types. *)
module Ptr : sig
  val create : t -> ?size:int -> ?addrspace:addr_space -> ?v:int -> unit -> ptr
  (** [create base ?size ?addrspace ?v ()] is a pointer to [base] with:
      - [size] element count. Defaults to [-1] (unbounded).
      - [addrspace] memory space. Defaults to {!Global}.
      - [v] pointer vector width. Defaults to [1].

      Raises [Invalid_argument] if [v < 1]. *)

  val vec : ptr -> int -> ptr
  (** [vec p n] is [p] with pointer vector width [n].

      Raises [Invalid_argument] if [n < 1]. *)

  val scalar : ptr -> ptr
  (** [scalar p] is [p] with pointer vector width [1].

      See also {!vec}. *)

  val vcount : ptr -> int
  (** [vcount p] is the pointer vector width of [p]. *)

  val equal : ptr -> ptr -> bool
  (** [equal a b] is [true] iff all fields of [a] and [b] are structurally equal
      ([base], [addrspace], [v], [size]).

      {b Note.} IR validators may use partial field comparisons (e.g., index
      validation ignores [size], pointer concatenation validation ignores [v]).
      Those are intentionally different from this structural equality. *)

  val compare : ptr -> ptr -> int
  (** [compare a b] is a total order over pointer types. *)

  val to_string : ptr -> string
  (** [to_string p] is a human-readable representation of [p] (e.g.,
      ["f32* [global]"]). *)

  val pp : Format.formatter -> ptr -> unit
  (** [pp] formats a pointer type using {!to_string}. *)
end

(** {1:cnames C type names} *)

val scalar_cname : scalar -> string
(** [scalar_cname s] is the C-language type name for [s], used by codegen
    renderers as a fallback when no device-specific type map override exists.
    For example, ["int"] for {!Int32}, ["signed char"] for {!Int8}, ["half"] for
    {!Float16}, ["__bf16"] for {!Bfloat16}. *)

(** {1:fp_conv Floating-point conversion}

    Precision-truncation utilities for constant folding. These functions round
    or convert between floating-point precisions using round-to-nearest-even
    semantics. *)

val float_to_fp16 : float -> float
(** [float_to_fp16 x] rounds [x] to IEEE 754 binary16 (half) precision using
    round-to-nearest-even. The result is a [float] holding the exact
    half-representable value. Overflows to infinity, underflows to zero or
    denormal, preserves NaN and infinity. *)

val float_to_bf16 : float -> float
(** [float_to_bf16 x] rounds [x] to bfloat16 precision using
    round-to-nearest-even. Non-finite values pass through unchanged. *)

val float_to_fp8 : scalar -> float -> int
(** [float_to_fp8 s x] converts [x] to an fp8 byte value.

    Raises [Invalid_argument] if [s] is not {!Fp8e4m3} or {!Fp8e5m2}.

    See also {!fp8_to_float}. *)

val fp8_to_float : scalar -> int -> float
(** [fp8_to_float s byte] converts fp8 byte value [byte] to a [float].

    Raises [Invalid_argument] if [s] is not {!Fp8e4m3} or {!Fp8e5m2}.

    See also {!float_to_fp8}. *)

val truncate_float : t -> float -> float
(** [truncate_float t x] truncates [x] to the precision of floating-point dtype
    [t]. For {!Float64} this is the identity. For {!Float32} it round-trips
    through [Int32.bits_of_float]. For narrower types it uses {!float_to_fp16},
    {!float_to_bf16}, or the fp8 conversion pair.

    Raises [Invalid_argument] if [t] is not floating-point.

    See also {!truncate_int}. *)

val truncate_int : t -> int -> int
(** [truncate_int t x] truncates integer [x] to the range of integer dtype [t].
    Unsigned types use bitwise masking. Signed types and {!Index} use modular
    arithmetic with sign extension. {!Bool} maps nonzero to [1] and zero to [0].

    Raises [Invalid_argument] if [t] is not an integer, index, or bool type.

    See also {!truncate_float}. *)
