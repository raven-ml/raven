(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Data types for tensor computations.

    This module defines three levels of data type:

    - {!t} — value dtypes (a {!scalar} identity with a vector width).
    - {!ptr} — pointer dtypes (a value dtype plus address space, buffer
      element count, and pointer vector width).
    - {!any} — the union of both, for IR nodes whose dtype can be either
      (e.g., [Index]).

    {b Promotion.} Scalar types are organized in a promotion lattice based
    on {{:https://jax.readthedocs.io/en/latest/jep/9407-type-promotion.html}
    JAX JEP-9407}. Promotion is total: any pair of numeric types has a
    common supertype, at the cost of some lossy edges (e.g., [Uint64]
    promotes through fp8 to reach floats). See {!least_upper_dtype}.

    {b Vectorization.} A value dtype represents [count] lanes of a
    [scalar]. Most operations work on the scalar component and ignore
    [count]. Use {!vec} and {!scalar_of} to convert.

    {b Index.} The {!Index} scalar is a symbolic type for loop counters and
    address arithmetic. It does not participate in promotion and is lowered
    to [int32] or [int64] by backends. *)

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

type t
(** Value dtype: a {!scalar} identity with a vector width ([count]).
    Invariant: [count >= 0]; [count = 0] only when [scalar = Index]. *)

type ptr
(** Pointer dtype: a value dtype ({!base}) plus address space, buffer
    element count, and pointer vector width.
    Invariant: pointer vector width [>= 1]. *)

type any = T of t | P of ptr
(** A dtype that is either a value or a pointer. Used in IR nodes
    whose dtype can be either (e.g., [Index] nodes that may or may
    not carry pointer semantics). *)

(** {2:val_accessors Value dtype accessors} *)

val scalar : t -> scalar
(** [scalar dt] is the element type of [dt]. *)

val count : t -> int
(** [count dt] is the vector width of [dt]. [1] for scalar types. *)

(** {2:ptr_accessors Pointer dtype accessors} *)

val base : ptr -> t
(** [base p] is the pointed-to value dtype. *)

val addrspace : ptr -> addr_space
(** [addrspace p] is the memory address space of [p]. *)

val ptr_size : ptr -> int
(** [ptr_size p] is the element count of [p], or [-1] for unbounded. *)

val ptr_v : ptr -> int
(** [ptr_v p] is the pointer vector width of [p]. *)

(** {2:any_accessors Any dtype accessors} *)

val any_scalar : any -> scalar
(** [any_scalar a] is the scalar identity. [any_scalar (T dt)] is
    [scalar dt]. [any_scalar (P p)] is [scalar (base p)]. *)

val any_count : any -> int
(** [any_count a] is the vector width. [any_count (T dt)] is
    [count dt]. [any_count (P p)] is [count (base p)]. *)

val vcount : any -> int
(** [vcount a] is the vector count matching tinygrad's [dtype.vcount].
    [vcount (T dt)] is [count dt]. [vcount (P p)] is [ptr_v p]. *)

val any_scalar_of : any -> any
(** [any_scalar_of a] is the scalar version of [a].
    Matches tinygrad's [dtype.scalar()].
    [any_scalar_of (T dt)] is [T (scalar_of dt)].
    [any_scalar_of (P p)] is [P (scalar_ptr_of p)]. *)

val any_to_val : any -> t
(** [any_to_val (T dt)] is [dt]. [any_to_val (P p)] is [base p]. *)

val any_is_ptr : any -> bool
(** [any_is_ptr a] is [true] iff [a] is [P _]. *)

(** {2:coercions Coercions} *)

val to_any : t -> any
(** [to_any dt] is [T dt]. *)

val ptr_to_any : ptr -> any
(** [ptr_to_any p] is [P p]. *)

val ptr_base_to_any : ptr -> any
(** [ptr_base_to_any p] is [T (base p)]. *)

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

val of_scalar : scalar -> t
(** [of_scalar s] is the scalar dtype for [s] with [count = 1]. *)

val default_float : t
(** [default_float] is {!float32}. Used by {!least_upper_float} and
    {!sum_acc_dtype} when promoting non-float types to a floating-point domain.
*)

val default_int : t
(** [default_int] is {!int32}. Used by {!sum_acc_dtype} to widen narrow integer
    accumulators. *)

(** {2:ptr_constructors Pointer dtype constructors} *)

val ptr_of : t -> addrspace:addr_space -> size:int -> ptr
(** [ptr_of base ~addrspace ~size] is a pointer to [base] in [addrspace]
    with [size] elements. Pointer vector width defaults to [1]. *)

val ptr_of_v : t -> addrspace:addr_space -> size:int -> v:int -> ptr
(** [ptr_of_v base ~addrspace ~size ~v] is like {!ptr_of} with explicit
    pointer vector width [v].

    Raises [Invalid_argument] if [v < 1]. *)

(** {2:ptr_with Pointer dtype transformers} *)

val ptr_with_v : ptr -> int -> ptr
(** [ptr_with_v p n] is [p] with pointer vector width [n].

    Raises [Invalid_argument] if [n < 1]. *)

val ptr_with_scalar : ptr -> ptr
(** [ptr_with_scalar p] is [p] with pointer vector width [1].

    See also {!ptr_with_v}. *)

val ptr_with_size : ptr -> int -> ptr
(** [ptr_with_size p n] is [p] with element count [n]. *)

val ptr_with_base : ptr -> t -> ptr
(** [ptr_with_base p dt] is [p] with pointed-to value dtype [dt]. *)

(** {1:predicates Predicates} *)

val is_float : t -> bool
(** [is_float dt] is [true] iff [scalar dt] is a floating-point type
    ({!Float16}, {!Bfloat16}, {!Float32}, {!Float64}, {!Fp8e4m3}, or
    {!Fp8e5m2}). *)

val is_int : t -> bool
(** [is_int dt] is [true] iff [scalar dt] is a signed or unsigned integer
    type, including {!Index}. *)

val is_unsigned : t -> bool
(** [is_unsigned dt] is [true] iff [scalar dt] is one of {!Uint8},
    {!Uint16}, {!Uint32}, or {!Uint64}. *)

val is_bool : t -> bool
(** [is_bool dt] is [true] iff [scalar dt] is {!Bool}. *)

val is_fp8 : t -> bool
(** [is_fp8 dt] is [true] iff [scalar dt] is {!Fp8e4m3} or {!Fp8e5m2}. *)

(** {1:properties Properties} *)

val bitsize : t -> int
(** [bitsize dt] is the total size of [dt] in bits, i.e., the scalar bit
    width multiplied by [count dt]. {!Void} has bitsize [0]. {!Index} has
    a sentinel bitsize of [800]. *)

val itemsize : t -> int
(** [itemsize dt] is [{!bitsize} dt] rounded up to the nearest byte. *)

val priority : t -> int
(** [priority dt] is the promotion priority of [scalar dt]. Higher priority
    types absorb lower ones in {!least_upper_dtype}. Ranges from [-1]
    ({!Void}, {!Index}) through [0] ({!Bool}) to [14] ({!Float64}). *)

(** {1:operations Operations} *)

val vec : t -> int -> t
(** [vec dt n] is a vector type with [n] lanes of [scalar dt].

    If [n = 1] or [scalar dt] is {!Void}, returns [dt] unchanged.
    [vec index 0] is permitted for empty shape vectors.

    Raises [Invalid_argument] if [count dt <> 1] (already vectorized),
    [n < 0], or [n = 0] on a non-{!Index} dtype.

    See also {!scalar_of}. *)

val scalar_of : t -> t
(** [scalar_of dt] is [dt] with [count = 1].

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
(** [min dt] is the smallest value representable by [scalar dt].

    Raises [Invalid_argument] if [scalar dt] is {!Void}.

    See also {!max}. *)

val max : t -> bound
(** [max dt] is the largest value representable by [scalar dt].

    Raises [Invalid_argument] if [scalar dt] is {!Void}.

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
    [scalar src] is exactly representable by [scalar dst]. {!Bool} casts
    losslessly to any type. {!Index} accepts all integer types as lossless
    sources.

    This checks exact representability, not promotion: for example,
    [can_lossless_cast int32 float32] is [false] (float32 cannot represent
    all 32-bit integers).

    See also {!least_upper_dtype}. *)

val sum_acc_dtype : t -> t
(** [sum_acc_dtype dt] is the accumulator dtype for sum-like reductions
    over [dt]. The result always has [count = 1].

    Widening rules:
    - Unsigned integers promote to at least {!uint32}.
    - Signed integers and booleans promote to at least {!int32}.
    - Floats promote to at least {!float32}.

    Raises [Invalid_argument] if [scalar dt] is {!Index}. *)

val finfo : t -> int * int
(** [finfo t] is [(exponent_bits, mantissa_bits)] for the floating-point dtype
    [t]. For example, [finfo float32] is [(8, 23)] and [finfo float16] is
    [(5, 10)].

    Raises [Invalid_argument] if [t] is not floating-point.

    See also {!is_float}. *)

(** {1:comparison Comparison} *)

val equal : t -> t -> bool
(** [equal a b] is [true] iff [a] and [b] have the same scalar and count. *)

val compare : t -> t -> int
(** [compare a b] is a total order over value dtypes. Orders first by scalar
    promotion priority and bit width, then by count. *)

val ptr_equal : ptr -> ptr -> bool
(** [ptr_equal a b] is [true] iff all fields of [a] and [b] are
    structurally equal (base, addrspace, v, size).

    {b Note.} IR validators may use partial field comparisons (e.g.,
    index validation ignores size). Those are intentionally different
    from this structural equality. *)

val ptr_compare : ptr -> ptr -> int
(** [ptr_compare a b] is a total order over pointer types. *)

val any_equal : any -> any -> bool
(** [any_equal a b] is [true] iff [a] and [b] carry the same dtype. *)

val any_compare : any -> any -> int
(** [any_compare a b] is a total order over {!any} values. *)

(** {1:fmt Formatting} *)

val scalar_to_string : scalar -> string
(** [scalar_to_string s] is the short name of [s] (e.g., ["f32"], ["i64"],
    ["bool"], ["void"], ["index"]). *)

val to_string : t -> string
(** [to_string dt] is the short name of [dt]. For scalar types this is
    [scalar_to_string (scalar dt)]. For vector types it appends the count
    (e.g., ["f32×4"]). *)

val ptr_to_string : ptr -> string
(** [ptr_to_string p] is a human-readable representation of [p] (e.g.,
    ["f32* \[global\]"]). *)

val pp_scalar : Format.formatter -> scalar -> unit
(** [pp_scalar] formats a {!scalar} using {!scalar_to_string}. *)

val pp : Format.formatter -> t -> unit
(** [pp] formats a value dtype using {!to_string}. *)

val pp_ptr : Format.formatter -> ptr -> unit
(** [pp_ptr] formats a pointer dtype using {!ptr_to_string}. *)

val pp_any : Format.formatter -> any -> unit
(** [pp_any] formats an {!any} dtype. *)

val addr_space_to_string : addr_space -> string
(** [addr_space_to_string a] is ["global"], ["local"], or ["reg"]. *)

val pp_addr_space : Format.formatter -> addr_space -> unit
(** [pp_addr_space] formats an address space. *)

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
(** [truncate_float dt x] truncates [x] to the precision of floating-point
    dtype [dt]. For {!Float64} this is the identity. For {!Float32} it
    round-trips through [Int32.bits_of_float]. For narrower types it uses
    {!float_to_fp16}, {!float_to_bf16}, or the fp8 conversion pair.

    Raises [Invalid_argument] if [dt] is not floating-point.

    See also {!truncate_int}. *)

val truncate_int : t -> int -> int
(** [truncate_int dt x] truncates integer [x] to the range of integer dtype
    [dt]. Unsigned types use bitwise masking. Signed types and {!Index} use
    modular arithmetic with sign extension. {!Bool} maps nonzero to [1] and
    zero to [0].

    Raises [Invalid_argument] if [dt] is not an integer, index, or bool type.

    See also {!truncate_float}. *)
