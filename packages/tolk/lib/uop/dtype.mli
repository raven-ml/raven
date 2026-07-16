(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Scalar data types for tensor computations.

    A dtype names a single scalar element type: its promotion priority, bit
    width, and value domain. Dtypes carry no vector width and no pointer or
    buffer metadata — a value's lane count comes from its shape at the IR
    level, and address spaces are tracked separately (see {!addr_space}).

    {b Promotion.} Numeric dtypes are organized in a promotion lattice: any
    pair of numeric dtypes has a least common supertype. The lattice is total
    at the cost of some lossy edges (for example [Uint64] promotes through
    {!Weakfloat} to reach the floats). See {!least_upper_dtype}.

    {b Weak dtypes.} {!Weakint} and {!Weakfloat} are abstract literal types
    with priority [0] and [9] respectively and a sentinel bit width of [800].
    They stand for untyped integer and floating-point literals and promote to
    any concrete type of their kind ([Weakint + Int8 = Int8],
    [Weakfloat + Float16 = Float16]). {!is_weak} tests for either.

    {b Index.} {!Index} is a dedicated integer dtype for shapes and loop
    indices. It has priority [0] and the same sentinel width [800], is
    classified as an integer, but is deliberately absent from the promotion
    lattice: index arithmetic never mixes dtypes, so {!least_upper_dtype} of
    {!Index} with anything else raises. *)

(** {1:types Dtype} *)

(** Scalar data type.

    Promotion priority ranges from [-1] ({!Void}) to [15] ({!Float64}).
    {!Weakint}, {!Index}, and {!Bool} share priority [0]. See {!priority} and
    {!least_upper_dtype} for the full ordering. *)
type t =
  | Void  (** Absence of a value. Zero bit width; has no numeric bounds. *)
  | Index
      (** Integer dtype for shapes and loop indices. Priority [0], sentinel
          width [800]. Classified as an integer but never participates in
          promotion. *)
  | Weakint
      (** Abstract integer literal. Priority [0], sentinel width [800].
          Promotes to any concrete integer type. *)
  | Bool  (** Boolean. 1 bit. *)
  | Int8  (** Signed 8-bit integer. *)
  | Int16  (** Signed 16-bit integer. *)
  | Int32  (** Signed 32-bit integer. *)
  | Int64  (** Signed 64-bit integer. *)
  | Uint8  (** Unsigned 8-bit integer. *)
  | Uint16  (** Unsigned 16-bit integer. *)
  | Uint32  (** Unsigned 32-bit integer. *)
  | Uint64  (** Unsigned 64-bit integer. *)
  | Uint128
      (** Private virtual 128-bit storage helper. Not classified as an integer
          or unsigned dtype. String parsing accepts only the private spelling
          ["_uint128"]. *)
  | Uint256
      (** Private virtual 256-bit storage helper. Not classified as an integer
          or unsigned dtype. String parsing accepts only the private spelling
          ["_uint256"]. *)
  | Weakfloat
      (** Abstract floating-point literal. Priority [9], sentinel width [800].
          Promotes to any concrete floating-point type. *)
  | Fp8e4m3  (** 8-bit float: 4-bit exponent, 3-bit mantissa. *)
  | Fp8e5m2  (** 8-bit float: 5-bit exponent, 2-bit mantissa. *)
  | Fp8e4m3fnuz
      (** 8-bit float, FNUz variant: 4 exp / 3 mant bits, no signed zero,
          NaN encoded as [0x80]. *)
  | Fp8e5m2fnuz  (** 8-bit float, FNUz variant: 5 exp / 2 mant bits. *)
  | Float16  (** IEEE 754 binary16 (half precision). *)
  | Bfloat16  (** Brain floating-point 16: 8-bit exponent, 7-bit mantissa. *)
  | Float32  (** IEEE 754 binary32 (single precision). *)
  | Float64  (** IEEE 754 binary64 (double precision). *)

(** {1:addr_spaces Address spaces} *)

(** GPU memory address space. *)
type addr_space =
  | Global  (** Global device memory. *)
  | Local  (** Shared/local memory (workgroup scope). *)
  | Reg  (** Register storage. *)
  | Alu
      (** Scalar ALU value space, used for symbolic variables and loaded
          values. *)

(** {1:constants Named dtypes} *)

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
val fp8e4m3fnuz : t
val fp8e5m2fnuz : t
val weakint : t
val index : t
val weakfloat : t

val default_float : t
(** [default_float] is the dtype named by the [DEFAULT_FLOAT] environment
    variable at module initialization, or {!float32} when unset. The value must
    name a floating-point dtype using a canonical name or public alias such as
    ["float32"], ["float"], ["half"], or ["default_float"]. Short display
    mnemonics such as ["f32"], ["f16"], and ["bf16"] are not accepted. Used by
    {!least_upper_float}. *)

val default_int : t
(** [default_int] is {!int32}. Used by {!sum_acc_dtype} to widen narrow integer
    accumulators. *)

(** {1:predicates Predicates} *)

val is_float : t -> bool
(** [is_float dt] is [true] iff [dt] is a floating-point dtype, including the
    fp8 dtypes and {!Weakfloat}. *)

val is_int : t -> bool
(** [is_int dt] is [true] iff [dt] is an integer dtype, including {!Weakint} and
    {!Index}. {!Uint128} and {!Uint256} are private storage helpers and are not
    classified as integers. *)

val is_unsigned : t -> bool
(** [is_unsigned dt] is [true] iff [dt] is a public unsigned integer dtype.
    {!Weakint}, {!Index}, {!Uint128}, and {!Uint256} are not classified as
    unsigned. *)

val is_bool : t -> bool
(** [is_bool dt] is [true] iff [dt] is {!Bool}. *)

val is_fp8 : t -> bool
(** [is_fp8 dt] is [true] iff [dt] is one of the four fp8 dtypes. *)

val is_weak : t -> bool
(** [is_weak dt] is [true] iff [dt] is {!Weakint} or {!Weakfloat}. {!Index} is
    not a weak dtype. *)

(** {1:properties Properties} *)

val bitsize : t -> int
(** [bitsize dt] is the size in bits. {!Void} is [0]; {!Weakint}, {!Index}, and
    {!Weakfloat} carry the sentinel width [800]. *)

val itemsize : t -> int
(** [itemsize dt] is {!bitsize} rounded up to whole bytes. *)

val priority : t -> int
(** [priority dt] is the promotion priority. Higher priorities absorb lower
    ones in {!least_upper_dtype}. Ranges from [-1] ({!Void}) through [0]
    ({!Bool}, {!Weakint}, {!Index}) to [15] ({!Float64}). *)

(** {1:promotion Promotion} *)

val least_upper_dtype : t list -> t
(** [least_upper_dtype ts] is the least upper bound of [ts] in the promotion
    lattice.

    Promotion is total over numeric types: any pair has a common supertype.
    Some edges are lossy (for example [Int64] to [Uint64] loses negative
    values; [Uint64] to fp8 loses most precision).

    Raises [Invalid_argument] if [ts] is empty, or if the inputs have no common
    supertype — which happens for any dtype outside the lattice, such as
    {!Index}, {!Void}, {!Uint128}, or {!Uint256}.

    See also {!least_upper_float} and {!can_lossless_cast}. *)

val least_upper_float : t -> t
(** [least_upper_float dt] is [dt] if [dt] is floating-point, or
    [least_upper_dtype [dt; default_float]] otherwise.

    See also {!least_upper_dtype}. *)

val can_lossless_cast : t -> t -> bool
(** [can_lossless_cast src dst] is [true] iff every value representable by
    [src] is exactly representable by [dst]. {!Bool} casts losslessly to any
    dtype; any signed or unsigned integer type casts losslessly to {!Weakint}
    and {!Index}.

    This checks exact representability, not promotion. For instance,
    [can_lossless_cast int32 float32] is [false] (float32 cannot represent
    every 32-bit integer).

    See also {!least_upper_dtype}. *)

val sum_acc_dtype : t -> t
(** [sum_acc_dtype dt] is the accumulator dtype for sum-like reductions over
    [dt].

    Widening rules:
    - Unsigned integers promote to at least {!uint32}.
    - Signed integers, {!Weakint}, and booleans promote to at least {!int32}.
    - Floats promote to at least the dtype named by the [SUM_DTYPE] environment
      variable, or {!float32} when unset. [SUM_DTYPE] is read on each call and
      must name a dtype using a canonical name or public alias such as
      ["float32"], ["double"], ["uchar"], or ["default_float"]. Short display
      mnemonics such as ["f32"], ["u8"], and ["bf16"] are not accepted.

    Raises [Invalid_argument] for dtypes outside the lattice (see
    {!least_upper_dtype}), such as {!Index}. *)

(** {1:bounds Bounds} *)

type bound =
  [ `Bool of bool | `SInt of int64 | `UInt of int64 | `Float of float ]
(** Numeric bounds for dtypes. Returned by {!min} and {!max}.

    - [`Bool b] for boolean bounds.
    - [`SInt n] for signed integer bounds, including {!Weakint} and {!Index},
      which report the [Int64] range as an approximation of their [800]-bit
      sentinel width.
    - [`UInt n] for unsigned integer bounds. Values are raw 64-bit unsigned bit
      patterns stored in an [int64] (for example {!Uint64}'s max is
      [`UInt Int64.minus_one]).
    - [`Float f] for floating-point bounds ([neg_infinity] and [infinity]). *)

val min : t -> bound
(** [min dt] is the smallest value representable by [dt]. {!Uint128} and
    {!Uint256}, which have no numeric bounds, report [`Bool false].

    Raises [Invalid_argument] if [dt] is {!Void}.

    See also {!max}. *)

val max : t -> bound
(** [max dt] is the largest value representable by [dt]. {!Uint128} and
    {!Uint256}, which have no numeric bounds, report [`Bool true].

    Raises [Invalid_argument] if [dt] is {!Void}.

    See also {!min}. *)

val finfo : t -> int * int
(** [finfo dt] is [(exponent_bits, mantissa_bits)] for the floating-point dtype
    [dt]. Mantissa bits exclude the implicit leading [1]. For example,
    [finfo float32] is [(8, 23)] and [finfo float16] is [(5, 10)].

    Raises [Invalid_argument] if [dt] has no exponent/mantissa layout — every
    non-float dtype, and the abstract {!Weakfloat}.

    See also {!is_float}. *)

(** {1:comparison Comparison} *)

val equal : t -> t -> bool
(** [equal a b] is [true] iff [a] and [b] are the same dtype. *)

val compare : t -> t -> int
(** [compare a b] is a total order over dtypes. Orders first by promotion
    priority, then bit width, then constructor. *)

(** {1:fmt Formatting} *)

val to_string : t -> string
(** [to_string dt] is the short display mnemonic of [dt] (for example ["f32"],
    ["i64"], ["bool"], ["void"], ["weakint"], ["index"]). *)

val repr : t -> string
(** [repr dt] is the qualified dtype representation, such as ["dtypes.int"] or
    ["dtypes.float"].

    Raises [Invalid_argument] if [dt] is a private wide helper ({!Uint128} or
    {!Uint256}), which have no public representation. *)

val pp : Format.formatter -> t -> unit
(** [pp] formats a dtype using {!to_string}. *)

val addr_space_to_string : addr_space -> string
(** [addr_space_to_string a] is ["global"], ["local"], ["reg"], or ["alu"]. *)

val pp_addr_space : Format.formatter -> addr_space -> unit
(** [pp_addr_space] formats an address space. *)

(** {1:fp_conv Floating-point conversion}

    Bit-exact precision-narrowing utilities used for constant folding. Rounding
    is round-to-nearest-even for all formats; conversions are implemented
    directly rather than delegated to hardware so results are reproducible
    across backends. *)

val float_to_fp16 : float -> float
(** [float_to_fp16 x] rounds [x] to IEEE 754 binary16 (half) precision using
    round-to-nearest-even. The result is a [float] holding a value exactly
    representable in half precision. Overflow produces an infinity; underflow
    produces zero or a subnormal; infinities pass through; NaNs produce a NaN
    (bit pattern not preserved). *)

val float_to_bf16 : float -> float
(** [float_to_bf16 x] rounds [x] to bfloat16 precision using
    round-to-nearest-even. Non-finite values pass through unchanged. *)

val float_to_fp8 : t -> float -> int
(** [float_to_fp8 dt x] encodes [x] as an fp8 byte value in [0..255].

    Raises [Invalid_argument] if [dt] is not an fp8 dtype.

    See also {!fp8_to_float}. *)

val fp8_to_float : t -> int -> float
(** [fp8_to_float dt byte] decodes the fp8 [byte] into a [float].

    Raises [Invalid_argument] if [dt] is not an fp8 dtype.

    See also {!float_to_fp8}. *)

val truncate_float : t -> float -> float
(** [truncate_float dt x] rounds [x] to the precision of floating-point dtype
    [dt]. {!Float64} and {!Weakfloat} are the identity. {!Float32} round-trips
    through [Int32.bits_of_float]. Narrower types delegate to {!float_to_fp16},
    {!float_to_bf16}, or the fp8 conversion pair.

    Raises [Invalid_argument] if [dt] is not floating-point.

    See also {!truncate_int}. *)

val truncate_int : t -> int -> int
(** [truncate_int dt x] reduces integer [x] into the range of integer dtype
    [dt]. Unsigned types mask to the low [bitsize dt] bits; signed types,
    {!Weakint}, and {!Index} apply the same mask and sign-extend. When the
    target width equals or exceeds [Sys.int_size] (notably for {!Int64},
    {!Uint64}, {!Weakint}, and {!Index} at their sentinel width) [x] is
    returned unchanged. {!Bool} maps [0] to [0] and any other value to [1].

    Raises [Invalid_argument] if [dt] is not an integer or bool type.

    See also {!truncate_float}. *)

(** {1:storage Storage conversion}

    Scalar storage helpers operating on one lane; callers handling vectors
    apply them lane-by-lane. *)

type storage_scalar = [ `Bool of bool | `Float of float | `Int of int64 ]
(** The type for scalar values at the storage boundary.

    [`Int n] carries raw signed or unsigned integer storage bits in an [int64].
    [`Float f] carries a host floating-point value. [`Bool b] carries a boolean
    storage value. *)

val storage_fmt_for_dtype : t -> char option
(** [storage_fmt_for_dtype dt] is the packing format character used for one
    scalar lane of [dt].

    Returns [None] for dtypes without a portable storage format: {!Void},
    {!Weakint}, {!Index}, {!Weakfloat}, {!Uint128}, and {!Uint256}.
    {!Bfloat16} stores as ['H'] and fp8 dtypes store as ['B']. *)

val to_storage_scalar : t -> storage_scalar -> storage_scalar
(** [to_storage_scalar dt x] maps [x] to the scalar value written to storage
    for [dt].

    {!Bfloat16} returns the high 16 bits of the rounded float as [`Int]. Fp8
    dtypes return the encoded byte as [`Int]. {!Float16} returns a
    precision-rounded [`Float]. Other dtypes return the corresponding boolean,
    integer, or float representation.

    Raises [Invalid_argument] if [dt] is {!Void}. *)

val from_storage_scalar : storage_scalar -> t -> storage_scalar
(** [from_storage_scalar x dt] maps one stored scalar value for [dt] back to a
    host scalar.

    {!Bfloat16} and fp8 storage integers decode to [`Float]. Other dtypes
    return the corresponding boolean, integer, or float representation.

    Raises [Invalid_argument] if [dt] is {!Void}. *)

val truncate : t -> storage_scalar -> storage_scalar
(** [truncate dt x] coerces [x] to the value domain of dtype [dt].

    Floating-point dtypes use {!truncate_float}. Integer dtypes use two's
    complement wrapping with the target width, including unsigned raw bit
    patterns. {!Bool} maps zero-like values to [`Bool false] and all other
    values to [`Bool true].

    Raises [Invalid_argument] if [dt] is {!Void}. *)
