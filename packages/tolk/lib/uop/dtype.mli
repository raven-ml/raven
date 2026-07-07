(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Data types for tensor computations.

    This module defines two levels of data type and their union:

    - {!Val.t} — value dtypes (a {!scalar} identity with a vector width).
    - {!Ptr.t} — pointer dtypes (a base value dtype plus address space,
      buffer element count, pointer vector width, and optional image metadata).
    - {!Image} — constructors and accessors for image pointer dtypes.
    - {!t} — the union of both, for IR nodes whose dtype can be either
      (e.g., [Index]).

    Val-specific operations live in {!Val}, pointer-specific operations
    in {!Ptr}, and dispatching operations at the top level. Kernel view
    fields use the most precise type: {!Val.t} for value-only nodes,
    {!Ptr.t} for pointer-only nodes, and {!t} where either is possible.

    {b Promotion.} Scalar types are organized in a promotion lattice
    based on
    {{:https://jax.readthedocs.io/en/latest/jep/9407-type-promotion.html}
    JAX JEP-9407}. Promotion is total over numeric types: any pair has
    a common supertype, at the cost of some lossy edges (e.g., [Uint64]
    promotes through fp8 to reach floats). See {!Val.least_upper_dtype}.

    {b Vectorization.} A value dtype represents [count] lanes of a
    [scalar]. Most predicates and properties work on the scalar
    component. Use {!Val.vec} and {!Val.scalarize} to convert.

    {b Weakint.} The {!Weakint} scalar is a weakly-typed integer used
    for loop counters, shape vectors, and untyped integer literals. It
    has priority [0] and promotes to any concrete integer type
    ([Weakint + Int8 = Int8]). Carries a sentinel bitsize of [800] so
    it cannot be confused with a machine width; backends lower it to
    [int32] or [int64]. *)

(** {1:scalars Scalar types} *)

(** Scalar data type identity.

    Promotion priority ranges from [-1] ({!Void}) to [14] ({!Float64}).
    {!Bool} and {!Weakint} share priority [0]. See {!priority} and
    {!Val.least_upper_dtype} for the full ordering. *)
type scalar =
  | Void  (** Absence of a value. Has zero bitsize. *)
  | Weakint
      (** Weakly-typed integer. Priority [0]. Used for loop counters,
          shape vectors, and untyped integer literals. Promotes to any
          concrete integer type. Carries a sentinel bitsize of [800]. *)
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
      (** Non-portable virtual 128-bit storage helper. Tinygrad keeps this as a
          private dtype; it is not classified as an integer or unsigned dtype.
          Environment dtype parsing accepts only tinygrad's private spelling
          ["_uint128"]. *)
  | Uint256
      (** Non-portable virtual 256-bit storage helper. Tinygrad keeps this as a
          private dtype; it is not classified as an integer or unsigned dtype.
          Environment dtype parsing accepts only tinygrad's private spelling
          ["_uint256"]. *)
  | Float16  (** IEEE 754 binary16 (half precision). *)
  | Bfloat16  (** Brain floating-point 16: 8-bit exponent, 7-bit mantissa. *)
  | Float32  (** IEEE 754 binary32 (single precision). *)
  | Float64  (** IEEE 754 binary64 (double precision). *)
  | Fp8e4m3  (** 8-bit float: 4-bit exponent, 3-bit mantissa. *)
  | Fp8e5m2  (** 8-bit float: 5-bit exponent, 2-bit mantissa. *)
  | Fp8e4m3fnuz
      (** 8-bit float, FNUz variant: 4 exp / 3 mant bits, no signed zero,
          NaN encoded as 0x80. *)
  | Fp8e5m2fnuz  (** 8-bit float, FNUz variant: 5 exp / 2 mant bits. *)

(** {1:addr_spaces Address spaces} *)

(** GPU memory address space. *)
type addr_space =
  | Global  (** Global device memory. *)
  | Local  (** Shared/local memory (workgroup scope). *)
  | Reg  (** Register storage. *)
  | Alu
      (** Scalar ALU value space, used for symbolic variables and loaded
          values. *)

(** Image storage kind. Mirrors tinygrad's [imageh] and [imagef] dtype
    constructors. *)
type image_kind = Imageh | Imagef

(** {1:vals Value dtypes} *)

module Val : sig
  type t
  (** Value dtype: a {!scalar} identity with a vector width ([count]).

      Invariant: [count >= 0]. Constructors produce scalar values with
      [count = 1]; use {!vec} for vector dtypes. A zero-lane vector is represented
      with [count = 0], matching tinygrad's raw dtype layer. *)

  (** {2 Accessors} *)

  val scalar : t -> scalar
  (** [scalar dt] is the element type of [dt]. *)

  val count : t -> int
  (** [count dt] is the vector width of [dt]. [1] for scalar types. *)

  (** {2 Constructors} *)

  val of_scalar : scalar -> t
  (** [of_scalar s] is the scalar dtype for [s] with [count = 1]. *)

  (** The named scalar constants below are all [of_scalar] applied to
      the matching {!scalar} constructor, with [count = 1]. *)

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

  val default_float : t
  (** [default_float] is the dtype named by the [DEFAULT_FLOAT] environment
      variable at module initialization, or {!float32} when unset. The value
      must name a floating-point scalar dtype using tinygrad dtype names or
      public aliases such as ["float32"], ["float"], ["half"], or
      ["default_float"]. Short display mnemonics such as ["f32"], ["f16"], and
      ["bf16"] are not accepted. Used by {!least_upper_float}. *)

  val default_int : t
  (** [default_int] is {!int32}. Used by {!sum_acc_dtype} to widen narrow
      integer accumulators. *)

  (** {2 Transformers} *)

  val scalarize : t -> t
  (** [scalarize dt] is [dt] with [count = 1].

      See also {!vec}. *)

  val vec : int -> t -> t
  (** [vec n dt] is a vector type with [n] lanes of [scalar dt].

      Returns [dt] unchanged when [n = 1] or [scalar dt] is {!Void}. For
      [n = 0] and a non-{!Void} scalar, returns a zero-lane vector with
      bitsize [0].

      Raises [Invalid_argument] if [count dt <> 1] (already vectorized),
        or [n < 0].

      See also {!scalarize}. *)

  val with_scalar : scalar -> t -> t
  (** [with_scalar s dt] is [dt] with its scalar identity replaced by
      [s], preserving [count]. No validation is performed; the caller
      is responsible for preserving the [count >= 0] invariant. *)

  (** {2 Predicates} *)

  val is_float : t -> bool
  (** [is_float dt] is [true] iff [scalar dt] is a floating-point type. *)

  val is_int : t -> bool
  (** [is_int dt] is [true] iff [scalar dt] is an integer type (including
      {!Weakint}). {!Uint128} and {!Uint256} are private storage helpers and
      are not classified as integers, matching tinygrad. *)

  val is_unsigned : t -> bool
  (** [is_unsigned dt] is [true] iff [scalar dt] is a public unsigned integer
      dtype. {!Uint128} and {!Uint256} are private storage helpers and are not
      classified as unsigned, matching tinygrad. *)

  val is_bool : t -> bool
  (** [is_bool dt] is [true] iff [scalar dt] is {!Bool}. *)

  val is_fp8 : t -> bool
  (** [is_fp8 dt] is [true] iff [scalar dt] is an fp8 dtype. *)

  (** {2 Properties} *)

  val bitsize : t -> int
  (** [bitsize dt] is the total size in bits: scalar bit width times
      [count]. {!Void} has bitsize [0], {!Weakint} has [800]. *)

  val itemsize : t -> int
  (** [itemsize dt] is {!bitsize} rounded up to whole bytes. *)

  val priority : t -> int
  (** [priority dt] is the promotion priority of [scalar dt]. See
      {!type-scalar} for the range. *)

  (** {2 Promotion} *)

  val least_upper_dtype : t list -> t
  (** [least_upper_dtype ts] is the least upper bound of the scalars of
      [ts] in the promotion lattice. The result always has
      [count = 1]; input [count]s are ignored.

      Promotion is total over numeric types: any pair has a common
      supertype. Some edges are lossy (e.g., [Int64] to [Uint64] loses
      negative values; [Uint64] to fp8 loses most precision).

      Raises [Invalid_argument] if [ts] is empty.

      See also {!least_upper_float} and {!can_lossless_cast}. *)

  val least_upper_float : t -> t
  (** [least_upper_float t] is [scalarize t] if [t] is floating-point, or
      [least_upper_dtype [scalarize t; default_float]] otherwise.

      See also {!least_upper_dtype}. *)

  val can_lossless_cast : t -> t -> bool
  (** [can_lossless_cast src dst] is [true] iff every value
      representable by [scalar src] is exactly representable by
      [scalar dst]. {!Bool} casts losslessly to any type; any signed
      or unsigned integer type casts losslessly to {!Weakint}. Only
      the [scalar] component is examined; [count] is ignored.

      This checks exact representability, not promotion. For instance,
      [can_lossless_cast int32 float32] is [false] (float32 cannot
      represent every 32-bit integer).

      See also {!least_upper_dtype}. *)

  val sum_acc_dtype : t -> t
  (** [sum_acc_dtype dt] is the accumulator dtype for sum-like reductions
      over [dt]. The result always has [count = 1].

      Widening rules:
      - Unsigned integers promote to at least {!uint32}.
      - Signed integers, {!Weakint}, and booleans promote to at least
        {!int32}.
      - Floats promote to at least the dtype named by the [SUM_DTYPE]
        environment variable, or {!float32} when unset. [SUM_DTYPE] is read
        when [sum_acc_dtype] is called and must name a scalar dtype using
        tinygrad dtype names or public aliases such as ["float32"], ["double"],
        ["uchar"], or ["default_float"]. Short display mnemonics such as
        ["f32"], ["u8"], and ["bf16"] are not accepted. *)

  (** {2 Comparison} *)

  val equal : t -> t -> bool
  (** [equal a b] is [true] iff [a] and [b] have the same scalar and count. *)

  val compare : t -> t -> int
  (** [compare a b] is a total order over value dtypes. Orders first
      by scalar promotion priority, then scalar bit width, then
      constructor ordinal, then by [count]. *)

  (** {2 Formatting} *)

  val to_string : t -> string
  (** [to_string dt] is the short name of [dt]. For [count = 1] this
      is the scalar mnemonic (e.g., ["f32"], ["i64"]). For vector
      types the lane count is appended with the multiplication sign
      (e.g., ["f32×4"]). *)

  val repr : t -> string
  (** [repr dt] is the tinygrad-style dtype representation, such as
      ["dtypes.int"] or ["dtypes.float.vec(4)"].

      Raises [Invalid_argument] if [dt] is a private wide helper dtype
      ({!Uint128} or {!Uint256}), matching tinygrad's lack of public repr
      entries for those helpers. *)

  val pp : Format.formatter -> t -> unit
  (** [pp] formats a value dtype using {!to_string}. *)
end

(** {1:ptrs Pointer dtypes} *)

module Ptr : sig
  type t
  (** Pointer dtype: a base {!Val.t} plus address space, buffer
      element count ([size]), pointer vector width ([v]), and optional image
      metadata.

      Invariant: [v >= 1]. [size = -1] indicates an unbounded buffer.
      Image pointers additionally have base {!Val.float32}, address space
      {!Global}, [size] equal to the product of their image shape, and priority
      [100]. Construct image pointers with {!Image.imageh} or {!Image.imagef}. *)

  (** {2 Accessors} *)

  val scalar : t -> scalar
  (** [scalar p] is the scalar identity of the base value dtype. *)

  val count : t -> int
  (** [count p] is the vector width of the base value dtype. *)

  val addrspace : t -> addr_space
  (** [addrspace p] is the memory address space of [p]. *)

  val v : t -> int
  (** [v p] is the pointer vector width of [p] (number of independent
      pointer lanes packed together). Always [>= 1]. *)

  val size : t -> int
  (** [size p] is the number of elements in the pointed-to buffer, or
      [-1] for unbounded. *)

  val base : t -> Val.t
  (** [base p] is the pointed-to value dtype (scalar and base count). *)

  val value : t -> Val.t
  (** [value p] is the dtype produced by reading through [p]. It preserves the
      pointer vector width: reading through [float.ptr(...).vec(4)] produces
      [float.vec(4)]. *)

  val image_kind : t -> image_kind option
  (** [image_kind p] is [Some kind] if [p] is an image pointer dtype and
      [None] otherwise. *)

  val image_shape : t -> int list option
  (** [image_shape p] is [Some shape] if [p] is an image pointer dtype and
      [None] otherwise. *)

  val is_image : t -> bool
  (** [is_image p] is [true] iff [p] carries image metadata. *)

  (** {2 Constructors} *)

  val create : Val.t -> addrspace:addr_space -> size:int -> t
  (** [create base ~addrspace ~size] is a pointer to [base] in
      [addrspace] with [size] elements ([-1] for unbounded). Pointer
      vector width defaults to [1]. *)

  val create_v : Val.t -> addrspace:addr_space -> size:int -> v:int -> t
  (** [create_v base ~addrspace ~size ~v] is like {!create} with
      explicit pointer vector width [v].

      Raises [Invalid_argument] if [v < 1]. *)

  (** {2 Transformers} *)

  val scalarize : t -> t
  (** [scalarize p] is [p] with pointer vector width [1] and base
      [count = 1]. The address space and [size] are preserved. *)

  val vec : int -> t -> t
  (** [vec n p] is [p] with pointer vector width set to [n].

      Raises [Invalid_argument] if [n < 1] or if [p] is already vectorized. *)

  val with_base : Val.t -> t -> t
  (** [with_base dt p] is [p] with base value dtype replaced by [dt].
      The address space, [v], and [size] are preserved. Image metadata is
      removed because image dtypes have a fixed {!Val.float32} base. *)

  val with_size : int -> t -> t
  (** [with_size n p] is [p] with buffer element count set to [n]. Image
      metadata is removed if [n] differs from the current image size, because
      image dtype size is derived from shape. *)

  (** {2 Properties} *)

  val bitsize : t -> int
  (** [bitsize p] is the base storage width in bits. For ordinary pointers it
      is the base scalar width times base [count]. For image pointers it is
      [16] for {!Imageh} and [32] for {!Imagef}. *)

  val itemsize : t -> int
  (** [itemsize p] is {!bitsize} rounded up to whole bytes. *)

  val nbytes : t -> int
  (** [nbytes p] is [size p * itemsize p].

      Raises [Invalid_argument] if [size p = -1]. *)

  val priority : t -> int
  (** [priority p] is the promotion priority. Image pointers have priority
      [100], matching tinygrad image dtypes. *)

  (** {2 Comparison} *)

  val equal : t -> t -> bool
  (** [equal a b] is [true] iff all fields of [a] and [b] are
      structurally equal (base scalar, base count, addrspace, v, size).

      {b Note.} IR validators may use partial field comparisons (e.g.,
      index validation ignores [size]); those checks are intentionally
      distinct from this structural equality. *)

  val compare : t -> t -> int
  (** [compare a b] is a total order over pointer types. Orders by
      priority, base scalar priority, base count, addrspace, [v], [size], and
      image metadata. *)

  (** {2 Formatting} *)

  val to_string : t -> string
  (** [to_string p] is a human-readable representation of [p] in the
      form [{base}*[.vec(v)] \[addrspace\]] (e.g.,
      ["f32* \[global\]"], ["i32*.vec(4) \[local\]"]). *)

  val repr : t -> string
  (** [repr p] is the tinygrad-style pointer or image dtype representation,
      such as ["dtypes.float.ptr(-1)"],
      ["dtypes.int.ptr(4, AddrSpace.LOCAL).vec(2)"], or
      ["dtypes.imageh((8, 16, 4))"]. *)

  val pp : Format.formatter -> t -> unit
  (** [pp] formats a pointer dtype using {!to_string}. *)
end

(** {1:images Image pointer dtypes} *)

module Image : sig
  type kind = image_kind = Imageh | Imagef
  (** Image storage kind. [Imageh] stores half-precision image elements;
      [Imagef] stores single-precision image elements. Both have
      {!Val.float32} as their loaded base dtype. *)

  type t = Ptr.t
  (** Image dtype. Images are specialized {!Ptr.t} values, matching
      tinygrad's [ImageDType] as a subclass of pointer dtype. *)

  val create : kind -> int list -> t
  (** [create kind shape] is an image pointer dtype in {!Global} address space
      with base {!Val.float32}, [size] equal to the product of [shape], pointer
      vector width [1], and image storage kind [kind]. *)

  val imageh : int list -> t
  (** [imageh shape] is [create Imageh shape]. It has priority [100],
      bitsize [16], base {!Val.float32}, address space {!Global}, and
      [size = List.fold_left ( * ) 1 shape]. *)

  val imagef : int list -> t
  (** [imagef shape] is [create Imagef shape]. It has priority [100],
      bitsize [32], base {!Val.float32}, address space {!Global}, and
      [size = List.fold_left ( * ) 1 shape]. *)

  val kind : t -> kind
  (** [kind image] is the image storage kind.

      Raises [Invalid_argument] if [image] is not an image pointer dtype. *)

  val shape : t -> int list
  (** [shape image] is the image shape.

      Raises [Invalid_argument] if [image] is not an image pointer dtype. *)

  val base : t -> Val.t
  (** [base image] is {!Val.float32}.

      Raises [Invalid_argument] if [image] is not an image pointer dtype. *)

  val size : t -> int
  (** [size image] is the product of {!shape}.

      Raises [Invalid_argument] if [image] is not an image pointer dtype. *)

  val pitch : t -> int
  (** [pitch image] is the tinygrad row pitch in bytes,
      [shape\[1\] * 4 * Ptr.itemsize image], with [shape\[1\]] rounded up to
      256 pixels on macOS.

      Raises [Invalid_argument] if [image] is not an image pointer dtype or
      has fewer than two dimensions. *)

  val addrspace : t -> addr_space
  (** [addrspace image] is {!Global}.

      Raises [Invalid_argument] if [image] is not an image pointer dtype. *)

  val vec : int -> t -> t
  (** [vec n image] is [image] with pointer vector width [n]. The image shape
      and storage kind are preserved.

      Raises [Invalid_argument] if [n < 1], if [image] is already vectorized,
      or if [image] is not an image pointer dtype. *)
end

(** {1:types Unified dtype} *)

type t = Val of Val.t | Ptr of Ptr.t
(** A dtype that is either a value or a pointer. Used in IR nodes whose
    dtype can be either (e.g., [Index] nodes that may or may not carry
    pointer semantics). *)

(** {2:dispatch_access Dispatching accessors} *)

val scalar : t -> scalar
(** [scalar dt] is the scalar identity.
    [scalar (Val v)] is [Val.scalar v].
    [scalar (Ptr p)] is [Ptr.scalar p]. *)

val count : t -> int
(** [count dt] is the value vector width.
    [count (Val v)] is [Val.count v].
    [count (Ptr p)] is [Ptr.count p]. *)

val vcount : t -> int
(** [vcount dt] is the outer vector width: lane count for values,
    pointer vector width for pointers.
    [vcount (Val v)] is [Val.count v].
    [vcount (Ptr p)] is [Ptr.v p]. *)

val is_ptr : t -> bool
(** [is_ptr dt] is [true] iff [dt] is [Ptr _]. *)

val val_of : t -> Val.t
(** [val_of dt] is the value component of [dt]: [v] for [Val v] and
    [Ptr.base p] for [Ptr p]. Pointer vector width [v] is discarded. *)

(** {2:dispatch_transform Dispatching transformers} *)

val scalarize : t -> t
(** [scalarize dt] dispatches to {!Val.scalarize} or {!Ptr.scalarize},
    preserving the [Val]/[Ptr] wrapper. *)

val vec : int -> t -> t
(** [vec n dt] dispatches to {!Val.vec} or {!Ptr.vec}, preserving the
    [Val]/[Ptr] wrapper. *)

(** {2:predicates Predicates} *)

val is_float : t -> bool
(** [is_float dt] is [true] iff [dt] is a value dtype whose scalar is a
    floating-point type
    ({!Float16}, {!Bfloat16}, {!Float32}, {!Float64}, {!Fp8e4m3}, or
    {!Fp8e5m2}) or an image pointer dtype. Ordinary pointer dtypes are not
    classified by their pointee scalar. *)

val is_int : t -> bool
(** [is_int dt] is [true] iff [dt] is a value dtype whose scalar is a signed or
    unsigned integer type, including {!Weakint}. {!Uint128} and {!Uint256} are
    private storage helpers and are not classified as integers. *)

val is_unsigned : t -> bool
(** [is_unsigned dt] is [true] iff [dt] is a value dtype whose scalar is an
    unsigned integer dtype. {!Uint128} and {!Uint256} are private storage
    helpers and are not classified as unsigned. *)

val is_bool : t -> bool
(** [is_bool dt] is [true] iff [dt] is a value dtype whose scalar is {!Bool}. *)

val is_fp8 : t -> bool
(** [is_fp8 dt] is [true] iff [dt] is a value dtype whose scalar is an fp8
    dtype. *)

(** {2:promotion Promotion} *)

val least_upper_dtype : t list -> t
(** [least_upper_dtype ts] is the least upper bound of [ts].

    If [ts] contains image pointer dtypes, the first image dtype is returned.
    Ordinary pointer dtypes are rejected. Otherwise this dispatches to
    {!Val.least_upper_dtype} and wraps the result in {!Val}.

    Raises [Invalid_argument] if [ts] is empty or contains an ordinary pointer
    dtype. *)

val least_upper_float : t -> t
(** [least_upper_float dt] is [dt] for image pointer dtypes. Ordinary pointer
    dtypes are rejected. Floating-point value dtypes are returned unchanged;
    other value dtypes promote with {!Val.default_float}.

    Raises [Invalid_argument] if [dt] is an ordinary pointer dtype. *)

(** {2:properties Properties} *)

val bitsize : t -> int
(** [bitsize dt] is the total size of [dt] in bits: the scalar bit
    width multiplied by [count dt]. {!Void} has bitsize [0];
    {!Weakint} has sentinel bitsize [800]. For ordinary pointers, this measures
    the pointed-to base width and ignores the pointer vector width [v]. For
    image pointers, this is [16] for {!Imageh} and [32] for {!Imagef}. *)

val itemsize : t -> int
(** [itemsize dt] is {!bitsize} rounded up to whole bytes. *)

val priority : t -> int
(** [priority dt] is the promotion priority of [scalar dt]. Higher
    priorities absorb lower ones in {!Val.least_upper_dtype}. Ranges
    from [-1] ({!Void}) through [0] ({!Bool}, {!Weakint}) to [14]
    ({!Float64}); image pointer dtypes have priority [100]. *)

(** {2:bounds Bounds} *)

type bound =
  [ `Bool of bool | `SInt of int64 | `UInt of int64 | `Float of float ]
(** Numeric bounds for dtypes. Returned by {!min} and {!max}.

    - [`Bool b] for boolean bounds.
    - [`SInt n] for signed integer bounds, including {!Weakint}, which
      reports the [Int64] range as an approximation.
    - [`UInt n] for unsigned integer bounds. Values are raw 64-bit
      unsigned bit patterns stored in an [int64] (e.g. [Uint64]'s max
      is [`UInt Int64.minus_one]).
    - [`Float f] for floating-point bounds ([neg_infinity] and
      [infinity]). *)

val min : t -> bound
(** [min dt] is the smallest value representable by [dt].

    Value dtypes use their scalar bounds. Image pointer dtypes are
    float-like and return [`Float neg_infinity]. Ordinary pointer dtypes are
    not classified by their pointee scalar and return [`Bool false].

    Raises [Invalid_argument] if [dt] is a value dtype whose scalar is
    {!Void}.

    See also {!max}. *)

val max : t -> bound
(** [max dt] is the largest value representable by [dt].

    Value dtypes use their scalar bounds. Image pointer dtypes are
    float-like and return [`Float infinity]. Ordinary pointer dtypes are not
    classified by their pointee scalar and return [`Bool true].

    Raises [Invalid_argument] if [dt] is a value dtype whose scalar is
    {!Void}.

    See also {!min}. *)

val finfo : t -> int * int
(** [finfo dt] is [(exponent_bits, mantissa_bits)] for the
    floating-point dtype [dt]. Mantissa bits exclude the implicit
    leading [1]. For example, [finfo (Val float32)] is [(8, 23)] and
    [finfo (Val float16)] is [(5, 10)].

    Raises [Invalid_argument] if [dt] is a pointer dtype or if [dt] is a value
    dtype whose scalar is not floating-point.

    See also {!is_float}. *)

(** {2:comparison Comparison} *)

val equal : t -> t -> bool
(** [equal a b] is [true] iff [a] and [b] are both [Val] or both
    [Ptr] with equal components. *)

val compare : t -> t -> int
(** [compare a b] is a total order over {!t}. [Val] sorts before
    [Ptr]; within each kind, compares by {!Val.compare} or
    {!Ptr.compare}. *)

(** {2:fmt Formatting} *)

val to_string : t -> string
(** [to_string dt] formats [dt] using {!Val.to_string} or {!Ptr.to_string}. *)

val repr : t -> string
(** [repr dt] formats [dt] using tinygrad-style dtype syntax. *)

val pp : Format.formatter -> t -> unit
(** [pp] formats a dtype using {!to_string}. *)

(** {1:scalar_fmt Scalar formatting} *)

val scalar_to_string : scalar -> string
(** [scalar_to_string s] is the short name of [s] (e.g., ["f32"], ["i64"],
    ["bool"], ["void"], ["weakint"]). *)

val pp_scalar : Format.formatter -> scalar -> unit
(** [pp_scalar] formats a {!scalar} using {!scalar_to_string}. *)

val addr_space_to_string : addr_space -> string
(** [addr_space_to_string a] is ["global"], ["local"], ["reg"], or ["alu"]. *)

val pp_addr_space : Format.formatter -> addr_space -> unit
(** [pp_addr_space] formats an address space. *)

(** {1:convenience Convenience constructors}

    Value dtype constants wrapped as {!t} for direct use in any context
    expecting a unified dtype. For the unwrapped {!Val.t} versions, use
    the {!Val} module directly. *)

val of_scalar : scalar -> t
(** [of_scalar s] is [Val (Val.of_scalar s)]. *)

(** The named constants below are [Val]-wrapped versions of the
    matching {!Val} constants. [default_float] follows [DEFAULT_FLOAT] as
    described by {!Val.default_float}; [default_int] is {!int32}. *)

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
val default_float : t
val default_int : t
val imageh : int list -> t
(** [imageh shape] is [Ptr (Image.imageh shape)]. *)

val imagef : int list -> t
(** [imagef shape] is [Ptr (Image.imagef shape)]. *)

(** {1:fp_conv Floating-point conversion}

    Bit-exact precision-narrowing utilities used for constant folding.
    Rounding is round-to-nearest-even for all formats; conversions are
    implemented in OCaml rather than delegated to hardware so results
    are reproducible across backends. *)

val float_to_fp16 : float -> float
(** [float_to_fp16 x] rounds [x] to IEEE 754 binary16 (half) precision
    using round-to-nearest-even. The result is a [float] holding a
    value exactly representable in half precision. Overflow produces
    an infinity; underflow produces zero or a subnormal; infinities
    pass through; NaNs produce a NaN (bit pattern not preserved). *)

val float_to_bf16 : float -> float
(** [float_to_bf16 x] rounds [x] to bfloat16 precision using
    round-to-nearest-even. Non-finite values pass through unchanged. *)

val float_to_fp8 : scalar -> float -> int
(** [float_to_fp8 s x] encodes [x] as an fp8 byte value in [0..255].

    Raises [Invalid_argument] if [s] is not an fp8 dtype.

    See also {!fp8_to_float}. *)

val fp8_to_float : scalar -> int -> float
(** [fp8_to_float s byte] decodes the fp8 [byte] into a [float].

    Raises [Invalid_argument] if [s] is not an fp8 dtype.

    See also {!float_to_fp8}. *)

val truncate_float : Val.t -> float -> float
(** [truncate_float dt x] rounds [x] to the precision of floating-point
    dtype [dt]. {!Float64} is the identity. {!Float32} round-trips
    through [Int32.bits_of_float]. Narrower types delegate to
    {!float_to_fp16}, {!float_to_bf16}, or the fp8 conversion pair.
    The [count] of [dt] is ignored.

    Raises [Invalid_argument] if [dt] is not floating-point.

    See also {!truncate_int}. *)

val truncate_int : Val.t -> int -> int
(** [truncate_int dt x] reduces integer [x] into the range of integer
    dtype [dt]. Unsigned types mask to the low [bitsize dt] bits;
    signed types and {!Weakint} apply the same mask and sign-extend.
    When the target width equals or exceeds [Sys.int_size] (notably
    for {!Int64}, {!Uint64}, and {!Weakint} at its 800-bit sentinel)
    [x] is returned unchanged. {!Bool} maps [0] to [0] and any other
    value to [1].

    Raises [Invalid_argument] if [dt] is not an integer or bool type.

    See also {!truncate_float}. *)

(** {1:storage Storage conversion}

    Tinygrad-compatible scalar storage helpers. These functions operate on one
    scalar lane; callers handling vectors should apply them lane-by-lane. *)

type storage_scalar = [ `Bool of bool | `Float of float | `Int of int64 ]
(** The type for scalar values at the storage boundary.

    [`Int n] carries raw signed or unsigned integer storage bits in an [int64].
    [`Float f] carries a host floating-point value. [`Bool b] carries a boolean
    storage value. *)

val storage_fmt_for_dtype : Val.t -> char option
(** [storage_fmt_for_dtype dt] is the Python [struct] format character used by
    tinygrad for one scalar lane of [dt].

    Returns [None] for dtypes without a portable storage format: {!Void},
    {!Weakint}, {!Uint128}, and {!Uint256}. {!Bfloat16} stores as ['H'] and fp8
    dtypes store as ['B']. The [count] of [dt] is ignored. *)

val to_storage_scalar : Val.t -> storage_scalar -> storage_scalar
(** [to_storage_scalar dt x] maps [x] to the scalar value written by tinygrad's
    storage path for [dt].

    {!Bfloat16} returns the high 16 bits of the rounded float as [`Int].
    Fp8 dtypes return the encoded byte as [`Int]. {!Float16} returns a
    precision-rounded [`Float]. Other scalar dtypes return the corresponding
    boolean, integer, or float representation.

    Raises [Invalid_argument] if [dt] is {!Void}. The [count] of [dt] is
    ignored. *)

val from_storage_scalar : storage_scalar -> Val.t -> storage_scalar
(** [from_storage_scalar x dt] maps one stored scalar value for [dt] back to a
    host scalar.

    {!Bfloat16} and fp8 storage integers decode to [`Float]. Other scalar
    dtypes return the corresponding boolean, integer, or float representation.

    Raises [Invalid_argument] if [dt] is {!Void}. The [count] of [dt] is
    ignored. *)

val truncate : Val.t -> storage_scalar -> storage_scalar
(** [truncate dt x] coerces [x] to the value domain of scalar dtype [dt].

    Floating-point dtypes use {!truncate_float}. Integer dtypes use two's
    complement wrapping with the target width, including unsigned raw bit
    patterns. {!Bool} maps zero-like values to [`Bool false] and all other
    values to [`Bool true].

    Raises [Invalid_argument] if [dt] is {!Void}. The [count] of [dt] is
    ignored. *)
