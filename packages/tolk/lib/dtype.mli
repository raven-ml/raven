(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Data types, their constants and their promotion.

    A data type names the type of one scalar element. A value with several
    elements gets them from its shape, and a pointer's address space is a
    property of the pointer, so neither is part of a data type.

    Integer and floating-point literals start {e weak}, as {!Weak_int} and
    {!Weak_float}: their width is not committed yet, and they take the width of
    what they are combined with. {!least_upper} combines data types and
    {!strong} commits a weak one to its default width. *)

(** {1:consts Constants} *)

type value = [ `Bool of bool | `Int of Bigint.t | `Float of float ]
(** The type for values. An [`Int] is a mathematical integer: its data type
    bounds it only when it is truncated. *)

type const = [ value | `Invalid ]
(** The type for constants: a value, or [`Invalid], the value of an element that
    is masked out. *)

val nan : float
(** [nan] is the positive quiet NaN whose payload is zero: the NaN a constant
    takes when an operation makes one from operands that are not NaNs. *)

val equal_const : [< const ] -> [< const ] -> bool
(** [equal_const c0 c1] is [true] iff [c0] and [c1] are the same constant: the
    same constructor with the same payload. Floats are the same when their bits
    are, so [0.0] and [-0.0] differ, and so do NaNs of different bits. *)

val hash_const : [< const ] -> int
(** [hash_const c] is a hash of [c], compatible with {!equal_const}. *)

val pp_const : Format.formatter -> [< const ] -> unit
(** [pp_const] formats a constant as a literal: [True], [False], [Invalid], an
    integer in decimal, and a float in the fewest significant digits that read
    back as the same float. A float has a [.0] if it has no fractional digit, is
    in exponent notation below [1e-4] and from [1e16] in magnitude ([1e-05],
    [1.5e+16]), and is [inf], [-inf] or [nan] if it is not finite. *)

(** {2:arith Arithmetic on values} *)

(** Arithmetic and comparison on values, as on numbers: a [`Bool] counts as [0]
    or [1], an operation on two integers gives an exact integer, and one that
    involves a float gives a float, in which an integer that rounds past the
    greatest double is the infinity of its sign. The operators compare an
    integer and a float exactly, without rounding the integer, and a comparison
    with NaN is [false], except [<>], which is [true]. *)
module Value : sig
  type t = value
  (** The type for values. *)

  val of_int : int -> t
  (** [of_int n] is [`Int n]. *)

  val to_float : t -> float
  (** [to_float v] is [v] as a float: [0.] or [1.] for a [`Bool], and an integer
      rounded to nearest, ties to even, or to the infinity of its sign if it
      rounds past the greatest double. *)

  val to_z : t -> Bigint.t
  (** [to_z v] is [v] as an integer: [0] or [1] for a [`Bool], and a float
      rounded towards zero.

      Raises [Invalid_argument] if [v] is an infinity or a NaN. *)

  val to_int : t -> int
  (** [to_int v] is [to_z v] as an [int].

      Raises [Invalid_argument] if [v] is an infinity or a NaN, or if [to_z v]
      does not fit an [int]. *)

  val to_bool : t -> bool
  (** [to_bool v] is [true] iff [v] is not zero: a NaN is [true]. *)

  val compare : t -> t -> int
  (** [compare v0 v1] is a total order on values by magnitude, where NaN is less
      than every other value and equal to itself. Values of equal magnitude
      compare equal whatever their kind, so [`Bool true], [`Int 1] and
      [`Float 1.] are equal. *)

  val ( = ) : t -> t -> bool
  (** [v0 = v1] is [true] iff [v0] and [v1] have the same magnitude. *)

  val ( <> ) : t -> t -> bool
  (** [v0 <> v1] is [not (v0 = v1)]. *)

  val ( < ) : t -> t -> bool
  (** [v0 < v1] is [true] iff [v0]'s magnitude is less than [v1]'s. *)

  val ( <= ) : t -> t -> bool
  (** [v0 <= v1] is [v0 < v1 || v0 = v1]. *)

  val ( > ) : t -> t -> bool
  (** [v0 > v1] is [v1 < v0]. *)

  val ( >= ) : t -> t -> bool
  (** [v0 >= v1] is [v1 <= v0]. *)

  val min : t -> t -> t
  (** [min v0 v1] is [v1] if [v1 < v0], and [v0] otherwise. *)

  val max : t -> t -> t
  (** [max v0 v1] is [v1] if [v1 > v0], and [v0] otherwise. *)

  val ( ~- ) : t -> t
  (** [-v] is the negation of [v]. *)

  val ( + ) : t -> t -> t
  (** [v0 + v1] is the sum of [v0] and [v1]. *)

  val ( - ) : t -> t -> t
  (** [v0 - v1] is the difference of [v0] and [v1]. *)

  val ( * ) : t -> t -> t
  (** [v0 * v1] is the product of [v0] and [v1]. *)

  val ( // ) : t -> t -> t
  (** [v0 // v1] is the quotient of [v0] by [v1] rounded down.

      Raises [Division_by_zero] if [v1] is zero. *)

  val ( % ) : t -> t -> t
  (** [v0 % v1] is [v0 - (v0 // v1) * v1], of the sign of [v1] when it is not
      zero: a float remainder is exact.

      Raises [Division_by_zero] if [v1] is zero. *)
end

(** {1:addr Address spaces} *)

(** The type for address spaces: the memory a pointer addresses. *)
type addr_space =
  | Global  (** Device memory, shared by every thread. *)
  | Local  (** Memory shared by the threads of a workgroup. *)
  | Reg  (** A thread's registers. *)
  | Alu  (** Values passed to a kernel by value. *)

val pp_addr_space : Format.formatter -> addr_space -> unit
(** [pp_addr_space] formats an address space as [AddrSpace.GLOBAL],
    [AddrSpace.LOCAL], [AddrSpace.REG] or [AddrSpace.ALU]. *)

val addr_space_of_string : string -> (addr_space, string) result
(** [addr_space_of_string s] is the address space named [s]: [GLOBAL], [LOCAL],
    [REG] or [ALU], the name {!pp_addr_space} formats after [AddrSpace.]. The
    error names [s]. *)

(** {1:dtypes Data types} *)

(** The type for data types. *)
type t =
  | Void  (** The type of no value. *)
  | Weak_int  (** An integer whose width is not committed yet. *)
  | Bool
  | Int8
  | Uint8
  | Int16
  | Uint16
  | Int32
  | Uint32
  | Int64
  | Uint64
  | Weak_float  (** A float whose width is not committed yet. *)
  | Fp8e4m3
      (** 8-bit float with 4 exponent and 3 mantissa bits and no infinities. *)
  | Fp8e5m2  (** 8-bit float with 5 exponent and 2 mantissa bits. *)
  | Fp8e4m3fnuz
      (** 8-bit float with 4 exponent and 3 mantissa bits, no infinities, no
          negative zero and one NaN. *)
  | Fp8e5m2fnuz
      (** 8-bit float with 5 exponent and 2 mantissa bits, no infinities, no
          negative zero and one NaN. *)
  | Float16
  | Bfloat16  (** 16-bit float with 8 exponent and 7 mantissa bits. *)
  | Float32
  | Float64

val priority : t -> int
(** [priority dt] is [dt]'s rank in the promotion order, from [-1] for {!Void}
    to [15] for {!Float64}. *)

val bitsize : t -> int
(** [bitsize dt] is the number of bits of a [dt] value: [0] for {!Void}, [1] for
    {!Bool}, and [800] for the weak data types, wider than any value a program
    stores. *)

val itemsize : t -> int
(** [itemsize dt] is the number of bytes of a [dt] value, [bitsize dt] rounded
    up to a whole byte. *)

val name : t -> string
(** [name dt] is the name of [dt] in C, such as ["unsigned char"], ["half"] or
    ["__bf16"]. {!Void} and the weak data types have the names ["void"],
    ["weakint"] and ["weakfloat"]. *)

val fmt : t -> char option
(** [fmt dt] is the format character of a [dt] value in a byte string: ['?'] for
    {!Bool}, ['b'], ['h'], ['i'] and ['q'] for the signed integers, ['B'],
    ['H'], ['I'] and ['Q'] for the unsigned ones, and ['e'], ['f'] and ['d'] for
    {!Float16}, {!Float32} and {!Float64}. It is [None] for the other data
    types. See {!storage_fmt}. *)

val min : t -> value
(** [min dt] is the least value of [dt]: [-infinity] for the floats that have
    infinities, the negated {!max} for the other floats, the least integer of
    [dt]'s {!bitsize} for the integers, weak included, and [false] for {!Bool}
    and {!Void}. *)

val max : t -> value
(** [max dt] is the greatest value of [dt]: [infinity] for the floats that have
    infinities, the greatest finite value for the other floats, the greatest
    integer of [dt]'s {!bitsize} for the integers, weak included, and [true] for
    {!Bool} and {!Void}. *)

val const : t -> [< const ] -> const
(** [const dt c] is [c] as a constant of [dt]. [`Invalid] is itself. For a float
    [dt] it is a [`Float], [c] truncated to [dt] ({!truncate}), except that a
    NaN keeps its sign and the payload [dt] holds, a signalling NaN staying one;
    an integer is rounded once, from its exact value, as a finite value. For
    {!Bool} it is [`Bool], [true] iff [c] is nonzero. Otherwise it is an [`Int]:
    a float rounded towards zero, and an integer unchanged, even out of [dt]'s
    bounds.

    Raises [Invalid_argument] if [c] is a NaN or an infinity and [dt] is neither
    a float nor {!Bool}. *)

val equal : t -> t -> bool
(** [equal d0 d1] is [true] iff [d0] and [d1] are the same data type. *)

val compare : t -> t -> int
(** [compare] is the promotion order: by {!priority}, then {!bitsize}, then
    {!name}. It orders {!Weak_int} after {!Bool}, and each 8-bit float just
    before its [fnuz] variant. *)

val hash : t -> int
(** [hash dt] is a hash of [dt], compatible with {!equal}. *)

val pp : Format.formatter -> t -> unit
(** [pp] formats a data type as [dtypes.] followed by its C-flavoured alias if
    it has one ([dtypes.char], [dtypes.uint], [dtypes.half], [dtypes.float],
    [dtypes.double]), and by its constructor's name in lowercase, without
    underscores, otherwise ([dtypes.bool], [dtypes.bfloat16], [dtypes.weakint]):
    [dtypes.] followed by a name {!of_string} reads. *)

(** {2:predicates Predicates and groups} *)

val is_float : t -> bool
(** [is_float dt] is [true] iff [dt] is in {!floats} or is {!Weak_float}. *)

val is_int : t -> bool
(** [is_int dt] is [true] iff [dt] is in {!ints} or is {!Weak_int}. *)

val is_unsigned : t -> bool
(** [is_unsigned dt] is [true] iff [dt] is in {!uints}. *)

val is_bool : t -> bool
(** [is_bool dt] is [true] iff [dt] is {!Bool}. *)

val fp8_ocp : t list
(** [fp8_ocp] is [[Fp8e4m3; Fp8e5m2]]. *)

val fp8_fnuz : t list
(** [fp8_fnuz] is [[Fp8e4m3fnuz; Fp8e5m2fnuz]]. *)

val fp8s : t list
(** [fp8s] is [fp8_ocp @ fp8_fnuz]. *)

val floats : t list
(** [floats] is the floats of known width: [fp8s], then {!Float16}, {!Bfloat16},
    {!Float32} and {!Float64}. *)

val uints : t list
(** [uints] is the unsigned integers, from {!Uint8} to {!Uint64}. *)

val sints : t list
(** [sints] is the signed integers, from {!Int8} to {!Int64}. *)

val ints : t list
(** [ints] is [uints @ sints]. *)

val weaks : t list
(** [weaks] is [[Weak_int; Weak_float]]. *)

val all : t list
(** [all] is every data type a program stores: [floats @ ints @ [Bool]]. *)

(** {2:literals Literals} *)

val of_const : [< const ] -> t
(** [of_const c] is the data type of the literal [c]: {!Bool} for [`Bool] and
    [`Invalid], {!Weak_int} for [`Int], and {!Weak_float} for [`Float]. *)

val of_consts : [< const ] list -> t
(** [of_consts cs] is the committed data type of the literals [cs]. If the
    greatest {!of_const} of [cs] by {!compare} is {!Weak_int}, it is
    [commit_int lo hi], where [lo] and [hi] bound the integers and booleans of
    [cs]; otherwise it is {!strong} of that greatest. It is [default_float ()]
    if [cs] is empty. *)

val finfo : t -> int * int
(** [finfo dt] is [(exponent, mantissa)], the number of exponent bits and of
    explicit mantissa bits of [dt].

    Raises [Invalid_argument] if [dt] is not in {!floats}. *)

(** {2:names Names and defaults} *)

val of_string : string -> (t, string) result
(** [of_string s] is the data type named [s], in any case: its constructor's
    name without underscores ([float32], [weakint]), a C-flavoured alias
    ([half], [float], [double], [char], [uchar], [short], [ushort], [int],
    [uint], [long], [ulong]), or [default_float] and [default_int] for
    {!default_float} and {!default_int}. The error names [s]. *)

val default_float : unit -> t
(** [default_float ()] is the data type named, in any case, by the current value
    of the setting {!Helpers.default_float}, [DEFAULT_FLOAT].

    Raises [Invalid_argument] if that is not a float of known width. *)

val default_int : unit -> t
(** [default_int ()] is the data type named, in any case, by the current value
    of the setting {!Helpers.default_int}, [DEFAULT_INT].

    Raises [Invalid_argument] if that is not an integer of known width. *)

val strong : t -> t
(** [strong dt] commits [dt] to a width: {!Weak_int} is [default_int ()],
    {!Weak_float} is [default_float ()], and any other [dt] is itself. *)

val commit_int : ?default_int:t -> Bigint.t -> Bigint.t -> t
(** [commit_int ~default_int lo hi] is the first of [default_int], {!Int32},
    {!Int64} and {!Uint64} that holds every integer from [lo] to [hi], and
    {!Int64} if none does. [default_int] defaults to [default_int ()].

    Raises [Invalid_argument] if [default_int] is not in {!ints}, or if
    [lo = hi] and no 64-bit integer holds it. *)

val weak : t -> t
(** [weak dt] is the weak data type of [dt]'s kind: {!Weak_float} for a float,
    {!Weak_int} for an integer, and [dt] itself for {!Bool} and {!Void}. *)

(** {1:promotion Promotion}

    The data types other than {!Void} are ordered by promotion: a data type
    promotes to itself and to those above it. {!Bool} is at the bottom, then
    {!Weak_int}, then the integers by width, each unsigned integer also
    promoting to the signed integer of twice its width. The 64-bit integers
    promote to {!Weak_float}, then come the 8-bit floats, then {!Float16} and
    {!Bfloat16}, which each 8-bit float promotes to, then {!Float32} and
    {!Float64}. *)

val least_upper : t list -> t
(** [least_upper dts] is the least data type by {!compare} that every data type
    of [dts] promotes to. Two different 8-bit floats promote to {!Float16}, the
    lesser of their two least common bounds, so [least_upper] is not
    associative: [least_upper [a; b; c]] is not always
    [least_upper [least_upper [a; b]; c]].

    Raises [Invalid_argument] if [dts] is empty or holds {!Void}. *)

val least_upper_float : t -> t
(** [least_upper_float dt] is the float that [dt] promotes to: {!Weak_float} for
    {!Weak_int}, [dt] itself for a float, and
    [least_upper [dt; default_float ()]] otherwise.

    Raises [Invalid_argument] if [dt] is {!Void}. *)

val can_lossless_cast : t -> t -> bool
(** [can_lossless_cast d0 d1] is [true] iff a cast from [d0] to [d1] is known to
    keep every value, that is iff [d0] is [d1] or {!Bool}, or [d1] is:
    - {!Float64} and [d0] is a float of known width other than {!Float64}, or an
      integer of 32 bits or fewer;
    - {!Float32} and [d0] is {!Float16}, {!Bfloat16}, an 8-bit float, or an
      integer of 16 bits or fewer;
    - {!Float16} and [d0] is an 8-bit float, {!Int8} or {!Uint8};
    - an unsigned integer and [d0] a narrower unsigned integer;
    - a signed integer and [d0] a narrower integer;
    - {!Weak_int} and [d0] is in {!ints}.

    It is [false] for some casts that keep every value, such as from an 8-bit
    float to {!Bfloat16}. *)

val sum_acc : t -> t
(** [sum_acc dt] is the data type that sums of [dt] values accumulate in: at
    least {!Uint32} for the unsigned integers, at least {!Int32} for {!Bool} and
    the other integers, and for floats at least the data type named by the
    environment variable [SUM_DTYPE], read when the program starts
    ({!Helpers.variable_string}), or {!Float32} if it is unset. rune's lowering
    takes the accumulator of a sum reduction from it.

    Raises [Invalid_argument] if [dt] is {!Void}, or if [SUM_DTYPE] does not
    name a data type. *)

(** {1:casts Casts} *)

val truncate : t -> value -> value
(** [truncate dt v] is [v] as a value of [dt]:
    - for a float of known width, [v] as a float rounded once to [dt]'s
      precision, to nearest with ties to even. A finite value that overflows a
      16-bit or wider float is an infinity, and one that overflows an 8-bit
      float is its greatest finite value of the same sign. An infinity is itself
      where the format has infinities, and its NaN where it has none. A NaN is
      quiet, as a conversion makes it: a signalling NaN becomes quiet with the
      same payload. It keeps its sign, except in the [fnuz] formats, whose one
      NaN decodes as a positive NaN; a 16-bit or wider float keeps the top of
      its payload, and an 8-bit float has one NaN per sign. An integer is
      rounded once, from its exact value, as a finite value: one that rounds
      past the greatest double is the infinity of its sign in {!Float64}, and
      overflows a narrower float.
    - for an integer of known width, [v] wrapped to [dt]'s width in two's
      complement;
    - for {!Bool}, [true] iff [v] is nonzero;
    - for a weak data type, [v] itself.

    Raises [Invalid_argument] if [dt] is {!Void}, or if [v] is a float and [dt]
    an integer of known width. *)

val storage_fmt : t -> char option
(** [storage_fmt dt] is the format character of the storage of [dt] values:
    {!fmt} for most, ['H'], a 16-bit unsigned integer, for {!Bfloat16}, and
    ['B'], an 8-bit unsigned integer, for the 8-bit floats. *)

val to_storage_scalar : t -> value -> value
(** [to_storage_scalar dt v] is the value that stores [v] in {!storage_fmt}: [v]
    rounded to {!Float16}, the [`Int] of the bits that encode [v] for
    {!Bfloat16} and the 8-bit floats, and [v] itself otherwise. Storage moves a
    NaN's bits and does not quiet it: a NaN keeps its sign and as much of its
    payload, quiet bit included, as the format holds. *)

val from_storage_scalar : t -> value -> value
(** [from_storage_scalar dt s] is the value stored by [s], the inverse of
    {!to_storage_scalar}: the [`Float] that the low bits of [s] encode for
    {!Bfloat16} and the 8-bit floats, a NaN with the sign and payload of its
    bits, and [s] itself otherwise.

    Raises [Invalid_argument] if [dt] is {!Bfloat16} or an 8-bit float and [s]
    is not an [`Int]. *)

val bitcast : t -> t -> value -> value
(** [bitcast d0 d1 v] is the [d1] value stored by the bits that store [v] as a
    [d0] value. Its bits are kept exactly, a signalling NaN's included, so
    [bitcast d1 d0 (bitcast d0 d1 c) = c] for every [c] that [d0] stores.

    Raises [Invalid_argument] if [d0] and [d1] have different {!itemsize}s, if
    either has no {!storage_fmt}, if [v] is a float and [d0] an integer, or if
    [v] is an integer out of the range of [d0]'s storage. *)
