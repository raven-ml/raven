(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Element formats.

    A dtype [('v, 's) t] names a storage format ['s] whose elements OCaml reads
    and writes as values of type ['v]. Operands of one format share a type, so a
    signature states which operands share one and the compiler checks every
    typed call. Where {!Bigarray} has the format, ['s] is Bigarray's element
    type, so arrays and bigarrays exchange bytes with their kinds checked by the
    type.

    A dtype's facts (its {!name}, {!bits}, {!kind} and {!float_format}) are rows
    of one table indexed by its {!code}, which the C header [nx_dtype.h] holds
    too. Every store of a [float] into a dtype follows one rule, stated with
    {!of_float}. *)

(** {1:elt Storage formats}

    The second parameter of a dtype. Bigarray's element types name the formats
    Bigarray has; the others are types of this module with no values. A dtype's
    name counts the bits of a whole element, Bigarray's element types one
    component's: [Complex64]'s elements are [complex32_elt]. *)

type float64_elt = Bigarray.float64_elt
(** IEEE 754 binary64. *)

type float32_elt = Bigarray.float32_elt
(** IEEE 754 binary32. *)

type float16_elt = Bigarray.float16_elt
(** IEEE 754 binary16. *)

(** bfloat16: binary32's sign, 8 exponent bits and the top 7 fraction bits. *)
type bfloat16_elt = |

(** float8 E4M3FN: 4 exponent bits, 3 fraction bits, exponent bias 7, no
    infinity; [S.1111.111] is NaN. Its largest finite value is 448. *)
type float8_e4m3fn_elt = |

(** float8 E5M2: 5 exponent bits, 2 fraction bits, exponent bias 15, with
    infinities and NaNs as in IEEE 754. Its largest finite value is 57344. *)
type float8_e5m2_elt = |

(** float4 E2M1FN: 2 exponent bits, 1 fraction bit, exponent bias 1, no infinity
    and no NaN. Its values are ±\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}. *)
type float4_e2m1fn_elt = |

type int64_elt = Bigarray.int64_elt
(** Signed 64-bit integers. *)

(** Unsigned 64-bit integers. *)
type uint64_elt = |

type int32_elt = Bigarray.int32_elt
(** Signed 32-bit integers. *)

(** Unsigned 32-bit integers. *)
type uint32_elt = |

type int16_signed_elt = Bigarray.int16_signed_elt
(** Signed 16-bit integers. *)

type int16_unsigned_elt = Bigarray.int16_unsigned_elt
(** Unsigned 16-bit integers. *)

type int8_signed_elt = Bigarray.int8_signed_elt
(** Signed 8-bit integers. *)

type int8_unsigned_elt = Bigarray.int8_unsigned_elt
(** Unsigned 8-bit integers. *)

(** Signed 4-bit integers, two's complement. *)
type int4_elt = |

(** Unsigned 4-bit integers. *)
type uint4_elt = |

type complex64_elt = Bigarray.complex64_elt
(** Complex numbers of two binary64 components, real part first. *)

type complex32_elt = Bigarray.complex32_elt
(** Complex numbers of two binary32 components, real part first. *)

(** Booleans, one to a byte: true iff the byte is not zero. *)
type bool_elt = |

(** Booleans, one to a bit. *)
type bit_elt = |

(** {1:dtypes Dtypes} *)

(** The type for dtypes. An element's OCaml type depends only on its format's
    width, on every target: [float] for every float format, [int] up to 16 bits,
    [int32] and [int64] for 32 and 64 bits, [Complex.t] and [bool].

    An unsigned format carries its bits in the type of its width, and the dtype
    says how to read them: {!pp_value} prints them unsigned, and the [Uint32]
    value [-1l] is 4294967295. *)
type ('v, 's) t =
  | Float64 : (float, float64_elt) t
  | Float32 : (float, float32_elt) t
  | Float16 : (float, float16_elt) t
  | Bfloat16 : (float, bfloat16_elt) t
  | Float8_e4m3fn : (float, float8_e4m3fn_elt) t
  | Float8_e5m2 : (float, float8_e5m2_elt) t
  | Float4_e2m1fn : (float, float4_e2m1fn_elt) t
  | Int64 : (int64, int64_elt) t
  | Uint64 : (int64, uint64_elt) t
  | Int32 : (int32, int32_elt) t
  | Uint32 : (int32, uint32_elt) t
  | Int16 : (int, int16_signed_elt) t
  | Uint16 : (int, int16_unsigned_elt) t
  | Int8 : (int, int8_signed_elt) t
  | Uint8 : (int, int8_unsigned_elt) t
  | Int4 : (int, int4_elt) t
  | Uint4 : (int, uint4_elt) t
  | Complex128 : (Complex.t, complex64_elt) t
  | Complex64 : (Complex.t, complex32_elt) t
  | Bool : (bool, bool_elt) t
  | Bit : (bool, bit_elt) t

(** The type for dtypes chosen at run time. {!equal_witness} recovers the static
    type. *)
type any = Any : ('v, 's) t -> any

val all : any list
(** [all] is every dtype, in {!code} order. *)

(** {1:facts Facts} *)

val code : ('v, 's) t -> int
(** [code dt] is [dt]'s code: its index in {!all} and the constant [NX_<NAME>]
    of [nx_dtype.h], as [NX_FLOAT32] for [Float32]. *)

val bits : ('v, 's) t -> int
(** [bits dt] is the width of one element of [dt] in bits: [1] for [Bit], [4]
    for the 4-bit formats, [8] for [Bool], up to [128] for [Complex128]. *)

val bytes : ('v, 's) t -> int -> int
(** [bytes dt n] is the number of bytes [n] elements of [dt] fill,
    ⌈[n · bits dt / 8]⌉.

    Raises [Invalid_argument] if [n < 0] or the number does not fit in an [int].
*)

val name : ('v, 's) t -> string
(** [name dt] is [dt]'s constructor name in lower case, as ["float32"] or
    ["float4_e2m1fn"]. *)

val of_name : string -> any option
(** [of_name s] is the dtype whose {!name} is [s], if any. *)

val pp : Format.formatter -> ('v, 's) t -> unit
(** [pp] formats a dtype's {!name}. *)

(** {1:kinds Kinds} *)

(** The type for kinds of number. ['v] is the value type of the kind's dtypes
    where the kind fixes one: integers read as [int], [int32] or [int64] by
    width. *)
type 'v kind =
  | Float : float kind
  | Complex : Complex.t kind
  | Signed : 'v kind
  | Unsigned : 'v kind
  | Boolean : bool kind

val kind : ('v, 's) t -> 'v kind
(** [kind dt] is the kind of number [dt] holds. Matching it learns the value
    type: in the [Float] arm of [match kind dt with Float -> …], values of [dt]
    are [float]s. *)

val is : 'k kind -> ('v, 's) t -> bool
(** [is k dt] is [true] iff [kind dt] is [k]. *)

(** {1:equality Equality} *)

val equal : ('v, 's) t -> ('w, 'r) t -> bool
(** [equal dt dt'] is [true] iff [dt] and [dt'] are the same dtype. *)

val equal_witness :
  ('v, 's) t -> ('w, 'r) t -> (('v, 's) t, ('w, 'r) t) Type.eq option
(** [equal_witness dt dt'] is [Some Equal] iff [equal dt dt']. *)

(** {1:values Values} *)

val zero : ('v, 's) t -> 'v
(** [zero dt] is [dt]'s additive identity: [0], and [false] for booleans. *)

val one : ('v, 's) t -> 'v
(** [one dt] is [dt]'s multiplicative identity: [1], and [true] for booleans. *)

val min_value : ('v, 's) t -> 'v
(** [min_value dt] is [dt]'s least value: [neg_infinity] for a float format with
    infinities, [-448.] for [Float8_e4m3fn] and [-6.] for [Float4_e2m1fn], and
    [false] for booleans.

    Raises [Invalid_argument] if [dt] is complex. *)

val max_value : ('v, 's) t -> 'v
(** [max_value dt] is [dt]'s greatest value: [infinity] for a float format with
    infinities, [448.] for [Float8_e4m3fn] and [6.] for [Float4_e2m1fn], and
    [true] for booleans. An unsigned format read as [int] gives its value, as
    [15], [255] and [65535]; [Uint32] and [Uint64] give the value with every bit
    set, [-1l] and [-1L].

    Raises [Invalid_argument] if [dt] is complex. *)

val pp_value : ('v, 's) t -> Format.formatter -> 'v -> unit
(** [pp_value dt] formats a value of [dt]: a float as the shortest decimal that
    a store into [dt] reads back as the same value ([nan], [inf] and [-inf] for
    the others), an unsigned integer unsigned, a complex number as [re+imi]. *)

(** {1:stores Stores} *)

val of_float : ('v, 's) t -> float -> 'v
(** [of_float dt x] is the value a store of [x] into [dt] holds. Every store of
    a [float] into a dtype follows this rule, a kernel's included:

    - A finite [x] rounds once, ties to even, to the nearest value of the format
      with its exponent range unbounded above. Below the least normal magnitude
      the result is a subnormal or a zero of [x]'s sign.
    - Past the largest finite value, a rounded result or an infinity is the
      infinity of its sign in [Float64], [Float32], [Float16] and [Bfloat16],
      and saturates in the formats of a byte or less: ±57344 in [Float8_e5m2],
      ±448 in [Float8_e4m3fn] and ±6 in [Float4_e2m1fn].
    - NaN is NaN, except in [Float4_e2m1fn], which has none: NaN stores as
      [+0.].
    - Integers truncate toward zero and saturate to their range, signed and
      unsigned alike; NaN stores as [0].
    - Complex numbers store [x] as their real part, rounded to their component's
      format, and a zero imaginary part. Booleans store [x <> 0.], [true] for
      NaN.

    So [of_float Float16 0.1] is [0x1.998p-4], the binary16 nearest to [0.1];
    [65519.] stores in [Float16] as [65504.] and [65520.] as infinity; and no
    store makes [Float8_e5m2]'s infinities, which come only from bytes already
    in a buffer. *)

(** {1:floats Float formats} *)

type float_format = {
  exponent_bits : int;  (** The width of the exponent field. *)
  fraction_bits : int;
      (** The width of the fraction field, without the implicit bit. *)
  infinities : bool;  (** Whether the format has infinities. *)
  nans : bool;  (** Whether the format has NaNs. *)
  epsilon : float;
      (** The gap between [1.] and the next value, [2{^-fraction_bits}]. *)
  min_normal : float;  (** The least positive normal value. *)
  max_finite : float;  (** The largest finite value. *)
}
(** The type for the facts of a float format. *)

val float_format : (float, 's) t -> float_format
(** [float_format dt] is the facts of [dt]'s format. *)
