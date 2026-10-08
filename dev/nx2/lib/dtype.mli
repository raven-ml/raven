(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Dtypes: the implementation of {!Nx_array.Dtype}. *)

type float64_elt = Bigarray.float64_elt
type float32_elt = Bigarray.float32_elt
type float16_elt = Bigarray.float16_elt
type bfloat16_elt = |
type float8_e4m3_elt = |
type float8_e5m2_elt = |
type float4_e2m1_elt = |
type int64_elt = Bigarray.int64_elt
type uint64_elt = |
type int32_elt = Bigarray.int32_elt
type uint32_elt = |
type int16_signed_elt = Bigarray.int16_signed_elt
type int16_unsigned_elt = Bigarray.int16_unsigned_elt
type int8_signed_elt = Bigarray.int8_signed_elt
type int8_unsigned_elt = Bigarray.int8_unsigned_elt
type int4_elt = |
type uint4_elt = |
type complex64_elt = Bigarray.complex64_elt
type complex32_elt = Bigarray.complex32_elt
type bool_elt = |
type bit_elt = |

(* The constructors' order is the codes' order, NX_DTYPES in nx_dtype.h: C reads
   a dtype as the immediate integer OCaml makes of it. *)
type ('v, 's) t =
  | Float64 : (float, float64_elt) t
  | Float32 : (float, float32_elt) t
  | Float16 : (float, float16_elt) t
  | Bfloat16 : (float, bfloat16_elt) t
  | Float8_e4m3 : (float, float8_e4m3_elt) t
  | Float8_e5m2 : (float, float8_e5m2_elt) t
  | Float4_e2m1 : (float, float4_e2m1_elt) t
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

type any = Any : ('v, 's) t -> any

val all : any list
val code : ('v, 's) t -> int
val bits : ('v, 's) t -> int
val bytes : ('v, 's) t -> int -> int
val name : ('v, 's) t -> string
val of_name : string -> any option
val pp : Format.formatter -> ('v, 's) t -> unit

type 'v kind =
  | Float : float kind
  | Complex : Complex.t kind
  | Signed : 'v kind
  | Unsigned : 'v kind
  | Boolean : bool kind

val kind : ('v, 's) t -> 'v kind
val is : 'k kind -> ('v, 's) t -> bool
val equal : ('v, 's) t -> ('w, 'r) t -> bool

val equal_witness :
  ('v, 's) t -> ('w, 'r) t -> (('v, 's) t, ('w, 'r) t) Type.eq option

val zero : ('v, 's) t -> 'v
val one : ('v, 's) t -> 'v
val min_value : ('v, 's) t -> 'v
val max_value : ('v, 's) t -> 'v
val of_float : ('v, 's) t -> float -> 'v
val pp_value : ('v, 's) t -> Format.formatter -> 'v -> unit

type float_format = {
  exponent_bits : int;
  fraction_bits : int;
  infinities : bool;
  nans : bool;
  epsilon : float;
  min_normal : float;
  max_finite : float;
}

val float_format : (float, 's) t -> float_format
