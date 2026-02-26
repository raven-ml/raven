(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Flat buffers for tensor storage.

    Flat, C-layout, one-dimensional buffers with support for both standard
    Bigarray element types and extended types (bfloat16, bool, int4, float8,
    uint32, uint64).

    The buffer type {!t} is abstract in this interface. Conversions to and from
    {!Bigarray} are explicit via {!of_bigarray1}, {!to_bigarray1},
    {!of_genarray}, and {!to_genarray}. *)

(** {1:elt Element types}

    Standard element types are aliases from {!Bigarray}. Extended types are
    defined here. *)

type float16_elt = Bigarray.float16_elt
type float32_elt = Bigarray.float32_elt
type float64_elt = Bigarray.float64_elt
type int8_signed_elt = Bigarray.int8_signed_elt
type int8_unsigned_elt = Bigarray.int8_unsigned_elt
type int16_signed_elt = Bigarray.int16_signed_elt
type int16_unsigned_elt = Bigarray.int16_unsigned_elt
type int32_elt = Bigarray.int32_elt
type int64_elt = Bigarray.int64_elt
type complex32_elt = Bigarray.complex32_elt
type complex64_elt = Bigarray.complex64_elt

type bfloat16_elt
(** Brain floating-point 16-bit. *)

type bool_elt
(** Boolean stored as a byte. *)

type int4_signed_elt
(** Signed 4-bit integer (two values packed per byte). *)

type int4_unsigned_elt
(** Unsigned 4-bit integer (two values packed per byte). *)

type float8_e4m3_elt
(** 8-bit float with 4 exponent and 3 mantissa bits. *)

type float8_e5m2_elt
(** 8-bit float with 5 exponent and 2 mantissa bits. *)

type uint32_elt
(** Unsigned 32-bit integer. *)

type uint64_elt
(** Unsigned 64-bit integer. *)

(** {1:kind Kind GADT} *)

type ('a, 'b) kind =
  | Float16 : (float, float16_elt) kind
  | Float32 : (float, float32_elt) kind
  | Float64 : (float, float64_elt) kind
  | Bfloat16 : (float, bfloat16_elt) kind
  | Float8_e4m3 : (float, float8_e4m3_elt) kind
  | Float8_e5m2 : (float, float8_e5m2_elt) kind
  | Int8_signed : (int, int8_signed_elt) kind
  | Int8_unsigned : (int, int8_unsigned_elt) kind
  | Int16_signed : (int, int16_signed_elt) kind
  | Int16_unsigned : (int, int16_unsigned_elt) kind
  | Int32 : (int32, int32_elt) kind
  | Uint32 : (int32, uint32_elt) kind
  | Int64 : (int64, int64_elt) kind
  | Uint64 : (int64, uint64_elt) kind
  | Int4_signed : (int, int4_signed_elt) kind
  | Int4_unsigned : (int, int4_unsigned_elt) kind
  | Complex32 : (Complex.t, complex32_elt) kind
  | Complex64 : (Complex.t, complex64_elt) kind
  | Bool : (bool, bool_elt) kind
      (** The type for element kinds. Nineteen constructors covering standard
          Bigarray kinds and extended types. *)

(** {2:kind_values Kind values} *)

val float16 : (float, float16_elt) kind
val float32 : (float, float32_elt) kind
val float64 : (float, float64_elt) kind
val bfloat16 : (float, bfloat16_elt) kind
val float8_e4m3 : (float, float8_e4m3_elt) kind
val float8_e5m2 : (float, float8_e5m2_elt) kind
val int8_signed : (int, int8_signed_elt) kind
val int8_unsigned : (int, int8_unsigned_elt) kind
val int16_signed : (int, int16_signed_elt) kind
val int16_unsigned : (int, int16_unsigned_elt) kind
val int32 : (int32, int32_elt) kind
val uint32 : (int32, uint32_elt) kind
val int64 : (int64, int64_elt) kind
val uint64 : (int64, uint64_elt) kind
val int4_signed : (int, int4_signed_elt) kind
val int4_unsigned : (int, int4_unsigned_elt) kind
val complex32 : (Complex.t, complex32_elt) kind
val complex64 : (Complex.t, complex64_elt) kind
val bool : (bool, bool_elt) kind

(** {2:kind_props Kind properties} *)

val kind_size_in_bytes : ('a, 'b) kind -> int
(** [kind_size_in_bytes k] is the storage size in bytes per element for kind
    [k]. For [Int4_signed] and [Int4_unsigned] this is [1] (two values packed
    per byte). *)

val to_stdlib_kind : ('a, 'b) kind -> ('a, 'b) Bigarray.kind option
(** [to_stdlib_kind k] is the standard {!Bigarray.kind} for [k], or [None] for
    extended types. *)

(** {1:buf Buffer type and operations} *)

type ('a, 'b) t
(** [('a, 'b) t] is a flat, C-layout, one-dimensional buffer. *)

(** {2:create Creation} *)

val create : ('a, 'b) kind -> int -> ('a, 'b) t
(** [create kind n] allocates a zero-initialized buffer of [n] elements. *)

(** {2:props Properties} *)

val kind : ('a, 'b) t -> ('a, 'b) kind
(** [kind buf] is the element kind of [buf]. *)

val length : ('a, 'b) t -> int
(** [length buf] is the number of elements in [buf]. *)

(** {2:access Element access} *)

val get : ('a, 'b) t -> int -> 'a
(** [get buf i] is the element at index [i].

    Raises [Invalid_argument] if [i] is out of bounds. *)

val set : ('a, 'b) t -> int -> 'a -> unit
(** [set buf i v] sets the element at index [i] to [v].

    Raises [Invalid_argument] if [i] is out of bounds. *)

val unsafe_get : ('a, 'b) t -> int -> 'a
(** [unsafe_get buf i] is like {!get} without bounds checking. *)

val unsafe_set : ('a, 'b) t -> int -> 'a -> unit
(** [unsafe_set buf i v] is like {!set} without bounds checking. *)

(** {2:bulk Bulk operations} *)

val fill : ('a, 'b) t -> 'a -> unit
(** [fill buf v] sets every element of [buf] to [v]. *)

val blit : src:('a, 'b) t -> dst:('a, 'b) t -> unit
(** [blit ~src ~dst] copies all elements from [src] to [dst].

    Raises [Invalid_argument] if dimensions differ. *)

val blit_from_bytes :
  ?src_off:int -> ?dst_off:int -> ?len:int -> bytes -> ('a, 'b) t -> unit
(** [blit_from_bytes ?src_off ?dst_off ?len bytes buf] copies [len] elements
    from [bytes] into [buf]. Offsets and length are in elements. [src_off] and
    [dst_off] default to [0]. [len] defaults to [length buf - dst_off]. *)

val blit_to_bytes :
  ?src_off:int -> ?dst_off:int -> ?len:int -> ('a, 'b) t -> bytes -> unit
(** [blit_to_bytes ?src_off ?dst_off ?len buf bytes] copies [len] elements from
    [buf] into [bytes]. Offsets and length are in elements. [src_off] and
    [dst_off] default to [0]. [len] defaults to [length buf - src_off]. *)

(** {1:ba Bigarray conversions} *)

val of_bigarray1 : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> ('a, 'b) t
(** [of_bigarray1 ba] is [ba] viewed as a buffer. Zero-copy for standard kinds.
*)

val to_bigarray1 : ('a, 'b) t -> ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t
(** [to_bigarray1 buf] is [buf] viewed as a one-dimensional bigarray. Zero-copy.
*)

val to_genarray :
  ('a, 'b) t -> int array -> ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t
(** [to_genarray buf shape] reshapes [buf] into a genarray with [shape]. The
    product of [shape] must equal [length buf]. *)

val of_genarray : ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t -> ('a, 'b) t
(** [of_genarray ga] flattens [ga] into a one-dimensional buffer. *)

(** {1:ga Genarray utilities}

    Operations on {!Bigarray.Genarray.t} that handle extended kinds. Used by I/O
    modules (npy, safetensors, images). *)

val genarray_create :
  ('a, 'b) kind ->
  'c Bigarray.layout ->
  int array ->
  ('a, 'b, 'c) Bigarray.Genarray.t
(** [genarray_create kind layout dims] allocates a genarray. Handles both
    standard and extended kinds. *)

val genarray_kind : ('a, 'b, 'c) Bigarray.Genarray.t -> ('a, 'b) kind
(** [genarray_kind ga] is the kind of [ga], including extended kinds. *)

val genarray_dims : ('a, 'b, 'c) Bigarray.Genarray.t -> int array
(** [genarray_dims ga] is the dimensions of [ga]. *)

val genarray_blit :
  ('a, 'b, 'c) Bigarray.Genarray.t -> ('a, 'b, 'c) Bigarray.Genarray.t -> unit
(** [genarray_blit src dst] copies [src] to [dst]. Handles extended kinds. *)

val genarray_change_layout :
  ('a, 'b, 'c) Bigarray.Genarray.t ->
  'd Bigarray.layout ->
  ('a, 'b, 'd) Bigarray.Genarray.t
(** [genarray_change_layout ga layout] changes the layout of [ga]. *)
