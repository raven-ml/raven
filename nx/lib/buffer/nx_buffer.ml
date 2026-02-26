(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Element types *)

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
type bfloat16_elt = |
type bool_elt = |
type int4_signed_elt = |
type int4_unsigned_elt = |
type float8_e4m3_elt = |
type float8_e5m2_elt = |
type uint32_elt = |
type uint64_elt = |

(* Kind GADT *)

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

(* Kind values *)

let float16 = Float16
let float32 = Float32
let float64 = Float64
let bfloat16 = Bfloat16
let float8_e4m3 = Float8_e4m3
let float8_e5m2 = Float8_e5m2
let int8_signed = Int8_signed
let int8_unsigned = Int8_unsigned
let int16_signed = Int16_signed
let int16_unsigned = Int16_unsigned
let int32 = Int32
let uint32 = Uint32
let int64 = Int64
let uint64 = Uint64
let int4_signed = Int4_signed
let int4_unsigned = Int4_unsigned
let complex32 = Complex32
let complex64 = Complex64
let bool = Bool

(* Kind properties *)

let kind_size_in_bytes : type a b. (a, b) kind -> int = function
  | Float16 -> 2
  | Float32 -> 4
  | Float64 -> 8
  | Bfloat16 -> 2
  | Float8_e4m3 -> 1
  | Float8_e5m2 -> 1
  | Int8_signed -> 1
  | Int8_unsigned -> 1
  | Int16_signed -> 2
  | Int16_unsigned -> 2
  | Int32 -> 4
  | Uint32 -> 4
  | Int64 -> 8
  | Uint64 -> 8
  | Int4_signed -> 1
  | Int4_unsigned -> 1
  | Complex32 -> 8
  | Complex64 -> 16
  | Bool -> 1

let to_stdlib_kind : type a b. (a, b) kind -> (a, b) Bigarray.kind option =
  function
  | Float16 -> Some Bigarray.Float16
  | Float32 -> Some Bigarray.Float32
  | Float64 -> Some Bigarray.Float64
  | Int8_signed -> Some Bigarray.Int8_signed
  | Int8_unsigned -> Some Bigarray.Int8_unsigned
  | Int16_signed -> Some Bigarray.Int16_signed
  | Int16_unsigned -> Some Bigarray.Int16_unsigned
  | Int32 -> Some Bigarray.Int32
  | Int64 -> Some Bigarray.Int64
  | Complex32 -> Some Bigarray.Complex32
  | Complex64 -> Some Bigarray.Complex64
  | Bfloat16 -> None
  | Bool -> None
  | Int4_signed -> None
  | Int4_unsigned -> None
  | Float8_e4m3 -> None
  | Float8_e5m2 -> None
  | Uint32 -> None
  | Uint64 -> None

(* Buffer type *)

type ('a, 'b) t = ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t

(* Genarray externals *)

external create_bfloat16_genarray :
  'c Bigarray.layout -> int array -> ('a, 'b, 'c) Bigarray.Genarray.t
  = "caml_nx_buffer_create_bfloat16"

external create_bool_genarray :
  'c Bigarray.layout -> int array -> ('a, 'b, 'c) Bigarray.Genarray.t
  = "caml_nx_buffer_create_bool"

external create_int4_signed_genarray :
  'c Bigarray.layout -> int array -> ('a, 'b, 'c) Bigarray.Genarray.t
  = "caml_nx_buffer_create_int4_signed"

external create_int4_unsigned_genarray :
  'c Bigarray.layout -> int array -> ('a, 'b, 'c) Bigarray.Genarray.t
  = "caml_nx_buffer_create_int4_unsigned"

external create_float8_e4m3_genarray :
  'c Bigarray.layout -> int array -> ('a, 'b, 'c) Bigarray.Genarray.t
  = "caml_nx_buffer_create_float8_e4m3"

external create_float8_e5m2_genarray :
  'c Bigarray.layout -> int array -> ('a, 'b, 'c) Bigarray.Genarray.t
  = "caml_nx_buffer_create_float8_e5m2"

external create_uint32_genarray :
  'c Bigarray.layout -> int array -> ('a, 'b, 'c) Bigarray.Genarray.t
  = "caml_nx_buffer_create_uint32"

external create_uint64_genarray :
  'c Bigarray.layout -> int array -> ('a, 'b, 'c) Bigarray.Genarray.t
  = "caml_nx_buffer_create_uint64"

(* Extended-kind genarray creation *)

let genarray_create : type a b c.
    (a, b) kind ->
    c Bigarray.layout ->
    int array ->
    (a, b, c) Bigarray.Genarray.t =
 fun kind layout dims ->
  match kind with
  | Bfloat16 -> create_bfloat16_genarray layout dims
  | Bool -> create_bool_genarray layout dims
  | Int4_signed -> create_int4_signed_genarray layout dims
  | Int4_unsigned -> create_int4_unsigned_genarray layout dims
  | Float8_e4m3 -> create_float8_e4m3_genarray layout dims
  | Float8_e5m2 -> create_float8_e5m2_genarray layout dims
  | Uint32 -> create_uint32_genarray layout dims
  | Uint64 -> create_uint64_genarray layout dims
  | _ -> (
      match to_stdlib_kind kind with
      | Some k -> Bigarray.Genarray.create k layout dims
      | None -> assert false)

(* Genarray externals *)

external genarray_get : ('a, 'b, 'c) Bigarray.Genarray.t -> int array -> 'a
  = "caml_nx_buffer_get"

external genarray_set :
  ('a, 'b, 'c) Bigarray.Genarray.t -> int array -> 'a -> unit
  = "caml_nx_buffer_set"

external genarray_kind_ext : ('a, 'b, 'c) Bigarray.Genarray.t -> ('a, 'b) kind
  = "caml_nx_buffer_kind"
[@@noalloc]

external genarray_blit_ext :
  ('a, 'b, 'c) Bigarray.Genarray.t -> ('a, 'b, 'c) Bigarray.Genarray.t -> unit
  = "caml_nx_buffer_blit"

external genarray_fill_ext : ('a, 'b, 'c) Bigarray.Genarray.t -> 'a -> unit
  = "caml_nx_buffer_fill"

external unsafe_blit_from_bytes :
  bytes -> int -> ('a, 'b, 'c) Bigarray.Genarray.t -> int -> int -> unit
  = "caml_nx_buffer_blit_from_bytes"
[@@noalloc]

external unsafe_blit_to_bytes :
  ('a, 'b, 'c) Bigarray.Genarray.t -> int -> bytes -> int -> int -> unit
  = "caml_nx_buffer_blit_to_bytes"
[@@noalloc]

(* Buffer creation *)

let create kind n =
  Bigarray.reshape_1 (genarray_create kind Bigarray.c_layout [| n |]) n

(* Buffer properties *)

let kind buf = genarray_kind_ext (Bigarray.genarray_of_array1 buf)
let length buf = Bigarray.Array1.dim buf

(* Element access *)

let get buf i = genarray_get (Bigarray.genarray_of_array1 buf) [| i |]
let set buf i v = genarray_set (Bigarray.genarray_of_array1 buf) [| i |] v

external unsafe_get : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> int -> 'a
  = "caml_nx_buffer_unsafe_get"

external unsafe_set :
  ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> int -> 'a -> unit
  = "caml_nx_buffer_unsafe_set"

(* Byte count for a span of elements, accounting for int4 packing *)
let elts_to_bytes : type a b. (a, b) kind -> int -> int =
 fun k n ->
  match k with
  | Int4_signed -> (n + 1) / 2
  | Int4_unsigned -> (n + 1) / 2
  | _ -> n * kind_size_in_bytes k

(* Bulk operations *)

let fill buf v = genarray_fill_ext (Bigarray.genarray_of_array1 buf) v

let blit ~src ~dst =
  genarray_blit_ext
    (Bigarray.genarray_of_array1 src)
    (Bigarray.genarray_of_array1 dst)

let blit_from_bytes ?(src_off = 0) ?(dst_off = 0) ?len bytes buf =
  let k = kind buf in
  let buf_len = length buf in
  let len = match len with Some l -> l | None -> buf_len - dst_off in
  if src_off < 0 then invalid_arg "blit_from_bytes: negative src_off";
  if dst_off < 0 then invalid_arg "blit_from_bytes: negative dst_off";
  if len < 0 then invalid_arg "blit_from_bytes: negative length";
  if dst_off + len > buf_len then
    invalid_arg "blit_from_bytes: dst_off + len > buffer length";
  let byte_len = elts_to_bytes k len in
  let src_byte_off = src_off * kind_size_in_bytes k in
  if src_byte_off + byte_len > Bytes.length bytes then
    invalid_arg "blit_from_bytes: src_off + len > bytes length";
  let dst_byte_off = elts_to_bytes k dst_off in
  unsafe_blit_from_bytes bytes src_byte_off
    (Bigarray.genarray_of_array1 buf)
    dst_byte_off byte_len

let blit_to_bytes ?(src_off = 0) ?(dst_off = 0) ?len buf bytes =
  let k = kind buf in
  let buf_len = length buf in
  let len = match len with Some l -> l | None -> buf_len - src_off in
  if src_off < 0 then invalid_arg "blit_to_bytes: negative src_off";
  if dst_off < 0 then invalid_arg "blit_to_bytes: negative dst_off";
  if len < 0 then invalid_arg "blit_to_bytes: negative length";
  if src_off + len > buf_len then
    invalid_arg "blit_to_bytes: src_off + len > buffer length";
  let byte_len = elts_to_bytes k len in
  let dst_byte_off = dst_off * kind_size_in_bytes k in
  if dst_byte_off + byte_len > Bytes.length bytes then
    invalid_arg "blit_to_bytes: dst_off + len > bytes length";
  let src_byte_off = elts_to_bytes k src_off in
  unsafe_blit_to_bytes
    (Bigarray.genarray_of_array1 buf)
    src_byte_off bytes dst_byte_off byte_len

(* Bigarray conversions *)

let of_bigarray1 buf = buf
let to_bigarray1 buf = buf

let to_genarray buf shape =
  Bigarray.reshape (Bigarray.genarray_of_array1 buf) shape

let of_genarray ga =
  let size = Array.fold_left ( * ) 1 (Bigarray.Genarray.dims ga) in
  Bigarray.array1_of_genarray (Bigarray.reshape ga [| size |])

(* Genarray utilities *)

let genarray_kind : type a b c. (a, b, c) Bigarray.Genarray.t -> (a, b) kind =
 fun ga -> genarray_kind_ext ga

let genarray_dims ga = Bigarray.Genarray.dims ga

let genarray_blit : type a b c.
    (a, b, c) Bigarray.Genarray.t -> (a, b, c) Bigarray.Genarray.t -> unit =
 fun src dst -> genarray_blit_ext src dst

let genarray_change_layout = Bigarray.Genarray.change_layout
