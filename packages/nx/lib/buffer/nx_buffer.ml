(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

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
    (a, b) Nx_dtype.t ->
    c Bigarray.layout ->
    int array ->
    (a, b, c) Bigarray.Genarray.t =
 fun kind layout dims ->
  match kind with
  | BFloat16 -> create_bfloat16_genarray layout dims
  | Bool -> create_bool_genarray layout dims
  | Int4 -> create_int4_signed_genarray layout dims
  | UInt4 -> create_int4_unsigned_genarray layout dims
  | Float8_e4m3 -> create_float8_e4m3_genarray layout dims
  | Float8_e5m2 -> create_float8_e5m2_genarray layout dims
  | UInt32 -> create_uint32_genarray layout dims
  | UInt64 -> create_uint64_genarray layout dims
  | _ -> (
      match Nx_dtype.to_bigarray_kind kind with
      | Some k -> Bigarray.Genarray.create k layout dims
      | None -> assert false)

(* Genarray externals *)

external genarray_get : ('a, 'b, 'c) Bigarray.Genarray.t -> int array -> 'a
  = "caml_nx_buffer_get"

external genarray_set :
  ('a, 'b, 'c) Bigarray.Genarray.t -> int array -> 'a -> unit
  = "caml_nx_buffer_set"

(* Not [@@noalloc]: the stub raises on kinds buffers do not support (char, int,
   nativeint). *)
external genarray_dtype :
  ('a, 'b, 'c) Bigarray.Genarray.t -> ('a, 'b) Nx_dtype.t
  = "caml_nx_buffer_kind"

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

let dtype buf = genarray_dtype (Bigarray.genarray_of_array1 buf)
let length buf = Bigarray.Array1.dim buf

(* Element access *)

let get buf i = genarray_get (Bigarray.genarray_of_array1 buf) [| i |]
let set buf i v = genarray_set (Bigarray.genarray_of_array1 buf) [| i |] v

external unsafe_get : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> int -> 'a
  = "caml_nx_buffer_unsafe_get"

external unsafe_set :
  ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> int -> 'a -> unit
  = "caml_nx_buffer_unsafe_set"

external unsafe_data_ptr :
  ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> nativeint
  = "caml_nx_buffer_data_ptr"

let is_int4 : type a b. (a, b) Nx_dtype.t -> bool =
 fun k -> match k with Int4 | UInt4 -> true | _ -> false

(* Byte count for a span of elements, accounting for int4 packing *)
let elts_to_bytes k n =
  if is_int4 k then (n + 1) / 2 else n * Nx_dtype.itemsize k

(* Byte offset of element [off]; int4 offsets must be even (checked by the
   callers) so the element starts a byte. *)
let elt_off_to_bytes k off =
  if is_int4 k then off / 2 else off * Nx_dtype.itemsize k

(* Reinterpretation *)

external unsafe_reinterpret :
  ('a, 'b) Nx_dtype.t -> ('c, 'd) t -> int -> int -> ('a, 'b) t
  = "caml_nx_buffer_reinterpret"

let reinterpret k buf =
  if is_int4 k || is_int4 (dtype buf) then
    invalid_arg "Nx_buffer.reinterpret: int4 and uint4 pack two elements a byte";
  let bytes = length buf * Nx_dtype.itemsize (dtype buf) in
  let size = Nx_dtype.itemsize k in
  if bytes mod size <> 0 then
    invalid_arg
      (Printf.sprintf
         "Nx_buffer.reinterpret: %d bytes is not a multiple of the %d-byte %s \
          element"
         bytes size (Nx_dtype.to_string k));
  unsafe_reinterpret k buf (bytes / size) size

(* Mapped files *)

type file = { path : string; size : int; mtime : float; inode : int }

external unsafe_register_file :
  ('a, 'b, 'c) Bigarray.Genarray.t -> string -> int -> float -> int -> unit
  = "caml_nx_buffer_register_file"

external unsafe_file_range :
  ('a, 'b, 'c) Bigarray.Genarray.t -> (string * int * float * int * int) option
  = "caml_nx_buffer_file_range"

let register_file file buf =
  unsafe_register_file
    (Bigarray.genarray_of_array1 buf)
    file.path file.size file.mtime file.inode

let file_range buf =
  match unsafe_file_range (Bigarray.genarray_of_array1 buf) with
  | None -> None
  | Some (path, size, mtime, inode, offset) ->
      Some ({ path; size; mtime; inode }, offset)

(* Bulk operations *)

let fill buf v = genarray_fill_ext (Bigarray.genarray_of_array1 buf) v

let blit ~src ~dst =
  genarray_blit_ext
    (Bigarray.genarray_of_array1 src)
    (Bigarray.genarray_of_array1 dst)

let blit_from_bytes ?(src_off = 0) ?(dst_off = 0) ?len bytes buf =
  let k = dtype buf in
  let buf_len = length buf in
  let len = match len with Some l -> l | None -> buf_len - dst_off in
  if src_off < 0 then invalid_arg "blit_from_bytes: negative src_off";
  if dst_off < 0 then invalid_arg "blit_from_bytes: negative dst_off";
  if len < 0 then invalid_arg "blit_from_bytes: negative length";
  if dst_off + len > buf_len then
    invalid_arg "blit_from_bytes: dst_off + len > buffer length";
  (* The copy moves whole bytes, and int4 packs two elements per byte: offsets
     must start a byte, and an odd length is only safe when the trailing nibble
     written is the buffer's padding. *)
  if is_int4 k then (
    if src_off land 1 <> 0 || dst_off land 1 <> 0 then
      invalid_arg "blit_from_bytes: int4 offsets must be even";
    if len land 1 <> 0 && dst_off + len <> buf_len then
      invalid_arg
        "blit_from_bytes: odd int4 length must reach the end of the buffer");
  let byte_len = elts_to_bytes k len in
  let src_byte_off = elt_off_to_bytes k src_off in
  if src_byte_off + byte_len > Bytes.length bytes then
    invalid_arg "blit_from_bytes: src_off + len > bytes length";
  let dst_byte_off = elt_off_to_bytes k dst_off in
  unsafe_blit_from_bytes bytes src_byte_off
    (Bigarray.genarray_of_array1 buf)
    dst_byte_off byte_len

let blit_to_bytes ?(src_off = 0) ?(dst_off = 0) ?len buf bytes =
  let k = dtype buf in
  let buf_len = length buf in
  let len = match len with Some l -> l | None -> buf_len - src_off in
  if src_off < 0 then invalid_arg "blit_to_bytes: negative src_off";
  if dst_off < 0 then invalid_arg "blit_to_bytes: negative dst_off";
  if len < 0 then invalid_arg "blit_to_bytes: negative length";
  if src_off + len > buf_len then
    invalid_arg "blit_to_bytes: src_off + len > buffer length";
  if is_int4 k && (src_off land 1 <> 0 || dst_off land 1 <> 0) then
    invalid_arg "blit_to_bytes: int4 offsets must be even";
  let byte_len = elts_to_bytes k len in
  let dst_byte_off = elt_off_to_bytes k dst_off in
  if dst_byte_off + byte_len > Bytes.length bytes then
    invalid_arg "blit_to_bytes: dst_off + len > bytes length";
  let src_byte_off = elt_off_to_bytes k src_off in
  unsafe_blit_to_bytes
    (Bigarray.genarray_of_array1 buf)
    src_byte_off bytes dst_byte_off byte_len

(* Bigarray conversions *)

(* Kinds whose values buffers cannot represent. *)
let unsupported_stdlib_kind : type a b. (a, b) Bigarray.kind -> string option =
  function
  | Bigarray.Char -> Some "char"
  | Bigarray.Int -> Some "int"
  | Bigarray.Nativeint -> Some "nativeint"
  | _ -> None

let of_bigarray1 buf =
  (match unsupported_stdlib_kind (Bigarray.Array1.kind buf) with
  | Some name -> invalid_arg ("Nx_buffer.of_bigarray1: unsupported kind " ^ name)
  | None -> ());
  buf

let to_bigarray1 buf =
  match Nx_dtype.to_bigarray_kind (dtype buf) with
  | Some _ -> buf
  | None ->
      invalid_arg
        ("Nx_buffer.to_bigarray1: no bigarray kind for "
        ^ Nx_dtype.to_string (dtype buf))

let to_genarray buf shape =
  Bigarray.reshape (Bigarray.genarray_of_array1 buf) shape

let of_genarray ga =
  (match unsupported_stdlib_kind (Bigarray.Genarray.kind ga) with
  | Some name -> invalid_arg ("Nx_buffer.of_genarray: unsupported kind " ^ name)
  | None -> ());
  let size = Array.fold_left ( * ) 1 (Bigarray.Genarray.dims ga) in
  Bigarray.array1_of_genarray (Bigarray.reshape ga [| size |])

(* Genarray utilities *)

let genarray_dims ga = Bigarray.Genarray.dims ga

let genarray_blit : type a b c.
    (a, b, c) Bigarray.Genarray.t -> (a, b, c) Bigarray.Genarray.t -> unit =
 fun src dst -> genarray_blit_ext src dst

let genarray_change_layout = Bigarray.Genarray.change_layout
