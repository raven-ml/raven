(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

(* Errors *)

let err_unsupported_ext ext = strf "unsupported image format: %s" ext
let err_bad_dims n s = strf "expected 2 or 3 dimensions, got %d (%s)" n s

(* Packed tensors *)

type packed = Packed_nx.t = P : ('a, 'b) Nx.t -> packed
type archive = (string, packed) Hashtbl.t
type packed_dtype = Dtype : ('a, 'b) Nx.dtype -> packed_dtype

let to_typed dtype packed = Packed_nx.to_typed dtype packed
let packed_dtype (P nx) = Dtype (Nx.dtype nx)
let packed_shape (P nx) = Nx.shape nx

(* Result unwrapping *)

let unwrap = function Ok v -> v | Error err -> failwith (Error.to_string err)

(* Images *)

let load_image ?(grayscale = false) path = Image_io.load_image ~grayscale path

(* [uint8_pixels img] is the height, width, channel count and bytes of the image
   tensor [img]. *)
let uint8_pixels img =
  let h, w, c =
    match Nx.shape img with
    | [| h; w |] -> (h, w, 1)
    | [| h; w; c |] -> (h, w, c)
    | s ->
        let dims =
          Array.to_list s |> List.map string_of_int |> String.concat "x"
        in
        failwith (err_bad_dims (Array.length s) dims)
  in
  let buf = Nx.to_buffer img in
  match Nx_buffer.kind buf with
  | UInt8 -> (h, w, c, Nx_buffer.to_bigarray1 buf)
  | _ -> failwith "expected uint8 tensor"

let png_channels c =
  if c <> 1 && c <> 3 && c <> 4 then
    failwith "PNG requires one, three, or four channels"

let save_image ?(overwrite = true) path img =
  let h, w, c, data = uint8_pixels img in
  let ext = String.lowercase_ascii (Filename.extension path) in
  match ext with
  | ".png" ->
      png_channels c;
      Image_io.save_png ~overwrite path data ~width:w ~height:h ~channels:c
  | ".jpg" | ".jpeg" ->
      if c <> 1 && c <> 3 then
        failwith "save_image: JPEG requires one or three channels";
      Image_io.save_jpeg ~overwrite path data ~width:w ~height:h ~channels:c
  | _ -> failwith (err_unsupported_ext ext)

let encode_png img =
  let h, w, c, data = uint8_pixels img in
  png_channels c;
  Image_io.encode_png data ~width:w ~height:h ~channels:c

(* NumPy *)

let load_npy path = Nx_npy.load_npy path |> unwrap
let save_npy ?overwrite path arr = Nx_npy.save_npy ?overwrite path arr |> unwrap
let load_npz path = Nx_npy.load_npz path |> unwrap
let load_npz_entry ~name path = Nx_npy.load_npz_entry ~name path |> unwrap

let save_npz ?overwrite path items =
  Nx_npy.save_npz ?overwrite path items |> unwrap

let gunzip ~src ~dst = Gzip_io.gunzip ~src ~dst

(* zlib streams *)

let bigarray_of_string s =
  let n = String.length s in
  let b = Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout n in
  for i = 0 to n - 1 do
    Bigarray.Array1.unsafe_set b i (Char.code (String.unsafe_get s i))
  done;
  b

let string_of_bigarray b =
  String.init (Bigarray.Array1.dim b) (fun i ->
      Char.chr (Bigarray.Array1.unsafe_get b i))

let deflate s =
  let src = bigarray_of_string s in
  let len = String.length s in
  let raw = Nx_io_codec.deflate_raw ~prefix:"" src ~off:0 ~len in
  let adler = Nx_io_codec.adler32 src ~off:0 ~len in
  let n = Bigarray.Array1.dim raw in
  let out = Bytes.create (n + 6) in
  Bytes.set out 0 '\x78';
  Bytes.set out 1 '\x01';
  for i = 0 to n - 1 do
    Bytes.unsafe_set out (i + 2) (Char.chr (Bigarray.Array1.unsafe_get raw i))
  done;
  Bytes.set_int32_be out (n + 2) adler;
  Bytes.unsafe_to_string out

let inflate s =
  let n = String.length s in
  if
    n < 6
    || Char.code s.[0] land 0x0f <> 8
    || ((Char.code s.[0] * 256) + Char.code s.[1]) mod 31 <> 0
  then failwith "inflate: not a zlib stream";
  let src = bigarray_of_string s in
  (* The output size is unknown: grow the bound until the stream ends before
     reaching it. *)
  let rec go max_output =
    let out =
      Nx_io_codec.inflate_raw_prefix src ~off:2 ~len:(n - 6) ~max_output
    in
    if Bigarray.Array1.dim out >= max_output then go (2 * max_output) else out
  in
  let out = go (Int.max 1024 (8 * n)) in
  let expected = String.get_int32_be s (n - 4) in
  let actual = Nx_io_codec.adler32 out ~off:0 ~len:(Bigarray.Array1.dim out) in
  if expected <> actual then failwith "inflate: checksum mismatch";
  string_of_bigarray out

(* SafeTensors *)

let load_safetensors path = Nx_safetensors.load_safetensors path |> unwrap

let save_safetensors ?overwrite path items =
  Nx_safetensors.save_safetensors ?overwrite path items |> unwrap

(* Text *)

let save_txt ?sep ?append ?newline ?header ?footer ?comments path arr =
  Nx_txt.save ?sep ?append ?newline ?header ?footer ?comments ~out:path arr
  |> unwrap

let load_txt ?sep ?comments ?skiprows ?max_rows path dtype =
  Nx_txt.load ?sep ?comments ?skiprows ?max_rows dtype path |> unwrap
