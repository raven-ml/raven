(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

(* Errors *)

let err_unsupported_ext ext = strf "unsupported image format: %s" ext
let err_bad_dims n s = strf "expected 2 or 3 dimensions, got %d (%s)" n s

module Archive = Archive

(* Result unwrapping *)

let unwrap = function Ok v -> v | Error err -> failwith (Error.to_string err)

(* Images *)

let load_image ?(grayscale = false) path = Image_io.load_image ~grayscale path

(* [with_pixels ~by img f] is [f h w c bytes] for the height, width, channel
   count and bytes of the image tensor [img], read by [by]. *)
let with_pixels ~by img f =
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
  match Nx.dtype img with
  | UInt8 -> Storage.reading ~by img (fun b -> f h w c (Storage.bytes b))
  | _ -> failwith "expected uint8 tensor"

let png_channels c =
  if c <> 1 && c <> 3 && c <> 4 then
    failwith "PNG requires one, three, or four channels"

let save_image ?(overwrite = true) path img =
  with_pixels ~by:"Nx_io.save_image" img @@ fun h w c data ->
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

(* The pHYs chunk holds a positive four-byte PNG integer, at most 2^31 - 1. *)
let pixels_per_metre dpi =
  let ppm = Float.round (dpi /. 0.0254) in
  if not (1. <= ppm && ppm <= 2147483647.) then
    invalid_arg (Printf.sprintf "Nx_io.encode_png: invalid dpi %g" dpi);
  int_of_float ppm

let encode_png ?dpi ?(srgb = false) img =
  let ppm = match dpi with None -> 0 | Some dpi -> pixels_per_metre dpi in
  with_pixels ~by:"Nx_io.encode_png" img @@ fun h w c data ->
  png_channels c;
  Image_io.encode_png data ~width:w ~height:h ~channels:c ~ppm ~srgb

(* NumPy *)

let load_npy path = Nx_npy.load_npy path |> unwrap
let save_npy ?overwrite path arr = Nx_npy.save_npy ?overwrite path arr |> unwrap
let load_npz path = Nx_npy.load_npz path |> unwrap
let load_npz_entry ~name path = Nx_npy.load_npz_entry ~name path |> unwrap

let save_npz ?overwrite path items =
  Nx_npy.save_npz ?overwrite path items |> unwrap

let gunzip ~src ~dst = Gzip_io.gunzip ~src ~dst

(* SafeTensors *)

let load_safetensors = Nx_safetensors.load_safetensors
let save_safetensors = Nx_safetensors.save_safetensors

(* GGUF *)

module Gguf = Gguf

let load_gguf = Gguf.load

(* Text *)

let save_txt ?sep ?append ?newline ?header ?footer ?comments path arr =
  Nx_txt.save ?sep ?append ?newline ?header ?footer ?comments ~out:path arr
  |> unwrap

let load_txt ?sep ?comments ?skiprows ?max_rows path dtype =
  Nx_txt.load ?sep ?comments ?skiprows ?max_rows dtype path |> unwrap
