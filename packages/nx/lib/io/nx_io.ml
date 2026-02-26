(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

(* Errors *)

let err_file_exists path = strf "file already exists: %s" path
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

let load_image ?(grayscale = false) path =
  let channels = if grayscale then 1 else 3 in
  match Stb_image.load ~channels path with
  | Error (`Msg msg) -> failwith msg
  | Ok img ->
      let h = Stb_image.height img in
      let w = Stb_image.width img in
      let c = Stb_image.channels img in
      let buf = Nx_buffer.of_bigarray1 (Stb_image.data img) in
      let n = Nx_buffer.length buf in
      let t = Nx.of_buffer buf ~shape:[| n |] in
      let shape = if c = 1 then [| h; w |] else [| h; w; c |] in
      Nx.reshape shape t

let save_image ?(overwrite = true) path img =
  if (not overwrite) && Sys.file_exists path then
    failwith (err_file_exists path);
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
  let data =
    match Nx_buffer.kind buf with
    | Int8_unsigned -> Nx_buffer.to_bigarray1 buf
    | _ -> failwith "save_image: expected uint8 tensor"
  in
  let ext = String.lowercase_ascii (Filename.extension path) in
  match ext with
  | ".png" -> Stb_image_write.png path ~w ~h ~c data
  | ".bmp" -> Stb_image_write.bmp path ~w ~h ~c data
  | ".tga" -> Stb_image_write.tga path ~w ~h ~c data
  | ".jpg" | ".jpeg" -> Stb_image_write.jpg path ~w ~h ~c ~quality:90 data
  | _ -> failwith (err_unsupported_ext ext)

(* NumPy *)

let load_npy path = Nx_npy.load_npy path |> unwrap
let save_npy ?overwrite path arr = Nx_npy.save_npy ?overwrite path arr |> unwrap
let load_npz path = Nx_npy.load_npz path |> unwrap
let load_npz_entry ~name path = Nx_npy.load_npz_entry ~name path |> unwrap

let save_npz ?overwrite path items =
  Nx_npy.save_npz ?overwrite path items |> unwrap

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
