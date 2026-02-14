(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx_buffer
open Error
module Cache_dir = Cache_dir
module Http = Http

(* ───── Type Definitions ───── *)

type packed_nx = Packed_nx.t = P : ('a, 'b) Nx.t -> packed_nx
type archive = (string, packed_nx) Hashtbl.t

module Safe = struct
  type error = Error.t =
    | Io_error of string
    | Format_error of string
    | Unsupported_dtype
    | Unsupported_shape
    | Missing_entry of string
    | Other of string

  (* Image dimensions *)

  type nx_dims = [ `Gray of int * int | `Color of int * int * int ]

  let get_nx_dims arr : nx_dims =
    match Nx.shape arr with
    | [| h; w |] -> `Gray (h, w)
    | [| h; w; c |] -> `Color (h, w, c)
    | s ->
        fail_msg "Invalid nx dimensions: expected 2 or 3, got %d (%s)"
          (Array.length s)
          (Array.to_list s |> List.map string_of_int |> String.concat "x")

  let load_image ?grayscale path =
    let grayscale = Option.value grayscale ~default:false in
    try
      let desired_channels = if grayscale then 1 else 3 in
      match Stb_image.load ~channels:desired_channels path with
      | Ok img ->
          let h = Stb_image.height img in
          let w = Stb_image.width img in
          let c = Stb_image.channels img in
          let buffer = Stb_image.data img in
          let nd = Nx.of_buffer (genarray_of_array1 buffer) in
          let shape = if c = 1 then [| h; w |] else [| h; w; c |] in
          Ok (Nx.reshape shape nd)
      | Error (`Msg msg) -> Error (Format_error msg)
    with
    | Sys_error msg -> Error (Io_error msg)
    | ex -> Error (Other (Printexc.to_string ex))

  let save_image ?(overwrite = true) path img =
    try
      (* Check if file exists and overwrite is false *)
      if (not overwrite) && Sys.file_exists path then
        Error (Io_error (Printf.sprintf "File '%s' already exists" path))
      else
        let h, w, c =
          match get_nx_dims img with
          | `Gray (h, w) -> (h, w, 1)
          | `Color (h, w, c) -> (h, w, c)
        in
        (* Ensure the input array is uint8 *)
        let data_gen = Nx.to_buffer img in
        let data =
          match Genarray.kind data_gen with
          | Int8_unsigned -> reshape_1 data_gen (h * w * c)
        in
        let extension = Filename.extension path |> String.lowercase_ascii in
        match extension with
        | ".png" ->
            Stb_image_write.png path ~w ~h ~c data;
            Ok ()
        | ".bmp" ->
            Stb_image_write.bmp path ~w ~h ~c data;
            Ok ()
        | ".tga" ->
            Stb_image_write.tga path ~w ~h ~c data;
            Ok ()
        | ".jpg" | ".jpeg" ->
            Stb_image_write.jpg path ~w ~h ~c ~quality:90 data;
            Ok ()
        | _ ->
            Error
              (Format_error
                 (Printf.sprintf
                    "Unsupported image format: '%s'. Use .png, .bmp, .tga, .jpg"
                    extension))
    with
    | Sys_error msg -> Error (Io_error msg)
    | Invalid_argument msg -> Error (Other msg)
    | Failure msg -> Error (Other msg)
    | ex -> Error (Other (Printexc.to_string ex))

  let load_npy path = Nx_npy.load_npy path

  let save_npy ?(overwrite = true) path arr =
    Nx_npy.save_npy ~overwrite path arr

  let load_npz path = Nx_npy.load_npz path
  let load_npz_member ~name path = Nx_npy.load_npz_member ~name path

  let save_npz ?(overwrite = true) path items =
    Nx_npy.save_npz ~overwrite path items

  (* Text I/O *)
  let save_txt = Nx_txt.save
  let load_txt = Nx_txt.load

  (* Conversions from packed arrays *)

  let as_float16 = Packed_nx.as_float16
  let as_bfloat16 = Packed_nx.as_bfloat16
  let as_float32 = Packed_nx.as_float32
  let as_float64 = Packed_nx.as_float64
  let as_int8 = Packed_nx.as_int8
  let as_int16 = Packed_nx.as_int16
  let as_int32 = Packed_nx.as_int32
  let as_int64 = Packed_nx.as_int64
  let as_uint8 = Packed_nx.as_uint8
  let as_uint16 = Packed_nx.as_uint16
  let as_bool = Packed_nx.as_bool
  let as_complex32 = Packed_nx.as_complex32
  let as_complex64 = Packed_nx.as_complex64

  (* SafeTensors support *)
  let load_safetensor path = Nx_safetensors.load_safetensor path

  let save_safetensor ?overwrite path items =
    Nx_safetensors.save_safetensor ?overwrite path items
end

(* Main module functions - these fail directly instead of returning results *)

let unwrap_result = function
  | Ok v -> v
  | Error err -> failwith (Error.to_string err)

let as_float16 packed = Packed_nx.as_float16 packed |> unwrap_result
let as_bfloat16 packed = Packed_nx.as_bfloat16 packed |> unwrap_result
let as_float32 packed = Packed_nx.as_float32 packed |> unwrap_result
let as_float64 packed = Packed_nx.as_float64 packed |> unwrap_result
let as_int8 packed = Packed_nx.as_int8 packed |> unwrap_result
let as_int16 packed = Packed_nx.as_int16 packed |> unwrap_result
let as_int32 packed = Packed_nx.as_int32 packed |> unwrap_result
let as_int64 packed = Packed_nx.as_int64 packed |> unwrap_result
let as_uint8 packed = Packed_nx.as_uint8 packed |> unwrap_result
let as_uint16 packed = Packed_nx.as_uint16 packed |> unwrap_result
let as_bool packed = Packed_nx.as_bool packed |> unwrap_result
let as_complex32 packed = Packed_nx.as_complex32 packed |> unwrap_result
let as_complex64 packed = Packed_nx.as_complex64 packed |> unwrap_result

let load_image ?grayscale path =
  Safe.load_image ?grayscale path |> unwrap_result

let save_image ?overwrite path img =
  Safe.save_image ?overwrite path img |> unwrap_result

let load_npy path = Safe.load_npy path |> unwrap_result

let save_npy ?overwrite path arr =
  Safe.save_npy ?overwrite path arr |> unwrap_result

let load_npz path = Safe.load_npz path |> unwrap_result

let load_npz_member ~name path =
  Safe.load_npz_member ~name path |> unwrap_result

let save_npz ?overwrite path items =
  Safe.save_npz ?overwrite path items |> unwrap_result

let load_safetensor path = Safe.load_safetensor path |> unwrap_result

let save_safetensor ?overwrite path items =
  Safe.save_safetensor ?overwrite path items |> unwrap_result

let save_txt ?sep ?append ?newline ?header ?footer ?comments ~out arr =
  Safe.save_txt ?sep ?append ?newline ?header ?footer ?comments ~out arr
  |> unwrap_result

let load_txt ?sep ?comments ?skiprows ?max_rows dtype path =
  Safe.load_txt ?sep ?comments ?skiprows ?max_rows dtype path |> unwrap_result
