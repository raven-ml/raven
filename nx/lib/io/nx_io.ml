open Bigarray_ext
open Utils

type packed_nx = Utils.packed_nx = P : ('a, 'b) Nx.t -> packed_nx
type nx_dims = [ `Gray of int * int | `Color of int * int * int ]

let get_nx_dims arr : nx_dims =
  match Nx.shape arr with
  | [| h; w |] -> `Gray (h, w)
  | [| h; w; c |] -> `Color (h, w, c)
  | s ->
      fail_msg "Invalid nx dimensions: expected 2 or 3, got %d (%s)"
        (Array.length s)
        (Array.to_list s |> List.map string_of_int |> String.concat "x")

let load_image ?(grayscale = false) path =
  try
    let desired_channels = if grayscale then 1 else 3 in
    match Stb_image.load ~channels:desired_channels path with
    | Ok img ->
        let h = Stb_image.height img in
        let w = Stb_image.width img in
        let c = Stb_image.channels img in
        let buffer = Stb_image.data img in
        let nd = Nx.of_bigarray (genarray_of_array1 buffer) in
        let shape = if c = 1 then [| h; w |] else [| h; w; c |] in
        Nx.reshape shape nd
    | Error (`Msg msg) -> fail_msg "STB Load Error (%s): %s" path msg
  with
  | Sys_error msg -> fail_msg "System error loading image '%s': %s" path msg
  | ex ->
      let err_msg = Printexc.to_string ex in
      let backtrace = Printexc.get_backtrace () in
      fail_msg "Unexpected error loading image '%s': %s\n%s" path err_msg
        backtrace

let save_image nd_img path =
  try
    let h, w, c =
      match get_nx_dims nd_img with
      | `Gray (h, w) -> (h, w, 1)
      | `Color (h, w, c) -> (h, w, c)
    in

    (* Ensure the input array is uint8 *)
    let data_gen = Nx.to_bigarray nd_img in
    let data =
      match Bigarray.Genarray.kind data_gen with
      | Bigarray.Int8_unsigned -> array1_of_genarray data_gen
    in

    let extension = Filename.extension path |> String.lowercase_ascii in
    match extension with
    | ".png" -> Stb_image_write.png path ~w ~h ~c data
    | ".bmp" -> Stb_image_write.bmp path ~w ~h ~c data
    | ".tga" -> Stb_image_write.tga path ~w ~h ~c data
    | ".jpg" | ".jpeg" -> Stb_image_write.jpg path ~w ~h ~c ~quality:90 data
    (* Note: Stb_image_write.hdr requires float32 data, not handled here *)
    | _ ->
        fail_msg
          "Unsupported image format for saving: '%s'. Use .png, .bmp, .tga, \
           .jpg"
          extension
  with
  | Sys_error msg -> fail_msg "System error saving image to '%s': %s" path msg
  | Invalid_argument msg ->
      (* Can be raised by Stb_image_write for bad dims/channels *)
      fail_msg "Invalid argument during saving '%s': %s" path msg
  | Failure msg ->
      (* Can be raised by Stb_image_write *)
      fail_msg "Failure during saving image to '%s': %s" path msg
  | ex ->
      let err_msg = Printexc.to_string ex in
      let backtrace = Printexc.get_backtrace () in
      fail_msg "Unexpected error saving image to '%s': %s\n%s" path err_msg
        backtrace

let load_npy path =
  try
    match Npy.read_copy path with
    | P genarray ->
        let genarray = Genarray.change_layout genarray Bigarray.c_layout in
        P (Nx.of_bigarray genarray)
  with
  | Unix.Unix_error (e, _, _) ->
      fail_msg "NPY Load Error (%s): %s" path (Unix.error_message e)
  | Sys_error msg -> fail_msg "NPY Load System Error (%s): %s" path msg
  | Failure msg -> fail_msg "NPY Load Failure (%s): %s" path msg
  | ex ->
      let err_msg = Printexc.to_string ex in
      fail_msg "Unexpected NPY Load Error (%s): %s" path err_msg

let save_npy nx path =
  try
    let genarray = Nx.to_bigarray nx in
    Npy.write genarray path
  with
  | Unix.Unix_error (e, _, _) ->
      fail_msg "NPY Save Error (%s): %s" path (Unix.error_message e)
  | Sys_error msg -> fail_msg "NPY Save System Error (%s): %s" path msg
  | Failure msg ->
      fail_msg "NPY Save Failure (%s): %s - Likely unsupported dtype" path msg
  | ex ->
      let err_msg = Printexc.to_string ex in
      fail_msg "Unexpected NPY Save Error (%s): %s" path err_msg

type npz_archive = (string, packed_nx) Hashtbl.t

let load_npz path =
  let archive = Hashtbl.create 16 in
  let zip_in = ref None in
  try
    let zi = Npy.Npz.open_in path in
    zip_in := Some zi;
    let entries = Npy.Npz.entries zi in
    List.iter
      (fun name ->
        match Npy.Npz.read zi name with
        | Npy.P genarray ->
            let genarray = Genarray.change_layout genarray Bigarray.c_layout in
            Hashtbl.add archive name (P (Nx.of_bigarray genarray)))
      entries;
    Npy.Npz.close_in zi;
    archive
  with
  | Zip.Error (name, func, msg) ->
      (match !zip_in with Some zi -> Npy.Npz.close_in zi | None -> ());
      fail_msg "NPZ Load Zip Error (%s): %s in %s: %s" path name func msg
  | Unix.Unix_error (e, _, _) ->
      (match !zip_in with Some zi -> Npy.Npz.close_in zi | None -> ());
      fail_msg "NPZ Load Error (%s): %s" path (Unix.error_message e)
  | Sys_error msg ->
      (match !zip_in with Some zi -> Npy.Npz.close_in zi | None -> ());
      fail_msg "NPZ Load System Error (%s): %s" path msg
  | Failure msg ->
      (match !zip_in with Some zi -> Npy.Npz.close_in zi | None -> ());
      fail_msg "NPZ Load Failure (%s): %s" path msg
  | ex ->
      (match !zip_in with Some zi -> Npy.Npz.close_in zi | None -> ());
      let err_msg = Printexc.to_string ex in
      fail_msg "Unexpected NPZ Load Error (%s): %s" path err_msg

let load_npz_member ~path ~name =
  let zip_in = ref None in
  try
    let zi = Npy.Npz.open_in path in
    zip_in := Some zi;
    let packed_npy =
      try Npy.Npz.read zi name
      with Not_found ->
        fail_msg "NPZ Load Error (%s): Member '%s' not found" path name
    in
    let result =
      match packed_npy with
      | Npy.P genarray ->
          let genarray = Genarray.change_layout genarray Bigarray.c_layout in
          P (Nx.of_bigarray genarray)
    in
    Npy.Npz.close_in zi;
    result
  with
  | Zip.Error (zip_name, func, msg) ->
      (match !zip_in with Some zi -> Npy.Npz.close_in zi | None -> ());
      fail_msg "NPZ Load Zip Error (%s): %s in %s: %s" path zip_name func msg
  | Unix.Unix_error (e, _, _) ->
      (match !zip_in with Some zi -> Npy.Npz.close_in zi | None -> ());
      fail_msg "NPZ Load Error (%s): %s" path (Unix.error_message e)
  | Sys_error msg ->
      (match !zip_in with Some zi -> Npy.Npz.close_in zi | None -> ());
      fail_msg "NPZ Load System Error (%s): %s" path msg
  | Failure msg ->
      (match !zip_in with Some zi -> Npy.Npz.close_in zi | None -> ());
      fail_msg "NPZ Load Failure (%s, member %s): %s" path name msg
  | ex ->
      (match !zip_in with Some zi -> Npy.Npz.close_in zi | None -> ());
      let err_msg = Printexc.to_string ex in
      fail_msg "Unexpected NPZ Load Error (%s, member %s): %s" path name err_msg

let save_npz items path =
  let zip_out = ref None in
  try
    let zo = Npy.Npz.open_out path in
    zip_out := Some zo;
    List.iter
      (fun (name, P nx) ->
        let genarray = Nx.to_bigarray nx in
        Npy.Npz.write zo name genarray)
      items;
    Npy.Npz.close_out zo
  with
  | Zip.Error (name, func, msg) ->
      (match !zip_out with Some zo -> Npy.Npz.close_out zo | None -> ());
      fail_msg "NPZ Save Zip Error (%s): %s in %s: %s" path name func msg
  | Unix.Unix_error (e, _, _) ->
      (match !zip_out with Some zo -> Npy.Npz.close_out zo | None -> ());
      fail_msg "NPZ Save Error (%s): %s" path (Unix.error_message e)
  | Sys_error msg ->
      (match !zip_out with Some zo -> Npy.Npz.close_out zo | None -> ());
      fail_msg "NPZ Save System Error (%s): %s" path msg
  | Failure msg ->
      (match !zip_out with Some zo -> Npy.Npz.close_out zo | None -> ());
      fail_msg "NPZ Save Failure (%s): %s - Likely unsupported dtype" path msg
  | ex ->
      (match !zip_out with Some zo -> Npy.Npz.close_out zo | None -> ());
      let err_msg = Printexc.to_string ex in
      fail_msg "Unexpected NPZ Save Error (%s): %s" path err_msg

(* Conversions from packed arrays *)

let to_float16 = convert "to_float16" Nx.float16
let to_float32 = convert "to_float32" Nx.float32
let to_float64 = convert "to_float64" Nx.float64
let to_int8 = convert "to_int8" Nx.int8
let to_int16 = convert "to_int16" Nx.int16
let to_int32 = convert "to_int32" Nx.int32
let to_int64 = convert "to_int64" Nx.int64
let to_uint8 = convert "to_uint8" Nx.uint8
let to_uint16 = convert "to_uint16" Nx.uint16
let to_complex32 = convert "to_complex32" Nx.complex32
let to_complex64 = convert "to_complex64" Nx.complex64
