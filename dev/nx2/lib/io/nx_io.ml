(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module A = Nx_array
module L = Nx_array.Layout
module Archive = Archive

(* NumPy *)

let load_npy path =
  Fail.file path @@ fun () ->
  let file = Storage.open_file path in
  let read pos n = Storage.string file ~pos n in
  Npy.value (Npy.header read ~length:(Rig.Buffer.length file)) file

let save_npy path x =
  let by = "Nx_io.save_npy" in
  let dt = Nx.dtype x in
  Npy.check ~by dt;
  let header = Npy.encode dt (Nx.shape x) in
  Storage.with_bytes ~by x @@ fun data ->
  Save.write path @@ fun fd ->
  Save.string fd header;
  Save.bytes fd data

let load_npz path =
  Fail.file path @@ fun () ->
  let zip = Zip.open_in (Storage.open_file path) in
  List.fold_left
    (fun archive name ->
      if name = "" then failwith "an entry has an empty name";
      Archive.add name (Zip.read_npy zip name) archive)
    Archive.empty (Zip.npy_entries zip)

let load_npz_entry ~name path =
  Fail.file path @@ fun () ->
  Zip.read_npy (Zip.open_in (Storage.open_file path)) name

let save_npz path archive =
  let by = "Nx_io.save_npz" in
  let entries = Archive.bindings archive in
  List.iter
    (fun (name, Nx.P x) ->
      if not (Zip.valid_name name) then
        Fail.invalid_argf "%s: %S is no valid entry name" by name;
      Npy.check ~by:(by ^ ": " ^ name) (Nx.dtype x))
    entries;
  Save.write path @@ fun fd ->
  let zip = Zip.open_out fd in
  List.iter
    (fun (name, Nx.P x) ->
      let header = Npy.encode (Nx.dtype x) (Nx.shape x) in
      Storage.with_bytes ~by x (Zip.add_npy zip name header))
    entries;
  Zip.close_out zip

(* SafeTensors *)

let load_safetensors path =
  Fail.file path @@ fun () -> Safetensors.load (Storage.open_file path)

let save_safetensors path archive =
  let by = "Nx_io.save_safetensors" in
  let entry (name, (Nx.P x as p)) =
    match Safetensors.code (Nx.dtype x) with
    | Some (code, rank) -> (rank, name, code, p)
    | None ->
        Fail.invalid_argf "%s: %s: SafeTensors has no %s" by name
          (Nx.Dtype.name (Nx.dtype x))
  in
  let entries =
    List.map entry (Archive.bindings archive)
    |> List.stable_sort (fun (r, _, _, _) (r', _, _, _) -> Int.compare r' r)
  in
  let header =
    Safetensors.header
      (List.map
         (fun (_, name, code, Nx.P x) -> (name, code, Nx.shape x, Nx.nbytes x))
         entries)
  in
  Save.write path @@ fun fd ->
  Save.string fd header;
  List.iter
    (fun (_, _, _, Nx.P x) -> Storage.with_bytes ~by x (Save.bytes fd))
    entries

(* GGUF *)

module Gguf = Gguf

let load_gguf path =
  Fail.file path @@ fun () -> Gguf.read (Storage.open_file path)

(* Text *)

let load_txt ?(sep = " ") ?(comments = "#") ?(skiprows = 0) path dt =
  Txt.load ~sep ~comments ~skiprows path dt

let save_txt ?(sep = " ") path x = Txt.save ~sep path x

(* Images *)

(* The pixels are decoded into host memory the value owns, which its donation
   may write, as an inflated npz entry's are. *)
let load_image path =
  Fail.file path @@ fun () ->
  Storage.read (Storage.pages (Storage.open_file path)) @@ fun src ->
  let h, w, c = Image.probe src in
  let pixels = Rig.Buffer.create Rig.host (h * w * c) in
  Rig.Buffer.write pixels Bigarray.int8_unsigned (Image.decode src);
  let shape = if c = 1 then [| h; w |] else [| h; w; c |] in
  let layout = L.contiguous (Iarray.of_array shape) in
  Nx.Repr.unsafe_of_array Nx.Host.v (A.v Nx_array.Dtype.Uint8 layout pixels)

(* [with_pixels ~by ~channels x f] is [f bytes ~width ~height ~channels] for the
   image [x], its pixels read on the host. *)
let with_pixels ~by ~channels x f =
  let h, w, c =
    match Nx.shape x with
    | [| h; w |] -> (h, w, 1)
    | [| h; w; c |] -> (h, w, c)
    | s ->
        Fail.invalid_argf "%s: %d axes, an image has 2 or 3" by (Array.length s)
  in
  if not (List.mem c channels) then
    Fail.invalid_argf "%s: %d channels, the format holds %s" by c
      (String.concat ", " (List.map string_of_int channels));
  if h = 0 || w = 0 then Fail.invalid_argf "%s: the image has no pixel" by;
  Storage.with_bytes ~by x (fun b -> f b ~width:w ~height:h ~channels:c)

(* The pHYs chunk holds a positive four-byte PNG integer, at most 2^31 - 1. *)
let pixels_per_metre dpi =
  let ppm = Float.round (dpi /. 0.0254) in
  if not (1. <= ppm && ppm <= 2147483647.) then
    Fail.invalid_argf "Nx_io.encode_png: the dpi %g" dpi;
  int_of_float ppm

let png ~by ~ppm ~srgb x =
  with_pixels ~by ~channels:[ 1; 2; 3; 4 ] x (Image.png_encode ~ppm ~srgb)

let encode_png ?dpi ?(srgb = false) x =
  let ppm = match dpi with None -> 0 | Some dpi -> pixels_per_metre dpi in
  png ~by:"Nx_io.encode_png" ~ppm ~srgb x

let save_image path x =
  let by = "Nx_io.save_image" in
  let file =
    match String.lowercase_ascii (Filename.extension path) with
    | ".png" -> png ~by ~ppm:0 ~srgb:false x
    | ".jpg" | ".jpeg" -> with_pixels ~by ~channels:[ 1; 3 ] x Image.jpeg_encode
    | ext ->
        Fail.invalid_argf "%s: the extension %S is neither .png, .jpg nor .jpeg"
          by ext
  in
  Save.write path (fun fd -> Save.string fd file)
