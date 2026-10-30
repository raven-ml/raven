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
    (fun archive (name, x) ->
      if name = "" then failwith {|the entry ".npy" has no name before .npy|};
      Archive.add name x archive)
    Archive.empty (Zip.read_npys zip)

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
    Safetensors.check_name ~by name;
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

let load_txt ?(sep = " ") ?(comments = "#") ?(skiprows = 0) dt path =
  Txt.load ~sep ~comments ~skiprows dt path

let save_txt ?(sep = " ") path x = Txt.save ~sep path x

(* Images *)

(* The pixels are decoded into host memory the value owns, which its donation
   may write, as an inflated npz entry's are. *)
let load_image path =
  Fail.file path @@ fun () ->
  Storage.read (Storage.pages (Storage.open_file path)) @@ fun src ->
  let h, w, c = Image.probe src in
  let write = Image.decode src in
  let pixels = Rig.Buffer.create Rig.host (h * w * c) in
  Rig.Buffer.write pixels Bigarray.int8_unsigned write;
  let layout = L.contiguous (Iarray.of_array [| h; w; c |]) in
  Nx.Repr.unsafe_of_array Nx.Host.v (A.v Nx_array.Dtype.Uint8 layout pixels)

(* The largest side of a JPEG image, whose frame header states each side in 16
   bits (T.81, B.2.2). *)
let jpeg_side = 65535

(* [with_pixels ~by ~channels ~side x f] is [f bytes ~width ~height ~channels]
   for the image [x], its pixels read on the host, after [x]'s shape is checked
   against the format's [channels] and largest [side]. *)
let with_pixels ~by ~channels ~side x f =
  let h, w, c =
    match Nx.shape x with
    | [| h; w; c |] -> (h, w, c)
    | s -> Fail.invalid_argf "%s: %d axes, an image has 3" by (Array.length s)
  in
  if not (List.mem c channels) then
    Fail.invalid_argf "%s: %d channels, the format holds %s" by c
      (String.concat ", " (List.map string_of_int channels));
  if h = 0 || w = 0 then Fail.invalid_argf "%s: the image has no pixel" by;
  if h > side || w > side then
    Fail.invalid_argf "%s: %d by %d pixels, the format holds at most %d a side"
      by h w side;
  Storage.with_bytes ~by x (fun b -> f b ~width:w ~height:h ~channels:c)

(* The pHYs chunk holds a positive PNG integer. *)
let pixels_per_metre ~by dpi =
  let ppm = Float.round (dpi /. 0.0254) in
  if not (1. <= ppm && ppm <= Float.of_int Image.png_max) then
    Fail.invalid_argf "%s: the dpi %g" by dpi;
  int_of_float ppm

(* [png ~by ?dpi ?srgb x write] is [write encode] once [x] is checked and its
   pixels are read, where [encode emit] gives [emit] each piece of the PNG
   file. *)
let png ~by ?dpi ?(srgb = false) x write =
  let ppm = match dpi with None -> 0 | Some dpi -> pixels_per_metre ~by dpi in
  with_pixels ~by ~channels:[ 1; 2; 3; 4 ] ~side:Image.png_max x
    (fun b ~width ~height ~channels ->
      write (Image.png_encode ~ppm ~srgb b ~width ~height ~channels))

let encode_png ?dpi ?srgb x =
  png ~by:"Nx_io.encode_png" ?dpi ?srgb x @@ fun encode ->
  let pieces = ref [] in
  encode (fun s -> pieces := s :: !pieces);
  String.concat "" (List.rev !pieces)

let save_png ?dpi ?srgb path x =
  png ~by:"Nx_io.save_png" ?dpi ?srgb x @@ fun encode ->
  Save.write path (fun fd -> encode (Save.string fd))

let save_jpeg path x =
  let by = "Nx_io.save_jpeg" in
  let file =
    with_pixels ~by ~channels:[ 1; 3 ] ~side:jpeg_side x Image.jpeg_encode
  in
  Save.write path (fun fd -> Save.bytes fd file)
