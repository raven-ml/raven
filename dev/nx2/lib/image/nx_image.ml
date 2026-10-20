(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module A = Nx_array
module L = Nx_array.Layout

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

(* Loading *)

let map_file path =
  let fd = Unix.openfile path [ Unix.O_RDONLY; O_CLOEXEC ] 0 in
  Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
  let size = (Unix.fstat fd).st_size in
  if size = 0 then Bigarray.(Array1.create int8_unsigned c_layout 0)
  else
    Bigarray.array1_of_genarray
      (Unix.map_file fd Bigarray.int8_unsigned Bigarray.c_layout false
         [| size |])

let load ?(grayscale = false) path =
  let pixels, h, w =
    Save.file path @@ fun () -> Codec.decode ~grayscale (map_file path)
  in
  let shape = if grayscale then [| h; w |] else [| h; w; 3 |] in
  (* The decoded pixels are the value's alone. *)
  Nx.Repr.unsafe_of_array Nx.Host.v
    (A.v A.Dtype.Uint8 (L.contiguous shape) pixels)

(* Saving *)

(* [with_pixels ~by ~channels x f] is [f bytes ~width ~height ~channels] for the
   image [x], its pixels read on the host. *)
let with_pixels ~by ~channels x f =
  let h, w, c =
    match Nx.shape x with
    | [| h; w |] -> (h, w, 1)
    | [| h; w; c |] -> (h, w, c)
    | s -> invalid_argf "%s: %d axes, an image has 2 or 3" by (Array.length s)
  in
  if not (List.mem c channels) then
    invalid_argf "%s: %d channels, the format holds %s" by c
      (String.concat ", " (List.map string_of_int channels));
  if h = 0 || w = 0 then invalid_argf "%s: the image has no pixel" by;
  let pixels a =
    let a = if L.is_contiguous (A.layout a) then a else A.copy a in
    let flat =
      Option.get (A.move (A.Move.Reshape (Iarray.of_array [| h * w * c |])) a)
    in
    A.read Bigarray.int8_unsigned flat @@ fun g ->
    f (Bigarray.array1_of_genarray g) ~width:w ~height:h ~channels:c
  in
  Fail.reading ~by (Nx.Repr.read x) pixels

(* The pHYs chunk holds a positive four-byte PNG integer, at most 2^31 - 1. *)
let pixels_per_metre dpi =
  let ppm = Float.round (dpi /. 0.0254) in
  if not (1. <= ppm && ppm <= 2147483647.) then
    invalid_argf "Nx_image.encode_png: the dpi %g" dpi;
  int_of_float ppm

let png ~by ~ppm ~srgb x =
  with_pixels ~by ~channels:[ 1; 3; 4 ] x (Codec.png_encode ~ppm ~srgb)

let encode_png ?dpi ?(srgb = false) x =
  let ppm = match dpi with None -> 0 | Some dpi -> pixels_per_metre dpi in
  png ~by:"Nx_image.encode_png" ~ppm ~srgb x

let save ?(overwrite = true) path x =
  let by = "Nx_image.save" in
  match String.lowercase_ascii (Filename.extension path) with
  | ".png" ->
      let png = png ~by ~ppm:0 ~srgb:false x in
      Save.write ~overwrite path (fun fd -> Save.string fd png)
  | ".jpg" | ".jpeg" ->
      with_pixels ~by ~channels:[ 1; 3 ] x
      @@ fun pixels ~width ~height ~channels ->
      Save.write ~overwrite path (fun fd ->
          Codec.jpeg_encode fd pixels width height channels)
  | ext ->
      invalid_argf "%s: the extension %S is neither .png, .jpg nor .jpeg" by ext
