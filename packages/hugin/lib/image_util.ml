(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Shared image encoding utilities *)

(* Base64 *)

let base64_alphabet =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

let base64_encode input =
  let len = String.length input in
  let out_len = (len + 2) / 3 * 4 in
  let out = Bytes.create out_len in
  let rec loop i j =
    if i < len then begin
      let b0 = Char.code (String.unsafe_get input i) in
      let b1 =
        if i + 1 < len then Char.code (String.unsafe_get input (i + 1)) else 0
      in
      let b2 =
        if i + 2 < len then Char.code (String.unsafe_get input (i + 2)) else 0
      in
      Bytes.unsafe_set out j (String.unsafe_get base64_alphabet (b0 lsr 2));
      Bytes.unsafe_set out (j + 1)
        (String.unsafe_get base64_alphabet (((b0 land 3) lsl 4) lor (b1 lsr 4)));
      Bytes.unsafe_set out (j + 2)
        (if i + 1 < len then
           String.unsafe_get base64_alphabet
             (((b1 land 0xf) lsl 2) lor (b2 lsr 6))
         else '=');
      Bytes.unsafe_set out (j + 3)
        (if i + 2 < len then String.unsafe_get base64_alphabet (b2 land 0x3f)
         else '=');
      loop (i + 3) (j + 4)
    end
  in
  loop 0 0;
  Bytes.unsafe_to_string out

(* Nx uint8 image -> Cairo ARGB32 surface *)

let nx_to_cairo_surface (data : Nx.uint8_t) =
  let shape = Nx.shape data in
  let img_h = shape.(0) and img_w = shape.(1) in
  let channels = if Array.length shape > 2 then shape.(2) else 1 in
  let stride = Ucairo.Image.stride_for_width img_w in
  let data_arr =
    Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout
      (stride * img_h)
  in
  let buf = Nx.data data in
  let base = Nx.offset data in
  let strides = Nx.strides data in
  (* uint8: byte strides = element strides *)
  let s0 = strides.(0) and s1 = strides.(1) in
  let s2 = if Array.length strides > 2 then strides.(2) else 0 in
  for row = 0 to img_h - 1 do
    let row_base = base + (row * s0) in
    for col = 0 to img_w - 1 do
      let off = (row * stride) + (col * 4) in
      let idx = row_base + (col * s1) in
      let r = Nx_buffer.unsafe_get buf idx in
      let g = Nx_buffer.unsafe_get buf (idx + s2) in
      let b = Nx_buffer.unsafe_get buf (idx + (2 * s2)) in
      let a =
        if channels >= 4 then Nx_buffer.unsafe_get buf (idx + (3 * s2)) else 255
      in
      (* Cairo ARGB32: premultiplied BGRA in memory on little-endian *)
      let premul c a = c * a / 255 in
      Bigarray.Array1.unsafe_set data_arr off (premul b a);
      Bigarray.Array1.unsafe_set data_arr (off + 1) (premul g a);
      Bigarray.Array1.unsafe_set data_arr (off + 2) (premul r a);
      Bigarray.Array1.unsafe_set data_arr (off + 3) a
    done
  done;
  Ucairo.Image.create_for_data8 data_arr ~w:img_w ~h:img_h ~stride

let nx_to_png_base64 data =
  let surface = nx_to_cairo_surface data in
  let png_buf = Buffer.create 4096 in
  Ucairo.Png.write_to_stream surface (Buffer.add_string png_buf);
  Ucairo.Surface.finish surface;
  base64_encode (Buffer.contents png_buf)
