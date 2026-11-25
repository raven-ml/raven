(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Images, as nx_io.mli's Images section states them. PNG files built here from
   PNG-3 load as the pixels their samples state; a PNG saved from any image
   loads back as its pixels, a JPEG within the error its quantisation allows;
   files other encoders wrote load as Pillow decodes them; a file cut short or
   corrupted raises [Failure] naming it, and never anything else. *)

open Windtrap
open Io_support

(* The pixels of the fixture image [name], as gen/generate.py states them. *)
let expected name =
  let npy = Filename.remove_extension name ^ ".npy" in
  Nx.unpack Nx.uint8 (Nx_io.load_npy (fixture npy))

let errors a b =
  Array.map2 (fun x y -> abs (x - y)) (Nx.to_array a) (Nx.to_array b)

(* PNG chunks *)

(* The chunks of the PNG file [png], as their types and data, in order. *)
let chunks png =
  let rec loop i acc =
    if i >= String.length png then List.rev acc
    else
      let n = Int32.to_int (String.get_int32_be png i) in
      let chunk = (String.sub png (i + 4) 4, String.sub png (i + 8) n) in
      loop (i + 12 + n) (chunk :: acc)
  in
  loop 8 []

let chunk_data ty png =
  List.filter_map
    (fun (ty', data) -> if ty = ty' then Some data else None)
    (chunks png)

(* [png] without its chunks of type [ty]. *)
let without ty png =
  let b = Buffer.create (String.length png) in
  Buffer.add_string b (String.sub png 0 8);
  let rec loop i =
    if i < String.length png then begin
      let n = Int32.to_int (String.get_int32_be png i) in
      if String.sub png (i + 4) 4 <> ty then
        Buffer.add_string b (String.sub png i (12 + n));
      loop (i + 12 + n)
    end
  in
  loop 8;
  Buffer.contents b

let be32 n = String.init 4 (fun i -> Char.chr ((n lsr (8 * (3 - i))) land 0xff))
let tiny = Nx.full Nx.uint8 [| 2; 3; 3 |] 9
let signature = "\137PNG\r\n\026\n"

(* The chunk of type [ty] and [data], with its CRC-32. *)
let chunk ty data =
  be32 (String.length data) ^ ty ^ data ^ be32 (Lz.Crc32.string (ty ^ data))

let ihdr ~width ~height =
  chunk "IHDR" (be32 width ^ be32 height ^ "\008\000\000\000\000")

(* An 8-bit grey PNG of [width] by [height] pixels whose image data is empty. *)
let header_png ~width ~height =
  signature ^ ihdr ~width ~height
  ^ chunk "IDAT" (Lz.Zlib.encode_string "")
  ^ chunk "IEND" ""

(* The data of the zlib stream [s], decoded into a destination doubled until it
   holds them. *)
let inflate s =
  let open Bigarray in
  let src = Array1.create int8_unsigned c_layout (String.length s) in
  String.iteri (fun i c -> src.{i} <- Char.code c) s;
  let rec decode n =
    let dst = Array1.create int8_unsigned c_layout n in
    match Lz.Zlib.decode src dst with
    | Ok { Lz.produced; _ } -> String.init produced (fun i -> Char.chr dst.{i})
    | Error { Lz.why = Too_long; _ } -> decode (2 * n)
    | Error e -> Format.kasprintf failwith "%a" Lz.pp_error e
  in
  decode (4 * String.length s)

(* The scanlines the image data of [png] decompresses to. *)
let scanlines png = inflate (String.concat "" (chunk_data "IDAT" png))

(* [png] with its IDAT chunks replaced by one holding [idat]. *)
let with_idat png idat =
  let rec loop placed = function
    | [] -> []
    | ("IDAT", _) :: rest when placed -> loop placed rest
    | ("IDAT", _) :: rest -> chunk "IDAT" idat :: loop true rest
    | (ty, data) :: rest -> chunk ty data :: loop placed rest
  in
  signature ^ String.concat "" (loop false (chunks png))

(* [s] with its byte [i] xored by [mask]. *)
let xor_byte s i mask =
  String.mapi
    (fun j c -> if i = j then Char.chr (Char.code c lxor mask) else c)
    s

(* A byte changed in a PNG file: in the type or data of a chunk, at an index
   into the type and data, or in the scanlines the image data decompresses
   to. *)
type flip = Chunk of int * int | Scanline of int

(* The file [png] with the byte [flip] names xored by [mask], its scanlines
   recompressed into its first IDAT chunk and every CRC-32 recomputed, so that
   the change reaches the decoder past the checksums. *)
let flipped png flip mask =
  let cs = chunks png in
  let first_idat = List.find_index (fun (ty, _) -> ty = "IDAT") cs in
  let change k (ty, data) =
    match flip with
    | Chunk (k', i) when k = k' ->
        let s = xor_byte (ty ^ data) i mask in
        Some (String.sub s 0 4, String.sub s 4 (String.length data))
    | Scanline i when Some k = first_idat ->
        let scanlines = xor_byte (scanlines png) i mask in
        Some ("IDAT", Lz.Zlib.encode_string scanlines)
    | Scanline _ when ty = "IDAT" -> None
    | Chunk _ | Scanline _ -> Some (ty, data)
  in
  signature
  ^ String.concat ""
      (List.map
         (fun (ty, data) -> chunk ty data)
         (List.filter_map Fun.id (List.mapi change cs)))

let pngs =
  [
    "png_filters.png";
    "png_gray1.png";
    "png_gray4.png";
    "png_rgb16.png";
    "png_palette_adam7.png";
    "png_palette_short_trns.png";
    "png_rgba.png";
    "png_la.png";
    "png_gray_trns.png";
    "png_rgb16_trns.png";
  ]

(* A PNG fixture with one byte changed, half the time in a chunk's type or data
   and half the time in its scanlines, and by any mask. *)
let png_flips =
  let open Gen in
  let* name = of_list ~pp:Format.pp_print_string pngs in
  let png = read (fixture name) in
  let cs = chunks png in
  let* flip =
    one_of
      [
        (let* k = int_range 0 (List.length cs - 1) in
         let ty, data = List.nth cs k in
         let+ i = int_range 0 (String.length ty + String.length data - 1) in
         Chunk (k, i));
        (let+ i = int_range 0 (String.length (scanlines png) - 1) in
         Scanline i);
      ]
  in
  let+ mask = int_range 1 255 in
  (name, flip, mask)

let png_flips =
  Gen.with_pp
    (fun ppf (name, flip, mask) ->
      match flip with
      | Chunk (k, i) ->
          Format.fprintf ppf "%s, chunk %d, byte %d xor %d" name k i mask
      | Scanline i ->
          Format.fprintf ppf "%s, scanline byte %d xor %d" name i mask)
    png_flips

(* PNG files built from PNG-3 *)

(* The colour types and bit depths of PNG-3's Table 11.1. *)
let color_depths =
  List.concat_map
    (fun (color, depths) -> List.map (fun d -> (color, d)) depths)
    [
      (0, [ 1; 2; 4; 8; 16 ]);
      (2, [ 8; 16 ]);
      (3, [ 1; 2; 4; 8 ]);
      (4, [ 8; 16 ]);
      (6, [ 8; 16 ]);
    ]

(* The samples a pixel of colour type [color] holds in the file (Table 11.1). *)
let file_channels = function 0 | 3 -> 1 | 4 -> 2 | 2 -> 3 | _ -> 4

(* A PNG file to build: its colour type and bit depth, interlaced by Adam7 or
   not, [h] rows of [w] pixels whose samples are [samples], row-major, a palette
   of RGB triples, the tRNS chunk's alphas of palette entries or the transparent
   sample values, and the filter types its scanlines take in turn. *)
type png = {
  color : int;
  depth : int;
  interlaced : bool;
  h : int;
  w : int;
  samples : int array;
  palette : int array;
  trns : int array option;
  filters : int array;
}

(* A scanline's samples packed most significant bit first at [depth] bits, a
   16-bit sample in two bytes, most significant first (7.2). *)
let pack depth samples =
  let n = Array.length samples in
  if depth = 16 then
    String.init (2 * n) (fun i ->
        let v = samples.(i / 2) in
        Char.chr (if i mod 2 = 0 then v lsr 8 else v land 0xff))
  else begin
    let b = Bytes.make (((n * depth) + 7) / 8) '\000' in
    Array.iteri
      (fun i v ->
        let bit = i * depth in
        let at = bit / 8 in
        let v = v lsl (8 - depth - (bit mod 8)) in
        Bytes.set_uint8 b at (Bytes.get_uint8 b at lor v))
      samples;
    Bytes.to_string b
  end

(* 9.4's Paeth predictor. *)
let paeth a b c =
  let p = a + b - c in
  let pa = abs (p - a) and pb = abs (p - b) and pc = abs (p - c) in
  if pa <= pb && pa <= pc then a else if pb <= pc then b else c

(* The scanline [line] after filter type [t] (9.2), its filter byte first:
   [prev] is the unfiltered scanline above it, or zeros for the first, and [bpp]
   a pixel's bytes, at least one. *)
let filtered t ~bpp prev line =
  String.init
    (String.length line + 1)
    (fun i ->
      if i = 0 then Char.chr t
      else
        let i = i - 1 in
        let byte s i = if i >= 0 then Char.code s.[i] else 0 in
        let a = byte line (i - bpp) and b = byte prev i in
        let c = byte prev (i - bpp) in
        let p =
          match t with
          | 0 -> 0
          | 1 -> a
          | 2 -> b
          | 3 -> (a + b) / 2
          | _ -> paeth a b c
        in
        Char.chr ((Char.code line.[i] - p) land 0xff))

(* Adam7's passes (8.2), each its first row and column and its steps between
   rows and columns. *)
let adam7 =
  [
    (0, 0, 8, 8);
    (0, 4, 8, 8);
    (4, 0, 8, 4);
    (0, 2, 4, 4);
    (2, 0, 4, 2);
    (0, 1, 2, 2);
    (1, 0, 2, 1);
  ]

(* The scanlines of [p]'s passes, filtered. A pass of no row or no column has no
   scanline (8.2). *)
let image_data p =
  let ch = file_channels p.color in
  let bpp = Int.max 1 (ch * p.depth / 8) in
  let passes = if p.interlaced then adam7 else [ (0, 0, 1, 1) ] in
  let b = Buffer.create 256 and k = ref 0 in
  List.iter
    (fun (r0, c0, dr, dc) ->
      let rows = if p.h > r0 then (p.h - r0 + dr - 1) / dr else 0 in
      let cols = if p.w > c0 then (p.w - c0 + dc - 1) / dc else 0 in
      let prev = ref "" in
      for i = 0 to (if cols = 0 then 0 else rows) - 1 do
        let y = r0 + (i * dr) in
        let sample j =
          p.samples.((((y * p.w) + c0 + (j / ch * dc)) * ch) + (j mod ch))
        in
        let line = pack p.depth (Array.init (cols * ch) sample) in
        if i = 0 then prev := String.make (String.length line) '\000';
        let t = p.filters.(!k mod Array.length p.filters) in
        Buffer.add_string b (filtered t ~bpp !prev line);
        prev := line;
        incr k
      done)
    passes;
  Buffer.contents b

let bytes_of a = String.init (Array.length a) (fun i -> Char.chr a.(i))

let png_file p =
  let header = [| p.depth; p.color; 0; 0; Bool.to_int p.interlaced |] in
  let trns t =
    if p.color = 3 then bytes_of t
    else
      String.concat "" (Array.to_list (Array.map (fun v -> pack 16 [| v |]) t))
  in
  signature
  ^ chunk "IHDR" (be32 p.w ^ be32 p.h ^ bytes_of header)
  ^ (if p.color = 3 then chunk "PLTE" (bytes_of p.palette) else "")
  ^ Option.fold ~none:"" ~some:(fun t -> chunk "tRNS" (trns t)) p.trns
  ^ chunk "IDAT" (Lz.Zlib.encode_string (image_data p))
  ^ chunk "IEND" ""

(* The image nx_io.mli states [p] loads as: a palette entry's colours, and its
   alpha in tRNS, 255 past the chunk's end; a 16-bit sample's high byte, and a
   sample of 1, 2 or 4 bits scaled to 8 by 255 / (2^depth - 1), which divides;
   and an alpha of 0 where a pixel's samples equal tRNS's at the file's depth,
   255 elsewhere. *)
let pixels p =
  let ch = file_channels p.color in
  let scale v =
    if p.depth = 16 then v lsr 8 else v * 255 / ((1 lsl p.depth) - 1)
  in
  let pixel k =
    let s = Array.sub p.samples (k * ch) ch in
    match (p.color, p.trns) with
    | 3, None -> Array.sub p.palette (3 * s.(0)) 3
    | 3, Some t ->
        let alpha = if s.(0) < Array.length t then t.(s.(0)) else 255 in
        Array.append (Array.sub p.palette (3 * s.(0)) 3) [| alpha |]
    | _, None -> Array.map scale s
    | _, Some t ->
        Array.append (Array.map scale s) [| (if s = t then 0 else 255) |]
  in
  let all = Array.concat (List.init (p.h * p.w) pixel) in
  Nx.create Nx.uint8 [| p.h; p.w; Array.length all / (p.h * p.w) |] all

(* [samples] and a tRNS chunk's values for them: a palette's alphas, fewer than
   or as many as its entries; grey or RGB samples' the value of one pixel, with
   in 16 bits a neighbour of the same high byte planted in another pixel; none
   for an alpha channel. *)
let trns_of ~color ~depth ~entries samples =
  let open Gen in
  let ch = file_channels color in
  let n = Array.length samples / ch in
  match color with
  | 3 ->
      let alphas k =
        map
          (fun a -> (samples, Some a))
          (array ~size:(constant k) (int_range 0 255))
      in
      frequency
        [
          (1, constant (samples, None));
          (1, alphas entries);
          (2, bind (int_range 1 entries) alphas);
        ]
  | 0 | 2 ->
      let planted =
        let+ k = int_range 0 (n - 1) and+ other = int_range 0 (n - 1) in
        let samples = Array.copy samples in
        let t = Array.sub samples (k * ch) ch in
        if depth = 16 && other <> k then begin
          Array.blit t 0 samples (other * ch) ch;
          samples.(other * ch) <- t.(0) lxor 1
        end;
        (samples, Some t)
      in
      frequency [ (1, constant (samples, None)); (2, planted) ]
  | _ -> constant (samples, None)

let built_pngs =
  let open Gen in
  let* color, depth = of_list color_depths in
  let* interlaced = bool in
  let* h, w = pair (int_range 1 9) (int_range 1 9) in
  let* entries =
    if color = 3 then int_range 1 (Int.min 256 (1 lsl depth)) else constant 0
  in
  let* palette = array ~size:(constant (3 * entries)) (int_range 0 255) in
  let top = if color = 3 then entries - 1 else (1 lsl depth) - 1 in
  let* samples =
    array ~size:(constant (h * w * file_channels color)) (int_range 0 top)
  in
  let* filters = array ~size:(constant 5) (int_range 0 4) in
  let+ samples, trns = trns_of ~color ~depth ~entries samples in
  { color; depth; interlaced; h; w; samples; palette; trns; filters }

let built_pngs =
  Gen.with_pp
    (fun ppf p ->
      Format.fprintf ppf "colour type %d of %d bits, %d by %d%s%s" p.color
        p.depth p.h p.w
        (if p.interlaced then ", Adam7" else "")
        (match p.trns with
        | None -> ""
        | Some t -> Printf.sprintf ", tRNS of %d values" (Array.length t)))
    built_pngs

(* The colour types and depths are drawn with p = 1/15 each, the rest with p =
   0.07 or more a case (a palette longer than its tRNS): at 400 cases a run
   misses one with (14/15)^400 = 1.0e-12. *)
let png_law =
  prop ~count:400 "a PNG built from PNG-3 loads as the pixels its samples state"
    built_pngs (fun p ->
      List.iter
        (fun (c, d) ->
          cover
            (Printf.sprintf "colour type %d of %d bits" c d)
            (p.color = c && p.depth = d))
        color_depths;
      cover "Adam7 with an empty pass" (p.interlaced && (p.w < 5 || p.h < 5));
      cover "a sample value in tRNS" (p.color <> 3 && Option.is_some p.trns);
      cover "a 16-bit neighbour of tRNS's value"
        (p.color <> 3 && p.depth = 16 && Option.is_some p.trns && p.h * p.w > 1);
      cover "a palette longer than its tRNS"
        (p.color = 3
        && Option.exists
             (fun t -> Array.length t < Array.length p.palette / 3)
             p.trns);
      cover "a palette as long as its tRNS"
        (p.color = 3
        && Option.exists
             (fun t -> Array.length t = Array.length p.palette / 3)
             p.trns);
      let path = file ~suffix:".png" (png_file p) in
      equal packed (Nx.P (pixels p)) (Nx.P (Nx_io.load_image path)))

(* JPEG *)

(* Images of noise, grey or colour of grey pixels, of 1 to 24 rows and columns:
   one row, sides a multiple of 8, whose blocks lie whole in them, or any. *)
let block_images =
  let open Gen in
  let* colour = bool in
  let side =
    frequency
      [ (1, constant 1); (2, of_list [ 8; 16; 24 ]); (2, int_range 1 24) ]
  in
  let* h, w = pair side side in
  let+ pixels = array ~size:(constant (h * w)) (int_range 0 255) in
  let x = Nx.create Nx.uint8 [| h; w; 1 |] pixels in
  if colour then Nx.broadcast_to [| h; w; 3 |] x else x

let block_images = Gen.with_pp Nx.pp block_images

(* The root mean square of the differences of two images' samples. *)
let rms a b =
  let e = errors a b in
  Float.sqrt
    (Array.fold_left (fun s e -> s +. Float.of_int (e * e)) 0. e
    /. Float.of_int (Array.length e))

(* save_jpeg's luminance table, T.81's Table K.1 scaled to 20%: q / 5 is never a
   half, so its rounding is round to nearest. *)
let luminance =
  Array.concat
    [
      [| 16; 11; 10; 16; 24; 40; 51; 61 |];
      [| 12; 12; 14; 19; 26; 58; 60; 55 |];
      [| 14; 13; 16; 24; 40; 57; 69; 56 |];
      [| 14; 17; 22; 29; 51; 87; 80; 62 |];
      [| 18; 22; 37; 56; 68; 109; 103; 77 |];
      [| 24; 35; 55; 64; 81; 104; 113; 92 |];
      [| 49; 64; 78; 87; 103; 121; 120; 101 |];
      [| 72; 92; 95; 98; 112; 100; 103; 99 |];
    ]

(* Table K.2. *)
let chrominance =
  Array.init 64 (fun i ->
      let u = i / 8 and v = i mod 8 in
      match (u, v) with
      | 0, 0 -> 17
      | 0, 1 | 1, 0 -> 18
      | 0, 2 | 2, 0 -> 24
      | 0, 3 | 3, 0 -> 47
      | 1, 1 -> 21
      | 1, 2 | 2, 1 -> 26
      | 1, 3 | 3, 1 -> 66
      | 2, 2 -> 56
      | _ -> 99)

let scaled table = Array.map (fun q -> Int.max 1 ((q + 2) / 5)) table

(* The root mean square error of an image of [h] rows and [w] columns, by
   Parseval: T.81 A.3.3's transform is orthonormal, and save_jpeg rounds each
   coefficient to an integer and then to a multiple of its step Q, so it errs by
   at most Q / 2 + 1 / 2, and a block's samples by sqrt (sum (Q / 2 + 1 / 2)^2 /
   64) in root mean square, 7.14 for [luminance] at 20%. The transforms'
   fixed-point coefficients lie within 2.6e-5 of the exact ones, an operator of
   norm 1.41e-4, which errs on a block of samples within 128, of norm 1024, by
   1.41e-4 x 1024 / 8 = 0.018 in root mean square, 0.04 for both transforms. The
   image's samples lie in [ceil (h / 8) x ceil (w / 8)] blocks, each padded to
   64 samples, whose error energy they can only share: their root mean square
   error is at most a block's times sqrt (64 blocks / (h w)). The decoder's
   rounding to an integer adds 1 / 2, and clamping to a byte only shrinks an
   error. A colour image of grey pixels has chroma of 128 alone, which every
   step keeps exactly. *)
let grey_bound h w =
  let q = scaled luminance in
  let block =
    Float.sqrt
      (Array.fold_left
         (fun s q -> s +. (((Float.of_int q /. 2.) +. 0.5) ** 2.))
         0. q
      /. 64.)
  in
  let blocks = (h + 7) / 8 * ((w + 7) / 8) in
  (block +. 0.04)
  *. Float.sqrt (Float.of_int (64 * blocks) /. Float.of_int (h * w))
  +. 0.5

(* T.81 Figure A.6: the zig-zag order of a block's coefficients, as indices in
   row-major order. *)
let zigzag =
  Array.concat
    [
      [| 0; 1; 8; 16; 9; 2; 3; 10 |];
      [| 17; 24; 32; 25; 18; 11; 4; 5 |];
      [| 12; 19; 26; 33; 40; 48; 41; 34 |];
      [| 27; 20; 13; 6; 7; 14; 21; 28 |];
      [| 35; 42; 49; 56; 57; 50; 43; 36 |];
      [| 29; 22; 15; 23; 30; 37; 44; 51 |];
      [| 58; 59; 52; 45; 38; 31; 39; 46 |];
      [| 53; 60; 61; 54; 47; 55; 62; 63 |];
    ]

(* The segments of a JPEG file [s] up to its first scan, as markers and data. *)
let segments s =
  let rec loop i acc =
    if i + 4 > String.length s then List.rev acc
    else
      let marker = Char.code s.[i + 1] in
      let n = String.get_uint16_be s (i + 2) in
      let acc = (marker, String.sub s (i + 4) (n - 2)) :: acc in
      if marker = 0xda then List.rev acc else loop (i + 2 + n) acc
  in
  loop 2 []

(* The quantisation tables of DQT segments, by their identifiers, entries in
   zig-zag order (B.2.4.1). *)
let tables segs =
  List.concat_map
    (fun (marker, data) ->
      if marker <> 0xdb then []
      else
        List.init
          (String.length data / 65)
          (fun k ->
            let at = 65 * k in
            ( Char.code data.[at] land 0xf,
              Array.init 64 (fun i -> Char.code data.[at + 1 + i]) )))
    segs

(* A smooth colour image: channel [c] of the pixel at [y], [x] is [a + 60 c + sy
   y + sx x], clamped to a byte. *)
let smooth h w (a, sy, sx) =
  Nx.init Nx.uint8 [| h; w; 3 |] (fun i ->
      Int.max 0 (Int.min 255 (a + (60 * i.(2)) + (sy * i.(0)) + (sx * i.(1)))))

(* Files of both formats, sequential and progressive or interlaced. *)
let ended =
  [
    "png_filters.png";
    "png_palette_adam7.png";
    "jpeg_baseline.jpg";
    "jpeg_progressive.jpg";
  ]

(* Files other encoders wrote, for the cuts and flips of every byte. *)
let damaged =
  [
    "png_filters.png";
    "png_palette_adam7.png";
    "jpeg_baseline.jpg";
    "jpeg_progressive.jpg";
    "jpeg_restart.jpg";
    "jpeg_cmyk.jpg";
    "jpeg_extended.jpg";
  ]

(* The format a save writes. *)
type format = Png | Jpeg

let images =
  group "images"
    [
      cases
        "each save writes its format and load_image reads the contents, \
         whatever the extension"
        ~id:Fun.id [ ".png"; ".jpg"; ".JPEG"; ".bmp"; "" ] (fun ext ->
          let x = Nx.full Nx.uint8 [| 4; 4; 3 |] 7 in
          let png = temp_file ~suffix:ext () in
          let jpeg = temp_file ~suffix:ext () in
          let reference = temp_file ~suffix:".jpg" () in
          Nx_io.save_png png x;
          Nx_io.save_jpeg jpeg x;
          Nx_io.save_jpeg reference x;
          equal string (Nx_io.encode_png x) (read png);
          equal string (read reference) (read jpeg);
          equal packed (Nx.P x) (Nx.P (Nx_io.load_image png));
          equal packed
            (Nx.P (Nx_io.load_image reference))
            (Nx.P (Nx_io.load_image jpeg)));
      cases
        "a shape the format does not hold is refused, and nothing is written"
        ~id:fst
        [
          ("a vector in PNG", (Png, [| 4 |]));
          ("no channel axis in PNG", (Png, [| 2; 2 |]));
          ("four axes in PNG", (Png, [| 2; 2; 3; 1 |]));
          ("five channels in PNG", (Png, [| 2; 2; 5 |]));
          ("no rows in PNG", (Png, [| 0; 3; 3 |]));
          ("no channels in PNG", (Png, [| 2; 2; 0 |]));
          ("no channel axis in JPEG", (Jpeg, [| 2; 2 |]));
          ("two channels in JPEG", (Jpeg, [| 2; 2; 2 |]));
          ("four channels in JPEG", (Jpeg, [| 2; 2; 4 |]));
          ("no columns in JPEG", (Jpeg, [| 3; 0; 1 |]));
          ("no channels in JPEG", (Jpeg, [| 2; 2; 0 |]));
          ("65536 rows in JPEG", (Jpeg, [| 65536; 1; 1 |]));
          ("65536 columns in JPEG", (Jpeg, [| 1; 65536; 1 |]));
        ]
        (fun (_, (format, shape)) ->
          let x = Nx.zeros Nx.uint8 shape in
          let dir = temp_dir () in
          let path = Filename.concat dir "x" in
          (match format with
          | Png ->
              invalid ~by:"Nx_io.save_png" (fun () -> Nx_io.save_png path x);
              invalid ~by:"Nx_io.encode_png" (fun () -> Nx_io.encode_png x)
          | Jpeg ->
              invalid ~by:"Nx_io.save_jpeg" (fun () -> Nx_io.save_jpeg path x));
          equal (array string) [||] (Sys.readdir dir));
      cases
        "a side past the largest PNG integer is refused, before a pixel is read"
        ~id:fst
        [
          ("rows", [| 2147483648; 1; 1 |]); ("columns", [| 1; 2147483648; 1 |]);
        ]
        (fun (_, shape) ->
          (* A view of one pixel: the refusal reads none of its elements. *)
          let x = Nx.broadcast_to shape (Nx.zeros Nx.uint8 [| 1; 1; 1 |]) in
          let dir = temp_dir () in
          let path = Filename.concat dir "x.png" in
          let side by =
            Exn.invalid_arg
              ~sub:
                (by ^ ": "
                ^ String.concat " by "
                    (List.map string_of_int [ shape.(0); shape.(1) ]))
          in
          raises_match (side "Nx_io.save_png") (fun () -> Nx_io.save_png path x);
          raises_match (side "Nx_io.encode_png") (fun () ->
              ignore (Nx_io.encode_png x));
          equal (array string) [||] (Sys.readdir dir));
      cases "a JPEG of 65535 rows or columns, its largest, loads back" ~id:fst
        [ ("rows", [| 65535; 1; 1 |]); ("columns", [| 1; 65535; 1 |]) ]
        (fun (_, shape) ->
          let path = temp_file ~suffix:".jpg" () in
          Nx_io.save_jpeg path (Nx.full Nx.uint8 shape 7);
          equal (array int) shape (Nx.shape (Nx_io.load_image path)));
    ]

let load =
  group "load_image"
    [
      png_law;
      test "a palette's tRNS longer than the palette fails" (fun () ->
          (* PNG-3 11.3.1.1: "The tRNS chunk shall not contain more alpha values
             than there are palette entries". *)
          let p =
            {
              color = 3;
              depth = 8;
              interlaced = false;
              h = 1;
              w = 2;
              samples = [| 0; 1 |];
              palette = [| 1; 2; 3; 4; 5; 6 |];
              trns = Some [| 0; 0; 0 |];
              filters = [| 0 |];
            }
          in
          let path = file (png_file p) in
          fails ~path (fun () -> Nx_io.load_image path));
      cases
        "files other encoders wrote load with the file's channels, as Pillow \
         decodes them"
        ~id:fst
        (* PNG decodes exactly. T.81 leaves the inverse transform's precision
           and the colour conversion to the decoder, so JPEG rows agree with
           Pillow's libjpeg-turbo within 3 levels, the largest difference seen
           on these files: a baseline, not a bound. *)
        (List.map
           (fun n -> ("png_" ^ n ^ ".png", 0))
           [
             "filters";
             "gray1";
             "gray4";
             "rgb16";
             "palette_adam7";
             "palette_short_trns";
             "rgba";
             "la";
             "gray_trns";
             "rgb16_trns";
           ]
        @ List.map
            (fun n -> ("jpeg_" ^ n ^ ".jpg", 3))
            [
              "baseline";
              "progressive";
              "progressive_dark";
              "progressive_restart";
              "gray";
              "late_table";
              "cmyk";
              "rgb";
              "444";
              "422";
              "restart";
              "extended";
              "table16_coarse";
            ])
        (fun (name, tolerance) ->
          let expected = expected name in
          let x = Nx_io.load_image (fixture name) in
          equal ~msg:"shape" (array int) (Nx.shape expected) (Nx.shape x);
          at_most ~msg:"largest error" int ~than:tolerance
            (Array.fold_left Int.max 0 (errors expected x)));
      cases
        "a JPEG whose 16-bit table scales its DC past a byte loads as its \
         inverse DCT's values, clamped"
        ~id:fst
        [ ("jpeg_table16.jpg", 32); ("jpeg_table16_large.jpg", 512) ]
        (fun (name, side) ->
          (* gen/generate.py's overflow_jpeg: each block has a DC coefficient
             alone, the sum of its differences of 2047 wrapped to 16 bits, which
             the table's 65535 scales far past a byte, so a DC-only block's
             samples (T.81 A.3.3) clamp to 255 or 0 by its sign. libjpeg masks
             its range-limit index instead, so Pillow differs. *)
          let blocks = side / 8 in
          let expected =
            Nx.init Nx.uint8 [| side; side; 1 |] (fun i ->
                let b = (i.(0) / 8 * blocks) + (i.(1) / 8) in
                let dc = ((((b + 1) * 2047) + 32768) land 0xffff) - 32768 in
                if dc > 0 then 255 else 0)
          in
          equal packed (Nx.P expected) (Nx.P (Nx_io.load_image (fixture name))));
      test
        "a progressive JPEG whose refinement skips 16 zeros decodes as its \
         baseline twin" (fun () ->
          equal packed
            (Nx.P (Nx_io.load_image (fixture "jpeg_zrl_baseline.jpg")))
            (Nx.P (Nx_io.load_image (fixture "jpeg_zrl_progressive.jpg"))));
      prop "bytes after a PNG's IEND or a JPEG's EOI are ignored"
        ~examples:(List.map (fun name -> (name, read (fixture name))) ended)
        (Gen.pair (Gen.of_list ~pp:Format.pp_print_string ended) Gen.string)
        (fun (name, tail) ->
          let path = file (read (fixture name) ^ tail) in
          equal packed
            (Nx.P (Nx_io.load_image (fixture name)))
            (Nx.P (Nx_io.load_image path)));
      prop "bytes after the zlib stream of a PNG's image data are ignored"
        (Gen.pair (Gen.of_list ~pp:Format.pp_print_string pngs) Gen.string)
        (fun (name, tail) ->
          cover "bytes after the stream" (tail <> "");
          let png = read (fixture name) in
          let idat = Lz.Zlib.encode_string (scanlines png) ^ tail in
          equal packed
            (Nx.P (Nx_io.load_image (fixture name)))
            (Nx.P (Nx_io.load_image (file (with_idat png idat)))));
      test
        "a PNG whose scanlines take more than 1032 times its image data is \
         refused at its header, and one at that bound at its inflate" (fun () ->
          (* One row of grey of [w] samples takes [w + 1] bytes with its filter
             byte; a zlib stream inflates each byte to at most 1032. Both
             refusals are Failures, told apart by their words, a baseline. *)
          let n = String.length (Lz.Zlib.encode_string "") in
          let refusal over =
            let path =
              file (header_png ~width:((1032 * n) - 1 + over) ~height:1)
            in
            within_budget @@ fun () ->
            match Nx_io.load_image path with
            | _ -> "loaded"
            | exception Failure m ->
                let k = String.length path + 2 in
                String.sub m k (String.length m - k)
          in
          expect (Printf.sprintf "past: %s\nat: %s" (refusal 1) (refusal 0))
          @@ __POS_OF__
               {|past: PNG image data too short for its dimensions
at: invalid PNG zlib stream: 0 bytes of scanlines, the image has 8256|});
      test
        "a PNG whose header states more pixels than its image data holds is \
         refused before its pixels are allocated" (fun () ->
          (* Width past 2^30 and height past 2^20: petabytes of pixels from a
             few hundred bytes of image data. *)
          let png = read (fixture "png_filters.png") in
          let png = flipped png (Chunk (0, 4)) 0x40 in
          let png = flipped png (Chunk (0, 9)) 0x10 in
          let path = file png in
          within_budget @@ fun () ->
          fails ~path ~naming:"PNG image data too short for its dimensions"
            (fun () -> Nx_io.load_image path));
      test
        "a JPEG whose frame states more blocks than its data can code is \
         refused" (fun () ->
          (* The frame's sides, at bytes 163 to 166, set to 65535: 2^26 luma
             blocks from 644 bytes. *)
          let jpeg = Bytes.of_string (read (fixture "jpeg_baseline.jpg")) in
          Bytes.fill jpeg 163 4 '\xff';
          let path = file (Bytes.to_string jpeg) in
          within_budget @@ fun () ->
          fails ~path ~naming:"JPEG data too short for its dimensions"
            (fun () -> Nx_io.load_image path));
      cases "a PNG side or chunk length past the largest PNG integer fails"
        ~id:fst
        [
          ( "a width of 2^31",
            (header_png ~width:0x80000000 ~height:1, "invalid PNG image header")
          );
          ( "a height of 2^31",
            (header_png ~width:1 ~height:0x80000000, "invalid PNG image header")
          );
          ( "a chunk length of 2^31",
            ( signature ^ ihdr ~width:1 ~height:1 ^ be32 0x80000000 ^ "IDAT"
              ^ be32 0,
              "invalid PNG chunk length" ) );
        ]
        (fun (_, (png, naming)) ->
          let path = file png in
          fails ~path ~naming (fun () -> Nx_io.load_image path));
      cases "a checksum that does not match fails" ~id:fst
        [
          ("a chunk's CRC-32", flip_byte (read (fixture "png_filters.png")) 30);
          ( "the image data's Adler-32",
            let png = read (fixture "png_filters.png") in
            let z = Lz.Zlib.encode_string (scanlines png) in
            with_idat png (xor_byte z (String.length z - 1) 1) );
        ]
        (fun (_, png) ->
          let path = file png in
          fails ~path (fun () -> Nx_io.load_image path));
      cases
        "PNG image data that decompresses to other than the image's scanlines \
         fails"
        ~id:fst
        [
          ("one byte short", fun s -> String.sub s 0 (String.length s - 1));
          ("one byte over", fun s -> s ^ "\000");
          ("none", fun _ -> "");
        ]
        (fun (_, change) ->
          let png = read (fixture "png_filters.png") in
          let idat = Lz.Zlib.encode_string (change (scanlines png)) in
          let path = file (with_idat png idat) in
          fails ~path ~naming:"invalid PNG zlib stream" (fun () ->
              Nx_io.load_image path));
      cases
        "a progressive JPEG that sends a coefficient's bit twice or out of \
         order fails"
        ~id:fst
        [
          ("a scan repeated", "jpeg_repeated_scan.jpg");
          ("a refinement of bits never sent", "jpeg_unsent_refinement.jpg");
          ("an AC scan before the DC scan", "jpeg_ac_before_dc.jpg");
        ]
        (fun (_, name) ->
          let path = fixture name in
          fails ~path ~naming:"invalid JPEG scan" (fun () ->
              Nx_io.load_image path));
      cases "a JPEG whose DC values outgrow 16 bits loads or fails" ~id:Fun.id
        [ "jpeg_dc_overflow.jpg"; "jpeg_dc_overflow_progressive.jpg" ]
        (fun name ->
          let path = fixture name in
          loads_or_fails ~path (fun () -> Nx_io.load_image path));
      cases "a file that is neither PNG nor JPEG fails" ~id:String.escaped
        [ ""; "GIF89a"; "\137PN" ] (fun s ->
          let path = file s in
          fails ~path (fun () -> Nx_io.load_image path));
      test "a directory fails, naming it" (fun () ->
          let path = temp_dir () in
          fails ~path (fun () -> Nx_io.load_image path));
      cases "every proper prefix of a file fails" ~id:Fun.id damaged
        (fun name -> prefixes_fail Nx_io.load_image (read (fixture name)));
      cases "a file with any byte flipped loads or fails" ~id:Fun.id damaged
        (fun name -> flips_load_or_fail Nx_io.load_image (read (fixture name)));
      prop
        "a PNG with a byte changed and its checksums recomputed loads or fails"
        png_flips (fun (name, flip, mask) ->
          cover "a chunk's type or data"
            (match flip with Chunk _ -> true | Scanline _ -> false);
          cover "a scanline"
            (match flip with Scanline _ -> true | Chunk _ -> false);
          let path = file (flipped (read (fixture name)) flip mask) in
          within_budget @@ fun () ->
          loads_or_fails ~path (fun () -> Nx_io.load_image path));
      prop "any bytes after a PNG or a JPEG signature load or fail"
        (Gen.pair
           (Gen.of_list ~pp:Format.pp_print_string [ "PNG"; "JPEG" ])
           Gen.string)
        (fun (format, bytes) ->
          let signature = if format = "PNG" then signature else "\xff\xd8" in
          let path = file (signature ^ bytes) in
          within_budget @@ fun () ->
          loads_or_fails ~path (fun () -> Nx_io.load_image path));
      test "a loaded image's donation writes in place" (fun () ->
          let n = 256 in
          let path = temp_file ~suffix:".png" () in
          Nx_io.save_png path
            (Nx.create Nx.uint8 [| n; n; 3 |] (Array.make (n * n * 3) 7));
          let y, grew =
            grown Rig.host
              (fun () -> Nx_io.load_image path)
              (fun x -> Nx.neg (Nx.donate x))
          in
          less ~msg:"no memory of the result's size" int ~than:(n * n * 3) grew;
          equal (array int) [| 249 |]
            (Nx.to_array (Nx.slice [ Nx.I 0; Nx.I 0; Nx.R (0, 1) ] y)));
    ]

(* Drawn images *)

(* Images of [channels], of 1 to 8 rows and columns, with any pixels, in any
   view. *)
let drawn_images channels =
  let open Gen in
  let* c = of_list channels in
  let* h, w = pair (int_range 1 8) (int_range 1 8) in
  let shape = [| h; w; c |] in
  let n = Array.fold_left ( * ) 1 shape in
  let* pixels = array ~size:(constant n) (int_range 0 255) in
  let+ view = of_list [ Plain; Transposed; Flipped; Broadcast ] in
  let x = Nx.create Nx.uint8 shape pixels in
  let x =
    match view with
    | Plain -> x
    | Transposed -> Nx.transpose ~axes:[ 1; 0; 2 ] x
    | Flipped -> Nx.flip ~axes:[ 0 ] x
    | Broadcast -> Nx.broadcast_to shape (Nx.slice [ Nx.R (0, 1) ] x)
  in
  (view, x)

let drawn_images channels =
  Gen.with_pp (fun ppf (_, x) -> Nx.pp ppf x) (drawn_images channels)

let save_png =
  group "save_png"
    [
      cases "writes 8 bits per sample in the colour type of its channels"
        ~id:(fun (c, _) -> Printf.sprintf "%d channels" c)
        (* PNG-3 Table 11.1: grey 0, grey and alpha 4, RGB 2, RGBA 6. *)
        [ (1, 0); (2, 4); (3, 2); (4, 6) ]
        (fun (c, color) ->
          let header =
            List.hd
              (chunk_data "IHDR"
                 (Nx_io.encode_png (Nx.zeros Nx.uint8 [| 2; 3; c |])))
          in
          equal (pair int int) (8, color)
            (Char.code header.[8], Char.code header.[9]));
      prop
        "writes encode_png's bytes, with or without ~dpi and ~srgb, which load \
         back as the image, of every channel count"
        (Gen.triple
           (drawn_images [ 1; 2; 3; 4 ])
           (Gen.option (Gen.float_range 1. 1000.))
           (Gen.option Gen.bool))
        (fun ((view, x), dpi, srgb) ->
          let rows = Nx.dim 0 x and columns = Nx.dim 1 x in
          cover "strided"
            ((view = Transposed && rows > 1 && columns > 1)
            || (view = Flipped && rows > 1));
          cover "broadcast" (view = Broadcast && rows > 1);
          let channels = Nx.dim 2 x in
          cover "grey" (channels = 1);
          cover "grey and alpha" (channels = 2);
          cover "colour" (channels = 3);
          cover "colour and alpha" (channels = 4);
          cover "~dpi" (Option.is_some dpi);
          cover "~srgb:true" (srgb = Some true);
          let save x =
            let path =
              saved ".png" (fun p x -> Nx_io.save_png ?dpi ?srgb p x) x
            in
            equal ~msg:"the file" string
              (Nx_io.encode_png ?dpi ?srgb x)
              (read path);
            path
          in
          Law.round_trip
            (Testable.contramap (fun x -> Nx.P x) packed)
            string save Nx_io.load_image x);
      test "an image whose file takes several IDAT chunks returns" (fun () ->
          let random = Random.State.make [| 0x6e78 |] in
          let noise =
            Nx.init Nx.uint8 [| 600; 640; 3 |] (fun _ ->
                Random.State.int random 256)
          in
          let path = temp_file ~suffix:".png" () in
          Nx_io.save_png path noise;
          greater ~msg:"IDAT chunks" int ~than:1
            (List.length (chunk_data "IDAT" (read path)));
          equal packed (Nx.P noise) (Nx.P (Nx_io.load_image path)));
      test
        "an image in host memory saves holding about 3 MiB beyond it, no copy \
         of it" (fun () ->
          (* The host's budget holds 8 MiB past what it holds now: the 3 MiB and
             a row the section states, and room, half the image. *)
          let x =
            Nx.create Nx.uint8 [| 2048; 2048; 4 |] (Array.make (1 lsl 24) 7)
          in
          let path = temp_file ~suffix:".png" () in
          let old = Rig.budget Rig.host in
          Rig.set_budget Rig.host (Rig.allocated Rig.host + (8 lsl 20));
          Fun.protect
            ~finally:(fun () -> Rig.set_budget Rig.host old)
            (fun () -> Nx_io.save_png path x);
          equal (array int) [| 2048; 2048; 4 |]
            (Nx.shape (Nx_io.load_image path)));
      test "leaves its image's memory to the image's next donation" (fun () ->
          let n = 256 in
          (* An image made from OCaml data, in memory of its own. *)
          let make () =
            Nx.create Nx.uint8 [| n; n; 3 |] (Array.make (n * n * 3) 7)
          in
          let saved_then_donated x =
            Nx_io.save_png (temp_file ~suffix:".png" ()) x;
            Nx.neg (Nx.donate x)
          in
          let y, grew = grown Rig.host make saved_then_donated in
          less ~msg:"no memory of the result's size" int ~than:(n * n * 3) grew;
          equal (array int) [| 249 |]
            (Nx.to_array (Nx.slice [ Nx.I 0; Nx.I 0; Nx.R (0, 1) ] y)));
    ]

let save_jpeg =
  group "save_jpeg"
    [
      cases "writes a baseline frame with T.81's Annex K tables at 20%" ~id:fst
        [
          ("grey", (1, [ (0, scaled luminance) ], [ 0x11 ]));
          ( "colour",
            ( 3,
              [ (0, scaled luminance); (1, scaled chrominance) ],
              [ 0x22; 0x11; 0x11 ] ) );
        ]
        (fun (_, (c, expected_tables, sampling)) ->
          let path = temp_file ~suffix:".jpg" () in
          Nx_io.save_jpeg path (Nx.zeros Nx.uint8 [| 16; 16; c |]);
          let segs = segments (read path) in
          let frame =
            match List.assoc_opt 0xc0 segs with
            | Some f -> f
            | None -> failf "no baseline frame (SOF0)"
          in
          equal ~msg:"components' sampling" (list int) sampling
            (List.init
               (Char.code frame.[5])
               (fun k -> Char.code frame.[7 + (3 * k)]));
          let in_zigzag (id, t) = (id, Array.map (fun i -> t.(i)) zigzag) in
          equal ~msg:"tables in zig-zag order"
            (list (pair int (array int)))
            (List.map in_zigzag expected_tables)
            (tables segs));
      (* Each kind is drawn with p = 1/2 a case, one row with p = 0.2, whole
         blocks with p = 0.45^2 = 0.2: at 150 cases a run misses one with
         0.8^150 = 2.9e-15. *)
      prop ~count:150
        "a grey image returns within the root mean square error its \
         quantisation allows"
        block_images (fun x ->
          let h = Nx.dim 0 x and w = Nx.dim 1 x in
          cover "grey" (Nx.dim 2 x = 1);
          cover "colour of grey pixels" (Nx.dim 2 x = 3);
          cover "one row" (h = 1);
          cover "whole blocks" (h mod 8 = 0 && w mod 8 = 0);
          cover "a partial block" (h mod 8 <> 0 || w mod 8 <> 0);
          let path = temp_file ~suffix:".jpg" () in
          Nx_io.save_jpeg path x;
          let back = Nx_io.load_image path in
          equal (array int) (Nx.shape x) (Nx.shape back);
          at_most float_exact ~than:(grey_bound h w) (rms x back));
      test "smooth colour images return with this baseline's mean errors"
        (fun () ->
          (* Chroma's table at 20% and 4:2:0 bound no error a reader can use:
             this baseline holds the mean errors on three ramps. *)
          let errs =
            List.map
              (fun (h, w, ramp) ->
                let x = smooth h w ramp in
                let path = temp_file ~suffix:".jpg" () in
                Nx_io.save_jpeg path x;
                let e = errors x (Nx_io.load_image path) in
                Float.of_int (Array.fold_left ( + ) 0 e)
                /. Float.of_int (Array.length e))
              [
                (24, 24, (40, 3, -2)); (17, 9, (0, 6, 6)); (1, 24, (200, 0, -6));
              ]
          in
          expect (String.concat " " (List.map (Printf.sprintf "%.2f") errs))
          @@ __POS_OF__ {|0.17 0.38 0.61|});
    ]

let encode_png =
  group "encode_png"
    [
      test "writes neither pHYs nor sRGB by default" (fun () ->
          let png = Nx_io.encode_png tiny in
          equal (list string) [] (chunk_data "pHYs" png);
          equal (list string) [] (chunk_data "sRGB" png));
      cases "~dpi writes the pixels per metre on both axes" ~id:fst
        (* PNG-3 11.3.4.3: unit 1 "unit is the metre". *)
        [
          ("72 dpi is 2835 pixels per metre", (72., 2835));
          ("144 dpi is 5669 pixels per metre", (144., 5669));
          ("0.0127 dpi rounds up to 1", (0.0127, 1));
          ("the largest PNG integer", (2147483647. *. 0.0254, 2147483647));
        ]
        (fun (_, (dpi, ppm)) ->
          equal (list string)
            [ be32 ppm ^ be32 ppm ^ "\001" ]
            (chunk_data "pHYs" (Nx_io.encode_png ~dpi tiny)));
      test "~srgb writes one sRGB chunk with the perceptual intent" (fun () ->
          (* PNG-3 11.3.2.5, Table 16: intent 0 is "Perceptual". *)
          equal (list string) [ "\000" ]
            (chunk_data "sRGB" (Nx_io.encode_png ~srgb:true tiny)));
      prop
        "~dpi and ~srgb add their chunks before the image data and change no \
         other"
        (Gen.pair (drawn_images [ 1; 2; 3; 4 ]) (Gen.float_range 1. 1000.))
        (fun ((_, x), dpi) ->
          let plain = Nx_io.encode_png x in
          let png = Nx_io.encode_png ~dpi ~srgb:true x in
          equal string plain (without "sRGB" (without "pHYs" png));
          (* PNG-3 places both chunks before the first IDAT. *)
          let rec before_idat = function
            | [] | ("IDAT", _) :: _ -> []
            | (ty, _) :: rest -> ty :: before_idat rest
          in
          equal ~msg:"chunks before IDAT"
            (slist string String.compare)
            [ "IHDR"; "pHYs"; "sRGB" ]
            (before_idat (chunks png)));
      cases "~dpi outside the range of a PNG integer is refused" ~id:fst
        [
          ("zero", 0.);
          ("a negative", -72.);
          ("one rounding to zero pixels per metre", 0.0126);
          ("one rounding past the largest PNG integer", 2147483648. *. 0.0254);
          ("an infinity", infinity);
          ("nan", nan);
        ]
        (fun (_, dpi) ->
          invalid ~by:"Nx_io.encode_png" (fun () -> Nx_io.encode_png ~dpi tiny);
          let dir = temp_dir () in
          let path = Filename.concat dir "x.png" in
          invalid ~by:"Nx_io.save_png" (fun () -> Nx_io.save_png ~dpi path tiny);
          equal (array string) [||] (Sys.readdir dir));
    ]

let () =
  exit (run "nx.io images" [ images; load; save_png; save_jpeg; encode_png ])
