(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Images. A PNG saved from any image loads back as its pixels, a smooth JPEG
   within its quality, and files other encoders wrote as Pillow decodes them, or
   as their format states them where Pillow differs; a file cut short or
   corrupted raises [Failure] naming it, and never anything else. *)

open Windtrap
open Io_support

(* The pixels of the fixture image [name], as gen/generate.py states them. *)
let expected name =
  let npy = Filename.remove_extension name ^ ".npy" in
  Nx.unpack Nx.uint8 (Nx_io.load_npy (fixture npy))

(* The pixels of gen/generate.py's overflow_jpeg file of [side] pixels and
   16-bit tables, as the inverse DCT gives them. Each block has only a DC
   coefficient, the sum of its differences of 2047 wrapped to 16 bits, and the
   table's 65535 scales it far past a byte, so its samples are 255 or 0 by the
   DC's sign. libjpeg masks its range-limit index instead of clamping, so
   Pillow's pixels differ. *)
let table16_pixels side =
  let blocks = side / 8 in
  Nx.init Nx.uint8 [| side; side; 1 |] (fun i ->
      let b = (i.(0) / 8 * blocks) + (i.(1) / 8) in
      let dc = ((((b + 1) * 2047) + 32768) land 0xffff) - 32768 in
      if dc > 0 then 255 else 0)

(* Drawn images *)

(* Images of [channels], of 1 to 8 rows and columns, with any pixels, in any
   view. *)
let images channels =
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

let images channels =
  Gen.with_pp (fun ppf (_, x) -> Nx.pp ppf x) (images channels)

let covers (view, x) =
  let rows = Nx.dim 0 x and columns = Nx.dim 1 x in
  cover "strided"
    ((view = Transposed && rows > 1 && columns > 1)
    || (view = Flipped && rows > 1));
  cover "broadcast" (view = Broadcast && rows > 1)

(* Smooth images, which JPEG keeps closely: channel [c] of the pixel at [y], [x]
   is [a + 60 c + sy y + sx x], clamped to a byte, for drawn [a] and slopes.
   Sides run from 1 to 24 pixels, across a block of 8 and a chroma block of 16,
   and one image in six is a single row. *)
let smooth channels =
  let open Gen in
  let side = int_range 1 24 in
  let slope = int_range (-6) 6 in
  let* c = of_list channels in
  let+ h = frequency [ (1, constant 1); (5, side) ]
  and+ w = side
  and+ a, sy, sx = triple (int_range 0 255) slope slope in
  Nx.init Nx.uint8 [| h; w; c |] (fun i ->
      Int.max 0 (Int.min 255 (a + (60 * i.(2)) + (sy * i.(0)) + (sx * i.(1)))))

let smooth channels = Gen.with_pp Nx.pp (smooth channels)

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

let png_chunks =
  group "PNG chunks"
    [
      test "encode_png writes neither pHYs nor sRGB by default" (fun () ->
          let png = Nx_io.encode_png tiny in
          equal (list string) [] (chunk_data "pHYs" png);
          equal (list string) [] (chunk_data "sRGB" png));
      cases ~name:fst "~dpi writes the pixels per metre on both axes"
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
          equal (list string) [ "\000" ]
            (chunk_data "sRGB" (Nx_io.encode_png ~srgb:true tiny)));
      prop
        "with ~dpi and ~srgb the file is encode_png's with the two chunks \
         added, and loads back the same"
        (Gen.pair (images [ 1; 2; 3; 4 ]) (Gen.float_range 1. 1000.))
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
            (before_idat (chunks png));
          equal packed
            (Nx.P (Nx_io.load_image (file ~suffix:".png" plain)))
            (Nx.P (Nx_io.load_image (file ~suffix:".png" png))));
      cases ~name:fst "~dpi outside the range of a PNG integer is refused"
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

let round_trips =
  group "round trips"
    [
      prop
        "save_png writes encode_png's bytes, with or without ~dpi and ~srgb, \
         which load back as the image, of every channel count"
        (Gen.triple
           (images [ 1; 2; 3; 4 ])
           (Gen.option (Gen.float_range 1. 1000.))
           (Gen.option Gen.bool))
        (fun ((view, x), dpi, srgb) ->
          covers (view, x);
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
      prop "a smooth image returns from JPEG within a mean error of 8 levels"
        (smooth [ 1; 3 ])
        (fun x ->
          let h = Nx.dim 0 x and w = Nx.dim 1 x in
          cover "one row" (h = 1);
          cover "both sides below 16 pixels" (h < 16 && w < 16);
          cover "a side past 16 pixels" (h > 16 || w > 16);
          let path = temp_file ~suffix:".jpg" () in
          Nx_io.save_jpeg path x;
          let back = Nx_io.load_image path in
          equal (array int) (Nx.shape x) (Nx.shape back);
          let e = errors x back in
          at_most int ~than:8 (Array.fold_left ( + ) 0 e / Array.length e));
      test "an image of noise returns from PNG" (fun () ->
          let random = Random.State.make [| 0x6e78 |] in
          let noise =
            Nx.init Nx.uint8 [| 64; 64; 3 |] (fun _ ->
                Random.State.int random 256)
          in
          let path = temp_file ~suffix:".png" () in
          Nx_io.save_png path noise;
          equal packed (Nx.P noise) (Nx.P (Nx_io.load_image path)));
      test "a save leaves its image's memory to the image's next donation"
        (fun () ->
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

(* The format a save writes. *)
type format = Png | Jpeg

let files =
  group "files"
    [
      cases ~name:fst
        "files other encoders wrote load with the file's channels: alpha and \
         transparency kept, grey as one channel, CMYK as colour"
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
              "gray";
              "late_table";
              "cmyk";
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
      cases ~name:Fun.id
        "each save writes its format and load_image reads the contents, \
         whatever the extension" [ ".png"; ".jpg"; ".JPEG"; ".bmp"; "" ]
        (fun ext ->
          let x = Nx.full Nx.uint8 [| 4; 4; 3 |] 7 in
          let png = temp_file ~suffix:ext () in
          let jpeg = temp_file ~suffix:ext () in
          let reference = temp_file ~suffix:".jpg" () in
          Nx_io.save_png png x;
          Nx_io.save_jpeg jpeg x;
          Nx_io.save_jpeg reference x;
          equal string (Nx_io.encode_png x) (read png);
          starts_with ~affix:"\xff\xd8" (read jpeg);
          equal string (read reference) (read jpeg);
          equal packed (Nx.P x) (Nx.P (Nx_io.load_image png));
          equal packed
            (Nx.P (Nx_io.load_image reference))
            (Nx.P (Nx_io.load_image jpeg)));
      cases ~name:fst
        "a shape the format does not hold is refused, and nothing is written"
        [
          ("a vector in PNG", (Png, [| 4 |]));
          ("no channel axis in PNG", (Png, [| 2; 2 |]));
          ("four axes in PNG", (Png, [| 2; 2; 3; 1 |]));
          ("five channels in PNG", (Png, [| 2; 2; 5 |]));
          ("no rows in PNG", (Png, [| 0; 3; 3 |]));
          ("no channel axis in JPEG", (Jpeg, [| 2; 2 |]));
          ("two channels in JPEG", (Jpeg, [| 2; 2; 2 |]));
          ("four channels in JPEG", (Jpeg, [| 2; 2; 4 |]));
          ("no columns in JPEG", (Jpeg, [| 3; 0; 1 |]));
          ("no channels in PNG", (Png, [| 2; 2; 0 |]));
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
      cases ~name:fst
        "a side past the largest PNG integer is refused, before a pixel is read"
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
              ~substring:
                (by ^ ": "
                ^ String.concat " by "
                    (List.map string_of_int [ shape.(0); shape.(1) ]))
          in
          raises_match (side "Nx_io.save_png") (fun () -> Nx_io.save_png path x);
          raises_match (side "Nx_io.encode_png") (fun () ->
              ignore (Nx_io.encode_png x));
          equal (array string) [||] (Sys.readdir dir));
      cases ~name:fst "a JPEG of 65535 rows or columns, its largest, loads back"
        [ ("rows", [| 65535; 1; 1 |]); ("columns", [| 1; 65535; 1 |]) ]
        (fun (_, shape) ->
          let path = temp_file ~suffix:".jpg" () in
          Nx_io.save_jpeg path (Nx.full Nx.uint8 shape 7);
          equal (array int) shape (Nx.shape (Nx_io.load_image path)));
      cases ~name:Fun.id "a JPEG whose DC values outgrow 16 bits loads or fails"
        [ "jpeg_dc_overflow.jpg"; "jpeg_dc_overflow_progressive.jpg" ]
        (fun name ->
          let path = fixture name in
          loads_or_fails ~path (fun () -> Nx_io.load_image path));
      cases ~name:fst
        "a JPEG whose 16-bit table scales its DC past a byte loads as its \
         inverse DCT's values, clamped"
        [ ("jpeg_table16.jpg", 32); ("jpeg_table16_large.jpg", 512) ]
        (fun (name, side) ->
          equal packed
            (Nx.P (table16_pixels side))
            (Nx.P (Nx_io.load_image (fixture name))));
      cases ~name:fst
        "a progressive JPEG that sends a coefficient's bit twice or out of \
         order fails"
        [
          ("a scan repeated", "jpeg_repeated_scan.jpg");
          ("a refinement of bits never sent", "jpeg_unsent_refinement.jpg");
          ("an AC scan before the DC scan", "jpeg_ac_before_dc.jpg");
        ]
        (fun (_, name) ->
          let path = fixture name in
          fails ~path ~naming:"invalid JPEG scan" (fun () ->
              Nx_io.load_image path));
      test "a PNG chunk whose checksum does not match fails" (fun () ->
          let path = file (flip_byte (read (fixture "png_filters.png")) 30) in
          fails ~path ~naming:"PNG chunk CRC mismatch" (fun () ->
              Nx_io.load_image path));
      test "a file that is neither PNG nor JPEG fails" (fun () ->
          let path = file "GIF89a" in
          fails ~path (fun () -> Nx_io.load_image path));
      cases ~name:Fun.id
        "every proper prefix fails, and any byte flipped loads or fails"
        [
          "png_filters.png";
          "png_palette_adam7.png";
          "jpeg_baseline.jpg";
          "jpeg_progressive.jpg";
        ] (fun name ->
          let whole = read (fixture name) in
          prefixes_fail Nx_io.load_image whole;
          flips_load_or_fail Nx_io.load_image whole);
      prop "any bytes after a PNG or a JPEG signature load or fail"
        (Gen.pair
           (Gen.of_list ~pp:Format.pp_print_string [ "PNG"; "JPEG" ])
           Gen.string)
        (fun (format, bytes) ->
          let signature =
            if format = "PNG" then "\137PNG\r\n\026\n" else "\xff\xd8"
          in
          let path = file (signature ^ bytes) in
          loads_or_fails ~path (fun () -> Nx_io.load_image path));
    ]

let () = exit (run "nx.io images" [ png_chunks; round_trips; files ])
