(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Images. A PNG saved from any image loads back as its pixels, a smooth JPEG
   within its quality, and files other encoders wrote as Pillow decodes them; a
   file cut short or corrupted raises [Failure] naming it, and never anything
   else. *)

open Windtrap

let fixture name = Filename.concat "support" name
let read path = In_channel.with_open_bin path In_channel.input_all

let file ?(suffix = "") s =
  let path = temp_file ~suffix () in
  Out_channel.with_open_bin path (fun oc -> Out_channel.output_string oc s);
  path

let flip_byte s i =
  String.mapi
    (fun j c -> if i = j then Char.chr (Char.code c lxor 0xff) else c)
    s

let fails ?(naming = "") ~path f =
  raises_match
    (function
      | Failure m ->
          String.starts_with ~prefix:(path ^ ": ") m
          && (naming = "" || Exn.failure ~substring:naming (Failure m))
      | _ -> false)
    (fun () -> ignore (f ()))

let invalid ~by f =
  raises_match
    (Exn.invalid_arg ~substring:(by ^ ": "))
    (fun () -> ignore (f ()))

let loads_or_fails ~path f =
  match f () with
  | _ -> ()
  | exception Failure m when String.starts_with ~prefix:(path ^ ": ") m -> ()
  | exception e -> failf "%s raised %s" path (Printexc.to_string e)

(* [f (make ())] and the bytes the host holds more after it, read once
   collected memory has returned. One uncounted run comes first. *)
let host_grown make f =
  let settled () =
    for _ = 1 to 4 do
      Gc.full_major ();
      ignore (Rig.Buffer.create Rig.host 0);
      Rig.free_cache Rig.host
    done;
    Rig.allocated Rig.host
  in
  ignore (f (make ()));
  let x = make () in
  let before = settled () in
  let y = f x in
  (y, Rig.allocated Rig.host - before)

(* Images compare as their shapes and pixels. *)
let image =
  Testable.make ~pp:Nx.pp ~equal:(fun a b ->
      Nx.shape a = Nx.shape b && Nx.to_array a = Nx.to_array b)

(* The pixels Pillow decodes from the fixture image [name]. *)
let pillow name =
  let npy = Filename.remove_extension name ^ ".npy" in
  Nx.unpack Nx.uint8 (Nx_io.load_npy (fixture npy))

(* Drawn images *)

(* How an image lies in memory. *)
type view = Plain | Transposed | Flipped | Broadcast

(* Images of [channels], [0] for no channel axis, of 1 to 8 rows and columns,
   with any pixels, in any view. *)
let images channels =
  let open Gen in
  let* c = of_list channels in
  let* h, w = pair (int_range 1 8) (int_range 1 8) in
  let shape = if c = 0 then [| h; w |] else [| h; w; c |] in
  let n = Array.fold_left ( * ) 1 shape in
  let* pixels = array ~size:(constant n) (int_range 0 255) in
  let+ view = of_list [ Plain; Transposed; Flipped; Broadcast ] in
  let x = Nx.create Nx.uint8 shape pixels in
  let x =
    match view with
    | Plain -> x
    | Transposed ->
        Nx.transpose ~axes:(if c = 0 then [ 1; 0 ] else [ 1; 0; 2 ]) x
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

(* Smooth images, gradients, which JPEG keeps closely. *)
let smooth channels =
  let open Gen in
  let* c = of_list channels in
  let+ h = int_range 8 24 and+ w = int_range 8 24 in
  let shape = if c = 0 then [| h; w |] else [| h; w; c |] in
  Nx.init Nx.uint8 shape (fun i ->
      let k = if Array.length i = 3 then i.(2) else 0 in
      ((i.(0) * 4) + (i.(1) * 3) + (k * 40)) land 255)

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
          let png = Nx_image.encode_png tiny in
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
            (chunk_data "pHYs" (Nx_image.encode_png ~dpi tiny)));
      test "~srgb writes one sRGB chunk with the perceptual intent" (fun () ->
          equal (list string) [ "\000" ]
            (chunk_data "sRGB" (Nx_image.encode_png ~srgb:true tiny)));
      prop
        "with ~dpi and ~srgb the file is encode_png's with the two chunks \
         added, and loads back the same"
        (Gen.pair (images [ 0; 1; 3; 4 ]) (Gen.float_range 1. 1000.))
        (fun ((_, x), dpi) ->
          let plain = Nx_image.encode_png x in
          let png = Nx_image.encode_png ~dpi ~srgb:true x in
          equal string plain (without "sRGB" (without "pHYs" png));
          equal image
            (Nx_image.load (file ~suffix:".png" plain))
            (Nx_image.load (file ~suffix:".png" png)));
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
          invalid ~by:"Nx_image.encode_png" (fun () ->
              Nx_image.encode_png ~dpi tiny));
    ]

let flat x = Nx.reshape [| Nx.dim 0 x; Nx.dim 1 x |] x

let round_trips =
  group "round trips"
    [
      prop
        "save writes encode_png's bytes, which load back exactly: grey in \
         three equal channels, alpha dropped, and colour in grayscale within \
         one level of its luma"
        (images [ 0; 1; 3; 4 ])
        (fun (view, x) ->
          covers (view, x);
          let h = Nx.dim 0 x and w = Nx.dim 1 x in
          cover "grey" (Nx.ndim x = 2 || Nx.dim 2 x = 1);
          cover "colour" (Nx.ndim x = 3 && Nx.dim 2 x = 3);
          cover "alpha" (Nx.ndim x = 3 && Nx.dim 2 x = 4);
          let path = temp_file ~suffix:".png" () in
          Nx_image.save path x;
          equal string (Nx_image.encode_png x) (read path);
          let colour = Nx_image.load path in
          let gray = Nx_image.load ~grayscale:true path in
          match Nx.shape x with
          | [| _; _; 3 |] ->
              equal image x colour;
              let px = Nx.to_array x and g = Nx.to_array gray in
              Array.iteri
                (fun i g ->
                  let p k = Float.of_int px.((3 * i) + k) in
                  let luma =
                    (0.299 *. p 0) +. (0.587 *. p 1) +. (0.114 *. p 2)
                  in
                  at_most float_exact ~than:1.
                    (Float.abs (Float.of_int g -. luma)))
                g
          | [| _; _; 4 |] ->
              equal image (Nx.slice [ Nx.A; Nx.A; Nx.R (0, 3) ] x) colour
          | _ ->
              equal image (flat x) gray;
              equal image
                (Nx.broadcast_to [| h; w; 3 |] (Nx.reshape [| h; w; 1 |] gray))
                colour);
      prop "a smooth image returns from JPEG within a mean error of 8 levels"
        (smooth [ 0; 1; 3 ])
        (fun x ->
          let path = temp_file ~suffix:".jpg" () in
          Nx_image.save path x;
          let colour = Nx.ndim x = 3 && Nx.dim 2 x = 3 in
          let back = Nx_image.load ~grayscale:(not colour) path in
          let x = if colour then x else flat x in
          equal (array int) (Nx.shape x) (Nx.shape back);
          let e = errors x back in
          at_most int ~than:8 (Array.fold_left ( + ) 0 e / Array.length e));
      test "a save leaves its image's memory to the image's next donation"
        (fun () ->
          let n = 256 in
          (* An image made from OCaml data, in memory of its own. *)
          let make () =
            Nx.create Nx.uint8 [| n; n; 3 |] (Array.make (n * n * 3) 7)
          in
          let saved_then_donated x =
            Nx_image.save (temp_file ~suffix:".png" ()) x;
            Nx.neg (Nx.donate x)
          in
          let y, grew = host_grown make saved_then_donated in
          less ~msg:"no memory of the result's size" int ~than:(n * n * 3)
            grew;
          equal (array int) [| 249 |]
            (Nx.to_array (Nx.slice [ Nx.I 0; Nx.I 0; Nx.R (0, 1) ] y)));
      test "a loaded image's donation writes in place" (fun () ->
          let n = 256 in
          let path = temp_file ~suffix:".png" () in
          Nx_image.save path
            (Nx.create Nx.uint8 [| n; n; 3 |] (Array.make (n * n * 3) 7));
          let y, grew =
            host_grown
              (fun () -> Nx_image.load path)
              (fun x -> Nx.neg (Nx.donate x))
          in
          less ~msg:"no memory of the result's size" int ~than:(n * n * 3)
            grew;
          equal (array int) [| 249 |]
            (Nx.to_array (Nx.slice [ Nx.I 0; Nx.I 0; Nx.R (0, 1) ] y)));
      test "a value of another set saves" (fun () ->
          let module M = (val Nx.devices [ Nx_support.memory 2 ]) in
          let x =
            Nx.init Nx.uint8 [| 3; 2; 3 |] (fun i -> i.(0) + i.(1) + i.(2))
          in
          let path = temp_file ~suffix:".png" () in
          Nx_image.save path (Nx.place M.on x);
          equal image x (Nx_image.load path));
    ]

let files =
  group "files"
    [
      cases ~name:fst "files other encoders wrote load as Pillow decodes them"
        (List.map
           (fun n -> ("png_" ^ n ^ ".png", 0))
           [ "filters"; "gray1"; "rgb16"; "palette_adam7" ]
        @ List.map
            (fun n -> ("jpeg_" ^ n ^ ".jpg", 3))
            [ "baseline"; "progressive"; "gray"; "cmyk"; "restart" ])
        (fun (name, tolerance) ->
          let expected = pillow name in
          let grayscale = Nx.ndim expected = 2 in
          let x = Nx_image.load ~grayscale (fixture name) in
          equal ~msg:"shape" (array int) (Nx.shape expected) (Nx.shape x);
          at_most ~msg:"largest error" int ~than:tolerance
            (Array.fold_left Int.max 0 (errors expected x)));
      cases ~name:Fun.id
        "the extension selects the encoder, whatever its case, and the \
         contents the decoder, whatever the extension"
        [ ".png"; ".PNG"; ".jpg"; ".JPG"; ".jpeg"; ".JpEg" ] (fun ext ->
          let path = temp_file ~suffix:ext () in
          Nx_image.save path (Nx.full Nx.uint8 [| 4; 4; 3 |] 7);
          let png = String.lowercase_ascii ext = ".png" in
          starts_with ~affix:(if png then "\137PNG" else "\xff\xd8") (read path);
          let other =
            file ~suffix:(if png then ".jpg" else ".png") (read path)
          in
          equal image (Nx_image.load path) (Nx_image.load other));
      cases
        ~name:(fun (name, _, _) -> name)
        "an extension or a shape the format does not hold is refused"
        [
          ("a BMP extension", "x.bmp", [| 2; 2; 3 |]);
          ("no extension", "x", [| 2; 2; 3 |]);
          ("a vector", "x.png", [| 4 |]);
          ("two channels", "x.png", [| 2; 2; 2 |]);
          ("no rows", "x.png", [| 0; 3; 3 |]);
          ("four channels in JPEG", "x.jpg", [| 2; 2; 4 |]);
          ("no columns in JPEG", "x.jpg", [| 3; 0 |]);
        ]
        (fun (_, name, shape) ->
          let path = Filename.concat (temp_dir ()) name in
          invalid ~by:"Nx_image.save" (fun () ->
              Nx_image.save path (Nx.zeros Nx.uint8 shape));
          if name = "x.png" then
            invalid ~by:"Nx_image.encode_png" (fun () ->
                Nx_image.encode_png (Nx.zeros Nx.uint8 shape)));
      test
        "a refusal before the write names the save, and an error during it \
         keeps its own message" (fun () ->
          let by = "Nx_image.save" in
          raises
            (Invalid_argument (by ^ ": the value is dead"))
            (fun () ->
              Fail.reading ~by
                (fun _ -> invalid_arg "Nx.Repr.read: the value is dead")
                (fun () -> ()));
          raises (Invalid_argument "codec: a bad image") (fun () ->
              Fail.reading ~by
                (fun f -> f ())
                (fun () -> invalid_arg "codec: a bad image")));
      test "a PNG chunk whose checksum does not match fails" (fun () ->
          let path = file (flip_byte (read (fixture "png_filters.png")) 30) in
          fails ~path ~naming:"PNG chunk CRC mismatch" (fun () ->
              Nx_image.load path));
      test "a file that is neither PNG nor JPEG, or absent, fails" (fun () ->
          let path = file "GIF89a" in
          fails ~path (fun () -> Nx_image.load path);
          let path = Filename.concat (temp_dir ()) "absent.png" in
          fails ~path (fun () -> Nx_image.load path));
      cases ~name:Fun.id
        "every proper prefix fails, and any byte flipped loads or fails"
        [
          "png_filters.png";
          "png_palette_adam7.png";
          "jpeg_baseline.jpg";
          "jpeg_progressive.jpg";
        ] (fun name ->
          let whole = read (fixture name) in
          for n = 0 to String.length whole - 1 do
            let path = file (String.sub whole 0 n) in
            fails ~path (fun () -> Nx_image.load path)
          done;
          for i = 0 to String.length whole - 1 do
            let path = file (flip_byte whole i) in
            loads_or_fails ~path (fun () -> Nx_image.load path);
            loads_or_fails ~path (fun () -> Nx_image.load ~grayscale:true path)
          done);
      test "without overwrite, a save to a path that names a file fails"
        (fun () ->
          let path = file ~suffix:".png" "old" in
          fails ~path (fun () -> Nx_image.save ~overwrite:false path tiny);
          equal string "old" (read path));
    ]

let () = exit (run "nx.image" [ png_chunks; round_trips; files ])
