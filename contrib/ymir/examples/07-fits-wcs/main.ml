(* World coordinates from a FITS header.

   [Wcs.read] turns a header's celestial keywords into a transform from 0-based
   pixel indices, in tensor axis order, to directions in the frame the caller
   names. It asks for each keyword's value, decoded by ymir.fits. [Wcs.write]
   returns the edits that spell a transform back into a header. A program
   composes the two libraries: here, an image and its error into an observation,
   and a cutout's header. *)

open Ymir
open Ymir_fits

let ( let* ) = Result.bind

(* A header as a .head file holds it: one record per line. *)
let text =
  String.concat "\n"
    [
      "NAXIS   =                    2";
      "NAXIS1  =                  200";
      "NAXIS2  =                  100";
      "CTYPE1  = 'RA---TAN'";
      "CTYPE2  = 'DEC--TAN'";
      "CRPIX1  =                100.5";
      "CRPIX2  =                 50.5";
      "CRVAL1  =                 45.0";
      "CRVAL2  =                -30.0";
      "CD1_1   =         -2.777778E-4";
      "CD1_2   =                  0.0";
      "CD2_1   =                  0.0";
      "CD2_2   =          2.777778E-4";
      "RADESYS = 'ICRS'";
      "END";
    ]

(* The header's keywords as [Wcs] reads them: decoded by ymir.fits. *)
let keywords h =
  {
    Wcs.float = (fun k -> Fits.Header.find Fits.Value.float k h);
    int = (fun k -> Fits.Header.find Fits.Value.int k h);
    text = (fun k -> Fits.Header.find Fits.Value.string k h);
  }

(* [edit h edits] is [h] with [Wcs.write]'s edits made. *)
let edit h edits =
  List.fold_left
    (fun h -> function
      | Wcs.Float (k, x) -> Fits.Header.set Fits.Value.float k x h
      | Int (k, n) -> Fits.Header.set Fits.Value.int k n h
      | Text (k, s) -> Fits.Header.set Fits.Value.string k s h
      | Remove k -> Fits.Header.remove k h)
    h edits

(* An image HDU named [name] holding [values] in MJy/sr, with JWST's PIXAR_SR,
   each pixel's solid angle. *)
let image header name values =
  let h =
    Fits.Header.(
      header
      |> set Fits.Value.string "EXTNAME" name
      |> set Fits.Value.string "BUNIT" "MJy/sr"
      |> set Fits.Value.float "PIXAR_SR" 2.35e-11)
  in
  Fits.Image.hdu h values

(* The SCI and ERR HDUs as an observation: the program decides that ERR shares
   SCI's unit, that a sample is valid where both are finite, and that PIXAR_SR
   is each pixel's area. *)
let observation hdus =
  let* sci = Fits.get "SCI" hdus in
  let* err = Fits.get "ERR" hdus in
  let h = Fits.header sci in
  let* wcs = Wcs.read Frame.icrs (keywords h) in
  let* bunit = Fits.Header.get Fits.Value.string "BUNIT" h in
  let* unit = Fits.Unit.parse bunit in
  let* pixar = Fits.Header.get Fits.Value.float "PIXAR_SR" h in
  let* data = Fits.Image.values Nx.float64 sci in
  let* sigma = Fits.Image.values Nx.float64 err in
  let grid = Grid.pixels ~shape:(Nx.shape data) Nx.float64 wcs in
  Ok
    (Observation.v
       ~variance:(Quantity.v (Unit.( ** ) unit 2) (Nx.square sigma))
       ~valid:Nx.(cast bit (logical_and (isfinite data) (isfinite sigma)))
       ~area:(Quantity.v Unit.steradian (Nx.scalar Nx.float64 pixar))
       grid (Quantity.v unit data))

let run () =
  let* header = Fits.Header.of_string ~name:"example.head" text in
  let* wcs = Wcs.read Frame.icrs (keywords header) in
  Format.printf "%a@." Transform.pp wcs;

  (* The image's first and last pixels, [row; column]. *)
  let pixels =
    Quantity.v Unit.one
      (Nx.create Nx.float64 [| 2; 2 |] [| 0.; 0.; 99.; 199. |])
  in
  let sky = Transform.apply wcs pixels in
  let lon = Nx.to_array (Quantity.value Unit.degree (Direction.lon sky)) in
  let lat = Nx.to_array (Quantity.value Unit.degree (Direction.lat sky)) in
  Array.iteri
    (fun i l -> Printf.printf "pixel %d -> (%.6f, %+.6f) deg\n" i l lat.(i))
    lon;

  (* A header in another frame is an [Error] that says how to read it. *)
  (match Wcs.read Frame.galactic (keywords header) with
  | Ok _ -> ()
  | Error e -> print_endline e);

  (* An image and its error under this header, as an observation. *)
  let data = Nx.full Nx.float32 [| 100; 200 |] 0.5 in
  let sigma = Nx.full Nx.float32 [| 100; 200 |] 0.05 in
  let* obs =
    observation [ image header "SCI" data; image header "ERR" sigma ]
  in
  let shape = Grid.shape (Observation.grid obs) in
  Format.printf "observation: %d x %d cells in %a@." shape.(0) shape.(1) Unit.pp
    (Quantity.unit (Observation.data obs));

  (* A cutout's WCS: the window shifts CRPIX, and the edits change only that. *)
  let* edits =
    Wcs.write ~window:[| (40, 60); (90, 110) |] wcs (keywords header)
  in
  List.iter
    (function
      | Wcs.Float (k, x) -> Printf.printf "set %s = %g\n" k x
      | Int (k, n) -> Printf.printf "set %s = %d\n" k n
      | Text (k, s) -> Printf.printf "set %s = '%s'\n" k s
      | Remove k -> Printf.printf "remove %s\n" k)
    edits;
  let cutout = edit header edits in
  let* crpix1 = Fits.Header.get Fits.Value.float "CRPIX1" cutout in
  let* crpix2 = Fits.Header.get Fits.Value.float "CRPIX2" cutout in
  Printf.printf "cutout CRPIX = (%g, %g)\n" crpix1 crpix2;
  Ok ()

let () =
  match run () with
  | Ok () -> ()
  | Error e ->
      prerr_endline e;
      exit 1
