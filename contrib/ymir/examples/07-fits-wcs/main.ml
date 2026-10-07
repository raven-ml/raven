(* World coordinates from a FITS header.

   [Fits.Wcs.read] turns a header's celestial keywords into a transform from
   0-based pixel indices, in tensor axis order, to directions in the frame the
   caller names. [Fits.Wcs.write] spells a transform back into a header. *)

open Ymir

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

let run () =
  let* header = Fits.Header.of_string ~name:"example.head" text in
  let* wcs = Fits.Wcs.read Frame.icrs header in
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
  (match Fits.Wcs.read Frame.galactic header with
  | Ok _ -> ()
  | Error e -> print_endline e);

  (* Writing a cutout's WCS: the window shifts CRPIX; every other record is
     kept. *)
  let* cutout = Fits.Wcs.write ~window:[| (40, 60); (90, 110) |] wcs header in
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
