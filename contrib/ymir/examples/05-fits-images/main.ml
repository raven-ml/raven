(* FITS images and headers.

   A FITS file is a list of HDUs, each a header of keyword records and a data
   unit. Reading copies the headers and reads no pixels until asked; every
   failure in a file is an [Error] naming where it happened. *)

open Ymir
open Ymir_fits

let ( let* ) = Result.bind

let image =
  Nx.init Nx.float32 [| 64; 48 |] (fun i ->
      let r = float_of_int i.(0) -. 31.5 and c = float_of_int i.(1) -. 23.5 in
      100. *. exp (-.((r *. r) +. (c *. c)) /. 50.))

let run path =
  (* A header holds keyword records. [Image.hdu] adds the structural ones
     (BITPIX, NAXIS, ...) itself; [~tiles] stores the image tile-compressed. *)
  let header =
    Fits.Header.(
      empty
      |> set Fits.Value.string "EXTNAME" "SCI"
      |> set Fits.Value.string "BUNIT" "electron/s"
      |> set ~comment:"seconds" Fits.Value.float "EXPTIME" 120.)
  in
  let sci = Fits.Image.hdu header image in
  let tiled =
    Fits.Image.hdu ~tiles:[| 16; 48 |]
      (Fits.Header.set Fits.Value.string "EXTNAME" "TILED" header)
      image
  in
  let* () = Fits.write path [ sci; tiled ] in

  (* Read it back. *)
  let* hdus = Fits.read path in
  List.iter (Format.printf "%a@." Fits.pp) hdus;
  let* sci = Fits.get "SCI" hdus in
  let* () = Fits.verify sci in
  print_endline "SCI checksums verify";
  let h = Fits.header sci in
  let* exptime = Fits.Header.get Fits.Value.float "EXPTIME" h in
  let* unit = Fits.unit sci in
  Printf.printf "EXPTIME = %g, BUNIT = %s\n" exptime
    (match unit with Some u -> Unit.to_string u | None -> "none");

  (* Pixels in the dtype the caller names. A window reads only the rows it
     covers. *)
  let* centre =
    Fits.Image.values ~window:[| (30, 34); (22, 26) |] Nx.float64 sci
  in
  Format.printf "centre of SCI:@.%a@." Nx.pp centre;

  (* The tile-compressed copy reads as the plain one, bit for bit. *)
  let* tiled = Fits.get "TILED" hdus in
  let* a = Fits.Image.values Nx.float32 sci in
  let* b = Fits.Image.values Nx.float32 tiled in
  Printf.printf "TILED equals SCI: %b\n" (Nx.array_equal a b |> Nx.item []);

  (* Print the header as records. *)
  Format.printf "%a@." Fits.Header.pp h;
  Ok ()

let () =
  let path = Filename.temp_file "ymir" ".fits" in
  let result = run path in
  Sys.remove path;
  match result with
  | Ok () -> ()
  | Error e ->
      prerr_endline e;
      exit 1
