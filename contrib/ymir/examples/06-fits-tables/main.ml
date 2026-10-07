(* FITS tables.

   A binary table is rows of typed columns: fixed-shape cells, variable-length
   lists, or text. A column carries its own keywords, such as its unit, and
   marks undefined cells with a validity mask. *)

open Ymir
open Ymir_fits

let ( let* ) = Result.bind
let no_cards = Fits.Header.empty

let run path =
  let n = 4 in
  (* A text column holds each row's bytes as one row of a ragged tensor. *)
  let name = Nx_ragged.of_strings [| "alpha"; "beta"; "gamma"; "delta" |] in
  let position =
    Nx.create Nx.float64 [| n; 2 |]
      [| 10.1; -2.3; 10.4; -2.1; 9.8; -2.6; 10.0; -2.0 |]
  in
  let flux = Nx.create Nx.float32 [| n |] [| 1.5; 0.2; 7.25; 0. |] in
  (* The last flux was not measured. *)
  let measured = Nx.create Nx.bit [| n |] [| true; true; true; false |] in
  let columns =
    [
      ("NAME", no_cards, Fits.Table.Text name);
      ( "POS",
        Fits.Header.set Fits.Value.string "TUNIT" "deg" no_cards,
        Fits.Table.Array { values = Nx.P position; validity = None } );
      ( "FLUX",
        Fits.Header.set Fits.Value.string "TUNIT" "mJy" no_cards,
        Fits.Table.Array { values = Nx.P flux; validity = Some measured } );
    ]
  in
  let header =
    Fits.Header.set Fits.Value.string "EXTNAME" "SOURCES" Fits.Header.empty
  in
  let* table = Fits.Table.hdu header columns in
  let* () = Fits.write path [ table ] in

  (* Read it back: the description, then columns by name. *)
  let* hdus = Fits.read path in
  let* hdu = Fits.get "SOURCES" hdus in
  let* t = Fits.Table.of_hdu hdu in
  Format.printf "%a@." Fits.Table.pp t;

  let* pos = Fits.Table.values Nx.float64 "POS" hdu in
  Format.printf "POS:@.%a@." Nx.pp pos;
  let* unit = Fits.Table.unit "POS" hdu in
  Option.iter (fun u -> Printf.printf "POS unit: %s\n" (Unit.to_string u)) unit;

  (* Undefined cells read as NaN, and [validity] says which they are. *)
  let* flux = Fits.Table.values ~rows:(1, 4) Nx.float32 "FLUX" hdu in
  Format.printf "FLUX rows 1-3: %a@." Nx.pp flux;
  let* valid = Fits.Table.validity ~rows:(1, 4) "FLUX" hdu in
  Option.iter (Format.printf "FLUX defined:  %a@." Nx.pp) valid;

  let* name = Fits.Table.ragged Nx.uint8 "NAME" hdu in
  Printf.printf "NAME: %s\n"
    (String.concat ", " (Array.to_list (Nx_ragged.to_strings name)));
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
