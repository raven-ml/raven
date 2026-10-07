(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* FITS unit strings: the spellings FITS 4.0 §4.3 lists, the values of its
   Tables 3-5, data-set symbols, and printing that reads back. *)

open Windtrap
open Ymir_fits
open Ymir_units
module P = Fits.Unit

let unit_w = Testable.make ~pp:Unit.pp ~equal:Unit.equal
let parsed = result unit_w string
let parse s = P.parse s

let same =
  let m2 = Unit.(metre ** 2) and m_3 = Unit.(one / (metre ** 3)) in
  let m32 = Unit.(root 2 (metre ** 3)) in
  cases ~name:fst "§4.3's spellings of one unit"
    (List.map
       (fun s -> (s, m2))
       [ "m**(2)"; "m**+2"; "m+2"; "m2"; "m^2"; "m^(+2)" ]
    @ List.map (fun s -> (s, m_3)) [ "m**-3"; "m-3"; "m^(-3)"; "/m3" ]
    @ List.map
        (fun s -> (s, m32))
        [ "m(1.5)"; "m^(1.5)"; "m**(1.5)"; "m(3/2)"; "m**(3/2)"; "m^(3/2)" ])
    (fun (s, u) -> equal parsed (Ok u) (parse s))

let values =
  let jy = Unit.(decimal "1e-26" * watt / (metre ** 2) / hertz) in
  cases ~name:fst "units at the standard's values"
    [
      ("pc", Unit.(decimal "3.0857e16" * metre));
      ("kpc", Unit.(decimal "3.0857e19" * metre));
      ("kg", Unit.kilogram);
      ("solMass", Unit.(decimal "1.9891e30" * kilogram));
      ("MJy/sr", Unit.(mega jy / steradian));
      ("uJy", Unit.(micro jy));
      ("erg/cm2 s", Unit.(decimal "1e-7" * joule / (centi metre ** 2) / second));
      ( "erg/cm**2/s/Angstrom",
        Unit.(
          decimal "1e-7" * joule
          / (centi metre ** 2)
          / second
          / (decimal "1e-10" * metre)) );
      ("10**-3 m", Unit.(milli metre));
      ("10+3 m", Unit.(kilo metre));
      ("10-3 m", Unit.(milli metre));
      ("10^(2) s", Unit.(int 100 * second));
      ("km/s/Mpc", Unit.(kilo metre / second / (decimal "3.0857e22" * metre)));
      ("deg", Unit.degree);
      ("mas", Unit.(milli arcsecond));
      ("Gyr", Unit.(int 31557600 * giga second));
      ("Pa", Unit.pascal);
      ("dam", Unit.(deca metre));
      ("mag", Unit.symbol "mag");
      ("mmag", Unit.(milli (symbol "mag")));
      ("ct/s", Unit.(symbol "count" / second));
      ("count s-1", Unit.(symbol "count" / second));
      ("ELECTRONS/S", Unit.(symbol "electron" / second));
      ("DN/s", Unit.(symbol "DN" / second));
      ("byte", Unit.(int 8 * symbol "bit"));
      ("kbyte", Unit.(int 8000 * symbol "bit"));
      ("Ry", Unit.(decimal "13.605692" * decimal "1.6021765e-19" * joule));
      ("", Unit.one);
    ]
    (fun (s, u) -> equal parsed (Ok u) (parse s))

let errors =
  cases ~name:Fun.id "strings FITS does not read"
    [
      "ms/2";
      "m1.5";
      "kdeg";
      "Kpc";
      "mx";
      "log(Hz)";
      "sqrt(Hz)";
      "exp(m)";
      "m**";
      "(m";
      "m)";
      "10 m";
      "10**1.5 m";
      "m^(1/0)";
      "1e-3 m";
      "pix";
      "beam";
      "Rm";
      "Qm";
    ] (fun s -> is_error (parse s))

let data_set () =
  let pix = Unit.scoped ~scope:"f#SCI" "pix" in
  equal parsed
    (Ok Unit.(Unit.arcsecond / pix))
    (P.parse ~scope:"f#SCI" "arcsec/pixel");
  equal parsed
    (Ok Unit.(Unit.arcsecond / pix))
    (P.parse ~scope:"f#SCI" "arcsec/pix");
  equal parsed
    (Ok (Unit.scoped ~scope:"f#SCI" "beam"))
    (P.parse ~scope:"f#SCI" "beam");
  equal (result string string) (Ok "MJy sr-1 pix-1")
    (P.print
       Unit.(
         mega (decimal "1e-26" * watt / (metre ** 2) / hertz) / steradian / pix))

let printing () =
  let p s = match parse s with Ok u -> P.print u | Error e -> Error e in
  expect
    (String.concat "\n"
       (List.map
          (fun s ->
            match p s with Ok t -> s ^ " -> " ^ t | Error e -> "error " ^ e)
          [
            "MJy/sr";
            "erg/cm2 s";
            "km/s/Mpc";
            "m(3/2)";
            "10**-3 m";
            "solMass";
            "Angstrom";
            "ct/s";
            "";
          ]))
  @@ __POS_OF__
       {|
    MJy/sr -> MJy sr-1
    erg/cm2 s -> g s-3
    km/s/Mpc -> mm pc-1 s-1
    m(3/2) -> m(3/2)
    10**-3 m -> mm
    solMass -> solMass
    Angstrom -> Angstrom
    ct/s -> count s-1
     ->
    |}

let fits_unit () =
  let h =
    Fits.Header.(
      empty
      |> set Fits.Value.string "BUNIT" "MJy/sr"
      |> set Fits.Value.string "EXTNAME" "SCI")
  in
  let hdu = Fits.Image.hdu h (Nx.zeros Nx.float32 [| 2; 2 |]) in
  equal
    (result (option unit_w) string)
    (Ok
       (Some
          Unit.(
            mega (decimal "1e-26" * watt / (metre ** 2) / hertz) / steradian)))
    (Fits.unit hdu);
  let pix =
    Fits.Image.hdu
      Fits.Header.(empty |> set Fits.Value.string "BUNIT" "count/pix")
      (Nx.zeros Nx.float32 [| 2 |])
  in
  equal
    (result (option unit_w) string)
    (Ok
       (Some
          Unit.(
            symbol "count" / scoped ~scope:(Fits.digest pix ^ "#HDU0") "pix")))
    (Fits.unit pix);
  let bad =
    Fits.Image.hdu
      Fits.Header.(empty |> set Fits.Value.string "BUNIT" "furlong")
      (Nx.zeros Nx.float32 [| 2 |])
  in
  match Fits.unit bad with
  | Ok _ -> fail "furlong read"
  | Error e -> contains ~sub:"card" e

(* Printing reads back *)

let words =
  [
    "m";
    "g";
    "s";
    "Jy";
    "pc";
    "deg";
    "erg";
    "solMass";
    "Angstrom";
    "eV";
    "mag";
    "count";
    "photon";
    "yr";
    "K";
  ]

let prefixes = [ ""; "k"; "M"; "m"; "u"; "n"; "G" ]

let round_trip =
  prop "parse (print u) is u"
    Gen.(
      list ~size:(int_range 0 4)
        (triple (of_list words) (of_list prefixes) (int_range (-3) 3)))
    (fun ws ->
      let text (w, p, e) =
        let p =
          if
            List.mem w
              [ "deg"; "erg"; "solMass"; "Angstrom"; "count"; "photon" ]
          then ""
          else p
        in
        p ^ w ^ if e = 0 then "" else string_of_int e
      in
      let s = String.concat " " (List.map text ws) in
      match parse s with
      | Error e -> fail e
      | Ok u -> (
          match P.print u with
          | Error _ -> collect "no spelling"
          | Ok t ->
              cover "printed" true;
              equal parsed (Ok u) (parse t)))

let () =
  exit
  @@ run "Fits.Unit"
       [
         same;
         values;
         errors;
         test "data-set symbols" data_set;
         test "printing" printing;
         test "BUNIT" fits_unit;
         round_trip;
       ]
