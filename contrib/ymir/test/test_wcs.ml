(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* FITS world coordinates: headers agree with WCSLIB and print back what they
   read, reals read back bit for bit, and unreadable descriptions are errors
   naming their keyword, or the view's own errors. *)

open Windtrap
open Ymir
open Ymir_fits
open Ymir_test.Frames
module Wcs_reference = Ymir_test.Wcs_reference

let f64 = Nx.float64
let ok r = require_ok ~pp:Format.pp_print_string r
let deg x = Quantity.v Unit.degree x
let tensor shape xs = Nx.create f64 shape xs
let end_record = "END" ^ String.make 77 ' '
let header text = ok (Fits.Header.of_string ~name:"header" (text ^ end_record))
let frame name = List.find (fun f -> Ymir_test.Frames.name f = name) fixed
let pixels xs = Quantity.v Unit.one (tensor [| Array.length xs / 2; 2 |] xs)

let lonlat_deg f xs =
  let t = tensor [| Array.length xs / 2; 2 |] xs in
  Direction.lonlat f
    ~lon:(deg (Nx.slice [ Nx.A; Nx.I 0 ] t))
    ~lat:(deg (Nx.slice [ Nx.A; Nx.I 1 ] t))

let keywords h =
  {
    Wcs.float = (fun k -> Fits.Header.find Fits.Value.float k h);
    int = (fun k -> Fits.Header.find Fits.Value.int k h);
    text = (fun k -> Fits.Header.find Fits.Value.string k h);
  }

(* [apply edits h] is [h] with [edits] made, as a program makes them. *)
let apply edits h =
  List.fold_left
    (fun h -> function
      | Wcs.Float (k, x) -> Fits.Header.set Fits.Value.float k x h
      | Int (k, n) -> Fits.Header.set Fits.Value.int k n h
      | Text (k, s) -> Fits.Header.set Fits.Value.string k s h
      | Remove k -> Fits.Header.remove k h)
    h edits

let pp_edit ppf = function
  | Wcs.Float (k, x) -> Format.fprintf ppf "%s = %h" k x
  | Int (k, n) -> Format.fprintf ppf "%s = %d" k n
  | Text (k, s) -> Format.fprintf ppf "%s = %S" k s
  | Remove k -> Format.fprintf ppf "remove %s" k

let edits = list (Testable.make ~pp:pp_edit ~equal:( = ))
let read ?alt f h = ok (Wcs.read ?alt f (keywords h))
let write ?alt ?window t h = ok (Wcs.write ?alt ?window t (keywords h))

(* [stages t] is [t]'s skeleton and each leaf's bits: equal for equal stages,
   leaf for leaf. *)
let stages t =
  let leaves, skeleton = Nx.Ptree.flatten (Transform.ptree ()) t in
  ( skeleton,
    List.map
      (fun (Nx.P x) ->
        Array.map Int64.bits_of_float (Nx.to_array (Nx.cast f64 x)))
      leaves )

let equal_stages a b =
  let sa, la = stages a and sb, lb = stages b in
  is_true ~msg:"skeletons" (Nx.Ptree.Skeleton.equal sa sb);
  equal (list (array int64)) la lb

let nircam = header (List.hd Wcs_reference.cases).header

(* WCSLIB *)

let wcslib =
  cases
    ~name:(fun (c : Wcs_reference.case) -> c.name)
    "WCSLIB" Wcs_reference.cases
    (fun c ->
      let (F f) = frame c.frame in
      let t = read ~alt:c.alt f (header c.header) in
      let world = Transform.apply t (pixels c.pixels) in
      (* WCSLIB solves ZPN's and AIR's deprojections to 1e-13 and 1e-12 in the
         radial function. *)
      let solved = c.name = "ZPN" || c.name = "AIR" in
      less float_exact
        ~than:(if solved then 5e-12 else 2e-13)
        (Array.fold_left max 0. (separation world (lonlat_deg f c.world))))

(* Law 1: a transform prints back what it read *)

let crota (c : Wcs_reference.case) = c.name = "CROTA2, Galactic"
let tan_pv (c : Wcs_reference.case) = c.name = "TAN with PV terms"

let round_trips =
  let reads_back (c : Wcs_reference.case) h =
    let (F f) = frame c.frame in
    let t = read ~alt:c.alt f (header c.header) in
    equal_stages t (read ~alt:c.alt f (apply (write ~alt:c.alt t h) h))
  in
  group "Law 1"
    [
      cases
        ~name:(fun (c : Wcs_reference.case) -> c.name)
        "a header read at float64 needs no edit"
        (List.filter (fun c -> not (crota c || tan_pv c)) Wcs_reference.cases)
        (fun c ->
          let (F f) = frame c.frame in
          let h = header c.header in
          equal edits [] (write ~alt:c.alt (read ~alt:c.alt f h) h));
      cases
        ~name:(fun (c : Wcs_reference.case) -> c.name)
        "edits to an empty header read back as equal stages" Wcs_reference.cases
        (fun c -> reads_back c Fits.Header.empty);
      cases
        ~name:(fun (c : Wcs_reference.case) -> c.name)
        "edits to the header read back as equal stages" Wcs_reference.cases
        (fun c -> reads_back c (header c.header));
      prop "a real reads back bit for bit"
        Gen.(
          pair
            (one_of [ float; float_range (-.Float.min_float) Float.min_float ])
            float)
        (fun (x, y) ->
          cover "subnormal" (Float.abs x < Float.min_float && x <> 0.);
          cover "past 1e20" (Float.abs y > 1e20);
          let h =
            Fits.Header.(
              nircam
              |> set Fits.Value.float "CRPIX1" x
              |> set Fits.Value.float "CRVAL1" y)
          in
          let t = read Frame.icrs h in
          let h' = apply (write t Fits.Header.empty) Fits.Header.empty in
          equal_stages t (read Frame.icrs h'));
      test "CROTA2's edits spell the PC matrix it reads as" (fun () ->
          let c = List.find crota Wcs_reference.cases in
          let h = header c.header in
          let t = read Frame.galactic h in
          let h' = apply (write t h) h in
          equal (option string) None
            (ok (Fits.Header.find Fits.Value.text "CROTA2" h'));
          equal_stages t (read Frame.galactic h'));
      test "TAN with PV terms's edits spell TPV" (fun () ->
          let c = List.find tan_pv Wcs_reference.cases in
          let h = header c.header in
          let t = read Frame.icrs h in
          let h' = apply (write t h) h in
          equal (option string) (Some "RA---TPV")
            (ok (Fits.Header.find Fits.Value.string "CTYPE1" h'));
          equal_stages t (read Frame.icrs h'));
      test "a PC transform's edits to a CD header set PC and remove CD"
        (fun () ->
          let t = read Frame.icrs nircam in
          let cd =
            Fits.Header.(
              nircam |> remove "PC1_1" |> remove "PC1_2" |> remove "PC2_1"
              |> remove "PC2_2"
              |> set Fits.Value.float "CD1_1" 1e-5)
          in
          equal edits
            [
              Wcs.Float ("PC1_1", 0.815256589635877);
              Float ("PC1_2", 0.5790998990288975);
              Float ("PC2_1", 0.5790998990288975);
              Float ("PC2_2", -0.815256589635877);
              Remove "CD1_1";
            ]
            (write t cd));
      test "a window moves CRPIX by its start" (fun () ->
          let t = read Frame.icrs nircam in
          let h =
            apply (write ~window:[| (10, 114); (20, 124) |] t nircam) nircam
          in
          let get k = ok (Fits.Header.get Fits.Value.float k h) in
          let was k = ok (Fits.Header.get Fits.Value.float k nircam) in
          equal float_exact (was "CRPIX1" -. 20.) (get "CRPIX1");
          equal float_exact (was "CRPIX2" -. 10.) (get "CRPIX2"));
    ]

(* Errors *)

let edit fs = List.fold_left (fun h f -> f h) nircam fs
let set_s k v h = Fits.Header.set Fits.Value.string k v h
let set_f k v h = Fits.Header.set Fits.Value.float k v h
let set_i k v h = Fits.Header.set Fits.Value.int k v h
let set_t k v h = Fits.Header.set Fits.Value.text k v h

let errors =
  let read_error h = require_error (Wcs.read Frame.icrs (keywords h)) in
  group "Errors"
    [
      test "a frame other than the caller's names the frame to read with"
        (fun () ->
          equal string
            "SCI: ICRS; the caller expects galactic. Read with Frame.icrs."
            (require_error
               (Wcs.read Frame.galactic
                  (keywords (edit [ set_s "EXTNAME" "SCI" ])))));
      cases ~name:fst "unread descriptions"
        [
          ( "FK5 at equinox 1950 needs the time stage, not read yet",
            [ set_s "RADESYS" "FK5"; set_f "EQUINOX" 1950. ] );
          ( "FK4 (EQUINOX 1950 without RADESYS) needs the time stage, not read \
             yet",
            [ Fits.Header.remove "RADESYS"; set_f "EQUINOX" 1950. ] );
          ( "CTYPE1: the SFL projection is not read yet",
            [ set_s "CTYPE1" "RA---SFL"; set_s "CTYPE2" "DEC--SFL" ] );
          ( "PV1_0: the fiducial offset is not read",
            [
              set_s "CTYPE1" "RA---SIN";
              set_s "CTYPE2" "DEC--SIN";
              set_f "PV1_0" 1.;
            ] );
          ( "PV2_3: the SIN projection's parameters are PV2_1 to PV2_2",
            [
              set_s "CTYPE1" "RA---SIN";
              set_s "CTYPE2" "DEC--SIN";
              set_f "PV2_3" 1.;
            ] );
          ( "PV2_1: the ZEA projection takes no parameter",
            [
              set_s "CTYPE1" "RA---ZEA";
              set_s "CTYPE2" "DEC--ZEA";
              set_f "PV2_1" 1.;
            ] );
          ( "PV1_40: TPV's terms are PV1_0 to PV1_39",
            [
              set_s "CTYPE1" "RA---TPV";
              set_s "CTYPE2" "DEC--TPV";
              set_f "PV1_40" 1.;
            ] );
          ( "B_ORDER is absent beside A_ORDER",
            [
              set_s "CTYPE1" "RA---TAN-SIP";
              set_s "CTYPE2" "DEC--TAN-SIP";
              set_i "A_ORDER" 2;
            ] );
          ( "CPDIS1: this distortion is not read yet",
            [ set_s "CPDIS1" "Lookup" ] );
          ("a header with both CD and PC is ambiguous", [ set_f "CD1_1" 1e-5 ]);
          ( "LONPOLE = 180 and PV1_3 = 150 spell one value and disagree",
            [ set_f "LONPOLE" 180.; set_f "PV1_3" 150. ] );
          ( "PV1_5: only PV1_1 to PV1_4 (the native reference point, LONPOLE \
             and LATPOLE) are read on the longitude axis",
            [ set_f "PV1_5" 1. ] );
          ("CROTA2 beside PC is ambiguous", [ set_f "CROTA2" 10. ]);
          ( "3 axes: a celestial pair is read, and a spectral, time or other \
             third axis is not",
            [ set_i "WCSAXES" 3 ] );
          ( "CTYPE1 = 'RA---TAN' and CTYPE2 = 'GLAT-TAN' are not a celestial \
             pair ymir reads",
            [ set_s "CTYPE2" "GLAT-TAN" ] );
          ("CUNIT1 = 'm' is not a celestial unit", [ set_s "CUNIT1" "m" ]);
          ( "CUNIT1 and CUNIT2 differ; one unit for both axes is read",
            [ set_s "CUNIT1" "arcsec" ] );
        ]
        (fun (message, fs) -> equal string message (read_error (edit fs)));
      cases
        ~name:(fun (k, _, _) -> k)
        "a value that does not decode is the view's error"
        [
          ( "CRPIX1",
            (fun h -> Result.map ignore ((keywords h).float "CRPIX1")),
            set_s "CRPIX1" "abc" );
          ( "CRVAL2",
            (fun h -> Result.map ignore ((keywords h).float "CRVAL2")),
            set_t "CRVAL2" "1E400" );
          ( "WCSAXES",
            (fun h -> Result.map ignore ((keywords h).int "WCSAXES")),
            set_t "WCSAXES" "2.0" );
          ( "CTYPE1",
            (fun h -> Result.map ignore ((keywords h).text "CTYPE1")),
            set_i "CTYPE1" 1 );
        ]
        (fun (_, view, f) ->
          let h = edit [ f ] in
          equal string (require_error (view h)) (read_error h));
      test "keywords whose cards disagree are the view's error" (fun () ->
          let h =
            header
              ((List.hd Wcs_reference.cases).header
             ^ "CRVAL1  =                 45.0" ^ String.make 50 ' ')
          in
          equal string
            (require_error ((keywords h).float "CRVAL1"))
            (read_error h));
      test "a stage list FITS cannot spell names the stage" (fun () ->
          let t = read Frame.icrs nircam in
          let c =
            Direction.lonlat Frame.icrs
              ~lon:(deg (scalar 110.))
              ~lat:(deg (scalar (-73.)))
          in
          let r = Quantity.v Unit.one (tensor [| 2 |] [| 1.; 2. |]) in
          let write t =
            require_error (Wcs.write t (keywords Fits.Header.empty))
          in
          equal string
            "Wcs.write: FITS cannot spell the inverse celestial ARC stage there"
            (write Transform.(t >> about c >> inverse (about c)));
          equal string "Wcs.write: FITS cannot spell the shift stage there"
            (write Transform.(shift r >> t));
          equal string
            "Wcs.write: FITS cannot spell the inverse shift stage there"
            (write
               Transform.(
                 axes [| 1; 0 |] ~origin:1 >> inverse (shift r) >> shift r >> t));
          equal string "Wcs.write: FITS cannot spell CRPIX1 = nan"
            (write
               (Nx.Ptree.map (Transform.ptree ())
                  (fun _ x ->
                    Nx.mul x (Nx.cast (Nx.dtype x) (scalar Float.nan)))
                  t)));
    ]

let () = exit (run "Wcs" [ wcslib; round_trips; errors ])
