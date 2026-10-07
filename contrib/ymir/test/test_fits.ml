(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* FITS readers: WCS headers agree with WCSLIB and print back what they read,
   unreadable descriptions are errors naming their keyword, and observations
   read from NIRCam cutouts agree with photutils on the whole mosaic, with
   gradients that match differences. *)

open Windtrap
open Ymir
open Ymir_test.Frames
module Wcs_reference = Ymir_test.Wcs_reference
module Nircam = Ymir_test.Nircam_reference

let f64 = Nx.float64
let ok r = require_ok ~pp:Format.pp_print_string r
let deg x = Quantity.v Unit.degree x
let arcsec x = Quantity.v Unit.arcsecond x
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

let text h = Fits.Header.to_string h
let unit = Testable.make ~pp:Unit.pp ~equal:Unit.equal

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
      let t = ok (Fits.Wcs.read ~alt:c.alt f (header c.header)) in
      let world = Transform.apply t (pixels c.pixels) in
      less float_exact ~than:2e-13
        (Array.fold_left max 0. (separation world (lonlat_deg f c.world))))

(* Law 1: a transform prints back what it read *)

let crota (c : Wcs_reference.case) = c.name = "CROTA2, Galactic"

let round_trips =
  let written_reads_back (c : Wcs_reference.case) h =
    let (F f) = frame c.frame in
    let t = ok (Fits.Wcs.read ~alt:c.alt f h) in
    let written = ok (Fits.Wcs.write ~alt:c.alt t Fits.Header.empty) in
    equal_stages t (ok (Fits.Wcs.read ~alt:c.alt f written))
  in
  group "Law 1"
    [
      cases
        ~name:(fun (c : Wcs_reference.case) -> c.name)
        "a header read at float64 prints back bit for bit"
        (List.filter (fun c -> not (crota c)) Wcs_reference.cases)
        (fun c ->
          let (F f) = frame c.frame in
          let h = header c.header in
          let t = ok (Fits.Wcs.read ~alt:c.alt f h) in
          equal string (text h) (text (ok (Fits.Wcs.write ~alt:c.alt t h))));
      cases
        ~name:(fun (c : Wcs_reference.case) -> c.name)
        "reading what was written gives equal stages" Wcs_reference.cases
        (fun c -> written_reads_back c (header c.header));
      test "CROTA2 prints back as the PC matrix it reads as" (fun () ->
          let c = List.find crota Wcs_reference.cases in
          let h = header c.header in
          let t = ok (Fits.Wcs.read Frame.galactic h) in
          let h' = ok (Fits.Wcs.write t h) in
          equal (option string) None
            (ok (Fits.Header.find Fits.Value.text "CROTA2" h'));
          equal_stages t (ok (Fits.Wcs.read Frame.galactic h')));
      test "a new card goes where the first removed card was" (fun () ->
          let t = ok (Fits.Wcs.read Frame.icrs nircam) in
          let cd =
            Fits.Header.(
              nircam |> remove "PC1_1" |> remove "PC1_2" |> remove "PC2_1"
              |> remove "PC2_2"
              |> set Fits.Value.float "CD1_1" 1e-5)
          in
          let keys h =
            List.map
              (fun r -> String.trim (String.sub r 0 8))
              (Fits.Header.records h)
          in
          equal (list string)
            [
              "NAXIS";
              "RADESYS";
              "WCSAXES";
              "CRPIX1";
              "CRPIX2";
              "CRVAL1";
              "CRVAL2";
              "CTYPE1";
              "CTYPE2";
              "CUNIT1";
              "CUNIT2";
              "CDELT1";
              "CDELT2";
              "PC1_1";
              "PC1_2";
              "PC2_1";
              "PC2_2";
            ]
            (keys (ok (Fits.Wcs.write t cd))));
      test "a window moves CRPIX by its start" (fun () ->
          let t = ok (Fits.Wcs.read Frame.icrs nircam) in
          let h =
            ok (Fits.Wcs.write ~window:[| (10, 114); (20, 124) |] t nircam)
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

let errors =
  let read_error h = require_error (Fits.Wcs.read Frame.icrs h) in
  group "Errors"
    [
      test "a frame other than the caller's names the frame to read with"
        (fun () ->
          equal string
            "SCI: ICRS; the caller expects galactic. Read with Frame.icrs."
            (require_error
               (Fits.Wcs.read Frame.galactic (edit [ set_s "EXTNAME" "SCI" ]))));
      cases ~name:fst "unread descriptions"
        [
          ( "FK5 at equinox 1950 needs the time stage, not read yet",
            [ set_s "RADESYS" "FK5"; set_f "EQUINOX" 1950. ] );
          ( "FK4 (EQUINOX 1950 without RADESYS) needs the time stage, not read \
             yet",
            [ Fits.Header.remove "RADESYS"; set_f "EQUINOX" 1950. ] );
          ( "CTYPE1 = 'RA---TAN-SIP': SIP distortion is not read yet",
            [ set_s "CTYPE1" "RA---TAN-SIP"; set_s "CTYPE2" "DEC--TAN-SIP" ] );
          ( "CTYPE1: the SIN projection is not read yet",
            [ set_s "CTYPE1" "RA---SIN"; set_s "CTYPE2" "DEC--SIN" ] );
          ("a header with both CD and PC is ambiguous", [ set_f "CD1_1" 1e-5 ]);
          ("CROTA2 beside PC is ambiguous", [ set_f "CROTA2" 10. ]);
          ( "PV2_1: TAN with PV terms is the TPV distortion, not read yet",
            [ set_f "PV2_1" 1. ] );
          ( "3 axes: a celestial pair is read, and a spectral, time or other \
             third axis is not",
            [ set_i "WCSAXES" 3 ] );
          ( "CTYPE1 = 'RA---TAN' and CTYPE2 = 'GLAT-TAN' are not a celestial \
             pair ymir reads",
            [ set_s "CTYPE2" "GLAT-TAN" ] );
        ]
        (fun (message, fs) -> equal string message (read_error (edit fs)));
      test "a stage list FITS cannot spell names the stage" (fun () ->
          let t = ok (Fits.Wcs.read Frame.icrs nircam) in
          let c =
            Direction.lonlat Frame.icrs
              ~lon:(deg (scalar 110.))
              ~lat:(deg (scalar (-73.)))
          in
          let r = Quantity.v Unit.one (tensor [| 2 |] [| 1.; 2. |]) in
          let write t = require_error (Fits.Wcs.write t Fits.Header.empty) in
          equal string
            "Fits.Wcs.write: FITS cannot spell the inverse celestial ARC stage \
             there"
            (write Transform.(t >> about c >> inverse (about c)));
          equal string "Fits.Wcs.write: FITS cannot spell the shift stage there"
            (write Transform.(shift r >> t));
          equal string
            "Fits.Wcs.write: FITS cannot spell the inverse shift stage there"
            (write
               Transform.(
                 axes [| 1; 0 |] ~origin:1 >> inverse (shift r) >> shift r >> t)));
    ]

(* Observations *)

let golden = lazy (ok (Fits.read "golden/nircam.fits"))

let observation ?window dtype k =
  ok
    (Fits.observation ~dtype ~frame:Frame.icrs ~data:"SCI" ~error:"ERR" ~ver:k
       ?window (Lazy.force golden))

let target (a : Nircam.aperture) =
  Direction.lonlat Frame.icrs ~lon:(deg (scalar a.ra)) ~lat:(deg (scalar a.dec))

let mjy = ok (Fits.Unit.parse "MJy")
let in_mjy q = Nx.item [] (Nx.cast f64 (Quantity.value mjy q))

let circle dtype a =
  Region.circle
    (Transform.about (target a))
    ~radius:(arcsec (Nx.scalar dtype Nircam.radius))

let annulus dtype a =
  Region.annulus
    (Transform.about (target a))
    ~inner:(arcsec (Nx.scalar dtype Nircam.inner))
    ~outer:(arcsec (Nx.scalar dtype Nircam.outer))

let indexed = List.mapi (fun i a -> (i + 1, a)) Nircam.targets

(* The goal post: |ymir − photutils| ≤ 1e-6 · Σ |data · w · PIXAR_SR|. *)
let goal_post (r : Nircam.sum) v =
  less float_exact ~than:(1e-6 *. r.scale) (Float.abs (v -. r.sum))

let photutils =
  let name (k, _) = Printf.sprintf "target %d" k in
  let sums (type e) (dtype : (float, e) Nx.dtype) (k, (a : Nircam.aperture)) =
    let obs = observation dtype k in
    let c = Observation.integrate (circle dtype a) obs in
    let s = Observation.integrate (annulus dtype a) obs in
    goal_post a.circle (in_mjy c.value);
    goal_post a.annulus (in_mjy s.value);
    (c, s)
  in
  group "photutils"
    [
      cases ~name "float64 sums meet the goal post" indexed (fun t ->
          ignore (sums f64 t));
      cases ~name "float32 sums meet the goal post" indexed (fun t ->
          ignore (sums Nx.float32 t));
      cases ~name "variances are the sums of w² PIXAR_SR² ERR²" indexed
        (fun ((_, (a : Nircam.aperture)) as t) ->
          let c, s = sums f64 t in
          let var (i : _ Observation.integral) =
            let v = Option.get i.variance in
            Nx.item [] (Quantity.value (Unit.( ** ) mjy 2) v)
          in
          equal (float_rel ~rel:1e-6 ~abs:0.) a.circle.variance (var c);
          equal (float_rel ~rel:1e-6 ~abs:0.) a.annulus.variance (var s));
    ]

let reading =
  group "Fits.observation"
    [
      test "the area is PIXAR_SR and the data a field in MJy/sr" (fun () ->
          let obs = observation f64 1 in
          let pixar =
            ok
              (Fits.Header.get Fits.Value.float "PIXAR_SR"
                 (Fits.header (ok (Fits.get ~ver:1 "SCI" (Lazy.force golden)))))
          in
          let area = Option.get (Observation.area obs) in
          equal float_exact pixar
            (Nx.item [] (Quantity.value Unit.steradian area));
          equal string "MJy sr-1"
            (ok (Fits.Unit.print (Quantity.unit (Observation.data obs)))));
      test "a sample is invalid where its value or error is not finite"
        (fun () ->
          let hdus = Lazy.force golden in
          let sci =
            ok (Fits.Image.values f64 (ok (Fits.get ~ver:5 "SCI" hdus)))
          in
          let err =
            ok (Fits.Image.values f64 (ok (Fits.get ~ver:5 "ERR" hdus)))
          in
          let expected = Nx.logical_and (Nx.isfinite sci) (Nx.isfinite err) in
          let valid = Option.get (Observation.valid (observation f64 5)) in
          is_false ~msg:"some sample is invalid" (Nx.item [] (Nx.all expected));
          equal (array bool) (Nx.to_array expected)
            (Nx.to_array (Nx.cast Nx.bool valid)));
      test "a window reads the cells of the whole image's grid" (fun () ->
          let window = [| (10, 90); (20, 96) |] in
          let part = observation ~window f64 2 in
          let whole =
            Observation.window
              ~start:(Nx.create Nx.int64 [| 2 |] [| 10L; 20L |])
              ~shape:[| 80; 76 |] (observation f64 2)
          in
          let a = List.nth Nircam.targets 1 in
          let sum o = in_mjy (Observation.integrate (circle f64 a) o).value in
          equal float_exact (sum whole) (sum part);
          equal (array float_exact)
            (Nx.to_array
               (Quantity.value
                  (Quantity.unit (Observation.data whole))
                  (Observation.data whole)))
            (Nx.to_array
               (Quantity.value
                  (Quantity.unit (Observation.data part))
                  (Observation.data part))));
      test "a BUNIT per pixel gives data per cell" (fun () ->
          let h =
            Fits.Header.(
              nircam
              |> set Fits.Value.string "EXTNAME" "SCI"
              |> set Fits.Value.string "BUNIT" "electron/s/pixel")
          in
          let hdu = Fits.Image.hdu h (Nx.ones Nx.float32 [| 8; 8 |]) in
          let obs =
            ok
              (Fits.observation ~dtype:f64 ~frame:Frame.icrs ~data:"SCI" [ hdu ])
          in
          equal unit
            Unit.(symbol "electron" / second / Grid.cell)
            (Quantity.unit (Observation.data obs)));
      test "data without BUNIT are an error" (fun () ->
          let h = Fits.Header.set Fits.Value.string "EXTNAME" "SCI" nircam in
          let hdu = Fits.Image.hdu h (Nx.ones Nx.float32 [| 8; 8 |]) in
          equal string
            "SCI: no BUNIT gives the data's unit; build the observation with \
             Observation.v"
            (require_error
               (Fits.observation ~dtype:f64 ~frame:Frame.icrs ~data:"SCI"
                  [ hdu ])));
    ]

(* Gradients *)

(* The Guide's flux about target [a]: the circle less the annulus's mean times
   the circle's area, in MJy, as a function of [east; north; radius] in
   arcseconds. *)
let flux (type e) (a : Nircam.aperture)
    (obs : (Frame.icrs Direction.t, e) Observation.t) (p : (float, e) Nx.t) =
  let dtype = Nx.dtype p in
  let centre = arcsec (Nx.cast f64 (Nx.slice [ Nx.R (0, 2) ] p)) in
  let at = Transform.(about (target a) >> shift centre) in
  let sum =
    Observation.integrate
      (Region.circle at ~radius:(arcsec (Nx.slice [ Nx.I 2 ] p)))
      obs
  in
  let sky =
    Observation.integrate
      (Region.annulus at
         ~inner:(arcsec (Nx.scalar dtype Nircam.inner))
         ~outer:(arcsec (Nx.scalar dtype Nircam.outer)))
      obs
  in
  Quantity.(value mjy (sub sum.value (mul (div sky.value sky.area) sum.area)))

let gradients =
  cases
    ~name:(fun (k, _) -> Printf.sprintf "target %d" k)
    "float32 gradients match float64 central differences"
    (List.filteri (fun i _ -> i = 1 || i = 4) indexed)
    (fun (k, a) ->
      let p = [| 0.02; -0.03; Nircam.radius |] in
      let obs = observation f64 k and obs32 = observation Nx.float32 k in
      let at q = Nx.item [] (flux a obs (tensor [| 3 |] q)) in
      let h = 1e-4 in
      let fd =
        Array.init 3 (fun i ->
            let up = Array.copy p and down = Array.copy p in
            up.(i) <- p.(i) +. h;
            down.(i) <- p.(i) -. h;
            (at up -. at down) /. (2. *. h))
      in
      let g = Rune.grad' (flux a obs32) (Nx.create Nx.float32 [| 3 |] p) in
      let scale =
        Array.fold_left (fun m x -> Float.max m (Float.abs x)) 0. fd
      in
      equal (array (float (1e-4 *. scale))) fd (Nx.to_array (Nx.cast f64 g)))

let () =
  exit
    (run "Fits"
       [
         wcslib;
         round_trips;
         errors;
         group "Observations" [ photutils; reading ];
         gradients;
       ])
