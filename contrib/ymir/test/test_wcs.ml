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
let read ?alt f h = ok (Wcs.read ?alt ~at:Nx.Host.on (Transform.Sky f) (keywords h))
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
    "WCSLIB" ~id:(fun (c : Wcs_reference.case) -> c.name) Wcs_reference.cases
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
        "a header read at float64 needs no edit" ~id:(fun (c : Wcs_reference.case) -> c.name)
        (List.filter (fun c -> not (crota c || tan_pv c)) Wcs_reference.cases)
        (fun c ->
          let (F f) = frame c.frame in
          let h = header c.header in
          equal edits [] (write ~alt:c.alt (read ~alt:c.alt f h) h));
      cases
        "edits to an empty header read back as equal stages" ~id:(fun (c : Wcs_reference.case) -> c.name) Wcs_reference.cases
        (fun c -> reads_back c Fits.Header.empty);
      cases
        "edits to the header read back as equal stages" ~id:(fun (c : Wcs_reference.case) -> c.name) Wcs_reference.cases
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
  let read_error h = require_error (Wcs.read ~at:Nx.Host.on (Transform.Sky Frame.icrs) (keywords h)) in
  group "Errors"
    [
      test "a frame other than the caller's names the frame to read with"
        (fun () ->
          equal string
            "SCI: ICRS; the caller expects galactic. Read with Frame.icrs."
            (require_error
               (Wcs.read ~at:Nx.Host.on (Transform.Sky Frame.galactic)
                  (keywords (edit [ set_s "EXTNAME" "SCI" ])))));
      cases "unread descriptions" ~id:fst
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
          ( "3 axes, 1 celestial pair and 1 linear axis; the caller expects 1 \
             celestial pair",
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
        "a value that does not decode is the view's error" ~id:(fun (k, _, _) -> k)
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
      test
        "LATPOLE alone fixes the pole for θ₀ = 0, CRVAL's latitude 0 and \
         LONPOLE 90° from φ₀"
        (fun () ->
          (* A plate carrée with unit steps from pixel (0, 0): pixel (90, 0)
             is the native pole. FITS gives LATPOLE no default here and takes
             it as the pole's latitude, where astropy (WCSLIB) reads the
             header without it and puts the pole at 90° whatever it says. *)
          let car =
            [
              set_s "CTYPE1" "RA---CAR"; set_s "CTYPE2" "DEC--CAR";
              set_f "CRVAL1" 10.; set_f "CRVAL2" 0.; set_f "CRPIX1" 1.;
              set_f "CRPIX2" 1.; set_f "CDELT1" 1.; set_f "CDELT2" 1.;
              set_f "PC1_1" 1.; set_f "PC1_2" 0.; set_f "PC2_1" 0.;
              set_f "PC2_2" 1.; set_f "LONPOLE" 90.;
            ]
          in
          equal string
            "LATPOLE is absent, and FITS gives it no default where θ₀ = 0, \
             CRVAL's latitude is 0 and LONPOLE is 90° from φ₀: it alone fixes \
             the celestial pole"
            (read_error (edit car));
          let h = edit (car @ [ set_f "LATPOLE" 30. ]) in
          let t = read Frame.icrs h in
          let world = Transform.apply t (pixels [| 0.; 0.; 90.; 0. |]) in
          let off =
            separation world (lonlat_deg Frame.icrs [| 10.; 0.; -80.; 30. |])
          in
          less float_exact ~than:1e-12 (Array.fold_left max 0. off);
          let h' = apply (write t Fits.Header.empty) Fits.Header.empty in
          equal (option float_exact) (Some 30.)
            (ok (Fits.Header.find Fits.Value.float "LATPOLE" h')));
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
      test "a celestial stage in a geocentric frame has no header" (fun () ->
          let t =
            Transform.(
              axes [| 1; 0 |] ~origin:1
              >> shift (Quantity.v Unit.one (tensor [| 2 |] [| 1.; 1. |]))
              >> linear (deg (tensor [| 2; 2 |] [| -1e-3; 0.; 0.; 1e-3 |]))
              >> celestial Tan Frame.cirs ~pv:(Nx.zeros Nx.float64 [| 0 |])
                   ~native:(deg (tensor [| 2 |] [| 0.; 90. |]))
                   ~crval:(deg (tensor [| 2 |] [| 10.; 20. |]))
                   ~lonpole:(deg (scalar 180.))
                   ~latpole:(deg (scalar 90.)))
          in
          equal string
            "Wcs.write: ymir writes no header for a celestial stage in cirs, \
             whose orientation depends on an instant"
            (require_error (Wcs.write t (keywords Fits.Header.empty))));
    ]

(* N axes *)

let cube_name (c : Wcs_reference.cube) = c.name
let cube_frame (c : Wcs_reference.cube) =
  if c.cube_frame = "" then F Frame.icrs else frame c.cube_frame

(* [world c] is the world [c]'s axes read as: its celestial pair first, then
   its linear axes in FITS order. *)
type each = { each : 'w. ('w, Nx.host) Transform.endpoint -> unit }

let with_world (c : Wcs_reference.cube) k =
  let (F f) = cube_frame c in
  match (Array.length c.celestial, Array.length c.shape - Array.length c.celestial) with
  | 2, 1 -> k.each Transform.(Both (Sky f, Planar))
  | 2, 2 -> k.each Transform.(Both (Sky f, Both (Planar, Planar)))
  | 0, 1 -> k.each Transform.Planar
  | _ -> invalid_arg "with_world"

let read_world ?alt w h = ok (Wcs.read ?alt ~at:Nx.Host.on w (keywords h))

let unit_of s = if s = "" then Unit.one else ok (Fits.Unit.parse s)

(* WCSLIB's linear axes are within a few units in the last place of their
   values. *)
let linear_close (c : Wcs_reference.cube) k (q : (float, Nx.float64_elt, Nx.host) Nx.t Quantity.t) =
  let n = Array.length c.shape in
  let rows = Array.length c.pixels / n in
  let cols = Array.length c.world / rows in
  let column = 2 * (Array.length c.celestial / 2) + k in
  let expected = Array.init rows (fun r -> c.world.((r * cols) + column)) in
  equal
    (array (float_rel ~rel:1e-13 ~abs:0.))
    expected
    (Nx.to_array (Quantity.value (unit_of c.units.(k)) q))

let sky_close (c : Wcs_reference.cube) f d =
  let n = Array.length c.shape in
  let rows = Array.length c.pixels / n in
  let cols = Array.length c.world / rows in
  let lonlat = Array.concat (List.init rows (fun r -> [| c.world.(r * cols); c.world.((r * cols) + 1) |])) in
  less float_exact ~than:2e-13
    (Array.fold_left max 0. (separation d (lonlat_deg f lonlat)))

let cube_points (c : Wcs_reference.cube) =
  let n = Array.length c.shape in
  Quantity.v Unit.one (tensor [| Array.length c.pixels / n; n |] c.pixels)

let cubes_wcslib =
  cases "WCSLIB, N axes" ~id:cube_name Wcs_reference.cubes (fun c ->
      let h = header c.header and x = cube_points c in
      let (F f) = cube_frame c in
      match (Array.length c.celestial, Array.length c.shape - Array.length c.celestial) with
      | 2, 1 ->
          let d, l = Transform.apply (read_world Transform.(Both (Sky f, Planar)) h) x in
          sky_close c f d;
          linear_close c 0 l
      | 2, 2 ->
          let d, (l, m) =
            Transform.apply (read_world Transform.(Both (Sky f, Both (Planar, Planar))) h) x
          in
          sky_close c f d;
          linear_close c 0 l;
          linear_close c 1 m
      | _ -> linear_close c 0 (Transform.apply (read_world Transform.Planar h) x))

(* The fixtures whose matrix joins the celestial pair to a linear axis. *)
let coupled (c : Wcs_reference.cube) =
  List.mem c.name
    [
      "PC3_1 couples VRAD with RA";
      "RA, FREQ, DEC with PC2_1 coupling FREQ with RA";
      "CD3_1 couples AWAV with RA";
    ]

let cubes_laws =
  group "Laws, N axes"
    [
      cases "a header read needs no edit" ~id:cube_name Wcs_reference.cubes (fun c ->
          let h = header c.header in
          with_world c { each = (fun w -> equal edits [] (write (read_world w h) h)) });
      cases "edits to an empty header read back as equal stages" ~id:cube_name
        Wcs_reference.cubes (fun c ->
          let h = header c.header in
          with_world c
            {
              each =
                (fun w ->
                  let t = read_world w h in
                  let e = Fits.Header.empty in
                  equal_stages t (read_world w (apply (write t e) e)));
            });
      cases "a cube's celestial side agrees with the header of its celestial axes alone"
        ~id:cube_name
        (List.filter (fun (c : Wcs_reference.cube) -> c.celestial <> [||] && not (coupled c)) Wcs_reference.cubes)
        (fun c ->
          let (F f) = cube_frame c in
          let side t =
            let g = Grid.pixels ~at:Nx.Host.on ~shape:c.shape f64 t in
            let sky = Grid.first g in
            let alone =
              Grid.pixels ~at:Nx.Host.on ~shape:(Grid.shape sky) f64
                (read f (header c.alone))
            in
            equal (array bool) [| true |] (Nx.to_array (Nx.reshape [| 1 |] (Grid.agree sky alone)))
          in
          match Array.length c.shape with
          | 3 -> side (read_world Transform.(Both (Sky f, Planar)) (header c.header))
          | _ -> side (read_world Transform.(Both (Sky f, Both (Planar, Planar))) (header c.header)));
      test "a coupled cube's cells have no measure and no region, naming the linear stage"
        (fun () ->
          let c = List.find coupled Wcs_reference.cubes in
          let centre =
            Direction.lonlat Frame.icrs ~lon:(deg (scalar 45.25)) ~lat:(deg (scalar 60.75))
          in
          let t = read_world Transform.(Both (Sky Frame.icrs, Planar)) (header c.header) in
          let g = Grid.pixels ~at:Nx.Host.on ~shape:c.shape f64 t in
          raises_match
            (Exn.invalid_arg
               ~sub:"Grid.measure: the cells are not products: stage 2, linear")
            (fun () -> Grid.measure g);
          raises_match
            (Exn.invalid_arg
               ~sub:"Region.weights: the cells are not products: stage 2, linear")
            (fun () ->
              Region.weights
                (Region.circle
                   Transform.(both (about centre) id >> inverse (split [| 0; 1 |]))
                   ~radius:(Quantity.v Unit.radian (scalar 1e-3)))
                g));
      test "a window moves each CRPIX by its start" (fun () ->
          let c = List.hd Wcs_reference.cubes in
          let h = header c.header in
          let t = read_world Transform.(Both (Sky Frame.fk5_j2000, Planar)) h in
          let h' = apply (write ~window:[| (2, 5); (1, 4); (3, 5) |] t h) h in
          let get k = ok (Fits.Header.get Fits.Value.float k h') in
          let was k = ok (Fits.Header.get Fits.Value.float k h) in
          equal (array float_exact)
            [| was "CRPIX1" -. 3.; was "CRPIX2" -. 1.; was "CRPIX3" -. 2. |]
            [| get "CRPIX1"; get "CRPIX2"; get "CRPIX3" |]);
    ]

let cube_errors =
  let muse = header (List.hd Wcs_reference.cubes).header in
  let radio = header (List.nth Wcs_reference.cubes 1).header in
  let edit fs = List.fold_left (fun h f -> f h) muse fs in
  let read_as world h = require_error (Wcs.read ~at:Nx.Host.on world (keywords h)) in
  let read_error h = read_as Transform.(Both (Sky Frame.fk5_j2000, Planar)) h in
  let linear_unit ctype cunit =
    let h = radio |> set_s "CTYPE3" ctype |> set_s "CUNIT3" cunit in
    let t = read_world Transform.(Both (Sky Frame.icrs, Both (Planar, Planar))) h in
    let _, (l, _) = Transform.apply t (cube_points (List.nth Wcs_reference.cubes 1)) in
    Unit.to_string (Quantity.unit l)
  in
  group "Errors, N axes"
    [
      cases "unread descriptions" ~id:fst
        [
          ( "CTYPE3 = 'WAVE-F2W': the spectral algorithm F2W is not read yet",
            [ set_s "CTYPE3" "WAVE-F2W" ] );
          ( "CTYPE3 = 'WAVE-LOG': the spectral algorithm LOG is not read yet",
            [ set_s "CTYPE3" "WAVE-LOG" ] );
          ( "CTYPE3 = 'WAVE-TAB': the spectral algorithm TAB is not read yet",
            [ set_s "CTYPE3" "WAVE-TAB" ] );
          ("CTYPE3 = 'TIME': a time axis is not read yet", [ set_s "CTYPE3" "TIME" ]);
          ( "CUNIT3: \"furlong\": furlong is not a FITS unit",
            [ set_s "CUNIT3" "furlong" ] );
          ("CUNIT3 = 'Hz' is not a unit of 'AWAV'", [ set_s "CUNIT3" "Hz" ]);
          ("PV3_1: a linear axis takes no parameter", [ set_f "PV3_1" 1. ]);
          ( "CD3_3 is absent: a CD header's linear axis needs it",
            [ Fits.Header.remove "CD3_3"; set_f "CDELT3" 1.25 ] );
          ("CD3_3 = 0 maps every pixel to CRVAL3", [ set_f "CD3_3" 0. ]);
        ]
        (fun (message, fs) -> equal string message (read_error (edit fs)));
      test "a linear axis whose CDELT is 0 maps every pixel to CRVAL" (fun () ->
          equal string "CDELT3 = 0 maps every pixel to CRVAL3"
            (read_as Transform.(Both (Sky Frame.icrs, Both (Planar, Planar)))
               (set_f "CDELT3" 0. radio)));
      test "a linear axis whose PC element is 0 maps every pixel to CRVAL"
        (fun () ->
          equal string "PC3_3 = 0 maps every pixel to CRVAL3"
            (read_as Transform.(Both (Sky Frame.icrs, Both (Planar, Planar)))
               (set_f "PC3_3" 0. radio)));
      test "a celestial CUNIT reads in FITS's unit syntax" (fun () ->
          let h =
            radio |> set_s "CUNIT1" "mrad" |> set_s "CUNIT2" "mrad"
          in
          let t = read_world Transform.(Both (Sky Frame.icrs, Both (Planar, Planar))) h in
          let e = Fits.Header.empty in
          equal_stages t
            (read_world Transform.(Both (Sky Frame.icrs, Both (Planar, Planar)))
               (apply (write t e) e));
          equal string (Unit.to_string Unit.(milli radian))
            (match Fits.Header.find Fits.Value.string "CUNIT1" (apply (write t e) e) with
             | Ok (Some s) -> Unit.to_string (ok (Fits.Unit.parse s))
             | _ -> "absent"));
      test "a header read as other axes than it has names both" (fun () ->
          equal string
            "3 axes, 1 celestial pair and 1 linear axis; the caller expects 1 \
             celestial pair"
            (read_as (Transform.Sky Frame.fk5_j2000) muse);
          equal string
            "3 axes, 1 celestial pair and 1 linear axis; the caller expects 1 \
             celestial pair and 2 linear axes"
            (read_as Transform.(Both (Sky Frame.fk5_j2000, Both (Planar, Planar))) muse));
      test "a world of two celestial pairs is an error" (fun () ->
          let two =
            List.fold_left
              (fun h f -> f h)
              (set_i "WCSAXES" 4 nircam)
              [ set_s "CTYPE3" "GLON-TAN"; set_s "CTYPE4" "GLAT-TAN" ]
          in
          equal string "SCI: the caller expects 2 celestial pairs; Wcs reads one"
            (read_as
               Transform.(Both (Sky Frame.icrs, Sky Frame.icrs))
               (set_s "EXTNAME" "SCI" two)));
      test "the keywords of a side's errors are the header's" (fun () ->
          let c = List.nth Wcs_reference.cubes 2 in
          equal string "CUNIT3 = 'm' is not a celestial unit"
            (require_error
               (Wcs.read ~at:Nx.Host.on Transform.(Both (Sky Frame.icrs, Planar))
                  (keywords (set_s "CUNIT3" "m" (header c.header))))));
      test "a linear axis's CUNIT reads in FITS's unit syntax" (fun () ->
          let unit s = Unit.to_string (ok (Fits.Unit.parse s)) in
          equal string (unit "km s-1") (linear_unit "VRAD" "km.s-1");
          equal string (unit "keV") (linear_unit "ENER" "keV");
          equal string (unit "arcsec") (linear_unit "OFFSET" "arcsec"));
      test "SPECSYS reads as stated beside another SSYSOBS" (fun () ->
          let h = radio |> set_s "SSYSOBS" "TOPOCENT" in
          let t = Transform.(Both (Sky Frame.icrs, Both (Planar, Planar))) in
          equal_stages (read_world t radio) (read_world t h));
    ]

(* Laws over drawn headers: a celestial pair at any two of three or four
   axes, CD or PC, the pair's matrix block diagonal or rotated, beside FREQ
   and STOKES. *)

let strf = Printf.sprintf

type drawn = { n : int; pair : int * int; cd : bool; rotated : bool }

let pp_drawn ppf d =
  Format.fprintf ppf "%d axes, pair (%d, %d), %s%s" d.n (fst d.pair) (snd d.pair)
    (if d.cd then "CD" else "PC")
    (if d.rotated then ", rotated" else "")

let drawn =
  Gen.with_pp pp_drawn
    Gen.(
      map
        (fun (n, (i, j), (cd, rotated)) ->
          let n = 3 + n in
          let i = 1 + (i mod n) in
          let j = 1 + (j mod (n - 1)) in
          let j = if j >= i then j + 1 else j in
          { n; pair = (min i j, max i j); cd; rotated })
        (triple (int_range 0 1) (pair (int_range 0 11) (int_range 0 11)) (pair bool bool)))

(* [drawn_header d] is [d]'s header, and its celestial axes' alone,
   renumbered. *)
let drawn_header d =
  let a, b = d.pair in
  let lin = List.filter (fun k -> k <> a && k <> b) (List.init d.n succ) in
  let c = if d.rotated then 0.8 else 1. and s = if d.rotated then 0.6 else 0. in
  let sky (a, b) =
    [
      set_s (strf "CTYPE%d" a) "RA---TAN"; set_s (strf "CTYPE%d" b) "DEC--TAN";
      set_f (strf "CRPIX%d" a) 10.5; set_f (strf "CRPIX%d" b) 8.25;
      set_f (strf "CRVAL%d" a) 210.5; set_f (strf "CRVAL%d" b) (-30.25);
    ]
    @
    if d.cd then
      [
        set_f (strf "CD%d_%d" a a) (-1e-3 *. c); set_f (strf "CD%d_%d" a b) (1e-3 *. s);
        set_f (strf "CD%d_%d" b a) (1e-3 *. s); set_f (strf "CD%d_%d" b b) (1e-3 *. c);
      ]
    else
      [
        set_f (strf "CDELT%d" a) (-1e-3); set_f (strf "CDELT%d" b) 1e-3;
        set_f (strf "PC%d_%d" a a) c; set_f (strf "PC%d_%d" a b) (-.s);
        set_f (strf "PC%d_%d" b a) s; set_f (strf "PC%d_%d" b b) c;
      ]
  in
  let linear k (ctype, crval, cdelt) =
    [ set_s (strf "CTYPE%d" k) ctype; set_f (strf "CRPIX%d" k) 1.;
      set_f (strf "CRVAL%d" k) crval ]
    @ if d.cd then [ set_f (strf "CD%d_%d" k k) cdelt ] else [ set_f (strf "CDELT%d" k) cdelt ]
  in
  let build fs n = List.fold_left (fun h f -> f h) (set_i "NAXIS" n Fits.Header.empty) fs in
  let axes = List.combine lin (List.filteri (fun k _ -> k < List.length lin) [ ("FREQ", 1.4e9, 2.5e5); ("STOKES", 1., 1.) ]) in
  ( build (sky (a, b) @ List.concat_map (fun (k, l) -> linear k l) axes) d.n,
    build (sky (1, 2)) 2 )

let drawn_world d k =
  if d.n = 3 then k.each Transform.(Both (Sky Frame.icrs, Planar))
  else k.each Transform.(Both (Sky Frame.icrs, Both (Planar, Planar)))

let drawn_laws =
  prop "a drawn header needs no edit, reads back from its edits, and its \
        celestial side agrees with its celestial axes alone" drawn (fun d ->
      cover "the pair's axes apart" (snd d.pair - fst d.pair > 1);
      cover "four axes" (d.n = 4);
      cover "CD" d.cd;
      cover "PC" (not d.cd);
      cover "a rotated pair" d.rotated;
      let h, alone = drawn_header d in
      drawn_world d
        {
          each =
            (fun w ->
              let t = read_world w h in
              equal edits [] (write t h);
              let e = Fits.Header.empty in
              equal_stages t (read_world w (apply (write t e) e)));
        };
      let side t =
        let g = Grid.pixels ~at:Nx.Host.on ~shape:(Array.make d.n 3) f64 t in
        let sky = Grid.first g in
        let own = Grid.pixels ~at:Nx.Host.on ~shape:(Grid.shape sky) f64 (read Frame.icrs alone) in
        equal (array bool) [| true |] (Nx.to_array (Nx.reshape [| 1 |] (Grid.agree sky own)))
      in
      if d.n = 3 then side (read_world Transform.(Both (Sky Frame.icrs, Planar)) h)
      else side (read_world Transform.(Both (Sky Frame.icrs, Both (Planar, Planar))) h))

let () =
  exit
    (run "Wcs"
       [
         wcslib;
         round_trips;
         errors;
         cubes_wcslib;
         cubes_laws;
         group "Drawn headers" [ drawn_laws ];
         cube_errors;
       ])
