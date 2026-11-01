(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The Earth and the geocentric frames: rotation angles, precession-nutation
   matrices, the ephemeris and apparent directions as SOFA's published values
   and ERFA's functions state them, and the laws of the rotations and
   conversions between them. *)

open Windtrap
open Ymir
module Reference = Ymir_test.Earth_reference

let pi = Float.pi
let seconds x = Quantity.v Unit.second x
let in_radians q = Quantity.value Unit.radian q
let scalar x = Nx.scalar Nx.float64 x
let vector xs = Nx.create Nx.float64 [| Array.length xs |] xs
let au = Unit.astronomical_unit
let au_per_day = Unit.(astronomical_unit / day)
let item x = Nx.item [] x

(* [epoch day second ut1] is the Earth at TT [second] past 0h of [day],
   with UT1 [ut1] seconds after TT's clock: ΔT = −[ut1]. *)
let epoch day second ut1 =
  let t =
    Time.v Time.tt
      ~day:(Nx.scalar Nx.int64 (Int64.of_int day))
      ~second:(seconds (scalar second))
  in
  Earth.of_tt t ~delta_t:(seconds (scalar (-.ut1)))

let at_mjd day = epoch day 0. 0.
let entries m = Nx.to_array m
let xyz d = Nx.to_array (Direction.xyz d)
let icrs v = Direction.of_xyz Frame.icrs (vector v)

(* [turn a] is [a] in [(-π, π]], so that angles that differ by whole turns
   compare. *)
let turn a =
  let a = Float.rem a (2. *. pi) in
  if a > pi then a -. (2. *. pi) else if a <= -.pi then a +. (2. *. pi) else a

let angle tol expected actual =
  at_most float_exact ~than:tol (Float.abs (turn (actual -. expected)))

let row_id (r : Reference.row) =
  let d, s, _ = r.instant in
  Printf.sprintf "MJD %d + %.6f s" d s

let of_row (r : Reference.row) =
  let d, s, ut1 = r.instant in
  epoch d s ut1

(* SOFA's published values *)

(* The expected values and tolerances of the IAU SOFA library's test
   program, as ERFA 2.0.1's t_erfa_c.c gives them. Each date there is TT and
   UT1 alike. *)
let sofa =
  group "SOFA's published values"
    [
      test "iauEra00 at UT1 MJD 54388 is 0.4022837240028158102" (fun () ->
          equal (float 1e-12) 0.4022837240028158102
            (item (in_radians (Earth.era (at_mjd 54388)))));
      test "iauGmst06 at MJD 53736 is 1.754174971870091203" (fun () ->
          equal (float 1e-12) 1.754174971870091203
            (item (in_radians (Earth.gmst (at_mjd 53736)))));
      test "iauGst06a at MJD 53736 is 1.754166137675019159" (fun () ->
          equal (float 1e-12) 1.754166137675019159
            (item (in_radians (Earth.gst (at_mjd 53736)))));
      test "iauEo06a at MJD 53736 is ERA - GST, -0.1332882371941833644e-2"
        (fun () ->
          let e = at_mjd 53736 in
          let era = item (in_radians (Earth.era e))
          and gst = item (in_radians (Earth.gst e)) in
          angle 1e-15 (-0.1332882371941833644e-2) (era -. gst));
      test "iauPnm06a at MJD 50123.9999 is the GCRS to true-of-date matrix"
        (fun () ->
          let e = epoch 50123 (0.9999 *. 86400.) 0. in
          equal (array (float 1e-12))
            [|
              0.9999995832794205484; 0.8372382772630962111e-3;
              0.3639684771140623099e-3; -0.8372533744743683605e-3;
              0.9999996486492861646; 0.4132905944611019498e-4;
              -0.3639337469629464969e-3; -0.4163377605910663999e-4;
              0.9999999329094260057;
            |]
            (entries (Frame.matrix_at e Frame.gcrs Frame.true_of_date)));
      test "iauC2i06a at MJD 53736 is the GCRS to CIRS matrix" (fun () ->
          equal (array (float 1e-12))
            [|
              0.9999998323037159379; 0.5581121329587613787e-9;
              -0.5791308487740529749e-3; -0.2384253169452306581e-7;
              0.9999999991917467827; -0.4020579392895682558e-4;
              0.5791308482835292617e-3; 0.4020580099454020310e-4;
              0.9999998314954628695;
            |]
            (entries (Frame.matrix_at (at_mjd 53736) Frame.gcrs Frame.cirs)));
      test "iauEpv00 at TDB MJD 53411.52501161" (fun () ->
          let tdb =
            Time.v Time.tdb
              ~day:(Nx.scalar Nx.int64 53411L)
              ~second:(seconds (scalar (0.52501161 *. 86400.)))
          in
          let p, v = Earth.heliocentric tdb in
          let pb, vb = Earth.barycentric tdb in
          let pos q = Nx.to_array (Quantity.value au q)
          and vel q = Nx.to_array (Quantity.value au_per_day q) in
          equal (array (float 1e-14))
            [|
              -0.7757238809297706813; 0.5598052241363340596;
              0.2426998466481686993;
            |]
            (pos p);
          equal (array (float 1e-15))
            [|
              -0.1091891824147313846e-1; -0.1247187268440845008e-1;
              -0.5407569418065039061e-2;
            |]
            (vel v);
          equal (array (float 1e-14))
            [|
              -0.7714104440491111971; 0.5598412061824171323;
              0.2425996277722452400;
            |]
            (pos pb);
          equal (array (float 1e-15))
            [|
              -0.1091874268116823295e-1; -0.1246525461732861538e-1;
              -0.5404773180966231279e-2;
            |]
            (vel vb));
      test "iauAtic13: CIRS (2.710121572969038991, 0.1729371367218230438) is \
            ICRS (2.710126504531716819, 0.1740632537627034482)"
        (fun () ->
          (* The test's date, JD 2456165.5 + 0.401182685, is TDB there; read
             as TT, it moves the result by under 1e-14 rad. *)
          let e = epoch 56165 (0.401182685 *. 86400.) 0. in
          let rad x = Quantity.v Unit.radian (scalar x) in
          let d =
            Direction.lonlat Frame.cirs ~lon:(rad 2.710121572969038991)
              ~lat:(rad 0.1729371367218230438)
          in
          let c = Direction.convert e Frame.icrs d in
          equal (float 1e-12) 2.710126504531716819
            (item (in_radians (Direction.lon c)));
          equal (float 1e-12) 0.1740632537627034482
            (item (in_radians (Direction.lat c))));
    ]

(* ERFA's functions *)

(* ERA and GMST are against their definitions evaluated exactly, the rest
   against pyerfa's ERFA 2.0.1 at the same instants. *)
let erfa =
  let rows = Reference.rows in
  group "ERFA's functions"
    [
      cases "ERA is IERS (5.14) within 4e-15 rad" ~id:row_id rows (fun r ->
          angle 4e-15 r.era (item (in_radians (Earth.era (of_row r)))));
      cases "GMST is IERS (5.32) within 4e-15 rad" ~id:row_id rows (fun r ->
          angle 4e-15 r.gmst (item (in_radians (Earth.gmst (of_row r)))));
      cases "ERA - GST is eraEo06a within 1e-15 rad" ~id:row_id rows (fun r ->
          let e = of_row r in
          let era = item (in_radians (Earth.era e))
          and gst = item (in_radians (Earth.gst e)) in
          angle 1e-15 r.eo (era -. gst));
      cases "GCRS to true of date is eraPnm06a within 4e-16" ~id:row_id rows
        (fun r ->
          equal (array (float 4e-16)) r.npb
            (entries
               (Frame.matrix_at (of_row r) Frame.icrs Frame.true_of_date)));
      cases "GCRS to CIRS is eraC2i06a within 5e-16" ~id:row_id rows (fun r ->
          equal (array (float 5e-16)) r.c2i
            (entries (Frame.matrix_at (of_row r) Frame.gcrs Frame.cirs)));
      cases "the ephemeris is eraEpv00's within 5e-15 au and 1e-16 au/d"
        ~id:row_id rows (fun r ->
          let tdb = Time.convert Time.tdb (Earth.tt (of_row r)) in
          let p, v = Earth.heliocentric tdb in
          let pb, vb = Earth.barycentric tdb in
          let pos q = Nx.to_array (Quantity.value au q)
          and vel q = Nx.to_array (Quantity.value au_per_day q) in
          equal (array (float 5e-15)) (fst r.heliocentric) (pos p);
          equal (array (float 1e-16)) (snd r.heliocentric) (vel v);
          equal (array (float 5e-15)) (fst r.barycentric) (pos pb);
          equal (array (float 1e-16)) (snd r.barycentric) (vel vb));
      cases
        "ICRS to GCRS is eraLdsun then eraAb, and to CIRS through eraC2i06a, \
         within 2e-15"
        ~id:row_id rows (fun r ->
          let e = of_row r in
          List.iter
            (fun (p, g, c) ->
              let to_ f = xyz (Direction.convert e f (icrs p)) in
              equal (array (float 2e-15)) g (to_ Frame.gcrs);
              equal (array (float 2e-15)) c (to_ Frame.cirs))
            r.directions);
    ]

(* Laws *)

type frame = F : 'f Frame.t -> frame

let frames =
  [ F Frame.icrs; F Frame.galactic; F Frame.gcrs; F Frame.cirs;
    F Frame.true_of_date; F Frame.tirs ]

let pp_frame ppf (F f) = Frame.pp ppf f
let frame = Gen.of_list ~pp:pp_frame frames
let geocentric =
  [ F Frame.gcrs; F Frame.cirs; F Frame.true_of_date; F Frame.tirs ]

let geocentric_frame = Gen.of_list ~pp:pp_frame geocentric
let mjd_1900 = 15020
let mjd_2100 = 88069

(* TT instants from 1900 to 2100, with UT1 within 2 minutes of TT. *)
let instants =
  let open Gen in
  triple (int_range mjd_1900 mjd_2100) (float_range 0. 86399.)
    (float_range (-120.) 120.)

let earth (d, s, u) = epoch d s u

let product a b =
  Array.init 9 (fun k ->
      let i = k / 3 and j = k mod 3 in
      (a.(3 * i) *. b.(j)) +. (a.((3 * i) + 1) *. b.(3 + j))
      +. (a.((3 * i) + 2) *. b.(6 + j)))

let transpose m = Array.init 9 (fun k -> m.((k mod 3 * 3) + (k / 3)))
let identity = [| 1.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 1. |]

let determinant m =
  (m.(0) *. ((m.(4) *. m.(8)) -. (m.(5) *. m.(7))))
  -. (m.(1) *. ((m.(3) *. m.(8)) -. (m.(5) *. m.(6))))
  +. (m.(2) *. ((m.(3) *. m.(7)) -. (m.(4) *. m.(6))))

let matrix e (F a) (F b) = entries (Frame.matrix_at e a b)

(* [near_sun e d] is the ICRS direction [d] radians from the Sun's centre, as
   the geocentre sees it, toward the north. *)
let near_sun e d =
  let tdb = Time.convert Time.tdb (Earth.tt e) in
  let p, _ = Earth.heliocentric tdb in
  let sun = Direction.of_xyz Frame.icrs (Nx.neg (Quantity.value au p)) in
  let lon = item (in_radians (Direction.lon sun))
  and lat = item (in_radians (Direction.lat sun)) in
  let rad x = Quantity.v Unit.radian (scalar x) in
  Direction.lonlat Frame.icrs ~lon:(rad lon) ~lat:(rad (lat +. d))

(* A direction anywhere, or within 2° of the Sun. *)
(* The Sun's deflection is limited within about 5′ of its centre. *)
let limiter = 1.4e-3

let sky =
  let open Gen in
  frequency
    [
      ( 2,
        map
          (fun (l, b) -> `Anywhere (l, b))
          (pair (float_range 0. (2. *. pi)) (float_range (-1.) 1.)) );
      (1, map (fun d -> `Near_sun d) (float_range 0. limiter));
      (1, map (fun d -> `Near_sun d) (float_range limiter 0.035));
    ]

let direction e = function
  | `Anywhere (l, z) ->
      let rad x = Quantity.v Unit.radian (scalar x) in
      Direction.lonlat Frame.icrs ~lon:(rad l) ~lat:(rad (Float.asin z))
  | `Near_sun d -> near_sun e d

let cover_sky = function
  | `Near_sun d ->
      cover ~share:0.15 "within the limiter, 5′ of the Sun" (d < limiter);
      cover ~share:0.15 "near the Sun, out of the limiter" (d >= limiter)
  | `Anywhere _ -> ()

let gap a b =
  Array.fold_left max 0. (Array.map2 (fun x y -> Float.abs (x -. y)) a b)

let laws =
  group "Laws"
    [
      prop "every frame's matrix at an instant is a rotation within 1e-15"
        Gen.(triple instants frame frame)
        (fun (i, a, b) ->
          let m = matrix (earth i) a b in
          equal (array (float 1e-15)) identity (product m (transpose m));
          equal (float 1e-15) 1. (determinant m));
      prop "matrix_at b a is matrix_at a b transposed within 2e-16"
        Gen.(triple instants frame frame)
        (fun (i, a, b) ->
          let e = earth i in
          equal (array (float 2e-16))
            (transpose (matrix e a b))
            (matrix e b a));
      prop "matrix_at a c is matrix_at b c times matrix_at a b within 1e-15"
        Gen.(quad instants frame frame frame)
        (fun (i, a, b, c) ->
          let e = earth i in
          equal (array (float 1e-15)) (matrix e a c)
            (product (matrix e b c) (matrix e a b)));
      prop "the equinox route is the CIO route: true of date to TIRS is \
            R3(GST) within 1e-15"
        instants (fun i ->
          let e = earth i in
          let g = item (in_radians (Earth.gst e)) in
          let c = Float.cos g and s = Float.sin g in
          equal (array (float 1e-15))
            [| c; s; 0.; -.s; c; 0.; 0.; 0.; 1. |]
            (matrix e (F Frame.true_of_date) (F Frame.tirs)));
      prop "ICRS to a geocentric frame and back is the direction within 1e-15"
        Gen.(triple instants sky geocentric_frame)
        (fun (i, where, F g) ->
          cover_sky where;
          let e = earth i in
          let d = direction e where in
          let back = Direction.convert e Frame.icrs (Direction.convert e g d) in
          less float_exact ~than:1e-15 (gap (xyz d) (xyz back)));
      prop "between geocentric frames, a conversion is the matrix's rotation"
        Gen.(triple instants sky (pair geocentric_frame geocentric_frame))
        (fun (i, where, (F a, F b)) ->
          let e = earth i in
          let d = Direction.convert e a (direction e where) in
          let m = entries (Frame.matrix_at e a b) and v = xyz d in
          let row r =
            (m.(3 * r) *. v.(0))
            +. (m.((3 * r) + 1) *. v.(1))
            +. (m.((3 * r) + 2) *. v.(2))
          in
          less float_exact ~than:4e-16
            (gap (Array.init 3 row) (xyz (Direction.convert e b d))));
    ]

(* Batches *)

(* [batch xs] is the Earth at the instants of [xs] as one batch, and their
   directions as one batch. *)
let batch xs =
  let column f = Array.of_list (List.map f xs) in
  let day =
    Nx.create Nx.int64 [| List.length xs |]
      (column (fun ((d, _, _), _) -> Int64.of_int d))
  in
  let second = vector (column (fun ((_, s, _), _) -> s)) in
  let ut1 = vector (column (fun ((_, _, u), _) -> u)) in
  let e =
    Earth.of_tt
      (Time.v Time.tt ~day ~second:(seconds second))
      ~delta_t:(seconds (Nx.neg ut1))
  in
  let d =
    Nx.stack ~axis:0
      (List.map (fun (i, w) -> Direction.xyz (direction (earth i) w)) xs)
  in
  (e, Direction.of_xyz Frame.icrs d)

let batches =
  group "Batches"
    [
      prop "a batch of instants and directions converts as each within 4e-16"
        Gen.(list ~size:(int_range 1 6) (pair instants sky))
        (fun xs ->
          let each (i, w) =
            let e = earth i in
            xyz (Direction.convert e Frame.tirs (direction e w))
          in
          let e, d = batch xs in
          equal (array (float 4e-16))
            (Array.concat (List.map each xs))
            (xyz (Direction.convert e Frame.tirs d)));
      test "an instant broadcasts over directions" (fun () ->
          let e = at_mjd 60000 in
          let d =
            Direction.of_xyz Frame.icrs
              (Nx.create Nx.float64 [| 2; 3 |] [| 1.; 0.; 0.; 0.; 0.; 1. |])
          in
          equal (array int) [| 2; 3 |]
            (Nx.shape (Direction.xyz (Direction.convert e Frame.cirs d)));
          equal (array int) [| 3; 3 |]
            (Nx.shape (Frame.matrix_at e Frame.icrs Frame.galactic)));
    ]

(* Derivatives and compilation *)

let star = icrs [| 0.3; -0.5; 0.8 |]

(* [on_day day second delta_t] is the Earth at TT [second] past 0h of
   [day], with ΔT [delta_t]. *)
let on_day day second delta_t =
  let t =
    Time.v Time.tt
      ~day:(Nx.scalar Nx.int64 (Int64.of_int day))
      ~second:(seconds second)
  in
  Earth.of_tt t ~delta_t:(seconds delta_t)

let tirs day second =
  Direction.convert
    (on_day day second (Nx.full_like second 69.))
    Frame.tirs star

(* [back day second] is a CIRS direction at TT [second] of [day] in ICRS,
   through the inverse of aberration and deflection. *)
let cirs_star = Direction.convert (epoch 60000 0. (-69.)) Frame.cirs star

let back day second =
  Direction.convert
    (on_day day second (Nx.full_like second 69.))
    Frame.icrs cirs_star

(* [central f s h] is [f]'s central difference at [s], step [h], each value
   read less whole turns of the first. *)
let central f s h =
  let a = f (s +. h) and b = f (s -. h) in
  Array.map2 (fun a b -> turn (a -. b) /. (2. *. h)) a b

let values f s = Nx.to_array (f (scalar s))

let derivatives =
  let days = Gen.int_range mjd_1900 mjd_2100 in
  let instant = Gen.(pair days (float_range 10. 86389.)) in
  group "Derivatives and compilation"
    [
      prop "d ERA / d ΔT is −2π 1.00273781191135448 / 86400 within 1e-16 rad/s"
        instant (fun (d, s) ->
          let era dt = in_radians (Earth.era (on_day d (scalar s) dt)) in
          let g = Rune.grad' (fun u -> Nx.sum (era u)) (scalar 69.) in
          equal (float 1e-16)
            (-2. *. pi *. 1.00273781191135448 /. 86400.)
            (item g));
      prop "d TIRS longitude / d TT is its central difference within 1e-12"
        instant (fun (d, s) ->
          let lon s = in_radians (Direction.lon (tirs d s)) in
          let g = Rune.grad' (fun s -> Nx.sum (lon s)) (scalar s) in
          equal (array (float 1e-12))
            (central (values lon) s 0.5)
            [| item g |]);
      prop "d ICRS direction / d TT through the inverse is its central \
            difference within 1e-12"
        instant (fun (d, s) ->
          let x s = Nx.slice [ Nx.I 0 ] (Direction.xyz (back d s)) in
          let g = Rune.grad' (fun s -> Nx.sum (x s)) (scalar s) in
          equal (array (float 1e-12))
            (central (values x) s 10.)
            [| item g |]);
      prop "the Earth's barycentric velocity is its position's derivative \
            within 1e-14 relative"
        instant (fun (d, s) ->
          let tdb s =
            Time.v Time.tdb
              ~day:(Nx.scalar Nx.int64 (Int64.of_int d))
              ~second:(seconds s)
          in
          let p s = Quantity.value au (fst (Earth.barycentric (tdb s))) in
          let v =
            Nx.to_array
              (Quantity.value au_per_day
                 (snd (Earth.barycentric (tdb (scalar s)))))
          in
          let _, dp = Rune.jvp' p (scalar s) (scalar 86400.) in
          let norm = sqrt (Array.fold_left (fun a x -> a +. (x *. x)) 0. v) in
          less float_exact ~than:(1e-14 *. norm) (gap v (Nx.to_array dp)));
      test "ICRS to TIRS compiles to the eager result within 2e-16" (fun () ->
          let second = Nx.linspace Nx.float64 0. 86399. 8 in
          let f s = Direction.xyz (tirs 60000 s) in
          equal (array (float 2e-16))
            (Nx.to_array (f second))
            (Nx.to_array (Rune.jit' f second)));
      test "CIRS to ICRS compiles to the eager result within 2e-16" (fun () ->
          let second = Nx.linspace Nx.float64 0. 86399. 8 in
          let f s = Direction.xyz (back 60000 s) in
          equal (array (float 2e-16))
            (Nx.to_array (f second))
            (Nx.to_array (Rune.jit' f second)));
    ]

(* UTC *)

let utc =
  let id ((y, m, d), (h, mi, s), _, _) =
    Printf.sprintf "%d-%02d-%02d %02d:%02d:%.6f" y m d h mi s
  in
  group "UTC"
    [
      cases "of_utc's ERA is at eraUtcut1's UT1 within 4e-15 rad" ~id
        Reference.utc (fun (date, clock, dut1, era) ->
          let e =
            Earth.of_utc
              (Time.of_date Time.utc date clock)
              ~dut1:(seconds (scalar dut1))
          in
          angle 4e-15 era (item (in_radians (Earth.era e))));
      test "before 1972, UT1 - UTC is read against UTC's clock" (fun () ->
          let era (h, mi, s) dut1 =
            item
              (in_radians
                 (Earth.era
                    (Earth.of_utc
                       (Time.of_date Time.utc (1965, 6, 15) (h, mi, s))
                       ~dut1:(seconds (scalar dut1)))))
          in
          angle 4e-15 (era (12, 30, 7.5) 0.) (era (12, 30, 7.25) 0.25));
    ]

(* Failures and structure *)

let failures =
  group "Failures and structure"
    [
      test "of_tt refuses an infinite ΔT" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"of_tt: delta_t is infinite")
            (fun () -> epoch 60000 0. Float.infinity));
      test "of_utc refuses a UT1 - UTC that is not a time" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Quantity.value") (fun () ->
              Earth.of_utc
                (Time.of_date Time.utc (2025, 1, 1) (0, 0, 0.))
                ~dut1:(Quantity.v Unit.metre (scalar 0.))));
      test "of_tt reads ΔT as TT − UT1" (fun () ->
          (* At ΔT = 64 s, TT 12:01:04 is UT1 12:00:00 of MJD 51544, J2000.0
             in UT1, whose ERA is 2π 0.7790572732640. *)
          let e = on_day 51544 (scalar 43264.) (scalar 64.) in
          angle 4e-15 (2. *. pi *. 0.7790572732640)
            (item (in_radians (Earth.era e))));
      test "convert refuses an infinite vector" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"is infinite") (fun () ->
              let d =
                Nx.Ptree.map (Direction.ptree ())
                  (fun _ x ->
                    Nx.mul x (Nx.cast (Nx.dtype x) (scalar Float.infinity)))
                  (icrs [| 1.; 0.; 0. |])
              in
              Direction.convert (at_mjd 60000) Frame.cirs d));
      test "names" (fun () ->
          equal (list string) [ "gcrs"; "cirs"; "true_of_date"; "tirs" ]
            (List.map (fun (F f) -> Frame.name f) geocentric));
      test "visits" (fun () ->
          List.iter
            (fun v -> Format.printf "%a@." Nx.Ptree.pp_visit v)
            (Nx.Ptree.visits (Earth.ptree ()) (at_mjd 60000));
          expect (output ())
          @@ __POS_OF__
               {|
            tt: case "tt"
            tt.day: a leaf
            tt.second: a leaf
            ut1_day: a leaf
            ut1_second: a leaf
            |});
    ]

let () =
  exit
    (run "Earth" [ sofa; erfa; utc; laws; batches; derivatives; failures ])
