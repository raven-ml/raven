(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Fixed frames: each orientation is its standard's definition rounded once, a
   conversion depends only on its two frames, and the conversions reproduce
   ERFA's and astropy's. *)

open Windtrap
open Ymir
open Ymir_test.Frames
module Reference = Ymir_test.Frames_reference

let bits = array float_exact
let entries a b = match (a, b) with F a, F b -> Nx.to_array (Frame.matrix a b)
let transpose m = Array.init 9 (fun k -> m.((k mod 3 * 3) + (k / 3)))
let frame = Gen.of_list ~pp:pp_fixed fixed
let pairs = Gen.pair frame frame
let triples = Gen.triple frame frame frame

(* [mul m n] is [m · n] for row-major [3; 3] arrays, in double-double and
   rounded once. *)
let mul m n =
  Array.init 9 (fun k ->
      let i = k / 3 and j = k mod 3 in
      List.fold_left
        (fun acc q ->
          dd_add acc (dd_mul (dd m.((3 * i) + q)) (dd n.((3 * q) + j))))
        (dd 0.) [ 0; 1; 2 ]
      |> fst)

let at_lonlat (F f) (l, la) =
  Direction.xyz
    (Direction.lonlat f ~lon:(radians (scalar l)) ~lat:(radians (scalar la)))

(* [rotate_lonlat a b (lon, lat)] is the vector at [(lon, lat)], in radians of
   frame [a], in frame [b]. *)
let rotate_lonlat (F a) (F b) (l, la) =
  Direction.xyz
    (Direction.rotate b
       (Direction.lonlat a ~lon:(radians (scalar l)) ~lat:(radians (scalar la))))

(* [same_ray ~tol v w] states that the vectors [v] and [w] are within [tol]
   radians. *)
let same_ray ~tol v w =
  less float_exact ~than:tol
    (one (separation (raw Frame.icrs v) (raw Frame.icrs w)))

(* Orientations *)

let erfa_galactic =
  [|
    "-0.054875560416215368492398900454";
    "-0.873437090234885048760383168409";
    "-0.483835015548713226831774175116";
    "+0.494109427875583673525222371358";
    "-0.444829629960011178146614061616";
    "+0.746982244497218890527388004556";
    "-0.867666149019004701181616534570";
    "-0.198076373431201528180486091412";
    "+0.455983776175066922272100478348";
  |]

let orientations =
  group "Each orientation is its definition, rounded once"
    [
      cases ~name "the orientation from ICRS is the 60-digit value's float"
        fixed (fun f ->
          equal bits (Array.map fst (orientation f)) (entries (F Frame.icrs) f));
      test "Galactic is ERFA's 30-digit table, bit for bit" (fun () ->
          equal bits
            (Array.map float_of_string erfa_galactic)
            (entries (F Frame.icrs) (F Frame.galactic)));
      test "the J2000 ecliptic is pyerfa's ecm06 at J2000, within 2e-16"
        (fun () ->
          equal
            (array (float 2e-16))
            Reference.ecm06_j2000
            (entries (F Frame.icrs) (F Frame.ecliptic_j2000)));
      test "matrix is [3; 3] float64" (fun () ->
          let m = Frame.matrix Frame.galactic Frame.ecliptic_j2000 in
          equal (array int) [| 3; 3 |] (Nx.shape m));
    ]

(* A conversion depends only on its two frames *)

let ulp_at_one = epsilon_float

let conversions =
  group "A conversion depends only on its two frames"
    [
      prop "matrix a a is the identity" frame (fun f ->
          equal bits [| 1.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 1. |] (entries f f));
      prop "matrix b a is the transpose of matrix a b, bit for bit" pairs
        (fun (a, b) -> equal bits (transpose (entries a b)) (entries b a));
      prop "matrix a b is R_b R_aᵀ within 2 ulp" pairs (fun (a, b) ->
          equal
            (array (float (2. *. ulp_at_one)))
            (Array.map fst (exact_matrix a b))
            (entries a b));
      prop "matrix a b is orthogonal within 2 ulp" pairs (fun (a, b) ->
          let m = entries a b in
          equal
            (array (float (2. *. ulp_at_one)))
            [| 1.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 1. |]
            (mul m (transpose m)));
      prop "a route through a third frame agrees within 3 ulp" triples
        (fun (a, b, c) ->
          equal
            (array (float (3. *. ulp_at_one)))
            (entries a c)
            (mul (entries b c) (entries a b)));
    ]

(* ERFA and astropy *)

let tol = 3e-15

let erfa =
  group "Conversions reproduce ERFA's test vectors"
    [
      test "t_icrs2g" (fun () ->
          same_ray ~tol
            (rotate_lonlat (F Frame.icrs) (F Frame.galactic)
               (5.9338074302227188048671087, -1.1784870613579944551540570))
            (at_lonlat (F Frame.galactic)
               (5.5850536063818546461558, -0.7853981633974483096157)));
      test "t_g2icrs" (fun () ->
          same_ray ~tol
            (rotate_lonlat (F Frame.galactic) (F Frame.icrs)
               (5.5850536063818546461558105, -0.7853981633974483096156608))
            (at_lonlat (F Frame.icrs)
               (5.9338074302227188048671, -1.1784870613579944551541)));
      test "t_h2fk5, position at J2000" (fun () ->
          same_ray ~tol
            (rotate_lonlat (F Frame.icrs) (F Frame.fk5_j2000)
               (1.767794352, -0.2917512594))
            (at_lonlat (F Frame.fk5_j2000)
               (1.767794455700065506, -0.2917513626469638890)));
      test "t_fk52h, position at J2000" (fun () ->
          same_ray ~tol
            (rotate_lonlat (F Frame.fk5_j2000) (F Frame.icrs)
               (1.76779433, -0.2917517103))
            (at_lonlat (F Frame.icrs)
               (1.767794226299947632, -0.2917516070530391757)));
      test "t_fk5hip: matrix fk5_j2000 icrs is r5h" (fun () ->
          equal
            (array (float 1e-17))
            [|
              0.9999999999999928638;
              0.1110223351022919694e-6;
              0.4411803962536558154e-7;
              -0.1110223308458746430e-6;
              0.9999999999999891830;
              -0.9647792498984142358e-7;
              -0.4411805033656962252e-7;
              0.9647792009175314354e-7;
              0.9999999999999943728;
            |]
            (entries (F Frame.fk5_j2000) (F Frame.icrs)));
    ]

let degrees_to_radians d = d *. Float.pi /. 180.

let astropy_ray (ra, dec, v) =
  let icrs =
    raw Frame.icrs
      (at_lonlat (F Frame.icrs) (degrees_to_radians ra, degrees_to_radians dec))
  in
  (icrs, v)

let astropy =
  group "Conversions against astropy"
    [
      cases
        ~name:(fun (ra, dec, _) -> Printf.sprintf "ecliptic at (%g, %g)" ra dec)
        "the barycentric mean ecliptic agrees within 1e-11 rad"
        Reference.astropy_ecliptic
        (fun row ->
          let icrs, v = astropy_ray row in
          same_ray ~tol:1e-11
            (Direction.xyz (Direction.rotate Frame.ecliptic_j2000 icrs))
            (Nx.create Nx.float64 [| 3 |] v));
      test "astropy's Galactic is up to 24.9 mas from this one" (fun () ->
          List.iter
            (fun row ->
              let icrs, v = astropy_ray row in
              let ours = Direction.rotate Frame.galactic icrs in
              let theirs =
                raw Frame.galactic (Nx.create Nx.float64 [| 3 |] v)
              in
              Printf.printf "%.1f mas\n"
                (one (separation ours theirs) *. 180. /. Float.pi *. 3.6e6))
            Reference.astropy_galactic;
          expect (output ()) @@ __POS_OF__ {|
            20.3 mas
            24.0 mas
            24.7 mas
            11.0 mas
            24.9 mas
            21.2 mas
            22.5 mas
            19.1 mas
            |});
    ]

let supergalactic =
  let pole = (degrees_to_radians 47.37, degrees_to_radians 6.32) in
  let origin = (degrees_to_radians 137.37, 0.) in
  group "Supergalactic is defined on Galactic"
    [
      test "its north pole is at Galactic (47.37°, +6.32°)" (fun () ->
          same_ray ~tol:1e-15
            (rotate_lonlat (F Frame.galactic) (F Frame.supergalactic) pole)
            (vector 0. 0. 1.));
      test "its origin is at Galactic (137.37°, 0°)" (fun () ->
          same_ray ~tol:1e-15
            (rotate_lonlat (F Frame.galactic) (F Frame.supergalactic) origin)
            (vector 1. 0. 0.));
    ]

(* Names *)

let visit_text visits =
  String.concat "\n" (List.map (Format.asprintf "%a" Nx.Ptree.pp_visit) visits)

let walk_visits (type f) (frame : f Frame.t) =
  let s =
    Nx.Ptree.instantiate
      (module struct
        type _ t = f Frame.t

        let walk = Frame.walk
      end)
  in
  Nx.Ptree.visits s frame

let names =
  group "A frame's name is its identity"
    [
      test "name and pp give the canonical names" (fun () ->
          List.iter
            (fun (F f) -> Format.printf "%s %a@." (Frame.name f) Frame.pp f)
            fixed;
          expect (output ())
          @@ __POS_OF__
               {|
            icrs icrs
            fk5_j2000 fk5_j2000
            galactic galactic
            ecliptic_j2000 ecliptic_j2000
            supergalactic supergalactic
          |});
      test "walk reports the name as one case" (fun () ->
          print_string (visit_text (walk_visits Frame.galactic));
          expect (output ()) @@ __POS_OF__ {| the root: case "galactic" |});
    ]

let () =
  exit
    (run "Frame"
       [ orientations; conversions; erfa; astropy; supergalactic; names ])
