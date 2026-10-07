(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Transforms: FITS celestial maps agree with WCSLIB, inverses invert, [about]
   agrees with the frames' measures, domains raise or mask, and gradients are
   finite and match differences. *)

open Windtrap
open Ymir
open Ymir_test.Frames
module Reference = Ymir_test.Grids_reference

let f64 = Nx.float64
let tensor shape xs = Nx.create f64 shape xs
let deg x = Quantity.v Unit.degree x
let one x = Quantity.v Unit.one x
let pixels xs = one (tensor [| Array.length xs / 2; 2 |] xs)
let values q u = Nx.to_array (Quantity.value u q)

let max_abs a b =
  Array.fold_left max 0. (Array.map2 (fun x y -> Float.abs (x -. y)) a b)

let no_pv = Nx.zeros f64 [| 0 |]

let code = function
  | "TAN" -> Transform.Tan
  | "ARC" -> Transform.Arc
  | c -> invalid_arg c

(* [fits c] is the transform a FITS header with [c]'s keywords reads as. *)
let fits (c : Reference.wcs) =
  Transform.(
    axes [| 1; 0 |] ~origin:1
    >> shift (one (tensor [| 2 |] c.crpix))
    >> linear (deg (tensor [| 2; 2 |] c.cd))
    >> celestial (code c.code) Frame.icrs ~pv:no_pv
         ~native:(deg (tensor [| 2 |] [| 0.; 90. |]))
         ~crval:(deg (tensor [| 2 |] c.crval))
         ~lonpole:(deg (scalar 180.))
         ~latpole:(deg (scalar 90.)))

let lonlat_deg xs =
  let n = Array.length xs / 2 in
  let t = tensor [| n; 2 |] xs in
  Direction.lonlat Frame.icrs
    ~lon:(deg (Nx.slice [ Nx.A; Nx.I 0 ] t))
    ~lat:(deg (Nx.slice [ Nx.A; Nx.I 1 ] t))

(* WCSLIB *)

let wcslib =
  cases
    ~name:(fun (c : Reference.wcs) -> c.code)
    "WCSLIB" Reference.wcs
    (fun c ->
      let t = fits c in
      let world = Transform.apply t (pixels c.pixels) in
      let off =
        Array.fold_left max 0. (separation world (lonlat_deg c.world))
      in
      less float_exact ~than:2e-13 off;
      let back = Transform.apply (Transform.inverse t) (lonlat_deg c.sky) in
      less float_exact ~than:1e-8 (max_abs (values back Unit.one) c.sky_pixels))

(* Inverses *)

let random_points =
  Gen.(
    map
      (fun (x, y) -> [| x; y |])
      (pair (float_range (-3000.) 3000.) (float_range (-3000.) 3000.)))

let inverses =
  group "Inverses"
    [
      test "inverse (inverse t) has t's stages" (fun () ->
          let t = fits (List.hd Reference.wcs) in
          let s = Nx.Ptree.visits (Transform.ptree ()) in
          let pp = Format.asprintf "%a" in
          equal string (pp Transform.pp t)
            (pp Transform.pp Transform.(inverse (inverse t)));
          equal int
            (List.length (s t))
            (List.length (s Transform.(inverse (inverse t)))));
      prop "a TAN chain round-trips its pixels" random_points (fun p ->
          let t = fits (List.hd Reference.wcs) in
          let back = Transform.(apply (inverse t) (apply t (pixels p))) in
          less float_exact ~than:1e-7 (max_abs (values back Unit.one) p));
      prop "an ARC chain round-trips its pixels" random_points (fun p ->
          let c = List.nth Reference.wcs 1 in
          let p = Array.map (fun x -> x /. 20.) p in
          let t = fits c in
          let back = Transform.(apply (inverse t) (apply t (pixels p))) in
          less float_exact ~than:1e-9 (max_abs (values back Unit.one) p));
      prop "a planar chain round-trips"
        Gen.(pair random_points (float_range 0.1 10.))
        (fun (p, s) ->
          let t =
            Transform.(
              axes [| 1; 0 |] ~origin:1
              >> shift (one (tensor [| 2 |] [| 3.5; -2. |]))
              >> linear (one (tensor [| 2; 2 |] [| 0.8; 0.1; -0.3; 1.2 |]))
              >> scale (deg (tensor [| 2 |] [| s; -1. /. s |])))
          in
          let back = Transform.(apply (inverse t) (apply t (pixels p))) in
          let scale =
            Array.fold_left (fun m x -> Float.max m (Float.abs x)) 1. p
          in
          less float_exact ~than:(1e-12 *. scale)
            (max_abs (values back Unit.one) p));
    ]

(* About *)

let directions =
  Gen.(
    map
      (fun (l, b) -> (l, b))
      (pair (float_range 0. 6.28) (float_range (-1.5707963) 1.5707963)))

let at (l, b) = lonlat_deg [| l *. 180. /. Float.pi; b *. 180. /. Float.pi |]

let about_law =
  group "about"
    [
      (* The bounds, from the rounding of each side, u = 2⁻⁵³ and r the offset's
         length. [about a] rotates [b'] into [a]'s native frame by a matrix
         whose entries are products of faithfully rounded sines and cosines of
         a's longitude and latitude, each read from [a]'s vector by [atan2]:
         the reference point is off by at most 7u, and each entry by at most
         9u, so each native component carries at most 7u + 9√3 u + 3u ≤ 26u
         of absolute error. The offset is that native vector's (u₁, u₂), of
         length r: its norm moves by at most 26√2 u ≤ 37u and its bearing by
         at most 37u / r. [separation] and [position_angle] work on [b' − a],
         exact where it cancels, and are within 4u·r and 4u; the final
         [hypot] and [atan2] add u each. *)
      prop "the norm is the separation and the bearing the position angle"
        Gen.(
          pair directions (pair (float_range (-1.) 1.) (float_range 1e-9 2.)))
        (fun ((l, b), (pa, r)) ->
          let u = Float.epsilon /. 2. in
          let a = at (l, b) in
          let b' =
            (* [b] at separation [r] and bearing [pa] from [a]. *)
            Transform.(apply (inverse (about a)))
              (Quantity.v Unit.radian
                 (tensor [| 1; 2 |] [| r *. Float.sin pa; r *. Float.cos pa |]))
          in
          cover "close" (r < 1e-6);
          cover "far" (r > 1.);
          let x = values (Transform.apply (Transform.about a) b') Unit.radian in
          let sep = (separation a b').(0)
          and bearing = (position_angle a b').(0) in
          less ~msg:"norm" float_exact
            ~than:((37. +. (4. *. r) +. 1.) *. u)
            (Float.abs (Float.hypot x.(0) x.(1) -. sep));
          let d = Float.atan2 x.(0) x.(1) -. bearing in
          let d =
            if d > Float.pi then d -. (2. *. Float.pi)
            else if d < -.Float.pi then d +. (2. *. Float.pi)
            else d
          in
          less ~msg:"bearing" float_exact
            ~than:(((37. /. r) +. 4. +. 1.) *. u)
            (Float.abs d));
      test "at the poles" (fun () ->
          List.iter
            (fun b ->
              let a = at (0.3, b) and c = at (1.1, b *. 0.999) in
              let x =
                values (Transform.apply (Transform.about a) c) Unit.radian
              in
              less float_exact ~than:1e-14
                (Float.abs (Float.hypot x.(0) x.(1) -. (separation a c).(0))))
            [ Float.pi /. 2.; -.Float.pi /. 2. ]);
      test "x is east and y north" (fun () ->
          let a = at (1., 0.2) in
          let east = at (1.001, 0.2) and north = at (1., 0.201) in
          let x =
            values (Transform.apply (Transform.about a) east) Unit.radian
          in
          let y =
            values (Transform.apply (Transform.about a) north) Unit.radian
          in
          greater float_exact ~than:0. x.(0);
          less float_exact ~than:1e-6 (Float.abs x.(1));
          greater float_exact ~than:0. y.(1);
          less float_exact ~than:1e-12 (Float.abs y.(0)));
      test "the centre maps to the origin" (fun () ->
          let a = at (2., -0.4) in
          equal
            (array (float 1e-16))
            [| 0.; 0. |]
            (values (Transform.apply (Transform.about a) a) Unit.radian));
    ]

(* Domains *)

let tan_about c = Transform.inverse (Transform.gnomonic c)

let domains =
  group "Domains"
    [
      test "TAN's projection raises beyond 90 degrees, naming the point"
        (fun () ->
          let a = at (1.9344, -1.282) in
          let pts = lonlat_deg [| 110.; -73.; 10.; 50. |] in
          raises_match
            (Exn.invalid_arg ~substring:"Transform.apply: point 1 is")
            (fun () -> Transform.apply (Transform.gnomonic a) pts));
      test "covers masks TAN's projection" (fun () ->
          let a = at (0., 0.) in
          let pts = lonlat_deg [| 10.; 0.; 95.; 0.; 180.; 0.; -89.; 0. |] in
          equal (array bool)
            [| true; false; false; true |]
            (Nx.to_array (Transform.covers (Transform.gnomonic a) pts)));
      test "covers is false at NaN, and apply maps NaN to NaN" (fun () ->
          let p =
            Quantity.v Unit.radian
              (tensor [| 2; 2 |] [| 0.1; Float.nan; 0.; 0. |])
          in
          let t = tan_about (at (0.5, 0.5)) in
          equal (array bool) [| false; true |]
            (Nx.to_array (Transform.covers t p));
          let d = Direction.xyz (Transform.apply t p) in
          equal float_exact Float.nan (Nx.item [ 0; 0 ] d));
      test "ARC's deprojection raises beyond 180 degrees" (fun () ->
          let t = Transform.inverse (Transform.about (at (0., 0.))) in
          let p =
            Quantity.v Unit.degree (tensor [| 2; 2 |] [| 10.; 0.; 150.; 150. |])
          in
          equal (array bool) [| true; false |]
            (Nx.to_array (Transform.covers t p));
          raises_match
            (Exn.invalid_arg
               ~substring:"point 1 is 212.132 deg from the ARC reference")
            (fun () -> Transform.apply t p));
      test "a transform with no stage covers everything" (fun () ->
          equal (array bool) [| true |]
            (Nx.to_array (Transform.covers Transform.id (pixels [| 1.; 2. |]))));
    ]

(* Constructors *)

let constructors =
  group "Constructors"
    [
      test "axes raises on a non-permutation" (fun () ->
          raises
            (Invalid_argument "Transform.axes: [0; 0] is not a permutation")
            (fun () -> Transform.axes [| 0; 0 |] ~origin:0));
      test "axes permutes and adds the origin" (fun () ->
          equal (array float_exact) [| 3.; 2. |]
            (values
               Transform.(
                 apply (axes [| 1; 0 |] ~origin:1) (pixels [| 1.; 2. |]))
               Unit.one));
      test "a misordered list raises where a stage reads the wrong unit"
        (fun () ->
          let cd = deg (tensor [| 2; 2 |] [| 1.; 0.; 0.; 1. |]) in
          raises_match (Exn.invalid_arg ~substring:"Quantity.value") (fun () ->
              Transform.(apply (linear cd >> linear cd) (pixels [| 1.; 2. |]))));
      test "celestial refuses pv of the wrong size" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"TAN takes 0 parameters")
            (fun () ->
              Transform.celestial Transform.Tan Frame.icrs
                ~pv:(Nx.zeros f64 [| 3 |])
                ~native:(deg (tensor [| 2 |] [| 0.; 90. |]))
                ~crval:(deg (tensor [| 2 |] [| 0.; 0. |]))
                ~lonpole:(deg (scalar 180.))
                ~latpole:(deg (scalar 90.))));
      test "a native latitude other than 90 degrees raises when applied"
        (fun () ->
          let t =
            Transform.celestial Transform.Tan Frame.icrs ~pv:no_pv
              ~native:(deg (tensor [| 2 |] [| 0.; 45. |]))
              ~crval:(deg (tensor [| 2 |] [| 0.; 0. |]))
              ~lonpole:(deg (scalar 180.))
              ~latpole:(deg (scalar 90.))
          in
          raises_match (Exn.invalid_arg ~substring:"native latitude") (fun () ->
              Transform.apply t
                (Quantity.v Unit.degree (tensor [| 2 |] [| 0.; 0. |]))));
    ]

(* Batches *)

let batches =
  group "Batches"
    [
      test "parameters and points broadcast" (fun () ->
          let centres = lonlat_deg [| 10.; 20.; 30.; -40.; 200.; 85. |] in
          let pts =
            Transform.apply
              (Transform.inverse (Transform.about centres))
              (Quantity.v Unit.arcsecond
                 (tensor [| 3; 2 |] [| 1.; 0.; 0.; 2.; -3.; 0. |]))
          in
          equal
            (array (float 1e-9))
            [| 1.; 2.; 3. |]
            (Array.map
               (fun r -> r *. 180. /. Float.pi *. 3600.)
               (separation centres pts)));
    ]

(* Gradients and compilation *)

let central f x h = (f (x +. h) -. f (x -. h)) /. (2. *. h)

let gradients =
  group "Gradients"
    [
      test "about's offsets differentiate in the centre" (fun () ->
          let b = at (1.2, 0.3) in
          let f lat =
            let a =
              Direction.lonlat Frame.icrs
                ~lon:(Quantity.v Unit.radian (scalar 1.2001))
                ~lat:(Quantity.v Unit.radian lat)
            in
            Nx.sum
              (Quantity.value Unit.radian
                 (Transform.apply (Transform.about a) b))
          in
          let g = Nx.item [] (Rune.grad' f (scalar 0.3002)) in
          let fd = central (fun x -> Nx.item [] (f (scalar x))) 0.3002 1e-6 in
          equal (float 1e-7) fd g);
      test "the derivative at the centre is finite" (fun () ->
          let f lat =
            let a =
              Direction.lonlat Frame.icrs
                ~lon:(Quantity.v Unit.radian (scalar 1.))
                ~lat:(Quantity.v Unit.radian (scalar 0.3))
            in
            let b =
              Direction.lonlat Frame.icrs
                ~lon:(Quantity.v Unit.radian (scalar 1.))
                ~lat:(Quantity.v Unit.radian lat)
            in
            Nx.sum
              (Quantity.value Unit.radian
                 (Transform.apply (Transform.about a) b))
          in
          equal (float 1e-12) 1. (Nx.item [] (Rune.grad' f (scalar 0.3))));
      test "compiled equals eager" (fun () ->
          let t = fits (List.hd Reference.wcs) in
          let f p = Direction.xyz (Transform.apply t (one p)) in
          let p = tensor [| 3; 2 |] [| 0.; 0.; 100.5; 7.25; 1000.; -30. |] in
          equal (array float_exact)
            (Nx.to_array (f p))
            (Nx.to_array (Rune.jit' f p)));
    ]

(* Structure *)

let structure =
  group "Structure"
    [
      test "a FITS chain's visits" (fun () ->
          let t = fits (List.hd Reference.wcs) in
          let visits = Nx.Ptree.visits (Transform.ptree ()) t in
          expect
            (String.concat "\n"
               (List.map (Format.asprintf "%a" Nx.Ptree.pp_visit) visits))
          @@ __POS_OF__
               {|
            the root: int 4
            0: case "axes"
            0: case "forward"
            0.perm: int 2
            0.perm: int 1
            0.perm: int 0
            0.origin: int 1
            1: case "shift"
            1: case "forward"
            1.offset: case "1"
            1.offset: a leaf
            2: case "linear"
            2: case "forward"
            2.matrix: case "1/180 pi rad"
            2.matrix: a leaf
            3: case "celestial"
            3: case "forward"
            3: case "TAN"
            3.frame: case "icrs"
            3.stated: int 0
            3.pv: a leaf
            3.native: case "1/180 pi rad"
            3.native: a leaf
            3.crval: case "1/180 pi rad"
            3.crval: a leaf
            3.lonpole: case "1/180 pi rad"
            3.lonpole: a leaf
            3.latpole: case "1/180 pi rad"
            3.latpole: a leaf
            |});
      test "pp" (fun () ->
          expect
            (Format.asprintf "%a" Transform.pp (Transform.about (at (0., 0.))))
          @@ __POS_OF__
               {|
            inverse (celestial ARC icrs ~crval:float64 [1,2] [[0, 0]] rad ~lonpole:
              180 1/180 pi rad)
            |});
    ]

let () =
  exit
    (run "Transform"
       [
         wcslib;
         inverses;
         about_law;
         domains;
         constructors;
         batches;
         gradients;
         structure;
       ])
