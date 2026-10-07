(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Observations: sky apertures agree with photutils, windows keep the sums, the
   unit decides what a cell counts as, invalid samples reach no result, a
   clipped region raises, and gradients match differences. *)

open Windtrap
open Ymir
open Ymir_test.Frames
module Reference = Ymir_test.Grids_reference

let f64 = Nx.float64
let deg x = Quantity.v Unit.degree x
let arcsec x = Quantity.v Unit.arcsecond x
let one x = Quantity.v Unit.one x
let jy = Unit.symbol "Jy"
let brightness = Unit.(jy / steradian)
let counts = Unit.(symbol "electron" / Grid.cell)
let pixel = 0.031 /. 3600. *. Float.pi /. 180.
let pixar = pixel *. pixel

let image dtype shape seed =
  Nx.init dtype shape (fun i ->
      let i, j = (i.(0), i.(1)) in
      float_of_int (((7919 * i) + (104729 * j) + (13 * i * j) + seed) mod 1009)
      /. 64.)

(* The fixture's TAN image. *)
let wcs =
  Transform.(
    axes [| 1; 0 |] ~origin:1
    >> shift (one (Nx.create f64 [| 2 |] Reference.sky_crpix))
    >> linear (deg (Nx.create f64 [| 2; 2 |] Reference.sky_cd))
    >> celestial Tan Frame.icrs ~pv:(Nx.zeros f64 [| 0 |])
         ~native:(deg (Nx.create f64 [| 2 |] [| 0.; 90. |]))
         ~crval:(deg (Nx.create f64 [| 2 |] Reference.sky_crval))
         ~lonpole:(deg (scalar 180.))
         ~latpole:(deg (scalar 90.)))

let crval =
  Direction.lonlat Frame.icrs
    ~lon:(deg (scalar Reference.sky_crval.(0)))
    ~lat:(deg (scalar Reference.sky_crval.(1)))

let sky ?area ?(unit = brightness) dtype =
  let g = Grid.pixels ~shape:Reference.sky_shape dtype wcs in
  Observation.v ?area g (Quantity.v unit (image dtype Reference.sky_shape 73))

let with_pixar dtype =
  sky ~area:(Quantity.v Unit.steradian (Nx.scalar dtype pixar)) dtype

(* The direction at [(east, north)] arcseconds from CRVAL. *)
let target east north =
  Transform.(apply (inverse (about crval)))
    (arcsec (Nx.create f64 [| 2 |] [| east; north |]))

let jansky (i : _ Observation.integral) = Nx.item [] (Quantity.value jy i.value)

(* photutils *)

(* The images jump from cell to cell, so a sum moves by about 0.1 of itself per
   pixel of the centre: photutils and ymir place the centre through different
   trigonometry, about 1e-15 rad or 1e-8 pixel apart, and agree to about
   1e-9. *)
let photutils =
  let obs = with_pixar f64 in
  group "photutils"
    [
      cases
        ~name:(fun (e, n, r, _) -> Printf.sprintf "circle (%g, %g) r=%g" e n r)
        "sky circles" Reference.sky_circles
        (fun (e, n, r, s) ->
          let c =
            Region.circle
              (Transform.about (target e n))
              ~radius:(arcsec (scalar r))
          in
          equal
            (float_rel ~rel:1e-8 ~abs:0.)
            (s *. pixar)
            (jansky (Observation.integrate c obs)));
      cases
        ~name:(fun (e, n, i, o, _) ->
          Printf.sprintf "annulus (%g, %g) %g-%g" e n i o)
        "sky annuli" Reference.sky_annuli
        (fun (e, n, i, o, s) ->
          let a =
            Region.annulus
              (Transform.about (target e n))
              ~inner:(arcsec (scalar i))
              ~outer:(arcsec (scalar o))
          in
          equal
            (float_rel ~rel:1e-8 ~abs:0.)
            (s *. pixar)
            (jansky (Observation.integrate a obs)));
      test "float32 data meet the goal post" (fun () ->
          let obs32 = with_pixar Nx.float32 in
          List.iter
            (fun (e, n, r, s) ->
              let c =
                Region.circle
                  (Transform.about (target e n))
                  ~radius:(arcsec (Nx.scalar Nx.float32 r))
              in
              let v =
                Nx.item []
                  (Quantity.value jy (Observation.integrate c obs32).value)
              in
              equal (float_rel ~rel:1e-6 ~abs:0.) (s *. pixar) v)
            Reference.sky_circles);
    ]

(* Windows *)

let windows =
  let obs = with_pixar f64 in
  let r = 0.5 in
  let circle t =
    Region.circle (Transform.about t) ~radius:(arcsec (scalar r))
  in
  group "Windows"
    [
      test "a stamp around the target sums as the whole image" (fun () ->
          let t = target 0.21 (-0.13) in
          let whole = jansky (Observation.integrate (circle t) obs) in
          let stamp = Observation.around t ~shape:[| 40; 40 |] obs in
          let i = Observation.integrate (circle t) stamp in
          equal (float_rel ~rel:1e-14 ~abs:0.) whole (jansky i);
          equal (float 1e-12) 1. (Nx.item [] i.coverage));
      test "a batch of stamps is a batch of integrals" (fun () ->
          let es = [| 0.; 0.21; -0.3 |] and ns = [| 0.; -0.13; 0.27 |] in
          let one_by_one =
            Array.map2
              (fun e n ->
                jansky (Observation.integrate (circle (target e n)) obs))
              es ns
          in
          let ts =
            Transform.(apply (inverse (about crval)))
              (arcsec
                 (Nx.create f64 [| 3; 2 |]
                    [| 0.; 0.; 0.21; -0.13; -0.3; 0.27 |]))
          in
          let stamps = Observation.around ts ~shape:[| 40; 40 |] obs in
          let i = Observation.integrate (circle ts) stamps in
          equal
            (array (float_rel ~rel:1e-14 ~abs:0.))
            one_by_one
            (Nx.to_array (Quantity.value jy i.value)));
      test "a stamp too small for the region raises" (fun () ->
          let t = target 0. 0. in
          let stamp = Observation.around t ~shape:[| 20; 20 |] obs in
          raises_match
            (Exn.invalid_arg
               ~substring:
                 "Observation.integrate: the region reaches the border of its \
                  20x20 window at sample (0, ") (fun () ->
              Observation.integrate (circle t) stamp));
      test "a region across the image's edge has its share as coverage"
        (fun () ->
          (* The image's corner cell (0, 0) as the centre: a quarter of the
             disc, to second order in the pixel, lies on the image. *)
          let g = Observation.grid obs in
          let corner =
            Grid.centres
              (Grid.window
                 ~start:(Nx.zeros Nx.int64 [| 2 |])
                 ~shape:[| 1; 1 |] g)
          in
          let corner =
            Nx.Ptree.map (Direction.ptree ())
              (fun _ v -> Nx.reshape [| 3 |] v)
              corner
          in
          let corner =
            Transform.(apply (inverse (about corner)))
              (Quantity.v Unit.radian (Nx.create f64 [| 2 |] [| 0.; 0. |]))
          in
          let stamp = Observation.around corner ~shape:[| 40; 40 |] obs in
          let i = Observation.integrate (circle corner) stamp in
          (* Cells (0, 0) and beyond: a quarter of the disc plus half of the two
             half-cell strips and a quarter cell. *)
          let rho = r /. 0.031 in
          let inside = (Float.pi *. rho *. rho /. 4.) +. rho +. 0.25 in
          equal (float 1e-3)
            (inside /. (Float.pi *. rho *. rho))
            (Nx.item [] i.coverage);
          (* The whole image reports the same edge. *)
          let whole = Observation.integrate (circle corner) obs in
          equal (float 1e-14) (Nx.item [] i.coverage)
            (Nx.item [] whole.coverage));
    ]

(* The unit rule *)

let unit_rule =
  let t = target 0.05 0.1 in
  let circle =
    Region.circle (Transform.about t) ~radius:(arcsec (scalar 0.4))
  in
  let weights obs = Region.weights circle (Observation.grid obs) in
  group "The unit rule"
    [
      test "values per cell count each cell once" (fun () ->
          let obs = sky ~unit:counts f64 in
          let i = Observation.integrate circle obs in
          let w = weights obs in
          let expected =
            Nx.item []
              (Nx.sum (Nx.mul w (Quantity.value counts (Observation.data obs))))
          in
          equal
            (float_rel ~rel:1e-13 ~abs:0.)
            expected
            (Nx.item [] (Quantity.value Unit.(symbol "electron") i.value));
          equal
            (float_rel ~rel:1e-13 ~abs:0.)
            (Nx.item [] (Nx.sum w))
            (Nx.item [] (Quantity.value Grid.cell i.area)));
      test "a field counts each cell by its measure" (fun () ->
          let obs = sky f64 in
          let i = Observation.integrate circle obs in
          let g = Observation.grid obs in
          let w = weights obs in
          let m = Quantity.value Unit.steradian (Grid.measure g) in
          let expected =
            Nx.item []
              (Nx.sum
                 (Nx.mul (Nx.mul w m)
                    (Quantity.value brightness (Observation.data obs))))
          in
          equal (float_rel ~rel:1e-13 ~abs:0.) expected (jansky i);
          (* The cap's solid angle, 2π(1 - cos ρ). *)
          let rho = 0.4 /. 3600. *. Float.pi /. 180. in
          let cap =
            2. *. Float.pi
            *. (2. *. Float.sin (rho /. 2.) *. Float.sin (rho /. 2.))
          in
          equal
            (float_rel ~rel:1e-8 ~abs:0.)
            cap
            (Nx.item [] (Quantity.value Unit.steradian i.area)));
      test "a stated area replaces the measure" (fun () ->
          let i = Observation.integrate circle (with_pixar f64) in
          let w = Nx.item [] (Nx.sum (weights (with_pixar f64))) in
          equal
            (float_rel ~rel:1e-13 ~abs:0.)
            (w *. pixar)
            (Nx.item [] (Quantity.value Unit.steradian i.area)));
      test "a per-cell area map windows with the data" (fun () ->
          let g = Grid.pixels ~shape:Reference.sky_shape f64 wcs in
          let obs =
            Observation.v ~area:(Grid.measure g) g
              (Quantity.v brightness (image f64 Reference.sky_shape 73))
          in
          let whole = jansky (Observation.integrate circle (sky f64)) in
          let stamp =
            Observation.window
              ~start:(Nx.create Nx.int64 [| 2 |] [| 20L; 25L |])
              ~shape:[| 60; 60 |] obs
          in
          equal
            (float_rel ~rel:1e-14 ~abs:0.)
            whole
            (jansky (Observation.integrate circle stamp)));
      test "an area of another dimension raises" (fun () ->
          raises_match
            (Exn.invalid_arg ~substring:"not a measure of the grid's cells")
            (fun () -> sky ~area:(Quantity.v Unit.metre (scalar 1.)) f64));
    ]

(* Validity and variance *)

let validity =
  let t = target 0. 0. in
  let circle r = Region.circle (Transform.about t) ~radius:(arcsec r) in
  let mask =
    Nx.init Nx.bool Reference.sky_shape (fun i -> (i.(0) + i.(1)) mod 7 <> 0)
  in
  group "Validity and variance"
    [
      test
        "invalid samples holding NaN give what zeros give, value and gradient"
        (fun () ->
          let data = image f64 Reference.sky_shape 73 in
          let g = Grid.pixels ~shape:Reference.sky_shape f64 wcs in
          let with_under under =
            let data = Nx.where mask data (Nx.full_like data under) in
            Observation.v ~valid:(Nx.cast Nx.bit mask) g
              (Quantity.v brightness data)
          in
          let f obs r =
            Quantity.value jy (Observation.integrate (circle r) obs).value
          in
          let nan = with_under Float.nan and zero = with_under 0. in
          equal float_exact
            (Nx.item [] (f zero (scalar 0.5)))
            (Nx.item [] (f nan (scalar 0.5)));
          equal float_exact
            (Nx.item [] (Rune.grad' (f zero) (scalar 0.5)))
            (Nx.item [] (Rune.grad' (f nan) (scalar 0.5))));
      test "coverage is the valid cells' share" (fun () ->
          let obs = Observation.restrict (Nx.cast Nx.bit mask) (sky f64) in
          let i = Observation.integrate (circle (scalar 0.5)) obs in
          equal (float 0.02) (6. /. 7.) (Nx.item [] i.coverage));
      test "restrict narrows" (fun () ->
          let half =
            Nx.init Nx.bool Reference.sky_shape (fun i -> i.(0) < 48)
          in
          let obs =
            Observation.restrict (Nx.cast Nx.bit half)
              (Observation.restrict (Nx.cast Nx.bit mask) (sky f64))
          in
          let valid = Nx.cast Nx.bool (Option.get (Observation.valid obs)) in
          equal (array bool)
            (Nx.to_array (Nx.logical_and half mask))
            (Nx.to_array valid));
      test "the variance sums variance × w² × a²" (fun () ->
          let g = Grid.pixels ~shape:Reference.sky_shape f64 wcs in
          let var = Nx.full f64 Reference.sky_shape 4. in
          let obs =
            Observation.v
              ~variance:(Quantity.v Unit.(brightness ** 2) var)
              ~area:(Quantity.v Unit.steradian (scalar pixar))
              g
              (Quantity.v brightness (image f64 Reference.sky_shape 73))
          in
          let r = circle (scalar 0.5) in
          let i = Observation.integrate r obs in
          let w = Region.weights r g in
          let expected =
            4. *. pixar *. pixar *. Nx.item [] (Nx.sum (Nx.square w))
          in
          equal
            (float_rel ~rel:1e-13 ~abs:0.)
            expected
            (Nx.item [] (Quantity.value Unit.(jy ** 2) (Option.get i.variance))));
    ]

(* Gradients *)

(* The Guide's flux of an aperture with its sky annulus subtracted, in Jy, as a
   function of [east; north; radius] in arcseconds. *)
let flux (type e) (obs : (Frame.icrs Direction.t, e) Observation.t)
    (p : (float, e) Nx.t) =
  let t = target 0.1 (-0.05) in
  let centre =
    Quantity.v Unit.arcsecond (Nx.cast f64 (Nx.slice [ Nx.R (0, 2) ] p))
  in
  let at = Transform.(about t >> shift centre) in
  let radius = arcsec (Nx.slice [ Nx.I 2 ] p) in
  let dtype = Nx.dtype p in
  let sum = Observation.integrate (Region.circle at ~radius) obs in
  let sky =
    Observation.integrate
      (Region.annulus at
         ~inner:(arcsec (Nx.scalar dtype 0.7))
         ~outer:(arcsec (Nx.scalar dtype 1.1)))
      obs
  in
  Quantity.(value jy (sub sum.value (mul (div sky.value sky.area) sum.area)))

let gradients =
  group "Gradients"
    [
      test "float32 gradients match float64 central differences" (fun () ->
          let p = [| 0.03; -0.02; 0.45 |] in
          let obs = with_pixar f64 and obs32 = with_pixar Nx.float32 in
          let at q = Nx.item [] (flux obs (Nx.create f64 [| 3 |] q)) in
          let h = 1e-5 in
          let fd =
            Array.init 3 (fun k ->
                let up = Array.copy p and down = Array.copy p in
                up.(k) <- p.(k) +. h;
                down.(k) <- p.(k) -. h;
                (at up -. at down) /. (2. *. h))
          in
          let g = Rune.grad' (flux obs32) (Nx.create Nx.float32 [| 3 |] p) in
          let scale =
            Array.fold_left (fun m x -> Float.max m (Float.abs x)) 0. fd
          in
          equal (array (float (1e-4 *. scale))) fd (Nx.to_array (Nx.cast f64 g)));
      test "compiled equals eager within rounding" (fun () ->
          let t = target 0.1 (-0.05) in
          let obs = Observation.around t ~shape:[| 40; 40 |] (with_pixar f64) in
          let f obs p =
            let centre =
              Quantity.v Unit.arcsecond (Nx.slice [ Nx.R (0, 2) ] p)
            in
            let r =
              Region.circle
                Transform.(about t >> shift centre)
                ~radius:(arcsec (Nx.slice [ Nx.I 2 ] p))
            in
            Quantity.value jy (Observation.integrate r obs).value
          in
          let p = Nx.create f64 [| 3 |] [| 0.03; -0.02; 0.45 |] in
          let c =
            Rune.jit
              Nx.Ptree.(Observation.ptree () @-> tensor @-> returns tensor)
              f
          in
          (* Compiled trigonometry rounds differently; on these images an ulp of
             a direction moves the sum by about 1e-11 of itself. *)
          equal
            (float_rel ~rel:1e-10 ~abs:0.)
            (Nx.item [] (f obs p))
            (Nx.item [] (c obs p)));
    ]

(* Arithmetic and worlds *)

let arithmetic =
  let obs ?variance ?valid ?area seed =
    let g = Grid.pixels ~shape:[| 4; 3 |] f64 Transform.id in
    Observation.v ?valid ?area
      ?variance:(Option.map (fun v -> Quantity.v Unit.(counts ** 2) v) variance)
      g
      (Quantity.v counts (image f64 [| 4; 3 |] seed))
  in
  let data o = Nx.to_array (Quantity.value counts (Observation.data o)) in
  group "Arithmetic"
    [
      test "add and sub combine data and add variances" (fun () ->
          let va = Nx.full f64 [| 4; 3 |] 2. and vb = Nx.full f64 [| 4; 3 |] 3. in
          let a = obs ~variance:va 1 and b = obs ~variance:vb 2 in
          let s = Observation.add a b and d = Observation.sub a b in
          equal (array float_exact)
            (Array.map2 ( +. ) (data a) (data b)) (data s);
          equal (array float_exact)
            (Array.map2 ( -. ) (data a) (data b)) (data d);
          equal (array float_exact) (Array.make 12 5.)
            (Nx.to_array
               (Quantity.value Unit.(counts ** 2)
                  (Option.get (Observation.variance d)))));
      test "a variance survives only where both have one" (fun () ->
          let a = obs ~variance:(Nx.ones f64 [| 4; 3 |]) 1 and b = obs 2 in
          equal bool true (Option.is_none (Observation.variance (Observation.add a b))));
      test "validity is the conjunction" (fun () ->
          let m k = Nx.cast Nx.bit (Nx.init Nx.bool [| 4; 3 |] (fun i -> (i.(0) + i.(1)) mod k <> 0)) in
          let a = obs ~valid:(m 2) 1 and b = obs ~valid:(m 3) 2 in
          let s = Observation.add a b in
          let v = Nx.to_array (Nx.cast Nx.bool (Option.get (Observation.valid s))) in
          let expected = Array.init 12 (fun k -> let i = k / 3 and j = k mod 3 in (i + j) mod 2 <> 0 && (i + j) mod 3 <> 0) in
          equal (array bool) expected v;
          Array.iteri (fun k ok -> if not ok then equal float_exact 0. (data s).(k)) expected);
      test "scale multiplies data and variance by k and k²" (fun () ->
          let a = obs ~variance:(Nx.full f64 [| 4; 3 |] 2.) 1 in
          let k = Quantity.v Unit.one (Nx.scalar f64 3.) in
          let s = Observation.scale k a in
          equal (array float_exact) (Array.map (fun x -> 3. *. x) (data a)) (data s);
          equal (array float_exact) (Array.make 12 18.)
            (Nx.to_array (Quantity.value Unit.(counts ** 2) (Option.get (Observation.variance s)))));
      test "grids that do not agree raise, naming the leaf" (fun () ->
          let shifted =
            Observation.v
              (Grid.pixels ~shape:[| 4; 3 |] f64
                 (Transform.shift (one (Nx.create f64 [| 2 |] [| 0.; 0.5 |]))))
              (Quantity.v counts (image f64 [| 4; 3 |] 0))
          in
          let other =
            Observation.v
              (Grid.pixels ~shape:[| 4; 3 |] f64
                 (Transform.shift (one (Nx.create f64 [| 2 |] [| 0.; 0.25 |]))))
              (Quantity.v counts (image f64 [| 4; 3 |] 0))
          in
          raises
            (Invalid_argument
               "Observation.sub: the grids do not agree at transform.0.offset: \
                element (1) is 0.5 in the first and 0.25 in the second")
            (fun () -> Observation.sub shifted other);
          raises_match (Exn.invalid_arg ~substring:"Observation.add: the grids do not agree at base: int 3 in the first, int 2 in the second")
            (fun () ->
              Observation.add (obs 1)
                (Observation.v (Grid.pixels ~shape:[| 4; 2 |] f64 Transform.id)
                   (Quantity.v counts (Nx.zeros f64 [| 4; 2 |])))));
      test "areas must be equal" (fun () ->
          let area x = Quantity.v Unit.(one ** 2) (Nx.scalar f64 x) in
          raises
            (Invalid_argument
               "Observation.add: one observation states an area and the other \
                does not")
            (fun () -> Observation.add (obs ~area:(area 1.) 1) (obs 2));
          raises
            (Invalid_argument
               "Observation.add: the areas differ: 1 in the first and 2 in the \
                second")
            (fun () -> Observation.add (obs ~area:(area 1.) 1) (obs ~area:(area 2.) 2)));
      test "map_world keeps the data and moves the integral's world" (fun () ->
          let o = sky Nx.float64 in
          let g = Observation.map_world (Transform.rotation Frame.icrs Frame.galactic) o in
          let c = target 0.1 (-0.2) in
          let r = arcsec (scalar 0.4) in
          let i = Observation.integrate (Region.circle (Transform.about c) ~radius:r) o in
          let j =
            Observation.integrate
              (Region.circle (Transform.about (Direction.rotate Frame.galactic c)) ~radius:r)
              g
          in
          equal (float_rel ~rel:1e-9 ~abs:0.) (jansky i) (jansky j));
    ]

let () =
  exit
    (run "Observation"
       [ photutils; windows; unit_rule; validity; arithmetic; gradients ])
