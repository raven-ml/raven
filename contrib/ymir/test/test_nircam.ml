(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* NIRCam cutouts read as observations, composing ymir.fits with ymir, agree
   with photutils on the whole mosaic at float64 and float32, a window reads the
   whole image's cells, and gradients match differences. *)

open Windtrap
open Ymir
open Ymir_fits
module Nircam = Ymir_test.Nircam_reference

let f64 = Nx.float64
let ok r = require_ok ~pp:Format.pp_print_string r
let deg x = Quantity.v Unit.degree x
let arcsec x = Quantity.v Unit.arcsecond x
let scalar x = Nx.scalar f64 x
let golden = lazy (ok (Fits.read "golden/nircam.fits"))

let observation ?window dtype k =
  ok (Ymir_test.I2d.observation ~ver:k ?window dtype (Lazy.force golden))

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

let windows =
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
      let values o =
        let d = Observation.data o in
        Nx.to_array (Quantity.value (Quantity.unit d) d)
      in
      equal float_exact (sum whole) (sum part);
      equal (array float_exact) (values whole) (values part))

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
      let at q = Nx.item [] (flux a obs (Nx.create f64 [| 3 |] q)) in
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

let () = exit (run "NIRCam" [ photutils; windows; gradients ])
