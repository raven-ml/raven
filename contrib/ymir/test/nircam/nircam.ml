(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Aperture photometry on the NIRCam F200W i2d mosaic of SMACS 0723 against
   photutils, with gradients in aperture centre and radius against central
   differences. It prints its numbers and exits 1 when a goal post fails:

   - sums of float64 and float32 observations of the file's float32 data
     within 1e-6 of Σ |data · w · PIXAR_SR| of photutils' sums, and
     variances within 1e-6, for the whole apertures among 200 over the
     mosaic and the five targets.
     photutils' isotropic circle differs from the cap by up to θ²/2 = 4e-7 of
     the boundary's offset; an aperture mostly over invalid samples divides
     that by a small valid area, so partial apertures are reported apart;
   - float32 gradients of the Guide's flux within 1e-4 of the largest float64
     central difference, eager and compiled. *)

open Ymir
module Nircam = Ymir_test.Nircam_reference

let f64 = Nx.float64
let deg x = Quantity.v Unit.degree x
let arcsec x = Quantity.v Unit.arcsecond x
let ok = function Ok x -> x | Error e -> failwith e
let mjy = ok (Fits.Unit.parse "MJy")
let in_mjy q = Nx.item [] (Nx.cast f64 (Quantity.value mjy q))
let stamp = [| Nircam.stamp; Nircam.stamp |]
let failed = ref false

let time f =
  let t = Unix.gettimeofday () in
  let x = f () in
  (x, Unix.gettimeofday () -. t)

let target (a : Nircam.aperture) =
  Direction.lonlat Frame.icrs
    ~lon:(deg (Nx.scalar f64 a.ra))
    ~lat:(deg (Nx.scalar f64 a.dec))

let circle dtype a =
  Region.circle
    (Transform.about (target a))
    ~radius:(arcsec (Nx.scalar dtype Nircam.radius))

let annulus dtype a =
  Region.annulus
    (Transform.about (target a))
    ~inner:(arcsec (Nx.scalar dtype Nircam.inner))
    ~outer:(arcsec (Nx.scalar dtype Nircam.outer))

let check name ok_ =
  if not ok_ then failed := true;
  Printf.printf "  %s: %s\n%!" name (if ok_ then "holds" else "FAILS")

(* Sums *)

type worst = { mutable error : float; mutable at : string; mutable count : int }

let worst () = { error = 0.; at = ""; count = 0 }

let note w at x =
  w.count <- w.count + 1;
  if x > w.error || Float.is_nan x then (
    w.error <- x;
    w.at <- at)

(* The largest errors, apart for whole apertures (every sample under them
   valid) and partial ones. *)
type errors = { whole : worst; partial : worst; variance : worst }

(* [sums obs] is, over the circle and annulus of every aperture, the largest
   error over [Σ |data · w · PIXAR_SR|] of the sums and the largest relative
   difference of the variances. *)
let sums (type e) (obs : (Frame.icrs Direction.t, e) Observation.t) =
  let data = Observation.data obs in
  let dtype = Nx.dtype (Quantity.value (Quantity.unit data) data) in
  let e = { whole = worst (); partial = worst (); variance = worst () } in
  let measure name (a : Nircam.aperture) =
    let s = Observation.around (target a) ~shape:stamp obs in
    List.iter
      (fun (kind, region, (r : Nircam.sum)) ->
        let i = Observation.integrate region s in
        let at = Printf.sprintf "%s %s (%.7f, %.7f)" name kind a.ra a.dec in
        let v = in_mjy i.value in
        let whole = Nx.item [] (Nx.cast f64 i.coverage) >= 1. -. 1e-6 in
        note
          (if whole then e.whole else e.partial)
          at
          (if r.scale = 0. then Float.abs v
           else Float.abs (v -. r.sum) /. r.scale);
        match i.variance with
        | Some q when whole ->
            let v =
              Nx.item [] (Nx.cast f64 (Quantity.value (Unit.( ** ) mjy 2) q))
            in
            note e.variance at (Float.abs (v -. r.variance) /. r.variance)
        | _ -> ())
      [
        ("circle", circle dtype a, a.circle);
        ("annulus", annulus dtype a, a.annulus);
      ]
  in
  List.iteri
    (fun k a -> measure (Printf.sprintf "target %d" (k + 1)) a)
    Nircam.targets;
  List.iteri (fun k a -> measure (Printf.sprintf "grid %d" k) a) Nircam.grid;
  e

let report name e =
  Printf.printf
    "%s:\n\
    \  %d whole apertures: largest |ymir - photutils| / sum |data w PIXAR_SR| \
     = %.3g (%s)\n\
    \  %d partly over invalid samples: %.3g (%s)\n\
    \  variances of whole apertures: largest relative difference = %.3g (%s)\n\
     %!"
    name e.whole.count e.whole.error e.whole.at e.partial.count e.partial.error
    e.partial.at e.variance.error e.variance.at

(* The same observation with the cells' own measure for area. *)
let without_area obs =
  Observation.v ?variance:(Observation.variance obs)
    ?valid:(Observation.valid obs) (Observation.grid obs) (Observation.data obs)

let area_convention obs =
  let worst = ref 0. in
  List.iter
    (fun (a : Nircam.aperture) ->
      let s = Observation.around (target a) ~shape:stamp obs in
      let v = in_mjy (Observation.integrate (circle f64 a) s).value in
      if a.circle.sum <> 0. then
        worst :=
          Float.max !worst
            (Float.abs (v -. a.circle.sum) /. Float.abs a.circle.sum))
    (Nircam.targets @ Nircam.grid);
  !worst

(* Gradients *)

(* The Guide's flux about [a]: the circle less the annulus's mean times the
   circle's area, in MJy, as a function of [east; north; radius] in
   arcseconds. *)
let flux (type e) (a : Nircam.aperture)
    (stamp : (Frame.icrs Direction.t, e) Observation.t) (p : (float, e) Nx.t) =
  let dtype = Nx.dtype p in
  let centre = arcsec (Nx.cast f64 (Nx.slice [ Nx.R (0, 2) ] p)) in
  let at = Transform.(about (target a) >> shift centre) in
  let sum =
    Observation.integrate
      (Region.circle at ~radius:(arcsec (Nx.slice [ Nx.I 2 ] p)))
      stamp
  in
  let sky =
    Observation.integrate
      (Region.annulus at
         ~inner:(arcsec (Nx.scalar dtype Nircam.inner))
         ~outer:(arcsec (Nx.scalar dtype Nircam.outer)))
      stamp
  in
  Quantity.(value mjy (sub sum.value (mul (div sky.value sky.area) sum.area)))

let p0 = [| 0.02; -0.03; Nircam.radius |]

let differences a stamp64 =
  let at q = Nx.item [] (flux a stamp64 (Nx.create f64 [| 3 |] q)) in
  let h = 1e-4 in
  Array.init 3 (fun i ->
      let up = Array.copy p0 and down = Array.copy p0 in
      up.(i) <- p0.(i) +. h;
      down.(i) <- p0.(i) -. h;
      (at up -. at down) /. (2. *. h))

let relative fd g =
  let scale = Array.fold_left (fun m x -> Float.max m (Float.abs x)) 0. fd in
  Array.fold_left Float.max 0.
    (Array.map2 (fun x y -> Float.abs (x -. y) /. scale) fd g)

let () =
  let path =
    match Sys.argv with
    | [| _ |] ->
        Filename.concat (Sys.getenv "HOME")
          (Filename.concat ".cache/ymir" Nircam.mosaic)
    | [| _; p |] -> p
    | _ ->
        prerr_endline "usage: nircam.exe [MOSAIC]";
        exit 2
  in
  if not (Sys.file_exists path) then begin
    Printf.eprintf
      "%s: no such file. uv run contrib/ymir/test/gen/nircam.py downloads it \
       (SHA-256 %s).\n"
      path Nircam.sha256;
    exit 2
  end;
  let read dtype =
    let hdus = ok (Fits.read path) in
    ok (Fits.observation ~dtype ~frame:Frame.icrs ~data:"SCI" ~error:"ERR" hdus)
  in
  let obs, t64 = time (fun () -> read f64) in
  Printf.printf "read %s at float64: %.1f s\n%!" path t64;
  let e, t = time (fun () -> sums obs) in
  report (Printf.sprintf "float64 observation (%.1f s)" t) e;
  check "whole apertures' sums within 1e-6" (e.whole.error <= 1e-6);
  check "whole apertures' variances within 1e-6" (e.variance.error <= 1e-6);
  Printf.printf
    "without PIXAR_SR (each pixel's own solid angle): circles differ from \
     photutils by up to %.3g relative\n\
     %!"
    (area_convention (without_area obs));
  let obs32, t32 = time (fun () -> read Nx.float32) in
  Printf.printf "read at float32: %.1f s\n%!" t32;
  let e32, t = time (fun () -> sums obs32) in
  report (Printf.sprintf "float32 observation (%.1f s)" t) e32;
  check "float32 whole apertures' sums within 1e-6" (e32.whole.error <= 1e-6);
  check "float32 whole apertures' variances within 1e-6"
    (e32.variance.error <= 1e-6);
  print_endline "gradients of the Guide's flux at [0.02; -0.03; 0.5] arcsec:";
  List.iteri
    (fun k (a : Nircam.aperture) ->
      let s64 = Observation.around (target a) ~shape:stamp obs in
      let s32 = Observation.around (target a) ~shape:stamp obs32 in
      let fd = differences a s64 in
      let p = Nx.create Nx.float32 [| 3 |] p0 in
      let grad s p = Rune.grad' (flux a s) p in
      let g, eager = time (fun () -> Nx.to_array (Nx.cast f64 (grad s32 p))) in
      let compiled =
        Rune.jit
          Nx.Ptree.(Observation.ptree () @-> tensor @-> returns tensor)
          grad
      in
      let gc, first =
        time (fun () -> Nx.to_array (Nx.cast f64 (compiled s32 p)))
      in
      let _, warm =
        time (fun () -> Nx.to_array (Nx.cast f64 (compiled s32 p)))
      in
      let e = relative fd g and ec = relative fd gc in
      Printf.printf
        "  target %d: d/d(east, north, radius) = (%.6g, %.6g, %.6g) \
         MJy/arcsec; eager %.2g, compiled %.2g of the largest; eager %.3f s, \
         compiled first call %.2f s, then %.3f s\n\
         %!"
        (k + 1) fd.(0) fd.(1) fd.(2) e ec eager first warm;
      check
        (Printf.sprintf "target %d gradients within 1e-4" (k + 1))
        (e <= 1e-4 && ec <= 1e-4))
    Nircam.targets;
  exit (if !failed then 1 else 0)
