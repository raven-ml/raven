(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Regions: weights are exact fractions that agree with photutils, sum to the
   shape's area, lie in [0, 1], ignore windows, and differentiate in the centre
   and the size. *)

open Windtrap
open Ymir
module Reference = Ymir_test.Grids_reference

let f64 = Nx.float64
let one x = Quantity.v Unit.one x
let counts = Unit.(symbol "electron" / Grid.cell)
let pi = Float.pi

(* A grid whose world is its own pixel coordinates. *)
let plane_at dtype shape = Grid.pixels ~shape dtype Transform.id
let plane shape = plane_at f64 shape
let at (row, col) = Transform.shift (one (Nx.create f64 [| 2 |] [| row; col |]))

let circle_at dtype centre r =
  Region.circle (at centre) ~radius:(one (Nx.scalar dtype r))

let circle centre r = circle_at f64 centre r

let annulus centre inner outer =
  Region.annulus (at centre)
    ~inner:(one (Nx.scalar f64 inner))
    ~outer:(one (Nx.scalar f64 outer))

let total w = Nx.item [] (Nx.sum w)

let image shape seed =
  Nx.init f64 shape (fun i ->
      let i, j = (i.(0), i.(1)) in
      float_of_int (((7919 * i) + (104729 * j) + (13 * i * j) + seed) mod 1009)
      /. 64.)

(* photutils *)

let photutils =
  let shape = Reference.image_shape in
  let obs = Observation.v (plane shape) (Quantity.v counts (image shape 0)) in
  let sum r =
    Nx.item []
      (Quantity.value
         Unit.(symbol "electron")
         (Observation.integrate r obs).value)
  in
  let near sum reference =
    (* Relative to the sum of |data · w|, all positive here. *)
    equal (float_rel ~rel:1e-12 ~abs:0.) reference sum
  in
  group "photutils"
    [
      cases
        ~name:(fun (row, col, r, _) ->
          Printf.sprintf "circle (%g, %g) r=%g" row col r)
        "exact circles" Reference.pixel_circles
        (fun (row, col, r, s) -> near (sum (circle (row, col) r)) s);
      cases
        ~name:(fun (row, col, i, o, _) ->
          Printf.sprintf "annulus (%g, %g) %g-%g" row col i o)
        "exact annuli" Reference.pixel_annuli
        (fun (row, col, i, o, s) -> near (sum (annulus (row, col) i o)) s);
    ]

(* Laws *)

let centres = Gen.(pair (float_range 14. 26.) (float_range 14. 26.))

let radii =
  Gen.one_of
    [
      Gen.float_range 1e-6 12.;
      Gen.of_list
        [ 0.5; 1.; Float.sqrt 2. /. 2.; Float.sqrt 0.5; 1e-12; 11.999 ];
    ]

let laws =
  group "Laws"
    [
      prop "weights sum to the disc's area and lie in [0, 1]"
        (Gen.pair centres radii) (fun (c, r) ->
          cover "cell-aligned centre" (fst c = Float.round (fst c));
          cover "tiny" (r < 1e-3);
          let w = Region.weights (circle c r) (plane [| 40; 40 |]) in
          equal (float_rel ~rel:1e-13 ~abs:1e-15) (pi *. r *. r) (total w);
          let a = Nx.to_array w in
          at_least float_exact ~than:0. (Array.fold_left min 1. a);
          at_most float_exact ~than:1. (Array.fold_left max 0. a));
      prop "cell-aligned and corner-aligned centres"
        Gen.(pair (int_range 15 25) (int_range 15 25))
        (fun (i, j) ->
          List.iter
            (fun (c, r) ->
              let w = Region.weights (circle c r) (plane [| 40; 40 |]) in
              equal (float_rel ~rel:1e-13 ~abs:0.) (pi *. r *. r) (total w))
            [
              ((float_of_int i, float_of_int j), 0.5);
              ((float_of_int i +. 0.5, float_of_int j +. 0.5), 0.5);
              ((float_of_int i +. 0.5, float_of_int j +. 0.5), Float.sqrt 0.5);
              ((float_of_int i, float_of_int j), 3.);
            ]);
      prop "an annulus weighs the outer disc's area minus the inner's"
        Gen.(triple centres (float_range 0. 6.) (float_range 0. 6.))
        (fun (c, a, b) ->
          let inner = Float.min a b and outer = Float.max a b in
          let w = Region.weights (annulus c inner outer) (plane [| 40; 40 |]) in
          equal
            (float_rel ~rel:1e-12 ~abs:1e-14)
            (pi *. ((outer *. outer) -. (inner *. inner)))
            (total w));
      test "cells wholly inside weigh exactly 1, wholly outside exactly 0"
        (fun () ->
          let w = Region.weights (circle (20., 20.) 5.) (plane [| 40; 40 |]) in
          equal float_exact 1. (Nx.item [ 20; 20 ] w);
          equal float_exact 1. (Nx.item [ 23; 23 ] w);
          equal float_exact 0. (Nx.item [ 26; 20 ] w);
          equal float_exact 0. (Nx.item [ 0; 0 ] w));
      test "a zero radius weighs 0" (fun () ->
          equal float_exact 0.
            (total
               (Region.weights (circle (20.3, 20.1) 0.) (plane [| 40; 40 |]))));
      prop "a window's weights are the base's at the same cells"
        Gen.(
          triple centres radii (pair (int_range (-10) 35) (int_range (-10) 35)))
        (fun (c, r, (i, j)) ->
          let g = plane [| 40; 40 |] in
          let start =
            Nx.create Nx.int64 [| 2 |] [| Int64.of_int i; Int64.of_int j |]
          in
          let w =
            Region.weights (circle c r)
              (Grid.window ~start ~shape:[| 12; 9 |] g)
          in
          let base = Region.weights (circle c r) g in
          cover "beyond the base" (i < 0 || j < 0 || i > 28 || j > 31);
          for a = 0 to 11 do
            for b = 0 to 8 do
              let k = i + a and l = j + b in
              if k >= 0 && l >= 0 && k < 40 && l < 40 then
                equal float_exact (Nx.item [ k; l ] base) (Nx.item [ a; b ] w)
            done
          done);
      test "a linear map's either orientation weighs the same" (fun () ->
          let flip =
            Transform.linear
              (one (Nx.create f64 [| 2; 2 |] [| 0.; 1.; 1.; 0. |]))
          in
          let g = plane [| 40; 40 |]
          and g' = Grid.pixels ~shape:[| 40; 40 |] f64 flip in
          let w = Region.weights (circle (20.2, 17.7) 6.3) g in
          let w' = Region.weights (circle (17.7, 20.2) 6.3) g' in
          equal (array (float 1e-14)) (Nx.to_array w) (Nx.to_array w'));
      test "float32 weights match float64 within float32's rounding" (fun () ->
          let g32 = plane_at Nx.float32 [| 40; 40 |] in
          let w32 =
            Region.weights (circle_at Nx.float32 (20.2, 17.7) 6.3) g32
          in
          let w =
            Region.weights (circle (20.2, 17.7) 6.3) (plane [| 40; 40 |])
          in
          equal
            (array (float 2e-6))
            (Nx.to_array w)
            (Array.map Fun.id (Nx.to_array (Nx.cast f64 w32))));
    ]

(* Errors *)

let errors =
  group "Errors"
    [
      test "a negative radius raises" (fun () ->
          raises
            (Invalid_argument "Region.weights: the radius is -1, not a size")
            (fun () ->
              Region.weights (circle (20., 20.) (-1.)) (plane [| 40; 40 |])));
      test "a batch names the radius's index" (fun () ->
          let r =
            Region.circle
              (at (20., 20.))
              ~radius:(one (Nx.create f64 [| 3 |] [| 1.; 2.; -3. |]))
          in
          raises
            (Invalid_argument
               "Region.weights: the radius at [2] is -3, not a size") (fun () ->
              Region.weights r (plane [| 40; 40 |])));
      test "a NaN radius raises" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"the radius is nan")
            (fun () ->
              Region.weights (circle (20., 20.) Float.nan) (plane [| 40; 40 |])));
      test "an inner radius above the outer raises" (fun () ->
          raises
            (Invalid_argument
               "Region.weights: the inner radius is above the outer one")
            (fun () ->
              Region.weights (annulus (20., 20.) 3. 2.) (plane [| 40; 40 |])));
    ]

(* Derivatives *)

let derivatives =
  let g = plane [| 40; 40 |] in
  let area r c =
    total (Region.weights (Region.circle (at c) ~radius:(one r)) g)
  in
  group "Derivatives"
    [
      prop "the area's derivative in the radius is the circumference"
        (Gen.pair centres (Gen.float_range 0.05 11.))
        (fun (c, r) ->
          let d =
            Rune.grad'
              (fun r ->
                Nx.sum (Region.weights (Region.circle (at c) ~radius:(one r)) g))
              (Nx.scalar f64 r)
          in
          ignore area;
          equal (float_rel ~rel:1e-10 ~abs:1e-12) (2. *. pi *. r) (Nx.item [] d));
      prop "the gradient in the centre matches central differences"
        (Gen.pair centres (Gen.float_range 0.05 10.))
        (fun ((row, col), r) ->
          let image = image [| 40; 40 |] 0 in
          let f c =
            let w =
              Region.weights
                (Region.circle
                   (Transform.shift (one c))
                   ~radius:(one (Nx.scalar f64 r)))
                g
            in
            Nx.sum (Nx.mul w image)
          in
          let at p = Nx.item [] (f (Nx.create f64 [| 2 |] p)) in
          (* Central differences at step e and e/2, combined by Richardson's
             extrapolation; the change of that estimate when e halves bounds
             its error. Near a cell edge the circle nearly touches, the sum's
             higher derivatives grow as (r - d)^(-3/2), and the differences
             do not converge at any step a float64 sum resolves: such a centre
             holds no reference. *)
          let central i =
            let d e =
              let up = [| row; col |] and down = [| row; col |] in
              up.(i) <- up.(i) +. e;
              down.(i) <- down.(i) -. e;
              (at up -. at down) /. (2. *. e)
            in
            let richardson e = ((4. *. d (e /. 2.)) -. d e) /. 3. in
            let fine = richardson 5e-7 in
            (fine, Float.abs (fine -. richardson 1e-6))
          in
          let fd = Array.init 2 central in
          (* A reference holds when its error is a tenth of the tolerance. *)
          Array.iter
            (fun (fd, error) ->
              assume (error <= 0.1 *. (1e-6 +. (1e-6 *. Float.abs fd))))
            fd;
          let d =
            Nx.to_array (Rune.grad' f (Nx.create f64 [| 2 |] [| row; col |]))
          in
          equal (array (float_rel ~rel:1e-6 ~abs:1e-6)) (Array.map fst fd) d);
      test "a zero radius has zero gradient" (fun () ->
          let d =
            Rune.grad'
              (fun r ->
                Nx.sum
                  (Region.weights
                     (Region.circle (at (20.3, 20.1)) ~radius:(one r))
                     g))
              (Nx.scalar f64 0.)
          in
          equal float_exact 0. (Nx.item [] d));
      prop "a circle tangent to a cell edge has a finite, exact derivative"
        Gen.(
          pair (int_range 15 25)
            (one_of [ constant 0.; float_range (-1e-9) 1e-9 ]))
        (fun (i, eps) ->
          (* The circle about (i, 20.3) of radius 3.5 + eps touches the edge at
             row i + 3.5. *)
          cover "exactly tangent" (eps = 0.);
          let d =
            Rune.grad'
              (fun r ->
                Nx.sum
                  (Region.weights
                     (Region.circle (at (float_of_int i, 20.3)) ~radius:(one r))
                     g))
              (Nx.scalar f64 (3.5 +. eps))
          in
          equal
            (float_rel ~rel:1e-6 ~abs:0.)
            (2. *. pi *. (3.5 +. eps))
            (Nx.item [] d));
    ]

let () = exit (run "Region" [ photutils; laws; errors; derivatives ])
