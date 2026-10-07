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

let ellipse_at (row, col) a b angle =
  Region.ellipse (at (row, col))
    ~a:(one (Nx.scalar f64 a))
    ~b:(one (Nx.scalar f64 b))
    ~angle:(Quantity.v Unit.degree (Nx.scalar f64 angle))

(* [polygon_of vs] is the polygon of [(row, column)] vertices [vs] in pixel
   coordinates. *)
let polygon_of vs =
  Region.polygon f64 Transform.id
    (one (Nx.create f64 [| Array.length vs / 2; 2 |] vs))

let shapes =
  let shape = Reference.image_shape in
  let obs = Observation.v (plane shape) (Quantity.v counts (image shape 0)) in
  let sum r =
    Nx.item []
      (Quantity.value
         Unit.(symbol "electron")
         (Observation.integrate r obs).value)
  in
  let near sum reference = equal (float_rel ~rel:1e-12 ~abs:0.) reference sum in
  group "photutils and regions"
    [
      cases
        ~name:(fun (row, col, a, b, angle, _) ->
          Printf.sprintf "ellipse (%g, %g) %g×%g at %g" row col a b angle)
        "exact ellipses" Reference.pixel_ellipses
        (fun (row, col, a, b, angle, s) ->
          near (sum (ellipse_at (row, col) a b angle)) s);
      cases
        ~name:(fun (vs, _) -> Printf.sprintf "%d vertices from (%g, %g)"
                                (Array.length vs / 2) vs.(0) vs.(1))
        "exact polygons" Reference.pixel_polygons
        (fun (vs, s) -> near (sum (polygon_of vs)) s);
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
          (* Each cell the boundary crosses adds a few roundings of its area,
             one square pixel: 16 ε for the four cells about a tiny disc. *)
          equal (float_rel ~rel:1e-13 ~abs:4e-15) (pi *. r *. r) (total w);
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
      test "a small disc near a long cell edge weighs its exact fraction"
        (fun () ->
          (* Cells 1 by 1e-8: the disc's radius, 3.7e-9, is below the rounding
             of the squared distance to a corner, and the edges 5.3e-9 away
             pass outside it. *)
          let k = 1.0536e-8 and r = 3.7253595981663415e-09 in
          let g =
            Grid.pixels ~shape:[| 40; 40 |] f64
              (Transform.linear
                 (one (Nx.create f64 [| 2; 2 |] [| 1.; 0.; 0.; k |])))
          in
          let disc =
            Region.circle
              (Transform.shift (one (Nx.create f64 [| 2 |] [| 14.; 14. *. k |])))
              ~radius:(one (Nx.scalar f64 r))
          in
          equal (float_rel ~rel:1e-6 ~abs:0.) (pi *. r *. r /. k)
            (Nx.item [ 14; 14 ] (Region.weights disc g)));
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

(* A star-shaped polygon about [c]: [k] vertices at increasing angles and
   radii in [[1, 9]]. *)
let stars =
  Gen.(
    map
      (fun ((c, k), (phase, radii)) ->
        let k = 3 + (k mod 6) in
        Array.concat
          (List.init k (fun j ->
               let a =
                 phase +. (2. *. Float.pi *. float_of_int j /. float_of_int k)
               in
               (* Shrinking may shorten the list. *)
               let r =
                 match radii with
                 | [] -> 5.
                 | _ -> List.nth radii (j mod List.length radii)
               in
               [| fst c +. (r *. Float.cos a); snd c +. (r *. Float.sin a) |])))
      (pair (pair centres (int_range 0 5))
         (pair (float_range 0. 6.3) (list ~size:(constant 8) (float_range 1. 9.)))))

let polygon_area vs =
  let n = Array.length vs / 2 in
  let acc = ref 0. in
  for j = 0 to n - 1 do
    let k = (j + 1) mod n in
    acc := !acc +. ((vs.(2 * j) *. vs.((2 * k) + 1)) -. (vs.(2 * k) *. vs.((2 * j) + 1)))
  done;
  Float.abs !acc /. 2.

let reverse vs =
  let n = Array.length vs / 2 in
  Array.init (2 * n) (fun i -> vs.((2 * (n - 1 - (i / 2))) + (i mod 2)))

let shape_laws =
  group "Shape laws"
    [
      prop "an ellipse's weights sum to π a b and lie in [0, 1]"
        Gen.(
          pair centres
            (triple (float_range 0.1 10.) (float_range 0.1 10.)
               (float_range (-180.) 180.)))
        (fun (c, (a, b, angle)) ->
          cover "thin" (Float.min a b < 0.5 *. Float.max a b);
          let w = Region.weights (ellipse_at c a b angle) (plane [| 40; 40 |]) in
          (* Axis ratios up to 100, where the rounding stays below 1e-13 of
             the area; a few roundings of a square pixel in each of the up to
             80 cells the boundary crosses. *)
          equal (float_rel ~rel:1e-13 ~abs:1e-13) (pi *. a *. b) (total w);
          let v = Nx.to_array w in
          at_least float_exact ~than:0. (Array.fold_left min 1. v);
          at_most float_exact ~than:1. (Array.fold_left max 0. v));
      prop "an ellipse with equal axes is the circle" Gen.(pair centres (pair radii (float_range (-180.) 180.)))
        (fun (c, (r, angle)) ->
          let g = plane [| 40; 40 |] in
          let r = Float.min r 11. in
          equal (array (float 1e-12))
            (Nx.to_array (Region.weights (circle c r) g))
            (Nx.to_array (Region.weights (ellipse_at c r r angle) g)));
      prop "a polygon's weights sum to its area and lie in [0, 1]" stars
        (fun vs ->
          let w = Region.weights (polygon_of vs) (plane [| 40; 40 |]) in
          equal (float_rel ~rel:1e-13 ~abs:1e-13) (polygon_area vs) (total w);
          let v = Nx.to_array w in
          at_least float_exact ~than:0. (Array.fold_left min 1. v);
          at_most float_exact ~than:1. (Array.fold_left max 0. v));
      prop "either orientation weighs the same" stars (fun vs ->
          let g = plane [| 40; 40 |] in
          equal (array (float 1e-14))
            (Nx.to_array (Region.weights (polygon_of vs) g))
            (Nx.to_array (Region.weights (polygon_of (reverse vs)) g)));
      prop "two halves of a polygon add to it"
        Gen.(pair centres (pair (float_range 1. 9.) (float_range (-1.) 1.)))
        (fun ((r0, c0), (s, t)) ->
          (* A square of side 2s about (r0, c0) cut by a line through its
             centre at slope t. *)
          let a = (r0 -. s, c0 -. s) and b = (r0 -. s, c0 +. s)
          and c = (r0 +. s, c0 +. s) and d = (r0 +. s, c0 -. s) in
          let left = (r0 -. (s *. t), c0 -. s) and right = (r0 +. (s *. t), c0 +. s) in
          let flat ps = Array.concat (List.map (fun (x, y) -> [| x; y |]) ps) in
          let g = plane [| 40; 40 |] in
          let w ps = Nx.to_array (Region.weights (polygon_of (flat ps)) g) in
          let whole = w [ a; b; c; d ] in
          let one_half = w [ a; b; right; left ] and other = w [ left; right; c; d ] in
          equal (array (float 1e-13)) whole (Array.map2 ( +. ) one_half other));
      test "a polygon on cell edges weighs its cells exactly 1, the rest 0"
        (fun () ->
          let w =
            Region.weights
              (polygon_of [| 9.5; 9.5; 9.5; 19.5; 19.5; 19.5; 19.5; 9.5 |])
              (plane [| 40; 40 |])
          in
          equal float_exact 1. (Nx.item [ 10; 10 ] w);
          equal float_exact 1. (Nx.item [ 19; 19 ] w);
          equal float_exact 0. (Nx.item [ 9; 10 ] w);
          equal float_exact 0. (Nx.item [ 20; 15 ] w);
          equal float_exact 100. (total w));
      test "a gnomonic polygon on a TAN cell's corners weighs that cell 1"
        (fun () ->
          let centre =
            Direction.lonlat Frame.icrs
              ~lon:(Quantity.v Unit.degree (Nx.scalar f64 30.))
              ~lat:(Quantity.v Unit.degree (Nx.scalar f64 (-50.)))
          in
          let pixel = 1. /. 3600. in
          let wcs =
            Transform.(
              shift (one (Nx.create f64 [| 2 |] [| 10.; 10. |]))
              >> linear
                   (Quantity.v Unit.degree
                      (Nx.create f64 [| 2; 2 |] [| 0.; pixel; pixel; 0. |]))
              >> inverse (gnomonic centre))
          in
          let g = Grid.pixels ~shape:[| 20; 20 |] f64 wcs in
          let corners = Grid.corners (Grid.window ~start:(Nx.create Nx.int64 [| 2 |] [| 7L; 12L |]) ~shape:[| 1; 1 |] g) in
          let vertices =
            Direction.of_xyz Frame.icrs
              (Nx.reshape [| 4; 3 |] (Direction.xyz corners))
          in
          let w = Region.weights (Region.polygon f64 (Transform.gnomonic centre) vertices) g in
          equal (float 1e-9) 1. (Nx.item [ 7; 12 ] w);
          equal (float 1e-9) 1. (total w));
      test "a batch of polygons weighs each" (fun () ->
          let a = [| 10.; 10.; 10.; 20.; 20.; 20.; 20.; 10. |]
          and b = [| 12.2; 11.; 25.7; 12.1; 18.4; 27.9 |] in
          let b4 = Array.append b [| 18.4; 27.9 |] in
          (* A repeated vertex adds an empty edge. *)
          let batch =
            Region.polygon f64 Transform.id
              (one (Nx.create f64 [| 2; 4; 2 |] (Array.append a b4)))
          in
          let g = plane [| 40; 40 |] in
          let w = Region.weights batch g in
          equal (array (float 1e-14))
            (Nx.to_array (Region.weights (polygon_of a) g))
            (Nx.to_array (Nx.slice [ Nx.I 0 ] w));
          equal (array (float 1e-13))
            (Nx.to_array (Region.weights (polygon_of b) g))
            (Nx.to_array (Nx.slice [ Nx.I 1 ] w)));
      test "integrate covers a polygon's own area" (fun () ->
          let g = plane [| 40; 40 |] in
          let obs = Observation.v g (Quantity.v counts (Nx.ones f64 [| 40; 40 |])) in
          let vs = [| 30.; 30.; 30.; 45.; 45.; 45.; 45.; 30. |] in
          let i = Observation.integrate (polygon_of vs) obs in
          (* A quarter of the square lies on the image: 9.5 × 9.5 of 15 × 15. *)
          equal (float 1e-12) (9.5 *. 9.5 /. 225.) (Nx.item [] i.coverage));
      test "crossing edges raise" (fun () ->
          raises
            (Invalid_argument "Region.weights: the polygon's edges 0 and 2 cross")
            (fun () ->
              Region.weights
                (polygon_of [| 10.; 10.; 20.; 20.; 10.; 20.; 20.; 10. |])
                (plane [| 40; 40 |])));
      test "the area differentiates in a, and in a vertex" (fun () ->
          let g = plane [| 40; 40 |] in
          let area a =
            Nx.sum
              (Region.weights
                 (Region.ellipse (at (20.2, 19.1)) ~a:(one a)
                    ~b:(one (Nx.scalar f64 3.))
                    ~angle:(Quantity.v Unit.degree (Nx.scalar f64 25.)))
                 g)
          in
          equal (float 1e-10) (pi *. 3.)
            (Nx.item [] (Rune.grad' area (Nx.scalar f64 5.)));
          let tri v =
            Nx.sum
              (Region.weights
                 (Region.polygon f64 Transform.id
                    (one
                       (Nx.concatenate ~axis:0
                          [
                            Nx.create f64 [| 2; 2 |] [| 10.2; 11.3; 25.7; 12.1 |];
                            Nx.reshape [| 1; 2 |] v;
                          ])))
                 g)
          in
          (* The area of (p, q, v) moves with v by half of (q - p) rotated. *)
          equal (array (float 1e-10))
            [| (12.1 -. 11.3) /. -2.; (25.7 -. 10.2) /. 2. |]
            (Nx.to_array (Rune.grad' tri (Nx.create f64 [| 2 |] [| 18.4; 27.9 |]))));
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

let () =
  exit
    (run "Region"
       [ photutils; shapes; laws; shape_laws; errors; derivatives ])
