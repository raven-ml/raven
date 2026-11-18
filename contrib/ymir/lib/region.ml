(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Quantity = Ymir_units.Quantity
module Unit = Ymir_units.Unit

let strf = Printf.sprintf

type ('e, 'd) q = (float, 'e, 'd) Nx.t Quantity.t

type ('w, 'e, 'd) shape =
  | Circle of ('e, 'd) q
  | Annulus of { inner : ('e, 'd) q; outer : ('e, 'd) q }
  | Ellipse of { a : ('e, 'd) q; b : ('e, 'd) q; angle : ('e, 'd) q }
  | Polygon of {
      dtype : (float, 'e) Nx.dtype;
      vertices : 'w;  (** On axis -2. *)
      world : ('w, 'd) Transform.endpoint;
    }

(* A planar shape centred on the origin of the plane its placement maps into.
   Weights map corners in float64 and cast the offsets to the shape's dtype, or
   to float32 for a narrower one. *)
type ('w, 'e, 'd) t = {
  shape : ('w, 'e, 'd) shape;
  placement : ('w, 'd Transform.plane, 'd) Transform.t;
}

let circle placement ~radius = { shape = Circle radius; placement }

let annulus placement ~inner ~outer =
  { shape = Annulus { inner; outer }; placement }

let ellipse placement ~a ~b ~angle =
  if not (Unit.convertible (Quantity.unit angle) Unit.radian) then
    invalid_arg
      (Format.asprintf "Region.ellipse: the angle is in %a, not an angle"
         Unit.pp (Quantity.unit angle));
  { shape = Ellipse { a; b; angle }; placement }

let polygon dtype placement vertices =
  let world = Transform.source placement Transform.Planar in
  let points =
    match Transform.value world vertices with
    | Transform.P q -> Quantity.value (Quantity.unit q) q
    | Transform.D d -> d.Direction.xyz
    | Transform.Pair _ ->
        invalid_arg "Region.polygon: vertices are points of a plane or the sky"
  in
  let s = Nx.shape points in
  let n = Array.length s in
  if n < 2 || s.(n - 2) < 3 then
    invalid_arg
      (Format.asprintf
         "Region.polygon: expected at least three vertices on axis -2, got \
          shape %a"
         Transform.pp_ints s);
  { shape = Polygon { dtype; vertices; world }; placement }

(* Checks *)

let payload q = Quantity.value (Quantity.unit q) q

(* [batch n i] drops the [n] cell axes a size carries from a check's index. *)
let batch n i = Direction.at (Array.sub i 0 (max 0 (Array.length i - n)))

let check_size n what r =
  Nx.check Nx.Ptree.tensor (Nx.greater_equal_s r 0.) r (fun i r ->
      Invalid_argument
        (strf "Region.weights: %s%s is %s, not a size" what (batch n i)
           (Transform.item r)))

(* [check_cells area xs ys] raises where a cell is not a convex quadrilateral
   oriented as its window's first cell: the transform folds the window. *)
let check_cells area xs ys =
  let edge k =
    let k = k mod 4 and l = (k + 1) mod 4 in
    Overlap.sub (xs.(l), ys.(l)) (xs.(k), ys.(k))
  in
  let turns =
    List.init 4 (fun k ->
        Nx.greater_s (Nx.mul (Overlap.cross (edge k) (edge (k + 1))) area) 0.)
  in
  let nd = Nx.ndim area in
  let first =
    Nx.slice
      (List.init (nd - 2) (fun _ -> Nx.A) @ [ Nx.R (0, 1); Nx.R (0, 1) ])
      area
  in
  let ok =
    List.fold_left Nx.logical_and (Nx.greater_s (Nx.mul area first) 0.) turns
  in
  Nx.check Nx.Ptree.unit ok () (fun i () ->
      let n = Array.length i in
      Invalid_argument
        (strf
           "Region.weights: cell (%d, %d)%s maps to a quadrilateral that is \
            not convex or not oriented as the window's first cell; the \
            transforms fold the grid there"
           i.(n - 2)
           i.(n - 1)
           (if n > 2 then
              strf " of window %s" (Direction.at (Array.sub i 0 (n - 2)))
            else "")))

let check_radii n inner outer =
  check_size n "the inner radius" inner;
  check_size n "the outer radius" outer;
  Nx.check Nx.Ptree.unit (Nx.less_equal inner outer) () (fun i () ->
      Invalid_argument
        (strf "Region.weights: the inner radius%s is above the outer one"
           (batch n i)))

(* [check_intervals length] raises where a cell of one axis maps to an
   interval oriented against its window's first cell: the transform folds the
   window. A NaN corner is no fold; its cell weighs NaN. *)
let check_intervals length =
  let nd = Nx.ndim length in
  let first =
    Nx.slice (List.init (nd - 1) (fun _ -> Nx.A) @ [ Nx.R (0, 1) ]) length
  in
  let both = Nx.mul length first in
  let ok = Nx.logical_or (Nx.greater_s both 0.) (Nx.isnan both) in
  Nx.check Nx.Ptree.unit ok () (fun i () ->
      let n = Array.length i in
      Invalid_argument
        (strf
           "Region.weights: cell %d%s maps to an interval oriented against \
            the window's first cell; the transforms fold the grid there"
           i.(n - 1)
           (if n > 1 then
              strf " of window %s" (Direction.at (Array.sub i 0 (n - 1)))
            else "")))

(* [check_simple vs] raises where two of the polygon's edges cross. *)
let check_simple vs =
  let n = Array.length vs in
  for i = 0 to n - 1 do
    for j = i + 2 to n - 1 do
      if not (i = 0 && j = n - 1) then
        let crossing =
          Overlap.crosses vs.(i) vs.((i + 1) mod n) vs.(j) vs.((j + 1) mod n)
        in
        Nx.check Nx.Ptree.unit (Nx.logical_not crossing) () (fun b () ->
            Invalid_argument
              (strf "Region.weights: the polygon's edges %d and %d cross%s" i
                 j (Direction.at b)))
    done
  done

(* Weights *)

(* [plane_of unit placement value ~cells] maps [value] through [placement] into
   the plane, in [unit] where given, else the plane's own. *)
let plane_of ?unit placement value ~cells =
  match Transform.run_cells ~cells placement value with
  | Transform.P q, _ ->
      let u = match unit with Some u -> u | None -> Quantity.unit q in
      (Quantity.value u q, u)
  | _ -> invalid_arg "Region.weights: the placement returned no plane"

(* [interval r x0 x1] weighs the cells from [x0] to [x1] on one axis against
   the interval [[-r, r]]: the length of the cell inside it over the cell's,
   exactly 1 for a cell inside and exactly 0 for one outside. *)
let interval r x0 x1 =
  let lo = Nx.minimum x0 x1 and hi = Nx.maximum x0 x1 in
  let length = Nx.sub hi lo in
  let inside =
    Nx.maximum_s (Nx.sub (Nx.minimum hi r) (Nx.maximum lo (Nx.neg r))) 0.
  in
  let empty = Nx.equal_s length 0. in
  let weight =
    Nx.where empty (Nx.zeros_like length)
      (Nx.div inside (Nx.where empty (Nx.ones_like length) length))
  in
  let weight = Nx.where (Nx.isnan length) length weight in
  {
    Overlap.weight;
    area = Nx.sub x1 x0;
    outside =
      Nx.logical_or (Nx.less_equal hi (Nx.neg r)) (Nx.greater_equal lo r);
    full = Nx.logical_and (Nx.greater_equal lo (Nx.neg r)) (Nx.less_equal hi r);
  }

(* [kernel1 r mapped] is {!kernel} on a grid of one axis, whose plane has one
   component. *)
let kernel1 (type w e d) ({ shape; placement } : (w, e, d) t) mapped =
  let ends size =
    let unit = Quantity.unit size in
    let dtype : (float, e) Nx.dtype = Nx.dtype (payload size) in
    let plane, _ = plane_of ~unit placement mapped ~cells:1 in
    if Nx.dim (-1) plane <> 1 then
      invalid_arg
        (strf
           "Region.weights: a grid of one axis is weighed in a plane of one \
            component, got %d"
           (Nx.dim (-1) plane));
    let x = Transform.component (Nx.cast dtype plane) 0 in
    let n = Nx.dim (-1) x - 1 in
    let at o =
      Nx.slice
        (List.init (Nx.ndim x - 1) (fun _ -> Nx.A) @ [ Nx.R (o, o + n) ])
        x
    in
    (unit, at 0, at 1)
  in
  let size unit q = Transform.expand_tensor 1 0 (Quantity.value unit q) in
  let o, own =
    match shape with
    | Circle r ->
        let unit, x0, x1 = ends r in
        let r' = size unit r in
        check_size 1 "the radius" r';
        (interval r' x0 x1, Nx.mul_s (Quantity.value unit r) 2.)
    | Annulus { inner; outer } ->
        let unit, x0, x1 = ends outer in
        let inner' = size unit inner and outer' = size unit outer in
        check_radii 1 inner' outer';
        let o = interval outer' x0 x1 and i = interval inner' x0 x1 in
        let wholly_inside = Nx.logical_and i.full (Nx.logical_not i.outside) in
        ( {
            o with
            weight = Nx.sub o.weight i.weight;
            outside = Nx.logical_or o.outside wholly_inside;
            full = Nx.logical_and o.full i.outside;
          },
          Nx.mul_s
            (Nx.sub (Quantity.value unit outer) (Quantity.value unit inner))
            2. )
    | Ellipse _ | Polygon _ ->
        invalid_arg
          "Region.weights: an ellipse or a polygon needs a plane of two \
           components, and the grid has one axis"
  in
  check_intervals o.area;
  (o, own)

(* [kernel r mapped] is the kernel's result for each cell of the corner lattice
   [mapped] against [r], [[batch; h; w]], and the shape's own area in its
   plane, [[batch]], both in the square of the shape's plane unit, computed at
   [r]'s dtype. *)
let kernel (type w e d) ({ shape; placement } : (w, e, d) t) mapped =
  let sized size =
    let unit = Quantity.unit size in
    let dtype : (float, e) Nx.dtype = Nx.dtype (payload size) in
    let plane, _ = plane_of ~unit placement mapped ~cells:2 in
    (unit, dtype, plane)
  in
  let corners dtype plane =
    let at = Grid.cells (Nx.cast dtype plane) in
    (Array.init 4 (fun k -> at k 0), Array.init 4 (fun k -> at k 1))
  in
  let size unit q = Transform.expand_tensor 2 0 (Quantity.value unit q) in
  let disc unit r = Nx.mul_s (Nx.square (Quantity.value unit r)) Float.pi in
  let o, own, xs, ys =
    match shape with
    | Circle r ->
        let unit, dtype, plane = sized r in
        let xs, ys = corners dtype plane in
        let r' = size unit r in
        check_size 2 "the radius" r';
        (Overlap.disc r' xs ys, disc unit r, xs, ys)
    | Annulus { inner; outer } ->
        let unit, dtype, plane = sized outer in
        let xs, ys = corners dtype plane in
        let inner' = size unit inner and outer' = size unit outer in
        check_radii 2 inner' outer';
        ( Overlap.annulus inner' outer' xs ys,
          Nx.sub (disc unit outer) (disc unit inner),
          xs,
          ys )
    | Ellipse { a; b; angle } ->
        let unit, dtype, plane = sized a in
        let xs, ys = corners dtype plane in
        let a' = size unit a and b' = size unit b in
        let angle' = size Unit.radian angle in
        check_size 2 "the semi-axis a" a';
        check_size 2 "the semi-axis b" b';
        let area =
          Nx.mul_s
            (Nx.mul (Quantity.value unit a) (Quantity.value unit b))
            Float.pi
        in
        (Overlap.ellipse a' b' angle' xs ys, area, xs, ys)
    | Polygon { dtype; vertices; world } ->
        let points, unit =
          plane_of placement (Transform.value world vertices) ~cells:1
        in
        let points = Nx.cast dtype points in
        let plane, _ = plane_of ~unit placement mapped ~cells:2 in
        let xs, ys = corners dtype plane in
        let n = Nx.dim (-2) points in
        let vertex j k =
          let v =
            Nx.slice
              (List.init (Nx.ndim points - 2) (fun _ -> Nx.A)
              @ [ Nx.I j; Nx.I k ])
              points
          in
          v
        in
        let vs = Array.init n (fun j -> (vertex j 0, vertex j 1)) in
        check_simple vs;
        let twice =
          let acc = ref (Nx.zeros_like (fst vs.(0))) in
          for j = 0 to n - 1 do
            acc := Nx.add !acc (Overlap.cross vs.(j) vs.((j + 1) mod n))
          done;
          !acc
        in
        let cellwise (x, y) =
          (Transform.expand_tensor 2 0 x, Transform.expand_tensor 2 0 y)
        in
        ( Overlap.polygon (Array.map cellwise vs) xs ys,
          Nx.mul_s (Nx.abs twice) 0.5,
          xs,
          ys )
  in
  check_cells o.area xs ys;
  (o, own)

(* A dtype narrower than float32 holds neither the cells' offsets nor the
   kernel's squares and cross products: float16 overflows past 65504, float8
   e4m3 saturates at 448 and merges corners 16 cells out. Such a region is
   weighed at float32, and its weights rounded once to its dtype. *)
let dtype_of (type w e d) ({ shape; _ } : (w, e, d) t) : (float, e) Nx.dtype =
  match shape with
  | Circle r -> Nx.dtype (payload r)
  | Annulus { outer; _ } -> Nx.dtype (payload outer)
  | Ellipse { a; _ } -> Nx.dtype (payload a)
  | Polygon { dtype; _ } -> dtype

let narrow (type e) (dtype : (float, e) Nx.dtype) =
  match dtype with
  | Float16 | Bfloat16 | Float8_e4m3fn | Float8_e5m2 | Float4_e2m1fn -> true
  | Float32 | Float64 -> false

(* [widened r] is [r] with its sizes at float32. *)
let widened (type w e d) ({ shape; placement } : (w, e, d) t) :
    (w, Nx.float32_elt, d) t =
  let q x = Quantity.v (Quantity.unit x) (Nx.cast Nx.float32 (payload x)) in
  let shape =
    match shape with
    | Circle r -> Circle (q r)
    | Annulus { inner; outer } -> Annulus { inner = q inner; outer = q outer }
    | Ellipse { a; b; angle } -> Ellipse { a = q a; b = q b; angle = q angle }
    | Polygon { vertices; world; _ } ->
        Polygon { dtype = Nx.float32; vertices; world }
  in
  { shape; placement }

let overlap (type w e d) (r : (w, e, d) t) (g : (w, e, d) Grid.t) :
    (e, d) Overlap.t * (float, e, d) Nx.t =
  let n = Array.length (Grid.shape g) in
  if n > 2 then (
    Grid.coupled "Region.weights" g;
    invalid_arg
      (strf
         "Region.weights: a grid of %d axes is weighed one side at a time \
          (Grid.first, Grid.second)"
         n));
  let kernel r mapped = if n = 1 then kernel1 r mapped else kernel r mapped in
  let mapped = Grid.mapped g in
  let dtype = dtype_of r in
  if not (narrow dtype) then kernel r mapped
  else
    let (o : (Nx.float32_elt, d) Overlap.t), own = kernel (widened r) mapped in
    ( {
        o with
        weight = Nx.cast dtype o.weight;
        area = Nx.cast dtype o.area;
      },
      Nx.cast dtype own )

let weights r g = (fst (overlap r g)).weight

(* Structure *)

module W = Nx.Ptree.Walk

let rec walk_points : type w d.
    (w, d) Transform.endpoint -> ('a, 'b, d, d) W.cursor -> w -> w =
 fun e c x ->
  match e with
  | Transform.Planar -> Transform.quantity c x
  | Transform.Sky _ -> W.structure (Direction.ptree ()) c x
  | Transform.Both (a, b) ->
      let x, y = x in
      ( W.field c "first" (walk_points a) x,
        W.field c "second" (walk_points b) y )

let walk (type w e d) c ({ shape; placement } : (w, e, d) t) : (w, e, d) t =
  let shape =
    match shape with
    | Circle r ->
        W.case c "circle";
        Circle (W.field c "radius" Transform.quantity r)
    | Annulus { inner; outer } ->
        W.case c "annulus";
        let inner = W.field c "inner" Transform.quantity inner in
        let outer = W.field c "outer" Transform.quantity outer in
        Annulus { inner; outer }
    | Ellipse { a; b; angle } ->
        W.case c "ellipse";
        let a = W.field c "a" Transform.quantity a in
        let b = W.field c "b" Transform.quantity b in
        let angle = W.field c "angle" Transform.quantity angle in
        Ellipse { a; b; angle }
    | Polygon { dtype; vertices; world } ->
        W.case c "polygon";
        W.case c (Format.asprintf "%a" Nx.pp_dtype dtype);
        let vertices = W.field c "vertices" (walk_points world) vertices in
        Polygon { dtype; vertices; world }
  in
  let placement =
    W.field c "placement" (W.structure (Transform.ptree ())) placement
  in
  { shape; placement }


let ptree () = Nx.Ptree.v walk
