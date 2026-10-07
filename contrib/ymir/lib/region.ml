(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Quantity = Ymir_units.Quantity
module Unit = Ymir_units.Unit

let strf = Printf.sprintf

type 'e q = (float, 'e) Nx.t Quantity.t

type ('w, 'e) shape =
  | Circle of 'e q
  | Annulus of { inner : 'e q; outer : 'e q }
  | Ellipse of { a : 'e q; b : 'e q; angle : 'e q }
  | Polygon of {
      dtype : (float, 'e) Nx.dtype;
      vertices : 'w;  (** On axis -2. *)
      world : 'w Transform.endpoint;
    }

(* A planar shape centred on the origin of the plane its placement maps into.
   Weights map corners in float64 and cast the offsets to the shape's dtype. *)
type ('w, 'e) t = {
  shape : ('w, 'e) shape;
  placement : ('w, Transform.plane) Transform.t;
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

(* [batch i] drops the two cell axes a size carries from a check's index. *)
let batch i = Direction.at (Array.sub i 0 (max 0 (Array.length i - 2)))

let check_size what r =
  Nx.check Nx.Ptree.tensor (Nx.greater_equal_s r 0.) r (fun i r ->
      Invalid_argument
        (strf "Region.weights: %s%s is %g, not a size" what (batch i)
           (Nx.item [] r)))

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
  | Transform.D _, _ ->
      invalid_arg "Region.weights: the placement returned directions"

(* [overlap r g] is the kernel's result for each of [g]'s cells against [r],
   [[batch; h; w]], and the shape's own area in its plane, [[batch]], both in
   the square of the shape's plane unit. *)
let overlap (type w e) ({ shape; placement } : (w, e) t) (g : (w, e) Grid.t) =
  let sized size =
    let unit = Quantity.unit size in
    let dtype : (float, e) Nx.dtype = Nx.dtype (payload size) in
    let plane, _ = plane_of ~unit placement (Grid.mapped g) ~cells:2 in
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
        check_size "the radius" r';
        (Overlap.disc r' xs ys, disc unit r, xs, ys)
    | Annulus { inner; outer } ->
        let unit, dtype, plane = sized outer in
        let xs, ys = corners dtype plane in
        let inner' = size unit inner and outer' = size unit outer in
        check_size "the inner radius" inner';
        check_size "the outer radius" outer';
        Nx.check Nx.Ptree.unit (Nx.less_equal inner' outer') () (fun i () ->
            Invalid_argument
              (strf "Region.weights: the inner radius%s is above the outer one"
                 (batch i)));
        ( Overlap.annulus inner' outer' xs ys,
          Nx.sub (disc unit outer) (disc unit inner),
          xs,
          ys )
    | Ellipse { a; b; angle } ->
        let unit, dtype, plane = sized a in
        let xs, ys = corners dtype plane in
        let a' = size unit a and b' = size unit b in
        let angle' = size Unit.radian angle in
        check_size "the semi-axis a" a';
        check_size "the semi-axis b" b';
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
        let plane, _ = plane_of ~unit placement (Grid.mapped g) ~cells:2 in
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

let weights r g = (fst (overlap r g)).weight

(* Structure *)

module W = Nx.Ptree.Walk

let walk_points : type w. w Transform.endpoint -> ('a, 'b) W.cursor -> w -> w =
 fun e c x ->
  match e with
  | Transform.Planar -> Transform.quantity c x
  | Transform.Sky _ -> W.structure (Direction.ptree ()) c x

let walk (type w e) c ({ shape; placement } : (w, e) t) : (w, e) t =
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

type ('w, 'e) region = ('w, 'e) t

let ptree (type w e) () : (w, e) t Nx.Ptree.t =
  Nx.Ptree.instantiate
    (module struct
      type _ t = (w, e) region

      let walk = walk
    end)
