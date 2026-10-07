(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Quantity = Ymir_units.Quantity
module Unit = Ymir_units.Unit

let strf = Printf.sprintf

type 'e q = (float, 'e) Nx.t Quantity.t
type 'e shape = Circle of 'e q | Annulus of { inner : 'e q; outer : 'e q }

(* A planar shape centred on the origin of the plane its placement maps into.
   Weights map corners in float64 and cast the offsets to the shape's dtype. *)
type ('w, 'e) t = {
  shape : 'e shape;
  placement : ('w, Transform.plane) Transform.t;
}

let circle placement ~radius = { shape = Circle radius; placement }

let annulus placement ~inner ~outer =
  { shape = Annulus { inner; outer }; placement }

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

(* Weights *)

(* [overlap r g] is the kernel's result for each of [g]'s cells against [r],
   [[batch; h; w]], and the shape's own area in its plane, [[batch]], both in
   the square of the shape's size unit. *)
let overlap (type w e) ({ shape; placement } : (w, e) t) (g : (w, e) Grid.t) =
  let size = match shape with Circle r -> r | Annulus { outer; _ } -> outer in
  let unit = Quantity.unit size in
  let dtype : (float, e) Nx.dtype = Nx.dtype (payload size) in
  let plane =
    match Transform.run_cells ~cells:2 placement (Grid.mapped g) with
    | Transform.P q, _ -> Quantity.value unit q
    | Transform.D _, _ ->
        invalid_arg "Region.weights: the placement returned directions"
  in
  let at = Grid.cells (Nx.cast dtype plane) in
  let xs = Array.init 4 (fun k -> at k 0)
  and ys = Array.init 4 (fun k -> at k 1) in
  let size q = Transform.expand_tensor 2 0 (Quantity.value unit q) in
  let disc r = Nx.mul_s (Nx.square (Quantity.value unit r)) Float.pi in
  let o, own =
    match shape with
    | Circle r ->
        let r' = size r in
        check_size "the radius" r';
        (Overlap.disc r' xs ys, disc r)
    | Annulus { inner; outer } ->
        let inner' = size inner and outer' = size outer in
        check_size "the inner radius" inner';
        check_size "the outer radius" outer';
        Nx.check Nx.Ptree.unit (Nx.less_equal inner' outer') () (fun i () ->
            Invalid_argument
              (strf "Region.weights: the inner radius%s is above the outer one"
                 (batch i)));
        (Overlap.annulus inner' outer' xs ys, Nx.sub (disc outer) (disc inner))
  in
  check_cells o.area xs ys;
  (o, own)

let weights r g = (fst (overlap r g)).weight

(* Structure *)

module W = Nx.Ptree.Walk

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
