(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Quantity = Ymir_units.Quantity
module Unit = Ymir_units.Unit

let strf = Printf.sprintf
let cell = Unit.symbol "cell"

(* [measured] counts the transform's first stages, those of the world the
   grid was built with, in which its measures are taken: {!map_world} appends
   stages and keeps the measures. *)
type ('w, 'e) t = {
  dtype : (float, 'e) Nx.dtype;
  base : int array;
  shape : int array;
  start : Nx.int64_t;
  transform : (Transform.plane, 'w) Transform.t;
  measured : int;
}

type ('w, 'e) kind =
  | Pixels : {
      base : int array;
      shape : int array;
      start : Nx.int64_t;
      transform : (Transform.plane, 'w) Transform.t;
    }
      -> ('w, 'e) kind

let kind g =
  Pixels
    { base = g.base; shape = g.shape; start = g.start; transform = g.transform }

let shape g = Array.copy g.shape
let pp_shape = Transform.pp_ints

let check_shape fn what s =
  if Array.length s <> 2 || Array.exists (fun n -> n < 0) s then
    invalid_arg
      (Format.asprintf "%s: %s %a is not two non-negative sizes" fn what
         pp_shape s)

let pixels ~shape dtype transform =
  check_shape "Grid.pixels" "~shape" shape;
  {
    dtype;
    base = Array.copy shape;
    shape = Array.copy shape;
    start = Nx.zeros Nx.int64 [| 2 |];
    transform;
    measured = Transform.length transform;
  }

let map_world t g =
  { g with transform = Transform.(g.transform >> t) }

(* Lattices *)

let component = Transform.component

(* [lattice start (h, w) offset] is the pixel coordinates [start + (i, j) +
   offset] for [i < h], [j < w], [[...; h; w; 2]] in {!Unit.one}: the batch
   axes are [start]'s. Each coordinate is an integer or a half-integer, exact
   in float64, so a window's points are its base's at the same cells, bit for
   bit. *)
let lattice start h w offset =
  let s = Nx.cast Nx.float64 start in
  let batch = Array.sub (Nx.shape s) 0 (Nx.ndim s - 1) in
  let at k = Nx.reshape (Array.append batch [| 1; 1 |]) (component s k) in
  let axis n = Nx.cast Nx.float64 (Nx.arange Nx.int32 0 n 1) in
  let rows = Nx.add (at 0) (Nx.reshape [| h; 1 |] (axis h)) in
  let cols = Nx.add (at 1) (Nx.reshape [| 1; w |] (axis w)) in
  let shift v = if offset = 0. then v else Nx.add_s v offset in
  Quantity.v Unit.one (Transform.stack [ shift rows; shift cols ])

let centre_points g = lattice g.start g.shape.(0) g.shape.(1) 0.
let corner_points g = lattice g.start (g.shape.(0) + 1) (g.shape.(1) + 1) (-0.5)

(* [quads v] turns a corner lattice [[...; h + 1; w + 1; c]] into each cell's
   corners [[...; h; w; 4; c]], counter-clockwise in the pixel plane: (i, j), (i
   + 1, j), (i + 1, j + 1), (i, j + 1). *)
let quads v =
  let s = Nx.shape v in
  let nd = Array.length s in
  let h = s.(nd - 3) - 1 and w = s.(nd - 2) - 1 in
  let at a b =
    Nx.slice
      (List.init (nd - 3) (fun _ -> Nx.A)
      @ [ Nx.R (a, a + h); Nx.R (b, b + w); Nx.A ])
      v
  in
  Nx.stack ~axis:(-2) [ at 0 0; at 1 0; at 1 1; at 0 1 ]

let world g = Transform.target Transform.Planar g.transform
let centres g = Transform.apply g.transform (centre_points g)

let corners g =
  Transform.map_points (world g) { f = quads }
    (Transform.apply g.transform (corner_points g))

(* [mapped g] is [g]'s corner lattice mapped, with [cells] axes for its
   lattice. *)
let mapped g =
  fst (Transform.run_cells ~cells:2 g.transform (Transform.P (corner_points g)))

(* Measures *)

(* [cells v] is each cell's corners as [k]-indexed components [c] of [[...; h;
   w]]. *)
let cells v =
  let q = quads v in
  let nd = Nx.ndim q in
  fun k c ->
    Nx.slice (List.init (nd - 2) (fun _ -> Nx.A) @ [ Nx.I k; Nx.I c ]) q

(* The solid angle of the triangle of unit vectors [a], [b], [c], by Van
   Oosterom and Strackee: [tan (Ω / 2) = a · (b × c) / (1 + a · b + b · c + c ·
   a)], the triple product taken on differences so it keeps its accuracy for
   small triangles. *)
let solid_angle a b c =
  let open Direction in
  let num = dot a (cross (sub3 b a) (sub3 c a)) in
  let den = Nx.add_s (Nx.add (Nx.add (dot a b) (dot b c)) (dot c a)) 1. in
  Nx.mul_s (Nx.atan2 num den) 2.

let measure g =
  let corners =
    fst
      (Transform.run_cells ~stages:g.measured ~cells:2 g.transform
         (Transform.P (corner_points g)))
  in
  match corners with
  | Transform.D d ->
      let at = cells d.xyz in
      let v k = (at k 0, at k 1, at k 2) in
      let omega =
        Nx.add (solid_angle (v 0) (v 1) (v 2)) (solid_angle (v 0) (v 2) (v 3))
      in
      Quantity.v Unit.steradian (Nx.cast g.dtype (Nx.abs omega))
  | Transform.P q ->
      let u = Quantity.unit q in
      let at = cells (Quantity.value u q) in
      let p k = (at k 0, at k 1) in
      let cross (ax, ay) (bx, by) = Nx.sub (Nx.mul ax by) (Nx.mul ay bx) in
      let sub (ax, ay) (bx, by) = (Nx.sub ax bx, Nx.sub ay by) in
      let c0 = p 0 in
      let twice =
        Nx.add
          (cross (sub (p 1) c0) (sub (p 2) c0))
          (cross (sub (p 2) c0) (sub (p 3) c0))
      in
      Quantity.v Unit.(u ** 2) (Nx.cast g.dtype (Nx.mul_s (Nx.abs twice) 0.5))

(* Windows *)

let check_start fn start =
  let s = Nx.shape start in
  if Array.length s = 0 || s.(Array.length s - 1) <> 2 then
    invalid_arg
      (Format.asprintf "%s: ~start takes positions [...; 2], got shape %a" fn
         pp_shape s)

let window ~start ~shape g =
  check_shape "Grid.window" "~shape" shape;
  check_start "Grid.window" start;
  { g with shape = Array.copy shape; start = Nx.add g.start start }

(* [locate x ~shape g] is the start of the [shape] block of [g]'s base centred
   on the cell holding each point [x], [[...; 2]]: the extra cell of an even
   size on the high side, the base's size for a point [g]'s inverse does not
   cover, every start clamped to [[-shape, base]]. *)
let locate x ~shape g =
  let fn = "Grid.around" in
  check_shape fn "~shape" shape;
  let v = Transform.value (world g) x in
  let p, ok = Transform.run_cells ~cells:0 (Transform.inverse g.transform) v in
  let p =
    match p with
    | Transform.P q -> Quantity.value Unit.one q
    | Transform.D _ ->
        invalid_arg "Grid.around: the inverse transform returned directions"
  in
  let ok =
    let n = Transform.numbers p in
    match ok with None -> n | Some ok -> Nx.logical_and ok n
  in
  let axis k =
    let s = shape.(k) and b = float_of_int g.base.(k) in
    let first =
      Nx.sub_s
        (Nx.floor (Nx.add_s (component p k) 0.5))
        (float_of_int ((s - 1) / 2))
    in
    let first = Nx.where ok first (Nx.full_like first b) in
    Nx.cast Nx.int64 (Nx.clamp ~min:(float_of_int (-s)) ~max:b first)
  in
  Transform.stack [ axis 0; axis 1 ]

let around x ~shape g =
  { g with shape = Array.copy shape; start = locate x ~shape g }

(* Structure *)

module W = Nx.Ptree.Walk

let walk c g =
  W.case c "pixels";
  let base = W.field c "base" Transform.ints g.base in
  let shape = W.field c "shape" Transform.ints g.shape in
  W.case c (Format.asprintf "%a" Nx.pp_dtype g.dtype);
  let measured = W.field c "measured" W.int g.measured in
  let start = W.field c "start" W.tensor g.start in
  let transform =
    W.field c "transform" (W.structure (Transform.ptree ())) g.transform
  in
  { g with base; shape; start; transform; measured }

type ('w, 'e) grid = ('w, 'e) t

let ptree (type w e) () : (w, e) t Nx.Ptree.t =
  Nx.Ptree.instantiate
    (module struct
      type _ t = (w, e) grid

      let walk = walk
    end)

(* Agreement

   Two grids agree where every static datum is equal and every leaf holds the
   same numbers, NaN equal to NaN: one boolean per batch element, each leaf's
   comparison reduced over its own axes. *)

(* [cores g] is the number of non-batch axes of each of [g]'s leaves, in walk
   order. *)
let cores g = 1 :: Transform.cores g.transform

let paths g =
  List.filter_map
    (function Nx.Ptree.Leaf p -> Some p | Nx.Ptree.Report _ -> None)
    (Nx.Ptree.visits (ptree ()) g)

(* [same a b] is where the leaves [a] and [b] hold the same number. *)
let same (Nx.P a) (Nx.P b) =
  if Nx_dtype.is Nx_dtype.Float (Nx.dtype a) then
    let a = Nx.cast Nx.float64 a and b = Nx.cast Nx.float64 b in
    Nx.logical_or (Nx.equal a b) (Nx.logical_and (Nx.isnan a) (Nx.isnan b))
  else Nx.equal (Nx.cast Nx.int64 a) (Nx.cast Nx.int64 b)

(* [compare a b] is [Error d] naming the first static difference, or each
   leaf's elementwise agreement with its path, its core and both sides. *)
let compare a b =
  let la, ka = Nx.Ptree.flatten (ptree ()) a
  and lb, kb = Nx.Ptree.flatten (ptree ()) b in
  match
    Nx.Ptree.Skeleton.diff ~this:"in the first" ka ~that:"in the second" kb
  with
  | Some d -> Error d
  | None ->
      Ok
        (List.map2
           (fun ((x, y), core) path -> (same x y, core, path, (x, y)))
           (List.combine la lb |> fun l -> List.combine l (cores a))
           (paths a))

let agree a b =
  match compare a b with
  | Error _ -> Nx.scalar Nx.bool false
  | Ok leaves ->
      List.fold_left
        (fun acc (eq, core, _, _) ->
          let n = Nx.ndim eq in
          let eq =
            if core = 0 then eq
            else Nx.all ~axes:(List.init core (fun k -> n - 1 - k)) eq
          in
          Nx.logical_and acc eq)
        (Nx.scalar Nx.bool true) leaves

let index_text i =
  "(" ^ String.concat ", " (Array.to_list (Array.map string_of_int i)) ^ ")"

let float64 (Nx.P x) = Nx.cast Nx.float64 x

(* [require fn a b] raises [Invalid_argument] naming [fn] where [a] and [b]
   do not agree: at trace time for a static difference, through [Nx.check]
   for a leaf. *)
let require fn a b =
  match compare a b with
  | Error d -> invalid_arg (strf "%s: the grids do not agree at %s" fn d)
  | Ok leaves ->
      List.iter
        (fun (eq, _, path, (x, y)) ->
          Nx.check
            Nx.Ptree.(pair tensor tensor)
            eq
            (float64 x, float64 y)
            (fun i (x, y) ->
              Invalid_argument
                (Format.asprintf
                   "%s: the grids do not agree at %a: element %s is %g in the \
                    first and %g in the second"
                   fn Nx.Ptree.Path.pp path (index_text i) (Nx.item [] x)
                   (Nx.item [] y))))
        leaves
