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
type ('w, 'e, 'd) t = {
  dtype : (float, 'e) Nx.dtype;
  base : int array;
  shape : int array;
  start : 'd Nx.int64_t;
  transform : ('d Transform.plane, 'w, 'd) Transform.t;
  measured : int;
}

type ('w, 'e, 'd) kind =
  | Pixels : {
      base : int array;
      shape : int array;
      start : 'd Nx.int64_t;
      transform : ('d Transform.plane, 'w, 'd) Transform.t;
    }
      -> ('w, 'e, 'd) kind

let kind g =
  Pixels
    { base = g.base; shape = g.shape; start = g.start; transform = g.transform }

let shape g = Array.copy g.shape
let pp_shape = Transform.pp_ints

let check_shape fn what s =
  if Array.length s = 0 || Array.exists (fun n -> n < 0) s then
    invalid_arg
      (Format.asprintf "%s: %s %a is not one or more non-negative sizes" fn
         what pp_shape s)

(* [check_rank fn what g s] raises unless [s] is sizes of [g]'s axes. *)
let check_rank fn what g s =
  let n = Array.length g.base in
  if Array.length s <> n || Array.exists (fun n -> n < 0) s then
    invalid_arg
      (Format.asprintf "%s: %s %a is not %d non-negative sizes" fn what
         pp_shape s n)

let pixels ~at ~shape dtype transform =
  check_shape "Grid.pixels" "~shape" shape;
  {
    dtype;
    base = Array.copy shape;
    shape = Array.copy shape;
    start = Nx.zeros_at at Nx.int64 [| Array.length shape |];
    transform;
    measured = Transform.length transform;
  }

let map_world t g =
  { g with transform = Transform.(g.transform >> t) }

(* Lattices *)

let component = Transform.component

(* [along n k m] is the shape of [n] axes of 1 but [m] at axis [k]. *)
let along n k m = Array.init n (fun i -> if i = k then m else 1)

(* [lattice start sizes offset] is the pixel coordinates [start + i + offset]
   for each [i] below [sizes], [[...; sizes; n]] in {!Unit.one}: the batch
   axes are [start]'s. Each coordinate is an integer or a half-integer, exact
   in float64, so a window's points are its base's at the same cells, bit for
   bit. *)
let lattice start sizes offset =
  let s = Nx.cast Nx.float64 start in
  let n = Array.length sizes in
  let batch = Array.sub (Nx.shape s) 0 (Nx.ndim s - 1) in
  let coordinate k =
    let at = Nx.reshape (Array.append batch (Array.make n 1)) (component s k) in
    let index = Nx.arange_at (Nx.beside s) Nx.int32 0 sizes.(k) 1 in
    let index = Nx.reshape (along n k sizes.(k)) (Nx.cast Nx.float64 index) in
    let v = Nx.add at index in
    if offset = 0. then v else Nx.add_s v offset
  in
  Quantity.v Unit.one (Transform.stack (List.init n coordinate))

let centre_points g = lattice g.start g.shape 0.
let corner_points g = lattice g.start (Array.map succ g.shape) (-0.5)

(* [quads n v] turns a corner lattice [[...; s + 1; c]] of [n] cell axes into
   each cell's [2ⁿ] corners [[...; s; 2ⁿ; c]]. Corner [k]'s offset on axis [a]
   is bit [a] of [k]'s reflected Gray code, so consecutive corners differ on
   one axis: for two axes (i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1),
   counter-clockwise in the pixel plane. *)
let quads n v =
  let s = Nx.shape v in
  let b = Array.length s - n - 1 in
  let corner k =
    let gray = k lxor (k lsr 1) in
    Nx.slice
      (List.init b (fun _ -> Nx.A)
      @ List.init n (fun a ->
          let o = (gray lsr a) land 1 in
          Nx.R (o, o + s.(b + a) - 1))
      @ [ Nx.A ])
      v
  in
  Nx.stack ~axis:(-2) (List.init (1 lsl n) corner)

let world g = Transform.target Transform.Planar g.transform
let centres g = Transform.apply g.transform (centre_points g)

let corners g =
  let n = Array.length g.shape in
  Transform.map_points (world g) { f = (fun v -> quads n v) }
    (Transform.apply g.transform (corner_points g))

(* [mapped g] is [g]'s corner lattice mapped, with [cells] axes for its
   lattice. *)
let mapped g =
  fst
    (Transform.run_cells ~cells:(Array.length g.shape) g.transform
       (Transform.P (corner_points g)))

(* Products

   A grid's cells are products when its transform is [split first] followed
   only by [both] stages: each cell is then a cell of the first side's axes
   times one of the other side's, and its measure the product of theirs. *)

(* The cells of a grid whose transform maps pixels to pairs: [Product] when
   it is [split first] then [both] stages alone, with each side's stages and
   how many of them lie within the grid's first [measured] stages;
   [Coupled why] naming the stage that maps both sides together; [Single]
   with no split. *)
type (_, 'd) cells =
  | Product : {
      first : int array;
      a : ('d Transform.plane, 'a, 'd) Transform.t;
      b : ('d Transform.plane, 'b, 'd) Transform.t;
      ma : int;
      mb : int;
    }
      -> ('a * 'b, 'd) cells
  | Coupled : string -> ('w, 'd) cells
  | Single : ('w, 'd) cells

type (_, _, _, 'd) sides =
  | Sides :
      ('x, 'a, 'd) Transform.t * ('y, 'b, 'd) Transform.t * int * int
      -> ('x, 'y, 'a * 'b, 'd) sides
  | Joined : string -> ('x, 'y, 'w, 'd) sides

let not_products = strf "the cells are not products: stage %d, %s, %s"

(* [sides_of ~measured t] walks [t] once. *)
let sides_of (type w d) ~measured (t : (d Transform.plane, w, d) Transform.t) :
    (w, d) cells =
  (* [after i t] is the sides of the [both] stages from stage [i]. *)
  let rec after : type x y w. int -> (x * y, w, d) Transform.t -> (x, y, w, d) sides =
   fun i -> function
    | Id -> Sides (Id, Id, 0, 0)
    | Stage (Parallel (t, u), rest) -> (
        match after (i + 1) rest with
        | Sides (a, b, ma, mb) ->
            let m k l = if i < measured then k + l else l in
            Sides
              ( Transform.(t >> a),
                Transform.(u >> b),
                m (Transform.length t) ma,
                m (Transform.length u) mb )
        | Joined why -> Joined why)
    | Stage (s, _) ->
        Joined (not_products i (Transform.stage_name s) "maps both sides of the split together")
  in
  (* [before i name t] finds a split after stage [i - 1], [name]. *)
  let rec before : type a w. int -> string -> (a, w, d) Transform.t -> (w, d) cells =
   fun i name -> function
    | Id -> Single
    | Stage (Split _, _) ->
        Coupled
          (not_products (i - 1) name (strf "maps every axis before stage %d splits them" i))
    | Stage (s, rest) -> before (i + 1) (Transform.stage_name s) rest
  in
  match t with
  | Stage (Split first, rest) -> (
      match after 1 rest with
      | Sides (a, b, ma, mb) -> Product { first; a; b; ma; mb }
      | Joined why -> Coupled why)
  | t -> before 0 "" t

let coupled (type w e d) fn (g : (w, e, d) t) =
  match sides_of ~measured:g.measured g.transform with
  | Coupled why -> invalid_arg (strf "%s: %s" fn why)
  | Product _ | Single -> ()

(* [side g axes t measured] is the grid of [g]'s [axes] through [t]. *)
let side g axes t measured =
  let pick a = Array.map (fun k -> a.(k)) axes in
  {
    dtype = g.dtype;
    base = pick g.base;
    shape = pick g.shape;
    start =
      Transform.stack (Array.to_list (Array.map (component g.start) axes));
    transform = t;
    measured;
  }

let factors (type a b e d) fn (g : (a * b, e, d) t) :
    (a, e, d) t * (b, e, d) t * int array =
  match sides_of ~measured:g.measured g.transform with
  | Product { first; a; b; ma; mb } ->
      let others = Transform.others first (Array.length g.shape) in
      (side g first a ma, side g others b mb, first)
  | Coupled why -> invalid_arg (strf "%s: %s" fn why)
  | Single -> invalid_arg (strf "%s: the cells are not products" fn)

let first g =
  let a, _, _ = factors "Grid.first" g in
  a

let second g =
  let _, b, _ = factors "Grid.second" g in
  b

(* Measures *)

(* [cells v] is each cell's corners as [k]-indexed components [c] of [[...; h;
   w]]. *)
let cells v =
  let q = quads 2 v in
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

(* [ends v] is the low and high ends of each cell of one axis, from a
   corner lattice [[...; s + 1; c]]. *)
let ends v =
  let nd = Nx.ndim v in
  let n = Nx.dim (-2) v - 1 in
  let at o =
    Nx.slice (List.init (nd - 2) (fun _ -> Nx.A) @ [ Nx.R (o, o + n); Nx.A ]) v
  in
  (at 0, at 1)

let rec measure64 : type w e d.
    (w, e, d) t -> (float, Nx.float64_elt, d) Nx.t Quantity.t =
 fun g ->
  let fn = "Grid.measure" in
  match sides_of ~measured:g.measured g.transform with
  | Coupled why -> invalid_arg (strf "%s: %s" fn why)
  | Product { first; a; b; ma; mb } when g.measured > 0 ->
      let others = Transform.others first (Array.length g.shape) in
      product first g
        (measure64 (side g first a ma))
        (measure64 (side g others b mb))
  | Product _ | Single -> (
      let corners =
        fst
          (Transform.run_cells ~stages:g.measured ~cells:(Array.length g.shape)
             g.transform
             (Transform.P (corner_points g)))
      in
      match (Array.length g.shape, corners) with
      | 2, Transform.D d ->
          let at = cells d.xyz in
          let v k = (at k 0, at k 1, at k 2) in
          let omega =
            Nx.add
              (solid_angle (v 0) (v 1) (v 2))
              (solid_angle (v 0) (v 2) (v 3))
          in
          Quantity.v Unit.steradian (Nx.abs omega)
      | 2, Transform.P q ->
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
          Quantity.v Unit.(u ** 2) (Nx.mul_s (Nx.abs twice) 0.5)
      | 1, Transform.P q ->
          let u = Quantity.unit q in
          let lo, hi = ends (Quantity.value u q) in
          let d = Nx.sub hi lo in
          let length =
            if Nx.dim (-1) d = 1 then Nx.abs (component d 0)
            else Nx.sqrt (Nx.sum ~axes:[ -1 ] (Nx.square d))
          in
          Quantity.v u length
      | n, _ ->
          invalid_arg
            (strf
               "%s: a cell of %s here has no measure; one has it on one or two \
                axes in a plane, two on the sky, or as a product of those \
                (Transform.split)"
               fn
               (if n = 1 then "1 axis" else strf "%d axes" n)))

(* [product first g ma mb] is each cell's measure, the product of its sides',
   each side's laid along its axes. *)
and product : type w e d.
    int array -> (w, e, d) t -> (float, Nx.float64_elt, d) Nx.t Quantity.t ->
    (float, Nx.float64_elt, d) Nx.t Quantity.t ->
    (float, Nx.float64_elt, d) Nx.t Quantity.t =
 fun first g ma mb ->
  let n = Array.length g.shape in
  let lay axes q =
    let v = Quantity.value (Quantity.unit q) q in
    let s = Nx.shape v in
    let b = Array.sub s 0 (Array.length s - Array.length axes) in
    let cells = Array.make n 1 in
    Array.iter (fun k -> cells.(k) <- g.shape.(k)) axes;
    Nx.reshape (Array.append b cells) v
  in
  Quantity.v
    Unit.(Quantity.unit ma * Quantity.unit mb)
    (Nx.mul (lay first ma) (lay (Transform.others first n) mb))

let measure g = Quantity.map (Nx.cast g.dtype) (measure64 g)

(* Windows *)

let check_start fn g start =
  let s = Nx.shape start in
  let n = Array.length g.base in
  if Array.length s = 0 || s.(Array.length s - 1) <> n then
    invalid_arg
      (Format.asprintf "%s: ~start takes positions [...; %d], got shape %a" fn
         n pp_shape s)

let window ~start ~shape g =
  check_rank "Grid.window" "~shape" g shape;
  check_start "Grid.window" g start;
  { g with shape = Array.copy shape; start = Nx.add g.start start }

(* [locate x ~shape g] is the start of the [shape] block of [g]'s base centred
   on the cell holding each point [x], [[...; n]]: the extra cell of an even
   size on the high side, the base's size for a point [g]'s inverse does not
   cover, every start clamped to [[-shape, base]]. *)
let locate x ~shape g =
  let fn = "Grid.around" in
  check_rank fn "~shape" g shape;
  let v = Transform.value (world g) x in
  let p, ok = Transform.run_cells ~cells:0 (Transform.inverse g.transform) v in
  let p =
    match p with
    | Transform.P q -> Quantity.value Unit.one q
    | _ -> invalid_arg "Grid.around: the inverse transform returned no plane"
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
  Transform.stack (List.init (Array.length shape) axis)

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


let ptree () = Nx.Ptree.v walk

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
  if Nx.Dtype.is Nx.Dtype.Float (Nx.dtype a) then
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
  let on = Nx.beside a.start in
  match compare a b with
  | Error _ -> Nx.scalar_at on Nx.bool false
  | Ok leaves ->
      List.fold_left
        (fun acc (eq, core, _, _) ->
          let n = Nx.ndim eq in
          let eq =
            if core = 0 then eq
            else Nx.all ~axes:(List.init core (fun k -> n - 1 - k)) eq
          in
          Nx.logical_and acc eq)
        (Nx.scalar_at on Nx.bool true) leaves

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
                   "%s: the grids do not agree at %a: element %s is %s in the \
                    first and %s in the second"
                   fn Nx.Ptree.Path.pp path (index_text i) (Transform.item x)
                   (Transform.item y))))
        leaves
