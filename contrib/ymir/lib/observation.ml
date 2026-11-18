(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Quantity = Ymir_units.Quantity
module Unit = Ymir_units.Unit

let strf = Printf.sprintf

type ('e, 'd) q = (float, 'e, 'd) Nx.t Quantity.t

(* [data] and [variance] are 0 wherever [valid] is false. *)
type ('w, 'e, 'd) t = {
  grid : ('w, 'e, 'd) Grid.t;
  data : ('e, 'd) q;
  variance : ('e, 'd) q option;
  valid : 'd Nx.bit_t option;
  area : ('e, 'd) q option;
}

let grid o = o.grid
let data o = o.data
let variance o = o.variance
let valid o = o.valid
let area o = o.area
let payload q = Quantity.value (Quantity.unit q) q
let pp_shape = Transform.pp_ints

(* [zero_invalid valid q] replaces [q] by 0 where [valid] is false, so a NaN or
   a sentinel there reaches no result under any derivative. *)
let zero_invalid valid q =
  Quantity.map (fun x -> Nx.where valid x (Nx.zeros_like x)) q

let zero_all valid o =
  let z = zero_invalid valid in
  { o with data = z o.data; variance = Option.map z o.variance }

(* Construction *)

let ends_with s suffix =
  let n = Array.length s and k = Array.length suffix in
  n >= k && Array.sub s (n - k) k = suffix

(* An area holds one value per cell when its last axes are the grid's, and one
   for all cells when they are 1 or absent. *)
let per_cell s gs = ends_with s gs

let constant s cells =
  let n = Array.length s in
  List.for_all (fun k -> n < k || s.(n - k) = 1) (List.init cells succ)

(* The unit of [g]'s measures: a steradian on the sky, the square of the plane's
   unit, which one cell's measure gives, in a plane. *)
let measure_unit (type w e d) (g : (w, e, d) Grid.t) =
  match Grid.world g with
  | Transform.Sky _ -> Unit.steradian
  | _ ->
      let one =
        Grid.window ~start:(Nx.zeros_like g.Grid.start)
          ~shape:(Array.map (fun _ -> 1) g.Grid.shape)
          g
      in
      Quantity.unit (Grid.measure one)

let v ?variance ?valid ?area grid data =
  let fn = "Observation.v" in
  let shape = Nx.shape (payload data) in
  let gs = Grid.shape grid in
  if not (ends_with shape gs) then
    invalid_arg
      (Format.asprintf
         "%s: data of shape %a do not end with the grid's shape %a" fn pp_shape
         shape pp_shape gs);
  let du = Quantity.unit data in
  Option.iter
    (fun var ->
      if not (Unit.convertible (Quantity.unit var) Unit.(du ** 2)) then
        invalid_arg
          (Format.asprintf
             "%s: the variance is in %a, not the data's unit squared, %a" fn
             Unit.pp (Quantity.unit var) Unit.pp
             Unit.(du ** 2)))
    variance;
  let area =
    Option.map
      (fun a ->
        let mu = measure_unit grid in
        if not (Unit.convertible (Quantity.unit a) mu) then
          invalid_arg
            (Format.asprintf
               "%s: the area is in %a, not a measure of the grid's cells (%a)"
               fn Unit.pp (Quantity.unit a) Unit.pp mu);
        let s = Nx.shape (payload a) in
        if
          Transform.broadcast s shape <> shape
          || not (per_cell s gs || constant s (Array.length gs))
        then
          invalid_arg
            (Format.asprintf
               "%s: an area of shape %a neither holds one value per cell of \
                data of shape %a nor one for all of them"
               fn pp_shape s pp_shape shape);
        a)
      area
  in
  let o = { grid; data; variance; valid; area } in
  match valid with None -> o | Some m -> zero_all (Nx.cast Nx.bool m) o

let restrict m o =
  let m = Nx.cast Nx.bool m in
  let m =
    match o.valid with
    | None -> m
    | Some v -> Nx.logical_and (Nx.cast Nx.bool v) m
  in
  zero_all m { o with valid = Some (Nx.cast Nx.bit m) }

let map_world t o = { o with grid = Grid.map_world t o.grid }

(* Arithmetic

   Sums, differences and scalings of observations on agreeing grids, whose
   variances combine exactly for independent samples. *)

(* [require_areas fn a b] raises where [a] and [b] are not one area. *)
let require_areas fn a b =
  match (a, b) with
  | None, None -> ()
  | Some _, None | None, Some _ ->
      invalid_arg
        (strf "%s: one observation states an area and the other does not" fn)
  | Some a, Some b ->
      let x = payload a and y = Quantity.value (Quantity.unit a) b in
      Nx.check
        Nx.Ptree.(pair tensor tensor)
        (Nx.equal x y) (x, y)
        (fun i (x, y) ->
          Invalid_argument
            (Format.asprintf
               "%s: the areas differ%s: %s in the first and %s in the second"
               fn
               (if Array.length i = 0 then "" else " at " ^ Grid.index_text i)
               (Transform.item x) (Transform.item y)))

let combine fn op a b =
  Grid.require fn a.grid b.grid;
  require_areas fn a.area b.area;
  let variance =
    match (a.variance, b.variance) with
    | Some va, Some vb -> Some (Quantity.add va vb)
    | _ -> None
  in
  let valid =
    match (a.valid, b.valid) with
    | None, None -> None
    | Some m, None | None, Some m -> Some (Nx.cast Nx.bool m)
    | Some m, Some n -> Some (Nx.logical_and (Nx.cast Nx.bool m) (Nx.cast Nx.bool n))
  in
  let o = { a with data = op a.data b.data; variance } in
  match valid with
  | None -> o
  | Some m -> zero_all m { o with valid = Some (Nx.cast Nx.bit m) }

let add a b = combine "Observation.add" Quantity.add a b
let sub a b = combine "Observation.sub" Quantity.sub a b

let scale k o =
  let variance = Option.map (fun v -> Quantity.mul v (Quantity.mul k k)) o.variance in
  let o = { o with data = Quantity.mul k o.data; variance } in
  match o.valid with None -> o | Some m -> zero_all (Nx.cast Nx.bool m) o

(* Windows *)

let component = Transform.component

(* [cell_index rel sizes] is the cells [rel + i] for each [i] below [sizes],
   one int64 index per axis, each [[...; sizes]] with 1 on the other axes. *)
let cell_index rel sizes =
  let n = Array.length sizes in
  let batch = Array.sub (Nx.shape rel) 0 (Nx.ndim rel - 1) in
  List.init n (fun k ->
      let at =
        Nx.reshape (Array.append batch (Array.make n 1)) (component rel k)
      in
      let along = Array.init n (fun i -> if i = k then sizes.(k) else 1) in
      let index = Nx.arange_at (Nx.beside rel) Nx.int64 0 sizes.(k) 1 in
      Nx.add at (Nx.reshape along index))

let all_of = function
  | [] -> invalid_arg "Observation: no axis"
  | m :: ms -> List.fold_left Nx.logical_and m ms

(* [gather rel sizes x] is the block of [sizes] cells of [x], [[...; S]], from
   the cell [rel], [[...; n]]: [[batch; sizes]] for the batch axes of [x] and
   [rel] broadcast, zero beyond [x]'s cells, and where the block is within
   them. *)
let gather rel sizes x =
  let s = Nx.shape x in
  let n = Array.length sizes and nd = Array.length s in
  let full = Array.sub s (nd - n) n in
  let index = cell_index rel sizes in
  let inside =
    all_of
      (List.mapi
         (fun k i ->
           Nx.logical_and (Nx.greater_equal_s i 0L)
             (Nx.less_s i (Int64.of_int full.(k))))
         index)
  in
  let flat =
    List.fold_left
      (fun acc (k, i) -> Nx.add (Nx.mul_s acc (Int64.of_int full.(k))) i)
      (List.hd index)
      (List.tl (List.mapi (fun k i -> (k, i)) index))
  in
  let flat = Nx.where inside flat (Nx.full_like flat (-1L)) in
  let count a = Array.fold_left ( * ) 1 a in
  let batch =
    Transform.broadcast
      (Array.sub s 0 (nd - n))
      (Array.sub (Nx.shape flat) 0 (Nx.ndim flat - n))
  in
  let xs =
    Nx.broadcast_to
      (Array.append batch [| count full |])
      (Nx.reshape (Array.append (Array.sub s 0 (nd - n)) [| count full |]) x)
  in
  let idx =
    Nx.reshape
      (Array.append batch [| count sizes |])
      (Nx.broadcast_to (Array.append batch sizes) flat)
  in
  let out =
    Nx.reshape (Array.append batch sizes) (Nx.take_along_axis ~axis:(-1) idx xs)
  in
  (out, inside)

let slice_window rel grid' o =
  let gs = Grid.shape o.grid in
  let s = Grid.shape grid' in
  let take q = Quantity.map (fun x -> fst (gather rel s x)) q in
  let area_take a = if per_cell (Nx.shape (payload a)) gs then take a else a in
  let valid =
    match o.valid with
    | Some m -> fst (gather rel s (Nx.cast Nx.bool m))
    | None -> snd (gather rel s (Nx.zeros_at (Nx.beside rel) Nx.bool gs))
  in
  {
    grid = grid';
    data = take o.data;
    variance = Option.map take o.variance;
    valid = Some (Nx.cast Nx.bit valid);
    area = Option.map area_take o.area;
  }

let window ~start ~shape o =
  let grid' = Grid.window ~start ~shape o.grid in
  slice_window start grid' o

let around x ~shape o =
  let grid' = Grid.around x ~shape o.grid in
  slice_window (Nx.sub grid'.Grid.start o.grid.Grid.start) grid' o

(* Sides *)

(* [moved fn axes ~cells o] is [o] on [g], one side of its grid whose cells
   are [o]'s on [axes], with the other side's cell axes moved before the
   batch axes, a view. *)
let moved fn g axes ~cells o =
  if Option.is_some o.area then
    invalid_arg
      (strf "%s: an area per cell of every axis has no share on one side" fn);
  let n = Array.length cells in
  let others = Transform.others axes n in
  let move x =
    let b = Nx.ndim x - n in
    Nx.transpose
      ~axes:
        (List.map (fun k -> b + k) (Array.to_list others)
        @ List.init b Fun.id
        @ List.map (fun k -> b + k) (Array.to_list axes))
      x
  in
  {
    grid = g;
    data = Quantity.map move o.data;
    variance = Option.map (Quantity.map move) o.variance;
    valid = Option.map move o.valid;
    area = None;
  }

let first o =
  let a, _, axes = Grid.factors "Observation.first" o.grid in
  moved "Observation.first" a axes ~cells:(Grid.shape o.grid) o

let second o =
  let _, b, axes = Grid.factors "Observation.second" o.grid in
  let n = Array.length (Grid.shape o.grid) in
  let cells = Grid.shape o.grid in
  moved "Observation.second" b (Transform.others axes n) ~cells o

(* Integrals *)

type ('e, 'd) integral = {
  value : ('e, 'd) q;
  variance : ('e, 'd) q option;
  area : ('e, 'd) q;
  coverage : (float, 'e, 'd) Nx.t;
}

(* Data whose unit has [Grid.cell⁻¹] are values per cell. *)
let per_cell_unit u =
  List.exists
    (function
      | Unit.Symbol { name = "cell"; scope = None }, -1, 1 -> true | _ -> false)
    (Unit.terms u)

(* [check_border ov g] raises where a border cell of [g]'s window lies in the
   base, the base continues beyond it, and the region is not exactly outside it:
   the window clipped the region. *)
let check_border outside (g : _ Grid.t) =
  let s = g.shape and n = Array.length g.shape in
  let cells = cell_index g.start s in
  let in_base =
    all_of
      (List.mapi
         (fun k c ->
           Nx.logical_and (Nx.greater_equal_s c 0L)
             (Nx.less_s c (Int64.of_int g.base.(k))))
         cells)
  in
  (* Cell [i] of the window on axis [k] is on its border where the base goes
     on past it. *)
  let continues =
    List.fold_left Nx.logical_or
      (Nx.zeros_like in_base)
      (List.mapi
         (fun k c ->
           let i =
             Nx.reshape
               (Array.init n (fun a -> if a = k then s.(k) else 1))
               (Nx.arange_at (Nx.beside c) Nx.int64 0 s.(k) 1)
           in
           let edge bound cond = Nx.logical_and (Nx.equal_s i bound) cond in
           Nx.logical_or
             (edge 0L (Nx.greater_s c 0L))
             (edge
                (Int64.of_int (s.(k) - 1))
                (Nx.less_s c (Int64.of_int (g.base.(k) - 1)))))
         cells)
  in
  let clipped =
    Nx.logical_and (Nx.logical_and in_base continues) (Nx.logical_not outside)
  in
  Nx.check Nx.Ptree.unit (Nx.logical_not clipped) () (fun i () ->
      let m = Array.length i in
      Invalid_argument
        (strf
           "Observation.integrate: the region reaches the border of its %s \
            window%s at sample %s; take a larger ~shape"
           (String.concat "x" (Array.to_list (Array.map string_of_int s)))
           (if m > n then " " ^ Transform.index (Array.sub i 0 (m - n)) else "")
           (Grid.index_text (Array.sub i (m - n) n))))

let integrate r o =
  let n = Array.length (Grid.shape o.grid) in
  let ov, own = Region.overlap r o.grid in
  check_border ov.outside o.grid;
  let valid =
    match o.valid with
    | Some m -> Nx.cast Nx.bool m
    | None -> Nx.scalar_at (Nx.beside (payload o.data)) Nx.bool true
  in
  let du = Quantity.unit o.data in
  let a_unit, a =
    if per_cell_unit du then (Grid.cell, None)
    else
      let a = match o.area with Some a -> a | None -> Grid.measure o.grid in
      (Quantity.unit a, Some (payload a))
  in
  let w = Nx.where valid ov.weight (Nx.zeros_like ov.weight) in
  let wa = match a with None -> w | Some a -> Nx.mul w a in
  let terms =
    [ Nx.mul (payload o.data) wa; wa; Nx.mul w (Nx.abs ov.area) ]
    @
    match o.variance with
    | None -> []
    | Some v -> [ Nx.mul (payload v) (Nx.square wa) ]
  in
  let sums =
    Nx_wide.hi
      (Nx_wide.sum
         ~axes:(List.init n (fun k -> k - n - 1))
         (Nx_wide.v (Transform.stack terms)))
  in
  let sum k = component sums k in
  (* The valid cells' overlap as a share of the region's own area, both in the
     region's plane: below 1 wherever the region reaches beyond the base, since
     a window that clips it inside the base raises. *)
  let empty = Nx.equal_s own 0. in
  let coverage =
    Nx.where empty (Nx.zeros_like own)
      (Nx.clamp ~max:1.
         (Nx.div (sum 2) (Nx.where empty (Nx.ones_like own) own)))
  in
  {
    value = Quantity.v Unit.(du * a_unit) (sum 0);
    area = Quantity.v a_unit (sum 1);
    variance =
      Option.map
        (fun v -> Quantity.v Unit.(Quantity.unit v * (a_unit ** 2)) (sum 3))
        o.variance;
    coverage;
  }

(* Structure *)

module W = Nx.Ptree.Walk

let walk c o =
  let grid = W.field c "grid" (W.structure (Grid.ptree ())) o.grid in
  let data = W.field c "data" Transform.quantity o.data in
  let variance =
    W.field c "variance" (W.option Transform.quantity) o.variance
  in
  let valid = W.field c "valid" (W.option W.tensor) o.valid in
  let area = W.field c "area" (W.option Transform.quantity) o.area in
  { grid; data; variance; valid; area }


let ptree () = Nx.Ptree.v walk
