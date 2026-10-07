(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Quantity = Ymir_units.Quantity
module Unit = Ymir_units.Unit

let strf = Printf.sprintf

type 'e q = (float, 'e) Nx.t Quantity.t

type ('w, 'e) t = {
  grid : ('w, 'e) Grid.t;
  data : 'e q;
  variance : 'e q option;
  valid : Nx.bit_t option;
  area : 'e q option;
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

let constant s =
  let n = Array.length s in
  (n < 1 || s.(n - 1) = 1) && (n < 2 || s.(n - 2) = 1)

(* The unit of [g]'s measures: a steradian on the sky, the square of the plane's
   unit, which one cell's measure gives, in a plane. *)
let measure_unit (type w e) (g : (w, e) Grid.t) =
  match Grid.world g with
  | Transform.Sky _ -> Unit.steradian
  | Transform.Planar ->
      let one =
        Grid.window ~start:(Nx.zeros Nx.int64 [| 2 |]) ~shape:[| 1; 1 |] g
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
          || not (per_cell s gs || constant s)
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

(* Windows *)

let component = Transform.component

(* [cell_index rel h w] is the cells [rel + (i, j)] as rows [[...; h; 1]] and
   columns [[...; 1; w]], int64. *)
let cell_index rel h w =
  let batch = Array.sub (Nx.shape rel) 0 (Nx.ndim rel - 1) in
  let at k = Nx.reshape (Array.append batch [| 1; 1 |]) (component rel k) in
  let axis n = Nx.arange Nx.int64 0 n 1 in
  ( Nx.add (at 0) (Nx.reshape [| h; 1 |] (axis h)),
    Nx.add (at 1) (Nx.reshape [| 1; w |] (axis w)) )

(* [gather rel h w x] is the [h × w] block of [x], [[...; H; W]], from the cell
   [rel], [[...; 2]]: [[batch; h; w]] for the batch axes of [x] and [rel]
   broadcast, zero beyond [x]'s cells, and where the block is within them. *)
let gather rel h w x =
  let s = Nx.shape x in
  let nd = Array.length s in
  let hh = s.(nd - 2) and ww = s.(nd - 1) in
  let rows, cols = cell_index rel h w in
  let inside =
    Nx.(
      logical_and
        (logical_and (greater_equal_s rows 0L) (less_s rows (Int64.of_int hh)))
        (logical_and (greater_equal_s cols 0L) (less_s cols (Int64.of_int ww))))
  in
  let flat = Nx.add (Nx.mul_s rows (Int64.of_int ww)) cols in
  let flat = Nx.where inside flat (Nx.full_like flat (-1L)) in
  let batch =
    Transform.broadcast
      (Array.sub s 0 (nd - 2))
      (Array.sub (Nx.shape flat) 0 (Nx.ndim flat - 2))
  in
  let xs =
    Nx.broadcast_to
      (Array.append batch [| hh * ww |])
      (Nx.reshape (Array.append (Array.sub s 0 (nd - 2)) [| hh * ww |]) x)
  in
  let idx =
    Nx.reshape
      (Array.append batch [| h * w |])
      (Nx.broadcast_to (Array.append batch [| h; w |]) flat)
  in
  let out =
    Nx.reshape
      (Array.append batch [| h; w |])
      (Nx.take_along_axis ~axis:(-1) ~indices:idx xs)
  in
  (out, inside)

let slice_window rel grid' o =
  let gs = Grid.shape o.grid in
  let s = Grid.shape grid' in
  let h = s.(0) and w = s.(1) in
  let take q = Quantity.map (fun x -> fst (gather rel h w x)) q in
  let area_take a = if per_cell (Nx.shape (payload a)) gs then take a else a in
  let valid =
    match o.valid with
    | Some m -> fst (gather rel h w (Nx.cast Nx.bool m))
    | None -> snd (gather rel h w (Nx.zeros Nx.bool gs))
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

(* Integrals *)

type 'e integral = {
  value : 'e q;
  variance : 'e q option;
  area : 'e q;
  coverage : (float, 'e) Nx.t;
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
  let h = g.shape.(0) and w = g.shape.(1) in
  let rows, cols = cell_index g.start h w in
  let b0 = Int64.of_int g.base.(0) and b1 = Int64.of_int g.base.(1) in
  let in_base =
    Nx.(
      logical_and
        (logical_and (greater_equal_s rows 0L) (less_s rows b0))
        (logical_and (greater_equal_s cols 0L) (less_s cols b1)))
  in
  let index n = Nx.arange Nx.int64 0 n 1 in
  let i = Nx.reshape [| h; 1 |] (index h)
  and j = Nx.reshape [| 1; w |] (index w) in
  let edge at bound cond = Nx.logical_and (Nx.equal_s at bound) cond in
  let continues =
    Nx.(
      logical_or
        (logical_or
           (edge i 0L (greater_s rows 0L))
           (edge i (Int64.of_int (h - 1)) (less_s rows (Int64.sub b0 1L))))
        (logical_or
           (edge j 0L (greater_s cols 0L))
           (edge j (Int64.of_int (w - 1)) (less_s cols (Int64.sub b1 1L)))))
  in
  let clipped =
    Nx.logical_and (Nx.logical_and in_base continues) (Nx.logical_not outside)
  in
  Nx.check Nx.Ptree.unit (Nx.logical_not clipped) () (fun i () ->
      let n = Array.length i in
      Invalid_argument
        (strf
           "Observation.integrate: the region reaches the border of its %dx%d \
            window%s at sample (%d, %d); take a larger ~shape"
           h w
           (if n > 2 then " " ^ Transform.index (Array.sub i 0 (n - 2)) else "")
           i.(n - 2)
           i.(n - 1)))

let integrate r o =
  let ov, own = Region.overlap r o.grid in
  check_border ov.outside o.grid;
  let valid =
    match o.valid with
    | Some m -> Nx.cast Nx.bool m
    | None -> Nx.scalar Nx.bool true
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
      (Nx_wide.sum ~axes:[ -3; -2 ] (Nx_wide.v (Transform.stack terms)))
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

type ('w, 'e) observation = ('w, 'e) t

let ptree (type w e) () : (w, e) t Nx.Ptree.t =
  Nx.Ptree.instantiate
    (module struct
      type _ t = (w, e) observation

      let walk = walk
    end)
