(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Nx_array.Dtype
module L = Nx_array.Layout
module M = Nx_array.Move
module P = Nx_kernel.Prog
module E = P.E

let is_cut (type v s d) (x : (v, s, d) Value.t) =
  match Prim.at x with Some p -> Grid.is_cut (Devices.grid p) | None -> false

let applies (type r) (op : r Value.prim) =
  match op with
  | (Value.Reduce _ | Value.Contract _) when Prim.exists is_cut op ->
      let rule = Prim.rule op and found = ref false in
      Prim.iteri
        (fun i x ->
          match Prim.at x with
          | None -> ()
          | Some p ->
              let axes = Route.reduced rule i (Prim.rank x) in
              let cuts = Grid.cuts (Devices.grid p) in
              if Array.exists (fun (a, _) -> Array.mem a axes) cuts then
                found := true)
        op;
      !found
  | _ -> false

(* Reading whole *)

(* [x], operand [i] of [rule], placed whole along the axes [rule] reduces and
   where it lies along the others: where the route reads an operand that lies
   uncut along them. *)
let whole_along (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q)
    ~by rule i (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  let p = Prim.placement x in
  let g =
    Array.fold_left
      (fun g axis -> Grid.uncut g ~axis)
      (Devices.grid p)
      (Route.reduced rule i (Prim.rank x))
  in
  if Grid.equal g (Devices.grid p) then x
  else apply ~by (Value.Place (Devices.v ~by (Devices.set p) g, x))

let gathered (type r) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (op : r Value.prim) : r =
  let rule = Prim.rule op in
  apply ~by (Prim.mapi (fun i x -> whole_along apply ~by rule i x) op)

(* Reductions as values *)

type 'd one = One : ('d, 'a) Value.reduction -> 'd one

(* Reductions of one loop, and how to read their results as values. *)
type 'd packed =
  | Packed : ('d, 'r) Value.reductions * ('r -> 'd Value.any list) -> 'd packed

let values : type d a. (d, a) Value.reduction -> a -> d Value.any list =
  function
  | Monoid _ -> fun v -> [ Value.Any v ]
  | Moments _ -> fun (m, v) -> [ Value.Any m; Value.Any v ]
  | Arg _ -> fun (e, p) -> [ Value.Any e; Value.Any p ]

let rec pack : type d. d one list -> d packed = function
  | [] -> Packed (Value.[], fun () -> [])
  | One r :: rest ->
      let (Packed (rs, read)) = pack rest in
      Packed (Value.( :: ) (r, rs), fun (a, q) -> values r a @ read q)

let rec ones : type d r. (d, r) Value.reductions -> d one list = function
  | [] -> []
  | r :: rest -> One r :: ones rest

(* [vs], one value per result of [rs] in order, as [rs]'s results. *)
let rec results : type d r.
    (d, r) Value.reductions -> d Value.any list list -> r =
 fun rs vs ->
  match (rs, vs) with
  | [], [] -> ()
  | Monoid (_, _, dt) :: rest, [ v ] :: vs ->
      (Prim.expect dt v, results rest vs)
  | Moments (_, dt) :: rest, [ m; v ] :: vs ->
      ((Prim.expect dt m, Prim.expect dt v), results rest vs)
  | Arg (_, _, dt) :: rest, [ e; p ] :: vs ->
      ((Prim.expect dt e, Prim.expect D.Int64 p), results rest vs)
  | _ -> invalid_arg "Partial.results: one list of values per reduction"

(* What a reduction ranked by its extreme gives: the extreme alone, or the
   extreme and its position. *)
type gives = Extreme | Extreme_and_position

(* How one reduction over cut axes computes: by [Exact] partials, the
   reduction itself into the carrier, combined by a kind; by [Ranked]
   partials, an [Arg] each, combined into the first extreme; or [Gathered],
   over its loads read whole along its axes. The [D.any] is the dtype its
   result rounds into. *)
type 'd plan =
  | Exact of P.binary * 'd one * D.any
  | Ranked of Nx_kernel.extreme * 'd one * D.any * gives
  | Gathered of 'd one

(* The dtype a reduction of an integer or a boolean of [dt] accumulates in. *)
let carrier (D.Any dt as d) =
  match dt with
  | D.Int4 -> D.Any D.Int8
  | D.Uint4 -> D.Any D.Uint8
  | D.Bit -> D.Any D.Bool
  | _ -> d

(* The plan of [r], a reduction of [prog]'s outputs. *)
let plan (type d a) (prog : P.t) (r : (d, a) Value.reduction) : d plan =
  let term k = P.dtype prog (Iarray.get prog.outs k) in
  let exact (k : P.binary) m o dt =
    let (D.Any c) = carrier (term o) in
    Exact (k, One (Monoid (m, o, c)), D.Any dt)
  in
  let ranked (e : Nx_kernel.extreme) o dt gives =
    let (D.Any t) = term o in
    Ranked (e, One (Arg (e, o, t)), D.Any dt, gives)
  in
  let exactly o =
    let (D.Any t) = term o in
    match D.kind t with
    | D.Signed | D.Unsigned | D.Boolean -> true
    | D.Float | D.Complex -> false
  in
  match r with
  | Monoid (Sum, o, dt) when exactly o -> exact Add Sum o dt
  | Monoid (Prod, o, dt) when exactly o -> exact Mul Prod o dt
  | Monoid (Max, o, dt) when exactly o -> exact Maximum Max o dt
  | Monoid (Min, o, dt) when exactly o -> exact Minimum Min o dt
  | Monoid (Max, o, dt) -> ranked Max o dt Extreme
  | Monoid (Min, o, dt) -> ranked Min o dt Extreme
  | Arg (e, o, dt) -> ranked e o dt Extreme_and_position
  | Monoid ((Sum | Prod | Logsumexp), _, _) | Moments _ -> Gathered (One r)

(* Combinations *)

(* [x] stored in [dt]. *)
let cast (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (D.Any dt) (Value.Any x : d Value.any) : d Value.any =
  match D.equal_witness (Prim.dtype x) dt with
  | Some Type.Equal -> Value.Any x
  | None -> Value.Any (fst (apply ~by (Prim.op1 ~by Cast dt x)))

(* [vs] combined by the kind [k], in order. *)
let fold_exact (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (k : P.binary) (vs : d Value.any list) : d Value.any =
  match vs with
  | [] -> invalid_arg "Partial.fold_exact: no partial"
  | Value.Any first :: rest ->
      let dt = Prim.dtype first in
      let add acc y =
        fst (apply ~by (Prim.op2 ~by (Binary k) dt acc (Prim.expect dt y)))
      in
      Value.Any (List.fold_left add first rest)

(* Element [j] of the last axis of [pairs], an axis of two after [shape]'s. *)
let half (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by shape
    (pairs : (v, s, d) Value.t) j : (v, s, d) Value.t =
  let ranges =
    Array.append
      (Array.map (fun e -> { M.start = 0; count = e; step = 1 }) shape)
      [| { M.start = j; count = 1; step = 1 } |]
  in
  let p = apply ~by (Value.Move (Slice ranges, pairs)) in
  apply ~by (Value.Move (Reshape (Array.copy shape), p))

(* The values [x] orders by, [x] itself or a complex number's parts. *)
let keys (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (x : (v, s, d) Value.t) : d Value.any list =
  let part (type w r) (pt : (w, r) D.t) =
    let pairs = apply ~by (Value.Bitcast (pt, x)) in
    let shape = Prim.shape x in
    let part j = Value.Any (half apply ~by shape pairs j) in
    [ part 0; part 1 ]
  in
  match Prim.dtype x with
  | D.Complex64 -> part D.Float32
  | D.Complex128 -> part D.Float64
  | _ -> [ Value.Any x ]

(* Of two partials of the extreme [e], each its extreme and its position in
   the whole's order, the first extreme: a NaN before any number, the smaller
   position between two NaNs; else the extreme in Prog's order, [-0] below
   [+0] and a complex number by its parts, the smaller position between two
   of one bits. *)
let first_of (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (e : Nx_kernel.extreme) ((Value.Any ea : d Value.any), pa)
    ((Value.Any eb : d Value.any), pb) : d Value.any * d Value.any =
  let dt = Prim.dtype ea in
  let eb = Prim.expect dt (Value.Any eb) in
  let pa = Prim.expect D.Int64 pa and pb = Prim.expect D.Int64 pb in
  let ka = keys apply ~by ea and kb = keys apply ~by eb in
  let a = [ Value.Any ea; Value.Any pa ] in
  let b = [ Value.Any eb; Value.Any pb ] in
  let loads = Array.of_list (a @ ka @ b @ kb) in
  let ins =
    Iarray.of_array
      (Array.map (fun (Value.Any y) -> D.Any (Prim.dtype y)) loads)
  in
  let n = 2 + List.length ka in
  let (Value.Any k0) = List.hd ka in
  let kdt = Prim.dtype k0 in
  let floats =
    match D.kind kdt with
    | D.Float -> true
    | D.Complex | D.Signed | D.Unsigned | D.Boolean -> false
  in
  let before x y =
    if floats then E.(x < y || (x = y && recip x < recip y)) else E.(x < y)
  in
  let same x y =
    if floats then E.(x = y && recip x = recip y) else E.(x = y)
  in
  let key base j = E.input (base + 2 + j) kdt in
  let count = List.length ka in
  (* Whether keys of [u] are below those of [v], comparing from key [j]. *)
  let rec below u v j =
    let lt = before (key u j) (key v j) in
    if j = count - 1 then lt
    else
      let rest = below u v (j + 1) in
      E.(lt || (same (key u j) (key v j) && rest))
  in
  let rec tie j =
    let s = same (key 0 j) (key n j) in
    if j = count - 1 then s
    else
      let rest = tie (j + 1) in
      E.(s && rest)
  in
  let past = match e with Max -> below 0 n 0 | Min -> below n 0 0 in
  let pb = n + 1 in
  let earlier = E.(input pb D.Int64 < input 1 D.Int64) in
  let wins =
    match D.kind dt with
    | D.Float | D.Complex ->
        let nan i = E.(input i dt <> input i dt) in
        let yes = E.const D.Bool true in
        let numbers = E.(where (nan n) yes (past || (tie 0 && earlier))) in
        E.(where (nan 0) (nan n && earlier) numbers)
    | D.Signed | D.Unsigned | D.Boolean -> E.(past || (tie 0 && earlier))
  in
  let pick i t = E.any (E.where wins (E.input (n + i) t) (E.input i t)) in
  let prog = P.of_exprs ~ins [| pick 0 dt; pick 1 D.Int64 |] in
  let layout = L.contiguous (Prim.shape ea) in
  let loads = Array.map (fun (Value.Any y) -> Value.Plain y) loads in
  let x, (p, ()) =
    apply ~by (Value.Map { layout; prog; outs = Value.[ dt; D.Int64 ]; loads })
  in
  (Value.Any x, Value.Any p)

(* [pos], positions in C order of the reduced indices of a window [w] along
   reduced axes of extents [whole], as positions in C order of the whole's. *)
let renumber (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (w : M.range array) whole (pos : d Value.any) : d Value.any =
  let full =
    Array.for_all2 (fun (r : M.range) n -> r.start = 0 && r.count = n) w whole
  in
  if full then pos
  else
    let pos = Prim.expect D.Int64 pos in
    let int64 n = E.const D.Int64 (Int64.of_int n) in
    let p = E.input 0 D.Int64 in
    let global = ref (int64 0) and inner = ref 1 and outer = ref 1 in
    for i = Array.length w - 1 downto 0 do
      let digit = E.((p / int64 !inner) mod int64 w.(i).count) in
      global := E.(!global + ((digit + int64 w.(i).start) * int64 !outer));
      inner := !inner * w.(i).count;
      outer := !outer * whole.(i)
    done;
    let prog = P.of_exprs ~ins:[| D.Any D.Int64 |] [| E.any !global |] in
    let layout = L.contiguous (Prim.shape pos) in
    let loads = [| Value.Plain pos |] in
    let y, () =
      apply ~by (Value.Map { layout; prog; outs = Value.[ D.Int64 ]; loads })
    in
    Value.Any y

(* Reductions over cut axes *)

let full d = { M.start = 0; count = d; step = 1 }

(* The position of device [k] in [p]'s devices. *)
let position p k =
  Option.get (Array.find_index (( = ) k) (Grid.devices (Devices.grid p)))

(* [l] read on device [k], at [at], over its window [w] there. *)
let view_load (type d) ~by k (at : d Devices.placement) (l : d Value.load) w :
    d Value.load =
  let view (type v s) (x : (v, s, d) Value.t) : (v, s, d) Value.t =
    Value.Array { at; a = Place.view ~by x k w; dead = Value.Live }
  in
  match l with
  | Plain x -> Plain (view x)
  | Padded p -> Padded { p with x = view p.x }

(* A device of the results' placement: its id, its window of the results, its
   window of the loop, its loads' windows, and its window along the reduced
   axes, its tile. *)
type device = {
  k : int;
  out : M.range array;
  loop : M.range array;
  owns : M.range array array;
  tile : M.range array;
}

(* The devices of [p], the results' placement of a loop of [shape] reducing
   [axes] over [loads] of [rule]. A device's window of the loop is its loads'
   windows there, but whole along an axis they read whole. *)
let devices (type d) ~by (p : d Devices.placement) rule shape axes
    (loads : d Value.load array) =
  let xs = Array.map Prim.load_any loads in
  let ranks = Array.map (fun (Value.Any x) -> Prim.rank x) xs in
  let row i a =
    match (rule : Route.rule) with
    | Elementwise -> Route.Becomes a
    | Axes t | Into t -> t.(i).(a)
  in
  let out = Prim.reduced shape axes in
  let device j k =
    let owns =
      Array.map
        (fun (Value.Any x) ->
          let q = Prim.placement x in
          Devices.window ~by q (Prim.shape x) (position q k))
        xs
    in
    let loop =
      Array.mapi
        (fun a d ->
          match Array.find_index (fun r -> r > a) ranks with
          | None -> full d
          | Some i -> (
              match row i a with
              | Route.Whole -> full d
              | Becomes _ | Reduced -> owns.(i).(a)))
        shape
    in
    let tile = Array.map (fun a -> loop.(a)) axes in
    { k; out = Devices.window ~by p out j; loop; owns; tile }
  in
  Array.mapi device (Grid.devices (Devices.grid p))

(* The partial plans' reductions, packed. *)
let partial_reductions plans =
  pack
    (List.filter_map
       (function
         | Exact (_, o, _) | Ranked (_, o, _, _) -> Some o
         | Gathered _ -> None)
       plans)

(* Device [dv]'s partials of [plans] over its own windows, by [prs] read by
   [read]: per partial plan, the reduction's partial, or the extreme and its
   position in C order of the reduced extents [whole]. *)
let partials (type d r) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (set : d Devices.t) layout axes prog (prs : (d, r) Value.reductions)
    (read : r -> d Value.any list) plans whole (loads : d Value.load array) dv
    =
  let at = Devices.one set dv.k in
  let viewed =
    Array.mapi (fun i l -> view_load ~by dv.k at l dv.owns.(i)) loads
  in
  let local =
    Prim.within
      (Value.Reduce { layout; axes; prog; reductions = prs; loads = viewed })
      dv.loop
  in
  let rec split plans vs =
    match (plans, vs) with
    | [], [] -> []
    | Gathered _ :: plans, vs -> split plans vs
    | Exact _ :: plans, v :: vs -> [ v ] :: split plans vs
    | Ranked _ :: plans, e :: q :: vs ->
        [ e; renumber apply ~by dv.tile whole q ] :: split plans vs
    | _ -> invalid_arg "Partial.partials: one partial per plan"
  in
  split plans (read (apply ~by local))

(* The partial plans' results from [per_tile], each tile's partials in tile
   order, combined in that order and rounded once. *)
let combined (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by plans
    (per_tile : d Value.any list list list) =
  let nth i = List.map (fun t -> List.nth t i) per_tile in
  let pair = function
    | [ x; q ] -> (x, q)
    | _ -> invalid_arg "Partial.combined: an extreme and its position"
  in
  let rec go plans i =
    match plans with
    | [] -> []
    | Gathered _ :: plans -> go plans i
    | Exact (k, _, dt) :: plans ->
        let v = fold_exact apply ~by k (List.map List.hd (nth i)) in
        [ cast apply ~by dt v ] :: go plans (i + 1)
    | Ranked (e, _, dt, gives) :: plans -> (
        let pairs = List.map pair (nth i) in
        let x, q =
          List.fold_left (first_of apply ~by e) (List.hd pairs) (List.tl pairs)
        in
        let x = cast apply ~by dt x in
        match gives with
        | Extreme -> [ x ] :: go plans (i + 1)
        | Extreme_and_position -> [ x; q ] :: go plans (i + 1))
  in
  go plans 0

(* Device [j]'s results: the partials of each tile its group holds, its own or
   the first holder's moved to it, combined in tile order. A group is the
   devices that hold one window of the results and some terms. *)
let final (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (set : d Devices.t) plans devices
    (computed : d Value.any list list option array) j =
  let dv = devices.(j) in
  let group =
    List.filter
      (fun i -> devices.(i).out = dv.out && Option.is_some computed.(i))
      (List.init (Array.length devices) Fun.id)
  in
  let tiles =
    List.sort_uniq compare (List.map (fun i -> devices.(i).tile) group)
  in
  let here = Devices.one set dv.k in
  let move (Value.Any v) = Value.Any (apply ~by (Value.Place (here, v))) in
  let taken t =
    match computed.(j) with
    | Some own when dv.tile = t -> own
    | Some _ | None ->
        let src = List.find (fun i -> devices.(i).tile = t) group in
        List.map (List.map move) (Option.get computed.(src))
  in
  combined apply ~by plans (List.map taken tiles)

(* The values of partial plan [i] at [p], of shape [out], from [finals], each
   device's results in [p]'s order. *)
let at_placement (type d) (p : d Devices.placement) out
    (finals : d Value.any list list array) i =
  let value v (Value.Any x0) =
    let dt = Prim.dtype x0 in
    let array f =
      let x = Prim.expect dt (List.nth (List.nth f i) v) in
      Iarray.get (Prim.arrays x) 0
    in
    let arrays = Iarray.of_array (Array.map array finals) in
    Value.Any (Prim.of_arrays p ~shape:out arrays)
  in
  List.mapi value (List.nth finals.(0) i)

(* The number of results of [o]. *)
let width (type d) (One r : d one) =
  match r with Monoid _ -> 1 | Moments _ | Arg _ -> 2

(* [vs] cut into lists of [widths]. *)
let rec cut widths vs =
  match widths with
  | [] -> []
  | w :: widths ->
      List.filteri (fun i _ -> i < w) vs
      :: cut widths (List.filteri (fun i _ -> i >= w) vs)

(* The gathered plans' values, from one reduction over [loads] read whole
   along the axes it reduces. *)
let gathered_values (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q)
    ~by rule layout axes prog plans (loads : d Value.load array) =
  match List.filter_map (function Gathered o -> Some o | _ -> None) plans with
  | [] -> []
  | os ->
      let (Packed (rs, read)) = pack os in
      let whole i (l : d Value.load) : d Value.load =
        match l with
        | Plain x -> Plain (whole_along apply ~by rule i x)
        | Padded q -> Padded { q with x = whole_along apply ~by rule i q.x }
      in
      let loads = Array.mapi whole loads in
      let op = Value.Reduce { layout; axes; prog; reductions = rs; loads } in
      cut (List.map width os) (read (apply ~by op))

let reduce (type d r) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (op : r Value.prim) layout axes prog (rs : (d, r) Value.reductions)
    (loads : d Value.load array) : r =
  let shape = L.shape layout in
  let plans = List.map (fun (One r) -> plan prog r) (ones rs) in
  let partial = function Exact _ | Ranked _ -> true | Gathered _ -> false in
  if
    Array.exists (fun a -> shape.(a) = 0) axes
    || not (List.exists partial plans)
  then gathered apply ~by op
  else
    let p = (Option.get (Prim.loads_route ~by op layout loads)).result in
    let set = Devices.set p and rule = Prim.rule op in
    let devices = devices ~by p rule shape axes loads in
    let whole = Array.map (fun a -> shape.(a)) axes in
    let (Packed (prs, read)) = partial_reductions plans in
    let computed =
      Array.map
        (fun dv ->
          if Array.exists (fun (r : M.range) -> r.count = 0) dv.tile then None
          else
            Some
              (partials apply ~by set layout axes prog prs read plans whole
                 loads dv))
        devices
    in
    let finals =
      Array.init (Array.length devices)
        (final apply ~by set plans devices computed)
    in
    let gathered =
      gathered_values apply ~by rule layout axes prog plans loads
    in
    let out = Prim.reduced shape axes in
    let rec assemble plans i g =
      match plans with
      | [] -> []
      | Gathered _ :: rest -> List.nth gathered g :: assemble rest i (g + 1)
      | (Exact _ | Ranked _) :: rest ->
          at_placement p out finals i :: assemble rest (i + 1) g
    in
    results rs (assemble plans 0 0)

let run (type r) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (op : r Value.prim) : r =
  match op with
  | Value.Reduce { layout; axes; prog; reductions; loads } ->
      reduce apply ~by op layout axes prog reductions loads
  | _ -> gathered apply ~by op
