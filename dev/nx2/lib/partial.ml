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

(* A float or complex [Sum] or [Prod] of output [o], of the dtype [term], into
   [into]. Its units reduce into [carrier], [float32] for a narrower float, and
   its result rounds into [rounds]: [into], or for a complex sum, which reads
   each complex number as two floats of [part] along a last axis, its part. *)
type lane = {
  monoid : Nx_kernel.monoid;
  o : int;
  term : D.any;
  into : D.any;
  part : D.any option;
  carrier : D.any;
  rounds : D.any;
}

let float_part (D.Any dt) =
  match dt with
  | D.Complex64 -> D.Any D.Float32
  | D.Complex128 -> D.Any D.Float64
  | _ -> D.Any dt

let lane monoid o (D.Any t as term) into =
  let base =
    { monoid; o; term; into; part = None; carrier = term; rounds = into }
  in
  match (D.kind t, monoid) with
  | D.Complex, Sum ->
      let part = float_part term in
      { base with part = Some part; carrier = part; rounds = float_part into }
  | D.Float, _ when D.bits t < 32 -> { base with carrier = D.Any D.Float32 }
  | _ -> base

(* How one reduction over cut axes computes: by [Exact] partials, the
   reduction itself into the carrier, combined by a kind; by [Ranked]
   partials, an [Arg] each, combined into the first extreme; by [Lane]
   partials, lane order's nodes; or [Expanded], by its expansion, whose
   reductions compute so. The [D.any] is the dtype its result rounds into. *)
type 'd plan =
  | Exact of P.binary * 'd one * D.any
  | Ranked of Nx_kernel.extreme * 'd one * D.any * gives
  | Lane of lane
  | Expanded of 'd one

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
  (* A complex sum adds each part as a float sum, into complex numbers. *)
  let lanes m o dt =
    let (D.Any t) = term o in
    match D.kind t with
    | D.Float -> true
    | D.Complex -> m = Nx_kernel.Prod || D.is D.Complex dt
    | D.Signed | D.Unsigned | D.Boolean -> false
  in
  match r with
  | Monoid (Sum, o, dt) when exactly o -> exact Add Sum o dt
  | Monoid (Prod, o, dt) when exactly o -> exact Mul Prod o dt
  | Monoid (Max, o, dt) when exactly o -> exact Maximum Max o dt
  | Monoid (Min, o, dt) when exactly o -> exact Minimum Min o dt
  | Monoid (Max, o, dt) -> ranked Max o dt Extreme
  | Monoid (Min, o, dt) -> ranked Min o dt Extreme
  | Arg (e, o, dt) -> ranked e o dt Extreme_and_position
  | Monoid (((Sum | Prod) as m), o, dt) when lanes m o dt ->
      Lane (lane m o (term o) (D.Any dt))
  | Monoid ((Sum | Prod | Logsumexp), _, _) | Moments _ -> Expanded (One r)

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
         | Lane _ | Expanded _ -> None)
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
    | (Lane _ | Expanded _) :: plans, vs -> split plans vs
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
    | (Lane _ | Expanded _) :: plans -> go plans i
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

(* The expanded plans' values, from the expansion of one reduction of them: its
   own reductions compute through [apply], over cut axes by partials. *)
let expanded_values (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    layout axes prog plans (loads : d Value.load array) =
  match List.filter_map (function Expanded o -> Some o | _ -> None) plans with
  | [] -> []
  | os ->
      let (Packed (rs, read)) = pack os in
      let op = Value.Reduce { layout; axes; prog; reductions = rs; loads } in
      (* No such reduction is core: each has an expansion. *)
      cut (List.map width os) (read (Option.get (Expand.run apply ~by op)))

(* Lane order *)

(* The terms of a block of lane order ({!Nx_kernel.section-orders}). *)
let block = 1024
let numel = Array.fold_left ( * ) 1
let counts w = Array.map (fun (r : M.range) -> r.count) w

(* A run of a tile's terms: the number [g] of its first in C order of the
   whole's reduced indices and [l] in its tile's, its count [len], and [owner],
   its tile's position among its group's. *)
type run = { g : int; l : int; len : int; owner : int }

(* The runs of [tile], at position [t], a window of the reduced extents [whole],
   in increasing number: one per index of the reduced axes before the last that
   [tile] does not hold whole. *)
let runs whole t (tile : M.range array) =
  let m = Array.length whole in
  let stride = Array.make m 1 in
  for a = m - 2 downto 0 do
    stride.(a) <- stride.(a + 1) * whole.(a + 1)
  done;
  let c = ref (m - 1) in
  while !c >= 0 && tile.(!c).count = whole.(!c) do
    decr c
  done;
  if !c < 0 then [ { g = 0; l = 0; len = numel whole; owner = t } ]
  else
    let c = !c in
    let len = tile.(c).count * stride.(c) in
    let outer = numel (Array.sub (counts tile) 0 c) in
    List.init outer (fun r ->
        let g = ref (tile.(c).start * stride.(c)) and q = ref r in
        for a = c - 1 downto 0 do
          g := !g + ((tile.(a).start + (!q mod tile.(a).count)) * stride.(a));
          q := !q / tile.(a).count
        done;
        { g = !g; l = r * len; len; owner = t })

(* Lane order's tree over a group's terms. A leaf is a unit, terms [lo, hi) that
   one device reduces: tile [dest]'s, which holds all of them or, for a block
   that window edges split, most; [parts] are the runs that hold them, cut to
   them. *)
type tree = Leaf of int | Node of tree * tree
type leaf = { lo : int; hi : int; dest : int; parts : run list }

(* The tree of [n] terms whose runs, of [tiles] tiles, are [runs], in increasing
   number, and its leaves in term order. A node of blocks [b, b + count) has the
   value of a lane-order reduction of exactly its terms: it is a leaf where one
   tile holds them all or it is one block. *)
let tree n tiles (runs : run array) =
  let find t =
    let lo = ref 0 and hi = ref (Array.length runs - 1) in
    while !lo < !hi do
      let mid = (!lo + !hi + 1) / 2 in
      if runs.(mid).g <= t then lo := mid else hi := mid - 1
    done;
    !lo
  in
  let clip lo hi r =
    let a = Int.max lo r.g and b = Int.min hi (r.g + r.len) in
    { r with g = a; l = r.l + a - r.g; len = b - a }
  in
  let leaves = ref [] and count = ref 0 in
  let leaf lf =
    leaves := lf :: !leaves;
    incr count;
    Leaf (!count - 1)
  in
  let rec walk b blocks =
    let lo = b * block and hi = Int.min n ((b + blocks) * block) in
    let s = find lo in
    let r = runs.(s) in
    if r.g + r.len >= hi then
      leaf { lo; hi; dest = r.owner; parts = [ clip lo hi r ] }
    else if blocks = 1 then begin
      let parts = ref [] and i = ref s in
      while !i < Array.length runs && runs.(!i).g < hi do
        parts := clip lo hi runs.(!i) :: !parts;
        incr i
      done;
      let held = Array.make tiles 0 in
      List.iter (fun p -> held.(p.owner) <- held.(p.owner) + p.len) !parts;
      let dest = ref 0 in
      Array.iteri (fun t k -> if k > held.(!dest) then dest := t) held;
      leaf { lo; hi; dest = !dest; parts = List.rev !parts }
    end
    else
      (* The left part holds the largest power of two of blocks below. *)
      let h = ref 1 in
      while 2 * !h < blocks do
        h := 2 * !h
      done;
      let left = walk b !h in
      let right = walk (b + !h) (blocks - !h) in
      Node (left, right)
  in
  let t = walk 0 ((n + block - 1) / block) in
  (t, Array.of_list (List.rev !leaves))

(* A unit's values: its terms reduced into the carrier [s], whether a term is a
   NaN [f], and reduced into the result's dtype [q], [s] itself where that is
   the carrier. *)
type 'd unit_value =
  | Unit : {
      s : ('v, 's, 'd) Value.t;
      f : (bool, D.bool_elt, 'd) Value.t;
      q : ('w, 'r, 'd) Value.t;
    }
      -> 'd unit_value

let same (D.Any a) (D.Any b) = D.equal a b

(* [prog] with output [o] and whether it is a NaN as its outputs. *)
let flagged (prog : P.t) o =
  let out = Iarray.get prog.outs o in
  let nodes =
    Iarray.append prog.nodes [| P.Op2 (Compare Not_equal, out, out) |]
  in
  P.v ~ins:prog.ins nodes ~outs:[| out; Iarray.length prog.nodes |]

(* [x] of [shape]. *)
let reshape (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by shape
    (Value.Any x : d Value.any) : d Value.any =
  if Prim.has_shape x shape then Value.Any x
  else Value.Any (apply ~by (Value.Move (Reshape shape, x)))

(* A unit's values from [op] of a loop of [layout] along [axes] of the program
   [flagged] gives over [loads], within [w] of the loop where given, each of
   shape [rows], one per output and part. *)
let unit_value (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by ln
    ~layout ~axes prog (loads : d Value.load array) w rows : d unit_value =
  let (D.Any c) = ln.carrier in
  let (D.Any q) = ln.rounds in
  let m = ln.monoid in
  let separate = not (same ln.carrier ln.rounds) in
  let (Packed (rs, read)) =
    pack
      ([ One (Monoid (m, 0, c)); One (Monoid (Max, 1, D.Bool)) ]
      @ if separate then [ One (Monoid (m, 0, q)) ] else [])
  in
  let op = Value.Reduce { layout; axes; prog; reductions = rs; loads } in
  let op = match w with None -> op | Some w -> Prim.within op w in
  let flat v = reshape apply ~by rows v in
  match List.map flat (read (apply ~by op)) with
  | [ s; f ] ->
      let s = Prim.expect c s in
      Unit { s; f = Prim.expect D.Bool f; q = s }
  | [ s; f; v ] ->
      Unit
        { s = Prim.expect c s; f = Prim.expect D.Bool f; q = Prim.expect q v }
  | _ -> invalid_arg "Partial.unit_value: two or three results"

(* Elements [start, start + count) along axis 1 of [x]. *)
let along (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (Value.Any x : d Value.any) start count : d Value.any =
  let r = Array.map full (Prim.shape x) in
  r.(1) <- { M.start; count; step = 1 };
  Value.Any (apply ~by (Value.Move (Slice r, x)))

(* [x] on [here]. *)
let moved (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (here : d Devices.placement) (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  if Devices.equal (Prim.placement x) here then x
  else apply ~by (Value.Place (here, x))

(* Output [o] of [prog] over device [dv]'s windows of [loads], laid out as
   [outputs; terms], the window's axes off [axes] then those along them, each in
   C order; a complex sum's as floats along a last axis of two. *)
let staged (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by ln
    (set : d Devices.t) layout axes (prog : P.t) (loads : d Value.load array) dv
    : d Value.any =
  let at = Devices.one set dv.k in
  let viewed =
    Array.mapi (fun i l -> view_load ~by dv.k at l dv.owns.(i)) loads
  in
  let terms (type v s) (term : (v, s) D.t) : (v, s, d) Value.t =
    let mapped () =
      let out = Iarray.get prog.outs ln.o in
      let one = P.v ~ins:prog.ins prog.nodes ~outs:[| out |] in
      let map =
        Value.Map { layout; prog = one; outs = Value.[ term ]; loads = viewed }
      in
      fst (apply ~by (Prim.within map dv.loop))
    in
    match viewed with
    | [| Plain x |] when P.is_identity prog -> (
        match D.equal_witness (Prim.dtype x) term with
        | Some Type.Equal -> x
        | None -> mapped ())
    | _ -> mapped ()
  in
  let (D.Any term) = ln.term in
  let x = terms term in
  let rank = Array.length dv.loop in
  let kept =
    List.filter (fun a -> not (Array.mem a axes)) (List.init rank Fun.id)
  in
  let order = Array.append (Array.of_list kept) axes in
  let x =
    if Array.for_all2 ( = ) order (Array.init rank Fun.id) then x
    else apply ~by (Value.Move (Permute order, x))
  in
  let extents = counts dv.loop in
  let terms = numel (Array.map (fun a -> extents.(a)) axes) in
  let (Value.Any x) =
    reshape apply ~by [| numel extents / terms; terms |] (Value.Any x)
  in
  match ln.part with
  | None -> Value.Any x
  | Some (D.Any pt) -> Value.Any (apply ~by (Value.Bitcast (pt, x)))

(* The shape of a unit's values on device [dv]: one per output, then a part's
   axis of two for a complex sum. *)
let rows ln dv axes =
  let extents = counts dv.loop in
  let outputs = numel extents / numel (Array.map (fun a -> extents.(a)) axes) in
  match ln.part with None -> [| outputs |] | Some _ -> [| outputs; 2 |]

(* Two units' values combined, the earlier [a] first: their reductions by the
   plan's monoid, whether either has a NaN term, and the earlier's value in the
   result's dtype where it has one, else the later's. *)
let node (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by ln
    (Unit a : d unit_value) (Unit b : d unit_value) : d unit_value =
  let c = Prim.dtype a.s and qt = Prim.dtype a.q in
  let sb = Prim.expect c (Value.Any b.s)
  and qb = Prim.expect qt (Value.Any b.q) in
  let k : P.binary = match ln.monoid with Prod -> Mul | _ -> Add in
  let ins : D.any iarray =
    [| D.Any c; D.Any D.Bool; D.Any qt; D.Any c; D.Any D.Bool; D.Any qt |]
  in
  let fa = E.input 1 D.Bool in
  let prog =
    P.of_exprs ~ins
      [|
        E.any (E.binary k (E.input 0 c) (E.input 3 c));
        E.any E.(fa || input 4 D.Bool);
        E.any (E.where fa (E.input 2 qt) (E.input 5 qt));
      |]
  in
  let layout = L.contiguous (Prim.shape a.s) in
  let loads =
    Value.[| Plain a.s; Plain a.f; Plain a.q; Plain sb; Plain b.f; Plain qb |]
  in
  let s, (f, (q, ())) =
    apply ~by
      (Value.Map { layout; prog; outs = Value.[ c; D.Bool; qt ]; loads })
  in
  Unit { s; f; q }

(* Operand [0], of [c], rounded into [qt]. *)
let rounded (type v s w r) (c : (v, s) D.t) (qt : (w, r) D.t) : (w, r) E.t =
  match D.equal_witness c qt with
  | Some Type.Equal -> E.input 0 c
  | None -> E.cast qt (E.input 0 c)

(* [s], of [qt], but a complex product with a NaN part, which is the quiet NaN
   in each part. *)
let quiet (type w r) ln (qt : (w, r) D.t) (s : (w, r) E.t) : (w, r) E.t =
  match (D.kind qt, ln.monoid) with
  | D.Complex, Prod ->
      let nan = Int64.float_of_bits 0x7FF8_0000_0000_0000L in
      E.(where (s <> s) (const qt { Complex.re = nan; im = nan }) s)
  | _ -> s

(* The result from the combined units [u]: the first unit with a NaN term
   reduced into the result's dtype where there is one, else [s] rounded into it;
   a complex product with a NaN part and no NaN term is the quiet NaN in each
   part, as the kernels make it. *)
let settle (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by ln
    (Unit u : d unit_value) : d Value.any =
  let c = Prim.dtype u.s and qt = Prim.dtype u.q in
  let s = quiet ln qt (rounded c qt) in
  let ins : D.any iarray = [| D.Any c; D.Any D.Bool; D.Any qt |] in
  let prog =
    P.of_exprs ~ins [| E.any (E.where (E.input 1 D.Bool) (E.input 2 qt) s) |]
  in
  let layout = L.contiguous (Prim.shape u.s) in
  let loads = Value.[| Plain u.s; Plain u.f; Plain u.q |] in
  let r, () =
    apply ~by (Value.Map { layout; prog; outs = Value.[ qt ]; loads })
  in
  Value.Any r

(* [u]'s values on [here]. *)
let unit_on (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by here
    (Unit u : d unit_value) : d unit_value =
  let s = moved apply ~by here u.s and f = moved apply ~by here u.f in
  match D.equal_witness (Prim.dtype u.q) (Prim.dtype u.s) with
  | Some Type.Equal when u.q == u.s -> Unit { s; f; q = s }
  | Some _ | None -> Unit { s; f; q = moved apply ~by here u.q }

(* A lane plan's results on [devices], each its window of the results: per
   group, the units of lane order's tree over the group's terms, each reduced on
   one device, their values moved to every device of the group and combined up
   the tree there. *)
let lane_results (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by ln
    (set : d Devices.t) layout axes prog (loads : d Value.load array)
    (devices : device array) whole : d Value.any array =
  let n = numel whole in
  let all = List.init (Array.length devices) Fun.id in
  let results = Array.make (Array.length devices) None in
  let group j =
    let members =
      List.filter (fun i -> devices.(i).out = devices.(j).out) all
    in
    let holders =
      List.filter (fun i -> numel (counts devices.(i).tile) > 0) members
    in
    let tiles =
      Array.of_list
        (List.sort_uniq compare (List.map (fun i -> devices.(i).tile) holders))
    in
    let holder =
      Array.map
        (fun t -> devices.(List.find (fun i -> devices.(i).tile = t) holders))
        tiles
    in
    let held =
      Array.of_list
        (List.sort
           (fun a b -> Int.compare a.g b.g)
           (List.concat (Array.to_list (Array.mapi (runs whole) tiles))))
    in
    let t, leaves = tree n (Array.length tiles) held in
    let terms =
      Array.map
        (fun dv -> lazy (staged apply ~by ln set layout axes prog loads dv))
        holder
    in
    let value lf =
      let dv = holder.(lf.dest) in
      let here = Devices.one set dv.k in
      let rows = rows ln dv axes in
      let of_terms (Value.Any x) =
        let layout = L.contiguous (Prim.shape x) in
        let prog =
          flagged (P.of_node ~ins:[| D.Any (Prim.dtype x) |] (In 0)) 0
        in
        unit_value apply ~by ln ~layout ~axes:[| 1 |] prog [| Value.Plain x |]
          None rows
      in
      match lf.parts with
      | [ p ] when p.l = 0 && p.len = numel (counts dv.tile) && ln.part = None
        ->
          (* The tile's terms are the unit's: reduce them where they lie. *)
          let viewed =
            Array.mapi (fun i l -> view_load ~by dv.k here l dv.owns.(i)) loads
          in
          unit_value apply ~by ln ~layout ~axes (flagged prog ln.o) viewed
            (Some dv.loop) rows
      | [ p ] ->
          of_terms (along apply ~by (Lazy.force terms.(p.owner)) p.l p.len)
      | parts ->
          (* Each tile's part of the block moves here once, then the runs are
             assembled in term order. *)
          let from t =
            let mine = List.filter (fun p -> p.owner = t) parts in
            let a = List.fold_left (fun a p -> Int.min a p.l) max_int mine in
            let b =
              List.fold_left (fun b p -> Int.max b (p.l + p.len)) 0 mine
            in
            let (Value.Any v) =
              along apply ~by (Lazy.force terms.(t)) a (b - a)
            in
            (a, Value.Any (moved apply ~by here v))
          in
          let owners =
            List.sort_uniq compare (List.map (fun p -> p.owner) parts)
          in
          let froms = List.map (fun t -> (t, from t)) owners in
          let (Value.Any x0) = snd (List.assoc lf.dest froms) in
          let dt = Prim.dtype x0 in
          let shape = Array.copy (Prim.shape x0) in
          shape.(1) <- lf.hi - lf.lo;
          let piece p =
            let a, v = List.assoc p.owner froms in
            let region = Array.map full shape in
            region.(1) <- { M.start = p.g - lf.lo; count = p.len; step = 1 };
            (region, Prim.expect dt (along apply ~by v (p.l - a) p.len))
          in
          let pieces = List.map piece parts in
          of_terms
            (Value.Any
               (apply ~by
                  (Value.Assemble
                     { dtype = dt; shape; fill = D.zero dt; pieces })))
    in
    let values = Array.map value leaves in
    let rec combine here = function
      | Leaf i -> unit_on apply ~by here values.(i)
      | Node (a, b) -> node apply ~by ln (combine here a) (combine here b)
    in
    List.iter
      (fun i ->
        let dv = devices.(i) in
        let here = Devices.one set dv.k in
        let r = settle apply ~by ln (combine here t) in
        let r =
          match (r, ln.into) with
          | Value.Any r, D.Any dt when Option.is_some ln.part ->
              Value.Any (apply ~by (Value.Bitcast (dt, r)))
          | r, _ -> r
        in
        results.(i) <- Some (reshape apply ~by (counts dv.out) r))
      members
  in
  List.iter (fun j -> if Option.is_none results.(j) then group j) all;
  Array.map Option.get results

let reduce (type d r) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (op : r Value.prim) layout axes prog (rs : (d, r) Value.reductions)
    (loads : d Value.load array) : r =
  let shape = L.shape layout in
  if Array.exists (fun a -> shape.(a) = 0) axes then gathered apply ~by op
  else
    let plans = List.map (fun (One r) -> plan prog r) (ones rs) in
    let p = (Option.get (Prim.loads_route ~by op layout loads)).result in
    let set = Devices.set p and rule = Prim.rule op in
    let devices = devices ~by p rule shape axes loads in
    let whole = Array.map (fun a -> shape.(a)) axes in
    let out = Prim.reduced shape axes in
    let exact = function
      | Exact _ | Ranked _ -> true
      | Lane _ | Expanded _ -> false
    in
    let finals =
      if not (List.exists exact plans) then [||]
      else
        let (Packed (prs, read)) = partial_reductions plans in
        let computed =
          Array.map
            (fun dv ->
              if Array.exists (fun (r : M.range) -> r.count = 0) dv.tile then
                None
              else
                Some
                  (partials apply ~by set layout axes prog prs read plans whole
                     loads dv))
            devices
        in
        Array.init (Array.length devices)
          (final apply ~by set plans devices computed)
    in
    let expanded = expanded_values apply ~by layout axes prog plans loads in
    let rec assemble plans i e =
      match plans with
      | [] -> []
      | Expanded _ :: rest -> List.nth expanded e :: assemble rest i (e + 1)
      | Lane ln :: rest ->
          let vs =
            lane_results apply ~by ln set layout axes prog loads devices whole
          in
          at_placement p out (Array.map (fun v -> [ [ v ] ]) vs) 0
          :: assemble rest i e
      | (Exact _ | Ranked _) :: rest ->
          at_placement p out finals i :: assemble rest (i + 1) e
    in
    results rs (assemble plans 0 0)

let run (type r) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (op : r Value.prim) : r =
  match op with
  | Value.Reduce { layout; axes; prog; reductions; loads } ->
      reduce apply ~by op layout axes prog reductions loads
  | _ -> gathered apply ~by op
