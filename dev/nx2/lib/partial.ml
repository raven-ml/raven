(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module M = Nx_array.Move
module P = Nx_kernel.Prog
module E = P.E

let invalid_argf fmt = Format.kasprintf invalid_arg fmt
let full d = { M.start = 0; count = d; step = 1 }
let numel = Array.fold_left ( * ) 1
let counts w = Array.map (fun (r : M.range) -> r.count) w
let is_empty w = Array.exists (fun (r : M.range) -> r.count = 0) w

let floats (D.Any dt) =
  match D.kind dt with
  | D.Float | D.Complex -> true
  | D.Signed | D.Unsigned | D.Boolean -> false

let identity d = P.of_node ~ins:[| d |] (In 0)

let binary : Nx_kernel.monoid -> P.binary = function
  | Sum -> Add
  | Prod -> Mul
  | Max -> Maximum
  | Min -> Minimum

(* The position of device [k] among [p]'s. *)
let position p k =
  Option.get (Array.find_index (( = ) k) (Grid.devices (Devices.grid p)))

(* The window of operand [i], of [shape], that a device reads by [rule]: the
   window its window [w] of the results reads ({!Route.window}), but [tile f]
   along an axis of the folded factor [f]. An axis of blocks of [f], or of
   its fences, is whole: its reader takes the blocks a tile spans, or the
   fences that bound it. *)
let reads rule i shape (w : M.range array) tile =
  match (rule : Route.rule) with
  | Elementwise -> w
  | Axes t ->
      let window = Route.window rule ~operand:i shape w in
      Array.mapi
        (fun a r ->
          match t.(i).(a) with
          | Route.Reduced f -> tile f
          | Becomes _ | Scanned _ | Regroups _ | Blocks _ | Whole -> r)
        window

(* The tile of each device of [route]'s result, in order: its window along
   [axes] of [operand], of [shape], where [route] reads it. The route reads
   every operand that holds them alike along them. *)
let tiles ~by (route : unit Route.t) ~operand shape axes =
  let q = route.operands.(operand) in
  Array.map
    (fun k ->
      let w = Devices.window ~by q shape (position q k) in
      Array.map (fun a -> w.(a)) axes)
    (Grid.devices (Devices.grid route.result))

(* Contraction pieces *)

let rec gcd a b = if b = 0 then a else gcd b (a mod b)

(* The ranges [lo, hi) splits into, in order, each within one block of
   every length of [rs] or from one of their common edges to another:
   [lo, hi) itself where [rs] is empty or [lo, hi) holds no element. *)
let pieces rs lo hi =
  let l = List.fold_left (fun l r -> l / gcd l r * r) 1 rs in
  let rec from x =
    if x >= hi then []
    else
      let body = hi / l * l in
      let y =
        if x mod l = 0 && body > x then body
        else List.fold_left (fun y r -> Int.min y (((x / r) + 1) * r)) hi rs
      in
      { M.start = x; count = y - x; step = 1 } :: from y
  in
  if lo >= hi then [ { M.start = lo; count = 0; step = 1 } ] else from lo

(* The pieces of [tile], its ranges of the folded factors, each factor's
   range cut by {!pieces} at [runs], its scale blocks' lengths, in C
   order. *)
let tile_pieces runs (tile : M.range array) =
  let rec from p =
    if p = Array.length tile then [ [] ]
    else
      let rest = from (p + 1) and r = tile.(p) in
      List.concat_map
        (fun x -> List.map (fun u -> x :: u) rest)
        (pieces runs.(p) r.start (r.start + r.count))
  in
  List.map Array.of_list (from 0)

(* Operations *)

(* [op] with its results at [p], each device of [p] computing its window
   over its operands' windows ({!Exec.windows}), which each operand holds
   there: by {!Exec.local} where it maps operands that lie beside each
   other. An operation that creates its values computes at [p] too. The
   first result takes [into] where it is given. *)
let at : type r.
    by:string -> ?into:A.any -> unit Devices.placement -> r Value.prim -> r =
 fun ~by ?into p op ->
  if Option.is_none into && Exec.is_local op then Exec.local ~by op
  else
    let (Prim.Operands xs) = Prim.operands op in
    let operands =
      Array.of_list (List.map (fun (Value.Any x) -> Prim.placement x) xs)
    in
    let route = { Route.operands; result = p; compute = Windows } in
    Exec.windows ~by ?into route (Prim.layouts ~by op) op

(* [op] at [x]'s placement. *)
let on ~by x op = at ~by (Prim.placement x) op

(* The one array of [x], a value on one device. *)
let array (type v s d) (x : (v, s, d) Value.t) = Iarray.get (Prim.arrays x) 0

(* [a] moved by [mv], a view: [a] is a fresh array or a slice of one, whose
   layout expresses a slice and a reshape of its unit or whole axes. *)
let moved mv a =
  match A.move mv a with
  | Some v -> v
  | None -> invalid_arg "Partial: a slot's layout expresses its move"

(* Values *)

(* [x] of [shape], its cut axes keeping their numbers. *)
let reshape (type v s d) ~by shape (x : (v, s, d) Value.t) : (v, s, d) Value.t
    =
  if Prim.has_shape x shape then x
  else on ~by x (Value.Move (Reshape (Iarray.of_array shape), x))

(* Elements [start, start + count) of [x] along [axis], an axis no cut
   names. *)
let slice (type v s d) ~by axis start count (x : (v, s, d) Value.t) :
    (v, s, d) Value.t =
  let r = Array.map full (Prim.shape x) in
  if start = 0 && count = r.(axis).count then x
  else begin
    r.(axis) <- { M.start; count; step = 1 };
    on ~by x (Value.Move (Slice (Iarray.of_array r), x))
  end

(* Elements [start] down to [start - count + 1] of [x] along [axis], in that
   order. *)
let backwards (type v s d) ~by axis start count (x : (v, s, d) Value.t) :
    (v, s, d) Value.t =
  let r = Array.map full (Prim.shape x) in
  r.(axis) <- { M.start; count; step = -1 };
  on ~by x (Value.Move (Slice (Iarray.of_array r), x))

(* [x] of [shape], its axes of extent 1 stretched. *)
let spread (type v s d) ~by shape (x : (v, s, d) Value.t) : (v, s, d) Value.t
    =
  if Prim.has_shape x shape then x
  else on ~by x (Value.Move (Broadcast (Iarray.of_array shape), x))

let cast (type d) ~by (D.Any dt) (Value.Any x : d Value.any) : d Value.any =
  match D.equal_witness (Prim.dtype x) dt with
  | Some Type.Equal -> Value.Any x
  | None -> Value.Any (fst (on ~by x (Prim.op1 ~by Cast dt x)))

(* [x]'s window [w] from its array on its set's device [k], at [at]. *)
let view (type v s d) ~by k (at : d Devices.placement) w
    (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  Value.Array { at; a = Place.view ~by x k w; dead = Value.Live }

let view_load (type d) ~by k at w (l : d Value.load) : d Value.load =
  match l with
  | Plain x -> Plain (view ~by k at w x)
  | Padded p -> Padded { p with x = view ~by k at w p.x }

(* The map over [vs], of one shape, its inputs in order, to [exprs], with
   its results at [p]. *)
let map (type d) ~by ?into p (vs : d Value.any list) exprs : d Value.any list =
  let dtype (Value.Any x : d Value.any) = D.Any (Prim.dtype x) in
  let prog =
    P.of_exprs
      ~ins:(Iarray.of_list (List.map dtype vs))
      (Iarray.of_list exprs)
  in
  let dts = List.map (P.dtype prog) (Iarray.to_list prog.outs) in
  let (Prim.Outputs (outs, read)) = Prim.outputs dts in
  let (Value.Any x) = List.hd vs in
  let layout = L.contiguous (Iarray.of_array (Prim.shape x)) in
  let plain (Value.Any x : d Value.any) = Value.Plain x in
  let loads = Iarray.of_list (List.map plain vs) in
  read (at ~by ?into p (Value.Map { layout; prog; outs; loads; at = None }))

(* Output [o] of [prog], of [term], over [loads] in the window [w] of a
   loop of [layout], on [here]: the load itself where [prog] is its
   identity. *)
let terms (type d) ~by here layout (prog : P.t) o (D.Any term)
    (loads : d Value.load iarray) w : d Value.any =
  match loads with
  | [| Plain x |] when P.is_identity prog && D.equal (Prim.dtype x) term ->
      Value.Any x
  | _ ->
      let one = P.subst ~outs:[| o |] ~ins:prog.ins (fun _ -> None) prog in
      let map =
        Value.Map
          { layout; prog = one; outs = Value.[ term ]; loads; at = None }
      in
      Value.Any (fst (at ~by here (Prim.within map w)))

(* The mean of [n] terms from their sum [s], in the carrier, at [s]'s
   placement. *)
let mean ~by n s =
  let (Value.Any x) = s in
  let c = Prim.dtype x in
  let count = E.const c (D.of_float c (Float.of_int n)) in
  map ~by (Prim.placement x) [ s ] [ E.any (E.fdiv (E.input 0 c) count) ]

(* [Moments] into [dt] of [n] terms from their mean [m], in the carrier,
   and the sums [s1] of their deviations [d] from it and [s2] of [d²], at
   [s1]'s placement: [(Σ d² - Σ d (Σ d / n)) / n], [+0] where that is
   negative or [Σ d²] where that is [+∞]. A NaN mean makes every deviation,
   so the variance, a NaN. *)
let moments ~by (D.Any dt) n m s1 s2 =
  let (Value.Any x) = s1 in
  let c = Prim.dtype x in
  let m' = E.input 2 c and v1 = E.input 0 c and v2 = E.input 1 c in
  let count = E.const c (D.of_float c (Float.of_int n)) in
  let zero = E.const c (D.zero c)
  and inf = E.const c (D.of_float c Float.infinity) in
  let q = E.(fdiv (v2 - (v1 * fdiv v1 count)) count) in
  let v = E.(where (v2 = inf) v2 (where (q < zero) zero q)) in
  map ~by (Prim.placement x) [ s1; s2; m ]
    [ E.any (E.cast dt m'); E.any (E.cast dt v) ]

(* Positions *)

(* The values [x] orders by: itself, or a complex number's parts. *)
let keys (type v s d) ~by (x : (v, s, d) Value.t) : d Value.any list =
  let parts (type w r) (pt : (w, r) D.t) =
    let pairs = on ~by x (Value.Bitcast (pt, x)) in
    let shape = Prim.shape x in
    let half j =
      Value.Any (reshape ~by shape (slice ~by (Array.length shape) j 1 pairs))
    in
    [ half 0; half 1 ]
  in
  match Prim.dtype x with
  | D.Complex64 -> parts D.Float32
  | D.Complex128 -> parts D.Float64
  | _ -> [ Value.Any x ]

(* Of two ranked partials of the extreme [e], each its extreme and its
   position in the whole's order, the first extreme: a NaN before any
   number, the smaller position between two NaNs; else the extreme in
   Prog's order, [-0] below [+0] and a complex number by its parts, the
   smaller position between two of one bits. *)
let first_of ~by (e : Nx_kernel.extreme) a b =
  match (a, b) with
  | [ Value.Any ea; pa ], [ eb; pb ] ->
      let dt = Prim.dtype ea in
      let eb = Prim.expect dt eb in
      let ka = keys ~by ea and kb = keys ~by eb in
      let count = List.length ka in
      let n = 2 + count in
      let (Value.Any k0) = List.hd ka in
      let kdt = Prim.dtype k0 in
      let fl = floats (D.Any kdt) in
      let before x y =
        if fl then E.(x < y || (x = y && recip x < recip y)) else E.(x < y)
      in
      let same x y = if fl then E.(x = y && recip x = recip y) else E.(x = y) in
      let key base j = E.input (base + 2 + j) kdt in
      (* Whether keys of [u] are below those of [v], from key [j]. *)
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
      let at = n + 1 in
      let earlier = E.(input at D.Int64 < input 1 D.Int64) in
      let wins =
        match D.kind dt with
        | D.Float | D.Complex ->
            let nan i = E.(input i dt <> input i dt) in
            let numbers =
              E.(where (nan n) (const D.Bool true) (past || (tie 0 && earlier)))
            in
            E.(where (nan 0) (nan n && earlier) numbers)
        | D.Signed | D.Unsigned | D.Boolean -> E.(past || (tie 0 && earlier))
      in
      let pick i t = E.any (E.where wins (E.input (n + i) t) (E.input i t)) in
      map ~by (Prim.placement ea)
        ((Value.Any ea :: pa :: ka) @ (Value.Any eb :: pb :: kb))
        [ pick 0 dt; pick 1 D.Int64 ]
  | _ -> invalid_arg "Partial.first_of: an extreme and its position each"

(* [pos] shifted by [s]. *)
let shift ~by s pos =
  if s = 0 then pos
  else
    let (Value.Any x) = pos in
    let s = E.const D.Int64 (Int64.of_int s) in
    List.hd
      (map ~by (Prim.placement x) [ pos ] [ E.any E.(input 0 D.Int64 + s) ])

(* [pos], positions in C order of the folded indices of a window [w] of
   the folded extents [whole], as positions in C order of the whole's. *)
let renumber ~by (w : M.range array) whole pos =
  if
    Array.for_all2 (fun (r : M.range) n -> r.start = 0 && r.count = n) w whole
  then pos
  else
    let (Value.Any x) = pos in
    let int64 n = E.const D.Int64 (Int64.of_int n) in
    let p = E.input 0 D.Int64 in
    let global = ref (int64 0) and inner = ref 1 and outer = ref 1 in
    for i = Array.length w - 1 downto 0 do
      let digit = E.((p / int64 !inner) mod int64 w.(i).count) in
      global := E.(!global + ((digit + int64 w.(i).start) * int64 !outer));
      inner := !inner * w.(i).count;
      outer := !outer * whole.(i)
    done;
    List.hd (map ~by (Prim.placement x) [ pos ] [ E.any !global ])

(* Closings *)

(* Where a reduction's or a contraction's partials close: [result], the
   route's; [stacked], the placement of the partials stacked along a new
   leading axis per cut folded axis, of [held] elements, each cut over the
   grid axes that cut its folded axis, the others cut as [result] cuts
   them; [placed], the stack's after its move, whole along the leading
   axes, the grid axes that cut them moved to an axis the result keeps
   whole, where it has one of at least a group's devices; and [parts], the
   reduced stack's, [placed] without the leading axes. [held] is, for each
   cut folded axis in order, the number of its tiles that hold terms, which
   come first: the route computes by partials only where every folded axis
   has elements ({!Route.computed}). *)
type closing = {
  result : unit Devices.placement;
  stacked : unit Devices.placement;
  placed : unit Devices.placement;
  parts : unit Devices.placement;
  held : int list;
}

(* The closing of a fold over the folded [axes] of [operand], in any order,
   each of [extent a] elements, into results of [shape] at [route]'s
   result. *)
let closing ~by (route : unit Route.t) ~operand ~axes ~extent shape =
  let r = route.result in
  let like = Devices.grid route.operands.(operand) and g = Devices.grid r in
  let set = Devices.set r in
  let cut_tiles a =
    Array.find_map
      (fun (b, t) -> if a = b then Some t else None)
      (Grid.cuts like)
  in
  (* Sorted, so that the slots lie in C order of the tiles' first indices
     whatever the order of [axes], as a contraction's pairs give them. *)
  let cut =
    List.sort Int.compare
      (List.filter (fun a -> Option.is_some (cut_tiles a)) axes)
  in
  let tiles = List.map (fun a -> Option.get (cut_tiles a)) cut in
  let held = List.map2 (fun a t -> Int.min (extent a) t) cut tiles in
  let group = List.fold_left ( * ) 1 tiles in
  let lead = List.length cut in
  let cut_at axes axis g =
    match Grid.cut_like g ~like ~axes ~axis with
    | Ok g -> g
    | Error e -> invalid_argf "%s: Partial.closing: %s" by e
  in
  let up = Grid.map_axes (( + ) lead) g in
  let stacked =
    List.fold_left
      (fun g (i, a) -> cut_at [ a ] i g)
      up
      (List.mapi (fun i a -> (i, a)) cut)
  in
  let whole a = not (Array.exists (fun (b, _) -> a = b) (Grid.cuts g)) in
  let split =
    List.find_opt
      (fun a -> whole a && shape.(a) >= group)
      (List.init (Array.length shape) Fun.id)
  in
  let placed, parts =
    match split with
    | Some k ->
        let placed = cut_at cut (k + lead) up in
        let parts = Grid.map_axes (fun a -> a - lead) placed in
        (Devices.v ~by set placed, Devices.v ~by set parts)
    | None -> (Devices.v ~by set up, r)
  in
  { result = r; stacked = Devices.v ~by set stacked; placed; parts; held }

(* The slots of [c], in order: their indices along the leading axes, in C
   order. *)
let slots c =
  List.fold_right
    (fun h rest ->
      List.concat_map
        (fun i -> List.map (fun s -> i :: s) rest)
        (List.init h Fun.id))
    c.held [ [] ]

(* The stack at [c.stacked] of [dt], of the results of [shape] after its
   leading axes: the slot of the device at position [j], its window of the
   results, is [partial j], and a device whose slot holds no element, as
   one whose tile is empty, holds an empty array. *)
let stack (type v s d) ~by c (dt : (v, s) D.t) shape
    (partial : int -> (v, s) A.t) : (v, s, d) Value.t =
  let set = Devices.set c.stacked
  and ks = Grid.devices (Devices.grid c.stacked) in
  let shape = Array.append (Array.of_list c.held) shape in
  let slot j =
    let s = counts (Devices.window ~by c.stacked shape j) in
    if numel s = 0 then A.create (Devices.rig set ks.(j)) dt (Iarray.of_array s)
    else moved (M.Reshape (Iarray.of_array s)) (partial j)
  in
  Prim.of_arrays c.stacked ~shape (Iarray.init (Array.length ks) slot)

(* The reduction [r] at [c.parts] of [s], the stack at [c.stacked], over
   its leading axes, after [s] moves to [c.placed]. *)
let reduced (type v s d a) ~by c (r : (d, a) Value.reduction)
    (s : (v, s, d) Value.t) : a =
  let s = Place.value ~by c.placed s in
  let op =
    Value.Reduce
      {
        layout = L.contiguous (Iarray.of_array (Prim.shape s));
        axes = Iarray.init (List.length c.held) Fun.id;
        prog = identity (D.Any (Prim.dtype s));
        reductions = Value.[ r ];
        loads = [| Plain s |];
      }
  in
  fst (at ~by c.parts op)

(* [y], at [c.parts], at the route's result: gathered along the axis that
   took the leading axes' cut. *)
let gathered (type v s d) ~by c (y : (v, s, d) Value.t) : (v, s, d) Value.t =
  if c.parts == c.result then y else Place.value ~by c.result y

(* The slot at [index] of [x], a stack at [c.placed], at [c.parts]: a view
   of each device's array. *)
let slot (type v s d) ~by c index (x : (v, s, d) Value.t) : (v, s, d) Value.t
    =
  let lead = List.length index in
  let shape = Prim.shape x in
  let out = Array.sub shape lead (Array.length shape - lead) in
  let one j a =
    let w = counts (Devices.window ~by c.placed shape j) in
    let r = Array.map full w in
    List.iteri (fun i k -> r.(i) <- { M.start = k; count = 1; step = 1 }) index;
    let part = Array.sub w lead (Array.length out) in
    moved (Reshape (Iarray.of_array part)) (moved (Slice (Iarray.of_array r)) a)
  in
  Prim.of_arrays c.parts ~shape:out (Iarray.mapi one (Prim.arrays x))

(* Reductions *)

let reduce (type d r) ~by (route : unit Route.t) rule layout axes prog
    (rs : (d, r) Value.reductions) (loads : d Value.load iarray) : r =
  let shape = Iarray.to_array (L.shape layout) in
  let ax = Array.of_list (List.sort Int.compare (Iarray.to_list axes)) in
  let p = route.result in
  let set = Devices.set p and ks = Grid.devices (Devices.grid p) in
  let out = Prim.reduced shape axes in
  let whole = Array.map (fun a -> shape.(a)) ax in
  let outw j = Devices.window ~by p out j in
  let here j = Devices.one set ks.(j) in
  let tiles =
    let (Value.Any x) = Prim.load_any (Iarray.get loads 0) in
    tiles ~by route ~operand:0 (Prim.shape x) ax
  in
  let cl =
    closing ~by route ~operand:0 ~axes:(Array.to_list ax)
      ~extent:(fun a -> shape.(a))
      out
  in
  let along j a = tiles.(j).(Option.get (Array.find_index (( = ) a) ax)) in
  let loop j =
    let w = outw j in
    let before a =
      Array.fold_left (fun n b -> if b < a then n + 1 else n) 0 ax
    in
    Array.init (Array.length shape) (fun a ->
        if Array.mem a ax then along j a else w.(a - before a))
  in
  let viewed j =
    Iarray.mapi
      (fun i l ->
        let (Value.Any x) = Prim.load_any l in
        view_load ~by ks.(j) (here j)
          (reads rule i (Prim.shape x) (outw j) (along j))
          l)
      loads
  in
  let local j prog rs =
    let op =
      Value.Reduce { layout; axes; prog; reductions = rs; loads = viewed j }
    in
    at ~by (here j) (Prim.within op (loop j))
  in
  let term o = P.dtype prog (Iarray.get prog.outs o) in
  let one (type a) (r : (d, a) Value.reduction) : a =
    let (D.Any c) = Nx_kernel.carrier (term (snd (Prim.reduction r))) in
    match r with
    | Monoid (m, o, dt) ->
        let own j = array (fst (local j prog Value.[ Monoid (m, o, c) ])) in
        let s = stack ~by cl c out own in
        gathered ~by cl (reduced ~by cl (Monoid (m, 0, dt)) s)
    | Logsumexp (o, dt) ->
        let own j = array (fst (local j prog Value.[ Logsumexp (o, c) ])) in
        let s = stack ~by cl c out own in
        gathered ~by cl (reduced ~by cl (Logsumexp (0, dt)) s)
    | Arg (e, o, dt) ->
        (* Each tile's extreme and its position in the whole's order,
           stacked, then the first of them in tile order. *)
        let (D.Any t) = term o in
        let pair j =
          let (x, q), () = local j prog Value.[ Arg (e, o, t) ] in
          let q = renumber ~by tiles.(j) whole (Value.Any q) in
          (array x, array (Prim.expect D.Int64 q))
        in
        let pairs = Array.init (Array.length ks) (fun j -> lazy (pair j)) in
        let extremes = stack ~by cl t out (fun j -> fst (Lazy.force pairs.(j)))
        and positions =
          stack ~by cl D.Int64 out (fun j -> snd (Lazy.force pairs.(j)))
        in
        let xs = Place.value ~by cl.placed extremes
        and qs = Place.value ~by cl.placed positions in
        let ranked i =
          [ Value.Any (slot ~by cl i xs); Value.Any (slot ~by cl i qs) ]
        in
        let first =
          match slots cl with
          | i :: rest ->
              List.fold_left
                (fun a i -> first_of ~by e a (ranked i))
                (ranked i) rest
          | [] -> invalid_arg "Partial.reduce: no tile holds terms"
        in
        let x, q =
          match first with
          | [ x; q ] -> (cast ~by (D.Any dt) x, q)
          | _ -> invalid_arg "Partial.reduce: an extreme and its position"
        in
        ( gathered ~by cl (Prim.expect dt x),
          gathered ~by cl (Prim.expect D.Int64 q) )
    | Moments (o, dt) ->
        (* Two passes, each closing its sums as a [Sum]: the mean from the
           tiles' sums, then the tiles' sums of the deviations from it and
           of their squares. *)
        let (D.Any t) = term o in
        let n = numel whole in
        let xs =
          Array.init (Array.length ks) (fun j ->
              lazy
                (terms ~by (here j) layout prog o (D.Any t) (viewed j)
                   (loop j)))
        in
        let x j = Prim.expect t (Lazy.force xs.(j)) in
        let over j prog rs loads =
          let layout = L.contiguous (Iarray.of_array (Prim.shape (x j))) in
          at ~by (here j)
            (Value.Reduce { layout; axes; prog; reductions = rs; loads })
        in
        let sum s = reduced ~by cl (Monoid (Sum, 0, c)) s in
        let total j =
          let s, () =
            over j (identity (D.Any t)) Value.[ Monoid (Sum, 0, c) ]
              [| Plain (x j) |]
          in
          array s
        in
        let s = sum (stack ~by cl c out total) in
        let m =
          gathered ~by cl (Prim.expect c (List.hd (mean ~by n (Value.Any s))))
        in
        let centred j =
          let shape = Prim.shape (x j) in
          let units =
            Array.mapi (fun a e -> if Iarray.mem a axes then 1 else e) shape
          in
          let mj = view ~by ks.(j) (here j) (outw j) m in
          let mj = spread ~by shape (reshape ~by units mj) in
          let d = E.(cast c (input 0 t) - input 1 c) in
          let ins : D.any iarray = [| D.Any t; D.Any c |] in
          let prog = P.of_exprs ~ins [| E.any d; E.any E.(d * d) |] in
          let s1, (s2, ()) =
            over j prog
              Value.[ Monoid (Sum, 0, c); Monoid (Sum, 1, c) ]
              [| Plain (x j); Plain mj |]
          in
          (array s1, array s2)
        in
        let ds = Array.init (Array.length ks) (fun j -> lazy (centred j)) in
        let s1 = sum (stack ~by cl c out (fun j -> fst (Lazy.force ds.(j))))
        and s2 = sum (stack ~by cl c out (fun j -> snd (Lazy.force ds.(j)))) in
        let mv =
          moments ~by (D.Any dt) n (Value.Any m) (Value.Any s1) (Value.Any s2)
        in
        let nth i = gathered ~by cl (Prim.expect dt (List.nth mv i)) in
        (nth 0, nth 1)
  in
  let rec each : type q. (d, q) Value.reductions -> q = function
    | [] -> ()
    | r :: rest ->
        let a = one r in
        (a, each rest)
  in
  each rs

(* Scans *)

let scan (type d a) ~by (route : unit Route.t) rule layout axis prog
    (r : (d, a) Value.reduction) (loads : d Value.load iarray) : a =
  let shape = Iarray.to_array (L.shape layout) in
  let p = route.result in
  let set = Devices.set p and ks = Grid.devices (Devices.grid p) in
  let n = Array.length ks in
  let win j = Devices.window ~by p shape j in
  let here j = Devices.one set ks.(j) in
  let viewed j =
    let tile _ = invalid_arg "Partial.scan: a reduced axis" in
    Iarray.mapi
      (fun i l ->
        let (Value.Any x) = Prim.load_any l in
        view_load ~by ks.(j) (here j)
          (reads rule i (Prim.shape x) (win j) tile)
          l)
      loads
  in
  (* The tiles' totals, each of the results' shape but one element along the
     axis, stacked along a new leading axis cut as the axis is, of one slot
     per tile holding terms; then placed whole on each device. The tiles
     holding terms come first, so the others hold no slot. *)
  let g = Devices.grid p in
  let tiles =
    Option.get
      (Array.find_map
         (fun (a, t) -> if a = axis then Some t else None)
         (Grid.cuts g))
  in
  let row = Array.copy shape in
  row.(axis) <- 1;
  let stack_shape = Array.append [| Int.min shape.(axis) tiles |] row in
  let lead = Grid.map_axes (fun a -> if a = axis then 0 else a + 1) g in
  let stacked = Devices.v ~by set lead
  and placed = Devices.v ~by set (Grid.uncut lead ~axis:0) in
  let index j = (Devices.window ~by stacked stack_shape j).(0).start in
  (* A device's scan of its tile alone, by [reduction]. *)
  let scanned j reduction =
    let op =
      Value.Scan
        { layout; axis; prog; reduction; loads = viewed j; carry = None }
    in
    at ~by (here j) (Prim.within op (win j))
  in
  (* [x]'s last element along the axis, as its slot of the stack. *)
  let last (type v s) j (x : (v, s, d) Value.t) : (v, s) A.t =
    let e = Prim.dim x axis in
    let slot = Devices.window ~by stacked stack_shape j in
    array (reshape ~by (counts slot) (slice ~by axis (e - 1) 1 x))
  in
  (* The totals [slot j] stacked: an empty array on a device whose tile
     holds no term. *)
  let totals (type v s) (dt : (v, s) D.t) (slot : int -> (v, s) A.t option)
      : (v, s, d) Value.t =
    let one j =
      match slot j with
      | Some a -> a
      | None ->
          let s = counts (Devices.window ~by stacked stack_shape j) in
          A.create (Devices.rig set ks.(j)) dt (Iarray.of_array s)
    in
    Prim.of_arrays stacked ~shape:stack_shape (Iarray.init n one)
  in
  (* Slot [i] of [x], a stack at [placed], on device [j]: its row of the
     results' shape. *)
  let earlier (type v s) j i (x : (v, s, d) Value.t) : (v, s, d) Value.t =
    let w = Devices.window ~by placed stack_shape j in
    w.(0) <- { M.start = i; count = 1; step = 1 };
    let row = Array.sub (counts w) 1 (Array.length shape) in
    reshape ~by row (view ~by ks.(j) (here j) w x)
  in
  let wide j (Value.Any x : d Value.any) =
    Value.Any (spread ~by (counts (win j)) x)
  in
  let fresh (type v s) (dt : (v, s) D.t) j =
    A.create (Devices.rig set ks.(j)) dt (Iarray.of_array (counts (win j)))
  in
  let held j = not (is_empty (win j)) in
  match r with
  | Monoid (m, o, dt) ->
      (* Each tile scans alone into the carrier; the totals' scan gives each
         device the combination of the tiles before its own, which it
         combines with its tile's scan, rounding once into [dt]. *)
      let (D.Any c) =
        Nx_kernel.carrier (P.dtype prog (Iarray.get prog.outs o))
      in
      let ys =
        Array.init n (fun j ->
            if held j then Some (scanned j (Monoid (m, o, c))) else None)
      in
      let prefix =
        let x = totals c (fun j -> Option.map (last j) ys.(j)) in
        let x = Place.value ~by placed x in
        let op =
          Value.Scan
            {
              layout = L.contiguous (Iarray.of_array stack_shape);
              axis = 0;
              prog = identity (D.Any c);
              reduction = Monoid (m, 0, c);
              loads = [| Plain x |];
              carry = None;
            }
        in
        at ~by placed op
      in
      let combined j =
        match ys.(j) with
        | None -> fresh dt j
        | Some y ->
            let t = index j in
            let into =
              match D.equal_witness c dt with
              | Some Type.Equal -> Some (A.Any (array y))
              | None -> None
            in
            if t = 0 then
              array (Prim.expect dt (cast ~by (D.Any dt) (Value.Any y)))
            else
              let carry = earlier j (t - 1) prefix in
              let carry = spread ~by (counts (win j)) carry in
              let e = E.binary (binary m) (E.input 0 c) (E.input 1 c) in
              let v =
                map ~by ?into (here j) [ Value.Any carry; Value.Any y ]
                  [ E.any (E.cast dt e) ]
              in
              array (Prim.expect dt (List.hd v))
      in
      Prim.of_arrays p ~shape (Iarray.init n combined)
  | Arg (e, o, dt) ->
      (* Each tile scans alone, its positions from the axis's start; the
         first extreme of the tiles before its own, in tile order, combines
         with each of its results. *)
      let pairs =
        Array.init n (fun j ->
            if held j then
              let v, q = scanned j (Arg (e, o, dt)) in
              let q = shift ~by (win j).(axis).start (Value.Any q) in
              Some (v, Prim.expect D.Int64 q)
            else None)
      in
      let xs =
        totals dt (fun j -> Option.map (fun (v, _) -> last j v) pairs.(j))
      and qs =
        totals D.Int64 (fun j -> Option.map (fun (_, q) -> last j q) pairs.(j))
      in
      let xs = Place.value ~by placed xs and qs = Place.value ~by placed qs in
      let results =
        Array.init n (fun j ->
            match pairs.(j) with
            | None -> (fresh dt j, fresh D.Int64 j)
            | Some (v, q) -> (
                let t = index j in
                if t = 0 then (array v, array q)
                else
                  let ranked i =
                    [ Value.Any (earlier j i xs); Value.Any (earlier j i qs) ]
                  in
                  let carry = ref (ranked 0) in
                  for i = 1 to t - 1 do
                    carry := first_of ~by e !carry (ranked i)
                  done;
                  let carry = List.map (wide j) !carry in
                  match first_of ~by e carry [ Value.Any v; Value.Any q ] with
                  | [ x; y ] ->
                      (array (Prim.expect dt x), array (Prim.expect D.Int64 y))
                  | _ ->
                      invalid_arg "Partial.scan: an extreme and its position"))
      in
      ( Prim.of_arrays p ~shape (Iarray.map fst (Iarray.of_array results)),
        Prim.of_arrays p ~shape (Iarray.map snd (Iarray.of_array results)) )
  | Logsumexp _ | Moments _ ->
      invalid_arg "Partial.scan: a scan that reads its axis whole"

(* Contractions *)

(* The fences of a device's tile [q] of [Rows]' groups, from the offsets
   [o], [g + 1] along their last axis, in the window [w] the device [k]
   holds whole along it, at [here]: fences [q.start] to [q.start + q.count]
   of [F e], the least of [max o.(0) o.(i)] for [i] from [e] to [g].
   {!Nx_kernel.type-segments} puts index [r] in group [e] where
   [max o.(0) o.(e) <= r] and [r] is below each of [o.(e + 1)] to
   [o.(g)]: that is [F e <= r < F (e + 1)], so the tile's groups are the
   whole's, whatever the offsets, and offsets that do not decrease are
   their own [F]. One scan of [Min] over [o.(g)] down to [o.(q.start)]
   gives them. *)
let fences (type d) ~by k (here : d Devices.placement) w (q : M.range)
    (o : d Value.index) : d Value.index =
  let last = Array.length w - 1 in
  let tail = (Prim.shape o).(last) - q.start in
  let along r =
    let w = Array.copy w in
    w.(last) <- r;
    view ~by k here w o
  in
  let down =
    backwards ~by last (tail - 1) tail
      (along { M.start = q.start; count = tail; step = 1 })
  in
  let first =
    spread ~by (Prim.shape down) (along { M.start = 0; count = 1; step = 1 })
  in
  let ins : D.any iarray = [| D.Any D.Int64; D.Any D.Int64 |] in
  let term = E.maximum (E.input 0 D.Int64) (E.input 1 D.Int64) in
  let least =
    at ~by here
      (Value.Scan
         {
           layout = L.contiguous (Iarray.of_array (Prim.shape down));
           axis = last;
           prog = P.of_exprs ~ins [| E.any term |];
           reduction = Monoid (Min, 0, D.Int64);
           loads = [| Plain down; Plain first |];
           carry = None;
         })
  in
  backwards ~by last (tail - 1) (q.count + 1) least

let contract (type v s d a b c e) ~by (route : unit Route.t) rule ~batch
    ~contracting ~acc ~segments (out : (v, s) D.t) (a : (a, b, d) Value.t)
    (b : (c, e, d) Value.t) (init : (v, s, d) Value.t option)
    (a_scale : d Value.scale option) (b_scale : d Value.scale option) :
    (v, s, d) Value.t =
  let shape =
    Prim.contract_shape ~by ~batch ~contracting ~segments ~acc a b init
      a_scale b_scale
  in
  let pairs = Iarray.to_array contracting in
  let p = route.result in
  let set = Devices.set p and ks = Grid.devices (Devices.grid p) in
  let resw j = Devices.window ~by p shape j in
  let here j = Devices.one set ks.(j) in
  (* The folded axes: [a]'s of the pairs, or [b]'s of [Rows]' groups, the
     factor after the pairs, which are then read whole. *)
  let operand, folded, of_folded =
    match (segments : d Value.index Nx_kernel.segments option) with
    | None -> (0, Array.map fst pairs, Prim.shape a)
    | Some (Rows { b_axis; _ }) -> (1, [| b_axis |], Prim.shape b)
    | Some (Contracting _) ->
        invalid_arg "Partial.contract: groups of a contracting pair"
  in
  let tiles = tiles ~by route ~operand of_folded folded in
  let cl =
    closing ~by route ~operand ~axes:(Array.to_list folded)
      ~extent:(Array.get of_folded) shape
  in
  (* Device [j]'s tile: its ranges of the folded factors, the pairs' in
     order, then [Rows]' groups', the pairs then whole. *)
  let tile j =
    match segments with
    | None -> tiles.(j)
    | Some _ ->
        let whole (x, _) = full (Prim.dim a x) in
        Array.append (Array.map whole pairs) tiles.(j)
  in
  (* Operand [i] on device [j], its factors' ranges [r]. *)
  let read (type w q) j i r (x : (w, q, d) Value.t) =
    let w = reads rule i (Prim.shape x) (resw j) (Array.get r) in
    view ~by ks.(j) (here j) w x
  in
  (* The scale [sc] of [x], operand [i], on device [j]: along an axis of
     blocks of a factor, the blocks that [x]'s range of it in [r] spans. *)
  let scale j i x r (sc : d Value.scale) : d Value.scale =
    let blocks (type w q) (v : (w, q, d) Value.t) =
      let w = reads rule i (Prim.shape v) (resw j) (Array.get r) in
      let narrow ax (kind : Route.axis) =
        match kind with
        | Blocks { factor; _ } ->
            let n = Prim.dim x ax / Prim.dim v ax and q = r.(factor) in
            let lo = q.M.start / n and hi = (q.start + q.count + n - 1) / n in
            w.(ax) <- { M.start = lo; count = hi - lo; step = 1 }
        | Becomes _ | Whole | Reduced _ | Scanned _ | Regroups _ -> ()
      in
      (match rule with
      | Axes t -> Array.iteri narrow t.(i)
      | Elementwise -> ());
      view ~by ks.(j) (here j) w v
    in
    match sc with
    | Exponent v -> Exponent (blocks v)
    | Value v -> Value (blocks v)
  in
  let first = if Option.is_some init then 3 else 2 in
  let b_at = first + Bool.to_int (Option.is_some a_scale) in
  (* Each factor's scale blocks' lengths: [a]'s and [b]'s along a pair,
     [b]'s along [Rows]' groups. A scale of an empty axis has none. *)
  let run x ax = function
    | Some sc when (Prim.scale_shape sc).(ax) > 0 ->
        [ Prim.dim x ax / (Prim.scale_shape sc).(ax) ]
    | Some _ | None -> []
  in
  let runs =
    Array.append
      (Array.map (fun (i, k) -> run a i a_scale @ run b k b_scale) pairs)
      (Array.map (fun x -> run b x b_scale)
         (match segments with Some _ -> folded | None -> [||]))
  in
  (* The groups of a piece [r] on device [j]: [Rows] bounded by the fences
     of its range of them. *)
  let groups j r =
    match segments with
    | None -> None
    | Some (Rows g) ->
        let i = b_at + Bool.to_int (Option.is_some b_scale) in
        let o = g.offsets in
        let w = reads rule i (Prim.shape o) (resw j) (Array.get r) in
        let q = r.(Array.length pairs) in
        let offsets = fences ~by ks.(j) (here j) w q o in
        Some (Nx_kernel.Rows { g with offsets })
    | Some (Contracting _) ->
        invalid_arg "Partial.contract: groups of a contracting pair"
  in
  let (D.Any ad) = acc in
  (* The contraction of a piece [r] of a tile on device [j] into the
     accumulator, from [from], the sum of the pieces before it. *)
  let piece j from r =
    at ~by (here j)
      (Value.Contract
         {
           batch;
           contracting;
           acc;
           out = ad;
           a = read j 0 r a;
           b = read j 1 r b;
           init = from;
           a_scale = Option.map (scale j first a r) a_scale;
           b_scale = Option.map (scale j b_at b r) b_scale;
           segments = groups j r;
         })
  in
  (* A device's partial: its tile's pieces in C order, the first from no
     [init] and each other from the sum of those before it. A piece of
     [Rows]' groups gives no products to a row of another group, which so
     keeps the sum before it, bit for bit. *)
  let partial j =
    match tile_pieces runs (tile j) with
    | r :: rs ->
        let from s r = piece j (Some s) r in
        array (List.fold_left from (piece j None r) rs)
    | [] -> invalid_arg "Partial.contract: a tile of no piece"
  in
  let s = stack ~by cl ad shape partial in
  match init with
  | None -> gathered ~by cl (reduced ~by cl (Monoid (Sum, 0, out)) s)
  | Some i ->
      (* The sums in the accumulator, then [init] as it holds it, rounded
         once. A segmented contraction's output without products sums to
         [+0] in every slot and takes [init] alone, as on one device. *)
      let y = reduced ~by cl (Monoid (Sum, 0, ad)) s in
      let total = E.input 0 ad and initial = E.cast ad (E.input 1 out) in
      let e =
        match segments with
        | None -> E.(total + initial)
        | Some _ ->
            E.(where (total = const ad (D.zero ad)) initial (total + initial))
      in
      let z =
        map ~by cl.parts [ Value.Any y; Value.Any i ] [ E.any (E.cast out e) ]
      in
      gathered ~by cl (Prim.expect out (List.hd z))

let run (type r) ~by route (op : r Value.prim) : r =
  let rule = Prim.rule op in
  match op with
  | Value.Reduce { layout; axes; prog; reductions; loads } ->
      reduce ~by route rule layout axes prog reductions loads
  | Value.Scan { layout; axis; prog; reduction; loads; carry = None } ->
      scan ~by route rule layout axis prog reduction loads
  | Value.Contract
      {
        batch;
        contracting;
        acc;
        out;
        a;
        b;
        init;
        a_scale;
        b_scale;
        segments = (None | Some (Rows _)) as segments;
      } ->
      contract ~by route rule ~batch ~contracting ~acc ~segments out a b init
        a_scale b_scale
  | Value.Contract { segments = Some (Contracting _); _ } ->
      invalid_arg "Partial.run: groups of a contracting pair by partials"
  | Value.Scan { carry = Some _; _ }
  | Value.Map _ | Value.Gather _ | Value.Scatter _ | Value.Sort _
  | Value.Fft _ | Value.Linalg _ | Value.Fold _ | Value.Assemble _
  | Value.Copy _ | Value.Move _ | Value.Bitcast _ | Value.Place _
  | Value.Check _ ->
      invalid_arg "Partial.run: an operation no route computes by partials"
