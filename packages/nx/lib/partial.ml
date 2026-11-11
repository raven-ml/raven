(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Nx_array.Dtype
module L = Nx_array.Layout
module M = Nx_array.Move
module P = Nx_kernel.Prog
module E = P.E

let invalid_argf fmt = Format.kasprintf invalid_arg fmt
let numel = Array.fold_left ( * ) 1

(* A window's extents as a value's shape, which {!reshape} and {!spread}
   take; {!Route.extents} gives an array's. *)
let counts w = Array.map (fun (r : M.range) -> r.count) w

let floats (D.Any dt) =
  match D.kind dt with
  | D.Float | D.Complex -> true
  | D.Signed | D.Unsigned | D.Boolean -> false

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
   along an axis of the folded factor [f]. An axis of blocks of [f] is
   whole: its reader takes the blocks a tile spans. *)
let reads rule i shape (w : M.range array) tile =
  match (rule : Route.rule) with
  | Elementwise -> w
  | Axes t ->
      let window = Route.window rule ~operand:i shape w in
      Array.mapi
        (fun a r ->
          match t.(i).(a) with
          | Route.Reduced f -> tile f
          | Becomes _ | Scanned _ | Indexed _ | Regroups _ | Blocks _ | Whole ->
              r)
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
   over its operands' windows ({!Partition.windows}), which each operand
   holds there: by [eng]'s [run] where they lie on [p]'s one device, which
   builds no route. An operation that creates its values computes at [p]
   too. The first result takes [into] where it is given. A loop computes
   the window of its loop's shape that starts at [origin]. *)
let at : type r d.
    Partition.engine ->
    by:string ->
    ?into:d Value.any ->
    ?origin:Prim.origin ->
    unit Devices.placement ->
    r Value.prim ->
    r =
 fun eng ~by ?into ?(origin = Prim.no_origin) p op ->
  let layouts = Prim.layouts ~by op in
  if Grid.count (Devices.grid p) = 1 && Prim.lies_at p op then
    eng.run ~by ?into ~origin p layouts op
  else
    let (Prim.Operands xs) = Prim.operands op in
    let operands =
      Array.of_list (List.map (fun (Value.Any x) -> Prim.placement x) xs)
    in
    let route = { Route.operands; result = p; compute = Windows } in
    Partition.windows eng ~by ?into ~origin route layouts op

(* [op] at [x]'s placement. *)
let on eng ~by x op = at eng ~by (Prim.placement x) op

(* For each device [j] of [p] that [holds] a window of a result of [shape],
   in order, [compute j], or, on a device that copies a replica
   ({!Partition.sources}), [copy here] of its source's, [here] its
   one-device placement. *)
let replicated (type a d) ~by (p : d Devices.placement) shape ~holds
    ~(compute : int -> a) ~(copy : d Devices.placement -> a -> a) :
    a option array =
  let set = Devices.set p and ks = Grid.devices (Devices.grid p) in
  let src = Partition.sources ~by p shape in
  let rs = Array.make (Array.length ks) None in
  for j = 0 to Array.length ks - 1 do
    if holds j then
      rs.(j) <-
        Some
          (if src.(j) = j then compute j
           else copy (Devices.one set ks.(j)) (Option.get rs.(src.(j))))
  done;
  rs

(* {!replicated} of values. *)
let values eng ~by p shape ~holds compute =
  replicated ~by p shape ~holds ~compute ~copy:(Partition.copy eng)

(* Values *)

(* [x] of [shape], its cut axes keeping their numbers. *)
let reshape (type v s d) eng ~by shape (x : (v, s, d) Value.t) :
    (v, s, d) Value.t =
  if Prim.has_shape x shape then x
  else on eng ~by x (Value.Move (Reshape (Iarray.of_array shape), x))

(* Elements [start, start + count) of [x] along [axis], an axis no cut
   names. *)
let slice (type v s d) eng ~by axis start count (x : (v, s, d) Value.t) :
    (v, s, d) Value.t =
  let r = Array.map Route.full (Prim.shape x) in
  if start = 0 && count = r.(axis).count then x
  else begin
    r.(axis) <- { M.start; count; step = 1 };
    on eng ~by x (Value.Move (Slice (Iarray.of_array r), x))
  end

(* [x] of [shape], its axes of extent 1 stretched. *)
let spread (type v s d) eng ~by shape (x : (v, s, d) Value.t) :
    (v, s, d) Value.t =
  if Prim.has_shape x shape then x
  else on eng ~by x (Value.Move (Broadcast (Iarray.of_array shape), x))

let cast (type d) eng ~by (D.Any dt) (Value.Any x : d Value.any) : d Value.any =
  match D.equal_witness (Prim.dtype x) dt with
  | Some Type.Equal -> Value.Any x
  | None -> Value.Any (fst (on eng ~by x (Prim.op1 ~by Cast dt x)))

(* [x]'s window [w] from its piece on its set's device [k]. *)
let view eng ~by k w x = Partition.read eng ~by x k w

let view_load (type d) eng ~by k w (l : d Value.load) : d Value.load =
  match l with
  | Plain x -> Plain (view eng ~by k w x)
  | Padded p -> Padded { p with x = view eng ~by k w p.x }

(* The map over [vs], of one shape, its inputs in order, to [exprs], with
   its results at [p], at [origin] where it is given. *)
let map (type d) eng ~by ?into ?origin p (vs : d Value.any list) exprs :
    d Value.any list =
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
  read
    (at eng ~by ?into ?origin p
       (Value.Map { layout; prog; outs; loads; at = None }))

(* Output [o] of [prog], of [term], over [loads] in the window [w] of a
   loop of [layout], on [here]: the load itself where [prog] is its
   identity. *)
let terms (type d) eng ~by here layout (prog : P.t) o (D.Any term)
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
      let map, origin = Prim.within map w in
      Value.Any (fst (at eng ~by ~origin here map))

(* The mean of [n] terms from their sum [s], in the carrier, at [s]'s
   placement. *)
let mean eng ~by n s =
  let (Value.Any x) = s in
  let c = Prim.dtype x in
  let count = E.const c (D.of_float c (Float.of_int n)) in
  map eng ~by (Prim.placement x) [ s ] [ E.any (E.fdiv (E.input 0 c) count) ]

(* The variance into [dt] of [n] terms from the sums, in the carrier, [s1]
   of their deviations [d] from their mean and [s2] of [d²], at [s1]'s
   placement: [(Σ d² - Σ d (Σ d / n)) / n], [+0] where that is negative or
   [Σ d²] where that is [+∞]. A NaN mean makes every deviation, so the
   variance, a NaN. *)
let variance eng ~by (D.Any dt) n s1 s2 =
  let (Value.Any x) = s1 in
  let c = Prim.dtype x in
  let v1 = E.input 0 c and v2 = E.input 1 c in
  let count = E.const c (D.of_float c (Float.of_int n)) in
  let zero = E.const c (D.zero c)
  and inf = E.const c (D.of_float c Float.infinity) in
  let q = E.(fdiv (v2 - (v1 * fdiv v1 count)) count) in
  let v = E.(where (v2 = inf) v2 (where (q < zero) zero q)) in
  map eng ~by (Prim.placement x) [ s1; s2 ] [ E.any (E.cast dt v) ]

(* Positions *)

(* The values [x] orders by: itself, or a complex number's parts. *)
let keys (type v s d) eng ~by (x : (v, s, d) Value.t) : d Value.any list =
  let parts (type w r) (pt : (w, r) D.t) =
    let pairs = on eng ~by x (Value.Bitcast (pt, x)) in
    let shape = Prim.shape x in
    let half j =
      let part = slice eng ~by (Array.length shape) j 1 pairs in
      Value.Any (reshape eng ~by shape part)
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
let first_of eng ~by (e : Nx_kernel.extreme) a b =
  match (a, b) with
  | [ Value.Any ea; pa ], [ eb; pb ] ->
      let dt = Prim.dtype ea in
      let eb = Prim.expect dt eb in
      let ka = keys eng ~by ea and kb = keys eng ~by eb in
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
      map eng ~by (Prim.placement ea)
        ((Value.Any ea :: pa :: ka) @ (Value.Any eb :: pb :: kb))
        [ pick 0 dt; pick 1 D.Int64 ]
  | _ -> invalid_arg "Partial.first_of: an extreme and its position each"

(* A tile's start reaches a device's program as a loop's coordinate at an
   origin ({!Partition.engine}), so that an engine may bind it when the
   program runs and each device's program is every device's. *)

(* [pos], [int64] positions on one device, and the start [s] of a nonempty
   window along an axis of [n] elements, combined by [f]: [pos] itself
   where [s] is [0]. One map over [pos] with a leading axis of one element,
   at [s] in a loop of [n] along it, reads the start as its coordinate
   there. A [pos] of the greatest rank, which has no room for that axis,
   reads it as a constant. *)
let shift (type d) eng ~by f s n (pos : d Value.any) : d Value.any =
  if s = 0 then pos
  else
    let (Value.Any x) = pos in
    let here = Prim.placement x and shape = Prim.shape x in
    let p = E.input 0 D.Int64 in
    if Array.length shape >= L.max_rank then
      let s = E.const D.Int64 (Int64.of_int s) in
      List.hd (map eng ~by here [ pos ] [ E.any (f p s) ])
    else
      let lead = Array.append [| 1 |] shape in
      let start = Array.mapi (fun a _ -> if a = 0 then s else 0) lead in
      let origin = { Prim.start; extents = Array.append [| n |] shape } in
      let lead = Value.Any (reshape eng ~by lead x) in
      let e = E.any (f p (E.coord 0)) in
      let (Value.Any y) = List.hd (map eng ~by ~origin here [ lead ] [ e ]) in
      Value.Any (reshape eng ~by shape y)

(* [pos], positions in C order of the folded indices of a device's tile
   [w], along the folded [axes] of a reduction's loop at [origin], as
   positions in C order of the folded extents [whole], over [pos]'s memory:
   [pos] is fresh, on one device, of the loop's shape but [axes]. The
   tile's starts are the loop's coordinates along [axes], over a loop of
   one element along each. *)
let renumber eng ~by ~(origin : Prim.origin) axes (w : M.range array) whole
    pos =
  if
    Array.for_all2 (fun (r : M.range) n -> r.start = 0 && r.count = n) w whole
  then pos
  else
    let (Value.Any x) = pos in
    let out = Prim.shape x in
    let kept = ref 0 in
    let units =
      Array.mapi
        (fun a _ ->
          if Array.mem a axes then 1
          else begin
            incr kept;
            out.(!kept - 1)
          end)
        origin.extents
    in
    let units = reshape eng ~by units x in
    let int64 n = E.const D.Int64 (Int64.of_int n) in
    let p = E.input 0 D.Int64 in
    let global = ref (int64 0) and inner = ref 1 and outer = ref 1 in
    for i = Array.length w - 1 downto 0 do
      let digit = E.((p / int64 !inner) mod int64 w.(i).count) in
      global := E.(!global + ((digit + coord axes.(i)) * int64 !outer));
      inner := !inner * w.(i).count;
      outer := !outer * whole.(i)
    done;
    let here = Prim.placement x and units = Value.Any units in
    let (Value.Any q) =
      List.hd (map eng ~by ~into:units ~origin here [ units ] [ E.any !global ])
    in
    Value.Any (reshape eng ~by out q)

(* Closings *)

(* Where a reduction's or a contraction's partials close: [result], the
   route's; [stacked], the placement of the partials stacked along a new
   leading axis per cut folded axis, of [held] elements, each cut over the
   grid axes that cut its folded axis, the others cut as [result] cuts
   them; and [parts], where each device computes its part of the results:
   [result] with the grid axes that cut the leading axes moved to an axis
   the result keeps whole, where it has one of at least a group's devices.
   [held] is, for each cut folded axis in order, the number of its tiles
   that hold terms, which come first: the route computes by partials only
   where every folded axis has elements ({!Route.computed}). *)
type closing = {
  result : unit Devices.placement;
  stacked : unit Devices.placement;
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
  let parts =
    match split with
    | Some k ->
        let placed = cut_at cut (k + lead) up in
        Devices.v ~by set (Grid.map_axes (fun a -> a - lead) placed)
    | None -> r
  in
  { result = r; stacked = Devices.v ~by set stacked; parts; held }

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
   one whose tile is empty, holds an empty piece. *)
let stack (type v s d) eng ~by c (dt : (v, s) D.t) shape
    (partial : int -> (v, s, d) Value.t) : (v, s, d) Value.t =
  let shape = Array.append (Array.of_list c.held) shape in
  let w j = Devices.window ~by c.stacked shape j in
  let slots =
    values eng ~by c.stacked shape
      ~holds:(fun j -> not (Route.empty (w j)))
      (fun j -> eng.Partition.view (M.Reshape (Route.extents (w j))) (partial j))
  in
  eng.assemble c.stacked shape dt slots

(* The most bytes of slots a piece gathers: a buffer of fixed size, so that
   a closing holds no more however large its result. A piece costs about
   1,100 words and 12 us of calls on four software devices, and reducing 256
   KiB of slots about 220 us, so its calls stay near a twentieth. *)
let piece_bytes = 262144

(* The pieces of a device's part [w], of [t] slots of [bits] bits an
   element, boxes in C order that tile it: [w] itself where its slots fit
   in {!piece_bytes} and where [w] is empty; else, for the first axis [a]
   on which the slots of one index fit, single indices of each axis before
   [a] and runs along [a] of as many indices as fit, every axis after [a]
   whole. Where no axis's index fits, as for more slots of one element
   than fit, [a] is the last axis, in runs of one. *)
let part_pieces t ~bits (w : M.range array) =
  let r = Array.length w in
  (* The bits of the slots of one index of axis [a]. *)
  let per a = t * bits * numel (counts (Array.sub w (a + 1) (r - a - 1))) in
  let budget = 8 * piece_bytes in
  if Route.empty w || r = 0 || numel (counts w) * t * bits <= budget then
    [ w ]
  else
    let a =
      match List.find_opt (fun a -> per a <= budget) (List.init r Fun.id) with
      | Some a -> a
      | None -> r - 1
    in
    let n = Int.max 1 (budget / per a) in
    (* The boxes over axes [i] and after, within [p], C order. *)
    let rec boxes i p =
      if i > a then [ Array.copy p ]
      else
        let { M.start; count = e; _ } = w.(i) in
        let step = if i = a then n else 1 in
        List.concat
          (List.init ((e + step - 1) / step) (fun k ->
               let count = Int.min step (e - (k * step)) in
               p.(i) <- { M.start = start + (k * step); count; step = 1 };
               boxes (i + 1) p))
    in
    boxes 0 (Array.copy w)

(* [inner], a window of the whole, as a window of [outer]'s. *)
let relative (outer : M.range array) (inner : M.range array) =
  Array.map2
    (fun (o : M.range) (i : M.range) -> { i with start = i.start - o.start })
    outer inner

type 'd held =
  | Held : ('v, 's, 'd) Value.t * ('v, 's, 'd) Partition.held -> 'd held

(* A piece's slots on its device: [slot i], each stack's slot at the index
   [i] of the leading axes, of the piece's shape, taken when asked for, and
   [all ()], each stack's slots at once, the leading axes kept. *)
type 'd taken = {
  slot : int list -> 'd Value.any list;
  all : unit -> 'd Value.any list;
}

(* [f]'s results, of [dts], at the route's result, from [stacks], each at
   [c.stacked], of the results of [shape] after their leading axes. Each
   device computes its part, its window at [c.parts], piece by piece:
   [f k here w taken] computes the piece [w] on the set's device [k] alone,
   at [here], from each stack's slots there, which it takes ({!taken}).
   Each device of the result then gathers its window from the pieces, its
   own where it holds them. So a device holds, beside its slots, at most
   one piece's slots and what [f] makes of them at a time, then its pieces
   and its window of the results. A part of a dtype narrower than a byte on a device whose memory
   the host does not address, where a piece may start inside a byte, is
   one piece. *)
let closed (type d) (eng : Partition.engine) ~by c shape dts
    (stacks : d Value.any list)
    (f :
      int ->
      d Devices.placement ->
      M.range array ->
      d taken ->
      d Value.any list) : d Value.any list =
  let set = Devices.set c.parts in
  let t = List.fold_left ( * ) 1 c.held in
  let lead = Array.of_list (List.map Route.full c.held) in
  let narrow = List.exists (fun (D.Any dt) -> D.bits dt < 8) dts in
  let bits =
    List.fold_left
      (fun n (Value.Any s) -> n + D.bits (Prim.dtype s))
      0 stacks
  in
  let held =
    List.map (fun (Value.Any s) -> Held (s, Partition.held ~by s)) stacks
  in
  (* Each device's pieces: its set device, their windows and results. *)
  let device j k =
    let here = Devices.one set k and d = Devices.rig set k in
    let w = Devices.window ~by c.parts shape j in
    let ps =
      if narrow && not (Rig.reaches Rig.host d) then [ w ]
      else part_pieces t ~bits w
    in
    let piece p =
      (* Each stack's window [sw], of the shape [s] where given. *)
      let at ?s sw =
        List.map (fun (Held (x, h)) ->
            let y =
              if Route.empty sw then
                eng.Partition.assemble here (counts sw) (Prim.dtype x)
                  [| None |]
              else Partition.take eng h here sw
            in
            match s with
            | None -> Value.Any y
            | Some s -> Value.Any (eng.view (Reshape (Iarray.of_array s)) y))
          held
      in
      let all () = at (Array.append lead p) in
      let slot index =
        let one i = { M.start = i; count = 1; step = 1 } in
        at ~s:(counts p) (Array.append (Array.of_list (List.map one index)) p)
      in
      (p, f k here p { slot; all })
    in
    (k, List.map piece ps)
  in
  (* A device that copies a replica of a part ({!Partition.sources}) copies
     its source's pieces. *)
  let ks = Grid.devices (Devices.grid c.parts) in
  let src = Partition.sources ~by c.parts shape in
  let parts = Array.make (Array.length ks) (0, []) in
  Array.iteri
    (fun j k ->
      let i = src.(j) in
      let empty = Route.empty (Devices.window ~by c.parts shape j) in
      parts.(j) <-
        (if i = j || empty then device j k
         else
           let here = Devices.one set k in
           let one (Value.Any y) = Value.Any (Partition.copy eng here y) in
           (k, List.map (fun (p, ys) -> (p, List.map one ys)) (snd parts.(i)))))
    ks;
  let r = c.result in
  let rs = Grid.devices (Devices.grid r) in
  List.mapi
    (fun i (D.Any dt) ->
      let nth ys = Prim.expect dt (List.nth ys i) in
      let window j =
        let k = rs.(j) in
        let w = Devices.window ~by r shape j in
        if Route.empty w then None
        else
          (* The pieces within [w], of distinct windows, [k]'s where it holds
             one, in C order of their starts: each device gathers a window of
             one shape from its boxes in one order, one program for all. *)
          let within = ref [] in
          let add own =
            Array.iter
              (fun (k', ps) ->
                if (k' = k) = own then
                  List.iter
                    (fun (p, ys) ->
                      if
                        Route.contains w p
                        && not (List.exists (fun (q, _) -> q = p) !within)
                      then within := (p, nth ys) :: !within)
                    ps)
              parts
          in
          add true;
          add false;
          let start (p, _) = Array.map (fun (q : M.range) -> q.start) p in
          match List.sort (fun a b -> compare (start a) (start b)) !within with
          | [ (p, y) ] when p = w -> Some y
          | boxes ->
              let here = Devices.one (Devices.set r) k in
              let whole =
                Array.map (fun (q : M.range) -> { q with start = 0 })
              in
              let box (p, y) = (relative w p, y, whole p) in
              let boxes = List.map box boxes in
              Some (eng.Partition.gather here dt (Route.extents w) boxes)
      in
      let ws = Array.init (Array.length rs) window in
      Value.Any (eng.Partition.assemble r shape dt ws))
    dts

(* [f] over a piece's slots ([taken]'s), from the first in C order of their
   indices, each later one combined with what [f] made of those before it,
   so that a device holds one slot it took at a time. *)
let running c (taken : 'd taken) f =
  match slots c with
  | i :: rest ->
      List.fold_left (fun acc i -> f acc (taken.slot i)) (taken.slot i) rest
  | [] -> invalid_arg "Partial: no tile holds terms"

(* The monoid [m] of [vs] and [ws], each in turn, at [here]. *)
let combined eng ~by here m vs ws =
  let k = List.length vs in
  let exprs =
    List.mapi
      (fun i (Value.Any v) ->
        let c = Prim.dtype v in
        E.any (E.binary (binary m) (E.input i c) (E.input (k + i) c)))
      vs
  in
  map eng ~by here (vs @ ws) exprs

(* The reduction [r] over the leading axes, [lead] of them, of [s], a stack
   on one device, at [here]. *)
let fold_slots (type v s w q d) eng ~by ~lead here
    (r : (d, (w, q, d) Value.t) Value.reduction) (s : (v, s, d) Value.t) :
    (w, q, d) Value.t =
  let op =
    Value.Reduce
      {
        layout = L.contiguous (Iarray.of_array (Prim.shape s));
        axes = Iarray.init lead Fun.id;
        prog = Prim.identity (Prim.dtype s);
        reductions = Value.[ r ];
        loads = [| Plain s |];
      }
  in
  fst (at eng ~by here op)

(* [then_] of the reduction [r], at the route's result, of [s], the stack
   at [c.stacked], of the results of [shape], over its leading axes, into
   [dt]: the reduction itself by default. A monoid's slots combine one by
   one in C order of their indices ({!running}), then round into [dt]; a
   logsumexp's reduce at once. [then_ k here w] finishes the piece [w] on
   the set's device [k], at [here]. *)
let reduced (type v s w q a b d) eng ~by c shape ?(then_ = fun _ _ _ y -> y)
    (dt : (w, q) D.t) (r : (d, (a, b, d) Value.t) Value.reduction)
    (s : (v, s, d) Value.t) : (w, q, d) Value.t =
  let lead = List.length c.held in
  let f k here w taken =
    let y =
      match r with
      | Monoid (m, _, rt) ->
          let sum = running c taken (combined eng ~by here m) in
          cast eng ~by (D.Any rt) (List.hd sum)
      | _ -> (
          match taken.all () with
          | [ Value.Any s ] -> Value.Any (fold_slots eng ~by ~lead here r s)
          | _ -> invalid_arg "Partial.reduced: one stack")
    in
    [ then_ k here w y ]
  in
  let ys = closed eng ~by c shape [ D.Any dt ] [ Value.Any s ] f in
  Prim.expect dt (List.hd ys)

(* Reductions *)

let reduce (type d r) eng ~by (route : unit Route.t) rule layout axes prog
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
        view_load eng ~by ks.(j)
          (reads rule i (Prim.shape x) (outw j) (along j))
          l)
      loads
  in
  let local j prog rs =
    let op =
      Value.Reduce { layout; axes; prog; reductions = rs; loads = viewed j }
    in
    let op, origin = Prim.within op (loop j) in
    at eng ~by ~origin (here j) op
  in
  let term o = P.dtype prog (Iarray.get prog.outs o) in
  let one (type a) (r : (d, a) Value.reduction) : a =
    let _, o, _ = Prim.reduction r in
    let (D.Any c) = Nx_kernel.carrier (term o) in
    match r with
    | Monoid (m, o, dt) ->
        let own j = fst (local j prog Value.[ Monoid (m, o, c) ]) in
        let s = stack eng ~by cl c out own in
        reduced eng ~by cl out dt (Monoid (m, 0, dt)) s
    | Logsumexp (o, dt) ->
        let own j = fst (local j prog Value.[ Logsumexp (o, c) ]) in
        let s = stack eng ~by cl c out own in
        reduced eng ~by cl out dt (Logsumexp (0, dt)) s
    | Arg (e, o, dt) ->
        (* Each tile's extreme and its position in the whole's order,
           stacked, then the first of them in tile order. *)
        let (D.Any t) = term o in
        let pair j =
          let (x, q), () = local j prog Value.[ Arg (e, o, t) ] in
          let origin = { Prim.start = Prim.starts (loop j); extents = shape } in
          let q = renumber eng ~by ~origin ax tiles.(j) whole (Value.Any q) in
          (x, Prim.expect D.Int64 q)
        in
        let pairs = Array.init (Array.length ks) (fun j -> lazy (pair j)) in
        let extremes =
          stack eng ~by cl t out (fun j -> fst (Lazy.force pairs.(j)))
        and positions =
          stack eng ~by cl D.Int64 out (fun j -> snd (Lazy.force pairs.(j)))
        in
        (* A piece's first extremes and positions: the slots combined in
           order, then the extremes cast. *)
        let first _ _ _ taken =
          match running cl taken (first_of eng ~by e) with
          | [ x; q ] -> [ cast eng ~by (D.Any dt) x; q ]
          | _ -> invalid_arg "Partial.reduce: an extreme and its position"
        in
        let stacks = [ Value.Any extremes; Value.Any positions ] in
        let dts = [ D.Any dt; D.Any D.Int64 ] in
        begin match closed eng ~by cl out dts stacks first with
        | [ x; q ] -> (Prim.expect dt x, Prim.expect D.Int64 q)
        | _ -> invalid_arg "Partial.reduce: an extreme and its position"
        end
    | Moments (o, dt) ->
        (* Two passes, each closing its sums as a [Sum]: the mean from the
           tiles' sums, then the tiles' sums of the deviations from it and
           of their squares. *)
        let (D.Any t) = term o in
        let n = numel whole in
        let xs =
          Array.init (Array.length ks) (fun j ->
              lazy
                (terms eng ~by (here j) layout prog o (D.Any t) (viewed j)
                   (loop j)))
        in
        let x j = Prim.expect t (Lazy.force xs.(j)) in
        let over j prog rs loads =
          let layout = L.contiguous (Iarray.of_array (Prim.shape (x j))) in
          at eng ~by (here j)
            (Value.Reduce { layout; axes; prog; reductions = rs; loads })
        in
        let total j =
          let s, () =
            over j (Prim.identity t) Value.[ Monoid (Sum, 0, c) ]
              [| Plain (x j) |]
          in
          s
        in
        let means _ _ _ y = List.hd (mean eng ~by n y) in
        let m =
          reduced eng ~by cl out ~then_:means c (Monoid (Sum, 0, c))
            (stack eng ~by cl c out total)
        in
        let mean = Prim.expect dt (cast eng ~by (D.Any dt) (Value.Any m)) in
        let centred j =
          let shape = Prim.shape (x j) in
          let units =
            Array.mapi (fun a e -> if Iarray.mem a axes then 1 else e) shape
          in
          let mj = view eng ~by ks.(j) (outw j) m in
          let mj = spread eng ~by shape (reshape eng ~by units mj) in
          let d = E.(cast c (input 0 t) - input 1 c) in
          let ins : D.any iarray = [| D.Any t; D.Any c |] in
          let prog = P.of_exprs ~ins [| E.any d; E.any E.(d * d) |] in
          let s1, (s2, ()) =
            over j prog
              Value.[ Monoid (Sum, 0, c); Monoid (Sum, 1, c) ]
              [| Plain (x j); Plain mj |]
          in
          (s1, s2)
        in
        (* The variance closes from the deviations' sums alone, so the mean
           goes once each device has them. *)
        let ds = Array.init (Array.length ks) (fun j -> lazy (centred j)) in
        let part f = stack eng ~by cl c out (fun j -> f (Lazy.force ds.(j))) in
        let stacks = [ Value.Any (part fst); Value.Any (part snd) ] in
        let var _ here _ taken =
          match running cl taken (combined eng ~by here Sum) with
          | [ a; b ] -> variance eng ~by (D.Any dt) n a b
          | _ -> invalid_arg "Partial.reduce: two sums"
        in
        let v = closed eng ~by cl out [ D.Any dt ] stacks var in
        (mean, Prim.expect dt (List.hd v))
  in
  let rec each : type q. (d, q) Value.reductions -> q = function
    | [] -> ()
    | r :: rest ->
        let a = one r in
        (a, each rest)
  in
  each rs

(* Attention *)

(* Attention over keys cut across devices: each device of the result
   computes [(o, lse)] over its tile of the keys, [o] in [acc], as one
   device does (its operands' windows, its program's coordinates offset to its
   rows and keys, its lengths less its first key); [lse] closes as the
   [Logsumexp] of the parts, and [o] as the sum of each part's [o] times
   [exp (lse_part - lse)], in C order of the parts, each product fused,
   rounded once to the call's [out]: [+0] where [lse] is [-∞]. A paged call's
   factor is its table's pages, of [page] keys each. *)
(* The bound nx.mli states, to first order. Part j's lse_j is within
   η + ε_j of its exact log-sum-exp, and the weight each of its keys takes
   in o_j within a relative w_j = 2η + ε_j + (2b + ln tk + 5) u (nx.mli's
   form). The parts' Logsumexp adds ε_n to the greatest error among them:
   lse within η + ε_max + ε_n. A key's weight in the whole is its weight in
   o_j times exp (lse_j - lse): that factor's exponent errs by
   (η + ε_j) + (η + ε_max + ε_n) and its rounding u |lse_j - lse|, at most
   (2b + ln tk) u, and exp adds 5u. In all,
   4η + 2ε_j + ε_max + ε_n + 2 (2b + ln tk + 5) u, at most nx.mli's w with
   η doubled and ε = 3 ε_max + ε_n + (2b + ln tk + 5) u, which bounds lse
   too. The fused sum over n parts adds n roundings to the parts' chains:
   γ(tk + n + 1, 2u). *)
let attention_parts (type v s w q a d) eng ~by (route : unit Route.t) rule
    prog scale (acc : (float, a) D.t) (out : (w, q) D.t) (q : (v, s, d) Value.t)
    (kv : (v, s, d) Value.kv) (lengths : d Value.index option)
    (loads : d Value.any iarray) : (w, q, d) Value.t * (float, a, d) Value.t =
  let r = Prim.rank q in
  let lead = r - 3 in
  let p = route.result in
  let set = Devices.set p and ks = Grid.devices (Devices.grid p) in
  let qs = Prim.shape q in
  let factor, page, dv =
    match kv with
    | Dense { k; v } -> ((1, Prim.shape k, r - 2), 1, Prim.dim v (r - 1))
    | Paged { k; v; table } -> ((3, Prim.shape table, lead), Prim.dim k 2, Prim.dim v 3)
  in
  let operand, fshape, faxis = factor in
  let os = Array.append (Array.sub qs 0 (r - 1)) [| dv |] in
  let ls = Array.sub qs 0 (r - 1) in
  let tiles = tiles ~by route ~operand fshape [| faxis |] in
  let closing_of shape =
    closing ~by route ~operand ~axes:[ faxis ]
      ~extent:(fun a -> fshape.(a))
      shape
  in
  let co = closing_of os and cl = closing_of ls in
  let outw j = Devices.window ~by p os j in
  let here j = Devices.one set ks.(j) in
  let part j =
    let w = outw j and tile = tiles.(j).(0) in
    let k = ks.(j) and h = here j in
    let first = tile.M.start * page and keys = tile.count * page in
    let read i x =
      let window = reads rule i (Prim.shape x) w (fun _ -> tile) in
      view eng ~by k window x
    in
    (* The loads' keys, read whole where paged, are the device's. *)
    let read_load i x =
      let window = reads rule i (Prim.shape x) w (fun _ -> tile) in
      window.(r - 1) <- { M.start = first; count = keys; step = 1 };
      view eng ~by k window x
    in
    let q = read 0 q in
    let kv : (v, s, d) Value.kv =
      match kv with
      | Dense { k; v } -> Dense { k = read 1 k; v = read 2 v }
      | Paged { k; v; table } ->
          Paged { k = read 1 k; v = read 2 v; table = read 3 table }
    in
    let ix = match kv with Dense _ -> 3 | Paged _ -> 4 in
    let lengths =
      Option.map
        (fun l ->
          let l = read ix l in
          Prim.expect D.Int64
            (List.hd
               (map eng ~by h [ Value.Any l ]
                  [ E.any E.(input 0 D.Int64 - const D.Int64 (Int64.of_int first)) ])))
        lengths
    in
    let ix = ix + Option.fold ~none:0 ~some:(fun _ -> 1) lengths in
    let loads =
      Iarray.mapi (fun i (Value.Any x) -> Value.Any (read_load (ix + i) x)) loads
    in
    let start = Array.map (fun (x : M.range) -> x.start) w in
    start.(r - 1) <- first;
    let origin = { Prim.start; extents = [||] } in
    at eng ~by ~origin h
      (Value.Attention { prog; scale; acc; out = acc; q; kv; lengths; loads })
  in
  let parts = Array.init (Array.length ks) (fun j -> lazy (part j)) in
  let o j = fst (Lazy.force parts.(j)) and l j = snd (Lazy.force parts.(j)) in
  let lse =
    reduced eng ~by cl ls acc (Logsumexp (0, acc)) (stack eng ~by cl acc ls l)
  in
  let os_ = stack eng ~by co acc os o in
  let ls_ =
    stack eng ~by co acc os (fun j ->
        let x = l j in
        let s = Prim.shape (o j) in
        spread eng ~by s
          (reshape eng ~by (Array.append (Prim.shape x) [| 1 |]) x))
  in
  let sum k here w taken =
    let lw = Array.sub w 0 (r - 1) in
    let total = view eng ~by k lw lse in
    let total =
      spread eng ~by (counts w)
        (reshape eng ~by (Array.append (counts lw) [| 1 |]) total)
    in
    let weight =
      E.(
        where
          (input 2 acc = const acc Float.neg_infinity)
          (const acc 0.)
          (exp (input 1 acc - input 2 acc)))
    in
    let step running i =
      match taken.slot i with
      | [ oj; lj ] ->
          let ins = [ oj; lj; Value.Any total ] in
          let ins, prior =
            match running with
            | Some a -> (ins @ [ a ], E.input 3 acc)
            | None -> (ins, E.const acc 0.)
          in
          Some
            (List.hd
               (map eng ~by here ins
                  [ E.any (E.fma weight (E.input 0 acc) prior) ]))
      | _ -> invalid_arg "Partial.attention: a part's o and lse"
    in
    match List.fold_left step None (slots co) with
    | Some y -> [ cast eng ~by (D.Any out) y ]
    | None -> invalid_arg "Partial.attention: no part holds keys"
  in
  let o =
    closed eng ~by co os [ D.Any out ] [ Value.Any os_; Value.Any ls_ ] sum
  in
  (Prim.expect out (List.hd o), lse)

let attention : type r.
    Partition.engine -> by:string -> unit Route.t -> Route.rule -> r Value.prim -> r =
 fun eng ~by route rule op ->
  match op with
  | Value.Attention { prog; scale; acc; out; q; kv; lengths; loads } ->
      attention_parts eng ~by route rule prog scale acc out q kv lengths loads
  | _ -> invalid_arg "Partial.attention: not attention"

(* Scans *)

let scan (type d a) eng ~by (route : unit Route.t) rule layout axis prog
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
        view_load eng ~by ks.(j)
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
    let op, origin = Prim.within op (win j) in
    at eng ~by ~origin (here j) op
  in
  (* [x]'s last element along the axis, as its slot of the stack. *)
  let last (type v s) j (x : (v, s, d) Value.t) : (v, s, d) Value.t =
    let e = Prim.dim x axis in
    let slot = Devices.window ~by stacked stack_shape j in
    reshape eng ~by (counts slot) (slice eng ~by axis (e - 1) 1 x)
  in
  (* The totals [slot j] stacked: an empty piece on a device whose tile
     holds no term. *)
  let totals (type v s) (dt : (v, s) D.t)
      (slot : int -> (v, s, d) Value.t option) : (v, s, d) Value.t =
    eng.assemble stacked stack_shape dt (Array.init n slot)
  in
  (* Slot [i] of [x], a stack at [placed], on device [j]: its row of the
     results' shape. The slot is brought from [j]'s own piece, which holds
     every slot: a move of a window, whose start an engine may bind when
     the program runs, so that each device's program is every device's. *)
  let earlier (type v s) j i (x : (v, s, d) Value.t) : (v, s, d) Value.t =
    let held = Devices.window ~by placed stack_shape j in
    let w = Array.map (fun (r : M.range) -> { r with start = 0 }) held in
    w.(0) <- { M.start = i; count = 1; step = 1 };
    let row = Array.sub (counts w) 1 (Array.length shape) in
    let here = here j in
    reshape eng ~by row (eng.bring here [ (eng.piece x j here, w) ])
  in
  let wide j (Value.Any x : d Value.any) =
    Value.Any (spread eng ~by (counts (win j)) x)
  in
  let held j = not (Route.empty (win j)) in
  match r with
  | Monoid (m, o, dt) ->
      (* Each tile scans alone into the carrier; the totals' scan gives each
         device the combination of the tiles before its own, which it
         combines with its tile's scan, rounding once into [dt]. *)
      let (D.Any c) =
        Nx_kernel.carrier (P.dtype prog (Iarray.get prog.outs o))
      in
      let ys =
        values eng ~by p shape ~holds:held (fun j ->
            scanned j (Monoid (m, o, c)))
      in
      let prefix =
        let x = totals c (fun j -> Option.map (last j) ys.(j)) in
        let x = Partition.place eng ~by placed x in
        let op =
          Value.Scan
            {
              layout = L.contiguous (Iarray.of_array stack_shape);
              axis = 0;
              prog = Prim.identity c;
              reduction = Monoid (m, 0, c);
              loads = [| Plain x |];
              carry = None;
            }
        in
        at eng ~by placed op
      in
      let combined j =
        let y = Option.get ys.(j) and t = index j in
        let into =
          match D.equal_witness c dt with
          | Some Type.Equal -> Some (Value.Any y)
          | None -> None
        in
        if t = 0 then Prim.expect dt (cast eng ~by (D.Any dt) (Value.Any y))
        else
          let carry = earlier j (t - 1) prefix in
          let carry = spread eng ~by (counts (win j)) carry in
          let e = E.binary (binary m) (E.input 0 c) (E.input 1 c) in
          let v =
            map eng ~by ?into (here j) [ Value.Any carry; Value.Any y ]
              [ E.any (E.cast dt e) ]
          in
          Prim.expect dt (List.hd v)
      in
      eng.assemble p shape dt (values eng ~by p shape ~holds:held combined)
  | Arg (e, o, dt) ->
      (* Each tile scans alone, its positions from the axis's start; the
         first extreme of the tiles before its own, in tile order, combines
         with each of its results. The totals' scan by [Arg] gives each
         device that first extreme and the slot holding it, the first of
         the slots with its bits since the positions grow with the slots:
         every device computes the same scan and reads its slot, so that
         each device's program is every device's. *)
      let pair here (v, q) =
        (Partition.copy eng here v, Partition.copy eng here q)
      in
      let pairs =
        replicated ~by p shape ~holds:held ~copy:pair ~compute:(fun j ->
            let v, q = scanned j (Arg (e, o, dt)) in
            let s = (win j).(axis).start in
            let q = shift eng ~by E.( + ) s shape.(axis) (Value.Any q) in
            (v, Prim.expect D.Int64 q))
      in
      let xs =
        totals dt (fun j -> Option.map (fun (v, _) -> last j v) pairs.(j))
      and qs =
        totals D.Int64 (fun j -> Option.map (fun (_, q) -> last j q) pairs.(j))
      in
      let xs = Partition.place eng ~by placed xs
      and qs = Partition.place eng ~by placed qs in
      let firsts, slots =
        at eng ~by placed
          (Value.Scan
             {
               layout = L.contiguous (Iarray.of_array stack_shape);
               axis = 0;
               prog = Prim.identity dt;
               reduction = Arg (e, 0, dt);
               loads = [| Plain xs |];
               carry = None;
             })
      in
      let positions =
        at eng ~by placed (Value.Gather { axis = 0; idx = slots; x = qs })
      in
      let result j =
        let v, q = Option.get pairs.(j) and t = index j in
        if t = 0 then (v, q)
        else
          let carry =
            [
              Value.Any (earlier j (t - 1) firsts);
              Value.Any (earlier j (t - 1) positions);
            ]
          in
          let carry = List.map (wide j) carry in
          let last = [ Value.Any v; Value.Any q ] in
          match first_of eng ~by e carry last with
          | [ x; y ] -> (Prim.expect dt x, Prim.expect D.Int64 y)
          | _ -> invalid_arg "Partial.scan: an extreme and its position"
      in
      let results =
        replicated ~by p shape ~holds:held ~compute:result ~copy:pair
      in
      ( eng.assemble p shape dt (Array.map (Option.map fst) results),
        eng.assemble p shape D.Int64 (Array.map (Option.map snd) results) )
  | Logsumexp _ | Moments _ ->
      invalid_arg "Partial.scan: a scan that reads its axis whole"

(* Contractions *)

(* The ids of a device's tile of [Rows]' groups from [start], its window
   [ids] of the whole's: group [e] of the tile is group [start + e] of the
   whole, so its ids are the whole's less [start], an id outside the tile
   then outside [\[0, count)] ({!Nx_kernel.type-segments}). *)
let local_ids (type d) eng ~by (here : d Devices.placement) start
    (ids : d Value.index) : d Value.index =
  if start = 0 then ids
  else
    let s = E.const D.Int64 (Int64.of_int start) in
    let e = E.any E.(input 0 D.Int64 - s) in
    Prim.expect D.Int64 (List.hd (map eng ~by here [ Value.Any ids ] [ e ]))

(* A contraction's partials closed where no sum is asked for: each device's
   [partial j], its slot in the accumulator [ad], added in C order of the
   tiles' first indices and rounded once into [out]. *)
let closed_sum (type v s d) eng ~by route ~operand ~folded ~of_folded shape
    (out : (v, s) D.t) (ad : (_, _) D.t) partial : (v, s, d) Value.t =
  let cl =
    closing ~by route ~operand ~axes:(Array.to_list folded)
      ~extent:(Array.get of_folded) shape
  in
  let s = stack eng ~by cl ad shape partial in
  reduced eng ~by cl shape out (Monoid (Sum, 0, out)) s

let contract (type v s d a b c e) eng ~by (route : unit Route.t) rule ~batch
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
  (* The folded axes: [a]'s of the pairs, or [b]'s of the pairs then of
     [Rows]' groups, the factor after the pairs. *)
  let operand, folded, of_folded =
    match (segments : d Value.index Nx_kernel.segments option) with
    | None -> (0, Array.map fst pairs, Prim.shape a)
    | Some (Rows { b_axis; _ }) ->
        (1, Array.append (Array.map snd pairs) [| b_axis |], Prim.shape b)
    | Some (Contracting _) ->
        invalid_arg "Partial.contract: groups of a contracting pair"
  in
  let tiles = tiles ~by route ~operand of_folded folded in
  (* Operand [i] on device [j], its factors' ranges [r]. *)
  let read (type w q) j i r (x : (w, q, d) Value.t) =
    let w = reads rule i (Prim.shape x) (resw j) (Array.get r) in
    view eng ~by ks.(j) w x
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
        | Becomes _ | Whole | Reduced _ | Scanned _ | Indexed _ | Regroups _ ->
            ()
      in
      (match rule with
      | Axes t -> Array.iteri narrow t.(i)
      | Elementwise -> ());
      view eng ~by ks.(j) w v
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
      (match segments with
      | Some (Rows { b_axis; _ }) -> [| run b b_axis b_scale |]
      | Some (Contracting _) | None -> [||])
  in
  (* The groups of a piece [r] on device [j]: [Rows]' ids of its rows,
     numbered from its range of the groups. *)
  let groups j r =
    match segments with
    | None -> None
    | Some (Rows g) ->
        let i = b_at + Bool.to_int (Option.is_some b_scale) in
        let w = reads rule i (Prim.shape g.ids) (resw j) (Array.get r) in
        let q = r.(Array.length pairs) in
        let ids = view eng ~by ks.(j) w g.ids in
        let ids = local_ids eng ~by (here j) q.start ids in
        Some (Nx_kernel.Rows { g with ids })
    | Some (Contracting _) ->
        invalid_arg "Partial.contract: groups of a contracting pair"
  in
  let (D.Any ad) = acc in
  (* The contraction of a piece [r] of a tile on device [j] into the
     accumulator, from [from], the sum of the pieces before it. *)
  let piece j from r =
    at eng ~by (here j)
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
           sum = false;
         })
  in
  (* A device's partial: its tile's pieces in C order, the first from
     [from] and each other from the sum of those before it. A piece of
     [Rows]' groups gives no products to a row of another group, which so
     keeps the sum before it, bit for bit. *)
  let partial j from =
    match tile_pieces runs tiles.(j) with
    | r :: rs ->
        let next s r = piece j (Some s) r in
        List.fold_left next (piece j from r) rs
    | [] -> invalid_arg "Partial.contract: a tile of no piece"
  in
  (* Under [Rows], an output's products lie on the device holding its row's
     group: [init] there, as the accumulator holds it, and [zero]
     elsewhere, the first device also taking the rows in no group. The ids,
     along the batch and [a_axis], are laid over the result's axes: the
     batch's, then [a]'s free axes, [a_axis] among them, then [b]'s. *)
  let owned j ~first (g : _ Nx_kernel.segments) i zero =
    match g with
    | Contracting _ ->
        invalid_arg "Partial.contract: groups of a contracting pair"
    | Rows { ids; a_axis; b_axis } ->
        let shape = counts (resw j) in
        let nb = Iarray.length batch in
        let taken ax =
          Iarray.exists (fun (x, _) -> x = ax) batch
          || Iarray.exists (fun (x, _) -> x = ax) contracting
        in
        let before = ref 0 in
        for ax = 0 to a_axis - 1 do
          if not (taken ax) then incr before
        done;
        let perm = Array.make (Array.length shape) (-1) in
        for d = 0 to nb - 1 do
          perm.(d) <- d
        done;
        perm.(nb + !before) <- nb;
        let k = b_at + Bool.to_int (Option.is_some b_scale) in
        let w = reads rule k (Prim.shape ids) (resw j) (Array.get tiles.(j)) in
        let ids = view eng ~by ks.(j) w ids in
        let ids = eng.view (M.Permute (Iarray.of_array perm)) ids in
        let ids = eng.view (M.Broadcast (Iarray.of_array shape)) ids in
        let q = tiles.(j).(Array.length pairs) in
        let c n = E.const D.Int64 (Int64.of_int n) in
        let id = E.input 1 D.Int64 in
        let hi = c (q.start + q.count) in
        let mine =
          if first then E.(id < hi || id >= c (Prim.dim b b_axis))
          else E.(id >= c q.start && id < hi)
        in
        let e = E.any (E.where mine (E.input 0 ad) (E.input 2 ad)) in
        let vs = [ Value.Any i; Value.Any ids; Value.Any zero ] in
        Prim.expect ad (List.hd (map eng ~by (here j) vs [ e ]))
  in
  (* The [init] device [j]'s term adds, where its sum stays partial:
     [init] where [first], else the identity, so that [init] joins the sum
     once, in the kernel that adds the first products; none without
     one. *)
  let from j ~first =
    let init_at i =
      let i = Value.Any (view eng ~by ks.(j) (resw j) i) in
      Prim.expect ad (cast eng ~by acc i)
    in
    let zero () = Partition.identity eng ~by (here j) ad (counts (resw j)) in
    match (init, segments) with
    | None, _ -> None
    | Some i, Some g -> Some (owned j ~first g (init_at i) (zero ()))
    | Some i, None when first -> Some (init_at i)
    | Some _, None -> Some (zero ())
  in
  if not (Devices.partial p) then
    match init with
    | None ->
        closed_sum eng ~by route ~operand ~folded ~of_folded shape out ad
          (fun j -> partial j None)
    | Some i ->
        (* Every partial adds the identity, which keeps an output without
           products at [-0]; the closed sum, in the accumulator, then adds
           [init] and rounds once into the result's dtype, so that [init]
           joins after every product, as on one device. *)
        let zero j =
          Some (Partition.identity eng ~by (here j) ad (counts (resw j)))
        in
        let sum =
          closed_sum eng ~by route ~operand ~folded ~of_folded shape ad ad
            (fun j -> partial j (zero j))
        in
        let joined j =
          let s = Value.Any (view eng ~by ks.(j) (resw j) sum) in
          let i = cast eng ~by acc (Value.Any (view eng ~by ks.(j) (resw j) i)) in
          let e = E.any (E.cast out E.(input 0 ad + input 1 ad)) in
          Prim.expect out (List.hd (map eng ~by (here j) [ s; i ] [ e ]))
        in
        let holds j = not (Route.empty (resw j)) in
        eng.assemble p shape out (values eng ~by p shape ~holds joined)
  else
    (* Each device's term, its window of the result: its partial, from
       [init] where its term is its group's first. *)
    let g = Devices.grid p in
    let term j = partial j (from j ~first:(Grid.term g j = 0)) in
    (* Each term rounds once from the accumulator into the result's dtype. *)
    let rounded j =
      Prim.expect out (cast eng ~by (D.Any out) (Value.Any (term j)))
    in
    let holds j = not (Route.empty (resw j)) in
    eng.assemble p shape out (values eng ~by p shape ~holds rounded)

(* Gathers *)

(* The dtype of [dt]'s bits whose maximum of one element and zeros is that
   element: [bit] itself, else the unsigned integer of [dt]'s width, two
   [uint64]s for a [complex128]. *)
let bits_of (type v s) (dt : (v, s) D.t) : D.any =
  match D.bits dt with
  | 1 -> D.Any D.Bit
  | 4 -> D.Any D.Uint4
  | 8 -> D.Any D.Uint8
  | 16 -> D.Any D.Uint16
  | 32 -> D.Any D.Uint32
  | 64 | 128 -> D.Any D.Uint64
  | n -> invalid_argf "Partial.bits_of: a dtype of %d bits" n

let gather (type v s d) eng ~by (route : unit Route.t) rule axis
    (idx : d Value.index) (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  let p = route.result in
  let set = Devices.set p and ks = Grid.devices (Devices.grid p) in
  let out = Prim.shape idx and sx = Prim.shape x in
  let dt = Prim.dtype x in
  let (D.Any u) = bits_of dt in
  let parts = D.bits dt / D.bits u in
  let bits = if parts = 1 then out else Array.append out [| parts |] in
  let tiles = tiles ~by route ~operand:1 sx [| axis |] in
  let cl =
    closing ~by route ~operand:1 ~axes:[ axis ] ~extent:(fun _ -> sx.(axis)) out
  in
  (* Device [j] gathers from its tile at the positions moved to the tile's
     start, one outside it reading zero bits, and reads the result's bits as
     [u]. *)
  let own j =
    let w = Devices.window ~by p out j and t = tiles.(j).(0) in
    let here = Devices.one set ks.(j) in
    let i = Value.Any (view eng ~by ks.(j) w idx) in
    let i = shift eng ~by E.( - ) t.start sx.(axis) i in
    let x = view eng ~by ks.(j) (reads rule 1 sx w (fun _ -> t)) x in
    let idx = Prim.expect D.Int64 i in
    let y = at eng ~by here (Value.Gather { axis; idx; x }) in
    at eng ~by here (Value.Bitcast (u, y))
  in
  (* Only the tile holding a position reads bits other than zeros there. *)
  let s = stack eng ~by cl u bits own in
  let y = reduced eng ~by cl bits u (Monoid (Max, 0, u)) s in
  on eng ~by y (Value.Bitcast (dt, y))

(* A range gather of [x] cut along its axis [a] closes as a gather does:
   device [j] gathers the runs from its tile, their starts moved to the
   tile's start, so that each position reads its element in the one tile
   holding it and zero bits in every other. Positions are [int64] sums
   modulo 2^64, so the move commutes with them however they wrap. *)
let gather_ranges (type v s d) eng ~by (route : unit Route.t) rule size
    (starts : d Value.index) (ends : d Value.index) (x : (v, s, d) Value.t) :
    (v, s, d) Value.t =
  let p = route.result in
  let set = Devices.set p and ks = Grid.devices (Devices.grid p) in
  let ss = Prim.shape starts and sx = Prim.shape x in
  let a = Array.length ss - 1 in
  let out = Array.copy sx in
  out.(a) <- size;
  let dt = Prim.dtype x in
  let (D.Any u) = bits_of dt in
  let parts = D.bits dt / D.bits u in
  let bits = if parts = 1 then out else Array.append out [| parts |] in
  let tiles = tiles ~by route ~operand:2 sx [| a |] in
  let cl =
    closing ~by route ~operand:2 ~axes:[ a ] ~extent:(fun _ -> sx.(a)) out
  in
  let own j =
    let w = Devices.window ~by p out j and t = tiles.(j).(0) in
    let here = Devices.one set ks.(j) in
    let runs i y = view eng ~by ks.(j) (Route.window rule ~operand:i ss w) y in
    let s = Value.Any (runs 0 starts) in
    let s = shift eng ~by E.( - ) t.start sx.(a) s in
    let starts = Prim.expect D.Int64 s and ends = runs 1 ends in
    let x = view eng ~by ks.(j) (reads rule 2 sx w (fun _ -> t)) x in
    let y = at eng ~by here (Value.Gather_ranges { size; starts; ends; x }) in
    at eng ~by here (Value.Bitcast (u, y))
  in
  let s = stack eng ~by cl u bits own in
  let y = reduced eng ~by cl bits u (Monoid (Max, 0, u)) s in
  on eng ~by y (Value.Bitcast (dt, y))

(* Scatters *)

let scatter (type v s d) eng ~by (route : unit Route.t) rule combine unique axis
    (idx : d Value.index) (updates : (v, s, d) Value.t)
    (into : (v, s, d) Value.t) : (v, s, d) Value.t =
  let p = route.result in
  let set = Devices.set p and ks = Grid.devices (Devices.grid p) in
  let shape = Prim.shape into in
  (* Device [j] writes into its window the updates at the positions moved
     to the window's start: one outside it writes nothing. *)
  let w j = Devices.window ~by p shape j in
  let own j =
    let w = w j and here = Devices.one set ks.(j) in
    let read i x =
      view eng ~by ks.(j) (Route.window rule ~operand:i (Prim.shape x) w) x
    in
    let i = Value.Any (read 0 idx) in
    let i = shift eng ~by E.( - ) w.(axis).start shape.(axis) i in
    let idx = Prim.expect D.Int64 i in
    let updates = read 1 updates and into = read 2 into in
    at eng ~by here (Value.Scatter { combine; unique; axis; idx; updates; into })
  in
  let holds j = not (Route.empty (w j)) in
  eng.assemble p shape (Prim.dtype into) (values eng ~by p shape ~holds own)

(* A scatter of updates cut along its axis: device [j] writes its tile of
   them, at the positions moved to its window's start, into its window,
   and the tiles' slots close in tile order. Two updates of one target
   differ only along the axis, so the later tile holds the later in C
   order.
   - [Add], in the carrier: the first tile into [into]'s window, every
     other into [-0], which adds nothing; the slots close by one map adding
     them, with no [+0] added, and the result rounds once.
   - [Max] and [Min]: every tile into [into]'s window, which the monoid's
     idempotence keeps; the slots close by the monoid, whose order puts
     [-0] below [+0], so its result has the bits a sequence of the updates
     gives, a NaN for a NaN.
   - [Set]: every tile into [into]'s window, beside a flag of the targets
     it wrote; a later tile's flagged target replaces the earlier's, so a
     target keeps its last tile's last update. *)
let summed (type v s d) eng ~by (route : unit Route.t) rule
    (combine : Nx_kernel.combine) unique axis (idx : d Value.index)
    (updates : (v, s, d) Value.t) (into : (v, s, d) Value.t) :
    (v, s, d) Value.t =
  let p = route.result in
  let set = Devices.set p and ks = Grid.devices (Devices.grid p) in
  let shape = Prim.shape into and su = Prim.shape updates in
  let dt = Prim.dtype into in
  let tiles = tiles ~by route ~operand:1 su [| axis |] in
  let cl =
    closing ~by route ~operand:1 ~axes:[ axis ] ~extent:(fun _ -> su.(axis))
      shape
  in
  (* Device [j]'s scatter of its tile, [f] of its tile's start, its
     here, its positions and its operands read there. *)
  let own (type a b) j
      (f : int -> d Devices.placement -> d Value.index -> (v, s, d) Value.t ->
        (v, s, d) Value.t -> (a, b, d) Value.t) : (a, b, d) Value.t =
    let w = Devices.window ~by p shape j and t = tiles.(j).(0) in
    let here = Devices.one set ks.(j) in
    let read i x =
      view eng ~by ks.(j) (reads rule i (Prim.shape x) w (fun _ -> t)) x
    in
    let i = Value.Any (read 0 idx) in
    let i = shift eng ~by E.( - ) w.(axis).start shape.(axis) i in
    f t.start here (Prim.expect D.Int64 i) (read 1 updates) (read 2 into)
  in
  let scatter here c ?into:fresh idx updates into =
    at eng ~by ?into:fresh here
      (Value.Scatter { combine = c; unique; axis; idx; updates; into })
  in
  let close stacks dts f =
    closed eng ~by cl shape dts stacks (fun _ here _ taken ->
        f here taken)
  in
  match combine with
  | Add ->
      let (D.Any c) = Nx_kernel.carrier (D.Any dt) in
      let tile j =
        own j (fun start here idx updates into ->
            let first =
              if start = 0 then E.any (E.cast c (E.input 0 dt))
              else E.any (E.neg (E.const c (D.zero c)))
            in
            let target = map eng ~by here [ Value.Any into ] [ first ] in
            let into = Prim.expect c (List.hd target) in
            let updates = cast eng ~by (D.Any c) (Value.Any updates) in
            let updates = Prim.expect c updates in
            (* The target is fresh: the updates add into its memory. *)
            scatter here Add ~into:(Value.Any into) idx updates into)
      in
      let s = Value.Any (stack eng ~by cl c shape tile) in
      Prim.expect dt
        (List.hd
           (close [ s ] [ D.Any dt ] (fun here taken ->
                let total = running cl taken (combined eng ~by here Sum) in
                [ cast eng ~by (D.Any dt) (List.hd total) ])))
  | Max | Min ->
      let m : Nx_kernel.monoid = if combine = Max then Max else Min in
      let tile j =
        own j (fun _ here idx updates into ->
            scatter here combine idx updates into)
      in
      let s = Value.Any (stack eng ~by cl dt shape tile) in
      Prim.expect dt
        (List.hd
           (close [ s ] [ D.Any dt ] (fun here taken ->
                running cl taken (combined eng ~by here m))))
  | Set ->
      let tile j =
        own j (fun _ here idx updates into ->
            scatter here Set idx updates into)
      in
      (* The targets a tile wrote: [true] scattered over [false]. *)
      let wrote j =
        own j (fun _ here idx updates into ->
            let constant v x =
              Prim.expect D.Bool
                (List.hd
                   (map eng ~by here [ Value.Any x ] [ E.any (E.const D.Bool v) ]))
            in
            scatter here Set idx (constant true updates) (constant false into))
      in
      let v = Value.Any (stack eng ~by cl dt shape tile)
      and w = Value.Any (stack eng ~by cl D.Bool shape wrote) in
      let later here acc slot =
        match (acc, slot) with
        | [ a; fa ], [ b; fb ] ->
            let x = E.input 0 dt and y = E.input 2 dt in
            let f = E.input 1 D.Bool and g = E.input 3 D.Bool in
            map eng ~by here [ a; fa; b; fb ]
              [ E.any (E.where g y x); E.any E.(f || g) ]
        | _ -> invalid_arg "Partial.summed: a value and its flag each"
      in
      Prim.expect dt
        (List.hd
           (close [ v; w ] [ D.Any dt ] (fun here taken ->
                [ List.hd (running cl taken (later here)) ])))

(* Sums *)

(* [x], a sum, closed at [p], which is no sum. Each device of a group adds
   the terms of its part, as a fold's slots add: [p] itself where [p] cuts an
   axis over the grid axes [x] sums over alone, a reduce-scatter; else its
   window of [x]'s placement closed, cut along the first axis it keeps whole
   that holds a term per device of a group, which each device then gathers,
   an all-reduce, then placed at [p]. *)
let close (type v s d e) eng ~by (p : e Devices.placement)
    (x : (v, s, d) Value.t) : (v, s, e) Value.t =
  let at = Prim.placement x and shape = Prim.shape x and dt = Prim.dtype x in
  let g = Devices.grid at and set = Devices.set at in
  let unsum g axis =
    match Grid.unsum g ~axis with
    | Ok u -> Devices.v ~by set u
    | Error e -> invalid_argf "%s: Partial.close: %s" by e
  in
  let whole = Devices.closed at in
  let e = Grid.extents g in
  let n = Array.fold_left (fun n a -> n * e.(a)) 1 (Grid.sums g) in
  let free a = not (Array.exists (fun (b, _) -> a = b) (Grid.cuts g)) in
  let scatters (a, _) =
    if free a && Devices.equal (unsum g a) p then Some p else None
  in
  let result, parts =
    match Array.find_map scatters (Grid.cuts (Devices.grid p)) with
    | Some p -> (p, p)
    | None -> (
        let holds a = free a && shape.(a) >= n in
        match List.find_opt holds (List.init (Array.length shape) Fun.id) with
        | Some a -> (whole, unsum g a)
        | None -> (whole, whole))
  in
  let stacked = unsum (Grid.map_axes (( + ) 1) g) 0 in
  let c = { result; stacked; parts; held = [ n ] } in
  let y =
    if numel shape = 0 then
      eng.Partition.assemble result shape dt
        (Array.make (Grid.count (Devices.grid result)) None)
    else
      let ks = Grid.devices g in
      let term j = eng.Partition.piece x j (Devices.one set ks.(j)) in
      let s = Value.Any (stack eng ~by c dt shape term) in
      let added _ here _ taken = running c taken (combined eng ~by here Sum) in
      Prim.expect dt (List.hd (closed eng ~by c shape [ D.Any dt ] [ s ] added))
  in
  Partition.place eng ~by p y

(* [x], which is no sum, at the sum [p]: on each device whose term is the
   first of its group its window, on every other the identity. *)
let spread_terms (type v s d e) eng ~by (p : e Devices.placement)
    (x : (v, s, d) Value.t) : (v, s, e) Value.t =
  let shape = Prim.shape x and dt = Prim.dtype x in
  Prim.summable ~by dt;
  let y = Partition.place eng ~by (Devices.closed p) x in
  let g = Devices.grid p and set = Devices.set p in
  let ks = Grid.devices g in
  let term j =
    let here = Devices.one set ks.(j) and w = Devices.window ~by p shape j in
    if Route.empty w then None
    else if Grid.term g j = 0 then Some (eng.Partition.piece y j here)
    else Some (Partition.identity eng ~by here dt (counts w))
  in
  eng.assemble p shape dt (Array.init (Array.length ks) term)

let place eng ~by p x =
  let at = Prim.placement x in
  if Devices.equal at p then Partition.place eng ~by p x
  else if Devices.partial p then
    let x =
      if Devices.partial at then close eng ~by (Devices.closed at) x else x
    in
    spread_terms eng ~by p x
  else if Devices.partial at then close eng ~by p x
  else Partition.place eng ~by p x

let run (type r) eng ~by route (op : r Value.prim) : r =
  let rule = Prim.rule op in
  match op with
  | Value.Gather { axis; idx; x } -> gather eng ~by route rule axis idx x
  | Value.Gather_ranges { size; starts; ends; x } ->
      gather_ranges eng ~by route rule size starts ends x
  | Value.Scatter { combine; unique; axis; idx; updates; into } ->
      (* The updates are read cut along the axis only where the route
         closes their tiles by partials. *)
      let cuts = Grid.cuts (Devices.grid route.operands.(1)) in
      if Array.exists (fun (a, _) -> a = axis) cuts then
        summed eng ~by route rule combine unique axis idx updates into
      else scatter eng ~by route rule combine unique axis idx updates into
  | Value.Reduce { layout; axes; prog; reductions; loads } ->
      reduce eng ~by route rule layout axes prog reductions loads
  | Value.Scan { layout; axis; prog; reduction; loads; carry = None } ->
      scan eng ~by route rule layout axis prog reduction loads
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
        sum = _;
      } ->
      contract eng ~by route rule ~batch ~contracting ~acc ~segments out a b
        init
        a_scale b_scale
  | Value.Contract { segments = Some (Contracting _); _ } ->
      invalid_arg "Partial.run: groups of a contracting pair by partials"
  | Value.Scan { carry = Some _; _ }
  | Value.Map _ | Value.Sort _ | Value.Unique _ | Value.Fft _
  | Value.Linalg _ | Value.Fold _ | Value.Assemble _ | Value.Copy _
  | Value.Move _ | Value.Bitcast _ | Value.Attention _ ->
      attention eng ~by route rule op
  | Value.Place _ | Value.Check _ ->
      invalid_arg "Partial.run: an operation no route computes by partials"
