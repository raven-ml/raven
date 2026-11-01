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
   too. The first result takes [into] where it is given. *)
let at : type r d.
    Partition.engine ->
    by:string ->
    ?into:d Value.any ->
    unit Devices.placement ->
    r Value.prim ->
    r =
 fun eng ~by ?into p op ->
  let layouts = Prim.layouts ~by op in
  if Grid.count (Devices.grid p) = 1 && Prim.lies_at p op then
    eng.run ~by ?into p layouts op
  else
    let (Prim.Operands xs) = Prim.operands op in
    let operands =
      Array.of_list (List.map (fun (Value.Any x) -> Prim.placement x) xs)
    in
    let route = { Route.operands; result = p; compute = Windows } in
    Partition.windows eng ~by ?into route layouts op

(* [op] at [x]'s placement. *)
let on eng ~by x op = at eng ~by (Prim.placement x) op

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

(* Elements [start] down to [start - count + 1] of [x] along [axis], in that
   order. *)
let backwards (type v s d) eng ~by axis start count (x : (v, s, d) Value.t) :
    (v, s, d) Value.t =
  let r = Array.map Route.full (Prim.shape x) in
  r.(axis) <- { M.start; count; step = -1 };
  on eng ~by x (Value.Move (Slice (Iarray.of_array r), x))

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
   its results at [p]. *)
let map (type d) eng ~by ?into p (vs : d Value.any list) exprs :
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
  read (at eng ~by ?into p (Value.Map { layout; prog; outs; loads; at = None }))

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
      Value.Any (fst (at eng ~by here (Prim.within map w)))

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

(* [pos] shifted by [s]. *)
let shift eng ~by s pos =
  if s = 0 then pos
  else
    let (Value.Any x) = pos in
    let s = E.const D.Int64 (Int64.of_int s) in
    List.hd
      (map eng ~by (Prim.placement x) [ pos ] [ E.any E.(input 0 D.Int64 + s) ])

(* [pos], positions in C order of the folded indices of a window [w] of
   the folded extents [whole], as positions in C order of the whole's,
   over [pos]'s memory: [pos] is fresh, on one device. *)
let renumber eng ~by (w : M.range array) whole pos =
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
    let p = Prim.placement x in
    List.hd (map eng ~by ~into:pos p [ pos ] [ E.any !global ])

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
  let ks = Grid.devices (Devices.grid c.stacked) in
  let shape = Array.append (Array.of_list c.held) shape in
  let slot j =
    let w = Devices.window ~by c.stacked shape j in
    if Route.empty w then None
    else Some (eng.Partition.view (M.Reshape (Route.extents w)) (partial j))
  in
  eng.assemble c.stacked shape dt (Array.init (Array.length ks) slot)

(* The most bytes of slots a piece gathers: a buffer of fixed size, so that
   a closing holds no more however large its result. A piece costs about
   1,100 words and 12 us of calls on four memory devices, and reducing 256
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

(* [f]'s results, of [dts], at the route's result, from [stacks], each at
   [c.stacked], of the results of [shape] after their leading axes. Each
   device computes its part, its window at [c.parts], piece by piece: it
   takes each stack's slots at a piece [w], and [f k here w] computes the
   piece from them on the set's device [k] alone, at [here]. Each device of
   the result then gathers its window from the pieces, its own where it
   holds them. So a device holds, beside its slots, one piece's slots and
   what [f] makes of them at a time, then its pieces and its window of the
   results. A part of a dtype narrower than a byte on a device whose memory
   the host does not address, where a piece may start inside a byte, is
   one piece. *)
let closed (type d) (eng : Partition.engine) ~by c shape dts
    (stacks : d Value.any list)
    (f :
      int ->
      d Devices.placement ->
      M.range array ->
      d Value.any list ->
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
      let sw = Array.append lead p in
      let slots (Held (s, h)) =
        if Route.empty sw then
          Value.Any
            (eng.Partition.assemble here (counts sw) (Prim.dtype s) [| None |])
        else Value.Any (Partition.take eng h here sw)
      in
      (p, f k here p (List.map slots held))
    in
    (k, List.map piece ps)
  in
  let parts = Array.mapi device (Grid.devices (Devices.grid c.parts)) in
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
   [dt]: the reduction itself by default. [then_ k here w] finishes the
   piece [w] on the set's device [k], at [here]. *)
let reduced (type v s w q a b d) eng ~by c shape ?(then_ = fun _ _ _ y -> y)
    (dt : (w, q) D.t) (r : (d, (a, b, d) Value.t) Value.reduction)
    (s : (v, s, d) Value.t) : (w, q, d) Value.t =
  let lead = List.length c.held in
  let f k here w = function
    | [ Value.Any s ] ->
        [ then_ k here w (Value.Any (fold_slots eng ~by ~lead here r s)) ]
    | _ -> invalid_arg "Partial.reduced: one stack"
  in
  let ys = closed eng ~by c shape [ D.Any dt ] [ Value.Any s ] f in
  Prim.expect dt (List.hd ys)

(* The slot at [index] of [x], a stack on one device: a view. *)
let slot (type v s d) (eng : Partition.engine) index (x : (v, s, d) Value.t) :
    (v, s, d) Value.t =
  let lead = List.length index in
  let shape = Prim.shape x in
  let out = Array.sub shape lead (Array.length shape - lead) in
  let r = Array.map Route.full shape in
  List.iteri (fun i k -> r.(i) <- { M.start = k; count = 1; step = 1 }) index;
  let v = eng.view (Slice (Iarray.of_array r)) x in
  eng.view (Reshape (Iarray.of_array out)) v

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
    at eng ~by (here j) (Prim.within op (loop j))
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
          let q = renumber eng ~by tiles.(j) whole (Value.Any q) in
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
        let first _ _ _ = function
          | [ Value.Any xs; Value.Any qs ] -> (
              let ranked i =
                [ Value.Any (slot eng i xs); Value.Any (slot eng i qs) ]
              in
              let fold =
                match slots cl with
                | i :: rest ->
                    List.fold_left
                      (fun a i -> first_of eng ~by e a (ranked i))
                      (ranked i) rest
                | [] -> invalid_arg "Partial.reduce: no tile holds terms"
              in
              match fold with
              | [ x; q ] -> [ cast eng ~by (D.Any dt) x; q ]
              | _ -> invalid_arg "Partial.reduce: an extreme and its position")
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
        let lead = List.length cl.held in
        let var _ here _ = function
          | [ Value.Any a; Value.Any b ] ->
              let sum s =
                let r = Value.Monoid (Sum, 0, c) in
                Value.Any (fold_slots eng ~by ~lead here r s)
              in
              variance eng ~by (D.Any dt) n (sum a) (sum b)
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
    at eng ~by (here j) (Prim.within op (win j))
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
     results' shape. *)
  let earlier (type v s) j i (x : (v, s, d) Value.t) : (v, s, d) Value.t =
    let w = Devices.window ~by placed stack_shape j in
    w.(0) <- { M.start = i; count = 1; step = 1 };
    let row = Array.sub (counts w) 1 (Array.length shape) in
    reshape eng ~by row (view eng ~by ks.(j) w x)
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
        Array.init n (fun j ->
            if held j then Some (scanned j (Monoid (m, o, c))) else None)
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
        match ys.(j) with
        | None -> None
        | Some y ->
            let t = index j in
            let into =
              match D.equal_witness c dt with
              | Some Type.Equal -> Some (Value.Any y)
              | None -> None
            in
            if t = 0 then
              Some (Prim.expect dt (cast eng ~by (D.Any dt) (Value.Any y)))
            else
              let carry = earlier j (t - 1) prefix in
              let carry = spread eng ~by (counts (win j)) carry in
              let e = E.binary (binary m) (E.input 0 c) (E.input 1 c) in
              let v =
                map eng ~by ?into (here j) [ Value.Any carry; Value.Any y ]
                  [ E.any (E.cast dt e) ]
              in
              Some (Prim.expect dt (List.hd v))
      in
      eng.assemble p shape dt (Array.init n combined)
  | Arg (e, o, dt) ->
      (* Each tile scans alone, its positions from the axis's start; the
         first extreme of the tiles before its own, in tile order, combines
         with each of its results. *)
      let pairs =
        Array.init n (fun j ->
            if held j then
              let v, q = scanned j (Arg (e, o, dt)) in
              let q = shift eng ~by (win j).(axis).start (Value.Any q) in
              Some (v, Prim.expect D.Int64 q)
            else None)
      in
      let xs =
        totals dt (fun j -> Option.map (fun (v, _) -> last j v) pairs.(j))
      and qs =
        totals D.Int64 (fun j -> Option.map (fun (_, q) -> last j q) pairs.(j))
      in
      let xs = Partition.place eng ~by placed xs
      and qs = Partition.place eng ~by placed qs in
      let results =
        Array.init n (fun j ->
            match pairs.(j) with
            | None -> None
            | Some (v, q) -> (
                let t = index j in
                if t = 0 then Some (v, q)
                else
                  let ranked i =
                    [ Value.Any (earlier j i xs); Value.Any (earlier j i qs) ]
                  in
                  let carry = ref (ranked 0) in
                  for i = 1 to t - 1 do
                    carry := first_of eng ~by e !carry (ranked i)
                  done;
                  let carry = List.map (wide j) !carry in
                  let last = [ Value.Any v; Value.Any q ] in
                  match first_of eng ~by e carry last with
                  | [ x; y ] ->
                      Some (Prim.expect dt x, Prim.expect D.Int64 y)
                  | _ ->
                      invalid_arg "Partial.scan: an extreme and its position"))
      in
      ( eng.assemble p shape dt (Array.map (Option.map fst) results),
        eng.assemble p shape D.Int64 (Array.map (Option.map snd) results) )
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
let fences (type d) eng ~by k (here : d Devices.placement) w (q : M.range)
    (o : d Value.index) : d Value.index =
  let last = Array.length w - 1 in
  let tail = (Prim.shape o).(last) - q.start in
  let along r =
    let w = Array.copy w in
    w.(last) <- r;
    view eng ~by k w o
  in
  let down =
    backwards eng ~by last (tail - 1) tail
      (along { M.start = q.start; count = tail; step = 1 })
  in
  let first =
    spread eng ~by (Prim.shape down)
      (along { M.start = 0; count = 1; step = 1 })
  in
  let ins : D.any iarray = [| D.Any D.Int64; D.Any D.Int64 |] in
  let term = E.maximum (E.input 0 D.Int64) (E.input 1 D.Int64) in
  let least =
    at eng ~by here
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
  backwards eng ~by last (tail - 1) (q.count + 1) least

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
        let whole (x, _) = Route.full (Prim.dim a x) in
        Array.append (Array.map whole pairs) tiles.(j)
  in
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
        let offsets = fences eng ~by ks.(j) (here j) w q o in
        Some (Nx_kernel.Rows { g with offsets })
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
        List.fold_left from (piece j None r) rs
    | [] -> invalid_arg "Partial.contract: a tile of no piece"
  in
  let s = stack eng ~by cl ad shape partial in
  match init with
  | None -> reduced eng ~by cl shape out (Monoid (Sum, 0, out)) s
  | Some i ->
      (* The sums in the accumulator, then [init] as it holds it, rounded
         once. A segmented contraction's output without products sums to
         [+0] in every slot and takes [init] alone, as on one device. *)
      let total = E.input 0 ad and initial = E.cast ad (E.input 1 out) in
      let e =
        match segments with
        | None -> E.(total + initial)
        | Some _ ->
            E.(where (total = const ad (D.zero ad)) initial (total + initial))
      in
      let with_init k here w y =
        let i = Value.Any (view eng ~by k w i) in
        List.hd (map eng ~by here [ y; i ] [ E.any (E.cast out e) ])
      in
      reduced eng ~by cl shape ~then_:with_init out (Monoid (Sum, 0, ad)) s

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
    let i = shift eng ~by (-t.start) (Value.Any (view eng ~by ks.(j) w idx)) in
    let x = view eng ~by ks.(j) (reads rule 1 sx w (fun _ -> t)) x in
    let idx = Prim.expect D.Int64 i in
    let y = at eng ~by here (Value.Gather { axis; idx; x }) in
    at eng ~by here (Value.Bitcast (u, y))
  in
  (* Only the tile holding a position reads bits other than zeros there. *)
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
  let own j =
    let w = Devices.window ~by p shape j in
    let here = Devices.one set ks.(j) in
    if Route.empty w then None
    else
      let read i x =
        view eng ~by ks.(j) (Route.window rule ~operand:i (Prim.shape x) w) x
      in
      let i = shift eng ~by (-w.(axis).start) (Value.Any (read 0 idx)) in
      let idx = Prim.expect D.Int64 i in
      let updates = read 1 updates and into = read 2 into in
      Some
        (at eng ~by here
           (Value.Scatter { combine; unique; axis; idx; updates; into }))
  in
  eng.assemble p shape (Prim.dtype into) (Array.init (Array.length ks) own)

(* An [Add] of updates cut along its axis: device [j] adds its tile of
   them, at the positions moved to its window's start, into its window,
   in the carrier: the first tile into [into]'s, every other into [-0],
   which adds nothing. The slots close by one map adding them in tile
   order, with no [+0] added, and the result rounds once. *)
let summed (type v s d) eng ~by (route : unit Route.t) rule unique axis
    (idx : d Value.index) (updates : (v, s, d) Value.t)
    (into : (v, s, d) Value.t) : (v, s, d) Value.t =
  let p = route.result in
  let set = Devices.set p and ks = Grid.devices (Devices.grid p) in
  let shape = Prim.shape into and su = Prim.shape updates in
  let dt = Prim.dtype into in
  let (D.Any c) = Nx_kernel.carrier (D.Any dt) in
  let tiles = tiles ~by route ~operand:1 su [| axis |] in
  let cl =
    closing ~by route ~operand:1 ~axes:[ axis ] ~extent:(fun _ -> su.(axis))
      shape
  in
  let own j =
    let w = Devices.window ~by p shape j and t = tiles.(j).(0) in
    let here = Devices.one set ks.(j) in
    let read i x =
      view eng ~by ks.(j) (reads rule i (Prim.shape x) w (fun _ -> t)) x
    in
    let i = shift eng ~by (-w.(axis).start) (Value.Any (read 0 idx)) in
    let idx = Prim.expect D.Int64 i in
    let start =
      if t.start = 0 then E.any (E.cast c (E.input 0 dt))
      else E.any (E.neg (E.const c (D.zero c)))
    in
    let target = map eng ~by here [ Value.Any (read 2 into) ] [ start ] in
    let into = Prim.expect c (List.hd target) in
    let updates = cast eng ~by (D.Any c) (Value.Any (read 1 updates)) in
    let updates = Prim.expect c updates in
    (* The target is fresh: the updates add into its memory. *)
    at eng ~by ~into:(Value.Any into) here
      (Value.Scatter { combine = Add; unique; axis; idx; updates; into })
  in
  let n = List.length (slots cl) in
  let sum =
    List.fold_left
      (fun acc k -> E.(acc + input k c))
      (E.input 0 c)
      (List.init (n - 1) (fun k -> k + 1))
  in
  (* A piece's slots added, then rounded into the dtype. *)
  let added _ here _ = function
    | [ Value.Any s ] ->
        let slots = List.map (fun i -> Value.Any (slot eng i s)) (slots cl) in
        let total = List.hd (map eng ~by here slots [ E.any sum ]) in
        [ cast eng ~by (D.Any dt) total ]
    | _ -> invalid_arg "Partial.summed: one stack"
  in
  let s = Value.Any (stack eng ~by cl c shape own) in
  Prim.expect dt (List.hd (closed eng ~by cl shape [ D.Any dt ] [ s ] added))

let run (type r) eng ~by route (op : r Value.prim) : r =
  let rule = Prim.rule op in
  match op with
  | Value.Gather { axis; idx; x } -> gather eng ~by route rule axis idx x
  | Value.Scatter { combine; unique; axis; idx; updates; into } ->
      (* An [Add]'s updates are read cut along the axis only where the
         route sums them. *)
      let cuts = Grid.cuts (Devices.grid route.operands.(1)) in
      if Array.exists (fun (a, _) -> a = axis) cuts then
        summed eng ~by route rule unique axis idx updates into
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
      } ->
      contract eng ~by route rule ~batch ~contracting ~acc ~segments out a b
        init
        a_scale b_scale
  | Value.Contract { segments = Some (Contracting _); _ } ->
      invalid_arg "Partial.run: groups of a contracting pair by partials"
  | Value.Scan { carry = Some _; _ }
  | Value.Map _ | Value.Sort _ | Value.Fft _ | Value.Linalg _ | Value.Fold _
  | Value.Assemble _ | Value.Copy _ | Value.Move _ | Value.Bitcast _
  | Value.Place _ | Value.Check _ ->
      invalid_arg "Partial.run: an operation no route computes by partials"
