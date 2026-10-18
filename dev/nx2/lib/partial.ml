(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)


module D = Nx_array.Dtype
module L = Nx_array.Layout
module M = Nx_array.Move
module P = Nx_kernel.Prog
module E = P.E

let full d = { M.start = 0; count = d; step = 1 }
let numel = Array.fold_left ( * ) 1
let counts w = Array.map (fun (r : M.range) -> r.count) w

let floats (D.Any dt) =
  match D.kind dt with
  | D.Float | D.Complex -> true
  | D.Signed | D.Unsigned | D.Boolean -> false

let complex (D.Any dt) =
  match D.kind dt with
  | D.Complex -> true
  | D.Float | D.Signed | D.Unsigned | D.Boolean -> false

(* The dtype of a complex dtype's parts; any other dtype itself. *)
let part (D.Any dt as d) =
  match dt with
  | D.Complex64 -> D.Any D.Float32
  | D.Complex128 -> D.Any D.Float64
  | _ -> d

let identity d = P.of_node ~ins:[| d |] (In 0)

(* [prog] with output [o] and whether it is a NaN as its outputs. *)
let flagged (prog : P.t) o =
  let out = Iarray.get prog.outs o in
  let nodes =
    Iarray.append prog.nodes [| P.Op2 (Compare Not_equal, out, out) |]
  in
  P.v ~ins:prog.ins nodes ~outs:[| out; Iarray.length prog.nodes |]

let binary : Nx_kernel.monoid -> P.binary = function
  | Sum -> Add
  | Prod -> Mul
  | Max -> Maximum
  | Min -> Minimum
  | Logsumexp -> invalid_arg "Partial.binary: Logsumexp"

(* The quiet NaN of a clear sign bit, the NaN Prog's kinds make from numbers
   ({!Nx_kernel.type-reduction}). *)
let made_nan = Int64.float_of_bits 0x7FF8_0000_0000_0000L

(* Operand [0], of [c], rounded into [qt]. *)
let rounded (type v s w r) (c : (v, s) D.t) (qt : (w, r) D.t) : (w, r) E.t =
  match D.equal_witness c qt with
  | Some Type.Equal -> E.input 0 c
  | None -> E.cast qt (E.input 0 c)

(* The terms of a block of lane order ({!Nx_kernel.section-orders}). *)
let block = 1024

(* The position of device [k] among [p]'s. *)
let position p k =
  Option.get (Array.find_index (( = ) k) (Grid.devices (Devices.grid p)))

(* The window of operand [i], of [shape], that a device reads by [rule]: of
   its window [w] of the results along an axis that becomes one of theirs,
   [tile f] along one of the folded factor [f], whole along any other. *)
let reads rule i shape (w : M.range array) tile =
  match (rule : Route.rule) with
  | Elementwise -> w
  | Axes t | Into t ->
      Array.mapi
        (fun a d ->
          match t.(i).(a) with
          | Route.Becomes a' | Scanned a' -> w.(a')
          | Reduced f -> tile f
          | Whole -> full d)
        shape

(* Groups *)

(* A run of a tile's terms: the number [g] of its first in C order of the
   whole's folded indices and [l] in its tile's, its count [len], and
   [owner], its tile's position among its group's. *)
type run = { g : int; l : int; len : int; owner : int }

(* The runs of [tile], at position [t], a window of the folded extents
   [whole], in increasing number: one per index of the folded axes before
   the last that [tile] does not hold whole. *)
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

(* [r] cut to the terms [lo, hi). *)
let clip lo hi r =
  let a = Int.max lo r.g and b = Int.min hi (r.g + r.len) in
  { r with g = a; l = r.l + a - r.g; len = b - a }

(* The devices of a placement, by position, that hold one window of the
   results off the folded axes, its [key]: its [members]; their [tiles],
   distinct windows along the folded axes of extents [whole] that hold
   terms, in order; the first member holding each, its [holder]; and the
   tiles' runs, [held], in increasing number. *)
type group = {
  members : int list;
  tiles : M.range array array;
  holder : int array;
  held : run array;
}

let groups n ~key ~tile whole =
  let keys = Array.init n key and tiles = Array.init n tile in
  let all = List.init n Fun.id in
  let group j =
    let members = List.filter (fun i -> keys.(i) = keys.(j)) all in
    let holding =
      List.filter (fun i -> numel (counts tiles.(i)) > 0) members
    in
    let ts =
      Array.of_list
        (List.sort_uniq compare (List.map (fun i -> tiles.(i)) holding))
    in
    let holder =
      Array.map (fun t -> List.find (fun i -> tiles.(i) = t) holding) ts
    in
    let held = List.concat (Array.to_list (Array.mapi (runs whole) ts)) in
    let held = List.sort (fun a b -> Int.compare a.g b.g) held in
    { members; tiles = ts; holder; held = Array.of_list held }
  in
  let first j = List.find (fun i -> keys.(i) = keys.(j)) all = j in
  List.map group (List.filter first all)

(* Units *)

(* The order partials combine in: a leaf is a unit, the partial of terms
   one device computes. *)
type tree = Leaf of int | Node of tree * tree

(* A leaf of lane order's tree: terms [lo, hi) that tile [dest] reduces,
   holding all of them or, for a block that window edges split, most;
   [parts] are the runs that hold them, cut to them. *)
type leaf = { lo : int; hi : int; dest : int; parts : run list }

(* Lane order's tree of [n] terms whose runs, of [tiles] tiles, are [runs],
   in increasing number, and its leaves in term order. A node of blocks
   [b, b + count) has the value of a lane-order reduction of exactly its
   terms: it is a leaf where one tile holds them all or it is one block. *)
let tree n tiles (runs : run array) =
  let find t =
    let lo = ref 0 and hi = ref (Array.length runs - 1) in
    while !lo < !hi do
      let mid = (!lo + !hi + 1) / 2 in
      if runs.(mid).g <= t then lo := mid else hi := mid - 1
    done;
    !lo
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

(* [n > 0] leaves combined in their order. *)
let chain n =
  let t = ref (Leaf 0) in
  for i = 1 to n - 1 do
    t := Node (!t, Leaf i)
  done;
  !t

(* What a ranked reduction gives: the extreme alone, or with its position. *)
type gives = Extreme | Extreme_and_position

(* Folds of partials through [A.apply] *)

module Make (A : sig
  val by : string
  val apply : by:string -> 'q Value.prim -> 'q
end) =
struct
  let by = A.by
  let apply op = A.apply ~by op

  (* Values *)

  (* [x] on [here]; a value of no load lies on no set until placed. *)
  let move (type v s d) (here : d Devices.placement) (x : (v, s, d) Value.t) :
      (v, s, d) Value.t =
    match Prim.at x with
    | Some p when Devices.equal p here -> x
    | Some _ | None -> apply (Value.Place (here, x))

  let moves here vs =
    List.map (fun (Value.Any x) -> Value.Any (move here x)) vs

  let reshape (type v s d) shape (x : (v, s, d) Value.t) : (v, s, d) Value.t =
    if Prim.has_shape x shape then x
    else apply (Value.Move (Reshape (Array.copy shape), x))

  (* Elements [start, start + count) of [x] along [axis]. *)
  let slice (type v s d) axis start count (x : (v, s, d) Value.t) :
      (v, s, d) Value.t =
    let r = Array.map full (Prim.shape x) in
    if start = 0 && count = r.(axis).count then x
    else begin
      r.(axis) <- { M.start; count; step = 1 };
      apply (Value.Move (Slice r, x))
    end

  let cast (type d) (D.Any dt) (Value.Any x : d Value.any) : d Value.any =
    match D.equal_witness (Prim.dtype x) dt with
    | Some Type.Equal -> Value.Any x
    | None -> Value.Any (fst (apply (Prim.op1 ~by Cast dt x)))

  let bitcast (type d) (D.Any dt) (Value.Any x : d Value.any) : d Value.any =
    Value.Any (apply (Value.Bitcast (dt, x)))

  (* [x] with each group of [groups]' axes, in order, made one axis. *)
  let grouped (type v s d) (groups : int array array) (x : (v, s, d) Value.t)
      : (v, s, d) Value.t =
    let s = Prim.shape x in
    let order = Array.concat (Array.to_list groups) in
    let x =
      if Array.for_all2 ( = ) order (Array.init (Array.length s) Fun.id) then x
      else apply (Value.Move (Permute order, x))
    in
    reshape (Array.map (fun g -> numel (Array.map (fun a -> s.(a)) g)) groups) x

  (* [x]'s window [w] from its array on its set's device [k], at [at]. *)
  let view (type v s d) k (at : d Devices.placement) w (x : (v, s, d) Value.t)
      : (v, s, d) Value.t =
    Value.Array { at; a = Place.view ~by x k w; dead = Value.Live }

  let view_load (type d) k at w (l : d Value.load) : d Value.load =
    match l with
    | Plain x -> Plain (view k at w x)
    | Padded p -> Padded { p with x = view k at w p.x }

  (* Output [o] of [prog], of [term], over [loads] in the window [w] of a
     loop of [layout]: the load itself where [prog] is its identity. *)
  let terms (type d) layout (prog : P.t) o (D.Any term)
      (loads : d Value.load array) w : d Value.any =
    match loads with
    | [| Plain x |]
      when P.is_identity prog && D.equal (Prim.dtype x) term ->
        Value.Any x
    | _ ->
        let one = P.v ~ins:prog.ins prog.nodes ~outs:[| Iarray.get prog.outs o |] in
        let map = Value.Map { layout; prog = one; outs = Value.[ term ]; loads } in
        Value.Any (fst (apply (Prim.within map w)))

  (* The value at [p] of [shape] whose array on each of [p]'s devices, in
     order, is [vs]'s. *)
  let at_placement (type d) (p : d Devices.placement) shape
      (vs : d Value.any array) : d Value.any =
    let (Value.Any x) = vs.(0) in
    let dt = Prim.dtype x in
    let array v = Iarray.get (Prim.arrays (Prim.expect dt v)) 0 in
    Value.Any (Prim.of_arrays p ~shape (Iarray.of_array (Array.map array vs)))

  (* The results of [r] at [p], of [shape], from each device's values, in
     [p]'s order. *)
  let result (type d a) (r : (d, a) Value.reduction) p shape
      (per : d Value.any list array) : a =
    let nth i = at_placement p shape (Array.map (fun vs -> List.nth vs i) per) in
    match r with
    | Monoid (_, _, dt) -> Prim.expect dt (nth 0)
    | Moments (_, dt) -> (Prim.expect dt (nth 0), Prim.expect dt (nth 1))
    | Arg (_, _, dt) -> (Prim.expect dt (nth 0), Prim.expect D.Int64 (nth 1))

  (* Maps *)

  type 'd outs = Outs : ('d, 'r) Value.outs * ('r -> 'd Value.any list) -> 'd outs

  let rec outs_of : type d. D.any list -> d outs = function
    | [] -> Outs (Value.[], fun () -> [])
    | D.Any dt :: rest ->
        let (Outs (outs, read)) = outs_of rest in
        Outs (Value.( :: ) (dt, outs), fun (x, r) -> Value.Any x :: read r)

  (* The map over [vs], of one shape, its inputs in order, to [exprs]. *)
  let map (type d) (vs : d Value.any list) exprs : d Value.any list =
    let dtype (Value.Any x : d Value.any) = D.Any (Prim.dtype x) in
    let prog = P.of_exprs ~ins:(Iarray.of_list (List.map dtype vs)) (Iarray.of_list exprs) in
    let dts = List.map (P.dtype prog) (Iarray.to_list prog.outs) in
    let (Outs (outs, read)) = outs_of dts in
    let (Value.Any x) = List.hd vs in
    let layout = L.contiguous (Prim.shape x) in
    let plain (Value.Any x : d Value.any) = Value.Plain x in
    read (apply (Value.Map { layout; prog; outs; loads = Array.of_list (List.map plain vs) }))

  (* Slabs *)

  (* A device's terms, laid out alike along an axis of each value. *)
  type 'd slab = ('d Value.any * int) list

  let cut (sl : 'd slab) l len : 'd slab =
    List.map (fun (Value.Any x, a) -> (Value.Any (slice a l len x), a)) sl

  (* The terms [lo, hi) of the runs [parts] on [here], [slabs t] holding
     owner [t]'s: each owner's span moved there once, and the parts
     assembled in term order. *)
  let gather here (slabs : int -> 'd slab) lo hi parts : 'd slab =
    let span t =
      let mine = List.filter (fun r -> r.owner = t) parts in
      let s = List.fold_left (fun s r -> Int.min s r.l) max_int mine in
      let e = List.fold_left (fun e r -> Int.max e (r.l + r.len)) 0 mine in
      let sl = cut (slabs t) s (e - s) in
      (t, (s, List.map (fun (Value.Any x, a) -> (Value.Any (move here x), a)) sl))
    in
    let owners = List.sort_uniq compare (List.map (fun r -> r.owner) parts) in
    let spans = List.map span owners in
    match parts with
    | [ r ] -> snd (List.assoc r.owner spans)
    | _ ->
        let piece r =
          let s, sl = List.assoc r.owner spans in
          (r.g - lo, cut sl (r.l - s) r.len)
        in
        let pieces = List.map piece parts in
        let assemble i (Value.Any x, a) =
          let dt = Prim.dtype x in
          let shape = Array.copy (Prim.shape x) in
          shape.(a) <- hi - lo;
          let region (o, sl) =
            let y = Prim.expect dt (fst (List.nth sl i)) in
            let r = Array.map full shape in
            r.(a) <- { M.start = o; count = (Prim.shape y).(a); step = 1 };
            (r, y)
          in
          let pieces = List.map region pieces in
          (Value.Any (apply (Value.Assemble { dtype = dt; shape; fill = D.zero dt; pieces })), a)
        in
        List.mapi assemble (snd (List.hd pieces))

  (* Folds *)

  (* The partials [value i] of [tree]'s leaves combined up it on [here], each
     moved there, by [node] of two, the earlier first. *)
  let rec fold node here value = function
    | Leaf i -> moves here (value i)
    | Node (a, b) -> node (fold node here value a) (fold node here value b)

  (* Kinds *)

  (* Two exact partials combined by [k], the earlier [a] first: a float
     extreme keeps the first NaN, as the first NaN term's. *)
  let exact (k : P.binary) a b =
    let vs = a @ b in
    let (Value.Any x) = List.hd vs in
    let dt = Prim.dtype x in
    let a = E.input 0 dt and b = E.input 1 dt in
    let e =
      match k with
      | (Maximum | Minimum) when floats (D.Any dt) ->
          E.(where (a <> a) a (where (b <> b) b (binary k a b)))
      | _ -> E.binary k a b
    in
    map vs [ E.any e ]

  (* The values [x] orders by: itself, or a complex number's parts. *)
  let keys (type v s d) (x : (v, s, d) Value.t) : d Value.any list =
    let parts (type w r) (pt : (w, r) D.t) =
      let pairs = apply (Value.Bitcast (pt, x)) in
      let shape = Prim.shape x in
      let half j = Value.Any (reshape shape (slice (Array.length shape) j 1 pairs)) in
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
  let first_of (e : Nx_kernel.extreme) a b =
    match (a, b) with
    | [ Value.Any ea; pa ], [ eb; pb ] ->
        let dt = Prim.dtype ea in
        let eb = Prim.expect dt eb in
        let ka = keys ea and kb = keys eb in
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
        map
          ((Value.Any ea :: pa :: ka) @ (Value.Any eb :: pb :: kb))
          [ pick 0 dt; pick 1 D.Int64 ]
    | _ -> invalid_arg "Partial.first_of: an extreme and its position each"

  (* [pos], positions in C order of the folded indices of a window [w] of
     the folded extents [whole], as positions in C order of the whole's. *)
  let renumber (w : M.range array) whole pos =
    if Array.for_all2 (fun (r : M.range) n -> r.start = 0 && r.count = n) w whole
    then pos
    else
      let int64 n = E.const D.Int64 (Int64.of_int n) in
      let p = E.input 0 D.Int64 in
      let global = ref (int64 0) and inner = ref 1 and outer = ref 1 in
      for i = Array.length w - 1 downto 0 do
        let digit = E.((p / int64 !inner) mod int64 w.(i).count) in
        global := E.(!global + ((digit + int64 w.(i).start) * int64 !outer));
        inner := !inner * w.(i).count;
        outer := !outer * whole.(i)
      done;
      List.hd (map [ pos ] [ E.any !global ])

  (* A lane partial's [q], its third value, else its first: [s] itself. *)
  let q_at vs base = if List.length vs = 3 then base + 2 else base

  (* Two lane partials [s; f] or [s; f; q] combined, the earlier [a]
     first: [s] by [k], whether either has a NaN term, and the earlier's
     [q] where it has one, else the later's. *)
  let lane (k : P.binary) a b =
    let na = List.length a in
    let (Value.Any s) = List.hd a in
    let (Value.Any q) = List.nth a (q_at a 0) in
    let c = Prim.dtype s and qt = Prim.dtype q in
    let fa = E.input 1 D.Bool and fb = E.input (na + 1) D.Bool in
    map (a @ b)
      [
        E.any (E.binary k (E.input 0 c) (E.input na c));
        E.any E.(fa || fb);
        E.any (E.where fa (E.input (q_at a 0) qt) (E.input (q_at b na) qt));
      ]

  (* [r], of [qt], but a complex product with a NaN part and no NaN term,
     the quiet NaN in each part, as kernels make it. *)
  let quiet (type w r) (m : Nx_kernel.monoid) (qt : (w, r) D.t) (r : (w, r) E.t)
      : (w, r) E.t =
    match (D.kind qt, m) with
    | D.Complex, Prod ->
        E.(where (r <> r) (const qt { Complex.re = made_nan; im = made_nan }) r)
    | _ -> r

  (* The result of a lane partial: [q] where a term is a NaN, else [s]
     rounded once into [q]'s dtype. *)
  let settle m vs =
    let (Value.Any s) = List.hd vs in
    let (Value.Any q) = List.nth vs (q_at vs 0) in
    let c = Prim.dtype s and qt = Prim.dtype q in
    let r = quiet m qt (rounded c qt) in
    List.hd (map vs [ E.any (E.where (E.input 1 D.Bool) (E.input (q_at vs 0) qt) r) ])

  (* A lane unit's partial, each of [rows]: by [reduce] of [flagged]'s
     outputs, its terms by [m] into the carrier [c], whether one is a NaN,
     and into [q] where that is not [c]. *)
  let lane_unit (type d) m (D.Any c) (D.Any q) rows
      (reduce : 'r. (d, 'r) Value.reductions -> 'r Value.prim) : d Value.any list
      =
    let flat (type v s) (x : (v, s, d) Value.t) = Value.Any (reshape rows x) in
    match D.equal_witness c q with
    | Some Type.Equal ->
        let s, (f, ()) =
          apply (reduce Value.[ Monoid (m, 0, c); Monoid (Max, 1, D.Bool) ])
        in
        [ flat s; flat f ]
    | None ->
        let s, (f, (v, ())) =
          apply
            (reduce
               Value.[ Monoid (m, 0, c); Monoid (Max, 1, D.Bool); Monoid (m, 0, q) ])
        in
        [ flat s; flat f; flat v ]

  (* Reductions *)

  let reduce (type d r) (route : d Route.t) layout axes prog
      (rs : (d, r) Value.reductions) (loads : d Value.load array) : r =
    let shape = L.shape layout in
    let p = route.result in
    let set = Devices.set p and ks = Grid.devices (Devices.grid p) in
    let rule = Prim.rule (Value.Reduce { layout; axes; prog; reductions = rs; loads }) in
    let out = Prim.reduced shape axes in
    let whole = Array.map (fun a -> shape.(a)) axes in
    let outw j = Devices.window ~by p out j in
    let here j = Devices.one set ks.(j) in
    (* Each device's tile, as it reads load 0: the route reads every load
       alike along the reduced axes. *)
    let tiles =
      let (Value.Any x) = Prim.load_any loads.(0) in
      let q = route.operands.(0) in
      Array.map
        (fun k ->
          let w = Devices.window ~by q (Prim.shape x) (position q k) in
          Array.map (fun a -> w.(a)) axes)
        ks
    in
    let along j a = tiles.(j).(Option.get (Array.find_index (( = ) a) axes)) in
    let loop j =
      let w = outw j in
      let before a = Array.fold_left (fun n b -> if b < a then n + 1 else n) 0 axes in
      Array.init (Array.length shape) (fun a ->
          if Array.mem a axes then along j a else w.(a - before a))
    in
    let viewed j =
      Array.mapi
        (fun i l ->
          let (Value.Any x) = Prim.load_any l in
          view_load ks.(j) (here j) (reads rule i (Prim.shape x) (outw j) (along j)) l)
        loads
    in
    let local j prog rs =
      Prim.within (Value.Reduce { layout; axes; prog; reductions = rs; loads = viewed j }) (loop j)
    in
    let gs = groups (Array.length ks) ~key:outw ~tile:(Array.get tiles) whole in
    let term o = P.dtype prog (Iarray.get prog.outs o) in
    let one (type a) (r : (d, a) Value.reduction) : a =
      let per = Array.make (Array.length ks) [] in
      (* Each group's partials [value t] of its leaves, combined up [tree]
         by [node] on each member and settled there. *)
      let combined g tree value node settle =
        List.iter
          (fun j -> per.(j) <- settle j (fold node (here j) value tree))
          g.members
      in
      (* One partial per tile, by [unit] on its holder, in tile order. *)
      let by_tiles unit node settle =
        List.iter
          (fun g ->
            let values = Array.map unit g.holder in
            combined g (chain (Array.length g.tiles)) (Array.get values) node settle)
          gs
      in
      let ranked e o (D.Any dt) gives =
        let (D.Any t) = term o in
        let unit j =
          let (x, q), () = apply (local j prog Value.[ Arg (e, o, t) ]) in
          [ Value.Any x; renumber tiles.(j) whole (Value.Any q) ]
        in
        let settle _ = function
          | [ x; q ] -> (
              let x = cast (D.Any dt) x in
              match gives with Extreme -> [ x ] | Extreme_and_position -> [ x; q ])
          | _ -> invalid_arg "Partial.reduce: an extreme and its position"
        in
        by_tiles unit (first_of e) settle;
        result r p out per
      in
      (* Lane order's tree over each group's terms, a complex sum's parts as
         floats. *)
      let lane_order m o (D.Any dt) =
        let term = term o in
        let parts = complex term && m = Nx_kernel.Sum in
        let t = if parts then part term else term in
        let q = if parts then part (D.Any dt) else D.Any dt in
        let c = Prim.carrier t in
        let rows j =
          let n = numel (counts (outw j)) in
          if parts then [| n; 2 |] else [| n |]
        in
        let kept = Array.of_list (List.filter (fun a -> not (Array.mem a axes)) (List.init (Array.length shape) Fun.id)) in
        (* A tile's terms on its holder, as [outputs; terms]. *)
        let slab j =
          let (Value.Any x) = terms layout prog o term (viewed j) (loop j) in
          let x = Value.Any (grouped [| kept; axes |] x) in
          [ ((if parts then bitcast t x else x), 1) ]
        in
        List.iter
          (fun g ->
            let tree, leaves = tree (numel whole) (Array.length g.tiles) g.held in
            let slabs = Array.map (fun j -> lazy (slab j)) g.holder in
            let value lf =
              let j = g.holder.(lf.dest) in
              match lf.parts with
              | [ r ] when r.l = 0 && r.len = numel (counts tiles.(j)) && not parts ->
                  (* The tile's terms are the unit's: reduced where they lie. *)
                  lane_unit m c q (rows j) (fun rs -> local j (flagged prog o) rs)
              | parts ->
                  let x = gather (here j) (fun t -> Lazy.force slabs.(t)) lf.lo lf.hi parts in
                  let (Value.Any x) = fst (List.hd x) in
                  let layout = L.contiguous (Prim.shape x) in
                  let prog = flagged (identity (D.Any (Prim.dtype x))) 0 in
                  lane_unit m c q (rows j) (fun rs ->
                      Value.Reduce { layout; axes = [| 1 |]; prog; reductions = rs; loads = [| Plain x |] })
            in
            let values = Array.map value leaves in
            let settle j vs =
              let v = settle m vs in
              let (Value.Any v) = if parts then bitcast (D.Any dt) v else v in
              [ Value.Any (reshape (counts (outw j)) v) ]
            in
            combined g tree (Array.get values) (lane (binary m)) settle)
          gs;
        result r p out per
      in
      let exactly o = not (floats (term o)) in
      (* A complex sum adds each part as a float sum, into complex numbers. *)
      let lanes m o dt =
        let (D.Any t) = term o in
        match D.kind t with
        | D.Float -> true
        | D.Complex -> m = Nx_kernel.Prod || D.is D.Complex dt
        | D.Signed | D.Unsigned | D.Boolean -> false
      in
      match r with
      | Monoid (((Sum | Prod | Max | Min) as m), o, dt) when exactly o ->
          let (D.Any c) = Prim.carrier (term o) in
          let unit j =
            let v, () = apply (local j prog Value.[ Monoid (m, o, c) ]) in
            [ Value.Any v ]
          in
          let settle _ vs = [ cast (D.Any dt) (List.hd vs) ] in
          by_tiles unit (exact (binary m)) settle;
          result r p out per
      | Monoid (((Max | Min) as m), o, dt) ->
          let e : Nx_kernel.extreme = if m = Max then Max else Min in
          ranked e o (D.Any dt) Extreme
      | Arg (e, o, dt) -> ranked e o (D.Any dt) Extreme_and_position
      | Monoid (((Sum | Prod) as m), o, dt) when lanes m o dt ->
          lane_order m o (D.Any dt)
      | Monoid ((Sum | Prod | Logsumexp), _, _) | Moments _ ->
          (* No such reduction is core: each has an expansion, whose own
             reductions compute as above. *)
          let op = Value.Reduce { layout; axes; prog; reductions = Value.[ r ]; loads } in
          fst (Option.get (Expand.run A.apply ~by op))
    in
    let rec each : type q. (d, q) Value.reductions -> q = function
      | [] -> ()
      | r :: rest ->
          let a = one r in
          (a, each rest)
    in
    each rs
end


(* Scans and contractions *)

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

(* A float or complex [Sum] or [Prod] of terms of [term] into [rounds], its
   units reducing into [carrier] ({!Prim.carrier}). *)
type lane = {
  monoid : Nx_kernel.monoid;
  term : D.any;
  carrier : D.any;
  rounds : D.any;
}

let float_part = part
let lane monoid term rounds = { monoid; term; carrier = Prim.carrier term; rounds }

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

(* [l] read on device [k], at [at], over its window [w] there. *)
let view_load (type d) ~by k (at : d Devices.placement) (l : d Value.load) w :
    d Value.load =
  let view (type v s) (x : (v, s, d) Value.t) : (v, s, d) Value.t =
    Value.Array { at; a = Place.view ~by x k w; dead = Value.Live }
  in
  match l with
  | Plain x -> Plain (view x)
  | Padded p -> Padded { p with x = view p.x }

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

(* [x] on [here]; a value of no load lies on no set until placed. *)
let moved (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (here : d Devices.placement) (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  match Prim.at x with
  | Some p when Devices.equal p here -> x
  | Some _ | None -> apply ~by (Value.Place (here, x))

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

(* [u]'s values on [here]. *)
let unit_on (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by here
    (Unit u : d unit_value) : d unit_value =
  let s = moved apply ~by here u.s and f = moved apply ~by here u.f in
  match D.equal_witness (Prim.dtype u.q) (Prim.dtype u.s) with
  | Some Type.Equal when u.q == u.s -> Unit { s; f; q = s }
  | Some _ | None -> Unit { s; f; q = moved apply ~by here u.q }

(* The terms of a scan's chunk ({!Nx_kernel.type-reduction}). *)
let chunk = 4096

(* [x], a value on one device, along [axis] from [start], [count] of them. *)
let cut_along (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    axis (x : (v, s, d) Value.t) start count : (v, s, d) Value.t =
  let r = Array.map full (Prim.shape x) in
  if start = 0 && count = r.(axis).count then x
  else begin
    r.(axis) <- { M.start; count; step = 1 };
    apply ~by (Value.Move (Slice r, x))
  end

(* The scan of [x], a value on one device, along [axis] by [r], from
   [carry]. *)
let scanned (type v s d a) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    axis (x : (v, s, d) Value.t) (r : (d, a) Value.reduction) carry : a =
  apply ~by
    (Value.Scan
       {
         layout = L.contiguous (Prim.shape x);
         axis;
         prog = identity (D.Any (Prim.dtype x));
         reduction = r;
         loads = [| Value.Plain x |];
         carry;
       })

(* [x] of [shape], its axes of extent 1 stretched. *)
let spread (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by shape
    (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  if Prim.has_shape x shape then x
  else apply ~by (Value.Move (Broadcast (Array.copy shape), x))

(* Output [o] of [prog], of the dtype [term], over [loads] in the window [w] of
   the loop of [layout], on device [k] of [set]: the load's view where [prog]
   is its identity. *)
let window_terms (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (set : d Devices.t) layout (prog : P.t) o (D.Any term)
    (loads : d Value.load array) k w : d Value.any =
  let at = Devices.one set k in
  let viewed = Array.map (fun l -> view_load ~by k at l w) loads in
  let mapped () =
    let out = Iarray.get prog.outs o in
    let one = P.v ~ins:prog.ins prog.nodes ~outs:[| out |] in
    let map =
      Value.Map { layout; prog = one; outs = Value.[ term ]; loads = viewed }
    in
    Value.Any (fst (apply ~by (Prim.within map w)))
  in
  match viewed with
  | [| Plain x |] when P.is_identity prog -> (
      match D.equal_witness (Prim.dtype x) term with
      | Some Type.Equal -> Value.Any x
      | None -> mapped ())
  | _ -> mapped ()

(* A tile of a group along the scanned axis: its terms [s, e) and the device
   that computes it, the first of the group holding them. *)
type tile = { s : int; e : int; holder : int }

(* Scans that combine exactly: each tile scans its own terms, and the carry of
   the tiles before it, their last results combined in order, combines with
   each result. Its partials and how two combine, the earlier first:
   [Exact_scan] in the carrier by a kind, [Extreme_scan] in the result's dtype
   by its extreme, a NaN first, and [Arg_scan] as an extreme and its position
   in the whole's order, the first extreme. *)
type exact_scan =
  | Exact_scan of Nx_kernel.monoid * P.binary
  | Extreme_scan of Nx_kernel.monoid * P.binary
  | Arg_scan of Nx_kernel.extreme

(* Two float or complex extremes [c] then [t] combined by [k]: a NaN [c], else
   a NaN [t], else their extreme. *)
let first_extreme (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (k : P.binary) (Value.Any c : d Value.any) (t : d Value.any) : d Value.any
    =
  let dt = Prim.dtype c in
  let t = Prim.expect dt t in
  let x i = E.input i dt in
  let nan i = E.(x i <> x i) in
  let prog =
    P.of_exprs
      ~ins:[| D.Any dt; D.Any dt |]
      [|
        E.any
          E.(where (nan 0) (x 0) (where (nan 1) (x 1) (binary k (x 0) (x 1))));
      |]
  in
  let layout = L.contiguous (Prim.shape c) in
  let loads = Value.[| Plain c; Plain t |] in
  let y, () =
    apply ~by (Value.Map { layout; prog; outs = Value.[ dt ]; loads })
  in
  Value.Any y

(* [pos] shifted by [s]. *)
let shifted (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by s
    (pos : d Value.any) : d Value.any =
  if s = 0 then pos
  else
    let pos = Prim.expect D.Int64 pos in
    let prog =
      P.of_exprs
        ~ins:[| D.Any D.Int64 |]
        [| E.any E.(input 0 D.Int64 + const D.Int64 (Int64.of_int s)) |]
    in
    let layout = L.contiguous (Prim.shape pos) in
    let y, () =
      apply ~by
        (Value.Map
           { layout; prog; outs = Value.[ D.Int64 ]; loads = [| Plain pos |] })
    in
    Value.Any y

(* The value of [r]'s results [vs]. *)
let one_result : type d a. (d, a) Value.reduction -> d Value.any list -> a =
 fun r vs ->
  match (r, vs) with
  | Monoid (_, _, dt), [ v ] -> Prim.expect dt v
  | Moments (_, dt), [ m; v ] -> (Prim.expect dt m, Prim.expect dt v)
  | Arg (_, _, dt), [ e; q ] -> (Prim.expect dt e, Prim.expect D.Int64 q)
  | _ -> invalid_arg "Partial.one_result: one value per result"

(* The results of [tiles] along [axis], each on its holder, from their terms
   [x]: [into] is the result's dtype. *)
let exact_tiles (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (set : d Devices.t) axis plan (D.Any into) (tiles : tile array)
    (x : d Value.any Lazy.t array) : d Value.any list array =
  let here t = Devices.one set tiles.(t).holder in
  let local t =
    let (Value.Any xv) = Lazy.force x.(t) in
    match plan with
    | Exact_scan (m, _) ->
        let (D.Any c) = Prim.carrier (D.Any (Prim.dtype xv)) in
        [ Value.Any (scanned apply ~by axis xv (Monoid (m, 0, c)) None) ]
    | Extreme_scan (m, _) ->
        [ Value.Any (scanned apply ~by axis xv (Monoid (m, 0, into)) None) ]
    | Arg_scan e ->
        let ex, pos = scanned apply ~by axis xv (Arg (e, 0, into)) None in
        [ Value.Any ex; shifted apply ~by tiles.(t).s (Value.Any pos) ]
  in
  let locals = Array.init (Array.length tiles) local in
  let last t =
    let n = tiles.(t).e - tiles.(t).s in
    List.map
      (fun (Value.Any v) -> Value.Any (cut_along apply ~by axis v (n - 1) 1))
      locals.(t)
  in
  let combine c l =
    match (plan, c, l) with
    | Exact_scan (_, k), [ c ], [ l ] -> [ fold_exact apply ~by k [ c; l ] ]
    | Extreme_scan (_, k), [ c ], [ l ] -> [ first_extreme apply ~by k c l ]
    | Arg_scan e, [ c0; c1 ], [ l0; l1 ] ->
        let y, q = first_of apply ~by e (c0, c1) (l0, l1) in
        [ y; q ]
    | _ -> invalid_arg "Partial.exact_tiles: partials of one kind"
  in
  let on h vs =
    List.map (fun (Value.Any v) -> Value.Any (moved apply ~by h v)) vs
  in
  let count = Array.length tiles in
  let carries = Array.make count [] in
  for t = 1 to count - 1 do
    let h = here t in
    carries.(t) <-
      (if t = 1 then on h (last 0)
       else combine (on h carries.(t - 1)) (on h (last (t - 1))))
  done;
  let finish (Value.Any v) =
    match plan with
    | Exact_scan _ -> cast apply ~by (D.Any into) (Value.Any v)
    | Extreme_scan _ | Arg_scan _ -> Value.Any v
  in
  Array.mapi
    (fun t l ->
      if t = 0 then List.map finish l
      else
        let wide =
          List.map2
            (fun (Value.Any c) (Value.Any v) ->
              Value.Any (spread apply ~by (Prim.shape v) c))
            carries.(t) l
        in
        List.map finish (combine wide l))
    locals

(* Float and complex sums and products: chunk totals as lane order's units,
   the carry into each chunk, and each tile's scan from the carry into its
   first whole chunk and, where it starts inside a chunk, from the running
   result before it. *)

(* Where an earlier term is a NaN [f], and the first such term in the result's
   dtype [q], per slice. *)
type 'd seen =
  | Seen : {
      f : (bool, D.bool_elt, 'd) Value.t;
      q : ('w, 'r, 'd) Value.t;
    }
      -> 'd seen

let seen_of (Unit u : 'd unit_value) = Seen { f = u.f; q = u.q }

let seen_on (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by here
    (Seen s : d seen) : d seen =
  Seen { f = moved apply ~by here s.f; q = moved apply ~by here s.q }

(* [a] then [b]: whether either has a NaN, and the earlier's first NaN. *)
let seen_node (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (Seen a : d seen) (Seen b : d seen) : d seen =
  let qt = Prim.dtype a.q in
  let qb = Prim.expect qt (Value.Any b.q) in
  let ins : D.any iarray =
    [| D.Any D.Bool; D.Any qt; D.Any D.Bool; D.Any qt |]
  in
  let fa = E.input 0 D.Bool in
  let prog =
    P.of_exprs ~ins
      [|
        E.any E.(fa || input 2 D.Bool);
        E.any (E.where fa (E.input 1 qt) (E.input 3 qt));
      |]
  in
  let layout = L.contiguous (Prim.shape a.f) in
  let loads = Value.[| Plain a.f; Plain a.q; Plain b.f; Plain qb |] in
  let f, (q, ()) =
    apply ~by (Value.Map { layout; prog; outs = Value.[ D.Bool; qt ]; loads })
  in
  Seen { f; q }

(* Unit [u]'s row [i] along [axis], of shape [rows]. *)
let unit_row (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by axis
    rows (Unit u : d unit_value) i : d unit_value =
  let row (type v s) (y : (v, s, d) Value.t) : (v, s, d) Value.t =
    apply ~by (Value.Move (Reshape rows, cut_along apply ~by axis y i 1))
  in
  let s = row u.s and f = row u.f in
  match D.equal_witness (Prim.dtype u.q) (Prim.dtype u.s) with
  | Some Type.Equal when u.q == u.s -> Unit { s; f; q = s }
  | Some _ | None -> Unit { s; f; q = row u.q }

(* [x] of [shape]. *)
let reshaped (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    shape (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  if Prim.has_shape x shape then x
  else apply ~by (Value.Move (Reshape (Array.copy shape), x))

(* The results of [tiles] along [axis] of [n] terms, each on its holder, from
   their terms [x] by [ln]. *)
let lane_tiles (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by ln
    (set : d Devices.t) axis n (tiles : tile array)
    (x : d Value.any Lazy.t array) : d Value.any array =
  let count = Array.length tiles in
  let here t = Devices.one set tiles.(t).holder in
  let chunks = (n + chunk - 1) / chunk in
  let tile_of i =
    let t = ref 0 in
    while tiles.(!t).e <= i do
      incr t
    done;
    !t
  in
  let (D.Any c) = ln.carrier in
  let flagged_terms = flagged (identity ln.term) 0 in
  let window t =
    let (Value.Any v) = Lazy.force x.(t) in
    Prim.shape v
  in
  let w0 = window 0 in
  let rank = Array.length w0 in
  (* Each tile's slices, one row each, and a carry's shape there. *)
  let rows = [| numel (Prim.reduced w0 [| axis |]) |] in
  let carry_shape t =
    let s = window t in
    s.(axis) <- 1;
    s
  in
  let units xs w =
    unit_value apply ~by ln ~layout:(L.contiguous w) ~axes:[| axis |]
      flagged_terms xs None rows
  in
  let piece t l len =
    let (Value.Any v) = Lazy.force x.(t) in
    Value.Any (cut_along apply ~by axis v l len)
  in
  (* The totals of the chunks before the last: of whole chunks where a tile
     holds them, one reduction per tile; of a chunk across tiles, lane order's
     units over its terms, combined on the tile holding the next chunk's
     start. *)
  let totals = Array.make chunks None in
  Array.iteri
    (fun t { s; e; _ } ->
      let lo = (s + chunk - 1) / chunk in
      let hi = Int.min (e / chunk) (chunks - 1) in
      if hi > lo then begin
        let (Value.Any v) = Lazy.force x.(t) in
        let v =
          cut_along apply ~by axis v ((lo * chunk) - s) ((hi - lo) * chunk)
        in
        let w = Prim.shape v in
        let split =
          Array.concat
            [
              Array.sub w 0 axis;
              [| hi - lo; chunk |];
              Array.sub w (axis + 1) (rank - axis - 1);
            ]
        in
        let v = reshaped apply ~by split v in
        let u =
          unit_value apply ~by ln ~layout:(L.contiguous split)
            ~axes:[| axis + 1 |] flagged_terms [| Value.Plain v |] None
            (Prim.reduced split [| axis + 1 |])
        in
        for k = lo to hi - 1 do
          totals.(k) <- Some (unit_row apply ~by axis rows u (k - lo))
        done
      end)
    tiles;
  for k = 0 to chunks - 2 do
    if Option.is_none totals.(k) then begin
      let lo = k * chunk and hi = Int.min n ((k + 1) * chunk) in
      let ts =
        Array.of_list
          (List.filter
             (fun t -> tiles.(t).s < hi && tiles.(t).e > lo)
             (List.init count Fun.id))
      in
      let runs =
        Array.mapi
          (fun i t ->
            let a = Int.max tiles.(t).s lo and b = Int.min tiles.(t).e hi in
            { g = a - lo; l = a - tiles.(t).s; len = b - a; owner = i })
          ts
      in
      let tr, leaves = tree (hi - lo) (Array.length ts) runs in
      let value lf =
        let terms =
          match lf.parts with
          | [ p ] -> piece ts.(p.owner) p.l p.len
          | parts ->
              (* Each part moves to the device of the unit, then the parts
                 are assembled in term order. *)
              let h = here ts.(lf.dest) in
              let (Value.Any x0) = Lazy.force x.(ts.(lf.dest)) in
              let dt = Prim.dtype x0 in
              let shape = window ts.(lf.dest) in
              shape.(axis) <- lf.hi - lf.lo;
              let part p =
                let region = Array.map full shape in
                region.(axis) <-
                  { M.start = p.g - lf.lo; count = p.len; step = 1 };
                let y = Prim.expect dt (piece ts.(p.owner) p.l p.len) in
                (region, moved apply ~by h y)
              in
              Value.Any
                (apply ~by
                   (Value.Assemble
                      {
                        dtype = dt;
                        shape;
                        fill = D.zero dt;
                        pieces = List.map part parts;
                      }))
        in
        let (Value.Any y) = terms in
        units [| Value.Plain y |] (Prim.shape y)
      in
      let values = Array.map value leaves in
      let h = here (tile_of hi) in
      let rec combine = function
        | Leaf i -> unit_on apply ~by h values.(i)
        | Node (a, b) -> node apply ~by ln (combine a) (combine b)
      in
      totals.(k) <- Some (combine tr)
    end
  done;
  (* The carry into each chunk, on the tile holding its start. *)
  let carries = Array.make chunks None in
  for k = 1 to chunks - 1 do
    let h = here (tile_of (k * chunk)) in
    let u = unit_on apply ~by h (Option.get totals.(k - 1)) in
    carries.(k) <-
      Some
        (match carries.(k - 1) with
        | None -> u
        | Some prev -> node apply ~by ln (unit_on apply ~by h prev) u)
  done;
  (* Whether each piece of a tile, its terms in one chunk, holds a NaN, and
     its first: a whole chunk's from its total, another's by a reduction. *)
  let seen_pieces t =
    let { s; e; _ } = tiles.(t) in
    let h = here t in
    let rec pieces i acc =
      if i >= e then List.rev acc
      else
        let k = i / chunk in
        let j = Int.min e ((k + 1) * chunk) in
        let sn =
          match totals.(k) with
          | Some u when i = k * chunk && j = (k + 1) * chunk ->
              seen_on apply ~by h (seen_of u)
          | Some _ | None ->
              let (Value.Any y) = piece t (i - s) (j - i) in
              seen_of (units [| Value.Plain y |] (Prim.shape y))
        in
        pieces j (sn :: acc)
    in
    pieces s []
  in
  let seen = Array.map (fun _ -> None) tiles in
  let pieces_of t =
    match seen.(t) with
    | Some l -> l
    | None ->
        let l = seen_pieces t in
        seen.(t) <- Some l;
        l
  in
  let fold = function
    | [] -> None
    | first :: rest -> Some (List.fold_left (seen_node apply ~by) first rest)
  in
  let after h a b =
    match (a, b) with
    | None, b -> Option.map (seen_on apply ~by h) b
    | a, None -> Option.map (seen_on apply ~by h) a
    | Some a, Some b ->
        let a = seen_on apply ~by h a and b = seen_on apply ~by h b in
        Some (seen_node apply ~by a b)
  in
  (* Whether the terms before each tile hold a NaN, and the first. *)
  let before = Array.make count None in
  for t = 1 to count - 1 do
    before.(t) <- after (here t) before.(t - 1) (fold (pieces_of (t - 1)))
  done;
  (* [res], of [qd], where [sn] has a NaN: its first, everywhere. *)
  let filled (type v s) t (qd : (v, s) D.t) sn (res : (v, s, d) Value.t) =
    match sn with
    | None -> res
    | Some (Seen sn) ->
        let shape = Prim.shape res in
        let wide (type w r) (y : (w, r, d) Value.t) =
          spread apply ~by shape (reshaped apply ~by (carry_shape t) y)
        in
        let fq = Prim.expect qd (Value.Any sn.q) in
        let prog =
          P.of_exprs
            ~ins:[| D.Any D.Bool; D.Any qd; D.Any qd |]
            [|
              E.any (E.where (E.input 0 D.Bool) (E.input 1 qd) (E.input 2 qd));
            |]
        in
        let loads = Value.[| Plain (wide sn.f); Plain (wide fq); Plain res |] in
        let y, () =
          apply ~by
            (Value.Map
               {
                 layout = L.contiguous shape;
                 prog;
                 outs = Value.[ qd ];
                 loads;
               })
        in
        y
  in
  (* A carry of rows, as a scan's, on [h]. *)
  let as_carry h t (Unit u) =
    Value.Any (reshaped apply ~by (carry_shape t) (moved apply ~by h u.s))
  in
  let monoid = ln.monoid and into = ln.rounds in
  let (D.Any qd) = into in
  let run = ref None in
  Array.mapi
    (fun t { s; e; _ } ->
      let h = here t in
      let (Value.Any v) = Lazy.force x.(t) in
      let carry_at b =
        if b >= chunk then Some (as_carry h t (Option.get carries.(b / chunk)))
        else None
      in
      let start = (s + chunk - 1) / chunk * chunk in
      let b0 = Int.max s start in
      let carried_in =
        Option.map (fun (Value.Any r) -> Value.Any (moved apply ~by h r)) !run
      in
      (* A tile starting inside a chunk continues its running result there;
         from its first chunk's start, it continues the carry into it. Where
         earlier terms hold a NaN, every result is the first. *)
      let head =
        if s mod chunk = 0 then None
        else
          let y =
            scanned apply ~by axis
              (cut_along apply ~by axis v 0 (Int.min e start - s))
              (Monoid (monoid, 0, qd)) carried_in
          in
          Some (filled t qd before.(t) y)
      in
      let body =
        if b0 >= e then None
        else
          let y =
            scanned apply ~by axis
              (cut_along apply ~by axis v (b0 - s) (e - b0))
              (Monoid (monoid, 0, qd)) (carry_at b0)
          in
          let head_seen =
            if Option.is_some head then fold [ List.hd (pieces_of t) ] else None
          in
          Some (filled t qd (after h before.(t) head_seen) y)
      in
      let res =
        match (head, body) with
        | Some a, None -> a
        | None, Some b -> b
        | Some a, Some b ->
            let shape = Prim.shape v in
            let region off len =
              let r = Array.map full shape in
              r.(axis) <- { M.start = off; count = len; step = 1 };
              r
            in
            apply ~by
              (Value.Assemble
                 {
                   dtype = qd;
                   shape = Array.copy shape;
                   fill = D.zero qd;
                   pieces =
                     [ (region 0 (b0 - s), a); (region (b0 - s) (e - b0), b) ];
                 })
        | None, None -> invalid_arg "Partial.lane_tiles: an empty tile"
      in
      (* The running result before the next tile, in the carrier, where that
         tile starts inside a chunk. *)
      (run :=
         if e >= n || e mod chunk = 0 then None
         else if same ln.carrier into then
           Some (Value.Any (cut_along apply ~by axis res (e - s - 1) 1))
         else
           let l0 = Int.max s (e / chunk * chunk) in
           let from =
             if l0 = s && s mod chunk <> 0 then carried_in else carry_at l0
           in
           let y =
             scanned apply ~by axis
               (cut_along apply ~by axis v (l0 - s) (e - l0))
               (Monoid (monoid, 0, c)) from
           in
           Some (Value.Any (cut_along apply ~by axis y (e - l0 - 1) 1)));
      Value.Any res)
    tiles

let scan (type d a) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (route : d Route.t) layout axis (prog : P.t) (r : (d, a) Value.reduction)
    (loads : d Value.load array) : a =
  let shape = L.shape layout in
  let n = shape.(axis) in
  let p = route.result in
  let set = Devices.set p in
  let ks = Grid.devices (Devices.grid p) in
  let windows = Array.mapi (fun j _ -> Devices.window ~by p shape j) ks in
  let o, (D.Any into as dt) =
    match r with
    | Monoid (_, o, dt) -> (o, D.Any dt)
    | Arg (_, o, dt) -> (o, D.Any dt)
    | Moments (o, dt) -> (o, D.Any dt)
  in
  let term = P.dtype prog (Iarray.get prog.outs o) in
  let (D.Any t) = term in
  let floats = match D.kind t with D.Float | D.Complex -> true | _ -> false in
  (* A complex sum scans each part as a float. *)
  let part =
    match (r, D.kind t) with
    | Monoid (Sum, _, _), D.Complex -> Some (float_part term)
    | _ -> None
  in
  let off w = Array.mapi (fun a rg -> if a = axis then full 0 else rg) w in
  let results = Array.make (Array.length ks) None in
  let group j =
    let members =
      List.filter
        (fun i -> off windows.(i) = off windows.(j))
        (List.init (Array.length ks) Fun.id)
    in
    let held =
      List.filter (fun i -> windows.(i).(axis).count > 0) members
    in
    let starts =
      List.sort_uniq compare
        (List.map (fun i -> windows.(i).(axis).start) held)
    in
    let tiles =
      Array.of_list
        (List.map
           (fun s ->
             let i = List.find (fun i -> windows.(i).(axis).start = s) held in
             let rg = windows.(i).(axis) in
             (i, { s; e = s + rg.count; holder = ks.(i) }))
           starts)
    in
    let terms (i, _) =
      lazy
        (let v =
           window_terms apply ~by set layout prog o term loads ks.(i)
             windows.(i)
         in
         match part with
         | None -> v
         | Some (D.Any pt) ->
             let (Value.Any v) = v in
             Value.Any (apply ~by (Value.Bitcast (pt, v))))
    in
    let x = Array.map terms tiles in
    let ts = Array.map snd tiles in
    let computed =
      if Array.length ts = 0 then [||]
      else
        match r with
        | Monoid (((Sum | Prod) as m), _, _) when floats ->
            let ln =
              match part with
              | None -> lane m term dt
              | Some pt -> lane m pt (float_part dt)
            in
            let ys = lane_tiles apply ~by ln set axis n ts x in
            Array.map
              (fun y ->
                match part with
                | None -> [ y ]
                | Some _ ->
                    let (Value.Any y) = y in
                    [ Value.Any (apply ~by (Value.Bitcast (into, y))) ])
              ys
        | Monoid (((Sum | Prod | Max | Min) as m), _, _) ->
            let k : P.binary =
              match m with
              | Sum -> Add
              | Prod -> Mul
              | Max -> Maximum
              | Min | Logsumexp -> Minimum
            in
            let plan =
              if floats then Extreme_scan (m, k) else Exact_scan (m, k)
            in
            exact_tiles apply ~by set axis plan dt ts x
        | Arg (e, _, _) -> exact_tiles apply ~by set axis (Arg_scan e) dt ts x
        | Monoid (Logsumexp, _, _) | Moments _ ->
            invalid_arg "Partial.scan: a scan that reads its axis whole"
    in
    List.iter
      (fun i ->
        let here = Devices.one set ks.(i) in
        let w = windows.(i) in
        let vs =
          if w.(axis).count = 0 then
            let viewed =
              Array.map (fun l -> view_load ~by ks.(i) here l w) loads
            in
            values r
              (apply ~by
                 (Prim.within
                    (Value.Scan
                       {
                         layout;
                         axis;
                         prog;
                         reduction = r;
                         loads = viewed;
                         carry = None;
                       })
                    w))
          else
            let t =
              Option.get (Array.find_index (fun ti -> ti.s = w.(axis).start) ts)
            in
            computed.(t)
        in
        (* Each result on its device: a scan of an empty window, or of one
           term, gives values of no load, which lie on no set. *)
        let on_here (Value.Any y) = Value.Any (moved apply ~by here y) in
        results.(i) <- Some (List.map on_here vs))
      members
  in
  Array.iteri (fun j _ -> if Option.is_none results.(j) then group j) ks;
  let finals = Array.map (fun v -> [ Option.get v ]) results in
  one_result r (at_placement p shape finals 0)

(* Contractions over cut contracting axes *)

(* The window [w] of [x]'s whole that device [k] holds, as a value on
   [k]. *)
let held_at (type v s d) ~by (set : d Devices.t) k w (x : (v, s, d) Value.t) :
    (v, s, d) Value.t =
  Value.Array
    { at = Devices.one set k; a = Place.view ~by x k w; dead = Value.Live }

(* Device [k]'s window of [x] where it lies, as a value on [k]. *)
let own (type v s d) ~by (set : d Devices.t) k (x : (v, s, d) Value.t) :
    (v, s, d) Value.t =
  let p = Prim.placement x in
  held_at ~by set k (Devices.window ~by p (Prim.shape x) (position p k)) x

(* [x] with each group of [groups]' axes, in order, made one axis. *)
let grouped (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (groups : int array array) (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  let s = Prim.shape x in
  let order = Array.concat (Array.to_list groups) in
  let x =
    if Array.for_all2 ( = ) order (Array.init (Array.length s) Fun.id) then x
    else apply ~by (Value.Move (Permute order, x))
  in
  reshaped apply ~by
    (Array.map (fun g -> numel (Array.map (fun a -> s.(a)) g)) groups)
    x

(* A tile's operands on one device: [a] laid out as [[batch; rows; terms]] and
   [b] as [[batch; terms; columns]], the terms in their number's order. *)
type ('a, 'b, 'c, 'e, 'd) operands = {
  la : ('a, 'b, 'd) Value.t;
  lb : ('c, 'e, 'd) Value.t;
}

(* The terms [l, l + len) of [x]. *)
let terms_of (type a b c e d) (apply : 'q. by:string -> 'q Value.prim -> 'q)
    ~by (x : (a, b, c, e, d) operands) l len : (a, b, c, e, d) operands =
  {
    la = cut_along apply ~by 2 x.la l len;
    lb = cut_along apply ~by 1 x.lb l len;
  }

(* [x] on [here]. *)
let operands_on (type a b c e d) (apply : 'q. by:string -> 'q Value.prim -> 'q)
    ~by here (x : (a, b, c, e, d) operands) : (a, b, c, e, d) operands =
  { la = moved apply ~by here x.la; lb = moved apply ~by here x.lb }

(* [x], of [shape] but along [axis], whose region from [start] along [axis] is
   the piece [y]. *)
let piece (type v s d) (shape : int array) axis start (y : (v, s, d) Value.t) =
  let r = Array.map full shape in
  r.(axis) <- { M.start; count = (Prim.shape y).(axis); step = 1 };
  (r, y)

(* [pieces] assembled into [shape]. *)
let assembled (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    shape (pieces : (M.range array * (v, s, d) Value.t) list) :
    (v, s, d) Value.t =
  let dt = Prim.dtype (snd (List.hd pieces)) in
  apply ~by
    (Value.Assemble { dtype = dt; shape; fill = D.zero dt; pieces })

(* Two values of one shape and dtype added. *)
let added (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (Value.Any x : d Value.any) (y : d Value.any) : d Value.any =
  let dt = Prim.dtype x in
  Value.Any
    (fst (apply ~by (Prim.op2 ~by (Binary Add) dt x (Prim.expect dt y))))

(* The contraction of [x] from [init], both in the accumulator [acc], into
   it. *)
let contracted (type a b c e d) (apply : 'q. by:string -> 'q Value.prim -> 'q)
    ~by order (D.Any acc) (x : (a, b, c, e, d) operands)
    (init : d Value.any option) : d Value.any =
  let init = Option.map (Prim.expect acc) init in
  Value.Any
    (apply ~by
       (Value.Contract
          {
            order;
            batch = [| (0, 0) |];
            contracting = [| (2, 1) |];
            acc = D.Any acc;
            out = acc;
            a = x.la;
            b = x.lb;
            init;
          }))

(* A group's tiles: the device holding each, its operands laid out there, and
   the runs of the group's terms, in increasing number. *)
type ('a, 'b, 'c, 'e, 'd) tiles = {
  holder : int array;
  laid : ('a, 'b, 'c, 'e, 'd) operands Lazy.t array;
  held : run array;
}

(* The operands of the terms [lo, hi) of [ts] on [here], the terms of
   [parts]: each tile's part moved there once, then the parts assembled in
   term order. *)
let block (type a b c e d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    here (ts : (a, b, c, e, d) tiles) lo hi parts : (a, b, c, e, d) operands =
  let from t =
    let mine = List.filter (fun r -> r.owner = t) parts in
    let s = List.fold_left (fun s r -> Int.min s r.l) max_int mine in
    let e = List.fold_left (fun e r -> Int.max e (r.l + r.len)) 0 mine in
    let x = terms_of apply ~by (Lazy.force ts.laid.(t)) s (e - s) in
    (s, operands_on apply ~by here x)
  in
  let owners = List.sort_uniq compare (List.map (fun r -> r.owner) parts) in
  let froms = List.map (fun t -> (t, from t)) owners in
  let pieces =
    List.map
      (fun r ->
        let s, x = List.assoc r.owner froms in
        (r.g - lo, terms_of apply ~by x (r.l - s) r.len))
      parts
  in
  let x0 = snd (List.hd pieces) in
  let sa = Array.copy (Prim.shape x0.la) in
  let sb = Array.copy (Prim.shape x0.lb) in
  sa.(2) <- hi - lo;
  sb.(1) <- hi - lo;
  let pa = List.map (fun (o, x) -> piece sa 2 o x.la) pieces in
  let pb = List.map (fun (o, x) -> piece sb 1 o x.lb) pieces in
  { la = assembled apply ~by sa pa; lb = assembled apply ~by sb pb }

(* The lane-order sum of the [n] products of [ts], from no [init], on each
   device of [here]. A complex sum's product is two terms of each part. *)
let lane_sums (type a b c e d) (apply : 'q. by:string -> 'q Value.prim -> 'q)
    ~by (set : d Devices.t) order acc n (ts : (a, b, c, e, d) tiles) here :
    d Value.any list =
  let (D.Any dt) = acc in
  let per = if D.is D.Complex dt then 2 else 1 in
  let scale f (r : run) = { r with g = f r.g; l = f r.l; len = f r.len } in
  let t, leaves =
    tree (n * per) (Array.length ts.holder)
      (Array.map (scale (fun x -> x * per)) ts.held)
  in
  let unit lf =
    let at = Devices.one set ts.holder.(lf.dest) in
    let x =
      match List.map (scale (fun x -> x / per)) lf.parts with
      | [ r ] -> terms_of apply ~by (Lazy.force ts.laid.(r.owner)) r.l r.len
      | parts -> block apply ~by at ts (lf.lo / per) (lf.hi / per) parts
    in
    contracted apply ~by order acc x None
  in
  let units = Array.map unit leaves in
  let rec combine at = function
    | Leaf i ->
        let (Value.Any v) = units.(i) in
        Value.Any (moved apply ~by at v)
    | Node (l, r) -> added apply ~by (combine at l) (combine at r)
  in
  List.map (fun k -> combine (Devices.one set k) t) here

(* The chain-order sum of the products of [ts], from [init k] on the device
   [k] holding the first, each run continuing the sum before it, on the
   device holding the last. *)
let chain_sum (type a b c e d) (apply : 'q. by:string -> 'q Value.prim -> 'q)
    ~by (set : d Devices.t) order acc (ts : (a, b, c, e, d) tiles) init :
    d Value.any =
  let step sum (r : run) =
    let k = ts.holder.(r.owner) in
    let x = terms_of apply ~by (Lazy.force ts.laid.(r.owner)) r.l r.len in
    let from =
      match sum with
      | None -> init k
      | Some (Value.Any v : d Value.any) ->
          Some (Value.Any (moved apply ~by (Devices.one set k) v))
    in
    Some (contracted apply ~by order acc x from)
  in
  Option.get (Array.fold_left step None ts.held)

let contract (type v s d a b c e) (apply : 'q. by:string -> 'q Value.prim -> 'q)
    ~by (route : d Route.t) ~order ~batch ~contracting ~acc (out : (v, s) D.t)
    (a : (a, b, d) Value.t) (b : (c, e, d) Value.t)
    (init : (v, s, d) Value.t option) : (v, s, d) Value.t =
  let whole = Array.map (fun (i, _) -> Prim.dim a i) contracting in
  let shape = Prim.contract_shape ~by ~batch ~contracting ~acc a b init in
  let p = route.result in
  let set = Devices.set p and ks = Grid.devices (Devices.grid p) in
  let free side rank =
    let named pairs x = Array.exists (fun pr -> side pr = x) pairs in
    Array.of_list
      (List.filter
         (fun x -> not (named batch x || named contracting x))
         (List.init rank Fun.id))
  in
  let pairs side = (Array.map side batch, Array.map side contracting) in
  let a_axes =
    let batch, pairs = pairs fst in
    [| batch; free fst (Prim.rank a); pairs |]
  and b_axes =
    let batch, pairs = pairs snd in
    [| batch; pairs; free snd (Prim.rank b) |]
  in
  let tile k =
    let pa = Prim.placement a in
    let w = Devices.window ~by pa (Prim.shape a) (position pa k) in
    Array.map (fun (i, _) -> w.(i)) contracting
  in
  let windows = Array.mapi (fun j _ -> Devices.window ~by p shape j) ks in
  let results = Array.make (Array.length ks) None in
  let group j =
    let members =
      List.filter
        (fun i -> windows.(i) = windows.(j))
        (List.init (Array.length ks) Fun.id)
    in
    let holders =
      List.filter (fun i -> numel (counts (tile ks.(i))) > 0) members
    in
    let tiles =
      Array.of_list
        (List.sort_uniq compare (List.map (fun i -> tile ks.(i)) holders))
    in
    let holder =
      Array.map
        (fun t -> ks.(List.find (fun i -> tile ks.(i) = t) holders))
        tiles
    in
    let laid k =
      lazy
        {
          la = grouped apply ~by a_axes (own ~by set k a);
          lb = grouped apply ~by b_axes (own ~by set k b);
        }
    in
    let held =
      Array.of_list
        (List.sort
           (fun r r' -> Int.compare r.g r'.g)
           (List.concat (Array.to_list (Array.mapi (runs whole) tiles))))
    in
    let ts = { holder; laid = Array.map laid holder; held } in
    let window = counts windows.(j) in
    (* [init] over [k]'s window of the result, as [[batch; rows;
       columns]], in the accumulator. The route reads [init] where it
       lies, which holds that window but may be wider: whole on every
       device beside a cut result. *)
    let init_on k =
      let nb = Array.length batch and nr = Array.length a_axes.(1) in
      let part lo len = numel (Array.sub window lo len) in
      let nc = Array.length window - nb - nr in
      let flat = [| part 0 nb; part nb nr; part (nb + nr) nc |] in
      Option.map
        (fun i ->
          cast apply ~by acc
            (reshape apply ~by flat
               (Value.Any (held_at ~by set k windows.(j) i))))
        init
    in
    let rounded v = reshape apply ~by window (cast apply ~by (D.Any out) v) in
    match order with
    | Nx_kernel.Lane ->
        let sums =
          lane_sums apply ~by set order acc (numel whole) ts
            (List.map (fun i -> ks.(i)) members)
        in
        List.iter2
          (fun i s ->
            let s =
              match init_on ks.(i) with
              | None -> s
              | Some x -> added apply ~by s x
            in
            results.(i) <- Some (rounded s))
          members sums
    | Chain ->
        let sum = chain_sum apply ~by set order acc ts init_on in
        let (Value.Any r) = rounded sum in
        let on i = Value.Any (moved apply ~by (Devices.one set ks.(i)) r) in
        List.iter (fun i -> results.(i) <- Some (on i)) members
  in
  Array.iteri (fun j _ -> if Option.is_none results.(j) then group j) ks;
  let finals = Array.map (fun v -> [ [ Option.get v ] ]) results in
  Prim.expect out (List.hd (at_placement p shape finals 0))

let reduce (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by route layout axes
    prog rs loads =
  let module F = Make (struct
    let by = by
    let apply = apply
  end) in
  F.reduce route layout axes prog rs loads
