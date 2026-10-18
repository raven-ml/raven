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

(* The terms of a block of lane order ({!Nx_kernel.section-orders}), and of
   a scan's chunk ({!Nx_kernel.type-reduction}). *)
let block = 1024
let chunk = 4096

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

  (* [x] of [shape], its axes of extent 1 stretched. *)
  let spread (type v s d) shape (x : (v, s, d) Value.t) : (v, s, d) Value.t =
    if Prim.has_shape x shape then x
    else apply (Value.Move (Broadcast (Array.copy shape), x))

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

  let listed : type d a. (d, a) Value.reduction -> a -> d Value.any list =
    function
    | Monoid _ -> fun v -> [ Value.Any v ]
    | Moments _ -> fun (m, v) -> [ Value.Any m; Value.Any v ]
    | Arg _ -> fun (e, p) -> [ Value.Any e; Value.Any p ]

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

  (* The partials [value i] of [tree]'s leaves combined up it by [node] on
     [here j] for each member [j] of [g], each given to [settle j]. *)
  let combine here g tree value node settle =
    List.iter (fun j -> settle j (fold node (here j) value tree)) g.members

  (* Lane order's tree over [g]'s [n] terms, whose runs are [held], its
     leaves' partials by [unit], combined as {!combine}. *)
  let lane_fold here g n held unit node settle =
    let tree, leaves = tree n (Array.length g.tiles) held in
    let values = Array.map unit leaves in
    combine here g tree (Array.get values) node settle

  (* The tile of each device of [route]'s result, in order: its window along
     [axes] of operand [0], of [shape], where [route] reads it. The route
     reads every operand alike along them. *)
  let tiles (route : 'd Route.t) shape axes =
    let q = route.operands.(0) in
    Array.map
      (fun k ->
        let w = Devices.window ~by q shape (position q k) in
        Array.map (fun a -> w.(a)) axes)
      (Grid.devices (Devices.grid route.result))

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

  (* [pos] shifted by [s]. *)
  let shift s pos =
    if s = 0 then pos
    else
      let s = E.const D.Int64 (Int64.of_int s) in
      List.hd (map [ pos ] [ E.any E.(input 0 D.Int64 + s) ])

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
    let tiles =
      let (Value.Any x) = Prim.load_any loads.(0) in
      tiles route (Prim.shape x) axes
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
      let store settle j vs = per.(j) <- settle j vs in
      (* One partial per tile, by [unit] on its holder, in tile order. *)
      let by_tiles unit node settle =
        List.iter
          (fun g ->
            let values = Array.map unit g.holder in
            combine here g (chain (Array.length g.tiles)) (Array.get values) node
              (store settle))
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
            let settle j vs =
              let v = settle m vs in
              let (Value.Any v) = if parts then bitcast (D.Any dt) v else v in
              [ Value.Any (reshape (counts (outw j)) v) ]
            in
            lane_fold here g (numel whole) g.held value (lane (binary m))
              (store settle))
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

  (* Scans *)

  let scan (type d a) (route : d Route.t) layout axis prog
      (r : (d, a) Value.reduction) (loads : d Value.load array) : a =
    let shape = L.shape layout in
    let n = shape.(axis) in
    let p = route.result in
    let set = Devices.set p and ks = Grid.devices (Devices.grid p) in
    let rule =
      Prim.rule
        (Value.Scan { layout; axis; prog; reduction = r; loads; carry = None })
    in
    let win j = Devices.window ~by p shape j in
    let here j = Devices.one set ks.(j) in
    let viewed j =
      let tile _ = invalid_arg "Partial.scan: a reduced axis" in
      Array.mapi
        (fun i l ->
          let (Value.Any x) = Prim.load_any l in
          view_load ks.(j) (here j) (reads rule i (Prim.shape x) (win j) tile) l)
        loads
    in
    let o = snd (Prim.reduction r) in
    let term = P.dtype prog (Iarray.get prog.outs o) in
    let into =
      match r with
      | Monoid (_, _, dt) -> D.Any dt
      | Moments (_, dt) -> D.Any dt
      | Arg (_, _, dt) -> D.Any dt
    in
    (* A complex sum scans each part as a float. *)
    let parts =
      match r with Monoid (Sum, _, _) -> complex term | _ -> false
    in
    let t = if parts then part term else term in
    let (D.Any qd) = if parts then part into else into in
    (* A device's terms over its window. *)
    let own j =
      let x = terms layout prog o term (viewed j) (win j) in
      if parts then bitcast t x else x
    in
    (* [x]'s first [m] units of [len] terms each, reduced by [rs] of
       [prog]'s outputs, one row each along the axis. *)
    let units (Value.Any x : d Value.any) m len prog rs =
      let w = Prim.shape x in
      let split =
        Array.concat
          [
            Array.sub w 0 axis;
            [| m; len |];
            Array.sub w (axis + 1) (Array.length w - axis - 1);
          ]
      in
      let loads = [| Value.Plain (reshape split x) |] in
      Value.Reduce
        { layout = L.contiguous split; axes = [| axis + 1 |]; prog; reductions = rs; loads }
    in
    let rows (Value.Any x : d Value.any) m =
      let s = Array.copy (Prim.shape x) in
      s.(axis) <- m;
      s
    in
    let scanned (Value.Any x : d Value.any) reduction carry =
      let prog = identity (D.Any (Prim.dtype x)) in
      let layout = L.contiguous (Prim.shape x) in
      apply
        (Value.Scan { layout; axis; prog; reduction; loads = [| Plain x |]; carry })
    in
    (* Each result where the terms before [x] hold a NaN, by [fill] of the
       carry's values and the result, stretched to it. *)
    let filled (Value.Any y) carry fill =
      let ys = Prim.shape y in
      let wide (Value.Any c) = Value.Any (spread ys c) in
      List.hd (map (List.map wide carry @ [ Value.Any y ]) [ fill ])
    in
    (* By kind: whether units are chunks, a unit's partials, [m] units of
       [len] terms of [x] from term [at], how two combine, the earlier
       first, and the scan of [x], from term [at], from the partials of the
       terms before it. *)
    let chunked, total, node, continued =
      match r with
      | Monoid (((Sum | Prod) as m), _, _) when floats term ->
          (* Each chunk continues from the carry of the chunks before it;
             where one of their terms is a NaN, every result is the
             first. *)
          let c = Prim.carrier t in
          let total x m' len _ =
            let prog = flagged (identity t) 0 in
            lane_unit m c (D.Any qd) (rows x m') (fun rs -> units x m' len prog rs)
          in
          let continued carry x _ =
            let y = Value.Any (scanned x (Monoid (m, 0, qd)) (Option.map List.hd carry)) in
            match carry with
            | None -> [ y ]
            | Some cv ->
                let f = List.nth cv 1 and q = List.nth cv (q_at cv 0) in
                let fill = E.(where (input 0 D.Bool) (input 1 qd) (input 2 qd)) in
                [ filled y [ f; q ] (E.any fill) ]
          in
          (true, total, lane (binary m), continued)
      | Monoid (((Sum | Prod | Max | Min) as m), _, _) ->
          (* Exact: each tile continues from the tiles before it. A float
             extreme's carry is a term before the tile's, their first NaN
             where one is. *)
          let (D.Any c) = Prim.carrier t in
          let total x m' len _ =
            let v, () = apply (units x m' len (identity t) Value.[ Monoid (m, 0, c) ]) in
            [ Value.Any v ]
          in
          let continued carry x _ =
            [ Value.Any (scanned x (Monoid (m, 0, qd)) (Option.map List.hd carry)) ]
          in
          (false, total, exact (binary m), continued)
      | Arg (e, _, _) ->
          (* Each tile scans alone, its first extremes then combined with
             those before it. *)
          let total x m' len at =
            let (v, q), () =
              apply (units x m' len (identity t) Value.[ Arg (e, 0, qd) ])
            in
            [ Value.Any v; shift at (Value.Any q) ]
          in
          let continued carry x at =
            let v, q = scanned x (Arg (e, 0, qd)) None in
            let local = [ Value.Any v; shift at (Value.Any q) ] in
            match carry with
            | None -> local
            | Some cv ->
                let wide (Value.Any c) = Value.Any (spread (Prim.shape v) c) in
                first_of e (List.map wide cv) local
          in
          (false, total, first_of e, continued)
      | Monoid (Logsumexp, _, _) | Moments _ ->
          invalid_arg "Partial.scan: a scan that reads its axis whole"
    in
    let per = Array.make (Array.length ks) [] in
    let off j =
      let w = Array.copy (win j) in
      w.(axis) <- full 0;
      w
    in
    let tile j = [| (win j).(axis) |] in
    let group g =
      let nt = Array.length g.tiles in
      let s t = g.tiles.(t).(0).start in
      let e t = s t + g.tiles.(t).(0).count in
      (* Where the units start: each chunk, or each tile. *)
      let bounds =
        if chunked then Array.init ((n + chunk - 1) / chunk) (fun u -> u * chunk)
        else Array.init nt s
      in
      let nu = Array.length bounds in
      let hi u = if u + 1 < nu then bounds.(u + 1) else n in
      let tile_of i =
        let t = ref 0 in
        while e !t <= i do
          incr t
        done;
        !t
      in
      (* The unit holding a tile's first term, where its scan starts. *)
      let first t =
        let u = ref 0 in
        while !u + 1 < nu && bounds.(!u + 1) <= s t do
          incr u
        done;
        !u
      in
      let start t = bounds.(first t) in
      let slabs = Array.map (fun j -> lazy (own j)) g.holder in
      (* Tile [t]'s terms from its scan's start, on member [j], [x] its
         own: those before them moved there once from the tiles holding
         them. *)
      let from j t x =
        if start t = s t then x
        else
          let before =
            List.filter_map
              (fun rn ->
                if rn.owner < t && rn.g + rn.len > start t then
                  Some (clip (start t) (s t) rn)
                else None)
              (Array.to_list g.held)
          in
          let mine = { g = s t; l = 0; len = e t - s t; owner = t } in
          let slab t' = [ ((if t' = t then x else Lazy.force slabs.(t')), axis) ] in
          fst (List.hd (gather (here j) slab (start t) (e t) (before @ [ mine ])))
      in
      let extended =
        Array.mapi (fun t j -> lazy (from j t (Lazy.force slabs.(t)))) g.holder
      in
      (* The partials of the units whose last term a tile holds, but the
         axis's last unit, from its holder's terms. *)
      let totals =
        Array.init nt (fun t ->
            lazy
              (let mine =
                 List.filter
                   (fun u -> u < nu - 1 && tile_of (hi u - 1) = t)
                   (List.init nu Fun.id)
               in
               match mine with
               | [] -> (0, [])
               | u0 :: _ ->
                   let m = List.length mine and len = hi u0 - bounds.(u0) in
                   let (Value.Any x) = Lazy.force extended.(t) in
                   let x = slice axis (bounds.(u0) - start t) (m * len) x in
                   (u0, total (Value.Any x) m len bounds.(u0))))
      in
      let unit u =
        let u0, rows = Lazy.force totals.(tile_of (hi u - 1)) in
        List.map (fun (Value.Any v) -> Value.Any (slice axis (u - u0) 1 v)) rows
      in
      let units = Array.init nu (fun u -> lazy (unit u)) in
      let member j =
        let w = win j in
        if w.(axis).count = 0 then
          (* An empty window scans alone. *)
          let op =
            Value.Scan { layout; axis; prog; reduction = r; loads = viewed j; carry = None }
          in
          per.(j) <- moves (here j) (listed r (apply (Prim.within op w)))
        else
          let t = Option.get (Array.find_index (fun tl -> tl.(0) = w.(axis)) g.tiles) in
          let x = if j = g.holder.(t) then Lazy.force extended.(t) else from j t (own j) in
          let u = first t in
          let carry =
            if u = 0 then None
            else Some (fold node (here j) (fun i -> Lazy.force units.(i)) (chain u))
          in
          let drop (Value.Any y) =
            let y = Value.Any (slice axis (s t - start t) (e t - s t) y) in
            if parts then bitcast into y else y
          in
          per.(j) <- moves (here j) (List.map drop (continued carry x (start t)))
      in
      List.iter member g.members
    in
    List.iter group (groups (Array.length ks) ~key:off ~tile [| n |]);
    result r p shape per

  (* Contractions *)

  let contract (type v s d a b c e) (route : d Route.t) ~order ~batch
      ~contracting ~acc (out : (v, s) D.t) (a : (a, b, d) Value.t)
      (b : (c, e, d) Value.t) (init : (v, s, d) Value.t option) :
      (v, s, d) Value.t =
    let shape = Prim.contract_shape ~by ~batch ~contracting ~acc a b init in
    let whole = Array.map (fun (i, _) -> Prim.dim a i) contracting in
    let p = route.result in
    let set = Devices.set p and ks = Grid.devices (Devices.grid p) in
    let rule =
      Prim.rule
        (Value.Contract { order; batch; contracting; acc; out; a; b; init })
    in
    let resw j = Devices.window ~by p shape j in
    let here j = Devices.one set ks.(j) in
    let tiles = tiles route (Prim.shape a) (Array.map fst contracting) in
    let read (type w r) j i (x : (w, r, d) Value.t) =
      view ks.(j) (here j)
        (reads rule i (Prim.shape x) (resw j) (Array.get tiles.(j)))
        x
    in
    let free side rank =
      let named pairs x = Array.exists (fun pr -> side pr = x) pairs in
      Array.of_list
        (List.filter
           (fun x -> not (named batch x || named contracting x))
           (List.init rank Fun.id))
    in
    let a_axes =
      [| Array.map fst batch; free fst (Prim.rank a); Array.map fst contracting |]
    and b_axes =
      [| Array.map snd batch; Array.map snd contracting; free snd (Prim.rank b) |]
    in
    (* A device's [a] as [[batch; rows; terms]] and [b] as
       [[batch; terms; columns]], the terms in their number's order. *)
    let slab j =
      [
        (Value.Any (grouped a_axes (read j 0 a)), 2);
        (Value.Any (grouped b_axes (read j 1 b)), 1);
      ]
    in
    let (D.Any ad) = acc in
    let contracted (sl : d slab) init =
      match sl with
      | [ (Value.Any la, _); (Value.Any lb, _) ] ->
          let init = Option.map (Prim.expect ad) init in
          Value.Any
            (apply
               (Value.Contract
                  {
                    order;
                    batch = [| (0, 0) |];
                    contracting = [| (2, 1) |];
                    acc;
                    out = ad;
                    a = la;
                    b = lb;
                    init;
                  }))
      | _ -> invalid_arg "Partial.contract: two operands"
    in
    (* [init] over device [j]'s window of the result, as
       [[batch; rows; columns]], in the accumulator. *)
    let init_on j =
      let w = counts (resw j) in
      let nb = Array.length batch and nr = Array.length a_axes.(1) in
      let part lo len = numel (Array.sub w lo len) in
      let flat =
        [| part 0 nb; part nb nr; part (nb + nr) (Array.length w - nb - nr) |]
      in
      Option.map (fun i -> cast acc (Value.Any (reshape flat (read j 2 i)))) init
    in
    let rounded j v =
      let (Value.Any r) = cast (D.Any out) v in
      Value.Any (reshape (counts (resw j)) r)
    in
    let per = Array.make (Array.length ks) [] in
    let gs = groups (Array.length ks) ~key:resw ~tile:(Array.get tiles) whole in
    let group g =
      let slabs = Array.map (fun j -> lazy (slab j)) g.holder in
      let slab t = Lazy.force slabs.(t) in
      match (order : Nx_kernel.order) with
      | Lane ->
          (* A complex sum's product is two terms, one of each part. *)
          let two = if complex acc then 2 else 1 in
          let scale f r = { r with g = f r.g; l = f r.l; len = f r.len } in
          let value lf =
            let j = g.holder.(lf.dest) in
            let parts = List.map (scale (fun x -> x / two)) lf.parts in
            let sl = gather (here j) slab (lf.lo / two) (lf.hi / two) parts in
            [ contracted sl None ]
          in
          let settle j s =
            let s = match init_on j with None -> s | Some x -> exact Add s [ x ] in
            per.(j) <- [ rounded j (List.hd s) ]
          in
          lane_fold here g (numel whole * two)
            (Array.map (scale (fun x -> x * two)) g.held)
            value (exact Add) settle
      | Chain ->
          (* Each run continues the sum before it, on its holder. *)
          let step sum r =
            let j = g.holder.(r.owner) in
            let from =
              match sum with
              | None -> init_on j
              | Some (Value.Any v) -> Some (Value.Any (move (here j) v))
            in
            Some (contracted (cut (slab r.owner) r.l r.len) from)
          in
          let sum = Option.get (Array.fold_left step None g.held) in
          let (Value.Any r) = rounded (List.hd g.members) sum in
          List.iter (fun j -> per.(j) <- [ Value.Any (move (here j) r) ]) g.members
    in
    List.iter group gs;
    Prim.expect out (at_placement p shape (Array.map List.hd per))
end

let run (type r) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (op : r Prim.partial) : r =
  let module F = Make (struct
    let by = by
    let apply = apply
  end) in
  match op with
  | Reduce_partial { route; layout; axes; prog; reductions; loads } ->
      F.reduce route layout axes prog reductions loads
  | Scan_partial { route; layout; axis; prog; reduction; loads } ->
      F.scan route layout axis prog reduction loads
  | Contract_partial { route; order; batch; contracting; acc; out; a; b; init }
    ->
      F.contract route ~order ~batch ~contracting ~acc out a b init
