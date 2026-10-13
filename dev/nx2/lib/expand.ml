(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Nx_array.Dtype
module L = Nx_array.Layout
module P = Nx_kernel.Prog
module M = Nx_array.Move
open Build

(* The node a one-node program computes from its operands in order, which a
   kernel computes alone, if it is one: every node before its output reads
   operand [i] at index [i]. *)
let single (prog : P.t) =
  let n = Iarray.length prog.nodes in
  let ins_first =
    let ok = ref true in
    for i = 0 to n - 2 do
      match Iarray.get prog.nodes i with
      | P.In j when j = i -> ()
      | _ -> ok := false
    done;
    !ok
  in
  if Iarray.length prog.outs <> 1 || Iarray.get prog.outs 0 <> n - 1
     || not ins_first
  then None
  else
    let m = n - 1 in
    match Iarray.get prog.nodes m with
    | (Const _ | Coord _) as nd -> Some nd
    | Op1 (_, _, 0) as nd when m = 1 -> Some nd
    | Op2 (_, 0, 1) as nd when m = 2 -> Some nd
    | Op3 (_, 0, 1, 2) as nd when m = 3 -> Some nd
    | _ -> None

(* Node [i] of [prog] over [loads], as a one-node map of [shape] over [value j]
   for each node [j] it reads. *)
let node_map (type d) (apply : 'r. by:string -> 'r Value.prim -> 'r) ~by layout
    (prog : P.t) (loads : d Value.any array) (value : int -> d Value.any) i :
    d Value.any =
  let (D.Any dt) = P.dtype prog i in
  let one node reads =
    let ins =
      Iarray.of_array
        (Array.map (fun (Value.Any x) -> D.Any (Prim.dtype x)) reads)
    in
    let loads = Array.map (fun (Value.Any x) -> Value.Plain x) reads in
    let prog = P.of_node ~ins node in
    let v, () =
      apply ~by (Value.Map { layout; prog; outs = Value.[ dt ]; loads })
    in
    Value.Any v
  in
  match Iarray.get prog.nodes i with
  | In j -> loads.(j)
  | (Const _ | Coord _) as node -> one node [||]
  | Op1 (k, t, j) -> one (Op1 (k, t, 0)) [| value j |]
  | Op2 (k, j, l) -> one (Op2 (k, 0, 1)) [| value j; value l |]
  | Op3 (k, j, l, m) -> one (Op3 (k, 0, 1, 2)) [| value j; value l; value m |]

let rec results : type d r. (d, r) Value.outs -> d Value.any list -> r =
 fun outs vs ->
  match (outs, vs) with
  | [], [] -> ()
  | dt :: outs, v :: vs -> (Prim.expect dt v, results outs vs)
  | _ -> invalid_arg "Expand.results: one value per output"

(* Reductions *)

let same (D.Any a) (D.Any b) = D.code a = D.code b

(* The dtype a reduction of [dt] accumulates in: [float32] for the floats
   narrower than 32 bits, the byte-wide dtype of the sub-byte ones, whose
   integers wrap to the same bits, [dt] otherwise. *)
let accumulator (D.Any dt as d) =
  match dt with
  | D.Float16 | D.Bfloat16 | D.Float8_e4m3fn | D.Float8_e5m2 | D.Float4_e2m1fn
    ->
      D.Any D.Float32
  | D.Int4 -> D.Any D.Int8
  | D.Uint4 -> D.Any D.Uint8
  | D.Bit -> D.Any D.Bool
  | _ -> d

let output (prog : P.t) k = P.dtype prog (Iarray.get prog.outs k)

(* The output a reduction reads and its result's dtype. *)
let target : type d a. (d, a) Value.reduction -> int * D.any = function
  | Monoid (_, k, dt) -> (k, D.Any dt)
  | Moments (k, dt) -> (k, D.Any dt)
  | Arg (_, k, dt) -> (k, D.Any dt)

(* The float format of a complex dtype's parts. *)
let parts (D.Any dt) =
  match dt with
  | D.Complex64 -> Some (D.Any D.Float32)
  | D.Complex128 -> Some (D.Any D.Float64)
  | _ -> None

(* Whether [r] is a sum of complex numbers, which adds their parts. *)
let complex_sum : type d a. (d, a) Value.reduction -> bool = function
  | Monoid (Sum, _, dt) -> parts (D.Any dt) <> None
  | Monoid _ | Moments _ | Arg _ -> false

(* Whether a loop of [prog] by [r] is plain: the identity of its operand,
   accumulated and rounded in the operand's own dtype, and no complex sum. Any
   other expands into plain loops. *)
let plain prog r =
  let k, dt = target r in
  P.is_identity prog && k = 0
  && same (output prog 0) dt
  && same (accumulator dt) dt
  && not (complex_sum r)

(* Whether a scan is core: a plain [Sum], [Prod], [Max] or [Min]. *)
let core_scan (type d a) prog (r : (d, a) Value.reduction) =
  match r with
  | Monoid ((Sum | Prod | Max | Min), _, _) -> plain prog r
  | Monoid (Logsumexp, _, _) | Moments _ | Arg _ -> false

(* Whether a reduction is core: one core scan's reduction. *)
let core_reduce (type d q) prog (rs : (d, q) Value.reductions) =
  match rs with [ r ] -> core_scan prog r | _ -> false

(* Output [k] of [prog] over [loads], of [layout]'s shape, stored in [acc]: the
   operand itself where [prog] is the identity and [acc] its dtype. *)
let body (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by layout
    (prog : P.t) k (loads : d Value.load array) (D.Any acc) : d Value.any =
  let out = Iarray.get prog.outs k in
  let cast = not (same (P.dtype prog out) (D.Any acc)) in
  match loads with
  | [| Value.Plain x |] when P.is_identity prog && not cast -> Value.Any x
  | _ ->
      let nodes, last =
        if cast then
          ( Iarray.append prog.nodes [| P.Op1 (Cast, D.Any acc, out) |],
            Iarray.length prog.nodes )
        else (prog.nodes, out)
      in
      let prog = P.v ~ins:prog.ins nodes ~outs:[| last |] in
      let v, () =
        apply ~by (Value.Map { layout; prog; outs = Value.[ acc ]; loads })
      in
      Value.Any v

(* [x] stored in [dt]. *)
let cast_to (type v s w r d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (dt : (w, r) D.t) (x : (v, s, d) Value.t) : (w, r, d) Value.t =
  match D.equal_witness (Prim.dtype x) dt with
  | Some Type.Equal -> x
  | None ->
      let y, () = apply ~by (Prim.op1 ~by Cast dt x) in
      y

(* The reduction [r] of [prog] over [loads] as a plain [loop] over its output in
   its accumulator, then rounded to its dtype; [None] where no plain loop gives
   it. *)
let reduction (type d a) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    ~(loop :
       'q.
       Nx_array.Layout.t ->
       P.t ->
       (d, 'q) Value.reduction ->
       d Value.load array ->
       'q) layout prog (loads : d Value.load array) (r : (d, a) Value.reduction)
    : a option =
  match r with
  | Monoid (m, k, dt) ->
      let (Value.Any b) =
        body apply ~by layout prog k loads (accumulator (output prog k))
      in
      let bt = Prim.dtype b in
      let plain (type v s) (x : (v, s, d) Value.t) =
        let xt = Prim.dtype x in
        loop
          (Nx_array.Layout.contiguous (Prim.shape x))
          (P.of_node ~ins:[| D.Any xt |] (In 0))
          (Value.Monoid (m, 0, xt))
          [| Value.Plain x |]
      in
      let v =
        match parts (D.Any bt) with
        | Some (D.Any pt) when complex_sum r ->
            (* The parts, an axis of two after the others, summed apart. *)
            let pairs = apply ~by (Value.Bitcast (pt, b)) in
            apply ~by (Value.Bitcast (bt, plain pairs))
        | Some _ | None -> plain b
      in
      Some (cast_to apply ~by dt v)
  | Moments _ | Arg _ -> None

(* Optional reductions *)

(* [x], [shape] reduced along [axes], broadcast back to [shape]. *)
let spread_back (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    shape axes (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  let units = Array.mapi (fun a e -> if Array.mem a axes then 1 else e) shape in
  let move mv x = apply ~by (Value.Move (mv, x)) in
  let x = if Prim.has_shape x units then x else move (M.Reshape units) x in
  if units = shape then x else move (M.Broadcast (Array.copy shape)) x

(* The map of [prog] over [loads], of [shape], into [dt]. *)
let map1 (type w r d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by shape
    prog (dt : (w, r) D.t) (loads : d Value.load array) : (w, r, d) Value.t =
  fst
    (apply ~by
       (Value.Map
          { layout = L.contiguous shape; prog; outs = Value.[ dt ]; loads }))

(* The monoid [m] of [prog]'s output over [loads], of [shape], along [axes], in
   [dt]. *)
let reduce1 (type w r d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by shape
    axes m prog (dt : (w, r) D.t) (loads : d Value.load array) :
    (w, r, d) Value.t =
  fst
    (apply ~by
       (Value.Reduce
          {
            layout = L.contiguous shape;
            axes;
            prog;
            reductions = Value.[ Monoid (m, 0, dt) ];
            loads;
          }))

(* [x] by the monoid [m] along [axes]. *)
let reduce_plain (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    axes m (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  let dt = Prim.dtype x in
  reduce1 apply ~by (Prim.shape x) axes m
    (P.of_node ~ins:[| D.Any dt |] (In 0))
    dt [| Value.Plain x |]

(* The nodes of [x] where it is a NaN, the first operand where so, else the
   second: [Where (x ≠ x) x y]. *)
let nan_or b x y =
  let nan = push b (Op2 (Compare Not_equal, x, x)) in
  push b (Op3 (Where, nan, x, y))

(* Logsumexp: [m' + log Σ exp (x - m')] along [axes], [m'] the terms' maximum
   where finite and [0] elsewhere, so that no term is shifted by an infinity;
   where the maximum is a NaN, the maximum itself, the first NaN term. No term:
   [log 0], [-∞]. *)
let logsumexp (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    shape axes (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  let dt = Prim.dtype x in
  let out = Prim.reduced shape axes in
  let f b v = const b dt (D.of_float dt v) in
  let sum_exp shift =
    let b = Build.make () in
    let t = input b (D.Any dt) in
    let shifted =
      match shift with
      | None -> t
      | Some _ -> push b (Op2 (Binary Sub, t, input b (D.Any dt)))
    in
    let e = push b (Op1 (Unary Exp, D.Any dt, shifted)) in
    let loads =
      match shift with
      | None -> [| Value.Plain x |]
      | Some m ->
          [| Value.Plain x; Value.Plain (spread_back apply ~by shape axes m) |]
    in
    reduce1 apply ~by shape axes Sum (program b [| e |]) dt loads
  in
  if Array.exists (fun a -> shape.(a) = 0) axes then
    map1 apply ~by out
      (P.of_node ~ins:[| D.Any dt |] (Op1 (Unary Log, D.Any dt, 0)))
      dt
      [| Value.Plain (sum_exp None) |]
  else
    let m = reduce_plain apply ~by axes Max x in
    (* [m'], from the maximum at node [m]. *)
    let finite b m =
      let a = push b (Op1 (Unary Abs, D.Any dt, m)) in
      let lt = push b (Op2 (Compare Less, a, f b Float.infinity)) in
      push b (Op3 (Where, lt, m, f b 0.))
    in
    let shift =
      let b = Build.make () in
      let mi = input b (D.Any dt) in
      map1 apply ~by out (program b [| finite b mi |]) dt [| Value.Plain m |]
    in
    let s = sum_exp (Some shift) in
    let b = Build.make () in
    let mi = input b (D.Any dt) in
    let si = input b (D.Any dt) in
    let r =
      push b
        (Op2 (Binary Add, finite b mi, push b (Op1 (Unary Log, D.Any dt, si))))
    in
    map1 apply ~by out
      (program b [| nan_or b mi r |])
      dt
      [| Value.Plain m; Value.Plain s |]

(* Moments in two passes: the mean, [Σ x / n], then the variance, [Σ (x - mean)²
   / n], each a sum. Where the mean is a NaN, both are the mean. *)
let moments (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    shape axes (x : (v, s, d) Value.t) : (v, s, d) Value.t * (v, s, d) Value.t =
  let dt = Prim.dtype x in
  let out = Prim.reduced shape axes in
  let n = Array.fold_left (fun n a -> n * shape.(a)) 1 axes in
  (* [s / n] at node [s], or [s] itself where it is a NaN, from [over]. *)
  let divided b s over =
    let count = const b dt (D.of_float dt (Float.of_int n)) in
    let q = push b (Op2 (Binary Fdiv, s, count)) in
    nan_or b over q
  in
  let mean =
    let b = Build.make () in
    let s = input b (D.Any dt) in
    map1 apply ~by out
      (program b [| divided b s s |])
      dt
      [| Value.Plain (reduce_plain apply ~by axes Sum x) |]
  in
  let squares =
    let b = Build.make () in
    let x' = input b (D.Any dt) in
    let mean' = input b (D.Any dt) in
    let d = push b (Op2 (Binary Sub, x', mean')) in
    let sq = push b (Op2 (Binary Mul, d, d)) in
    reduce1 apply ~by shape axes Sum
      (program b [| sq |])
      dt
      [| Value.Plain x; Value.Plain (spread_back apply ~by shape axes mean) |]
  in
  let var =
    let b = Build.make () in
    let s = input b (D.Any dt) in
    let m = input b (D.Any dt) in
    map1 apply ~by out
      (program b [| divided b s m |])
      dt
      [| Value.Plain squares; Value.Plain mean |]
  in
  (mean, var)

(* Element [j] of the last axis of [pairs], an axis of two after [shape]'s, as a
   view of [shape]. *)
let half (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by shape
    (pairs : (v, s, d) Value.t) j : (v, s, d) Value.t =
  let ranges =
    Array.append
      (Array.map (fun e -> { M.start = 0; count = e; step = 1 }) shape)
      [| { M.start = j; count = 1; step = 1 } |]
  in
  let p = apply ~by (Value.Move (Slice ranges, pairs)) in
  apply ~by (Value.Move (Reshape (Array.copy shape), p))

(* Values whose elements are all equal at two indices iff [x]'s bits are there:
   [x] itself for an integer or a boolean, its bits read as an unsigned integer
   of its width for a float or a complex number, each part's for a
   complex128. *)
let keys (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (x : (v, s, d) Value.t) : d Value.any list =
  let dt = Prim.dtype x in
  let read (type w r) (u : (w, r) D.t) =
    Value.Any (apply ~by (Value.Bitcast (u, x)))
  in
  match D.kind dt with
  | D.Signed | D.Unsigned | D.Boolean -> [ Value.Any x ]
  | D.Float | D.Complex -> (
      match D.bits dt with
      | 4 -> [ read D.Uint4 ]
      | 8 -> [ read D.Uint8 ]
      | 16 -> [ read D.Uint16 ]
      | 32 -> [ read D.Uint32 ]
      | 64 -> [ read D.Uint64 ]
      | _ ->
          let words = apply ~by (Value.Bitcast (D.Uint64, x)) in
          let part j = Value.Any (half apply ~by (Prim.shape x) words j) in
          [ part 0; part 1 ])

(* Arg: the extreme [m], then the least position along [axes], in C order of the
   reduced indices, whose term has [m]'s bits, or, where [m] is a NaN, is a
   NaN. *)
let arg (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by shape
    axes (e : Nx_kernel.extreme) (x : (v, s, d) Value.t) :
    (v, s, d) Value.t * (int64, D.int64_elt, d) Value.t =
  let dt = Prim.dtype x in
  let m =
    reduce_plain apply ~by axes (match e with Max -> Max | Min -> Min) x
  in
  let mb = spread_back apply ~by shape axes m in
  let kx = keys apply ~by x and km = keys apply ~by mb in
  let nan = match D.kind dt with D.Float | D.Complex -> true | _ -> false in
  let values = if nan then [ Value.Any x; Value.Any mb ] else [] in
  let loads = Array.of_list (values @ kx @ km) in
  (* Nodes [0] to [n - 1] are the [n] loads: [x] and [m] where they may be NaNs,
     then [x]'s keys, then [m]'s. *)
  let b = Build.make () in
  let input (Value.Any y) = ignore (input b (D.Any (Prim.dtype y))) in
  Array.iter input loads;
  let v = List.length values and k = List.length kx in
  let equal j = push b (Op2 (Compare Equal, v + j, v + k + j)) in
  let same =
    List.fold_left
      (fun a j -> push b (Op2 (Binary And, a, equal j)))
      (equal 0)
      (List.init (k - 1) succ)
  in
  let found =
    if not nan then same
    else
      let isnan i = push b (Op2 (Compare Not_equal, i, i)) in
      let both = push b (Op2 (Binary And, isnan 0, isnan 1)) in
      push b (Op2 (Binary Or, same, both))
  in
  let int64 n = const b D.Int64 (Int64.of_int n) in
  let position, _ =
    Array.fold_right
      (fun a (pos, stride) ->
        let c = push b (Coord a) in
        let t = push b (Op2 (Binary Mul, c, int64 stride)) in
        (push b (Op2 (Binary Add, pos, t)), stride * shape.(a)))
      axes
      (int64 0, 1)
  in
  let none = const b D.Int64 Int64.max_int in
  let pick = push b (Op3 (Where, found, position, none)) in
  let p =
    reduce1 apply ~by shape axes Min (program b [| pick |]) D.Int64
      (Array.map (fun (Value.Any y) -> Value.Plain y) loads)
  in
  (m, p)

(* Running reductions *)

(* The outputs of a map whose dtypes are known at run time, and how to read its
   results as values. *)
type 'd outputs =
  | Outputs : ('d, 'r) Value.outs * ('r -> 'd Value.any list) -> 'd outputs

let rec outputs : type d. D.any list -> d outputs = function
  | [] -> Outputs (Value.[], fun () -> [])
  | D.Any dt :: dts ->
      let (Outputs (outs, read)) = outputs dts in
      Outputs (Value.( :: ) (dt, outs), fun (v, r) -> Value.Any v :: read r)

(* The map of [prog] over [loads], of [shape], into [dts]. *)
let map_n (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by shape
    prog dts (loads : d Value.load array) : d Value.any array =
  let (Outputs (outs, read)) = outputs (Array.to_list dts) in
  let layout = L.contiguous shape in
  Array.of_list (read (apply ~by (Value.Map { layout; prog; outs; loads })))

(* The ranges of [shape] that keep its elements [start + j step], [j < count],
   along [axis] and all others elsewhere. *)
let along_axis shape axis start count step =
  let keep a e =
    if a = axis then { M.start; count; step }
    else { M.start = 0; count = e; step = 1 }
  in
  Array.mapi keep shape

(* [x]'s elements [start + j step], [j < count], along [axis]. *)
let part (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by axis
    start count step (Value.Any x : d Value.any) : d Value.any =
  let ranges = along_axis (Prim.shape x) axis start count step in
  Value.Any (apply ~by (Value.Move (Slice ranges, x)))

let plain (Value.Any x) = Value.Plain x

(* Ladner and Fischer's prefix circuit at its least size, the tree of Brent
   and Kung:

     s     s0  s1  s2  s3  s4
     p     c(s0,s1)  c(s2,s3)          the pairs
     q     the scan of p, recursively
     y     s0  q0  c(q0,s2)  q1  c(q1,s4)

   Output [i]'s depth [f(i)] follows [f(0) = 0], [f(2j+1) = 1 + f(j)] and
   [f(2j) = 2 + f(j-1)], so [f(i) <= max(0, 2⌈log₂(i+1)⌉ - 1)]. *)
let pairwise (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by shape
    axis ?first combine (state : d Value.any array) : d Value.any array =
  let dts = Array.map (fun (Value.Any x) -> D.Any (Prim.dtype x)) state in
  let along e =
    let s = Array.copy shape in
    s.(axis) <- e;
    s
  in
  let parts start count step =
    Array.map (part apply ~by axis start count step)
  in
  (* The states [combine earlier later] of the [e] indices of the two. *)
  let combined e earlier later =
    let b = Build.make () in
    let ea = Array.map (input b) dts in
    let la = Array.map (input b) dts in
    let prog = program b (Iarray.of_array (combine b ea la)) in
    let loads = Array.map plain (Array.append earlier later) in
    map_n apply ~by (along e) prog dts loads
  in
  (* Index [0]'s result, from [state]'s first. *)
  let head first state =
    let s0 = parts 0 1 1 state in
    match first with
    | None -> s0
    | Some first ->
        let b = Build.make () in
        let s = Array.map (input b) dts in
        let prog = program b (Iarray.of_array (first b s)) in
        map_n apply ~by (along 1) prog dts (Array.map plain s0)
  in
  let rec level first state m =
    if m = 0 || (m = 1 && Option.is_none first) then state
    else if m = 1 then head first state
    else
      let half = m / 2 and rows = (m + 1) / 2 in
      let p = combined half (parts 0 half 2 state) (parts 1 half 2 state) in
      let q = level None p half in
      let evens =
        if rows = 1 then [||]
        else
          combined (rows - 1) (parts 0 (rows - 1) 1 q)
            (parts 2 (rows - 1) 2 state)
      in
      let h = head first state in
      let shape = along m in
      Array.mapi
        (fun k (Value.Any y0) ->
          let dt = Prim.dtype y0 in
          let piece start count v =
            (along_axis shape axis start count 2, Prim.expect dt v)
          in
          let pieces =
            piece 0 1 (Value.Any y0)
            :: piece 1 half q.(k)
            :: (if rows = 1 then [] else [ piece 2 (rows - 1) evens.(k) ])
          in
          let fill = D.zero dt in
          Value.Any
            (apply ~by (Value.Assemble { dtype = dt; shape; fill; pieces })))
        h
  in
  level first state shape.(axis)

(* The node of whether [v] is finite: [|v| < ∞], [false] for a NaN. *)
let finite b (D.Any dt) v =
  let inf = const b dt (D.of_float dt Float.infinity) in
  push b (Op2 (Compare Less, push b (Op1 (Unary Abs, D.Any dt, v)), inf))

(* Logsumexp by {!pairwise}: results [a] and [b] combine as
   [M + log1p (exp (m - M))], [M] their [Maximum] and [m] their [Minimum], or
   as [M] where [M] is not finite: [-∞] for terms all [-∞], [+∞] past a [+∞],
   the first NaN with its bits. Index [0] is its term [x] as [x + 0], so that
   a lone [-0] gives [+0], as the reduction does. Each combination shifts by
   its own maximum, so no prefix is lost below a later one. *)
let running_logsumexp (type v s d)
    (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by shape axis
    (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  let dt = Prim.dtype x in
  let combine b earlier later =
    let a = earlier.(0) and y = later.(0) in
    let big = push b (Op2 (Binary Maximum, a, y)) in
    let small = push b (Op2 (Binary Minimum, a, y)) in
    let gap = push b (Op2 (Binary Sub, small, big)) in
    let e = push b (Op1 (Unary Exp, D.Any dt, gap)) in
    let l = push b (Op1 (Unary Log1p, D.Any dt, e)) in
    let sum = push b (Op2 (Binary Add, big, l)) in
    [| push b (Op3 (Where, finite b (D.Any dt) big, sum, big)) |]
  in
  let first b s =
    [| push b (Op2 (Binary Add, s.(0), const b dt (D.of_float dt 0.))) |]
  in
  let y = pairwise apply ~by shape axis ~first combine [| Value.Any x |] in
  Prim.expect dt y.(0)

(* {!Nx_kernel.reduction}'s Moments scan by {!pairwise}, over Chan, Golub
   and LeVeque's combination, their (1.5b), of states [(μ, S, n, z)], [z]
   whether a term is a NaN.

   The derivation of its bounds, to first order, at index [i] of [N = i + 1]
   terms through [D = D_i] combinations, with [R] the terms' range and [V]
   their variance. The shift [c] is one of the terms, so each shifted term,
   each state's mean and each [δ] is within [R] of [0].
   - Mean. A term's shift rounds once, within [u R]. A combination's [μ] is
     [(1 - w) μa + w μb], [w = nb / n], a convex combination of its states'
     means, plus the roundings of [δ], [nb / n] and their product,
     [3 u |δ| w <= 3 u R], and of the sum, [u R]: its error is its states'
     largest plus [4 u R]. Through [D] combinations, [(4 D + 1) u R]; the
     sum [c + μ] adds [u |μ|].
   - Variance. Errors in [S] add up the tree. A combination rounds
     [Sa + Sb] within [u (Sa + Sb)], [δ δ (na w)] within a relative
     [6 u] (the rounding of [δ] twice through its square, the square,
     [w], [na w] and the product) and the sum within [u S]: at
     most [7 u S], as [Sa + Sb] and [δ² na nb / n] are at most [S]. The
     nodes at one distance from the root hold disjoint terms, and a union's
     [S] is at least the sum of its parts', so the [D] distances add at most
     [7 D u N V]. The states' mean errors [ea] and [eb] reach [δ²] as
     [2 |δ| (ea + eb)], and below a node at distance [k] from the root each
     is at most [(4 (D - k) - 3) u R]. With [w' = na nb / n <= n / 4],
     Cauchy and Schwarz over the nodes at distance [k] give
     [Σ |δ| w' <= √(Σ δ² w') √(Σ w') <= √(N V) √(N / 4)], so distance [k]
     adds [2 (4 (D - k) - 3) u R N √V], and the [D] distances
     [(4 D² - 2 D) u R N √V]. Divided by [N] and rounded once more, the
     variance is within [(7 D + 1) u V + (4 D² - 2 D) u R √V]. *)
let running_moments (type v s d)
    (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by shape axis
    (x : (v, s, d) Value.t) : (v, s, d) Value.t * (v, s, d) Value.t =
  let dt = Prim.dtype x in
  let f b v = const b dt (D.of_float dt v) in
  let layout = L.contiguous shape in
  let two prog loads =
    let a, (b, ()) =
      apply ~by (Value.Map { layout; prog; outs = Value.[ dt; dt ]; loads })
    in
    (a, b)
  in
  let finite b v = finite b (D.Any dt) v in
  (* [c], from the first term at node [t]: [t] where finite, else [0]. *)
  let shift b t = push b (Op3 (Where, finite b t, t, f b 0.)) in
  if shape.(axis) = 0 then
    let b = Build.make () in
    let zero = f b 0. in
    two (program b [| zero; zero |]) [||]
  else
    let first =
      let keep a e =
        { M.start = 0; count = (if a = axis then 1 else e); step = 1 }
      in
      let x0 = apply ~by (Value.Move (Slice (Array.mapi keep shape), x)) in
      apply ~by (Value.Move (Broadcast (Array.copy shape), x0))
    in
    let dts = [| D.Any dt; D.Any dt; D.Any dt; D.Any D.Bool |] in
    let terms =
      let b = Build.make () in
      let t = input b (D.Any dt) in
      let t0 = input b (D.Any dt) in
      let d = push b (Op2 (Binary Sub, t, shift b t0)) in
      let z = push b (Op2 (Compare Not_equal, t, t)) in
      map_n apply ~by shape
        (program b [| d; f b 0.; f b 1.; z |])
        dts
        [| Value.Plain x; Value.Plain first |]
    in
    let combine b earlier later =
      let ma = earlier.(0) and sa = earlier.(1) and na = earlier.(2) in
      let mb = later.(0) and sb = later.(1) and nb = later.(2) in
      let za = earlier.(3) and zb = later.(3) in
      let op2 k x y = push b (Op2 (k, x, y)) in
      let where c x y = push b (Op3 (Where, c, x, y)) in
      let n = op2 (Binary Add) na nb in
      let delta = op2 (Binary Sub) mb ma in
      let w = op2 (Binary Fdiv) nb n in
      let chan = op2 (Binary Add) ma (op2 (Binary Mul) delta w) in
      let both = op2 (Binary And) (finite b ma) (finite b mb) in
      let past = where za ma (where zb mb (op2 (Binary Add) ma mb)) in
      let mean = where both chan past in
      let naw = op2 (Binary Mul) na w in
      let between = op2 (Binary Mul) (op2 (Binary Mul) delta delta) naw in
      let s = op2 (Binary Add) (op2 (Binary Add) sa sb) between in
      [| mean; s; n; op2 (Binary Or) za zb |]
    in
    let state = pairwise apply ~by shape axis combine terms in
    let b = Build.make () in
    let m = input b (D.Any dt) in
    let s = input b (D.Any dt) in
    let c = shift b (input b (D.Any dt)) in
    let at = push b (Coord axis) in
    let at = push b (Op1 (Cast, D.Any dt, at)) in
    let n = push b (Op2 (Binary Add, at, f b 1.)) in
    let mean = push b (Op2 (Binary Add, c, m)) in
    let var = push b (Op2 (Binary Fdiv, s, n)) in
    let nan = push b (Op2 (Binary Sub, m, m)) in
    let var = push b (Op3 (Where, finite b m, var, nan)) in
    two
      (program b [| mean; var |])
      [|
        Value.Plain (Prim.expect dt state.(0));
        Value.Plain (Prim.expect dt state.(1));
        Value.Plain first;
      |]

(* Arg by {!pairwise} over the pair of an extreme and its position: the
   earlier pair holds unless the later's extreme is past it in [Max]'s or
   [Min]'s order, so ties keep the first position and the first NaN holds.
   The order reads the extreme's keys, itself or a complex number's parts,
   carried beside it: a NaN key is past every number, then the keys compare
   in turn, by [Less], and a float's equal keys by their reciprocals, so that
   [-0] is below [+0]. Exact. *)
let running_arg (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q)
    ~by shape axis (e : Nx_kernel.extreme) (x : (v, s, d) Value.t) :
    (v, s, d) Value.t * (int64, D.int64_elt, d) Value.t =
  let dt = Prim.dtype x in
  let position =
    let at = P.of_node ~ins:[||] (Coord axis) in
    map1 apply ~by shape at D.Int64 [||]
  in
  (* The keys' dtype, and the state's indices that hold them. *)
  let kt, keys, carried =
    match parts (D.Any dt) with
    | Some (D.Any pt) ->
        let pairs = apply ~by (Value.Bitcast (pt, x)) in
        let part j = Value.Any (half apply ~by shape pairs j) in
        (D.Any pt, [| 2; 3 |], [| part 0; part 1 |])
    | None -> (D.Any dt, [| 0 |], [||])
  in
  let (D.Any kdt) = kt in
  let floats = match D.kind kdt with D.Float -> true | _ -> false in
  let combine b earlier later =
    let op2 k x y = push b (Op2 (k, x, y)) in
    let where c x y = push b (Op3 (Where, c, x, y)) in
    let recip x = push b (Op1 (Unary Recip, kt, x)) in
    let before x y =
      let less = op2 (Compare Less) x y in
      if not floats then less
      else
        let zeros = op2 (Compare Less) (recip x) (recip y) in
        op2 (Binary Or) less (op2 (Binary And) (op2 (Compare Equal) x y) zeros)
    in
    let same x y =
      let equal = op2 (Compare Equal) x y in
      if not floats then equal
      else op2 (Binary And) equal (op2 (Compare Equal) (recip x) (recip y))
    in
    (* Whether keys [u] are below keys [v], comparing from key [i]. *)
    let rec below u v i =
      let lt = before u.(i) v.(i) in
      if i = Array.length u - 1 then lt
      else
        let rest = op2 (Binary And) (same u.(i) v.(i)) (below u v (i + 1)) in
        op2 (Binary Or) lt rest
    in
    let ka = Array.map (fun i -> earlier.(i)) keys in
    let kb = Array.map (fun i -> later.(i)) keys in
    let past = match e with Max -> below ka kb 0 | Min -> below kb ka 0 in
    let yes = const b D.Bool true and no = const b D.Bool false in
    let past =
      if not floats then past
      else
        let nan k =
          let isnan c i = op2 (Binary Or) c (op2 (Compare Not_equal) i i) in
          Array.fold_left isnan no k
        in
        where (nan ka) no (where (nan kb) yes past)
    in
    Array.map2 (fun a l -> where past l a) earlier later
  in
  let state =
    pairwise apply ~by shape axis combine
      (Array.append [| Value.Any x; Value.Any position |] carried)
  in
  (Prim.expect dt state.(0), Prim.expect D.Int64 state.(1))

let reduce (type d q) (apply : 'a. by:string -> 'a Value.prim -> 'a) ~by layout
    axes prog (rs : (d, q) Value.reductions) (loads : d Value.load array) :
    q option =
  let loop : type a.
      Nx_array.Layout.t ->
      P.t ->
      (d, a) Value.reduction ->
      d Value.load array ->
      a =
   fun layout prog r loads ->
    let v, () =
      apply ~by
        (Value.Reduce { layout; axes; prog; reductions = Value.[ r ]; loads })
    in
    v
  in
  let shape = L.shape layout in
  let one : type a. (d, a) Value.reduction -> a option = function
    | Monoid (Logsumexp, k, dt) ->
        let (Value.Any x) =
          body apply ~by layout prog k loads (accumulator (output prog k))
        in
        Some (cast_to apply ~by dt (logsumexp apply ~by shape axes x))
    | Moments (k, dt) ->
        let (Value.Any x) =
          body apply ~by layout prog k loads (accumulator (output prog k))
        in
        let mean, var = moments apply ~by shape axes x in
        Some (cast_to apply ~by dt mean, cast_to apply ~by dt var)
    | Arg (e, k, dt) ->
        let (Value.Any x) =
          body apply ~by layout prog k loads (output prog k)
        in
        let m, p = arg apply ~by shape axes e x in
        Some (cast_to apply ~by dt m, p)
    | Monoid _ as r -> reduction apply ~by ~loop layout prog loads r
  in
  let rec all : type q. (d, q) Value.reductions -> q option = function
    | [] -> Some ()
    | r :: rest -> (
        match one r with
        | None -> None
        | Some a -> Option.map (fun q -> (a, q)) (all rest))
  in
  all rs

let scan (type d a) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by layout
    axis prog (r : (d, a) Value.reduction) (loads : d Value.load array) :
    a option =
  let loop : type q.
      Nx_array.Layout.t ->
      P.t ->
      (d, q) Value.reduction ->
      d Value.load array ->
      q =
   fun layout prog reduction loads ->
    apply ~by (Value.Scan { layout; axis; prog; reduction; loads })
  in
  match r with
  | Moments (k, dt) ->
      let (Value.Any x) =
        body apply ~by layout prog k loads (accumulator (output prog k))
      in
      let mean, var = running_moments apply ~by (L.shape layout) axis x in
      Some (cast_to apply ~by dt mean, cast_to apply ~by dt var)
  | Monoid (Logsumexp, k, dt) ->
      let (Value.Any x) =
        body apply ~by layout prog k loads (accumulator (output prog k))
      in
      let y = running_logsumexp apply ~by (L.shape layout) axis x in
      Some (cast_to apply ~by dt y)
  | Arg (e, k, dt) ->
      let (Value.Any x) = body apply ~by layout prog k loads (output prog k) in
      let m, p = running_arg apply ~by (L.shape layout) axis e x in
      Some (cast_to apply ~by dt m, p)
  | Monoid _ -> reduction apply ~by ~loop layout prog loads r

(* Narrow dtypes *)

(* Whether [dt] is a base dtype, which every library's kernels compute: float32,
   float64, the 8- to 64-bit integers and bool. *)
let base (D.Any dt) =
  match dt with
  | D.Float32 | D.Float64 | D.Int64 | D.Uint64 | D.Int32 | D.Uint32 | D.Int16
  | D.Uint16 | D.Int8 | D.Uint8 | D.Bool ->
      true
  | D.Float16 | D.Bfloat16 | D.Float8_e4m3fn | D.Float8_e5m2 | D.Float4_e2m1fn
  | D.Int4 | D.Uint4 | D.Bit | D.Complex64 | D.Complex128 ->
      false

(* The bits [b] of an element of [from], as an element of [into], which holds
   its value exactly. *)
let widened_bits (D.Any from) b (D.Any into) =
  let read (type v s) (dt : (v, s) D.t) : v =
    Nx_array.get
      (Nx_array.v dt (Nx_array.Layout.contiguous [||]) (Rig.Buffer.of_string b))
      [||]
  in
  match (D.kind from, into) with
  | D.Float, D.Float32 -> P.bits D.Float32 (read from)
  | _ -> (
      match (from, into) with
      | D.Int4, D.Int8 -> P.bits D.Int8 (read D.Int4)
      | D.Uint4, D.Uint8 -> P.bits D.Uint8 (read D.Uint4)
      | D.Bit, D.Bool -> P.bits D.Bool (read D.Bit)
      | _ -> invalid_arg "Expand.widened_bits: no wider dtype")

(* float32 would not do for the floats of 8 or 16 bits: a store from it merges
   float8_e5m2's NaNs into one code and quiets the 16-bit floats' signaling
   NaNs. *)
let kept (D.Any dt as d) =
  if base d then d
  else
    match D.bits dt with
    | 8 -> D.Any D.Uint8
    | 16 -> D.Any D.Uint16
    | b when b < 8 -> accumulator d
    | _ -> d

(* [x] held in [dt]: its bits where [dt] has its width, else its value. *)
let held (type v s w r d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (dt : (w, r) D.t) (x : (v, s, d) Value.t) : (w, r, d) Value.t =
  match D.equal_witness (Prim.dtype x) dt with
  | Some Type.Equal -> x
  | None when D.bits (Prim.dtype x) = D.bits dt ->
      apply ~by (Value.Bitcast (dt, x))
  | None -> cast_to apply ~by dt x

(* A one-node map at dtypes some library declines, as the node at dtypes that
   hold its values exactly, its result rounded once to its dtype: their
   accumulators ({!accumulator}), or for a selection, a constant and a copy the
   dtypes {!kept} gives, its result read back as {!held} reads it. [None] for a
   node whose dtypes all stay, and for a cast and a bitcast, which no other
   dtype computes exactly. *)
let widened (type d r) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by layout
    (prog : P.t) (outs : (d, r) Value.outs) (loads : d Value.load array) :
    r option =
  let n = Iarray.length prog.nodes in
  let node = Iarray.get prog.nodes (n - 1) in
  let dts = Array.map Prim.load_dtype loads in
  let wide =
    match (node : P.node) with
    | Const _ | Op1 (Copy, _, _) | Op3 (Where, _, _, _) -> kept
    | _ -> accumulator
  in
  let result = P.dtype prog (n - 1) in
  let stays d = same (wide d) d in
  let all = Array.append dts [| result |] in
  let widens =
    match (node : P.node) with
    | Op1 ((Cast | Bitcast), _, _) | In _ | Coord _ -> false
    | Const _ | Op1 _ | Op2 _ | Op3 _ ->
        Array.for_all (fun d -> base (wide d) || stays d) all
        && not (Array.for_all stays all)
  in
  if not widens then None
  else
    let load l =
      let (Value.Any x) = Prim.load_any l in
      let (D.Any w) = wide (D.Any (Prim.dtype x)) in
      Value.Any (held apply ~by w x)
    in
    let operands = Array.map load loads in
    let ins =
      Iarray.of_array
        (Array.map (fun (Value.Any x) -> D.Any (Prim.dtype x)) operands)
    in
    let node : P.node =
      match node with
      | Const ((D.Any from as dt), b) ->
          let (D.Any into as w) = wide dt in
          if D.bits from = D.bits into then Const (w, b)
          else Const (w, widened_bits dt b w)
      | Op1 (k, dt, i) -> Op1 (k, wide dt, i)
      | (Coord _ | Op2 _ | Op3 _ | In _) as nd -> nd
    in
    let (D.Any w) = wide result in
    let v, () =
      apply ~by
        (Value.Map
           {
             layout;
             prog = P.of_node ~ins node;
             outs = Value.[ w ];
             loads = Array.map (fun (Value.Any x) -> Value.Plain x) operands;
           })
    in
    match outs with [ dt ] -> Some (held apply ~by dt v, ()) | _ -> None

(* The program of a region's flat positions in a C-contiguous value whose axes
   have [strides]: [Σ (start + i·step)·stride] at the region's index [i]. *)
let positions (rs : Nx_array.Move.range array) strides =
  let first = ref 0 in
  Array.iteri
    (fun d (g : Nx_array.Move.range) ->
      first := !first + (g.start * strides.(d)))
    rs;
  let b = Build.make () in
  let int64 n = const b D.Int64 (Int64.of_int n) in
  let sum = ref (int64 !first) in
  Array.iteri
    (fun d (g : Nx_array.Move.range) ->
      let c = push b (P.Coord d) in
      let k = int64 (g.step * strides.(d)) in
      let m = push b (P.Op2 (Binary Mul, c, k)) in
      sum := push b (P.Op2 (Binary Add, !sum, m)))
    rs;
  program b [| !sum |]

(* An assembly as a fill of the flat result, then per piece in order a scatter
   of its elements at their flat positions: O(n) per piece. *)
let assemble (type v s d) (apply : 'r. by:string -> 'r Value.prim -> 'r) ~by
    (dt : (v, s) Value.dtype) shape (fill : v)
    (pieces : (Nx_array.Move.range array * (v, s, d) Value.t) list) :
    (v, s, d) Value.t =
  let create : type w q.
      (w, q) Value.dtype -> int array -> P.t -> (w, q, d) Value.t =
   fun dt shape prog ->
    let v, () =
      apply ~by
        (Value.Map
           {
             layout = L.contiguous shape;
             prog;
             outs = Value.[ dt ];
             loads = [||];
           })
    in
    v
  in
  let n = Array.fold_left ( * ) 1 shape in
  let strides = Array.make (Array.length shape) 1 in
  for d = Array.length shape - 2 downto 0 do
    strides.(d) <- strides.(d + 1) * shape.(d + 1)
  done;
  let flat =
    create dt [| n |] (P.of_node ~ins:[||] (Const (D.Any dt, P.bits dt fill)))
  in
  let place acc (rs, x) =
    let s = Prim.shape x in
    let m = Array.fold_left ( * ) 1 s in
    if m = 0 then acc
    else
      let line : type w q. (w, q, d) Value.t -> (w, q, d) Value.t =
       fun v -> apply ~by (Value.Move (Reshape [| m |], v))
      in
      let targets = create D.Int64 s (positions rs strides) in
      apply ~by
        (Value.Scatter
           {
             combine = Set;
             unique = true;
             axis = 0;
             idx = line targets;
             updates = line x;
             into = acc;
           })
  in
  apply ~by
    (Value.Move (Reshape (Array.copy shape), List.fold_left place flat pieces))

(* Padded loads *)

(* [x] padded with [fill] as [p] says, before its windows: an assembly of the
   elements [p] keeps at their padded positions, [lo + j·(interior + 1)] along
   each axis, and [fill] elsewhere; a negative [lo] or [hi] keeps fewer. *)
let padded (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (x : (v, s, d) Value.t) (fill : v) (p : Nx_kernel.pad) : (v, s, d) Value.t =
  let s = Prim.shape x in
  let extent a =
    let d = s.(a) in
    p.lo.(a) + p.hi.(a) + d + (p.interior.(a) * max 0 (d - 1))
  in
  let shape = Array.init (Array.length s) extent in
  (* The elements of axis [a] the padding keeps, [first] and on: those whose
     padded position lies in [0, extent). *)
  (* CR: Use Int64 for kept geometry and a quotient-based ceiling.
     shape[1], lo=-2, hi=2, interior=max_int is valid and all fill,
     but gap wraps and creates start=-2. shape[2], lo=1-max_int,
     hi=0, interior=max_int-2 also overflows the ceiling.
     Narrow only retained ranges; use step 1 when count=1, since
     the true gap may exceed max_int. Cover both as tiny public maps. *)
  let kept a =
    let gap = p.interior.(a) + 1 and lo = p.lo.(a) in
    let first = if lo >= 0 then 0 else (-lo + gap - 1) / gap in
    let room = shape.(a) - 1 - lo in
    let last = if room < 0 then -1 else min (s.(a) - 1) (room / gap) in
    (first, max 0 (last - first + 1), gap)
  in
  let ks = Array.init (Array.length s) kept in
  let pieces =
    if Array.exists (fun (_, count, _) -> count = 0) ks then []
    else
      let region =
        Array.mapi
          (fun a (first, count, gap) ->
            { M.start = p.lo.(a) + (first * gap); count; step = gap })
          ks
      and slice =
        Array.map (fun (start, count, _) -> { M.start; count; step = 1 }) ks
      in
      [ (region, apply ~by (Value.Move (Slice slice, x))) ]
  in
  apply ~by (Value.Assemble { dtype = Prim.dtype x; shape; fill; pieces })

(* A load read plainly: a padded one as {!padded}'s assembly, viewed through its
   windows. *)
let unpadded (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (l : d Value.load) : d Value.load =
  match l with
  | Plain _ -> l
  | Padded { x; fill; pad } ->
      let y = padded apply ~by x fill pad in
      if pad.windows = [||] then Plain y
      else Plain (apply ~by (Value.Move (Window pad.windows, y)))

(* Folds *)

(* The most taps one map of a fold adds: with the running sum and the result, a
   program's most operands. *)
let run_taps = P.max_operands - 2

(* A fold as maps, each adding a run of at most {!run_taps} taps, in C order of
   the taps, to the running sum, the first from [+0]. Tap [t], [x] at [t] along
   the windows' axes, reaches the result at [i] along an axis from its element
   [w], where [w·step + t·dilation = lo + i·(interior + 1)], [step] [1] and [t]
   [0] off the windows: a padded load of the tap, interior-padded by [step - 1],
   from [t·dilation - lo], read every [interior + 1] by a window of one. One map
   per run, O(taps/14) passes over the result. *)
let fold (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by shape
    (p : Nx_kernel.pad) (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  let dt = Prim.dtype x and r = Array.length shape in
  let xs = Prim.shape x and nw = Array.length p.windows in
  let along = Array.make r None in
  Array.iteri (fun k (w : M.window) -> along.(w.axis) <- Some (k, w)) p.windows;
  let sizes = Array.map (fun (w : M.window) -> w.size) p.windows in
  let taps = Array.fold_left ( * ) 1 sizes in
  (* CR: Derive bounded source/result progressions for each tap. A valid
     shape [2], interior [L.max_numel], lo [-L.max_numel], hi [0] needs
     [0; x[1]], but this reverse pad exceeds max_numel; rank-32 inputs can
     also gain a 33rd axis. Solve gap*i - step*w = off-lo with gcd and
     checked Int64 arithmetic, clip before multiplying, then Slice the
     source and pad directly to the result without Window axes. Use step 1
     for singleton progressions and retain the ordered map runs. Cover
     both boundaries through a public Fold on a declining provider. *)
  (* The loop's shape: the result's, then an axis of one per axis read every
     [interior + 1] with [interior > 0], in axis order. *)
  let strided =
    List.filter (fun a -> p.interior.(a) > 0) (List.init r Fun.id)
  in
  let wide = Array.append shape (Array.make (List.length strided) 1) in
  (* Tap [t]'s padded load, [j] its indices along the windows' axes. *)
  let tap t : d Value.load =
    let j = Array.make nw 0 and t = ref t in
    for k = nw - 1 downto 0 do
      j.(k) <- !t mod sizes.(k);
      t := !t / sizes.(k)
    done;
    let ranges =
      Array.init (r + nw) (fun a ->
          if a < r then { M.start = 0; count = xs.(a); step = 1 }
          else { M.start = j.(a - r); count = 1; step = 1 })
    in
    let xt = apply ~by (Value.Move (Slice ranges, x)) in
    let xt = apply ~by (Value.Move (Reshape (Array.sub xs 0 r), xt)) in
    let lo = Array.make r 0 and hi = Array.make r 0 in
    let interior = Array.make r 0 in
    for a = 0 to r - 1 do
      let c = xs.(a) in
      let step, off =
        match along.(a) with
        | Some (k, w) -> (w.step, j.(k) * w.dilation)
        | None -> (1, 0)
      in
      let extent = ((shape.(a) - 1) * (p.interior.(a) + 1)) + 1 in
      interior.(a) <- step - 1;
      lo.(a) <- off - p.lo.(a);
      hi.(a) <- extent - lo.(a) - c - ((step - 1) * (c - 1))
    done;
    let windows =
      Array.of_list
        (List.map
           (fun a ->
             { M.axis = a; size = 1; step = p.interior.(a) + 1; dilation = 1 })
           strided)
    in
    Padded { x = xt; fill = D.zero dt; pad = { lo; hi; interior; windows } }
  in
  let zero b = const b dt (D.zero dt) in
  (* The sum so far plus taps [first] to [first + count - 1]. *)
  let add sum first count =
    let b = Build.make () in
    let total, loads =
      match sum with
      | None -> (ref (zero b), [])
      | Some s -> (ref (input b (D.Any dt)), [ Value.Plain s ])
    in
    let loads =
      Array.of_list (loads @ List.init count (fun i -> tap (first + i)))
    in
    for _ = 1 to count do
      total := push b (Op2 (Binary Add, !total, input b (D.Any dt)))
    done;
    map1 apply ~by wide (program b [| !total |]) dt loads
  in
  let rec runs sum first =
    if first >= taps then Option.get sum
    else
      let count = min run_taps (taps - first) in
      runs (Some (add sum first count)) (first + count)
  in
  if Array.exists (( = ) 0) shape || Array.exists (( = ) 0) xs then
    let b = Build.make () in
    map1 apply ~by shape (program b [| zero b |]) dt [||]
  else apply ~by (Value.Move (Reshape (Array.copy shape), runs None 0))

(* Fourier transforms

   A transform computes on pairs: a complex value of shape [s] as a float value
   of shape [2; s], its real parts, then its imaginary parts, so that each step
   is a map of floats, which every library computes. *)

let whole e = { M.start = 0; count = e; step = 1 }

(* The value of [shape] whose every element is [+0]. *)
let zeros (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (dt : (v, s) D.t) shape : (v, s, d) Value.t =
  map1 apply ~by shape
    (P.of_node ~ins:[||] (Const (D.Any dt, P.bits dt (D.zero dt))))
    dt [||]

(* The complex [x] as pairs of [pt], its parts' dtype. *)
let to_pairs (type v s w r d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (pt : (w, r) D.t) (x : (v, s, d) Value.t) : (w, r, d) Value.t =
  let z = apply ~by (Value.Bitcast (pt, x)) in
  let r = Prim.rank x in
  let order = Array.init (r + 1) (fun i -> if i = 0 then r else i - 1) in
  apply ~by (Value.Move (Permute order, z))

(* [pairs] as a complex value of [dt]. *)
let of_pairs (type v s w r d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (dt : (v, s) D.t) (pairs : (w, r, d) Value.t) : (v, s, d) Value.t =
  let r = Prim.rank pairs - 1 in
  let order = Array.init (r + 1) (fun i -> if i = r then 0 else i + 1) in
  let z = apply ~by (Value.Move (Permute order, pairs)) in
  apply ~by (Value.Bitcast (dt, z))

(* The map of pairs of shape [out], [2; …], whose result's parts are the nodes
   [f b parts], [parts] each load's real and imaginary nodes: each load is pairs
   whose shape broadcasts to [out]. *)
let pairs_map (type w r d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    out (loads : (w, r, d) Value.t array) f : (w, r, d) Value.t =
  let dt = Prim.dtype loads.(0) in
  let b = Build.make () in
  let parts =
    Array.map
      (fun _ ->
        let re = input b (D.Any dt) in
        (re, input b (D.Any dt)))
      loads
  in
  let re, im = f b parts in
  let first = push b (Coord 0) in
  let imaginary = push b (Op2 (Compare Equal, first, const b D.Int64 1L)) in
  let part c x =
    let s = Prim.shape x in
    let ranges =
      Array.mapi
        (fun a e ->
          if a = 0 then { M.start = c; count = 1; step = 1 } else whole e)
        s
    in
    let y = apply ~by (Value.Move (Slice ranges, x)) in
    Value.Plain (apply ~by (Value.Move (Broadcast (Array.copy out), y)))
  in
  let views =
    Array.concat
      (Array.to_list (Array.map (fun x -> [| part 0 x; part 1 x |]) loads))
  in
  map1 apply ~by out
    (program b [| push b (Op3 (Where, imaginary, im, re)) |])
    dt views

(* The nodes of [x·y] for nodes of complex numbers' parts. *)
let cmul b (xr, xi) (yr, yi) =
  let op k x y = push b (P.Op2 (Binary k, x, y)) in
  (op Sub (op Mul xr yr) (op Mul xi yi), op Add (op Mul xr yi) (op Mul xi yr))

(* The nodes of [e^(i·a)] at the node [a] of [dt]. *)
let unit_at b dt a =
  ( push b (P.Op1 (Unary Cos, D.Any dt, a)),
    push b (P.Op1 (Unary Sin, D.Any dt, a)) )

(* The transform of exponent sign [sign] of [pairs], of shape [2; b; n] with [n]
   a power of two, along its last axis, by Stockham's autosort: at each of [log₂
   n] levels, a map gives the sums and the twiddled differences of the two
   halves of each of [n / 2^l]-point transforms, interleaved, so the result
   needs no reordering. *)
let stockham (type w r d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    sign (pairs : (w, r, d) Value.t) : (w, r, d) Value.t =
  let dt = Prim.dtype pairs and s0 = Prim.shape pairs in
  let batch = s0.(1) and n = s0.(2) in
  let move mv x = apply ~by (Value.Move (mv, x)) in
  (* [x] holds [nl]-point transforms at stride [s]: [2; b; nl; s]. *)
  let rec level x nl s =
    if nl <= 1 then x
    else
      let m = nl / 2 in
      let out = [| 2; batch; m; 2; s |] in
      let v = move (Reshape [| 2; batch; nl; s |]) x in
      let part first c =
        let ranges =
          [|
            { M.start = c; count = 1; step = 1 };
            whole batch;
            { M.start = first; count = m; step = 1 };
            whole s;
          |]
        in
        let y =
          move (Reshape [| 1; batch; m; 1; s |]) (move (Slice ranges) v)
        in
        Value.Plain (move (Broadcast out) y)
      in
      let b = Build.make () in
      let ins = Array.init 4 (fun _ -> input b (D.Any dt)) in
      let op k x y = push b (P.Op2 (Binary k, x, y)) in
      let sr = op Add ins.(0) ins.(2) and si = op Add ins.(1) ins.(3) in
      let d = (op Sub ins.(0) ins.(2), op Sub ins.(1) ins.(3)) in
      let p = push b (Op1 (Cast, D.Any dt, push b (Coord 2))) in
      let turn =
        const b dt (D.of_float dt (sign *. 2. *. Float.pi /. Float.of_int nl))
      in
      let tr, ti = cmul b d (unit_at b dt (op Mul p turn)) in
      let is axis =
        push b (Op2 (Compare Equal, push b (Coord axis), const b D.Int64 1L))
      in
      let im = is 0 in
      let sum = push b (Op3 (Where, im, si, sr))
      and diff = push b (Op3 (Where, im, ti, tr)) in
      let o = push b (Op3 (Where, is 3, diff, sum)) in
      let y =
        map1 apply ~by out
          (program b [| o |])
          dt
          [| part 0 0; part 0 1; part m 0; part m 1 |]
      in
      level y m (2 * s)
  in
  move (Reshape [| 2; batch; n |]) (level pairs n 1)

(* The transform of exponent sign [sign] of [pairs], of shape [2; b; n], along
   its last axis, by Bluestein's chirp: [jk = (j² + k² - (k - j)²) / 2] makes it
   [c_k Σ_j (x_j c_j) conj c_(k-j)], [c_j = e^(sign·iπ j²/n)], a convolution,
   computed by {!stockham} at the power of two [M ≥ 2n - 1]. [j²] is taken
   modulo [2n], exact in int64, so the chirp's angle stays below [2π]. *)
let bluestein (type w r d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    sign (pairs : (w, r, d) Value.t) : (w, r, d) Value.t =
  let dt = Prim.dtype pairs and s0 = Prim.shape pairs in
  let batch = s0.(1) and n = s0.(2) in
  let m =
    let rec up p = if p >= (2 * n) - 1 then p else up (2 * p) in
    up 1
  in
  let i64 b v = const b D.Int64 (Int64.of_int v) in
  let f b v = const b dt (D.of_float dt v) in
  (* [e^(sg·iπ (t² mod 2n)/n)] at the int64 node [t]. *)
  let chirp b sg t =
    let sq = push b (P.Op2 (Binary Mul, t, t)) in
    let q = push b (P.Op2 (Binary Mod, sq, i64 b (2 * n))) in
    let q = push b (P.Op1 (Cast, D.Any dt, q)) in
    unit_at b dt
      (push b (P.Op2 (Binary Mul, q, f b (sg *. Float.pi /. Float.of_int n))))
  in
  let padded =
    apply ~by
      (Value.Assemble
         {
           dtype = dt;
           shape = [| 2; batch; m |];
           fill = D.zero dt;
           pieces = [ ([| whole 2; whole batch; whole n |], pairs) ];
         })
  in
  let a =
    pairs_map apply ~by [| 2; batch; m |] [| padded |] (fun b parts ->
        cmul b parts.(0) (chirp b sign (push b (Coord 2))))
  in
  (* [conj c_j] at [t = j] and [t = M - j], [0] elsewhere. *)
  let kernel =
    let b = Build.make () in
    let t = push b (Coord 2) in
    let below = push b (P.Op2 (Compare Less, t, i64 b n)) in
    let mirror = push b (P.Op2 (Binary Sub, i64 b m, t)) in
    let above = push b (P.Op2 (Compare Less, mirror, i64 b n)) in
    let j = push b (P.Op3 (Where, below, t, mirror)) in
    let inside = push b (P.Op2 (Binary Or, below, above)) in
    let cr, ci = chirp b (-.sign) j in
    let zero = f b 0. in
    let im = push b (P.Op2 (Compare Equal, push b (Coord 0), i64 b 1)) in
    let part = push b (P.Op3 (Where, im, ci, cr)) in
    map1 apply ~by [| 2; 1; m |]
      (program b [| push b (P.Op3 (Where, inside, part, zero)) |])
      dt [||]
  in
  let fa = stockham apply ~by (-1.) a
  and fk = stockham apply ~by (-1.) kernel in
  let product =
    pairs_map apply ~by [| 2; batch; m |] [| fa; fk |] (fun b parts ->
        cmul b parts.(0) parts.(1))
  in
  let conv = stockham apply ~by 1. product in
  let conv =
    apply ~by (Value.Move (Slice [| whole 2; whole batch; whole n |], conv))
  in
  pairs_map apply ~by [| 2; batch; n |] [| conv |] (fun b parts ->
      let re, im = cmul b parts.(0) (chirp b sign (push b (Coord 2))) in
      let scale = f b (1. /. Float.of_int m) in
      ( push b (P.Op2 (Binary Mul, re, scale)),
        push b (P.Op2 (Binary Mul, im, scale)) ))

(* The transform of exponent sign [sign] of [pairs] along axis [axis] of the
   complex value they hold: its lines laid as [2; b; n], {!stockham} for a power
   of two [n], {!bluestein} otherwise. *)
let along (type w r d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by sign
    axis (pairs : (w, r, d) Value.t) : (w, r, d) Value.t =
  let s = Prim.shape pairs in
  let r = Array.length s and a = axis + 1 in
  let n = s.(a) in
  if n <= 1 then pairs
  else
    let move mv x = apply ~by (Value.Move (mv, x)) in
    let order =
      Array.of_list (List.filter (( <> ) a) (List.init r Fun.id) @ [ a ])
    in
    let moved = move (Permute order) pairs in
    let ms = Prim.shape moved in
    let batch = Array.fold_left ( * ) 1 (Array.sub ms 1 (r - 2)) in
    let lines = move (Reshape [| 2; batch; n |]) moved in
    let y =
      if n land (n - 1) = 0 then stockham apply ~by sign lines
      else bluestein apply ~by sign lines
    in
    let back = Array.make r 0 in
    Array.iteri (fun i o -> back.(o) <- i) order;
    move (Permute back) (move (Reshape ms) y)

let sign_of : Nx_kernel.direction -> float = function
  | Forward -> -1.
  | Inverse -> 1.

(* A transform as {!along} each of its axes, on pairs. A real operand's pairs
   have [+0] imaginary parts; [R2c] keeps bins [0] to [n/2] of the last axis;
   [C2r] completes the last axis's [n] bins from their conjugates, as
   {!Nx_kernel.S.c2r} states, and keeps the real parts. Where the operand or
   the result has no element, the result is [+0]: the empty sum. *)
let fft : type d r.
    ('q. by:string -> 'q Value.prim -> 'q) -> by:string -> (d, r) Value.fft -> r
    =
 fun apply ~by f ->
  let numel s = Array.fold_left ( * ) 1 s in
  let set s a e = Array.mapi (fun i x -> if i = a then e else x) s in
  let slice x a range =
    let s = Prim.shape x in
    apply ~by
      (Value.Move
         (Slice (Array.mapi (fun i e -> if i = a then range else whole e) s), x))
  in
  match f with
  | C2c { direction; axes; x } ->
      let dt = Prim.dtype x and s = Prim.shape x in
      if numel s = 0 then zeros apply ~by dt s
      else
        let (D.Any pt) = Option.get (parts (D.Any dt)) in
        let p = to_pairs apply ~by pt x in
        let p =
          Array.fold_left
            (fun p a -> along apply ~by (sign_of direction) a p)
            p axes
        in
        of_pairs apply ~by dt p
  | R2c { dtype; axes; x } ->
      let s = Prim.shape x and l = axes.(Array.length axes - 1) in
      let out = set s l ((s.(l) / 2) + 1) in
      if numel s = 0 then zeros apply ~by dtype out
      else
        let pt = Prim.dtype x in
        let p =
          apply ~by
            (Value.Assemble
               {
                 dtype = pt;
                 shape = Array.append [| 2 |] s;
                 fill = D.zero pt;
                 pieces =
                   [
                     ( Array.append
                         [| { M.start = 0; count = 1; step = 1 } |]
                         (Array.map whole s),
                       apply ~by
                         (Value.Move (Reshape (Array.append [| 1 |] s), x)) );
                   ];
               })
        in
        let p = Array.fold_left (fun p a -> along apply ~by (-1.) a p) p axes in
        of_pairs apply ~by dtype (slice p (l + 1) (whole out.(l)))
  | C2r { dtype; n; axes; x } ->
      let s = Prim.shape x and k = Array.length axes - 1 in
      let l = axes.(k) in
      let out = set s l n in
      if numel out = 0 then zeros apply ~by dtype out
      else
        let (D.Any pt) = Option.get (parts (D.Any (Prim.dtype x))) in
        let p = to_pairs apply ~by pt x in
        let p =
          Array.fold_left
            (fun p a -> along apply ~by 1. a p)
            p (Array.sub axes 0 k)
        in
        (* Bins [h] to [n - 1] are [conj X[n - k]]: bins [n - h] down to [1]. *)
        let h = s.(l) and mirrored = n - s.(l) in
        let full =
          if mirrored = 0 then p
          else
            let rest =
              slice p (l + 1) { M.start = n - h; count = mirrored; step = -1 }
            in
            let conj =
              pairs_map apply ~by (Prim.shape rest) [| rest |] (fun b parts ->
                  let re, im = parts.(0) in
                  (re, push b (P.Op1 (Unary Neg, D.Any pt, im))))
            in
            let region from count =
              Array.mapi
                (fun i e ->
                  if i = l + 1 then { M.start = from; count; step = 1 }
                  else whole e)
                (Array.append [| 2 |] out)
            in
            apply ~by
              (Value.Assemble
                 {
                   dtype = pt;
                   shape = Array.append [| 2 |] out;
                   fill = D.zero pt;
                   pieces = [ (region 0 h, p); (region h mirrored, conj) ];
                 })
        in
        let p = along apply ~by 1. l full in
        let re =
          apply ~by
            (Value.Move
               (Reshape out, slice p 0 { M.start = 0; count = 1; step = 1 }))
        in
        cast_to apply ~by dtype (apply ~by (Value.Copy re))

(* Contractions *)

(* [x], whose axis [i] is [full]'s axis [at.(i)], at the shape [full]: its axes
   put in that order, a unit axis where it has none, and broadcast. *)
let spread (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (x : (v, s, d) Value.t) at full : (v, s, d) Value.t =
  let r = Array.length at in
  let order = Array.init r Fun.id in
  Array.sort (fun i j -> compare at.(i) at.(j)) order;
  let move mv x = apply ~by (Value.Move (mv, x)) in
  let x = if order = Array.init r Fun.id then x else move (M.Permute order) x in
  let units = Array.mapi (fun k e -> if Array.mem k at then e else 1) full in
  let x = if Prim.has_shape x units then x else move (M.Reshape units) x in
  if units = full then x else move (M.Broadcast full) x

(* The contraction of [a] and [b] by the pairs [batch] and [contracting] from
   [init], computed in [acc] into [out], as other operations. A float result
   narrower than its accumulator is the contraction into the accumulator,
   rounded once. Any other is the products in the accumulator at every index of
   the result's axes then the contracted ones, a sum over the contracted ones,
   [init] added, and one rounding to [out]. *)
let contract (type v s a b c e d) (apply : 'q. by:string -> 'q Value.prim -> 'q)
    ~by ~batch ~contracting ~acc:(D.Any acc) (out : (v, s) D.t)
    (a : (a, b, d) Value.t) (b : (c, e, d) Value.t)
    (init : (v, s, d) Value.t option) : (v, s, d) Value.t =
  if D.is D.Float acc && not (D.equal acc out) then
    let init = Option.map (cast_to apply ~by acc) init in
    let y =
      apply ~by
        (Value.Contract
           { batch; contracting; acc = D.Any acc; out = acc; a; b; init })
    in
    cast_to apply ~by out y
  else
    let sa = Prim.shape a and sb = Prim.shape b in
    let free s side =
      List.filter
        (fun i ->
          not
            (Array.exists (fun p -> side p = i) batch
            || Array.exists (fun p -> side p = i) contracting))
        (List.init (Array.length s) Fun.id)
    in
    let fa = free sa fst and fb = free sb snd in
    let nb = Array.length batch and nfa = List.length fa in
    let rshape =
      Array.concat
        [
          Array.map (fun (i, _) -> sa.(i)) batch;
          Array.of_list (List.map (fun i -> sa.(i)) fa);
          Array.of_list (List.map (fun i -> sb.(i)) fb);
        ]
    in
    let nr = Array.length rshape in
    let full =
      Array.append rshape (Array.map (fun (i, _) -> sa.(i)) contracting)
    in
    let positions s side frees offset =
      let at = Array.make (Array.length s) 0 in
      Array.iteri (fun k p -> at.(side p) <- k) batch;
      List.iteri (fun j i -> at.(i) <- nb + offset + j) frees;
      Array.iteri (fun c p -> at.(side p) <- nr + c) contracting;
      at
    in
    let va = spread apply ~by a (positions sa fst fa 0) full in
    let vb = spread apply ~by b (positions sb snd fb nfa) full in
    let prog =
      P.v
        ~ins:[| D.Any (Prim.dtype a); D.Any (Prim.dtype b) |]
        [|
          In 0;
          In 1;
          Op1 (Cast, D.Any acc, 0);
          Op1 (Cast, D.Any acc, 1);
          Op2 (Binary Mul, 2, 3);
        |]
        ~outs:[| 4 |]
    in
    let layout = Nx_array.Layout.contiguous full in
    let loads = [| Value.Plain va; Value.Plain vb |] in
    let sum =
      if Array.length contracting = 0 then
        fst
          (apply ~by (Value.Map { layout; prog; outs = Value.[ acc ]; loads }))
      else
        fst
          (apply ~by
             (Value.Reduce
                {
                  layout;
                  axes = Array.init (Array.length contracting) (fun c -> nr + c);
                  prog;
                  reductions = Value.[ Monoid (Sum, 0, acc) ];
                  loads;
                }))
    in
    let sum =
      match init with
      | None -> sum
      | Some i ->
          fst
            (apply ~by
               (Prim.op2 ~by (Binary Add) acc sum (cast_to apply ~by acc i)))
    in
    cast_to apply ~by out sum

(* Gathers and scatters *)

(* The accumulator of [x]'s dtype where it is sub-byte, which kernels may
   decline to gather or scatter: [None] for a dtype of a byte or more. *)
let sub_byte (type v s d) (x : (v, s, d) Value.t) =
  let dt = Prim.dtype x in
  if D.bits dt >= 8 then None else Some (accumulator (D.Any dt))

(* A sub-byte gather at its accumulator, cast back once. *)
let gather (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by axis
    idx (x : (v, s, d) Value.t) : (v, s, d) Value.t option =
  match sub_byte x with
  | None -> None
  | Some (D.Any w) ->
      let x' = cast_to apply ~by w x in
      let y = apply ~by (Value.Gather { axis; idx; x = x' }) in
      Some (cast_to apply ~by (Prim.dtype x) y)

(* A scatter whose targets may repeat as one unique scatter per position along
   [axis], in order: a position's updates differ off [axis], so their targets
   do, and updates to one target land in C order. A sum associates left to
   right. *)
let repeated (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    combine axis idx (updates : (v, s, d) Value.t) (into : (v, s, d) Value.t) :
    (v, s, d) Value.t =
  let s = Prim.shape updates in
  let at : type w q. int -> (w, q, d) Value.t -> (w, q, d) Value.t =
   fun i x ->
    let w =
      Array.mapi
        (fun a n ->
          if a = axis then { M.start = i; count = 1; step = 1 }
          else { M.start = 0; count = n; step = 1 })
        s
    in
    apply ~by (Value.Move (Slice w, x))
  in
  let acc = ref into in
  for i = 0 to s.(axis) - 1 do
    acc :=
      apply ~by
        (Value.Scatter
           {
             combine;
             unique = true;
             axis;
             idx = at i idx;
             updates = at i updates;
             into = !acc;
           })
  done;
  !acc

(* A narrow float's [Add] of repeated targets as {!repeated} in its accumulator,
   rounded once; a target no update reaches, found by a [Set] of [true] at every
   target, keeps [into]'s bits. *)
let rounded_once (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (w : ('w, 'r) D.t) axis idx (updates : (v, s, d) Value.t)
    (into : (v, s, d) Value.t) : (v, s, d) Value.t =
  let dt = Prim.dtype into in
  let wide x = cast_to apply ~by w x in
  let sum =
    repeated apply ~by Nx_kernel.Add axis idx (wide updates) (wide into)
  in
  let fill b shape : (bool, D.bool_elt, d) Value.t =
    let prog = P.of_node ~ins:[||] (Const (D.Any D.Bool, P.bits D.Bool b)) in
    let v, () =
      apply ~by
        (Value.Map
           {
             layout = L.contiguous shape;
             prog;
             outs = Value.[ D.Bool ];
             loads = [||];
           })
    in
    v
  in
  let touched =
    repeated apply ~by Nx_kernel.Set axis idx
      (fill true (Prim.shape updates))
      (fill false (Prim.shape into))
  in
  let v, () =
    apply ~by (Prim.op3 ~by Where touched (cast_to apply ~by dt sum) into)
  in
  v

(* A sub-byte scatter at its accumulator, cast back once. An integer's [Add]
   wraps there to the same bits as at its own dtype. A scatter whose targets may
   repeat at a dtype of a byte or more is {!repeated}, a narrow float's sum
   {!rounded_once}. *)
let scatter (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    combine ~unique axis idx (updates : (v, s, d) Value.t)
    (into : (v, s, d) Value.t) : (v, s, d) Value.t option =
  match sub_byte into with
  | None when unique -> None
  | None -> (
      match (combine, accumulator (D.Any (Prim.dtype into))) with
      | Nx_kernel.Add, D.Any w
        when not (same (D.Any w) (D.Any (Prim.dtype into))) ->
          Some (rounded_once apply ~by w axis idx updates into)
      | _ -> Some (repeated apply ~by combine axis idx updates into))
  | Some (D.Any w) ->
      let wide x = cast_to apply ~by w x in
      let y =
        apply ~by
          (Value.Scatter
             {
               combine;
               unique;
               axis;
               idx;
               updates = wide updates;
               into = wide into;
             })
      in
      Some (cast_to apply ~by (Prim.dtype into) y)

(* Sorts *)

(* The unsigned integer dtype of [w] bits, 8 to 64, and its element of an
   int64's low bits. *)
type unsigned = U : ('w, 'r) D.t * (int64 -> 'w) -> unsigned

let unsigned w =
  match w with
  | 8 -> U (D.Uint8, Int64.to_int)
  | 16 -> U (D.Uint16, Int64.to_int)
  | 32 -> U (D.Uint32, Int64.to_int32)
  | _ -> U (D.Uint64, Fun.id)

let uconst b (U (ut, of_int64)) n = const b ut (of_int64 n)

let ones w = if w = 64 then -1L else Int64.(sub (shift_left 1L w) 1L)

(* The nodes of the order key of the float of [w] bits whose bits are node [u],
   a NaN where node [nan] is: its bits with the sign set where it is clear, all
   flipped where it is set, so that [-0] keys just below [+0]; all ones for
   every NaN. *)
let float_key b w u nan =
  let ut = unsigned w in
  let sign = uconst b ut (Int64.shift_left 1L (w - 1))
  and all = uconst b ut (ones w) in
  let negative = push b (Op2 (Compare Less_equal, sign, u)) in
  let flipped = push b (Op2 (Binary Xor, u, all)) in
  let raised = push b (Op2 (Binary Or, u, sign)) in
  let k = push b (Op3 (Where, negative, flipped, raised)) in
  push b (Op3 (Where, nan, all, k))

(* The key of [w] bits at node [k] as int64 nodes below [2^32], the high half
   first. *)
let halves b w k =
  let wide n = push b (Op1 (Cast, D.Any D.Int64, n)) in
  if w <= 32 then [ wide k ]
  else
    let ut = unsigned w in
    let hi = push b (Op2 (Binary Idiv, k, uconst b ut 0x1_0000_0000L)) in
    let lo = push b (Op2 (Binary And, k, uconst b ut 0xffff_ffffL)) in
    [ wide hi; wide lo ]

(* [x]'s order keys, most significant first: int64 values below [2^32] whose
   lexicographic order is {!Nx_kernel.S.sort}'s order of [x]'s elements,
   every NaN, and every complex number with a NaN part, one key above [+∞]. A
   signed integer keys as its bits with the sign flipped, a float as
   {!float_key} gives, a complex number as its real part's key, then its
   imaginary part's. *)
let sort_keys (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (x : (v, s, d) Value.t) : (int64, D.int64_elt, d) Value.t list =
  let dt = Prim.dtype x and shape = Prim.shape x in
  let w = D.bits dt in
  (* The keys [build b ins] gives over [loads], [ins] their nodes, one map
     each. *)
  let keys loads build =
    let built () =
      let b = Build.make () in
      let input (Value.Any y) = input b (D.Any (Prim.dtype y)) in
      (b, build b (Array.map input loads))
    in
    let loads = Array.map (fun (Value.Any y) -> Value.Plain y) loads in
    List.init
      (List.length (snd (built ())))
      (fun j ->
        let b, outs = built () in
        map1 apply ~by shape (program b [| List.nth outs j |]) D.Int64 loads)
  in
  match D.kind dt with
  | D.Boolean ->
      keys [| Value.Any x |] (fun b ins ->
          [ push b (Op1 (Cast, D.Any D.Int64, ins.(0))) ])
  | D.Unsigned -> keys [| Value.Any x |] (fun b ins -> halves b w ins.(0))
  | D.Signed ->
      keys [| Value.Any x |] (fun b ins ->
          let (U (ut, _) as u) = unsigned w in
          let bits = push b (Op1 (Bitcast, D.Any ut, ins.(0))) in
          let sign = uconst b u (Int64.shift_left 1L (w - 1)) in
          halves b w (push b (Op2 (Binary Xor, bits, sign))))
  | D.Float ->
      keys [| Value.Any x |] (fun b ins ->
          let (U (ut, _)) = unsigned w in
          let v = ins.(0) in
          let nan = push b (Op2 (Compare Not_equal, v, v)) in
          let bits = push b (Op1 (Bitcast, D.Any ut, v)) in
          halves b w (float_key b w bits nan))
  | D.Complex ->
      let w = w / 2 in
      let (D.Any pt) = Option.get (parts (D.Any dt)) in
      let pairs = apply ~by (Value.Bitcast (pt, x)) in
      let part j = Value.Any (half apply ~by shape pairs j) in
      keys
        [| part 0; part 1 |]
        (fun b ins ->
          let (U (ut, _)) = unsigned w in
          let re = ins.(0) and im = ins.(1) in
          let isnan v = push b (Op2 (Compare Not_equal, v, v)) in
          let nan = push b (Op2 (Binary Or, isnan re, isnan im)) in
          let key v =
            halves b w (float_key b w (push b (Op1 (Bitcast, D.Any ut, v))) nan)
          in
          key re @ key im)

let bit_length n =
  let rec go n b = if n = 0 then b else go (n lsr 1) (b + 1) in
  go n 0

(* The positions along [axis] that sort the keys [k], below [2^32], stably: a
   bitonic network over each key packed above its position, which makes the
   packed integers distinct, so the network's order is the stable one. The axis
   is padded to a power of two [N] with integers that sort last. Each of the
   network's [log N (log N + 1) / 2] steps is a gather of each element's
   partner, the element whose position differs in one bit, and a map keeping the
   lesser or the greater of the two: O(N log² N). A descending sort complements
   the positions, so that equal keys keep increasing positions. *)
let bitonic (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by axis
    descending (k : (int64, D.int64_elt, d) Value.t) :
    (int64, D.int64_elt, d) Value.t =
  let shape = Prim.shape k in
  let n = shape.(axis) in
  let low = bit_length (n - 1) in
  let mask = Int64.(sub (shift_left 1L low) 1L) in
  let int64 b v = const b D.Int64 v in
  let coord b = push b (Coord axis) in
  (* A position, or its complement in a descending sort. *)
  let tie b p =
    if descending then push b (Op2 (Binary Sub, int64 b mask, p)) else p
  in
  let packed =
    let b = Build.make () in
    let key = input b (D.Any D.Int64) in
    let shift = int64 b (Int64.shift_left 1L low) in
    let high = push b (Op2 (Binary Mul, key, shift)) in
    map1 apply ~by shape
      (program b [| push b (Op2 (Binary Add, high, tie b (coord b))) |])
      D.Int64 [| Value.Plain k |]
  in
  let stages = bit_length (n - 1) in
  let wide = Array.copy shape in
  wide.(axis) <- 1 lsl stages;
  let whole s = Array.map (fun e -> { M.start = 0; count = e; step = 1 }) s in
  let x =
    if wide.(axis) = n then packed
    else
      apply ~by
        (Value.Assemble
           {
             dtype = D.Int64;
             shape = wide;
             fill = (if descending then -1L else Int64.max_int);
             pieces = [ (whole shape, packed) ];
           })
  in
  (* Step [j] of the stage of blocks of [block]: the element at [i] keeps the
     lesser of it and its partner at [i lxor j] where [i] is the first of the
     two in a block sorting up, or the second in one sorting down. *)
  let step x block j =
    let partner =
      let b = Build.make () in
      map1 apply ~by wide
        (program b
           [| push b (Op2 (Binary Xor, coord b, int64 b (Int64.of_int j))) |])
        D.Int64 [||]
    in
    let y = apply ~by (Value.Gather { axis; idx = partner; x }) in
    let b = Build.make () in
    let a = input b (D.Any D.Int64) in
    let c = input b (D.Any D.Int64) in
    let clear bit =
      let m = push b (Op2 (Binary And, coord b, int64 b (Int64.of_int bit))) in
      push b (Op2 (Compare Equal, m, int64 b 0L))
    in
    let up = clear block and first = clear j in
    let lesser =
      push b
        (Op2 (Compare (if descending then Not_equal else Equal), up, first))
    in
    let lo = push b (Op2 (Binary Minimum, a, c))
    and hi = push b (Op2 (Binary Maximum, a, c)) in
    map1 apply ~by wide
      (program b [| push b (Op3 (Where, lesser, lo, hi)) |])
      D.Int64
      [| Value.Plain x; Value.Plain y |]
  in
  let x = ref x in
  for s = 1 to stages do
    let block = 1 lsl s in
    let j = ref (block / 2) in
    while !j >= 1 do
      x := step !x block !j;
      j := !j / 2
    done
  done;
  let sorted =
    if wide.(axis) = n then !x
    else apply ~by (Value.Move (Slice (whole shape), !x))
  in
  let b = Build.make () in
  let s = input b (D.Any D.Int64) in
  let p = push b (Op2 (Binary And, s, int64 b mask)) in
  map1 apply ~by shape
    (program b [| tie b p |])
    D.Int64 [| Value.Plain sorted |]

(* A sort as positions from its keys, the least significant first, each pass a
   stable {!bitonic} sort of one key in the order of the passes before it, then
   a gather of the elements: O(K N log² N) for K keys, K 1 to 4. A sub-byte sort
   runs at its accumulator, which holds each code and orders them alike, its
   values cast back once; no sub-byte dtype has a NaN code. *)
let sort (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by axis
    descending k (x : (v, s, d) Value.t) :
    ((v, s, d) Value.t * (int64, D.int64_elt, d) Value.t) option =
  match sub_byte x with
  | Some (D.Any w) ->
      let x' = cast_to apply ~by w x in
      let v, p = apply ~by (Value.Sort { axis; descending; k; x = x' }) in
      Some (cast_to apply ~by (Prim.dtype x) v, p)
  | None ->
      let shape = Prim.shape x in
      let positions =
        if shape.(axis) <= 1 then
          map1 apply ~by shape
            (P.of_node ~ins:[||] (Const (D.Any D.Int64, P.bits D.Int64 0L)))
            D.Int64 [||]
        else
          let gather idx y = apply ~by (Value.Gather { axis; idx; x = y }) in
          List.fold_left
            (fun order key ->
              let key =
                Option.fold ~none:key ~some:(fun o -> gather o key) order
              in
              let p = bitonic apply ~by axis descending key in
              Some (Option.fold ~none:p ~some:(fun o -> gather p o) order))
            None
            (List.rev (sort_keys apply ~by x))
          |> Option.get
      in
      (* CR: Select the first k positions, make them contiguous, then gather
         values. Slicing both outputs here gives strides [4;1] for input
         [2;4], axis 1, k=2; Prim.results promises [2;1]. Copy positions
         only when their prefix is not contiguous at offset 0. Pin both
         result forms on a declining device; gather only the kept values. *)
      let values = apply ~by (Value.Gather { axis; idx = positions; x }) in
      let first y =
        match k with
        | None -> y
        | Some k ->
            let rs =
              Array.map
                (fun e -> { M.start = 0; count = e; step = 1 })
                (Prim.shape y)
            in
            rs.(axis) <- { M.start = 0; count = k; step = 1 };
            apply ~by (Value.Move (Slice rs, y))
      in
      Some (first values, first positions)

let run : type r.
    ('q. by:string -> 'q Value.prim -> 'q) ->
    by:string ->
    r Value.prim ->
    r option =
 fun apply ~by op ->
  match op with
  | Value.Map { layout; prog; outs; loads }
    when Array.exists Prim.is_padded loads ->
      let loads = Array.map (unpadded apply ~by) loads in
      Some (apply ~by (Value.Map { layout; prog; outs; loads }))
  | Value.Map { layout; prog; outs; loads } when Option.is_none (single prog) ->
      let loads = Array.map Prim.load_any loads in
      let values = Array.make (Iarray.length prog.nodes) None in
      let value j = Option.get values.(j) in
      Array.iteri
        (fun i _ ->
          values.(i) <- Some (node_map apply ~by layout prog loads value i))
        values;
      Some (results outs (List.map value (Iarray.to_list prog.outs)))
  | Value.Reduce { layout; axes; prog; reductions; loads }
    when Array.exists Prim.is_padded loads || not (core_reduce prog reductions)
    ->
      reduce apply ~by layout axes prog reductions loads
  | Value.Scan { layout; axis; prog; reduction = r; loads }
    when Array.exists Prim.is_padded loads || not (core_scan prog r) ->
      scan apply ~by layout axis prog r loads
  | Value.Contract { batch; contracting; acc; out; a; b; init } ->
      Some (contract apply ~by ~batch ~contracting ~acc out a b init)
  | Value.Map { layout; prog; outs; loads } ->
      widened apply ~by layout prog outs loads
  | Value.Assemble { dtype; shape; fill; pieces } ->
      Some (assemble apply ~by dtype shape fill pieces)
  | Value.Gather { axis; idx; x } -> gather apply ~by axis idx x
  | Value.Scatter { combine; unique; axis; idx; updates; into } ->
      scatter apply ~by combine ~unique axis idx updates into
  | Value.Sort { axis; descending; k; x } -> sort apply ~by axis descending k x
  | Value.Fold { shape; pad; x } -> Some (fold apply ~by shape pad x)
  | Value.Fft f -> Some (fft apply ~by f)
  | Value.Reduce _ | Value.Scan _ | Value.Linalg _ | Value.Copy _ | Value.Move _
  | Value.Bitcast _ | Value.Place _ | Value.Check _ ->
      None
