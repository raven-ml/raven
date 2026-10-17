(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Nx_array.Dtype
module L = Nx_array.Layout
module P = Nx_kernel.Prog
module M = Nx_array.Move
module E = P.E

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

(* A value of floats: Logsumexp and Moments reduce floats alone
   ({!Nx_kernel.accepts}). *)
type 'd floats = Floats : (float, 's, 'd) Value.t -> 'd floats

let floats ~by (Value.Any x) =
  let dt = Prim.dtype x in
  match D.kind dt with
  | D.Float -> Floats x
  | D.Complex | D.Signed | D.Unsigned | D.Boolean ->
      invalid_arg (Printf.sprintf "%s: %s is not a float dtype" by (D.name dt))

(* The output a reduction reads and its result's dtype. *)
let target : type d a. (d, a) Value.reduction -> int * D.any = function
  | Monoid (_, k, dt) -> (k, D.Any dt)
  | Moments (k, dt) -> (k, D.Any dt)
  | Arg (_, k, dt) -> (k, D.Any dt)

(* The float format of a complex dtype's parts. *)
type part = Part : (float, 's) D.t -> part

let parts (D.Any dt) =
  match dt with
  | D.Complex64 -> Some (Part D.Float32)
  | D.Complex128 -> Some (Part D.Float64)
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
        | Some (Part pt) when complex_sum r ->
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

(* [x] where it is a NaN, else [y]: [Where (x ≠ x) x y]. *)
let nan_or x y = E.(where (x <> x) x y)

(* The least integer whose [exp] is a normal float of [dt], float32 or float64:
   nx.cpu takes an [exp] below it as [0], so that no lane makes a subnormal. *)
let exp_cutoff (type s) (dt : (float, s) D.t) =
  match dt with D.Float64 -> -708. | _ -> -87.

(* Logsumexp: [m' + log Σ exp (x - m')] along [axes], [m'] the terms' maximum
   where finite and [0] elsewhere, so that no term is shifted by an infinity,
   and an [exp] below {!exp_cutoff} taken as [0]; where the maximum is a NaN,
   the maximum itself, the first NaN term. No term: [log 0], [-∞]. These are
   nx.cpu's two passes, so the two give the same bits. *)
let logsumexp (type s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    shape axes (x : (float, s, d) Value.t) : (float, s, d) Value.t =
  let dt = Prim.dtype x in
  let out = Prim.reduced shape axes in
  let f v = E.const dt (D.of_float dt v) in
  let sum_exp shift =
    let loads =
      match shift with
      | None -> [| Value.Plain x |]
      | Some m ->
          [| Value.Plain x; Value.Plain (spread_back apply ~by shape axes m) |]
    in
    let ins = Iarray.init (Array.length loads) (fun _ -> D.Any dt) in
    let t = E.input 0 dt in
    let y = match shift with None -> t | Some _ -> E.(t - input 1 dt) in
    let e = E.(where (y < f (exp_cutoff dt)) (f 0.) (exp y)) in
    reduce1 apply ~by shape axes Sum (P.of_exprs ~ins [| E.any e |]) dt loads
  in
  if Array.exists (fun a -> shape.(a) = 0) axes then
    map1 apply ~by out
      (P.of_node ~ins:[| D.Any dt |] (Op1 (Unary Log, D.Any dt, 0)))
      dt
      [| Value.Plain (sum_exp None) |]
  else
    let m = reduce_plain apply ~by axes Max x in
    (* [m'], from the maximum [m]. *)
    let finite m = E.(where (abs m < f Float.infinity) m (f 0.)) in
    let shift =
      let ins : D.any iarray = [| D.Any dt |] in
      let prog = P.of_exprs ~ins [| E.any (finite (E.input 0 dt)) |] in
      map1 apply ~by out prog dt [| Value.Plain m |]
    in
    let s = sum_exp (Some shift) in
    let ins : D.any iarray = [| D.Any dt; D.Any dt |] in
    let m' = E.input 0 dt in
    let r = E.(finite m' + log (input 1 dt)) in
    map1 apply ~by out
      (P.of_exprs ~ins [| E.any (nan_or m' r) |])
      dt
      [| Value.Plain m; Value.Plain s |]

(* The unsigned integer of a narrow float's width, 4, 8 or 16 bits. *)
type narrow = Narrow : (int, 'r) D.t -> narrow

(* [r], [x] reduced along [axes] of [shape] into [x]'s dtype, a float narrower
   than its accumulator, with each NaN result the first NaN term as [x] holds
   it, as the kernels give it: of the NaN terms, the least
   [position · 2^w + bits], [w] the dtype's width and [position] the term's in
   C order of the reduced indices. The accumulator's NaN would be quiet. *)
let held_nan (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    shape axes (x : (v, s, d) Value.t) (r : (v, s, d) Value.t) :
    (v, s, d) Value.t =
  let dt = Prim.dtype x in
  let (Narrow u) =
    match D.bits dt with
    | 4 -> Narrow D.Uint4
    | 8 -> Narrow D.Uint8
    | _ -> Narrow D.Uint16
  in
  let int64 n = E.const D.Int64 (Int64.of_int n) in
  let codes = 1 lsl D.bits dt and none = E.const D.Int64 Int64.max_int in
  let position, _ =
    Array.fold_right
      (fun a (pos, stride) ->
        (E.(pos + (coord a * int64 stride)), stride * shape.(a)))
      axes
      (int64 0, 1)
  in
  let t = E.input 0 dt in
  let packed = E.((position * int64 codes) + cast D.Int64 (bitcast u t)) in
  let pick = E.(where (t <> t) packed none) in
  let first =
    reduce1 apply ~by shape axes Min
      (P.of_exprs ~ins:[| D.Any dt |] [| E.any pick |])
      D.Int64 [| Value.Plain x |]
  in
  let q = E.input 1 D.Int64 in
  let held = E.(bitcast dt (cast u (q mod int64 codes))) in
  map1 apply ~by (Prim.reduced shape axes)
    (P.of_exprs
       ~ins:[| D.Any dt; D.Any D.Int64 |]
       [| E.any E.(where (q < none) held (input 0 dt)) |])
    dt
    [| Value.Plain r; Value.Plain first |]

(* Moments in two passes: the mean, [Σ x / n], then the variance by the
   corrected two-pass algorithm (Chan, Golub and LeVeque 1983; Higham,
   Accuracy and Stability, §1.9): [(Σ d² - Σ d (Σ d / n)) / n], [d = x -
   mean], or [+0] where that is negative. [Σ (x - c)² - (Σ (x - c))² / n] is
   [Σ (x - μ)²] for any [c], so the mean's rounding reaches the variance at
   second order, where [Σ d² / n] alone carries it squared. [(Σ d)²] can reach
   [n Σ d²] and overflow where [Σ d²] does not; [Σ d (Σ d / n)] is at most [Σ
   d²]. Where [Σ d²] overflows, it may too, and their difference is a NaN: the
   variance is then [Σ d²], [+∞]. Where the mean is a NaN, both are the mean. *)
let moments (type s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    shape axes (x : (float, s, d) Value.t) :
    (float, s, d) Value.t * (float, s, d) Value.t =
  let dt = Prim.dtype x in
  let out = Prim.reduced shape axes in
  let n = Array.fold_left (fun n a -> n * shape.(a)) 1 axes in
  let count = E.const dt (D.of_float dt (Float.of_int n)) in
  let one : D.any iarray = [| D.Any dt |]
  and two : D.any iarray = [| D.Any dt; D.Any dt |]
  and three : D.any iarray = [| D.Any dt; D.Any dt; D.Any dt |] in
  let mean =
    let s = E.input 0 dt in
    map1 apply ~by out
      (P.of_exprs ~ins:one [| E.any (nan_or s (E.fdiv s count)) |])
      dt
      [| Value.Plain (reduce_plain apply ~by axes Sum x) |]
  in
  let spread = spread_back apply ~by shape axes mean in
  let deviations f =
    let d = E.(input 0 dt - input 1 dt) in
    reduce1 apply ~by shape axes Sum
      (P.of_exprs ~ins:two [| E.any (f d) |])
      dt
      [| Value.Plain x; Value.Plain spread |]
  in
  let firsts = deviations Fun.id and squares = deviations E.(fun d -> d * d) in
  let var =
    let s1 = E.input 1 dt and s2 = E.input 2 dt in
    let zero = E.const dt (D.zero dt)
    and inf = E.const dt (D.of_float dt Float.infinity) in
    let q = E.(fdiv (s2 - (s1 * fdiv s1 count)) count) in
    let r = E.(where (s2 = inf) s2 (where (q < zero) zero q)) in
    map1 apply ~by out
      (P.of_exprs ~ins:three [| E.any (nan_or (E.input 0 dt) r) |])
      dt
      [| Value.Plain mean; Value.Plain firsts; Value.Plain squares |]
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
  (* The operands are the loads: [x] and [m] where they may be NaNs, then
     [x]'s keys, then [m]'s. *)
  let ins = Iarray.of_array (Array.map (fun (Value.Any y) -> D.Any (Prim.dtype y)) loads) in
  let v = List.length values and k = List.length kx in
  let matches j =
    let (Value.Any a) = loads.(v + j) in
    let kt = Prim.dtype a in
    E.equal (E.input (v + j) kt) (E.input (v + k + j) kt)
  in
  let same =
    List.fold_left
      (fun a j -> E.(a land matches j))
      (matches 0)
      (List.init (k - 1) succ)
  in
  let found =
    if not nan then same
    else
      let isnan i = E.(i <> i) in
      E.(same lor (isnan (input 0 dt) land isnan (input 1 dt)))
  in
  let int64 n = E.const D.Int64 (Int64.of_int n) in
  let position, _ =
    Array.fold_right
      (fun a (pos, stride) ->
        (E.(pos + (coord a * int64 stride)), stride * shape.(a)))
      axes
      (int64 0, 1)
  in
  let pick = E.(where found position (const D.Int64 Int64.max_int)) in
  let p =
    reduce1 apply ~by shape axes Min (P.of_exprs ~ins [| E.any pick |]) D.Int64
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

(* Whether [v] is finite: [|v| < ∞], [false] for a NaN. *)
let finite v =
  let dt = E.dtype v in
  E.(abs v < const dt (D.of_float dt Float.infinity))

(* Logsumexp by {!Pairwise.scan}: results [a] and [b] combine as
   [M + log1p (exp (m - M))], [M] their [Maximum] and [m] their [Minimum], or
   as [M] where [M] is not finite: [-∞] for terms all [-∞], [+∞] past a [+∞],
   the first NaN with its bits. Index [0] is its term [x] as [x + 0], so that
   a lone [-0] gives [+0], as the reduction does. Each combination shifts by
   its own maximum, so no prefix is lost below a later one. *)
let running_logsumexp (type s d)
    (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by shape axis
    (x : (float, s, d) Value.t) : (float, s, d) Value.t =
  let dt = Prim.dtype x in
  let a = E.input 0 dt and y = E.input 1 dt in
  let big = E.maximum a y and small = E.minimum a y in
  let combine =
    [| E.(any (where (finite big) (big + log1p (exp (small - big))) big)) |]
  in
  let first = [| E.(any (input 0 dt + const dt (D.of_float dt 0.))) |] in
  let y = Pairwise.scan apply ~by shape axis ~first combine [| Value.Any x |] in
  Prim.expect dt y.(0)

(* {!Nx_kernel.reduction}'s Moments scan by {!Pairwise.scan}, over Chan, Golub
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
let running_moments (type s d)
    (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by shape axis
    (x : (float, s, d) Value.t) : (float, s, d) Value.t * (float, s, d) Value.t =
  let dt = Prim.dtype x in
  let f v = E.const dt (D.of_float dt v) in
  let layout = L.contiguous shape in
  let two prog loads =
    let a, (b, ()) =
      apply ~by (Value.Map { layout; prog; outs = Value.[ dt; dt ]; loads })
    in
    (a, b)
  in
  (* [c], from the first term [t]: [t] where finite, else [0]. *)
  let shift t = E.where (finite t) t (f 0.) in
  if shape.(axis) = 0 then
    two (P.of_exprs ~ins:[||] [| E.any (f 0.); E.any (f 0.) |]) [||]
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
      let ins : D.any iarray = [| D.Any dt; D.Any dt |] in
      let t = E.input 0 dt in
      let d = E.(t - shift (input 1 dt)) in
      map_n apply ~by shape
        (P.of_exprs ~ins E.[| any d; any (f 0.); any (f 1.); any (t <> t) |])
        dts
        [| Value.Plain x; Value.Plain first |]
    in
    let combine =
      let ma = E.input 0 dt and sa = E.input 1 dt and na = E.input 2 dt in
      let za = E.input 3 D.Bool in
      let mb = E.input 4 dt and sb = E.input 5 dt and nb = E.input 6 dt in
      let zb = E.input 7 D.Bool in
      let n = E.(na + nb) in
      let delta = E.(mb - ma) in
      let w = E.fdiv nb n in
      let chan = E.(ma + (delta * w)) in
      let both = E.(finite ma && finite mb) in
      let past = E.(where za ma (where zb mb (ma + mb))) in
      let mean = E.where both chan past in
      let between = E.(delta * delta * (na * w)) in
      let s = E.(sa + sb + between) in
      E.[| any mean; any s; any n; any (za || zb) |]
    in
    let state = Pairwise.scan apply ~by shape axis combine terms in
    let ins : D.any iarray = [| D.Any dt; D.Any dt; D.Any dt |] in
    let m = E.input 0 dt and s = E.input 1 dt and c = shift (E.input 2 dt) in
    let n = E.(cast dt (coord axis) + f 1.) in
    let var = E.(where (finite m) (fdiv s n) (m - m)) in
    two
      (P.of_exprs ~ins [| E.(any (c + m)); E.any var |])
      [|
        Value.Plain (Prim.expect dt state.(0));
        Value.Plain (Prim.expect dt state.(1));
        Value.Plain first;
      |]

(* Arg by {!Pairwise.scan} over the pair of an extreme and its position: the
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
    | Some (Part pt) ->
        let pairs = apply ~by (Value.Bitcast (pt, x)) in
        let part j = Value.Any (half apply ~by shape pairs j) in
        (D.Any pt, [| 2; 3 |], [| part 0; part 1 |])
    | None -> (D.Any dt, [| 0 |], [||])
  in
  let (D.Any kdt) = kt in
  let floats = match D.kind kdt with D.Float -> true | _ -> false in
  let n = 2 + Array.length carried in
  let before x y =
    if not floats then E.(x < y)
    else E.(x < y || (x = y && recip x < recip y))
  in
  let same x y =
    if not floats then E.(x = y) else E.(x = y && recip x = recip y)
  in
  (* Whether keys [u] are below keys [v], comparing from key [i]. *)
  let rec below u v i =
    let lt = before u.(i) v.(i) in
    if i = Array.length u - 1 then lt
    else
      let rest = below u v (i + 1) in
      E.(lt || (same u.(i) v.(i) && rest))
  in
  let ka = Array.map (fun i -> E.input i kdt) keys in
  let kb = Array.map (fun i -> E.input (n + i) kdt) keys in
  let past = match e with Max -> below ka kb 0 | Min -> below kb ka 0 in
  let yes = E.const D.Bool true and no = E.const D.Bool false in
  let past =
    if not floats then past
    else
      let nan k = Array.fold_left (fun c i -> E.(c || i <> i)) no k in
      E.(where (nan ka) no (where (nan kb) yes past))
  in
  (* State element [i], of [dt]: the later's where it is past. *)
  let pick i dt = E.any (E.where past (E.input (n + i) dt) (E.input i dt)) in
  let combine =
    Array.append
      [| pick 0 dt; pick 1 D.Int64 |]
      (Array.mapi (fun j _ -> pick (2 + j) kdt) carried)
  in
  let state =
    Pairwise.scan apply ~by shape axis combine
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
        let out = output prog k in
        let (Floats x) =
          floats ~by (body apply ~by layout prog k loads (accumulator out))
        in
        let r = cast_to apply ~by dt (logsumexp apply ~by shape axes x) in
        let narrow = (not (same (accumulator out) out)) && same out (D.Any dt) in
        if (not narrow) || Array.exists (fun a -> shape.(a) = 0) axes then
          Some r
        else
          let y = Prim.expect dt (body apply ~by layout prog k loads out) in
          Some (held_nan apply ~by shape axes y r)
    | Moments (k, dt) ->
        let (Floats x) =
          floats ~by
            (body apply ~by layout prog k loads (accumulator (output prog k)))
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
    axis prog (r : (d, a) Value.reduction) (loads : d Value.load array)
    (carry : d Value.any option) : a option =
  let loop : type q.
      Nx_array.Layout.t ->
      P.t ->
      (d, q) Value.reduction ->
      d Value.load array ->
      q =
   fun layout prog reduction loads ->
    (* A complex sum scans its parts as floats, and its carry's. *)
    let carry =
      match (carry, reduction) with
      | Some (Value.Any c), Value.Monoid (_, _, dt)
        when Prim.rank c < L.rank layout ->
          Some (Value.Any (apply ~by (Value.Bitcast (dt, c))))
      | _ -> carry
    in
    apply ~by (Value.Scan { layout; axis; prog; reduction; loads; carry })
  in
  match r with
  | Moments (k, dt) ->
      let (Floats x) =
        floats ~by
          (body apply ~by layout prog k loads (accumulator (output prog k)))
      in
      let mean, var = running_moments apply ~by (L.shape layout) axis x in
      Some (cast_to apply ~by dt mean, cast_to apply ~by dt var)
  | Monoid (Logsumexp, k, dt) ->
      let (Floats x) =
        floats ~by
          (body apply ~by layout prog k loads (accumulator (output prog k)))
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

(* float32 would not do for the floats of 8 or 16 bits: a store from it quiets
   their signalling NaNs. *)
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
  let int64 n = E.const D.Int64 (Int64.of_int n) in
  let sum = ref (int64 !first) in
  Array.iteri
    (fun d (g : Nx_array.Move.range) ->
      let k = int64 (g.step * strides.(d)) in
      sum := E.(!sum + (coord d * k)))
    rs;
  P.of_exprs ~ins:[||] [| E.any !sum |]

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
  let zero = E.const dt (D.zero dt) in
  (* The sum so far plus taps [first] to [first + count - 1]. *)
  let add sum first count =
    let held = match sum with None -> [] | Some s -> [ Value.Plain s ] in
    let loads =
      Array.of_list (held @ List.init count (fun i -> tap (first + i)))
    in
    let ins = Iarray.init (Array.length loads) (fun _ -> D.Any dt) in
    let at = List.length held in
    let total = ref (if at = 0 then zero else E.input 0 dt) in
    for i = at to at + count - 1 do
      total := E.(!total + input i dt)
    done;
    map1 apply ~by wide (P.of_exprs ~ins [| E.any !total |]) dt loads
  in
  let rec runs sum first =
    if first >= taps then Option.get sum
    else
      let count = min run_taps (taps - first) in
      runs (Some (add sum first count)) (first + count)
  in
  if Array.exists (( = ) 0) shape || Array.exists (( = ) 0) xs then
    map1 apply ~by shape (P.of_exprs ~ins:[||] [| E.any zero |]) dt [||]
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

(* The map of pairs of shape [out], [2; …], whose result's parts are
   [f parts], [parts] each load's real and imaginary parts: each load is pairs
   whose shape broadcasts to [out]. *)
let pairs_map (type r d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    out (loads : (float, r, d) Value.t array) f : (float, r, d) Value.t =
  let dt = Prim.dtype loads.(0) in
  let ins = Iarray.init (2 * Array.length loads) (fun _ -> D.Any dt) in
  let x k = E.input k dt in
  let parts = Array.mapi (fun i _ -> (x (2 * i), x ((2 * i) + 1))) loads in
  let re, im = f parts in
  let imaginary = E.(coord 0 = const D.Int64 1L) in
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
  map1 apply ~by out (P.of_exprs ~ins [| E.(any (where imaginary im re)) |]) dt views

(* [x·y] for complex numbers' parts. *)
let cmul (xr, xi) (yr, yi) = E.((xr * yr) - (xi * yi), (xr * yi) + (xi * yr))

(* [e^(i·a)]. *)
let unit_at a = (E.cos a, E.sin a)

(* The transform of exponent sign [sign] of [pairs], of shape [2; b; n] with [n]
   a power of two, along its last axis, by Stockham's autosort: at each of [log₂
   n] levels, a map gives the sums and the twiddled differences of the two
   halves of each of [n / 2^l]-point transforms, interleaved, so the result
   needs no reordering. *)
let stockham (type r d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    sign (pairs : (float, r, d) Value.t) : (float, r, d) Value.t =
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
      let ins = Iarray.init 4 (fun _ -> D.Any dt) in
      let x k = E.input k dt in
      let sr = E.(x 0 + x 2) and si = E.(x 1 + x 3) in
      let d = E.(x 0 - x 2, x 1 - x 3) in
      let p = E.cast dt (E.coord 2) in
      let turn =
        E.const dt (D.of_float dt (sign *. 2. *. Float.pi /. Float.of_int nl))
      in
      let tr, ti = cmul d (unit_at E.(p * turn)) in
      let is axis = E.(coord axis = const D.Int64 1L) in
      let im = is 0 in
      let sum = E.where im si sr and diff = E.where im ti tr in
      let y =
        map1 apply ~by out
          (P.of_exprs ~ins [| E.(any (where (is 3) diff sum)) |])
          dt
          [| part 0 0; part 0 1; part m 0; part m 1 |]
      in
      level y m (2 * s)
  in
  move (Reshape [| 2; batch; n |]) (level pairs n 1)

let bit_length n =
  let rec go n b = if n = 0 then b else go (n lsr 1) (b + 1) in
  go n 0

(* [a = t mod m] times itself by Horner's rule over [a]'s digits of
   [c = 62 - ⌈log₂ m⌉] bits, the most significant first: [r] below [m] times
   [2^c], plus [a] times a digit, stays below [2^63]. For [m] up to [2^31],
   one digit: [a · a mod m]. *)
let square_mod m t =
  let i64 v = E.const D.Int64 (Int64.of_int v) in
  let l = bit_length (m - 1) in
  let c = 62 - l in
  let digits = max 1 ((l + c - 1) / c) in
  let a = E.(t mod i64 m) and base = i64 (1 lsl c) in
  let mask = i64 ((1 lsl c) - 1) in
  let digit i =
    let d = if i = 0 then a else E.idiv a (i64 (1 lsl (c * i))) in
    if i = digits - 1 then d else E.(d land mask)
  in
  let top = digit (digits - 1) in
  let r = ref E.(a * top mod i64 m) in
  for i = digits - 2 downto 0 do
    let d = digit i in
    r := E.(((!r * base) + (a * d)) mod i64 m)
  done;
  !r

(* The transform of exponent sign [sign] of [pairs], of shape [2; b; n], along
   its last axis, by Bluestein's chirp: [jk = (j² + k² - (k - j)²) / 2] makes it
   [c_k Σ_j (x_j c_j) conj c_(k-j)], [c_j = e^(sign·iπ j²/n)], a convolution,
   computed by {!stockham} at the power of two [M ≥ 2n - 1]. [j²] is taken
   modulo [2n] by {!square_mod}, exact in int64, so the chirp's angle stays
   below [2π]. *)
let bluestein (type r d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    sign (pairs : (float, r, d) Value.t) : (float, r, d) Value.t =
  let dt = Prim.dtype pairs and s0 = Prim.shape pairs in
  let batch = s0.(1) and n = s0.(2) in
  let m =
    let rec up p = if p >= (2 * n) - 1 then p else up (2 * p) in
    up 1
  in
  let i64 v = E.const D.Int64 (Int64.of_int v) in
  let f v = E.const dt (D.of_float dt v) in
  (* [e^(sg·iπ (t² mod 2n)/n)] at the int64 [t]. *)
  let chirp sg t =
    let q = E.cast dt (square_mod (2 * n) t) in
    unit_at E.(q * f (sg *. Float.pi /. Float.of_int n))
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
    pairs_map apply ~by [| 2; batch; m |] [| padded |] (fun parts ->
        cmul parts.(0) (chirp sign (E.coord 2)))
  in
  (* [conj c_j] at [t = j] and [t = M - j], [0] elsewhere. *)
  let kernel =
    let t = E.coord 2 in
    let below = E.(t < i64 n) and mirror = E.(i64 m - t) in
    let inside = E.(below || mirror < i64 n) in
    let cr, ci = chirp (-.sign) (E.where below t mirror) in
    let part = E.(where (coord 0 = i64 1) ci cr) in
    map1 apply ~by [| 2; 1; m |]
      (P.of_exprs ~ins:[||] [| E.(any (where inside part (f 0.))) |])
      dt [||]
  in
  let fa = stockham apply ~by (-1.) a
  and fk = stockham apply ~by (-1.) kernel in
  let product =
    pairs_map apply ~by [| 2; batch; m |] [| fa; fk |] (fun parts ->
        cmul parts.(0) parts.(1))
  in
  let conv = stockham apply ~by 1. product in
  let conv =
    apply ~by (Value.Move (Slice [| whole 2; whole batch; whole n |], conv))
  in
  pairs_map apply ~by [| 2; batch; n |] [| conv |] (fun parts ->
      let re, im = cmul parts.(0) (chirp sign (E.coord 2)) in
      let scale = f (1. /. Float.of_int m) in
      E.(re * scale, im * scale))

(* The transform of exponent sign [sign] of [pairs] along axis [axis] of the
   complex value they hold: its lines laid as [2; b; n], {!stockham} for a power
   of two [n], {!bluestein} otherwise. *)
let along (type r d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by sign
    axis (pairs : (float, r, d) Value.t) : (float, r, d) Value.t =
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
        let (Part pt) = Option.get (parts (D.Any dt)) in
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
        let (Part pt) = Option.get (parts (D.Any (Prim.dtype x))) in
        let p = to_pairs apply ~by pt x in
        let p =
          Array.fold_left
            (fun p a -> along apply ~by 1. a p)
            p (Array.sub axes 0 k)
        in
        (* Bins [0] and, for an even [n], [n/2] keep their real parts alone. *)
        let p =
          let dt = Prim.dtype p and t = E.coord (l + 1) in
          let half = Int64.of_int (if n mod 2 = 0 then n / 2 else 0) in
          let edge = E.(t = const D.Int64 0L || t = const D.Int64 half) in
          pairs_map apply ~by (Prim.shape p) [| p |] (fun parts ->
              let re, im = parts.(0) in
              (re, E.where edge (E.const dt (D.zero dt)) im))
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
              pairs_map apply ~by (Prim.shape rest) [| rest |] (fun parts ->
                  let re, im = parts.(0) in
                  (re, E.neg im))
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
   [init], computed in [acc] into [out], as other operations. A result of
   another dtype than its accumulator is the contraction into the
   accumulator, which kernels compute in [order], rounded once. Any other is
   the products in the accumulator at every index of the result's axes then
   the contracted ones, each rounded, a sum over the contracted ones, and
   [init] added. That sum is a reduction's, in the reduction's order
   whatever [order]: no kernel's bits, and the same for any window of the
   result only where that order depends on the count of terms alone, as
   nx.cpu's reduction order does. *)
let contract (type v s a b c e d) (apply : 'q. by:string -> 'q Value.prim -> 'q)
    ~by ~order ~batch ~contracting ~acc:(D.Any acc) (out : (v, s) D.t)
    (a : (a, b, d) Value.t) (b : (c, e, d) Value.t)
    (init : (v, s, d) Value.t option) : (v, s, d) Value.t =
  if not (D.equal acc out) then
    let init = Option.map (cast_to apply ~by acc) init in
    let y =
      apply ~by
        (Value.Contract
           { order; batch; contracting; acc = D.Any acc; out = acc; a; b; init })
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
  (* No position leaves [into], copied: no result is an operand. *)
  if s.(axis) = 0 then apply ~by (Value.Copy into)
  else
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

let ones w = if w = 64 then -1L else Int64.(sub (shift_left 1L w) 1L)

(* The element of the unsigned dtype [dt] of an int64's low bits. *)
let unsigned_of (type v s) (dt : (v, s) D.t) : int64 -> v =
  match dt with
  | D.Uint8 -> Int64.to_int
  | D.Uint16 -> Int64.to_int
  | D.Uint32 -> Int64.to_int32
  | D.Uint64 -> Fun.id
  | _ -> invalid_arg "Expand.unsigned_of: not an unsigned dtype"

(* The key [k], of an unsigned dtype of [w] bits, as int64s of at most [c]
   bits, the most significant first: each chunk taken in [k]'s dtype, whose
   division and mask are unsigned, then cast. *)
let chunks (type v s) c (k : (v, s) E.t) =
  let dt = E.dtype k in
  let w = D.bits dt and of_int64 = unsigned_of dt in
  let count = (w + c - 1) / c in
  let word v = E.const dt (of_int64 v) in
  List.init count (fun i ->
      let shift = (count - 1 - i) * c in
      let high =
        if shift = 0 then k else E.idiv k (word (Int64.shift_left 1L shift))
      in
      E.cast D.Int64 (if i = 0 then high else E.(high land word (ones c))))

(* The {!chunks} of [c] bits of the order key of [x], a float of [w] bits, a
   NaN where [nan] is: its bits with the sign set where it is clear, all
   flipped where it is set, so that [-0] keys just below [+0]; all ones for
   every NaN. *)
let float_key c w x nan =
  let (U (ut, of_int64)) = unsigned w in
  let u = E.bitcast ut x in
  let top = E.const ut (of_int64 (Int64.shift_left 1L (w - 1))) in
  let all = E.const ut (of_int64 (ones w)) in
  chunks c E.(where nan all (where (top <= u) (u lxor all) (u lor top)))

let key_bits n = 63 - bit_length (n - 1)

(* [x]'s order keys, most significant first: int64 values of [c] bits,
   whose lexicographic order is
   {!Nx_kernel.S.sort}'s order of [x]'s elements, every NaN, and every
   complex number with a NaN part, one key above [+∞]. A signed integer keys
   as its bits with the sign flipped, a float as {!float_key} gives, a
   complex number as its real part's key, then its imaginary part's. *)
let sort_keys (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    c (x : (v, s, d) Value.t) : (int64, D.int64_elt, d) Value.t list =
  let dt = Prim.dtype x and shape = Prim.shape x in
  let w = D.bits dt in
  (* The keys [ks] over [loads], one map each. *)
  let keys loads ks =
    let ins = Iarray.of_array (Array.map (fun (Value.Any y) -> D.Any (Prim.dtype y)) loads) in
    let loads = Array.map (fun (Value.Any y) -> Value.Plain y) loads in
    List.map
      (fun k -> map1 apply ~by shape (P.of_exprs ~ins [| E.any k |]) D.Int64 loads)
      ks
  in
  let v = E.input 0 dt in
  match D.kind dt with
  | D.Boolean -> keys [| Value.Any x |] [ E.cast D.Int64 v ]
  | D.Unsigned -> keys [| Value.Any x |] (chunks c v)
  | D.Signed ->
      let (U (ut, of_int64)) = unsigned w in
      let top = E.const ut (of_int64 (Int64.shift_left 1L (w - 1))) in
      keys [| Value.Any x |] (chunks c E.(bitcast ut v lxor top))
  | D.Float -> keys [| Value.Any x |] (float_key c w v E.(v <> v))
  | D.Complex ->
      let w = w / 2 in
      let (Part pt) = Option.get (parts (D.Any dt)) in
      let pairs = apply ~by (Value.Bitcast (pt, x)) in
      let part j = Value.Any (half apply ~by shape pairs j) in
      let re = E.input 0 pt and im = E.input 1 pt in
      let nan = E.(re <> re || im <> im) in
      keys [| part 0; part 1 |] (float_key c w re nan @ float_key c w im nan)

(* The positions along [axis] that sort the keys [k], of {!key_bits} bits or
   fewer, stably: a bitonic network over each key packed above its position,
   which makes the packed integers distinct, so the network's order is the
   stable one. The axis is padded to a power of two [N] with integers that sort last.
   Each of the network's [log N (log N + 1) / 2] steps is a gather of each
   element's partner, the element whose position differs in one bit, and a
   map keeping the lesser or the greater of the two: O(N log² N). A
   descending sort complements the positions, so that equal keys keep
   increasing positions. *)
let bitonic (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by axis
    descending (k : (int64, D.int64_elt, d) Value.t) :
    (int64, D.int64_elt, d) Value.t =
  let shape = Prim.shape k in
  let n = shape.(axis) in
  let low = bit_length (n - 1) in
  let mask = Int64.(sub (shift_left 1L low) 1L) in
  let int64 v = E.const D.Int64 v in
  let one : D.any iarray = [| D.Any D.Int64 |] in
  let two : D.any iarray = [| D.Any D.Int64; D.Any D.Int64 |] in
  (* A position, or its complement in a descending sort. *)
  let tie p = if descending then E.(int64 mask - p) else p in
  let packed =
    let high = E.(input 0 D.Int64 * int64 (Int64.shift_left 1L low)) in
    map1 apply ~by shape
      (P.of_exprs ~ins:one [| E.(any (high + tie (coord axis))) |])
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
      let j = int64 (Int64.of_int j) in
      map1 apply ~by wide
        (P.of_exprs ~ins:[||] [| E.(any (coord axis lxor j)) |])
        D.Int64 [||]
    in
    let y = apply ~by (Value.Gather { axis; idx = partner; x }) in
    let a = E.input 0 D.Int64 and b = E.input 1 D.Int64 in
    let clear bit = E.(coord axis land int64 (Int64.of_int bit) = int64 0L) in
    let up = clear block and first = clear j in
    let lesser = E.compare (if descending then Not_equal else Equal) up first in
    let lo = E.minimum a b and hi = E.maximum a b in
    map1 apply ~by wide
      (P.of_exprs ~ins:two [| E.(any (where lesser lo hi)) |])
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
  let s = E.input 0 D.Int64 in
  map1 apply ~by shape
    (P.of_exprs ~ins:one [| E.any (tie E.(s land int64 mask)) |])
    D.Int64 [| Value.Plain sorted |]

(* A sort as positions from its keys of [bits] bits, the least significant
   first, each pass a stable {!bitonic} sort of one key in the order of the
   passes before it, then
   a gather of the elements: O(K N log² N) for K keys. A sub-byte sort
   runs at its accumulator, which holds each code and orders them alike, its
   values cast back once; no sub-byte dtype has a NaN code. *)
let sort (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by ~bits
    axis descending k (x : (v, s, d) Value.t) :
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
            (List.rev (sort_keys apply ~by bits x))
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
      (* An output that is an operand is its copy: no result is an operand. *)
      let out o =
        match Iarray.get prog.nodes o with
        | In _ ->
            let (Value.Any x) = value o in
            Value.Any (apply ~by (Value.Copy x))
        | Const _ | Coord _ | Op1 _ | Op2 _ | Op3 _ -> value o
      in
      Some (results outs (List.map out (Iarray.to_list prog.outs)))
  | Value.Reduce { layout; axes; prog; reductions; loads }
    when Array.exists Prim.is_padded loads || not (core_reduce prog reductions)
    ->
      reduce apply ~by layout axes prog reductions loads
  | Value.Scan { layout; axis; prog; reduction = r; loads; carry }
    when Array.exists Prim.is_padded loads || not (core_scan prog r) ->
      scan apply ~by layout axis prog r loads carry
  | Value.Contract { order; batch; contracting; acc; out; a; b; init } ->
      Some (contract apply ~by ~order ~batch ~contracting ~acc out a b init)
  | Value.Map { layout; prog; outs; loads } ->
      widened apply ~by layout prog outs loads
  | Value.Assemble { dtype; shape; fill; pieces } ->
      Some (assemble apply ~by dtype shape fill pieces)
  | Value.Gather { axis; idx; x } -> gather apply ~by axis idx x
  | Value.Scatter { combine; unique; axis; idx; updates; into } ->
      scatter apply ~by combine ~unique axis idx updates into
  | Value.Sort { axis; descending; k; x } ->
      let bits = key_bits (Prim.dim x axis) in
      sort apply ~by ~bits axis descending k x
  | Value.Fold { shape; pad; x } -> Some (fold apply ~by shape pad x)
  | Value.Fft f -> Some (fft apply ~by f)
  | Value.Copy x ->
      (* A copy is the map of one [Copy]. *)
      let dt = Prim.dtype x in
      let prog = P.of_node ~ins:[| D.Any dt |] (Op1 (Copy, D.Any dt, 0)) in
      let layout = L.contiguous (Prim.shape x) in
      Option.map fst
        (widened apply ~by layout prog Value.[ dt ] [| Value.Plain x |])
  | Value.Reduce _ | Value.Scan _ | Value.Linalg _ | Value.Move _
  | Value.Bitcast _ | Value.Place _ | Value.Check _ ->
      None
