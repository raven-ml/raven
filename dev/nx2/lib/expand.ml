(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Nx_array.Dtype
module L = Nx_array.Layout
module P = Nx_kernel.Prog
module M = Nx_array.Move
module E = P.E

(* Node [i] of [prog] over [loads], as a one-node map of [shape] over [value j]
   for each node [j] it reads. *)
(* [xs] as plain loads. *)
let plains (type d) (xs : d Value.any array) : d Value.load iarray =
  Iarray.init (Array.length xs) (fun i ->
      let (Value.Any x) = xs.(i) in
      Value.Plain x)

let node_map (type d) (apply : 'r. by:string -> 'r Value.prim -> 'r) ~by layout
    (prog : P.t) (loads : d Value.any array) (value : int -> d Value.any) i :
    d Value.any =
  let (D.Any dt) = P.dtype prog i in
  let one node reads =
    let ins =
      Iarray.of_array
        (Array.map (fun (Value.Any x) -> D.Any (Prim.dtype x)) reads)
    in
    let loads = plains reads in
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
  | Logsumexp (k, dt) -> (k, D.Any dt)
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
  | Monoid _ | Logsumexp _ | Moments _ | Arg _ -> false

(* Whether a loop of [prog] by [r] is plain: the identity of its operand,
   accumulated and rounded in the operand's own dtype, and no complex sum. Any
   other expands into plain loops. *)
let plain prog r =
  let k, dt = target r in
  P.is_identity prog && k = 0
  && same (output prog 0) dt
  && same (Nx_kernel.carrier dt) dt
  && not (complex_sum r)

(* Whether a scan has no expansion: a plain [Sum], [Prod], [Max] or [Min]. *)
let simple_scan (type d a) prog (r : (d, a) Value.reduction) =
  match r with
  | Monoid ((Sum | Prod | Max | Min), _, _) -> plain prog r
  | Logsumexp _ | Moments _ | Arg _ -> false

(* Whether a reduction has no expansion: one simple scan's reduction. *)
let simple_reduce (type d q) prog (rs : (d, q) Value.reductions) =
  match rs with [ r ] -> simple_scan prog r | _ -> false

(* Output [k] of [prog] over [loads], of [layout]'s shape, stored in [acc]: the
   operand itself where [prog] is the identity and [acc] its dtype. *)
let body (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by layout
    (prog : P.t) k (loads : d Value.load iarray) (D.Any acc) : d Value.any =
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
       d Value.load iarray ->
       'q) layout prog (loads : d Value.load iarray)
    (r : (d, a) Value.reduction)
    : a option =
  match r with
  | Monoid (m, k, dt) ->
      let (Value.Any b) =
        body apply ~by layout prog k loads (Nx_kernel.carrier (output prog k))
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
  | Logsumexp _ | Moments _ | Arg _ -> None

(* Optional reductions *)

(* [x], [shape] reduced along [axes], broadcast back to [shape]. *)
let spread_back (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    shape axes (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  let units =
    Array.mapi (fun a e -> if Iarray.mem a axes then 1 else e) shape
  in
  let move mv x = apply ~by (Value.Move (mv, x)) in
  let x =
    if Prim.has_shape x units then x
    else move (M.Reshape (Iarray.of_array units)) x
  in
  if units = shape then x else move (M.Broadcast (Iarray.of_array shape)) x

(* The map of [prog] over [loads], of [shape], into [dt]. *)
let map1 (type w r d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by shape
    prog (dt : (w, r) D.t) (loads : d Value.load iarray) : (w, r, d) Value.t =
  fst
    (apply ~by
       (Value.Map
          { layout = L.contiguous shape; prog; outs = Value.[ dt ]; loads }))

(* The monoid [m] of [prog]'s output over [loads], of [shape], along [axes], in
   [dt]. *)
let reduce1 (type w r d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by shape
    axes m prog (dt : (w, r) D.t) (loads : d Value.load iarray) :
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

(* The reduction's cut ({!Nx_kernel.type-reduction}): the least integer whose
   [exp] is a normal float of [dt], float32 or float64, below which an [exp] is
   taken as [0]. *)
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
    let loads : _ Value.load iarray =
      match shift with
      | None -> [| Value.Plain x |]
      | Some m ->
          [| Value.Plain x; Value.Plain (spread_back apply ~by shape axes m) |]
    in
    let ins = Iarray.init (Iarray.length loads) (fun _ -> D.Any dt) in
    let t = E.input 0 dt in
    let y = match shift with None -> t | Some _ -> E.(t - input 1 dt) in
    let e = E.(where (y < f (exp_cutoff dt)) (f 0.) (exp y)) in
    reduce1 apply ~by shape axes Sum (P.of_exprs ~ins [| E.any e |]) dt loads
  in
  if Iarray.exists (fun a -> shape.(a) = 0) axes then
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
  let n = Iarray.fold_left (fun n a -> n * shape.(a)) 1 axes in
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
  let r = Array.length shape in
  let ranges =
    Iarray.init (r + 1) (fun a ->
        if a < r then { M.start = 0; count = shape.(a); step = 1 }
        else { M.start = j; count = 1; step = 1 })
  in
  let p = apply ~by (Value.Move (Slice ranges, pairs)) in
  apply ~by (Value.Move (Reshape (Iarray.of_array shape), p))

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
    Iarray.fold_right
      (fun a (pos, stride) ->
        (E.(pos + (coord a * int64 stride)), stride * shape.(a)))
      axes
      (int64 0, 1)
  in
  let pick = E.(where found position (const D.Int64 Int64.max_int)) in
  let p =
    reduce1 apply ~by shape axes Min (P.of_exprs ~ins [| E.any pick |]) D.Int64
      (plains loads)
  in
  (m, p)

(* Running reductions *)

(* [t], of the float [dt], quieted where it is a NaN: its quiet bit, the top
   of its fraction, set, its sign and the rest of its payload kept, as the
   kernels return a NaN term. float8_e4m3fn's one NaN of each sign is quiet,
   and float4_e2m1fn has none. *)
let quiet (type v s) (dt : (v, s) D.t) (t : (v, s) E.t) : (v, s) E.t =
  let set (type w r) (u : (w, r) D.t) (bit : w) =
    E.(where (t <> t) (bitcast dt (bitcast u t lor const u bit)) t)
  in
  match dt with
  | D.Float64 -> set D.Uint64 (Int64.shift_left 1L 51)
  | D.Float32 -> set D.Uint32 (Int32.shift_left 1l 22)
  | D.Float16 -> set D.Uint16 0x200
  | D.Bfloat16 -> set D.Uint16 0x40
  | D.Float8_e5m2 -> set D.Uint8 0x2
  | _ -> t

(* [x], of floats or complex numbers, with each NaN part quieted. *)
let quieted (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  let floats (type w r) (y : (w, r, d) Value.t) : (w, r, d) Value.t =
    let yt = Prim.dtype y in
    let prog =
      P.of_exprs ~ins:[| D.Any yt |] [| E.any (quiet yt (E.input 0 yt)) |]
    in
    map1 apply ~by (Prim.shape y) prog yt [| Value.Plain y |]
  in
  let dt = Prim.dtype x in
  match parts (D.Any dt) with
  | Some (Part pt) ->
      let pairs = apply ~by (Value.Bitcast (pt, x)) in
      apply ~by (Value.Bitcast (dt, floats pairs))
  | None -> (
      match D.kind dt with D.Float -> floats x | _ -> x)

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
    prog dts (loads : d Value.load iarray) : d Value.any array =
  let (Outputs (outs, read)) = outputs (Array.to_list dts) in
  let layout = L.contiguous shape in
  Array.of_list (read (apply ~by (Value.Map { layout; prog; outs; loads })))

(* Whether [v] is finite: [|v| < ∞], [false] for a NaN. *)
let finite v =
  let dt = E.dtype v in
  E.(abs v < const dt (D.of_float dt Float.infinity))

(* {!Nx_kernel.reduction}'s Logsumexp scan by {!Pairwise.scan}, over states
   [(m, s)] of a maximum and a sum of [e (x - c)], [c] being [m] where finite
   and [0] elsewhere, [e] an [exp] cut as the reduction cuts it.

   The derivation of its bound, to first order, at index [i] of [N = i + 1]
   terms through [D = D_i] combinations, with [M] the terms' maximum, finite,
   and [S = Σ w], [w = exp (x - M)], so [1 <= S <= N]. A term's [w] reaches
   [s] through at most [D] combinations. One where its state's maximum is
   the combined one multiplies by [exp 0 = 1], exactly; any other rescales
   by an [exp], within [5 u] (2 ulps and the rounding), and a product,
   within [u]; each adds within [u]: [7 u] a combination. The rescales'
   arguments [ma - c] round within [u (c - ma)], whose sum along the term's
   path is at most [u (M - x)]: they move [w] by a relative [u (M - x)], and
   [Σ w (M - x) = Σ w ln (1 / w) <= S ln N]. So [s] is within a relative
   [7 D u + u ln N], [log s] within that plus [Log]'s [5 u ln N], and
   [M + log s] rounds within [u |r|]. *)
let running_logsumexp (type s d)
    (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by shape axis
    (x : (float, s, d) Value.t) : (float, s, d) Value.t =
  let dt = Prim.dtype x in
  let f v = E.const dt (D.of_float dt v) in
  let shift m = E.where (finite m) m (f 0.) in
  let e z = E.(where (z < f (exp_cutoff dt)) (f 0.) (exp z)) in
  let two : D.any iarray = [| D.Any dt; D.Any dt |] in
  let terms =
    let t = E.input 0 dt in
    map_n apply ~by shape
      (P.of_exprs ~ins:[| D.Any dt |] E.[| any t; any (e (t - shift t)) |])
      [| D.Any dt; D.Any dt |]
      [| Value.Plain x |]
  in
  let combine =
    let ma = E.input 0 dt and sa = E.input 1 dt in
    let mb = E.input 2 dt and sb = E.input 3 dt in
    let m = E.maximum ma mb in
    let c = shift m in
    E.[| any m; any ((sa * e (ma - c)) + (sb * e (mb - c))) |]
  in
  let state = Pairwise.scan apply ~by shape axis combine terms in
  let m = E.input 0 dt and s = E.input 1 dt in
  let y =
    map1 apply ~by shape
      (P.of_exprs ~ins:two [| E.any E.(where (finite m) (m + log s) m) |])
      dt
      [|
        Value.Plain (Prim.expect dt state.(0));
        Value.Plain (Prim.expect dt state.(1));
      |]
  in
  quieted apply ~by y

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
      let rank = Array.length shape and keep a e =
        { M.start = 0; count = (if a = axis then 1 else e); step = 1 }
      in
      let x0 =
        apply ~by
          (Value.Move (Slice (Iarray.init rank (fun a -> keep a shape.(a))), x))
      in
      apply ~by (Value.Move (Broadcast (Iarray.of_array shape), x0))
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
      (P.of_exprs ~ins [| E.any (quiet dt E.(c + m)); E.any (quiet dt var) |])
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
   [-0] is below [+0]. Exact; a NaN extreme is then quieted. *)
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
  ( quieted apply ~by (Prim.expect dt state.(0)),
    Prim.expect D.Int64 state.(1) )

let reduce (type d q) (apply : 'a. by:string -> 'a Value.prim -> 'a) ~by layout
    axes prog (rs : (d, q) Value.reductions) (loads : d Value.load iarray) :
    q option =
  let loop : type a.
      Nx_array.Layout.t ->
      P.t ->
      (d, a) Value.reduction ->
      d Value.load iarray ->
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
    | Logsumexp (k, dt) ->
        let (Floats x) =
          floats ~by
            (body apply ~by layout prog k loads
               (Nx_kernel.carrier (output prog k)))
        in
        Some (cast_to apply ~by dt (logsumexp apply ~by shape axes x))
    | Moments (k, dt) ->
        let (Floats x) =
          floats ~by
            (body apply ~by layout prog k loads
               (Nx_kernel.carrier (output prog k)))
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
    axis prog (r : (d, a) Value.reduction) (loads : d Value.load iarray)
    (carry : d Value.any option) : a option =
  let loop : type q.
      Nx_array.Layout.t ->
      P.t ->
      (d, q) Value.reduction ->
      d Value.load iarray ->
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
          (body apply ~by layout prog k loads
             (Nx_kernel.carrier (output prog k)))
      in
      let mean, var = running_moments apply ~by (L.shape layout) axis x in
      Some (cast_to apply ~by dt mean, cast_to apply ~by dt var)
  | Logsumexp (k, dt) ->
      let (Floats x) =
        floats ~by
          (body apply ~by layout prog k loads
             (Nx_kernel.carrier (output prog k)))
      in
      let y = running_logsumexp apply ~by (L.shape layout) axis x in
      Some (cast_to apply ~by dt y)
  | Arg (e, k, dt) ->
      let (Value.Any x) = body apply ~by layout prog k loads (output prog k) in
      let m, p = running_arg apply ~by (L.shape layout) axis e x in
      Some (cast_to apply ~by dt m, p)
  | Monoid _ -> reduction apply ~by ~loop layout prog loads r

(* Narrow dtypes *)

(* Whether [dt] is a base dtype: float32, float64, the 8- to 64-bit integers
   and bool. *)
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

(* The dtype a selection, a constant or a copy of [d] computes in: [d] itself
   for a base dtype ({!base}); for a float of 8 or 16 bits, the
   unsigned integer of its width, over the same bits; for a sub-byte dtype, its
   byte-wide accumulator, which holds each of its codes and gives it back; [d]
   itself otherwise. float32 would not do for the floats of 8 or 16 bits: a
   store from it quiets their signalling NaNs. *)
let kept (D.Any dt as d) =
  if base d then d
  else
    match D.bits dt with
    | 8 -> D.Any D.Uint8
    | 16 -> D.Any D.Uint16
    | b when b < 8 -> Nx_kernel.carrier d
    | _ -> d

(* [x] held in [dt]: its bits where [dt] has its width, else its value. *)
let held (type v s w r d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (dt : (w, r) D.t) (x : (v, s, d) Value.t) : (w, r, d) Value.t =
  match D.equal_witness (Prim.dtype x) dt with
  | Some Type.Equal -> x
  | None when D.bits (Prim.dtype x) = D.bits dt ->
      apply ~by (Value.Bitcast (dt, x))
  | None -> cast_to apply ~by dt x

(* A one-node map at a dtype other than a base one, as the node at dtypes that
   hold its values exactly, its result rounded once to its dtype: their
   carriers ({!Nx_kernel.carrier}), or for a selection, a constant and a copy
   the dtypes {!kept} gives, its result read back as {!held} reads it. [None]
   for a node whose dtypes all stay, and for a cast and a bitcast, which no
   other dtype computes exactly. *)
let widened (type d r) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by layout
    (prog : P.t) (outs : (d, r) Value.outs) (loads : d Value.load iarray) :
    r option =
  let n = Iarray.length prog.nodes in
  let node = Iarray.get prog.nodes (n - 1) in
  let dts = Iarray.to_array (Iarray.map Prim.load_dtype loads) in
  let wide =
    match (node : P.node) with
    | Const _ | Op1 (Copy, _, _) | Op3 (Where, _, _, _) -> kept
    | _ -> Nx_kernel.carrier
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
    let operands = Iarray.to_array (Iarray.map load loads) in
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
             loads = plains operands;
           })
    in
    match outs with [ dt ] -> Some (held apply ~by dt v, ()) | _ -> None

(* The program of a region's flat positions in a C-contiguous value whose axes
   have [strides]: [Σ (start + i·step)·stride] at the region's index [i]. *)
let positions (rs : Nx_array.Move.range iarray) strides =
  let first = ref 0 in
  Iarray.iteri
    (fun d (g : Nx_array.Move.range) ->
      first := !first + (g.start * strides.(d)))
    rs;
  let int64 n = E.const D.Int64 (Int64.of_int n) in
  let sum = ref (int64 !first) in
  Iarray.iteri
    (fun d (g : Nx_array.Move.range) ->
      let k = int64 (g.step * strides.(d)) in
      sum := E.(!sum + (coord d * k)))
    rs;
  P.of_exprs ~ins:[||] [| E.any !sum |]

(* An assembly as a fill of the flat result, then per piece in order a scatter
   of its elements at their flat positions: O(n) per piece. *)
let assemble (type v s d) (apply : 'r. by:string -> 'r Value.prim -> 'r) ~by
    (dt : (v, s) Value.dtype) shape (fill : v)
    (pieces : (Nx_array.Move.range iarray * (v, s, d) Value.t) list) :
    (v, s, d) Value.t =
  let shape = Iarray.to_array shape in
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
    (Value.Move
       (Reshape (Iarray.of_array shape), List.fold_left place flat pieces))

(* Padded loads *)

(* [x] padded with [fill] as [p] says, before its windows: an assembly of the
   elements [p] keeps at their padded positions, [lo + j·(interior + 1)] along
   each axis, and [fill] elsewhere; a negative [lo] or [hi] keeps fewer. *)
let padded (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (x : (v, s, d) Value.t) (fill : v) (p : Nx_kernel.pad) : (v, s, d) Value.t =
  let s = Prim.shape x in
  let extent a =
    let d = s.(a) in
    Iarray.get p.lo a + Iarray.get p.hi a + d
    + (Iarray.get p.interior a * max 0 (d - 1))
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
    let gap = Iarray.get p.interior a + 1 and lo = Iarray.get p.lo a in
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
        Iarray.init (Array.length ks) (fun a ->
            let first, count, gap = ks.(a) in
            { M.start = Iarray.get p.lo a + (first * gap); count; step = gap })
      and slice =
        Iarray.init (Array.length ks) (fun a ->
            let start, count, _ = ks.(a) in
            { M.start; count; step = 1 })
      in
      [ (region, apply ~by (Value.Move (Slice slice, x))) ]
  in
  let shape = Iarray.of_array shape in
  apply ~by (Value.Assemble { dtype = Prim.dtype x; shape; fill; pieces })

(* A load read plainly: a padded one as {!padded}'s assembly, viewed through its
   windows. *)
let unpadded (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (l : d Value.load) : d Value.load =
  match l with
  | Plain _ -> l
  | Padded { x; fill; pad } ->
      let y = padded apply ~by x fill pad in
      if Iarray.length pad.windows = 0 then Plain y
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
  let xs = Prim.shape x and nw = Iarray.length p.windows in
  let along = Array.make r None in
  Iarray.iteri
    (fun k (w : M.window) -> along.(w.axis) <- Some (k, w))
    p.windows;
  let sizes = Array.init nw (fun k -> (Iarray.get p.windows k).size) in
  let taps = Array.fold_left ( * ) 1 sizes in
  (* CR: Derive bounded source/result progressions for each tap. A valid
     shape [2], interior [L.max_numel], lo [-L.max_numel], hi [0] needs
     [0; x[1]], but this reverse pad exceeds max_numel; rank-32 inputs can
     also gain a 33rd axis. Solve gap*i - step*w = off-lo with gcd and
     checked Int64 arithmetic, clip before multiplying, then Slice the
     source and pad directly to the result without Window axes. Use step 1
     for singleton progressions and retain the ordered map runs. Cover
     both boundaries through a public Fold that expands. *)
  (* The loop's shape: the result's, then an axis of one per axis read every
     [interior + 1] with [interior > 0], in axis order. *)
  let strided =
    List.filter (fun a -> Iarray.get p.interior a > 0) (List.init r Fun.id)
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
      Iarray.init (r + nw) (fun a ->
          if a < r then { M.start = 0; count = xs.(a); step = 1 }
          else { M.start = j.(a - r); count = 1; step = 1 })
    in
    let xt = apply ~by (Value.Move (Slice ranges, x)) in
    let lead = Iarray.init r (Array.get xs) in
    let xt = apply ~by (Value.Move (Reshape lead, xt)) in
    let lo = Array.make r 0 and hi = Array.make r 0 in
    let interior = Array.make r 0 in
    for a = 0 to r - 1 do
      let c = xs.(a) in
      let step, off =
        match along.(a) with
        | Some (k, w) -> (w.step, j.(k) * w.dilation)
        | None -> (1, 0)
      in
      let extent = ((shape.(a) - 1) * (Iarray.get p.interior a + 1)) + 1 in
      interior.(a) <- step - 1;
      lo.(a) <- off - Iarray.get p.lo a;
      hi.(a) <- extent - lo.(a) - c - ((step - 1) * (c - 1))
    done;
    let windows =
      Iarray.of_list
        (List.map
           (fun a ->
             let step = Iarray.get p.interior a + 1 in
             { M.axis = a; size = 1; step; dilation = 1 })
           strided)
    in
    let lo = Iarray.of_array lo and hi = Iarray.of_array hi in
    let interior = Iarray.of_array interior in
    Padded { x = xt; fill = D.zero dt; pad = { lo; hi; interior; windows } }
  in
  let zero = E.const dt (D.zero dt) in
  (* The sum so far plus taps [first] to [first + count - 1]. *)
  let add sum first count =
    let held = match sum with None -> [] | Some s -> [ Value.Plain s ] in
    let loads =
      Iarray.of_list (held @ List.init count (fun i -> tap (first + i)))
    in
    let ins = Iarray.init (Iarray.length loads) (fun _ -> D.Any dt) in
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
  else apply ~by (Value.Move (Reshape (Iarray.of_array shape), runs None 0))

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
  let order = Iarray.init (r + 1) (fun i -> if i = 0 then r else i - 1) in
  apply ~by (Value.Move (Permute order, z))

(* [pairs] as a complex value of [dt]. *)
let of_pairs (type v s w r d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (dt : (v, s) D.t) (pairs : (w, r, d) Value.t) : (v, s, d) Value.t =
  let r = Prim.rank pairs - 1 in
  let order = Iarray.init (r + 1) (fun i -> if i = r then 0 else i + 1) in
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
      Iarray.init (Array.length s) (fun a ->
          if a = 0 then { M.start = c; count = 1; step = 1 } else whole s.(a))
    in
    let y = apply ~by (Value.Move (Slice ranges, x)) in
    Value.Plain (apply ~by (Value.Move (Broadcast (Iarray.of_array out), y)))
  in
  let views =
    Iarray.init (2 * Array.length loads) (fun i -> part (i mod 2) loads.(i / 2))
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
        let ranges : M.range iarray =
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
        Value.Plain (move (Broadcast (Iarray.of_array out)) y)
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
    let moved = move (Permute (Iarray.of_array order)) pairs in
    let ms = Prim.shape moved in
    let batch = Array.fold_left ( * ) 1 (Array.sub ms 1 (r - 2)) in
    let lines = move (Reshape [| 2; batch; n |]) moved in
    let y =
      if n land (n - 1) = 0 then stockham apply ~by sign lines
      else bluestein apply ~by sign lines
    in
    let back = Array.make r 0 in
    Array.iteri (fun i o -> back.(o) <- i) order;
    let y = move (Reshape (Iarray.of_array ms)) y in
    move (Permute (Iarray.of_array back)) y

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
         ( Slice
             (Iarray.init (Array.length s) (fun i ->
                  if i = a then range else whole s.(i))),
           x ))
  in
  match f with
  | C2c { direction; axes; x } ->
      let dt = Prim.dtype x and s = Prim.shape x in
      if numel s = 0 then zeros apply ~by dt s
      else
        let (Part pt) = Option.get (parts (D.Any dt)) in
        let p = to_pairs apply ~by pt x in
        let p =
          Iarray.fold_left
            (fun p a -> along apply ~by (sign_of direction) a p)
            p axes
        in
        of_pairs apply ~by dt p
  | R2c { dtype; axes; x } ->
      let s = Prim.shape x and l = Iarray.get axes (Iarray.length axes - 1) in
      let out = set s l ((s.(l) / 2) + 1) in
      if numel s = 0 then zeros apply ~by dtype out
      else
        let pt = Prim.dtype x in
        let p =
          apply ~by
            (Value.Assemble
               {
                 dtype = pt;
                 shape = Iarray.of_array (Array.append [| 2 |] s);
                 fill = D.zero pt;
                 pieces =
                   [
                     ( Iarray.of_array
                         (Array.append
                            [| { M.start = 0; count = 1; step = 1 } |]
                            (Array.map whole s)),
                       apply ~by
                         (Value.Move
                            ( Reshape
                                (Iarray.append [| 1 |] (Iarray.of_array s)),
                              x ))
                     );
                   ];
               })
        in
        let p =
          Iarray.fold_left (fun p a -> along apply ~by (-1.) a p) p axes
        in
        of_pairs apply ~by dtype (slice p (l + 1) (whole out.(l)))
  | C2r { dtype; n; axes; x } ->
      let s = Prim.shape x and k = Iarray.length axes - 1 in
      let l = Iarray.get axes k in
      let out = set s l n in
      if numel out = 0 then zeros apply ~by dtype out
      else
        let (Part pt) = Option.get (parts (D.Any (Prim.dtype x))) in
        let p = to_pairs apply ~by pt x in
        let p =
          Iarray.fold_left
            (fun p a -> along apply ~by 1. a p)
            p (Iarray.sub axes ~pos:0 ~len:k)
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
            let shape = Iarray.of_array (Array.append [| 2 |] out) in
            let region from count =
              Iarray.mapi
                (fun i e ->
                  if i = l + 1 then { M.start = from; count; step = 1 }
                  else whole e)
                shape
            in
            apply ~by
              (Value.Assemble
                 {
                   dtype = pt;
                   shape;
                   fill = D.zero pt;
                   pieces = [ (region 0 h, p); (region h mirrored, conj) ];
                 })
        in
        let p = along apply ~by 1. l full in
        let re =
          apply ~by
            (Value.Move
               ( Reshape (Iarray.of_array out),
                 slice p 0 { M.start = 0; count = 1; step = 1 } ))
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
  let x =
    if order = Array.init r Fun.id then x
    else move (M.Permute (Iarray.of_array order)) x
  in
  let units = Array.mapi (fun k e -> if Array.mem k at then e else 1) full in
  let x =
    if Prim.has_shape x units then x
    else move (M.Reshape (Iarray.of_array units)) x
  in
  if units = full then x else move (M.Broadcast (Iarray.of_array full)) x

(* [x] moved by [mv], where it changes [x]. *)
let moved (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by mv
    (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  match (mv : M.t) with
  | Reshape s | Broadcast s when Prim.has_shape x (Iarray.to_array s) -> x
  | _ -> apply ~by (Value.Move (mv, x))

(* [x] times its scale, each element cast to [acc] and multiplied by its
   block's scale read into [acc], rounded once: each axis split into its
   blocks and the elements within one, the scale given a unit axis for the
   latter and broadcast along it. A code [e] reads as float32's bits of
   [2^(e - 127)], [2^-127] a subnormal at [0] and a NaN at [255], then casts
   to [acc]. An operand of no element is cast alone. *)
let scaled (type w r d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (acc : (w, r) D.t) (Value.Any x : d Value.any)
    (scale : d Value.scale option) : d Value.any =
  match scale with
  | None -> Value.Any x
  | Some _ when Array.exists (( = ) 0) (Prim.shape x) ->
      Value.Any (cast_to apply ~by acc x)
  | Some sc ->
      let factor, Value.Any v =
        match sc with
        | Exponent c ->
            let k v = E.const D.Int32 v in
            let e = E.cast D.Int32 (E.input 1 D.Uint8) in
            let bits =
              E.(
                where (e = k 0l) (k 0x00400000l)
                  (where (e = k 255l) (k 0x7FC00000l) (e * k 0x00800000l)))
            in
            (E.cast acc (E.bitcast D.Float32 bits), Value.Any c)
        | Value v -> (E.cast acc (E.input 1 (Prim.dtype v)), Value.Any v)
      in
      let s = Prim.shape x and ss = Prim.shape v in
      let r = Array.length s in
      let split =
        Array.init (2 * r) (fun i ->
            if i mod 2 = 0 then ss.(i / 2) else s.(i / 2) / ss.(i / 2))
      in
      let units = Array.mapi (fun i e -> if i mod 2 = 0 then e else 1) split in
      let x' = moved apply ~by (Reshape (Iarray.of_array split)) x in
      let v' =
        moved apply ~by
          (Broadcast (Iarray.of_array split))
          (moved apply ~by (Reshape (Iarray.of_array units)) v)
      in
      let xt = Prim.dtype x and vt = Prim.dtype v in
      let prog =
        P.of_exprs
          ~ins:[| D.Any xt; D.Any vt |]
          [| E.any E.(cast acc (input 0 xt) * factor) |]
      in
      let y = map1 apply ~by split prog acc [| Plain x'; Plain v' |] in
      Value.Any (moved apply ~by (Reshape (Iarray.of_array s)) y)

(* The group of each index [r] of an axis of [len] by [offsets], of the
   batch extents then [g + 1] ({!Nx_kernel.type-segments}), at the batch
   extents then [len]: the largest [e < g] with [offsets.(e) <= r], a
   [Max] over the groups, where [offsets.(0) <= r < offsets.(g)], and [-1]
   elsewhere, as for no group. *)
let group_ids (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (offsets : d Value.index) len : d Value.index =
  let os = Prim.shape offsets in
  let nb = Array.length os - 1 in
  let g = os.(nb) - 1 and lead = Array.sub os 0 nb in
  let at start count =
    let r = Array.map (fun d -> { M.start = 0; count = d; step = 1 }) os in
    r.(nb) <- { M.start; count; step = 1 };
    apply ~by (Value.Move (Slice (Iarray.of_array r), offsets))
  in
  let lifted shape x =
    let units = Array.append lead (Array.make (Array.length shape - nb) 1) in
    units.(Array.length shape - 1) <- (Prim.shape x).(nb);
    moved apply ~by
      (Broadcast (Iarray.of_array shape))
      (moved apply ~by (Reshape (Iarray.of_array units)) x)
  in
  let o i = E.input i D.Int64 and k v = E.const D.Int64 v in
  let rows = Array.append lead [| len |] in
  let none = P.of_exprs ~ins:[||] [| E.any (k (-1L)) |] in
  if g = 0 then map1 apply ~by rows none D.Int64 [||]
  else
    let each = Array.concat [ lead; [| len; g |] ] in
    let r = E.coord nb and e = E.coord (nb + 1) in
    let below =
      P.of_exprs ~ins:[| D.Any D.Int64 |]
        [| E.(any (where (o 0 <= r) e (k (-1L)))) |]
    in
    let id =
      reduce1 apply ~by each [| nb + 1 |] Max below D.Int64
        [| Plain (lifted each (at 0 g)) |]
    in
    let edge i = Value.Plain (lifted rows (at i 1)) in
    let member =
      P.of_exprs
        ~ins:[| D.Any D.Int64; D.Any D.Int64; D.Any D.Int64 |]
        [| E.(any (where (o 1 <= r && r < o 2) (o 0) (k (-1L)))) |]
    in
    map1 apply ~by rows member D.Int64 [| Plain id; edge 0; edge g |]

(* Whether each group [e < g] of [ids] ({!group_ids}) holds an index: [1]
   or [0] at [g] then [ids]' batch extents. *)
let held (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (ids : d Value.index) g : d Value.index =
  let s = Prim.shape ids in
  let nb = Array.length s - 1 in
  let each = Array.append [| g |] s in
  let v = spread apply ~by ids (Array.init (nb + 1) (fun i -> i + 1)) each in
  let k x = E.const D.Int64 x in
  let mine =
    P.of_exprs ~ins:[| D.Any D.Int64 |]
      [| E.(any (where (input 0 D.Int64 = coord 0) (k 1L) (k 0L))) |]
  in
  reduce1 apply ~by each [| nb + 1 |] Max mine D.Int64 [| Plain v |]

(* The contraction of [a] and [b], each cast to [acc], by the pairs [batch]
   and [contracting], from [init]: the products in [acc] at every index of
   the result's axes then the contracted ones, each rounded, a sum over the
   contracted ones, and [init] added. With [groups] [(ids, pair, g)], the
   result gains a leading axis of [g], and slice [e] keeps the products
   whose index along [pair] has the group id [e] in [ids], at [batch]'s
   extents then that pair's, and [+0] for the others; a group of no index
   is [init] alone, whose [-0] the [+0] of its empty sum would turn. *)
let summed (type w r d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    ~batch ~contracting ?groups (acc : (w, r) D.t) (Value.Any a : d Value.any)
    (Value.Any b : d Value.any) (init : (w, r, d) Value.t option) :
    (w, r, d) Value.t =
  let sa = Prim.shape a and sb = Prim.shape b in
  let free s side =
    List.filter
      (fun i ->
        not
          (Iarray.exists (fun p -> side p = i) batch
          || Iarray.exists (fun p -> side p = i) contracting))
      (List.init (Array.length s) Fun.id)
  in
  let fa = free sa fst and fb = free sb snd in
  let nb = Iarray.length batch and nfa = List.length fa in
  let lead = match groups with Some (_, _, g) -> [| g |] | None -> [||] in
  let nl = Array.length lead in
  let rshape =
    Array.concat
      [
        lead;
        Iarray.to_array (Iarray.map (fun (i, _) -> sa.(i)) batch);
        Array.of_list (List.map (fun i -> sa.(i)) fa);
        Array.of_list (List.map (fun i -> sb.(i)) fb);
      ]
  in
  let nr = Array.length rshape in
  let full =
    Array.append rshape
      (Iarray.to_array (Iarray.map (fun (i, _) -> sa.(i)) contracting))
  in
  let positions s side frees offset =
    let at = Array.make (Array.length s) 0 in
    Iarray.iteri (fun k p -> at.(side p) <- nl + k) batch;
    List.iteri (fun j i -> at.(i) <- nl + nb + offset + j) frees;
    Iarray.iteri (fun c p -> at.(side p) <- nr + c) contracting;
    at
  in
  let va = spread apply ~by a (positions sa fst fa 0) full in
  let vb = spread apply ~by b (positions sb snd fb nfa) full in
  let at = Prim.dtype a and bt = Prim.dtype b in
  let product = E.(cast acc (input 0 at) * cast acc (input 1 bt)) in
  let prog, (loads : d Value.load iarray) =
    match groups with
    | None ->
        ( P.of_exprs ~ins:[| D.Any at; D.Any bt |] [| E.any product |],
          [| Value.Plain va; Value.Plain vb |] )
    | Some (ids, pair, _) ->
        (* Selecting the products, where a mask would multiply them, keeps
           the NaNs and infinities of other groups out. *)
        let into =
          Array.init (nb + 1) (fun i -> if i < nb then nl + i else nr + pair)
        in
        let vi = spread apply ~by ids into full in
        let zero = E.const acc (D.zero acc) in
        let kept = E.(where (input 2 D.Int64 = coord 0) product zero) in
        ( P.of_exprs
            ~ins:[| D.Any at; D.Any bt; D.Any D.Int64 |]
            [| E.any kept |],
          [| Value.Plain va; Value.Plain vb; Value.Plain vi |] )
  in
  let sum =
    if Iarray.length contracting = 0 then map1 apply ~by full prog acc loads
    else
      reduce1 apply ~by full
        (Iarray.init (Iarray.length contracting) (fun c -> nr + c))
        Sum prog acc loads
  in
  match (init, groups) with
  | None, _ -> sum
  | Some i, None -> fst (apply ~by (Prim.op2 ~by (Binary Add) acc sum i))
  | Some i, Some (ids, _, g) ->
      let has = held apply ~by ids g in
      let vh = spread apply ~by has (Array.init (nb + 1) Fun.id) rshape in
      let s = E.input 1 acc and i' = E.input 2 acc in
      let prog =
        P.of_exprs
          ~ins:[| D.Any D.Int64; D.Any acc; D.Any acc |]
          [| E.(any (where (input 0 D.Int64 = const D.Int64 1L) (s + i') i')) |]
      in
      map1 apply ~by rshape prog acc [| Plain vh; Plain sum; Plain i |]

(* [Rows]: each row's group of [b] gathered along [b_axis] into an axis of
   [a]'s rows, contracted with it as a batch pair, its axis moved to
   [a_axis]'s place; a row in no group is [init] or [+0]. *)
let rows (type w r d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    ~batch ~contracting ~a_axis ~b_axis (acc : (w, r) D.t)
    (Value.Any a : d Value.any) (Value.Any b : d Value.any) offsets
    (init : (w, r, d) Value.t option) : (w, r, d) Value.t =
  let nb = Iarray.length batch and len = Prim.dim a a_axis in
  let ids = group_ids apply ~by offsets len in
  let into = Array.copy (Prim.shape b) in
  into.(b_axis) <- len;
  let at =
    Array.init (nb + 1) (fun i ->
        if i < nb then snd (Iarray.get batch i) else b_axis)
  in
  let idx = spread apply ~by ids at into in
  let b = apply ~by (Value.Gather { axis = b_axis; idx; x = b }) in
  let batch' = Iarray.append batch [| (a_axis, b_axis) |] in
  let y =
    summed apply ~by ~batch:batch' ~contracting acc (Any a) (Any b) None
  in
  (* [a_axis]'s place among [a]'s free axes, after the batch. *)
  let named ps i = Iarray.exists (fun p -> fst p = i) ps in
  let q = ref 0 in
  for i = 0 to a_axis - 1 do
    if not (named batch i || named contracting i) then incr q
  done;
  let q = !q in
  let perm =
    Array.init (Prim.rank y) (fun t ->
        if t < nb || t > nb + q then t else if t = nb + q then nb else t + 1)
  in
  let y =
    if q = 0 then y
    else apply ~by (Value.Move (Permute (Iarray.of_array perm), y))
  in
  let shape = Prim.shape y in
  let rows = Array.init (nb + 1) (fun i -> if i < nb then i else nb + q) in
  let vi = spread apply ~by ids rows shape in
  let member = E.(input 0 D.Int64 >= const D.Int64 0L) in
  let sum = E.input 1 acc in
  let prog, (loads : d Value.load iarray) =
    match init with
    | None ->
        ( P.of_exprs ~ins:[| D.Any D.Int64; D.Any acc |]
            [| E.(any (where member sum (const acc (D.zero acc)))) |],
          [| Value.Plain vi; Value.Plain y |] )
    | Some i ->
        let i' = E.input 2 acc in
        ( P.of_exprs ~ins:[| D.Any D.Int64; D.Any acc; D.Any acc |]
            [| E.(any (where member (sum + i') i')) |],
          [| Value.Plain vi; Value.Plain y; Value.Plain i |] )
  in
  map1 apply ~by shape prog acc loads

(* The contraction of [a] and [b] by the pairs [batch] and [contracting] from
   [init], computed in [acc] into [out], as other operations. A result of
   another dtype than its accumulator is the contraction into the
   accumulator, which kernels compute in [order], rounded once. Any other
   scales each scaled operand first ({!scaled}), then sums its products
   ({!summed}): by group of the contracted pair for [Contracting], and over
   each row's group of [b] for [Rows] ({!rows}). A contraction of no
   products is [init] alone, or [+0]. That sum is a reduction's,
   in the reduction's order whatever [order]: no kernel's bits, and the same
   for any window of the result only where that order depends on the count
   of terms alone, as nx.cpu's reduction order does. *)
let contract (type v s a b c e d) (apply : 'q. by:string -> 'q Value.prim -> 'q)
    ~by ~order ~batch ~contracting ~segments ~acc:(D.Any acc) (out : (v, s) D.t)
    (a : (a, b, d) Value.t) (b : (c, e, d) Value.t)
    (init : (v, s, d) Value.t option) a_scale b_scale : (v, s, d) Value.t =
  let init = Option.map (cast_to apply ~by acc) init in
  if not (D.equal acc out) then
    let y =
      apply ~by
        (Value.Contract
           {
             order;
             batch;
             contracting;
             acc = D.Any acc;
             out = acc;
             a;
             b;
             init;
             a_scale;
             b_scale;
             segments;
           })
    in
    cast_to apply ~by out y
  else
    let depth =
      Iarray.fold_left (fun n (i, _) -> n * Prim.dim a i) 1 contracting
    in
    let a = scaled apply ~by acc (Any a) a_scale in
    let b = scaled apply ~by acc (Any b) b_scale in
    let y =
      match (init, (segments : d Value.index Nx_kernel.segments option)) with
      | Some i, _ when depth = 0 -> i
      | _, None -> summed apply ~by ~batch ~contracting acc a b init
      | _, Some (Rows { offsets; a_axis; b_axis }) ->
          rows apply ~by ~batch ~contracting ~a_axis ~b_axis acc a b offsets
            init
      | _, Some (Contracting { offsets; pair }) ->
          let (Value.Any x) = a in
          let p = Iarray.get contracting pair in
          let ids = group_ids apply ~by offsets (Prim.dim x (fst p)) in
          let g = Prim.dim offsets (Prim.rank offsets - 1) - 1 in
          let groups = (ids, pair, g) in
          summed apply ~by ~batch ~contracting ~groups acc a b init
    in
    cast_to apply ~by out y

(* Gathers and scatters *)

(* The carrier of [x]'s dtype where it is sub-byte, which a gather or a
   scatter computes in: [None] for a dtype of a byte or more. *)
let sub_byte (type v s d) (x : (v, s, d) Value.t) =
  let dt = Prim.dtype x in
  if D.bits dt >= 8 then None else Some (Nx_kernel.carrier (D.Any dt))

(* A sub-byte gather at its accumulator, cast back once. *)
let gather (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by axis
    idx (x : (v, s, d) Value.t) : (v, s, d) Value.t option =
  match sub_byte x with
  | None -> None
  | Some (D.Any w) ->
      let x' = cast_to apply ~by w x in
      let y = apply ~by (Value.Gather { axis; idx; x = x' }) in
      Some (cast_to apply ~by (Prim.dtype x) y)

(* A scatter whose targets may repeat as a loop of one unique scatter per
   position along [axis], in order: a position's updates differ off [axis], so
   their targets do, and updates to one target land in C order. A sum
   associates left to right. A trip gathers its position of [idx] and
   [updates] at its index, a 0-d int64 value, broadcast over a position's
   slice. *)
let repeated (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    ~(loop :
       int ->
       (d Value.index -> d Value.any list -> d Value.any list) ->
       d Value.any list ->
       d Value.any list) combine axis idx (updates : (v, s, d) Value.t)
    (into : (v, s, d) Value.t) : (v, s, d) Value.t =
  let s = Prim.shape updates in
  let r = Array.length s in
  let dt = Prim.dtype into in
  (* No position leaves [into], copied: no result is an operand. *)
  if s.(axis) = 0 then apply ~by (Value.Copy into)
  else
    let ones = Iarray.init r (fun _ -> 1) in
    let slice = Iarray.init r (fun a -> if a = axis then 1 else s.(a)) in
    let step k carry =
      (* The index, broadcast once per trip over a position's slice. *)
      let k = apply ~by (Value.Move (Reshape ones, k)) in
      let pos = apply ~by (Value.Move (Broadcast slice, k)) in
      let at (type w q) (x : (w, q, d) Value.t) : (w, q, d) Value.t =
        apply ~by (Value.Gather { axis; idx = pos; x })
      in
      match carry with
      | [ acc ] ->
          let into = Prim.expect dt acc in
          let idx = at idx and updates = at updates in
          [
            Value.Any
              (apply ~by
                 (Value.Scatter
                    { combine; unique = true; axis; idx; updates; into }));
          ]
      | _ -> invalid_arg "Expand: a scatter's loop carries its target alone"
    in
    match loop s.(axis) step [ Value.Any into ] with
    | [ y ] -> Prim.expect dt y
    | _ -> invalid_arg "Expand: a scatter's loop gives its target alone"

(* A narrow float's [Add] of repeated targets as {!repeated} in its accumulator,
   rounded once; a target no update reaches, found by a [Set] of [true] at every
   target, keeps [into]'s bits. *)
let rounded_once (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    ~loop (w : ('w, 'r) D.t) axis idx (updates : (v, s, d) Value.t)
    (into : (v, s, d) Value.t) : (v, s, d) Value.t =
  let dt = Prim.dtype into in
  let wide x = cast_to apply ~by w x in
  let sum =
    repeated apply ~by ~loop Nx_kernel.Add axis idx (wide updates)
      (wide into)
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
    repeated apply ~by ~loop Nx_kernel.Set axis idx
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
    ~loop combine ~unique axis idx (updates : (v, s, d) Value.t)
    (into : (v, s, d) Value.t) : (v, s, d) Value.t option =
  match sub_byte into with
  | None when unique -> None
  | None -> (
      match (combine, Nx_kernel.carrier (D.Any (Prim.dtype into))) with
      | Nx_kernel.Add, D.Any w
        when not (same (D.Any w) (D.Any (Prim.dtype into))) ->
          Some (rounded_once apply ~by ~loop w axis idx updates into)
      | _ -> Some (repeated apply ~by ~loop combine axis idx updates into))
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
    let loads = plains loads in
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
  let whole s =
    Iarray.init (Array.length s) (fun a ->
        { M.start = 0; count = s.(a); step = 1 })
  in
  let x =
    if wide.(axis) = n then packed
    else
      apply ~by
        (Value.Assemble
           {
             dtype = D.Int64;
             shape = Iarray.of_array wide;
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
         result forms where the sort expands; gather only the kept values. *)
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
            apply ~by (Value.Move (Slice (Iarray.of_array rs), y))
      in
      Some (first values, first positions)

(* Matrices

   A routine computes on [b; m; n], its matrices' batch flattened into one axis,
   through maps, gathers, reductions, sorts and contractions. Its steps are a
   loop whose every trip applies the same operations at full size: where a step
   of the reference touches a part that shrinks, such as the rows below a pivot,
   a mask of the coordinates against the trip's index keeps the rest. *)

(* A loop of [n] trips from [init]: [loop n step init]. *)
type 'd loop =
  int ->
  ('d Value.index -> 'd Value.any list -> 'd Value.any list) ->
  'd Value.any list ->
  'd Value.any list

(* The parts of a float or complex dtype's elements: the float format [re] of
   its real parts and the functions between them. *)
type ('v, 's) field =
  | Field : {
      re : (float, 'r) D.t;
      real : ('v, 's) E.t -> (float, 'r) E.t;
      imag : ('v, 's) E.t -> (float, 'r) E.t;
      conj : ('v, 's) E.t -> ('v, 's) E.t;
      of_real : (float, 'r) E.t -> ('v, 's) E.t;
      nan : 'v;
    }
      -> ('v, 's) field

(* On complex numbers the real part is a cast; the imaginary part the real part
   of [-i z], [b] of [a + bi], exact for finite parts. *)
let field (type v s) (dt : (v, s) D.t) : (v, s) field =
  let reals (type r) (re : (float, r) D.t) : (float, r) field =
    Field
      {
        re;
        real = Fun.id;
        imag = (fun _ -> E.const re 0.);
        conj = Fun.id;
        of_real = Fun.id;
        nan = Float.nan;
      }
  in
  let complex (type c r) (dt : (Complex.t, c) D.t) (re : (float, r) D.t) :
      (Complex.t, c) field =
    let real z = E.cast re z and of_real x = E.cast dt x in
    let imag z = real E.(z * const dt { Complex.re = 0.; im = -1. }) in
    let conj z =
      E.(of_real (real z) - (const dt Complex.i * of_real (imag z)))
    in
    let nan = { Complex.re = Float.nan; im = Float.nan } in
    Field { re; real; imag; conj; of_real; nan }
  in
  match dt with
  | D.Float32 -> reals D.Float32
  | D.Float64 -> reals D.Float64
  | D.Complex64 -> complex D.Complex64 D.Float32
  | D.Complex128 -> complex D.Complex128 D.Float64
  | _ -> invalid_arg (Printf.sprintf "Expand: no routine at %s" (D.name dt))

(* [p], the unit roundoff of [dt]'s parts being [2^-p]. *)
let digits (type v s) (dt : (v, s) D.t) =
  match dt with D.Float32 | D.Complex64 -> 24 | _ -> 53

let i64 v = E.const D.Int64 (Int64.of_int v)
let not_ e = E.(e = const D.Bool false)
let rows = E.coord 1
let cols = E.coord 2

let move (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by mv x =
  apply ~by (Value.Move (mv, x))

let reshape (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by s x =
  move apply ~by (M.Reshape (Iarray.of_array s)) x

(* [x] broadcast to [shape], unit axes put before its own. *)
let fit (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by shape
    (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  let r = Array.length shape and s = Prim.shape x in
  let x =
    if Array.length s = r then x
    else
      reshape apply ~by (Array.append (Array.make (r - Array.length s) 1) s) x
  in
  if Prim.has_shape x shape then x
  else move apply ~by (M.Broadcast (Iarray.of_array shape)) x

(* The map of [e] into [dt] at every index of [shape], [e] reading [xs]'s
   element [i], fit to [shape], as [E.input i]. *)
let mapx (type w r d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by shape
    (dt : (w, r) D.t) (xs : d Value.any list) (e : (w, r) E.t) :
    (w, r, d) Value.t =
  let load (Value.Any x) = Value.Plain (fit apply ~by shape x) in
  let ins = List.map (fun (Value.Any x) -> D.Any (Prim.dtype x)) xs in
  map1 apply ~by shape
    (P.of_exprs ~ins:(Iarray.of_list ins) [| E.any e |])
    dt
    (Iarray.of_list (List.map load xs))

(* [x]'s matrices as [b; m; n], and its batch's shape. *)
let matrices (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (x : (v, s, d) Value.t) : int array * (v, s, d) Value.t =
  let s = Prim.shape x in
  let r = Array.length s in
  let lead = Array.sub s 0 (r - 2) in
  let b = Array.fold_left ( * ) 1 lead in
  (lead, reshape apply ~by [| b; s.(r - 2); s.(r - 1) |] x)

(* [x], of a flattened batch, with the batch's shape [lead]. *)
let batched (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by lead x =
  let s = Prim.shape x in
  reshape apply ~by (Array.append lead (Array.sub s 1 (Array.length s - 1))) x

(* Row [i] of [x], [b; m; n], as [b; 1; n], and column [j] as [b; m; 1], for
   indices [i] and [j] that fit those shapes. *)
let row (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by i x =
  let s = Prim.shape x in
  let idx = fit apply ~by [| s.(0); 1; s.(2) |] i in
  apply ~by (Value.Gather { axis = 1; idx; x })

let column (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by j x =
  let s = Prim.shape x in
  let idx = fit apply ~by [| s.(0); s.(1); 1 |] j in
  apply ~by (Value.Gather { axis = 2; idx; x })

(* The contraction of [x] and [y] per matrix of the batch along the pair of axes
   [contracting]. *)
let product (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    contracting (x : (v, s, d) Value.t) (y : (v, s, d) Value.t) :
    (v, s, d) Value.t =
  let batch : (int * int) iarray = [| (0, 0) |]
  and contracting : (int * int) iarray = [| contracting |] in
  let dt = Prim.dtype x in
  let order = Prim.contract_order ~batch ~contracting x y in
  apply ~by
    (Value.Contract
       {
         order;
         batch;
         contracting;
         acc = D.Any dt;
         out = dt;
         a = x;
         b = y;
         init = None;
         a_scale = None;
         b_scale = None;
         segments = None;
       })

let conjugate (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  let dt = Prim.dtype x in
  let (Field f) = field dt in
  match D.kind dt with
  | D.Complex ->
      mapx apply ~by (Prim.shape x) dt [ Value.Any x ] (f.conj (E.input 0 dt))
  | D.Float | D.Signed | D.Unsigned | D.Boolean -> x

(* [x]'s matrices transposed, [b; n; m]: a view. *)
let transposed (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by x =
  move apply ~by (M.Permute [| 0; 2; 1 |]) x

(* [x] reduced by [m] along [axis], kept as a unit axis. *)
let along (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by m axis
    (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  let s = Array.copy (Prim.shape x) in
  let y = reduce_plain apply ~by [| axis |] m x in
  s.(axis) <- 1;
  reshape apply ~by s y

(* How many elements of each matrix of [xs], fit to [b; m; n], are [true] by
   [e], as [b; 1; 1]. *)
let count_of (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by shape
    (xs : d Value.any list) e : d Value.index =
  let flag = mapx apply ~by shape D.Int64 xs E.(where e (i64 1) (i64 0)) in
  let count = reduce_plain apply ~by [| 1; 2 |] Sum flag in
  reshape apply ~by [| shape.(0); 1; 1 |] count

(* [x] with each matrix whose [count] is positive NaN. *)
let failing (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (count : d Value.index) (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  let dt = Prim.dtype x and s = Prim.shape x in
  let (Field f) = field dt in
  let units = Array.mapi (fun a e -> if a = 0 then e else 1) s in
  mapx apply ~by s dt
    [ Value.Any x; Value.Any (reshape apply ~by units count) ]
    E.(where (input 1 D.Int64 > i64 0) (const dt f.nan) (input 0 dt))

(* Whether each matrix of [x], [b; m; n], holds an infinity or a NaN. *)
let not_finite (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (x : (v, s, d) Value.t) =
  let dt = Prim.dtype x in
  let (Field f) = field dt in
  let z = E.input 0 dt in
  count_of apply ~by (Prim.shape x) [ Value.Any x ]
    (not_ E.(finite (f.real z) && finite (f.imag z)))

(* The matrix of [b; n; n] whose elements are [one] on the diagonal and [zero]
   elsewhere. *)
let identity (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (dt : (v, s) D.t) b n : (v, s, d) Value.t =
  mapx apply ~by [| b; n; n |] dt []
    E.(where (rows = cols) (const dt (D.one dt)) (const dt (D.zero dt)))

(* Cholesky's columns, dpotf2's: at trip [j], [v = a_j - L l_jᴴ], [l_j] row [j]
   of [L] so far, then [L_jj = √(re v_j)] and [L_ij = v_i / L_jj] below it. The
   lower triangle and the diagonal's real parts alone are read. A matrix whose
   pivot is not positive, or NaN, is NaN. *)
let cholesky (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    ~(loop : d loop) (triangle : Nx_kernel.triangle) (a : (v, s, d) Value.t) :
    (v, s, d) Value.t =
  let dt = Prim.dtype a in
  let (Field f) = field dt in
  let lead, a = matrices apply ~by a in
  let shape = Prim.shape a in
  let b = shape.(0) and n = shape.(1) in
  let x i = E.input i dt and k = E.input 3 D.Int64 in
  let zero = E.const dt (D.zero dt) in
  let step j = function
    | [ l ] ->
        let l = Prim.expect dt l in
        let s =
          product apply ~by (2, 2) l (conjugate apply ~by (row apply ~by j l))
        in
        let v =
          mapx apply ~by [| b; n; 1 |] dt
            [ Value.Any (column apply ~by j a); Value.Any s ]
            E.(x 0 - x 1)
        in
        let d = f.of_real (E.sqrt (f.real (x 2))) in
        let next = E.(where (rows > k) (x 1 / d) (where (rows = k) d zero)) in
        [
          Value.Any
            (mapx apply ~by shape dt
               [
                 Value.Any l;
                 Value.Any v;
                 Value.Any (row apply ~by j v);
                 Value.Any j;
               ]
               E.(where (cols = k) next (x 0)));
        ]
    | _ -> invalid_arg "Expand: Cholesky's loop carries its factor alone"
  in
  let l =
    match loop n step [ Value.Any (zeros apply ~by dt shape) ] with
    | [ l ] -> Prim.expect dt l
    | _ -> invalid_arg "Expand: Cholesky's loop gives its factor alone"
  in
  let bad =
    count_of apply ~by shape [ Value.Any l ]
      E.(rows = cols && not_ (f.real (x 0) > const f.re 0.))
  in
  let l =
    match triangle with
    | Lower -> l
    | Upper ->
        conjugate apply ~by (apply ~by (Value.Copy (transposed apply ~by l)))
  in
  batched apply ~by lead (failing apply ~by bad l)

(* Substitution, dtrsm's, by rows: [x] of [T x = b], at trip [i] [x_i = (b_i -
   t_i x) / t_ii], [x] zero below row [i] so far. [T] is [a]'s triangle, or
   [aᴴ]'s, with ones on its diagonal where [unit_diagonal]; an upper one is
   solved as the lower [J T J], [J] the reversal, of [J b]. A zero on a diagonal
   it reads makes [x] NaN. *)
let solve (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    ~(loop : d loop) (triangle : Nx_kernel.triangle) ~adjoint ~unit_diagonal
    (a : (v, s, d) Value.t) (rhs : (v, s, d) Value.t) : (v, s, d) Value.t =
  let dt = Prim.dtype a in
  let (Field f) = field dt in
  let _, a = matrices apply ~by a and lead, rhs = matrices apply ~by rhs in
  let sb = Prim.shape rhs in
  let b = sb.(0) and n = sb.(1) in
  let lower = triangle = Lower <> adjoint in
  (* [x] reversed along its axes from 1 to [last]. *)
  let reversed last x =
    let ranges =
      Iarray.init 3 (fun i ->
          let e = Prim.dim x i in
          if i = 0 || i > last then whole e
          else { M.start = e - 1; count = e; step = -1 })
    in
    move apply ~by (M.Slice ranges) x
  in
  let a = if adjoint then transposed apply ~by a else a in
  let a, rhs = if lower then (a, rhs) else (reversed 2 a, reversed 1 rhs) in
  let x i = E.input i dt and k i = E.input i D.Int64 in
  let zero = E.const dt (D.zero dt) and one = E.const dt (D.one dt) in
  let t =
    let v = if adjoint then f.conj (x 0) else x 0 in
    let diagonal = if unit_diagonal then one else v in
    mapx apply ~by [| b; n; n |] dt [ Value.Any a ]
      E.(where (rows = cols) diagonal (where (rows > cols) v zero))
  in
  let step i = function
    | [ y ] ->
        let y = Prim.expect dt y in
        let ti = row apply ~by i t in
        let s = product apply ~by (2, 1) ti y in
        let tii = column apply ~by i ti in
        [
          Value.Any
            (mapx apply ~by sb dt
               [
                 Value.Any y;
                 Value.Any (row apply ~by i rhs);
                 Value.Any s;
                 Value.Any tii;
                 Value.Any i;
               ]
               E.(where (rows = k 4) ((x 1 - x 2) / x 3) (x 0)));
        ]
    | _ -> invalid_arg "Expand: a solve's loop carries its solution alone"
  in
  let y =
    match loop n step [ Value.Any (zeros apply ~by dt sb) ] with
    | [ y ] -> Prim.expect dt y
    | _ -> invalid_arg "Expand: a solve's loop gives its solution alone"
  in
  let bad =
    count_of apply ~by [| b; n; n |] [ Value.Any t ]
      E.(rows = cols && x 0 = zero)
  in
  let y = if lower then y else reversed 1 y in
  batched apply ~by lead (failing apply ~by bad y)

(* Gaussian elimination with partial pivoting, dgetf2's right-looking steps: at
   trip [j], the pivot {!Nx_kernel.S.lu} states, rows [j] and [p] interchanged
   in [a] and in [perm], column [j] below the diagonal divided by a pivot that
   is not zero, and the trailing rows and columns less its product with row
   [j]. *)
let lu (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    ~(loop : d loop) (a : (v, s, d) Value.t) :
    (v, s, d) Value.t * d Value.index * d Value.index =
  let dt = Prim.dtype a in
  let (Field f) = field dt in
  let lead, a = matrices apply ~by a in
  let shape = Prim.shape a in
  let b = shape.(0) and m = shape.(1) and n = shape.(2) in
  let kk = Stdlib.min m n in
  let x i = E.input i dt
  and k i = E.input i D.Int64
  and re i = E.input i f.re in
  let flat x = reshape apply ~by [| b; 1 |] x in
  (* Row [j] and row [p] interchanged: the row each row reads. *)
  let swapped j p r = E.(where (r = j) p (where (r = p) j r)) in
  let step j = function
    | [ w; perm; pivots ] ->
        let w = Prim.expect dt w in
        let perm = Prim.expect D.Int64 perm
        and pivots = Prim.expect D.Int64 pivots in
        let c = column apply ~by j w in
        (* The magnitudes the scan compares, below row [j], [-1] elsewhere: a
           NaN at row [j] is greatest, so the scan stays there, and below it is
           never taken. *)
        let mags =
          let mag = E.(abs (f.real (x 0)) + abs (f.imag (x 0))) in
          let fl v = E.const f.re v in
          mapx apply ~by [| b; m; 1 |] f.re
            [ Value.Any c; Value.Any j ]
            E.(
              where
                (rows < k 1)
                (fl (-1.))
                (where (mag <> mag)
                   (where (rows = k 1) (fl Float.infinity) (fl (-1.)))
                   mag))
        in
        let top = along apply ~by Max 1 mags in
        let p =
          along apply ~by Min 1
            (mapx apply ~by [| b; m; 1 |] D.Int64
               [ Value.Any mags; Value.Any top ]
               E.(where (re 0 = re 1) rows (i64 m)))
        in
        let w =
          let idx =
            mapx apply ~by shape D.Int64
              [ Value.Any j; Value.Any p ]
              (swapped (k 0) (k 1) rows)
          in
          apply ~by (Value.Gather { axis = 1; idx; x = w })
        in
        let p2 = flat p in
        let perm =
          let r = E.coord 1 in
          let idx =
            mapx apply ~by [| b; m |] D.Int64
              [ Value.Any j; Value.Any p2 ]
              (swapped (k 0) (k 1) r)
          in
          apply ~by (Value.Gather { axis = 1; idx; x = perm })
        in
        let pivots =
          mapx apply ~by [| b; kk |] D.Int64
            [ Value.Any pivots; Value.Any j; Value.Any p2 ]
            E.(where (coord 1 = k 1) (k 2) (k 0))
        in
        let c = column apply ~by j w and r = row apply ~by j w in
        let d = column apply ~by j r in
        let zero = E.const dt (D.zero dt) in
        let l = E.(where (x 3 = zero) (x 1) (x 1 / x 3)) in
        let w =
          mapx apply ~by shape dt
            [ Value.Any w; Value.Any c; Value.Any r; Value.Any d; Value.Any j ]
            E.(
              where
                (rows > k 4)
                (where
                   (cols > k 4)
                   (x 0 - (l * x 2))
                   (where (cols = k 4) l (x 0)))
                (x 0))
        in
        [ Value.Any w; Value.Any perm; Value.Any pivots ]
    | _ -> invalid_arg "Expand: LU's loop carries its factors and positions"
  in
  let init =
    [
      Value.Any (apply ~by (Value.Copy a));
      Value.Any (mapx apply ~by [| b; m |] D.Int64 [] (E.coord 1));
      Value.Any (zeros apply ~by D.Int64 [| b; kk |]);
    ]
  in
  match loop kk step init with
  | [ w; perm; pivots ] ->
      let positions x = batched apply ~by lead (Prim.expect D.Int64 x) in
      ( batched apply ~by lead (Prim.expect dt w),
        positions pivots,
        positions perm )
  | _ -> invalid_arg "Expand: LU's loop gives its factors and positions"

(* The greatest magnitude of [z]'s parts, which a scaled sum of squares divides
   by. *)
let largest real imag z = E.(maximum (abs (real z)) (abs (imag z)))

(* A power of two in [t]'s binade, [t] of [dt]'s parts, a subnormal one
   included, [0] for [t = 0]: dividing by it and multiplying back are exact
   where the results are normal. *)
let binade (type v s r) (dt : (v, s) D.t) (t : (float, r) E.t) =
  let re = E.dtype t in
  let fl v = E.const re v in
  let lo, hi = if digits dt = 24 then (-149., 127.) else (-1074., 1023.) in
  let e = E.(minimum (maximum (floor (log2 t)) (fl lo)) (fl hi)) in
  E.(where (t > fl 0.) (exp2 e) (fl 0.))

(* Each matrix of [x], [b; m; n], times the power of two nearest the reciprocal
   of its greatest part's magnitude, which is exact, and that power, [b; 1; 1]
   of [x]'s dtype: the squares of its elements then neither overflow nor
   underflow where they matter, as dsyev scales its matrix "to allowable range".
   A matrix of zeros, or with an infinity or a NaN, keeps a power of [1]. *)
let scaled (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (x : (v, s, d) Value.t) : (v, s, d) Value.t * (v, s, d) Value.t =
  let dt = Prim.dtype x and shape = Prim.shape x in
  let (Field f) = field dt in
  let b = shape.(0) in
  let one = E.const dt (D.one dt) in
  if Array.exists (( = ) 0) shape then
    (x, mapx apply ~by [| b; 1; 1 |] dt [] one)
  else
    let mags =
      mapx apply ~by shape f.re [ Value.Any x ]
        (largest f.real f.imag (E.input 0 dt))
    in
    let top =
      reshape apply ~by [| b; 1; 1 |]
        (reduce_plain apply ~by [| 1; 2 |] Max mags)
    in
    let fl v = E.const f.re v in
    let lo, hi = if digits dt = 24 then (-126., 127.) else (-1022., 1023.) in
    let t = E.input 0 f.re in
    let e = E.(minimum (maximum (neg (round (log2 t))) (fl lo)) (fl hi)) in
    let usable = E.(t > fl 0. && t < fl Float.infinity) in
    let power =
      mapx apply ~by [| b; 1; 1 |] dt [ Value.Any top ]
        (f.of_real E.(where usable (exp2 e) (fl 1.)))
    in
    let y i = E.input i dt in
    ( mapx apply ~by shape dt [ Value.Any x; Value.Any power ] E.(y 0 * y 1),
      power )

(* Householder QR, dgeqr2's reflectors as zlarfg makes them: at trip [j], [x]
   column [j] of [R] from row [j], [α = x_0], [β = -sign(re α) ‖x‖₂], [τ = (β -
   α) / β] and [v = (1, x_1.. / (α - β))]; [τ = 0] where [x_1..] is zero and [α]
   real. [‖x‖₂] is [μ √(Σ |x_i / μ|²)], [μ] a power of two near the greatest
   part's magnitude, so that no square overflows or underflows and the scalings
   add no rounding. [R] takes [Hᴴ R], [H = I - τ v vᴴ], its column [j] [β] then
   zeros. The result is [R], of [a]'s shape, the reflectors' vectors as the
   columns of [b; m; k] and their [τ] as [b; 1; k]. *)
let householder (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    ~(loop : d loop) (a : (v, s, d) Value.t) :
    (v, s, d) Value.t * (v, s, d) Value.t * (v, s, d) Value.t =
  let dt = Prim.dtype a in
  let (Field f) = field dt in
  let shape = Prim.shape a in
  let b = shape.(0) and m = shape.(1) and n = shape.(2) in
  let k = Stdlib.min m n in
  let x i = E.input i dt
  and j i = E.input i D.Int64
  and re i = E.input i f.re in
  let zero = E.const dt (D.zero dt) and one = E.const dt (D.one dt) in
  let fl v = E.const f.re v in
  let col = [| b; m; 1 |] and one_by = [| b; 1; 1 |] in
  let step i = function
    | [ r; vs; taus ] ->
        let r = Prim.expect dt r in
        let vs = Prim.expect dt vs and taus = Prim.expect dt taus in
        let c = column apply ~by i r in
        let alpha = row apply ~by i c in
        let size = largest f.real f.imag (x 0) in
        let tail =
          along apply ~by Max 1
            (mapx apply ~by col f.re
               [ Value.Any c; Value.Any i ]
               E.(where (rows > j 1) size (fl 0.)))
        in
        let mu =
          mapx apply ~by one_by f.re
            [ Value.Any alpha; Value.Any tail ]
            (binade dt E.(maximum (largest f.real f.imag (x 0)) (re 1)))
        in
        let squares =
          along apply ~by Sum 1
            (mapx apply ~by col f.re
               [ Value.Any c; Value.Any i; Value.Any mu ]
               (let y = E.(x 0 / f.of_real (re 2)) in
                E.(
                  where
                    (rows > j 1 && re 2 > fl 0.)
                    (f.real (y * f.conj y))
                    (fl 0.))))
        in
        (* Over [α], the tail's greatest part, [μ] and the tail's scaled
           squares: [β] as [x]'s, or [α] where [H] is the identity, [τ] and [α -
           β]. *)
        let scalar e =
          mapx apply ~by one_by dt
            [ Value.Any alpha; Value.Any tail; Value.Any mu; Value.Any squares ]
            e
        in
        (* [τ] and [v] are the same of [x / μ] as of [x], so they are computed
           there, as zlarfg rescales a column whose [β] would be subnormal: only
           [β = μ β'] meets the scale, where it is a result. *)
        let ar = f.real (x 0) and ai = f.imag (x 0) in
        let alpha' = E.(x 0 / f.of_real (re 2)) in
        let sr = E.(ar / re 2) and si = E.(ai / re 2) in
        let norm = E.(sqrt ((sr * sr) + (si * si) + re 3)) in
        let beta' = f.of_real E.(where (ar < fl 0.) norm (neg norm)) in
        let plain = E.(re 1 = fl 0. && ai = fl 0.) in
        let diagonal = scalar E.(where plain (x 0) (beta' * f.of_real (re 2)))
        and tau = scalar E.(where plain zero ((beta' - alpha') / beta'))
        and gap = scalar E.(where plain one (alpha' - beta')) in
        (* The tail over [μ (α' - β')], divided rather than multiplied by a
           reciprocal, which overflows where [α' - β'] is subnormal. *)
        let v =
          mapx apply ~by col dt
            [ Value.Any c; Value.Any gap; Value.Any i; Value.Any mu ]
            E.(
              where
                (rows < j 2)
                zero
                (where
                   (rows = j 2)
                   one
                   (where (re 3 > fl 0.) (x 0 / f.of_real (re 3) / x 1) zero)))
        in
        let w = product apply ~by (1, 1) (conjugate apply ~by v) r in
        let r =
          mapx apply ~by shape dt
            [
              Value.Any r;
              Value.Any v;
              Value.Any w;
              Value.Any tau;
              Value.Any diagonal;
              Value.Any i;
            ]
            E.(
              where
                (cols = j 5)
                (where (rows > j 5) zero (where (rows = j 5) (x 4) (x 0)))
                (x 0 - (f.conj (x 3) * x 1 * x 2)))
        in
        let put shape v from =
          mapx apply ~by shape dt
            [ Value.Any from; Value.Any v; Value.Any i ]
            E.(where (cols = j 2) (x 1) (x 0))
        in
        [
          Value.Any r;
          Value.Any (put [| b; m; k |] v vs);
          Value.Any (put [| b; 1; k |] tau taus);
        ]
    | _ -> invalid_arg "Expand: QR's loop carries its factor and reflectors"
  in
  let init =
    [
      Value.Any (apply ~by (Value.Copy a));
      Value.Any (zeros apply ~by dt [| b; m; k |]);
      Value.Any (zeros apply ~by dt [| b; 1; k |]);
    ]
  in
  match loop k step init with
  | [ r; vs; taus ] -> (Prim.expect dt r, Prim.expect dt vs, Prim.expect dt taus)
  | _ -> invalid_arg "Expand: QR's loop gives its factor and reflectors"

(* The first [width] columns of [Q = H_0 ⋯ H_(k-1)], dorg2r's: from the
   identity's first [width] columns, at trip [t], [H_i] applied from the left,
   [i = k - 1 - t], [Q ← Q - τ_i v_i (v_iᴴ Q)]. O(m k width). *)
let accumulate (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    ~(loop : d loop) (vs : (v, s, d) Value.t) (taus : (v, s, d) Value.t) width :
    (v, s, d) Value.t =
  let dt = Prim.dtype vs and s = Prim.shape vs in
  let b = s.(0) and m = s.(1) and k = s.(2) in
  let x i = E.input i dt and last = i64 (k - 1) in
  let q0 =
    mapx apply ~by [| b; m; width |] dt []
      E.(where (rows = cols) (const dt (D.one dt)) (const dt (D.zero dt)))
  in
  let step t = function
    | [ q ] ->
        let q = Prim.expect dt q in
        let i =
          mapx apply ~by [||] D.Int64
            [ Value.Any t ]
            E.(last - input 0 D.Int64)
        in
        let v = column apply ~by i vs and tau = column apply ~by i taus in
        let w = product apply ~by (1, 1) (conjugate apply ~by v) q in
        [
          Value.Any
            (mapx apply ~by [| b; m; width |] dt
               [ Value.Any q; Value.Any v; Value.Any w; Value.Any tau ]
               E.(x 0 - (x 3 * x 1 * x 2)));
        ]
    | _ -> invalid_arg "Expand: Q's loop carries Q alone"
  in
  match loop k step [ Value.Any q0 ] with
  | [ q ] -> Prim.expect dt q
  | _ -> invalid_arg "Expand: Q's loop gives Q alone"

(* [x]'s matrices cut to [rows] rows and [cols] columns, copied. *)
let corner (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by rows cols x =
  let s = Prim.shape x in
  let ranges : M.range iarray = [| whole s.(0); whole rows; whole cols |] in
  apply ~by (Value.Copy (move apply ~by (M.Slice ranges) x))

let qr (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by ~loop
    (factors : Nx_kernel.factors) (a : (v, s, d) Value.t) :
    (v, s, d) Value.t * (v, s, d) Value.t =
  let lead, a = matrices apply ~by a in
  let s = Prim.shape a in
  let m = s.(1) and n = s.(2) in
  let r, vs, taus = householder apply ~by ~loop a in
  let width = match factors with Reduced -> Stdlib.min m n | Complete -> m in
  let q = accumulate apply ~by ~loop vs taus width in
  (batched apply ~by lead q, batched apply ~by lead (corner apply ~by width n r))

(* Jacobi's method

   A rotation of the pair [(p, q)], [p < q], of indices of a Hermitian [G],
   whose element [g_pq = |g_pq| e] is not zero, has [c = 1 / √(1 + t²)] and [s =
   t c], [t = sign ζ / (|ζ| + √(1 + ζ²))] (Drmač and Veselić, LAWN 169, §1).
   Over a value's columns or a matrix's rows and columns it takes index [i] into
   [c x_i + κ x_π(i)], [π(i)] its partner, [κ_p = -s conj e] and [κ_q = s e],
   and annihilates [g_pq]. A trip rotates disjoint pairs, a round of a
   1-factorisation of the indices: of [N] players, [n] for an even [n] and [n +
   1] for an odd one, its last a stand-in, round [r] of [R = N - 1] pairs player
   [N - 1] with [r] and [i] with [(2r - i) mod R]. *)

let rounds n = if n mod 2 = 0 then n - 1 else n

(* Index [i]'s partner in round [r], int64 expressions; [n], past the indices,
   for the stand-in. *)
let partner n r i =
  let last = i64 (rounds n) in
  E.(
    where (i = last) r
      (where (i = r) last (((((i64 2 * r) - i) mod last) + last) mod last)))

(* Per index [i], [b; n; 1], of the pairs [π] gives, the cosine [c] and [κ] of
   the rotation of [ζ], [|g_pq|] and the phase [e] that [ζ], [size] and [phase]
   compute from [loads], where [rotated]. *)
let rotation (type v s r d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (dt : (v, s) D.t) (re : (float, r) D.t) ~of_real ~conj shape loads ~first
    ~rotated ~zeta ~phase : (v, s, d) Value.t * (v, s, d) Value.t =
  let one = E.const re 1. in
  (* [|ζ| + √(1 + ζ²)], as [|ζ| (1 + √(1 + ζ⁻²))] past [1], where [ζ²] would
     overflow: dgesvj's [t = 1 / (2ζ)] past its BIGTHETA, a rotation that still
     clears a small column of a large one. *)
  let a = E.abs zeta in
  let denominator =
    E.(
      where (a > one)
        (a * (one + sqrt (one + (one / (a * a)))))
        (a + sqrt (one + (a * a))))
  in
  let t = E.(where (zeta < const re 0.) (neg one) one / denominator) in
  let c = E.(one / sqrt (one + (t * t))) in
  let cos = mapx apply ~by shape dt loads (of_real E.(where rotated c one)) in
  let kappa =
    let s = of_real E.(t * c) and e = phase in
    mapx apply ~by shape dt loads
      E.(
        where rotated
          (where first (neg (s * conj e)) (s * e))
          (const dt (D.zero dt)))
  in
  (cos, kappa)

(* [|z|] without a square that overflows or underflows. *)
let modulus re real imag z =
  let a = E.abs (real z) and b = E.abs (imag z) in
  let big = E.maximum a b in
  let zero = E.const re 0. in
  let x = E.(a / big) and y = E.(b / big) in
  E.(where (big = zero) zero (big * sqrt ((x * x) + (y * y))))

(* The trips for [off(G)²] to fall by [2^-(2e)] at least: the least [T] with [(1
   - 1/R)^T <= 2^-(2e)]. *)
let jacobi_trips n e =
  if n <= 1 then 0
  else
    let r = rounds n in
    if r = 1 then 1
    else
      let fall = Float.log1p (-1. /. Float.of_int r) in
      int_of_float (Float.ceil (-2. *. Float.of_int e *. Float.log 2. /. fall))

(* The rotations of the round of greatest weight [Σ |g_iπ(i)|²] of each matrix
   of [g], Hermitian [b; n; n], given the table of each index's partners per
   round: [π], [c] and [κ] per index, [b; n; 1], with [ζ = (g_qq - g_pp) / (2
   |g_pq|)]. *)
let rotations (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    table (g : (v, s, d) Value.t) =
  let dt = Prim.dtype g in
  let (Field f) = field dt in
  let s = Prim.shape g in
  let b = s.(0) and n = s.(1) in
  let col = [| b; n; 1 |] in
  let x i = E.input i dt and k i = E.input i D.Int64 in
  let round =
    let weights =
      let w =
        apply ~by
          (Value.Gather
             { axis = 2; idx = fit apply ~by [| b; n; rounds n |] table; x = g })
      in
      let w2 =
        mapx apply ~by (Prim.shape w) f.re [ Value.Any w ]
          (f.real E.(x 0 * f.conj (x 0)))
      in
      reduce_plain apply ~by [| 1 |] Sum w2
    in
    let _, r = arg apply ~by (Prim.shape weights) [| 1 |] Max weights in
    reshape apply ~by [| b; 1; 1 |] r
  in
  let pi =
    mapx apply ~by col D.Int64 [ Value.Any round ] (partner n (k 0) rows)
  in
  let diagonal = column apply ~by (mapx apply ~by col D.Int64 [] rows) g in
  let theirs = apply ~by (Value.Gather { axis = 1; idx = pi; x = diagonal }) in
  let g_i = apply ~by (Value.Gather { axis = 2; idx = pi; x = g }) in
  (* [g_pq] as [p] reads it, which [q] takes from [p]: computed [g] need not be
     Hermitian to the bit, and both indices of a pair must rotate by one
     rotation for [J] to be orthogonal. *)
  let g_pair = apply ~by (Value.Gather { axis = 1; idx = pi; x = g_i }) in
  let first = E.(rows < k 3) in
  let gpq = E.(where first (x 2) (x 4)) in
  let mine = f.real (x 0) and other = f.real (x 1) in
  let gpp = E.where first mine other and gqq = E.where first other mine in
  let size = modulus f.re f.real f.imag gpq in
  let loads =
    [
      Value.Any diagonal;
      Value.Any theirs;
      Value.Any g_i;
      Value.Any pi;
      Value.Any g_pair;
    ]
  in
  let cos, kappa =
    rotation apply ~by dt f.re ~of_real:f.of_real ~conj:f.conj col loads ~first
      ~rotated:E.(size > const f.re 0.)
      ~zeta:E.((gqq - gpp) / (const f.re 2. * size))
      ~phase:E.(gpq / f.of_real size)
  in
  (pi, cos, kappa)

(* [x J]: column [i] of [x], [b; m; n], as [c_i x_i + κ_i x_π(i)]. *)
let rotate_columns (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q)
    ~by (pi, cos, kappa) (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  let dt = Prim.dtype x and s = Prim.shape x in
  let across v = reshape apply ~by [| s.(0); 1; s.(2) |] v in
  let theirs =
    apply ~by (Value.Gather { axis = 2; idx = fit apply ~by s (across pi); x })
  in
  let e i = E.input i dt in
  mapx apply ~by s dt
    [
      Value.Any x;
      Value.Any theirs;
      Value.Any (across cos);
      Value.Any (across kappa);
    ]
    E.((e 2 * e 0) + (e 3 * e 1))

(* [Jᴴ x]: row [i] of [x], [b; n; n], as [c_i x_i + conj κ_i x_π(i)], then the
   rotated pairs' elements [+0] and the diagonal real, as the rotations leave
   them in exact arithmetic. *)
let rotate_rows (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    (pi, cos, kappa) (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  let dt = Prim.dtype x and s = Prim.shape x in
  let (Field f) = field dt in
  let theirs =
    apply ~by (Value.Gather { axis = 1; idx = fit apply ~by s pi; x })
  in
  let y = E.((input 2 dt * input 0 dt) + (f.conj (input 3 dt) * input 1 dt)) in
  mapx apply ~by s dt
    [
      Value.Any x;
      Value.Any theirs;
      Value.Any cos;
      Value.Any kappa;
      Value.Any pi;
    ]
    E.(
      where
        (cols = input 4 D.Int64)
        (const dt (D.zero dt))
        (where (rows = cols) (f.of_real (f.real y)) y))

(* Each index's partner per round, [n; R], for [rounds n] rounds. *)
let partners (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by n =
  mapx apply ~by
    [| n; Stdlib.max 1 (rounds n) |]
    D.Int64 []
    (partner n (E.coord 1) (E.coord 0))

(* The eigenvalues of a Hermitian matrix by two-sided Jacobi: [A] from [a]'s
   lower triangle and its diagonal's real parts, {!scaled}, then [jacobi_trips n
   p] trips of [A ← Jᴴ A J] and [V ← V J], each of the round of greatest weight,
   then [A]'s diagonal ascending, scaled back, and [V]'s columns in its order. A
   rotation moves [2 |g_pq|²] of [off(A)²] to the diagonal (LAWN 169, §1), and
   the rounds share [off(A)² / 2] out, so a trip takes at least [off(A)² / R]. A
   matrix with an infinity or a NaN where it is read is NaN. The values come
   with the function that gives [V] where [vectors] carried it. *)
let eigh (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    ~(loop : d loop) ~vectors (a : (v, s, d) Value.t) :
    (v, s, d) Value.t * (unit -> (v, s, d) Value.t) =
  let dt = Prim.dtype a in
  let (Field f) = field dt in
  let lead, a = matrices apply ~by a in
  let shape = Prim.shape a in
  let b = shape.(0) and n = shape.(1) in
  let x i = E.input i dt in
  let a =
    mapx apply ~by shape dt
      [ Value.Any a; Value.Any (transposed apply ~by a) ]
      E.(
        where (rows > cols) (x 0)
          (where (rows = cols) (f.of_real (f.real (x 0))) (f.conj (x 1))))
  in
  let bad = not_finite apply ~by a in
  let a, power = scaled apply ~by a in
  let table = partners apply ~by n in
  let step _ = function
    | w :: vs ->
        let w = Prim.expect dt w in
        let r = rotations apply ~by table w in
        let w = rotate_rows apply ~by r (rotate_columns apply ~by r w) in
        let turned v =
          Value.Any (rotate_columns apply ~by r (Prim.expect dt v))
        in
        Value.Any w :: List.map turned vs
    | [] -> invalid_arg "Expand: eigh's loop carries its matrix"
  in
  let init =
    if vectors then [ Value.Any a; Value.Any (identity apply ~by dt b n) ]
    else [ Value.Any a ]
  in
  let w, vs =
    match loop (jacobi_trips n (digits dt)) step init with
    | w :: vs -> (Prim.expect dt w, List.map (Prim.expect dt) vs)
    | [] -> invalid_arg "Expand: eigh's loop gives its matrix"
  in
  let diagonal =
    column apply ~by (mapx apply ~by [| b; n; 1 |] D.Int64 [] rows) w
  in
  let values =
    mapx apply ~by [| b; n |] f.re
      [
        Value.Any (reshape apply ~by [| b; n |] diagonal);
        Value.Any (reshape apply ~by [| b; 1 |] power);
      ]
      E.(f.real (x 0) / f.real (x 1))
  in
  let sorted, order =
    apply ~by
      (Value.Sort { axis = 1; descending = false; k = None; x = values })
  in
  let w =
    mapx apply ~by [| b; n |] dt [ Value.Any sorted ]
      (f.of_real (E.input 0 f.re))
  in
  let w = batched apply ~by lead (failing apply ~by bad w) in
  let vectors () =
    match vs with
    | [ v ] ->
        let idx = fit apply ~by shape (reshape apply ~by [| b; 1; n |] order) in
        let v = apply ~by (Value.Gather { axis = 2; idx; x = v }) in
        batched apply ~by lead (failing apply ~by bad v)
    | _ -> invalid_arg "Expand: eigh's loop gives its vectors"
  in
  (w, vectors)

(* The sweeps of one-sided Jacobi, dgesvj's NSWEEP. *)
let sweeps = 30

(* The singular values by one-sided Jacobi as dgesvj computes them (Drmač and
   Veselić, LAWN 169 and 170; Demmel and Veselić, LAWN 15, §4), on [a], or [aᴴ]
   where it is wide, of [m'] rows and [k] columns, {!scaled}: {!householder}
   first reduces it to its triangle [R], [k × k], then [sweeps] sweeps of [R]
   rounds rotate the columns of [Rᴴ], at trip [t] round [t mod R]. A pair
   rotates where its columns' cosine exceeds [tol], each column divided by its
   norm, {!householder}'s, before its products, so no square overflows or
   underflows. A matrix whose last sweep still rotated, or with an infinity or a
   NaN, is NaN. The values are the columns' norms, descending, scaled back; the
   left vectors are [Q] of the first reduction times the rotations, and the
   right ones the columns over their norms completed by {!householder} of [k ×
   k] to an orthonormal basis, each column of its [Q] by the sign of its [R]'s
   diagonal. The values come with the function that gives [u] and [vh] of
   given factors where [vectors] carried the rotations. *)
let svd (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    ~(loop : d loop) ~vectors (a : (v, s, d) Value.t) :
    (v, s, d) Value.t
    * (Nx_kernel.factors -> (v, s, d) Value.t * (v, s, d) Value.t) =
  let dt = Prim.dtype a in
  let (Field f) = field dt in
  let lead, a = matrices apply ~by a in
  let shape = Prim.shape a in
  let b = shape.(0) and m = shape.(1) and n = shape.(2) in
  let k = Stdlib.min m n and mm = Stdlib.max m n in
  let tall = m >= n in
  let x i = E.input i dt
  and re i = E.input i f.re
  and j i = E.input i D.Int64 in
  let zero = E.const dt (D.zero dt) and fl v = E.const f.re v in
  (* dgesvj's tolerance, plus the floor a pair's cosine reaches here: each
     column of a rotated pair within [ρ] of exact (LAWN 169, Fact 1) and the
     cosine computed within [γ(k + 2)]. *)
  let tol =
    let unit = Float.ldexp 1. (-digits dt) in
    let complex = match D.kind dt with D.Complex -> true | _ -> false in
    let nu = (if complex then 4. else 2.) *. unit in
    let rho = (if complex then 24. else 6.) *. unit in
    let g =
      Float.of_int (k + 2) *. nu /. (1. -. (Float.of_int (k + 2) *. nu))
    in
    let ctol =
      if vectors then Float.sqrt (Float.of_int k) else Float.of_int k
    in
    (ctol *. unit) +. (2. *. rho) +. g
  in
  (* A column of a norm below the least normal number counts as zero, as its
     value does where results underflow: it neither rotates nor fails the
     check. *)
  let least = if digits dt = 24 then 0x1p-126 else 0x1p-1022 in
  let bad = not_finite apply ~by a in
  let y =
    if tall then a
    else
      mapx apply ~by [| b; mm; k |] dt
        [ Value.Any (transposed apply ~by a) ]
        (f.conj (x 0))
  in
  let y, power = scaled apply ~by y in
  let r0, reflectors, taus = householder apply ~by ~loop y in
  let square = [| b; k; k |] and across = [| b; 1; k |] in
  (* Jacobi runs on [Rᴴ], as LAWN 169's Corollary 2.5 takes [X = R̃ᵀ] and dgejsv
     does: [Rᴴ V = Y], so [R = V Σ Wᴴ], [W] [Y]'s columns over their norms, and
     the left vectors are [Q V], the right ones [W]. *)
  let x0 =
    mapx apply ~by square dt
      [ Value.Any (transposed apply ~by (corner apply ~by k k r0)) ]
      (f.conj (x 0))
  in
  (* Each column's norm, [b; 1; k], and the columns over their norms, [0] for a
     zero column. *)
  let norms y =
    if k = 0 then (zeros apply ~by f.re across, y)
    else
      let mu =
        let top =
          along apply ~by Max 1
            (mapx apply ~by square f.re [ Value.Any y ]
               (largest f.real f.imag (x 0)))
        in
        mapx apply ~by across f.re [ Value.Any top ] (binade dt (re 0))
      in
      let sums =
        along apply ~by Sum 1
          (mapx apply ~by square f.re
             [ Value.Any y; Value.Any mu ]
             (let z = E.(x 0 / f.of_real (re 1)) in
              E.(where (re 1 > fl 0.) (f.real (z * f.conj z)) (fl 0.))))
      in
      let norm =
        mapx apply ~by across f.re
          [ Value.Any mu; Value.Any sums ]
          E.(re 0 * sqrt (re 1))
      in
      let units =
        mapx apply ~by square dt
          [ Value.Any y; Value.Any norm ]
          E.(where (re 1 > fl 0.) (x 0 / f.of_real (re 1)) zero)
      in
      (norm, units)
  in
  let step t = function
    | y :: count :: vs ->
        let y = Prim.expect dt y and count = Prim.expect D.Int64 count in
        let col = [| b; k; 1 |] in
        let pi =
          mapx apply ~by col D.Int64 [ Value.Any t ]
            (partner k E.(j 0 mod i64 (rounds k)) rows)
        in
        let norm, units = norms y in
        let norm = reshape apply ~by col norm in
        let theirs =
          apply ~by (Value.Gather { axis = 1; idx = pi; x = norm })
        in
        let cosine =
          let partners =
            apply ~by
              (Value.Gather
                 {
                   axis = 2;
                   idx = fit apply ~by square (reshape apply ~by across pi);
                   x = units;
                 })
          in
          let terms =
            mapx apply ~by square dt
              [ Value.Any units; Value.Any partners ]
              E.(f.conj (x 0) * x 1)
          in
          reshape apply ~by col (along apply ~by Sum 1 terms)
        in
        (* [p]'s cosine for both of the pair, as {!rotations} takes [g_pq]. *)
        let pair =
          apply ~by (Value.Gather { axis = 1; idx = pi; x = cosine })
        in
        let first = E.(rows < j 3) in
        let cpq = E.(where first (x 2) (x 4)) in
        let size = modulus f.re f.real f.imag cpq in
        let dp = E.where first (re 0) (re 1)
        and dq = E.where first (re 1) (re 0) in
        let loads =
          [
            Value.Any norm;
            Value.Any theirs;
            Value.Any cosine;
            Value.Any pi;
            Value.Any pair;
          ]
        in
        let rotated = E.(size > fl tol && dp >= fl least && dq >= fl least) in
        let cos, kappa =
          rotation apply ~by dt f.re ~of_real:f.of_real ~conj:f.conj col loads
            ~first ~rotated
            ~zeta:E.(((dq / dp) - (dp / dq)) / (fl 2. * size))
            ~phase:E.(cpq / f.of_real size)
        in
        (* The rotations of this sweep so far, [0] at its first round. *)
        let count =
          let here =
            along apply ~by Sum 1
              (mapx apply ~by col D.Int64 loads
                 E.(where rotated (i64 1) (i64 0)))
          in
          mapx apply ~by [| b; 1; 1 |] D.Int64
            [ Value.Any count; Value.Any here; Value.Any t ]
            E.(where (j 2 mod i64 (rounds k) = i64 0) (j 1) (j 0 + j 1))
        in
        let turned v =
          Value.Any
            (rotate_columns apply ~by (pi, cos, kappa) (Prim.expect dt v))
        in
        turned (Value.Any y) :: Value.Any count :: List.map turned vs
    | _ -> invalid_arg "Expand: svd's loop carries its columns and count"
  in
  let init =
    let count = zeros apply ~by D.Int64 [| b; 1; 1 |] in
    if vectors then
      [ Value.Any x0; Value.Any count; Value.Any (identity apply ~by dt b k) ]
    else [ Value.Any x0; Value.Any count ]
  in
  let trips = if k <= 1 then 0 else sweeps * rounds k in
  let y, count, vs =
    match loop trips step init with
    | y :: count :: vs ->
        ( Prim.expect dt y,
          Prim.expect D.Int64 count,
          List.map (Prim.expect dt) vs )
    | _ -> invalid_arg "Expand: svd's loop gives its columns and count"
  in
  let norm, units = norms y in
  (* dgesvj's convergence: a sweep that rotates no pair. One that rotates none
     leaves the matrix as it was, so the next rotates none either: the last
     sweep rotated nothing iff some sweep did not. *)
  let failed =
    mapx apply ~by [| b; 1; 1 |] D.Int64
      [ Value.Any bad; Value.Any count ]
      E.(j 0 + j 1)
  in
  let values =
    mapx apply ~by [| b; k |] f.re
      [
        Value.Any (reshape apply ~by [| b; k |] norm);
        Value.Any (reshape apply ~by [| b; 1 |] power);
      ]
      E.(re 0 / f.real (x 1))
  in
  let sorted, order =
    apply ~by (Value.Sort { axis = 1; descending = true; k = None; x = values })
  in
  let out x = batched apply ~by lead (failing apply ~by failed x) in
  let s =
    out
      (mapx apply ~by [| b; k |] dt [ Value.Any sorted ]
         (f.of_real (E.input 0 f.re)))
  in
  let vectors (factors : Nx_kernel.factors) =
    match vs with
    | [ v ] ->
        let ordered x =
          let idx =
            fit apply ~by (Prim.shape x) (reshape apply ~by across order)
          in
          apply ~by (Value.Gather { axis = 2; idx; x })
        in
        let units = ordered units and v = ordered v in
        let rr, cvs, ctaus = householder apply ~by ~loop units in
        let completed = accumulate apply ~by ~loop cvs ctaus k in
        let signs =
          let idx = mapx apply ~by across D.Int64 [] cols in
          apply ~by (Value.Gather { axis = 1; idx; x = rr })
        in
        let w =
          mapx apply ~by square dt
            [ Value.Any completed; Value.Any signs ]
            E.(where (f.real (x 1) < fl 0.) (neg (x 0)) (x 0))
        in
        let width = match factors with Reduced -> k | Complete -> mm in
        let q = accumulate apply ~by ~loop reflectors taus width in
        let rest = { M.start = k; count = width - k; step = 1 } in
        let u =
          let main = product apply ~by (2, 1) (corner apply ~by mm k q) v in
          if width = k then main
          else
            apply ~by
              (Value.Assemble
                 {
                   dtype = dt;
                   shape = [| b; mm; width |];
                   fill = D.zero dt;
                   pieces =
                     [
                       ([| whole b; whole mm; whole k |], main);
                       ( [| whole b; whole mm; rest |],
                         move apply ~by
                           (M.Slice [| whole b; whole mm; rest |])
                           q );
                     ];
                 })
        in
        let adjoint z =
          let s = Prim.shape z in
          mapx apply ~by
            [| b; s.(2); s.(1) |]
            dt
            [ Value.Any (transposed apply ~by z) ]
            (f.conj (x 0))
        in
        if tall then (out u, out (adjoint w)) else (out w, out (adjoint u))
    | _ -> invalid_arg "Expand: svd's loop gives its vectors"
  in
  (s, vectors)

(* A routine's expansion; [Eig] and [Eig_values] have none. *)
let linalg : type d r.
    ('q. by:string -> 'q Value.prim -> 'q) ->
    by:string ->
    loop:d loop ->
    (d, r) Value.linalg ->
    r option =
 fun apply ~by ~loop l ->
  match l with
  | Cholesky { triangle; a } -> Some (cholesky apply ~by ~loop triangle a)
  | Lu a -> Some (lu apply ~by ~loop a)
  | Qr { factors; a } -> Some (qr apply ~by ~loop factors a)
  | Svd_values a -> Some (fst (svd apply ~by ~loop ~vectors:false a))
  | Svd { factors; a } ->
      let s, vectors = svd apply ~by ~loop ~vectors:true a in
      let u, vh = vectors factors in
      Some (s, u, vh)
  | Eigh_values a -> Some (fst (eigh apply ~by ~loop ~vectors:false a))
  | Eigh a ->
      let w, vectors = eigh apply ~by ~loop ~vectors:true a in
      Some (w, vectors ())
  | Eig_values _ | Eig _ -> None
  | Solve_triangular { triangle; adjoint; unit_diagonal; a; b } ->
      Some (solve apply ~by ~loop triangle ~adjoint ~unit_diagonal a b)

let reduced (type d q) (apply : 'a. by:string -> 'a Value.prim -> 'a) ~by
    layout axes prog (rs : (d, q) Value.reductions)
    (loads : d Value.load iarray) : q option =
  if Iarray.exists Prim.is_padded loads || not (simple_reduce prog rs) then
    reduce apply ~by layout axes prog rs loads
  else None

let run : type r.
    ('q. by:string -> 'q Value.prim -> 'q) ->
    by:string ->
    repeat:
      ('d.
      int ->
      ('d Value.index -> 'd Value.any list -> 'd Value.any list) ->
      'd Value.any list ->
      'd Value.any list) ->
    r Value.prim ->
    r option =
 fun apply ~by ~repeat:loop op ->
  match op with
  | Value.Map { layout; prog; outs; loads }
    when Iarray.exists Prim.is_padded loads ->
      let loads = Iarray.map (unpadded apply ~by) loads in
      Some (apply ~by (Value.Map { layout; prog; outs; loads }))
  | Value.Map { layout; prog; outs; loads }
    when Option.is_none (P.single prog) ->
      let loads =
        Array.init (Iarray.length loads) (fun i ->
            Prim.load_any (Iarray.get loads i))
      in
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
  | Value.Reduce { layout; axes; prog; reductions; loads } ->
      reduced apply ~by layout axes prog reductions loads
  | Value.Scan { layout; axis; prog; reduction = r; loads; carry }
    when Iarray.exists Prim.is_padded loads || not (simple_scan prog r) ->
      scan apply ~by layout axis prog r loads carry
  | Value.Contract
      {
        order;
        batch;
        contracting;
        acc;
        out;
        a;
        b;
        init;
        a_scale;
        b_scale;
        segments;
      } ->
      Some
        (contract apply ~by ~order ~batch ~contracting ~segments ~acc out a b
           init a_scale b_scale)
  | Value.Map { layout; prog; outs; loads } ->
      widened apply ~by layout prog outs loads
  | Value.Assemble { dtype; shape; fill; pieces } ->
      Some (assemble apply ~by dtype shape fill pieces)
  | Value.Gather { axis; idx; x } -> gather apply ~by axis idx x
  | Value.Scatter { combine; unique; axis; idx; updates; into } ->
      scatter apply ~by ~loop combine ~unique axis idx updates into
  | Value.Sort { axis; descending; k; x } ->
      let bits = key_bits (Prim.dim x axis) in
      sort apply ~by ~bits axis descending k x
  | Value.Fold { shape; pad; x } ->
      Some (fold apply ~by (Iarray.to_array shape) pad x)
  | Value.Fft f -> Some (fft apply ~by f)
  | Value.Linalg l -> linalg apply ~by ~loop l
  | Value.Copy x ->
      (* A copy is the map of one [Copy]. *)
      let dt = Prim.dtype x in
      let prog = P.of_node ~ins:[| D.Any dt |] (Op1 (Copy, D.Any dt, 0)) in
      let layout = L.contiguous (Prim.shape x) in
      Option.map fst
        (widened apply ~by layout prog Value.[ dt ] [| Value.Plain x |])
  | Value.Scan _ | Value.Move _ | Value.Bitcast _ | Value.Place _
  | Value.Check _ ->
      None

