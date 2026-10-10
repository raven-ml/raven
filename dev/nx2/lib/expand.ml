(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Nx_array.Dtype
module L = Nx_array.Layout
module P = Nx_kernel.Prog
module S = Nx_kernel.Spec
module M = Nx_array.Move

(* Whether [prog] is one node over its operands in order, which a kernel
   computes alone. *)
let one_node prog =
  let n = P.length prog in
  let ins = Array.length (P.ins prog) in
  n = ins + 1
  && P.outs prog = [| ins |]
  && List.for_all (fun i -> P.node prog i = P.In i) (List.init ins Fun.id)

(* Node [i] of [prog] over [loads], as a one-node map of [shape] over [value j]
   for each node [j] it reads. *)
let node_map (type d) (apply : 'r. by:string -> 'r Value.prim -> 'r) ~by layout
    prog (loads : d Value.any array) (value : int -> d Value.any) i :
    d Value.any =
  let (D.Any dt) = P.dtype prog i in
  let one node reads =
    let ins = Array.map (fun (Value.Any x) -> D.Any (Prim.dtype x)) reads in
    let loads = Array.map (fun (Value.Any x) -> Value.Plain x) reads in
    let prog = Prim.program node ins in
    let v, () =
      apply ~by (Value.Map { layout; prog; outs = Value.[ dt ]; loads })
    in
    Value.Any v
  in
  match P.node prog i with
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

(* Whether [prog] reads its one operand and gives it back. *)
let identity prog =
  Array.length (P.ins prog) = 1
  && Array.length (P.outs prog) = 1
  && P.node prog (P.outs prog).(0) = P.In 0

let output prog k = P.dtype prog (P.outs prog).(k)

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
  identity prog && k = 0
  && same (output prog 0) dt
  && same (accumulator dt) dt
  && not (complex_sum r)

(* Whether a reduction is core: one plain [Sum], [Prod], [Max] or [Min]. *)
let core_reduce (type d q) prog (rs : (d, q) Value.reductions) =
  match rs with
  | [ (Monoid ((Sum | Prod | Max | Min), _, _) as r) ] -> plain prog r
  | _ -> false

(* Output [k] of [prog] over [loads], of [layout]'s shape, stored in [acc]: the
   operand itself where [prog] is the identity and [acc] its dtype. *)
let body (type d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by layout prog
    k (loads : d Value.load array) (D.Any acc) : d Value.any =
  let out = (P.outs prog).(k) in
  let cast = not (same (P.dtype prog out) (D.Any acc)) in
  if identity prog && not cast then
    let (Value.Plain x) = loads.(0) in
    Value.Any x
  else
    let nodes = Array.init (P.length prog) (P.node prog) in
    let nodes, last =
      if cast then
        (Array.append nodes [| P.Op1 (Cast, D.Any acc, out) |], P.length prog)
      else (nodes, out)
    in
    let prog = P.v ~ins:(P.ins prog) nodes ~outs:[| last |] in
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
          (Prim.program (P.In 0) [| D.Any xt |])
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

(* A program built node by node: [push b n] appends [n] and is its index. *)
type builder = { mutable nodes : P.node list; mutable count : int }

let builder () = { nodes = []; count = 0 }

let push b n =
  b.nodes <- n :: b.nodes;
  b.count <- b.count + 1;
  b.count - 1

let program b ~ins out =
  P.v ~ins (Array.of_list (List.rev b.nodes)) ~outs:[| out |]

let const (type v s) (dt : (v, s) D.t) (x : v) = P.Const (D.Any dt, P.bits dt x)

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
let fold1 (type w r d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by shape
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
let fold_plain (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    axes m (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  let dt = Prim.dtype x in
  fold1 apply ~by (Prim.shape x) axes m
    (Prim.program (P.In 0) [| D.Any dt |])
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
  let ins = [| D.Any dt; D.Any dt |] in
  let out = Prim.reduced shape axes in
  let f v = const dt (D.of_float dt v) in
  let sum_exp shift =
    let b = builder () in
    let t = push b (In 0) in
    let shifted =
      match shift with
      | None -> t
      | Some _ -> push b (Op2 (Binary Sub, t, push b (In 1)))
    in
    let e = push b (Op1 (Unary Exp, D.Any dt, shifted)) in
    let loads =
      match shift with
      | None -> [| Value.Plain x |]
      | Some m ->
          [| Value.Plain x; Value.Plain (spread_back apply ~by shape axes m) |]
    in
    let ins = Array.sub ins 0 (Array.length loads) in
    fold1 apply ~by shape axes Sum (program b ~ins e) dt loads
  in
  if Array.exists (fun a -> shape.(a) = 0) axes then
    map1 apply ~by out
      (Prim.program (Op1 (Unary Log, D.Any dt, 0)) [| D.Any dt |])
      dt
      [| Value.Plain (sum_exp None) |]
  else
    let m = fold_plain apply ~by axes Max x in
    (* [m'], from the maximum at node [m]. *)
    let finite b m =
      let a = push b (Op1 (Unary Abs, D.Any dt, m)) in
      let lt = push b (Op2 (Compare Less, a, push b (f Float.infinity))) in
      push b (Op3 (Where, lt, m, push b (f 0.)))
    in
    let shift =
      let b = builder () in
      map1 apply ~by out
        (program b ~ins:[| D.Any dt |] (finite b (push b (In 0))))
        dt [| Value.Plain m |]
    in
    let s = sum_exp (Some shift) in
    let b = builder () in
    let mi = push b (In 0) and si = push b (In 1) in
    let r =
      push b
        (Op2 (Binary Add, finite b mi, push b (Op1 (Unary Log, D.Any dt, si))))
    in
    map1 apply ~by out
      (program b ~ins (nan_or b mi r))
      dt
      [| Value.Plain m; Value.Plain s |]

(* Moments in two passes: the mean, [Σ x / n], then the variance, [Σ (x - mean)²
   / n], each a sum. Where the mean is a NaN, both are the mean. *)
let moments (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    shape axes (x : (v, s, d) Value.t) : (v, s, d) Value.t * (v, s, d) Value.t =
  let dt = Prim.dtype x in
  let out = Prim.reduced shape axes in
  let n = Array.fold_left (fun n a -> n * shape.(a)) 1 axes in
  let count = const dt (D.of_float dt (Float.of_int n)) in
  (* [s / n] at node [s], or [s] itself where it is a NaN, from [over]. *)
  let divided b s over =
    let q = push b (Op2 (Binary Fdiv, s, push b count)) in
    nan_or b over q
  in
  let mean =
    let b = builder () in
    let s = push b (In 0) in
    map1 apply ~by out
      (program b ~ins:[| D.Any dt |] (divided b s s))
      dt
      [| Value.Plain (fold_plain apply ~by axes Sum x) |]
  in
  let squares =
    let b = builder () in
    let d = push b (Op2 (Binary Sub, push b (In 0), push b (In 1))) in
    let sq = push b (Op2 (Binary Mul, d, d)) in
    fold1 apply ~by shape axes Sum
      (program b ~ins:[| D.Any dt; D.Any dt |] sq)
      dt
      [| Value.Plain x; Value.Plain (spread_back apply ~by shape axes mean) |]
  in
  let var =
    let b = builder () in
    let s = push b (In 0) and m = push b (In 1) in
    map1 apply ~by out
      (program b ~ins:[| D.Any dt; D.Any dt |] (divided b s m))
      dt
      [| Value.Plain squares; Value.Plain mean |]
  in
  (mean, var)

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
          let s = Prim.shape x in
          let part j =
            let ranges =
              Array.append
                (Array.map (fun e -> { M.start = 0; count = e; step = 1 }) s)
                [| { M.start = j; count = 1; step = 1 } |]
            in
            let p = apply ~by (Value.Move (Slice ranges, words)) in
            Value.Any (apply ~by (Value.Move (Reshape (Array.copy s), p)))
          in
          [ part 0; part 1 ])

(* Arg: the extreme [m], then the least position along [axes], in C order of the
   reduced indices, whose term has [m]'s bits, or, where [m] is a NaN, is a
   NaN. *)
let arg (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by shape
    axes (e : S.extreme) (x : (v, s, d) Value.t) :
    (v, s, d) Value.t * (int64, D.int64_elt, d) Value.t =
  let dt = Prim.dtype x in
  let m =
    fold_plain apply ~by axes (match e with Max -> Max | Min -> Min) x
  in
  let mb = spread_back apply ~by shape axes m in
  let kx = keys apply ~by x and km = keys apply ~by mb in
  let nan = match D.kind dt with D.Float | D.Complex -> true | _ -> false in
  let values = if nan then [ Value.Any x; Value.Any mb ] else [] in
  let loads = Array.of_list (values @ kx @ km) in
  let ins = Array.map (fun (Value.Any y) -> D.Any (Prim.dtype y)) loads in
  (* Nodes [0] to [n - 1] are the [n] loads: [x] and [m] where they may be NaNs,
     then [x]'s keys, then [m]'s. *)
  let b = builder () in
  Array.iteri (fun i _ -> ignore (push b (In i))) loads;
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
  let int64 n = const D.Int64 (Int64.of_int n) in
  let r = Array.length shape in
  let position, _ =
    Array.fold_right
      (fun a (pos, stride) ->
        let c = push b (Coord (r - 1 - a)) in
        let t = push b (Op2 (Binary Mul, c, push b (int64 stride))) in
        (push b (Op2 (Binary Add, pos, t)), stride * shape.(a)))
      axes
      (push b (int64 0), 1)
  in
  let none = push b (const D.Int64 Int64.max_int) in
  let pick = push b (Op3 (Where, found, position, none)) in
  let p =
    fold1 apply ~by shape axes Min (program b ~ins pick) D.Int64
      (Array.map (fun (Value.Any y) -> Value.Plain y) loads)
  in
  (m, p)

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
  reduction apply ~by ~loop layout prog loads r

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

(* float32 would not do for the floats of 8 or 16 bits: a store from it
   saturates float8_e5m2's infinities, merges its NaNs into one code and quiets
   the 16-bit floats' signaling NaNs. *)
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
    prog (outs : (d, r) Value.outs) (loads : d Value.load array) : r option =
  let n = P.length prog in
  let node = P.node prog (n - 1) in
  let dts = Array.map (fun (Value.Plain x) -> D.Any (Prim.dtype x)) loads in
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
    let load (Value.Plain x) =
      let (D.Any w) = wide (D.Any (Prim.dtype x)) in
      Value.Any (held apply ~by w x)
    in
    let operands = Array.map load loads in
    let ins = Array.map (fun (Value.Any x) -> D.Any (Prim.dtype x)) operands in
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
             prog = Prim.program node ins;
             outs = Value.[ w ];
             loads = Array.map (fun (Value.Any x) -> Value.Plain x) operands;
           })
    in
    match outs with [ dt ] -> Some (held apply ~by dt v, ()) | _ -> None

(* The program of a region's flat positions in a C-contiguous value whose axes
   have [strides]: [Σ (start + i·step)·stride] at the region's index [i]. *)
let positions (rs : Nx_array.Move.range array) strides =
  let r = Array.length rs in
  let int64 n = const D.Int64 (Int64.of_int n) in
  let first = ref 0 in
  Array.iteri
    (fun d (g : Nx_array.Move.range) ->
      first := !first + (g.start * strides.(d)))
    rs;
  let b = builder () in
  let sum = ref (push b (int64 !first)) in
  Array.iteri
    (fun d (g : Nx_array.Move.range) ->
      let c = push b (P.Coord (r - 1 - d)) in
      let k = push b (int64 (g.step * strides.(d))) in
      let m = push b (P.Op2 (Binary Mul, c, k)) in
      sum := push b (P.Op2 (Binary Add, !sum, m)))
    rs;
  program b ~ins:[||] !sum

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

(* The contraction [spec] of [a] and [b] from [init], into [out], as other
   operations. A float result narrower than its accumulator is the contraction
   into the accumulator, rounded once. Any other is the products in the
   accumulator at every index of the result's axes then the contracted ones, a
   sum over the contracted ones, [init] added, and one rounding to [out]. *)
let contract (type v s a b c e d) (apply : 'q. by:string -> 'q Value.prim -> 'q)
    ~by spec (out : (v, s) D.t) (a : (a, b, d) Value.t) (b : (c, e, d) Value.t)
    (init : (v, s, d) Value.t option) : (v, s, d) Value.t =
  let (D.Any acc) = S.acc spec in
  let batch = S.batch spec and contracting = S.contracting spec in
  if D.is D.Float acc && not (D.equal acc out) then
    let spec =
      S.contract ~batch ~contracting ~acc:(D.Any acc) ~out:(D.Any acc)
        ~init:(S.init spec)
    in
    let init = Option.map (cast_to apply ~by acc) init in
    let y = apply ~by (Value.Contract { spec; out = acc; a; b; init }) in
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
let gather (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    axis idx (x : (v, s, d) Value.t) : (v, s, d) Value.t option =
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

(* A narrow float's [Add] of repeated targets as {!repeated} in its
   accumulator, rounded once; a target no update reaches, found by a [Set] of
   [true] at every target, keeps [into]'s bits. *)
let rounded_once (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q)
    ~by (w : ('w, 'r) D.t) axis idx (updates : (v, s, d) Value.t)
    (into : (v, s, d) Value.t) : (v, s, d) Value.t =
  let dt = Prim.dtype into in
  let wide x = cast_to apply ~by w x in
  let sum = repeated apply ~by S.Add axis idx (wide updates) (wide into) in
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
    repeated apply ~by S.Set axis idx
      (fill true (Prim.shape updates))
      (fill false (Prim.shape into))
  in
  let v, () =
    apply ~by (Prim.op3 ~by Where touched (cast_to apply ~by dt sum) into)
  in
  v

(* A sub-byte scatter at its accumulator, cast back once. An integer's [Add]
   wraps there to the same bits as at its own dtype. A scatter whose targets
   may repeat at a dtype of a byte or more is {!repeated}, a narrow float's
   sum {!rounded_once}. *)
let scatter (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by
    combine ~unique axis idx (updates : (v, s, d) Value.t)
    (into : (v, s, d) Value.t) : (v, s, d) Value.t option =
  match sub_byte into with
  | None when unique -> None
  | None -> (
      match (combine, accumulator (D.Any (Prim.dtype into))) with
      | S.Add, D.Any w when not (same (D.Any w) (D.Any (Prim.dtype into))) ->
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

(* A sub-byte sort at its accumulator, which holds each code and orders them
   alike, its values cast back once. No sub-byte dtype has a NaN code. *)
let sort (type v s d) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by axis
    descending k (x : (v, s, d) Value.t) :
    ((v, s, d) Value.t * (int64, D.int64_elt, d) Value.t) option =
  match sub_byte x with
  | None -> None
  | Some (D.Any w) ->
      let x' = cast_to apply ~by w x in
      let v, p = apply ~by (Value.Sort { axis; descending; k; x = x' }) in
      Some (cast_to apply ~by (Prim.dtype x) v, p)

let run : type r.
    ('q. by:string -> 'q Value.prim -> 'q) ->
    by:string ->
    r Value.prim ->
    r option =
 fun apply ~by op ->
  match op with
  | Value.Map { layout; prog; outs; loads } when not (one_node prog) ->
      let loads = Array.map (fun (Value.Plain x) -> Value.Any x) loads in
      let values = Array.make (P.length prog) None in
      let value j = Option.get values.(j) in
      Array.iteri
        (fun i _ ->
          values.(i) <- Some (node_map apply ~by layout prog loads value i))
        values;
      Some (results outs (List.map value (Array.to_list (P.outs prog))))
  | Value.Reduce { layout; axes; prog; reductions; loads }
    when not (core_reduce prog reductions) ->
      reduce apply ~by layout axes prog reductions loads
  | Value.Scan { layout; axis; prog; reduction = r; loads }
    when not (plain prog r) ->
      scan apply ~by layout axis prog r loads
  | Value.Contract { spec; out; a; b; init } ->
      Some (contract apply ~by spec out a b init)
  | Value.Map { layout; prog; outs; loads } ->
      widened apply ~by layout prog outs loads
  | Value.Assemble { dtype; shape; fill; pieces } ->
      Some (assemble apply ~by dtype shape fill pieces)
  | Value.Gather { axis; idx; x } -> gather apply ~by axis idx x
  | Value.Scatter { combine; unique; axis; idx; updates; into } ->
      scatter apply ~by combine ~unique axis idx updates into
  | Value.Sort { axis; descending; k; x } -> sort apply ~by axis descending k x
  | Value.Reduce _ | Value.Scan _ | Value.Copy _ | Value.Move _
  | Value.Bitcast _ | Value.Place _ | Value.Check _ ->
      None
