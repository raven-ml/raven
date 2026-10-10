(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Nx_array.Dtype
module P = Nx_kernel.Prog

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

let plain_reduce (type d q) prog (rs : (d, q) Value.reductions) =
  match rs with [ r ] -> plain prog r | _ -> false

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

let rec reductions : type d q.
    ('a. by:string -> 'a Value.prim -> 'a) ->
    by:string ->
    loop:
      ('a.
       Nx_array.Layout.t ->
       P.t ->
       (d, 'a) Value.reduction ->
       d Value.load array ->
       'a) ->
    Nx_array.Layout.t ->
    P.t ->
    d Value.load array ->
    (d, q) Value.reductions ->
    q option =
 fun apply ~by ~loop layout prog loads -> function
  | [] -> Some ()
  | r :: rest -> (
      match reduction apply ~by ~loop layout prog loads r with
      | None -> None
      | Some a -> (
          match reductions apply ~by ~loop layout prog loads rest with
          | Some q -> Some (a, q)
          | None -> None))

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
  reductions apply ~by ~loop layout prog loads rs

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

(* A one-node map at dtypes some library declines, as the node at their
   accumulators ({!accumulator}), which hold every value exactly, its result
   rounded once to its dtype: [None] for a node no accumulator computes exactly,
   as a cast, a bitcast or a copy whose bits are its meaning, or a complex
   dtype. *)
let widened (type d r) (apply : 'q. by:string -> 'q Value.prim -> 'q) ~by layout
    prog (outs : (d, r) Value.outs) (loads : d Value.load array) : r option =
  let n = P.length prog in
  let node = P.node prog (n - 1) in
  let dts = Array.map (fun (Value.Plain x) -> D.Any (Prim.dtype x)) loads in
  let wide d = accumulator d in
  let narrow d = (not (base d)) && base (wide d) in
  let result = P.dtype prog (n - 1) in
  let widens =
    match (node : P.node) with
    | Op1 ((Copy | Cast | Bitcast), _, _) | In _ | Coord _ -> false
    | Const (dt, _) -> narrow dt
    | Op1 _ | Op2 _ | Op3 _ ->
        Array.for_all (fun d -> base d || narrow d) dts
        && (base result || narrow result)
        && Array.exists narrow (Array.append dts [| result |])
  in
  if not widens then None
  else
    let load (Value.Plain x) =
      let (D.Any w) = wide (D.Any (Prim.dtype x)) in
      Value.Any (cast_to apply ~by w x)
    in
    let operands = Array.map load loads in
    let ins = Array.map (fun (Value.Any x) -> D.Any (Prim.dtype x)) operands in
    let node : P.node =
      match node with
      | Const (dt, b) -> Const (wide dt, widened_bits dt b (wide dt))
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
    match outs with [ dt ] -> Some (cast_to apply ~by dt v, ()) | _ -> None

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
    when not (plain_reduce prog reductions) ->
      reduce apply ~by layout axes prog reductions loads
  | Value.Scan { layout; axis; prog; reduction = r; loads }
    when not (plain prog r) ->
      scan apply ~by layout axis prog r loads
  | Value.Map { layout; prog; outs; loads } ->
      widened apply ~by layout prog outs loads
  | Value.Reduce _ | Value.Scan _ | Value.Copy _ | Value.Move _
  | Value.Bitcast _ | Value.Place _ | Value.Check _ ->
      None
