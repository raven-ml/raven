(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Nx_array.Dtype
module P = Nx_kernel.Prog

type apply = { apply : 'r. by:string -> 'r Value.prim -> 'r }

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
let node_map (type d) a ~by shape prog (loads : d Value.any array)
    (value : int -> d Value.any) i : d Value.any =
  let (D.Any dt) = P.dtype prog i in
  let one node reads =
    let ins = Array.map (fun (Value.Any x) -> D.Any (Prim.dtype x)) reads in
    let loads = Array.map (fun (Value.Any x) -> Value.Plain x) reads in
    let prog = Prim.program node ins in
    let v, () =
      a.apply ~by (Value.Map { shape; prog; outs = Value.[ dt ]; loads })
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

let run : type r. apply -> by:string -> r Value.prim -> r option =
 fun a ~by op ->
  match op with
  | Value.Map { shape; prog; outs; loads } when not (one_node prog) ->
      let loads = Array.map (fun (Value.Plain x) -> Value.Any x) loads in
      let values = Array.make (P.length prog) None in
      let value j = Option.get values.(j) in
      Array.iteri
        (fun i _ ->
          values.(i) <- Some (node_map a ~by shape prog loads value i))
        values;
      Some (results outs (List.map value (Array.to_list (P.outs prog))))
  | Value.Map _ | Value.Copy _ | Value.Move _ | Value.Bitcast _ | Value.Place _
  | Value.Check _ ->
      None
