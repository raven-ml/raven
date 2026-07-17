(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Tolk_uop
module U = Uop

type t = { name : string; size : string; sink : Uop.t }

let name t = t.name
let size t = t.size
let sink t = t.sink

let kernels kg =
  List.filter_map
    (fun node ->
      match (U.op node, U.as_call node) with
      | Ops.Call, Some { body; _ } when U.as_kernel_info body <> None ->
          Some body
      | _ -> None)
    (U.toposort kg)

(* Workloads *)

let elementwise =
  let a = Helpers.mk_param ~idx:0 [ 256; 256 ] in
  let b = Helpers.mk_param ~idx:1 [ 256; 256 ] in
  let c = Helpers.mk_param ~idx:2 [ 256; 256 ] in
  let bc = U.alu_binary ~op:Ops.Mul ~lhs:b ~rhs:c in
  let r = U.alu_binary ~op:Ops.Add ~lhs:a ~rhs:bc in
  { name = "elementwise"; size = "256x256"; sink = Helpers.wrap_sink [ r ] }

let reduce =
  let x = Helpers.mk_param ~idx:0 [ 512; 512 ] in
  let r = U.reduce_axis ~src:x ~op:Ops.Add ~axes:[ 1 ] in
  { name = "reduce"; size = "512x512"; sink = Helpers.wrap_sink [ r ] }

let matmul_small =
  let m, n, k = (128, 128, 128) in
  let a = Helpers.mk_param ~idx:0 [ m; k ] in
  let b = Helpers.mk_param ~idx:1 [ k; n ] in
  let ar = U.reshape ~src:a ~shape:(Helpers.mk_shape [ m; 1; k ]) in
  let ae = U.broadcast_to ~src:ar ~shape:(Helpers.mk_shape [ m; n; k ]) in
  let bt = U.permute ~src:b ~order:[ 1; 0 ] in
  let br = U.reshape ~src:bt ~shape:(Helpers.mk_shape [ 1; n; k ]) in
  let be = U.broadcast_to ~src:br ~shape:(Helpers.mk_shape [ m; n; k ]) in
  let mul = U.alu_binary ~op:Ops.Mul ~lhs:ae ~rhs:be in
  let red = U.reduce_axis ~src:mul ~op:Ops.Add ~axes:[ 2 ] in
  { name = "matmul_small"; size = "128x128x128"; sink = Helpers.wrap_sink [ red ] }

(* Headline scaling workloads. Subtraction and negation are lowered to the
   same primitive form the tensor frontend uses — [a - b] is [a + b * (-1)],
   [-b] is [b * (-1)] — so a benchmarked graph is node-for-node identical to
   the reference graph the comparative driver builds. Coefficient constants are
   exactly representable in float32 so their rendered literals match too. *)

let lorenz_width = 64
let lorenz_ladder = [ 10; 25; 50; 100; 200 ]

let lorenz n_steps =
  let w = lorenz_width in
  let bcast v =
    U.expand
      ~src:(U.const (Const.float Dtype.float32 v))
      ~dims:(Helpers.mk_shape [ w ])
  in
  let mul a b = U.alu_binary ~op:Ops.Mul ~lhs:a ~rhs:b in
  let add a b = U.alu_binary ~op:Ops.Add ~lhs:a ~rhs:b in
  let neg_one = bcast (-1.0) in
  let sub a b = add a (mul b neg_one) in
  let sigma = bcast 10.0
  and rho = bcast 28.0
  and beta = bcast 2.5
  and dt = bcast 0.0625 in
  let x0 = Helpers.mk_param ~idx:0 [ w ] in
  let y0 = Helpers.mk_param ~idx:1 [ w ] in
  let z0 = Helpers.mk_param ~idx:2 [ w ] in
  let step (x, y, z) =
    let dx = mul sigma (sub y x) in
    let dy = sub (mul x (sub rho z)) y in
    let dz = sub (mul x y) (mul beta z) in
    (add x (mul dt dx), add y (mul dt dy), add z (mul dt dz))
  in
  let rec fold i st = if i = 0 then st else fold (i - 1) (step st) in
  let x, y, z = fold n_steps (x0, y0, z0) in
  let state = add (add x y) z in
  let out = U.reduce_axis ~src:state ~op:Ops.Add ~axes:[ 0 ] in
  {
    name = "lorenz";
    size = Printf.sprintf "n%d" n_steps;
    sink = Helpers.wrap_sink [ out ];
  }

let rnn_batch = 32
let rnn_dim = 32
let rnn_ladder = [ 2; 5; 10; 20 ]

let rnn horizon =
  let b = rnn_batch and d = rnn_dim in
  let add a b = U.alu_binary ~op:Ops.Add ~lhs:a ~rhs:b in
  let mul a b = U.alu_binary ~op:Ops.Mul ~lhs:a ~rhs:b in
  let matmul ~m ~k ~n lhs rhs =
    let ar = U.reshape ~src:lhs ~shape:(Helpers.mk_shape [ m; 1; k ]) in
    let ae = U.broadcast_to ~src:ar ~shape:(Helpers.mk_shape [ m; n; k ]) in
    let bt = U.permute ~src:rhs ~order:[ 1; 0 ] in
    let br = U.reshape ~src:bt ~shape:(Helpers.mk_shape [ 1; n; k ]) in
    let be = U.broadcast_to ~src:br ~shape:(Helpers.mk_shape [ m; n; k ]) in
    U.reduce_axis ~src:(mul ae be) ~op:Ops.Add ~axes:[ 2 ]
  in
  let w_in = Helpers.mk_param ~idx:0 [ d; d ] in
  let w_rec = Helpers.mk_param ~idx:1 [ d; d ] in
  let h0 = Helpers.mk_param ~idx:2 [ b; d ] in
  let step_loss h = U.reduce_axis ~src:(mul h h) ~op:Ops.Add ~axes:[ 0; 1 ] in
  let rec loop t h acc =
    if t = horizon then acc
    else
      let x = Helpers.mk_param ~idx:(3 + t) [ b; d ] in
      let h' =
        add (matmul ~m:b ~k:d ~n:d x w_in) (matmul ~m:b ~k:d ~n:d h w_rec)
      in
      let loss = step_loss h' in
      let acc = match acc with None -> loss | Some a -> add a loss in
      loop (t + 1) h' (Some acc)
  in
  let out = match loop 0 h0 None with Some a -> a | None -> step_loss h0 in
  {
    name = "rnn";
    size = Printf.sprintf "h%d" horizon;
    sink = Helpers.wrap_sink [ out ];
  }

let all = [ elementwise; reduce; matmul_small ]
let scaling = List.map lorenz lorenz_ladder @ List.map rnn rnn_ladder
