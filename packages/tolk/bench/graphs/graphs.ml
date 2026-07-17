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

let all = [ elementwise; reduce; matmul_small ]
