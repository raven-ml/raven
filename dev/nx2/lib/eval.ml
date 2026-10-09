(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Value

(* Computes each constant operand where the operation reads it. *)
let computing = { Prim.place = (fun p x -> Exec.at p x) }
let at_host = { Prim.map = (fun x -> Exec.at Devices.anywhere x) }

let delivered : type r. by:string -> r prim -> r prim =
 fun ~by op ->
  match op with
  | Place _ | Check _ -> Prim.map at_host op
  | Map _ | Copy _ | Move _ | Bitcast _ -> Prim.prepare ~by computing op

let eval ~by op =
  match Interp.receiver ~by op with
  | None -> Exec.run ~by op
  | Some i -> Interp.apply i ~by (delivered ~by op)

(* The one-node maps, built and evaluated: the paths off the fast one. *)
let eval1 ~by k dt x =
  let v, () = eval ~by (Prim.op1 k dt x) in
  v

let eval2 ~by k dt x y =
  let v, () = eval ~by (Prim.op2 k dt x y) in
  v

let eval3 ~by k c x y =
  let v, () = eval ~by (Prim.op3 k c x y) in
  v

let apply1 ~by k dt x =
  if Interp.quiet () then Exec.apply1 ~slow:eval1 ~by k dt x
  else eval1 ~by k dt x

let apply2 ~by k dt x y =
  if Interp.quiet () then Exec.apply2 ~slow:eval2 ~by k dt x y
  else eval2 ~by k dt x y

let apply3 ~by k c x y =
  if Interp.quiet () then Exec.apply3 ~slow:eval3 ~by k c x y
  else eval3 ~by k c x y

let expand i ~by op =
  Interp.expanding i (fun () -> Expand.run { apply = eval } ~by op)
