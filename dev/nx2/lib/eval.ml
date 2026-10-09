(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Value

(* Computes each constant operand where the operation reads it. An operation
   over values of every set alone has no placement to read them at: they stay
   formulas, which an interpreter places as any caller does. *)
let computing (type v s d) (p : d Devices.placement option) (x : (v, s, d) t) :
    (v, s, d) t =
  let x = Exec.live x in
  match p with Some p -> Exec.at p x | None -> x

let delivered : type r. by:string -> r prim -> r prim =
 fun ~by op ->
  match op with
  | Place (p, x) -> Place (p, Exec.at (Devices.rebrand p) (Exec.live x))
  | Check _ -> Prim.map (fun x -> Exec.read (Exec.live x)) op
  | Map _ | Copy _ | Move _ | Bitcast _ -> Prim.prepare ~by computing op

let eval ~by op =
  match Interp.receiver ~by op with
  | None -> Exec.run ~by op
  | Some i ->
      let (Prim.Operands xs) = Prim.operands op in
      List.iteri (fun k (Any x) -> Prim.alive ~by k x) xs;
      Interp.apply i ~by (delivered ~by op)

(* The one-node maps, built and evaluated: the paths off the fast one. *)
let eval1 ~by k dt x =
  let v, () = eval ~by (Prim.op1 ~by k dt x) in
  v

let eval2 ~by k dt x y =
  let v, () = eval ~by (Prim.op2 ~by k dt x y) in
  v

let eval3 ~by k c x y =
  let v, () = eval ~by (Prim.op3 ~by k c x y) in
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

let place ~by p x =
  match x with
  | Array r
    when Devices.rebrand r.at == p
         && String.length r.dead = 0
         && Interp.quiet () ->
      (* A second value over the memory shares it. *)
      Rig.Claim.share (Nx_array.buffer r.a);
      Array { at = p; a = r.a; dead = Prim.live }
  | Array _ | Shards _ | Donated _ | Deferred _ | Traced _ ->
      eval ~by (Place (p, x))

let expand i ~by op = Interp.expanding i (fun () -> Expand.run eval ~by op)
