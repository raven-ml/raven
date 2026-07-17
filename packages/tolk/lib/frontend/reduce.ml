(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Capture before [open Tolk_uop] shadows it with the uop movement module. *)
module Movement_ops = Movement
open Tolk_uop
module Movement = Movement_ops
module U = Uop
module D = Dtype
module T = Tensor

(* Reduction is not allowed over a provably size-one axis: those axes are
   dropped by reshape instead. A symbolic axis whose size cannot be shown to
   be one is reduced. *)
let rop t op axis =
  let sh = T.symbolic_shape t in
  let axis = List.sort_uniq compare axis in
  let reduce_axis =
    List.filter
      (fun x -> U.resolve (U.O.ne (List.nth sh x) (U.const_int 1)))
      axis
  in
  let ret =
    if reduce_axis <> [] then
      T.of_uop (U.reduce_axis ~src:(T.uop t) ~op ~axes:reduce_axis)
    else t
  in
  if axis <> reduce_axis then
    Movement.symbolic_reshape ret
      (List.filteri (fun idx _ -> not (List.mem idx axis)) sh)
  else ret

let reduce t op ?axis ?(keepdim = false) () =
  let axes = match axis with None -> List.init (T.ndim t) Fun.id | Some l -> l in
  let axes = List.map (T.resolve_dim t) axes in
  let axes = if T.ndim t = 0 then [] else axes in
  let ret = rop t op axes in
  if keepdim then
    Movement.symbolic_reshape ret
      (List.mapi
         (fun idx s -> if List.mem idx axes then U.const_int 1 else s)
         (T.symbolic_shape t))
  else ret

let is_narrow_float dt =
  D.equal dt D.float16 || D.equal dt D.bfloat16
  || D.is_fp8 dt

let sum ?axis ?(keepdim = false) ?dtype t =
  let src_dt = T.val_dtype t in
  let acc = match dtype with Some d -> d | None -> D.sum_acc_dtype src_dt in
  let ret = reduce (Dtype_ops.cast t acc) Ops.Add ?axis ~keepdim () in
  if dtype = None && is_narrow_float src_dt then Dtype_ops.cast ret (T.dtype t)
  else ret

let prod ?axis ?(keepdim = false) ?dtype t =
  let dt = match dtype with Some d -> d | None -> T.val_dtype t in
  reduce (Dtype_ops.cast t dt) Ops.Mul ?axis ~keepdim ()

let max ?axis ?(keepdim = false) t = reduce t Ops.Max ?axis ~keepdim ()

let min ?axis ?(keepdim = false) t =
  Elementwise.inverse (max ?axis ~keepdim (Elementwise.inverse t))

let any ?axis ?(keepdim = false) t = max ?axis ~keepdim (Dtype_ops.bool t)
let all ?axis ?(keepdim = false) t = prod ?axis ~keepdim (Dtype_ops.bool t)
