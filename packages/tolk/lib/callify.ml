(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_uop
module U = Uop

let is_store_after u =
  match U.op u, U.src u with
  | Ops.After, src when Array.length src > 1 && not (U.is_bound_var u) ->
      U.op (U.base src.(0)) <> Ops.Alloc || U.op src.(1) = Ops.Store
  | _ -> false

let contiguous_view u =
  let rec has_effect u =
    match U.op u with
    | Ops.After -> true
    | op when Ops.Group.is_movement op || op = Ops.Bitcast -> has_effect (U.src u).(0)
    | _ -> false in
  let base = U.buf_uop u in
  if U.op base <> Ops.Buffer || has_effect u then None
  else match U.contiguous_view_offset u with
    | None -> None
    | Some offset ->
        let view = U.slice ~src:base ~offset:(U.const_int offset)
            ~size:(U.max_numel u) ~dtype:(U.dtype u) in
        let dims = match U.shape u with [d] -> d | dims -> U.stack dims in
        Some (U.reshape ~src:view ~shape:dims)

let rec canonicalize_scope root =
  let allocs = U.Ref_tbl.create 16 in
  U.graph_rewrite ~bottom_up:true (fun u ->
      match U.op u, U.Arg.as_param_arg (U.arg u) with
      | Ops.Alloc, Some p when p.slot >= 0 ->
          (match U.Ref_tbl.find_opt allocs u with
           | Some v -> Some v
           | None ->
               let slot = -1 - U.Ref_tbl.length allocs in
               let v = U.replace u ~arg:(U.Arg.Param_arg {p with slot}) () in
               U.Ref_tbl.add allocs u v;
               Some v)
      | Ops.Call, _ ->
          let src = Array.copy (U.src u) in
          let body = canonicalize_scope src.(0) in
          if body == src.(0) then None else begin
            src.(0) <- body;
            Some (U.replace u ~src ())
          end
      | _ -> None) root

let transform_to_call sink =
  let stores = ref [] in
  ignore (U.graph_rewrite (fun u ->
      if is_store_after u then stores := u :: !stores;
      None) sink);
  let body = canonicalize_scope (U.sink (List.rev !stores)) in
  let replacements = ref [] in
  let replace_input u =
    let slot = List.length !replacements in
    replacements := u :: !replacements;
    Some (U.param_like u ~slot) in
  let body = U.graph_rewrite ~bottom_up:true ~walk:true (fun u ->
      match U.op u with
      | Ops.Buffer when U.addrspace u = Some Dtype.Global -> replace_input u
      | Ops.Slice when U.op (U.src u).(0) = Ops.Buffer -> replace_input u
      | Ops.After when U.is_bound_var u -> replace_input u
      | _ -> None) body in
  let body = U.graph_rewrite ~enter_calls:true (fun u ->
      match U.node_tag u with None -> None | Some _ -> Some (U.replace u ~node_tag:None ())) body in
  let info : U.call_info =
    {grad_fxn = None; name = None; precompile = true;
     precompile_backward = false; dtype = Dtype.void; aux = None} in
  U.call ~body ~args:(List.rev !replacements) ~info
