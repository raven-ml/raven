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

let rec contiguous_view_with_base u =
  let rec has_effect u =
    match U.op u with
    | Ops.After -> true
    | op when Ops.Group.is_movement op || op = Ops.Bitcast -> has_effect (U.src u).(0)
    | _ -> false in
  let rec storage_anchor u =
    if U.op u = Ops.Buffer then Some (u, 0)
    else if U.op u = Ops.Bitcast then storage_anchor (U.src u).(0)
    else view_anchor u
  and view_anchor u =
    match Prepare.contiguous_view u with
    | Some (base, offset) when not (U.equal base u) || U.op base = Ops.Buffer ->
        Option.map (fun (storage, base_offset) ->
            storage, Bound.(to_int (add (int base_offset) (int offset))))
          (storage_anchor base)
    | _ -> None in
  (* A view of a split buffer is a view of each shard when multi_pm lowers
     it to one: that view, split the same way. The tinygrad counterpart
     declines when the view feeds a copy to one device, and drops a copy to
     a device list for the view itself; tolk keeps the copy in both cases,
     which reads the per-shard views in place. *)
  let split () =
    match U.device_of u with
    | Some (U.Multi _) ->
        let lowered = U.graph_rewrite ~name:"multi buffer view" Multi.multi_pm u in
        if U.op lowered = Ops.Unshard then Some lowered else None
    | _ -> None
  in
  if has_effect u then None
  else match split () with
  | Some lowered ->
      Option.map (fun (view, flat) ->
          U.unshard ~src:view ~axes:(List.map fst (U.sharding lowered))
            ~ranges:(List.map snd (U.sharding lowered)) (), flat)
        (contiguous_view_with_base (U.src lowered).(0))
  | None -> match view_anchor u with
    | Some (base, offset) ->
        let bytes = U.bitcast ~src:base ~dtype:Dtype.int8 in
        let size = Bound.(to_int (mul (int (U.max_numel u))
            (int (Dtype.itemsize (U.dtype u))))) in
        let bytes = U.shrink ~src:bytes ~offset:(U.const_int offset)
            ~size:(U.const_int size) in
        let flat = U.bitcast ~src:bytes ~dtype:(U.dtype u) in
        let dims_node = function [d] -> d | dims -> U.stack dims in
        let dims = U.shape u in
        let max_dims = List.map U.const_int (U.max_shape u) in
        let view = U.reshape ~src:flat ~shape:(dims_node max_dims) in
        let view = if List.equal U.equal dims max_dims then view else
          U.shrink ~src:view ~offset:(dims_node (List.map (fun _ -> U.const_int 0) dims))
            ~size:(dims_node dims) in
        Some (view, flat)
    | _ -> None

let contiguous_view u = Option.map fst (contiguous_view_with_base u)

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
  let views = U.Ref_tbl.create 16 in
  let materialized_view src =
    match contiguous_view_with_base src with
    | Some (view, flat) ->
        U.Ref_tbl.replace views flat ();
        Some view
    | None -> None in
  let sink = U.graph_rewrite (fun u ->
      match U.op u, U.src u with
      | (Ops.Copy | Ops.Stage), [|src|]
        when Ops.Group.is_movement (U.op src) || U.op src = Ops.Bitcast ->
          (match materialized_view src with
           | Some view when U.op u = Ops.Stage -> Some view
           | Some view when not (U.equal view src) -> Some (U.replace u ~src:[|view|] ())
           | _ -> None)
      | Ops.Store, src when Array.length src >= 2
          && (U.op src.(0) = Ops.Bitcast
              || Ops.Group.is_movement (U.op src.(0))
                 && (match U.device_of src.(0), U.device_of src.(1) with
                     | Some dst, Some value -> dst <> value
                     | _ -> false)) ->
          (match materialized_view src.(0) with
           | Some view when not (U.equal view src.(0)) ->
               let src = Array.copy src in
               src.(0) <- view;
               Some (U.replace u ~src ())
           | _ -> None)
      | _ -> None) sink in
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
      | (Ops.Shrink | Ops.Bitcast) when U.Ref_tbl.mem views u -> replace_input u
      | Ops.After when U.is_bound_var u -> replace_input u
      | _ -> None) body in
  let body = U.graph_rewrite ~enter_calls:true (fun u ->
      match U.op u, U.node_tag u with
      | _, None -> None
      | Ops.Param, Some tag when tag <> "" -> None
      | _ -> Some (U.replace u ~node_tag:None ())) body in
  let info : U.call_info =
    {grad_fxn = None; name = None; precompile = true;
     precompile_backward = false; dtype = Dtype.void; aux = None} in
  U.call ~body ~args:(List.rev !replacements) ~info
