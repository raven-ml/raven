(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_uop
module U = Uop

let dims_node = function [d] -> d | dims -> U.stack dims
let shrink_to src dims =
  if List.equal U.equal (U.shape src) dims then src
  else U.shrink ~src ~offset:(dims_node (List.map (fun _ -> U.const_int 0) dims))
      ~size:(dims_node dims)

let run sink =
  List.iter (fun u ->
      if Dtype.is_weak (U.dtype u) && U.device_of u <> None then
        invalid_arg "Bufferize: cast a placed weak output to a concrete dtype") (U.children sink);
  let bases = U.Ref_tbl.create 16 in
  List.iter (fun u ->
      let base = U.base u in
      U.Ref_tbl.replace bases base ();
      let rec peel u = match U.op u with
        | Ops.Stage when U.arg u = U.Arg.Empty -> peel (U.base (U.src u).(0))
        | Ops.Detach | Ops.Contiguous_backward -> peel (U.base (U.src u).(0))
        | _ -> u in
      let storage = U.storage_base (peel base) in
      if U.op storage = Ops.Alloc then U.Ref_tbl.replace bases storage ()) (U.children sink);
  let tensor_map = U.Ref_tbl.create 64 in
  let mapped u = Option.value (U.Ref_tbl.find_opt tensor_map u) ~default:u in
  List.iter (fun original ->
      let u = U.replace original ~src:(Array.map mapped (U.src original)) () in
      let u =
        match U.op original, U.Arg.as_param_arg (U.arg original) with
        | Ops.Alloc, Some p when p.bind_on_realize || U.Ref_tbl.mem bases original ->
            U.buffer ~slot:(U.fresh_buffer_slot ()) ~dtype:p.dtype
              ?shape:(Option.map U.const_int p.size) ?device:p.device ()
        | _ when U.Ref_tbl.mem bases original && not (U.is_virtual u)
                 && (U.op (U.storage_base u) = Ops.Alloc || not (U.has_buffer_identity u)) ->
            let rec peel contiguous src = match U.op src with
              | Ops.Stage when U.arg src = U.Arg.Empty -> peel true (U.src src).(0)
              | Ops.Detach | Ops.Contiguous_backward -> peel contiguous (U.src src).(0)
              | _ -> contiguous, src in
            let contiguous, src = peel false u in
            if U.is_virtual src || List.exists (fun d -> U.const_int_value d = Some 0) (U.shape src)
               || U.has_buffer_identity ~after_ok:true src then src
            else if U.op src = Ops.After && Array.length (U.src src) > 1
              && (U.op (U.src src).(1) = Ops.Store
                  || (not contiguous && U.has_buffer_identity (U.storage_base src))) then src
            else
              (match if contiguous then Callify.contiguous_view src else None with
               | Some view -> view
               | None ->
                   let dims = U.max_shard_shape src in
                   let buffer = U.buffer ~slot:(U.fresh_buffer_slot ()) ~dtype:(U.dtype src)
                       ~shape:(dims_node (List.map U.const_int (if dims = [] then [1] else dims)))
                       ?device:(U.device_of src) () in
                   let buffer = U.reshape ~src:buffer ~shape:(dims_node (List.map U.const_int dims)) in
                   let buffer = shrink_to buffer (U.shard_shape src) in
                   let buffer = match U.device_of src, U.axis src with
                     | Some (U.Multi _), Some axis -> U.unshard ~src:buffer ~axes:[axis] ()
                     | _ -> buffer in
                   U.after ~src:buffer ~deps:[U.store ~dst:buffer ~value:src ()])
        | _ -> u in
      if u != original then U.Ref_tbl.replace tensor_map original u)
    (U.toposort ~enter_calls:false sink);
  let sink = mapped sink in
  let becomes = U.toposort ~enter_calls:false sink
    |> List.filter_map (fun u ->
        if not (Callify.is_store_after u) then None else
          let storage = U.graph_rewrite ~bottom_up:true (fun n ->
              if U.op n = Ops.After then Some (U.src n).(0) else None) (U.src u).(0) in
          Some (u, shrink_to storage (U.shape u))) in
  let result = Hashtbl.create (U.Ref_tbl.length tensor_map + List.length becomes) in
  List.iter (fun (u, value) -> Hashtbl.replace result (U.tag u) value) becomes;
  U.Ref_tbl.iter (fun u value ->
      Hashtbl.replace result (U.tag u) (U.substitute ~walk:true becomes value)) tensor_map;
  sink, result
