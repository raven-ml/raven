(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Multi-device collective reduction.

   Implements naive, hierarchical, ring, and all-to-all allreduce strategies for reducing
   buffers across multiple devices. *)

open Tolk_uop
module U = Uop

(* Shape encoding

   Shapes and bounds are Uop nodes: a single dim is a scalar const,
   multiple dims become a stack of scalar consts. *)

let dim = U.const_int

let emit_shape = function
  | [ d ] -> dim d
  | dims -> U.stack (List.map dim dims)

(* Int-list wrappers over Uop shape/bounds APIs. *)

let reshape src dims = U.reshape ~src ~shape:(emit_shape dims)

let shrink src bounds =
  let offset = emit_shape (List.map fst bounds) in
  let size = emit_shape (List.map (fun (b, e) -> e - b) bounds) in
  U.shrink ~src ~offset ~size

let pad_to_shape src ~offset ~shape =
  U.pad ~src ~offset:(emit_shape offset) ~size:(emit_shape shape)

let copy_to_device src dev =
  if U.device_of src = Some (U.Single dev) then src
  else U.copy ~src ~device:(Single dev) ()

let emit = function [d] -> d | dims -> U.stack dims
let shrink_to src shape =
  if List.equal U.equal (U.shape src) shape then src
  else U.shrink ~src ~offset:(emit (List.map (fun _ -> dim 0) shape)) ~size:(emit shape)

(* Canonical device placement: canonical per-device names, and a
   single-element group collapses to that device. *)
let canonicalize_device (device : U.device) : U.device =
  match device with
  | Single d -> Single (Helpers.canonicalize_device_name d)
  | Multi [ d ] -> Single (Helpers.canonicalize_device_name d)
  | Multi ds -> Multi (List.map Helpers.canonicalize_device_name ds)
  | Index _ as d -> d

(* Reduction *)

let reduce op lhs rhs = U.alu_binary ~op ~lhs ~rhs

let fold_reduce op = function
  | [] -> failwith "fold_reduce: empty list"
  | x :: xs -> List.fold_left (reduce op) x xs

let hierarchical buf ~op ~device ~shape ~ndev ~hdev devs =
  let numel = List.fold_left ( * ) 1 shape in
  let flat = reshape buf [numel] in
  let chunks = Array.init hdev (fun k -> numel * k / hdev, numel * (k + 1) / hdev) in
  let fold = fold_reduce op in
  (* Replicas must agree bit for bit, so every device folds the boxes' partial
     sums of a chunk in box order, over stored partials: a partial fused into
     that fold renders as one flat sum with the copies, which C reassociates. *)
  let owned = Array.init ndev (fun i ->
      let k = i mod hdev and box = i / hdev * hdev in
      U.contiguous ~src:(fold (List.init hdev (fun j ->
          let shard = U.mselect ~src:flat ~index:(box + j) in
          copy_to_device (shrink shard [chunks.(k)]) devs.(i)))) ()) in
  let summed = Array.init ndev (fun i ->
      fold (List.init (ndev / hdev) (fun box ->
          let j = box * hdev + i mod hdev in
          if j = i then owned.(i) else copy_to_device owned.(j) devs.(i)))) in
  let gathered = Array.init hdev (fun k ->
      match device with
      | U.Single target -> copy_to_device summed.(k) target
      | _ -> U.mstack (List.init ndev (fun j -> copy_to_device summed.(j / hdev * hdev + k) devs.(j)))) in
  let result = U.usum (List.init hdev (fun k ->
      pad_to_shape gathered.(k) ~offset:[fst chunks.(k)] ~shape:[numel])) in
  reshape result shape

(* handle_allreduce *)

(* The reduction with [op] of [buf]'s shards on [devs], placed on
   [device]. *)
let reduce_shards buf ~op ~device devs =
  let logical_shape = U.shape buf in
  let concrete = List.for_all (fun d -> Option.is_some (U.const_int_value d)) logical_shape in
  let shape = U.max_shape buf in
  let devs = Array.of_list devs in
  let ndev = Array.length devs in
  let numel = List.fold_left ( * ) 1 shape in
  let threshold =
    Helpers.Context_var.get Helpers.ring_allreduce_threshold
  in
  let all2all = Helpers.Context_var.get Helpers.all2all in
  let ring = Helpers.Context_var.get Helpers.ring in
  (* Ring allreduce doesn't benefit with <=2 nodes or <256k elements —
     fall back to naive to save on dispatch and chunking. *)
  let use_all2all =
    concrete && (all2all >= 2 || (ndev > 2 && numel > threshold && all2all >= 1))
  in
  let use_ring =
    concrete && (not use_all2all)
    && (ring >= 2 || (ndev > 2 && numel > threshold && ring >= 1))
  in
  let padded = if concrete then buf else
      U.pad ~src:buf ~offset:(emit_shape (List.map (fun _ -> 0) shape)) ~size:(emit_shape shape) in
  let buf = U.contiguous ~src:padded () in
  let hdev = Helpers.Context_var.get Helpers.allreduce_node_ndevs in
  if concrete && hdev > 0 && ndev mod hdev = 0 then
    hierarchical buf ~op ~device ~shape ~ndev ~hdev devs
  else if (not use_ring) && not use_all2all then
    (* Naive: copy every shard to the target device and reduce. *)
    let shards =
      List.init ndev (fun i ->
          U.copy ~src:(U.mselect ~src:buf ~index:i) ~device ())
    in
    shrink_to (fold_reduce op shards) logical_shape
  else
    (* Divide into ndev chunks, aligned to the largest power-of-2 factor
       (up to 32) that divides numel. Larger chunks go to earlier
       devices. *)
    let factor =
      Option.value ~default:1
        (List.find_opt (fun f -> numel mod f = 0) [ 32; 16; 8; 4; 2 ])
    in
    let base = numel / factor / ndev in
    let left = numel / factor mod ndev in
    let chunks =
      Array.init ndev (fun i ->
          (if i < left then base + 1 else base) * factor)
    in
    (* Prefix-sum to get (start, end) pairs. *)
    let bounds =
      let pos = ref 0 in
      Array.map
        (fun sz ->
          let s = !pos in
          pos := s + sz;
          (s, s + sz))
        chunks
    in
    (* Reduce-scatter: each device ends up with one fully-reduced chunk. *)
    let reduced_chunks =
      Array.mapi
        (fun i (s, e) ->
          if use_all2all then
            (* All-to-all: gather chunk [s,e) from every device onto
               device i. *)
            let chunks_on_i =
              List.init ndev (fun j ->
                  let shard = U.mselect ~src:buf ~index:j in
                  copy_to_device
                    (shrink (reshape shard [ numel ]) [ (s, e) ])
                    devs.(i))
            in
            fold_reduce op chunks_on_i
          else
            (* Ring: walk chunk around the ring, accumulating at each
               hop. *)
            let flat = reshape buf [ numel ] in
            let chunk = shrink flat [ (s, e) ] in
            let reduced = ref (shrink flat [ (s, e) ]) in
            for step = 0 to ndev - 2 do
              let src_idx = (i + step) mod ndev in
              let dest_idx = (i + step + 1) mod ndev in
              (* On the first step, reduced is still multi-device
                 (inherits from buf) and needs mselect. After that it
                 lives on a single device. *)
              let r =
                if step = 0 then U.mselect ~src:!reduced ~index:src_idx
                else !reduced
              in
              let cp = copy_to_device r devs.(dest_idx) in
              let ch =
                copy_to_device
                  (U.mselect ~src:chunk ~index:dest_idx)
                  devs.(dest_idx)
              in
              reduced := reduce op cp ch
            done;
            !reduced)
        bounds
    in
    (* Allgather: broadcast each reduced chunk to all devices. *)
    let copied_chunks =
      Array.mapi
        (fun i rc ->
          match device with
          | Single target ->
              (* Target is a single device — just copy there. *)
              copy_to_device rc target
          | _ when use_all2all ->
              (* All-to-all: copy to every device and stack. *)
              U.mstack
                (List.init ndev (fun j -> copy_to_device rc devs.(j)))
          | _ ->
              (* Ring: chain copies around the ring, then reorder. *)
              let chain = Array.make ndev rc in
              let current = ref rc in
              for step = 0 to ndev - 2 do
                current :=
                  copy_to_device !current devs.((i + step) mod ndev);
                chain.(step + 1) <- !current
              done;
              U.mstack
                (List.init ndev (fun j ->
                     chain.((j - i + 1 + ndev) mod ndev))))
        reduced_chunks
    in
    (* Reassemble: pad each chunk back to full size and sum. *)
    let padded =
      List.init ndev (fun i ->
          let s = fst bounds.(i) in
          pad_to_shape copied_chunks.(i) ~offset:[ s ] ~shape:[ numel ])
    in
    reshape (U.usum padded) shape

let handle_allreduce buf ~op ~device =
  match U.device_of buf with
  | Some (Multi devs) -> Some (reduce_shards buf ~op ~device devs)
  | _ -> None

(* Collectives *)

let collective ~name ~device ~like src body =
  let shape = U.shape like and max_shape = U.max_shape like in
  let alloc = U.alloc ~slot:(U.fresh_buffer_slot ()) ~device:(canonicalize_device device)
      ~dtype:(U.dtype like) ~shape:(dim (List.fold_left ( * ) 1 max_shape)) () in
  (* The call takes the whole allocation and views it inside its body. A view
     argument was scheduled as a copy of the viewed values, so the call's
     writes missed the storage its consumers read. *)
  let view storage = shrink_to (reshape storage max_shape) shape in
  let dst = U.param_like alloc ~slot:0 in
  let stores = body ~dst:(view dst) ~src:(U.param_like src ~slot:1) in
  let info : U.call_info = {
    grad_fxn = None; name = Some name; precompile = true;
    precompile_backward = false; dtype = Dtype.void; aux = None } in
  let call = U.call ~body:(U.sink [U.after ~src:dst ~deps:stores])
      ~args:[alloc; U.contiguous ~src ()] ~info in
  U.after ~src:(view alloc) ~deps:[call]

let create_allreduce_function buf ~op ~device =
  let like = U.allreduce ~src:buf ~op ~device in
  Option.map (fun result ->
      collective ~name:"allreduce" ~device ~like buf (fun ~dst ~src:_ ->
          [U.store ~dst ~value:result ()]))
    (handle_allreduce (U.param_like buf ~slot:1) ~op ~device)
