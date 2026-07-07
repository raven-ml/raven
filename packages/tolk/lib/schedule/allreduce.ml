(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Multi-device collective reduction.

   Implements naive, ring, and all-to-all allreduce strategies for reducing
   buffers across multiple devices. *)

open Tolk_uop
module U = Uop

(* Environment *)

let ring_var = Helpers.Context_var.int ~key:"RING" ~default:1
let all2all_var = Helpers.Context_var.int ~key:"ALL2ALL" ~default:0

let ring_allreduce_threshold =
  Helpers.Context_var.int ~key:"RING_ALLREDUCE_THRESHOLD" ~default:256_000

(* Shape encoding

   Shapes and bounds are Uop nodes: a single dim is a scalar const,
   multiple dims become a stack of scalar consts. *)

let dim d = U.const (Const.int Dtype.Val.weakint d)

let emit_shape = function
  | [ d ] -> dim d
  | dims -> U.stack (List.map dim dims)

let emit_pairs pairs =
  (emit_shape (List.map fst pairs), emit_shape (List.map snd pairs))

(* Int-list wrappers over Uop shape/bounds APIs. *)

let reshape src dims = U.reshape ~src ~shape:(emit_shape dims)

let shrink src bounds =
  let offset = emit_shape (List.map fst bounds) in
  let size = emit_shape (List.map (fun (b, e) -> e - b) bounds) in
  U.shrink ~src ~offset ~size

let pad_to_shape src ~offset ~shape =
  U.pad ~src ~offset:(emit_shape offset) ~size:(emit_shape shape)

let copy_to_device src dev = U.copy ~src ~device:(Single dev) ()

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

(* handle_allreduce *)

let handle_allreduce buf ~op ~device ~shape =
  match U.device_of buf with
  | Some (Multi devs) ->
      let devs = Array.of_list devs in
      let ndev = Array.length devs in
      let numel = List.fold_left ( * ) 1 shape in
      let threshold = Helpers.Context_var.get ring_allreduce_threshold in
      let all2all = Helpers.Context_var.get all2all_var in
      let ring = Helpers.Context_var.get ring_var in
      (* Ring allreduce doesn't benefit with <=2 nodes or <256k elements —
         fall back to naive to save on dispatch and chunking. *)
      let use_all2all =
        all2all >= 2 || (ndev > 2 && numel > threshold && all2all >= 1)
      in
      let use_ring =
        (not use_all2all)
        && (ring >= 2 || (ndev > 2 && numel > threshold && ring >= 1))
      in
      let buf = U.contiguous ~src:buf () in
      if (not use_ring) && not use_all2all then
        (* Naive: copy every shard to the target device and reduce. *)
        let shards =
          List.init ndev (fun i ->
              U.copy ~src:(U.mselect ~src:buf ~index:i) ~device ())
        in
        Some (fold_reduce op shards)
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
              let s, e = bounds.(i) in
              ignore e;
              pad_to_shape copied_chunks.(i) ~offset:[ s ] ~shape:[ numel ])
        in
        Some (reshape (U.usum padded) shape)
  | _ -> None

(* create_allreduce_function *)

let create_allreduce_function buf ~op ~device ~dtype ~shape ?output () =
  let output =
    match output with
    | Some o -> o
    | None ->
        (* A shaped Invalid placeholder cloned onto [device]: a fresh flat
           output buffer viewed at [shape], initialised by storing the
           (deviceless) Invalid const, so no cross-device copy is needed. *)
        let shape_node = emit_shape shape in
        let numel = List.fold_left ( * ) 1 shape in
        let invalid_shaped =
          U.const_of_dtype ~shape:shape_node (Dtype.val_of dtype)
            U.Const_invalid
        in
        let buffer =
          U.buffer ~slot:(U.fresh_buffer_slot ())
            ~device:(canonicalize_device device)
            ~shape:(emit_shape [ numel ]) ~addrspace:Dtype.Global ~dtype ()
        in
        let view = U.reshape ~src:buffer ~shape:shape_node in
        U.after ~src:view ~deps:[ U.store ~dst:view ~value:invalid_shaped () ]
  in
  (* Build params mirroring the output and source signatures. *)
  let to_ =
    U.param ~slot:0 ~dtype ~shape:(emit_shape shape) ~device
      ~addrspace:Dtype.Global ()
  in
  let src_device =
    match U.device_of buf with Some d -> d | None -> device
  in
  let src =
    U.param ~slot:1 ~dtype ~shape:(emit_shape shape) ~device:src_device
      ~addrspace:Dtype.Global ()
  in
  match handle_allreduce src ~op ~device ~shape with
  | Some result ->
      let assigned =
        U.after ~src:to_ ~deps:[ U.store ~dst:to_ ~value:result () ]
      in
      let sink = U.sink [ assigned ] in
      let info : U.call_info =
        {
          grad_fxn = None;
          metadata = [];
          name = Some "allreduce";
          precompile = true;
          precompile_backward = false;
          aux = None;
        }
      in
      let kernel =
        U.call ~body:sink
          ~args:[ output; U.contiguous ~src:buf () ]
          ~info
      in
      Some (U.after ~src:output ~deps:[ kernel ])
  | None -> None
