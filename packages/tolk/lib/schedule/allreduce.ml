(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Multi-device collective reduction.

   Implements naive, ring, and all-to-all allreduce strategies for reducing
   buffers across multiple devices. *)

open Tolk_ir

module T = Tensor

(* Environment *)

let ring_var = Helpers.Context_var.int ~key:"RING" ~default:0
let all2all_var = Helpers.Context_var.int ~key:"ALL2ALL" ~default:0

let ring_allreduce_threshold =
  Helpers.Context_var.int ~key:"RING_ALLREDUCE_THRESHOLD" ~default:256_000

(* Shape encoding

   Shapes and bounds are tensor nodes: a single dim is a scalar const,
   multiple dims become a vectorize of scalar consts. *)

let dim d = T.const (Const.int Dtype.Val.index d) Dtype.index

let emit_shape = function
  | [d] -> dim d
  | dims -> T.vectorize ~srcs:(List.map dim dims)

let emit_pairs pairs =
  (emit_shape (List.map fst pairs), emit_shape (List.map snd pairs))

(* Int-list wrappers over tensor-node shape/bounds APIs. *)

let reshape src dims = T.reshape ~src ~shape:(emit_shape dims)

let shrink src bounds =
  let before, after = emit_pairs bounds in
  T.shrink ~src ~before ~after

let pad src padding =
  let before, after = emit_pairs padding in
  T.pad ~src ~before ~after

let copy_to_device src dev = T.copy ~src ~device:(T.device (Single dev)) ()

(* Reduction *)

let reduce op lhs rhs = T.binary ~op:(op :> Op.binary) ~lhs ~rhs

let fold_reduce op = function
  | [] -> failwith "fold_reduce: empty list"
  | x :: xs -> List.fold_left (reduce op) x xs

(* handle_allreduce *)

let handle_allreduce buf ~op ~device =
  let devices = T.compute_devices buf in
  match devices buf with
  | Some (Multi devs) ->
      let devs = Array.of_list devs in
      let ndev = Array.length devs in
      let shapes = T.compute_shapes buf in
      let shape = match shapes buf with
        | Some s -> s
        | None -> failwith "handle_allreduce: buf has no shape"
      in
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
        not use_all2all
        && (ring >= 2 || (ndev > 2 && numel > threshold && ring >= 1))
      in
      let buf = T.contiguous ~src:buf () in
      if not use_ring && not use_all2all then
        (* Naive: copy every shard to the target device and reduce. *)
        let shards = List.init ndev (fun i ->
          T.copy ~src:(T.mselect ~src:buf ~index:i) ~device ()) in
        Some (fold_reduce op shards)
      else begin
        (* Divide into ndev chunks, aligned to the largest power-of-2 factor
           (up to 32) that divides numel. Larger chunks go to earlier devices. *)
        let factor =
          match List.find_opt (fun f -> numel mod f = 0) [32; 16; 8; 4; 2] with
          | Some f -> f | None -> 1
        in
        let base = numel / factor / ndev in
        let left = numel / factor mod ndev in
        let chunks = Array.init ndev (fun i ->
          (if i < left then base + 1 else base) * factor) in
        (* Prefix-sum to get (start, end) pairs. *)
        let bounds =
          let pos = ref 0 in
          Array.map (fun sz -> let s = !pos in pos := s + sz; (s, s + sz)) chunks
        in
        (* Reduce-scatter: each device ends up with one fully-reduced chunk. *)
        let reduced_chunks = Array.mapi (fun i (s, e) ->
          if use_all2all then
            (* All-to-all: gather chunk [s,e) from every device onto device i. *)
            let chunks_on_i = List.init ndev (fun j ->
              let shard = T.mselect ~src:buf ~index:j in
              copy_to_device (shrink (reshape shard [numel]) [(s, e)]) devs.(i)) in
            fold_reduce op chunks_on_i
          else begin
            (* Ring: walk chunk around the ring, accumulating at each hop. *)
            let flat = reshape buf [numel] in
            let chunk = shrink flat [(s, e)] in
            let reduced = ref (shrink flat [(s, e)]) in
            for step = 0 to ndev - 2 do
              let src_idx = (i + step) mod ndev in
              let dest_idx = (i + step + 1) mod ndev in
              (* On the first step, reduced is still multi-device (inherits from
                 buf) and needs mselect. After that it lives on a single device. *)
              let r = if step = 0 then T.mselect ~src:!reduced ~index:src_idx
                      else !reduced in
              let cp = copy_to_device r devs.(dest_idx) in
              let ch = copy_to_device (T.mselect ~src:chunk ~index:dest_idx)
                         devs.(dest_idx) in
              reduced := reduce op cp ch
            done;
            !reduced
          end) bounds
        in
        (* Allgather: broadcast each reduced chunk to all devices. *)
        let copied_chunks = Array.mapi (fun i rc ->
          match T.view device with
          | Device { device = Single target } ->
              (* Target is a single device — just copy there. *)
              copy_to_device rc target
          | _ when use_all2all ->
              (* All-to-all: copy to every device and stack. *)
              T.mstack ~srcs:(List.init ndev (fun j ->
                copy_to_device rc devs.(j)))
          | _ ->
              (* Ring: chain copies around the ring, then reorder. *)
              let chain = Array.make ndev rc in
              let current = ref rc in
              for step = 0 to ndev - 2 do
                current := copy_to_device !current devs.((i + step) mod ndev);
                chain.(step + 1) <- !current
              done;
              T.mstack ~srcs:(List.init ndev (fun j ->
                chain.((j - i + 1 + ndev) mod ndev)))) reduced_chunks
        in
        (* Reassemble: pad each chunk back to full size and sum. *)
        let padded = List.init ndev (fun i ->
          let (s, e) = bounds.(i) in
          pad copied_chunks.(i) [(s, numel - e)]) in
        Some (reshape (fold_reduce `Add padded) shape)
      end
  | _ -> None

(* create_allreduce_function *)

let create_allreduce_function buf ~op ~device ~dtype ~shape ?output () =
  let output = match output with
    | Some o -> o
    | None ->
        let size = List.fold_left ( * ) 1 shape in
        let unique = T.noop ~dtype () in
        T.contiguous ~src:(reshape (T.buffer ~unique ~device ~size ~dtype) shape) ()
  in
  (* Build params mirroring the output and source signatures. *)
  let to_ = T.param ~slot:0 ~dtype ~shape:(emit_shape shape) ~device () in
  let buf_shapes = T.compute_shapes buf in
  let buf_devices = T.compute_devices buf in
  let src_shape = Option.value ~default:shape (buf_shapes buf) in
  let src_device = match buf_devices buf with
    | Some dev -> T.device dev
    | None -> device
  in
  let src = T.param ~slot:1 ~dtype ~shape:(emit_shape src_shape)
              ~device:src_device () in
  match handle_allreduce src ~op ~device with
  | Some result ->
      let assigned = T.assign ~target:to_ ~value:result () in
      let sink = T.sink [assigned] in
      let info : T.call_info =
        { grad_fxn = None; metadata = []; name = Some "allreduce";
          precompile = true }
      in
      let kernel = T.call ~callee:(Ref sink) ~args:[output; T.contiguous ~src:buf ()]
                     ~info ~dtype in
      Some (T.after ~src:output ~deps:[kernel])
  | None -> None
