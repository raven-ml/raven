(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_ir

module T = Tensor

(* Environment *)

let ring_var = Device.Context.int ~name:"RING" ~default:0
let all2all_var = Device.Context.int ~name:"ALL2ALL" ~default:0

let ring_allreduce_threshold =
  Device.Context.int ~name:"RING_ALLREDUCE_THRESHOLD" ~default:256_000

(* Shape encoding *)

let emit_shape b dims =
  match dims with
  | [ d ] -> T.const b (Const.int Dtype.index d)
  | dims ->
      let srcs = List.map (fun d -> T.const b (Const.int Dtype.index d)) dims in
      T.emit b (Vectorize { srcs; dtype = Dtype.vec Dtype.index (List.length srcs) })

let emit_pairs b pairs =
  (emit_shape b (List.map fst pairs), emit_shape b (List.map snd pairs))

(* Builder-level operations *)

let reshape b src dims = T.reshape b ~src ~shape:(emit_shape b dims)

let shrink b src bounds =
  let before, after = emit_pairs b bounds in
  T.shrink b ~src ~before ~after

let pad b src padding =
  let before, after = emit_pairs b padding in
  T.pad b ~src ~before ~after

let copy_to_device b src device_name =
  T.copy b ~src ~device:(T.device b (Single device_name)) ()

(* reduce op as binary: all three reduce ops (`Add, `Mul, `Max) are also binary *)
let alu_reduce b op x y = T.binary b ~op:(op :> Op.binary) ~lhs:x ~rhs:y

let fold_reduce b op = function
  | [] -> failwith "fold_reduce: empty list"
  | x :: rest -> List.fold_left (alu_reduce b op) x rest

(* handle_allreduce *)

(* Three-phase collective reduction across multi-device shards.
   Phase 1 (reduce-scatter): partitions the flat buffer into ndev chunks and
   reduces each chunk to its owner device — ring transport sends each shard
   around the ring accumulating partial results, while all-to-all copies every
   chunk directly to the owner and reduces locally.
   Phase 2 (allgather): copies each reduced chunk to the target device(s) —
   ring forwards around the ring, all-to-all broadcasts to all devices.
   Phase 3 (reassemble): pads chunks back to full size, sums them (the pads
   make the non-owned regions zero), and reshapes to the original shape.
   Strategy selection: uses all-to-all or ring when ndev > 2 and numel exceeds
   RING_ALLREDUCE_THRESHOLD; otherwise falls back to naive copy-all-and-reduce. *)
let handle_allreduce b ~shapes ~(devices : T.device option array) ~buf ~red_op
    ~red_device_id =
  match devices.(buf) with
  | Some (Multi devs) ->
      let ndev = List.length devs in
      let shape =
        match shapes.(buf) with
        | Some s -> s
        | None -> failwith "allreduce: buf has no shape"
      in
      let numel = List.fold_left ( * ) 1 shape in
      let threshold = Device.Context.get ring_allreduce_threshold in
      let all2all = Device.Context.get all2all_var in
      let ring = Device.Context.get ring_var in
      let use_all2all =
        all2all >= 2 || (ndev > 2 && numel > threshold && all2all >= 1)
      in
      let use_ring =
        (not use_all2all)
        && (ring >= 2 || (ndev > 2 && numel > threshold && ring >= 1))
      in
      let buf = T.contiguous b ~src:buf () in
      if (not use_ring) && not use_all2all then begin
        (* naive: copy all shards to the target device and reduce *)
        let on_target =
          List.init ndev (fun i ->
              let sel = T.mselect b ~src:buf ~index:i in
              T.copy b ~src:sel ~device:red_device_id ())
        in
        Some (fold_reduce b red_op on_target)
      end
      else begin
        let factor =
          [ 32; 16; 8; 4; 2 ]
          |> List.find_opt (fun f -> numel mod f = 0)
          |> Option.value ~default:1
        in
        let base_size = numel / factor / ndev in
        let left = numel / factor mod ndev in
        let chunk_sizes =
          List.init ndev (fun i ->
              if i < left then (base_size + 1) * factor
              else base_size * factor)
        in
        let chunks =
          let _, pairs =
            List.fold_left
              (fun (pos, acc) sz -> (pos + sz, (pos, pos + sz) :: acc))
              (0, []) chunk_sizes
          in
          List.rev pairs
        in
        (* reduce-scatter *)
        let reduced_chunks =
          List.mapi
            (fun i (s, e) ->
              if use_all2all then begin
                let chunks_on_i =
                  List.init ndev (fun j ->
                      let sel = T.mselect b ~src:buf ~index:j in
                      let r = reshape b sel [ numel ] in
                      let sh = shrink b r [ (s, e) ] in
                      copy_to_device b sh (List.nth devs i))
                in
                fold_reduce b red_op chunks_on_i
              end
              else begin
                let chunk = shrink b (reshape b buf [ numel ]) [ (s, e) ] in
                let reduced = ref (shrink b (reshape b buf [ numel ]) [ (s, e) ]) in
                for step = 0 to ndev - 2 do
                  let src_dev = (i + step) mod ndev in
                  let dest = (i + step + 1) mod ndev in
                  let reduced_src =
                    match devices.(!reduced) with
                    | Some (Multi _) -> T.mselect b ~src:!reduced ~index:src_dev
                    | _ -> !reduced
                  in
                  let dest_dev = T.device b (Single (List.nth devs dest)) in
                  let cp = T.copy b ~src:reduced_src ~device:dest_dev () in
                  let ch_dev = T.device b (Single (List.nth devs dest)) in
                  let ch_src = T.mselect b ~src:chunk ~index:dest in
                  let ch = T.copy b ~src:ch_src ~device:ch_dev () in
                  reduced := alu_reduce b red_op cp ch
                done;
                !reduced
              end)
            chunks
        in
        (* allgather *)
        let copied_chunks =
          List.mapi
            (fun i rc ->
              match T.view (T.finish b) red_device_id with
              | Device { device = Single target } ->
                  copy_to_device b rc target
              | _ ->
                  if use_all2all then
                    T.mstack b ~srcs:(List.map (fun d -> copy_to_device b rc d) devs)
                  else begin
                    let chain = Array.make ndev rc in
                    let current = ref rc in
                    for step = 0 to ndev - 2 do
                      let target = (i + step) mod ndev in
                      current := copy_to_device b !current (List.nth devs target);
                      chain.((step + 1) mod ndev) <- !current
                    done;
                    T.mstack b
                      ~srcs:(List.init ndev (fun j ->
                                 chain.((j - i + 1 + ndev) mod ndev)))
                  end)
            reduced_chunks
        in
        (* reassemble: sum padded chunks, reshape back *)
        let padded =
          List.map2 (fun (s, e) c -> pad b c [ (s, numel - e) ]) chunks copied_chunks
        in
        Some (reshape b (fold_reduce b `Add padded) shape)
      end
  | _ -> None

(* create_allreduce_function *)

let create_allreduce_function b ~shapes ~(devices : T.device option array) ~buf
    ~red_op ~red_device_id ~red_dtype ~red_shape ~red_size ?output () =
  let output =
    match output with
    | Some o -> o
    | None ->
        let noop = T.noop b ~dtype:red_dtype () in
        let out_buf =
          T.buffer b ~unique:noop ~device:red_device_id ~size:red_size
            ~dtype:red_dtype
        in
        reshape b out_buf red_shape
  in
  let to_param =
    T.param b ~slot:0 ~dtype:red_dtype
      ~shape:(emit_shape b red_shape)
      ~device:red_device_id ()
  in
  let src_shape = match shapes.(buf) with Some s -> s | None -> red_shape in
  let buf_dev =
    match devices.(buf) with
    | Some (Single d) -> T.device b (Single d)
    | Some (Multi ds) -> T.device b (Multi ds)
    | None -> red_device_id
  in
  let src_param =
    T.param b ~slot:1 ~dtype:red_dtype
      ~shape:(emit_shape b src_shape)
      ~device:buf_dev ()
  in
  match handle_allreduce b ~shapes ~devices ~buf:src_param ~red_op ~red_device_id with
  | Some result ->
      let assigned = T.assign b ~target:to_param ~value:result () in
      let sink_node = T.sink b [ assigned ] in
      let buf_contig = T.contiguous b ~src:buf () in
      let call_info : T.call_info =
        { grad_fxn = None; metadata = []; name = Some "allreduce"; precompile = true }
      in
      let kernel =
        T.call b ~callee:(Ref sink_node) ~args:[ output; buf_contig ]
          ~info:call_info ~dtype:red_dtype
      in
      Some (T.after b ~src:output ~deps:[ kernel ])
  | None -> None
