(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_ir
module K = Kernel

(* Index dtype lowering *)

let is_index_dtype dt = Dtype.equal (Dtype.scalar_of dt) Dtype.index

let concrete_index_dtype dt =
  if is_index_dtype dt then Dtype.vec Dtype.int32 (Dtype.count dt) else dt

let lower_index_dtype node =
  match K.view node with
  | Cast { dtype; _ }
    when is_index_dtype (Dtype.any_to_val dtype) ->
      Some (K.replace node ~dtype:(concrete_index_dtype (Dtype.any_to_val dtype)) ())
  | Const { dtype; _ } | Range { dtype; _ }
  | Unary { dtype; _ } | Binary { dtype; _ } | Ternary { dtype; _ }
  | Gep { dtype; _ }
  | Special { dtype; _ } | Define_var { dtype; _ }
    when is_index_dtype dtype ->
      Some (K.replace node ~dtype:(concrete_index_dtype dtype) ())
  | Vectorize { dtype; _ }
    when is_index_dtype (Dtype.any_to_val dtype) ->
      Some (K.replace node ~dtype:(concrete_index_dtype (Dtype.any_to_val dtype)) ())
  | Invalid_index { dtype } ->
      let scalar_zero = K.const (Const.int (Dtype.scalar_of (concrete_index_dtype dtype)) 0) in
      if Dtype.count dtype > 1 then Some (K.broadcast scalar_zero (Dtype.count dtype))
      else Some scalar_zero
  | _ -> None

let strip_index_cast_children node =
  match K.view node with
  | Sink _ | End _ ->
      let children = K.children node in
      let stripped =
        List.map
          (fun c ->
            match K.view c with
            | Cast { src; dtype = Dtype.T dt } when is_index_dtype dt -> src
            | Cast { src; dtype = Dtype.T dt } -> begin
                (* Also strip identity casts where src dtype = cast dtype
                   (left over after lowering index→int) *)
                match K.dtype src with
                | Some src_dt when Dtype.equal src_dt dt -> src
                | _ -> c end
            | _ -> c)
          children
      in
      if List.for_all2 (fun a b -> a == b) children stripped then None
      else Some (K.replace node ~children:stripped ())
  | _ -> None

(* Remove hanging index casts from INDEX children.
   INDEX(buf, idx.cast(int)) → INDEX(buf, idx) when idx is already integral. *)
let strip_index_cast_from_index node =
  match K.view node with
  | Index { ptr; idxs; gate; dtype = Dtype.P _ } ->
      let stripped = List.map (fun idx ->
        match K.view idx with
        | Cast { src; dtype = Dtype.T dt } when Dtype.is_int dt ->
            (match K.dtype src with
             | Some src_dt when Dtype.is_int src_dt -> src
             | _ -> idx)
        | _ -> idx) idxs
      in
      if List.for_all2 (fun a b -> a == b) idxs stripped then None
      else Some (K.index ~ptr ~idxs:stripped ?gate ())
  | _ -> None

let pm_lower_index_dtype_rule =
  K.first_match [ lower_index_dtype; strip_index_cast_children; strip_index_cast_from_index ]

(* Split multi-range End nodes into nested single-range Ends.
   Filters to actual RANGE nodes, sorts by axis (reverse), and builds a
   nested chain. If no ranges remain (e.g., all were replaced by
   SPECIAL/DEFINE_VAR), the End is stripped entirely. *)
let range_sort_key node =
  match K.view node with
  | Range { axis; kind; _ } -> (axis, kind)
  | _ -> (max_int, Axis_kind.Placeholder)

let split_end node =
  match K.view node with
  | End { value; ranges } ->
      let actual_ranges = List.filter K.is_range ranges in
      let sorted =
        List.sort (fun a b ->
          compare (range_sort_key b) (range_sort_key a)) actual_ranges
      in
      let result =
        List.fold_left (fun v r -> K.end_ ~value:v ~ranges:[ r ] ()) value sorted
      in
      if result == node then None else Some result
  | _ -> None

(* Codegen-level bufferize lowering.
   Converts Bufferize nodes (created by fix_group_for_reduce) to
   DEFINE_LOCAL + INDEX + STORE + END + BARRIER. *)

let bufferize_range_size ranges =
  List.fold_left (fun acc r ->
    match K.view r with
    | K.Range { size; _ } ->
        (match K.view size with
         | K.Const { value; _ } ->
             (match Const.view value with
              | Const.Int i -> acc * Int64.to_int i
              | _ -> failwith "bufferize_range_size: non-integer range extent")
         | _ -> failwith "bufferize_range_size: non-constant range extent")
    | _ -> acc) 1 ranges

let add_buffers_local_rule (node : K.t) : K.t option =
  match K.view node with
  | K.Bufferize { src; ranges; dtype; _ } ->
      let size = bufferize_range_size ranges in
      if size <= 0 then None
      else begin
        let sorted_rngs =
          List.sort (fun a b ->
            compare (K.range_axis a) (K.range_axis b))
            ranges
        in
        let range_ids =
          List.filter K.is_range sorted_rngs
        in
        let ptr_dt =
          Dtype.ptr_of (Dtype.base dtype) ~addrspace:Dtype.Local ~size
        in
        let def_local = K.define_local ~size ~dtype:ptr_dt in
        let idx = K.index ~ptr:def_local ~idxs:sorted_rngs () in
        let store = K.store ~dst:idx ~value:src ~ranges:[] in
        let ended = K.end_ ~value:store ~ranges:range_ids () in
        let bar = K.after ~src:K.barrier ~deps:[ ended ] in
        Some (K.after ~src:def_local ~deps:[ bar ])
      end
  | _ -> None

(* Pipeline *)

let lower ren sink =
  (* 8: postopt symbolic *)
  let sink = K.graph_rewrite ~name:"postopt symbolic" (K.first_match [ Symbolic.sym ]) sink in
  (* 9: expander *)
  let sink = Expander.expand sink in
  let dbg = Device.Context.get (Device.Context.int ~name:"DEBUG" ~default:0) in
  (* 10: add local buffers — lower remaining Bufferize nodes *)
  let sink = K.graph_rewrite ~name:"add local buffers" (K.first_match [ add_buffers_local_rule ]) sink in
  (* 11: remove reduce *)
  let sink = Devectorizer.pm_reduce sink in
  (* 12: gpu dims *)
  let sink = Gpudims.pm_add_gpudims ren sink in
  (* 13: add loads *)
  let sink = Devectorizer.pm_add_loads sink in
  (* 14: devectorize *)
  let sink =
    K.graph_rewrite ~name:"devectorize"
      (K.first_match [
        Symbolic.sym;
        Devectorizer.pm_devectorize_rule;
        Devectorizer.load_store_folding_rule;
        Devectorizer.pm_correct_load_store_rule ren;
        Devectorizer.load_store_indexing_rule;
      ])
      sink
  in
  (* 15: lower index dtype + load_store_indexing + gep_pushing *)
  let sink =
    K.graph_rewrite ~name:"lower all index dtypes"
      (K.first_match [
        pm_lower_index_dtype_rule;
        Devectorizer.load_store_indexing_rule;
        Symbolic.gep_pushing;
      ])
      sink
  in
  (* 16: post-index symbolic *)
  let sink = K.graph_rewrite ~name:"post index symbolic" (K.first_match [ Symbolic.sym ]) sink in
  (* 17: pre_matcher (renderer-specific) *)
  let sink =
    match Renderer.pre_matcher ren with
    | Some pm -> K.graph_rewrite ~name:"pre_matcher" pm sink
    | None -> sink
  in
  (* 18-21: decompositions *)
  let ops =
    let base = Renderer.supported_ops ren in
    let disable_fast_idiv =
      Device.Context.get (Device.Context.int ~name:"DISABLE_FAST_IDIV" ~default:0) <> 0
    in
    { base with disable_fast_idiv }
  in
  let pm_decomp =
    K.first_match [
      Symbolic.symbolic_simple;
      Decompositions.get_late_rewrite_patterns ops;
    ]
  in
  let sink = K.graph_rewrite ~name:"decompositions" pm_decomp sink in
  (* 19-20: dtype decompositions (long + float) *)
  let sink =
    if Renderer.supports_dtype ren Dtype.int64 then sink
    else K.graph_rewrite ~name:"decomp long -> int" Decompositions.pm_long_decomp sink
  in
  let sink =
    List.fold_left
      (fun sink (fr, to_) ->
        let ctx : Decompositions.float_decomp_ctx = { from_dtype = fr; to_dtype = to_ } in
        K.graph_rewrite (Decompositions.pm_float_decomp ctx) sink)
      sink (Renderer.emulated_float_dtypes ren)
  in
  if dbg >= 6 then K.print_uops ~label:"decomp dtypes" sink;
  (* 21: transcendentals *)
  let sink =
    K.graph_rewrite ~name:"transcendental"
      (K.first_match [
        Symbolic.symbolic_simple;
        Decompositions.get_transcendental_patterns ops;
      ])
      sink
  in
  (* 22: final rewrite — decomp + render + extra_matcher + split_ends *)
  let pm_final =
    let extra = match Renderer.extra_matcher ren with Some m -> [m] | None -> [] in
    K.first_match ([pm_decomp; Devectorizer.pm_render_rule] @ extra @ [split_end])
  in
  let sink = K.graph_rewrite ~name:"final rewrite" pm_final sink in
  (* 23: add control flow — explicit ordering deps between RANGE nodes are
     added inline by the linearizer rather than as a separate pass. Print the
     final state here for debug inspection. *)
  if dbg >= 6 then K.print_uops ~label:"add control flow" sink;
  sink

let lower_and_linearize ren sink = Linearizer.linearize (lower ren sink)

let compile dev ren sink =
  Device.compile_program dev (lower_and_linearize ren sink)
