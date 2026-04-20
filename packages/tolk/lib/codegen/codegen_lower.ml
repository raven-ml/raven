(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Codegen lowering — all passes after optimization, up to linearization.

   This module has no dependency on Search, Postrange, or Heuristic,
   so beam search can safely call [lower_and_linearize] without cycles. *)

open Tolk_ir
module K = Kernel

(* Environment *)

let debug = Helpers.getenv "DEBUG" 0
let devectorize = Helpers.getenv "DEVECTORIZE" 1

(* Index dtype lowering — replace abstract index dtype with concrete int32. *)

let is_index_dtype dt = Dtype.equal (Dtype.scalarize dt) Dtype.index

let concrete_index_dtype dt =
  if is_index_dtype dt then Dtype.vec (Dtype.count dt) Dtype.int32 else dt

let lower_index_dtype_rule node =
  match K.view node with
  | Cast { dtype; _ } when is_index_dtype dtype ->
      Some (K.replace node ~dtype:(concrete_index_dtype dtype) ())
  | Const { dtype; _ } | Range { dtype; _ }
  | Unary { dtype; _ } | Binary { dtype; _ } | Ternary { dtype; _ }
  | Gep { dtype; _ }
  | Special { dtype; _ } | Define_var { dtype; _ }
    when is_index_dtype (Dtype.Val dtype) ->
      Some (K.replace node ~dtype:(concrete_index_dtype (Dtype.Val dtype)) ())
  | Vectorize { dtype; _ } when is_index_dtype dtype ->
      Some (K.replace node ~dtype:(concrete_index_dtype dtype) ())
  | Invalid_index { dtype } ->
      let cdt = concrete_index_dtype (Dtype.Val dtype) in
      let scalar_zero = K.const (Const.int (Dtype.val_of (Dtype.scalarize cdt)) 0) in
      if Dtype.Val.count dtype > 1 then Some (K.broadcast scalar_zero (Dtype.Val.count dtype))
      else Some scalar_zero
  | _ -> None

(* Strip index casts from Sink/End children. *)
let strip_index_cast_children node =
  match K.view node with
  | Sink _ | End _ ->
      let children = K.children node in
      let stripped = List.map (fun c -> match K.view c with
        | Cast { src; dtype = Dtype.Val dt } when is_index_dtype (Dtype.Val dt) -> src
        | Cast { src; dtype = Dtype.Val dt } ->
            (match K.dtype_opt src with
             | Some src_dt when Dtype.equal src_dt (Dtype.Val dt) -> src | _ -> c)
        | _ -> c) children in
      if List.for_all2 (fun a b -> a == b) children stripped then None
      else Some (K.replace node ~children:stripped ())
  | _ -> None

(* Strip index casts from INDEX children. *)
let strip_index_cast_from_index node =
  match K.view node with
  | Index { ptr; idxs; gate; dtype = Dtype.Ptr _ } ->
      let stripped = List.map (fun idx -> match K.view idx with
        | Cast { src; dtype = Dtype.Val dt } when Dtype.Val.is_int dt ->
            (match K.dtype_opt src with
             | Some src_dt when Dtype.is_int src_dt -> src | _ -> idx)
        | _ -> idx) idxs in
      if List.for_all2 (fun a b -> a == b) idxs stripped then None
      else Some (K.index ~ptr ~idxs:stripped ?gate ())
  | _ -> None

let pm_lower_index_dtype =
  K.first_match [lower_index_dtype_rule; strip_index_cast_children;
                 strip_index_cast_from_index]

(* Bufferize lowering — convert Bufferize nodes to
   DEFINE_LOCAL + INDEX + STORE + END + BARRIER. *)
let bufferize_range_size ranges =
  List.fold_left (fun acc r -> match K.view r with
    | Range { size; _ } ->
        (match K.const_arg size with
         | Some (Int i) -> acc * Int64.to_int i
         | _ -> failwith "bufferize_range_size: non-constant range extent")
    | _ -> acc) 1 ranges

let add_buffers_local_rule node =
  match K.view node with
  | Bufferize { src; ranges; dtype; _ } ->
      let size = bufferize_range_size ranges in
      if size <= 0 then None
      else
        let sorted_rngs = List.sort (fun a b ->
          compare (K.range_axis a) (K.range_axis b)) ranges in
        let range_ids = List.filter K.is_range sorted_rngs in
        let ptr_dt = Dtype.Ptr.create (Dtype.Ptr.base dtype) ~addrspace:Dtype.Local ~size in
        let def_local = K.define_local ~size ~dtype:ptr_dt in
        let idx = K.index ~ptr:def_local ~idxs:sorted_rngs () in
        let store = K.store ~dst:idx ~value:src ~ranges:[] in
        let ended = K.end_ ~value:store ~ranges:range_ids () in
        let bar = K.after ~src:K.barrier ~deps:[ended] in
        Some (K.after ~src:def_local ~deps:[bar])
  | _ -> None

(* Lower an optimized kernel AST to a form ready for linearization.
   Runs expansion, devectorization, GPU dimension mapping, index dtype
   concretization, operation decomposition, and renderer-specific rewrites. *)
let lower ren sink =
  (* Normalize symbolic expressions before expansion. *)
  let sink = K.graph_rewrite ~name:"postopt symbolic"
    (K.first_match [Symbolic.sym; Symbolic.pm_move_where_on_load]) sink in

  (* Expand UPCAST/UNROLL ranges into explicit vector operations. *)
  let sink = Expander.expand sink in

  (* Convert Bufferize nodes into DEFINE_LOCAL + INDEX + STORE + END + BARRIER. *)
  let sink = K.graph_rewrite ~name:"add local buffers"
    (K.first_match [add_buffers_local_rule]) sink in

  (* Scalarize reductions: lower Reduce nodes and push GEPs through. *)
  let sink = Devectorizer.pm_reduce sink in

  (* Map logical ranges to physical GPU grid dimensions (SPECIAL nodes). *)
  let sink = Gpudims.pm_add_gpudims ren sink in

  (* Insert explicit Load/Store operations from Index nodes. *)
  let sink = Devectorizer.pm_add_loads sink in

  (* Scalarize remaining vector operations for targets without native vectors. *)
  let sink = Devectorizer.pm_devectorize ren sink in

  (* Lower image Param_image loads/stores to read_imagef/write_imagef. *)
  let sink = Images.rewrite ren sink in

  (* Replace abstract index dtype with concrete int32. *)
  let sink = K.graph_rewrite ~name:"lower all index dtypes"
    (K.first_match [pm_lower_index_dtype; Devectorizer.load_store_indexing;
                    Symbolic.gep_pushing]) sink in
  let sink = K.graph_rewrite ~name:"post index symbolic"
    (K.first_match [Symbolic.symbolic]) sink in

  (* Apply renderer-specific pre-processing if provided. *)
  let sink = match Renderer.pre_matcher ren with
    | Some pm -> K.graph_rewrite ~name:"pre_matcher" pm sink
    | None -> sink in

  (* Decompose compound operations into primitives supported by the target. *)
  let ops =
    let base = Renderer.supported_ops ren in
    let disable_fast_idiv = Helpers.getenv "DISABLE_FAST_IDIV" 0 <> 0 in
    { base with disable_fast_idiv } in
  let pm_decomp = K.first_match [
    Symbolic.symbolic_simple;
    Decompositions.get_late_rewrite_patterns ops] in
  let sink = K.graph_rewrite ~name:"decompositions" pm_decomp sink in

  (* Lower unsupported dtypes: int64 → int32, emulated floats. *)
  let sink =
    if Renderer.supports_dtype ren Dtype.int64 then sink
    else K.graph_rewrite ~name:"decomp long -> int"
           Decompositions.pm_long_decomp sink in
  let sink = List.fold_left (fun sink (fr, to_) ->
    let ctx : Decompositions.float_decomp_ctx =
      { from_dtype = fr; to_dtype = to_ } in
    K.graph_rewrite (Decompositions.pm_float_decomp ctx) sink)
    sink (Renderer.emulated_float_dtypes ren) in

  (* Expand transcendental functions (exp2, log2, sin, etc.). *)
  let sink = K.graph_rewrite ~name:"transcendental"
    (K.first_match [
      Symbolic.symbolic_simple;
      Decompositions.get_transcendental_patterns ops]) sink in

  (* Final cleanup: re-apply decompositions, renderer emit rules, and
     split multi-range End nodes into nested single-range Ends. *)
  let extra = match Renderer.extra_matcher ren with
    | Some m -> [m] | None -> [] in
  let sink = K.graph_rewrite ~name:"final rewrite"
    (K.first_match
       ([pm_decomp; Devectorizer.pm_render_rule] @ extra
        @ [Linearizer.do_split_ends]))
    sink in

  (* Add control-flow ordering edges between sibling loops. *)
  let sink = Linearizer.pm_add_control_flow sink in

  if debug >= 6 then K.print_uops ~label:"lower" sink;
  sink

let lower_and_linearize ren sink =
  Linearizer.linearize (lower ren sink)

let compile dev ren sink =
  Device.compile_program dev (lower_and_linearize ren sink)
