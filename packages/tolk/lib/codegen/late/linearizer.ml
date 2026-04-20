(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_ir
module K = Kernel

(* Priority-based topological sort.

   Assigns each node a priority (run_count, op_priority, extra) and produces
   a linear order that respects dependencies while keeping nodes with similar
   priorities adjacent.  Lower priority numbers appear earlier in output. *)

(* Run count for a range node: the product of its extent. *)
let range_extent node = match K.view node with
  | Range { size; _ } -> (
      match K.const_arg size with Some (Int n) -> Int64.to_int n | _ -> 1)
  | _ -> 1

(* Priority triple: (run_count, op_priority, extra).
   Constructor order matters: OCaml's generic [compare] on this type gives
   the same lexicographic ordering as tinygrad's Python tuple comparison. *)
type extra = No_extra | Idx of int | Name of string

let priority_of live node =
  let run_count = List.fold_left (fun acc r -> acc * range_extent r) 1
    (match K.Ref_tbl.find_opt live node with Some rs -> rs | None -> []) in
  let op_pri, extra = match K.view node with
    | Param { idx; _ }       -> -20, Idx idx
    | Define_var { name; _ } -> -19, Name name
    | Define_local _         -> -18, No_extra
    | Define_reg _           -> -17, No_extra
    | Load _                 ->  -1, No_extra
    | Store _                ->   1, No_extra
    | Range _                ->   5, No_extra
    | End _                  ->  -5, No_extra
    | _                      ->   0, No_extra
  in
  (run_count, op_pri, extra)

(* Heap for priority extraction during toposort.
   Keys are unique (assigned by List.iteri), so int comparison suffices. *)
module Heap = Set.Make (struct
  type t = int * K.t
  let compare (a, _) (b, _) = compare a b
end)

let linearize_order topo =
  let n = List.length topo in
  let sink = match List.rev topo with
    | x :: _ -> x | [] -> failwith "Linearizer: empty topo" in
  let live = K.live_ranges_tbl sink in
  (* Assign priorities and compute ideal ordering *)
  let priorities = K.Ref_tbl.create n in
  List.iter (fun u ->
    K.Ref_tbl.replace priorities u (priority_of live u)) topo;
  let nkey = K.Ref_tbl.create n in
  List.iteri (fun i u -> K.Ref_tbl.replace nkey u i)
    (List.stable_sort (fun a b ->
       compare (K.Ref_tbl.find priorities a) (K.Ref_tbl.find priorities b))
       topo);
  (* Compute out-degrees *)
  let out_degree = K.Ref_tbl.create n in
  List.iter (fun u ->
    List.iter (fun s ->
      let d = match K.Ref_tbl.find_opt out_degree s with
        | Some d -> d | None -> 0 in
      K.Ref_tbl.replace out_degree s (d + 1))
      (K.children u)) topo;
  (* Heap-based toposort: work backwards from sink, release nodes when
     all consumers are placed, prefer nodes closest to ideal position. *)
  let get_nkey u = match K.Ref_tbl.find_opt nkey u with
    | Some k -> k | None -> 0 in
  let heap = ref (Heap.singleton (- get_nkey sink, sink)) in
  let result = ref [] in
  while not (Heap.is_empty !heap) do
    let ((_, u) as elt) = Heap.min_elt !heap in
    heap := Heap.remove elt !heap;
    result := u :: !result;
    List.iter (fun v ->
      let d = (match K.Ref_tbl.find_opt out_degree v with
        | Some d -> d | None -> 0) - 1 in
      K.Ref_tbl.replace out_degree v d;
      if d = 0 then heap := Heap.add (- get_nkey v, v) !heap)
      (K.children u)
  done;
  !result

(* Control-flow context: ordering edges between sibling loops.

   For each pair of sibling END nodes (loops nested under the same parent),
   adds an edge from the later loop's RANGE to the earlier loop's END,
   ensuring sequential emission of loops that must not interleave. *)

let end_range node = match K.view node with
  | End { ranges = [r]; _ } when K.is_range r -> Some r
  | _ -> None

type cfg_context = { edges : K.t K.Ref_tbl.t }

let build_cfg_context topo =
  let n = List.length topo in
  (* Phase 1: compute transitive deps and find nesting relationships. *)
  let deps = K.Ref_tbl.create n in
  let nesting = K.Ref_tbl.create 32 in
  List.iter (fun node ->
    let cdeps = K.Ref_tbl.create 16 in
    List.iter (fun child ->
      match K.Ref_tbl.find_opt deps child with
      | Some s -> K.Ref_tbl.iter (fun k () -> K.Ref_tbl.replace cdeps k ()) s
      | None -> ())
      (K.children node);
    (match K.view node with
     | End _ | Sink _ ->
         K.Ref_tbl.iter (fun x () ->
           match K.view x with
           | End _ when not (K.Ref_tbl.mem nesting x) ->
               let is_nested = match K.view node with
                 | Sink _ -> true
                 | _ -> (match end_range node, K.Ref_tbl.find_opt deps x with
                   | Some rr, Some xd -> K.Ref_tbl.mem xd rr
                   | _ -> false) in
               if is_nested then K.Ref_tbl.replace nesting x node
           | _ -> ()) cdeps
     | _ -> ());
    (match K.view node with
     | Range _ | End _ -> K.Ref_tbl.replace cdeps node ()
     | _ -> ());
    K.Ref_tbl.replace deps node cdeps) topo;
  (* Phase 2: group siblings and build ordering edges. *)
  let siblings = K.Ref_tbl.create 32 in
  K.Ref_tbl.iter (fun child parent ->
    let cur = match K.Ref_tbl.find_opt siblings parent with
      | Some l -> l | None -> [] in
    K.Ref_tbl.replace siblings parent (child :: cur)) nesting;
  let edges = K.Ref_tbl.create 16 in
  K.Ref_tbl.iter (fun parent ends ->
    let dep_count node =
      match K.Ref_tbl.find_opt deps node with
      | Some nd ->
          List.fold_left (fun acc u ->
            if K.Ref_tbl.mem nd u then acc + 1 else acc) 0 ends
      | None -> 0 in
    let order = List.sort (fun a b -> compare (dep_count a) (dep_count b)) ends in
    let add_edge rn pred =
      assert (not (K.in_backward_slice rn pred));
      K.Ref_tbl.replace edges rn pred in
    let rec chain prev = function
      | y :: ys ->
          (match end_range y with
           | Some rr -> add_edge rr prev; chain y ys
           | None -> chain prev ys)
      | [] -> () in
    (match K.view parent with
     | Sink _ -> (match order with x :: rest -> chain x rest | [] -> ())
     | _ -> (match end_range parent with
       | Some rr -> chain rr order | None -> ())))
    siblings;
  { edges }

(* Split multi-range END into nested single-range ENDs.
   Extracts actual RANGE nodes from the ranges' dependency graph, sorts
   by axis (descending), and nests innermost-first. *)
let do_split_ends node = match K.view node with
  | End { value; ranges } ->
      let result =
        K.toposort (K.sink ranges)
        |> List.filter K.is_range
        |> List.sort (fun a b -> compare (K.range_axis b) (K.range_axis a))
        |> List.fold_left (fun v r -> K.end_ ~value:v ~ranges:[r] ()) value in
      if result == node then None else Some result
  | _ -> None

let pm_split_ends root = K.graph_rewrite do_split_ends root

(* Kernel -> Program emission *)

module P = Program

(* Resolve the dtype of an After/Group/End chain (transparent wrappers). *)
let rec after_dtype node = match K.view node with
  | Barrier | Store _ -> Some Dtype.void
  | End { value; _ } | After { src = value; _ } -> after_dtype value
  | Group { srcs = src :: _ } -> after_dtype src
  | Group { srcs = [] } -> None
  | _ -> K.dtype_opt node

(* Walk through Cast/Bitcast/After to find a gated Index. *)
let rec find_gate node = match K.view node with
  | Index { gate = Some g; _ } -> Some g
  | After { src; _ } | Cast { src; _ } | Bitcast { src; _ } -> find_gate src
  | _ -> None

type emitter = {
  builder : P.builder;
  k2p : P.id K.Ref_tbl.t;           (* kernel node -> program id *)
  mutable open_ranges : K.t list;    (* ranges opened but not yet closed *)
}

let lookup em node =
  match K.Ref_tbl.find_opt em.k2p node with
  | Some id -> id
  | None -> failwith "Linearizer: missing kernel ref mapping"

let emit_instr em node instr =
  let id = P.emit em.builder instr in
  K.Ref_tbl.replace em.k2p node id

let maps em = List.map (lookup em)

(* Alias: transparent node maps to another node's program id. *)
let alias em node target =
  K.Ref_tbl.replace em.k2p node (lookup em target)

(* Open a range: emit if not yet emitted, track as open. *)
let ensure_range em node =
  match K.Ref_tbl.find_opt em.k2p node with
  | Some id -> id
  | None -> match K.view node with
    | Range { size; dtype; axis; sub; kind } ->
        em.open_ranges <- node :: em.open_ranges;
        emit_instr em node
          (Range { size = lookup em size; dtype; axis; sub; kind });
        lookup em node
    | _ -> failwith "Linearizer: expected Range node"

(* Resolve a child: ranges are opened lazily, everything else is looked up. *)
let resolve em node = match K.view node with
  | Range _ -> ensure_range em node
  | _ -> lookup em node

let emit em node =
  let m = lookup em and ms = maps em in
  match K.view node with
  (* Transparent: alias to source, produce no instruction. *)
  | Sink _ -> ()
  | Group { srcs = src :: _ } -> alias em node src
  | Group { srcs = [] } -> failwith "Linearizer: empty Group"
  | After { src; _ } when K.is_ptr src -> alias em node src
  | After { src; deps } ->
      let dtype = match after_dtype src with
        | Some dt -> Dtype.val_of dt | None -> failwith "Linearizer: After src has no dtype" in
      emit_instr em node (After { src = m src; deps = ms deps; dtype })

  (* Range lifecycle *)
  | Range _ -> ignore (ensure_range em node)
  | End { value; ranges = [] } -> alias em node value
  | End { value; ranges = [range] } ->
      let dep = resolve em value in
      let range_id = ensure_range em range in
      ignore (P.emit em.builder (End_range { dep; range = range_id }));
      em.open_ranges <-
        List.filter (fun r -> not (r == range)) em.open_ranges;
      K.Ref_tbl.replace em.k2p node dep
  | End _ -> failwith "Linearizer: End must have 0 or 1 range after split"

  (* Gated store: wrap in If/Endif *)
  | Store { dst; value; _ } -> (
      match find_gate dst with
      | Some gate ->
          let gate_id = m gate and dst_id = m dst in
          let if_id = P.emit em.builder
            (If { cond = gate_id; idx_for_dedup = dst_id }) in
          emit_instr em node (Store { dst = dst_id; value = m value });
          ignore (P.emit em.builder (Endif { if_ = if_id }))
      | None ->
          emit_instr em node (Store { dst = m dst; value = m value }))

  (* 1:1 translations *)
  | Param { idx; dtype } ->
      emit_instr em node (Param { idx; dtype })
  | Param_image { idx; dtype; width; height } ->
      emit_instr em node (Param_image { idx; dtype; width; height })
  | Define_local { size; dtype } ->
      emit_instr em node (Define_local { size; dtype })
  | Define_reg { size; dtype; _ } ->
      emit_instr em node (Define_reg { size; dtype })
  | Define_var { name; lo; hi; dtype } ->
      emit_instr em node (Define_var { name; lo; hi; dtype })
  | Const { value; dtype } ->
      emit_instr em node (Const { value; dtype })
  | Index { ptr; idxs; gate; dtype = Dtype.Ptr pty } ->
      emit_instr em node
        (Index { ptr = m ptr; idxs = ms idxs;
                 gate = Option.map m gate; dtype = pty })
  | Index { dtype = Dtype.Val _; _ } ->
      failwith "Linearizer: Index must be ptr-typed after pm_add_loads"
  | Load { src; alt; dtype } ->
      let has_gate = find_gate src <> None in
      if has_gate && alt = None then
        failwith "Linearizer: gated loads require an alt value before linearize";
      if (not has_gate) && alt <> None then
        failwith "Linearizer: Load alt requires gated Index";
      emit_instr em node
        (Load { src = m src; alt = Option.map m alt; dtype })
  | Unary { op; src; dtype } ->
      emit_instr em node (Unary { op; src = m src; dtype })
  | Binary { op; lhs; rhs; dtype } ->
      emit_instr em node
        (Binary { op; lhs = m lhs; rhs = m rhs; dtype })
  | Ternary { op; a; b; c; dtype } ->
      emit_instr em node
        (Ternary { op; a = m a; b = m b; c = m c; dtype })
  | Cast { src; dtype } ->
      emit_instr em node
        (Cast { src = m src; dtype = Dtype.val_of dtype })
  | Bitcast { src; dtype } ->
      emit_instr em node (Bitcast { src = m src; dtype })
  | Vectorize { srcs; dtype } ->
      emit_instr em node
        (Vectorize { srcs = ms srcs; dtype = Dtype.val_of dtype })
  | Gep { src; idxs; dtype } ->
      emit_instr em node (Gep { src = m src; idxs; dtype })
  | Barrier -> emit_instr em node Barrier
  | Special { dim; size; dtype } ->
      emit_instr em node (Special { dim; size = m size; dtype })
  | Wmma { name; a; b; c; dtype; dims; dtype_in; dtype_out;
           device; threads; upcast_axes; reduce_axes } ->
      emit_instr em node
        (Wmma { name; a = m a; b = m b; c = m c; dtype;
                dims; dtype_in; dtype_out; device; threads;
                upcast_axes; reduce_axes })
  | Custom { fmt; args } ->
      emit_instr em node (Custom { fmt; args = ms args })
  | Custom_inline { fmt; args; dtype } ->
      emit_instr em node (Custom_inline { fmt; args = ms args; dtype })

  (* Must be lowered before linearization *)
  | Invalid_index _ | Vconst _ | Ptrcat _ | Vcat _
  | Reduce _ | Unroll _ | Contract _ | Bufferize _ ->
      failwith ("Linearizer: " ^ K.view_op_name (K.view node)
                ^ " must be lowered before linearize")

(* Add control-flow edges: RANGE nodes gain a dependency on the
   predecessor END/RANGE determined by build_cfg_context. *)
let add_control_flow cfg node = match K.view node with
  | Range _ -> (
      match K.Ref_tbl.find_opt cfg.edges node with
      | Some pred ->
          let children = K.children node in
          Some (K.replace node ~children:(children @ [pred]) ())
      | None -> None)
  | _ -> None

let pm_add_control_flow sink =
  let cfg = build_cfg_context (K.toposort sink) in
  K.graph_rewrite ~name:"add control flow" (add_control_flow cfg) sink

(* Priority-based topological ordering followed by Kernel → Program emission.
   The input must already have split Ends and control-flow edges applied. *)

let linearize sink =
  let topo = K.toposort sink in
  let order = linearize_order topo in
  let em = {
    builder = P.create ();
    k2p = K.Ref_tbl.create (List.length topo);
    open_ranges = [];
  } in
  List.iter (emit em) order;
  if em.open_ranges <> [] then
    failwith "Linearizer: unclosed ranges after emission (missing End?)";
  P.finish em.builder
