(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_ir
module K = Kernel
module P = Program

(* Error strings *)

let err_unlowered kind =
  Printf.sprintf "Linearizer: %s must be lowered before linearize" kind
let err_missing_ref = "Linearizer: missing kernel ref mapping"
let err_not_range = "Linearizer: ensure_range expects a Range node"
let err_empty_group = "Linearizer: empty Group"
let err_after_no_dtype = "Linearizer: After src has no dtype"
let err_end_multi = "Linearizer: End must have 0 or 1 range after split"
let err_unclosed = "Linearizer: unclosed ranges after emission (missing End?)"
let err_gated_no_alt = "Linearizer: gated loads require an alt value before linearize"
let err_alt_no_gate = "Linearizer: Load alt requires gated Index"

(* Split ends *)

let axis_kind_rank = function
  | Axis_kind.Global -> 0 | Warp -> 1 | Local -> 2 | Thread -> 3
  | Loop -> 4 | Group_reduce -> 5 | Reduce -> 6
  | Upcast -> 7 | Unroll -> 8 | Placeholder -> 9

let range_sort_key node =
  match K.view node with
  | Range { axis; kind; _ } -> (axis, axis_kind_rank kind)
  | _ -> (max_int, max_int)

let split_ends root =
  let cache = K.Ref_tbl.create 128 in
  let rec go node =
    match K.Ref_tbl.find_opt cache node with
    | Some n -> n
    | None ->
      let result =
        match K.view node with
        | End { value; ranges } ->
          let value = go value in
          let ranges =
            List.map go ranges
            |> List.sort (fun a b -> compare (range_sort_key b) (range_sort_key a))
          in
          List.fold_left (fun v r -> K.end_ ~value:v ~ranges:[ r ] ()) value ranges
        | _ -> K.replace node ~children:(List.map go (K.children node)) ()
      in
      K.Ref_tbl.replace cache node result;
      result
  in
  go root

(* CFG context *)

let end_range node =
  match K.view node with
  | End { ranges = [ r ]; _ } when K.is_range r -> Some r
  | _ -> None

type cfg_context = { edges : K.t K.Ref_tbl.t }

(* Build CFG ordering edges from the DAG's nesting structure.  End nodes that
   share a parent (sibling loops) are chained so that inner ranges depend on
   the preceding sibling's End, enforcing sequential emission of loops that
   must not be interleaved. *)
let build_cfg_context topo =
  let n = List.length topo in
  let deps = K.Ref_tbl.create n in
  let nesting = K.Ref_tbl.create 32 in
  List.iter
    (fun node ->
      let cdeps = K.Ref_tbl.create 16 in
      List.iter
        (fun child ->
          match K.Ref_tbl.find_opt deps child with
          | Some s -> K.Ref_tbl.iter (fun k () -> K.Ref_tbl.replace cdeps k ()) s
          | None -> ())
        (K.children node);
      (match K.view node with
      | End _ | Sink _ ->
        K.Ref_tbl.iter
          (fun x () ->
            match K.view x with
            | End _ when not (K.Ref_tbl.mem nesting x) ->
              let nest =
                match K.view node, end_range node with
                | Sink _, _ -> true
                | End _, Some rr ->
                  (match K.Ref_tbl.find_opt deps x with
                  | Some xd -> K.Ref_tbl.mem xd rr
                  | None -> false)
                | _ -> false
              in
              if nest then K.Ref_tbl.replace nesting x node
            | _ -> ())
          cdeps
      | _ -> ());
      (match K.view node with
      | Range _ | End _ -> K.Ref_tbl.replace cdeps node ()
      | _ -> ());
      K.Ref_tbl.replace deps node cdeps)
    topo;
  let siblings = K.Ref_tbl.create 32 in
  K.Ref_tbl.iter
    (fun child parent ->
      let cur = Option.value ~default:[] (K.Ref_tbl.find_opt siblings parent) in
      K.Ref_tbl.replace siblings parent (child :: cur))
    nesting;
  let edges = K.Ref_tbl.create 16 in
  let dep_count ends node =
    match K.Ref_tbl.find_opt deps node with
    | Some nd -> List.fold_left (fun acc u -> if K.Ref_tbl.mem nd u then acc + 1 else acc) 0 ends
    | None -> 0
  in
  let add_edge rn pred =
    if not (K.in_backward_slice rn pred) then K.Ref_tbl.replace edges rn pred
  in
  K.Ref_tbl.iter
    (fun parent ends ->
      let order =
        List.sort (fun a b -> compare (dep_count ends a) (dep_count ends b)) ends
      in
      match K.view parent with
      | Sink _ ->
        let rec chain = function
          | x :: (y :: _ as rest) ->
            (match end_range y with Some rr -> add_edge rr x | None -> ());
            chain rest
          | _ -> ()
        in
        chain order
      | End _ ->
        (match end_range parent with
        | Some rr ->
          let rec chain prev = function
            | y :: ys ->
              (match end_range y with Some rr -> add_edge rr prev | None -> ());
              chain (Option.value ~default:prev (end_range y)) ys
            | [] -> ()
          in
          chain rr order
        | None -> ())
      | _ -> ())
    siblings;
  { edges }

(* Big-nat: int list, base 1_000_000_000, least-significant digit first. *)

type big_nat = int list

let big_base = 1_000_000_000L

let normalize digits =
  let rec drop_leading_zeros = function
    | [] -> [] | 0 :: rest -> drop_leading_zeros rest | xs -> xs
  in
  match List.rev digits |> drop_leading_zeros |> List.rev with
  | [] -> [ 0 ]
  | xs -> xs

let big_mul_small digits factor =
  if factor <= 0 || digits = [ 0 ] then [ 0 ]
  else
    let rec go carry acc = function
      | [] ->
        if carry = 0L then normalize (List.rev acc)
        else
          go Int64.(div carry big_base)
            (Int64.to_int Int64.(rem carry big_base) :: acc) []
      | d :: rest ->
        let prod = Int64.add (Int64.mul (Int64.of_int d) (Int64.of_int factor)) carry in
        go Int64.(div prod big_base)
          (Int64.to_int Int64.(rem prod big_base) :: acc) rest
    in
    go 0L [] digits

let compare_big_nat a b =
  let la = List.length a and lb = List.length b in
  if la <> lb then compare la lb
  else
    (* Least-significant first: compare recursively so most-significant wins. *)
    let rec cmp = function
      | [], [] -> 0
      | x :: xs, y :: ys ->
        let c = cmp (xs, ys) in
        if c <> 0 then c else compare x y
      | [], _ -> -1
      | _, [] -> 1
    in
    cmp (a, b)

(* Priority *)

type extra = None_ | Int_ of int | String_ of string

let compare_extra a b =
  match a, b with
  | None_, None_ -> 0
  | None_, _ -> -1 | _, None_ -> 1
  | Int_ x, Int_ y -> compare x y
  | String_ x, String_ y -> String.compare x y
  | Int_ _, String_ _ -> -1 | String_ _, Int_ _ -> 1

type priority = { run_count : big_nat; op_priority : int; extra : extra }

let compare_priority a b =
  let c = compare_big_nat a.run_count b.run_count in
  if c <> 0 then c
  else
    let c = compare a.op_priority b.op_priority in
    if c <> 0 then c else compare_extra a.extra b.extra

(* Tuplize: recursive structural key for deterministic tie-breaking. *)

type tuplize_arg =
  | TA_none | TA_int of int | TA_str of string
  | TA_pair of int * int | TA_list of tuplize_arg list

type tuplize_key = TK of int * tuplize_arg * tuplize_key list

let view_ordinal_and_arg = function
  | K.Define_var { name; lo; hi; _ } ->
    (1, TA_list [ TA_str name; TA_int lo; TA_int hi ])
  | K.Special { dim; _ } ->
    let tag = match dim with Group_id _ -> 0 | Local_id _ -> 1 | Global_idx _ -> 2 in
    (3, TA_int (tag * 100 + Special_dim.axis dim))
  | K.Define_local { size; _ } -> (4, TA_int size)
  | K.Define_reg { size; _ } -> (5, TA_int size)
  | K.Param { idx; _ } -> (8, TA_int idx)
  | K.Param_image { idx; width; height; _ } ->
    (9, TA_list [ TA_int idx; TA_int width; TA_int height ])
  | K.Sink _ -> (15, TA_none) | K.After _ -> (16, TA_none)
  | K.Group _ -> (17, TA_none) | K.Gep { idxs; _ } -> (18, TA_int (Hashtbl.hash idxs))
  | K.Vectorize _ -> (19, TA_none) | K.Index _ -> (20, TA_none)
  | K.Load _ -> (21, TA_none) | K.Store _ -> (22, TA_none)
  | K.Wmma { name; _ } -> (23, TA_str name)
  | K.Cast _ -> (24, TA_none) | K.Bitcast _ -> (25, TA_none)
  | K.Unary { op; _ } -> (26, TA_int (Hashtbl.hash op))
  | K.Binary { op; _ } -> (27, TA_int (Hashtbl.hash op))
  | K.Ternary { op; _ } -> (28, TA_int (Hashtbl.hash op))
  | K.Barrier -> (29, TA_none)
  | K.Range { axis; kind; _ } -> (30, TA_pair (axis, axis_kind_rank kind))
  | K.End _ -> (31, TA_none)
  | K.Const { value; _ } -> (33, TA_int (Hashtbl.hash (Const.view value)))
  | K.Custom { fmt; _ } -> (34, TA_str fmt)
  | K.Custom_inline { fmt; _ } -> (35, TA_str fmt)
  | K.Reduce { op; _ } -> (100, TA_int (Hashtbl.hash op))
  | K.Cat _ | K.Ptrcat _ | K.Unroll _ | K.Contract _ | K.Bufferize _
  | K.Invalid_index _ ->
    (100, TA_none)

let dtype_ordinal node =
  match K.dtype node with
  | Some dt -> Dtype.compare dt Dtype.void
  | None -> 0

let build_compare_tuplize topo =
  let cache = K.Ref_tbl.create (List.length topo) in
  List.iter
    (fun node ->
      let ord, arg = view_ordinal_and_arg (K.view node) in
      let children_keys =
        List.map
          (fun c ->
            match K.Ref_tbl.find_opt cache c with
            | Some k -> k
            | None -> TK (0, TA_none, []))
          (K.children node)
      in
      K.Ref_tbl.replace cache node
        (TK (ord + dtype_ordinal node * 1000, arg, children_keys)))
    topo;
  fun a b -> compare (K.Ref_tbl.find cache a) (K.Ref_tbl.find cache b)

(* Priority sort *)

let active_ranges topo children_fn =
  let tbl = K.Ref_tbl.create (List.length topo) in
  List.iter
    (fun node ->
      let from_children =
        List.fold_left
          (fun acc child ->
            match K.Ref_tbl.find_opt tbl child with
            | Some rs ->
              List.fold_left (fun a r -> if List.memq r a then a else r :: a) acc rs
            | None -> acc)
          [] (children_fn node)
      in
      let ranges =
        match K.view node with
        | End { ranges; _ } ->
          List.filter (fun r -> not (List.memq r ranges)) from_children
        | Range _ -> node :: from_children
        | _ -> from_children
      in
      K.Ref_tbl.replace tbl node ranges)
    topo;
  fun node ->
    match K.Ref_tbl.find_opt tbl node with Some rs -> rs | None -> []

let range_extent node =
  match K.view node with
  | Range { size; _ } ->
    (match K.view size with
    | Const { value; _ } ->
      (match Const.view value with Int n -> Int64.to_int n | _ -> 1)
    | _ -> 1)
  | _ -> 1

(* Compute an emission priority for each node: the product of the extents of
   all enclosing ranges (as a big-nat to avoid overflow), combined with an
   op-kind ordinal.  Nodes inside deeper/larger loop nests get higher priority
   so they are scheduled together. *)
let compute_priorities topo children_fn =
  let ranges = active_ranges topo children_fn in
  let priorities = K.Ref_tbl.create (List.length topo) in
  List.iter
    (fun node ->
      let run_count =
        List.fold_left (fun acc r -> big_mul_small acc (range_extent r)) [ 1 ] (ranges node)
      in
      let op_priority, extra =
        match K.view node with
        | Param { idx; _ } -> (-20, Int_ idx)
        | Define_var { name; _ } -> (-19, String_ name)
        | Define_local _ -> (-18, None_) | Define_reg _ -> (-17, None_)
        | Load _ -> (-1, None_) | Store _ -> (1, None_)
        | Range _ -> (5, None_) | End _ -> (-5, None_)
        | _ -> (0, None_)
      in
      K.Ref_tbl.replace priorities node { run_count; op_priority; extra })
    topo;
  priorities

(* Linearize order *)

module Heap = Set.Make (struct
  type t = int * int * K.t
  let compare (a1, a2, _) (b1, b2, _) =
    let c = compare a1 b1 in
    if c <> 0 then c else compare a2 b2
end)

(* Produce a linear emission order via a max-heap topological sort: nodes are
   released when all users are emitted, then ranked by iteration-count priority
   and a structural comparison key to keep related ops adjacent. *)
let linearize_order topo children_fn (cfg : cfg_context) =
  let children_with_cfg node =
    let cs = children_fn node in
    match K.view node with
    | Range _ ->
      (match K.Ref_tbl.find_opt cfg.edges node with
      | Some dep -> cs @ [ dep ]
      | None -> cs)
    | _ -> cs
  in
  let priorities = compute_priorities topo children_with_cfg in
  let compare_tuplize = build_compare_tuplize topo in
  let sorted =
    List.stable_sort
      (fun a b ->
        let c =
          compare_priority (K.Ref_tbl.find priorities a) (K.Ref_tbl.find priorities b)
        in
        if c <> 0 then c else compare_tuplize a b)
      topo
  in
  let n = List.length topo in
  let nkey = K.Ref_tbl.create n in
  List.iteri (fun i node -> K.Ref_tbl.replace nkey node i) sorted;
  let out_degree = K.Ref_tbl.create n in
  List.iter (fun node -> K.Ref_tbl.replace out_degree node 0) topo;
  List.iter
    (fun u ->
      List.iter
        (fun v ->
          let d = match K.Ref_tbl.find_opt out_degree v with Some d -> d | None -> 0 in
          K.Ref_tbl.replace out_degree v (d + 1))
        (children_with_cfg u))
    topo;
  let topo_id = K.Ref_tbl.create n in
  List.iteri (fun i node -> K.Ref_tbl.replace topo_id node i) topo;
  let get k tbl = match K.Ref_tbl.find_opt tbl k with Some v -> v | None -> 0 in
  let sink = match List.rev topo with x :: _ -> x | [] -> failwith "Linearizer: empty topo" in
  let heap = ref (Heap.singleton (get sink nkey, get sink topo_id, sink)) in
  let result = ref [] in
  while not (Heap.is_empty !heap) do
    let _, _, u = Heap.max_elt !heap in
    heap := Heap.remove (Heap.max_elt !heap) !heap;
    result := u :: !result;
    List.iter
      (fun v ->
        let d = get v out_degree - 1 in
        K.Ref_tbl.replace out_degree v d;
        if d = 0 then heap := Heap.add (get v nkey, get v topo_id, v) !heap)
      (children_with_cfg u)
  done;
  !result

(* DAG -> Program emission *)

let rec after_dtype node =
  match K.view node with
  | Barrier | Store _ -> Some Dtype.void
  | End { value; _ } | After { src = value; _ } -> after_dtype value
  | Group { srcs = src :: _ } -> after_dtype src
  | Group { srcs = [] } -> None
  | _ -> K.dtype node

let rec has_gate node =
  match K.view node with
  | Index { gate; _ } -> Option.is_some gate
  | After { src; _ } | Cast { src; _ } | Bitcast { src; _ } -> has_gate src
  | _ -> false

let rec find_gate node =
  match K.view node with
  | Index { gate = Some g; _ } -> Some g
  | After { src; _ } | Cast { src; _ } | Bitcast { src; _ } -> find_gate src
  | _ -> None

type emit_ctx = {
  builder : P.builder;
  k2p : P.id K.Ref_tbl.t;
  range_closed : bool K.Ref_tbl.t;
  open_ranges : K.t list ref;
}

let emit_set ctx node id = K.Ref_tbl.replace ctx.k2p node id
let emit_set_emit ctx node v = emit_set ctx node (P.emit ctx.builder v)

let emit_lookup ctx node =
  match K.Ref_tbl.find_opt ctx.k2p node with
  | Some id -> id
  | None -> failwith err_missing_ref

let rec emit_ensure_range ctx node =
  match K.view node with
  | Range { size; dtype; axis; sub; kind } ->
    (match K.Ref_tbl.find_opt ctx.k2p node with
    | Some _ -> ()
    | None ->
      let id = P.emit ctx.builder (Range { size = emit_lookup ctx size; dtype; axis; sub; kind }) in
      emit_set ctx node id;
      K.Ref_tbl.replace ctx.range_closed node false;
      ctx.open_ranges := node :: !(ctx.open_ranges));
    K.Ref_tbl.find ctx.k2p node
  | _ -> failwith err_not_range

let emit_map_id ctx node =
  match K.view node with
  | Range _ -> emit_ensure_range ctx node
  | _ -> emit_lookup ctx node

let m ctx = emit_map_id ctx
let ms ctx xs = List.map (m ctx) xs

let emit_node ctx node =
  match K.view node with
  | Sink _ -> ()
  | Group { srcs = src :: _ } -> emit_set ctx node (m ctx src)
  | Group { srcs = [] } -> failwith err_empty_group
  | After { src; deps } ->
    if K.is_ptr src then emit_set ctx node (m ctx src)
    else
      let dtype = match after_dtype src with Some dt -> dt | None -> failwith err_after_no_dtype in
      emit_set_emit ctx node (After { src = m ctx src; deps = ms ctx deps; dtype })
  | Param { idx; dtype } -> emit_set_emit ctx node (Param { idx; dtype })
  | Param_image { idx; dtype; width; height } ->
    emit_set_emit ctx node (Param_image { idx; dtype; width; height })
  | Define_local { size; dtype } -> emit_set_emit ctx node (Define_local { size; dtype })
  | Define_reg { size; dtype } -> emit_set_emit ctx node (Define_reg { size; dtype })
  | Define_var { name; lo; hi; dtype } -> emit_set_emit ctx node (Define_var { name; lo; hi; dtype })
  | Const { value; dtype } -> emit_set_emit ctx node (Const { value; dtype })
  | Index { ptr; idxs; gate; dtype } ->
    let pty = match dtype with
      | Dtype.P p -> p
      | Dtype.T _ -> failwith "Linearizer: Index must be ptr-typed after pm_add_loads"
    in
    emit_set_emit ctx node
      (Index { ptr = m ctx ptr; idxs = ms ctx idxs; gate = Option.map (m ctx) gate; dtype = pty })
  | Load { src; alt; dtype } ->
    (match has_gate src, alt with
    | true, None -> failwith err_gated_no_alt
    | false, Some _ -> failwith err_alt_no_gate
    | _ -> ());
    emit_set_emit ctx node (Load { src = m ctx src; alt = Option.map (m ctx) alt; dtype })
  | Store { dst; value; _ } ->
    (match find_gate dst with
    | Some gate ->
      let gate_id = m ctx gate and dst_id = m ctx dst in
      let if_id = P.emit ctx.builder (If { cond = gate_id; idx_for_dedup = dst_id }) in
      emit_set_emit ctx node (Store { dst = dst_id; value = m ctx value });
      ignore (P.emit ctx.builder (Endif { if_ = if_id }))
    | None ->
      emit_set_emit ctx node (Store { dst = m ctx dst; value = m ctx value }))
  | End { value; ranges = [] } ->
    emit_set ctx node (m ctx value)
  | End { value; ranges = [ range ] } ->
    let dep = m ctx value in
    let range_id = emit_ensure_range ctx range in
    ignore (P.emit ctx.builder (End_range { dep; range = range_id }));
    K.Ref_tbl.replace ctx.range_closed range true;
    ctx.open_ranges := List.filter (fun r -> not (r == range)) !(ctx.open_ranges);
    emit_set ctx node dep
  | End _ -> failwith err_end_multi
  | Range _ -> ignore (emit_ensure_range ctx node)
  | Unary { op; src; dtype } ->
    emit_set_emit ctx node (Unary { op; src = m ctx src; dtype })
  | Binary { op; lhs; rhs; dtype } ->
    emit_set_emit ctx node (Binary { op; lhs = m ctx lhs; rhs = m ctx rhs; dtype })
  | Ternary { op; a; b; c; dtype } ->
    emit_set_emit ctx node (Ternary { op; a = m ctx a; b = m ctx b; c = m ctx c; dtype })
  | Cast { src; dtype } -> emit_set_emit ctx node (Cast { src = m ctx src; dtype = Dtype.any_to_val dtype })
  | Bitcast { src; dtype } -> emit_set_emit ctx node (Bitcast { src = m ctx src; dtype })
  | Vectorize { srcs; dtype } -> emit_set_emit ctx node (Vectorize { srcs = ms ctx srcs; dtype = Dtype.any_to_val dtype })
  | Gep { src; idxs; dtype } -> emit_set_emit ctx node (Gep { src = m ctx src; idxs; dtype })
  | Barrier -> emit_set_emit ctx node Barrier
  | Special { dim; size; dtype } ->
    emit_set_emit ctx node (Special { dim; size = m ctx size; dtype })
  | Wmma { name; a; b; c; dtype; dims; dtype_in; dtype_out; device; threads;
           upcast_axes; reduce_axes } ->
    emit_set_emit ctx node
      (Wmma { name; a = m ctx a; b = m ctx b; c = m ctx c; dtype; dims;
              dtype_in; dtype_out; device; threads; upcast_axes; reduce_axes })
  | Custom { fmt; args } ->
    emit_set_emit ctx node (Custom { fmt; args = ms ctx args })
  | Custom_inline { fmt; args; dtype } ->
    emit_set_emit ctx node (Custom_inline { fmt; args = ms ctx args; dtype })
  | Invalid_index _ -> failwith (err_unlowered "Invalid_index")
  | Ptrcat _ -> failwith (err_unlowered "Ptrcat")
  | Cat _ -> failwith (err_unlowered "Cat")
  | Reduce _ -> failwith (err_unlowered "Reduce")
  | Unroll _ -> failwith (err_unlowered "Unroll")
  | Contract _ -> failwith (err_unlowered "Contract")
  | Bufferize _ -> failwith (err_unlowered "Bufferize")

(* Entry point *)

let linearize sink =
  let root = split_ends sink in
  let topo = K.toposort root in
  let cfg = build_cfg_context topo in
  let order = linearize_order topo K.children cfg in
  let ctx =
    { builder = P.create ();
      k2p = K.Ref_tbl.create (List.length topo);
      range_closed = K.Ref_tbl.create 32;
      open_ranges = ref [] }
  in
  List.iter (emit_node ctx) order;
  if !(ctx.open_ranges) <> [] then failwith err_unclosed;
  P.finish ctx.builder
