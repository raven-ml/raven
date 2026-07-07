(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Port of tinygrad/codegen/late/linearizer.py to the tolk_uop IR. *)

open Tolk_uop
module U = Uop

(* Priority *)

(* Priority triple [(run_count, op_priority, extra)], lexicographic, lower
   first.  High [run_count] nodes land later; within a run count, the op tag
   tunes placement (LOADs early, STOREs late, RANGE late, END early). *)

type extra = No_extra | Idx of int

let range_size r =
  match U.as_range r with Some _ -> U.vmax r + 1 | None -> 1

let run_count u = List.fold_left (fun acc r -> acc * range_size r) 1 (U.ranges u)

let priority_of u =
  let op_pri, extra = match U.op u with
  | Ops.Param ->
      let idx =
        match U.as_param u with
        | Some { param; _ } -> param.slot
        | None -> 0
      in
      -20, Idx idx
  | Ops.Buffer ->
      let pri =
        match U.addrspace u with
        | Some Dtype.Local -> -17
        | _ -> -18
      in
      pri, No_extra
  | Ops.Load         ->  -1, No_extra
  | Ops.Store        ->   1, No_extra
  | Ops.Range        ->   5, No_extra
  | Ops.End          ->  -5, No_extra
  | _                ->   0, No_extra
  in
  run_count u, op_pri, extra

(* Heap *)

(* Min-heap of [(-nkey, node)]: extracting the minimum picks the node with
   the highest ideal-order index, matching tinygrad's [-nkey] trick. *)
module Heap = Set.Make (struct
  type t = int * U.t
  let compare (a, ua) (b, ub) =
    let c = Int.compare a b in
    if c <> 0 then c else U.compare ua ub
end)

(* Linearize *)

let out_degree_of tbl u =
  match U.Ref_tbl.find_opt tbl u with Some n -> n | None -> 0

let remap_sources replacements u =
  let src = U.src u in
  let changed = ref false in
  let src =
    Array.map
      (fun s ->
        match U.Ref_tbl.find_opt replacements s with
        | None -> s
        | Some s' ->
            changed := true;
            s')
      src
  in
  if !changed then U.replace u ~src () else u

let gated_store_dst u =
  match U.op u, U.src u with
  | (Ops.Index | Ops.Shrink), _ -> true
  | Ops.Cast, [| inner |] -> U.op inner = Ops.Index || U.op inner = Ops.Shrink
  | _ -> false

let linearize_cleanups (program : U.t list) : U.t list =
  let replacements = U.Ref_tbl.create 16 in
  let rec loop acc = function
    | [] -> List.rev acc
    | u :: rest ->
        let original = u in
        let u = remap_sources replacements original in
        if not (U.equal original u) then U.Ref_tbl.replace replacements original u;
        (match U.op u with
        | Ops.If | Ops.Endif ->
            failwith "IF/ENDIF must be inserted by linearize cleanups"
        | Ops.Store -> (
            match U.as_store u with
            | Some { dst; value; gate = Some gate }
              when Dtype.equal (U.dtype gate) Dtype.bool
                   && gated_store_dst dst ->
                let if_ = U.if_ ~cond:gate ~idx_for_dedup:dst in
                let store = U.store ~dst ~value () in
                let endif = U.endif ~if_ in
                U.Ref_tbl.replace replacements original store;
                U.Ref_tbl.replace replacements u store;
                loop (endif :: store :: if_ :: acc) rest
            | _ -> loop (u :: acc) rest)
        | Ops.After -> loop (u :: acc) rest
        | _ -> loop (u :: acc) rest)
  in
  loop [] program

let validate_linearize_ready sink =
  U.toposort sink
  |> List.iter (fun u ->
         match U.op u with
         | Ops.Reduce -> failwith "Reduce must be lowered before linearize"
         | Ops.Stage -> failwith "Stage must be lowered before linearize"
         | Ops.If | Ops.Endif ->
             failwith "IF/ENDIF must be inserted by linearize cleanups"
         | Ops.Group when Array.length (U.src u) = 0 ->
             failwith "empty Group"
         | Ops.Load when Array.length (U.src u) = 2 ->
             failwith "gated loads require an alt value before linearize"
         | _ -> ())

let linearize_raw (sink : U.t) : U.t list =
  let lst = U.toposort sink in
  let n = List.length lst in

  (* Out-degrees and priorities. *)
  let out_degree = U.Ref_tbl.create n in
  let priorities = U.Ref_tbl.create n in
  List.iter (fun u ->
    Array.iter (fun s ->
      U.Ref_tbl.replace out_degree s (out_degree_of out_degree s + 1))
      (U.src u);
    U.Ref_tbl.replace priorities u (priority_of u))
    lst;

  (* Assign ideal order by sorting on (priority, structure). *)
  let nkey = U.Ref_tbl.create n in
  let order_cmp a b =
    let c =
      compare (U.Ref_tbl.find priorities a) (U.Ref_tbl.find priorities b)
    in
    if c <> 0 then c else U.compare_structure a b
  in
  List.iteri (fun i u -> U.Ref_tbl.replace nkey u i)
    (List.stable_sort order_cmp lst);
  let nkey_of u = U.Ref_tbl.find nkey u in
  (* Heap-driven toposort: release a node when all its consumers are placed,
     preferring nodes closest to their ideal position. *)
  let heap = ref (Heap.singleton (-nkey_of sink, sink)) in
  let ret = ref [] in
  while not (Heap.is_empty !heap) do
    let ((_, u) as elt) = Heap.min_elt !heap in
    heap := Heap.remove elt !heap;
    ret := u :: !ret;
    Array.iter (fun v ->
      let d = out_degree_of out_degree v - 1 in
      U.Ref_tbl.replace out_degree v d;
      if d = 0 then heap := Heap.add (-nkey_of v, v) !heap)
      (U.src u)
  done;
  !ret

let linearize (sink : U.t) : U.t list =
  validate_linearize_ready sink;
  linearize_cleanups (linearize_raw sink)

(* CFGContext

   Three relationships between ranges: nested, dependent, independent.
   Everything is nested inside the sink.  Build a parent map for END nodes
   from their enclosing END/SINK, then for each sibling set emit ordering
   edges that sequence them. *)

type cfg_context = { edges : U.t U.Ref_tbl.t }

(* [end_range e] is the single range closed by [e].  After [pm_split_ends],
   every END has exactly one range. *)
let end_range e =
  match U.as_end e with
  | Some { ranges = [ r ]; _ } when U.op r = Ops.Range -> Some r
  | _ -> None

let build_cfg_context (sink : U.t) : cfg_context =
  let topo = U.toposort sink in
  let n = List.length topo in
  let topo_index = U.Ref_tbl.create n in
  List.iteri (fun i u -> U.Ref_tbl.replace topo_index u i) topo;

  (* Phase 1: transitive deps per node, and nesting parent for each END. *)
  let deps = U.Ref_tbl.create n in
  let nesting = U.Ref_tbl.create 32 in
  let record_nesting u d =
    U.Ref_tbl.iter (fun x () ->
      match U.op x with
      | Ops.End when not (U.Ref_tbl.mem nesting x) ->
          let is_nested = match U.op u with
          | Ops.Sink -> true
          | _ ->
              (match end_range u, U.Ref_tbl.find_opt deps x with
               | Some rr, Some xd -> U.Ref_tbl.mem xd rr
               | _ -> false)
          in
          if is_nested then U.Ref_tbl.replace nesting x u
      | _ -> ())
      d
  in
  List.iter (fun u ->
    let d = U.Ref_tbl.create 8 in
    Array.iter (fun s ->
      match U.Ref_tbl.find_opt deps s with
      | Some sd -> U.Ref_tbl.iter (fun k () -> U.Ref_tbl.replace d k ()) sd
      | None -> ())
      (U.src u);
    (match U.op u with Ops.End | Ops.Sink -> record_nesting u d | _ -> ());
    (match U.op u with
     | Ops.Range | Ops.End -> U.Ref_tbl.replace d u ()
     | _ -> ());
    U.Ref_tbl.replace deps u d)
    topo;

  (* Phase 2: group siblings by parent and emit ordering edges. *)
  let siblings = U.Ref_tbl.create 32 in
  U.Ref_tbl.iter (fun child parent ->
    let cur = match U.Ref_tbl.find_opt siblings parent with
    | Some l -> l | None -> []
    in
    U.Ref_tbl.replace siblings parent (child :: cur))
    nesting;

  let edges = U.Ref_tbl.create 16 in
  let add_edge rn pred =
    if pred == rn || U.in_backward_slice rn pred then
      failwith "linearizer control-flow cycle";
    U.Ref_tbl.replace edges rn pred
  in
  let rec chain prev = function
  | [] -> ()
  | y :: ys ->
      (match end_range y with
       | Some rr -> add_edge rr prev; chain y ys
       | None -> chain prev ys)
  in
  U.Ref_tbl.iter (fun parent ends ->
    let dep_count node =
      match U.Ref_tbl.find_opt deps node with
      | Some nd ->
          List.fold_left (fun acc u ->
            if U.Ref_tbl.mem nd u then acc + 1 else acc) 0 ends
      | None -> 0
    in
    let order =
      List.stable_sort
        (fun a b ->
          let c = compare (dep_count a) (dep_count b) in
          if c <> 0 then c
          else
            compare (U.Ref_tbl.find topo_index a) (U.Ref_tbl.find topo_index b))
        ends
    in
    match U.op parent, order with
    | Ops.Sink, x :: rest -> chain x rest
    | Ops.Sink, [] -> ()
    | _, _ ->
        (match end_range parent with
         | Some rr -> chain rr order
         | None -> ()))
    siblings;
  { edges }

(* Split multi-range END into nested single-range ENDs, innermost first
   by full range argument (descending). *)

(* Raven encodes tinygrad RANGE.arg as [(axis, sub, kind)] for split-end
   ordering. *)
let range_key r =
  match U.as_range r with
  | Some v -> (v.axis, v.sub, v.kind)
  | None -> (0, [], Axis_type.Loop)

let do_split_ends (e : U.t) : U.t option =
  match U.as_end e with
  | None -> None
  | Some { value; ranges } ->
      let nested =
        U.ranges (U.sink ranges)
        |> List.stable_sort (fun a b -> compare (range_key b) (range_key a))
      in
      let result =
        List.fold_left (fun v r -> U.end_ ~value:v ~ranges:[ r ]) value nested
      in
      if result == e then None else Some result

let pm_split_ends (root : U.t) : U.t = U.graph_rewrite do_split_ends root

(* Rewrite pass: attach each RANGE to its predecessor END/RANGE. *)
let pm_add_control_flow (sink : U.t) : U.t =
  let cfg = build_cfg_context sink in
  let rule node =
    match U.op node with
    | Ops.Range -> (
        match U.Ref_tbl.find_opt cfg.edges node with
        | Some pred ->
            let srcs = Array.to_list (U.src node) in
            Some (U.replace node ~src:(Array.of_list (srcs @ [ pred ])) ())
        | None -> None)
    | _ -> None
  in
  U.graph_rewrite ~name:"add control flow" ~bottom_up:true rule sink
