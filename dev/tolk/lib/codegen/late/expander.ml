(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_ir
module K = Kernel

(* Helpers *)

let prod lst = List.fold_left ( * ) 1 lst

let all_same = function
  | [] -> true
  | x :: xs -> List.for_all (( = ) x) xs

let is_broadcast srcs =
  match srcs with
  | [] -> false
  | x :: xs -> List.for_all (fun y -> x == y) xs (* physical eq: same DAG node *)

(* View predicates *)

let is_unroll n = match K.view n with K.Unroll _ -> true | _ -> false
let is_range n = match K.view n with K.Range _ -> true | _ -> false
let is_const n = match K.view n with K.Const _ -> true | _ -> false

let unroll_axes nodes =
  List.concat_map
    (fun x -> match K.view x with K.Unroll { axes; _ } -> axes | _ -> [])
    nodes

(* range_start: index at which range args begin for ops that have them *)

let range_start = function
  | K.Bufferize _ -> Some 1
  | K.Reduce _ -> Some 1
  | K.Store _ -> Some 2
  | K.Wmma _ -> Some 3
  | K.End _ -> Some 1
  | _ -> None

(* Axis index helpers *)

(* _expand_arg_to_idx: compute a flat index from axes and an assoc list *)
let expand_arg_to_idx args rpk =
  List.fold_right
    (fun (axis, m) (idx, mul) ->
      let v = List.assoc_opt axis rpk |> Option.value ~default:0 in
      (v * mul + idx, mul * m))
    args (0, 1)
  |> fst

(* _choices_from_args: cartesian product of per-axis ranges.
   Matches itertools.product order: rightmost axis varies fastest. *)
let choices_from_args args =
  List.fold_left
    (fun acc (axis, m) ->
      List.concat_map
        (fun rest -> List.init m (fun v -> rest @ [ (axis, v) ]))
        acc)
    [ [] ] args

(* _swizzle_args: for each choice of cargs, compute the flat index into eargs,
   zeroing out exclude_args *)
let swizzle_args cargs eargs exclude_args =
  List.map
    (fun rpk ->
      let rpk =
        if exclude_args = [] then rpk
        else rpk @ List.map (fun x -> (x, 0)) exclude_args
      in
      expand_arg_to_idx eargs rpk)
    (choices_from_args cargs)

(* Pattern functions *)

let do_expand root =
  let children = K.children root in
  let expands =
    List.filter is_unroll children
  in
  if expands = [] then None
  else
    let exclude_args =
      match K.view root with
      | K.Wmma { reduce_axes; upcast_axes = ua, ub, uc; _ } ->
          List.sort_uniq compare (reduce_axes @ List.map fst (ua @ ub @ uc))
      | _ -> []
    in
    let expands_args =
      List.map
        (fun e -> match K.view e with K.Unroll { axes; _ } -> axes | _ -> assert false)
        expands
    in
    let expand_args =
      if all_same expands_args && exclude_args = [] then List.hd expands_args
      else
        List.filter
          (fun (a, _) -> not (List.mem a exclude_args))
          (List.sort_uniq compare (List.concat expands_args))
    in
    let expand_sz = prod (List.map snd expand_args) in
    let new_srcs =
      List.mapi
        (fun i src ->
          match K.view src with
          | K.Unroll { src = inner; axes = src_axes; _ } ->
              if expand_args = src_axes then inner
              else
                let lst = swizzle_args expand_args src_axes exclude_args in
                let lst =
                  let sc = (K.dtype_or Dtype.void src).count in
                  if sc > 1 then
                    List.concat_map (fun idx -> List.init sc (fun j -> (idx * sc) + j)) lst
                  else lst
                in
                K.gep_multi ~src:inner ~idxs:lst
          | _ -> (
              match range_start (K.view root) with
              | Some rs when i >= rs -> src
              | _ ->
              if (K.dtype_or Dtype.void src).count > 1 then
                K.cat ~srcs:(List.init expand_sz (fun _ -> src))
              else K.broadcast src expand_sz))
        children
    in
    let root_dt = K.dtype_or Dtype.void root in
    let nsrc =
      match K.view root with
      | K.Gep { idx = gep_idx; _ } ->
          assert (root_dt.count = 1);
          let src0 = List.hd new_srcs in
          let src0_count = (K.dtype_or Dtype.void src0).count in
          let stride = src0_count / expand_sz in
          let idxs = List.init expand_sz (fun k -> gep_idx + (k * stride)) in
          K.gep_multi ~src:src0 ~idxs
      | _ ->
          (* Nodes with ptr or void dtype (Index, Store, End) keep their
             original dtype — the expansion width is tracked in the UNROLL
             wrapper, not in the node's dtype. Value nodes get a widened dtype.
             Pointer types keep their scalar form — the expansion width
             lives in the UNROLL wrapper, not the pointer dtype. *)
          if root_dt = Dtype.void then
            K.replace root ~children:new_srcs ()
          else
            let new_dt =
              Dtype.vec (Dtype.scalar_of root_dt) (root_dt.count * expand_sz)
            in
            K.replace root ~children:new_srcs ~dtype:new_dt ()
    in
    Some (K.unroll ~src:nsrc ~axes:expand_args ~dtype:root_dt)

let do_contract con =
  match K.view con with
  | K.Contract { src = ex; axes = con_axes; dtype = con_dt } ->
      assert (con_dt = Dtype.void || con_dt.count = prod (List.map snd con_axes));
      (match K.view ex with
      | K.Unroll { src = inner; axes = ex_axes; _ } ->
          let new_ex_args =
            List.filter (fun x -> not (List.mem x con_axes)) ex_axes
          in
          let idxs =
            List.concat_map
              (fun rpk ->
                List.map
                  (fun lrpk -> expand_arg_to_idx ex_axes (rpk @ lrpk))
                  (choices_from_args con_axes))
              (choices_from_args new_ex_args)
          in
          Some
            (K.unroll
               ~src:(K.gep_multi ~src:inner ~idxs)
               ~axes:new_ex_args ~dtype:con_dt)
      | _ ->
          (* CONTRACT without UNROLL repeats the element VECTORIZED.
             Void CONTRACTs unwrap to the source (void.count = 1,
             void can't be vectorized). *)
          if con_dt = Dtype.void then Some ex
          else
            Some
              (K.vectorize
                 ~srcs:(List.init con_dt.count (fun _ -> ex))))
  | _ -> None

let end_unrolls u =
  match K.view u with
  | K.End { value; ranges } ->
      let unrolls, rest = List.partition is_unroll ranges in
      if unrolls = [] then None
      else
        let all_axes = unroll_axes unrolls in
        let contracted =
          K.contract ~src:value ~axes:all_axes ~dtype:Dtype.void
        in
        Some (K.end_ ~value:contracted ~ranges:rest)
  | _ -> None

(* Expander rules *)

let expander_rule node =
  match K.view node with
  (* push broadcast through AFTER; otherwise do_expand *)
  | K.After { src; deps } -> (
      match K.view src with
      | K.Vectorize { srcs; _ } when is_broadcast srcs ->
          let x = List.hd srcs in
          let n = List.length srcs in
          Some (K.broadcast (K.after ~src:x ~deps) n)
      | _ -> do_expand node)
  (* push broadcast through END; then end_unrolls; then do_expand *)
  | K.End { value; ranges } -> (
      match K.view value with
      | K.Vectorize { srcs; _ } when is_broadcast srcs ->
          let x = List.hd srcs in
          let n = List.length srcs in
          Some (K.broadcast (K.end_ ~value:x ~ranges) n)
      | _ -> (
          match end_unrolls node with Some _ as r -> r | None -> do_expand node))
  (* BUFFERIZE(UNROLL, UNROLL) -> replace both with CONTRACT; otherwise do_expand *)
  | K.Bufferize { src; ranges = [ range_src ]; _ } -> (
      match (K.view src, K.view range_src) with
      | K.Unroll _, K.Unroll { axes; _ } ->
          let inner_count =
            match K.view range_src with
            | K.Unroll { src = inner; _ } -> (K.dtype_or Dtype.void inner).count
            | _ -> assert false
          in
          let new_srcs =
            List.map
              (fun s ->
                match K.view s with
                | K.Unroll { dtype = s_dt; _ } ->
                    K.contract ~src:s ~axes
                      ~dtype:(Dtype.vec (Dtype.scalar_of s_dt) inner_count)
                | _ -> s)
              (K.children node)
          in
          Some (K.replace node ~children:new_srcs ())
      | _ -> do_expand node)
  | K.Bufferize _ -> do_expand node
  (* double UNROLL: combine axes *)
  | K.Unroll { src = inner; axes = outer_axes; dtype = outer_dt } -> (
      match K.view inner with
      | K.Unroll { src = deepest; axes = inner_axes; _ } ->
          Some
            (K.unroll ~src:deepest
               ~axes:(inner_axes @ outer_axes)
               ~dtype:outer_dt)
      | _ ->
          (* empty UNROLL is NOOP *)
          if outer_axes = [] then Some inner else None)
  (* CONTRACT -> do_contract *)
  | K.Contract _ -> do_contract node
  (* do_expand on ALU/CAST/BITCAST/GEP/WMMA/LOAD/STORE/INDEX/VECTORIZE/REDUCE *)
  | K.Unary _ | K.Binary _ | K.Ternary _ | K.Cast _ | K.Bitcast _ | K.Gep _
  | K.Wmma _ | K.Load _ | K.Store _ | K.Index _ | K.Vectorize _ | K.Reduce _
    ->
      do_expand node
  | _ -> None

(* Pre-expander rules *)

let const_range_vec dt size =
  (* Build a Vectorize of [0; 1; ...; size-1] as index constants *)
  let scalar = Dtype.scalar_of dt in
  K.vectorize ~srcs:(List.init size (fun i ->
    K.const (Const.int scalar i)))

let range_to_unroll node =
  match K.view node with
  | K.Range { size; dtype; axis; kind } -> (
      match kind with
      | Axis_kind.Upcast | Axis_kind.Unroll -> (
          match K.view size with
          | K.Const { value; _ } -> (
              match Const.view value with
              | Int n ->
                  let s = Int64.to_int n in
                  let vec = const_range_vec dtype s in
                  Some
                    (K.unroll ~src:vec
                       ~axes:[ (axis, s) ]
                       ~dtype)
              | _ -> None)
          | _ -> None)
      | _ -> None)
  | _ -> None

let fix_reduce_unroll node =
  match K.view node with
  | K.Reduce { op; src; ranges; dtype } ->
      let reduce_range, reduce_expand =
        List.partition is_range ranges
      in
      if reduce_expand = [] then None
      else
        let reduce_expand =
          List.filter (fun x -> not (is_const x)) reduce_expand
        in
        let contract_axis =
          List.concat_map
            (fun x ->
              match K.view x with
              | K.Unroll { axes; _ } -> axes
              | _ -> failwith "expected UNROLL in reduce expand")
            reduce_expand
        in
        let ret =
          if contract_axis <> [] then
            K.contract ~src
              ~axes:contract_axis
              ~dtype:(Dtype.vec (Dtype.scalar_of dtype) (prod (List.map snd contract_axis)))
          else src
        in
        Some
          (K.reduce ~op ~src:ret ~ranges:reduce_range ~dtype)
  | _ -> None

let fix_store_unroll node =
  match K.view node with
  | K.Store { dst; value; ranges } ->
      let store_expand, store_range =
        List.partition is_unroll ranges
      in
      if store_expand = [] then None
      else
        let all_axes = unroll_axes store_expand in
        let inner_store = K.store ~dst ~value ~ranges:store_range in
        Some (K.contract ~src:inner_store ~axes:all_axes ~dtype:Dtype.void)
  | _ -> None

let pre_expander_rule =
  K.first_match [ range_to_unroll; fix_reduce_unroll; fix_store_unroll ]

(* Group for reduce rules *)

let fix_group_for_reduce node =
  match K.view node with
  | K.Reduce { op; src; ranges; dtype } ->
      let reduce_gfr, reduce_r =
        List.partition
          (fun u ->
            match K.view u with
            | K.Range { kind = Axis_kind.Group_reduce; _ } -> true
            | _ -> false)
          ranges
      in
      if reduce_gfr = [] then None
      else
        let upstream_locals =
          List.filter
            (fun u ->
              match K.view u with
              | K.Range { kind = Axis_kind.Local; _ } -> true
              | _ -> false)
            (K.toposort node)
        in
        let ret = K.reduce ~op ~src ~ranges:reduce_r ~dtype in
        let reduce_loop =
          List.map
            (fun r ->
              match K.view r with
              | K.Range { size; dtype = rdt; axis; _ } ->
                  K.range ~size ~axis:(axis + 100) ~kind:Axis_kind.Reduce
                    ~dtype:rdt ()
              | _ -> assert false)
            reduce_gfr
        in
        let gfr_axis =
          match K.view (List.hd reduce_gfr) with
          | K.Range { axis; _ } -> axis
          | _ -> assert false
        in
        let buf_ranges = upstream_locals @ reduce_gfr in
        let buf_dt = K.dtype_or Dtype.void ret in
        let buf_ptr =
          Dtype.Ptr.create buf_dt ~addrspace:Dtype.Local ~size:(-1) ()
        in
        let buf =
          K.bufferize ~src:ret ~ranges:buf_ranges ~dtype:buf_ptr
            ~opts:
              {
                device = Some (Device_index gfr_axis);
                addrspace = Dtype.Local;
                removable = true;
              }
        in
        let idx =
          K.index ~ptr:buf ~idxs:(upstream_locals @ reduce_loop) ()
        in
        Some (K.reduce ~op ~src:idx ~ranges:reduce_loop ~dtype)
  | _ -> None

(* Entry point *)

let expand root =
  K.rewrite_fixpoint
    (K.first_match
       [ Symbolic.sym; pre_expander_rule; fix_group_for_reduce; expander_rule ])
    root
