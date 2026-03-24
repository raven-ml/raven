(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_ir
module K = Kernel

let prod lst = List.fold_left ( * ) 1 lst

let is_broadcast = function
  | [] -> false
  | x :: xs -> List.for_all (fun y -> x == y) xs

let is_unroll n = match K.view n with K.Unroll _ -> true | _ -> false

let unroll_axes nodes =
  List.concat_map
    (fun x -> match K.view x with K.Unroll { axes; _ } -> axes | _ -> [])
    nodes

let expand_arg_to_idx args rpk =
  List.fold_right
    (fun (axis, m) (idx, mul) ->
      let v = List.assoc_opt axis rpk |> Option.value ~default:0 in
      (v * mul + idx, mul * m))
    args (0, 1)
  |> fst

(* Cartesian product; rightmost axis varies fastest (itertools.product order) *)
let choices_from_args args =
  List.fold_left
    (fun acc (axis, m) ->
      List.concat_map
        (fun rest -> List.init m (fun v -> rest @ [ (axis, v) ]))
        acc)
    [ [] ] args

let swizzle_args cargs eargs exclude_args =
  List.map
    (fun rpk ->
      let rpk =
        if exclude_args = [] then rpk
        else rpk @ List.map (fun x -> (x, 0)) exclude_args
      in
      expand_arg_to_idx eargs rpk)
    (choices_from_args cargs)

(* Expand an op's Unroll children into a single wider vector operation.
   Each Unroll source is swizzled or replicated to match the combined expand
   axes, non-Unroll scalars are broadcast, and the result is wrapped in a
   new Unroll with the unified axis list. *)
let do_expand root =
  let children = K.children root in
  let expands = List.filter is_unroll children in
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
        (fun e ->
          match K.view e with K.Unroll { axes; _ } -> axes | _ -> assert false)
        expands
    in
    let expand_args =
      let all_same = match expands_args with
        | [] -> true
        | x :: xs -> List.for_all (( = ) x) xs
      in
      if all_same && exclude_args = [] then List.hd expands_args
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
                let sc = Dtype.count (K.dtype_or Dtype.void src) in
                let lst =
                  if sc > 1 then
                    List.concat_map
                      (fun idx -> List.init sc (fun j -> (idx * sc) + j))
                      lst
                  else lst
                in
                K.gep_multi ~src:inner ~idxs:lst
          | _ -> (
              match K.range_start root with
              | Some rs when i >= rs -> src
              | _ ->
                  let is_non_ptr_index = match K.view root with
                    | K.Index { dtype = Dtype.T _; _ } -> true | _ -> false in
                  if is_non_ptr_index && i >= 1 then src
                  else if Dtype.count (K.dtype_or Dtype.void src) > 1 then
                    K.cat ~srcs:(List.init expand_sz (fun _ -> src))
                  else K.broadcast src expand_sz))
        children
    in
    (* REG buffer special case: for non-PtrDType INDEX on REG-addressed buffers,
       expand into individual scalar INDEXes wrapped in VECTORIZE+UNROLL.
       This avoids creating a VECTORIZE of REG pointers the devectorizer cannot
       resolve. *)
    let root_dt = K.dtype_or Dtype.void root in
    let is_non_ptr_index = match K.view root with
      | K.Index { dtype = Dtype.T _; _ } -> true | _ -> false in
    let buf_is_reg =
      is_non_ptr_index &&
      (match K.get_ptr_dtype (List.hd children) with
       | Some pty -> Dtype.addrspace pty = Dtype.Reg | None -> false) in
    if buf_is_reg then begin
      let idxs = List.init expand_sz (fun j ->
        let idx_srcs = List.map (fun s ->
          if K.is_ptr s || Dtype.count (K.dtype_or Dtype.void s) > 1
          then K.gep ~src:s ~idx:j
          else s) new_srcs
        in
        K.replace root ~children:idx_srcs ()) in
      Some (K.unroll ~src:(K.vectorize ~srcs:idxs)
              ~axes:expand_args ~dtype:root_dt)
    end
    else
    let nsrc =
      match K.view root with
      | K.Gep { idxs = [gep_idx]; _ } ->
          assert (Dtype.count root_dt = 1);
          let src0 = List.hd new_srcs in
          let src0_count = Dtype.count (K.dtype_or Dtype.void src0) in
          let stride = src0_count / expand_sz in
          let new_idxs =
            List.init expand_sz (fun k -> gep_idx + (k * stride))
          in
          K.gep_multi ~src:src0 ~idxs:new_idxs
      | K.Gep { idxs = gep_idxs; _ } ->
          let src0 = List.hd new_srcs in
          let src0_count = Dtype.count (K.dtype_or Dtype.void src0) in
          let stride = src0_count / expand_sz in
          let new_idxs =
            List.concat_map (fun k ->
              List.map (fun gi -> gi + (k * stride)) gep_idxs)
              (List.init expand_sz Fun.id)
          in
          K.gep_multi ~src:src0 ~idxs:new_idxs
      | _ ->
          if root_dt = Dtype.void then
            K.replace root ~children:new_srcs ()
          else begin
            match K.view root with
            | K.Index { dtype = Dtype.P pty; idxs; gate; _ } ->
                (* Ptr-typed Index: vectorize the ptr vector width, not the
                   element count. *)
                let new_pty = Dtype.ptr_with_v pty expand_sz in
                let ptr = List.hd new_srcs in
                let rest = List.tl new_srcs in
                let n_idxs = List.length idxs in
                let new_idxs = List.filteri (fun i _ -> i < n_idxs) rest in
                let new_gate = if Option.is_some gate then
                  Some (List.nth rest n_idxs) else None in
                K.index_raw ~ptr ~idxs:new_idxs ?gate:new_gate
                  ~dtype:(Dtype.ptr_to_any new_pty) ()
            | _ ->
                let new_dt =
                  Dtype.vec (Dtype.scalar_of root_dt) (Dtype.count root_dt * expand_sz)
                in
                K.replace root ~children:new_srcs ~dtype:new_dt ()
          end
    in
    Some (K.unroll ~src:nsrc ~axes:expand_args ~dtype:root_dt)

(* Contract an Unroll(Expand(...)) back to scalar form by generating GEP index
   permutations for all axes except the contracted ones, effectively selecting
   elements from the expanded vector. *)
let do_contract con =
  match K.view con with
  | K.Contract { src = ex; axes = con_axes; dtype = con_dt } ->
      assert (con_dt = Dtype.void || Dtype.count con_dt = prod (List.map snd con_axes));
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
            (K.unroll ~src:(K.gep_multi ~src:inner ~idxs)
               ~axes:new_ex_args ~dtype:con_dt)
      | _ ->
          if con_dt = Dtype.void then Some ex
          else Some (K.vectorize ~srcs:(List.init (Dtype.count con_dt) (fun _ -> ex))))
  | _ -> None

let end_unrolls u =
  match K.view u with
  | K.End { value; ranges } ->
      let unrolls, rest = List.partition is_unroll ranges in
      if unrolls = [] then None
      else
        let axes = unroll_axes unrolls in
        Some (K.end_ ~value:(K.contract ~src:value ~axes ~dtype:Dtype.void)
                ~ranges:rest ())
  | _ -> None

(* Expander rules *)

let push_broadcast_through ~rebuild ~src ~rest =
  match K.view src with
  | K.Vectorize { srcs; _ } when is_broadcast srcs ->
      let x = List.hd srcs in
      Some (K.broadcast (rebuild x) (List.length srcs))
  | _ -> rest ()

let expander_rule node =
  match K.view node with
  | K.After { src; deps } ->
      push_broadcast_through
        ~rebuild:(fun x -> K.after ~src:x ~deps) ~src
        ~rest:(fun () -> do_expand node)
  | K.End { value; ranges } ->
      push_broadcast_through
        ~rebuild:(fun x -> K.end_ ~value:x ~ranges ()) ~src:value
        ~rest:(fun () ->
          match end_unrolls node with Some _ as r -> r | None -> do_expand node)
  | K.Bufferize { src; ranges = [ range_src ]; _ } -> (
      match (K.view src, K.view range_src) with
      | K.Unroll _, K.Unroll { src = inner; axes; _ } ->
          let inner_count = Dtype.count (K.dtype_or Dtype.void inner) in
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
  | K.Unroll { src = inner; axes = outer_axes; dtype = outer_dt } -> (
      match K.view inner with
      | K.Unroll { src = deepest; axes = inner_axes; _ } ->
          Some (K.unroll ~src:deepest ~axes:(inner_axes @ outer_axes)
                  ~dtype:outer_dt)
      | _ -> if outer_axes = [] then Some inner else None)
  | K.Contract _ -> do_contract node
  | K.Unary _ | K.Binary _ | K.Ternary _ | K.Cast _ | K.Bitcast _ | K.Gep _
  | K.Wmma _ | K.Load _ | K.Store _ | K.Index _ | K.Vectorize _ | K.Reduce _ ->
      do_expand node
  | _ -> None

(* Pre-expander rules *)

let range_to_unroll node =
  match K.view node with
  | K.Range { size; dtype; axis; kind = (Axis_kind.Upcast | Axis_kind.Unroll); _ } -> (
      match K.view size with
      | K.Const { value; _ } -> (
          match Const.view value with
          | Int n ->
              let s = Int64.to_int n in
              let scalar = Dtype.scalar_of dtype in
              let vec =
                K.vconst
                  ~values:(List.init s (fun i -> Const.int scalar i))
                  ~dtype:(Dtype.vec scalar s)
              in
              Some (K.unroll ~src:vec ~axes:[ (axis, s) ] ~dtype)
          | _ -> None)
      | _ -> None)
  | _ -> None

let fix_reduce_unroll node =
  match K.view node with
  | K.Reduce { op; src; ranges; dtype } ->
      let reduce_range, reduce_expand = List.partition K.is_range ranges in
      if reduce_expand = [] then None
      else
        let reduce_expand =
          List.filter (fun x -> not (K.is_const x)) reduce_expand
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
            K.contract ~src ~axes:contract_axis
              ~dtype:
                (Dtype.vec (Dtype.scalar_of dtype)
                   (prod (List.map snd contract_axis)))
          else src
        in
        Some (K.reduce ~op ~src:ret ~ranges:reduce_range ~dtype)
  | _ -> None

let fix_store_unroll node =
  match K.view node with
  | K.Store { dst; value; ranges } ->
      let store_expand, store_range = List.partition is_unroll ranges in
      if store_expand = [] then None
      else
        let all_axes = unroll_axes store_expand in
        let inner_store = K.store ~dst ~value ~ranges:store_range in
        Some (K.contract ~src:inner_store ~axes:all_axes ~dtype:Dtype.void)
  | _ -> None

let pre_expander_rule =
  K.first_match [ range_to_unroll; fix_reduce_unroll; fix_store_unroll ]

(* Group-for-reduce *)

(* Convert a Reduce with Group_reduce ranges into a local-buffer-based
   reduction: the partial result is stored into a shared local buffer indexed
   by upstream local ranges, then reloaded and reduced over fresh Reduce-kind
   loop ranges, turning a cross-workgroup reduce into an explicit two-phase
   pattern the backend can lower to barrier + local memory. *)
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
            (fun u -> K.is_range u && K.range_kind u = Axis_kind.Local)
            (K.toposort node)
        in
        let ret = K.reduce ~op ~src ~ranges:reduce_r ~dtype in
        let reduce_loop =
          List.map
            (fun r ->
              K.range ~size:(K.range_size r) ~axis:(K.range_axis r + 100)
                ~kind:Axis_kind.Reduce ~dtype:(K.dtype_or Dtype.index r) ())
            reduce_gfr
        in
        let gfr_axis = K.range_axis (List.hd reduce_gfr) in
        let buf_dt = K.dtype_or Dtype.void ret in
        let buf_ptr =
          Dtype.ptr_of buf_dt ~addrspace:Dtype.Local ~size:(-1)
        in
        let buf =
          K.bufferize ~src:ret ~ranges:(upstream_locals @ reduce_gfr)
            ~dtype:buf_ptr
            ~opts:{ device = Some (Device_index gfr_axis);
                    addrspace = Dtype.Local; removable = true }
        in
        let idx =
          K.index ~ptr:buf ~idxs:(upstream_locals @ reduce_loop) ()
        in
        Some (K.reduce ~op ~src:idx ~ranges:reduce_loop ~dtype)
  | _ -> None

(* Entry point *)

let expand root =
  K.graph_rewrite ~name:"expander"
    (K.first_match
       [ Symbolic.sym; pre_expander_rule; fix_group_for_reduce; expander_rule ])
    root
