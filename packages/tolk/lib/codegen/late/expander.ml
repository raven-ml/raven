(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_ir
module K = Kernel

(* Helpers *)

let prod lst = List.fold_left ( * ) 1 lst

(* Flatten a multi-axis position into a linear index.
   args = [(axis, size); ...], rpk = [(axis, value); ...].
   Rightmost axis varies fastest. *)
let expand_arg_to_idx args rpk =
  List.fold_right
    (fun (axis, m) (idx, mul) ->
      let v = List.assoc axis rpk in
      (v * mul + idx, mul * m))
    args (0, 1)
  |> fst

(* Cartesian product of all axis values.
   [(0,2); (1,3)] -> [{0:0,1:0}; {0:0,1:1}; ...; {0:1,1:2}] as assoc lists. *)
let choices_from_args args =
  List.fold_left
    (fun acc (axis, m) ->
      List.concat_map (fun rest ->
        List.init m (fun v -> (axis, v) :: rest)) acc)
    [[]] args

(* For each choice in cargs, compute the flat index into eargs's space.
   Excluded axes are zeroed.
   XXX tinygrad memoizes this with @functools.cache. *)
let swizzle_args cargs eargs exclude_args =
  List.map
    (fun rpk ->
      let rpk =
        if exclude_args = [] then rpk
        else List.map (fun x -> (x, 0)) exclude_args @ rpk
      in
      expand_arg_to_idx eargs rpk)
    (choices_from_args cargs)

let all_same = function
  | [] -> true
  | x :: xs -> List.for_all (( = ) x) xs

let is_unroll n = match K.view n with Unroll _ -> true | _ -> false

let unroll_axes nodes =
  List.concat_map (fun n ->
    match K.view n with Unroll { axes; _ } -> axes | _ -> []) nodes

(* Expand an op's Unroll children into a single wider vector operation. *)
let do_expand root =
  let expandable = match K.view root with
    | Unary _ | Binary _ | Ternary _ | Cast _ | Bitcast _ | Gep _
    | Wmma _ | Load _ | Store _ | Index _ | Bufferize _
    | Vectorize _ | Reduce _ | End _ | After _ -> true
    | _ -> false in
  if not expandable then None
  else
  let children = K.children root in
  let expands = List.filter is_unroll children in
  if expands = [] then None
  else
    let root_view = K.view root in
    let root_dt = match K.dtype_opt root with Some dt -> dt | None -> Dtype.void in
    let dcount n = match K.sort n with
      | Effect -> 1 | _ -> Dtype.count (K.dtype n) in
    let exclude_args = match root_view with
      | Wmma { reduce_axes; upcast_axes = ua, ub, uc; _ } ->
          List.sort_uniq compare (reduce_axes @ List.map fst (ua @ ub @ uc))
      | _ -> [] in
    let expands_args = List.map (fun e ->
      match K.view e with Unroll { axes; _ } -> axes | _ -> []) expands in
    let expand_args =
      if all_same expands_args && exclude_args = [] then List.hd expands_args
      else
        List.filter (fun (a, _) -> not (List.mem a exclude_args))
          (List.sort_uniq compare (List.concat expands_args)) in
    let expand_sz = prod (List.map snd expand_args) in
    let is_non_ptr_index = match root_view with
      | Index { dtype = Dtype.Val _; _ } -> true | _ -> false in
    (* Build new sources *)
    let new_srcs = List.mapi (fun i src ->
      match K.view src with
      | Unroll { src = inner; axes = src_axes; _ } ->
          if expand_args = src_axes then inner
          else
            let lst = swizzle_args expand_args src_axes exclude_args in
            let sc = dcount src in
            let lst = if sc > 1 then
              List.concat_map (fun idx ->
                List.init sc (fun j -> idx * sc + j)) lst
              else lst in
            K.gep_multi ~src:inner ~idxs:lst
      | _ ->
          let is_passthrough =
            (match K.range_start root with
             | Some rs -> i >= rs | None -> false)
            || (is_non_ptr_index && i >= 1) in
          if is_passthrough then src
          else if dcount src > 1 then
            K.vcat ~srcs:(List.init expand_sz (fun _ -> src))
          else K.broadcast src expand_sz)
      children in
    (* Build result node — one match for all cases *)
    let buf_reg = K.is_ptr (List.hd children) &&
      (let pty = K.ptr_dtype (List.hd children) in
       Dtype.Ptr.addrspace pty = Dtype.Reg) in
    let nsrc = match root_view with
      | Index { dtype = Dtype.Val _; _ } when buf_reg ->
          (* REG buffer: expand into individual scalar INDEXes *)
          K.vectorize ~srcs:(List.init expand_sz (fun j ->
            K.replace root ~children:(List.map (fun s ->
              if K.is_ptr s || dcount s > 1
              then K.gep ~src:s ~idx:j else s) new_srcs) ()))
      | Gep { idxs = [gep_idx]; _ } ->
          assert (Dtype.count root_dt = 1);
          let src0 = List.hd new_srcs in
          let stride = dcount src0 / expand_sz in
          K.gep_multi ~src:src0
            ~idxs:(List.init expand_sz (fun k -> gep_idx + k * stride))
      | Index { dtype = Dtype.Ptr pty; idxs; gate; _ } ->
          let rest = List.tl new_srcs in
          let n = List.length idxs in
          K.index_raw ~ptr:(List.hd new_srcs)
            ~idxs:(List.filteri (fun i _ -> i < n) rest)
            ?gate:(Option.map (fun _ -> List.nth rest n) gate)
            ~dtype:(Dtype.vec expand_sz (Dtype.Ptr pty)) ()
      | _ when root_dt = Dtype.void ->
          K.replace root ~children:new_srcs ()
      | _ ->
          K.replace root ~children:new_srcs
            ~dtype:(Dtype.vec (Dtype.count root_dt * expand_sz)
                      (Dtype.scalarize root_dt)) ()
    in
    Some (K.unroll ~src:nsrc ~axes:expand_args ~dtype:(Dtype.val_of root_dt))

(* Contract an Unroll back to scalar form via GEP index permutations. *)
let do_contract con =
  match K.view con with
  | Contract { src = ex; axes = con_axes; dtype = con_dt } ->
      (match K.view ex with
       | Unroll { src = inner; axes = ex_axes; _ } ->
           assert (Dtype.Val.equal con_dt Dtype.Val.void
                   || Dtype.Val.count con_dt = prod (List.map snd con_axes));
           let new_ex_args =
             List.filter (fun x -> not (List.mem x con_axes)) ex_axes in
           let idxs =
             List.concat_map (fun rpk ->
               List.map
                 (fun lrpk -> expand_arg_to_idx ex_axes (rpk @ lrpk))
                 (choices_from_args con_axes))
               (choices_from_args new_ex_args) in
           Some (K.unroll ~src:(K.gep_multi ~src:inner ~idxs)
                   ~axes:new_ex_args ~dtype:con_dt)
       | _ ->
           if Dtype.Val.equal con_dt Dtype.Val.void then Some ex
           else Some (K.vectorize
                   ~srcs:(List.init (Dtype.Val.count con_dt) (fun _ -> ex))))
  | _ -> None

(* Wrap END's value in CONTRACT for any Unroll ranges. *)
let end_unrolls node =
  match K.view node with
  | End { value; ranges } ->
      let unrolls, rest = List.partition is_unroll ranges in
      if unrolls = [] then None
      else
        let axes = unroll_axes unrolls in
        Some (K.end_ ~value:(K.contract ~src:value ~axes ~dtype:Dtype.Val.void)
                ~ranges:rest ())
  | _ -> None

(* Detect Vectorize that is a broadcast (all srcs physically identical). *)
let peel_broadcast n =
  match K.view n with
  | Vectorize { srcs = (x :: _) as srcs; _ }
    when List.for_all (fun y -> x == y) srcs -> Some (x, List.length srcs)
  | _ -> None

(* Push broadcast through After: After(Broadcast(x, n), deps) -> Broadcast(After(x, deps), n). *)
let broadcast_after node =
  match K.view node with
  | After { src; deps } ->
      Option.map (fun (x, n) ->
        K.broadcast (K.after ~src:x ~deps) n) (peel_broadcast src)
  | _ -> None

(* Push broadcast through End: End(Broadcast(x, n), ranges) -> Broadcast(End(x, ranges), n). *)
let broadcast_end node =
  match K.view node with
  | End { value; ranges } ->
      Option.map (fun (x, n) ->
        K.broadcast (K.end_ ~value:x ~ranges ()) n) (peel_broadcast value)
  | _ -> None

(* Bufferize(Unroll, Unroll) contracts the range Unroll. *)
let bufferize_contract node =
  match K.view node with
  | Bufferize { src = val_src; ranges = [range_src]; _ } -> (
      match K.view val_src, K.view range_src with
      | Unroll _, Unroll { src = inner; axes; _ } ->
          let cnt = Dtype.count (K.dtype inner) in
          Some (K.replace node ~children:(List.map (fun s ->
            match K.view s with
            | Unroll { dtype = s_dt; _ } ->
                K.contract ~src:s ~axes
                  ~dtype:(Dtype.Val.vec cnt (Dtype.Val.scalarize s_dt))
            | _ -> s) (K.children node)) ())
      | _ -> None)
  | _ -> None

(* Flatten nested Unroll(Unroll(x, inner_axes), outer_axes) -> Unroll(x, inner+outer). *)
let double_unroll node =
  match K.view node with
  | Unroll { src = inner; axes = outer_axes; dtype = outer_dt } -> (
      match K.view inner with
      | Unroll { src = deepest; axes = inner_axes; _ } ->
          Some (K.unroll ~src:deepest ~axes:(inner_axes @ outer_axes)
                  ~dtype:outer_dt)
      | _ -> None)
  | _ -> None

(* Empty Unroll is a no-op. *)
let empty_unroll node =
  match K.view node with
  | Unroll { src; axes = []; _ } -> Some src
  | _ -> None

(* Core expander: broadcast pushing, unroll/contract engine, and
   expansion of ALU/Cast/Index/etc with Unroll children. *)
let expander_rule =
  K.first_match [
    broadcast_after; broadcast_end;
    end_unrolls;
    bufferize_contract;
    double_unroll;
    do_expand;
    do_contract;
    empty_unroll;
  ]

(* pre-expander *)

(* Rewrite UPCAST/UNROLL Range to Unroll(Vconst(0..s-1)). *)
let range_to_unroll node =
  match K.view node with
  | Range { size; dtype; axis; kind = (Axis_kind.Upcast | Axis_kind.Unroll); _ } -> (
      match K.const_arg size with
      | Some (Int n) ->
          let s = Int64.to_int n in
          let scalar = Dtype.Val.scalarize dtype in
          let vec = K.vconst
            ~values:(List.init s (fun i -> Const.int scalar i))
            ~dtype:(Dtype.Val.vec s scalar) in
          Some (K.unroll ~src:vec ~axes:[(axis, s)] ~dtype)
      | _ -> None)
  | _ -> None

(* Partition Reduce ranges into RANGE and UNROLL, contract the UNROLLs. *)
let fix_reduce_unroll node =
  match K.view node with
  | Reduce { op; src; ranges; dtype } ->
      let reduce_range, reduce_expand = List.partition K.is_range ranges in
      if reduce_expand = [] then None
      else
        let reduce_expand =
          List.filter (fun x -> K.const_arg x = None) reduce_expand in
        let axes = unroll_axes reduce_expand in
        let ret =
          if axes <> [] then
            K.with_tag "1" (K.contract ~src ~axes
              ~dtype:(Dtype.Val.vec (prod (List.map snd axes))
                        (Dtype.Val.scalarize dtype)))
          else src in
        Some (K.reduce ~op ~src:ret ~ranges:reduce_range ~dtype)
  | _ -> None

(* Partition Store ranges into UNROLL and rest, contract the UNROLLs. *)
let fix_store_unroll node =
  match K.view node with
  | Store { dst; value; ranges } ->
      let store_expand, store_range = List.partition is_unroll ranges in
      if store_expand = [] then None
      else
        let axes = unroll_axes store_expand in
        let inner = K.store ~dst ~value ~ranges:store_range in
        Some (K.with_tag "1"
                (K.contract ~src:inner ~axes ~dtype:Dtype.Val.void))
  | _ -> None

(* Convert Reduce with Group_reduce ranges into a local-buffer reduction. *)
let fix_group_for_reduce node =
  match K.view node with
  | Reduce { op; src; ranges; dtype } ->
      let reduce_gfr, reduce_r =
        List.partition (fun u ->
          match K.view u with
          | Range { kind = Axis_kind.Group_reduce; _ } -> true
          | _ -> false)
          ranges in
      if reduce_gfr = [] then None
      else
        let upstream_locals =
          List.filter
            (fun u -> K.is_range u && K.range_kind u = Axis_kind.Local)
            (K.toposort node) in
        let ret = K.reduce ~op ~src ~ranges:reduce_r ~dtype in
        let reduce_loop =
          List.map (fun r ->
            K.range ~size:(K.range_size r) ~axis:(K.range_axis r + 100)
              ~kind:Axis_kind.Reduce ~dtype:(Dtype.val_of (K.dtype r)) ())
            reduce_gfr in
        let gfr_axis = K.range_axis (List.hd reduce_gfr) in
        let buf_dt = K.dtype ret in
        let buf =
          K.bufferize ~src:ret ~ranges:(upstream_locals @ reduce_gfr)
            ~dtype:(Dtype.Ptr.create (Dtype.val_of buf_dt) ~addrspace:Dtype.Local ~size:(-1))
            ~opts:{ device = Some (Device_index gfr_axis);
                    addrspace = Dtype.Local; removable = true } in
        let idx =
          K.index ~ptr:buf ~idxs:(upstream_locals @ reduce_loop) () in
        Some (K.reduce ~op ~src:idx ~ranges:reduce_loop ~dtype)
  | _ -> None

let pre_expander_rule =
  K.first_match [range_to_unroll; fix_reduce_unroll; fix_store_unroll]

(* Run all expander passes to fixpoint: symbolic simplification,
   range-to-unroll conversion, group-for-reduce lowering, and the
   main expand/contract engine. *)
let expand root =
  K.graph_rewrite ~name:"expander"
    (K.first_match [
      Symbolic.sym;
      pre_expander_rule;
      fix_group_for_reduce;
      expander_rule;
    ]) root
