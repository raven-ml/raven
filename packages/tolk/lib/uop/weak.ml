(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

module U = Uop

let is_weak u = Dtype.is_weak (U.dtype u)
let select_dtype u = U.commit_dtype ~default_int:Dtype.int32 u
let weak_pat = Upat.any_dtype
    [ Upat.exact_dtype Dtype.weakint; Upat.exact_dtype Dtype.weakfloat ]
let promo src = Dtype.least_upper_dtype (Array.to_list (Array.map U.dtype src))
let changed u ret = if U.equal u ret then None else Some ret

let derived_dtypes u src =
  if not (Ops.Group.is_broadcastable (U.op u)) then None
  else
    let meet = promo src in
    let result = U.dtype (U.replace u ~src ()) in
    if Dtype.is_weak meet || Dtype.is_weak result then None else Some (meet, result)

let commit_srcs_at u dt =
  let bare = Option.is_some (derived_dtypes u (U.src u)) in
  let src = Array.map (fun s ->
      if not (is_weak s) then s
      else match U.op s, U.arg s with
      | Ops.Const, U.Arg.Value value when bare ->
          let value = Const.of_view dt (Const.view value) in
          U.const (Const.of_view (Dtype.weak_dtype dt) (Const.view value))
      | _ -> U.ccast ~src:s ~dtype:dt) (U.src u) in
  changed u (U.replace u ~src ())

let commit_weak_srcs u =
  if not (Array.exists is_weak (U.src u)) then None
  else let dt = promo (U.src u) in
    if Dtype.is_weak dt then None else commit_srcs_at u dt

let cast_weak_srcs c u =
  if is_weak c || not (Dtype.equal (Dtype.weak_dtype (U.dtype c)) (U.dtype u))
  then None
  else
    let widths = Array.fold_left (fun widths s ->
        if is_weak s then select_dtype s :: widths else widths)
        [ U.dtype c; select_dtype u ] (U.src u) in
    Option.map (fun result -> U.cast ~src:result ~dtype:(U.dtype c))
      (commit_srcs_at u (Dtype.least_upper_dtype widths))

let pm_cast_weak =
  let open Upat in
  Pattern_matcher.make [
    cast ~name:"c" (ops_src ~dtype:weak_pat ~name:"u" Ops.Group.alu)
    => fun bs -> cast_weak_srcs (bs $ "c") (bs $ "u");
  ]

let pm_commit_weak =
  let open Upat in
  Pattern_matcher.(make [
    ops ~name:"u" Ops.Group.broadcastable
    => (fun bs -> commit_weak_srcs (bs $ "u"));
    op ~name:"u" Ops.Store => (fun bs ->
      let u = bs $ "u" in
      let src = U.src u in
      if Array.length src < 2 || not (is_weak src.(1)) then None
      else
        let src = Array.copy src in
        src.(1) <- U.ccast ~src:src.(1) ~dtype:(U.dtype src.(0));
        changed u (U.replace u ~src ()));
  ] ++ pm_cast_weak)

let absorb_weak_src s =
  match U.op s, U.src s with
  | Ops.Cast, [| inner |] when is_weak s ->
      if Dtype.equal (U.dtype s) Dtype.weakint && not (Dtype.is_int (U.dtype inner))
      then U.cast ~src:inner ~dtype:(select_dtype s)
      else inner
  | _ -> s

let is_committed_const u =
  match U.op u, U.src u with Ops.Cast, [| c |] -> U.op c = Ops.Const | _ -> false

let lower_weak_node u =
  if is_committed_const u then None
  else
    let src = Array.map absorb_weak_src (U.src u) in
    let src = if Option.is_some (derived_dtypes u src) then src else
        Array.map (fun s -> if U.op s = Ops.Const && is_weak s
            then U.ccast ~src:s ~dtype:(select_dtype s) else s) src in
    if Array.for_all2 U.equal src (U.src u) then None
    else
      let start = if U.op u = Ops.Where then 1 else 0 in
      let lower_op = Ops.Group.is_binary (U.op u) || Ops.Group.is_unary (U.op u)
          || List.mem (U.op u) [ Ops.Where; Ops.Range; Ops.Stack; Ops.Special ] in
      let unresolved = Array.exists (fun s -> is_weak s && U.op s <> Ops.Const) src in
      if not lower_op || unresolved then Some (U.replace u ~src ())
      else
        let dt = Dtype.strong_dtype
            (if Ops.Group.is_binary (U.op u) then
               Dtype.least_upper_dtype (select_dtype u :: Array.to_list (Array.map U.dtype src))
             else U.dtype (U.replace u ~src ())) in
        let src = Array.mapi (fun i s ->
            if i < start || U.is_invalid_const (U.base s) || is_weak s then s
            else U.ccast ~src:s ~dtype:dt) src in
        Some (U.cast ~src:(U.replace u ~src ()) ~dtype:(U.dtype u))

let lower_weak_cast u =
  match U.src u with
  | [| inner |] when is_weak inner ->
      (match U.op inner, U.src inner with
       | Ops.Cast, [| x |] when not (is_weak x) ->
           Some (U.cast ~dtype:(U.dtype u)
             ~src:(U.cast ~dtype:(select_dtype u)
               ~src:(U.cast ~src:x ~dtype:(select_dtype inner))))
       | _ -> None)
  | _ -> None

let lower_weak_param u =
  match U.arg u with
  | U.Arg.Param_arg p when p.addrspace = Dtype.Alu ->
      let arg = U.Arg.Param_arg { p with dtype = select_dtype u } in
      Some (U.cast ~src:(U.replace u ~arg ()) ~dtype:Dtype.weakint)
  | _ -> None

(* A valid index into an n-element buffer lives in [0,n): a gated long index
   narrows when n-1 fits int32. Out-of-gate values wrap, and the gate discards
   them. *)
let narrow_gated_long_index u =
  let src = U.src u in
  if Array.length src < 2 then None
  else
    let buf = src.(0) in
    match U.op src.(1), U.src src.(1) with
    | Ops.Where, [| gate; idx; inv |]
      when Dtype.equal (U.dtype idx) Dtype.int64
           && U.is_invalid_const inv
           && Option.is_some (U.shape_opt buf)
           && Bound.le
                (List.fold_left (fun n dim -> Bound.mul n (U.vmax dim))
                   Bound.one (U.shape buf))
                (Bound.succ (Dtype.max Dtype.int32)) ->
        let src = Array.copy src in
        src.(1) <-
          U.valid ~src:(U.cast ~src:idx ~dtype:Dtype.int32) ~cond:gate;
        Some (U.replace u ~src ())
    | _ -> None

let pm_lower_index_dtype () =
  let open Upat in
  Pattern_matcher.make [
    ops ~name:"u" [ Ops.Index; Ops.Shrink ]
    => (fun bs -> narrow_gated_long_index (bs $ "u"));
    op_src ~dtype:weak_pat ~name:"u" Ops.Cast
    => (fun bs -> lower_weak_cast (bs $ "u"));
    ops_src ~dtype:(exact_dtype Dtype.weakint) ~name:"u" [ Ops.Param; Ops.Buffer ]
    => (fun bs -> lower_weak_param (bs $ "u"));
    ops ~name:"u" Ops.Group.all => (fun bs -> lower_weak_node (bs $ "u"));
  ]

let uncast_const u =
  let src = Array.map (fun s ->
      match U.op s, U.src s with
      | Ops.Cast, [| c |] when not (is_weak s) && U.op c = Ops.Const && is_weak c -> c
      | _ -> s) (U.src u) in
  if Array.for_all2 U.equal src (U.src u) then None
  else match derived_dtypes u src with
    | Some (meet, result) when Dtype.equal meet (promo (U.src u))
                           && Dtype.equal result (U.dtype u) -> Some (U.replace u ~src ())
    | _ -> None

let pm_uncast_const =
  let open Upat in
  Pattern_matcher.make [ ops ~name:"u" Ops.Group.broadcastable
    => fun bs -> uncast_const (bs $ "u") ]

let cast_consts u =
  if U.op u = Ops.Const || is_committed_const u then None
  else
    let meet = Option.map fst (derived_dtypes u (U.src u)) in
    let src = Array.map (fun s ->
        let s = match meet with
          | Some dt when U.op s = Ops.Const && is_weak s -> U.ccast ~src:s ~dtype:dt
          | _ -> s in
        match U.op s, U.arg s with
        | Ops.Const, U.Arg.Value value when not (U.is_invalid_const s) ->
            U.cconst value (select_dtype s)
        | _ -> s) (U.src u) in
    changed u (U.replace u ~src ())

let pm_cast_const =
  let open Upat in
  Pattern_matcher.make [ ops ~name:"u" Ops.Group.all => fun bs -> cast_consts (bs $ "u") ]
