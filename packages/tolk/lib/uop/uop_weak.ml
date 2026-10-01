(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops

let is_weak dt = List.mem dt Dtype.weaks
let weak u = is_weak (dtype u)
let is_const u = op u = Op.Const
let committed u = commit_dtype ~default_int:Dtype.Int32 u
let unchanged src u = List.equal ( == ) src (Ops.src u)

(* The decompositions and float emulation commit bare constants at a type
   another source already states. *)
let commit_weak_consts u = function
  | None -> u
  | Some dt ->
      let commit s = if is_const s && weak s then ccast s dt else s in
      replace ~src:(List.map commit (src u)) u

(* The committed types [u] commits its sources at: the operands' meet and [u]'s
   own derived type, [None] if either is weak. *)
let derived_dtypes u src =
  if not (Op.Set.mem (op u) Op.Set.broadcastable) then None
  else
    let meet = promo_dtype src in
    if is_weak meet then None
    else
      let result = dtype_of (op u) src (arg u) in
      if is_weak result then None else Some (meet, result)

let commit_srcs_at u dt =
  (* The root re-derives: a shift's type is its left operand's, so committing
     that operand commits the node too. *)
  let bare = Option.is_some (derived_dtypes u (src u)) in
  let commit s =
    if not (weak s) then s
    else if bare && is_const s then const (Dtype.const dt (value s))
    else ccast s dt
  in
  let ret = replace ~src:(List.map commit (src u)) u in
  if ret == u then None else Some ret

let commit_weak_srcs u =
  if not (List.exists weak (src u)) then None
  else
    let dt = Dtype.least_upper (List.map dtype (src u)) in
    if is_weak dt then None else commit_srcs_at u dt

(* A committed cast over a weak node states the width the value will live at.
   That width is a floor, never a narrowing. *)
let cast_weak_srcs c u =
  (* Only within the kind: an integer cast of a weak float node converts the
     value, and says nothing of the node's width. *)
  if weak c || not (Dtype.equal (Dtype.weak (dtype c)) (dtype u)) then None
  else
    (* Every weak source commits at the one width: the node's own bounds and
       each source's, none of them narrowed. *)
    let srcs =
      List.filter_map
        (fun s -> if weak s then Some (committed s) else None)
        (src u)
    in
    let widths = committed u :: srcs in
    let dt = Dtype.least_upper (dtype c :: widths) in
    (* An integer cast never computes in a float, which the lattice puts above a
       64-bit unsigned and a signed integer (DIVERGENCES D44): the node computes
       at 64 bits, unsigned only where its bounds pass a signed integer's. *)
    let dt =
      if Dtype.is_int (dtype c) && Dtype.is_float dt then
        if List.exists (Dtype.equal Dtype.Uint64) widths then Dtype.Uint64
        else Dtype.Int64
      else dt
    in
    Option.map (fun ret -> cast ret (dtype c)) (commit_srcs_at u dt)

(* Rides every rewrite that can build a weak constant, and must reach its fixed
   point before pm_lower_weak gives one its default. *)
let pm_commit_weak =
  Pattern_matcher.(
    v
      (fun () -> [
        rule (Upat.v ~op:Op.Set.broadcastable ~name:"u" ()) (fun m ->
            commit_weak_srcs (m "u"));
        rule
          (Upat.op Op.Store
             ~src:[ Upat.var "dst"; Upat.var "x" ~dtype:Dtype.weaks ]
             ~allow_any_len:true ~name:"u")
          (fun m ->
            let u = m "u" and dst = m "dst" in
            let x = ccast (m "x") (dtype dst) in
            Some (replace ~src:(dst :: x :: List.drop 2 (src u)) u));
        (* No constant arm: a committed cast over a weak constant is already
           committed, built that way by Ops.const. *)
        rule
          (Upat.op Op.Cast ~name:"c"
             ~src:[ Upat.v ~op:Op.Set.alu ~dtype:Dtype.weaks ~name:"u" () ])
          (fun m -> cast_weak_srcs (m "c") (m "u"));
      ]))

(* Consumers absorb the weak cast off their sources and default the constants
   they cannot derive; the operations that produce a type settle here. A weak
   float unary must resolve before the transcendental decomposition. *)
let lower_weak_ops =
  Op.Set.(
    union (union binary unary)
      (of_list [ Op.Where; Op.Range; Op.Stack; Op.Special ]))

(* A weak cast states a width, which the consumer restates. A weak integer cast
   of a boolean or a float is a conversion: it commits here. *)
let absorb_weak_src s =
  if op s <> Op.Cast || not (weak s) then s
  else
    let x = nth s 0 in
    if Dtype.equal (dtype s) Dtype.Weak_int && not (Dtype.is_int (dtype x)) then
      cast x (committed s)
    else x

let lower_weak_node u =
  (* A committed constant, not a consumer. *)
  if op u = Op.Cast && is_const (nth u 0) then None
  else
    let src = List.map absorb_weak_src (Ops.src u) in
    let src =
      if Option.is_some (derived_dtypes u src) then src
      else
        List.map
          (fun s -> if is_const s && weak s then ccast s (committed s) else s)
          src
    in
    if unchanged src u then None
    else
      (* A Where's condition is a boolean, never part of the width. *)
      let start = if op u = Op.Where then 1 else 0 in
      let operands = List.drop start src in
      if
        (not (Op.Set.mem (op u) lower_weak_ops))
        || List.exists (fun s -> weak s && not (is_const s)) operands
      then Some (replace ~src u)
      else
        (* Resolve whole once every weak expression is lowered: a binary node
           widens from its own bounds too, and derivable constants wait. *)
        let dt =
          Dtype.strong
            (if Op.Set.mem (op u) Op.Set.binary then
               Dtype.least_upper (committed u :: List.map dtype src)
             else dtype_of (op u) src (arg u))
        in
        let commit s =
          if is_invalid (base s) || weak s then s else ccast s dt
        in
        let src = List.take start src @ List.map commit operands in
        Some (cast (replace ~src u) (dtype u))

let pm_lower_weak =
  Pattern_matcher.(
    v
      (fun () -> [
        (* A guarded long index into small storage narrows: the values outside
           the guard are discarded. *)
        rule
          (Upat.v
             ~op:(Op.Set.of_list [ Op.Index; Op.Shrink ])
             ~src:
               [
                 Upat.var "buf";
                 Upat.where (Upat.var "gate")
                   (Upat.var "idx" ~dtype:[ Dtype.Int64 ])
                   (Upat.const `Invalid);
               ]
             ~allow_any_len:true ~name:"u" ())
          (fun m ->
            let u = m "u" and buf = m "buf" in
            let last = Dtype.Value.of_int (max_numel buf - 1) in
            if Dtype.Value.(last <= Dtype.max Dtype.Int32) then
              let idx = valid (cast (m "idx") Dtype.Int32) (m "gate") in
              Some (replace ~src:(buf :: idx :: List.drop 2 (src u)) u)
            else None);
        (* Two stacked weak casts are two conversions of kind: each resolves at
           its own kind's default. *)
        rule
          (Upat.op Op.Cast ~dtype:Dtype.weaks ~name:"u"
             ~src:[ Upat.op Op.Cast ~dtype:Dtype.weaks ~src:[ Upat.var "x" ] ])
          (fun m ->
            let u = m "u" and x = m "x" in
            if weak x then None
            else
              Some
                (cast
                   (cast (cast x (committed (nth u 0))) (committed u))
                   (dtype u)));
        rule
          (Upat.v
             ~op:(Op.Set.of_list [ Op.Param; Op.Buffer; Op.Alloc ])
             ~dtype:[ Dtype.Weak_int ] ~name:"u" ())
          (fun m ->
            let u = m "u" in
            match (addrspace u, arg u) with
            | Some Dtype.Alu, Param p ->
                let arg = Param { p with dtype = committed u } in
                Some (cast (replace ~arg u) Dtype.Weak_int)
            | _ -> None);
        rule (Upat.v ~op:Op.Set.all ~name:"u" ()) (fun m ->
            lower_weak_node (m "u"));
      ]))

(* Drop the cast off a committed constant where the consumer derives it anyway,
   so rules keyed on bare constants keep matching. The drop must change nothing
   the consumer derives: neither the operands' meet nor the node's own type. *)
let uncast_const u =
  (* A weak cast over a constant is not a commit: it is still resolving. The
     literal left is the constant at the width it was committed to, as a machine
     holds it. A NaN or an infinity has no integer value: its literal is left as
     it is written, and the consumer's derived type keeps the cast. *)
  let uncast s =
    if op s = Op.Cast && (not (weak s)) && is_const (nth s 0) && weak (nth s 0)
    then
      let dt = dtype s in
      match value s with
      | `Float f
        when not (Float.is_finite f || Dtype.is_float dt || Dtype.is_bool dt) ->
          nth s 0
      | c -> (
          match Dtype.const dt c with
          | #Dtype.value as v -> const (Dtype.truncate dt v :> Dtype.const)
          | `Invalid -> s)
    else s
  in
  let src = List.map uncast (Ops.src u) in
  if unchanged src u then None
  else
    match derived_dtypes u src with
    | Some (meet, result)
      when Dtype.equal meet (promo_dtype (Ops.src u))
           && Dtype.equal result (dtype u) ->
        Some (replace ~src u)
    | _ -> None

let pm_uncast_const =
  Pattern_matcher.(
    v
      (fun () -> [
        rule (Upat.v ~op:Op.Set.broadcastable ~name:"u" ()) (fun m ->
            uncast_const (m "u"));
      ]))

(* Commit every remaining bare constant, keyed on the consumer: being bare is a
   property of the edge. *)
let cast_consts u =
  (* A committed constant's literal is its value, not an edge. *)
  if op u = Op.Cast && is_const (nth u 0) then None
  else
    let u =
      match derived_dtypes u (src u) with
      | Some (meet, _) -> commit_weak_consts u (Some meet)
      | None -> u
    in
    (* A cast folds at the types a bare constant derives, so the width is
       forced. Invalid never commits. *)
    let commit s =
      if is_const s && not (is_invalid s) then cconst (committed s) (value s)
      else s
    in
    Some (replace ~src:(List.map commit (src u)) u)

let pm_cast_const =
  Pattern_matcher.(
    v
      (fun () -> [
        rule (Upat.v ~op:Op.Set.all ~name:"u" ~early_reject:[ Op.Const ] ())
          (fun m -> cast_consts (m "u"));
      ]))
