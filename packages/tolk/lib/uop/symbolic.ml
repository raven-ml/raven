(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
open Shape
module V = Dtype.Value

let equals u v =
  match value u with #Dtype.value as x -> V.(x = v) | _ -> false

let pm = Pattern_matcher.v
let ops = Op.Set.of_list
let zero = V.of_int 0
let one = V.of_int 1
let const_v u (v : V.t) = const_like u (v :> Dtype.const)
let is_const u = op u = Op.Const

let conj = function
  | u :: us -> uprod u us
  | [] -> invalid_arg "empty conjunction"

let invalid_pat = Upat.op Op.Const ~arg:(Const `Invalid) ~name:"i"
let boolean = [ Dtype.Bool ]

(* Valids *)

(* if it's X <= c, returns X, true, c; if it's X >= c, returns X, false, c *)
let parse_valid v =
  (* a<b gives a<=b.vmax-1 (b>=a+1 for a const a), its negation a>=b.vmin
     (b<=a for a const a) *)
  let neg = op v = Op.Cmpeq && is_const (nth v 1) && equals (nth v 1) zero in
  let lt = if neg then nth v 0 else v in
  if op lt <> Op.Cmplt || not (Dtype.is_int (dtype (nth lt 0))) then None
  else
    let a = nth lt 0 in
    if is_const a then
      match value a with
      | #Dtype.value as c ->
          let c = V.to_z c in
          Some (nth lt 1, neg, if neg then c else Bigint.succ c)
      | `Invalid -> None
    else if neg then Some (a, false, V.to_z (vmin (nth lt 1)))
    else Some (a, true, Bigint.pred (V.to_z (vmax (nth lt 1))))

(* A range is a loop, whatever the values it computes with: assuming a valid
   inside one's bound would make another loop of the same identity, which runs
   apart. A substitution leaves each range it does not replace whole. *)
let keep_ranges =
  Pattern_matcher.append
    (pm (fun () ->
         [
           Pattern_matcher.rule_ctx (Upat.op Op.Range ~name:"r") (fun subs m ->
               if Tbl.mem subs (m "r") then None else raise Bottom_up_gate);
         ]))
    pm_substitute

let assuming u subs =
  let tbl = Tbl.create 8 in
  List.iter (fun (k, x) -> if k != x then Tbl.replace tbl k x) subs;
  if Tbl.length tbl = 0 then u
  else
    graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:tbl u
      (Before_sources keep_ranges)

let uop_given_valid ?(try_simplex = true) valid u =
  (* first, parse valid into [expr, (lower bound, upper bound)] *)
  let bound bounds stmt =
    match parse_valid stmt with
    | None -> bounds
    | Some (e, upper, c) ->
        let lo, hi =
          Option.value (List.assq_opt e bounds) ~default:(None, None)
        in
        let b = if upper then (lo, Some c) else (Some c, hi) in
        if List.mem_assq e bounds then
          List.map (fun (k, v) -> (k, if k == e then b else v)) bounds
        else bounds @ [ (e, b) ]
  in
  let bounds = List.fold_left bound [] (split_uop valid Op.And) in
  let fake i e lo hi =
    variable ~dtype:(dtype e) ("fake" ^ string_of_int i) lo hi
  in
  let or_bound f e = function Some c -> `Int c | None -> f e in
  let exprs =
    List.mapi
      (fun i (e, (lo, hi)) -> (i, e, or_bound vmin e lo, or_bound vmax e hi))
      bounds
  in
  (* simplify uop given that valid is True *)
  let simplex u (i, e, lo, _) =
    let terms = split_uop e Op.Add in
    let irreducible t =
      Op.Set.mem (op t) Op.Set.irreducible && op t <> Op.Const
    in
    if
      not
        (try_simplex
        && op e = Op.Add
        && V.(lo = one)
        && List.for_all irreducible terms)
    then u
    else
      (* For X0 + X1 + ... > 0, check whether every Xi > 0 gives the same
         simplified output. *)
      let candidate = List.map (fun t -> (t, fake i t one (vmax t))) terms in
      let slice = backward_slice_with_self ~calls:Skip u in
      if List.exists (fun (t, _) -> not (Nodes.mem t slice)) candidate then u
      else
        let given (x, nx) =
          simplify (assuming (simplify (assuming u [ (x, nx) ])) [ (nx, x) ])
        in
        match List.map given candidate with
        | n :: news when List.for_all (( == ) n) news -> n
        | n :: _ as news
          when op u = Op.Stack && List.compare_length_with (src u) 2 = 0 ->
            let same k = List.for_all (fun w -> nth w k == nth n k) news in
            let u = if same 0 then replace u ~src:[ nth n 0; nth u 1 ] else u in
            if same 1 then replace u ~src:[ nth u 0; nth n 1 ] else u
        | _ -> u
  in
  let u = List.fold_left simplex u exprs in
  (* try all the valids together (but only the whole expressions) *)
  let subs = List.map (fun (i, e, lo, hi) -> (e, fake i e lo hi)) exprs in
  let s = assuming u subs in
  if s == u then u
  else simplify (assuming (simplify s) (List.map (fun (e, x) -> (x, e)) subs))

(* prioritize dependencies, then tighter bounds, so weaker clauses don't hide
   useful simplifications *)
let valid_priority v valids =
  match parse_valid v with
  | None -> (0, Bigint.zero)
  | Some (e, upper, c) ->
      let depends o = e == o || Nodes.mem e (backward_slice ~calls:Skip o) in
      (-List.length (List.filter depends valids), if upper then c else Bigint.neg c)

let simplify_valid valid =
  (* this should only be for indexing, skip if there's a INDEX *)
  if op_in_backward_slice_with_self ~calls:Skip valid [ Op.Index ] then None
  else
    let valids = split_uop valid Op.And in
    let keyed = List.map (fun v -> (valid_priority v valids, v)) valids in
    let order ((d0, c0), _) ((d1, c1), _) =
      match Int.compare d0 d1 with 0 -> Bigint.compare c0 c1 | c -> c
    in
    let valids = List.map snd (List.stable_sort order keyed) in
    let given ret stmt =
      (match ret with
      | [] -> stmt
      | _ -> uop_given_valid (conj (List.rev ret)) stmt)
      :: ret
    in
    let ret = List.rev (List.fold_left given [] (dedup_nodes valids)) in
    if List.equal ( == ) ret valids then None else Some (conj ret)

(* Phase 3: the complete symbolic *)

(* A float factor moved out of a sum changes its rounding, and out of a maximum
   its NaN and signed zeros. The maximum of x * m is the maximum of x times m
   only where x * m grows with x, which an unsigned product that wraps does not:
   over r < 4, the uint8 maximum of r * 255 is 255, where 3 * 255 wraps to 253.
   So a factor leaves no unsigned maximum, and a signed one only where no
   product wraps, its own included. A maximum over a range whose size may be 0
   is its identity when it is, which the factor would scale: it keeps the
   identity there. *)
let reduce_mul_chain r =
  match arg r with
  | Reduce { op = Op.Max; _ } when Dtype.is_unsigned (dtype r) -> None
  | Reduce { op = (Op.Add | Op.Max) as rop; _ }
    when not (Dtype.is_float (dtype r)) -> (
      let ranges = List.tl (src r) in
      let outside m =
        let parents = backward_slice ~calls:Skip m in
        (not (List.memq m ranges))
        && List.for_all (fun rg -> not (Nodes.mem rg parents)) ranges
        && (rop <> Op.Max || V.(vmin m >= zero))
      in
      let prod = List.fold_left mul (int 1) in
      let body = function [] -> const_v (nth r 0) one | ms -> prod ms in
      let wraps factor body =
        let products b = V.[ vmin factor * b; vmax factor * b ] in
        V.(vmin factor < zero)
        || not (exact (dtype r) (products (vmin body) @ products (vmax body)))
      in
      match List.partition outside (split_uop (nth r 0) Op.Mul) with
      | [], _ -> None
      | out, inside when rop = Op.Max && wraps (prod out) (body inside) -> None
      | out, inside -> (
          let moved = mul (replace r ~src:(body inside :: ranges)) (prod out) in
          let runs rg = O.(int 0 < nth rg 0) in
          match List.filter (fun rg -> V.(vmin (nth rg 0) <= zero)) ranges with
          | [] -> Some moved
          | _ when rop = Op.Add -> Some moved
          | rg :: rest ->
              let all = List.fold_left (fun c rg -> O.(c land runs rg)) in
              let identity = identity_element rop (dtype r) in
              Some (where (all (runs rg) rest) moved (const_like r identity))))
  | _ -> None

let drop_and_clauses cond x i =
  let xs = Ops.ranges x in
  let in_x c =
    List.exists (fun r -> Nodes.mem r xs) (Nodes.to_list (Ops.ranges c))
  in
  match List.partition in_x (split_uop cond Op.And) with
  | _, [] -> None
  | keep, _ -> Some (where (uprod (bool true) keep) x i)

let pm_drop_and_clauses =
  pm
    (fun () -> [ rule invalid_gate (fun m -> drop_and_clauses (m "cond") (m "x") (m "i")) ])

(* move conditions from where to load's valid, drop clauses already in load *)
let where_on_load cond buf idx or_cast =
  let where_clauses = split_uop cond Op.And and load_valid = get_valid idx in
  let in_load = split_uop load_valid Op.And in
  let idx_index =
    List.filter
      (fun u -> op u = Op.Index)
      (Nodes.to_list (backward_slice_with_self ~calls:Skip idx))
  in
  let idx_ranges = Ops.ranges idx in
  (* can move if: not a const, condition's ranges are subset of idx's ranges,
     and no data dependent INDEX (only idx's INDEX allowed) *)
  let can_move c =
    let own u = op u <> Op.Index || List.memq u idx_index in
    (not (is_const c))
    && List.for_all
         (fun r -> Nodes.mem r idx_ranges)
         (Nodes.to_list (Ops.ranges c))
    && List.for_all own (Nodes.to_list (backward_slice_with_self ~calls:Skip c))
  in
  let clauses =
    List.filter (fun c -> not (List.memq c in_load)) where_clauses
  in
  let moved, keep = List.partition can_move clauses in
  if List.compare_lengths keep where_clauses = 0 then None
  else
    let idx = index buf [ valid (get_idx idx) (uprod load_valid moved) ] in
    let ret = if op or_cast = Op.Cast then cast idx (dtype or_cast) else idx in
    Some (where (uprod (bool true) keep) ret (const_v ret zero))

(* where after gated load becomes alt value. A gated load reads +0. where its
   gate fails, so a selection of -0. stays. *)
let pm_move_where_on_load =
  let loaded =
    Upat.(or_casted ~name:"or_cast" (index (var "buf") [ var "idx" ]))
  in
  let zero = Upat.named "zero" (Upat.int 0) in
  let on_load m cond =
    match value (m "zero") with
    | `Float z when Float.sign_bit z -> None
    | _ -> where_on_load cond (m "buf") (m "idx") (m "or_cast")
  in
  pm
    (fun () -> [
      rule Upat.(where (var "cond") loaded zero) (fun m -> on_load m (m "cond"));
      rule
        Upat.(where (var "cond") zero loaded)
        (fun m -> on_load m (logical_not (m "cond")));
    ])

(* pure index math only: a LOAD in x executes even where cond is false, so its
   INDEX valid must survive the assumption *)
let gated_given_valid cond x i =
  if
    (not (Dtype.equal (dtype x) Dtype.Weak_int))
    || op_in_backward_slice_with_self ~calls:Skip x [ Op.Index ]
  then None
  else Some (where cond (uop_given_valid ~try_simplex:false cond x) i)

let pm_simplify_valid =
  pm
    (fun () -> [
      (* simplify valid *)
      rule (Upat.op Op.And ~dtype:boolean ~name:"valid") (fun m ->
          simplify_valid (m "valid"));
      rule invalid_gate (fun m -> gated_given_valid (m "cond") (m "x") (m "i"));
    ])

let remove_from_sink_like = ops [ Op.Noop; Op.Stack; Op.Sink; Op.After ]

let pm_clean_up_group_sink =
  pm
    (fun () -> [
      (* clean up SINK *)
      rule
        (Upat.op Op.Sink ~name:"root")
        (fun m ->
          let root = m "root" in
          let spliced x = Op.Set.mem (op x) remove_from_sink_like in
          if not (List.exists spliced (src root)) then None
          else
            let srcs =
              List.concat_map
                (fun x -> if spliced x then src x else [ x ])
                (src root)
            in
            Some (v (op root) ~src:srcs ~arg:(arg root)));
    ])

let sym =
  let indexed = Upat.op Op.Index ~name:"index" in
  let gated_store idx cond x =
    store (index (nth idx 0) [ valid (nth idx 1) cond ]) x
  in
  Pattern_matcher.concat
    [
      symbolic;
      pm_simplify_valid;
      pm
        (fun () -> [
          (* Pow *)
          rule (Upat.op Op.Pow ~name:"p") (fun m ->
              let p = m "p" in
              if Dtype.is_int (dtype p) then
                Transcendental.xpow_int (nth p 0) (nth p 1) (dtype p)
              else Some (Transcendental.xpow (nth p 0) (nth p 1)));
          (* Load/store folding *)
          rule
            (Upat.store ~name:"x" indexed [ Upat.load indexed [] ])
            (fun m -> Some (after (v Op.Noop) (src (m "x"))));
          rule
            Upat.(
              store indexed [ where (var "gate") (var "alt") (load indexed []) ])
            (fun m -> Some (gated_store (m "index") (m "gate") (m "alt")));
          rule
            Upat.(
              store indexed
                [ where (var "gate") (load indexed []) (var "alt") ])
            (fun m ->
              let gate = logical_not (m "gate") in
              Some (gated_store (m "index") gate (m "alt")));
          (* fold gated LOAD/STORE *)
          rule
            (Upat.op Op.Store ~name:"x" ~src:[ Upat.wild; invalid_pat ])
            (fun m -> Some (after (v Op.Noop) (src (m "x"))));
          (* store of where with invalid -> gated store *)
          rule
            (Upat.op Op.Store
               ~src:Upat.[ indexed; where (var "cond") (var "val") invalid_pat ])
            (fun m -> Some (gated_store (m "index") (m "cond") (m "val")));
          (* reduce mul chain, move muls after the reduce *)
          rule
            (Upat.reduce ~name:"r" ~allow_any_len:true (Upat.op Op.Mul) [])
            (fun m -> reduce_mul_chain (m "r"));
          (* Combine terms (opinionated) *)
          (* (x+y)*c -> x*c+y*c. only for int, float has inf*0=nan issue *)
          rule
            Upat.(
              named "a" (var ~dtype:[ Dtype.Weak_int ] "x" + var "y")
              * cvar "c")
            (fun m ->
              let a = m "a" and c = m "c" in
              let fit v = Shape.fits a V.(v * vmin c) in
              let bounds u = [ vmin u; vmax u ] in
              if List.for_all fit (bounds (m "x") @ bounds (m "y")) then
                Some O.((m "x" * c) + (m "y" * c))
              else None);
        ]);
      pm_clean_up_group_sink;
    ]
