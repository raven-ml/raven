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

let dedup l =
  Helpers.dedup
    (module struct
      type t = Ops.t

      let equal = ( == )
      let hash = hash
    end)
    l

let invalid_pat = Upat.op Op.Const ~arg:(Const `Invalid) ~name:"i"
let boolean = [ Dtype.Bool ]
let int_or_bool = Dtype.Bool :: Dtype.Weak_int :: Dtype.ints

(* Valids *)

(* if it's X <= c, returns X, true, c; if it's X >= c, returns X, false, c *)
let parse_valid v =
  let int_lt u = op u = Op.Cmplt && Dtype.is_int (dtype (nth u 0)) in
  if
    op v = Op.Cmpne
    && is_const (nth v 1)
    && equals (nth v 1) one
    && int_lt (nth v 0)
  then
    (* (X < c).ne(True) -> X >= c *)
    let s0 = nth v 0 in
    Some (nth s0 0, false, V.to_z (vmin (nth s0 1)))
    (* c < X -> X >= c+1 (a const on the left is a lower bound on the right),
       and X < c -> X <= c-1 *)
  else if int_lt v && is_const (nth v 0) then
    match value (nth v 0) with
    | #Dtype.value as c -> Some (nth v 1, false, Bigint.succ (V.to_z c))
    | `Invalid -> None
  else if int_lt v then Some (nth v 0, true, Bigint.pred (V.to_z (vmax (nth v 1))))
  else None

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
    let irreducible t = Op.Set.mem (op t) Op.Set.irreducible in
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
          simplify
            (substitute ~calls:Skip ~pass:Fixed_point
               (simplify
                  (substitute ~calls:Skip ~pass:Fixed_point u [ (x, nx) ]))
               [ (nx, x) ])
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
  let s = substitute ~calls:Skip ~pass:Fixed_point u subs in
  if s == u then u
  else
    simplify
      (substitute ~calls:Skip ~pass:Fixed_point (simplify s)
         (List.map (fun (e, x) -> (x, e)) subs))

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
    let ret = List.rev (List.fold_left given [] (dedup valids)) in
    if List.equal ( == ) ret valids then None else Some (conj ret)

(* Phase 3: the complete symbolic *)

(* A float factor moved out of a sum changes its rounding, and out of a maximum
   its NaN and signed zeros. *)
let reduce_mul_chain r =
  match arg r with
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
      match List.partition outside (split_uop (nth r 0) Op.Mul) with
      | [], _ -> None
      | out, inside ->
          let body =
            match inside with [] -> const_v (nth r 0) one | _ -> prod inside
          in
          Some (mul (replace r ~src:(body :: ranges)) (prod out)))
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

let remove_from_sink_like = ops [ Op.Noop; Op.Stack; Op.Sink; Op.Group ]

let pm_clean_up_group_sink =
  pm
    (fun () -> [
      (* clean up GROUP/SINK *)
      rule (Upat.op Op.Group ~src:[ Upat.var "x" ]) (fun m -> Some (m "x"));
      rule
        (Upat.v ~op:(ops [ Op.Sink; Op.Group ]) ~name:"root" ())
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
              Some (Transcendental.xpow (nth p 0) (nth p 1)));
          (* Load/store folding *)
          rule
            (Upat.store indexed [ Upat.load indexed [] ])
            (fun _ -> Some (v Op.Noop));
          rule
            Upat.(
              store indexed [ where (var "gate") (var "alt") (load indexed []) ])
            (fun m -> Some (gated_store (m "index") (m "gate") (m "alt")));
          (* fold gated LOAD/STORE *)
          rule
            (Upat.op Op.Store ~src:[ Upat.wild; invalid_pat ])
            (fun _ -> Some (v Op.Noop));
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
          rule
            Upat.(int (-1) * (var ~dtype:int_or_bool "x" + var "y"))
            (fun m -> Some O.(-m "x" + -m "y"));
          (* (x+y)*c -> x*c+y*c. only for int, float has inf*0=nan issue *)
          rule
            Upat.((var ~dtype:[ Dtype.Weak_int ] "x" + var "y") * cvar "c")
            (fun m ->
              let c = m "c" in
              Some O.((m "x" * c) + (m "y" * c)));
        ]);
      pm_clean_up_group_sink;
    ]
