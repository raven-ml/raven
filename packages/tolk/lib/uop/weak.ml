(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

module U = Uop

let is_weak u = Dtype.is_weak (U.dtype u)

let weak_pat =
  Upat.any_dtype [ Upat.exact_dtype Dtype.weakint;
                   Upat.exact_dtype Dtype.weakfloat ]

(* The width a weak node takes when nothing else demands one. *)
let select_dtype u =
  if Dtype.equal (U.dtype u) Dtype.weakfloat then Dtype.default_float
  else if U.vmin u < -0x8000_0000 || U.vmax u > 0x7fff_ffff then Dtype.int64
  else Dtype.int32

(* Casting every source of [u] to [dt] gives the node itself [dt], except for
   comparisons, which are boolean whatever their operands are. *)
let rebuilt_dtype u dt =
  if Ops.Group.is_comparison (U.op u) then Dtype.bool else dt

let strip_weak_cast s =
  match U.op s, U.src s with
  | Ops.Cast, [| inner |] when is_weak s -> inner
  | _ -> s

let promo srcs = Dtype.least_upper_dtype (List.map U.dtype srcs)

(* The width [u] lives at once its sources are the concrete [src]. Binary ops
   widen from their own bounds; every other node derives from its sources. *)
let target_dtype u src =
  let dts = Array.to_list (Array.map U.dtype src) in
  let dt =
    if Ops.Group.is_binary (U.op u) then
      Dtype.least_upper_dtype (select_dtype u :: dts)
    else
      match U.op u with
      | Ops.Where -> promo (List.tl (Array.to_list src))
      | Ops.Stack -> promo (Array.to_list src)
      | Ops.Range | Ops.Special -> U.dtype src.(0)
      | Ops.Sin | Ops.Log2 | Ops.Exp2 | Ops.Sqrt | Ops.Reciprocal ->
          Dtype.least_upper_float (U.dtype src.(0))
      | op when Ops.Group.is_unary op -> U.dtype src.(0)
      | op ->
          invalid_arg
            (Printf.sprintf "Weak.target_dtype: no rule for %s" (Ops.name op))
  in
  Dtype.strong_dtype dt

(* Rebuild [u] at a concrete width, then re-wrap it in a cast to the weak dtype
   so the consumer's edge still reads weak. Sources that already carry a weak
   cast are unwrapped first; a source still weak after unwrapping means the
   node is not ready yet. *)
let lower_weak_node u =
  let start = if U.op u = Ops.Where then 1 else 0 in
  let src = Array.map strip_weak_cast (U.src u) in
  let unchanged = Array.for_all2 U.equal (U.src u) src
  and still_weak =
    let rec loop i =
      i < Array.length src && (is_weak src.(i) || loop (i + 1))
    in
    loop start
  in
  if unchanged || still_weak then None
  else
    let dt = target_dtype u src in
    let src =
      Array.mapi
        (fun i s ->
          if i < start || U.is_invalid_const (U.base s) then s
          else U.cast ~src:s ~dtype:dt)
        src
    in
    Some
      (U.cast
         ~src:(U.replace u ~src ~dtype:(rebuilt_dtype u dt) ())
         ~dtype:(U.dtype u))

let lower_weak_const u =
  match U.arg u with
  | U.Arg.Value c ->
      Some
        (U.cast
           ~src:(U.const (Const.of_view (select_dtype u) (Const.view c)))
           ~dtype:(U.dtype u))
  | _ -> None

(* Two stacked weak casts are a weakint value used as weakfloat (or the
   reverse): resolve the inner one at the outer kind's default. A single weak
   cast is never rewritten here — each consumer absorbs it on its own edge,
   see lower_weak_srcs. *)
let lower_weak_cast u =
  match U.src u with
  | [| inner |] when is_weak inner -> (
      match U.op inner, U.src inner with
      | Ops.Cast, [| x |] when not (is_weak x) ->
          Some
            (U.cast
               ~src:(U.cast ~src:x ~dtype:(select_dtype u))
               ~dtype:(U.dtype u))
      | _ -> None)
  | _ -> None

let lower_weak_param u =
  match U.as_param u, U.addrspace u with
  | Some { param; _ }, Some Dtype.Alu ->
      let dt = select_dtype u in
      let arg = U.Arg.Param_arg { param with dtype = dt } in
      Some (U.cast ~src:(U.replace u ~dtype:dt ~arg ()) ~dtype:Dtype.weakint)
  | _ -> None

let pm_lower_weak : Upat.Pattern_matcher.t =
  let open Upat in
  Pattern_matcher.make [
    (op_src ~dtype:weak_pat ~name:"u" Ops.Const
     => fun bs -> lower_weak_const (bs $ "u"));

    (op_src ~dtype:weak_pat ~name:"u" Ops.Cast
     => fun bs -> lower_weak_cast (bs $ "u"));

    (ops ~name:"u"
       (Ops.Group.binary @ Ops.Group.unary
        @ [ Ops.Where; Ops.Range; Ops.Stack; Ops.Special ])
     => fun bs -> lower_weak_node (bs $ "u"));

    (op_src ~dtype:(exact_dtype Dtype.weakint) ~name:"u" Ops.Param
     => fun bs -> lower_weak_param (bs $ "u"));
  ]

(* Lower each weak source of [u] independently, memoising per source: the same
   index expression usually feeds many consumers. A comparison is lowered whole
   instead, so the binary rule unifies both operands at one width. *)
let lower_weak_srcs memo u =
  let lower s =
    match U.Ref_tbl.find_opt memo s with
    | Some r -> r
    | None ->
        let r =
          U.graph_rewrite (Upat.Pattern_matcher.rewrite pm_lower_weak) s
        in
        (* the consumer absorbs the cast on its own edge *)
        let r =
          match U.op r, U.src r with
          | Ops.Cast, [| inner |] when is_weak r -> inner
          | _ -> r
        in
        U.Ref_tbl.add memo s r;
        r
  in
  let ret =
    if Ops.Group.is_comparison (U.op u) then lower u
    else
      let lower_if s = if is_weak s then lower s else s in
      U.replace u ~src:(Array.map lower_if (U.src u)) ()
  in
  if U.equal ret u then None else Some ret

(* A bare weak CONST commits directly — the value stays mathematical and
   emission truncates it — while a weak non-const takes the demand cast. *)
let commit_weak s dt =
  match U.op s, U.arg s with
  | Ops.Const, U.Arg.Value c -> U.const (Const.of_view dt (Const.view c))
  | _ -> U.cast ~src:s ~dtype:dt

let commit_weak_srcs u =
  let src = U.src u in
  if not (Array.exists is_weak src) then None
  else
    let dt = promo (Array.to_list src) in
    if Dtype.is_weak dt then None
    else
      (* The root re-derives: a shift's dtype is its lhs's, so committing the
         lhs commits the node too. *)
      let src =
        Array.map (fun s -> if is_weak s then commit_weak s dt else s) src
      in
      Some (U.replace u ~src ~dtype:(rebuilt_dtype u dt) ())

let commit_store_value u =
  match U.src u with
  | [||] | [| _ |] -> None
  | src when is_weak src.(1) ->
      let src = Array.copy src in
      src.(1) <- commit_weak src.(1) (U.dtype src.(0));
      Some (U.replace u ~src ())
  | _ -> None

(* Demand from a peer. Runs both in index lowering and in the decomps: a rule
   that mints a weak const must commit it in the same rewrite, so none reaches
   the renderer. *)
let pm_commit_weak : Upat.Pattern_matcher.t =
  let open Upat in
  Pattern_matcher.make [
    (ops ~name:"u" Ops.Group.broadcastable
     => fun bs -> commit_weak_srcs (bs $ "u"));

    (* demand from the destination: a store's weak value commits at the
       destination's dtype *)
    (op ~name:"u" Ops.Store => fun bs -> commit_store_value (bs $ "u"));
  ]

(* Demand from a consumer's cast: a concrete cast over a weak ALU node states
   the width the value will live at, and that width is a floor, never a
   narrowing. *)
let cast_weak_srcs c u =
  if
    Dtype.is_weak (U.dtype c)
    || not (Dtype.equal (Dtype.weak_dtype (U.dtype c)) (U.dtype u))
  then None
  else
    let dt = Dtype.least_upper_dtype [ U.dtype c; select_dtype u ] in
    let src =
      Array.map (fun s -> if is_weak s then commit_weak s dt else s) (U.src u)
    in
    Some
      (U.cast
         ~src:(U.replace u ~src ~dtype:(rebuilt_dtype u dt) ())
         ~dtype:(U.dtype c))

let pm_cast_weak : Upat.Pattern_matcher.t =
  let open Upat in
  Pattern_matcher.make [
    (cast ~name:"c" (ops_src ~dtype:weak_pat ~name:"u" Ops.Group.alu)
     => fun bs -> cast_weak_srcs (bs $ "c") (bs $ "u"));
  ]

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
           && List.fold_left ( * ) 1 (U.max_shape buf) - 1 <= 0x7fff_ffff ->
        let src = Array.copy src in
        src.(1) <-
          U.valid ~src:(U.cast ~src:idx ~dtype:Dtype.int32) ~cond:gate;
        Some (U.replace u ~src ())
    | _ -> None

let pm_lower_index_dtype () : Upat.Pattern_matcher.t =
  let open Upat in
  let memo = U.Ref_tbl.create 64 in
  Pattern_matcher.(
    pm_commit_weak ++ pm_cast_weak
    ++ make [
         (ops ~name:"u" Ops.Group.all => fun bs ->
            let u = bs $ "u" in
            if (not (is_weak u)) && Array.exists is_weak (U.src u) then
              lower_weak_srcs memo u
            else None);

         (ops ~name:"u" [ Ops.Index; Ops.Shrink ] => fun bs ->
            narrow_gated_long_index (bs $ "u"));
       ])
