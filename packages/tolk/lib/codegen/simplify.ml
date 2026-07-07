(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Port of tinygrad/codegen/simplify.py to the tolk_uop IR. *)

open Tolk_uop
module U = Uop

(* Helpers *)

(* Child index at which this pass's range-closing ops carry ranges. *)
let range_start_of_op = function
  | Ops.Reduce | Ops.End -> Some 1
  | _ -> None

let is_range u = U.op u = Ops.Range
let is_const u = U.op u = Ops.Const
let is_load u = U.op u = Ops.Index

let const_int_value u =
  match U.arg u with
  | U.Arg.Value c ->
      (match Const.view c with
       | Const.Int n -> Some (Int64.to_int n)
       | _ -> None)
  | _ -> None

let is_zero_const u =
  match U.op u, U.arg u with
  | Ops.Const, U.Arg.Value c ->
      (match Const.view c with
       | Const.Int n -> Int64.equal n 0L
       | Const.Float f -> Float.equal f 0.0
       | Const.Bool b -> not b
       | Const.Invalid -> false)
  | _ -> false

let mem_phys x xs = List.exists (fun y -> y == x) xs

let split_and c = U.split_uop c Ops.And

let count_divmod x =
  List.fold_left (fun n u ->
    match U.op u with
    | Ops.Floordiv | Ops.Floormod -> n + 1
    | _ -> n) 0 (U.backward_slice x)

let no_range u =
  not (is_range u || List.exists is_range (U.backward_slice u))

let no_load u =
  not (is_load u || List.exists is_load (U.backward_slice u))

let symbolic =
  Upat.Pattern_matcher.(Symbolic.symbolic ++ Symbolic.index_pushing)

(* [ended_ranges u] are the children [u] closes around its body. *)
let ended_ranges u =
  match range_start_of_op (U.op u) with
  | None -> []
  | Some k ->
      let s = U.src u in
      let n = Array.length s in
      if k >= n then [] else Array.to_list (Array.sub s k (n - k))

let range_size r = (U.src r).(0)

let range_kind r =
  match U.as_range r with
  | Some v -> v.kind
  | None -> invalid_arg "range_kind: not a Range"

(* Rebuild [r] with a new [size], preserving axis/sub/kind/dtype. *)
let range_with_size r size =
  match U.as_range r with
  | Some v ->
      U.replace r ~src:(Array.of_list (size :: v.parents)) ()
  | None -> invalid_arg "range_with_size: not a Range"

let range_split r ~outer_size ~inner_size =
  match U.as_range r with
  | Some v ->
      let make sub size =
        U.replace r
          ~src:(Array.of_list (size :: v.parents))
          ~arg:
            (U.Arg.Range_info
               { axis = v.axis; sub = v.sub @ [ sub ]; kind = v.kind })
          ()
      in
      (make 0 outer_size, make 1 inner_size)
  | None -> invalid_arg "range_split: not a Range"

(* Flatten range *)

(* Reattach the range children of a Reduce/End in toposort order. *)
let flatten_range r =
  match range_start_of_op (U.op r) with
  | None -> None
  | Some off ->
      let s = U.src r in
      let n = Array.length s in
      let rngs =
        if off >= n then [] else Array.to_list (Array.sub s off (n - off))
      in
      if rngs = [] then None
      else
        let new_rngs = List.filter is_range (U.toposort (U.sink rngs)) in
        let head = Array.sub s 0 off in
        let src = Array.append head (Array.of_list new_rngs) in
        let r' = U.replace r ~src () in
        if U.equal r r' then None else Some r'

let pm_flatten_range =
  let open Upat in
  Pattern_matcher.make [
    ops [ Ops.Reduce; Ops.End ] ~name:"r"
    => (fun bs -> flatten_range (bs $ "r"));
  ]

(* Merge adjacent ranges *)

(* Merge pairs of ranges of the same kind into a single [merged] of size
   [s0 * s1], provided doing so does not increase the divmod count. *)
let simplify_merge_adjacent u =
  let u_ended = ended_ranges u in
  if u_ended = [] then None
  else
    let reduce_ranges =
      List.filter_map (fun x ->
        Option.map (fun (v : U.reduce_view) -> v.ranges) (U.as_reduce x))
        (u :: U.backward_slice u)
    in
    let pairs = match U.op u with
      | Ops.End ->
          let rec adj = function
            | a :: (b :: _ as rest) -> (a, b) :: adj rest
            | _ -> []
          in
          adj u_ended
      | _ ->
          List.concat_map (fun r0 ->
            List.filter_map (fun r1 ->
              if r0 == r1 then None else Some (r0, r1)) u_ended) u_ended
    in
    let result = ref u in
    List.iter (fun (r0, r1) ->
      if range_kind r0 = range_kind r1
         && List.for_all (fun rngs ->
              mem_phys r0 rngs = mem_phys r1 rngs) reduce_ranges
      then begin
        let open U.O in
        let s0 = range_size r0 and s1 = range_size r1 in
        let merged = range_with_size r0 (s0 * s1) in
        let nidx =
          U.substitute [ (r0, merged // s1); (r1, merged mod s1) ] !result
        in
        let nidx =
          U.graph_rewrite ~name:"check_merge"
            (U.first_match
               [
                 Upat.Pattern_matcher.rewrite symbolic;
                 Upat.Pattern_matcher.rewrite pm_flatten_range;
               ])
            nidx
        in
        if count_divmod nidx <= count_divmod !result then result := nidx
      end) pairs;
    if !result == u then None else Some !result

(* Simplify ranges *)

(* Flush [ctx] by substituting each captured range with [sub k v], then
   simplify the result with the symbolic rewriter. *)
let do_substitute ctx x ~sub =
  let mappings =
    U.Ref_tbl.fold (fun k v acc ->
      match v with Some v -> (k, sub k v) :: acc | None -> acc) ctx []
  in
  U.Ref_tbl.reset ctx;
  if mappings = [] then None
  else
    let ret =
      U.graph_rewrite
        (Upat.Pattern_matcher.rewrite symbolic)
        (U.substitute mappings x)
    in
    if U.equal ret x then None else Some ret

(* True iff [ctx]'s recorded bound for [r] already dominates [c]. *)
let dominated_by ctx r c =
  match U.Ref_tbl.find_opt ctx r with
  | Some (Some existing) ->
      (match const_int_value existing, const_int_value c with
       | Some ei, Some ci -> ci <= ei
       | _ -> true)
  | Some None -> true
  | None -> false

let is_invalid u =
  match U.op u, U.arg u with
  | Ops.Const, U.Arg.Value c -> Const.view c = Const.Invalid
  | _ -> false

(* [(idx, valid)] from scalar [WHERE(valid, idx, Invalid)], matching
   tinygrad's direct INDEX-child matcher path. *)
let get_idx_valid u =
  match U.op u with
  | Ops.Where ->
      let s = U.src u in
      if Array.length s = 3 && is_invalid s.(2) then (s.(1), s.(0))
      else (u, U.const_bool true)
  | _ -> (u, U.const_bool true)

let replace_guard guards r c =
  match U.Ref_tbl.find_opt guards r with
  | Some existing ->
      (match const_int_value existing, const_int_value c with
       | Some ei, Some ci when ci <= ei -> ()
       | _ -> U.Ref_tbl.replace guards r c)
  | None -> U.Ref_tbl.replace guards r c

let collect_guards guards cond =
  match U.op cond with
  | _ ->
      List.iter (fun v ->
        match U.op v with
        | Ops.Cmplt ->
            let r = (U.src v).(0) and c = (U.src v).(1) in
            if is_range r && is_const c then replace_guard guards r c
        | _ -> ()) (split_and cond)

let apply_lane_guards ctx x cond =
  let guards = U.Ref_tbl.create 8 in
  collect_guards guards cond;
  (* Keep the largest c_i for each guarded range r. *)
  U.Ref_tbl.iter
    (fun r c ->
      if not (dominated_by ctx r c) then U.Ref_tbl.replace ctx r (Some c))
    guards;
  (* Any range that is ever ungated cannot be shrunk. *)
  List.iter
    (fun r ->
      if not (U.Ref_tbl.mem guards r) then
        U.Ref_tbl.replace ctx r (Some (range_size r)))
    (U.ranges x)

let mark_gated_value ctx idx_value =
  let x, cond = get_idx_valid idx_value in
  apply_lane_guards ctx x cond

let mark_gated ctx idx =
  match U.as_index idx with
  | None -> ()
  | Some { idxs = first :: _; _ } when U.op first = Ops.Where ->
      mark_gated_value ctx first
  | Some _ -> mark_gated_value ctx idx

let mark_unshrinkable ctx r =
  U.Ref_tbl.replace ctx r (Some (range_size r))

let simplify_ranges_rule ctx node =
  match U.op node with
  | Ops.End | Ops.Reduce ->
      (match simplify_merge_adjacent node with
       | Some _ as merged -> merged
       | None ->
           Option.iter (fun (v : U.reduce_view) ->
             List.iter (mark_unshrinkable ctx) v.ranges)
             (U.as_reduce node);
           None)
  | Ops.Index -> mark_gated ctx node; None
  | Ops.Sink ->
      do_substitute ctx node ~sub:(fun r c -> range_with_size r c)
  | _ -> None

(* Drive [simplify_ranges_rule] + [flatten_range] to a fixed point. *)
let simplify_ranges root =
  let rec loop u =
    let ctx : U.t option U.Ref_tbl.t = U.Ref_tbl.create 16 in
    let u' =
      U.graph_rewrite ~name:"simplify ranges"
        (fun n ->
          match flatten_range n with
          | Some _ as r -> r
          | None -> simplify_ranges_rule ctx n)
        u
    in
    if U.equal u u' then u else loop u'
  in
  loop root

(* Split ranges *)

(* Split [range(N) floormod C] into [outer(N//C) * C + inner(C)] whenever
   [C] divides [range_size]. *)

let can_split_range r c =
  is_range r && is_const c
  && range_kind r <> Axis_type.Warp
  && is_const (range_size r)
  &&
  match const_int_value c with
  | Some n -> U.divides (range_size r) n <> None
  | None -> false

let split_ranges_rule ctx node =
  match U.op node with
  | Ops.Floormod ->
      let r = (U.src node).(0) and c = (U.src node).(1) in
      if not (U.Ref_tbl.mem ctx r) && can_split_range r c then
        U.Ref_tbl.replace ctx r (Some c);
      None
  | Ops.Sink ->
      do_substitute ctx node ~sub:(fun r c ->
        let open U.O in
        let outer_size = range_size r // c in
        let (outer, inner) = range_split r ~outer_size ~inner_size:c in
        (outer * c) + inner)
  | _ -> None

let split_ranges root =
  let rec loop u =
    let ctx : U.t option U.Ref_tbl.t = U.Ref_tbl.create 16 in
    let u' =
      U.graph_rewrite ~name:"split ranges"
        (fun n ->
          match split_ranges_rule ctx n with
          | Some _ as r -> r
          | None -> flatten_range n)
        u
    in
    if U.equal u u' then u else loop u'
  in
  loop root

(* Reduce unparented *)

(* Remove ranges from a REDUCE that aren't referenced in the reduce
   source. ADD: compensate with a multiplication by the range size.
   MUL: compensate by exponentiating. MAX: no compensation. *)
let reduce_unparented node =
  match U.as_reduce node with
  | Some { op; src; ranges }
    when (op = Ops.Add || op = Ops.Max || op = Ops.Mul)
         && List.for_all is_range ranges ->
      let src_ranges = U.ranges src in
      let parented, unparented =
        List.partition (fun r -> mem_phys r src_ranges) ranges
      in
      if unparented = [] then None
      else
        let dtype = Dtype.val_of (U.dtype node) in
        let ret =
          if parented <> [] || not (Dtype.equal (U.dtype node) (U.dtype src))
          then U.reduce ~op ~src ~ranges:parented ~dtype
          else src
        in
        let compensate binop acc r =
          let s =
            U.cast ~src:(range_size r)
              ~dtype:(Dtype.Val (Dtype.Val.scalarize dtype))
          in
          let b = U.broadcast s (Dtype.Val.count dtype) in
          U.alu_binary ~op:binop ~lhs:acc ~rhs:b
        in
        let ret = match op with
          | Ops.Add -> List.fold_left (compensate Ops.Mul) ret unparented
          | Ops.Mul -> List.fold_left (compensate Ops.Pow) ret unparented
          | _ -> ret
        in
        Some ret
  | _ -> None

let pm_reduce_unparented =
  let open Upat in
  Pattern_matcher.make [
    op ~name:"red" Ops.Reduce => (fun bs -> reduce_unparented (bs $ "red"));
  ]

(* Reduce collapse *)

(* Toposort of [root]'s DAG restricted to nodes satisfying [gate]. *)
let toposort_gated gate root =
  let visited = U.Ref_tbl.create 64 in
  let order = ref [] in
  let rec visit node =
    if not (U.Ref_tbl.mem visited node) && gate node then begin
      U.Ref_tbl.replace visited node ();
      Array.iter visit (U.src node);
      order := node :: !order
    end
  in
  visit root;
  List.rev !order

let fold_result count v =
  U.alu_binary ~op:Ops.Mul
    ~lhs:(U.cast ~src:count ~dtype:(U.dtype v)) ~rhs:v

let maximum a b = U.alu_binary ~op:Ops.Max ~lhs:a ~rhs:b

let as_lowered_add_reduce u =
  match U.as_reduce u with
  | Some ({ op = Ops.Add; axes = []; _ } as v) -> Some v
  | _ -> None

(* The reference builds [a - b] as [a + b * (-1)]; a raw SUB node would
   block symbolic term collection (cancellation, comparison lifting). *)
let sub_add_neg a b =
  U.alu_binary ~op:Ops.Add ~lhs:a
    ~rhs:(U.alu_binary ~op:Ops.Mul ~lhs:b ~rhs:(U.const_like b (-1)))

(* [minimum] mirrors the reference: [~max(~a, ~b)] on ints, where [~x] is
   [x lxor -1]. The XOR pair cancels under symbolic once the MAX folds. *)
let bitnot x = U.alu_binary ~op:Ops.Xor ~lhs:x ~rhs:(U.const_like x (-1))
let minimum a b = bitnot (maximum (bitnot a) (bitnot b))

(* sum over r in [0,N) of [lower <= r < upper] * val collapses to
   [clamp(min(upper,N) - max(lower,0), 0, N) * val]. *)
let clamp_count ?lower ?upper r =
  let n = range_size r in
  let hi = match upper with Some u -> minimum u n | None -> n in
  let zero = U.const_int 0 in
  let lo = match lower with Some l -> maximum l zero | None -> zero in
  minimum (maximum (sub_add_neg hi lo) zero) n

(* [(x + y).or_casted < c -> x < (c.cast(y.dtype) - y)] when [y] and
   [c] carry no ranges. *)
let rule_lift_add_lt =
  let open Upat in
  let x = var "x" and y = var "y" and c = var "c" in
  let add = O.(x + y) in
  let body bs =
    let y = bs $ "y" and c = bs $ "c" in
    if no_range y && no_range c then
      let x = bs $ "x" in
      Some U.O.(x < sub_add_neg (U.cast ~src:c ~dtype:(U.dtype y)) y)
    else None
  in
  [ O.(add < c) => body; O.(cast add < c) => body ]

(* [x * y < c -> x < (c + y - 1) // y] when [y] and [c] carry no ranges,
   [y]'s dtype is integral, and [y.vmin > 0]. *)
let rule_lift_mul_lt =
  let open Upat in
  let x = var "x" and y = var "y" and c = var "c" in
  O.(x * y < c) => fun bs ->
    let x = bs $ "x" and y = bs $ "y" and c = bs $ "c" in
    if no_range y && no_range c && Dtype.is_int (U.dtype y) && U.vmin y > 0
    then
      let open U.O in
      let numerator = c + y - U.const_like y 1 in
      Some (x < (U.alu_binary ~op:Ops.Floordiv ~lhs:numerator ~rhs:y))
    else None

(* [(r < cut).where(0, val)].reduce(r, Add) *)
let rule_reduce_fold_lower =
  let open Upat in
  let r = op ~name:"r" Ops.Range
  and cut = var "cut" and z = var "zero" and v = var "val" in
  let w = where O.(r < cut) z v in
  op ~src:[ w; var "r" ] ~name:"red" Ops.Reduce
  => fun bs ->
    let red = bs $ "red" and r = bs $ "r"
    and cut = bs $ "cut" and v = bs $ "val" and z = bs $ "zero" in
    if Option.is_none (as_lowered_add_reduce red) || not (no_range v)
       || not (is_zero_const z) then None
    else
      Some (fold_result (clamp_count ~lower:cut r) v)

(* [((r < lower).not & (r < upper)).where(val, 0)].reduce(r, Add) *)
let rule_reduce_fold_between =
  let open Upat in
  let r = op ~name:"r" Ops.Range in
  let lower = var "lower" and upper = var "upper" and v = var "val"
  and z = var "zero" in
  let not_lt = op ~src:[ O.(r < lower); true_ ] Ops.Cmpne in
  let cond = alu [ not_lt; O.(r < upper) ] Ops.And in
  let w = where cond v z in
  op ~src:[ w; var "r" ] ~name:"red" Ops.Reduce
  => fun bs ->
    let red = bs $ "red" and r = bs $ "r" and lower = bs $ "lower"
    and upper = bs $ "upper" and v = bs $ "val" and z = bs $ "zero" in
    if Option.is_none (as_lowered_add_reduce red) || not (no_range v)
       || not (is_zero_const z) then None
    else
      Some (fold_result (clamp_count ~lower ~upper r) v)

(* [(r < cut).where(val, 0)].reduce(r, Add) *)
let rule_reduce_fold_upper =
  let open Upat in
  let r = op ~name:"r" Ops.Range
  and cut = var "cut" and v = var "val" and z = var "zero" in
  let w = where O.(r < cut) v z in
  op ~src:[ w; var "r" ] ~name:"red" Ops.Reduce
  => fun bs ->
    let red = bs $ "red" and r = bs $ "r"
    and cut = bs $ "cut" and v = bs $ "val" and z = bs $ "zero" in
    if Option.is_none (as_lowered_add_reduce red) || not (no_range v)
       || not (is_zero_const z) then None
    else
      Some (fold_result (clamp_count ~upper:cut r) v)

(* [(x + y).reduce(r, Add) -> x.reduce(r) + y.reduce(r)]. *)
let rule_reduce_split_add =
  let open Upat in
  let x = var "x" and y = var "y" in
  op ~src:[ O.(x + y) ] ~name:"red" ~allow_any_len:true Ops.Reduce
  => fun bs ->
    let red = bs $ "red" and x = bs $ "x" and y = bs $ "y" in
    match as_lowered_add_reduce red with
    | None -> None
    | Some { ranges; _ } ->
        let dtype = Dtype.val_of (U.dtype red) in
        Some
          (U.alu_binary ~op:Ops.Add
             ~lhs:(U.reduce ~op:Ops.Add ~src:x ~ranges ~dtype)
             ~rhs:(U.reduce ~op:Ops.Add ~src:y ~ranges ~dtype))

(* [(x & y).where(c, 0)].reduce(Add) -> y.where(c, 0).reduce * x.cast *)
let rule_reduce_and_where =
  let open Upat in
  let x = op ~name:"x" Ops.Param
  and y = var "y" and c = var "c" and z = var "zero" in
  let w = where (alu [ x; y ] Ops.And) c z in
  op ~src:[ w ] ~name:"red" ~allow_any_len:true Ops.Reduce
  => fun bs ->
    let red = bs $ "red" and x = bs $ "x"
    and y = bs $ "y" and c = bs $ "c" and z = bs $ "zero" in
    match as_lowered_add_reduce red with
    | None -> None
    | Some { ranges; _ } ->
        if not (is_zero_const z) then None
        else
          let dtype = Dtype.val_of (U.dtype red) in
          let body =
            U.alu_ternary ~op:Ops.Where ~a:y ~b:c ~c:(U.zero_like c)
          in
          Some
            (U.alu_binary ~op:Ops.Mul
               ~lhs:(U.reduce ~op:Ops.Add ~src:body ~ranges ~dtype)
               ~rhs:(U.cast ~src:x ~dtype:(U.dtype c)))

(* [x * gate.cast] with [gate:bool] -> [gate.where(x, 0)]. *)
let rule_mul_casted_bool =
  let open Upat in
  let x = var "x" and gate = var_dtype "gate" (exact_dtype Dtype.bool) in
  let body bs =
    let x = bs $ "x" and gate = bs $ "gate" in
    Some (U.alu_ternary ~op:Ops.Where ~a:gate ~b:x ~c:(U.zero_like x))
  in
  [ O.(x * cast gate) => body; O.(cast gate * x) => body ]

let pm_reduce_collapse =
  Upat.Pattern_matcher.(
    pm_reduce_unparented
    ++ make
         (rule_lift_add_lt
         @ [
             rule_lift_mul_lt;
             rule_reduce_fold_lower;
             rule_reduce_fold_between;
             rule_reduce_fold_upper;
             rule_reduce_split_add;
             rule_reduce_and_where;
           ]
         @ rule_mul_casted_bool)
    ++ symbolic)

(* Reduce load collapse *)

(* [(x + y).or_casted != c -> x != (c.cast(y.dtype) - y)]. *)
let rule_lift_add_ne =
  let open Upat in
  let x = var "x" and y = var "y" and c = var "c" in
  let add = O.(x + y) in
  let body bs =
    let y = bs $ "y" and c = bs $ "c" in
    if no_range y && no_range c then
      let x = bs $ "x" in
      Some
        (U.alu_binary ~op:Ops.Cmpne ~lhs:x
           ~rhs:(sub_add_neg (U.cast ~src:c ~dtype:(U.dtype y)) y))
    else None
  in
  [ O.(ne add c) => body; O.(ne (cast add) c) => body ]

(* [(idx != r.or_casted).where(0, expr)].reduce(r, Add) lifts a gated
   tensor load: replace [r] in [expr] with [idx.valid(cond)]. *)
let rule_reduce_gated_load_ne =
  let open Upat in
  let r = op ~name:"r" Ops.Range in
  let idx = var "idx" and expr = var "expr" in
  let body bs =
    let r = bs $ "r" and idx = bs $ "idx" and expr = bs $ "expr" in
    let r_dt = U.dtype r in
    let idx_cast = U.cast ~src:idx ~dtype:r_dt in
    let zero_cast = U.cast ~src:(U.const_int 0) ~dtype:r_dt in
    let lo =
      U.alu_binary ~op:Ops.Cmpne
        ~lhs:(U.alu_binary ~op:Ops.Cmplt ~lhs:idx_cast ~rhs:zero_cast)
        ~rhs:(U.const_bool true)
    in
    let hi =
      U.alu_binary ~op:Ops.Cmplt ~lhs:idx_cast ~rhs:(range_size r)
    in
    let v = U.alu_binary ~op:Ops.And ~lhs:lo ~rhs:hi in
    let valid_idx =
      U.alu_ternary ~op:Ops.Where ~a:v ~b:idx_cast ~c:(U.invalid ())
    in
    Some
      (U.alu_ternary ~op:Ops.Where ~a:v
         ~b:(U.substitute [ (r, valid_idx) ] expr)
         ~c:(U.zero_like expr))
  in
  [
    op ~src:[ where O.(ne idx r) (var "zero") expr; var "r" ]
      ~name:"red" Ops.Reduce
    => (fun bs ->
         match as_lowered_add_reduce (bs $ "red") with
         | Some _ when is_zero_const (bs $ "zero") -> body bs
         | _ -> None);
    op ~src:[ where O.(ne idx (cast r)) (var "zero") expr; var "r" ]
      ~name:"red" Ops.Reduce
    => (fun bs ->
         match as_lowered_add_reduce (bs $ "red") with
         | Some _ when is_zero_const (bs $ "zero") -> body bs
         | _ -> None);
    op ~src:[ where (alu [ idx; r ] Ops.Cmpeq) expr (var "zero"); var "r" ]
      ~name:"red" Ops.Reduce
    => (fun bs ->
         match as_lowered_add_reduce (bs $ "red") with
         | Some _ when is_zero_const (bs $ "zero") -> body bs
         | _ -> None);
    op ~src:[ where (alu [ idx; cast r ] Ops.Cmpeq) expr (var "zero"); var "r" ]
      ~name:"red" Ops.Reduce
    => (fun bs ->
         match as_lowered_add_reduce (bs $ "red") with
         | Some _ when is_zero_const (bs $ "zero") -> body bs
         | _ -> None);
  ]

let pm_reduce_load_collapse =
  Upat.Pattern_matcher.(
    pm_reduce_collapse
    ++ make (rule_lift_add_ne @ rule_reduce_gated_load_ne))

(* Reduce collapse driver *)

(* For each range in a REDUCE: isolate the range-dependent subgraph,
   replace externals with PARAM proxies, rebuild a standalone Reduce,
   simplify with [pm], and substitute back. *)

let is_leaf n =
  match U.op n with
  | Ops.Const | Ops.Param | Ops.Buffer -> true
  | _ -> false

let has_store_or_reduce nodes =
  List.exists (fun x ->
    match U.op x with Ops.Store | Ops.Reduce -> true | _ -> false) nodes

let collect_proxies ~included ~in_set =
  let proxies = U.Ref_tbl.create 16 in
  let n = ref 0 in
  List.iter (fun u_node ->
    Array.iter (fun s ->
      if not (U.Ref_tbl.mem in_set s
              || U.Ref_tbl.mem proxies s
              || is_leaf s) then begin
        let dv =
          U.variable ~name:(Printf.sprintf "in%d" !n)
            ~min_val:(U.vmin s) ~max_val:(U.vmax s)
            ~dtype:(Dtype.val_of (U.dtype s)) ()
        in
        U.Ref_tbl.replace proxies s dv;
        incr n
      end) (U.src u_node)) included;
  proxies

let rewrite_fixpoint ~name pm root =
  let rewrite_once u =
    U.graph_rewrite ~name (Upat.Pattern_matcher.rewrite pm) u
  in
  let rec loop fuel u =
    if fuel = 0 then u
    else
      let u' = rewrite_once u in
      if U.equal u u' then u else loop (fuel - 1) u'
  in
  loop 16 root

let reduce_collapse_inner ~pm red u =
  match U.as_reduce red with
  | Some { op = Ops.Add; ranges; _ } ->
      let result = ref u in
      let failed = ref false in
      List.iter (fun r ->
        if not !failed then begin
          let included =
            toposort_gated (fun x -> mem_phys r (U.ranges x)) !result
          in
          if has_store_or_reduce included then failed := true
          else begin
            let in_set = U.Ref_tbl.create 32 in
            List.iter (fun x -> U.Ref_tbl.replace in_set x ()) included;
            let proxies = collect_proxies ~included ~in_set in
            let fwd =
              U.Ref_tbl.fold (fun k v acc -> (k, v) :: acc) proxies []
            in
            let collapse_fxn =
              U.reduce ~op:Ops.Add ~ranges:[ r ]
                ~src:(U.substitute fwd !result)
                ~dtype:(Dtype.val_of (U.dtype !result))
            in
            let sink =
              rewrite_fixpoint ~name:"reduce_collapse" pm collapse_fxn
            in
            if not (no_range sink) then failed := true
            else
              let rev =
                U.Ref_tbl.fold (fun k v acc -> (v, k) :: acc) proxies []
              in
              result := U.substitute rev sink
          end
        end) ranges;
      if !failed || !result == u then None else Some !result
  | _ -> None

let reduce_collapse red u = reduce_collapse_inner ~pm:pm_reduce_collapse red u

let reduce_load_collapse red u =
  reduce_collapse_inner ~pm:pm_reduce_load_collapse red u

(* Reduce simplify *)

let pm_reduce_simplify =
  let open Upat in
  Upat.Pattern_matcher.(
    pm_reduce_unparented
    ++ make [
      op ~src:[ var "u" ] ~allow_any_len:true ~name:"red" Ops.Reduce
      => (fun bs ->
           let red = bs $ "red" in
           match as_lowered_add_reduce red with
           | Some _ -> reduce_collapse red (bs $ "u")
           | None -> None);
    ])

(* Load collapse *)

let is_weakint_dtype dtype = Dtype.scalar dtype = Dtype.Weakint

(* Undo the inner-lift rule on weakint expressions that carry a load:
   math on a loaded index can overflow. *)
let rule_undo_add_lt_on_load =
  let open Upat in
  let x = var "x" and y = var "y" and c = var "c" in
  O.(x + y < c) => fun bs ->
    let x = bs $ "x" and y = bs $ "y" and c = bs $ "c" in
    if is_weakint_dtype (U.dtype x) && no_load y && no_load c
       && not (no_load x)
    then
      let open U.O in
      Some (x < c - y)
    else None

let pm_load_collapse =
  let open Upat in
  Upat.Pattern_matcher.make [
    (op ~src:[ var "u"; any ] ~name:"red" Ops.Reduce
     => (fun bs ->
          let red = bs $ "red" in
          match as_lowered_add_reduce red with
          | Some _ -> reduce_load_collapse red (bs $ "u")
          | None -> None));
    rule_undo_add_lt_on_load;
  ]

(* Drivers *)

(* Whole-tree rewriter for a single pattern matcher. *)
let apply pm root = U.graph_rewrite (Upat.Pattern_matcher.rewrite pm) root

let flatten_range_all root = apply pm_flatten_range root

let reduce_unparented_all root = apply pm_reduce_unparented root

let reduce_simplify_all root = apply pm_reduce_simplify root

let load_collapse_all root = apply pm_load_collapse root
