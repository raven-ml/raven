(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Range simplification pattern matchers.

   Each exported function is a Kernel.t -> Kernel.t rewrite pass composed
   via Kernel.graph_rewrite. *)

open Tolk_ir
module K = Kernel

(* Helpers *)

let none_in_slice pred u = not (List.exists pred (K.backward_slice u))
let no_range u = none_in_slice K.is_range u

let no_load u =
  none_in_slice (fun n -> match K.view n with K.Index _ -> true | _ -> false) u

let count_divmod (x : K.t) =
  List.fold_left
    (fun acc n ->
      match K.view n with
      | Binary { op = `Idiv | `Mod; _ } -> acc + 1
      | _ -> acc)
    0 (K.backward_slice x)

let minimum a b =
  K.ternary ~op:`Where ~a:(K.binary ~op:`Cmplt ~lhs:a ~rhs:b) ~b:a ~c:b

let maximum a b = K.binary ~op:`Max ~lhs:a ~rhs:b

let peel_cast n = match K.view n with Cast { src; _ } -> src | _ -> n

let is_zero n =
  match K.view n with
  | Const { value; _ } -> (
      match Const.view value with
      | Int 0L | Bool false -> true
      | Float f -> f = 0.0
      | _ -> false)
  | _ -> false

(* Node bounds *)

let bound_to_int : Dtype.bound -> int = function
  | `Bool b -> Bool.to_int b
  | `SInt n | `UInt n ->
      if n < Int64.of_int Int.min_int then Int.min_int
      else if n > Int64.of_int Int.max_int then Int.max_int
      else Int64.to_int n
  | `Float f ->
      if Float.is_nan f then 0
      else if f <= Float.of_int Int.min_int then Int.min_int
      else if f >= Float.of_int Int.max_int then Int.max_int
      else Float.to_int f

let dtype_bounds n =
  match K.dtype n with
  | Some dt -> bound_to_int (Dtype.min dt), bound_to_int (Dtype.max dt)
  | None -> 0, Int.max_int

let rec node_bounds (n : K.t) : int * int =
  match K.view n with
  | Const { value; _ } -> (
      match Const.view value with
      | Int v -> let v = Int64.to_int v in v, v
      | Bool b -> let v = Bool.to_int b in v, v
      | Float _ -> dtype_bounds n)
  | Range { size; _ } | Special { size; _ } -> 0, snd (node_bounds size) - 1
  | Define_var { lo; hi; _ } -> lo, hi
  | Unary { op = `Neg; src; _ } ->
      let lo, hi = node_bounds src in -hi, -lo
  | Ternary { op = `Where; b; c; dtype } when Dtype.is_int dtype ->
      let b_lo, b_hi = node_bounds b in
      let c_lo, c_hi = node_bounds c in
      min b_lo c_lo, max b_hi c_hi
  | Gep { src; _ } | Unroll { src; _ } -> node_bounds src
  | Vectorize { srcs; _ } ->
      List.fold_left
        (fun (lo, hi) s ->
          let s_lo, s_hi = node_bounds s in
          min lo s_lo, max hi s_hi)
        (Int.max_int, Int.min_int) srcs
  | Cast { src; dtype } when Dtype.is_int (Dtype.any_to_val dtype) || Dtype.any_scalar dtype = Dtype.Index ->
      let dt = Dtype.any_to_val dtype in
      let s_lo, s_hi = node_bounds src in
      max (bound_to_int (Dtype.min dt)) s_lo,
      min s_hi (bound_to_int (Dtype.max dt))
  | Cast { src; _ } | Bitcast { src; _ } -> node_bounds src
  | Binary { op; lhs; rhs; dtype } when not (Dtype.is_float dtype) ->
      let s0_lo, s0_hi = node_bounds lhs in
      let s1_lo, s1_hi = node_bounds rhs in
      (match op with
      | `Add -> s0_lo + s1_lo, s0_hi + s1_hi
      | `Sub -> s0_lo - s1_hi, s0_hi - s1_lo
      | `Mul ->
          let a = s0_lo * s1_lo and b = s0_lo * s1_hi in
          let c = s0_hi * s1_lo and d = s0_hi * s1_hi in
          min (min a b) (min c d), max (max a b) (max c d)
      | `And when Dtype.is_int dtype && s1_lo = s1_hi && s1_lo >= 0 ->
          0, (if s0_lo < 0 then s1_hi else min s0_hi s1_hi)
      | `And when Dtype.is_bool dtype ->
          Bool.to_int (s0_lo > 0 && s1_lo > 0),
          Bool.to_int (s0_hi > 0 && s1_hi > 0)
      | `Or when Dtype.is_bool dtype ->
          Bool.to_int (s0_lo > 0 || s1_lo > 0),
          Bool.to_int (s0_hi > 0 || s1_hi > 0)
      | `Shl when s1_lo = s1_hi && s1_lo >= 0 && s1_lo < Sys.int_size - 1 ->
          s0_lo lsl s1_lo, s0_hi lsl s1_lo
      | `Shr when s1_lo = s1_hi && s1_lo >= 0 && s1_lo < Sys.int_size - 1 ->
          s0_lo asr s1_lo, s0_hi asr s1_lo
      | `Max -> max s0_lo s1_lo, max s0_hi s1_hi
      | `Mod ->
          if s1_lo = s1_hi && s1_lo > 0 then
            (if s0_lo >= 0 then 0
             else if s0_lo > -s1_lo then s0_lo
             else -(s1_hi - 1)),
            (if s0_hi < 0 then 0
             else if s0_hi < s1_lo then s0_hi
             else s1_lo - 1)
          else if s1_lo > 0 then
            (if s0_lo >= 0 then 0
             else if s0_hi <= 0 then -(s1_hi - 1)
             else -(s1_hi - 1)),
            (if s0_hi <= 0 then 0
             else if s0_lo >= 0 then s1_hi - 1
             else s1_hi - 1)
          else dtype_bounds n
      | `Idiv ->
          if s1_lo * s1_hi > 0 then
            let a = s0_lo / s1_lo and b = s0_lo / s1_hi in
            let c = s0_hi / s1_lo and d = s0_hi / s1_hi in
            min (min a b) (min c d), max (max a b) (max c d)
          else dtype_bounds n
      | `Cmplt ->
          Bool.to_int (s0_hi < s1_lo), Bool.to_int (s0_lo < s1_hi)
      | `Cmpne ->
          Bool.to_int (s0_hi < s1_lo || s1_hi < s0_lo),
          Bool.to_int (not (s0_lo = s0_hi && s0_lo = s1_lo && s1_lo = s1_hi))
      | `Cmpeq ->
          Bool.to_int (s0_lo = s0_hi && s0_lo = s1_lo && s1_lo = s1_hi),
          Bool.to_int (s0_lo <= s1_hi && s1_lo <= s0_hi)
      | _ -> dtype_bounds n)
  | _ -> dtype_bounds n

let node_vmin n = fst (node_bounds n)
let node_vmax n = snd (node_bounds n)

(* Apply substitutions from ctx, clear ctx, simplify result.
   Returns None if substitution is a no-op. *)
let do_substitute ctx x sub_fxn =
  let mappings =
    K.Ref_tbl.fold
      (fun k v acc ->
        match v with Some v -> (k, sub_fxn k v) :: acc | None -> acc)
      ctx []
  in
  K.Ref_tbl.reset ctx;
  if mappings = [] then None
  else
    let ret = K.graph_rewrite Symbolic.symbolic (K.substitute mappings x) in
    if ret = x then None else Some ret

(* Flatten range *)

let flatten_range (node : K.t) : K.t option =
  match K.view node with
  | Reduce _ | Store _ | End _ -> (
      match K.range_start node with
      | None -> None
      | Some off ->
          let ch = K.children node in
          let prefix = List.filteri (fun i _ -> i < off) ch in
          let rngs = List.filteri (fun i _ -> i >= off) ch in
          if rngs = [] then None
          else
            let new_rngs = List.filter K.is_range (K.toposort (K.sink rngs)) in
            let result = K.replace node ~children:(prefix @ new_rngs) () in
            if result = node then None else Some result)
  | _ -> None

let pm_flatten_range root = K.graph_rewrite flatten_range root

(* Split ranges *)

let split_ranges (root : K.t) : K.t =
  let ctx : K.t option K.Ref_tbl.t = K.Ref_tbl.create 16 in
  let rule node =
    match K.view node with
    | Binary { op = `Mod; lhs; rhs; _ } -> (
        match K.view lhs, K.view rhs with
        | Range { size; _ }, Const { value = c_val; _ } ->
            if not (K.Ref_tbl.mem ctx lhs) then begin
              match K.view size with
              | Const { value = sz_val; _ } -> (
                  match Const.view sz_val, Const.view c_val with
                  | Int sz_int, Int c_int
                    when c_int <> 0L && Int64.rem sz_int c_int = 0L ->
                      K.Ref_tbl.replace ctx lhs (Some rhs)
                  | _ -> ())
              | _ -> ()
            end;
            None
        | _ -> None)
    (* Image stores: don't substitute their ranges *)
    | Store { dst; _ } -> (
        match K.view dst with
        | Index { ptr; _ } -> (
            match K.view ptr with
            | Param_image _ ->
                List.iter
                  (fun r -> K.Ref_tbl.replace ctx r None)
                  (K.live_ranges dst)
            | _ -> ())
        | _ -> ());
        None
    | Sink _ ->
        do_substitute ctx node (fun k v ->
            let open K.O in
            let size = K.range_size k and axis = K.range_axis k in
            let sub = K.range_sub k and kind = K.range_kind k in
            let dt = K.dtype_or Dtype.index k in
            let outer =
              K.range ~size:(size / v) ~axis ~sub:(sub @ [0]) ~kind ~dtype:dt ()
            in
            let inner =
              K.range ~size:v ~axis ~sub:(sub @ [1]) ~kind ~dtype:dt ()
            in
            (outer * v) + inner)
    | _ -> None
  in
  K.graph_rewrite ~name:"split ranges" rule root

let pm_split_ranges root =
  let rec loop node =
    let node' = split_ranges node in
    if node' = node then node else loop node'
  in
  loop root

(* Simplify ranges *)

(* Merge two adjacent Range nodes in an End or Reduce into a single Range whose
   size is the product of the originals.  The inner references are replaced with
   divmod expressions (merged/s1, merged mod s1) and the result is kept only
   when it does not increase the total divmod count. *)
let simplify_merge_adjacent (u : K.t) : K.t option =
  match K.view u with
  | End _ | Reduce _ ->
      let u_ended = K.ended_ranges u in
      if u_ended = [] then None
      else begin
        let reduce_ranges_list =
          List.filter_map
            (fun x ->
              match K.view x with
              | Reduce { ranges; _ } -> Some ranges
              | _ -> None)
            (K.backward_slice u)
        in
        let pairs =
          match K.view u with
          | End _ ->
              let rec adj = function
                | a :: (b :: _ as rest) -> (a, b) :: adj rest
                | _ -> []
              in
              adj u_ended
          | _ ->
              List.concat_map
                (fun r0 ->
                  List.filter_map
                    (fun r1 -> if r0 == r1 then None else Some (r0, r1))
                    u_ended)
                u_ended
        in
        let result = ref u in
        List.iter
          (fun (r0, r1) ->
            if K.is_range r0 && K.is_range r1
               && K.range_kind r0 = K.range_kind r1
            then begin
              let same_reduces =
                List.for_all
                  (fun rngs ->
                    let has_r0 = List.exists (fun x -> x == r0) rngs in
                    let has_r1 = List.exists (fun x -> x == r1) rngs in
                    has_r0 = has_r1)
                  reduce_ranges_list
              in
              if same_reduces then begin
                let s0 = K.range_size r0 and s1 = K.range_size r1 in
                let open K.O in
                let merged =
                  K.range ~size:(s0 * s1) ~axis:(K.range_axis r0)
                    ~kind:(K.range_kind r0)
                    ~dtype:(K.dtype_or Dtype.index r0) ()
                in
                let nidx =
                  K.substitute [(r0, merged / s1); (r1, merged mod s1)] !result
                in
                let nidx =
                  K.graph_rewrite
                    (K.first_match [Symbolic.symbolic; flatten_range])
                    nidx
                in
                if count_divmod nidx <= count_divmod !result then
                  result := nidx
              end
            end)
          pairs;
        if !result == u then None else Some !result
      end
  | _ -> None

(* Walk the DAG to tighten Range sizes: extract guard conditions (r < C) from
   masked Where/Invalid_index patterns inside Index nodes, keep the tightest
   bound per range, then substitute narrowed ranges at every Sink. *)
let simplify_ranges (root : K.t) : K.t =
  let ctx : K.t option K.Ref_tbl.t = K.Ref_tbl.create 16 in
  let rule node =
    match K.view node with
    | End _ | Reduce _ -> (
        let merged = simplify_merge_adjacent node in
        (match K.view node with
        | Reduce { ranges; _ } ->
            List.iter
              (fun r -> K.Ref_tbl.replace ctx r (Some (K.range_size r)))
              ranges
        | _ -> ());
        merged)
    | Index _ ->
        let ch = K.children node in
        let idx_value =
          match ch with _ :: v :: _ -> v | _ -> List.hd ch
        in
        let x, guards =
          match K.view idx_value with
          | Ternary { op = `Where; a = cond; b = x; c = invalid; _ } -> (
              match K.view invalid with
              | Invalid_index _ ->
                  let guard_tbl = K.Ref_tbl.create 8 in
                  let rec split_and c =
                    match K.view c with
                    | Binary { op = `And; lhs; rhs; _ } ->
                        split_and lhs @ split_and rhs
                    | _ -> [c]
                  in
                  List.iter
                    (fun v ->
                      match K.view v with
                      | Binary { op = `Cmplt; lhs = r; rhs = c; _ }
                        when K.is_range r && K.is_const c ->
                          K.Ref_tbl.replace guard_tbl r c
                      | _ -> ())
                    (split_and cond);
                  (x, guard_tbl)
              | _ -> (idx_value, K.Ref_tbl.create 0))
          | _ -> (idx_value, K.Ref_tbl.create 0)
        in
        (* Filter guards that aren't tighter than the range size, for
           robustness when symbolic normalization is incomplete. *)
        K.Ref_tbl.iter
          (fun r c ->
            let is_tighter =
              match K.view c, K.view (K.range_size r) with
              | Const { value = cv; _ }, Const { value = sv; _ } -> (
                  match Const.view cv, Const.view sv with
                  | Int ci, Int si -> Int64.compare ci si <= 0
                  | _ -> true)
              | _ -> true
            in
            if is_tighter then begin
              let update =
                match K.Ref_tbl.find_opt ctx r with
                | None -> true
                | Some (Some existing) -> (
                    match K.view existing, K.view c with
                    | Const { value = ev; _ }, Const { value = cv; _ } -> (
                        match Const.view ev, Const.view cv with
                        | Int ei, Int ci -> Int64.compare ci ei > 0
                        | _ -> false)
                    | _ -> false)
                | Some None -> false
              in
              if update then K.Ref_tbl.replace ctx r (Some c)
            end)
          guards;
        List.iter
          (fun r ->
            if not (K.Ref_tbl.mem guards r) then
              K.Ref_tbl.replace ctx r (Some (K.range_size r)))
          (K.live_ranges x);
        None
    | Sink _ ->
        do_substitute ctx node (fun r c ->
            K.range ~size:c ~axis:(K.range_axis r) ~kind:(K.range_kind r)
              ~dtype:(K.dtype_or Dtype.index r) ())
    | _ -> None
  in
  K.graph_rewrite ~name:"simplify ranges"
    (fun node ->
      match rule node with Some _ as r -> r | None -> flatten_range node)
    root

let pm_simplify_ranges root =
  let rec loop node =
    let node' = simplify_ranges node in
    if node' = node then node else loop node'
  in
  loop root

(* Reduce unparented *)

let reduce_unparented (node : K.t) : K.t option =
  match K.view node with
  | Reduce { op; src; ranges; dtype } ->
      assert (List.for_all K.is_range ranges);
      let src_range_set = K.Ref_tbl.create 16 in
      List.iter (fun r -> K.Ref_tbl.replace src_range_set r ()) (K.live_ranges src);
      let parented, unparented =
        List.partition (fun r -> K.Ref_tbl.mem src_range_set r) ranges
      in
      if unparented = [] then None
      else
        let ret =
          if parented <> []
             || not (Dtype.equal dtype (K.dtype_or Dtype.void src))
          then K.reduce ~op ~src ~ranges:parented ~dtype
          else src
        in
        let compensate combine_op acc =
          List.fold_left
            (fun acc r ->
              let scalar_dt = Dtype.scalar_of dtype in
              let casted = K.cast ~src:(K.range_size r) ~dtype:(Dtype.to_any scalar_dt) in
              let v =
                if Dtype.count dtype > 1 then K.broadcast casted (Dtype.count dtype)
                else casted
              in
              K.binary ~op:combine_op ~lhs:acc ~rhs:v)
            acc unparented
        in
        Some
          (match op with
          | `Add -> compensate `Mul ret
          | `Mul -> compensate `Pow ret
          | `Max -> ret)
  | _ -> None

let pm_reduce_unparented root = K.graph_rewrite reduce_unparented root

(* Reduce collapse *)

let toposort_gated (gate : K.t -> bool) (root : K.t) : K.t list =
  let visited = K.Ref_tbl.create 64 in
  let order = ref [] in
  let rec visit node =
    if not (K.Ref_tbl.mem visited node) && gate node then begin
      K.Ref_tbl.replace visited node ();
      List.iter visit (K.children node);
      order := node :: !order
    end
  in
  visit root;
  List.rev !order

(* Collapse a single-range additive Reduce by isolating the range-dependent
   subgraph, replacing external dependencies with fresh define_var proxies,
   building a standalone Reduce over that subgraph, simplifying it, and
   substituting the proxies back.  Fails if the collapsed result still
   contains live ranges (meaning the reduction could not be fully resolved). *)
let reduce_collapse_inner ~(pm : K.t -> K.t option) (red : K.t) (u : K.t) :
    K.t option =
  match K.view red with
  | Reduce { op = `Add; ranges; _ } ->
      let result = ref u in
      let failed = ref false in
      List.iter
        (fun r ->
          if not !failed then begin
            let lr_tbl = K.live_ranges_tbl !result in
            let included =
              toposort_gated
                (fun x ->
                  match K.Ref_tbl.find_opt lr_tbl x with
                  | Some rngs -> List.exists (fun xr -> xr == r) rngs
                  | None -> false)
                !result
            in
            if List.exists
                 (fun x ->
                   match K.view x with
                   | Store _ | Reduce _ -> true
                   | _ -> false)
                 included
            then failed := true
            else begin
              let included_set = K.Ref_tbl.create 32 in
              List.iter
                (fun x -> K.Ref_tbl.replace included_set x ())
                included;
              let replaces = K.Ref_tbl.create 16 in
              let var_count = ref 0 in
              List.iter
                (fun u_node ->
                  List.iter
                    (fun s ->
                      if K.Ref_tbl.mem included_set s
                         || K.Ref_tbl.mem replaces s
                      then ()
                      else
                        match K.view s with
                        | Const _ | Define_var _ | Param _
                        | Define_local _ -> ()
                        | _ ->
                            let name = Printf.sprintf "in%d" !var_count in
                            incr var_count;
                            K.Ref_tbl.replace replaces s
                              (K.define_var ~name
                                 ~lo:(node_vmin s) ~hi:(node_vmax s) ()))
                    (K.children u_node))
                included;
              let fwd_mappings =
                K.Ref_tbl.fold (fun k v acc -> (k, v) :: acc) replaces []
              in
              let collapse_fxn =
                K.reduce ~op:`Add ~src:(K.substitute fwd_mappings !result)
                  ~ranges:[r] ~dtype:(K.dtype_or Dtype.void !result)
              in
              let sink =
                K.graph_rewrite
                  (K.first_match [reduce_unparented; pm; Symbolic.symbolic])
                  collapse_fxn
              in
              if not (no_range sink) then failed := true
              else
                let rev_mappings =
                  K.Ref_tbl.fold (fun k v acc -> (v, k) :: acc) replaces []
                in
                result := K.substitute rev_mappings sink
            end
          end)
        ranges;
      if !failed || !result == u then None else Some !result
  | _ -> None

(* Reduce collapse rules: single-range folds, general rules, and lift rules. *)

let reduce_fold_rule r dtype src =
  match K.view src with
  | Ternary { op = `Where; a = cond; b = val_true; c = val_false; _ } -> (
      match K.view cond with
      | Binary { op = `Cmplt; lhs = cond_r; rhs = cut; _ }
        when cond_r == r && is_zero val_false && no_range val_true ->
          let open K.O in
          let clamped = minimum (maximum cut (int_ 0)) (K.range_size r) in
          Some (K.cast ~src:clamped ~dtype:(Dtype.to_any (Dtype.scalar_of dtype)) * val_true)
      | Binary { op = `Cmplt; lhs = cond_r; rhs = cut; _ }
        when cond_r == r && is_zero val_true && no_range val_false ->
          let open K.O in
          let r_size = K.range_size r in
          let count = minimum (maximum (r_size + neg cut) (int_ 0)) r_size in
          Some (K.cast ~src:count ~dtype:(Dtype.to_any (Dtype.scalar_of dtype)) * val_false)
      | Binary { op = `And; lhs = and_lhs; rhs = and_rhs; _ }
        when is_zero val_false -> (
          match K.view and_lhs, K.view and_rhs with
          | Binary { op = `Cmpeq; lhs = lower_cond; rhs = false_const; _ },
            Binary { op = `Cmplt; lhs = upper_r; rhs = upper; _ } -> (
              match K.view lower_cond with
              | Binary { op = `Cmplt; lhs = lower_r; rhs = lower; _ }
                when lower_r == r && upper_r == r
                     && is_zero false_const && no_range val_true ->
                  let open K.O in
                  let r_size = K.range_size r in
                  let count =
                    minimum
                      (maximum
                         (minimum upper r_size + neg (maximum lower (int_ 0)))
                         (int_ 0))
                      r_size
                  in
                  Some
                    (K.cast ~src:count ~dtype:(Dtype.to_any (Dtype.scalar_of dtype)) * val_true)
              | _ -> None)
          | _ -> None)
      | _ -> None)
  | _ -> None

let reduce_general_rule ranges dtype src =
  match K.view src with
  | Binary { op = `Add; lhs = x; rhs = y; _ } ->
      Some
        (K.binary ~op:`Add
           ~lhs:(K.reduce ~op:`Add ~src:x ~ranges ~dtype)
           ~rhs:(K.reduce ~op:`Add ~src:y ~ranges ~dtype))
  | Ternary { op = `Where; a = cond; b = val_true; c = val_false; _ }
    when is_zero val_false -> (
      match K.view cond with
      | Binary { op = `And; lhs = and_lhs; rhs = and_rhs; _ } -> (
          match K.view and_lhs with
          | Define_var _ ->
              let inner =
                K.ternary ~op:`Where ~a:and_rhs ~b:val_true ~c:val_false
              in
              Some
                (K.binary ~op:`Mul
                   ~lhs:(K.reduce ~op:`Add ~src:inner ~ranges ~dtype)
                   ~rhs:(K.cast ~src:and_lhs
                            ~dtype:(Dtype.to_any (K.dtype_or Dtype.index val_true))))
          | _ -> None)
      | _ -> None)
  | _ -> None

(* Lift addition/multiplication out of comparisons for reduce collapse *)
let lift_add_from_cmp ~cmp_op lhs c =
  let inner = peel_cast lhs in
  match K.view inner with
  | Binary { op = `Add; lhs = x; rhs = y; _ } ->
      if no_range y && no_range c then
        let open K.O in
        let y_dt = K.dtype_or Dtype.index y in
        Some (K.binary ~op:cmp_op ~lhs:x ~rhs:(K.cast ~src:c ~dtype:(Dtype.to_any y_dt) + neg y))
      else None
  | _ -> None

let pm_reduce_collapse_rule (node : K.t) : K.t option =
  match K.view node with
  | Binary { op = `Cmplt; lhs; rhs = c; _ } -> (
      match lift_add_from_cmp ~cmp_op:`Cmplt lhs c with
      | Some _ as r -> r
      | None ->
          let inner = peel_cast lhs in
          match K.view inner with
          | Binary { op = `Mul; lhs = x; rhs = y; _ }
            when no_range y && no_range c
                 && Dtype.is_int (K.dtype_or Dtype.void y)
                 && node_vmin y > 0 ->
              let open K.O in
              Some (x < ((c + y + neg (int_ 1)) / y))
          | _ -> None)
  | Reduce { op = `Add; src; ranges; dtype } when ranges <> [] -> (
      match ranges with
      | [r] -> (
          match reduce_fold_rule r dtype src with
          | Some _ as r -> r
          | None -> reduce_general_rule ranges dtype src)
      | _ -> reduce_general_rule ranges dtype src)
  | Binary { op = `Mul; lhs = x; rhs = gate_cast; _ } -> (
      match K.view gate_cast with
      | Cast { src = gate; _ } -> (
          match K.dtype gate with
          | Some dt when Dtype.scalar dt = Dtype.Bool ->
              Some (K.ternary ~op:`Where ~a:gate ~b:x ~c:(K.zero_like x))
          | _ -> None)
      | _ -> None)
  | _ -> None

let reduce_collapse red u =
  reduce_collapse_inner ~pm:pm_reduce_collapse_rule red u

(* pm_reduce_simplify: combines reduce_unparented + reduce_collapse *)

let pm_reduce_simplify_rule (node : K.t) : K.t option =
  match reduce_unparented node with
  | Some _ as r -> r
  | None -> (
      match K.view node with
      | Reduce { op = `Add; src = u; ranges; _ } when ranges <> [] ->
          reduce_collapse node u
      | _ -> None)

let pm_reduce_simplify root = K.graph_rewrite pm_reduce_simplify_rule root

(* Load collapse *)

let pm_reduce_load_collapse_rule (node : K.t) : K.t option =
  match K.view node with
  | Binary { op = `Cmpne; lhs; rhs = c; _ } ->
      lift_add_from_cmp ~cmp_op:`Cmpne lhs c
  | Reduce { op = `Add; src; ranges = [r]; _ } -> (
      match K.view src with
      | Ternary { op = `Where; a = cond; b = zero; c = expr; _ }
        when is_zero zero -> (
          match K.view cond with
          | Binary { op = `Cmpne; lhs = idx; rhs = ne_rhs; _ } ->
              if peel_cast ne_rhs == r then
                let open K.O in
                let r_dt = K.dtype_or Dtype.index r in
                let idx_cast = K.cast ~src:idx ~dtype:(Dtype.to_any r_dt) in
                let valid_cond =
                  K.binary ~op:`And
                    ~lhs:(K.binary ~op:`Cmpeq
                            ~lhs:(K.binary ~op:`Cmplt ~lhs:idx_cast
                                    ~rhs:(K.cast ~src:(int_ 0) ~dtype:(Dtype.to_any r_dt)))
                            ~rhs:(K.const_bool false))
                    ~rhs:(idx_cast < K.range_size r)
                in
                let valid_idx =
                  K.ternary ~op:`Where ~a:valid_cond ~b:idx_cast
                    ~c:(K.invalid_index ())
                in
                let subst_expr = K.substitute [(r, valid_idx)] expr in
                Some
                  (K.ternary ~op:`Where ~a:valid_cond ~b:subst_expr
                     ~c:(K.zero_like expr))
              else None
          | _ -> None)
      | _ -> pm_reduce_collapse_rule node)
  | _ -> pm_reduce_collapse_rule node

let reduce_load_collapse red u =
  reduce_collapse_inner ~pm:pm_reduce_load_collapse_rule red u

let pm_load_collapse_rule (node : K.t) : K.t option =
  match K.view node with
  | Reduce { op = `Add; src = u; ranges = [_]; _ } ->
      reduce_load_collapse node u
  | Binary { op = `Cmplt; lhs = x; rhs = c; _ } -> (
      match K.view x with
      | Binary { op = `Add; lhs = x_inner; rhs = y; _ } -> (
          match K.dtype x_inner with
          | Some dt when Dtype.scalar dt = Dtype.Index ->
              if no_load y && no_load c && not (no_load x_inner) then
                let open K.O in
                Some (x_inner < (c + neg y))
              else None
          | _ -> None)
      | _ -> None)
  | _ -> None

let pm_load_collapse root = K.graph_rewrite ~name:"load collapse" pm_load_collapse_rule root
