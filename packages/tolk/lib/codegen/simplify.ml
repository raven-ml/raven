(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_ir
module K = Kernel

(* Helpers *)

let no_range u =
  not (List.exists K.is_range (K.backward_slice u))

let no_load u =
  not (List.exists (fun n ->
    match K.view n with Index _ -> true | _ -> false)
    (K.backward_slice u))

let is_divmod n = match K.view n with
  | Binary { op = `Idiv | `Mod; _ } -> true | _ -> false

let count_divmod x =
  List.length (List.filter (fun n -> n != x && is_divmod n)
    (K.backward_slice x))

let is_zero n = match K.const_arg n with
  | Some (Int 0L) | Some (Bool false) -> true
  | Some (Float f) -> f = 0.0
  | _ -> false

let peel_cast n = match K.view n with Cast { src; _ } -> src | _ -> n

let minimum a b =
  K.ternary ~op:`Where ~a:(K.binary ~op:`Cmplt ~lhs:a ~rhs:b) ~b:a ~c:b

let maximum a b = K.binary ~op:`Max ~lhs:a ~rhs:b

let mem_phys x xs = List.exists (fun y -> y == x) xs

let split_and c =
  let rec go c = match K.view c with
    | Binary { op = `And; lhs; rhs; _ } -> go lhs @ go rhs
    | _ -> [c] in
  go c

let rec list_take n = function
  | _ when n <= 0 -> []
  | x :: xs -> x :: list_take (n - 1) xs
  | [] -> []

let rec list_drop n = function
  | l when n <= 0 -> l
  | _ :: xs -> list_drop (n - 1) xs
  | [] -> []

(* Toposort-reorder range children of Reduce/Store/End. *)
let flatten_range node =
  match K.view node with
  | Reduce _ | Store _ | End _ ->
      (match K.range_start node with
       | None -> None
       | Some off ->
           let ch = K.children node in
           let rngs = list_drop off ch in
           if rngs = [] then None
           else
             let new_rngs =
               List.filter K.is_range (K.toposort (K.sink rngs)) in
             let result =
               K.replace node ~children:(list_take off ch @ new_rngs) () in
             if result = node then None else Some result)
  | _ -> None

let pm_flatten_range root = K.graph_rewrite flatten_range root

(* Apply substitutions from ctx, clear ctx, simplify result. *)
let do_substitute ctx x sub_fxn =
  let mappings = K.Ref_tbl.fold (fun k v acc ->
    match v with Some v -> (k, sub_fxn k v) :: acc | None -> acc) ctx [] in
  K.Ref_tbl.reset ctx;
  if mappings = [] then None
  else
    let ret = K.graph_rewrite Symbolic.symbolic (K.substitute mappings x) in
    if ret = x then None else Some ret

(* Merge two adjacent ranges into one whose size is the product of the
   originals.  Kept only when divmod count does not increase. *)
let simplify_merge_adjacent u =
  match K.view u with
  | End _ | Reduce _ ->
      let u_ended = K.ended_ranges u in
      if u_ended = [] then None
      else begin
        let reduce_ranges =
          List.filter_map (fun x -> match K.view x with
            | Reduce { ranges; _ } -> Some ranges | _ -> None)
            (K.backward_slice u) in
        let pairs = match K.view u with
          | End _ ->
              let rec adj = function
                | a :: (b :: _ as rest) -> (a, b) :: adj rest | _ -> [] in
              adj u_ended
          | _ ->
              List.concat_map (fun r0 ->
                List.filter_map (fun r1 ->
                  if r0 == r1 then None else Some (r0, r1)) u_ended)
                u_ended in
        let result = ref u in
        List.iter (fun (r0, r1) ->
          if K.range_kind r0 = K.range_kind r1
             && List.for_all (fun rngs ->
                  mem_phys r0 rngs = mem_phys r1 rngs) reduce_ranges
          then begin
            let open K.O in
            let s0 = K.range_size r0 and s1 = K.range_size r1 in
            let merged = K.range ~size:(s0 * s1) ~axis:(K.range_axis r0)
              ~kind:(K.range_kind r0)
              ~dtype:(Dtype.val_of (K.dtype r0)) () in
            let nidx = K.substitute
              [(r0, merged / s1); (r1, merged mod s1)] !result in
            let nidx = K.graph_rewrite
              (K.first_match [Symbolic.symbolic; flatten_range]) nidx in
            if count_divmod nidx <= count_divmod !result then
              result := nidx
          end) pairs;
        if !result == u then None else Some !result
      end
  | _ -> None

(* Extract r<C guards from gated Index nodes, track the tightest bound
   per range, mark reduce ranges unshrinkable, substitute at Sink. *)
let simplify_ranges root =
  let ctx : K.t option K.Ref_tbl.t = K.Ref_tbl.create 16 in
  let mark_unshrinkable r =
    K.Ref_tbl.replace ctx r (Some (K.range_size r)) in
  let extract_guards idx_value =
    match K.view idx_value with
    | Ternary { op = `Where; a = cond; b = x; c = invalid; _ } ->
        (match K.view invalid with
         | Invalid_index _ ->
             let tbl = K.Ref_tbl.create 8 in
             List.iter (fun v -> match K.view v with
               | Binary { op = `Cmplt; lhs = r; rhs = c; _ }
                 when K.is_range r && K.is_const c ->
                   K.Ref_tbl.replace tbl r c
               | _ -> ()) (split_and cond);
             (x, tbl)
         | _ -> (idx_value, K.Ref_tbl.create 0))
    | _ -> (idx_value, K.Ref_tbl.create 0) in
  let rule node =
    match K.view node with
    | End _ | Reduce _ ->
        (match simplify_merge_adjacent node with
         | Some _ as merged -> merged
         | None ->
             (match K.view node with
              | Reduce { ranges; _ } -> List.iter mark_unshrinkable ranges
              | _ -> ());
             None)
    | Index _ ->
        let ch = K.children node in
        let idx_value = match ch with _ :: v :: _ -> v | _ -> List.hd ch in
        let x, guards = extract_guards idx_value in
        let x = if K.Ref_tbl.length guards = 0 then node else x in
        K.Ref_tbl.iter (fun r c ->
          let dominated = match K.Ref_tbl.find_opt ctx r with
            | Some (Some existing) ->
                (match K.const_arg existing, K.const_arg c with
                 | Some (Int ei), Some (Int ci) -> Int64.compare ci ei <= 0
                 | _ -> true)
            | Some None -> true
            | None -> false in
          if not dominated then K.Ref_tbl.replace ctx r (Some c)) guards;
        List.iter (fun r ->
          if not (K.Ref_tbl.mem guards r) then mark_unshrinkable r)
          (K.live_ranges x);
        None
    | Sink _ ->
        do_substitute ctx node (fun r c ->
          K.range ~size:c ~axis:(K.range_axis r) ~kind:(K.range_kind r)
            ~dtype:(Dtype.val_of (K.dtype r)) ())
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

let is_image_store node = match K.view node with
  | Store { dst; _ } ->
      (match K.view dst with
       | Index { ptr; _ } ->
           (match K.view ptr with Param_image _ -> true | _ -> false)
       | _ -> false)
  | _ -> false

let can_split_range r c =
  K.is_range r && K.is_const c
  && K.range_kind r <> Axis_kind.Warp
  && K.is_const (K.range_size r)
  && K.divides (K.range_size r) (K.const_to_int c) <> None

(* Split ranges where range_size divides the modulus constant.
   range(N) % C where N|C becomes outer(N/C)*C + inner(C). *)
let split_ranges root =
  let ctx : K.t option K.Ref_tbl.t = K.Ref_tbl.create 16 in
  let rule node =
    match K.view node with
    | Binary { op = `Mod; lhs = r; rhs = c; _ }
      when can_split_range r c && not (K.Ref_tbl.mem ctx r) ->
        K.Ref_tbl.replace ctx r (Some c); None
    | _ when is_image_store node ->
        let dst = List.hd (K.children node) in
        List.iter (fun r -> K.Ref_tbl.replace ctx r None)
          (K.live_ranges dst);
        None
    | Sink _ ->
        do_substitute ctx node (fun k v ->
          let open K.O in
          let size = K.range_size k and axis = K.range_axis k in
          let sub = K.range_sub k and kind = K.range_kind k in
          let dt = Dtype.val_of (K.dtype k) in
          let outer = K.range ~size:(size / v) ~axis
            ~sub:(sub @ [0]) ~kind ~dtype:dt () in
          let inner = K.range ~size:v ~axis
            ~sub:(sub @ [1]) ~kind ~dtype:dt () in
          (outer * v) + inner)
    | _ -> None
  in
  K.graph_rewrite ~name:"split ranges"
    (fun node ->
      match rule node with Some _ as r -> r | None -> flatten_range node)
    root

let pm_split_ranges root =
  let rec loop node =
    let node' = split_ranges node in
    if node' = node then node else loop node'
  in
  loop root

(* Remove ranges from a Reduce that aren't referenced in the source.
   Compensate: ADD → multiply by range size, MUL → exponentiate. *)
let reduce_unparented node =
  match K.view node with
  | Reduce { op; src; ranges; dtype }
    when op = `Add || op = `Max || op = `Mul ->
      assert (List.for_all K.is_range ranges);
      let src_ranges = K.live_ranges src in
      let parented, unparented =
        List.partition (fun r -> mem_phys r src_ranges) ranges in
      if unparented = [] then None
      else
        let ret =
          if parented <> [] || not (Dtype.equal (Dtype.Val dtype) (K.dtype src))
          then K.reduce ~op ~src ~ranges:parented ~dtype
          else src in
        let range_size_broadcast r =
          let s = K.cast ~src:(K.range_size r)
            ~dtype:(Dtype.scalarize (Dtype.Val dtype)) in
          K.broadcast s (Dtype.Val.count dtype) in
        let compensate binop acc r =
          K.binary ~op:binop ~lhs:acc ~rhs:(range_size_broadcast r) in
        let ret = match op with
          | `Add -> List.fold_left (compensate `Mul) ret unparented
          | `Mul -> List.fold_left (compensate `Pow) ret unparented
          | _ -> ret in
        Some ret
  | _ -> None

let pm_reduce_unparented root = K.graph_rewrite reduce_unparented root

(* Gated toposort: only follow children where gate holds. *)
let toposort_gated gate root =
  let visited = K.Ref_tbl.create 64 in
  let order = ref [] in
  let rec visit node =
    if not (K.Ref_tbl.mem visited node) && gate node then begin
      K.Ref_tbl.replace visited node ();
      List.iter visit (K.children node);
      order := node :: !order
    end in
  visit root;
  List.rev !order

(* Fold rules for single-range reduce(add, where(r<cut, ...)) patterns.
   Each rule computes a count expression and returns cast(count, val.dtype) * val. *)
let fold_result count v =
  let dt = K.dtype v in
  K.binary ~op:`Mul
    ~lhs:(K.cast ~src:count ~dtype:dt) ~rhs:v

let reduce_fold_rule r _dtype src =
  let open K.O in
  let r_size = K.range_size r in
  match K.view src with
  | Ternary { op = `Where; a = cond; b = val_true; c = val_false; _ } ->
      (match K.view cond with
       | Binary { op = `Cmplt; lhs = cond_r; rhs = cut; _ }
         when cond_r == r && is_zero val_false && no_range val_true ->
           Some (fold_result (minimum (maximum cut (int_ 0)) r_size) val_true)
       | Binary { op = `Cmplt; lhs = cond_r; rhs = cut; _ }
         when cond_r == r && is_zero val_true && no_range val_false ->
           Some (fold_result
             (minimum (maximum (r_size + neg cut) (int_ 0)) r_size) val_false)
       | Binary { op = `And; lhs; rhs; _ } when is_zero val_false ->
           (match K.view lhs, K.view rhs with
            | Binary { op = `Cmpeq; lhs = lower_cond; rhs = false_const; _ },
              Binary { op = `Cmplt; lhs = upper_r; rhs = upper; _ } ->
                (match K.view lower_cond with
                 | Binary { op = `Cmplt; lhs = lower_r; rhs = lower; _ }
                   when lower_r == r && upper_r == r
                        && is_zero false_const && no_range val_true ->
                     let count = minimum
                       (maximum (minimum upper r_size + neg (maximum lower (int_ 0)))
                          (int_ 0)) r_size in
                     Some (fold_result count val_true)
                 | _ -> None)
            | _ -> None)
       | _ -> None)
  | _ -> None

(* General reduce rules: split ADD across reduce, AND-WHERE factoring. *)
let reduce_general_rule ranges dtype src =
  match K.view src with
  | Binary { op = `Add; lhs = x; rhs = y; _ } ->
      Some (K.binary ~op:`Add
        ~lhs:(K.reduce ~op:`Add ~src:x ~ranges ~dtype)
        ~rhs:(K.reduce ~op:`Add ~src:y ~ranges ~dtype))
  | Ternary { op = `Where; a = cond; b = val_true; c = val_false; _ }
    when is_zero val_false ->
      (match K.view cond with
       | Binary { op = `And; lhs = dv; rhs = rest; _ } ->
           (match K.view dv with
            | Define_var _ ->
                let inner = K.ternary ~op:`Where
                  ~a:rest ~b:val_true ~c:val_false in
                Some (K.binary ~op:`Mul
                  ~lhs:(K.reduce ~op:`Add ~src:inner ~ranges ~dtype)
                  ~rhs:(K.cast ~src:dv
                    ~dtype:(K.dtype val_true)))
            | _ -> None)
       | _ -> None)
  | _ -> None

(* Lift addition/multiplication out of comparisons for reduce collapse. *)
let lift_add_from_cmp ~cmp_op lhs c =
  let inner = peel_cast lhs in
  match K.view inner with
  | Binary { op = `Add; lhs = x; rhs = y; _ }
    when no_range y && no_range c ->
      let open K.O in
      let y_dt = K.dtype y in
      Some (K.binary ~op:cmp_op ~lhs:x
        ~rhs:(K.cast ~src:c ~dtype:y_dt + neg y))
  | _ -> None

(* Combined reduce collapse rewrite rule. *)
let pm_reduce_collapse_rule node =
  match K.view node with
  | Binary { op = `Cmplt; lhs; rhs = c; _ } ->
      (match lift_add_from_cmp ~cmp_op:`Cmplt lhs c with
       | Some _ as r -> r
       | None ->
           match K.view lhs with
           | Binary { op = `Mul; lhs = x; rhs = y; _ }
             when no_range y && no_range c
                  && Dtype.is_int (K.dtype y)
                  && K.vmin y > 0 ->
               let open K.O in
               Some (x < ((c + y + neg (int_ 1)) / y))
           | _ -> None)
  | Reduce { op = `Add; src; ranges; dtype } when ranges <> [] ->
      let folded = match ranges with
        | [r] -> reduce_fold_rule r dtype src | _ -> None in
      (match folded with
       | Some _ -> folded
       | None -> reduce_general_rule ranges dtype src)
  | Binary { op = `Mul; lhs = x; rhs = gate_cast; _ } ->
      (match K.view gate_cast with
       | Cast { src = gate; _ } ->
           (match K.dtype_opt gate with
            | Some dt when Dtype.scalar dt = Dtype.Bool ->
                Some (K.ternary ~op:`Where ~a:gate ~b:x
                  ~c:(K.zero_like x))
            | _ -> None)
       | _ -> None)
  | _ -> None

(* Nodes that don't need proxy replacement in reduce collapse. *)
let is_leaf n = match K.view n with
  | Const _ | Vconst _ | Define_var _ | Param _ | Define_local _ -> true
  | _ -> false

let has_store_or_reduce nodes =
  List.exists (fun x -> match K.view x with
    | Store _ | Reduce _ -> true | _ -> false) nodes

(* Isolate range-dependent subgraph, replace externals with define_var
   proxies, build a standalone Reduce, simplify, substitute back. *)
let reduce_collapse_inner ~pm red u =
  match K.view red with
  | Reduce { op = `Add; ranges; _ } ->
      let result = ref u in
      let failed = ref false in
      List.iter (fun r ->
        if not !failed then begin
          let lr_tbl = K.live_ranges_tbl !result in
          let included = toposort_gated (fun x ->
            match K.Ref_tbl.find_opt lr_tbl x with
            | Some rngs -> mem_phys r rngs | None -> false) !result in
          if has_store_or_reduce included then failed := true
          else begin
            let in_set = K.Ref_tbl.create 32 in
            List.iter (fun x -> K.Ref_tbl.replace in_set x ()) included;
            let proxies = K.Ref_tbl.create 16 in
            let n = ref 0 in
            List.iter (fun u_node ->
              List.iter (fun s ->
                if not (K.Ref_tbl.mem in_set s || K.Ref_tbl.mem proxies s
                        || is_leaf s) then begin
                  K.Ref_tbl.replace proxies s
                    (K.define_var ~name:(Printf.sprintf "in%d" !n)
                       ~lo:(K.vmin s) ~hi:(K.vmax s)
                       ~dtype:(Dtype.val_of (K.dtype s)) ());
                  incr n
                end) (K.children u_node)) included;
            let fwd = K.Ref_tbl.fold (fun k v acc -> (k, v) :: acc) proxies [] in
            let collapse_fxn = K.reduce ~op:`Add
              ~src:(K.substitute fwd !result) ~ranges:[r]
              ~dtype:(Dtype.val_of (K.dtype !result)) in
            let sink = K.graph_rewrite
              (K.first_match [reduce_unparented; pm; Symbolic.symbolic])
              collapse_fxn in
            if not (no_range sink) then failed := true
            else
              let rev = K.Ref_tbl.fold (fun k v acc -> (v, k) :: acc) proxies [] in
              result := K.substitute rev sink
          end
        end) ranges;
      if !failed || !result == u then None else Some !result
  | _ -> None

let reduce_collapse red u =
  reduce_collapse_inner ~pm:pm_reduce_collapse_rule red u

(* idx >= 0 & idx < size, as a validity condition for index substitution. *)
let valid_index_cond idx_cast r_dt r_size =
  let open K.O in
  let ge_zero = K.binary ~op:`Cmpeq
    ~lhs:(idx_cast < K.cast ~src:(int_ 0) ~dtype:r_dt)
    ~rhs:(K.const_bool false) in
  K.binary ~op:`And ~lhs:ge_zero ~rhs:(idx_cast < r_size)

(* Load-specific collapse rules: lift ne, gated-load substitution. *)
let pm_reduce_load_collapse_rule node =
  match K.view node with
  | Binary { op = `Cmpne; lhs; rhs = c; _ } ->
      lift_add_from_cmp ~cmp_op:`Cmpne lhs c
  | Reduce { op = `Add; src; ranges = [r]; _ } ->
      (match K.view src with
       | Ternary { op = `Where; a = cond; b = zero; c = expr; _ }
         when is_zero zero ->
           (match K.view cond with
            | Binary { op = `Cmpne; lhs = idx; rhs = ne_rhs; _ }
              when peel_cast ne_rhs == r ->
                let r_dt = K.dtype r in
                let idx_cast = K.cast ~src:idx ~dtype:r_dt in
                let valid = valid_index_cond idx_cast r_dt (K.range_size r) in
                let valid_idx = K.ternary ~op:`Where ~a:valid
                  ~b:idx_cast ~c:(K.invalid_index ()) in
                Some (K.ternary ~op:`Where ~a:valid
                  ~b:(K.substitute [(r, valid_idx)] expr)
                  ~c:(K.zero_like expr))
            | _ -> None)
       | _ -> pm_reduce_collapse_rule node)
  | _ -> pm_reduce_collapse_rule node

let reduce_load_collapse red u =
  reduce_collapse_inner ~pm:pm_reduce_load_collapse_rule red u

(* pm_reduce_simplify: reduce_unparented + reduce_collapse. *)
let pm_reduce_simplify_rule node =
  match reduce_unparented node with
  | Some _ as r -> r
  | None -> match K.view node with
    | Reduce { op = `Add; src = u; ranges; _ } when ranges <> [] ->
        reduce_collapse node u
    | _ -> None

let pm_reduce_simplify root = K.graph_rewrite pm_reduce_simplify_rule root

(* pm_load_collapse: reduce_load_collapse + lift rule for loaded indices. *)
let pm_load_collapse_rule node =
  match K.view node with
  | Reduce { op = `Add; src = u; ranges = [_]; _ } ->
      reduce_load_collapse node u
  | Binary { op = `Cmplt; lhs = x; rhs = c; _ } ->
      (match K.view x with
       | Binary { op = `Add; lhs = x_inner; rhs = y; _ } ->
           (match K.dtype_opt x_inner with
            | Some dt when Dtype.scalar dt = Dtype.Index ->
                if no_load y && no_load c && not (no_load x_inner) then
                  let open K.O in
                  Some (x_inner < (c + neg y))
                else None
            | _ -> None)
       | _ -> None)
  | _ -> None

let pm_load_collapse root =
  K.graph_rewrite ~name:"load collapse" pm_load_collapse_rule root
