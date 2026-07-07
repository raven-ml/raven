(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Port of tinygrad/schedule/indexing.py to the tolk_uop IR. *)

open Tolk_uop
module U = Uop

(* Ops that never need realization: they produce contiguous output, so
   their consumers can always index directly. *)
let always_contiguous = function
  | Ops.Contiguous | Ops.After | Ops.Copy | Ops.Buffer | Ops.Slice
  | Ops.Const | Ops.Bind | Ops.Mselect | Ops.Mstack | Ops.Param | Ops.Load
  | Ops.Call | Ops.Function ->
      true
  | _ -> false

(* Small helpers *)

let idx n = U.const (Const.int Dtype.Val.weakint n)
let btrue = U.const_bool true
let bfalse = U.const_bool false

let ( +! ) a b = U.alu_binary ~op:Ops.Add ~lhs:a ~rhs:b
let ( -! ) a b = U.alu_binary ~op:Ops.Sub ~lhs:a ~rhs:b
let ( *! ) a b = U.alu_binary ~op:Ops.Mul ~lhs:a ~rhs:b

(* Reshape re-derives index arithmetic via mod/div and then simplifies;
   tinygrad uses [symbolic + pm_simplify_valid + pm_drop_and_clauses] here. *)
let reshape_pm =
  Upat.Pattern_matcher.(
    Symbolic.symbolic ++ Symbolic.pm_simplify_valid
    ++ Symbolic.pm_drop_and_clauses)

(* Pad validity predicate simplification: [symbolic + pm_simplify_valid]. *)
let pad_pm =
  Upat.Pattern_matcher.(Symbolic.symbolic ++ Symbolic.pm_simplify_valid)

let select_axes axes xs = List.filteri (fun i _ -> List.mem i axes) xs

let all_same = function
  | [] -> true
  | x :: rest -> List.for_all (fun y -> y == x) rest

let rec zip_shortest xs ys =
  match xs, ys with
  | x :: xs, y :: ys -> (x, y) :: zip_shortest xs ys
  | _ -> []

let is_range u = U.op u = Ops.Range
let is_movement_op u = Ops.Group.is_movement (U.op u)

(* True iff [u]'s op is an elementwise ALU/cast op or a reduce axis. *)
let is_elementwise_or_reduce u =
  let o = U.op u in
  Ops.Group.is_elementwise o || o = Ops.Reduce

let is_invalid_const u =
  U.op u = Ops.Const
  && (match U.arg u with
      | U.Arg.Value c -> Const.view c = Const.Invalid
      | _ -> false)

(* Does [n] or its backward slice contain a non-injective movement op? *)
let has_non_injective_view n =
  List.exists (fun x ->
    match U.op x with
    | Ops.Shrink | Ops.Permute | Ops.Flip | Ops.Pad -> true
    | _ -> false)
    (n :: U.backward_slice n)

(* A possibly-gated range is encoded as [Where(valid, idx, invalid)];
   these accessors strip the wrapper when present. *)

let rec get_idx r =
  match U.op r with
  | Ops.Stack -> U.stack (List.map get_idx (U.children r))
  | Ops.Where ->
      let s = U.src r in
      if Array.length s = 3 && is_invalid_const s.(2) then s.(1) else r
  | _ -> r

let rec get_valid r =
  match U.op r with
  | Ops.Stack -> U.stack (List.map get_valid (U.children r))
  | Ops.Where ->
      let s = U.src r in
      if Array.length s = 3 && is_invalid_const s.(2) then s.(0) else btrue
  | Ops.Const when is_invalid_const r -> bfalse
  | _ -> btrue

(* Boolean fold: MUL for conjunction, ADD for disjunction. *)
let prod_valid = function
  | [] -> btrue
  | x :: rest -> List.fold_left ( *! ) x rest

let sum_valid = function
  | [] -> bfalse
  | x :: rest -> List.fold_left ( +! ) x rest

(* Simplify [expr] under the installed symbolic rules. *)
let simplify_expr = U.simplify

let is_zero e =
  match U.const_int_value (simplify_expr e) with
  | Some 0 -> true
  | _ -> U.vmin e = 0 && U.vmax e = 0

let is_one e =
  match U.const_int_value (simplify_expr e) with
  | Some 1 -> true
  | _ -> U.vmin e = 1 && U.vmax e = 1

let same_expr a b =
  U.equal a b || is_zero (simplify_expr (a -! b))

let shape_expr_of ?shape_exprs ~shapes u =
  let fallback () =
    match shapes u with
    | Some sh -> Some (List.map idx sh)
    | None ->
        try Some (U.shape u) with Invalid_argument _ -> None
  in
  match shape_exprs with
  | Some shape_exprs ->
      (match shape_exprs u with Some _ as sh -> sh | None -> fallback ())
  | None -> fallback ()

(* Indexing context *)

type realize_state = Marked | Realized of int list

type indexing_context = {
  realize_map : (int, realize_state) Hashtbl.t;
  range_map : (int, U.t list * U.t list) Hashtbl.t;
  nodes : (int, U.t) Hashtbl.t;
  mutable range_idx : int;
}

let create_context () = {
  realize_map = Hashtbl.create 256;
  range_map = Hashtbl.create 256;
  nodes = Hashtbl.create 256;
  range_idx = 0;
}

let realize_get ctx n = Hashtbl.find_opt ctx.realize_map (U.tag n)
let realize_set ctx n v = Hashtbl.replace ctx.realize_map (U.tag n) v
let realize_del ctx n = Hashtbl.remove ctx.realize_map (U.tag n)
let realize_mem ctx n = Hashtbl.mem ctx.realize_map (U.tag n)

let range_get ctx n = Hashtbl.find_opt ctx.range_map (U.tag n)
let range_set ctx n v = Hashtbl.replace ctx.range_map (U.tag n) v

(* Size-1 dimensions collapse to constant 0 — mirrors tinygrad's
   [new_range] when [size == 1]. *)
let new_range_expr ctx size ?(kind = Axis_type.Loop) () =
  if U.op size = Ops.Range then size
  else if U.const_int_value (simplify_expr size) = Some 1
          || (U.vmin size = 1 && U.vmax size = 1)
  then idx 0
  else
    let axis = ctx.range_idx in
    ctx.range_idx <- ctx.range_idx + 1;
    U.range ~size ~axis ~kind ()

let new_range ctx size ?(kind = Axis_type.Loop) () =
  new_range_expr ctx (idx size) ~kind ()

(* Phase 1: realize map *)

let mark_non_contiguous ctx s =
  if not (always_contiguous (U.op (U.base s))) then realize_set ctx s Marked

let generate_realize_map ctx root =
  List.iter (fun n ->
    (match U.op n with
     | Ops.Copy | Ops.Contiguous | Ops.Store -> realize_set ctx n Marked
     | _ -> ());
    (match U.op n with
     | Ops.Copy | Ops.Mselect ->
         let s = U.src n in
         if Array.length s >= 1 then mark_non_contiguous ctx s.(0)
     | Ops.Mstack ->
         Array.iter (mark_non_contiguous ctx) (U.src n)
     | _ -> ());
    (* Conditionally unrealize or force-realize the value in STORE(dst, value).
       Matches [realize_store_after_src] in indexing.py. *)
    (match U.op n with
     | Ops.Store ->
         let s = U.src n in
         if Array.length s >= 2 then begin
           let dest = s.(0) and src = s.(1) in
           (match U.op src with
            | Ops.Copy | Ops.Slice
              when realize_mem ctx src && not (has_non_injective_view dest) ->
                realize_del ctx src
            | _ -> ());
           if List.exists (fun x -> x == dest)
                (src :: U.backward_slice src) then
             realize_set ctx src Marked
         end
     | _ -> ()))
    (U.toposort root)

(* Phase 2: range propagation *)

(* Reshape.

   Linearise the output-axis ranges into a scalar and decompose it along
   the input shape via mod/div. The placeholder substitution trick keeps
   [simplify_expr] from confusing range identities with the arithmetic it
   needs to reduce. *)

let apply_reshape in_shape out_shape rngs =
  let rng_sink = simplify_expr (U.sink rngs) in
  let rngs = U.children rng_sink in
  let all_ranges = U.ranges rng_sink in
  let placeholders = List.mapi (fun i r ->
    let size = match U.as_range r with Some v -> v.size | None -> idx 1 in
    (r, U.range ~size ~axis:i ~kind:Axis_type.Placeholder ())) all_ranges in
  let back = List.map (fun (k, v) -> (v, k)) placeholders in
  let rngs = List.map (U.substitute placeholders) rngs in
  let _, terms =
    List.fold_left
      (fun (stride, ts) (s, r) ->
         let t = if is_one stride then r else r *! stride in
         (simplify_expr (stride *! s), t :: ts))
      (idx 1, []) (List.rev (zip_shortest out_shape rngs))
  in
  let combined = List.fold_left ( +! ) (idx 0) (List.rev terms) in
  let acc = ref combined in
  let axes = List.rev_map (fun s ->
    let r = U.alu_binary ~op:Ops.Floormod ~lhs:!acc ~rhs:s in
    acc := U.alu_binary ~op:Ops.Floordiv ~lhs:!acc ~rhs:s;
    r) (List.rev in_shape) in
  let sink =
    U.sink axes
    |> U.graph_rewrite ~name:"reshape"
         (Upat.Pattern_matcher.rewrite reshape_pm)
  in
  U.children (U.substitute back sink)

let argsort order =
  let indexed = List.mapi (fun i v -> (v, i)) order in
  List.map snd (List.sort (fun (a, _) (b, _) -> compare a b) indexed)

(* [r >= k] expressed as [not (r < k)]. *)
let ge r k =
  let lt = U.alu_binary ~op:Ops.Cmplt ~lhs:r ~rhs:k in
  U.alu_binary ~op:Ops.Cmpne ~lhs:lt ~rhs:btrue

let apply_movement_op ?shape_exprs ~shapes n rngs =
  let src = (U.src n).(0) in
  match U.op n with
  | Ops.Shrink ->
      (match U.marg n with
	       | U.Marg_bounds pairs ->
	           zip_shortest rngs pairs
	           |> List.map (fun (r, (offset, _)) ->
	             if is_zero offset then r else r +! offset)
	       | _ -> rngs)
  | Ops.Permute ->
      (match U.marg n with
       | U.Marg_permute order -> List.map (fun p -> List.nth rngs p) (argsort order)
       | _ -> rngs)
  | Ops.Flip ->
      (match U.marg n, shape_expr_of ?shape_exprs ~shapes src with
	       | U.Marg_flip dims, Some in_shape ->
	           zip_shortest rngs (zip_shortest dims in_shape)
	           |> List.map (fun (r, (f, sh)) ->
	             if not f then r else (sh -! idx 1) -! r)
	       | _ -> rngs)
  | Ops.Expand ->
      (match shape_expr_of ?shape_exprs ~shapes src,
             shape_expr_of ?shape_exprs ~shapes n with
	       | Some in_shape, Some out_shape ->
	           zip_shortest rngs (zip_shortest in_shape out_shape)
	           |> List.map (fun (r, (in_s, out_s)) ->
	             if same_expr in_s out_s then r else idx 0)
	       | _ -> rngs)
  | Ops.Pad ->
      (match shape_expr_of ?shape_exprs ~shapes src, U.marg n with
	       | Some in_shape, U.Marg_bounds pairs ->
	           zip_shortest (zip_shortest rngs in_shape) pairs
	           |> List.map (fun ((r, sh), (offset, size)) ->
	             if same_expr size sh && is_zero offset then r
	             else
	               let upper =
                 U.alu_binary ~op:Ops.Cmplt ~lhs:r ~rhs:(offset +! sh) in
	               let valid =
	                  U.alu_binary ~op:Ops.And ~lhs:(ge r offset) ~rhs:upper
	                  |> U.graph_rewrite ~name:"pad" (Upat.Pattern_matcher.rewrite pad_pm) in
	               U.alu_ternary ~op:Ops.Where ~a:valid ~b:(r -! offset)
	                 ~c:(U.invalid ()))
	       | _ -> rngs)
  | Ops.Reshape ->
      (match shape_expr_of ?shape_exprs ~shapes src,
             shape_expr_of ?shape_exprs ~shapes n with
       | Some in_shape, Some out_shape -> apply_reshape in_shape out_shape rngs
       | _ -> rngs)
  | _ -> assert false

(* Build the direct-consumer map for [root] from its toposort. *)
let consumer_map root =
  let tbl : (int, U.t list) Hashtbl.t = Hashtbl.create 256 in
  let topo = U.toposort root in
  List.iter (fun u -> Hashtbl.replace tbl (U.tag u) []) topo;
  List.iter (fun u ->
    Array.iter (fun s ->
      match Hashtbl.find_opt tbl (U.tag s) with
      | Some prev -> Hashtbl.replace tbl (U.tag s) (u :: prev)
      | None -> ()) (U.src u)) topo;
  (fun u -> Option.value ~default:[] (Hashtbl.find_opt tbl (U.tag u))),
  topo

(* Transpose [[a0;a1;...]; [b0;b1;...]; ...] to per-index lists. *)
let transpose = function
  | [] -> []
  | first :: _ as lists ->
      List.mapi (fun i _ -> List.map (fun l -> List.nth l i) lists) first

(* Used only on nodes that come out of [U.ranges], always Range. *)
let range_axis r = match U.as_range r with
  | Some v -> v.axis | None -> assert false

let pcontig_var = Helpers.Context_var.int ~key:"PCONTIG" ~default:0

(* After choosing out_rngs, force additional axes to be realized when a
   reduce closes ranges earlier than the surrounding elementwise would. *)
let check_ending_ranges ctx ~pcontig ~ending_get ~ending_set ~out_shape x out_rngs =
  if ending_get x = [] then out_rngs
  else begin
    let existing = match realize_get ctx x with
      | Some (Realized a) -> a | _ -> []
    in
    let axes = ref existing in
    List.iteri (fun i r ->
      if not (List.mem i !axes) then
        if pcontig <= 1
           || List.exists (fun rr ->
                List.exists (fun e -> range_axis rr > range_axis e)
                  (ending_get x))
                (U.ranges r)
        then axes := !axes @ [i]) out_rngs;
    ending_set x [];
    if !axes = [] then out_rngs
    else begin
      realize_set ctx x (Realized !axes);
      List.mapi (fun i r ->
        if List.mem i !axes then
          let size =
            match List.nth_opt out_shape i with
            | Some size -> size
            | None ->
                let extra =
                  match U.as_reduce x with
                  | Some { src; axes; _ } ->
                      Printf.sprintf " axes=%d src=%s" (List.length axes)
                        (Ops.name (U.op src))
                  | None -> ""
                in
                failwith
                  (Printf.sprintf
                     "check_ending_ranges: %s out_shape=%d out_rngs=%d axis=%d%s"
                     (Ops.name (U.op x)) (List.length out_shape)
                     (List.length out_rngs) i extra)
          in
          new_range_expr ctx size ()
        else r) out_rngs
  end
  end

(* Skip nodes that don't take part in range propagation. *)
let skip_for_rangeify x =
  match U.op x with
  | Ops.Store | Ops.End -> false
  | Ops.Call | Ops.Function | Ops.Linear | Ops.Mselect | Ops.Mstack -> true
  | Ops.After -> true
  | _ ->
      match U.dtype x with
      | Dtype.Val v -> Dtype.Val.scalar v = Dtype.Weakint
      | _ -> false

(* Merge consumer ranges for nodes with multiple consumers agreeing on
   rank. Non-trivially new axes get fresh ranges and are recorded in the
   realize map. *)
let merge_consumer_rngs ctx ~pcontig ~out_shape x consumer_rngs =
  let per_axis = transpose consumer_rngs in
  let per_axis =
    List.filteri (fun i _ -> i < List.length out_shape) per_axis in
  let pairs = List.map (fun axis_rngs ->
    List.map get_idx axis_rngs, List.map get_valid axis_rngs) per_axis in
  let all_all_same = List.for_all (fun (lr, _) -> all_same lr) pairs in
  let out = ref [] and realize_axes = ref [] in
  List.iteri (fun i (local_rngs, valids) ->
    if all_all_same || (pcontig > 0 && all_same local_rngs) then
      let merged = simplify_expr
        (U.alu_ternary ~op:Ops.Where ~a:(sum_valid valids)
           ~b:(List.hd local_rngs) ~c:(U.invalid ())) in
      out := merged :: !out
    else begin
      out := new_range_expr ctx (List.nth out_shape i) () :: !out;
      realize_axes := i :: !realize_axes
    end) pairs;
  let realize_axes = List.rev !realize_axes in
  if realize_axes <> [] then realize_set ctx x (Realized realize_axes);
  List.rev !out

let choose_consumer_rngs ctx ~pcontig ~out_shape x consumer_rngs =
  match consumer_rngs with
  | [] -> None
  | [rs] -> Some rs
  | _ ->
      let n = List.length (List.hd consumer_rngs) in
      if not (List.for_all (fun l -> List.length l = n) consumer_rngs) then
        let n_out = List.length out_shape in
        let out = List.map (fun s -> new_range_expr ctx s ()) out_shape in
        realize_set ctx x (Realized (List.init n_out Fun.id));
        Some out
      else Some (merge_consumer_rngs ctx ~pcontig ~out_shape x consumer_rngs)

let run_rangeify ?shape_exprs root ~shapes =
  let ctx = create_context () in
  let shape_exprs =
    match shape_exprs with
    | Some shape_exprs -> shape_exprs
    | None -> fun x -> Option.map (List.map idx) (shapes x)
  in
  generate_realize_map ctx root;
  let consumers, topo = consumer_map root in
  let ending : (int, U.t list) Hashtbl.t = Hashtbl.create 256 in
  let ending_get x =
    Option.value ~default:[] (Hashtbl.find_opt ending (U.tag x)) in
  let ending_set x v = Hashtbl.replace ending (U.tag x) v in
  let pcontig = Helpers.Context_var.get pcontig_var in

  let step x =
    Hashtbl.replace ctx.nodes (U.tag x) x;
    if skip_for_rangeify x then () else begin
      ending_set x (List.concat_map ending_get (consumers x));
      let out_shape = Option.value ~default:[] (shape_exprs x) in
      let consumer_rngs =
        List.filter_map
          (fun c -> match range_get ctx c with
             | Some (in_rngs, _) -> Some in_rngs
             | None -> None)
          (consumers x)
      in
      let chosen =
        if realize_mem ctx x then begin
          let out = List.map (fun s -> new_range_expr ctx s ()) out_shape in
          ending_set x [];
          realize_set ctx x
            (Realized (List.init (List.length out_shape) Fun.id));
          Some out
        end
        else choose_consumer_rngs ctx ~pcontig ~out_shape x consumer_rngs
      in
      match chosen with
      | None -> ()
      | Some out_rngs ->
          let out_rngs =
            if is_elementwise_or_reduce x then
              check_ending_ranges ctx ~pcontig ~ending_get ~ending_set
                ~out_shape x out_rngs
            else out_rngs
          in
          let rngs =
            if is_movement_op x then
              apply_movement_op ~shape_exprs ~shapes x out_rngs
            else out_rngs
          in
          let rngs =
            match U.as_reduce x with
            | Some { src; axes; _ } when axes <> [] ->
                (match shape_exprs src with
                 | Some in_shape_expr ->
                     let rec build i outs = function
                     | [] -> []
                     | size :: sizes ->
                         if List.mem i axes then
                           new_range_expr ctx size ~kind:Axis_type.Reduce ()
                           :: build (i + 1) outs sizes
                         else
                           match outs with
                           | r :: rs -> r :: build (i + 1) rs sizes
                           | [] -> []
                     in
                     build 0 rngs in_shape_expr
                 | None -> rngs)
            | _ -> rngs
          in
          (if U.op x = Ops.Expand
              && List.for_all (fun s -> U.op s <> Ops.Range) out_shape
           then
             let diff =
               List.filter_map
                 (fun (ri, ro) -> if ri != ro then Some ro else None)
                 (zip_shortest rngs out_rngs)
             in
             if diff <> [] then
               ending_set x (ending_get x @ U.ranges (U.sink diff)));
          range_set ctx x (rngs, out_rngs)
    end
  in
  List.iter step (List.rev topo);
  ctx

(* Phase 3: apply rangeify *)

let direct_buffer_src u =
  match U.op u with
  | Ops.Param | Ops.Buffer | Ops.Stage | Ops.Slice | Ops.Mstack | Ops.Mselect
  | Ops.After -> true
  | _ -> false

(* Movement ops disappear — their effect is captured in the range_map. *)
let remove_movement_op ctx x =
  if is_movement_op x then
    let s = (U.src x).(0) in
    if Option.is_some (range_get ctx x) || U.ranges x <> [] then Some s
    else if U.op s = Ops.Index then Some s
    else None
  else None

(* Direct buffer source: insert an INDEX into [s] using [x]'s input ranges. *)
let index_direct_buffer in_rngs s =
  (* A non-pointer tensor buffer needs [as_ptr:true]: [U.index] rejects
     non-pointer [ptr] otherwise. A later kernel-split pass promotes such
     tensor-level buffers to real pointers. *)
  let as_ptr =
    match U.dtype s with
    | Dtype.Ptr _ -> false
    | Dtype.Val _ -> true
  in
  U.index ~ptr:s ~idxs:in_rngs ~as_ptr ()

(* Realized source: wrap in STAGE (or END for STORE) and INDEX. *)
let wrap_realized_src ctx ~parent_is_copy ~parent_rngs ~realized_axes s =
  let out_rngs =
    match range_get ctx s with
    | Some (_, o) -> o
    | None ->
        let shape =
          try U.max_shape s with Invalid_argument _ -> []
        in
        let out =
          List.map (fun size -> new_range_expr ctx (idx size) ()) shape
        in
        if out <> [] then range_set ctx s (out, out);
        out
  in
  let closed = select_axes realized_axes out_rngs in
  match U.op s with
  | Ops.Store ->
      let ranges = List.filter is_range closed in
      realize_del ctx s;
      U.end_ ~value:s ~ranges
  | _ ->
      let removable =
        not parent_is_copy && not (always_contiguous (U.op s)) in
      let is_local =
        List.length out_rngs <> List.length realized_axes in
      let addrspace = if is_local then Dtype.Local else Dtype.Global in
      let opts : U.stage_opts =
        { device = U.device_of s; addrspace; removable } in
      let buf = U.stage ~src:s ~ranges:closed ~opts in
      match parent_rngs with
      | Some (in_rngs, _) ->
          let idxs = select_axes realized_axes in_rngs in
          U.index ~ptr:buf ~idxs ~as_ptr:true ()
      | None -> buf

(* For each child of [x], insert STAGE/INDEX/END as needed. *)
let create_stage_and_index ctx ~shapes ~shape_exprs x =
  match U.op x with
  | Ops.Stage | Ops.Index -> None
  | _ ->
      let parent_is_copy = U.op x = Ops.Copy in
      let parent_rngs = range_get ctx x in
      let changed = ref false in
      let rewrite_child i s =
        if is_movement_op x && i > 0 then s
        else if direct_buffer_src s then
          match parent_rngs with
          | Some (in_rngs, _) ->
              changed := true;
              index_direct_buffer in_rngs s
          | _ -> s
        else match realize_get ctx s with
        | Some (Realized realized_axes) ->
            changed := true;
            wrap_realized_src ctx ~parent_is_copy ~parent_rngs
              ~realized_axes s
        | _ -> s
      in
      let new_children =
        List.mapi rewrite_child (Array.to_list (U.src x))
      in
      if !changed then (
        let x' = U.replace x ~src:(Array.of_list new_children) () in
        (match realize_get ctx x with
         | Some v -> realize_set ctx x' v | None -> ());
        (match range_get ctx x with
         | Some v -> range_set ctx x' v | None -> ());
        Some x')
      else None

let with_indexed_children ctx ~shapes ~shape_exprs x =
  Option.value (create_stage_and_index ctx ~shapes ~shape_exprs x) ~default:x

(* REDUCE(op, axes) -> REDUCE(op, []) with explicit range children. *)
let convert_reduce ctx ~shapes ~shape_exprs x =
  match U.as_reduce x, range_get ctx x with
  | Some { src = _; op; axes; _ }, Some (in_rngs, _ as entry)
    when axes <> [] ->
      let ranges =
        List.mapi
          (fun i r -> if List.mem i axes then Some r else None)
          in_rngs
        |> List.filter_map Fun.id
      in
      range_set ctx x entry;
      let bx = with_indexed_children ctx ~shapes ~shape_exprs x in
      let src = (U.src bx).(0) in
      Some (U.reduce ~src ~op ~ranges ~dtype:(Dtype.val_of (U.dtype x)))
  | _ -> None

(* PAD -> WHERE(valid, src, 0). *)
let convert_pad_to_where ctx ~shapes ~shape_exprs x =
  match U.op x, range_get ctx x with
  | Ops.Pad, Some ((in_rngs, _) as entry) ->
      let valid = prod_valid (List.map get_valid in_rngs) in
      let bx = with_indexed_children ctx ~shapes ~shape_exprs x in
      let src = (U.src bx).(0) in
      let ret =
        U.alu_ternary ~op:Ops.Where ~a:valid ~b:src ~c:(U.zero_like x)
      in
      range_set ctx ret entry;
      Some ret
  | _ -> None

let fix_deviceless_stage ~device n =
  match U.as_stage n with
  | Some { opts = { device = None; addrspace = Dtype.Global; _ } as opts; _ } ->
      Some
        (U.replace n
           ~arg:(U.Arg.Stage_info { opts with device = Some device })
           ())
  | _ -> None

let fix_deviceless_stages device root =
  match device with
  | None -> root
  | Some device ->
      U.graph_rewrite ~name:"fix deviceless stages"
        (fix_deviceless_stage ~device) root

let apply_rangeify_pass ?shape_exprs ctx ~shapes root =
  let shape_exprs =
    match shape_exprs with
    | Some shape_exprs -> shape_exprs
    | None -> fun x -> Option.map (List.map idx) (shapes x)
  in
  let on_rebuild ~old_n ~new_n =
    if U.tag old_n <> U.tag new_n then begin
      (match realize_get ctx old_n with
       | Some v when not (realize_mem ctx new_n) -> realize_set ctx new_n v
       | _ -> ());
      (match range_get ctx old_n with
       | Some v when Option.is_none (range_get ctx new_n) -> range_set ctx new_n v
       | _ -> ())
    end
  in
  let device = U.device_of root in
  let root =
    U.graph_rewrite ~name:"apply rangeify" ~bottom_up:true ~on_rebuild
      (fun n ->
        match U.op n with
        | Ops.Reduce -> convert_reduce ctx ~shapes ~shape_exprs n
        | Ops.Pad -> convert_pad_to_where ctx ~shapes ~shape_exprs n
        | _ ->
            (match create_stage_and_index ctx ~shapes ~shape_exprs n with
             | Some _ as r -> r
             | None -> remove_movement_op ctx n))
      root
  in
  fix_deviceless_stages device root
