(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Port of tinygrad/schedule/rangeify.py to the tolk_uop IR.
   Transforms a tensor-level SINK into a kernel graph with CALL nodes
   wrapping kernel ASTs. *)

open Tolk_uop
module U = Uop

let symbolic =
  Upat.Pattern_matcher.(Symbolic.symbolic ++ Symbolic.index_pushing)

(* Context vars *)

let v_openpilot = Helpers.Context_var.int ~key:"OPENPILOT_HACKS" ~default:0
let v_float16 = Helpers.Context_var.int ~key:"FLOAT16" ~default:0
let v_split_red = Helpers.Context_var.int ~key:"SPLIT_REDUCEOP" ~default:1
let v_split_thr =
  Helpers.Context_var.int ~key:"REDUCEOP_SPLIT_THRESHOLD" ~default:32768
let v_split_sz = Helpers.Context_var.int ~key:"REDUCEOP_SPLIT_SIZE" ~default:22
let v_max_bufs = Helpers.Context_var.int ~key:"MAX_KERNEL_BUFFERS" ~default:0
let v_pcontig = Helpers.Context_var.int ~key:"PCONTIG" ~default:0
let getv = Helpers.Context_var.get

(* Helpers *)

let prod l = List.fold_left ( * ) 1 l
let int_ n = U.const (Const.int Dtype.Val.int32 n)

let src0 u = (U.src u).(0)
let src_list u = Array.to_list (U.src u)
let src_tail u =
  let s = U.src u in
  Array.to_list (Array.sub s 1 (Array.length s - 1))

let shape_node = function [ d ] -> int_ d | ds -> U.stack (List.map int_ ds)

let ptr_size_shape n =
  match U.dtype n with
  | Dtype.Ptr p ->
      let size = Dtype.Ptr.size p in
      if size >= 0 then Some [ size ] else None
  | Dtype.Val _ -> None

let movement_src u =
  match U.op u with
  | Ops.Reshape | Ops.Expand | Ops.Pad | Ops.Shrink | Ops.Permute | Ops.Flip ->
      Some (src0 u)
  | _ -> None

let is_movement u = Option.is_some (movement_src u)

let rec shape_of_node n =
  match U.op n with
  | Ops.Const ->
      (match U.arg n with
       | U.Arg.Value v ->
           (match Const.view v with
            | Const.Int i -> Some [ Int64.to_int i ]
            | _ -> None)
       | _ -> None)
  | Ops.Stack ->
      let rec collect = function
        | [] -> Some []
        | x :: xs ->
            (match shape_of_node x with
             | Some [ v ] -> Option.map (fun r -> v :: r) (collect xs)
             | _ -> None)
      in
      collect (src_list n)
  | _ -> None

let shape_expr_of_node n =
  match U.op n with
  | Ops.Const ->
      (match U.const_int_value n with Some _ -> Some [ n ] | None -> None)
  | Ops.Stack -> Some (U.children n)
  (* A symbolic integer expression is a rank-1 shape argument. *)
  | _ -> if Dtype.is_int (U.dtype n) then Some [ n ] else None

let map_order shape order =
  if List.length shape < List.length order then None
  else
    try Some (List.map (List.nth shape) order) with Failure _ -> None

let combine_same_rank a b =
  if List.length a = List.length b then Some (List.combine a b) else None

let broadcast_shape shapes =
  let shapes = List.filter (fun s -> s <> []) shapes in
  match shapes with
  | [] -> Some []
  | first :: rest ->
      let rank =
        List.fold_left
          (fun acc s -> max acc (List.length s))
          (List.length first) rest
      in
      let align s = List.init (rank - List.length s) (fun _ -> 1) @ s in
      let shapes = List.map align shapes in
      Some
        (List.init rank (fun i ->
             List.fold_left
               (fun acc s ->
                 let d = List.nth s i in
                 if d = 0 then 0
                 else if acc = 1 then d
                 else if d = 1 || d = acc then acc
                 else max acc d)
               1 shapes))

let is_one_expr u = match U.const_int_value u with Some 1 -> true | _ -> false

let broadcast_shape_expr shapes =
  let shapes = List.filter (fun s -> s <> []) shapes in
  match shapes with
  | [] -> Some []
  | first :: rest ->
      let rank =
        List.fold_left
          (fun acc s -> max acc (List.length s))
          (List.length first) rest
      in
      let align s = List.init (rank - List.length s) (fun _ -> int_ 1) @ s in
      let shapes = List.map align shapes in
      Some
        (List.init rank (fun i ->
             List.fold_left
               (fun acc s ->
                 let d = List.nth s i in
                 if is_one_expr acc then d
                 else if is_one_expr d || U.equal d acc then acc
                 else d)
               (int_ 1) shapes))

let rec shape_of n =
  match U.op n with
  | Ops.Param | Ops.Buffer ->
      (match U.children n with
       | shape :: _ ->
           (match shape_of_node shape with
            | Some _ as s -> s
            | None -> ptr_size_shape n)
       | [] -> ptr_size_shape n)
  | Ops.Reshape | Ops.Expand -> shape_of_node (U.src n).(1)
  | Ops.Pad | Ops.Shrink ->
      (match
         shape_of (src0 n), shape_of_node (U.src n).(1),
         shape_of_node (U.src n).(2)
       with
       | Some _, Some _, Some size -> Some size
       | _ -> None)
  | Ops.Permute ->
      let order = match U.arg n with U.Arg.Ints i -> i | _ -> [] in
      Option.bind (shape_of (src0 n)) (fun s -> map_order s order)
  | Ops.Flip -> shape_of (src0 n)
  | Ops.Reduce ->
      (match U.as_reduce n, shape_of (src0 n) with
       | Some { axes = []; _ }, _ ->
           let count = Dtype.count (U.dtype n) in
           Some (if count > 1 then [ count ] else [])
       | Some { axes; _ }, Some s ->
           Some (List.filteri (fun i _ -> not (List.mem i axes)) s)
       | _ -> None)
  | Ops.Contiguous | Ops.Contiguous_backward | Ops.Detach | Ops.Copy
  | Ops.After | Ops.Noop | Ops.Bitcast | Ops.Cast | Ops.Store | Ops.End ->
      if Array.length (U.src n) = 0 then None else shape_of (src0 n)
  | op when Ops.Group.is_elementwise op ->
      let child_shapes = List.filter_map shape_of (U.children n) in
      broadcast_shape child_shapes
  | _ -> None

let rec shape_expr_of n =
  match U.op n with
  | Ops.Param | Ops.Buffer ->
      (match U.children n with
       | shape :: _ ->
           (match shape_expr_of_node shape with
            | Some _ as s -> s
            | None -> Option.map (List.map int_) (ptr_size_shape n))
       | [] -> Option.map (List.map int_) (ptr_size_shape n))
  | Ops.Reshape | Ops.Expand -> shape_expr_of_node (U.src n).(1)
  | Ops.Pad | Ops.Shrink ->
      (match shape_expr_of (src0 n), shape_expr_of_node (U.src n).(2) with
       | Some _, (Some _ as size) -> size
       | _ -> None)
  | Ops.Permute ->
      let order = match U.arg n with U.Arg.Ints i -> i | _ -> [] in
      Option.bind (shape_expr_of (src0 n)) (fun s -> map_order s order)
  | Ops.Flip -> shape_expr_of (src0 n)
  | Ops.Reduce ->
      (match U.as_reduce n, shape_expr_of (src0 n) with
       | Some { axes = []; _ }, _ -> Option.map (List.map int_) (shape_of n)
       | Some { axes; _ }, Some s ->
           Some (List.filteri (fun i _ -> not (List.mem i axes)) s)
       | _ -> None)
  | Ops.Contiguous | Ops.Contiguous_backward | Ops.Detach | Ops.Copy
  | Ops.After | Ops.Noop | Ops.Bitcast | Ops.Cast | Ops.Store | Ops.End ->
      if Array.length (U.src n) = 0 then None else shape_expr_of (src0 n)
  | op when Ops.Group.is_elementwise op ->
      U.children n |> List.filter_map shape_expr_of |> broadcast_shape_expr
  | _ -> (
      match shape_of n with
      | Some sh -> Some (List.map int_ sh)
      | None -> (
          (* Symbolic dimensions (e.g. a SHRINK sized by a variable) are
             invisible to the concrete [shape_of]; fall back to the
             expression-level shape. *)
          try Some (U.shape n) with Invalid_argument _ -> None))

let argsort order =
  List.map snd (List.sort compare (List.mapi (fun i o -> (o, i)) order))

let device_max_bufs = function "METAL" -> 31 | "WEBGPU" -> 8 | _ -> 0

let is_ptr_dtype = function Dtype.Ptr _ -> true | _ -> false

let base n = U.base n

(* Movement-op rewrites *)

let pm_mop_through_index n =
  match U.as_index n with
  | Some { ptr; _ } when is_movement ptr ->
      let src = Option.get (movement_src ptr) in
      let idxs = src_tail n in
      let mop_shape u =
        match shape_of u with
        | Some _ as shape -> shape
        | None -> (
            try Some (List.map U.vmax (U.shape u))
            with Invalid_argument _ -> None)
      in
      (match mop_shape src, mop_shape ptr with
       | Some _, Some ps when List.length idxs = List.length ps ->
           let new_idxs =
             Indexing.apply_movement_op ~shapes:shape_of ptr idxs
           in
           Some (U.replace n ~src:(Array.of_list (src :: new_idxs)) ())
       | Some src_shape, Some ptr_shape when U.op ptr = Ops.Reshape ->
           let nidxs = List.length idxs in
           let ptr_suffix =
             List.filteri (fun i _ -> i >= nidxs) ptr_shape
           in
           let src_prefix = List.length src_shape - List.length ptr_suffix in
           if src_prefix < 0 then None
           else
             let src_suffix =
               List.filteri (fun i _ -> i >= src_prefix) src_shape
             in
             if src_suffix <> ptr_suffix then None
             else if src_prefix = 0 then
               if Dtype.equal (U.dtype src) (U.dtype n) then Some src
               else None
             else
               let src_prefix_shape =
                 List.filteri (fun i _ -> i < src_prefix) src_shape
               in
               let ptr_prefix_shape =
                 List.filteri (fun i _ -> i < nidxs) ptr_shape
               in
               let shapes u =
                 if u == src then Some src_prefix_shape
                 else if u == ptr then Some ptr_prefix_shape
                 else shape_of u
               in
               let new_idxs = Indexing.apply_movement_op ~shapes ptr idxs in
               let ret = U.replace n ~src:(Array.of_list (src :: new_idxs)) () in
               if shape_of ret = shape_of n then Some ret else None
       | _ -> None)
  | _ -> None

let pm_mop_past_after n =
  match U.op n with
  | Ops.After ->
      let r = src0 n in
      let op = U.op r in
      if not (Ops.Group.is_movement op || op = Ops.Index) then None
      else
        let src = Array.copy (U.src r) in
        src.(0) <- U.after ~src:(src0 r) ~deps:(src_tail n);
        Some (U.replace r ~src ())
  | _ -> None

let pm_mop_past_end n =
  match U.as_end n with
  | Some { value; ranges } when is_movement value ->
      Some (U.end_ ~value:(Option.get (movement_src value)) ~ranges)
  | _ -> None

let pm_index_concat n =
  match U.as_index n with
  | Some { ptr; _ }
    when U.op ptr = Ops.Index
      && is_ptr_dtype (U.dtype ptr) && not (is_ptr_dtype (U.dtype n)) ->
      Some
        (U.replace n ~src:(Array.append (U.src ptr) (Array.of_list (src_tail n))) ())
  | _ -> None

let movement_ops n =
  match
    U.first_match [ pm_mop_through_index; pm_mop_past_after; pm_mop_past_end ] n
  with
  | Some n' when not (U.equal n n') -> Some n'
  | Some _ | None -> None

let pm_early_rangeify n =
  match U.as_index n with
  | Some { ptr; _ } when
      (let op = U.op ptr in Ops.Group.is_elementwise op || op = Ops.Const) ->
      let tail = src_tail n in
      let push s = U.replace n ~src:(Array.of_list (s :: tail)) () in
      let new_children = List.map push (U.children ptr) in
      Some (U.replace ptr ~src:(Array.of_list new_children) ())
  | _ -> None

let shape_for_store u =
  try Some (U.shape u) with Invalid_argument _ -> None

let shape_numel dims =
  List.fold_left (fun acc d -> acc * U.vmax d) 1 dims

let same_shape a b =
  List.length a = List.length b && List.for_all2 U.equal a b

let pm_add_ranges_to_store ctr n =
  match U.as_store n with
  | Some { dst; value; _ } ->
      (match shape_for_store dst, shape_for_store value with
       | Some [], _ | _, Some [] -> None
       | Some d_sh, Some v_sh when shape_numel d_sh = 1 && shape_numel v_sh = 1
         ->
           None
       | Some d_sh, Some v_sh ->
           if not (same_shape d_sh v_sh) then
             invalid_arg "Rangeify.add_ranges_to_store: bad store shape";
           let idxs =
             List.map (fun size ->
                 let axis = !ctr in
                 incr ctr;
                 U.range ~size ~axis ~kind:Axis_type.Loop ())
               d_sh
           in
           let mk s = U.index ~ptr:s ~idxs ~as_ptr:true () in
           Some (U.end_
                   ~value:(U.store ~dst:(mk dst) ~value:(mk value) ())
                   ~ranges:idxs)
       | _ -> None)
  | _ -> None

let early_movement_pass sink =
  let ctr = ref 1000 in
  let rules =
    [
      pm_mop_through_index;
      pm_mop_past_after;
      pm_mop_past_end;
      pm_index_concat;
      pm_early_rangeify;
      pm_add_ranges_to_store ctr;
    ]
  in
  U.graph_rewrite ~bottom_up:true ~name:"early movement ops"
    (U.first_match rules)
    sink

let rewrite_movement_ops sink =
  U.graph_rewrite ~bottom_up:true ~name:"early movement ops"
    (U.first_match [ movement_ops; pm_index_concat ])
    sink

(* Fold moved AFTERs (openpilot hack) *)

let is_invalid u =
  U.op u = Ops.Const
  && (match U.arg u with
      | U.Arg.Value v -> Const.view v = Const.Invalid | _ -> false)

let found_after ctx ~after ~value =
  let x = ref value and a = ref after in
  if getv v_float16 <> 0 && U.op !x = Ops.Cast
     && Dtype.equal (U.dtype !x) Dtype.float16
  then begin
    a := U.cast ~src:!a ~dtype:Dtype.float32;
    x := src0 !x
  end;
  let continue_ = ref true in
  while !continue_ do
    match U.op !x with
    | Ops.Permute ->
        let order = match U.arg !x with U.Arg.Ints o -> o | _ -> [] in
        a := U.permute ~src:!a ~order:(argsort order);
        x := src0 !x
    | Ops.Reshape ->
        (match shape_of (src0 !x) with
         | Some s ->
             a := U.reshape ~src:!a ~shape:(shape_node s);
             x := src0 !x
         | None -> continue_ := false)
    | Ops.Where ->
        let s = U.src !x in
        if is_invalid s.(2) && U.op s.(1) = Ops.Pad then x := src0 s.(1);
        continue_ := false
    | _ -> continue_ := false
  done;
  U.Ref_tbl.replace ctx !x !a

let pm_fold_moved_after ctx n =
  match U.op n with
  | Ops.After ->
      let deps = src_tail n in
      (match List.find_opt (fun d -> U.op d = Ops.Store) deps with
       | Some s ->
           let value = (Option.get (U.as_store s)).value in
           (match U.op value with
            | Ops.Reshape | Ops.Expand | Ops.Pad | Ops.Shrink | Ops.Permute
            | Ops.Flip | Ops.Cast | Ops.Where ->
                found_after ctx ~after:n ~value; None
            | _ -> None)
       | None -> None)
  | op when Ops.Group.is_alu op || op = Ops.Cast || op = Ops.Bitcast ->
      let children = U.children n in
      let new_children =
        List.map (fun s ->
            Option.value (U.Ref_tbl.find_opt ctx s) ~default:s)
          children
      in
      if List.for_all2 ( == ) children new_children then None
      else Some (U.replace n ~src:(Array.of_list new_children) ())
  | _ -> None

(* Earliest rewrites *)

let fix_store_hazard ~target ~value =
  let target_has_shrink =
    List.exists (fun u -> U.op u = Ops.Shrink) (U.toposort target)
  in
  let unsafe op =
    op = Ops.Permute || op = Ops.Flip
    || (op = Ops.Shrink && target_has_shrink)
  in
  let b = base target in
  let slice =
    U.toposort ~gate:(fun s -> U.op s <> Ops.Contiguous) value
  in
  let reaches = U.Ref_tbl.create (List.length slice) in
  let found = ref false in
  List.iter (fun s ->
      if not !found then begin
        let r = s == b
          || List.exists (fun c ->
                 U.Ref_tbl.find_opt reaches c = Some true)
               (U.children s)
        in
        U.Ref_tbl.replace reaches s r;
        if r && unsafe (U.op s) then found := true
      end) slice;
  if !found then
    Some (U.store ~dst:target ~value:(U.contiguous ~src:value ()) ())
  else None

let resolve_function n =
  match U.as_call n with
  | Some { body; args; info } when U.op n = Ops.Function && not info.precompile ->
      let params =
        List.filter (fun u -> U.op u = Ops.Param) (U.toposort body)
      in
      let idx_of p =
        match U.as_param p with
        | Some { param = { slot; _ }; _ } -> slot
        | None -> -1
      in
      let params = List.sort (fun a b -> compare (idx_of a) (idx_of b)) params in
      let n_args = List.length args in
      let mappings =
        List.filter_map (fun p ->
            let i = idx_of p in
            if i >= 0 && i < n_args then Some (p, List.nth args i) else None)
          params
      in
      Some (U.substitute mappings body)
  | _ -> None

let detect_expanded src =
  let sh = Option.value (shape_of src) ~default:[] in
  let n = List.length sh in
  if n = 0 then []
  else
    let rngs =
      List.mapi (fun i s ->
          if s > 1 then U.range ~size:(int_ s) ~axis:i ~kind:Axis_type.Loop ()
          else int_ 0) sh
    in
    let rec push node rngs =
      match U.op node with
      | Ops.Reshape | Ops.Expand | Ops.Pad | Ops.Shrink | Ops.Permute | Ops.Flip ->
          push (src0 node) (Indexing.apply_movement_op
                              ~shapes:shape_of node rngs)
      | _ -> rngs
    in
    let final = push src rngs in
    let live =
      List.concat_map (fun r ->
          List.filter_map (fun x ->
              Option.map (fun (v : U.range_view) -> v.axis) (U.as_range x))
            (r :: U.backward_slice r))
        final
    in
    List.init n (fun i -> not (List.mem i live))

let pow2 n =
  if n < 0 then 1 else
    let rec loop acc i = if i = 0 then acc else loop (acc * 2) (i - 1) in
    loop 1 n

let range_down from_ until =
  let rec loop acc n = if n < until then List.rev acc else loop (n :: acc) (n - 1) in
  loop [] from_

let split_reduceop_rule n =
  match U.as_reduce n with
  | Some { src; op; axes; _ } when axes <> [] ->
      (match shape_of src, shape_of n with
       | Some in_shape, Some out_shape
         when prod out_shape <> 0
              && getv v_split_red <> 0
              && prod in_shape / max 1 (prod out_shape) >= getv v_split_thr ->
           let expanded = detect_expanded src in
           let cap = min 256 (pow2 (getv v_split_sz) / max 1 (prod out_shape)) in
           let candidates =
             List.concat_map
               (fun axis ->
                 if axis < 0 || axis >= List.length in_shape then []
                 else
                   let dim = List.nth in_shape axis in
                   range_down cap 8
                   |> List.filter_map (fun divisor ->
                       if dim mod divisor = 0
                          && not (List.nth expanded axis)
                       then Some (axis, divisor)
                       else None))
               axes
           in
           (match candidates with
            | [] -> None
            | (axis, divisor) :: _ ->
                let split_shape =
                  List.concat
                    [
                      List.filteri (fun i _ -> i < axis) in_shape;
                      [ divisor; List.nth in_shape axis / divisor ];
                      List.filteri (fun i _ -> i > axis) in_shape;
                    ]
                in
                let order =
                  List.filter (fun i -> i <> axis)
                    (List.init (List.length split_shape) Fun.id)
                  @ [ axis ]
                in
                let splitted =
                  U.reshape ~src ~shape:(shape_node split_shape)
                  |> fun u -> U.permute ~src:u ~order
                in
                let first =
                  U.contiguous
                    ~src:(U.reduce_axis ~src:splitted ~op ~axes)
                    ()
                in
                let second_axis = List.length out_shape in
                let second =
                  U.reduce_axis ~src:first ~op ~axes:[ second_axis ]
                in
                Some (U.reshape ~src:second ~shape:(shape_node out_shape)))
       | _ -> None)
  | _ -> None

let mop_cleanup n =
  match U.op n with
  | Ops.Reshape when U.op (src0 n) = Ops.Reshape ->
      Some (U.replace n ~src:[| src0 (src0 n); (U.src n).(1) |] ())
  | _ -> None

let identity_of op dtv = match op with
  | Ops.Add -> Const.zero dtv
  | Ops.Mul -> Const.one dtv
  | Ops.Max -> Const.min_value dtv
  | _ -> Const.zero dtv

let earliest_rewrites =
  U.first_match
    [ pm_index_concat; pm_early_rangeify; pm_mop_through_index;
      pm_mop_past_after; pm_mop_past_end; mop_cleanup; resolve_function;
      (fun n -> match U.as_allreduce n with
         | Some { src; device; op } ->
             (match shape_of n with
              | Some shape ->
                  Allreduce.create_allreduce_function src ~device ~op
                    ~dtype:(U.dtype n) ~shape ()
              | None -> None)
         | None -> None);
      split_reduceop_rule;
      (fun n -> match U.op n with
         | Ops.Detach | Ops.Contiguous_backward -> Some (src0 n)
         | _ -> None);
      (fun n -> match U.op n with
         | Ops.Copy when is_movement (src0 n) ->
             let s = src0 n in
             if shape_of (base s) <> shape_of s then
               let sr = Array.copy (U.src n) in
               sr.(0) <- U.contiguous ~src:s ();
               Some (U.replace n ~src:sr ())
             else None
         | _ -> None);
      (fun n -> match U.op n with
         | Ops.Copy ->
             let s = src0 n in
             (match U.device_of s, U.device_of n with
              | Some d1, Some d2 when d1 = d2 ->
                  Some (U.noop ~src:s ~dtype:(U.dtype s) ())
              | _ -> None)
         | _ -> None);
      (fun n -> match U.op n with
         | Ops.Sink ->
             let children = U.children n in
             let new_children =
               List.map
                 (fun child ->
                    match U.op child, U.src child with
                    | Ops.After, srcs when Array.length srcs > 1 -> child
                    | _ -> base child)
                 children
             in
             if List.for_all2 ( == ) children new_children then None
             else Some (U.replace n ~src:(Array.of_list new_children) ())
         | _ -> None);
      (fun n -> match U.as_store n with
         | Some { dst = target; value; _ } -> fix_store_hazard ~target ~value
         | _ -> None);
      (fun n -> match U.as_store n with
         | Some { dst; value; _ } when U.op dst = Ops.Bitcast ->
             let inner = src0 dst in
             Some (U.store ~dst:inner
                     ~value:(U.bitcast ~src:value
                               ~dtype:(Dtype.Val (Dtype.val_of (U.dtype inner))))
                     ())
         | _ -> None);
      (fun n -> match U.as_reduce n with
         | Some { src; op; _ } ->
             (match shape_of src, shape_of n with
              | Some s, Some t when List.mem 0 s && not (List.mem 0 t) ->
                  Some (U.const (identity_of op (Dtype.val_of (U.dtype n))))
              | _ -> None)
         | None -> None);
      (fun n ->
        if U.op n = Ops.Sink then None
        else match shape_of n with
          | Some s when List.mem 0 s ->
              Some (U.const (Const.zero (Dtype.val_of (U.dtype n))))
          | _ -> None);
    ]

(* Post-rangeify *)

let is_always_run op = op = Ops.Contiguous || op = Ops.Copy || op = Ops.Noop

let remove_noop_stage n =
  match U.as_stage n with
  | Some { src; ranges; _ } ->
      (match U.as_index src with
       | Some _ when Array.length (U.src src) - 1 = List.length ranges ->
           let idxs = src_tail src in
           if not (List.equal ( == ) idxs ranges) then None
           else
             let ptr = src0 src in
            if U.op ptr = Ops.Slice then None
             else
               (match shape_of n with
                | Some sh when sh <> [] ->
                    let zeros = shape_node (List.map (fun _ -> 0) sh) in
                    Some (U.shrink ~src:ptr ~offset:zeros ~size:(shape_node sh))
                | _ -> Some ptr)
       | _ -> None)
  | _ -> None

let cleanup_dead_axes n =
  match U.as_stage n with
  | Some { src; ranges; opts; _ } when not (is_always_run (U.op src))
                                       && U.op src <> Ops.After ->
      let sh = Option.value (shape_of n) ~default:[] in
      if List.length sh <> List.length ranges then None
      else if List.exists (fun r -> match U.as_range r with
          | Some v -> U.op v.size <> Ops.Const | None -> false) ranges
      then None
      else
        let src_ranges = U.ranges src in
        let hit = ref false and new_ranges = ref [] and new_sh = ref [] in
        List.iter2 (fun s rng ->
            let dead = match U.op rng with
              | Ops.Const -> true
              | Ops.Range -> not (List.exists (U.equal rng) src_ranges)
              | _ -> false
            in
            if dead then begin new_sh := 1 :: !new_sh; hit := true end
            else begin
              new_sh := s :: !new_sh;
              new_ranges := rng :: !new_ranges
            end) sh ranges;
        if not !hit then None
        else
          let new_ranges = List.rev !new_ranges in
          let new_sh = List.rev !new_sh in
          let b = U.stage ~src ~ranges:new_ranges ~opts in
          let r = U.reshape ~src:b ~shape:(shape_node new_sh) in
          Some (U.expand ~src:r ~shape:(shape_node sh))
  | _ -> None

let is_reduce_range r =
  match U.as_range r with
  | Some v -> v.kind = Axis_type.Reduce | None -> false

let range_size_expr r =
  match U.as_range r with Some v -> v.size | None -> int_ 1

let prod_expr = function
  | [] -> int_ 1
  | x :: xs ->
      let open U.O in
      List.fold_left ( * ) x xs

let unflatten_stage_index flat ranges =
  let sizes = List.map range_size_expr ranges in
  let open U.O in
  let rec loop acc flat = function
    | [] -> List.rev acc
    | [ _ ] -> List.rev (flat :: acc)
    | _ :: rest ->
        let stride = prod_expr rest in
        let axis = flat // stride in
        loop (axis :: acc) (flat mod stride) rest
  in
  loop [] flat sizes

let stage_index_sources buf idx =
  let srcs = src_tail idx in
  let ranges = buf.U.ranges in
  let removable_range r =
    match U.op r with Ops.Range | Ops.Const -> true | _ -> false
  in
  let compatible ranges srcs =
    let range_int_size r =
      match U.as_range r with
      | Some v -> Option.value (U.const_int_value v.size) ~default:(U.vmax r + 1)
      | None -> 1
    in
    List.length ranges = List.length srcs
    && List.for_all removable_range ranges
    && List.for_all2
         (fun range idx ->
           let r_size = range_int_size range in
           let i_size = U.vmax idx + 1 in
           r_size = i_size || r_size = 1)
         ranges srcs
  in
  if compatible ranges srcs then Some srcs
  else if not (List.for_all removable_range ranges) then None
  else match srcs, ranges with
    | [ flat ], _ :: _ :: _ -> Some (unflatten_stage_index flat ranges)
    | _ -> None

let substitute_stage_ranges mappings src =
  let mappings = List.filter (fun (k, v) -> not (U.equal k v)) mappings in
  if mappings = [] then src
  else
    let keys = List.map fst mappings in
    let rewrite u =
      match
        List.find_map
          (fun (k, v) -> if U.equal u k then Some v else None)
          mappings
      with
      | Some _ as r -> r
      | None ->
          if List.exists
               (fun key -> List.exists (U.equal key) (U.ranges u))
               keys
          then None
          else raise U.Bottom_up_gate
    in
    U.graph_rewrite ~bottom_up:true rewrite src

let remove_stage src (buf : U.stage_view) idx =
  if is_always_run (U.op src) || not buf.opts.removable then None
  else
    let accessed = U.Ref_tbl.create 8 in
    let indexes = ref [] and reduces = ref [] in
    ignore
      (U.toposort src ~gate:(fun x ->
           match U.op x with
           | Ops.Stage ->
               (match U.as_stage x with
                | Some { opts = { addrspace = Dtype.Global; _ }; _ } ->
                    U.Ref_tbl.replace accessed x (); false
                | _ -> true)
           | Ops.Mstack -> U.Ref_tbl.replace accessed x (); false
           | Ops.After -> U.Ref_tbl.replace accessed (U.buf_uop x) (); false
           | Ops.Store -> false
           | Ops.Param -> U.Ref_tbl.replace accessed x (); true
           | Ops.Index -> indexes := x :: !indexes; true
           | Ops.Reduce -> reduces := x :: !reduces; true
           | _ -> true));
    let pc = getv v_pcontig in
    if U.Ref_tbl.length accessed > 3 && pc <= 2 then None
    else
      let buffer_in_reduce =
        !reduces <> [] &&
        let found = ref false in
        let srcs = List.map src0 !reduces in
        ignore
          (U.toposort (U.sink srcs) ~gate:(fun x ->
               if !found then false
               else match U.op x with
                 | Ops.Param | Ops.Stage | Ops.After -> found := true; false
                 | _ -> true));
        !found
      in
      match stage_index_sources buf idx with
      | None -> None
      | Some buf_src ->
      if buffer_in_reduce then begin
        if pc <= 2 then None
        else
          let local_indexes =
            List.filter (fun x ->
                match U.as_index x with
                | Some { ptr; _ } ->
                    (match U.as_stage ptr with
                     | Some { opts = { addrspace = Dtype.Local; _ }; _ } -> true
                     | _ -> false)
                | _ -> false) !indexes
          in
          let exclude_ranges =
            List.concat_map (fun x -> U.ranges (U.sink (src_tail x)))
              local_indexes
          in
          let subs =
            List.filter_map (fun (k, v) ->
                if U.op k = Ops.Const then None else Some (k, v))
              (List.combine buf.ranges buf_src)
          in
          let is_pcontig, is_subs =
            List.partition (fun (k, v) ->
                List.exists (U.equal k) exclude_ranges
                || List.exists is_reduce_range (U.ranges v))
              subs
          in
          if is_subs = [] then None
          else
            let ret = substitute_stage_ranges is_subs src in
            if is_pcontig = [] then Some ret
            else
              let pc_rngs = List.map fst is_pcontig in
              let pc_idxs = List.map snd is_pcontig in
              let b =
                U.stage ~src:ret ~ranges:pc_rngs
                  ~opts:{ device = None; addrspace = Dtype.Local;
                          removable = true }
              in
              Some
                (U.replace idx ~src:(Array.of_list (b :: pc_idxs)) ())
      end else
        let mappings =
          List.filter_map (fun (k, v) ->
              if U.op k = Ops.Const then None
              else match U.arg v with
                | U.Arg.Value c when Const.view c = Const.Invalid -> None
                | _ -> Some (k, v))
            (List.combine buf.ranges buf_src)
        in
        Some (substitute_stage_ranges mappings src)

let remove_stage_index n =
  match U.as_index n with
  | Some { ptr; _ } -> (
      match U.as_stage ptr with
      | Some buf -> remove_stage buf.src buf n
      | None -> None)
  | None -> None

let starts_with ~prefix s =
  String.length s >= String.length prefix
  && String.sub s 0 (String.length prefix) = prefix

let late_buffer_slice n =
  match U.as_stage n with
  | Some { src; ranges; _ }
    when U.op src = Ops.Bitcast || U.op src = Ops.Contiguous ->
      let is_disk = match U.device_of n with
        | Some (Single d) -> starts_with ~prefix:"DISK" d || starts_with ~prefix:"TINYFS" d
        | _ -> false
      in
      if not is_disk then None
      else
        let range_size r = match U.as_range r with
          | Some v -> Option.value (U.const_int_value v.size) ~default:(U.vmax r + 1)
          | None -> 1
        in
        let size = prod (List.map range_size ranges) in
        let rec find_idx x = match List.find_opt (fun u -> U.op u = Ops.Index)
                                     (U.children x) with
          | Some i -> i
          | None -> (match U.children x with
                     | c :: _ -> find_idx c | [] -> x)
        in
        let idx = find_idx src in
        let offset =
          match U.as_index idx with
          | Some _ when Array.length (U.src idx) = 2 ->
              (U.src idx).(1)
          | Some _ ->
              let terms = src_tail idx in
              (match terms with
               | [] -> int_ 0
               | t :: rest -> List.fold_left (fun acc i ->
                   U.alu_binary ~op:Ops.Add ~lhs:acc ~rhs:i) t rest)
          | None -> int_ 0
        in
        let src_base = match U.as_index idx with
          | Some v -> v.ptr | None -> idx
        in
        let slice =
          U.slice ~src:src_base ~offset ~size ~dtype:(U.dtype src)
        in
        Some (U.replace n ~src:[| slice; List.hd ranges |] ())
  | _ -> None

let limit_bufs (ctx : Indexing.indexing_context) n =
  match U.op n with
  | op when Ops.Group.is_binary op || Ops.Group.is_ternary op ->
      let dname = match U.device_of n with
        | Some (Single d) -> Some (List.hd (String.split_on_char ':' d))
        | Some (Multi ds) -> Some (List.hd (String.split_on_char ':' (List.hd ds)))
        | _ -> None
      in
      Option.bind dname (fun d ->
          let max_bufs = match getv v_max_bufs with
            | 0 -> device_max_bufs d | n -> n
          in
          if max_bufs = 0 then None
          else
            let bufs = U.Ref_tbl.create 16 in
            ignore
              (U.toposort n ~gate:(fun u ->
                   match U.op u with
                   | Ops.Stage | Ops.After | Ops.Param
                   | Ops.Mselect | Ops.Mstack ->
                       U.Ref_tbl.replace bufs u (); false
                   | _ -> true));
            if U.Ref_tbl.length bufs <= max_bufs - 1 then None
            else
              let children = U.children n in
              let new_children =
                List.map (fun s ->
                    if Ops.Group.is_elementwise (U.op s)
                       && U.device_of s <> None then
                      let orig = U.ranges s in
                      let renum =
                        List.map (fun x -> match U.as_range x with
                            | Some v ->
                                let axis = ctx.range_idx in
                                ctx.range_idx <- ctx.range_idx + 1;
                                U.range ~size:v.size ~axis ~sub:v.sub
                                  ~kind:Axis_type.Loop
                                  ~dtype:(Dtype.val_of (U.dtype x))
                                  ~parents:v.parents ()
                            | None -> x) orig
                      in
                      let subst = U.substitute (List.combine orig renum) s in
                      let opts : U.stage_opts =
                        { device = U.device_of s; addrspace = Dtype.Global;
                          removable = true }
                      in
                      let b =
                        U.stage ~src:subst ~ranges:renum ~opts
                      in
                      U.replace s ~op:Ops.Index
                        ~src:(Array.of_list (b :: orig)) ()
                    else s) children
              in
              if List.for_all2 ( == ) children new_children then None
              else Some (U.replace n ~src:(Array.of_list new_children) ()))
  | _ -> None

(* Add buffers *)

let range_int_size r =
  match U.as_range r with
  | Some v -> Option.value (U.const_int_value v.size) ~default:(U.vmax r + 1)
  | None -> 1

let flat_index_of_ranges ?dims ranges =
  let range_dims ranges =
    match dims with
    | Some dims when List.length dims = List.length ranges -> dims
    | _ -> List.map range_int_size ranges
  in
  match ranges with
  | [] -> int_ 0
  | [ r ] -> r
  | ranges ->
      let dims = Array.of_list (range_dims ranges) in
      let ranges = Array.of_list ranges in
      let n_axes = Array.length ranges in
      let acc = ref ranges.(n_axes - 1) in
      let stride = ref 1 in
      for i = n_axes - 2 downto 0 do
        stride := !stride * dims.(i + 1);
        let open U.O in
        let term =
          if !stride = 0 then int_ 0
          else if !stride = 1 then ranges.(i)
          else ranges.(i) * int_ !stride
        in
        acc := !acc + term
      done;
      !acc

let flatten_stage n =
  match U.as_stage n with
  | Some { src; ranges; opts; _ }
    when List.length ranges > 1
         && List.for_all
              (fun r -> match U.op r with Ops.Range | Ops.Const -> true | _ -> false)
              ranges ->
      let shape =
        try U.max_shape n with Invalid_argument _ ->
          List.map (fun r -> U.vmax r + 1) ranges
      in
      let flat_src =
        U.buffer ~slot:(-1) ~dtype:(U.dtype src)
          ~shape:(shape_node [ prod shape ]) ()
      in
      let flat_view = U.reshape ~src:flat_src ~shape:(shape_node shape) in
      let flat_idx =
        match Indexing.apply_movement_op ~shapes:shape_of flat_view ranges with
        | [ idx ] -> idx
        | _ -> invalid_arg "Rangeify.flatten_stage: reshape did not flatten"
      in
      let flat = U.stage ~src ~ranges:[ flat_idx ] ~opts in
      let ret = U.reshape ~src:flat ~shape:(shape_node shape) in
      let sym_shape =
        List.map
          (fun r ->
             match U.as_range r with
             | Some v when U.op v.size <> Ops.Const -> Some v.size
             | _ -> None)
          ranges
      in
      if List.for_all Option.is_none sym_shape then Some ret
      else
        let sym =
          sym_shape
          |> List.map (function Some dim -> dim | None -> int_ 1)
        in
        let size = match sym with [ dim ] -> dim | dims -> U.stack dims in
        let zeros = shape_node (List.map (fun _ -> 0) sym_shape) in
        Some (U.shrink ~src:ret ~offset:zeros ~size)
  | _ -> None

let range_axis_cmp a b =
  let ax u = match U.as_range u with Some v -> v.axis | None -> 0 in
  compare (ax a) (ax b)

let stage_to_store ?(allow_locals = true) counter n =
  match U.as_stage n with
  | Some { src; ranges; opts } ->
      let shape =
        match shape_of n with
        | Some _ as shape -> shape
        | None -> (try Some (U.max_shape n) with Invalid_argument _ -> None)
      in
      let range_dims = List.map range_int_size ranges in
      let stage_dims =
        match shape with
        | Some shape when shape <> [] && List.length shape = List.length ranges ->
            shape
        | _ -> range_dims
      in
      let idx_expr =
        match ranges with
        | [ idx ] -> idx
        | _ -> flat_index_of_ranges ~dims:stage_dims ranges
      in
      let idx_ranges = List.sort range_axis_cmp (U.ranges idx_expr) in
      let size =
        match stage_dims with
        | _ :: _ -> prod stage_dims
        | _ ->
            let size_ranges =
              match idx_ranges with [] -> U.ranges src | ranges -> ranges
            in
            prod (List.map range_int_size size_ranges)
      in
      if size <= 0 then None
      else
        let base_dt = Dtype.val_of (U.dtype n) in
        let val_dt = Dtype.Val base_dt in
        let ptr_dt =
          Dtype.Ptr.create base_dt ~addrspace:opts.addrspace ~size
        in
        let idx_dt = Dtype.Ptr ptr_dt in
        (match U.op src with
        | Ops.After ->
            let stores =
              List.filter (fun d -> match U.as_store d with
                  | Some { dst; _ } -> U.op dst = Ops.Index
                  | _ -> false) (src_tail src)
            in
            let buf = U.buf_uop (src0 src) in
            let cmp a b =
              let c = range_axis_cmp a b in
              if c = 0 then compare (U.tag a) (U.tag b) else c
            in
            let ended_stores =
              List.filter_map
                (fun store ->
                   let { U.dst; value; gate } =
                     Option.get (U.as_store store)
                   in
                   let target =
                     match U.as_index dst with
                     | Some { ptr = p; _ } ->
                         (match U.as_stage p with
                          | Some { src = inner; _ } when U.op inner = Ops.Index ->
                              inner
                          | _ -> dst)
                     | None -> dst
                   in
                   if value == target then None
                   else
                     let ranges =
                       List.sort_uniq cmp (U.ranges target @ idx_ranges)
                     in
                     let target = U.replace target ~dtype:idx_dt () in
                     Some
                       (U.end_
                          ~value:(U.store ~dst:target ~value ?gate ())
                          ~ranges))
                stores
            in
            if ended_stores = [] then Some buf
            else Some (U.after ~src:buf ~deps:ended_stores)
        | _ when opts.addrspace = Dtype.Global ->
            let id = !counter in
            incr counter;
            let buf =
              U.buffer ~slot:id ?device:opts.device ~shape:(shape_node [ size ])
                ~addrspace:Dtype.Global ~dtype:val_dt ()
            in
            let idx =
              U.replace
                (U.index ~ptr:buf ~idxs:[idx_expr] ~as_ptr:true ())
                ~dtype:idx_dt ()
            in
            let ended =
              U.end_ ~value:(U.store ~dst:idx ~value:src ()) ~ranges:idx_ranges
            in
            Some (U.after ~src:buf ~deps:[ ended ])
        | _ when opts.addrspace = Dtype.Local && allow_locals ->
            let id = !counter in
            incr counter;
            let buf =
              U.buffer ~slot:id ~shape:(shape_node [ size ]) ~dtype:val_dt
                ~addrspace:Dtype.Local ()
            in
            let idx =
              U.replace
                (U.index ~ptr:buf ~idxs:[idx_expr] ~as_ptr:true ())
                ~dtype:idx_dt ()
            in
            let st =
              U.end_ ~value:(U.store ~dst:idx ~value:src ()) ~ranges:idx_ranges
            in
            Some (U.after ~src:buf ~deps:[ U.barrier ~srcs:[ st ] () ])
        | _ -> None)
  | None -> None

let refresh_index_ptr_dtype n =
  match U.as_index n with
  | Some { ptr; _ } ->
      (match U.dtype n, U.dtype ptr with
       | Dtype.Ptr pdt, Dtype.Ptr fresh when not (Dtype.Ptr.equal pdt fresh) ->
           Some (U.replace n ~dtype:(Dtype.Ptr fresh) ())
       | Dtype.Val vdt, Dtype.Ptr fresh
         when not (Dtype.Val.equal vdt (Dtype.Ptr.value fresh)) ->
           Some (U.replace n ~dtype:(Dtype.Val (Dtype.Ptr.value fresh)) ())
       | _ -> None)
  | None -> None

(* Split kernels *)

type split_context = {
  mutable slot : int;
  buf_map : U.t U.Ref_tbl.t;
  mutable formals : (int * U.t) list;
  (* BINDs unbound inside the kernel, most recent first. *)
  mutable vars : U.t list;
  mutable range_ctr : int;
  mutable opts : U.Opt.t list option;
  buf_shapes : int list U.Ref_tbl.t;
}

let create_split_context () =
  { slot = 0; buf_map = U.Ref_tbl.create 16; formals = [];
    vars = [];
    range_ctr = 0;
    opts = None; buf_shapes = U.Ref_tbl.create 16 }

let same_split_buffer a b =
  if a == b then true
  else
    let identity n =
      let b = U.buf_uop n in
      match U.as_buffer b, U.as_param b with
      | Some { buffer; _ }, _ ->
          Some (`Buffer (buffer.slot, buffer.addrspace))
      | _, Some { param; _ } ->
          Some (`Param (param.slot, param.addrspace))
      | None, None -> None
    in
    match identity a, identity b with
    | Some ia, Some ib -> ia = ib
    | _ -> false

let find_buf_arg ctx key =
  U.Ref_tbl.find_opt ctx.buf_map key

let replace_formal_arg ctx old_arg new_arg =
  let same_arg arg =
    same_split_buffer arg old_arg
  in
  ctx.formals <-
    List.map
      (fun (slot, arg) -> (slot, if same_arg arg then new_arg else arg))
      ctx.formals

let debuf ctx n =
  let dtype = U.dtype n in
  let shape =
    match shape_of n with
    | Some sh when sh <> [] -> sh
    | _ ->
        (match U.Ref_tbl.find_opt ctx.buf_shapes n with
         | Some sh when sh <> [] -> sh
         | _ ->
             match ptr_size_shape n with
             | Some sh -> sh
             | None -> [ 1 ])
  in
  let max_shape =
    match (try U.max_shape n with Invalid_argument _ -> []) with
    | _ :: _ as sh -> sh
    | [] -> shape
  in
  let size = prod max_shape in
  let addrspace =
    match U.addrspace n with Some a -> a | None -> Dtype.Global
  in
  let ptr_dt =
    Dtype.Ptr.create (Dtype.val_of dtype) ~addrspace ~size
  in
  let slot = ctx.slot in
  ctx.slot <- ctx.slot + 1;
  let ret =
    let device = U.device_of n in
    let param =
      U.param ~slot ~dtype:(Dtype.Ptr ptr_dt) ~shape:(shape_node [ size ]) ?device
        ~addrspace ()
    in
    let reshaped = U.reshape ~src:param ~shape:(shape_node max_shape) in
    (* Symbolic buffers: the ptr is sized for [max_shape]; shrink the
       max-sized view down to the actual [shape] when they differ. *)
    if max_shape <> shape then
      U.shrink ~src:reshaped
        ~offset:(shape_node (List.map (fun _ -> 0) shape))
        ~size:(shape_node shape)
    else reshaped
  in
  let arg = match find_buf_arg ctx n with
    | Some arg -> arg
    | None ->
        U.Ref_tbl.replace ctx.buf_map n n;
        n
  in
  ctx.formals <- (slot, arg) :: ctx.formals;
  Some ret

let handle_after ctx n =
  let op = U.op n in
  let is_local = match U.dtype n with
    | Dtype.Ptr p -> Dtype.Ptr.addrspace p = Dtype.Local && op = Ops.After
    | _ -> false
  in
  if is_local then None
  else
    let buf = match op with
      | Ops.After | Ops.Mstack | Ops.Mselect -> U.buf_uop n
      | _ -> n
    in
  (match find_buf_arg ctx buf with
     | None -> U.Ref_tbl.replace ctx.buf_map buf n
     | Some existing
       when same_split_buffer existing buf && op = Ops.After && src_tail n <> [] ->
         U.Ref_tbl.replace ctx.buf_map buf n;
         replace_formal_arg ctx existing n
     | Some _ -> ());
    Some buf

let unbind_kernel ctx n =
  if not (List.exists (( == ) n) ctx.vars) then ctx.vars <- n :: ctx.vars;
  Option.map (fun (v : U.bind_view) -> v.var) (U.as_bind n)

let renumber_range ctx n =
  match U.as_range n, U.node_tag n with
  | Some v, Some "" ->
      let axis = ctx.range_ctr in
      ctx.range_ctr <- ctx.range_ctr + 1;
      Some
        (U.range ~size:v.size ~axis ~sub:v.sub ~kind:v.kind
           ~dtype:(Dtype.val_of (U.dtype n)) ~parents:v.parents ())
  | _ -> None

let renumber_kernel_ranges root =
  let add_unique acc r =
    if U.op r <> Ops.Range || List.exists (fun x -> x == r) acc then acc
    else acc @ [ r ]
  in
  let topo = U.toposort root in
  let reduce_ranges =
    List.fold_left
      (fun acc n ->
        match U.as_reduce n with
        | Some { ranges; _ } -> List.fold_left add_unique acc ranges
        | None -> acc)
      [] topo
  in
  let end_ranges =
    List.fold_left
      (fun acc n ->
        match U.as_end n with
        | Some { ranges; _ } -> List.fold_left add_unique acc ranges
        | None -> acc)
      reduce_ranges topo
  in
  let all_ranges =
    List.fold_left add_unique end_ranges
      (List.filter (fun n -> U.op n = Ops.Range) topo)
  in
  let mappings =
    List.mapi
      (fun axis r ->
        match U.as_range r with
        | Some v ->
            let r' =
              U.range ~size:v.size ~axis ~sub:v.sub ~kind:v.kind
                ~dtype:(Dtype.val_of (U.dtype r)) ~parents:v.parents ()
            in
            (r, r')
        | None -> assert false)
      all_ranges
  in
  U.substitute mappings root

let get_contiguous ctx n =
  (match U.as_contiguous_opts n with
   | Some opts when opts <> [] -> ctx.opts <- Some opts
   | _ -> ());
  Some (src0 n)

let find_bufs n =
  (* A base buffer read through two INDEXes whose immediate pointer has a
     different op (e.g. a raw BUFFER vs an AFTER/STAGE over it) is a
     read/write cycle within the kernel. Key on the pointer op, matching
     tinygrad's [read_from.setdefault(buf, idx.src[0].op)]. *)
  let read_from : Ops.t U.Ref_tbl.t = U.Ref_tbl.create 8 in
  List.iter (fun s ->
      match U.as_index s with
      | Some { ptr; _ } ->
          let b = U.buf_uop ptr in
          (match U.op b with
           | Ops.Buffer | Ops.Param ->
               let ptr_op = U.op ptr in
               (match U.Ref_tbl.find_opt read_from b with
                | Some prev when not (Ops.equal prev ptr_op) ->
                    failwith "cycle detected while indexing buffer"
                | _ -> U.Ref_tbl.replace read_from b ptr_op)
           | _ -> ())
      | None -> ())
    (U.toposort n ~gate:(fun x -> U.op x <> Ops.After));
  None

let to_define_global ctx n =
  match U.op n with
  | Ops.Store -> find_bufs n
  | Ops.Buffer | Ops.Mstack | Ops.Mselect -> debuf ctx n
  | Ops.Param -> (
      match U.as_param n, U.dtype n with
      (* A named, ranged PARAM normalises to the canonical variable so
         binding identity survives the kernel split. *)
      | Some { param = { name = Some name; vmin_vmax = Some (lo, hi); _ }; _ },
        dtype ->
          Some
            (U.variable ~name ~min_val:lo ~max_val:hi
               ~dtype:(Dtype.val_of dtype) ())
      | Some { param = { name = None; _ }; shape }, Dtype.Val _
        when U.op shape <> Ops.Noop ->
          debuf ctx n
      | _ -> None)
  | Ops.Bind -> unbind_kernel ctx n
  | Ops.After -> handle_after ctx n
  (* ALU params are scalar symbolic values, not buffers. *)
  | Ops.Index
    when Array.length (U.src n) = 1
         && (match U.as_param (src0 n) with
            | Some { param = { addrspace = Dtype.Alu; _ }; _ } -> true
            | _ -> false) ->
      Some (src0 n)
  | Ops.Stage ->
      (match U.as_stage n with
       | Some { opts; _ } when opts.device <> None ->
           Some
             (U.replace n
                ~arg:(U.Arg.Stage_info { opts with device = None }) ())
       | _ -> None)
  | Ops.Const when Array.length (U.src n) > 0 ->
      Some (U.replace n ~src:[||] ())
  | Ops.Range -> None
  | Ops.Contiguous -> get_contiguous ctx n
  | Ops.Noop when Array.length (U.src n) > 0 -> Some (src0 n)
  | _ -> None

let compact_kernel_params ctx body =
  let add_unique xs x = if List.exists (( == ) x) xs then xs else xs @ [ x ] in
  let topo = U.toposort body in
  let params =
    List.fold_left
      (fun acc n ->
         match U.as_param n with
         | Some { param = { slot; addrspace; _ }; _ }
           when slot >= 0 && addrspace <> Dtype.Alu ->
             add_unique acc n
         | _ -> acc)
      [] topo
  in
  let buffer_slot_map = List.mapi (fun slot param -> param, slot) params in
  let find_buffer_slot old =
    List.find_map
      (fun (param, slot) -> if param == old then Some slot else None)
      buffer_slot_map
  in
  let body =
    U.graph_rewrite ~name:"compact kernel params" ~walk:true
      (fun n ->
         match U.as_param n with
         | Some { param; _ }
           when param.slot >= 0 && param.addrspace <> Dtype.Alu -> (
             match find_buffer_slot n with
             | Some slot when slot <> param.slot ->
                 Some
                   (U.replace n
                      ~arg:(U.Arg.Param_arg { param with slot })
                      ())
             | _ -> None)
         | _ -> None)
      body
  in
  let bufs =
    List.map
      (fun param_node ->
         let old_slot =
           match U.as_param param_node with
           | Some { param; _ } -> param.slot
           | None -> assert false
         in
         match
           List.find_map
             (fun (slot, arg) -> if slot = old_slot then Some arg else None)
             ctx.formals
         with
         | Some arg -> arg
         | None -> param_node)
      params
  in
  body, bufs @ List.rev ctx.vars

let split_store n =
  match U.op n with
  | Ops.Store | Ops.End ->
      if U.ranges n <> [] then None
      else
        let ctx = create_split_context () in
        let record ptr idxs =
          match U.op ptr with
          | Ops.Buffer | Ops.Param ->
              let dims = List.map (fun r -> match U.as_range r with
                  | Some v -> U.const_int_value v.size
                  | None -> if U.op r = Ops.Const then Some 1 else None) idxs
              in
              if List.for_all Option.is_some dims then
                U.Ref_tbl.replace ctx.buf_shapes ptr (List.map Option.get dims)
          | _ -> ()
        in
        let nodes = U.toposort n in
        ctx.slot <-
          List.fold_left
            (fun acc nd ->
               match U.as_param nd with
               | Some { param = { slot; _ }; _ } when slot >= 0 ->
                   max acc (slot + 1)
               | _ -> acc)
            ctx.slot nodes;
        List.iter (fun nd -> match U.as_store nd with
            | Some { dst; _ } ->
                (match U.as_index dst with
                 | Some { ptr; _ } ->
                     let tail = src_tail dst in
                     if tail <> [] then record ptr tail
                     else (match U.op ptr with
                         | Ops.Buffer | Ops.Param ->
                             U.Ref_tbl.replace ctx.buf_shapes ptr []
                         | _ -> ())
                 | None -> ())
            | None -> ()) nodes;
        List.iter (fun nd -> match U.as_index nd with
            | Some { ptr; _ } ->
                let tail = src_tail nd in
                if List.length tail > 1
                   && not (U.Ref_tbl.mem ctx.buf_shapes ptr)
                then record ptr tail
            | None -> ()) nodes;
        let rewrite =
          U.first_match
            [ to_define_global ctx; Simplify.flatten_range; pm_mop_through_index ]
        in
        let ret =
          U.graph_rewrite ~bottom_up:true ~name:"kernel_split" rewrite n
        in
        let ret =
          match U.as_end ret with
          | Some { value; ranges } -> (
              match U.as_store value with
              | Some { dst; value = stored; gate } -> (
                  match U.as_index dst with
                  | Some { ptr; idxs } ->
                      let is_range r = U.op r = Ops.Range in
                      let stored_ranges =
                        List.filter is_range (U.ranges stored)
                      in
                      let same_size a b = U.vmax a = U.vmax b in
                      let flat_size ranges =
                        List.fold_left
                          (fun acc r -> acc * (U.vmax r + 1))
                          1 ranges
                      in
                      let same_flat_size dst stored =
                        flat_size dst = flat_size stored
                      in
                      let range_mem r ranges =
                        List.exists (U.equal r) ranges
                      in
                      let same_ranges a b =
                        List.length a = List.length b
                        && List.for_all (fun r -> range_mem r b) a
                      in
                      let sort_ranges ranges =
                        List.sort_uniq
                          (fun a b ->
                             let c = range_axis_cmp a b in
                             if c = 0 then compare (U.tag a) (U.tag b) else c)
                          ranges
                      in
                      let rewrite_flat_idx idx =
                        let dst_ranges =
                          List.filter is_range (U.ranges idx)
                        in
                        match dst_ranges, stored_ranges with
                        | [ dst ], [ stored ] when same_size dst stored ->
                            Some (`Substitute [ (dst, stored) ])
                        | _ :: _, _ :: _
                          when (not (same_ranges dst_ranges stored_ranges))
                               && same_flat_size dst_ranges stored_ranges ->
                            let gate_ranges =
                              match gate with
                              | None -> []
                              | Some gate -> U.ranges gate
                            in
                            if
                              List.exists
                                (fun r -> range_mem r gate_ranges)
                                dst_ranges
                            then None
                            else
                              Some
                                (`Replace
                                  ( flat_index_of_ranges stored_ranges,
                                    dst_ranges ))
                        | _ -> None
                      in
                      (match idxs with
                       | [ idx ] -> (
                           match rewrite_flat_idx idx with
                           | Some (`Substitute subs) ->
                           let idxs = List.map (U.substitute subs) idxs in
                           let dst =
                             U.index ~ptr ~idxs
                               ~as_ptr:(Dtype.is_ptr (U.dtype dst)) ()
                           in
                           let ranges = List.map (U.substitute subs) ranges in
                           U.end_ ~value:(U.store ~dst ~value:stored ?gate ())
                             ~ranges
                           | Some (`Replace (idx, dst_ranges)) ->
                               let dst =
                                 U.index ~ptr ~idxs:[ idx ]
                                   ~as_ptr:(Dtype.is_ptr (U.dtype dst)) ()
                               in
                               let ranges =
                                 ranges
                                 |> List.filter
                                      (fun r -> not (range_mem r dst_ranges))
                                 |> fun rs -> sort_ranges (rs @ stored_ranges)
                               in
                               U.end_
                                 ~value:(U.store ~dst ~value:stored ?gate ())
                                 ~ranges
                           | None -> ret)
                       | _ -> ret)
                  | None -> ret)
              | None -> ret)
          | None -> ret
        in
        let ret = renumber_kernel_ranges ret in
	        let stored = match U.as_store ret with
          | Some { value; _ } -> Some value
          | None -> (match U.as_end ret with
              | Some { value; _ } ->
                  (match U.as_store value with
                   | Some { value = v; _ } -> Some v
                   | None ->
                       if U.op value = Ops.Call then None
                       else failwith "split_store: END wraps non-STORE")
              | None ->
                  if U.op ret = Ops.Call then None
                  else failwith "split_store: unexpected result")
        in
        (match stored with
         | None -> None
         | Some stored ->
             let info : U.call_info =
               {
                 grad_fxn = None;
                 metadata = [];
                 name = None;
                 precompile = false;
                 precompile_backward = false;
                 aux = None;
               }
             in
             let body = match U.op stored with
               | Ops.Copy | Ops.Slice ->
                   let ended = match U.as_end ret with
                     | Some { ranges; _ } -> ranges | None -> []
                   in
                   U.replace stored
                     ~src:(Array.of_list (U.children stored @ ended)) ()
             | _ ->
                 U.sink ~kernel_info:{
                   name = ""; axis_types = []; dont_use_locals = false;
                   applied_opts = []; opts_to_apply = ctx.opts;
                   estimates = None; beam = 0 } [ ret ]
             in
             let body, args = compact_kernel_params ctx body in
             Some (U.call ~body ~args ~info))
  | _ -> None

(* WAR deps *)

let fix_war_deps root =
  let afters =
    List.filter (fun n -> U.op n = Ops.After) (U.toposort root)
  in
  if afters = [] then root
  else
    let buf_of n = match U.op n with
      | Ops.After -> U.buf_uop n | _ -> n
    in
    let kernel_assign : U.t U.Ref_tbl.t = U.Ref_tbl.create 16 in
    List.iter (fun u -> U.Ref_tbl.replace kernel_assign (buf_of u) u) afters;
    let call_of u = match U.op u with
      | Ops.After -> List.find_opt (fun d -> U.op d = Ops.Call) (src_tail u)
      | _ -> None
    in
    let assign_rep : U.t list U.Ref_tbl.t = U.Ref_tbl.create 16 in
    List.iter (fun u ->
        let u_buf = buf_of u in
        let reads = match call_of u with
          | Some c ->
              List.filter (fun a -> U.op a = Ops.Buffer || U.op a = Ops.Param)
                (src_tail c)
          | None -> []
        in
        List.iter (fun s ->
            if s != u_buf then
              match U.Ref_tbl.find_opt kernel_assign s with
              | Some a when not (call_of a <> None && call_of a = call_of u) ->
                  let prev =
                    Option.value (U.Ref_tbl.find_opt assign_rep a) ~default:[]
                  in
                  if not (List.exists (( == ) u) prev) then
                    U.Ref_tbl.replace assign_rep a (u :: prev)
              | _ -> ()) reads) afters;
    if U.Ref_tbl.length assign_rep = 0 then root
    else
      U.graph_rewrite ~name:"fix_war_deps" (fun n ->
          match U.Ref_tbl.find_opt assign_rep n with
          | Some extra when U.op n = Ops.After ->
              Some (U.after ~src:(src0 n) ~deps:(src_tail n @ extra))
          | _ -> None) root

(* Main pipeline *)

let post_rangeify_rules =
  U.first_match [
    pm_mop_through_index;
    pm_mop_past_after;
    pm_mop_past_end;
    Upat.Pattern_matcher.rewrite symbolic;
    Upat.Pattern_matcher.rewrite Simplify.pm_reduce_simplify;
    cleanup_dead_axes;
    remove_noop_stage;
    remove_stage_index;
    (fun n -> match U.as_stage n with
       | Some { src; _ } when U.op src = Ops.Const ->
           (match U.arg src with
            | U.Arg.Value v -> Some (U.const v)
            | _ -> None)
       | _ -> None);
    (fun n -> match U.as_index n with
       | Some { ptr; _ } when U.op ptr = Ops.Const -> Some ptr
       | _ -> None);
    (fun n -> match U.op n with
       | Ops.Copy ->
           let s = src0 n in
           (match U.arg s with
            | U.Arg.Value v when U.op s = Ops.Const -> Some (U.const v)
            | _ -> None)
       | _ -> None);
    (fun n -> match U.op n with
       | Ops.Noop when Array.length (U.src n) > 0
                       && U.op (src0 n) = Ops.Const ->
           Some (src0 n)
       | _ -> None);
    (fun n -> match U.as_index n with
       | Some { ptr; _ } when U.op ptr = Ops.Mstack ->
           (match U.children ptr with
            | s :: _ ->
                let b = base s in
                (match U.arg b with
                 | U.Arg.Value v when U.op b = Ops.Const -> Some (U.const v)
                 | _ -> None)
            | [] -> None)
       | _ -> None);
  ]

let add_buffers_rules ?(allow_locals = true) counter =
  U.first_match [
    pm_mop_through_index; pm_mop_past_after; pm_mop_past_end;
    flatten_stage;
    late_buffer_slice;
    stage_to_store ~allow_locals counter;
    refresh_index_ptr_dtype;
    (fun n -> match U.as_range n, U.node_tag n with
       | Some _, None -> Some (U.with_tag "" n)
       | _ -> None);
    (* RESHAPEs through MSELECT/MSTACK *)
    (fun n -> match U.op n with
       | Ops.Mselect | Ops.Mstack ->
           let children = U.children n in
           if children <> []
              && List.for_all (fun c -> U.op c = Ops.Reshape) children
           then
             let unwrapped = List.map (fun c -> base (src0 c)) children in
             let inner = U.replace n ~src:(Array.of_list unwrapped) () in
             (match shape_of n with
              | Some sh when sh <> [] ->
                  Some (U.reshape ~src:inner ~shape:(shape_node sh))
              | _ -> Some inner)
           else None
       | _ -> None);
    (* Strip RESHAPE on CALL args *)
    (fun n -> match U.op n with
       | Ops.Call ->
           let args = src_tail n in
           let new_args =
             List.map (fun a -> if U.op a = Ops.Reshape then src0 a else a) args
           in
           if List.for_all2 ( == ) args new_args then None
           else
             Some
               (U.replace n
                  ~src:(Array.of_list (src0 n :: new_args)) ())
       | _ -> None);
    (* Strip MOP on AFTER deps; flatten nested AFTERs *)
    (fun n -> match U.op n with
       | Ops.After ->
           let deps = src_tail n in
           let new_deps =
             List.map (fun d ->
                 Option.value (movement_src d) ~default:d) deps
           in
           let flat =
             List.concat_map (fun d -> match U.op d with
                 | Ops.After -> src_tail d | _ -> [ d ]) new_deps
           in
           if List.length flat = List.length deps
              && List.for_all2 ( == ) flat deps then None
           else Some (U.after ~src:(src0 n) ~deps:flat)
       | _ -> None);
    (* Remove invalid writes *)
    (fun n -> match U.op n with
       | Ops.After ->
           let deps = src_tail n in
           let real =
             List.filter (fun d -> match U.op d with
                 | Ops.Noop when Array.length (U.src d) = 0 -> false
                 | Ops.End ->
                     let v = src0 d in
                     not (U.op v = Ops.Noop && Array.length (U.src v) = 0)
                 | _ -> true) deps
           in
           if List.length real < List.length deps then
             match real with
             | [] -> Some (src0 n)
             | _ -> Some (U.after ~src:(src0 n) ~deps:real)
           else None
      | _ -> None);
  ]

let get_kernel_graph root =
  let root =
    U.graph_rewrite ~name:"multi_pm"
      (Multi.multi_pm ~shapes:shape_of ~devices:U.device_of)
      root
  in
  let root =
    if getv v_openpilot = 0 then root
    else
      let ctx = U.Ref_tbl.create 16 in
      U.graph_rewrite ~name:"fold moved afters" (pm_fold_moved_after ctx) root
  in
  let root =
    U.graph_rewrite ~bottom_up:true ~name:"earliest rewrites"
      earliest_rewrites root
  in
  let rctx =
    Indexing.run_rangeify root ~shapes:shape_of ~shape_exprs:shape_expr_of
  in
  let root =
    Indexing.apply_rangeify_pass rctx ~shapes:shape_of
      ~shape_exprs:shape_expr_of root
  in
  let root =
    U.graph_rewrite ~name:"post_rangeify" post_rangeify_rules root
  in
  let root =
    U.graph_rewrite ~name:"limit_bufs" (limit_bufs rctx) root
  in
  let buffer_slot_start =
    List.fold_left (fun acc x ->
        let slot =
          match U.as_buffer x, U.as_param x with
          | Some { buffer = { slot; _ }; _ }, _ -> Some slot
          | None, Some { param = { slot; _ }; _ } -> Some slot
          | None, None -> None
        in
        match slot with
        | Some slot when slot >= 0 -> max acc (slot + 1)
        | Some _ | None -> acc)
      0 (U.toposort root)
  in
  let counter = ref buffer_slot_start in
  let root =
    U.graph_rewrite ~name:"add_buffers" ~bottom_up:true
      (add_buffers_rules ~allow_locals:false counter) root
  in
  let root =
    U.graph_rewrite ~enter_calls:false ~bottom_up:true
      ~name:"split_kernels" split_store root
  in
  let root =
    U.graph_rewrite ~enter_calls:false ~bottom_up:true
      ~name:"split_kernels_fixpoint" split_store root
  in
  fix_war_deps root
