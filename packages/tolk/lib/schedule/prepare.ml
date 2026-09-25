(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Tensor preparation owns scheduling rewrites before range assignment. *)

open Tolk_uop
module U = Uop

let getv = Helpers.Context_var.get

let prod l = List.fold_left ( * ) 1 l
let int_ n = U.const_int n

let src0 u = (U.src u).(0)
let src_tail u =
  let s = U.src u in
  Array.to_list (Array.sub s 1 (Array.length s - 1))

let shape_node = function [ d ] -> int_ d | ds -> U.stack (List.map int_ ds)

let movement_src u =
  match U.op u with
  | Ops.Reshape | Ops.Expand | Ops.Pad | Ops.Shrink | Ops.Permute | Ops.Flip ->
      Some (src0 u)
  | _ -> None

let is_movement u = Option.is_some (movement_src u)

let shape_of n =
  let rec concrete = function
    | [] -> Some []
    | dim :: rest ->
        Option.bind (U.const_int_value dim) (fun value ->
            Option.map (fun dims -> value :: dims) (concrete rest))
  in
  Option.bind (U.shape_opt n) concrete

let argsort order =
  List.map snd (List.sort compare (List.mapi (fun i o -> (o, i)) order))

let base = U.base

let pm_mop_through_index n =
  match U.as_index n with
  | Some { ptr; _ } when is_movement ptr ->
      let src = Option.get (movement_src ptr) in
      let idxs = src_tail n in
      let mop_shape u =
        match shape_of u with
        | Some _ as shape -> shape
        | None -> (
            try Some (List.map (fun dim -> Bound.to_int (U.vmax dim)) (U.shape u))
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

let movement_ops n =
  match
    U.first_match [ pm_mop_through_index; pm_mop_past_after; pm_mop_past_end ] n
  with
  | Some n' when not (U.equal n n') -> Some n'
  | Some _ | None -> None

(* Fold moved AFTERs (openpilot hack) *)

let is_invalid u =
  U.op u = Ops.Const
  && (match U.arg u with
      | U.Arg.Value v -> Const.view v = Const.Invalid | _ -> false)

let found_after ctx ~after ~value =
  let x = ref value and a = ref after in
  if getv Helpers.float16 <> 0 && U.op !x = Ops.Cast
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
  let boundary s =
    match U.op s with
    | Ops.Stage | Ops.Copy -> false
    | Ops.After -> not (List.exists (fun dep ->
        match U.as_store dep with
        | Some {dst; _} -> U.base dst == U.base (src0 s)
        | None -> false) (src_tail s))
    | _ -> true in
  let slice = U.toposort ~enter_calls:false ~gate:boundary value in
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
        if r && unsafe (U.op s) && not (s == target && U.op s = Ops.Shrink) then
          found := true
      end) slice;
  if !found then
    Some (U.store ~dst:target ~value:(U.contiguous ~src:value ()) ())
  else None

let flat_storage a =
  let dims = U.max_shard_shape a in
  let max_dims = List.map U.const_int dims in
  let shape_arg = function [d] -> d | ds -> U.stack ds in
  let a = match U.op a, U.src a with
    | Ops.Shrink, [|src; offset; _|]
      when List.equal U.equal (U.shape src) max_dims
           && List.for_all (fun d -> U.const_int_value d = Some 0) (U.as_shape offset) -> src
    | _ -> a in
  let size = List.fold_left (fun n dim -> Bound.mul n (Bound.int dim)) Bound.one dims |> Bound.to_int in
  let a = if List.equal U.equal (U.shape a) max_dims then a
    else U.pad ~src:a ~offset:(shape_arg (List.map (fun _ -> int_ 0) dims))
        ~size:(shape_arg max_dims) in
  size, U.reshape ~src:a ~shape:(int_ size)

let inline_call n =
  match U.as_call n with
  | Some {body; args; info} when not info.precompile && U.op body = Ops.Sink
      && Option.is_none (U.as_kernel_info body) ->
      let nodes = U.toposort ~enter_calls:false body in
      let mappings = List.filter_map (fun p ->
          match U.as_param p with
          | Some {param; _} when param.slot >= 0 ->
              if param.slot >= List.length args then
                invalid_arg "Prepare.inline_call: missing argument";
              let a = List.nth args param.slot in
              if not (Dtype.equal (U.dtype p) (U.dtype a)) then
                invalid_arg "Prepare.inline_call: argument dtype mismatch";
              let a = match param.size with
                | Some expected ->
                    let size, flat = flat_storage a in
                    if size <> expected then invalid_arg "Prepare.inline_call: argument capacity mismatch";
                    flat
                | None ->
                    if U.shape a <> [] then invalid_arg "Prepare.inline_call: expected scalar argument";
                    a in
              Some (p, a)
          | _ when U.op p = Ops.Alloc ->
              let arg = Option.get (U.Arg.as_param_arg (U.arg p)) in
              Some (p, U.replace p ~arg:(U.Arg.Param_arg {arg with slot = U.fresh_buffer_slot ()}) ())
          | _ -> None) nodes in
      Some (U.substitute ~walk:true mappings body)
  | _ -> None

let returned_after n =
  match U.op n, U.src n with
  | Ops.After, [|result; effects|] when U.op effects = Ops.Sink ->
      let stores = List.filter (fun st -> match U.as_store st with
          | Some {dst; _} -> U.base dst == U.base result
          | None -> false) (U.children effects) in
      (match stores with
       | [store] ->
           if U.op (U.base result) = Ops.Param then Some (U.after ~src:result ~deps:[store])
           else Some (Option.get (U.as_store store)).value
       | _ -> None)
  | _ -> None

let forward_call_outputs sink =
  let placed = U.Ref_tbl.create 16 in
  let rec peel u = if U.op u = Ops.After then peel (U.src u).(0) else u in
  let items = List.map (fun item ->
      let store = match U.op item, U.src item with
        | Ops.After, [|target; st|] when U.op st = Ops.Store && (U.src st).(0) == target -> st
        | _ -> item in
      match U.as_store store with
      | Some {dst = target; value; gate = None} ->
          let src = peel value in
          let base = U.storage_base src in
          let key = if U.op base = Ops.Alloc then base else src in
          if (item != store && U.op base <> Ops.Alloc)
             || U.Ref_tbl.mem placed key
             || List.exists (( == ) (U.storage_base target)) (U.toposort ~enter_calls:false value)
          then item
          else begin
            let replacement =
              if U.op base = Ops.Alloc && U.has_buffer_identity src
                 && U.has_buffer_identity target
                 && U.max_numel base = U.max_numel (U.storage_base target)
              then Some (U.storage_base target)
              else if U.op src = Ops.Stage && U.arg src = U.Arg.Empty then
                Some (U.after ~src:target ~deps:[U.store ~dst:target ~value:(U.src src).(0) ()])
              else if (U.op src = Ops.Buffer || U.op src = Ops.Unshard)
                      && U.has_buffer_identity src && U.has_buffer_identity target then Some target
              else None in
            match replacement with
            | Some replacement ->
                U.Ref_tbl.add placed key replacement;
                if item != store then U.Ref_tbl.add placed item value;
                value
            | None -> U.after ~src:target ~deps:[store]
          end
      | _ -> item) (U.children sink) in
  let mappings = U.Ref_tbl.fold (fun key value mappings -> (key, value) :: mappings) placed [] in
  U.substitute ~walk:true mappings (U.sink items)

let rec push_movement node rngs =
  match U.op node with
  | Ops.Reshape | Ops.Expand | Ops.Pad | Ops.Shrink | Ops.Permute | Ops.Flip ->
      push_movement (src0 node)
        (Indexing.apply_movement_op ~shapes:shape_of node rngs)
  | _ -> (node, rngs)

let live_axes rngs =
  List.concat_map (fun r ->
      List.filter_map (fun x ->
          Option.map (fun (v : U.range_view) -> v.axis) (U.as_range x))
        (r :: U.backward_slice r))
    rngs

let axis_ranges sh =
  List.mapi (fun i s ->
      if s > 1 then U.range ~size:(int_ s) ~axis:i ~kind:Axis_type.Weak ()
      else int_ 0) sh

let detect_expanded src =
  let sh = Option.value (shape_of src) ~default:[] in
  let n = List.length sh in
  if n = 0 then []
  else
    let live = live_axes (snd (push_movement src (axis_ranges sh))) in
    List.init n (fun i -> not (List.mem i live))

(* One-hot sum

   No tinygrad counterpart. A gather is a sum over
   [where (index = arange) x 0], and the reduce that sums it collapses to one
   gated load once the kernel is lowered. Splitting that reduce first puts a
   buffer between the two halves, and neither half collapses: the gather then
   costs a pass over the table. [is_one_hot_sum] recognises the shape so the
   split leaves it whole. *)

let const_view u = Option.map Const.view (U.as_const u)

let is_not c =
  U.op c = Ops.Cmpne && const_view (U.src c).(1) = Some (Const.Bool true)

let is_zero_const u =
  match const_view u with
  | Some (Const.Int n) -> Z.equal n Z.zero
  | Some (Const.Float f) -> Float.equal f 0.0
  | _ -> false

(* [rngs] indexes [parent]; the result indexes [child], an operand that
   [parent] broadcasts. *)
let operand_rngs ~parent ~child rngs =
  match shape_of parent, shape_of child with
  | Some psh, Some csh ->
      let nleft = List.length psh - List.length csh in
      if nleft < 0 then None
      else
        Some
          (List.filteri (fun j _ -> j >= nleft) rngs
           |> List.mapi (fun j r ->
                if List.nth csh j = 1 && List.nth psh (j + nleft) <> 1
                then U.const_like r 0
                else r))
  | _ -> None

let is_one_hot_sum ~src ~op ~num_axes =
  op = Ops.Add
  &&
  match shape_of src with
  | None -> false
  | Some sh ->
      let where, rngs = push_movement src (axis_ranges sh) in
      U.op where = Ops.Where
      &&
      let srcs = U.src where in
      (is_zero_const (U.base srcs.(1)) || is_zero_const (U.base srcs.(2)))
      &&
      let rec condition parent rngs child =
        match operand_rngs ~parent ~child rngs with
        | Some rngs when is_not child -> condition child rngs (src0 child)
        | Some rngs -> Some (child, rngs)
        | None -> None
      in
      match condition where rngs srcs.(0) with
      | Some (cond, cond_rngs)
        when (U.op cond = Ops.Cmpne || U.op cond = Ops.Cmpeq)
             && Dtype.is_int (U.dtype (src0 cond)) ->
          let live operand =
            match operand_rngs ~parent:cond ~child:operand cond_rngs with
            | Some rngs -> Some (live_axes (snd (push_movement operand rngs)))
            | None -> None
          in
          let reduced a = a < num_axes in
          let one_hot a b =
            a <> [] && List.for_all reduced a
            && not (List.exists reduced b)
          in
          (match live (U.src cond).(0), live (U.src cond).(1) with
           | Some a, Some b -> one_hot a b || one_hot b a
           | _ -> false)
      | _ -> false

let pow2 n =
  if n < 0 then 1 else
    let rec loop acc i = if i = 0 then acc else loop (acc * 2) (i - 1) in
    loop 1 n

let range_down from_ until =
  let rec loop acc n = if n < until then List.rev acc else loop (n :: acc) (n - 1) in
  loop [] from_

let split_reduceop_rule n =
  match U.as_reduce n with
  | Some { src; op; num_axes; _ } when num_axes > 0 ->
      (match shape_of src, shape_of n with
       | Some in_shape, Some out_shape
         when prod out_shape <> 0
              && getv Helpers.split_reduceop <> 0
              && prod in_shape / max 1 (prod out_shape)
                 >= getv Helpers.reduceop_split_threshold
              && not (is_one_hot_sum ~src ~op ~num_axes) ->
           let expanded = detect_expanded src in
           let cap =
             min 256
               (pow2 (getv Helpers.reduceop_split_size)
                / max 1 (prod out_shape))
           in
           (* Reduced axes are permuted to the front, so they are exactly the
              first [num_axes] axes of [src]. *)
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
               (List.init num_axes Fun.id)
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
                    ~src:
                      (U.reduce_axis ~src:splitted ~op
                         ~axes:(List.init num_axes Fun.id))
                    ()
                in
                let second_axis = List.length out_shape in
                let second =
                  U.reduce_axis ~src:first ~op ~axes:[ second_axis ]
                in
                Some (U.reshape ~src:second ~shape:(shape_node out_shape)))
       | _ -> None)
  | _ -> None

let identity_of op dtv = match op with
  | Ops.Add -> Const.zero dtv
  | Ops.Mul -> Const.one dtv
  | Ops.Max -> Const.min_value dtv
  | _ -> Const.zero dtv

(* tinygrad/schedule/prepare.py: repack the trailing dimension through
   unsigned integer lanes before scalar indexing fixes each element's width. *)
let expand_bitcast bc =
  if U.op bc <> Ops.Bitcast then None
  else
    let x = src0 bc in
    let os = Dtype.itemsize (U.dtype x) in
    let ns = Dtype.itemsize (U.dtype bc) in
    if os = ns || U.on_disk x then None
    else
      let uint = function
        | 1 -> Dtype.uint8 | 2 -> Dtype.uint16
        | 4 -> Dtype.uint32 | 8 -> Dtype.uint64
        | _ -> invalid_arg "Prepare.expand_bitcast: unsupported element width"
      in
      let dims = function [ d ] -> d | ds -> U.stack ds in
      let reshape x shape = U.reshape ~src:x ~shape:(dims shape) in
      let shape = U.shape x in
      if shape = [] then
        invalid_arg "Prepare.expand_bitcast: size-changing bitcast needs an axis";
      let target = U.shape bc in
      let tmp = U.bitcast ~src:x ~dtype:(uint os) in
      let repacked =
        if ns > os then begin
          let rate = ns / os in
          let tmp = reshape tmp (target @ [ int_ rate ]) in
          let prefix = List.map (fun _ -> int_ 0) target in
          let parts =
            List.init rate (fun i ->
                let part = U.shrink ~src:tmp
                    ~offset:(dims (prefix @ [ int_ i ]))
                    ~size:(dims (target @ [ int_ 1 ])) in
                U.alu_binary ~op:Ops.Shl
                  ~lhs:(U.cast ~src:part ~dtype:(uint ns))
                  ~rhs:(int_ (8 * i * os)))
          in
          match parts with
          | first :: rest ->
              reshape (List.fold_left (fun lhs rhs ->
                  U.alu_binary ~op:Ops.Add ~lhs ~rhs) first rest) target
          | [] -> assert false
        end else begin
          let parts = List.init (os / ns) (fun i ->
              U.alu_binary ~op:Ops.Shr ~lhs:tmp ~rhs:(int_ (8 * i * ns))) in
          let order = List.init (List.length shape) (fun i -> i + 1) @ [ 0 ] in
          U.cast ~dtype:(uint ns)
            ~src:(reshape (U.permute ~src:(U.stack parts) ~order) target)
        end
      in
      Some (U.bitcast ~src:repacked ~dtype:(U.dtype bc))

(* Copies to stores *)

let dims_node = function [ d ] -> d | ds -> U.stack ds

(* Reshapes as the tensor layer builds them: an identity reshape is the
   value itself, so a view shared between consumers stays one node. *)
let reshape_to u dims =
  if List.equal U.equal (U.shape u) dims then u
  else U.reshape ~src:u ~shape:(dims_node dims)

let convert_copy_to_store copy =
  let input = src0 copy in
  let dims = U.shape input and max_dims = List.map int_ (U.max_shape input) in
  let input = if List.equal U.equal dims max_dims then input else
      U.pad ~src:input ~offset:(dims_node (List.map (fun _ -> int_ 0) dims))
        ~size:(dims_node max_dims) in
  let device = U.Arg.as_device (U.arg copy) in
  let buffer = U.alloc ~slot:(U.fresh_buffer_slot ()) ~dtype:(U.dtype copy)
      ~shape:(int_ (U.max_numel input)) ?device () in
  let buffer = reshape_to buffer max_dims in
  let stored = U.after ~src:buffer ~deps:[U.store ~dst:buffer ~value:input ()] in
  Some (if List.equal U.equal dims max_dims then stored else
      U.shrink ~src:stored ~offset:(dims_node (List.map (fun _ -> int_ 0) dims))
        ~size:(dims_node dims))

let stage_to_store stage =
  let input = src0 stage in
  if U.has_buffer_identity ~after_ok:true input || U.op input = Ops.Copy then Some input
  else
    let dims = U.shape stage in
    let max_dims = List.map int_ (U.max_shape stage) in
    let buffer = U.alloc ~slot:(U.fresh_buffer_slot ()) ~dtype:(U.dtype stage)
        ~shape:(dims_node max_dims) ?device:(U.device_of input) () in
    let view = if List.equal U.equal dims max_dims then buffer else
        U.shrink ~src:buffer ~offset:(dims_node (List.map (fun _ -> int_ 0) dims))
          ~size:(dims_node dims) in
    Some (U.after ~src:view ~deps:[U.store ~dst:view ~value:input ()])

(* Storage a copy can write or read in place: a buffer, or a contiguous
   window of one (a STAGE counts as the buffer it becomes), which the copy
   reaches as a byte view. The tinygrad counterpart takes whole buffers only:
   it stages a window it reads, and a window it writes gets a staging buffer
   that a kernel copies again.

   Rule 3 below stages a cross-device value that is not such storage, and
   the stage it adds counts as storage, so it does not fire twice on one
   value. [stage_to_store] later turns a symbolic stage into a SHRINK of a
   fresh buffer, which is not a window when an inner axis is symbolic, and a
   store of it could match rule 3 again. None arises: a cross-device store
   comes from [convert_copy_to_store], which pads its value to the maximum
   shape, or from rule 1, whose destination is a window of the value's own
   shape and so has no symbolic inner axis either. *)
let storage_view value = Option.is_some (U.storage_window value)

let materialize n =
  match U.op n with
  | Ops.Store -> (
      match U.as_store n with
      | Some { dst; value; gate = None }
        when U.op value = Ops.Copy && U.device_of dst = U.device_of value
          && storage_view dst ->
          Some (U.store ~dst ~value:(src0 value) ())
      | Some { dst; value; gate = None }
        when U.op dst = Ops.Reshape && U.op value = Ops.Reshape
          && List.equal U.equal (U.shape (src0 dst)) (U.shape (src0 value)) ->
          Some (U.store ~dst:(src0 dst) ~value:(src0 value) ())
      | Some { dst; value; gate = None }
        when Option.is_some (U.device_of value)
          && U.device_of dst <> U.device_of value
          && not (storage_view value) ->
          Some (U.store ~dst ~value:(U.contiguous ~src:value ()) ())
      | _ -> None)
  | Ops.Copy -> convert_copy_to_store n
  | Ops.Stage when U.arg n = U.Arg.Empty -> stage_to_store n
  | _ -> None

let disk_copy n =
  match U.op n, U.src n with
  | Ops.Copy, [|stage|] when U.op stage = Ops.Stage && U.arg stage = U.Arg.Empty ->
      let input = src0 stage in
      if is_movement input && U.on_disk input then
        Some (U.replace n ~src:[|input|] ()) else None
  | Ops.Copy, [|input|] when is_movement input && U.on_disk input ->
      let moved = Array.copy (U.src input) in
      moved.(0) <- U.replace n ~src:[|src0 input|] ();
      Some (U.replace input ~src:moved ())
  | _ -> None

let earliest_rewrites =
  let shaped_const n value =
    let dims = match U.shape n with [ d ] -> d | ds -> U.stack ds in
    U.expand ~src:(U.const value) ~dims
  in
  U.first_match
    [ pm_mop_through_index;
      pm_mop_past_after; pm_mop_past_end;
      Upat.Pattern_matcher.rewrite Movement.mop_cleanup;
      (fun n -> match U.as_allreduce n with
         | Some { src; device; op } ->
             Allreduce.create_allreduce_function src ~device ~op
         | None -> None);
      split_reduceop_rule;
      (fun n -> match U.op n with
         | Ops.Detach | Ops.Contiguous_backward -> Some (src0 n)
         | _ -> None);
      (fun n -> match U.op n with
         | Ops.Copy ->
             let s = src0 n in
             (match U.device_of s, U.device_of n with
              | Some d1, Some d2 when d1 = d2 ->
                  Some s
              | _ -> None)
         | _ -> None);
      materialize;
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
      (* Two STOREs of the same value into the same buffer: keep the first. *)
      (fun n -> match U.op n with
         | Ops.After -> (
             match src_tail n with
             | [ store2 ] ->
                 let a1 = src0 n in
                 if U.op a1 <> Ops.After then None
                 else
                   (match U.as_store store2, src_tail a1 with
                    | Some { dst = d2; value = v2; _ }, [ store1 ]
                      when d2 == a1 ->
                        (match U.as_store store1 with
                         | Some { dst = d1; value = v1; _ }
                           when d1 == src0 a1 && v1 == v2 -> Some a1
                         | _ -> None)
                    | _ -> None)
             | _ -> None)
         | _ -> None);
      (* A buffer storing its own already-stored contents back into itself. *)
      (fun n -> match U.op n with
         | Ops.After -> (
             match src_tail n with
             | [ store ] ->
                 let buf = src0 n in
                 (match U.as_store store with
                  | Some { dst; value = a1; _ }
                    when dst == buf && U.op a1 = Ops.After && src0 a1 == buf ->
                      (match src_tail a1 with
                       | [ store1 ] ->
                           (match U.as_store store1 with
                            | Some { dst = d1; _ } when d1 == buf -> Some a1
                            | _ -> None)
                       | _ -> None)
                  | _ -> None)
             | _ -> None)
         | _ -> None);
      (fun n -> match U.as_store n with
         | Some { dst; value; _ } when U.op dst = Ops.Bitcast ->
             let inner = src0 dst in
             Some (U.store ~dst:inner
                     ~value:(U.bitcast ~src:value ~dtype:(U.dtype inner))
                     ())
         | _ -> None);
      expand_bitcast;
      (fun n -> match U.as_reduce n with
         | Some { src; op; _ } ->
             (match shape_of src, shape_of n with
              | Some s, Some t when List.mem 0 s && not (List.mem 0 t) ->
                  Some (shaped_const n (identity_of op (U.dtype n)))
              | _ -> None)
         | None -> None);
      (fun n ->
        if U.op n = Ops.Sink then None
        else match shape_of n with
          | Some s when List.mem 0 s ->
              Some (shaped_const n (Const.zero (U.dtype n)))
          | _ -> None);
    ]

let prepare_rangeify root =
  let root = forward_call_outputs root in
  (* The tinygrad counterpart forwards outputs only before multi_pm. The
     collectives multi_pm lowers allocate their results, so outputs are
     forwarded again: a realized collective writes the result's storage
     instead of an allocation it then copies. *)
  let root = forward_call_outputs (U.graph_rewrite ~name:"multi_pm" Multi.multi_pm root) in
  let root = U.graph_rewrite ~name:"inline calls"
      (U.first_match [movement_ops; inline_call; returned_after; disk_copy]) root in
  let root =
    if getv Helpers.openpilot_hacks = 0 then root
    else
      let ctx = U.Ref_tbl.create 16 in
      U.graph_rewrite ~name:"fold moved afters" (pm_fold_moved_after ctx) root
  in
  let root =
    U.graph_rewrite ~bottom_up:true ~name:"earliest rewrites"
      earliest_rewrites root
  in
  root
