(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops

let rule = Pattern_matcher.rule
let rule_ctx = Pattern_matcher.rule_ctx
let with_ctx = Pattern_matcher.with_ctx
let ops = Op.Set.of_list
let var = Upat.var
let is_range_of_size u = op u = Op.Range && op (nth u 0) <> Op.Const

let opts u =
  match arg u with
  | Bufferize o -> o
  | _ -> invalid_arg "a stage needs its buffer's options"

let compare_range r0 r1 =
  match List.compare Int.compare (axis_id r0) (axis_id r1) with
  | 0 -> Axis_type.compare (axis_type r0) (axis_type r1)
  | c -> c

let sorted_ranges rs = List.sort_uniq compare_range rs
let is_tagged u = Option.equal Tag.equal (tag u) (Some (Tag.Tuple []))

let rec zip l0 l1 =
  match (l0, l1) with x :: r0, y :: r1 -> (x, y) :: zip r0 r1 | _ -> []

(* Cleanups *)

let always_run u = op u = Op.Noop
let transcendental = ops Op.[ Exp2; Log2; Sin; Pow ]

(* Whether an axis dies is only known once an expand to its left is seen. *)
let cleanup_dead_axes b =
  let value = nth b 0 in
  (* An after is storage: its ranges say how consumers read it. *)
  if (opts b).keep = Whole || always_run value || op value = Op.After then None
  else
    let axes = zip (shape b) (List.tl (src b)) in
    let dead rng =
      op rng = Op.Const
      || (op rng = Op.Range && not (Nodes.mem rng (ranges value)))
    in
    if List.exists (fun (_, r) -> is_range_of_size r) axes then None
    else if not (List.exists (fun (_, r) -> dead r) axes) then None
    else
      let live = List.filter (fun (_, r) -> not (dead r)) axes in
      let reshape = List.map (fun (s, r) -> if dead r then Int 1 else s) axes in
      Some
        (expand
           (Ops.reshape (replace b ~src:(value :: List.map snd live)) reshape)
           (shape b))

let pm_gate_substitute =
  Pattern_matcher.v
    (fun () -> [
      rule_ctx (Upat.v ~op:Op.Set.all ~name:"b" ()) (fun ctx m ->
          let rs = ranges (m "b") in
          if Tbl.to_seq_keys ctx |> Seq.exists (fun r -> Nodes.mem r rs) then
            None
          else raise Bottom_up_gate);
    ])

(* A buffer stored only to be read through movements is removed: the indices of
   the read are expressed in terms of the stored value's. *)
let remove_bufferize src buf idx =
  if List.compare_lengths (Ops.src buf) (Ops.src idx) <> 0 then
    invalid_arg "an index of a stage has one index per range";
  if
    not
      (List.for_all
         (fun x -> List.mem (op x) Op.[ Range; Const ])
         (List.tl (Ops.src buf)))
  then invalid_arg "a stage's ranges are ranges or constants";
  (* A user's materialisation is never removed. *)
  if always_run src || (opts buf).keep = Whole then None
  else
    (* The cost: the buffers the value reads, and whether a reduction reads
       one. *)
    let accessed = Tbl.create 8 in
    let access u = Tbl.replace accessed u () in
    let red_gate x =
      match op x with
      | Op.After ->
          access (buf_uop x);
          false
      | Op.Stage when (opts x).addrspace = Dtype.Global ->
          access x;
          false
      | Op.Mstack ->
          access x;
          false
      (* Stores do not count as buffer accesses. *)
      | Op.Store -> false
      | Op.Param ->
          access x;
          true
      | _ -> true
    in
    let computed = toposort ~gate:red_gate src in
    let reduces = List.filter (fun x -> op x = Op.Reduce) computed in
    (* Inlined where it is broadcast, the value would be computed again for each
       element of the ranges it does not vary along: one whose computing runs a
       transcendental function stays stored. Its producers are already stored
       or inlined, so the walk sees what inlining would compute. *)
    let recomputes =
      (opts buf).keep = Broadcast
      && List.exists (fun x -> Op.Set.mem (op x) transcendental) computed
    in
    if Tbl.length accessed > 3 || recomputes then None
    else
      let reads_buffer x = List.mem (op x) Op.[ Param; Stage; After ] in
      if
        List.exists reads_buffer
          (toposort (sink (List.map (fun r -> nth r 0) reduces)))
      then None
      else
        (* A constant range is not replaced, nor is a range read by a dead
           load. *)
        let replaced =
          List.filter
            (fun (k, v) ->
              op k <> Op.Const && not (op v = Op.Const && is_invalid v))
            (zip (List.tl (Ops.src buf)) (List.tl (Ops.src idx)))
        in
        Some (substitute ~extra_pm:pm_gate_substitute src replaced)

let remove_noop_bufferize idx b2 =
  if not (List.equal ( == ) (List.tl (src idx)) (List.tl (src b2))) then None
  else
    match shape b2 with
    | [] -> Some (nth idx 0)
    | s -> Some (shrink (nth idx 0) (List.map (fun s -> Some (Int 0, s)) s))

let after_all_invalid after =
  let buf = buf_uop (nth after 0) in
  let all_invalid s =
    op s = Op.End
    &&
    let st = nth s 0 in
    let ended = ended_ranges s in
    op st = Op.Store
    && is_invalid (base (nth st 1))
    && buf_uop (nth st 0) == buf
    && List.for_all (fun r -> Nodes.mem r (ranges (nth st 0))) ended
    && resolve ~default:false
         (eq
            (List.fold_left (fun p r -> mul p (nth r 0)) (int 1) ended)
            (sint_to_uop (numel buf)))
  in
  List.for_all all_invalid (List.tl (src after))

let pm_const_buffer_folding =
  Pattern_matcher.append (with_ctx Prepare.pm_mops)
    (Pattern_matcher.v
       (fun () -> [
         rule (Upat.op Op.Stage ~name:"b") (fun m -> cleanup_dead_axes (m "b"));
         (* A stage of an index by the stage's own ranges is the storage. *)
         rule
           (Upat.f
              (Upat.op Op.Index ~name:"idx")
              Op.Stage ~allow_any_len:true ~name:"b2")
           (fun m -> remove_noop_bufferize (m "idx") (m "b2"));
         (* A constant needs no buffer, in either spelling. *)
         rule
           (Upat.f
              (Upat.or_casted (Upat.cvar "c"))
              Op.Stage ~allow_any_len:true ~name:"b")
           (fun m -> Some (const_like (m "b") (value (m "c"))));
         (* An index of a constant is the constant. *)
         rule
           (Upat.op Op.Index ~src:[ Upat.or_casted ~name:"c" (Upat.cvar "k") ])
           (fun m -> Some (m "c"));
         (* An index of storage whose every store is invalid is invalid. *)
         rule
           (Upat.op Op.Index ~name:"idx" ~allow_any_len:true
              ~src:[ Upat.op Op.After ~name:"after" ])
           (fun m ->
             if after_all_invalid (m "after") then
               Some (const_like (m "idx") `Invalid)
             else None);
         (* A stack's source without a device is the same value on every device,
            so an index of the stack is an index of that value. *)
         rule
           (Upat.f
              (Upat.op Op.Mstack ~allow_any_len:true ~src:[ var "s" ])
              Op.Index ~allow_any_len:true ~name:"idx")
           (fun m ->
             let s = m "s" and idx = m "idx" in
             if Option.is_none (device s) then
               Some (replace idx ~src:(s :: List.tl (src idx)))
             else None);
       ]))

let pm_remove_bufferize =
  Pattern_matcher.v
    (fun () -> [
      rule
        (Upat.f
           (Upat.f (var "src") Op.Stage ~allow_any_len:true ~name:"buf")
           Op.Index ~allow_any_len:true ~name:"idx")
        (fun m -> remove_bufferize (m "src") (m "buf") (m "idx"));
      (* A store of a value into itself does nothing. *)
      rule (Upat.store (var "x") [ var "x" ]) (fun _ -> Some (v Op.Noop));
      rule
        (Upat.op Op.End ~allow_any_len:true ~src:[ Upat.op Op.Noop ~name:"x" ])
        (fun m -> Some (m "x"));
    ])

let strip_zero_offset_shrink x =
  match op x with
  | Op.Shrink -> (
      match marg x with
      | Shrink b when List.for_all (fun (s, _) -> Sint.equal s (Int 0)) b ->
          nth x 0
      | _ -> x)
  | _ -> x

(* A call's arguments that have consumers can be indexed; the call reads its
   storage. An empty argument of a precompiled call is the constant prepare made
   of it: the body reaches no element of it, and the call passes the scalar. *)
let no_indexing_calls u =
  let precompiled = match arg u with Call c -> c.precompile | _ -> false in
  let arg x =
    match op x with
    | Op.Index -> nth x 0
    | Op.Shrink -> strip_zero_offset_shrink x
    | Op.Mstack -> replace x ~src:(List.map strip_zero_offset_shrink (src x))
    | (Op.Expand | Op.Reshape)
      when precompiled && Sint.equal (numel x) (Int 0) ->
        base x
    | _ -> x
  in
  replace u ~src:(List.map arg (src u))

let pm_no_indexing_calls =
  Pattern_matcher.v
    (fun () -> [
      rule (Upat.op Op.Call ~name:"u") (fun m ->
          Some (no_indexing_calls (m "u")));
    ])

let loop_range r = Axis_type.equal (axis_type r) Axis_type.Loop

(* The kernel graph is what runs: it has no views, and a value's storage is the
   storage. A view that moves with a range is what a call in that range reads on
   each trip, and stays. *)
let pm_no_views =
  Pattern_matcher.v
    (fun () -> [
      rule
        (Upat.v
           ~op:(ops Op.[ Reshape; Shrink ])
           ~name:"v" ~allow_any_len:true
           ~src:
             [
               Upat.v
                 ~op:(ops Op.[ After; Param; Unshard; Mstack; Buffer; Alloc ])
                 ();
             ]
           ())
        (fun m ->
          let v = m "v" in
          if List.exists loop_range (Nodes.to_list (ranges v)) then None
          else Some (nth v 0));
      (* A loop's range, outside every kernel, loses the tag that kernels
         renumber their ranges by. *)
      rule (Upat.op Op.Range ~name:"r") (fun m ->
          let r = m "r" in
          if is_tagged r && loop_range r then Some (replace r ~tag:None)
          else None);
    ])

module Bufs = Set.Make (struct
  type t = Ops.t

  let compare = Ops.compare
end)

type limit_bufs_ctx = { buf_cache : Bufs.t Tbl.t; mutable range_idx : int }

let limit_bufs ctx root =
  let max_bufs = Helpers.Context_var.value Helpers.max_kernel_buffers in
  (* Without a device, the node computes indices. *)
  if Option.is_none (device root) || max_bufs = 0 then None
  else
    let visitor u =
      match (op u, src u) with
      | (Op.Stage | Op.After | Op.Param | Op.Mselect | Op.Mstack), _ ->
          Bufs.singleton u
      | _, [ s ] -> Tbl.find ctx.buf_cache s
      | _, srcs ->
          List.fold_left
            (fun acc s -> Bufs.union acc (Tbl.find ctx.buf_cache s))
            Bufs.empty srcs
    in
    (* One buffer is the output. *)
    if Bufs.cardinal (topovisit root visitor ctx.buf_cache) <= max_bufs - 1 then
      None
    else
      let each s =
        if
          not (Op.Set.mem (op s) Op.Set.elementwise && Option.is_some (device s))
        then s
        else
          (* The value is stored first: its reductions are weak ranges of the
             stage, and the device range stays a launched axis. *)
          let orig = Nodes.to_list (ranges s) in
          let fresh r =
            if
              op r = Op.Range
              && not (Axis_type.equal (axis_type r) Axis_type.Device)
            then begin
              let n = ctx.range_idx in
              ctx.range_idx <- n + 1;
              replace r
                ~arg:(Range { axis_id = [ n ]; axis_type = Axis_type.Weak })
            end
            else r
          in
          let ends = List.map fresh orig in
          let opts =
            { device = device s; addrspace = Dtype.Global; keep = Removable }
          in
          index
            (bufferize ~opts (substitute s (List.combine orig ends)) ends)
            orig
      in
      Some (replace root ~src:(List.map each (src root)))

let pm_limit_bufs =
  Pattern_matcher.v
    (fun () -> [
      rule_ctx
        (Upat.v ~op:(Op.Set.union Op.Set.binary Op.Set.ternary) ~name:"root" ())
        (fun ctx m -> limit_bufs ctx (m "root"));
    ])

(* Buffers *)

let bufferize_to_store ctx x idx =
  let dtype = commit_dtype x in
  let rngs = sorted_ranges (Nodes.to_list (ranges idx)) in
  (match numel x with
  | Int n when (n > 0) [@mutate off "Prepare removes empty values"] -> ()
  | _ -> invalid_arg "a stage has no elements or a symbolic size");
  let value = nth x 0 in
  if op value = Op.After then
    (* The stores of the storage end over the stage's ranges, and the storage is
       read after them. *)
    let buf = base (buf_uop (nth value 0)) in
    let stores =
      List.filter
        (fun s -> op s = Op.Store && op (nth s 0) = Op.Index)
        (List.tl (src value))
    in
    if stores = [] then Some buf
    else
      let end_store st =
        let target = nth st 0 in
        (* A stage of an index stores through the underlying index. *)
        let target =
          if
            op (nth target 0) = Op.Stage && op (nth (nth target 0) 0) = Op.Index
          then nth (nth target 0) 0
          else target
        in
        if nth st 1 == target then None
        else
          let ends = sorted_ranges (Nodes.to_list (ranges target) @ rngs) in
          Some (end_ (store target (nth st 1)) ends)
      in
      Some (after buf (List.filter_map end_store stores))
  else
    match opts x with
    | { addrspace = Dtype.Global; device; _ } ->
        let slot = !ctx in
        incr ctx;
        let buf =
          v Op.Alloc ~src:(device_range_src device)
            ~arg:(Param (param_arg ~slot ~size:(max_numel x) ?device dtype))
        in
        let do_store =
          end_ (store (index buf [ idx ]) (cast (nth x 0) dtype)) rngs
        in
        Some (cast (after buf [ do_store ]) (Ops.dtype x))
    | _ -> None

(* A stage over several ranges is a stage over their flat index, reshaped. *)
let flatten_bufferize x =
  match src x with
  | [ _; _ ] -> None
  | value :: rngs ->
      let flat =
        Helpers.get_single_element
          (Indexing.apply_movement_op [ numel x ] (Reshape (shape x)) rngs)
      in
      let ret = reshape (replace x ~src:[ value; flat ]) (shape x) in
      if List.exists is_range_of_size rngs then
        let size r = if op r = Op.Const then Int 1 else Sym (nth r 0) in
        Some (shrink ret (List.map (fun r -> Some (Int 0, size r)) rngs))
      else Some ret
  | [] -> None

let rec is_noop_after_dep x =
  (op x = Op.Noop && src x = []) [@mutate off "a noop has no sources"]
  || (op x = Op.End && is_noop_after_dep (nth x 0))

let remove_noop_afters x =
  match src x with
  | first :: deps ->
      let kept = List.filter (fun s -> not (is_noop_after_dep s)) deps in
      if List.compare_lengths kept deps = 0 then None
      else if kept = [] then Some first
      else Some (replace x ~src:(first :: kept))
  | [] -> None

let pm_add_buffers =
  Pattern_matcher.concat
    [
      with_ctx Prepare.pm_mops;
      Pattern_matcher.v
        (fun () -> [
          rule (Upat.op Op.Stage ~name:"x") (fun m -> flatten_bufferize (m "x"));
        ]);
      Pattern_matcher.v
        (fun () -> [
          rule_ctx
            (Upat.op Op.Stage ~name:"x" ~src:[ Upat.wild; Upat.var "idx" ])
            (fun ctx m -> bufferize_to_store ctx (m "x") (m "idx"));
          (* An index of a buffer through the weak cast added above indexes the
             buffer and casts the value read. This must run in the rewrite that
             adds the cast, or the expander expands the whole cast buffer into
             one vector. *)
          rule
            (Upat.op Op.Index ~name:"u" ~allow_any_len:true
               ~src:[ Upat.op Op.Cast ~dtype:Dtype.weaks ~src:[ var "buf" ] ])
            (fun m ->
              let u = m "u" in
              Some
                (cast (replace u ~src:(m "buf" :: List.tl (src u))) (dtype u)));
          (* Reshapes move through shard selections and stacks. *)
          rule
            (Upat.v
               ~op:(ops Op.[ Mselect; Mstack ])
               ~name:"m" ~each:(Upat.op Op.Reshape) ())
            (fun m ->
              let m = m "m" in
              Some
                (reshape
                   (replace m ~src:(List.map (fun x -> base (nth x 0)) (src m)))
                   (shape m)));
          (* A kernel's arguments lose their reshapes. *)
          rule (Upat.op Op.Call ~name:"k") (fun m ->
              let k = m "k" in
              Some
                (replace k
                   ~src:
                     (List.map
                        (fun x -> if op x = Op.Reshape then nth x 0 else x)
                        (src k))));
          (* Invalid writes are dropped. *)
          rule
            (Upat.op Op.Store
               ~src:[ Upat.wild; Upat.op Op.Const ~arg:(Const `Invalid) ])
            (fun _ -> Some (v Op.Noop));
          rule (Upat.op Op.After ~name:"x") (fun m ->
              remove_noop_afters (m "x"));
        ]);
    ]

(* Scalar parameters keep their identity across the call boundary. *)
let pm_add_param_range_tags =
  Pattern_matcher.v
    (fun () -> [
      rule
        (Upat.v ~op:(ops Op.[ Param; Range ]) ~name:"x" ())
        (fun m ->
          let x = m "x" in
          if op x = Op.Param && addrspace x = Some Dtype.Alu then None
          else Some (rtag ~tag:(Tag.Tuple []) x));
    ])

(* Kernels *)

type local_ctx = {
  mutable dg : int;
  map : t Tbl.t;
  mutable order : t list; (* The keys of [map], last first. *)
  mutable range : int;
}

let add_arg ctx buf value =
  if not (Tbl.mem ctx.map buf) then begin
    Tbl.replace ctx.map buf value;
    ctx.order <- buf :: ctx.order
  end

let debuf ctx buf =
  let align, phase = storage_phase buf in
  let param =
    v Op.Param
      ~arg:
        (Param
           (param_arg ~slot:ctx.dg ~size:(max_numel buf)
              ~addrspace:(addrspace buf) ?device:(device buf) ~phase ~align
              (dtype buf)))
  in
  let ret = reshape param (List.map (fun n -> Int n) (max_shape buf)) in
  (* A buffer of symbolic shape is its greatest view, shrunk. *)
  let ret =
    if
      List.equal Sint.equal
        (List.map (fun n -> Int n) (max_shape buf))
        (shape buf)
    then ret
    else shrink ret (List.map (fun s -> Some (Int 0, s)) (shape buf))
  in
  add_arg ctx buf buf;
  ctx.dg <- ctx.dg + 1;
  ret

let handle_after ctx after =
  if addrspace after = Some Dtype.Local then None
  else
    let buf = buf_uop after in
    (* Bottom up, so it is added once. *)
    add_arg ctx buf after;
    Some buf

(* Ranges are renumbered from 0, so equal kernels dedupe. *)
let renumber_range ctx r =
  if not (is_tagged r) then None
  else
    let ret =
      replace r ~tag:None
        ~arg:(Range { axis_id = [ ctx.range ]; axis_type = axis_type r })
    in
    ctx.range <- ctx.range + 1;
    Some ret

let check_buf_states x =
  let idxs =
    List.filter
      (fun s -> op s = Op.Index)
      (toposort ~gate:(fun x -> op x <> Op.After) x)
  in
  let read_from = Tbl.create 8 in
  List.iter
    (fun idx ->
      let buf = buf_uop idx and state = nth idx 0 in
      if List.mem (op buf) Op.[ Buffer; Alloc; Param ] then
        match Tbl.find_opt read_from buf with
        | Some s when s != state ->
            invalid_arg
              (Format.asprintf "cycle detected while indexing %a" pp buf)
        | Some _ -> ()
        | None -> Tbl.replace read_from buf state)
    idxs

let to_define_global =
  Pattern_matcher.v
    (fun () -> [
      rule_ctx (Upat.op Op.Store ~name:"x") (fun _ m ->
          check_buf_states (m "x");
          None);
      rule_ctx
        (Upat.v ~op:(ops Op.[ Buffer; Alloc; Mstack; Mselect ]) ~name:"buf" ())
        (fun ctx m -> Some (debuf ctx (m "buf")));
      (* Only storage parameters get kernel-local slots; scalar parameters keep
         the slots of their enclosing call. *)
      rule_ctx (Upat.op Op.Param ~name:"buf") (fun ctx m ->
          let buf = m "buf" in
          if
            (not (is_tagged buf))
            || (addrspace buf = Some Dtype.Alu
               || shape_opt buf = None)
               [@mutate off "a scalar parameter has no shape"]
          then None
          else Some (debuf ctx buf));
      (* Scalar parameters are values, not buffers. *)
      rule_ctx
        (Upat.op Op.Index ~src:[ Upat.op Op.Param ~name:"v" ])
        (fun _ m ->
          if addrspace (m "v") = Some Dtype.Alu then Some (m "v") else None);
      rule_ctx (Upat.op Op.After ~name:"after") (fun ctx m ->
          handle_after ctx (m "after"));
      (* A local stage has no device. *)
      rule_ctx (Upat.op Op.Stage ~name:"b") (fun _ m ->
          let b = m "b" in
          Some (replace b ~arg:(Bufferize { (opts b) with device = None })));
      rule_ctx (Upat.op Op.Range ~name:"r") (fun ctx m ->
          renumber_range ctx (m "r"));
    ])

let split_store x =
  (* Open device ranges are bound per device at launch. A loop around a call
     runs that call, and is no kernel. *)
  if
    List.exists
      (fun r -> not (Axis_type.equal (axis_type r) Axis_type.Device))
      (Nodes.to_list (ranges x))
    || op x = Op.End
       && op (nth x 0) = Op.Call
       && List.exists loop_range (List.tl (src x))
  then None
  else
    let lctx = { dg = 0; map = Tbl.create 8; order = []; range = 0 } in
    let ret =
      graph_rewrite ~bottom_up:true ~ctx:lctx x
        (Pattern_matcher.append to_define_global
           (with_ctx Simplify.pm_flatten_range))
    in
    (* Buffers can be on different devices here: the schedule compiles such
       kernels to copies. *)
    let args = List.rev_map (Tbl.find lctx.map) lctx.order in
    Some (call (sink ~kernel:(kernel_info ()) [ ret ]) args)

let split_kernels =
  Pattern_matcher.v
    (fun () -> [
      rule
        (Upat.v ~op:(ops Op.[ Store; End ]) ~name:"x" ())
        (fun m -> split_store (m "x"));
    ])

let get_kernel_graph tsink =
  let setting = Helpers.Context_var.value in
  let tsink =
    Indexing.run_rangeify ~debug:(setting Helpers.debug_rangeify) tsink
  in
  (* Cleanups for speed and runnability. *)
  let tsink =
    graph_rewrite ~ctx:() tsink
      (Pattern_matcher.concat
         [
           Symbolic.symbolic;
           Simplify.pm_reduce_simplify;
           pm_const_buffer_folding;
           pm_remove_bufferize;
         ])
  in
  let ranges = List.filter (fun x -> op x = Op.Range) (toposort tsink) in
  let next_range =
    1 + List.fold_left (fun m r -> max m (List.hd (axis_id r))) (-1) ranges
  in
  let tsink =
    graph_rewrite
      ~ctx:{ buf_cache = Tbl.create 64; range_idx = next_range }
      tsink pm_limit_bufs
  in
  let slots =
    List.filter_map
      (fun x ->
        match (op x, arg x) with Op.Alloc, Param p -> Some p.slot | _ -> None)
      (toposort tsink)
  in
  let next_slot = ref (1 + List.fold_left max (-1) slots) in
  let tsink =
    graph_rewrite ~bottom_up:true ~ctx:next_slot tsink
      (Pattern_matcher.append pm_add_buffers (with_ctx pm_add_param_range_tags))
  in
  let tsink = graph_rewrite ~bottom_up:true ~ctx:() tsink split_kernels in
  let tsink = graph_rewrite ~ctx:() tsink pm_no_indexing_calls in
  let tsink = graph_rewrite ~ctx:() tsink pm_no_views in
  if setting Helpers.spec <> 0 then
    Spec.type_verify ~enter_calls:false Spec.kernel_graph tsink;
  tsink
