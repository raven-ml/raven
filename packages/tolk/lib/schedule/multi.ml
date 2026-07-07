(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Port of tinygrad/schedule/multi.py to the tolk_uop IR. *)

open Tolk_uop
module U = Uop

(* Helpers *)

let prod = List.fold_left ( * ) 1

let index_of x l =
  let rec loop i = function
    | [] -> invalid_arg "index_of: element not found"
    | y :: _ when y = x -> i
    | _ :: rest -> loop (i + 1) rest
  in
  loop 0 l

let int_ n = U.const (Const.int Dtype.Val.int32 n)

let allreduce_cast = Helpers.Context_var.int ~key:"ALLREDUCE_CAST" ~default:1

(* True iff [dt]'s scalar is bf16 or half — the dtypes tinygrad allreduces
   in their pre-cast (narrow) form to avoid precision loss. *)
let is_bf16_or_half dt =
  match dt with
  | Dtype.Val v -> (
      match Dtype.Val.scalar v with
      | Dtype.Float16 | Dtype.Bfloat16 -> true
      | _ -> false)
  | _ -> false

(* Shape encoding *)

let emit_shape = function
  | [ d ] -> int_ d
  | ds -> U.stack (List.map int_ ds)

let emit_pairs pairs =
  (emit_shape (List.map fst pairs), emit_shape (List.map snd pairs))

let emit_symbolic = function [ d ] -> d | ds -> U.stack ds

let sub_expr lhs rhs = U.alu_binary ~op:Ops.Sub ~lhs ~rhs

(* Decompose a stacked shape node into its per-axis scalars. *)
let shape_elems node =
  match U.op node with
  | Ops.Stack -> Array.to_list (U.src node)
  | _ -> [ node ]

let ndev_of devices node =
  match (devices node : U.device option) with
  | Some (Multi ds) -> List.length ds
  | _ -> 1

(* Partition [src] along [axis] using a symbolic device index. Each device
   takes its slice: [dnum * sz .. dnum * sz + sz). *)
let shard shape ndev src axis =
  let dim = List.nth shape axis in
  if dim mod ndev <> 0 then failwith "multi axis uneven";
  let sz = dim / ndev in
  let dnum =
    U.variable ~name:"_device_num" ~min_val:0 ~max_val:(ndev - 1)
      ~dtype:Dtype.Val.int32 ()
  in
  let off = U.alu_binary ~op:Ops.Mul ~lhs:dnum ~rhs:(int_ sz) in
  let before =
    List.mapi (fun i _ -> if i <> axis then int_ 0 else off) shape
  in
  let size =
    List.mapi (fun i s -> if i <> axis then int_ s else int_ sz) shape
  in
  U.shrink ~src ~offset:(emit_symbolic before) ~size:(emit_symbolic size)

(* Inverse of [shard]: pad each device's shard so it covers the full range,
   with zeros outside its slice. Summing across devices reconstructs the
   full tensor. *)
let unshard shape ndev src axis =
  let bsz = List.nth shape axis in
  let dnum =
    U.variable ~name:"_device_num" ~min_val:0 ~max_val:(ndev - 1)
      ~dtype:Dtype.Val.int32 ()
  in
  let off = U.alu_binary ~op:Ops.Mul ~lhs:(int_ bsz) ~rhs:dnum in
  let before =
    List.mapi (fun i _ -> if i <> axis then int_ 0 else off) shape
  in
  let size = List.map int_ shape in
  U.pad ~src ~offset:(emit_symbolic before) ~size:(emit_symbolic size)

(* Shape extraction: decode an int list from a shape node, when every
   element is an integer constant. *)
let extract_int_shape node =
  let elems = shape_elems node in
  let rec go acc = function
    | [] -> Some (List.rev acc)
    | x :: rest -> (
        match U.const_int_value x with
        | Some n -> go (n :: acc) rest
        | None -> None)
  in
  go [] elems

(* Walk the DAG collecting every [_device_num] symbolic parameter. *)
let device_num_vars node =
  U.find_nodes
    (fun u ->
      match U.Arg.as_param_arg (U.arg u) with
      | Some { name = Some "_device_num"; _ } -> U.op u = Ops.Param
      | Some _ -> false
      | None -> false)
    node

(* Substitute every [_device_num] variable in [node] with constant [i]. *)
let subst_device_num node i =
  match device_num_vars node with
  | [] -> node
  | dvars ->
      let mappings =
        List.map
          (fun dv ->
            let dt = Dtype.val_of (U.dtype dv) in
            (dv, U.const (Const.int dt i)))
          dvars
      in
      U.substitute mappings node

(* Predicates and light view helpers *)

let is_multi x = U.op x = Ops.Multi

let multi_axis x =
  match U.arg x with
  | U.Arg.Int a -> a
  | _ -> invalid_arg "multi_axis: not a Multi"

let unwrap_multi x = if is_multi x then (U.src x).(0) else x

(* For a MULTI node, return (inner, axis). *)
let inner_axis m = (U.src m).(0), multi_axis m

(* MSELECT/MSTACK rewrite *)

(* Move SHRINK before MSTACK: substitute [_device_num] with each device
   index and apply the shrink to each MSTACK element individually. *)
let mstack_early_shrink ms before after =
  let bs = shape_elems before and es = shape_elems after in
  let apply_shrink s i =
    let bs' = List.map (fun b -> subst_device_num b i) bs in
    let es' = List.map (fun e -> subst_device_num e i) es in
    let size = List.map2 sub_expr es' bs' in
    U.shrink ~src:s ~offset:(emit_symbolic bs') ~size:(emit_symbolic size)
  in
  let new_srcs =
    List.mapi
        (fun i x ->
          match U.op x with
          | Ops.Copy ->
            let device =
              match U.Arg.as_device (U.arg x) with
              | Some d -> d
              | None -> invalid_arg "mstack_early_shrink: copy without device"
            in
            U.copy ~src:(apply_shrink (U.src x).(0) i) ~device ()
        | _ -> U.contiguous ~src:(apply_shrink x i) ())
      (Array.to_list (U.src ms))
  in
  U.mstack new_srcs

(* BROADCAST: copy from single to multi-device -> per-device copies in MSTACK. *)
let broadcast_copy ~devices node =
  match U.op node with
  | Ops.Copy -> (
      let src = (U.src node).(0) in
      match (devices src : U.device option), U.Arg.as_device (U.arg node) with
      | Some (Single _), Some (Multi ds) ->
          let copies = List.map (fun d -> U.copy ~src ~device:(Single d) ()) ds in
          Some (U.mstack copies)
      | _ -> None)
  | _ -> None

(* COPY_TO_ONE: copy from multi-device to single -> select shard 0 and copy. *)
let copy_to_one ~devices node =
  match U.op node with
  | Ops.Copy -> (
      let src = (U.src node).(0) in
      match (devices src : U.device option), U.Arg.as_device (U.arg node) with
      | Some (Multi _), Some (Single _ as device) ->
          Some (U.copy ~src:(U.mselect ~src ~index:0) ~device ())
      | _ -> None)
  | _ -> None

(* MSELECT(MSTACK) -> direct indexing. *)
let mselect_mstack node =
  match U.op node with
  | Ops.Mselect when U.op (U.src node).(0) = Ops.Mstack ->
      let mstack = (U.src node).(0) in
      let index = match U.arg node with U.Arg.Int i -> i | _ -> 0 in
      Some (U.src mstack).(index)
  | _ -> None

(* MSELECT(movement(s)) -> movement(MSELECT(s)): push select inside. *)
let mselect_before_movement node =
  match U.op node with
  | Ops.Mselect when Ops.Group.is_movement (U.op (U.src node).(0)) ->
      let src = (U.src node).(0) in
      let index = match U.arg node with U.Arg.Int i -> i | _ -> 0 in
      let c = U.src src in
      let inner = U.mselect ~src:c.(0) ~index in
      (match U.op src, U.arg src with
      | Ops.Reshape, _ -> Some (U.reshape ~src:inner ~shape:c.(1))
      | Ops.Expand, _ -> Some (U.expand ~src:inner ~shape:c.(1))
      | Ops.Permute, U.Arg.Ints order ->
          Some (U.permute ~src:inner ~order)
      | Ops.Flip, U.Arg.Bools dims -> Some (U.flip ~src:inner ~dims)
      | Ops.Pad, _ -> Some (U.pad ~src:inner ~offset:c.(1) ~size:c.(2))
      | Ops.Shrink, _ -> Some (U.shrink ~src:inner ~offset:c.(1) ~size:c.(2))
      | _ -> None)
  | _ -> None

(* Multi functions *)

(* Gather a sharded MULTI tensor onto [device]: extract the inner shard,
   unshard it (symbolic pad per device), then allreduce-sum. *)
let copy_multi ~shapes ~devices multi device =
  let inner = (U.src multi).(0) in
  let axis = multi_axis multi in
  let inner_shape =
    match shapes inner with
    | Some sh -> sh
    | None -> failwith "copy_multi: unknown inner shape"
  in
  let ndev = ndev_of devices multi in
  match (device : U.device) with
  | Single _ ->
      (* Gather onto a single device: select each shard, copy it there, and
         concatenate along the shard axis. [cat] is pad-each-to-full + sum. *)
      let sz = List.nth inner_shape axis in
      let full =
        List.mapi (fun i s -> if i = axis then s * ndev else s) inner_shape
      in
      let full_node = emit_shape full in
      let pieces =
        List.init ndev (fun i ->
            let piece = U.copy ~src:(U.mselect ~src:inner ~index:i) ~device () in
            let offset =
              List.mapi (fun j _ -> if j = axis then i * sz else 0) inner_shape
            in
            U.pad ~src:piece ~offset:(emit_shape offset) ~size:full_node)
      in
      U.usum pieces
  | _ ->
      let unsharded = unshard inner_shape ndev inner axis in
      U.allreduce ~src:unsharded ~device ~op:Ops.Add

let alu_multi ~shapes ~devices root =
  let children = Array.to_list (U.src root) in
  (* Result shard axis: last deduped axis among MULTI sources. *)
  let dedup l =
    List.fold_left (fun acc x -> if List.mem x acc then acc else x :: acc) [] l
    |> List.rev
  in
  let axes =
    children
    |> List.filter_map (fun s -> if is_multi s then Some (multi_axis s) else None)
    |> dedup
  in
  match List.rev axes with
  | [] -> None
  | axis :: _ ->
      let multi_len s =
        match (devices s : U.device option) with
        | Some (Multi ds) -> Some (List.length ds)
        | _ -> None
      in
      let ndev =
        match List.find_map multi_len children with
        | Some n -> n
        | None -> failwith "alu_multi: no multi device"
      in
      let shape_exn s =
        match shapes s with
        | Some sh -> sh
        | None -> failwith "alu_multi: unknown shape"
      in
      (* Align each source to [axis]: unwrap matching MULTIs, shard
         non-sharded sources, gather-then-reshard mismatched ones. *)
      let aligned =
        List.map
          (fun s ->
            if is_multi s then
              let a = multi_axis s in
              if a = axis then (U.src s).(0)
              else
                let dev =
                  match devices s with
                  | Some d -> d
                  | None -> failwith "alu_multi: no device"
                in
                shard (shape_exn s) ndev (copy_multi ~shapes ~devices s dev) axis
            else shard (shape_exn s) ndev s axis)
          children
      in
      let result =
        match U.op root, aligned with
        | op, [ s ] when Ops.Group.is_unary op -> U.alu_unary ~op ~src:s
        | op, [ l; r ] when Ops.Group.is_binary op ->
            U.alu_binary ~op ~lhs:l ~rhs:r
        | op, [ a; b; c ] when Ops.Group.is_ternary op ->
            U.alu_ternary ~op ~a ~b ~c
        | _ -> failwith "alu_multi: unexpected"
      in
      Some (U.multi ~src:result ~axis)

let reduce_multi ~devices op axes ranges dtype src axis multi =
  let reduced =
    match ranges with
    | [] -> U.reduce_axis ~src ~op ~axes
    | _ -> U.reduce ~src ~op ~ranges ~dtype
  in
  if List.mem axis axes then
    (* Shard axis is being reduced — allreduce across devices. *)
    let dev =
      match devices multi with
      | Some d -> d
      | None -> failwith "reduce_multi: no device"
    in
    (* ALLREDUCE_CAST: when the sharded input is a CAST up from bf16/half,
       allreduce in the original (narrow) dtype and cast back, to keep the
       reduction in the accumulation precision tinygrad chose. *)
    let cast_from_narrow =
      U.op src = Ops.Cast
      && Array.length (U.src src) = 1
      && is_bf16_or_half (U.dtype (U.src src).(0))
    in
    if Helpers.Context_var.get allreduce_cast <> 0 && cast_from_narrow then
      let cast_src = (U.src src).(0) in
      let reduced_in_narrow =
        U.cast ~src:reduced ~dtype:(U.dtype cast_src)
      in
      U.cast
        ~src:(U.allreduce ~src:reduced_in_narrow ~device:dev ~op)
        ~dtype:(U.dtype reduced)
    else U.allreduce ~src:reduced ~device:dev ~op
  else
    (* Reducing axes below the shard axis removes them, so the shard axis
       shifts down by the number of reduced axes that precede it. *)
    let new_axis =
      axis - List.length (List.filter (fun a -> a < axis) axes)
    in
    U.multi ~src:reduced ~axis:new_axis

(* Find the last position in [new_shape] where the cumulative product of
   all preceding dimensions equals [prior_prod]. Returns [None] when the
   shard boundary cannot be placed (e.g. dimensions were merged across it). *)
let find_shard_axis prior_prod new_shape =
  let acc = ref 1 in
  let found = ref None in
  List.iteri
    (fun i s ->
      if !acc = prior_prod then found := Some i;
      acc := !acc * s)
    new_shape;
  !found

let reshape_multi ~shapes ~devices shape src axis multi =
  match extract_int_shape shape, shapes multi with
  | None, _ | _, None -> None
  | Some new_shape, Some multi_shape ->
      let ndev = ndev_of devices multi in
      if prod multi_shape <> prod new_shape then
        failwith "reshape must maintain prod(shape)";
      let prior_prod = prod (List.filteri (fun i _ -> i < axis) multi_shape) in
      (match find_shard_axis prior_prod new_shape with
      | None -> None
      | Some new_axis ->
          let adjusted =
            List.mapi (fun i s -> if i = new_axis then s / ndev else s) new_shape
          in
          Some
            (U.multi
               ~src:(U.reshape ~src ~shape:(emit_shape adjusted))
               ~axis:new_axis))

(* The expand target shape uses the full multi-device size at the shard
   axis, but the inner source has the per-shard size — keep it. *)
let expand_multi ~shapes shape src axis =
  match extract_int_shape shape with
  | None -> None
  | Some target ->
      let adjusted =
        match shapes src with
        | Some src_shape ->
            List.mapi
              (fun i s -> if i = axis then List.nth src_shape axis else s)
              target
        | None -> target
      in
      Some (U.multi ~src:(U.expand ~src ~shape:(emit_shape adjusted)) ~axis)

let pad_multi ~shapes offset size src axis =
  match extract_int_shape offset, extract_int_shape size, shapes src with
  | Some offsets, Some sizes, Some src_shape ->
      if List.nth offsets axis <> 0
         || List.nth sizes axis <> List.nth src_shape axis
      then
        failwith "padding not supported on sharded axis";
      Some (U.multi ~src:(U.pad ~src ~offset ~size) ~axis)
  | _ -> None

let permute_multi order src axis =
  U.multi ~src:(U.permute ~src ~order) ~axis:(index_of axis order)

let flip_multi dims src axis =
  if List.nth dims axis then failwith "flipping not supported on sharded axis";
  U.multi ~src:(U.flip ~src ~dims) ~axis

let shrink_multi ~shapes ~devices offset size src axis multi =
  match
    ( extract_int_shape offset,
      extract_int_shape size,
      shapes src,
      shapes multi,
      devices multi )
  with
  | Some offsets, Some sizes, Some src_shape, Some multi_shape, Some dev ->
      let pairs = List.combine offsets sizes in
      let shard_pair = List.nth pairs axis in
      let shard_dim = List.nth src_shape axis in
      let full_pair = 0, List.nth multi_shape axis in
      let ndev = ndev_of devices multi in
      let bounds =
        List.init ndev (fun i -> i * shard_dim, shard_dim)
      in
      if shard_pair <> full_pair && not (List.mem shard_pair bounds) then
        failwith "shrinking not supported on sharded axis";
      let replace_shard p =
        List.mapi
          (fun i (s, e) -> if i = axis then 0, shard_dim else s, e)
          p
      in
      if shard_pair <> full_pair then
        (* Shrink targets exactly one partition — select that shard,
           copy to all devices, drop the MULTI wrapper. *)
        let idx = index_of shard_pair bounds in
        let offset, size = emit_pairs (replace_shard pairs) in
        Some
          (U.shrink
             ~src:
               (U.copy ~src:(U.mselect ~src:multi ~index:idx)
                  ~device:dev ())
             ~offset ~size)
      else
        (* Full-axis shrink: adjust to per-shard range, shrink independently. *)
        let offset, size = emit_pairs (replace_shard pairs) in
        Some (U.multi ~src:(U.shrink ~src ~offset ~size) ~axis)
  | _ -> None

(* store_after_multi receives the MULTI-wrapped dest and uses it directly —
   inner MULTIs are stripped by later rewrite passes. *)
let store_after_multi dest src_inner src_axis =
  U.multi
    ~src:(U.after ~src:dest ~deps:[ U.store ~dst:dest ~value:src_inner () ])
    ~axis:src_axis

(* Apply op to inner shard, unwrap any other MULTI sources, re-wrap. *)
let passthrough_multi root src axis =
  let wrap inner = Some (U.multi ~src:inner ~axis) in
  let tail_unwrapped () =
    U.src root |> Array.to_list |> List.tl |> List.map unwrap_multi
  in
  match U.op root with
  | Ops.Cast -> wrap (U.cast ~src ~dtype:(U.dtype root))
  | Ops.Bitcast -> wrap (U.bitcast ~src ~dtype:(U.dtype root))
  | Ops.Contiguous -> wrap (U.contiguous ~src ~ranges:(tail_unwrapped ()) ())
  | Ops.Detach -> wrap (U.detach ~src)
  | Ops.Contiguous_backward -> wrap (U.contiguous_backward ~src)
  | Ops.After -> wrap (U.after ~src ~deps:(tail_unwrapped ()))
  | _ -> None

(* Param layout: mandatory shape child, device in the Param arg. *)
let param_parts node =
  let shape = match U.as_param node with
    | Some { shape; _ } -> shape
    | None -> U.shape_to_shape_arg None
  in
  let device =
    match U.Arg.as_param_arg (U.arg node) with
    | Some p -> p.device
    | None -> None
  in
  shape, device

let shard_param_shape ~axis ~ndev shape =
  let s = unwrap_multi shape in
  match extract_int_shape s with
  | Some dims ->
      let adjusted =
        List.mapi (fun i d -> if i = axis then d / ndev else d) dims
      in
      emit_shape adjusted
  | None -> s

(* PARAM: if a PARAM has a MULTI child (indicating it lives on multiple
   devices), rebuild with per-shard shape and wrap in MULTI. *)
let param_to_multi ~devices node =
  let children = U.children node in
  match List.find_map (fun c ->
    if is_multi c then Some (multi_axis c) else None) children
  with
  | None -> None
  | Some axis ->
      let param =
        match U.Arg.as_param_arg (U.arg node) with
        | Some p -> p
        | None -> assert false
      in
      let dtype = U.dtype node in
      let ndev = ndev_of devices node in
      let shape, device = param_parts node in
      let shape = shard_param_shape ~axis ~ndev shape in
      Some
        (U.multi
           ~src:
             (U.param ~slot:param.slot ~dtype ~shape ?device
                ?vmin_vmax:param.vmin_vmax ?name:param.name
                ~addrspace:param.addrspace ?axis:param.axis ())
           ~axis)

(* Pattern matcher *)

let rec multi_pm ~shapes ~devices node =
  match U.op node with
  (* PARAM with MULTI children -> shard shape, wrap in MULTI. *)
  | Ops.Param -> param_to_multi ~devices node

  (* ALU: align shard axes across sources, apply per-shard. *)
  | op
    when Ops.Group.is_alu op && List.exists is_multi (U.children node) ->
      alu_multi ~shapes ~devices node

  (* Movement/reduction ops with MULTI source. *)
  | Ops.Reduce when is_multi (U.src node).(0) ->
      let m = (U.src node).(0) in
      let inner, axis = inner_axis m in
      let { U.ranges; op; axes; _ } =
        match U.as_reduce node with Some r -> r | None -> assert false
      in
      let axes =
        match axes with
        | [] ->
            List.filter_map
              (fun r ->
                match U.as_range r with Some v -> Some v.axis | None -> None)
              ranges
        | axes -> axes
      in
      Some
        (reduce_multi ~devices op axes ranges (Dtype.val_of (U.dtype node)) inner
           axis m)
  | Ops.Reshape when is_multi (U.src node).(0) ->
      let m = (U.src node).(0) and shape = (U.src node).(1) in
      let inner, axis = inner_axis m in
      reshape_multi ~shapes ~devices shape inner axis m
  | Ops.Expand when is_multi (U.src node).(0) ->
      let m = (U.src node).(0) and shape = (U.src node).(1) in
      let inner, axis = inner_axis m in
      expand_multi ~shapes shape inner axis
  | Ops.Pad when is_multi (U.src node).(0) ->
      let m = (U.src node).(0) in
      let offset = (U.src node).(1) and size = (U.src node).(2) in
      let inner, axis = inner_axis m in
      pad_multi ~shapes offset size inner axis
  | Ops.Permute when is_multi (U.src node).(0) ->
      let m = (U.src node).(0) in
      let order =
        match U.arg node with U.Arg.Ints xs -> xs | _ -> assert false
      in
      let inner, axis = inner_axis m in
      Some (permute_multi order inner axis)
  | Ops.Flip when is_multi (U.src node).(0) ->
      let m = (U.src node).(0) in
      let dims =
        match U.arg node with U.Arg.Bools xs -> xs | _ -> assert false
      in
      let inner, axis = inner_axis m in
      Some (flip_multi dims inner axis)

  (* SHRINK: multi_pm rule (MULTI source) or replace_allreduce (MSTACK). *)
  | Ops.Shrink -> (
      let src = (U.src node).(0) in
      let before = (U.src node).(1) and after = (U.src node).(2) in
      match U.op src with
      | Ops.Multi ->
          let inner, axis = inner_axis src in
          shrink_multi ~shapes ~devices before after inner axis src
      | Ops.Mstack -> Some (mstack_early_shrink src before after)
      | _ -> None)

  (* AFTER(MULTI, STORE(MULTI, MULTI)) -> store_after_multi;
     AFTER(MULTI, ...) -> passthrough. *)
  | Ops.After when is_multi (U.src node).(0) ->
      let src = (U.src node).(0) in
      let deps = Array.to_list (U.src node) |> List.tl in
      let try_store =
        match deps with
        | [ dep ] when U.op dep = Ops.Store ->
            let dst = (U.src dep).(0) and value = (U.src dep).(1) in
            if is_multi dst && is_multi value then
              Some (store_after_multi dst (unwrap_multi value) (multi_axis value))
            else None
        | _ -> None
      in
      (match try_store with
      | Some _ as r -> r
      | None -> passthrough_multi node (unwrap_multi src) (multi_axis src))

  (* COPY(MULTI, device) -> gather via unshard + allreduce.
     COPY(single->multi) -> broadcast. COPY(multi->single) -> select shard 0. *)
  | Ops.Copy ->
      let src = (U.src node).(0) in
      if is_multi src then
        match U.Arg.as_device (U.arg node) with
        | Some device -> Some (copy_multi ~shapes ~devices src device)
        | None -> None
      else (
        match broadcast_copy ~devices node with
        | Some _ as r -> r
        | None -> copy_to_one ~devices node)

  (* ALLREDUCE(MULTI, device) -> unwrap, allreduce inner, re-wrap. *)
  | Ops.Allreduce when is_multi (U.src node).(0) ->
      let src = (U.src node).(0) in
      let op, device =
        match U.arg node with
        | U.Arg.Op_device (op, device) -> op, device
        | _ -> assert false
      in
      let inner = (U.src src).(0) and axis = multi_axis src in
      Some (U.multi ~src:(U.allreduce ~src:inner ~device ~op) ~axis)

  (* CALL/FUNCTION: resolve body recursively, then passthrough or void strip. *)
  | Ops.Call | Ops.Function ->
      call_multi ~shapes ~devices node

  (* Passthrough: CAST, BITCAST, CONTIGUOUS, DETACH, CONTIGUOUS_BACKWARD. *)
  | (Ops.Cast | Ops.Bitcast | Ops.Contiguous | Ops.Detach
    | Ops.Contiguous_backward)
    when is_multi (U.src node).(0) ->
      let src = (U.src node).(0) in
      passthrough_multi node (unwrap_multi src) (multi_axis src)

  (* STORE: strip MULTI from dst and value. *)
  | Ops.Store when is_multi (U.src node).(0) ->
      let dst = (U.src node).(0) and value = (U.src node).(1) in
      Some (U.store ~dst:(unwrap_multi dst) ~value:(unwrap_multi value) ())

  (* MSELECT: resolve on MSTACK, or push inside movement ops. *)
  | Ops.Mselect -> (
      match mselect_mstack node with
      | Some _ as r -> r
      | None -> mselect_before_movement node)

  (* GETTUPLE(TUPLE) -> direct indexing. *)
  | Ops.Gettuple when U.op (U.src node).(0) = Ops.Tuple ->
      let t = (U.src node).(0) in
      let index = match U.arg node with U.Arg.Int i -> i | _ -> assert false in
      Some (U.src t).(index)

  (* GETTUPLE(MULTI(TUPLE|FUNCTION)): pass MULTI through the projection. *)
  | Ops.Gettuple when is_multi (U.src node).(0) ->
      let m = (U.src node).(0) in
      let inner = unwrap_multi m in
      let index = match U.arg node with U.Arg.Int i -> i | _ -> assert false in
      (match U.op inner with
      | Ops.Tuple | Ops.Function ->
          Some (U.multi ~src:(U.gettuple ~src:inner ~index) ~axis:(multi_axis m))
      | _ -> Some m)

  | _ -> None

and rewrite_into_function ~shapes ~devices ~info body args =
  let new_body = U.graph_rewrite (multi_pm ~shapes ~devices) body in
  let new_args = List.map unwrap_multi args in
  match U.op new_body with
  | Ops.Tuple when List.exists is_multi (U.children new_body) ->
      let elems = U.children new_body in
      let stripped =
        U.tuple (List.map unwrap_multi elems)
      in
      let shard_call = U.call ~body:stripped ~args:new_args ~info in
      let wrapped =
        List.mapi
          (fun i s ->
            let g = U.gettuple ~src:shard_call ~index:i in
            if is_multi s then U.multi ~src:g ~axis:(multi_axis s) else g)
          elems
      in
      U.tuple wrapped
  | _ -> U.call ~body:new_body ~args:new_args ~info

and call_multi ~shapes ~devices node =
  let info =
    match U.arg node with U.Arg.Call_info i -> i | _ -> assert false
  in
  let body = (U.src node).(0) in
  let args = Array.to_list (U.src node) |> List.tl in
  let is_function = U.op node = Ops.Function in
  let make = U.call in
  if is_function && not info.precompile then
    Some (rewrite_into_function ~shapes ~devices ~info body args)
  else if is_multi body then
    let axis = multi_axis body in
    let new_body = unwrap_multi body in
    let new_args = List.map unwrap_multi args in
    Some (U.multi ~src:(make ~body:new_body ~args:new_args ~info) ~axis)
  else if
    Dtype.equal (U.dtype node) Dtype.void
    && (is_multi body || List.exists is_multi args)
  then
    let new_body = unwrap_multi body in
    let new_args = List.map unwrap_multi args in
    Some (make ~body:new_body ~args:new_args ~info)
  else None
