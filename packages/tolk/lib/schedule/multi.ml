(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Multi-device sharding transformations.

   Transforms operations on MULTI-wrapped (sharded) buffers into per-shard
   operations.  Each handler strips the MULTI wrapper, applies the operation
   to the inner per-shard tensor, and re-wraps the result. *)

open Tolk_ir
module T = Tensor

(* Helpers *)

let prod l = List.fold_left ( * ) 1 l

let index_of x l =
  let rec loop i = function
    | [] -> invalid_arg "index_of: element not found"
    | y :: _ when y = x -> i
    | _ :: rest -> loop (i + 1) rest
  in
  loop 0 l

let ndev_of devices node =
  match (devices node : T.device option) with
  | Some (T.Multi ds) -> List.length ds
  | _ -> 1

let int_ n = T.const (Const.int Dtype.Val.int32 n) Dtype.int32

(* Build a shape-like vectorized node from scalar tensor expressions. *)
let emit_symbolic = function
  | [d] -> d | ds -> T.vectorize ~srcs:ds

(* Partition [src] along [axis] using a symbolic device index.
   Each device takes its slice: [dnum*sz .. dnum*sz + sz). *)
let shard shape ndev src axis =
  let dim = List.nth shape axis in
  if dim mod ndev <> 0 then failwith "multi axis uneven";
  let sz = dim / ndev in
  let dnum = T.define_var ~name:"_device_num" ~lo:0 ~hi:(ndev - 1)
    ~dtype:Dtype.int32 () in
  let off = T.binary ~op:`Mul ~lhs:dnum ~rhs:(int_ sz) in
  let before = List.mapi (fun i _ ->
    if i <> axis then int_ 0 else off) shape in
  let after = List.mapi (fun i s ->
    if i <> axis then int_ s
    else T.binary ~op:`Add ~lhs:off ~rhs:(int_ sz)) shape in
  T.shrink ~src ~before:(emit_symbolic before) ~after:(emit_symbolic after)

(* Inverse of [shard]: pad each device's shard so it covers the full range,
   with zeros outside its slice.  Summing across devices reconstructs the
   full tensor. *)
let unshard shape ndev src axis =
  let bsz = List.nth shape axis in
  let dnum = T.define_var ~name:"_device_num" ~lo:0 ~hi:(ndev - 1)
    ~dtype:Dtype.int32 () in
  let off = T.binary ~op:`Mul ~lhs:(int_ bsz) ~rhs:dnum in
  let before = List.mapi (fun i _ ->
    if i <> axis then int_ 0 else off) shape in
  let after = List.mapi (fun i _ ->
    if i <> axis then int_ 0
    else T.binary ~op:`Sub ~lhs:(int_ (bsz * (ndev - 1))) ~rhs:off) shape in
  T.pad ~src ~before:(emit_symbolic before) ~after:(emit_symbolic after)

(* MSELECT/MSTACK rewrite *)

(* Substitute every [_device_num] variable in [node] with constant [i]. *)
let subst_device_num node i =
  let is_device_num v = match T.view v with
    | Define_var { name; _ } -> name = "_device_num" | _ -> false
  in
  match List.find_opt is_device_num (T.variables node) with
  | None -> node
  | Some dvar ->
    let dt = Option.value ~default:Dtype.int32 (T.dtype dvar) in
    T.substitute [(dvar, T.const (Const.int (Dtype.val_of dt) i) dt)] node

(* Decompose a vectorized shape node into its per-axis scalars. *)
let shape_elems node = match T.view node with
  | Vectorize { srcs; _ } -> srcs | _ -> [node]

(* Move SHRINK before MSTACK: substitute [_device_num] with each device
   index and apply the shrink to each MSTACK element individually. *)
let mstack_early_shrink ms before after =
  let bs = shape_elems before and es = shape_elems after in
  let srcs = match T.view ms with
    | Mstack { srcs; _ } -> srcs
    | _ -> failwith "mstack_early_shrink: expected MSTACK"
  in
  let apply_shrink s i =
    let bs' = List.map (fun b -> subst_device_num b i) bs in
    let es' = List.map (fun e -> subst_device_num e i) es in
    T.shrink ~src:s ~before:(emit_symbolic bs') ~after:(emit_symbolic es')
  in
  let new_srcs = List.mapi (fun i x ->
    match T.view x with
    | Copy { src; device; _ } ->
        T.copy ~src:(apply_shrink src i) ~device ()
    | _ ->
        T.contiguous ~src:(apply_shrink x i) ()) srcs
  in
  T.mstack ~srcs:new_srcs

(* BROADCAST: copy from single to multi-device → per-device copies in MSTACK. *)
let broadcast_copy ~devices node =
  match T.view node with
  | Copy { src; device; _ } ->
    (match (devices src : T.device option), T.view device with
     | Some (T.Single _), Device { device = Multi ds } ->
         let copies = List.map (fun d ->
           T.copy ~src ~device:(T.device (Single d)) ()) ds in
         Some (T.mstack ~srcs:copies)
     | _ -> None)
  | _ -> None

(* COPY_TO_ONE: copy from multi-device to single → select shard 0 and copy. *)
let copy_to_one ~devices node =
  match T.view node with
  | Copy { src; device; _ } ->
    (match (devices src : T.device option), T.view device with
     | Some (T.Multi _), Device { device = T.Single _ } ->
         Some (T.copy ~src:(T.mselect ~src ~index:0) ~device ())
     | _ -> None)
  | _ -> None

(* MSELECT(MSTACK) → direct indexing. *)
let mselect_mstack node =
  match T.view node with
  | Mselect { src; index; _ } ->
    (match T.view src with
     | Mstack { srcs; _ } -> List.nth_opt srcs index
     | _ -> None)
  | _ -> None

(* MSELECT(movement(s)) → movement(MSELECT(s)): push select inside. *)
let mselect_before_movement node =
  match T.view node with
  | Mselect { src; index; _ } ->
    let sel inner = T.mselect ~src:inner ~index in
    (match T.view src with
     | Reshape { src = inner; shape; _ } ->
         Some (T.reshape ~src:(sel inner) ~shape)
     | Expand { src = inner; shape; _ } ->
         Some (T.expand ~src:(sel inner) ~shape)
     | Permute { src = inner; order; _ } ->
         Some (T.permute ~src:(sel inner) ~order)
     | Flip { src = inner; dims; _ } ->
         Some (T.flip ~src:(sel inner) ~dims)
     | Pad { src = inner; before; after; _ } ->
         Some (T.pad ~src:(sel inner) ~before ~after)
     | Shrink { src = inner; before; after; _ } ->
         Some (T.shrink ~src:(sel inner) ~before ~after)
     | _ -> None)
  | _ -> None

(* Multi functions *)

(* The expand target shape uses the full multi-device size at the shard
   axis, but the inner source has the per-shard size — keep it. *)
let expand_multi ~shapes shape src axis =
  match T.extract_int_shape shape with
  | None -> None
  | Some target ->
    let adjusted = match shapes src with
      | Some src_shape ->
        List.mapi (fun i s ->
          if i = axis then List.nth src_shape axis else s) target
      | None -> target
    in
    Some (T.multi ~src:(T.expand ~src ~shape:(Allreduce.emit_shape adjusted))
            ~axis)

let pad_multi before after src axis =
  match T.extract_int_shape before, T.extract_int_shape after with
  | Some bs, Some es ->
    if List.nth bs axis <> 0 || List.nth es axis <> 0 then
      failwith "padding not supported on sharded axis";
    Some (T.multi ~src:(T.pad ~src ~before ~after) ~axis)
  | _ -> None

let permute_multi order src axis =
  T.multi ~src:(T.permute ~src ~order) ~axis:(index_of axis order)

let flip_multi dims src axis =
  if List.nth dims axis then failwith "flipping not supported on sharded axis";
  T.multi ~src:(T.flip ~src ~dims) ~axis

let reduce_multi ~devices op axes src axis multi =
  let reduced = T.reduce_axis ~src ~op ~axes in
  if List.mem axis axes then
    (* Shard axis is being reduced — allreduce across devices. *)
    let dev = match devices multi with
      | Some d -> T.device d
      | None -> failwith "reduce_multi: no device"
    in
    let dt = Option.value ~default:Dtype.void (T.dtype reduced) in
    T.allreduce ~src:reduced ~device:dev ~op ~dtype:dt
  else
    T.multi ~src:reduced ~axis

(* In tinygrad, store_after_multi receives the MULTI-wrapped dest and
   uses it directly — inner MULTIs are stripped by later rewrite passes.
   We match that: the caller should pass the MULTI-wrapped dest. *)
let store_after_multi dest src_inner src_axis =
  T.multi ~src:(T.after ~src:dest ~deps:[T.store ~dst:dest ~value:src_inner])
    ~axis:src_axis

let unwrap_multi x = match T.view x with T.Multi { src; _ } -> src | _ -> x

(* Apply op to inner shard, unwrap any other MULTI sources, re-wrap. *)
let passthrough_multi root src axis =
  let wrap inner = Some (T.multi ~src:inner ~axis) in
  match T.view root with
  | Cast { dtype; _ } -> wrap (T.cast ~src ~dtype)
  | Bitcast { dtype; _ } -> wrap (T.bitcast ~src ~dtype)
  | Contiguous { ranges; opts; _ } ->
    wrap (T.contiguous ~src ~ranges:(List.map unwrap_multi ranges) ~opts ())
  | Detach _ -> wrap (T.detach ~src)
  | Contiguous_backward _ -> wrap (T.contiguous_backward ~src)
  | After { deps; _ } ->
    wrap (T.after ~src ~deps:(List.map unwrap_multi deps))
  | _ -> None

(* Find the last position in [new_shape] where the cumulative product of
   all preceding dimensions equals [prior_prod]. Returns [None] when the
   shard boundary cannot be placed (e.g. dimensions were merged across it). *)
let find_shard_axis prior_prod new_shape =
  let acc = ref 1 in
  let found = ref None in
  List.iteri (fun i s ->
    if !acc = prior_prod then found := Some i;
    acc := !acc * s)
    new_shape;
  !found

let reshape_multi ~shapes ~devices shape src axis multi =
  match T.extract_int_shape shape, shapes multi with
  | None, _ | _, None -> None
  | Some new_shape, Some multi_shape ->
    let ndev = ndev_of devices multi in
    if prod multi_shape <> prod new_shape then
      failwith "reshape must maintain prod(shape)";
    let prior_prod = prod (List.filteri (fun i _ -> i < axis) multi_shape) in
    match find_shard_axis prior_prod new_shape with
    | None -> None
    | Some new_axis ->
      let adjusted = List.mapi (fun i s ->
        if i = new_axis then s / ndev else s) new_shape in
      Some (T.multi ~src:(T.reshape ~src ~shape:(Allreduce.emit_shape adjusted))
              ~axis:new_axis)

let shrink_multi ~shapes ~devices before after src axis multi =
  match T.extract_int_shape before, T.extract_int_shape after,
        shapes src, shapes multi, devices multi with
  | Some starts, Some ends, Some src_shape, Some multi_shape, Some dev ->
    let pairs = List.combine starts ends in
    let shard_pair = List.nth pairs axis in
    let shard_dim = List.nth src_shape axis in
    let full_pair = (0, List.nth multi_shape axis) in
    let ndev = ndev_of devices multi in
    let bounds = List.init ndev (fun i ->
      (i * shard_dim, (i + 1) * shard_dim)) in
    if shard_pair <> full_pair && not (List.mem shard_pair bounds) then
      failwith "shrinking not supported on sharded axis";
    let replace_shard p =
      List.mapi (fun i (s, e) ->
        if i = axis then (0, shard_dim) else (s, e)) p
    in
    if shard_pair <> full_pair then begin
      (* Shrink targets exactly one partition — select that shard,
         copy to all devices, drop the MULTI wrapper. *)
      let idx = index_of shard_pair bounds in
      let bef, aft = Allreduce.emit_pairs (replace_shard pairs) in
      Some (T.shrink
              ~src:(T.copy ~src:(T.mselect ~src:multi ~index:idx)
                      ~device:(T.device dev) ())
              ~before:bef ~after:aft)
    end else begin
      (* Full-axis shrink: adjust to per-shard range, shrink independently. *)
      let bef, aft = Allreduce.emit_pairs (replace_shard pairs) in
      Some (T.multi ~src:(T.shrink ~src ~before:bef ~after:aft) ~axis)
    end
  | _ -> None

(* Gather a sharded MULTI tensor onto [device]: extract the inner shard,
   unshard it (symbolic pad per device), then allreduce-sum. *)
let copy_multi ~shapes ~devices multi device =
  match T.view multi with
  | Multi { src = inner; axis; _ } ->
    let inner_shape = match shapes inner with
      | Some sh -> sh
      | None -> failwith "copy_multi: unknown inner shape"
    in
    let ndev = ndev_of devices multi in
    let unsharded = unshard inner_shape ndev inner axis in
    let dt = Option.value ~default:Dtype.void (T.dtype unsharded) in
    T.allreduce ~src:unsharded ~device ~op:`Add ~dtype:dt
  | _ -> failwith "copy_multi: expected MULTI"

let alu_multi ~shapes ~devices root =
  let srcs = match T.view root with
    | Unary { src; _ } -> [src]
    | Binary { lhs; rhs; _ } -> [lhs; rhs]
    | Ternary { a; b; c; _ } -> [a; b; c]
    | _ -> []
  in
  (* Result shard axis: last axis among MULTI sources. *)
  let axes = List.filter_map (fun s ->
    match T.view s with Multi { axis; _ } -> Some axis | _ -> None) srcs in
  match axes with
  | [] -> None
  | _ ->
    let axis = List.nth axes (List.length axes - 1) in
    let ndev = match List.find_map (fun s ->
      match devices s with
      | Some (dev : T.device) -> (match dev with T.Multi ds -> Some (List.length ds) | T.Single _ -> None)
      | _ -> None) srcs with
      | Some n -> n
      | None -> failwith "alu_multi: no multi device"
    in
    let shape_exn s = match shapes s with
      | Some sh -> sh
      | None -> failwith "alu_multi: unknown shape"
    in
    (* Align each source to [axis]: unwrap matching MULTIs, shard
       non-sharded sources, gather-then-reshard mismatched ones. *)
    let aligned = List.map (fun s ->
      match T.view s with
      | Multi { src = inner; axis = a; _ } when a = axis -> inner
      | Multi _ ->
          let dev = match devices s with
            | Some d -> T.device d
            | None -> failwith "alu_multi: no device"
          in
          shard (shape_exn s) ndev
            (copy_multi ~shapes ~devices s dev) axis
      | _ -> shard (shape_exn s) ndev s axis) srcs
    in
    let result = match T.view root, aligned with
      | Unary { op; _ }, [s] -> T.unary ~op ~src:s
      | Binary { op; _ }, [l; r] -> T.binary ~op ~lhs:l ~rhs:r
      | Ternary { op; _ }, [a; b; c] -> T.ternary ~op ~a ~b ~c
      | _ -> failwith "alu_multi: unexpected"
    in
    Some (T.multi ~src:result ~axis)

(* PARAM: if a PARAM has a MULTI child (indicating it lives on multiple
   devices), rebuild with per-shard shape and wrap in MULTI. *)
let param_to_multi ~shapes ~devices node =
  match T.view node with
  | Param { slot; dtype; shape; device } ->
      let axis_opt = List.find_map (fun c ->
        match T.view c with T.Multi { axis; _ } -> Some axis | _ -> None)
        (T.children node) in
      (match axis_opt with
       | None -> None
       | Some axis ->
           let ndev = ndev_of devices node in
           let shard_shape = match shape with
             | Some s ->
                 let s = unwrap_multi s in
                 (match T.extract_int_shape s with
                  | Some dims ->
                      let adjusted = List.mapi (fun i d ->
                        if i = axis then d / ndev else d) dims in
                      Some (Allreduce.emit_shape adjusted)
                  | None -> Some s)
             | None -> None
           in
           let device = Option.map unwrap_multi device in
           Some (T.multi
                   ~src:(T.param ~slot ~dtype ?shape:shard_shape ?device ())
                   ~axis))
  | _ -> None

(* Don't resolve CALL bodies that are already compiled kernels. *)
let should_resolve_call callee (info : T.call_info) =
  if info.precompile then false
  else match callee with
  | T.Ast _ -> false
  | T.Ref body ->
      (match T.view body with
       | Sink { kernel_info = Some _; _ } | Linear _ | Copy _ -> false
       | _ -> true)

(* Pattern matcher *)

let is_multi x = match T.view x with T.Multi _ -> true | _ -> false

let multi_axis x = match T.view x with
  | T.Multi { axis; _ } -> axis | _ -> assert false

let rec multi_pm ~shapes ~devices node =
  match T.view node with
  (* PARAM with MULTI children → shard shape, wrap in MULTI. *)
  | Param _ -> param_to_multi ~shapes ~devices node

  (* ALU: align shard axes across sources, apply per-shard. *)
  | (Unary _ | Binary _ | Ternary _)
    when List.exists is_multi (T.children node) ->
      alu_multi ~shapes ~devices node

  (* Movement/reduction ops with MULTI source. *)
  | Reduce_axis { src; op; axes; _ } when is_multi src ->
      (match T.view src with
       | Multi { src = inner; axis; _ } ->
           Some (reduce_multi ~devices op axes inner axis src)
       | _ -> None)

  | Reshape { src; shape; _ } when is_multi src ->
      (match T.view src with
       | Multi { src = inner; axis; _ } ->
           reshape_multi ~shapes ~devices shape inner axis src
       | _ -> None)

  | Expand { src; shape; _ } when is_multi src ->
      (match T.view src with
       | Multi { src = inner; axis; _ } ->
           expand_multi ~shapes shape inner axis
       | _ -> None)

  | Pad { src; before; after; _ } when is_multi src ->
      (match T.view src with
       | Multi { src = inner; axis; _ } ->
           pad_multi before after inner axis
       | _ -> None)

  | Permute { src; order; _ } when is_multi src ->
      (match T.view src with
       | Multi { src = inner; axis; _ } ->
           Some (permute_multi order inner axis)
       | _ -> None)

  | Flip { src; dims; _ } when is_multi src ->
      (match T.view src with
       | Multi { src = inner; axis; _ } ->
           Some (flip_multi dims inner axis)
       | _ -> None)

  (* SHRINK: multi_pm rule (MULTI source) or replace_allreduce (MSTACK). *)
  | Shrink { src; before; after; _ } ->
      (match T.view src with
       | Multi { src = inner; axis; _ } ->
           shrink_multi ~shapes ~devices before after inner axis src
       | Mstack _ ->
           Some (mstack_early_shrink src before after)
       | _ -> None)

  (* AFTER(MULTI, STORE(MULTI, MULTI)) → store_after_multi;
     AFTER(MULTI, ...) → passthrough. *)
  | After { src; deps; _ } when is_multi src ->
      let try_store = match deps with
        | [dep] ->
            (match T.view dep with
             | Store { dst; value } when is_multi dst && is_multi value ->
                 Some (store_after_multi dst (unwrap_multi value)
                         (multi_axis value))
             | _ -> None)
        | _ -> None
      in
      (match try_store with
       | Some _ as r -> r
       | None -> passthrough_multi node (unwrap_multi src) (multi_axis src))

  (* COPY(MULTI, device) → gather via unshard + allreduce.
     COPY(single→multi) → broadcast.  COPY(multi→single) → select shard 0. *)
  | Copy { src; device; _ } ->
      if is_multi src then
        Some (copy_multi ~shapes ~devices src device)
      else
        (match broadcast_copy ~devices node with
         | Some _ as r -> r
         | None -> copy_to_one ~devices node)

  (* ALLREDUCE(MULTI, device) → unwrap, allreduce inner, re-wrap. *)
  | Allreduce { src; device; op; _ } when is_multi src ->
      (match T.view src with
       | Multi { src = inner; axis; _ } ->
           let dt = Option.value ~default:Dtype.void (T.dtype inner) in
           Some (T.multi ~src:(T.allreduce ~src:inner ~device ~op ~dtype:dt)
                   ~axis)
       | _ -> None)

  (* CALL: resolve body through multi_pm, then passthrough or void strip.
     Tinygrad's GETTUPLE/TUPLE rules have no equivalent here — our CALL
     nodes return typed values directly, not through a TUPLE wrapper. *)
  | Call { callee; args; info; dtype } ->
      (* 1. Recursive body resolution (tinygrad's rewrite_into_call). *)
      let resolved = match callee with
        | Ref body when should_resolve_call callee info ->
            let rewrite = multi_pm ~shapes ~devices in
            let new_body = T.graph_rewrite ~name:"subcall" rewrite body in
            let new_args = List.map unwrap_multi args in
            if is_multi new_body then
              let axis = multi_axis new_body in
              Some (T.multi
                      ~src:(T.call ~callee:(Ref (unwrap_multi new_body))
                              ~args:new_args ~info ~dtype)
                      ~axis)
            else if new_body == body
                    && List.for_all2 (fun a b -> a == b) new_args args
            then None
            else Some (T.call ~callee:(Ref new_body) ~args:new_args
                         ~info ~dtype)
        | _ -> None
      in
      (match resolved with
       | Some _ -> resolved
       | None ->
           (* 2. Passthrough: callee ref is MULTI. *)
           (match callee with
            | Ref r when is_multi r ->
                let axis = multi_axis r in
                Some (T.multi
                        ~src:(T.call ~callee:(Ref (unwrap_multi r))
                                ~args:(List.map unwrap_multi args) ~info ~dtype)
                        ~axis)
            | _ ->
                (* 3. void CALL: strip MULTI from all sources. *)
                let all_srcs = match callee with
                  | Ref r -> r :: args | Ast _ -> args in
                if dtype = Dtype.void && List.exists is_multi all_srcs then
                  let callee = match callee with
                    | Ref r -> T.Ref (unwrap_multi r) | c -> c in
                  Some (T.call ~callee ~args:(List.map unwrap_multi args)
                          ~info ~dtype)
                else None))

  (* Passthrough: CAST, BITCAST, CONTIGUOUS, DETACH, CONTIGUOUS_BACKWARD. *)
  | (Cast { src; _ } | Bitcast { src; _ } | Contiguous { src; _ }
    | Detach { src; _ } | Contiguous_backward { src; _ })
    when is_multi src ->
      passthrough_multi node (unwrap_multi src) (multi_axis src)

  (* STORE: strip MULTI from dst and value. *)
  | Store { dst; value } when is_multi dst ->
      Some (T.store ~dst:(unwrap_multi dst) ~value:(unwrap_multi value))

  (* MSELECT: resolve on MSTACK, or push inside movement ops. *)
  | Mselect _ ->
      (match mselect_mstack node with
       | Some _ as r -> r
       | None -> mselect_before_movement node)

  | _ -> None
