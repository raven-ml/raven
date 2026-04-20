(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Buffer allocation.

   Transforms a tensor-level SINK into a CALL with explicit buffer
   allocations and a buffer_map tracking which original tensor nodes
   map to which allocated buffers.

   Three phases:
   1. Tag nodes that need realization (CONTIGUOUS, AFTER+STORE, bases).
   2. Replace tagged nodes with explicit buffer allocations.
   3. Finalize: strip tags, collect assigns, replace buffers with PARAMs. *)

open Tolk_ir
module T = Tensor
module D = Dtype
module C = Const

(* Helpers *)

let int_ n = T.const (C.int D.Val.index n) D.index
let shape_prod = List.fold_left ( * ) 1
let dtype_or_void n = match T.dtype n with Some d -> d | None -> D.void

let shape_node dims =
  match List.map int_ dims with [d] -> d | ds -> T.vectorize ~srcs:ds

(* Follow movement ops (not MULTI, not DETACH) plus DETACH to the
   underlying node.  Equivalent to tinygrad's UOp.multibase. *)
let rec multibase x = match T.view x with
  | Reshape { src; _ } | Expand { src; _ } | Pad { src; _ }
  | Shrink { src; _ } | Permute { src; _ } | Flip { src; _ }
  | Detach { src; _ } -> multibase src
  | _ -> x

(* Follow AFTER chains to the underlying source. *)
let rec base_through_after x = match T.view x with
  | After { src; _ } -> base_through_after src
  | _ -> x

(* Is the base of [x] a buffer or buffer-view? *)
let has_buffer_identity x = match T.view (T.base x) with
  | Buffer _ | Buffer_view _ -> true
  | _ -> false

(* Ops that do not need buffer realization. *)
let dont_realize = function
  | T.Const _ | T.Buffer _ | T.Bind _ | T.Define_var _ | T.After _ -> true
  | _ -> false

(* Shrink [src] to [target_shape].  Each dimension is kept from 0 to
   the target size — a no-op when shapes already match. *)
let shrink_to shapes src target_shape =
  match shapes src with
  | Some s when s = target_shape -> src
  | _ ->
      let before = shape_node (List.map (fun _ -> 0) target_shape) in
      let after = shape_node target_shape in
      T.shrink ~src ~before ~after

(* If movement ops on [src] collapse to a contiguous range backed by a
   buffer, return the element offset.  Returns [None] when the view is
   non-contiguous or too complex to analyse statically. *)
let contiguous_view_offset shapes src =
  (* Walk the movement-op chain and track whether the view stays
     contiguous.  We handle the common patterns; the full analysis
     would require the rangeify index pipeline. *)
  let rec walk node = match T.view node with
    | Buffer _ | Buffer_view _ -> Some 0
    | Reshape { src; _ } -> walk src
    | Shrink { src; _ } ->
        let inner = match shapes src with Some s -> s | None -> [] in
        if inner = [] then None
        else
          let pairs = match T.extract_marg_pairs (T.view node) with
            | Some p -> p | None -> [] in
          if pairs = [] then None
          else
            let n = List.length pairs in
            (* All leading dimensions must be kept in full. *)
            let all_full = List.for_all2 (fun (b, e) d ->
              b = 0 && e = d) (List.filteri (fun i _ -> i < n - 1) pairs)
              (List.filteri (fun i _ -> i < n - 1) inner) in
            if not all_full then None
            else
              let last_b = fst (List.nth pairs (n - 1)) in
              if last_b = 0 then walk src
              else
                (* Contiguous slice starting at last_b. *)
                let strides = List.rev (List.fold_left (fun acc d ->
                  (List.hd acc * d) :: acc) [1]
                  (List.rev (List.tl (List.rev inner)))) in
                let offset = last_b * List.nth strides (n - 1) in
                (match walk src with
                 | Some base_off -> Some (base_off + offset)
                 | None -> None)
    | _ -> None
  in
  let base = T.base src in
  match T.view base with
  | Buffer _ | Buffer_view _ -> walk src
  | _ -> None

(* Context *)

type ctx = {
  uop_tbl : (int, T.t) Hashtbl.t;
  mutable uop_count : int;
  buffer_map : (int, T.t) Hashtbl.t;
  bases : (int, unit) Hashtbl.t;
  mutable assigns : T.t list;
  mutable replacements : T.t list;
  tags : (int, int list) Hashtbl.t;
  shapes : T.t -> int list option;
  devices : T.t -> T.device option;
  mutable uid : int;
}

(* Tag side-table *)

let get_tags ctx n = Hashtbl.find_opt ctx.tags (T.tag n)

let get_tags_or_empty ctx n =
  match Hashtbl.find_opt ctx.tags (T.tag n) with
  | Some t -> t | None -> []

let has_tag ctx n = Hashtbl.mem ctx.tags (T.tag n)
let set_tags ctx n ts = Hashtbl.replace ctx.tags (T.tag n) ts
let remove_tags ctx n = Hashtbl.remove ctx.tags (T.tag n)

(* When graph_rewrite rebuilds a node with new children, propagate its
   tag entry to the replacement. *)
let propagate_tags ctx ~old_n ~new_n =
  if old_n != new_n then
    match Hashtbl.find_opt ctx.tags (T.tag old_n) with
    | Some t -> Hashtbl.replace ctx.tags (T.tag new_n) t
    | None -> ()

(* Assign the next tag index to [x] and record it. *)
let tag_uop ctx x =
  if has_tag ctx x then ()
  else begin
    let idx = ctx.uop_count in
    ctx.uop_count <- ctx.uop_count + 1;
    Hashtbl.replace ctx.uop_tbl idx x;
    set_tags ctx x [idx]
  end

(* Phase 1 — add_tags *)

(* Number the nodes that need realization and populate buffer_map for
   plain AFTER nodes.  Runs bottom-up so children are tagged before
   parents. *)
let add_tags ctx node =
  match T.view node with
  | After { src; deps; _ } ->
      if List.exists (fun d ->
        match T.view d with Store _ -> true | _ -> false) deps
      then tag_uop ctx node;
      Hashtbl.replace ctx.buffer_map (T.tag node) (base_through_after src);
      None
  | Contiguous _ ->
      tag_uop ctx node;
      None
  | _ when Hashtbl.mem ctx.bases (T.tag node) ->
      tag_uop ctx node;
      None
  | _ -> None

(* Phase 2 — early transform *)

(* Create a fresh buffer matching [src]'s device, shape, and [dtype].
   For multi-device tensors the buffer covers one shard and is wrapped
   in MULTI. *)
let buffer_like ctx src dtype =
  let shape = match ctx.shapes src with
    | Some s -> s | None -> failwith "buffer_like: unknown shape" in
  let dev = match ctx.devices src with
    | Some d -> d | None -> failwith "buffer_like: unknown device" in
  let axis = match T.view src with
    | Multi { axis; _ } -> Some axis | _ -> None in
  let ndev = match dev with
    | T.Multi ds -> List.length ds | T.Single _ -> 1 in
  (* Per-shard shape: divide the sharding axis by the device count. *)
  let shard_shape = match axis with
    | Some ax when ndev > 1 ->
        List.mapi (fun i d -> if i = ax then d / ndev else d) shape
    | _ -> shape in
  let size = shape_prod shard_shape in
  let dev_node = T.device dev in
  let uid = ctx.uid in
  ctx.uid <- ctx.uid + 1;
  let buf = T.buffer ~unique:(T.unique ~id:uid) ~device:dev_node ~size ~dtype in
  let buf = T.reshape ~src:buf ~shape:(shape_node shard_shape) in
  (* Shrink to actual shard shape when it differs from max shard shape.
     For evenly divisible axes this is a no-op. *)
  let buf = shrink_to ctx.shapes buf shard_shape in
  match axis with
  | Some ax when ndev > 1 -> T.multi ~src:buf ~axis:ax
  | _ -> buf

(* If movement ops on [src] collapse to a contiguous range, return a
   BUFFER_VIEW reshaped to [src]'s shape. *)
let make_buffer_view shapes src =
  match contiguous_view_offset shapes src with
  | None -> None
  | Some offset ->
      let base = T.base src in
      let size = match shapes src with
        | Some s -> shape_prod s | None -> 0 in
      (* Chain BUFFER_VIEW offsets when the base is already a view. *)
      let offset, buf = match T.view base with
        | Buffer_view { offset = bv_off; src = bv_src; _ } ->
            offset + bv_off, bv_src
        | _ -> offset, base
      in
      let bv_dtype = dtype_or_void src in
      let bv = T.buffer_view ~src:buf ~size ~offset ~dtype:bv_dtype in
      let shape = match shapes src with Some s -> s | None -> [] in
      Some (T.reshape ~src:bv ~shape:(shape_node shape))

(* CONTIGUOUS(movement-ops(BUFFER)) → CONTIGUOUS(BUFFER_VIEW) when the
   movement ops collapse to a contiguous range. *)
let contiguous_mops_to_view ctx node =
  match T.view node with
  | Contiguous { src; _ } ->
      let base = T.base src in
      (match T.view base with
       | Buffer _ | Buffer_view _ ->
           (* RESHAPE directly on a buffer already has buffer identity,
              handled by merge_contiguous_after — skip. *)
           let trivial_reshape = match T.view src with
             | Reshape { src = inner; _ } ->
                 (match T.view inner with
                  | Buffer _ | Buffer_view _ -> true | _ -> false)
             | _ -> false in
           if trivial_reshape then None
           else if ctx.shapes node = None then None (* symbolic shapes *)
           else
             (* XXX: should check that the device allocator supports
                offset views.  All current tolk devices (CPU, Metal)
                do, so we skip the check for now. *)
             (match make_buffer_view ctx.shapes src with
              | None -> None
              | Some view ->
                  let c = T.contiguous ~src:view () in
                  (match get_tags ctx node with
                   | Some ts -> set_tags ctx c ts
                   | None -> ());
                  Some c)
       | _ -> None)
  | _ -> None

(* Transform precompiled CALL nodes to have explicit output buffers.
   Currently only single-output (SINK body) precompiled calls exist in
   tolk; multi-output calls would need TUPLE/GETTUPLE IR support. *)
let transform_precompiled_call _ctx node =
  match T.view node with
  | Call { info; callee = Ref body; _ } when info.precompile ->
      (match T.view body with
       | Sink _ -> None
       | _ -> None)
  | _ -> None

(* Rule: tagged non-CONTIGUOUS/AFTER/STORE → wrap in CONTIGUOUS and
   move the tag onto it. *)
let wrap_tagged ctx node =
  match T.view node with
  | Contiguous _ | After _ | Store _ -> None
  | _ ->
      (match get_tags ctx node with
       | Some ts ->
           remove_tags ctx node;
           let c = T.contiguous ~src:node () in
           set_tags ctx c ts;
           Some c
       | None -> None)

(* Rule: CONTIGUOUS(AFTER) where AFTER's source has buffer identity →
   remove the redundant CONTIGUOUS and merge tags into the AFTER. *)
let merge_contiguous_after ctx node =
  match T.view node with
  | Contiguous { src = a; _ } ->
      (match T.view a with
       | After { src = a_src; _ } when has_buffer_identity a_src ->
           let merged = get_tags_or_empty ctx a @ get_tags_or_empty ctx node in
           remove_tags ctx node;
           set_tags ctx a merged;
           Some a
       | _ -> None)
  | _ -> None

(* Rule: AFTER(_, STORE(_, src)) → CONTIGUOUS(src) when the store's
   target is not a BUFFER. *)
let revert_store_to_contiguous ctx node =
  match T.view node with
  | After { deps; _ } ->
      let store_src = List.find_map (fun d ->
        match T.view d with
        | Store { value; _ } -> Some value
        | _ -> None) deps in
      (match store_src with
       | None -> None
       | Some src ->
           let rec find_target n = match T.view n with
             | Bitcast { src; _ } | After { src; _ } -> find_target (T.base src)
             | _ -> n
           in
           let target = find_target node in
           (match T.view target with
            | Buffer _ -> None
            | _ ->
                let c = T.contiguous ~src () in
                (match get_tags ctx node with
                 | Some ts -> set_tags ctx c ts
                 | None -> ());
                Some c))
  | _ -> None

(* Rule: CONTIGUOUS → BUFFER + STORE + AFTER.  The core allocation. *)
let contig_to_store_after ctx node =
  match T.view node with
  | Contiguous { src; dtype; _ } ->
      let has_dev = ctx.devices src <> None in
      if not has_dev then None
      else
        let shape = match ctx.shapes src with Some s -> s | None -> [] in
        if shape_prod shape = 0 then Some src
        else begin
          let buf = buffer_like ctx src dtype in
          let store = T.store ~dst:buf ~value:src in
          let result = T.after ~src:buf ~deps:[store] in
          (match get_tags ctx node with
           | Some ts -> set_tags ctx result ts
           | None -> ());
          Some result
        end
  | _ -> None

(* Rule: remove DETACH / CONTIGUOUS_BACKWARD. *)
let remove_detach node =
  match T.view node with
  | Detach { src; _ } | Contiguous_backward { src; _ } -> Some src
  | _ -> None

(* Phase 3 — finalize *)

(* Strip tags, map each original numbered node to its final buffer,
   and collect assigns. *)
let pm_finalize ctx node =
  match T.view node with
  | After _ ->
      (match get_tags ctx node with
       | Some tag_indices ->
           remove_tags ctx node;
           let replace_uop = base_through_after node in
           List.iter (fun t ->
             let original = Hashtbl.find ctx.uop_tbl t in
             let original_shape =
               match ctx.shapes original with Some s -> s | None -> [] in
             let buf = shrink_to ctx.shapes replace_uop original_shape in
             Hashtbl.replace ctx.buffer_map (T.tag original) buf)
             tag_indices
       | None -> ());
      ctx.assigns <- node :: ctx.assigns;
      None
  | Const { value; dtype; srcs = [u; d] }
    when (match T.view u with Unique _ -> true | _ -> false)
      && (match T.view d with Device _ -> true | _ -> false) ->
      Some (T.const ~srcs:[d] value dtype)
  | _ -> None

(* Replace BUFFER, BUFFER_VIEW, and BIND with PARAM for cache-key
   normalisation. *)
let pm_replace_buf ctx node =
  let replace_input b =
    ctx.replacements <- b :: ctx.replacements;
    let slot = List.length ctx.replacements - 1 in
    let dtype = dtype_or_void b in
    let device = match T.view b with
      | Buffer { device; _ } -> Some device
      | _ -> None
    in
    Some (T.param ~slot ~dtype ?device ())
  in
  match T.view node with
  | Buffer { unique; device; _ }
    when (match T.view unique with Unique _ -> true | _ -> false)
      && (match T.view device with Device _ -> true | _ -> false) ->
      replace_input node
  | Buffer_view { src; _ }
    when (match T.view src with Buffer _ -> true | _ -> false) ->
      replace_input node
  | Bind { var; value = Some v; _ }
    when (match T.view var with Define_var _ -> true | _ -> false)
      && (match T.view v with Const _ -> true | _ -> false) ->
      replace_input node
  | _ -> None

(* Entry point *)

let transform_to_call (big_sink : T.t) : T.t * (int, T.t) Hashtbl.t =
  let shapes = T.compute_shapes big_sink in
  let devices = T.compute_devices big_sink in
  let bases = Hashtbl.create 16 in
  (match T.view big_sink with
   | Sink { srcs; _ } ->
       List.iter (fun x ->
         if not (dont_realize (T.view (T.base x))) then
           Hashtbl.replace bases (T.tag (multibase x)) ())
         srcs
   | _ -> ());
  let uid_start =
    List.fold_left (fun acc x ->
      match T.view x with
      | Unique { id; _ } -> max acc (id + 1)
      | _ -> acc) 0 (T.toposort big_sink)
  in
  let ctx = {
    uop_tbl = Hashtbl.create 64;
    uop_count = 0;
    buffer_map = Hashtbl.create 64;
    bases;
    assigns = [];
    replacements = [];
    tags = Hashtbl.create 64;
    shapes;
    devices;
    uid = uid_start;
  } in
  (* Phase 1: number the nodes that need realization. *)
  let big_sink =
    T.graph_rewrite ~name:"add_tags" (add_tags ctx) big_sink in
  (* Phase 2: replace tagged nodes with buffer allocations. *)
  let big_sink =
    T.graph_rewrite ~name:"early_transform"
      ~on_rebuild:(propagate_tags ctx)
      (T.first_match [
        transform_precompiled_call ctx;
        contiguous_mops_to_view ctx;
        wrap_tagged ctx;
        merge_contiguous_after ctx;
        revert_store_to_contiguous ctx;
        contig_to_store_after ctx;
        remove_detach;
      ]) big_sink in
  (* Phase 3a: finalize — strip tags and collect assigns. *)
  ignore (T.graph_rewrite ~name:"finalize" (pm_finalize ctx) big_sink);
  (* Phase 3b: replace buffers with PARAMs and wrap in a CALL. *)
  let assigns_sink = T.sink (List.rev ctx.assigns) in
  let body =
    T.graph_rewrite ~name:"replace_bufs" (pm_replace_buf ctx) assigns_sink in
  let args = List.rev ctx.replacements in
  let dtype = dtype_or_void body in
  let info =
    { T.grad_fxn = None; metadata = []; name = None; precompile = false } in
  let ret = T.call ~callee:(Ref body) ~args ~info ~dtype in
  (ret, ctx.buffer_map)
