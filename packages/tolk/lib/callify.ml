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

open Tolk_uop
module U = Uop
module D = Dtype
module C = Const

(* Helpers *)

let index_ n = U.const (C.int D.weakint n)
let shape_prod = List.fold_left ( * ) 1
let dtype_or_void n = U.dtype n

(* A shape descriptor is metadata, not tensor data. It stays out of range
   propagation by position, not by dtype: it only ever occupies a source slot
   that [Indexing.data_srcs] excludes, so no consumer indexes it and no pass
   assigns it ranges. Its entries are ordinary weak integers. *)
let shape_node dims =
  match List.map index_ dims with [ d ] -> d | ds -> U.stack ds

let concrete_shape n =
  let rec loop acc = function
    | [] -> Some (List.rev acc)
    | dim :: dims ->
        Option.bind (U.const_int_value dim) (fun d -> loop (d :: acc) dims)
  in
  Option.bind (U.shape_opt n) (loop [])

(* Address space of an input buffer replaced by a PARAM: the node's own
   address space, defaulting to global for nodes that carry none (a scalar binding, a
   plain scalar). *)
let replacement_addrspace node =
  match U.addrspace node with Some a -> a | None -> D.Global

let is_op op n = Ops.equal (U.op n) op

let src0 n =
  match U.children n with
  | x :: _ -> Some x
  | [] -> None

let after_parts n =
  match U.op n, U.children n with
  | Ops.After, src :: deps -> Some (src, deps)
  | _ -> None

let single_src n =
  match U.children n with
  | [ src ] -> Some src
  | _ -> None

let first_src n =
  match U.children n with
  | src :: _ -> Some src
  | [] -> None

(* Follow movement ops (not MULTI, not DETACH) plus DETACH to the
   underlying node.  Equivalent to tinygrad's UOp.multibase. *)
let rec multibase x =
  match U.op x, single_src x, first_src x with
  | (Ops.Reshape | Ops.Expand | Ops.Pad | Ops.Shrink | Ops.Permute
    | Ops.Flip | Ops.Detach), _, Some src -> multibase src
  | _ -> x

(* Follow AFTER chains to the underlying source. *)
let rec base_through_after x =
  match after_parts x with
  | Some (src, _) -> base_through_after src
  | None -> x

let base x = base_through_after (multibase x)

(* Ops that do not need buffer realization. *)
let dont_realize = function
  | Ops.Const | Ops.Buffer | Ops.Param | Ops.After -> true
  | _ -> false

(* Shrink [src] to [target_shape].  Each dimension is kept from 0 to
   the target size — a no-op when shapes already match. *)
let shrink_to src target_shape =
  if List.equal U.equal (U.shape src) target_shape then src
  else
    let offset = shape_node (List.map (fun _ -> 0) target_shape) in
    let size = match target_shape with [ d ] -> d | ds -> U.stack ds in
    U.shrink ~src ~offset ~size


(* Context *)

type ctx = {
  uop_tbl : (int, U.t) Hashtbl.t;
  mutable uop_count : int;
  buffer_map : (int, U.t) Hashtbl.t;
  bases : (int, unit) Hashtbl.t;
  mutable assigns : U.t list;
  mutable replacements : (int * U.t) list;
  replacement_slots : (int, int) Hashtbl.t;
  tags : (int, int list) Hashtbl.t;
  shapes : U.t -> int list option;
  devices : U.t -> U.device option;
}

(* Tag side-table *)

let get_tags ctx n = Hashtbl.find_opt ctx.tags (U.tag n)

let get_tags_or_empty ctx n =
  match Hashtbl.find_opt ctx.tags (U.tag n) with
  | Some t -> t | None -> []

let has_tag ctx n = Hashtbl.mem ctx.tags (U.tag n)
let set_tags ctx n ts = Hashtbl.replace ctx.tags (U.tag n) ts
let remove_tags ctx n = Hashtbl.remove ctx.tags (U.tag n)

(* When graph_rewrite rebuilds a node with new children, propagate its
   tag entry to the replacement. *)
let propagate_tags ctx ~old_n ~new_n =
  if old_n != new_n then
    match Hashtbl.find_opt ctx.tags (U.tag old_n) with
    | Some t -> Hashtbl.replace ctx.tags (U.tag new_n) t
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
  if U.is_bound_var node then None
  else match after_parts node with
  | Some (src, deps) ->
      if List.exists (fun d ->
        match U.op d with Ops.Store -> true | _ -> false) deps
      then tag_uop ctx node;
      Hashtbl.replace ctx.buffer_map (U.tag node) (base_through_after src);
      None
  | None ->
      if is_op Ops.Contiguous node || Hashtbl.mem ctx.bases (U.tag node) then begin
        tag_uop ctx node;
        None
      end
      else None

(* Phase 2 — early transform *)

(* Create a fresh buffer matching [src]'s device, shape, and [dtype].
   For multi-device tensors the buffer covers one shard and is wrapped
   in MULTI. *)
let buffer_like ctx src dtype =
  match ctx.shapes src with
  | None ->
      (* Symbolic shape: allocate at the maximum size and shrink the view
         down to the symbolic shape. *)
      let dims = U.shape src in
      let dev = match ctx.devices src with
        | Some d -> d | None -> failwith "buffer_like: unknown device" in
      let max_shape = List.map (fun dim -> Bound.to_int (U.vmax dim)) dims in
      let buf =
        U.buffer ~slot:(U.fresh_buffer_slot ()) ~device:dev
          ~shape:(shape_node max_shape) ~addrspace:D.Global ~dtype ()
      in
      U.shrink ~src:buf
        ~offset:(shape_node (List.map (fun _ -> 0) dims))
        ~size:(match dims with [ d ] -> d | ds -> U.stack ds)
  | Some shape ->
  let dev = match ctx.devices src with
    | Some d -> d | None -> failwith "buffer_like: unknown device" in
  let axis =
    match U.op src with
    | Ops.Unshard -> U.Arg.as_int (U.arg src)
    | _ -> None
  in
  let ndev = match dev with
    | Multi ds -> List.length ds
    | Single _ | Index _ -> 1
  in
  (* Per-shard shape: divide the sharding axis by the device count. *)
  let shard_shape = match axis with
    | Some ax when ndev > 1 ->
        List.mapi (fun i d -> if i = ax then d / ndev else d) shape
    | _ -> shape in
  let buf =
    U.buffer ~slot:(U.fresh_buffer_slot ()) ~device:dev
      ~shape:(shape_node shard_shape) ~addrspace:D.Global ~dtype ()
  in
  (* Shrink to actual shard shape when it differs from max shard shape.
     For evenly divisible axes this is a no-op. *)
  let buf = shrink_to buf (List.map index_ shard_shape) in
  match axis with
  | Some ax when ndev > 1 -> U.multi ~src:buf ~axis:ax
  | _ -> buf

(* If movement ops on [src] collapse to a contiguous range, return a
   Slice reshaped to [src]'s shape. *)
let make_slice shapes src =
  (* A view fold cannot discard the effects carried by its base. Preparation
     must first forward that storage through the call producing it. *)
  match U.op (multibase src), U.contiguous_view_offset src with
  | Ops.After, _ | _, None -> None
  | _, Some offset ->
      let base = base src in
      let size = match shapes src with
        | Some s -> shape_prod s | None -> 0 in
      (* Chain onto an existing view in byte units so a dtype change between
         the outer view and the underlying buffer stays correct; a byte offset
         that is not a whole number of underlying elements cannot collapse to a
         plain slice. *)
      let chained =
        match U.as_slice base with
        | Some { offset = slice_off; src = slice_src; _ } -> (
            match U.const_int_value slice_off with
            | Some slice_off ->
                let inner_itemsize = D.itemsize (U.dtype slice_src) in
                let byte_offset =
                  (slice_off * inner_itemsize) + (offset * D.itemsize (U.dtype src))
                in
                if byte_offset mod inner_itemsize <> 0 then None
                else Some (byte_offset / inner_itemsize, slice_src)
            | None -> Some (offset, base))
        | None -> Some (offset, base)
      in
      (match chained with
       | None -> None
       | Some (offset, buf) ->
           let slice_dtype = dtype_or_void src in
           let slice =
             U.slice ~src:buf ~size ~offset:(index_ offset) ~dtype:slice_dtype
           in
           let shape = match shapes src with Some s -> s | None -> [] in
           Some (U.reshape ~src:slice ~shape:(shape_node shape)))

(* CONTIGUOUS(movement-ops(BUFFER)) → CONTIGUOUS(SLICE) when the
   movement ops collapse to a contiguous range. *)
let contiguous_mops_to_slice ctx node =
  match U.op node, first_src node with
  | Ops.Contiguous, Some src ->
      let base = base src in
      (match U.op base with
       | Ops.Buffer | Ops.Slice | Ops.Param ->
           (* RESHAPE directly on a buffer already has buffer identity,
              handled by merge_contiguous_after — skip. *)
           let trivial_reshape =
             match U.op src, first_src src with
             | Ops.Reshape, Some inner ->
                 (match U.op inner with
                  | Ops.Buffer | Ops.Slice | Ops.Param -> true
                  | _ -> false)
             | _ -> false
           in
           if trivial_reshape then None
           else if ctx.shapes node = None then None (* symbolic shapes *)
           else
             (* Devices that cannot take offset views are excluded upstream of
                here by the memory planner's device filter, so no per-device
                check is needed at this point. *)
             (match make_slice ctx.shapes src with
              | None -> None
              | Some view ->
                  let c = U.contiguous ~src:view () in
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
  match U.as_call node with
  | Some { info; body; _ } when info.precompile ->
      (match U.op body with Ops.Sink -> None | _ -> None)
  | _ -> None

(* Rule: tagged non-CONTIGUOUS/AFTER/STORE → wrap in CONTIGUOUS and
   move the tag onto it. *)
let wrap_tagged ctx node =
  match U.op node with
  | Ops.Contiguous | Ops.After | Ops.Store -> None
  | _ ->
      (match get_tags ctx node with
       | Some ts ->
           remove_tags ctx node;
           let c = U.contiguous ~src:node () in
           set_tags ctx c ts;
           Some c
       | None -> None)

(* Rule: CONTIGUOUS(AFTER) where AFTER's source has buffer identity →
   remove the redundant CONTIGUOUS and merge tags into the AFTER. *)
let merge_contiguous_after ctx node =
  match U.op node, first_src node with
  | Ops.Contiguous, Some a ->
      (match after_parts a with
       | Some (a_src, _) when U.has_buffer_identity a_src ->
           let merged = get_tags_or_empty ctx a @ get_tags_or_empty ctx node in
           remove_tags ctx node;
           set_tags ctx a merged;
           Some a
       | _ -> None)
  | _ -> None

(* Rule: AFTER(_, STORE(_, src)) → CONTIGUOUS(src) when the store's
   target is not a BUFFER. *)
let revert_store_to_contiguous ctx node =
  match after_parts node with
  | Some (_, deps) ->
      let store_src = List.find_map (fun d ->
        match U.as_store d with Some { value; _ } -> Some value | None -> None)
        deps
      in
      (match store_src with
       | None -> None
       | Some src ->
           let target = U.storage_base node in
           (match U.op target with
            | Ops.Buffer | Ops.Slice -> None
            | _ ->
                let c = U.contiguous ~src () in
                (match get_tags ctx node with
                 | Some ts -> set_tags ctx c ts
                 | None -> ());
                Some c))
  | _ -> None

(* Rule: CONTIGUOUS → BUFFER + STORE + AFTER.  The core allocation. *)
let contig_to_store_after ctx node =
  match U.op node, first_src node with
  | Ops.Contiguous, Some src ->
      let has_dev = ctx.devices src <> None in
      if not has_dev then None
      else
        if List.exists (fun dim -> U.const_int_value dim = Some 0) (U.shape src)
        then Some src
        else begin
          let dtype = U.commit_dtype node in
          let buf = buffer_like ctx src dtype in
          let store = U.store ~dst:buf ~value:(U.cast ~src ~dtype) () in
          let result = U.after ~src:buf ~deps:[store] in
          (match get_tags ctx node with
           | Some ts -> set_tags ctx result ts
           | None -> ());
          Some (U.cast ~src:result ~dtype:(U.dtype node))
        end
  | _ -> None

(* Rule: remove DETACH / CONTIGUOUS_BACKWARD. *)
let remove_detach node =
  match U.op node, first_src node with
  | (Ops.Detach | Ops.Contiguous_backward), Some src -> Some src
  | _ -> None

(* Phase 3 — finalize *)

(* Strip tags, map each original numbered node to its final buffer,
   and collect assigns. *)
let pm_finalize ctx node =
  if U.is_bound_var node then None
  else match U.op node with
  | Ops.After ->
      (match get_tags ctx node with
       | Some tag_indices ->
           remove_tags ctx node;
           let replace_uop = base_through_after node in
           List.iter (fun t ->
             let original = Hashtbl.find ctx.uop_tbl t in
             let buf = shrink_to replace_uop (U.shape original) in
             Hashtbl.replace ctx.buffer_map (U.tag original) buf)
             tag_indices
       | None -> ());
      ctx.assigns <- node :: ctx.assigns;
      None
  | _ -> None

(* Replace input BUFFER, SLICE(BUFFER), and bound variables with dense PARAMs for
   cache-key normalisation. *)
let pm_replace_buf ctx node =
  let replacement_slot b =
    let tag = U.tag b in
    match Hashtbl.find_opt ctx.replacement_slots tag with
    | Some idx -> idx
    | None ->
        let idx = List.length ctx.replacements in
        Hashtbl.replace ctx.replacement_slots tag idx;
        ctx.replacements <- (idx, b) :: ctx.replacements;
        idx
  in
  let replace_input b =
    let idx = replacement_slot b in
    let dtype = dtype_or_void b in
    let device =
      match U.as_slice b with
      | Some { src; _ } -> U.device_of src
      | None -> U.device_of b
    in
    let shape =
      match ctx.shapes b with
      | Some sh -> shape_node sh
      | None -> U.shape_to_shape_arg None
    in
    let addrspace = if U.is_bound_var b then D.Alu else replacement_addrspace b in
    let volatile =
      match U.Arg.as_param_arg (U.arg (U.buf_uop b)) with
      | Some param -> param.volatile
      | None -> false
    in
    match U.as_bind b, ctx.shapes b with
    | Some { var; _ }, _ ->
        let p = Option.get (U.Arg.as_param_arg (U.arg var)) in
        Some (U.replace var ~op:Ops.Param
          ~arg:(U.Arg.Param_arg { p with slot = idx; name = Some ("p" ^ string_of_int idx) }) ())
    (* A buffer is always numel-shaped: a scalar output is a size-1 buffer
       viewed as a scalar. Emitting a bare scalar PARAM loses the size-1
       dimension that scheduling needs to index the store at offset 0, so
       give it shape [1] and reshape to the scalar shape. *)
    | None, Some [] ->
        let param =
          U.param ~slot:idx ~dtype ~shape:(shape_node [ 1 ]) ?device
            ~addrspace ~volatile ()
        in
        Some (U.reshape ~src:param ~shape:(shape_node []))
    | None, _ -> Some (U.param ~slot:idx ~dtype ~shape ?device ~addrspace ~volatile ())
  in
  match U.op node, U.children node with
  | Ops.Buffer, _ when not (U.is_variable node) -> replace_input node
  | Ops.Slice, src :: _ when U.op src = Ops.Buffer -> replace_input node
  | Ops.After, _ when U.is_bound_var node -> replace_input node
  | _ -> None

(* Entry point *)

let transform_to_call (big_sink : U.t) : U.t * (int, U.t) Hashtbl.t =
  let shapes = concrete_shape in
  let devices = U.device_of in
  let bases = Hashtbl.create 16 in
  (match U.op big_sink with
   | Ops.Sink ->
       List.iter (fun x ->
         if not (dont_realize (U.op (base x))) then
           Hashtbl.replace bases (U.tag (multibase x)) ())
         (U.children big_sink)
   | _ -> ());
  (* Fresh allocations must not collide with buffers already numbered by
     hand in this graph; the slot counter itself is process-global so node
     identities stay unique across calls. *)
  U.reserve_buffer_slots
    (List.fold_left (fun acc x ->
         match U.as_buffer x, U.as_param x with
         | Some { buffer = { slot; _ }; _ }, _
         | None, Some { param = { slot; _ }; _ }
           when slot >= 0 ->
             max acc (slot + 1)
         | _ -> acc)
       0 (U.toposort big_sink));
  let ctx = {
    uop_tbl = Hashtbl.create 64;
    uop_count = 0;
    buffer_map = Hashtbl.create 64;
    bases;
    assigns = [];
    replacements = [];
    replacement_slots = Hashtbl.create 16;
    tags = Hashtbl.create 64;
    shapes;
    devices;
  } in
  (* Phase 1: number the nodes that need realization. *)
  let big_sink =
    U.graph_rewrite ~name:"add_tags" (add_tags ctx) big_sink in
  (* Phase 2: replace tagged nodes with buffer allocations. *)
  let big_sink =
    U.graph_rewrite ~name:"early_transform"
      ~on_rebuild:(propagate_tags ctx)
      (U.first_match [
        transform_precompiled_call ctx;
        contiguous_mops_to_slice ctx;
        wrap_tagged ctx;
        merge_contiguous_after ctx;
        revert_store_to_contiguous ctx;
        contig_to_store_after ctx;
        remove_detach;
      ]) big_sink in
  (* Phase 3a: finalize — strip tags and collect assigns. *)
  ignore (U.graph_rewrite ~name:"finalize" (pm_finalize ctx) big_sink);
  (* Phase 3b: replace buffers with PARAMs and wrap in a CALL. *)
  let assigns_sink = U.sink (List.rev ctx.assigns) in
  let body =
    (* A storage view is one argument. Match it before its underlying buffer
       becomes a PARAM and the view's identity can no longer be recovered. *)
    U.graph_rewrite ~name:"replace_bufs" ~bottom_up:true ~walk:true
      (pm_replace_buf ctx) assigns_sink in
  let args =
    ctx.replacements
    |> List.sort (fun (a, _) (b, _) -> Int.compare a b)
    |> List.map snd
  in
  let info =
    { U.grad_fxn = None; name = None; precompile = false;
      precompile_backward = false; dtype = Dtype.void; aux = None } in
  let ret = U.call ~body ~args ~info in
  (ret, ctx.buffer_map)
