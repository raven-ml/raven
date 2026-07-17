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

let int_ n = U.const (C.int D.weakint n)
let index_ n = U.const (C.int D.index n)
let shape_prod = List.fold_left ( * ) 1
let dtype_or_void n = U.dtype n

let shape_node dims =
  match List.map int_ dims with [ d ] -> d | ds -> U.stack ds

let concrete_shape n =
  try
    let dims = U.shape n in
    let rec loop acc = function
      | [] -> Some (List.rev acc)
      | dim :: dims -> (
          match U.const_int_value dim with
          | Some d -> loop (d :: acc) dims
          | None -> None)
    in
    loop [] dims
  with Invalid_argument _ -> None

(* Address space of an input buffer replaced by a PARAM: the node's own
   address space, defaulting to global for nodes that carry none (a BIND, a
   plain scalar). *)
let replacement_addrspace node =
  match U.addrspace node with Some a -> a | None -> D.Global

let is_op op n = Ops.equal (U.op n) op

let src0 n =
  match U.children n with
  | x :: _ -> Some x
  | [] -> None

let src1 n =
  match U.children n with
  | _ :: x :: _ -> Some x
  | _ -> None

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

let const_ints n =
  match U.op n with
  | Ops.Stack ->
      let rec loop acc = function
        | [] -> Some (List.rev acc)
        | x :: xs ->
            (match U.const_int_value x with
             | Some v -> loop (v :: acc) xs
             | None -> None)
      in
      loop [] (U.children n)
  | Ops.Const -> Option.map (fun v -> [ v ]) (U.const_int_value n)
  | _ -> None

let shrink_pairs n =
  match U.op n, src1 n, U.children n with
  | Ops.Shrink, Some offset, [ _src; _offset; size ] ->
      (match const_ints offset, const_ints size with
       | Some offsets, Some sizes ->
           (try Some (List.combine offsets sizes) with Invalid_argument _ -> None)
       | _ -> None)
  | _ -> None

let map_order xs order =
  try Some (List.map (List.nth xs) order) with Failure _ -> None

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

(* Is the base of [x] a buffer or buffer-view? *)
let has_buffer_identity x =
  match U.op (base x) with
  | Ops.Buffer | Ops.Slice | Ops.Param -> true
  | _ -> false

(* Ops that do not need buffer realization. *)
let dont_realize = function
  | Ops.Const | Ops.Buffer | Ops.Param | Ops.Bind | Ops.After -> true
  | _ -> false

let compute_shapes root =
  let cache = U.Ref_tbl.create 64 in
  let rec shape n =
    match U.Ref_tbl.find_opt cache n with
    | Some s -> s
    | None ->
        let s =
          match U.op n with
          | Ops.Buffer ->
              (match (U.as_buffer n : U.buffer_view option) with
               | Some { shape; _ } ->
                   (match const_ints shape with
                    | Some _ as s -> s
                    | None -> None)
               | _ -> None)
          | Ops.Param ->
              (match (U.as_param n : U.param_view option) with
               | Some { shape; _ } ->
                   (match const_ints shape with
                    | Some _ as s -> s
                    | None -> None)
               | _ -> None)
          | Ops.Slice ->
              Option.map (fun (v : U.slice_view) -> [ v.size ]) (U.as_slice n)
          | Ops.Reshape | Ops.Expand ->
              (match src1 n with Some sh -> const_ints sh | None -> None)
          | Ops.Pad ->
              (match U.children n with
               | [ _src; _offset; size ] -> const_ints size
               | _ -> None)
          | Ops.Shrink ->
              (match shrink_pairs n with
               | Some pairs -> Some (List.map snd pairs)
               | None -> Option.bind (src0 n) shape)
          | Ops.Permute ->
              let order = match U.arg n with U.Arg.Ints i -> i | _ -> [] in
              Option.bind (Option.bind (src0 n) shape) (fun s ->
                map_order s order)
          | Ops.Flip ->
              Option.bind (src0 n) shape
          | Ops.Multi ->
              (* A MULTI presents the global shape: [U.shape] multiplies the
                 shard axis of its per-shard source back up by the device
                 count, whatever form the source takes (a symbolic
                 [_device_num] view or a per-shard buffer). *)
              concrete_shape n
          | Ops.Detach | Ops.Contiguous | Ops.Contiguous_backward
          | Ops.After ->
              Option.bind (src0 n) shape
          | _ -> concrete_shape n
        in
        U.Ref_tbl.add cache n s;
        s
  in
  ignore (shape root);
  shape

(* Shrink [src] to [target_shape].  Each dimension is kept from 0 to
   the target size — a no-op when shapes already match. *)
let shrink_to shapes src target_shape =
  match shapes src with
  | Some s when s = target_shape -> src
  | _ ->
      let before = shape_node (List.map (fun _ -> 0) target_shape) in
      let size = shape_node target_shape in
      U.shrink ~src ~offset:before ~size

(* If movement ops on [src] collapse to a contiguous range backed by a
   buffer, return the element offset.  Returns [None] when the view is
   non-contiguous or too complex to analyse statically. *)
let contiguous_view_offset shapes src =
  (* Walk the movement-op chain and track whether the view stays
     contiguous.  We handle the common patterns; the full analysis
     would require the rangeify index pipeline. *)
  let rec walk node =
    match U.op node, first_src node with
    | (Ops.Buffer | Ops.Param), _ -> Some 0
    | Ops.Slice, _ ->
        (match U.as_slice node with
         | Some { src; offset; _ } ->
             (match U.const_int_value offset, walk src with
              | Some off, Some base_off -> Some (base_off + off)
              | _ -> None)
         | None -> None)
    | Ops.Reshape, Some src -> walk src
    | Ops.Shrink, Some src ->
        let inner = match shapes src with Some s -> s | None -> [] in
        if inner = [] then None
        else
          let pairs = match shrink_pairs node with Some p -> p | None -> [] in
          if pairs = [] then None
          else
            let n = List.length pairs in
            (* All leading dimensions must be kept in full. *)
            let all_full = List.for_all2 (fun (offset, size) dim ->
              offset = 0 && size = dim)
              (List.filteri (fun i _ -> i < n - 1) pairs)
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
  let base = base src in
  match U.op base with
  | Ops.Buffer | Ops.Slice | Ops.Param -> walk src
  | _ -> None

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
  match after_parts node with
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
      let dims =
        try U.shape src
        with Invalid_argument _ -> failwith "buffer_like: unknown shape"
      in
      let dev = match ctx.devices src with
        | Some d -> d | None -> failwith "buffer_like: unknown device" in
      let max_shape = List.map U.vmax dims in
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
    | Ops.Multi -> U.Arg.as_int (U.arg src)
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
  let buf = shrink_to ctx.shapes buf shard_shape in
  match axis with
  | Some ax when ndev > 1 -> U.multi ~src:buf ~axis:ax
  | _ -> buf

(* If movement ops on [src] collapse to a contiguous range, return a
   Slice reshaped to [src]'s shape. *)
let make_slice shapes src =
  match contiguous_view_offset shapes src with
  | None -> None
  | Some offset ->
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
       | Some (a_src, _) when has_buffer_identity a_src ->
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
           let rec find_target n =
             match U.op n, first_src n, after_parts n with
             | Ops.Bitcast, Some src, _ -> find_target (base src)
             (* A multi-device store target wraps its per-shard buffer in
                MULTI; look through it, or the AFTER just built for it
                reverts and the rewrite cycles. *)
             | Ops.Multi, Some src, _ -> find_target (base src)
             | _, _, Some (src, _) -> find_target (base src)
             | _ -> n
           in
           let target = find_target node in
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
        let shape = match ctx.shapes src with Some s -> s | None -> [] in
        if shape_prod shape = 0 then Some src
        else begin
          let buf = buffer_like ctx src (U.dtype node) in
          let store = U.store ~dst:buf ~value:src () in
          let result = U.after ~src:buf ~deps:[store] in
          (match get_tags ctx node with
           | Some ts -> set_tags ctx result ts
           | None -> ());
          Some result
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
  match U.op node with
  | Ops.After ->
      (match get_tags ctx node with
       | Some tag_indices ->
           remove_tags ctx node;
           let replace_uop = base_through_after node in
           List.iter (fun t ->
             let original = Hashtbl.find ctx.uop_tbl t in
             let original_shape =
               match ctx.shapes original with Some s -> s | None -> [] in
             let buf = shrink_to ctx.shapes replace_uop original_shape in
             Hashtbl.replace ctx.buffer_map (U.tag original) buf)
             tag_indices
       | None -> ());
      ctx.assigns <- node :: ctx.assigns;
      None
  | _ -> None

(* Replace input BUFFER, SLICE(BUFFER), and BIND with dense PARAMs for
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
    let addrspace = replacement_addrspace b in
    match U.as_bind b, ctx.shapes b with
    | Some { var; _ }, _ ->
        (* A bound variable keeps its name and range so the kernel graph can
           recover the canonical variable; the value is stripped so different
           bind values hit the same schedule cache. *)
        let vmin_vmax, name =
          match U.as_param var with
          | Some { param = { vmin_vmax; name; _ }; _ } -> vmin_vmax, name
          | None -> None, None
        in
        Some
          (U.param ~slot:idx ~dtype ~shape ?device ?vmin_vmax ?name ~addrspace
             ())
    (* A buffer is always numel-shaped: a scalar output is a size-1 buffer
       viewed as a scalar. Emitting a bare scalar PARAM loses the size-1
       dimension that scheduling needs to index the store at offset 0, so
       give it shape [1] and reshape to the scalar shape. *)
    | None, Some [] ->
        let param =
          U.param ~slot:idx ~dtype ~shape:(shape_node [ 1 ]) ?device
            ~addrspace ()
        in
        Some (U.reshape ~src:param ~shape:(shape_node []))
    | None, _ -> Some (U.param ~slot:idx ~dtype ~shape ?device ~addrspace ())
  in
  match U.op node, U.children node with
  | Ops.Buffer, _ ->
      replace_input node
  | Ops.Slice, src :: _ when U.op src = Ops.Buffer -> replace_input node
  | Ops.Bind, [ var; v ] when is_op Ops.Param var && is_op Ops.Const v ->
      replace_input node
  | _ -> None

(* Entry point *)

let transform_to_call (big_sink : U.t) : U.t * (int, U.t) Hashtbl.t =
  let shapes = compute_shapes big_sink in
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
    U.graph_rewrite ~name:"replace_bufs" ~walk:true (pm_replace_buf ctx)
      assigns_sink in
  let args =
    ctx.replacements
    |> List.sort (fun (a, _) (b, _) -> Int.compare a b)
    |> List.map snd
  in
  let info =
    { U.grad_fxn = None; name = None; precompile = false;
      precompile_backward = false; aux = None } in
  let ret = U.call ~body ~args ~info in
  (ret, ctx.buffer_map)
