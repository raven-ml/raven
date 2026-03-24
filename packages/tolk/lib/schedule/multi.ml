(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Multi-device sharding transformations.

   Transforms operations on MULTI-wrapped (sharded) buffers. *)

open Tolk_ir
module T = Tensor

let prod l = List.fold_left ( * ) 1 l

let last = function
  | [] -> None
  | l -> Some (List.nth l (List.length l - 1))

let index_of x l =
  let rec loop i = function
    | [] -> None
    | p :: _ when p = x -> Some i
    | _ :: rest -> loop (i + 1) rest
  in
  loop 0 l

let first_some fs =
  let rec loop = function
    | [] -> None
    | f :: rest -> (match f () with Some _ as r -> r | None -> loop rest)
  in
  loop fs

let err_axis_mismatch a b =
  Printf.sprintf "axis must match in assign %d != %d" a b

(* Helpers *)

let is_multi program id =
  match T.view program id with Multi _ -> true | _ -> false

let get_multi program id =
  match T.view program id with
  | Multi { src; axis; _ } -> Some (src, axis)
  | _ -> None

let ndev_of (devices : T.device option array) id =
  match devices.(id) with Some (Multi ds) -> List.length ds | _ -> 1

let devices_of (devices : T.device option array) id =
  match devices.(id) with Some (Multi ds) -> ds | _ -> []

(* Recursively computes the sharding axis a node inherits from its sources.
   Multi nodes define their axis directly. Elementwise and cast ops propagate
   from their source; binary/ternary ops take the last available axis. Reduce
   returns None if the reduction eliminates the sharded axis. Reshape remaps
   by matching cumulative dimension products: finds the output axis whose
   prefix-product equals the input axis's prefix-product, handling splits and
   merges of dimensions across the reshape boundary. Permute applies argsort. *)
let rec compute_axis program devices id =
  match T.view program id with
  | Multi { axis; _ } -> Some axis
  | Copy _ -> None
  | Param _ ->
    List.find_map
      (fun c -> match T.view program c with Multi { axis; _ } -> Some axis | _ -> None)
      (T.children program id)
  | Unary { src; _ } | Cast { src; _ } | Bitcast { src; _ }
  | Detach { src; _ } | Contiguous { src; _ } | Contiguous_backward { src; _ } ->
    compute_axis program devices src
  | Binary { lhs; rhs; _ } ->
    last (List.filter_map (fun s -> compute_axis program devices s) [ lhs; rhs ])
  | Ternary { a; b; c; _ } ->
    last (List.filter_map (fun s -> compute_axis program devices s) [ a; b; c ])
  | Reduce_axis { src; axes; _ } ->
    (match compute_axis program devices src with
     | Some ax when List.mem ax axes -> None
     | other -> other)
  | Reshape { src; shape; _ } ->
    (match compute_axis program devices src with
     | None -> None
     | Some src_axis ->
       Option.bind (T.extract_int_shape program shape) (fun new_shape ->
         let shapes_src =
           Option.value ~default:[] (T.compute_shapes program).(src)
         in
         let prior_prod = prod (List.filteri (fun i _ -> i < src_axis) shapes_src) in
         let acc = ref 1 in
         let new_axis = ref 0 in
         List.iteri (fun i s ->
           if !acc = prior_prod then new_axis := i;
           acc := !acc * s)
           new_shape;
         Some !new_axis))
  | Permute { src; order; _ } ->
    Option.bind (compute_axis program devices src) (fun ax -> index_of ax order)
  | Shrink { src; _ } | Flip { src; _ } | Expand { src; _ } | Pad { src; _ } ->
    compute_axis program devices src
  | _ -> None

(* Adjust shard-axis bounds: replace the shard-axis pair with (0, inner_size). *)
let adjust_shard_pairs shapes multi_src multi_axis pairs =
  List.mapi (fun i (s, e) ->
    if i = multi_axis then
      let inner_size = match shapes.(multi_src) with
        | Some ss when i < List.length ss -> List.nth ss i
        | _ -> e - s
      in
      (0, inner_size)
    else (s, e))
    pairs

(* Replace-allreduce patterns *)

let broadcast_copy b program (devices : T.device option array) = function
  | T.Copy { src; device; dtype } ->
    (match devices.(src), T.view program device with
     | Some (Single _), Device { device = Multi target_devs } ->
       let copies = List.map (fun d ->
         T.copy b ~src ~device:(T.device b (Single d)) ()) target_devs
       in
       Some (T.Mstack { srcs = copies; dtype })
     | _ -> None)
  | _ -> None

let copy_to_one b program (devices : T.device option array) = function
  | T.Copy { src; device; _ } ->
    (match devices.(src), T.view program device with
     | Some (Multi _), Device { device = Single _ } ->
       let copied = T.copy b ~src:(T.mselect b ~src ~index:0) ~device () in
       Some (T.view (T.finish b) copied)
     | _ -> None)
  | _ -> None

let mselect_mstack program = function
  | T.Mselect { src; index; _ } ->
    (match T.view program src with
     | Mstack { srcs; _ } ->
       Option.map (fun s -> T.view program s) (List.nth_opt srcs index)
     | _ -> None)
  | _ -> None

(* mselect(movement(s, ...), i) -> movement(mselect(s, i), ...) *)
let mselect_before_movement b program = function
  | T.Mselect { src; index; _ } ->
    let sel inner = T.mselect b ~src:inner ~index in
    let done_ id = Some (T.view (T.finish b) id) in
    (match T.view program src with
     | Reshape { src = inner; shape; _ } -> done_ (T.reshape b ~src:(sel inner) ~shape)
     | Expand { src = inner; shape; _ }  -> done_ (T.expand b ~src:(sel inner) ~shape)
     | Permute { src = inner; order; _ } -> done_ (T.permute b ~src:(sel inner) ~order)
     | Flip { src = inner; dims; _ }     -> done_ (T.flip b ~src:(sel inner) ~dims)
     | Pad { src = inner; before; after; _ } ->
       done_ (T.pad b ~src:(sel inner) ~before ~after)
     | Shrink { src = inner; before; after; _ } ->
       done_ (T.shrink b ~src:(sel inner) ~before ~after)
     | _ -> None)
  | _ -> None

(* Multi functions *)

let alu_multi b program v =
  let srcs = match v with
    | T.Unary { src; _ } -> [ src ]
    | Binary { lhs; rhs; _ } -> [ lhs; rhs ]
    | Ternary { a; b; c; _ } -> [ a; b; c ]
    | _ -> []
  in
  if srcs = [] || not (List.exists (is_multi program) srcs) then None
  else
    let axes = List.filter_map (fun s -> get_multi program s |> Option.map snd) srcs in
    match last axes with
    | None -> None
    | Some axis ->
      let unwrapped = List.map (fun s ->
        match T.view program s with
        | Multi { src = inner; _ } -> inner
        | _ -> s) srcs
      in
      let result = match v, unwrapped with
        | T.Unary { op; _ }, [ s ] -> T.unary b ~op ~src:s
        | Binary { op; _ }, [ l; r ] -> T.binary b ~op ~lhs:l ~rhs:r
        | Ternary { op; _ }, [ a; b_; c ] -> T.ternary b ~op ~a ~b:b_ ~c
        | _ -> failwith "alu_multi: unexpected"
      in
      Some (T.view (T.finish b) (T.multi b ~src:result ~axis))

let reduce_multi b ~(devices : T.device option array) program root_view multi_id =
  match root_view, T.view program multi_id with
  | T.Reduce_axis { op; axes; _ },
    Multi { src = multi_src; axis = multi_axis; _ } ->
    let reduced = T.reduce_axis b ~src:multi_src ~op ~axes in
    if List.mem multi_axis axes then
      let dev = match devices.(multi_id) with
        | Some (Multi ds) -> T.device b (Multi ds)
        | Some (Single d) -> T.device b (Single d)
        | None -> failwith "reduce_multi: no device"
      in
      Some (T.allreduce b ~src:reduced ~device:dev ~op)
    else
      Some (T.multi b ~src:reduced ~axis:multi_axis)
  | _ -> None

let reshape_multi b ~shapes ~devices program root_view multi_id =
  match root_view, T.view program multi_id with
  | T.Reshape { shape; _ }, Multi { src = multi_src; _ } ->
    Option.bind (T.extract_int_shape program shape) (fun new_shape ->
      let multi_shape = Option.value ~default:[] shapes.(multi_id) in
      if prod multi_shape <> prod new_shape then
        failwith "reshape must maintain prod(shape)";
      let new_axis = compute_axis program devices multi_id in
      let inner_shape = match new_axis with
        | Some ax ->
          let ndev = ndev_of devices multi_id in
          List.mapi (fun i s -> if i = ax then s / ndev else s) new_shape
        | None -> new_shape
      in
      let reshaped =
        T.reshape b ~src:multi_src ~shape:(Allreduce.emit_shape b inner_shape)
      in
      match new_axis with
      | Some ax -> Some (T.multi b ~src:reshaped ~axis:ax)
      | None -> Some reshaped)
  | _ -> None

let expand_multi b ~shapes program root_view multi_id =
  match root_view, T.view program multi_id with
  | T.Expand { shape; _ }, Multi { src = multi_src; axis = multi_axis; _ } ->
    Option.bind (T.extract_int_shape program shape) (fun new_shape ->
      let inner_shape = match shapes.(multi_src) with
        | Some src_shape ->
          List.mapi (fun i s ->
            if i = multi_axis then List.nth src_shape multi_axis else s)
            new_shape
        | None -> new_shape
      in
      let expanded =
        T.expand b ~src:multi_src ~shape:(Allreduce.emit_shape b inner_shape)
      in
      Some (T.multi b ~src:expanded ~axis:multi_axis))
  | _ -> None

let pad_multi b program root_view multi_id =
  match root_view, T.view program multi_id with
  | T.Pad { before; after; _ }, Multi { src = multi_src; axis = multi_axis; _ } ->
    (match T.extract_int_shape program before, T.extract_int_shape program after with
     | Some bs, Some es ->
       let pairs = List.combine bs es in
       if multi_axis >= 0
          && multi_axis < List.length pairs
          && List.nth pairs multi_axis <> (0, 0)
       then failwith "padding not supported on sharded axis";
       let bef, aft = Allreduce.emit_pairs b pairs in
       Some (T.multi b ~src:(T.pad b ~src:multi_src ~before:bef ~after:aft) ~axis:multi_axis)
     | _ -> None)
  | _ -> None

let shrink_multi b ~shapes ~devices program root_view multi_id =
  match root_view, T.view program multi_id with
  | T.Shrink { before; after; _ }, Multi { src = multi_src; axis = multi_axis; _ } ->
    (match T.extract_int_shape program before, T.extract_int_shape program after with
     | Some starts, Some ends ->
       let pairs = List.combine starts ends in
       let multi_shape = Option.value ~default:[] shapes.(multi_id) in
       let shard_bounds = List.nth_opt pairs multi_axis in
       let full_axis = match shard_bounds with
         | Some (s, e) ->
           s = 0 && multi_axis < List.length multi_shape
           && e = List.nth multi_shape multi_axis
         | None -> true
       in
       let adjusted = adjust_shard_pairs shapes multi_src multi_axis pairs in
       let bef, aft = Allreduce.emit_pairs b adjusted in
       if not full_axis && shard_bounds <> None then
         let dev_list = devices_of devices multi_id in
         let dev = T.device b
           (Single (match dev_list with [] -> "CPU" | d :: _ -> d))
         in
         Some (T.shrink b ~src:(T.copy b ~src:multi_src ~device:dev ()) ~before:bef ~after:aft)
       else
         Some (T.multi b ~src:(T.shrink b ~src:multi_src ~before:bef ~after:aft) ~axis:multi_axis)
     | _ -> None)
  | _ -> None

let permute_multi b ~devices program root_view multi_id =
  match root_view, T.view program multi_id with
  | T.Permute { order; _ }, Multi { src = multi_src; _ } ->
    let permuted = T.permute b ~src:multi_src ~order in
    let root_axis = match compute_axis program devices multi_id with
      | Some ax -> Option.value ~default:ax (index_of ax order)
      | None -> 0
    in
    Some (T.multi b ~src:permuted ~axis:root_axis)
  | _ -> None

let flip_multi b program root_view multi_id =
  match root_view, T.view program multi_id with
  | T.Flip { dims; _ }, Multi { src = multi_src; axis = multi_axis; _ } ->
    if multi_axis >= 0 && multi_axis < List.length dims && List.nth dims multi_axis
    then failwith "flipping not supported on sharded axis";
    Some (T.multi b ~src:(T.flip b ~src:multi_src ~dims) ~axis:multi_axis)
  | _ -> None

let copy_multi b program multi_id device_id =
  match T.view program multi_id with
  | Multi { src = multi_src; axis = multi_axis; _ } ->
    if multi_axis < 0 then failwith "all multi ops have axis";
    let dev = match T.view program device_id with
      | Device { device } -> T.device b device
      | _ -> device_id
    in
    Some (T.allreduce b ~src:multi_src ~device:dev ~op:`Add)
  | _ -> None

let assign_multi b program dest_id src_id =
  match T.view program dest_id, T.view program src_id with
  | Multi { src = dest_src; axis = dest_axis; _ },
    Multi { src = src_src; axis = src_axis; _ } ->
    if dest_axis <> src_axis then failwith (err_axis_mismatch dest_axis src_axis);
    let assigned = T.assign b ~target:dest_src ~value:src_src () in
    Some (T.multi b ~src:assigned ~axis:src_axis)
  | _ -> None

let passthrough_multi b program root_view multi_id =
  match T.view program multi_id with
  | Multi { src; axis; _ } ->
    let wrap inner = Some (T.multi b ~src:inner ~axis) in
    (match root_view with
     | T.Cast { dtype; _ }               -> wrap (T.cast b ~src ~dtype)
     | Bitcast { dtype; _ }              -> wrap (T.bitcast b ~src ~dtype)
     | Contiguous _                      -> wrap (T.contiguous b ~src ())
     | Detach _                          -> wrap (T.detach b ~src)
     | Contiguous_backward _             -> wrap (T.contiguous_backward b ~src)
     | After { deps; _ }                 -> wrap (T.after b ~src ~deps)
     | Store { value; _ }                -> wrap (T.store b ~dst:src ~value)
     | _ -> None)
  | _ -> None

(* Combined pattern matcher *)

let multi_pm b ~shapes ~devices program v =
  let id_to_view id = T.view (T.finish b) id in
  first_some [
    (fun () -> mselect_mstack program v);
    (fun () -> mselect_before_movement b program v);
    (fun () -> broadcast_copy b program devices v);
    (fun () -> copy_to_one b program devices v);
    (fun () -> alu_multi b program v);
    (fun () ->
      match v with
      | T.Reduce_axis { src; _ } when is_multi program src ->
        Option.map id_to_view (reduce_multi b ~devices program v src)
      | Reshape { src; _ } when is_multi program src ->
        Option.map id_to_view (reshape_multi b ~shapes ~devices program v src)
      | Expand { src; _ } when is_multi program src ->
        Option.map id_to_view (expand_multi b ~shapes program v src)
      | Pad { src; _ } when is_multi program src ->
        Option.map id_to_view (pad_multi b program v src)
      | Shrink { src; _ } when is_multi program src ->
        Option.map id_to_view (shrink_multi b ~shapes ~devices program v src)
      | Permute { src; _ } when is_multi program src ->
        Option.map id_to_view (permute_multi b ~devices program v src)
      | Flip { src; _ } when is_multi program src ->
        Option.map id_to_view (flip_multi b program v src)
      | After { src = target; deps; _ }
        when List.exists (fun d ->
          match T.view program d with Store _ -> true | _ -> false) deps
          && is_multi program target ->
        let value = List.find_map (fun d ->
          match T.view program d with
          | Store { value; _ } -> Some value | _ -> None) deps in
        (match value with
         | Some value when is_multi program value ->
             Option.map id_to_view (assign_multi b program target value)
         | _ -> None)
      | Copy { src; device; _ } when is_multi program src ->
        Option.map id_to_view (copy_multi b program src device)
      | Allreduce { src; device; op; _ } when is_multi program src ->
        (match get_multi program src with
         | Some (inner, ax) ->
           Some (id_to_view (T.multi b ~src:(T.allreduce b ~src:inner ~device ~op) ~axis:ax))
         | None -> None)
      | After { src; _ } when is_multi program src ->
        Option.map id_to_view (passthrough_multi b program v src)
      | Store { dst; _ } when is_multi program dst ->
        Option.map id_to_view (passthrough_multi b program v dst)
      | (Cast { src; _ } | Bitcast { src; _ } | Contiguous { src; _ }
        | Detach { src; _ } | Contiguous_backward { src; _ })
        when is_multi program src ->
        Option.map id_to_view (passthrough_multi b program v src)
      | _ -> None);
  ]
