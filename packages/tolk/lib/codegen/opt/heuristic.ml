(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Port of tinygrad/codegen/opt/heuristic.py to the tolk_uop IR. *)

open Tolk_uop
module U = Uop
module P = Postrange

(* Environment *)

let nolocals_var = Helpers.Context_var.int ~key:"NOLOCALS" ~default:0

let getenv key default = Helpers.getenv key default

let use_tc () = getenv "TC" 1

let tc_select () = getenv "TC_SELECT" (-1)
let tc_opt () = getenv "TC_OPT" 0
let image () = getenv "IMAGE" 0 <> 0
let occupancy_floor () = getenv "OCCUPANCY_FLOOR" 4096
let mv_blocksize () = getenv "MV_BLOCKSIZE" 4
let mv_threads_per_row () = getenv "MV_THREADS_PER_ROW" 8
let mv_rows_per_thread () = getenv "MV_ROWS_PER_THREAD" 4
let mv () = getenv "MV" 1

(* Helpers *)

let const_int_or default u =
  match U.const_int_value u with Some n -> n | None -> default

let last lst = List.nth lst (List.length lst - 1)

let nth_size k axis = List.nth (P.full_shape k) axis
let nth_rng k axis = List.nth (P.rngs k) axis

let prod_at shape axes =
  List.fold_left (fun acc a -> const_int_or 1 (List.nth shape a) * acc) 1 axes

let is_range u = Option.is_some (U.as_range u)

let range_size u = match U.as_range u with
  | Some r -> r.size
  | None -> invalid_arg "range_size: not a range"

let is_const u = U.op u = Ops.Const

let divides_by rng n = U.divides (range_size rng) n <> None

let index_of_rng rngs rng =
  match List.find_index (fun r -> r == rng) rngs with
  | Some i -> i
  | None -> -1

(* Unwrap a Where/Invalid guard [Where (_, b, Invalid)] -> [b]. Acts as an
   identity on any other node. *)
let strip_where_invalid u =
  match U.op u, U.src u with
  | Ops.Where, [| _; b; c |] when
      (match U.op c, U.arg c with
       | Ops.Const, U.Arg.Value v ->
           (match Const.view v with Invalid -> true | _ -> false)
       | _ -> false) -> b
  | _ -> u

(* Kernel-stage buffer [buf] is an INDEX node; extract its idx expression,
   stripping any Where/Invalid guard. *)
let get_idx buf = match U.as_index buf with
  | Some { idxs = [ idx ]; _ } -> Some (strip_where_invalid idx)
  | Some _ -> None
  | None -> None

(* [u] is [rng * c] or [c * rng] for an integer constant [c]; return [c]. *)
let mul_by_rng_const rng u = match U.op u, U.src u with
  | Ops.Mul, [| a; b |] when a == rng && is_const b -> U.const_int_value b
  | Ops.Mul, [| a; b |] when b == rng && is_const a -> U.const_int_value a
  | _ -> None

let try_apply k opt =
  try ignore (P.apply_opt k opt : _ option) with P.Opt_error _ -> ()

(* Find the first size in [sizes] that divides [rng], apply [mk_opt] at that
   axis, and return the apply_opt result. *)
let try_opt_on_rng tk rng sizes mk_opt =
  match List.find_opt (divides_by rng) sizes with
  | None -> None
  | Some sz ->
      let axis = index_of_rng (P.rngs tk) rng in
      if axis < 0 then None else P.apply_opt tk (mk_opt axis sz)

(* Try tensor core optimization. Returns [Some k] on success. *)
let try_tensor_cores k =
  let reduce_axes = P.axes_of k [ Axis_type.Group_reduce; Axis_type.Reduce ] in
  let use_tc = use_tc () in
  let tc_opt = tc_opt () in
  if use_tc <= 0 || (List.length reduce_axes <> 1 && tc_opt < 1) then None
  else
    let tk = P.copy k in
    let tc_result =
      try
        P.apply_opt tk
          (U.Opt.Tc
             { axis = 0; tc_select = tc_select (); tc_opt; use_tc })
      with P.Opt_error _ -> None
    in
    match tc_result with
    | Some (n_rng, m_rng) ->
        let rngs = [| n_rng; m_rng |] in
        List.iter (fun d ->
          let upcast axis amount = U.Opt.Upcast { axis; amount } in
          match try_opt_on_rng tk rngs.(d) [ 5; 4; 3; 2 ] upcast with
          | Some (replaced, _) -> rngs.(d) <- replaced
          | None -> ()) [ 1; 0 ];
        let local axis amount = U.Opt.Local { axis; amount } in
        ignore (try_opt_on_rng tk rngs.(0) [ 4; 2 ] local);
        Some tk
    | None when P.tensor_core tk <> None -> Some tk
    | None -> None

let is_valid_image_buf k buf = match U.as_index buf with
  | Some { ptr; _ } ->
      Coalese.image_valid_dims
        ~image_pitch_alignment:(Renderer.image_pitch_alignment (P.ren k))
        ~base:(U.dtype ptr)
        ~size:(List.fold_left ( * ) 1 (U.max_shape ptr))
        ()
      <> []
  | None -> false

let upcast_image_buf k buf =
  match get_idx buf with
  | None -> ()
  | Some idx ->
      let axes = List.filter_map (fun c ->
        if is_range c && (U.vmax c + 1) mod 4 = 0 then
          let i = index_of_rng (P.rngs k) c in
          if i >= 0 then Some i else None
        else None) (U.split_uop idx Ops.Add) in
      match axes with
      | [] -> ()
      | axis :: _ when List.mem axis (P.upcastable_dims k) ->
          ignore (P.apply_opt k (U.Opt.Upcast { axis; amount = 4 }))
      | axis :: _ ->
          match List.find_index (( = ) axis) (P.unrollable_dims k) with
          | Some ui -> ignore (P.apply_opt k (U.Opt.Unroll { axis = ui; amount = 4 }))
          | None -> ()

let upcast_images k =
  if image () then
    List.iter (fun b -> if is_valid_image_buf k b then upcast_image_buf k b)
      (P.bufs k)

let index_operand u =
  match U.as_index u with
  | Some _ -> Some u
  | None -> None

(* Detect matrix-vector pattern: reduce(add, mul(INDEX, INDEX)) where the
   first reduce range appears as an addend in idx0, and all idx0 ranges
   appear in idx1. Returns the first reduce range on match. *)
let detect_matvec k =
  let ( let* ) = Option.bind in
  let* red = P.reduceop k in
  let* rv = U.as_reduce red in
  if rv.op <> Ops.Add then None
  else match U.op rv.src, U.src rv.src with
  | Ops.Mul, [| in0_raw; in1_raw |] ->
      let* in0 = index_operand in0_raw in
      let* in1 = index_operand in1_raw in
      let* idx0 = get_idx in0 in
      let* idx1 = get_idx in1 in
      (match P.ranges_of k [ Axis_type.Reduce ] with
       | [] -> None
       | first_red :: _ ->
           let idx0_rngs = U.ranges idx0 in
           let idx1_rngs = U.ranges idx1 in
           let first_red_in_addends =
             List.exists (fun u -> u == first_red) (U.split_uop idx0 Ops.Add)
           in
           let idx0_ranges_covered =
             List.for_all (fun r -> List.memq r idx1_rngs) idx0_rngs
           in
           if first_red_in_addends && idx0_ranges_covered then Some first_red
           else None)
  | _ -> None

(* Apply matvec opts (GROUP + LOCAL + UPCAST) if the pattern matches. *)
let try_matvec k =
  let mv = mv () in
  let mv_blocksize = mv_blocksize () in
  let mv_threads_per_row = mv_threads_per_row () in
  let mv_rows_per_thread = mv_rows_per_thread () in
  let mv_off =
    mv = 0
    || (mv_blocksize <= 1 && mv_threads_per_row <= 1 && mv_rows_per_thread <= 1)
  in
  if not (Renderer.has_local (P.ren k)) || mv_off
     || List.length (P.full_shape k) < 2
     || not (Renderer.has_shared (P.ren k))
  then None
  else
    let ( let* ) = Option.bind in
    let* first_red = detect_matvec k in
    let tile = mv_blocksize * mv_rows_per_thread in
    let* gi = List.find_opt (fun gi ->
      divides_by first_red mv_threads_per_row
      && U.divides (nth_size k gi) tile <> None)
      (P.axes_of k [ Axis_type.Global ]) in
    if mv_threads_per_row > 1 then
      try_apply k (U.Opt.Group { axis = 0; amount = mv_threads_per_row });
    if mv_blocksize > 1 then
      ignore (P.apply_opt k (U.Opt.Local { axis = gi; amount = mv_blocksize }));
    if mv_rows_per_thread > 1 then
      ignore
        (P.apply_opt k (U.Opt.Upcast { axis = gi; amount = mv_rows_per_thread }));
    Some k

(* Try GROUPTOP if output shape is small. *)
let try_grouping k =
  let threshold =
    if Helpers.Context_var.get nolocals_var <> 0 then 240 else 2048
  in
  if prod_at (P.output_shape k) (P.upcastable_dims k) <= threshold then
    (try List.iter (fun axis ->
      try
        ignore (P.apply_opt k (U.Opt.Grouptop { axis; amount = 16 }));
        raise_notrace Exit
      with P.Opt_error _ -> ()) [ 0; 1; 2 ]
     with Exit -> ());
  P.group_for_reduces k > 0

(* Upcast small masked dims. *)
let upcast_masked k =
  let ast_slice = U.backward_slice (P.ast k) in
  let is_masked rng =
    List.exists (fun u -> match U.op u, U.src u with
      | Ops.Where, [| cond; _; _ |] ->
          List.exists (fun n -> n == rng) (U.backward_slice cond)
      | _ -> false) ast_slice
  in
  (* Walk upcastable dims; collect small masked axes whose cumulative product
     stays under 7*7. Built reversed so iterating applies the leading axes
     last, matching the original [to_upcast[::-1]] order. *)
  let to_upcast =
    List.fold_left (fun acc axis ->
      let sz = const_int_or 0 (nth_size k axis) in
      let image_occupancy_ok =
        if not (image ()) then true
        else
          match List.nth (P.axis_types k) axis with
          | Axis_type.Global ->
              let global_upcast =
                prod_at (P.full_shape k)
                  (List.filter
                     (fun a -> List.nth (P.axis_types k) a = Axis_type.Global)
                     acc)
                * sz
              in
              let global_prod =
                prod_at (P.full_shape k) (P.axes_of k [ Axis_type.Global ])
              in
              global_prod / max global_upcast 1 >= occupancy_floor ()
          | _ -> true
      in
      if sz > 7 || not image_occupancy_ok then acc
      else if is_masked (nth_rng k axis)
              && prod_at (P.full_shape k) acc * sz <= 49
      then axis :: acc
      else acc) [] (P.upcastable_dims k)
  in
  List.iter (fun axis ->
    ignore (P.apply_opt k (U.Opt.Upcast { axis; amount = 0 })))
    to_upcast

(* Sum of stride-like contributions of [rng] to [idx]: [rng] itself counts as
   1, [rng * const] or [const * rng] counts as [const], everything else 0. *)
let stride_contribution rng idx =
  List.fold_left (fun acc c ->
    if c == rng then acc + 1
    else match mul_by_rng_const rng c with
    | Some n -> acc + n
    | None -> acc) 0 (U.split_uop idx Ops.Add)

(* A buffer broadcasts on [rng] iff [rng] does not appear in its index but
   every upcast/unroll range does. *)
let buf_broadcasts_on rng upcast_unroll b = match get_idx b with
  | None -> false
  | Some idx ->
      let bslice = U.backward_slice idx in
      not (List.memq rng bslice)
      && List.for_all (fun r2 -> List.memq r2 bslice) upcast_unroll

let axis_choice k axis amt =
  if const_int_or 0 (nth_size k axis) mod amt <> 0 then None
  else
    let rng = nth_rng k axis in
    let upcast_unroll = P.ranges_of k [ Axis_type.Upcast; Axis_type.Unroll ] in
    if not (List.exists (buf_broadcasts_on rng upcast_unroll) (P.bufs k)) then
      None
    else
      let num_strides = ref 0 in
      let sum_strides = ref 0 in
      List.iter (fun b -> match get_idx b with
        | None -> ()
        | Some idx ->
            if List.memq rng (U.backward_slice idx) then incr num_strides;
            sum_strides := !sum_strides + stride_contribution rng idx)
        (P.bufs k);
      Some (!num_strides, !sum_strides, axis, amt)

(* Upcast non-reduce axes based on stride analysis: prefer axes where some
   buffer broadcasts (stride 0) while all upcast/unroll axes have nonzero
   stride. Pick the axis with fewest strides first. *)
let upcast_heuristic k =
  let is_dsp = Renderer.device (P.ren k) = "DSP" in
  let upcasted = Hashtbl.create 8 in
  let continue_ = ref true in
  while
    !continue_
    && prod_at (P.output_shape k) (P.upcastable_dims k) >= 1024
    && P.upcast_size k < 32
  do
    let upcast_amounts =
      if is_dsp then (if Hashtbl.length upcasted = 0 then [ 128 ] else [])
      else [ 3; 4 ]
    in
    let choices = List.concat_map (fun axis ->
      if Hashtbl.mem upcasted axis then []
      else List.filter_map (fun amt -> axis_choice k axis amt) upcast_amounts)
      (P.upcastable_dims k)
      |> List.sort compare
    in
    match choices with
    | [] -> continue_ := false
    | (_, _, axis, amt) :: _ ->
        ignore (P.apply_opt k (U.Opt.Upcast { axis; amount = amt }));
        Hashtbl.replace upcasted axis ()
  done

(* Unroll last reduce dim if small. *)
let unroll_reduce k =
  try
    let ud = P.unrollable_dims k in
    if ud <> []
       && (P.upcast_size k <= 4 || P.axes_of k [ Axis_type.Unroll ] = [])
       && P.upcast_size k < 64
    then begin
      let s = const_int_or 0 (nth_size k (last ud)) in
      if s <= 32 then begin
        ignore (P.apply_opt k
          (U.Opt.Unroll { axis = List.length ud - 1; amount = 0 }));
        let ud2 = P.unrollable_dims k in
        if ud2 <> [] && s <= 3 && const_int_or 0 (nth_size k (last ud2)) <= 3
        then
          ignore (P.apply_opt k
            (U.Opt.Unroll { axis = List.length ud2 - 1; amount = 0 }))
      end
      else if const_int_or 0 (nth_size k (last ud)) mod 4 = 0 then
        ignore (P.apply_opt k
          (U.Opt.Unroll { axis = List.length ud - 1; amount = 4 }))
    end
  with P.Opt_error _ -> ()

(* Upcast by 4 if nothing is upcasted yet. *)
let upcast_default k =
  let ud = P.upcastable_dims k in
  if P.upcasted k = 0 && ud <> []
     && const_int_or 0 (nth_size k (last ud)) mod 4 = 0
  then ignore (P.apply_opt k (U.Opt.Upcast { axis = last ud; amount = 4 }))

(* Pick a local size for [axis] given the budget already used by [taken]. *)
let local_size_for k taken axis =
  let used = List.fold_left (fun p (_, sz) -> p * sz) 1 taken in
  let ax_sz = const_int_or 0 (nth_size k axis) in
  let candidates = (if axis = 0 then [ 32 ] else []) @ [ 16; 8; 4; 3; 2 ] in
  List.find_opt (fun x -> ax_sz mod x = 0 && used * x <= 128) candidates

(* Choose local sizes for global/loop axes, prioritising expand axes. *)
let apply_locals k =
  if not (Renderer.has_local (P.ren k)) then ()
  else if Helpers.Context_var.get nolocals_var <> 0 then
    ignore (P.apply_opt k U.Opt.Nolocals)
  else
    (* Rank axes: expand axes (broadcast in some buffer) sort first. *)
    let ranking = List.filter_map (fun axis ->
      let rng = nth_rng k axis in
      if not (is_const (range_size rng)) then None
      else
        let is_expand = List.exists (fun b -> match get_idx b with
          | Some idx -> not (List.memq rng (U.backward_slice idx))
          | None -> false) (P.bufs k) in
        Some (is_expand, axis))
      (P.axes_of k [ Axis_type.Global; Axis_type.Loop ])
    in
    let sorted_ranking = List.sort (fun (e1, a1) (e2, a2) ->
      let c = compare e2 e1 in if c <> 0 then c else compare a2 a1) ranking
    in
    (* Pick a local size for each axis, respecting the 128-thread budget. *)
    let to_local = List.fold_left (fun acc (_, axis) ->
      match local_size_for k acc axis with
      | Some sz -> (axis, sz) :: acc
      | None -> acc) [] sorted_ranking
    in
    (* Apply at most 3 locals, sorted by axis, adjusting for deleted shapes. *)
    let to_apply = to_local |> List.rev
      |> List.filteri (fun i _ -> i < 3)
      |> List.sort (fun (a1, _) (a2, _) -> compare a1 a2)
    in
    let deleted = ref 0 in
    List.iter (fun (axis, local_sz) ->
      let axis = axis - !deleted in
      let will_delete = const_int_or 0 (nth_size k axis) = local_sz in
      ignore (P.apply_opt k (U.Opt.Local { axis; amount = local_sz }));
      if will_delete then incr deleted) to_apply

(* Try splitting the first divisible LOOP axis by [threads]. *)
let try_thread_split k threads =
  try List.iter (fun axis ->
    if const_int_or 0 (nth_size k axis) mod threads = 0 then begin
      try_apply k (U.Opt.Thread { axis; amount = threads });
      raise_notrace Exit
    end) (P.axes_of k [ Axis_type.Loop ])
  with Exit -> ()

let last_is_thread opts = opts <> [] && (match last opts with
  | U.Opt.Thread _ -> true | _ -> false)

(* Pick a thread count for LOOP axes. *)
let apply_threading k =
  if not (Renderer.has_threads (P.ren k)) then ()
  else match Renderer.global_max (P.ren k) with
  | None | Some [] -> ()
  | Some (gmax :: _) ->
      let total =
        List.fold_left (fun acc s -> const_int_or 1 s * acc) 1 (P.full_shape k)
      in
      (* Heuristic: use about 128K ops per thread. *)
      try List.iter (fun threads ->
        if threads <= gmax && total / (128 lsl 10) >= threads then begin
          try_thread_split k threads;
          if last_is_thread (P.applied_opts k) then raise_notrace Exit
        end) [ 32; 16; 12; 8; 6; 5; 4; 3; 2 ]
      with Exit -> ()

let hand_coded_optimizations k =
  match try_tensor_cores k with
  | Some k -> k
  | None ->
      let k = P.copy k in
      upcast_images k;
      match try_matvec k with
      | Some k -> k
      | None ->
          if try_grouping k then k
          else begin
            upcast_masked k;
            upcast_heuristic k;
            unroll_reduce k;
            upcast_default k;
            apply_locals k;
            apply_threading k;
            k
          end
