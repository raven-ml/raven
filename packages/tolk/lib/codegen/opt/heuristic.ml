(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_ir
module K = Kernel
module P = Postrange

(* Environment *)

let use_tc = Helpers.getenv "USE_TC" 1
let tc_select = Helpers.getenv "TC_SELECT" (-1)
let tc_opt = Helpers.getenv "TC_OPT" 0
let amx = Helpers.getenv "AMX" 0 <> 0
let mv_blocksize = Helpers.getenv "MV_BLOCKSIZE" 4
let mv_threads_per_row = Helpers.getenv "MV_THREADS_PER_ROW" 8
let mv_rows_per_thread = Helpers.getenv "MV_ROWS_PER_THREAD" 4
let mv = Helpers.getenv "MV" 1
let nolocals_var = Helpers.Context_var.int ~key:"NOLOCALS" ~default:0

(* Helpers *)

let const_int_or default node =
  match K.const_arg node with Some (Int n) -> Int64.to_int n | _ -> default

let last lst = List.nth lst (List.length lst - 1)

let prod_at shape axes =
  List.fold_left (fun acc a ->
    const_int_or 1 (List.nth shape a) * acc) 1 axes

let divides_by rng n = K.divides (K.range_size rng) n <> None

let index_of_rng rngs rng =
  match List.find_index (fun r -> r == rng) rngs with
  | Some i -> i | None -> -1

(* Unwrap Where/Invalid guard from an Index's first idx. *)
let get_idx buf = match K.view buf with
  | Index { idxs = idx :: _; _ } -> Some (
      match K.view idx with
      | Ternary { op = `Where; b; c; _ }
        when (match K.view c with Invalid_index _ -> true | _ -> false) -> b
      | _ -> idx)
  | _ -> None

(* Flatten ADD tree into a list of addends. *)
let split_add node =
  let rec go acc = function
    | [] -> List.rev acc
    | n :: rest -> match K.view n with
      | Binary { op = `Add; lhs; rhs; _ } -> go acc (lhs :: rhs :: rest)
      | _ -> go (n :: acc) rest
  in
  go [] [node]

let try_apply k opt =
  try ignore (P.apply_opt k opt : _ option) with P.Opt_error _ -> ()

(* Find the first size in [sizes] that divides [rng], apply [mk_opt]
   at that axis, and return the shift_to result. *)
let try_opt_on_rng tk rng sizes mk_opt =
  match List.find_opt (divides_by rng) sizes with
  | Some sz ->
      let axis = index_of_rng (P.rngs tk) rng in
      if axis >= 0 then P.apply_opt tk (mk_opt axis sz) else None
  | None -> None

(* Try tensor core optimization.  Returns Some k on success. *)
let try_tensor_cores k =
  if use_tc <= 0
     || (List.length (P.axes_of k [Axis_kind.Group_reduce; Axis_kind.Reduce]) <> 1
         && tc_opt < 1)
  then None
  else
    let tk = P.copy k in
    let tc_result =
      try P.apply_opt tk (K.Opt.Tc { axis = 0; tc_select; tc_opt; use_tc })
      with P.Opt_error _ -> None in
    match tc_result with
    | Some (n_rng, m_rng) when not amx ->
        let rngs = [| n_rng; m_rng |] in
        List.iter (fun d ->
          match try_opt_on_rng tk rngs.(d) [5;4;3;2]
            (fun axis amount -> K.Opt.Upcast { axis; amount }) with
          | Some (replaced, _) -> rngs.(d) <- replaced
          | None -> ()) [1; 0];
        ignore (try_opt_on_rng tk rngs.(0) [4; 2]
          (fun axis amount -> K.Opt.Local { axis; amount }));
        Some tk
    | _ -> None

(* Upcast float4 image axes.  Must run early before locals are added. *)
let is_image_buf buf = match K.view buf with
  | Index { ptr; _ } -> (match K.view ptr with Param_image _ -> true | _ -> false)
  | _ -> false

let upcast_images k =
  List.iter (fun buf ->
    if is_image_buf buf then
      Option.iter (fun idx ->
        let axes = List.filter_map (fun c ->
          if K.is_range c && const_int_or 0 (K.range_size c) mod 4 = 0
          then match index_of_rng (P.rngs k) c with
            | i when i >= 0 -> Some i | _ -> None
          else None) (split_add idx) in
        match axes with
        | axis :: _ when List.mem axis (P.upcastable_dims k) ->
            ignore (P.apply_opt k (K.Opt.Upcast { axis; amount = 4 }))
        | axis :: _ ->
            (match List.find_index (( = ) axis) (P.unrollable_dims k) with
             | Some ui -> ignore (P.apply_opt k (K.Opt.Unroll { axis = ui; amount = 4 }))
             | None -> ())
        | [] -> ())
        (get_idx buf))
    (P.bufs k)

(* Detect matrix-vector pattern: reduce(add, mul(INDEX, INDEX)) where
   the first reduce range appears as an addend in idx0, and all idx0
   ranges appear in idx1.  Returns the first reduce range on match. *)
let detect_matvec k =
  let open Option in
  bind (P.reduceop k) (fun red ->
  match K.view red with
  | Reduce { op = `Add; src = mul_src; _ } -> (
    match K.view mul_src with
    | Binary { op = `Mul; lhs = in0; rhs = in1; _ }
      when (match K.view in0 with Index _ -> true | _ -> false)
        && (match K.view in1 with Index _ -> true | _ -> false) ->
        bind (get_idx in0) (fun idx0 ->
        bind (get_idx in1) (fun _idx1 ->
        match P.ranges_of k [Axis_kind.Reduce] with
        | first_red :: _ ->
            let idx0_rngs = List.filter K.is_range (K.backward_slice idx0) in
            let idx1_rngs = List.filter K.is_range (K.backward_slice _idx1) in
            if List.exists (fun u -> u == first_red) (split_add idx0)
               && List.for_all (fun r -> List.memq r idx1_rngs) idx0_rngs
            then Some first_red
            else None
        | [] -> None))
    | _ -> None)
  | _ -> None)

(* Apply matvec opts (GROUP + LOCAL + UPCAST) if the pattern matches. *)
let try_matvec k =
  if not (Renderer.has_local (P.ren k)) || mv = 0
     || (mv_blocksize <= 1 && mv_threads_per_row <= 1 && mv_rows_per_thread <= 1)
     || List.length (P.full_shape k) < 2
     || not (Renderer.has_shared (P.ren k))
  then None
  else match detect_matvec k with
  | None -> None
  | Some first_red ->
      let gi = List.find_opt (fun gi ->
        divides_by first_red mv_threads_per_row
        && const_int_or 0 (List.nth (P.full_shape k) gi)
           mod (mv_blocksize * mv_rows_per_thread) = 0)
        (P.axes_of k [Axis_kind.Global]) in
      match gi with
      | None -> None
      | Some gi ->
          if mv_threads_per_row > 1 then
            try_apply k (K.Opt.Group { axis = 0; amount = mv_threads_per_row });
          if mv_blocksize > 1 then
            ignore (P.apply_opt k (K.Opt.Local { axis = gi; amount = mv_blocksize }));
          if mv_rows_per_thread > 1 then
            ignore (P.apply_opt k (K.Opt.Upcast { axis = gi; amount = mv_rows_per_thread }));
          Some k

(* Try GROUPTOP if output shape is small. *)
let try_grouping k =
  let threshold = if Helpers.Context_var.get nolocals_var <> 0 then 240 else 2048 in
  if prod_at (P.output_shape k) (P.upcastable_dims k) <= threshold then
    (try List.iter (fun axis ->
       try
         ignore (P.apply_opt k (K.Opt.Grouptop { axis; amount = 16 }));
         raise_notrace Exit
       with P.Opt_error _ -> ()) [0; 1; 2]
     with Exit -> ());
  P.group_for_reduces k > 0

(* Upcast small masked dims (e.g. from Tensor.stack). *)
let upcast_masked k =
  let ast_slice = K.backward_slice (P.ast k) in
  let is_masked rng = List.exists (fun u -> match K.view u with
    | Ternary { op = `Where; a = cond; _ } ->
        List.exists (fun n -> n == rng) (K.backward_slice cond)
    | _ -> false) ast_slice in
  let to_upcast = List.fold_left (fun acc axis ->
    let sz = const_int_or 0 (List.nth (P.full_shape k) axis) in
    if sz > 7 then acc
    else if is_masked (List.nth (P.rngs k) axis)
         && prod_at (P.full_shape k) acc * sz <= 49
    then acc @ [axis] else acc)
    [] (P.upcastable_dims k) in
  List.iter (fun axis ->
    ignore (P.apply_opt k (K.Opt.Upcast { axis; amount = 0 })))
    (List.rev to_upcast)

(* Upcast non-reduce axes based on stride analysis: prefer axes where
   some buffer broadcasts (stride 0) while all upcast/unroll axes have
   nonzero stride.  Pick the axis with fewest strides first. *)
let upcast_heuristic k =
  let is_dsp = Renderer.device (P.ren k) = "DSP" in
  let upcasted = Hashtbl.create 8 in
  let continue_ = ref true in
  while !continue_
        && prod_at (P.output_shape k) (P.upcastable_dims k) >= 1024
        && P.upcast_size k < 32 do
    let upcast_amounts =
      if is_dsp then (if Hashtbl.length upcasted = 0 then [128] else [])
      else [3; 4] in
    let xb = List.fold_left (fun acc axis ->
      List.fold_left (fun acc amt ->
        if Hashtbl.mem upcasted axis then acc
        else if const_int_or 0 (List.nth (P.full_shape k) axis) mod amt <> 0 then acc
        else
          let rng = List.nth (P.rngs k) axis in
          let upcast_unroll = P.ranges_of k [Axis_kind.Upcast; Axis_kind.Unroll] in
          let has_broadcast = List.exists (fun b ->
            match get_idx b with
            | Some idx ->
                let bslice = K.backward_slice idx in
                not (List.memq rng bslice)
                && List.for_all (fun r2 -> List.memq r2 bslice) upcast_unroll
            | None -> false) (P.bufs k) in
          if not has_broadcast then acc
          else
            let num_strides = ref 0 in
            let sum_strides = ref 0 in
            List.iter (fun b ->
              match get_idx b with
              | Some idx ->
                  if List.memq rng (K.backward_slice idx) then
                    incr num_strides;
                  List.iter (fun c ->
                    if c == rng then incr sum_strides
                    else match K.view c with
                    | Binary { op = `Mul; lhs; rhs; _ } ->
                        if lhs == rng && K.is_const rhs then
                          sum_strides := !sum_strides + K.const_to_int rhs
                        else if rhs == rng && K.is_const lhs then
                          sum_strides := !sum_strides + K.const_to_int lhs
                    | _ -> ()) (split_add idx)
              | None -> ()) (P.bufs k);
            (!num_strides, !sum_strides, axis, amt) :: acc)
        acc upcast_amounts)
      [] (P.upcastable_dims k)
      |> List.sort compare in
    match xb with
    | (_, _, axis, amt) :: _ ->
        ignore (P.apply_opt k (K.Opt.Upcast { axis; amount = amt }));
        Hashtbl.replace upcasted axis ()
    | [] -> continue_ := false
  done

(* Unroll last reduce dim if small. *)
let unroll_reduce k =
  try
    let ud = P.unrollable_dims k in
    if ud <> []
       && (P.upcast_size k <= 4 || P.axes_of k [Axis_kind.Unroll] = [])
       && P.upcast_size k < 64
    then begin
      let s = const_int_or 0 (List.nth (P.full_shape k) (last ud)) in
      if s <= 32 then begin
        ignore (P.apply_opt k
          (K.Opt.Unroll { axis = List.length ud - 1; amount = 0 }));
        let ud2 = P.unrollable_dims k in
        if ud2 <> [] && s <= 3
           && const_int_or 0 (List.nth (P.full_shape k) (last ud2)) <= 3 then
          ignore (P.apply_opt k
            (K.Opt.Unroll { axis = List.length ud2 - 1; amount = 0 }))
      end else if const_int_or 0 (List.nth (P.full_shape k) (last ud)) mod 4 = 0 then
        ignore (P.apply_opt k
          (K.Opt.Unroll { axis = List.length ud - 1; amount = 4 }))
    end
  with P.Opt_error _ -> ()

(* Upcast by 4 if nothing is upcasted yet. *)
let upcast_default k =
  let ud = P.upcastable_dims k in
  if P.upcasted k = 0 && ud <> []
     && const_int_or 0 (List.nth (P.full_shape k) (last ud)) mod 4 = 0 then
    ignore (P.apply_opt k (K.Opt.Upcast { axis = last ud; amount = 4 }))

(* Choose local sizes for global/loop axes, prioritising expand axes. *)
let apply_locals k =
  if not (Renderer.has_local (P.ren k)) then ()
  else if Helpers.Context_var.get nolocals_var <> 0 then
    ignore (P.apply_opt k K.Opt.Nolocals)
  else begin
    (* Rank axes: expand axes (broadcast in some buffer) sort first. *)
    let ranking = List.filter_map (fun axis ->
      let rng = List.nth (P.rngs k) axis in
      if not (K.is_const (K.range_size rng)) then None
      else
        let is_expand = List.exists (fun b ->
          match get_idx b with
          | Some idx -> not (List.memq rng (K.backward_slice idx))
          | None -> false) (P.bufs k) in
        Some (is_expand, axis))
      (P.axes_of k [Axis_kind.Global; Axis_kind.Loop]) in
    let sorted_ranking = List.sort (fun (e1, a1) (e2, a2) ->
      let c = compare e2 e1 in
      if c <> 0 then c else compare a2 a1) ranking in
    (* Pick a local size for each axis, respecting the 128-thread budget. *)
    let to_local = List.fold_left (fun acc (_, axis) ->
      let local_size = List.fold_left (fun p (_, sz) -> p * sz) 1 acc in
      let candidates =
        (if axis = 0 then [32] else []) @ [16; 8; 4; 3; 2] in
      let ax_sz = const_int_or 0 (List.nth (P.full_shape k) axis) in
      match List.find_opt (fun x ->
        ax_sz mod x = 0 && local_size * x <= 128) candidates with
      | Some sz -> (axis, sz) :: acc
      | None -> acc) [] sorted_ranking in
    (* Apply at most 3 locals, sorted by axis, adjusting for deleted shapes. *)
    let to_apply = to_local
      |> List.rev |> List.filteri (fun i _ -> i < 3)
      |> List.sort (fun (a1, _) (a2, _) -> compare a1 a2) in
    let deleted = ref 0 in
    List.iter (fun (axis, local_sz) ->
      let axis = axis - !deleted in
      let will_delete = const_int_or 0 (List.nth (P.full_shape k) axis) = local_sz in
      ignore (P.apply_opt k (K.Opt.Local { axis; amount = local_sz }));
      if will_delete then incr deleted) to_apply
  end

(* Pick a thread count for LOOP axes. *)
let apply_threading k =
  if Renderer.has_threads (P.ren k) then
    match Renderer.global_max (P.ren k) with
    | Some (gmax :: _) ->
        let total = List.fold_left (fun acc s -> const_int_or 1 s * acc) 1
          (P.full_shape k) in
        (try List.iter (fun threads ->
          if threads <= gmax && total / (128 lsl 10) >= threads then begin
            (try List.iter (fun axis ->
              if const_int_or 0 (List.nth (P.full_shape k) axis) mod threads = 0
              then begin
                try_apply k (K.Opt.Thread { axis; amount = threads });
                raise_notrace Exit
              end) (P.axes_of k [Axis_kind.Loop])
             with Exit -> ());
            let opts = P.applied_opts k in
            if opts <> [] && (match last opts with
              K.Opt.Thread _ -> true | _ -> false)
            then raise_notrace Exit
          end) [32; 16; 12; 8; 6; 5; 4; 3; 2]
         with Exit -> ())
    | _ -> ()

let hand_coded_optimizations k =
  match try_tensor_cores k with Some k -> k | None ->
  let k = P.copy k in
  upcast_images k;
  match try_matvec k with Some k -> k | None ->
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
