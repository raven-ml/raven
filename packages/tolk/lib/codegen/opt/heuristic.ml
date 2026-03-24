(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Hand-coded kernel optimizations. *)

open Tolk_ir
module K = Kernel
module P = Postrange

exception Early_return of P.t
exception Break

let getenv_int name default =
  match Sys.getenv_opt name with
  | Some s -> (try int_of_string s with Failure _ -> default)
  | None -> default

let last lst = List.nth lst (List.length lst - 1)

let const_or default node =
  if K.is_const node then K.const_to_int node else default

let prod_at_axes k axes =
  let fs = P.full_shape k in
  List.fold_left
    (fun acc a ->
      let s = List.nth fs a in
      if K.is_const s then acc * K.const_to_int s else acc)
    1 axes

let divides_by rng n =
  let sz = K.range_size rng in
  if K.is_const sz && K.const_to_int sz mod n = 0 then Some n else None

let split_add node =
  let rec aux acc = function
    | [] -> List.rev acc
    | n :: rest ->
        match K.view n with
        | Binary { op = `Add; lhs; rhs; _ } -> aux acc (lhs :: rhs :: rest)
        | _ -> aux (n :: acc) rest
  in
  aux [] [ node ]

let unwrap_idx idx =
  match K.view idx with
  | Ternary { op = `Where; b = real_idx; c = invalid; _ } ->
      (match K.view invalid with Invalid_index _ -> real_idx | _ -> idx)
  | _ -> idx

let get_idx buf =
  match K.view buf with
  | Index { idxs = idx :: _; _ } -> Some (unwrap_idx idx)
  | _ -> None

let is_param_image buf =
  match K.view buf with
  | Index { ptr; _ } ->
      (match K.view ptr with Param_image _ -> true | _ -> false)
  | _ -> false

let index_of_rng rngs rng =
  let rec find i = function
    | [] -> -1
    | r :: _ when r == rng -> i
    | _ :: rest -> find (i + 1) rest
  in
  find 0 rngs

let try_apply k opt =
  try ignore (P.apply_opt k opt) with P.Opt_error _ -> ()

(* reduce(add, mul(INDEX, INDEX)) where the first reduce range appears in
   idx0's addends and all idx0 ranges appear in idx1. *)
let detect_matvec k =
  match P.reduceop k with
  | None -> None
  | Some red ->
      match K.view red with
      | Reduce { op = `Add; src = mul_src; _ } ->
          (match K.view mul_src with
          | Binary { op = `Mul; lhs = in0; rhs = in1; _ } ->
              (match K.view in0, K.view in1 with
              | Index _, Index _ ->
                  (match get_idx in0, get_idx in1 with
                  | Some idx0, Some idx1 ->
                      (match P.ranges_of k [ Axis_kind.Reduce ] with
                      | first_reduce_rng :: _
                        when List.exists
                               (fun u -> u == first_reduce_rng)
                               (split_add idx0) ->
                          let idx0_rngs =
                            List.filter K.is_range (K.backward_slice idx0)
                          in
                          let idx1_rngs =
                            List.filter K.is_range (K.backward_slice idx1)
                          in
                          if
                            List.for_all
                              (fun r ->
                                List.exists (fun r2 -> r2 == r) idx1_rngs)
                              idx0_rngs
                          then Some first_reduce_rng
                          else None
                      | _ -> None)
                  | _ -> None)
              | _ -> None)
          | _ -> None)
      | _ -> None

let hand_coded_optimizations_impl k =
  let use_tc = getenv_int "USE_TC" 1 in
  let tc_select = getenv_int "TC_SELECT" (-1) in
  let tc_opt = getenv_int "TC_OPT" 0 in
  let amx = getenv_int "AMX" 0 in

  (* Step 1: Tensor cores *)
  if
    use_tc > 0
    && (List.length (P.axes_of k [ Axis_kind.Group_reduce; Axis_kind.Reduce ]) = 1
       || tc_opt >= 1)
  then begin
    let tk = P.copy k in
    let tc_result =
      try Some (P.apply_opt tk (K.Opt.Tc { axis = 0; tc_select; tc_opt; use_tc }))
      with P.Opt_error _ -> None
    in
    match tc_result with
    | Some rngs when amx = 0 ->
        (match rngs with
        | Some (n_rng, m_rng) ->
            let rngs = [| n_rng; m_rng |] in
            List.iter
              (fun tc_dim ->
                match List.find_map (divides_by rngs.(tc_dim)) [ 5; 4; 3; 2 ] with
                | Some sz ->
                    let axis = index_of_rng (P.rngs tk) rngs.(tc_dim) in
                    if axis >= 0 then
                      (match P.apply_opt tk (K.Opt.Upcast { axis; amount = sz }) with
                      | Some (replaced, _) -> rngs.(tc_dim) <- replaced
                      | None -> ())
                | None -> ())
              [ 1; 0 ];
            (match List.find_map (divides_by rngs.(0)) [ 4; 2 ] with
            | Some sz ->
                let axis = index_of_rng (P.rngs tk) rngs.(0) in
                if axis >= 0 then
                  ignore (P.apply_opt tk (K.Opt.Local { axis; amount = sz }))
            | None -> ())
        | None -> ());
        raise (Early_return tk)
    | _ -> ()
  end;

  let k = P.copy k in

  (* Step 2: Image upcast *)
  List.iter
    (fun buf ->
      if is_param_image buf then
        match get_idx buf with
        | Some idx ->
            let addends = split_add idx in
            let unit_stride_axes_mul_4 =
              List.filter_map
                (fun c ->
                  if K.is_range c then
                    let sz = K.range_size c in
                    if K.is_const sz && K.const_to_int sz mod 4 = 0 then
                      let i = index_of_rng (P.rngs k) c in
                      if i >= 0 then Some i else None
                    else None
                  else None)
                addends
            in
            (match unit_stride_axes_mul_4 with
            | axis :: _ ->
                if List.mem axis (P.upcastable_dims k) then
                  ignore (P.apply_opt k (K.Opt.Upcast { axis; amount = 4 }))
                else begin
                  let unrollable = P.unrollable_dims k in
                  match
                    List.find_mapi
                      (fun i x -> if x = axis then Some i else None)
                      unrollable
                  with
                  | Some ui ->
                      ignore
                        (P.apply_opt k (K.Opt.Unroll { axis = ui; amount = 4 }))
                  | None -> ()
                end
            | [] -> ())
        | None -> ())
    (P.bufs k);

  (* Step 3: Matvec *)
  let mv = getenv_int "MV" 1 in
  let mv_blocksize = getenv_int "MV_BLOCKSIZE" 4 in
  let mv_threads_per_row = getenv_int "MV_THREADS_PER_ROW" 8 in
  let mv_rows_per_thread = getenv_int "MV_ROWS_PER_THREAD" 4 in
  if
    Renderer.has_local (P.ren k)
    && mv <> 0
    && (mv_blocksize > 1 || mv_threads_per_row > 1 || mv_rows_per_thread > 1)
    && P.reduceop k <> None
    && List.length (P.full_shape k) >= 2
    && Renderer.has_shared (P.ren k)
  then begin
    match detect_matvec k with
    | Some first_reduce_rng ->
        let did_matvec =
          List.exists
            (fun global_idx ->
              let rsz = K.range_size first_reduce_rng in
              let gsz = List.nth (P.full_shape k) global_idx in
              if
                K.is_const rsz
                && K.const_to_int rsz mod mv_threads_per_row = 0
                && K.is_const gsz
                && K.const_to_int gsz mod (mv_blocksize * mv_rows_per_thread) = 0
              then begin
                if mv_threads_per_row > 1 then
                  try_apply k
                    (K.Opt.Group { axis = 0; amount = mv_threads_per_row });
                if mv_blocksize > 1 then
                  ignore
                    (P.apply_opt k
                       (K.Opt.Local { axis = global_idx; amount = mv_blocksize }));
                if mv_rows_per_thread > 1 then
                  ignore
                    (P.apply_opt k
                       (K.Opt.Upcast
                          { axis = global_idx; amount = mv_rows_per_thread }));
                true
              end
              else false)
            (P.axes_of k [ Axis_kind.Global ])
        in
        if did_matvec then raise (Early_return k)
    | None -> ()
  end;

  (* Step 4: Grouping *)
  let nolocals = getenv_int "NOLOCALS" 0 in
  let group_threshold = if nolocals <> 0 then 240 else 2048 in
  if prod_at_axes k (P.upcastable_dims k) <= group_threshold then
    (try
       List.iter
         (fun axis ->
           (try
              ignore (P.apply_opt k (K.Opt.Grouptop { axis; amount = 16 }));
              raise_notrace Break
            with P.Opt_error _ -> ()))
         [ 0; 1; 2 ]
     with Break -> ());

  (* Step 5: No more opt if grouping *)
  if P.group_for_reduces k > 0 then raise (Early_return k);

  (* Step 6: Masked upcast *)
  let to_upcast = ref [] in
  let ast_slice = K.backward_slice (P.ast k) in
  List.iter
    (fun axis ->
      let rng = List.nth (P.rngs k) axis in
      let fs = P.full_shape k in
      let sz = List.nth fs axis in
      if K.is_const sz && K.const_to_int sz <= 7 then begin
        let is_masked =
          List.exists
            (fun u ->
              match K.view u with
              | Ternary { op = `Where; a = cond; _ } ->
                  List.exists (fun n -> n == rng) (K.backward_slice cond)
              | _ -> false)
            ast_slice
        in
        let already_prod =
          List.fold_left
            (fun acc j ->
              let s = List.nth fs j in
              if K.is_const s then acc * K.const_to_int s else acc)
            1 !to_upcast
        in
        if is_masked && already_prod * K.const_to_int sz <= 49 then
          to_upcast := axis :: !to_upcast
      end)
    (P.upcastable_dims k);
  (* Prepending with :: already reversed the order, matching the desired
     upcast application sequence. *)
  List.iter
    (fun axis -> ignore (P.apply_opt k (K.Opt.Upcast { axis; amount = 0 })))
    !to_upcast;

  (* Step 7: Heuristic upcast *)
  let is_dsp = Renderer.device (P.ren k) = "DSP" in
  let upcasted_axis = Hashtbl.create 8 in
  let continue_upcast = ref true in
  while
    !continue_upcast
    && prod_at_axes k (P.upcastable_dims k) >= 1024
    && P.upcast_size k < 32
  do
    let xb_choices = ref [] in
    let upcast_amounts =
      if is_dsp then
        if Hashtbl.length upcasted_axis = 0 then [ 128 ] else []
      else [ 3; 4 ]
    in
    List.iter
      (fun axis ->
        List.iter
          (fun upcast_amount ->
            if not (Hashtbl.mem upcasted_axis axis) then begin
              let s = List.nth (P.full_shape k) axis in
              if K.is_const s && K.const_to_int s mod upcast_amount = 0 then begin
                let rng = List.nth (P.rngs k) axis in
                let bufs_list = P.bufs k in
                let upcast_unroll_rngs =
                  P.ranges_of k [ Axis_kind.Upcast; Axis_kind.Unroll ]
                in
                let has_broadcast =
                  List.exists
                    (fun b ->
                      match get_idx b with
                      | Some idx ->
                          let bslice = K.backward_slice idx in
                          not (List.exists (fun n -> n == rng) bslice)
                          && List.for_all
                               (fun r2 -> List.exists (fun n -> n == r2) bslice)
                               upcast_unroll_rngs
                      | None -> false)
                    bufs_list
                in
                if has_broadcast then begin
                  let num_strides = ref 0 in
                  let sum_strides = ref 0 in
                  List.iter
                    (fun b ->
                      match get_idx b with
                      | Some idx ->
                          let bslice = K.backward_slice idx in
                          if List.exists (fun n -> n == rng) bslice then
                            incr num_strides;
                          List.iter
                            (fun c ->
                              if c == rng then incr sum_strides
                              else
                                match K.view c with
                                | Binary { op = `Mul; lhs; rhs; _ } ->
                                    if lhs == rng && K.is_const rhs then
                                      sum_strides :=
                                        !sum_strides + K.const_to_int rhs
                                    else if rhs == rng && K.is_const lhs then
                                      sum_strides :=
                                        !sum_strides + K.const_to_int lhs
                                | _ -> ())
                            (split_add idx)
                      | None -> ())
                    bufs_list;
                  xb_choices :=
                    (!num_strides, !sum_strides, axis, upcast_amount)
                    :: !xb_choices
                end
              end
            end)
          upcast_amounts)
      (P.upcastable_dims k);
    let sorted = List.sort compare !xb_choices in
    match sorted with
    | (_, _, axis, upcast_amount) :: _ ->
        ignore (P.apply_opt k (K.Opt.Upcast { axis; amount = upcast_amount }));
        Hashtbl.replace upcasted_axis axis ()
    | [] -> continue_upcast := false
  done;

  (* Step 8: Reduce unroll *)
  (try
     let unrollable = P.unrollable_dims k in
     if
       unrollable <> []
       && (P.upcast_size k <= 4 || P.axes_of k [ Axis_kind.Unroll ] = [])
       && P.upcast_size k < 64
     then begin
       let last_axis = last unrollable in
       let s = const_or 0 (List.nth (P.full_shape k) last_axis) in
       if s <= 32 then begin
         ignore
           (P.apply_opt k
              (K.Opt.Unroll { axis = List.length unrollable - 1; amount = 0 }));
         let unrollable2 = P.unrollable_dims k in
         if unrollable2 <> [] && s <= 3 then begin
           let s2 = const_or 0 (List.nth (P.full_shape k) (last unrollable2)) in
           if s2 <= 3 then
             ignore
               (P.apply_opt k
                  (K.Opt.Unroll { axis = List.length unrollable2 - 1; amount = 0 }))
         end
       end
       else begin
         let sz = List.nth (P.full_shape k) last_axis in
         if K.is_const sz && K.const_to_int sz mod 4 = 0 then
           ignore
             (P.apply_opt k
                (K.Opt.Unroll { axis = List.length unrollable - 1; amount = 4 }))
       end
     end
   with P.Opt_error _ -> ());

  (* Step 9: Default upcast *)
  let upcastable = P.upcastable_dims k in
  if P.upcasted k = 0 && upcastable <> [] then begin
    let last_axis = last upcastable in
    let sz = List.nth (P.full_shape k) last_axis in
    if K.is_const sz && K.const_to_int sz mod 4 = 0 then
      ignore (P.apply_opt k (K.Opt.Upcast { axis = last_axis; amount = 4 }))
  end;

  (* Step 10: Local groups *)
  if Renderer.has_local (P.ren k) then begin
    if nolocals <> 0 then ignore (P.apply_opt k K.Opt.Nolocals)
    else begin
      let local_axis_ranking =
        List.filter_map
          (fun axis ->
            let rng = List.nth (P.rngs k) axis in
            let sz = K.range_size rng in
            if K.is_const sz then
              let is_expand =
                List.exists
                  (fun b ->
                    match get_idx b with
                    | Some idx ->
                        not (List.exists (fun n -> n == rng) (K.backward_slice idx))
                    | None -> false)
                  (P.bufs k)
              in
              Some (is_expand, axis)
            else None)
          (P.axes_of k [ Axis_kind.Global; Axis_kind.Loop ])
      in
      let sorted_ranking =
        List.sort
          (fun (e1, a1) (e2, a2) ->
            let c = compare e2 e1 in
            if c <> 0 then c else compare a2 a1)
          local_axis_ranking
      in
      let to_local = ref [] in
      List.iter
        (fun (_, axis) ->
          let local_size =
            List.fold_left (fun acc (_, sz) -> acc * sz) 1 !to_local
          in
          let ax_sz = const_or 0 (List.nth (P.full_shape k) axis) in
          let candidates =
            (if axis = 0 then [ 32 ] else []) @ [ 16; 8; 4; 3; 2 ]
          in
          match
            List.find_opt
              (fun x -> ax_sz mod x = 0 && local_size * x <= 128)
              candidates
          with
          | Some sz -> to_local := (axis, sz) :: !to_local
          | None -> ())
        sorted_ranking;
      let to_apply =
        let lst = List.filteri (fun i _ -> i < 3) (List.rev !to_local) in
        List.sort (fun (a1, _) (a2, _) -> compare a1 a2) lst
      in
      let deleted_shape = ref 0 in
      List.iter
        (fun (axis, local_sz) ->
          let axis = axis - !deleted_shape in
          let will_delete =
            let s = List.nth (P.full_shape k) axis in
            K.is_const s && local_sz = K.const_to_int s
          in
          ignore (P.apply_opt k (K.Opt.Local { axis; amount = local_sz }));
          if will_delete then incr deleted_shape)
        to_apply
    end
  end;

  (* Step 11: Threading *)
  if Renderer.has_threads (P.ren k) then begin
    match Renderer.global_max (P.ren k) with
    | Some (gmax :: _) ->
        let total_shape = prod_at_axes k (List.init (P.shape_len k) Fun.id) in
        (try
           List.iter
             (fun threads ->
               if
                 threads <= gmax
                 && total_shape / (128 lsl 10) >= threads
               then begin
                 let loop_axes = P.axes_of k [ Axis_kind.Loop ] in
                 let tried = ref false in
                 List.iter
                   (fun axis ->
                     if not !tried then begin
                       let s = List.nth (P.full_shape k) axis in
                       if K.is_const s && K.const_to_int s mod threads = 0 then begin
                         try_apply k (K.Opt.Thread { axis; amount = threads });
                         tried := true
                       end
                     end)
                   loop_axes;
                 match List.rev (P.applied_opts k) with
                 | K.Opt.Thread _ :: _ -> raise_notrace Break
                 | _ -> ()
               end)
             [ 32; 16; 12; 8; 6; 5; 4; 3; 2 ]
         with Break -> ())
    | _ -> ()
  end;

  k

let hand_coded_optimizations k =
  try hand_coded_optimizations_impl k with Early_return k -> k
