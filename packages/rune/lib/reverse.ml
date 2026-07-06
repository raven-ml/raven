(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Reverse-mode differentiation as an effect handler over Nx operations.

   Forward pass: every intercepted operation computes its primal by
   re-performing the operation in the enclosing context (so nested grads
   compose), and, if any input is tracked on the tape, marks its output tracked
   and records a pull thunk. Operations whose inputs are all untracked are
   constants with respect to the differentiated inputs and are recorded nowhere.

   Backward pass: [Tape.backward] runs the pull thunks in reverse. A pull thunk
   reads its output cotangent from the tape and accumulates input contributions.
   Pull thunks execute ordinary Nx operations, so an enclosing grad
   differentiates them: higher-order derivatives work.

   Every Nx effect constructor is matched explicitly. Operations without a
   gradient fall into three deliberate categories: - zero derivative
   (comparisons, bitwise and integer ops, rounding, argmax/argmin/argsort, RNG,
   tensor creation): fall through untracked, which yields the correct zero
   gradient; - no rule implemented (svd, eig, eigh, rfft, irfft, psum, mod):
   raise when an input is tracked instead of silently producing a zero gradient
   — detach the input if differentiation should not flow through it; - in-place
   mutation (assign): always raises during differentiation. *)

open Nx_effect
module T = Nx

(* Reduce a cotangent to the shape of a broadcast source. *)
let unbroadcast (type a b) (g : (a, b) T.t) (src_shape : int array) : (a, b) T.t
    =
  let dst_shape = T.shape g in
  if src_shape = dst_shape then g
  else
    let src_rank = Array.length src_shape in
    let dst_rank = Array.length dst_shape in
    let axes = ref [] in
    for i = 0 to dst_rank - src_rank - 1 do
      axes := i :: !axes
    done;
    for i = 0 to src_rank - 1 do
      if src_shape.(i) = 1 && dst_shape.(i + (dst_rank - src_rank)) > 1 then
        axes := (i + (dst_rank - src_rank)) :: !axes
    done;
    match !axes with
    | [] -> g
    | ax ->
        let summed = T.sum g ~axes:ax ~keepdims:true in
        if T.shape summed <> src_shape then T.reshape src_shape summed
        else summed

let err_no_rule op =
  invalid_arg
    (Printf.sprintf
       "Rune: the gradient of %s is not implemented; detach its input if \
        differentiation should not flow through it"
       op)

(* Handler *)

let handler (tape : Tape.t) =
  let open Effect.Deep in
  let tracked x = Tape.tracked tape x in
  let track x = Tape.track tape x in

  (* [pull1 out x f] records: cotangent of [x] += [f] applied to the cotangent
     of [out]. Skips recording when [x] is untracked. *)
  let pull1 (type a b c d) k (out : (a, b) t) (x : (c, d) t)
      (f : (a, b) t -> (c, d) t) =
    if tracked x then begin
      track out;
      Tape.record tape (fun () ->
          match Tape.find tape out with
          | None -> ()
          | Some g -> Tape.accumulate tape x (f g))
    end;
    continue k out
  in

  (* [pull2 out a b fa fb] is [pull1] for binary arithmetic: contributions are
     reduced back to each input's shape to undo broadcasting. *)
  let pull2 (type a b) k (out : (a, b) t) (a_in : (a, b) t) (b_in : (a, b) t)
      (fa : (a, b) t -> (a, b) t) (fb : (a, b) t -> (a, b) t) =
    let ta = tracked a_in and tb = tracked b_in in
    if ta || tb then begin
      track out;
      Tape.record tape (fun () ->
          match Tape.find tape out with
          | None -> ()
          | Some g ->
              if ta then
                Tape.accumulate tape a_in (unbroadcast (fa g) (T.shape a_in));
              if tb then
                Tape.accumulate tape b_in (unbroadcast (fb g) (T.shape b_in)))
    end;
    continue k out
  in

  let no_rule (type c) k (op : string) (inputs_tracked : bool) (out : unit -> c)
      =
    if inputs_tracked then err_no_rule op else continue k (out ())
  in

  let effc : type c. c Effect.t -> ((c, _) continuation -> _) option =
   fun eff ->
    if not !Gate.enabled then None
    else
      match eff with
      (* Constants: creation, RNG, metadata. Fresh outputs are untracked. *)
      | E_view _ -> None
      | E_to_host _ -> None
      | E_buffer _ -> None
      | E_const_scalar _ -> None
      | E_from_host _ -> None
      | E_threefry _ -> None
      | E_to_device _ -> None
      (* Zero derivative: boolean, bitwise and integer results. *)
      | E_cmpeq _ -> None
      | E_cmpne _ -> None
      | E_cmplt _ -> None
      | E_cmple _ -> None
      | E_xor _ -> None
      | E_or _ -> None
      | E_and _ -> None
      | E_idiv _ -> None
      | E_argmax _ -> None
      | E_argmin _ -> None
      | E_argsort _ -> None
      (* Zero derivative: piecewise-constant real functions. *)
      | E_sign _ -> None
      | E_trunc _ -> None
      | E_ceil _ -> None
      | E_floor _ -> None
      | E_round _ -> None
      (* Mutation is incompatible with identity-keyed tracking. *)
      | E_assign _ ->
          Some
            (fun _k ->
              invalid_arg
                "in-place mutation (set_item, set_slice, blit, assign) cannot \
                 be used inside grad/value_and_grad — use scatter instead")
      (* Binary arithmetic *)
      | E_add { a; b } -> Some (fun k -> pull2 k (add a b) a b Fun.id Fun.id)
      | E_sub { a; b } -> Some (fun k -> pull2 k (sub a b) a b Fun.id T.neg)
      | E_mul { a; b } ->
          Some
            (fun k ->
              pull2 k (mul a b) a b (fun g -> T.mul g b) (fun g -> T.mul g a))
      | E_fdiv { a; b } ->
          Some
            (fun k ->
              pull2 k (div a b) a b
                (fun g -> T.div g b)
                (fun g -> T.mul (T.neg g) (T.div a (T.mul b b))))
      | E_pow { a; b } ->
          Some
            (fun k ->
              let out = pow a b in
              pull2 k out a b
                (fun g -> T.mul g (Derivs.pow_wrt_base a b))
                (fun g -> T.mul g (Derivs.pow_wrt_exp a out)))
      | E_max { a; b } ->
          Some
            (fun k ->
              let out = max a b in
              let mask g = T.cast (T.dtype g) (T.cmpgt a b) in
              pull2 k out a b
                (fun g -> T.mul g (mask g))
                (fun g -> T.mul g (T.sub (T.ones_like (mask g)) (mask g))))
      | E_min { a; b } ->
          Some
            (fun k ->
              let out = min a b in
              let mask g = T.cast (T.dtype g) (T.cmplt a b) in
              pull2 k out a b
                (fun g -> T.mul g (mask g))
                (fun g -> T.mul g (T.sub (T.ones_like (mask g)) (mask g))))
      | E_atan2 { a; b } ->
          Some
            (fun k ->
              let denom () = T.add (T.mul a a) (T.mul b b) in
              pull2 k (atan2 a b) a b
                (fun g -> T.mul g (T.div b (denom ())))
                (fun g -> T.mul g (T.neg (T.div a (denom ())))))
      | E_mod { a; b } ->
          Some
            (fun k ->
              no_rule k "mod" (tracked a || tracked b) (fun () -> mod_ a b))
      (* Unary arithmetic *)
      | E_neg { t_in } -> Some (fun k -> pull1 k (neg t_in) t_in T.neg)
      | E_sin { t_in } ->
          Some
            (fun k -> pull1 k (sin t_in) t_in (fun g -> T.mul g (T.cos t_in)))
      | E_cos { t_in } ->
          Some
            (fun k ->
              pull1 k (cos t_in) t_in (fun g -> T.mul g (T.neg (T.sin t_in))))
      | E_tan { t_in } ->
          Some
            (fun k ->
              pull1 k (tan t_in) t_in (fun g -> T.mul g (Derivs.tan' t_in)))
      | E_asin { t_in } ->
          Some
            (fun k ->
              pull1 k (asin t_in) t_in (fun g -> T.mul g (Derivs.asin' t_in)))
      | E_acos { t_in } ->
          Some
            (fun k ->
              pull1 k (acos t_in) t_in (fun g ->
                  T.mul g (T.neg (Derivs.asin' t_in))))
      | E_atan { t_in } ->
          Some
            (fun k ->
              pull1 k (atan t_in) t_in (fun g -> T.mul g (Derivs.atan' t_in)))
      | E_sinh { t_in } ->
          Some
            (fun k -> pull1 k (sinh t_in) t_in (fun g -> T.mul g (T.cosh t_in)))
      | E_cosh { t_in } ->
          Some
            (fun k -> pull1 k (cosh t_in) t_in (fun g -> T.mul g (T.sinh t_in)))
      | E_tanh { t_in } ->
          Some
            (fun k ->
              let out = tanh t_in in
              pull1 k out t_in (fun g -> T.mul g (Derivs.tanh' out)))
      | E_exp { t_in } ->
          Some
            (fun k ->
              let out = exp t_in in
              pull1 k out t_in (fun g -> T.mul g out))
      | E_log { t_in } ->
          Some
            (fun k -> pull1 k (log t_in) t_in (fun g -> T.mul g (T.recip t_in)))
      | E_sqrt { t_in } ->
          Some
            (fun k ->
              let out = sqrt t_in in
              pull1 k out t_in (fun g -> T.mul g (Derivs.sqrt' out)))
      | E_recip { t_in } ->
          Some
            (fun k ->
              pull1 k (recip t_in) t_in (fun g -> T.mul g (Derivs.recip' t_in)))
      | E_abs { t_in } ->
          Some
            (fun k -> pull1 k (abs t_in) t_in (fun g -> T.mul g (T.sign t_in)))
      | E_erf { t_in } ->
          Some
            (fun k ->
              pull1 k (erf t_in) t_in (fun g -> T.mul g (Derivs.erf' t_in)))
      (* Selection *)
      | E_where { condition; if_true; if_false } ->
          Some
            (fun k ->
              let out = where condition if_true if_false in
              let tt = tracked if_true and tf = tracked if_false in
              if tt || tf then begin
                track out;
                Tape.record tape (fun () ->
                    match Tape.find tape out with
                    | None -> ()
                    | Some g ->
                        let mask = T.cast (T.dtype g) condition in
                        if tt then
                          Tape.accumulate tape if_true
                            (unbroadcast (T.mul g mask) (T.shape if_true));
                        if tf then
                          Tape.accumulate tape if_false
                            (unbroadcast
                               (T.mul g (T.sub (T.ones_like mask) mask))
                               (T.shape if_false)))
              end;
              continue k out)
      (* Movement: linear ops whose pull is the transpose movement. *)
      | E_reshape { t_in; new_shape } ->
          Some
            (fun k ->
              pull1 k (reshape t_in new_shape) t_in (fun g ->
                  T.reshape (T.shape t_in) g))
      | E_permute { t_in; axes } ->
          Some
            (fun k ->
              let inv = Array.make (Array.length axes) 0 in
              Array.iteri (fun i d -> inv.(d) <- i) axes;
              pull1 k (permute t_in axes) t_in (fun g ->
                  T.transpose g ~axes:(Array.to_list inv)))
      | E_expand { t_in; new_target_shape } ->
          Some
            (fun k ->
              pull1 k (expand t_in new_target_shape) t_in (fun g ->
                  unbroadcast g (T.shape t_in)))
      | E_pad { t_in; padding_config; fill_value } ->
          Some
            (fun k ->
              let limits =
                Array.mapi
                  (fun i (pre, _) -> (pre, pre + (T.shape t_in).(i)))
                  padding_config
              in
              pull1 k (pad t_in padding_config fill_value) t_in (fun g ->
                  T.shrink limits g))
      | E_shrink { t_in; limits } ->
          Some
            (fun k ->
              let out = shrink t_in limits in
              let pads =
                Array.mapi
                  (fun i (start, _) ->
                    let total = (T.shape t_in).(i) in
                    let len = (T.shape out).(i) in
                    (start, total - start - len))
                  limits
              in
              pull1 k out t_in (fun g ->
                  pad g pads (Nx_core.Dtype.zero (dtype t_in))))
      | E_flip { t_in; dims_to_flip } ->
          Some
            (fun k ->
              pull1 k (flip t_in dims_to_flip) t_in (fun g ->
                  flip g dims_to_flip))
      | E_cat { t_list; axis } ->
          Some
            (fun k ->
              let out = cat t_list ~axis in
              if List.exists tracked t_list then begin
                track out;
                Tape.record tape (fun () ->
                    match Tape.find tape out with
                    | None -> ()
                    | Some g ->
                        let g_shape = T.shape g in
                        let off = ref 0 in
                        List.iter
                          (fun x ->
                            let len = (T.shape x).(axis) in
                            let limits =
                              Array.init (Array.length g_shape) (fun i ->
                                  if i = axis then (!off, !off + len)
                                  else (0, g_shape.(i)))
                            in
                            off := !off + len;
                            if tracked x then
                              Tape.accumulate tape x (T.shrink limits g))
                          t_list)
              end;
              continue k out)
      | E_cast { t_in; target_dtype } ->
          Some
            (fun k ->
              pull1 k (cast ~dtype:target_dtype t_in) t_in (fun g ->
                  T.cast (dtype t_in) g))
      | E_contiguous { t_in } ->
          Some (fun k -> pull1 k (contiguous t_in) t_in Fun.id)
      | E_copy { t_in } -> Some (fun k -> pull1 k (copy t_in) t_in Fun.id)
      (* Reductions *)
      | E_reduce_sum { t_in; axes; keepdims } ->
          Some
            (fun k ->
              let shape_in = T.shape t_in in
              pull1 k (reduce_sum ~axes ~keepdims t_in) t_in (fun g ->
                  let g =
                    if keepdims then g
                    else
                      let kept =
                        T.shape
                          (T.sum t_in ~axes:(Array.to_list axes) ~keepdims:true)
                      in
                      T.reshape kept g
                  in
                  T.broadcast_to shape_in g))
      | E_reduce_max { t_in; axes; keepdims } ->
          Some
            (fun k ->
              let out = reduce_max ~axes ~keepdims t_in in
              let shape_in = T.shape t_in in
              let broadcast_kept x =
                if keepdims then T.broadcast_to shape_in x
                else
                  let kept =
                    T.shape
                      (T.max t_in ~axes:(Array.to_list axes) ~keepdims:true)
                  in
                  T.broadcast_to shape_in (T.reshape kept x)
              in
              pull1 k out t_in (fun g ->
                  let mask =
                    T.cast (dtype out) (T.equal t_in (broadcast_kept out))
                  in
                  T.mul (broadcast_kept g) mask))
      | E_reduce_min { t_in; axes; keepdims } ->
          Some
            (fun k ->
              let out = reduce_min ~axes ~keepdims t_in in
              let shape_in = T.shape t_in in
              let broadcast_kept x =
                if keepdims then T.broadcast_to shape_in x
                else
                  let kept =
                    T.shape
                      (T.min t_in ~axes:(Array.to_list axes) ~keepdims:true)
                  in
                  T.broadcast_to shape_in (T.reshape kept x)
              in
              pull1 k out t_in (fun g ->
                  let mask =
                    T.cast (dtype out) (T.equal t_in (broadcast_kept out))
                  in
                  T.mul (broadcast_kept g) mask))
      | E_reduce_prod { t_in; axes; keepdims } ->
          Some
            (fun k ->
              let out = reduce_prod ~axes ~keepdims t_in in
              let shape_in = T.shape t_in in
              let broadcast_kept x =
                if keepdims then T.broadcast_to shape_in x
                else
                  let kept =
                    T.shape
                      (T.prod t_in ~axes:(Array.to_list axes) ~keepdims:true)
                  in
                  T.broadcast_to shape_in (T.reshape kept x)
              in
              pull1 k out t_in (fun g ->
                  T.mul (broadcast_kept g) (T.div (broadcast_kept out) t_in)))
      (* Sorting: a sort is a gather at the argsort indices. *)
      | E_sort { t_in; axis; descending } ->
          Some
            (fun k ->
              pull1 k (sort ~axis ~descending t_in) t_in (fun g ->
                  let indices = argsort ~axis ~descending t_in in
                  scatter ~mode:`Add (T.zeros_like t_in) ~indices ~updates:g
                    ~axis))
      (* Scans *)
      | E_associative_scan { t_in; axis; op } ->
          Some
            (fun k ->
              let out = associative_scan ~axis ~op t_in in
              let shape_in = T.shape t_in in
              let axis_norm =
                let rank = Array.length shape_in in
                if axis < 0 then axis + rank else axis
              in
              pull1 k out t_in (fun g ->
                  match op with
                  | `Sum ->
                      let flipped = T.flip g ~axes:[ axis_norm ] in
                      let scanned = T.cumsum ~axis:axis_norm flipped in
                      T.flip scanned ~axes:[ axis_norm ]
                  | `Prod ->
                      let prefix_exclusive axis x =
                        let shape = T.shape x in
                        let pad_config =
                          Array.mapi
                            (fun i _ -> if i = axis then (1, 0) else (0, 0))
                            shape
                        in
                        let one = Nx_core.Dtype.one (T.dtype x) in
                        let padded = T.pad pad_config one x in
                        let slice_specs =
                          Array.map (fun dim -> T.R (0, dim)) shape
                        in
                        T.slice
                          (Array.to_list slice_specs)
                          (T.cumprod ~axis padded)
                      in
                      let suffix_exclusive axis x =
                        let shape = T.shape x in
                        let one = Nx_core.Dtype.one (T.dtype x) in
                        let flipped = T.flip x ~axes:[ axis ] in
                        let suffix_inclusive =
                          T.flip (T.cumprod ~axis flipped) ~axes:[ axis ]
                        in
                        let pad_config =
                          Array.mapi
                            (fun i _ -> if i = axis then (0, 1) else (0, 0))
                            shape
                        in
                        let padded = T.pad pad_config one suffix_inclusive in
                        let slice_specs =
                          Array.mapi
                            (fun i dim ->
                              if i = axis then T.R (1, dim + 1) else T.R (0, dim))
                            shape
                        in
                        T.slice (Array.to_list slice_specs) padded
                      in
                      let divide_no_nan num denom =
                        let zero_mask = T.equal denom (T.zeros_like denom) in
                        let safe_denom =
                          T.where zero_mask (T.ones_like denom) denom
                        in
                        let base = T.div num safe_denom in
                        T.where zero_mask (T.zeros_like base) base
                      in
                      let reverse_cumsum x axis =
                        let flipped = T.flip x ~axes:[ axis ] in
                        T.flip (T.cumsum ~axis flipped) ~axes:[ axis ]
                      in
                      let prefix = prefix_exclusive axis_norm t_in in
                      let suffix = suffix_exclusive axis_norm t_in in
                      let h = divide_no_nan g suffix in
                      let tail_sum = T.sub (reverse_cumsum h axis_norm) h in
                      let inner = T.add g (T.mul suffix tail_sum) in
                      T.mul prefix inner
                  | `Max | `Min ->
                      (* The cotangent flows to positions where the running
                         extremum strictly improves. *)
                      let shape = T.shape out in
                      let dt = dtype t_in in
                      let boundary =
                        match op with
                        | `Max -> Nx_core.Dtype.min_value dt
                        | _ -> Nx_core.Dtype.max_value dt
                      in
                      let pad_left =
                        Array.mapi
                          (fun i _ -> if i = axis_norm then (1, 0) else (0, 0))
                          shape
                      in
                      let padded = T.pad pad_left boundary out in
                      let slice_specs =
                        Array.map (fun dim -> T.R (0, dim)) shape
                      in
                      let shifted =
                        T.slice (Array.to_list slice_specs) padded
                      in
                      let active =
                        match op with
                        | `Max -> T.cmpgt out shifted
                        | _ -> T.cmplt out shifted
                      in
                      T.mul g (T.cast dt active)))
      (* Gather / scatter *)
      | E_gather { data; indices; axis } ->
          Some
            (fun k ->
              pull1 k (gather data indices ~axis) data (fun g ->
                  scatter ~mode:`Add (T.zeros_like data) ~indices ~updates:g
                    ~axis))
      | E_scatter
          { data_template; indices; updates; axis; mode; unique_indices } ->
          Some
            (fun k ->
              let out =
                scatter ~mode ~unique_indices data_template ~indices ~updates
                  ~axis
              in
              let tt = tracked data_template and tu = tracked updates in
              if tt || tu then begin
                track out;
                Tape.record tape (fun () ->
                    match Tape.find tape out with
                    | None -> ()
                    | Some g ->
                        if tu then
                          Tape.accumulate tape updates (gather g indices ~axis);
                        if tt then begin
                          (* Under [`Set] the written positions shadow the
                             template; under [`Add] the template passes through
                             everywhere. *)
                          let gt =
                            match mode with
                            | `Add -> g
                            | `Set ->
                                let mask =
                                  scatter
                                    (T.ones_like data_template)
                                    ~indices ~updates:(T.zeros_like updates)
                                    ~axis
                                in
                                T.mul g mask
                          in
                          Tape.accumulate tape data_template gt
                        end)
              end;
              continue k out)
      (* Windowing: unfold and fold are duals. *)
      | E_unfold { t_in; kernel_size; stride; dilation; padding } ->
          Some
            (fun k ->
              let input_shape = T.shape t_in in
              let num_spatial = Array.length kernel_size in
              let output_size =
                Array.sub input_shape
                  (Array.length input_shape - num_spatial)
                  num_spatial
              in
              pull1 k (unfold t_in ~kernel_size ~stride ~dilation ~padding) t_in
                (fun g ->
                  fold g ~output_size ~kernel_size ~stride ~dilation ~padding))
      | E_fold { t_in; output_size; kernel_size; stride; dilation; padding } ->
          Some
            (fun k ->
              pull1 k
                (fold t_in ~output_size ~kernel_size ~stride ~dilation ~padding)
                t_in (fun g -> unfold g ~kernel_size ~stride ~dilation ~padding))
      (* Matrix multiplication *)
      | E_matmul { a; b } ->
          Some
            (fun k ->
              let out = matmul a b in
              let ta = tracked a and tb = tracked b in
              if ta || tb then begin
                track out;
                Tape.record tape (fun () ->
                    match Tape.find tape out with
                    | None -> ()
                    | Some g ->
                        let a_shape = T.shape a and b_shape = T.shape b in
                        let g_shape = T.shape g in
                        let a_ndim = Array.length a_shape in
                        let b_ndim = Array.length b_shape in
                        let g_ndim = Array.length g_shape in
                        let transpose_last2 x =
                          let nd = Array.length (T.shape x) in
                          if nd < 2 then x
                          else
                            let axes =
                              List.init nd (fun i ->
                                  if i = nd - 2 then -1
                                  else if i = nd - 1 then -2
                                  else i)
                            in
                            T.transpose ~axes x
                        in
                        if ta then begin
                          let grad_a =
                            if a_ndim = 2 && b_ndim >= 3 then
                              let g_bt = T.matmul g (transpose_last2 b) in
                              let batch_dims = List.init (g_ndim - 2) Fun.id in
                              if batch_dims = [] then g_bt
                              else T.sum g_bt ~axes:batch_dims ~keepdims:false
                            else if a_ndim >= 3 && b_ndim >= 3 then
                              T.matmul g (transpose_last2 b)
                            else T.matmul g (T.transpose b)
                          in
                          Tape.accumulate tape a grad_a
                        end;
                        if tb then begin
                          let grad_b =
                            if b_ndim = 2 && a_ndim >= 3 then
                              let at_g = T.matmul (transpose_last2 a) g in
                              let batch_dims = List.init (g_ndim - 2) Fun.id in
                              if batch_dims = [] then at_g
                              else T.sum at_g ~axes:batch_dims ~keepdims:false
                            else if a_ndim = 2 && b_ndim >= 3 then
                              let a_t = T.transpose a in
                              let batch_shape =
                                Array.sub g_shape 0 (g_ndim - 2)
                              in
                              let a_t_shape = T.shape a_t in
                              let target_shape =
                                Array.concat [ batch_shape; a_t_shape ]
                              in
                              let a_t_expanded =
                                T.broadcast_to target_shape
                                  (T.reshape
                                     (Array.concat [ [| 1 |]; a_t_shape ])
                                     a_t)
                              in
                              T.matmul a_t_expanded g
                            else if a_ndim >= 3 && b_ndim >= 3 then
                              T.matmul (transpose_last2 a) g
                            else T.matmul (T.transpose a) g
                          in
                          Tape.accumulate tape b grad_b
                        end)
              end;
              continue k out)
      (* FFT: fft and ifft are duals; the real-valued variants have no rule
         yet. *)
      | E_fft { t; axes } ->
          Some (fun k -> pull1 k (fft t ~axes) t (fun g -> ifft g ~axes))
      | E_ifft { t; axes } ->
          Some (fun k -> pull1 k (ifft t ~axes) t (fun g -> fft g ~axes))
      | E_rfft { t; axes } ->
          Some
            (fun k ->
              no_rule k "rfft" (tracked t) (fun () ->
                  rfft t ~dtype:Nx_core.Dtype.complex128 ~axes))
      | E_irfft { t; axes; s } ->
          Some
            (fun k ->
              no_rule k "irfft" (tracked t) (fun () ->
                  irfft t ~axes ?s ~dtype:Nx_core.Dtype.float64))
      | E_psum { t_in } ->
          Some
            (fun k -> no_rule k "psum" (tracked t_in) (fun () -> op_psum t_in))
      (* Linear algebra *)
      | E_cholesky { t_in; upper } ->
          Some
            (fun k ->
              let l = cholesky ~upper t_in in
              pull1 k l t_in (fun dl ->
                  let l_lower, dl_lower =
                    if upper then (T.transpose l, T.transpose dl) else (l, dl)
                  in
                  let c = T.matmul (T.transpose l_lower) dl_lower in
                  let p =
                    let diag_c = T.diagonal c in
                    let two = T.add (T.ones_like diag_c) (T.ones_like diag_c) in
                    T.sub (T.tril c) (T.diag (T.div diag_c two))
                  in
                  let z =
                    triangular_solve ~upper:false ~transpose:true
                      ~unit_diag:false l_lower p
                  in
                  let y =
                    triangular_solve ~upper:false ~transpose:true
                      ~unit_diag:false l_lower (T.transpose z)
                  in
                  let s = T.transpose y in
                  let da_sym =
                    T.sub (T.add s (T.transpose s)) (T.diag (T.diagonal s))
                  in
                  T.tril da_sym))
      | E_triangular_solve { a; b; upper; transpose; unit_diag } ->
          Some
            (fun k ->
              let out = triangular_solve ~upper ~transpose ~unit_diag a b in
              let ta = tracked a and tb = tracked b in
              if ta || tb then begin
                track out;
                Tape.record tape (fun () ->
                    match Tape.find tape out with
                    | None -> ()
                    | Some g ->
                        let grad_b =
                          triangular_solve ~upper ~transpose:(not transpose)
                            ~unit_diag a g
                        in
                        if tb then Tape.accumulate tape b grad_b;
                        if ta then begin
                          let out_2d, grad_b_2d =
                            if Array.length (T.shape g) = 1 then
                              ( T.expand_dims [ -1 ] out,
                                T.expand_dims [ -1 ] grad_b )
                            else (out, grad_b)
                          in
                          let grad_a_full =
                            if transpose then
                              T.neg (T.matmul out_2d (T.transpose grad_b_2d))
                            else T.neg (T.matmul grad_b_2d (T.transpose out_2d))
                          in
                          let grad_a =
                            if upper then T.triu grad_a_full
                            else T.tril grad_a_full
                          in
                          Tape.accumulate tape a grad_a
                        end)
              end;
              continue k out)
      | E_qr { t_in; reduced } ->
          Some
            (fun k ->
              let q, r = qr ~reduced t_in in
              if tracked t_in then begin
                track q;
                track r;
                Tape.record tape (fun () ->
                    match (Tape.find tape q, Tape.find tape r) with
                    | None, None -> ()
                    | found_q, found_r ->
                        let gq =
                          match found_q with
                          | Some g -> g
                          | None -> T.zeros_like q
                        in
                        let gr =
                          match found_r with
                          | Some g -> T.transpose (T.tril (T.transpose g))
                          | None -> T.zeros_like r
                        in
                        let m =
                          T.sub
                            (T.matmul r (T.transpose gr))
                            (T.matmul (T.transpose gq) q)
                        in
                        let lower_strict = T.tril ~k:(-1) m in
                        let diag_mat = T.diag (T.contiguous (T.diagonal m)) in
                        let copyltu =
                          T.add
                            (T.add lower_strict (T.transpose lower_strict))
                            diag_mat
                        in
                        let rhs = T.add gq (T.matmul q copyltu) in
                        let da_t =
                          triangular_solve ~upper:true ~transpose:false
                            ~unit_diag:false r (T.transpose rhs)
                        in
                        Tape.accumulate tape t_in (T.transpose da_t))
              end;
              continue k (q, r))
      | E_svd { t_in; full_matrices } ->
          Some
            (fun k ->
              no_rule k "svd" (tracked t_in) (fun () -> svd ~full_matrices t_in))
      | E_eig { t_in; vectors } ->
          Some
            (fun k ->
              no_rule k "eig" (tracked t_in) (fun () -> eig ~vectors t_in))
      | E_eigh { t_in; vectors } ->
          Some
            (fun k ->
              no_rule k "eigh" (tracked t_in) (fun () -> eigh ~vectors t_in))
      (* Custom rules. The forward function runs in the enclosing context: this
         handler replaces its internals with the user's rule, while enclosing
         transformations see the forward computation itself. *)
      | Custom.E_custom_vjp (Custom.Vjp_call { tree; params; fwd; bwd }) ->
          Some
            (fun k ->
              let (module Q) = tree in
              let any = ref false in
              Q.iter (fun leaf -> if tracked leaf then any := true) params;
              let y, res = fwd params in
              if !any then begin
                track y;
                Tape.record tape (fun () ->
                    match Tape.find tape y with
                    | None -> ()
                    | Some ct ->
                        let grads = bwd res ct in
                        ignore
                          (Q.map2
                             (fun leaf g ->
                               if tracked leaf then Tape.accumulate tape leaf g;
                               leaf)
                             params grads))
              end;
              continue k y)
      | Custom.E_custom_jvp (Custom.Jvp_call { tree; params; f; _ }) ->
          Some
            (fun k ->
              let (module Q) = tree in
              let any = ref false in
              Q.iter (fun leaf -> if tracked leaf then any := true) params;
              if !any then
                invalid_arg
                  "Rune: a custom_jvp function is not reverse-differentiable; \
                   define a custom_vjp rule instead"
              else continue k (f params))
      (* Effects from other libraries fall through. A new Nx tensor operation
         must be added to this match: an unmatched tensor effect would be
         differentiated as a constant. *)
      | _ -> None
  in
  { retc = Fun.id; exnc = raise; effc }
