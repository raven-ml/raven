(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Forward-mode differentiation as an effect handler over Nx operations.

   Tangents propagate eagerly: every intercepted operation computes its primal
   by re-performing the operation in the enclosing context, and, if any input
   has a tangent in the store, computes and stores the output tangent
   immediately. There is no tape and no second pass. A tensor absent from the
   store is a constant with zero tangent.

   Tangent arithmetic is re-performed in the enclosing context too, so composing
   with grad (forward-over-reverse and reverse-over-forward) and nesting jvp
   both work.

   Every Nx effect constructor is matched explicitly, with the same three
   deliberate categories as the reverse engine: zero-derivative operations fall
   through untracked; operations with no rule yet raise when an input is active;
   mutation always raises. *)

open Nx_effect
module T = Nx

let err_no_rule op =
  invalid_arg
    (Printf.sprintf
       "Rune: the tangent of %s is not implemented; detach its input if \
        differentiation should not flow through it"
       op)

let handler (tangents : Tensor_map.t) =
  let open Effect.Deep in
  let tangent x = Tensor_map.find tangents x in
  let active x = Option.is_some (tangent x) in
  let tan_or_zeros x =
    match tangent x with Some dx -> dx | None -> T.zeros_like x
  in
  (* Materialize stored tangents: rule outputs can be lazy views (broadcasts,
     transposes), and later rules may reshape them. *)
  let set_tangent out v = Tensor_map.set tangents out (T.contiguous v) in

  (* [lift1 out x dfun] stores [dfun dx] as the tangent of [out] when [x] has
     tangent [dx]. *)
  let lift1 (type a b c d) k (out : (a, b) t) (x : (c, d) t)
      (dfun : (c, d) t -> (a, b) t) =
    (match tangent x with None -> () | Some dx -> set_tangent out (dfun dx));
    continue k out
  in

  (* [lift2 out a b make] stores [make da db] as the tangent of [out] when
     either input is active; the inactive side gets a zero tangent. *)
  let lift2 (type a b) k (out : (a, b) t) (a_in : (a, b) t) (b_in : (a, b) t)
      (make : (a, b) t -> (a, b) t -> (a, b) t) =
    if active a_in || active b_in then
      set_tangent out (make (tan_or_zeros a_in) (tan_or_zeros b_in));
    continue k out
  in

  let no_rule (type c) k (op : string) (inputs_active : bool) (out : unit -> c)
      =
    if inputs_active then err_no_rule op else continue k (out ())
  in

  let effc : type c. c Effect.t -> ((c, _) continuation -> _) option =
   fun eff ->
    if not !Gate.enabled then None
    else
      match eff with
      (* Constants: creation, RNG, metadata. Fresh outputs are inactive. *)
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
                 be used inside jvp — use scatter instead")
      (* Binary arithmetic *)
      | E_add { a; b } -> Some (fun k -> lift2 k (add a b) a b T.add)
      | E_sub { a; b } -> Some (fun k -> lift2 k (sub a b) a b T.sub)
      | E_mul { a; b } ->
          Some
            (fun k ->
              lift2 k (mul a b) a b (fun da db ->
                  T.add (T.mul da b) (T.mul a db)))
      | E_fdiv { a; b } ->
          Some
            (fun k ->
              lift2 k (div a b) a b (fun da db ->
                  T.sub (T.div da b) (T.mul (T.div a (T.mul b b)) db)))
      | E_pow { a; b } ->
          Some
            (fun k ->
              let out = pow a b in
              lift2 k out a b (fun da db ->
                  T.add
                    (T.mul da (Derivs.pow_wrt_base a b))
                    (T.mul db (Derivs.pow_wrt_exp a out))))
      | E_max { a; b } ->
          Some
            (fun k ->
              let out = max a b in
              lift2 k out a b (fun da db ->
                  let mask = T.cast (dtype out) (T.cmpgt a b) in
                  T.add (T.mul da mask)
                    (T.mul db (T.sub (T.ones_like mask) mask))))
      | E_min { a; b } ->
          Some
            (fun k ->
              let out = min a b in
              lift2 k out a b (fun da db ->
                  let mask = T.cast (dtype out) (T.cmplt a b) in
                  T.add (T.mul da mask)
                    (T.mul db (T.sub (T.ones_like mask) mask))))
      | E_atan2 { a; b } ->
          Some
            (fun k ->
              lift2 k (atan2 a b) a b (fun da db ->
                  let denom = T.add (T.mul a a) (T.mul b b) in
                  T.sub (T.mul da (T.div b denom)) (T.mul db (T.div a denom))))
      | E_mod { a; b } ->
          Some
            (fun k ->
              no_rule k "mod" (active a || active b) (fun () -> mod_ a b))
      (* Unary arithmetic *)
      | E_neg { t_in } -> Some (fun k -> lift1 k (neg t_in) t_in T.neg)
      | E_sin { t_in } ->
          Some
            (fun k -> lift1 k (sin t_in) t_in (fun dx -> T.mul dx (T.cos t_in)))
      | E_cos { t_in } ->
          Some
            (fun k ->
              lift1 k (cos t_in) t_in (fun dx -> T.mul dx (T.neg (T.sin t_in))))
      | E_tan { t_in } ->
          Some
            (fun k ->
              lift1 k (tan t_in) t_in (fun dx -> T.mul dx (Derivs.tan' t_in)))
      | E_asin { t_in } ->
          Some
            (fun k ->
              lift1 k (asin t_in) t_in (fun dx -> T.mul dx (Derivs.asin' t_in)))
      | E_acos { t_in } ->
          Some
            (fun k ->
              lift1 k (acos t_in) t_in (fun dx ->
                  T.mul dx (T.neg (Derivs.asin' t_in))))
      | E_atan { t_in } ->
          Some
            (fun k ->
              lift1 k (atan t_in) t_in (fun dx -> T.mul dx (Derivs.atan' t_in)))
      | E_sinh { t_in } ->
          Some
            (fun k ->
              lift1 k (sinh t_in) t_in (fun dx -> T.mul dx (T.cosh t_in)))
      | E_cosh { t_in } ->
          Some
            (fun k ->
              lift1 k (cosh t_in) t_in (fun dx -> T.mul dx (T.sinh t_in)))
      | E_tanh { t_in } ->
          Some
            (fun k ->
              let out = tanh t_in in
              lift1 k out t_in (fun dx -> T.mul dx (Derivs.tanh' out)))
      | E_exp { t_in } ->
          Some
            (fun k ->
              let out = exp t_in in
              lift1 k out t_in (fun dx -> T.mul dx out))
      | E_log { t_in } ->
          Some
            (fun k ->
              lift1 k (log t_in) t_in (fun dx -> T.mul dx (T.recip t_in)))
      | E_sqrt { t_in } ->
          Some
            (fun k ->
              let out = sqrt t_in in
              lift1 k out t_in (fun dx -> T.mul dx (Derivs.sqrt' out)))
      | E_recip { t_in } ->
          Some
            (fun k ->
              lift1 k (recip t_in) t_in (fun dx ->
                  T.mul dx (Derivs.recip' t_in)))
      | E_abs { t_in } ->
          Some
            (fun k ->
              lift1 k (abs t_in) t_in (fun dx -> T.mul dx (T.sign t_in)))
      | E_erf { t_in } ->
          Some
            (fun k ->
              lift1 k (erf t_in) t_in (fun dx -> T.mul dx (Derivs.erf' t_in)))
      (* Selection *)
      | E_where { condition; if_true; if_false } ->
          Some
            (fun k ->
              let out = where condition if_true if_false in
              if active if_true || active if_false then begin
                let mask = T.cast (dtype out) condition in
                let dt_ = tan_or_zeros if_true and df = tan_or_zeros if_false in
                set_tangent out
                  (T.add (T.mul dt_ mask)
                     (T.mul df (T.sub (T.ones_like mask) mask)))
              end;
              continue k out)
      (* Movement: linear ops apply to the tangent unchanged. *)
      | E_reshape { t_in; new_shape } ->
          Some
            (fun k ->
              lift1 k (reshape t_in new_shape) t_in (fun dx ->
                  reshape dx new_shape))
      | E_permute { t_in; axes } ->
          Some
            (fun k ->
              lift1 k (permute t_in axes) t_in (fun dx -> permute dx axes))
      | E_expand { t_in; new_target_shape } ->
          Some
            (fun k ->
              lift1 k (expand t_in new_target_shape) t_in (fun dx ->
                  expand dx new_target_shape))
      | E_pad { t_in; padding_config; fill_value } ->
          Some
            (fun k ->
              (* The fill value is a constant: the tangent pads with zero. *)
              lift1 k (pad t_in padding_config fill_value) t_in (fun dx ->
                  pad dx padding_config (Nx_core.Dtype.zero (dtype t_in))))
      | E_shrink { t_in; limits } ->
          Some
            (fun k ->
              lift1 k (shrink t_in limits) t_in (fun dx -> shrink dx limits))
      | E_flip { t_in; dims_to_flip } ->
          Some
            (fun k ->
              lift1 k (flip t_in dims_to_flip) t_in (fun dx ->
                  flip dx dims_to_flip))
      | E_cat { t_list; axis } ->
          Some
            (fun k ->
              let out = cat t_list ~axis in
              if List.exists active t_list then
                set_tangent out (cat (List.map tan_or_zeros t_list) ~axis);
              continue k out)
      | E_cast { t_in; target_dtype } ->
          Some
            (fun k ->
              lift1 k (cast ~dtype:target_dtype t_in) t_in (fun dx ->
                  T.cast target_dtype dx))
      | E_contiguous { t_in } ->
          Some (fun k -> lift1 k (contiguous t_in) t_in Fun.id)
      | E_copy { t_in } -> Some (fun k -> lift1 k (copy t_in) t_in Fun.id)
      (* Reductions *)
      | E_reduce_sum { t_in; axes; keepdims } ->
          Some
            (fun k ->
              lift1 k (reduce_sum ~axes ~keepdims t_in) t_in (fun dx ->
                  T.sum dx ~axes:(Array.to_list axes) ~keepdims))
      | E_reduce_max { t_in; axes; keepdims } ->
          Some
            (fun k ->
              let out = reduce_max ~axes ~keepdims t_in in
              lift1 k out t_in (fun dx ->
                  let shape_in = T.shape t_in in
                  let out_bc =
                    if keepdims then T.broadcast_to shape_in out
                    else
                      let kept =
                        T.max t_in ~axes:(Array.to_list axes) ~keepdims:true
                      in
                      T.broadcast_to shape_in kept
                  in
                  let mask = T.cast (dtype out) (T.equal t_in out_bc) in
                  T.sum (T.mul dx mask) ~axes:(Array.to_list axes) ~keepdims))
      | E_reduce_min { t_in; axes; keepdims } ->
          Some
            (fun k ->
              let out = reduce_min ~axes ~keepdims t_in in
              lift1 k out t_in (fun dx ->
                  let shape_in = T.shape t_in in
                  let out_bc =
                    if keepdims then T.broadcast_to shape_in out
                    else
                      let kept =
                        T.min t_in ~axes:(Array.to_list axes) ~keepdims:true
                      in
                      T.broadcast_to shape_in kept
                  in
                  let mask = T.cast (dtype out) (T.equal t_in out_bc) in
                  T.sum (T.mul dx mask) ~axes:(Array.to_list axes) ~keepdims))
      | E_reduce_prod { t_in; axes; keepdims } ->
          Some
            (fun k ->
              let out = reduce_prod ~axes ~keepdims t_in in
              lift1 k out t_in (fun dx ->
                  let shape_in = T.shape t_in in
                  let out_bc =
                    if keepdims then T.broadcast_to shape_in out
                    else
                      let kept =
                        T.prod t_in ~axes:(Array.to_list axes) ~keepdims:true
                      in
                      T.broadcast_to shape_in kept
                  in
                  T.sum
                    (T.mul (T.div out_bc t_in) dx)
                    ~axes:(Array.to_list axes) ~keepdims))
      (* Sorting: a sort is a gather at the argsort indices. *)
      | E_sort { t_in; axis; descending } ->
          Some
            (fun k ->
              lift1 k (sort ~axis ~descending t_in) t_in (fun dx ->
                  let indices = argsort ~axis ~descending t_in in
                  gather dx indices ~axis))
      (* Scans *)
      | E_associative_scan { t_in; axis; op } ->
          Some
            (fun k ->
              let out = associative_scan ~axis ~op t_in in
              lift1 k out t_in (fun dx ->
                  match op with
                  | `Sum -> associative_scan ~axis ~op:`Sum dx
                  | `Prod ->
                      (* d cumprod_k = cumprod_k * sum_{i<=k} dx_i / x_i;
                         requires nonzero inputs, like the reverse rule. *)
                      let ratio = T.div dx t_in in
                      T.mul out (associative_scan ~axis ~op:`Sum ratio)
                  | `Max | `Min ->
                      (* The tangent flows from positions where the running
                         extremum strictly improves. *)
                      let shape = T.shape out in
                      let ndim = Array.length shape in
                      let axis_norm = if axis < 0 then axis + ndim else axis in
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
                      let active_mask =
                        match op with
                        | `Max -> T.cmpgt out shifted
                        | _ -> T.cmplt out shifted
                      in
                      (* Positions where the extremum does not improve keep a
                         zero tangent rather than carrying the previous
                         extremum's tangent; this matches the reverse rule (they
                         are transposes of each other). *)
                      T.mul dx (T.cast dt active_mask)))
      (* Gather / scatter *)
      | E_gather { data; indices; axis } ->
          Some
            (fun k ->
              lift1 k (gather data indices ~axis) data (fun dx ->
                  gather dx indices ~axis))
      | E_scatter
          { data_template; indices; updates; axis; mode; unique_indices } ->
          Some
            (fun k ->
              let out =
                scatter ~mode ~unique_indices data_template ~indices ~updates
                  ~axis
              in
              if active data_template || active updates then begin
                let d_template =
                  match mode with
                  | `Add -> tan_or_zeros data_template
                  | `Set ->
                      let mask =
                        scatter
                          (T.ones_like data_template)
                          ~indices ~updates:(T.zeros_like updates) ~axis
                      in
                      T.mul (tan_or_zeros data_template) mask
                in
                let d_updates =
                  scatter ~mode
                    (T.zeros_like data_template)
                    ~indices ~updates:(tan_or_zeros updates) ~axis
                in
                set_tangent out (T.add d_template d_updates)
              end;
              continue k out)
      (* Windowing: unfold and fold are linear. *)
      | E_unfold { t_in; kernel_size; stride; dilation; padding } ->
          Some
            (fun k ->
              lift1 k (unfold t_in ~kernel_size ~stride ~dilation ~padding) t_in
                (fun dx -> unfold dx ~kernel_size ~stride ~dilation ~padding))
      | E_fold { t_in; output_size; kernel_size; stride; dilation; padding } ->
          Some
            (fun k ->
              lift1 k
                (fold t_in ~output_size ~kernel_size ~stride ~dilation ~padding)
                t_in (fun dx ->
                  fold dx ~output_size ~kernel_size ~stride ~dilation ~padding))
      (* Matrix multiplication *)
      | E_matmul { a; b } ->
          Some
            (fun k ->
              let out = matmul a b in
              (match (tangent a, tangent b) with
              | None, None -> ()
              | da, db ->
                  let terms =
                    List.filter_map Fun.id
                      [
                        Option.map (fun da -> matmul da b) da;
                        Option.map (fun db -> matmul a db) db;
                      ]
                  in
                  let tan =
                    match terms with
                    | [ t ] -> t
                    | [ t1; t2 ] -> T.add t1 t2
                    | _ -> assert false
                  in
                  set_tangent out tan);
              continue k out)
      (* FFT: linear operations apply to the tangent. *)
      | E_fft { t; axes } ->
          Some (fun k -> lift1 k (fft t ~axes) t (fun dx -> fft dx ~axes))
      | E_ifft { t; axes } ->
          Some (fun k -> lift1 k (ifft t ~axes) t (fun dx -> ifft dx ~axes))
      | E_rfft { t; dtype; axes } ->
          Some
            (fun k ->
              no_rule k "rfft" (active t) (fun () -> rfft t ~dtype ~axes))
      | E_irfft { t; dtype; axes; s } ->
          Some
            (fun k ->
              no_rule k "irfft" (active t) (fun () -> irfft t ~axes ?s ~dtype))
      | E_psum { t_in } ->
          Some
            (fun k -> no_rule k "psum" (active t_in) (fun () -> op_psum t_in))
      (* Linear algebra *)
      | E_cholesky { t_in; upper } ->
          Some
            (fun k ->
              let l = cholesky ~upper t_in in
              lift1 k l t_in (fun da ->
                  (* dL = L phi(L^-1 dA L^-T), phi = strict lower + half
                     diagonal. *)
                  let l_lower, da_lower =
                    if upper then (T.transpose l, T.transpose da) else (l, da)
                  in
                  let w =
                    triangular_solve ~upper:false ~transpose:false
                      ~unit_diag:false l_lower da_lower
                  in
                  let m =
                    T.transpose
                      (triangular_solve ~upper:false ~transpose:false
                         ~unit_diag:false l_lower (T.transpose w))
                  in
                  let phi =
                    let diag_m = T.diagonal m in
                    let two = T.add (T.ones_like diag_m) (T.ones_like diag_m) in
                    T.sub (T.tril m) (T.diag (T.div diag_m two))
                  in
                  let dl_lower = T.matmul l_lower phi in
                  if upper then T.transpose dl_lower else dl_lower))
      | E_triangular_solve { a; b; upper; transpose; unit_diag } ->
          Some
            (fun k ->
              let out = triangular_solve ~upper ~transpose ~unit_diag a b in
              if active a || active b then begin
                (* A_op X = B, so A_op dX = dB - dA_op X, with dA restricted to
                   the triangle the solve reads. *)
                let db = tan_or_zeros b in
                let rhs =
                  match tangent a with
                  | None -> db
                  | Some da ->
                      let da_used =
                        let tri = if upper then T.triu da else T.tril da in
                        if unit_diag then T.sub tri (T.diag (T.diagonal tri))
                        else tri
                      in
                      let da_op =
                        if transpose then T.transpose da_used else da_used
                      in
                      let out_2d, was_1d =
                        if Array.length (T.shape out) = 1 then
                          (T.expand_dims [ -1 ] out, true)
                        else (out, false)
                      in
                      let prod = T.matmul da_op out_2d in
                      let prod =
                        if was_1d then T.reshape (T.shape b) prod else prod
                      in
                      T.sub db prod
                in
                set_tangent out
                  (triangular_solve ~upper ~transpose ~unit_diag a rhs)
              end;
              continue k out)
      | E_qr { t_in; reduced } ->
          Some
            (fun k ->
              if active t_in then err_no_rule "qr"
              else continue k (qr ~reduced t_in))
      | E_svd { t_in; full_matrices } ->
          Some
            (fun k ->
              no_rule k "svd" (active t_in) (fun () -> svd ~full_matrices t_in))
      | E_eig { t_in; vectors } ->
          Some
            (fun k ->
              no_rule k "eig" (active t_in) (fun () -> eig ~vectors t_in))
      | E_eigh { t_in; vectors } ->
          Some
            (fun k ->
              no_rule k "eigh" (active t_in) (fun () -> eigh ~vectors t_in))
      (* Custom rules. *)
      | Custom.E_custom_jvp (Custom.Jvp_call { tree; params; f; jvp }) ->
          Some
            (fun k ->
              let (module Q) = tree in
              let any = ref false in
              Q.iter (fun leaf -> if active leaf then any := true) params;
              if not !any then continue k (f params)
              else begin
                let dparams = Q.map (fun leaf -> tan_or_zeros leaf) params in
                let y, dy = jvp params dparams in
                set_tangent y dy;
                continue k y
              end)
      | Custom.E_custom_vjp (Custom.Vjp_call { tree; params; fwd; _ }) ->
          Some
            (fun k ->
              let (module Q) = tree in
              let any = ref false in
              Q.iter (fun leaf -> if active leaf then any := true) params;
              if !any then
                invalid_arg
                  "Rune: a custom_vjp function is not forward-differentiable; \
                   define a custom_jvp rule instead"
              else continue k (fst (fwd params)))
      (* Effects from other libraries fall through. A new Nx tensor operation
         must be added to this match: an unmatched tensor effect would be
         differentiated as a constant. *)
      | _ -> None
  in
  { retc = Fun.id; exnc = raise; effc }
