[@@@ocaml.warning "-27"]

(* High-level tensor operations built on backend [B]. *)

module Make (B : Backend_intf.S) = struct
  type ('a, 'b) t = ('a, 'b) B.t
  type context = B.context

  let create_context () = B.create_context ()
  let shape t = View.shape (B.view t)
  let dtype t = B.dtype t
  let strides t = View.strides (B.view t)
  let stride i t = View.stride i (B.view t)
  let dims t = View.dims (B.view t)
  let dim i t = View.dim i (B.view t)
  let ndim t = View.ndim (B.view t)
  let itemsize t = Dtype.itemsize (B.dtype t)
  let size t = View.size (B.view t)
  let numel t = size t
  let nbytes t = numel t * itemsize t
  let offset t = View.offset (B.view t)
  let layout t = View.layout (B.view t)

  (* infer the dimension corresponding to [-1] in [new_shape_spec] *)
  let resolve_neg_one_shape current_shape new_shape_spec =
    let new_shape_spec_l = Array.to_list new_shape_spec in
    let current_numel = View.prod current_shape in
    let neg_one_count =
      new_shape_spec_l |> List.filter (( = ) (-1)) |> List.length
    in
    if neg_one_count > 1 then
      invalid_arg "Reshape target shape can only contain one -1"
    else if neg_one_count = 0 then new_shape_spec
    else
      let specified_numel =
        List.filter (( <> ) (-1)) new_shape_spec_l |> Array.of_list |> View.prod
      in
      (* when shape_spec includes zero dimensions *)
      if specified_numel = 0 then
        if current_numel = 0 then
          Array.map (fun x -> if x = -1 then 0 else x) new_shape_spec
        else
          invalid_arg
            "Reshape cannot infer -1 when other dimensions multiply to 0 but \
             total size is non-zero"
      else if current_numel mod specified_numel <> 0 then
        invalid_arg
          (Printf.sprintf
             "Reshape size mismatch: Cannot reshape %d elements into shape \
              with specified elements %d"
             current_numel specified_numel)
      else
        let inferred_dim = current_numel / specified_numel in
        Array.map (fun s -> if s = -1 then inferred_dim else s) new_shape_spec

  let reshape ctx x shape_spec =
    let new_shape = resolve_neg_one_shape (shape x) shape_spec in
    if shape x = new_shape then x else B.op_reshape ctx x new_shape

  (* reshape and expand [x] to [new_shape] following numpy-style rules *)
  let broadcast_to ctx x new_shape =
    let current_shape = shape x in
    if current_shape = new_shape then x
    else
      let rank_current = Array.length current_shape in
      let rank_new = Array.length new_shape in
      if rank_current > rank_new then
        invalid_arg "Cannot broadcast tensor to fewer dimensions"
      else
        let padded_shape =
          if rank_current < rank_new then
            Array.append (Array.make (rank_new - rank_current) 1) current_shape
          else current_shape
        in
        let compatible = ref true in
        for i = 0 to rank_new - 1 do
          if not (padded_shape.(i) = new_shape.(i) || padded_shape.(i) = 1) then
            compatible := false
        done;
        if not !compatible then
          invalid_arg
            (Printf.sprintf "Cannot broadcast shape %s to %s (padded: %s)"
               (View.pp_int_array current_shape)
               (View.pp_int_array new_shape)
               (View.pp_int_array padded_shape));
        let x_reshaped =
          if padded_shape <> current_shape then reshape ctx x padded_shape
          else x
        in
        B.op_expand ctx x_reshaped new_shape

  (* return [x] and [y] broadcasted to a common shape *)
  let broadcasted ctx ?(reverse = false) x y =
    let a, b = if reverse then (y, x) else (x, y) in
    let broadcast_shape = View.broadcast_shapes (shape a) (shape b) in
    let a_broad = broadcast_to ctx a broadcast_shape in
    let b_broad = broadcast_to ctx b broadcast_shape in
    (a_broad, b_broad)

  (* like [broadcast_to] but [-1] keeps the original dimension *)
  let expand ctx x shape_spec =
    let current_shape = shape x in
    let rank_current = Array.length current_shape in
    let rank_spec = Array.length shape_spec in
    let rank_new = max rank_current rank_spec in
    let current_aligned =
      if rank_current < rank_new then
        Array.append (Array.make (rank_new - rank_current) 1) current_shape
      else current_shape
    in
    let spec_aligned =
      if rank_spec < rank_new then
        Array.append (Array.make (rank_new - rank_spec) (-1)) shape_spec
      else shape_spec
    in
    let new_shape =
      Array.mapi
        (fun i spec_dim ->
          if spec_dim = -1 then current_aligned.(i) else spec_dim)
        spec_aligned
    in
    broadcast_to ctx x new_shape

  let cast (type a b c d) ctx (x : (a, b) t) (dt : (c, d) Dtype.t) : (c, d) t =
    match Dtype.eq_gadt (dtype x) dt with
    | Some Refl ->
        (* Here the compiler now *knows* that [x] has type [(c,d) t], so this
           type-safe “no-op” copy type-checks. *)
        B.op_copy ctx x
    | None -> B.op_cast ctx x dt

  (* ────────── creation ops ────────── *)

  let create ctx dtype shape arr = failwith "todo: create from OCaml array/list"
  let init ctx dtype shape f = failwith "todo: init with function"

  let empty ctx dtype shape_arr =
    let numel = View.prod shape_arr in
    let buf = B.op_buffer ctx dtype numel in
    reshape ctx buf shape_arr

  let full ctx dt target_shape fill_value =
    let scalar_tensor = B.op_const_scalar ctx fill_value dt in
    if Array.length target_shape = 0 then scalar_tensor
    else if View.prod target_shape = 0 then empty ctx dt target_shape
    else
      let rank = Array.length target_shape in
      let intermediate_shape = Array.make rank 1 in
      let reshaped_scalar = reshape ctx scalar_tensor intermediate_shape in
      expand ctx reshaped_scalar target_shape

  let zeros ctx dtype shape_arr =
    let zero_val = Dtype.zero dtype in
    full ctx dtype shape_arr zero_val

  let ones ctx dtype shape_arr =
    let one_val = Dtype.one dtype in
    full ctx dtype shape_arr one_val

  let full_like ctx x_ref fill_value =
    let target_shape = shape x_ref in
    let self_dtype = B.dtype x_ref in
    full ctx self_dtype target_shape fill_value

  let zeros_like ctx x =
    let self_dtype = B.dtype x in
    let zero_val = Dtype.zero self_dtype in
    full_like ctx x zero_val

  let ones_like ctx x =
    let self_dtype = B.dtype x in
    let one_val = Dtype.one self_dtype in
    full_like ctx x one_val

  (* ────────── element-wise binary ops ────────── *)

  let add ctx a b =
    let a', b' = broadcasted ctx a b in
    B.op_add ctx a' b'

  let mul ctx a b =
    let a', b' = broadcasted ctx a b in
    B.op_mul ctx a' b'

  (* neg is defined under unary ops *)
  let sub ctx a b =
    let a', b' = broadcasted ctx a b in
    let neg_b = B.op_neg ctx b' in
    (* neg is defined below *)
    B.op_add ctx a' neg_b

  (* True division, result is float *)
  let div ctx a b =
    let target_dtype = Dtype.Float32 in
    (* Or a more sophisticated Dtype.least_upper_float logic *)
    let a_f = cast ctx a target_dtype in
    let b_f = cast ctx b target_dtype in
    let a_b, b_b = broadcasted ctx a_f b_f in
    B.op_fdiv ctx a_b b_b

  (* Integer division, truncating (already present) *)
  let idiv (type a b) ctx (a : (a, b) t) (b : (a, b) t) : (a, b) t =
    if not (Dtype.is_int (dtype a)) then
      invalid_arg "idiv: operands must be integer types.";
    let a_broad, b_broad = broadcasted ctx a b in
    B.op_idiv ctx a_broad b_broad

  (* Float division (already present) *)
  let fdiv ctx a b =
    let a_broad, b_broad = broadcasted ctx a b in
    B.op_fdiv ctx a_broad b_broad

  let pow ctx a b =
    (* TODO: Tinygrad notes base needs to be float and result might be cast back
       to int. For now, assume B.op_pow handles dtypes appropriately or inputs
       are pre-cast. *)
    let a', b' = broadcasted ctx a b in
    B.op_pow ctx a' b'

  let maximum ctx a b =
    let a', b' = broadcasted ctx a b in
    B.op_max ctx a' b'

  (* minimum defined under unary ops, uses neg and maximum *)

  let modulus ctx a b =
    let a', b' = broadcasted ctx a b in
    B.op_mod ctx a' b'

  let bitwise_xor ctx a b =
    let a', b' = broadcasted ctx a b in
    B.op_xor ctx a' b'

  let bitwise_or ctx a b =
    let a', b' = broadcasted ctx a b in
    B.op_or ctx a' b'

  let bitwise_and ctx a b =
    let a', b' = broadcasted ctx a b in
    B.op_and ctx a' b'

  (* ────────── comparison ops ────────── *)

  let cmplt ctx a b =
    let a', b' = broadcasted ctx a b in
    B.op_cmplt ctx a' b'

  let cmpne ctx a b =
    let a', b' = broadcasted ctx a b in
    B.op_cmpne ctx a' b'

  (* logical_not is defined under unary ops *)

  let cmpeq ctx a b =
    let ne_result = cmpne ctx a b in
    B.op_neg ctx ne_result

  let cmpgt ctx a b = cmplt ctx b a
  let cmple ctx a b = B.op_neg ctx (cmpgt ctx a b)
  let cmpge ctx a b = B.op_neg ctx (cmplt ctx a b)

  (* ────────── logical ops (for boolean tensors, typically uint8) ────────── *)

  let logical_and ctx a b =
    let a_b, b_b = broadcasted ctx a b in
    B.op_and ctx a_b b_b

  let logical_or ctx a b =
    let a_b, b_b = broadcasted ctx a b in
    B.op_or ctx a_b b_b

  let logical_xor ctx a b =
    let a_b, b_b = broadcasted ctx a b in
    B.op_xor ctx a_b b_b

  (* ────────── element-wise unary ops ────────── *)

  let neg ctx x = B.op_neg ctx x
  let logical_not ctx a = B.op_neg ctx a

  let bitwise_not ctx x =
    let dt = dtype x in
    let minus_one_val = Dtype.minus_one dt in
    let minus_one_tensor = B.op_const_scalar ctx minus_one_val dt in
    let minus_one_b = broadcast_to ctx minus_one_tensor (shape x) in
    B.op_xor ctx x minus_one_b

  (* Math functions - assume float inputs as per B.op signatures *)
  let log2 ctx x = B.op_log2 ctx x
  let exp2 ctx x = B.op_exp2 ctx x
  let sin ctx x = B.op_sin ctx x
  let sqrt ctx x = B.op_sqrt ctx x
  let recip ctx x = B.op_recip ctx x

  let log ctx x =
    let log2_x = log2 ctx x in
    let ln_2_val = Stdlib.log 2.0 in
    let dt = dtype x in
    let ln_2_tensor = B.op_const_scalar ctx ln_2_val dt in
    let ln_2_b = broadcast_to ctx ln_2_tensor (shape log2_x) in
    B.op_mul ctx log2_x ln_2_b

  let exp ctx x =
    let one_over_ln_2_val = 1.0 /. Stdlib.log 2.0 in
    let dt = dtype x in
    let factor_tensor = B.op_const_scalar ctx one_over_ln_2_val dt in
    let factor_b = broadcast_to ctx factor_tensor (shape x) in
    let x_scaled = B.op_mul ctx x factor_b in
    B.op_exp2 ctx x_scaled

  let cos ctx x =
    let pi_half_val = Stdlib.acos 0.0 in
    let dt = dtype x in
    let pi_half_tensor = B.op_const_scalar ctx pi_half_val dt in
    let pi_half_b = broadcast_to ctx pi_half_tensor (shape x) in
    let arg_to_sin = sub ctx pi_half_b x in
    B.op_sin ctx arg_to_sin

  let tan ctx x =
    let sin_x = sin ctx x in
    let cos_x = cos ctx x in
    B.op_fdiv ctx sin_x cos_x

  let square ctx x = mul ctx x x

  let abs ctx x =
    let dt = dtype x in
    if Dtype.is_uint dt then x
    else
      let zero_val = Dtype.zero dt in
      let zero_tensor = B.op_const_scalar ctx zero_val dt in
      let zero_b = broadcast_to ctx zero_tensor (shape x) in
      let cond = cmplt ctx x zero_b in
      (* x < 0 *)
      let neg_x = neg ctx x in
      B.op_where ctx cond neg_x x

  let sign ctx x =
    let dt = dtype x in
    let zero_val = Dtype.zero dt in
    let one_val = Dtype.one dt in
    if Dtype.is_uint dt then full_like ctx x one_val
    else
      let zero_t = full_like ctx x zero_val in
      let one_t = full_like ctx x one_val in
      let minus_one_val = Dtype.minus_one dt in
      let minus_one_t = full_like ctx x minus_one_val in

      let is_positive = cmpgt ctx x zero_t in
      let is_negative = cmplt ctx x zero_t in

      let result = B.op_where ctx is_positive one_t zero_t in
      B.op_where ctx is_negative minus_one_t result

  (* Activations & related *)
  let relu ctx x =
    let dt = dtype x in
    let zero_val = Dtype.zero dt in
    let zero_tensor = B.op_const_scalar ctx zero_val dt in
    let zero_b = broadcast_to ctx zero_tensor (shape x) in
    maximum ctx x zero_b (* equivalent to (x > 0).where(x, 0) *)

  let sigmoid ctx x =
    (* 1 / (1 + exp(-x)) = 1 / (1 + (exp2(-x / log(2)))) *)
    let dt = dtype x in
    let neg_one_over_log2 = B.op_const_scalar ctx (-1.0 /. Stdlib.log 2.0) dt in
    let one_t = ones_like ctx x in
    let exp_term = exp2 ctx (mul ctx x neg_one_over_log2) in
    recip ctx (add ctx one_t exp_term)

  let hardsigmoid ctx ?(alpha = 1.0 /. 6.0) ?(beta = 0.5) x =
    let dt = dtype x in
    let alpha_t = B.op_const_scalar ctx alpha dt in
    let beta_t = B.op_const_scalar ctx beta dt in
    let one_t = B.op_const_scalar ctx 1.0 dt in

    let term1_arg = add ctx (mul ctx alpha_t x) beta_t in
    let term1 = relu ctx term1_arg in

    let term2_arg = sub ctx term1_arg one_t in
    let term2 = relu ctx term2_arg in
    sub ctx term1 term2

  let rsqrt ctx x = recip ctx (sqrt ctx x)

  (* More trig and hyperbolic, assuming float inputs *)
  let _poly_n_horner_coeffs_first ctx x_tensor coeffs =
    (* coeffs are [P_N, P_{N-1}, ..., P_0] for P_N x^N + ... + P_0 *)
    match coeffs with
    | [] -> invalid_arg "_poly_n_horner_coeffs_first: empty coefficients list"
    | p_n :: ps_from_n_minus_1_to_0 ->
        let dt = dtype x_tensor in
        let acc = full ctx dt (shape x_tensor) p_n in
        (* Initialize with P_N *)
        List.fold_left
          (fun current_acc p_i_val ->
            let p_i_tensor = full ctx dt (shape x_tensor) p_i_val in
            add ctx (mul ctx current_acc x_tensor) p_i_tensor)
          acc ps_from_n_minus_1_to_0

  let asin ctx x =
    (* Based on tinygrad: x = math.pi / 2 - (1.0 - self.abs()).sqrt() * polyN(self.abs(), coefficients); self.sign() * x *)
    (* Coefficients for polyN are P_N, ..., P_0 for P_N t^N + ... + P_0
       tinygrad coeffs: [-0.0012624911, 0.0066700901, -0.0170881256, 0.0308918810, -0.0501743046, 0.0889789874, -0.2145988016, 1.5707963050] *)
    let coeffs =
      [
        -0.0012624911;
        0.0066700901;
        -0.0170881256;
        0.0308918810;
        -0.0501743046;
        0.0889789874;
        -0.2145988016;
        1.5707963050;
      ]
    in
    let dt = dtype x in
    let pi_half_t = full ctx dt (shape x) (Stdlib.Float.pi /. 2.0) in
    let one_t = full ctx dt (shape x) 1.0 in
    let abs_x = abs ctx x in
    let term_sqrt = sqrt ctx (sub ctx one_t abs_x) in
    let poly_val = _poly_n_horner_coeffs_first ctx abs_x coeffs in
    let val_before_sign = sub ctx pi_half_t (mul ctx term_sqrt poly_val) in
    mul ctx (sign ctx x) val_before_sign

  let acos ctx x =
    let dt = dtype x in
    let pi_half_t = full ctx dt (shape x) (Stdlib.Float.pi /. 2.0) in
    sub ctx pi_half_t (asin ctx x)

  let atan ctx x =
    (* (self / (1 + self * self).sqrt()).asin() *)
    let dt = dtype x in
    let one_t = full ctx dt (shape x) 1.0 in
    let x_squared = square ctx x in
    let denominator = sqrt ctx (add ctx one_t x_squared) in
    asin ctx (fdiv ctx x denominator)

  let sinh ctx x =
    (* (exp(x) - exp(-x)) / 2 *)
    let dt = dtype x in
    let two_t = full ctx dt (shape x) 2.0 in
    let exp_x = exp ctx x in
    let exp_neg_x = exp ctx (neg ctx x) in
    fdiv ctx (sub ctx exp_x exp_neg_x) two_t

  let cosh ctx x =
    (* (exp(x) + exp(-x)) / 2 *)
    let dt = dtype x in
    let two_t = full ctx dt (shape x) 2.0 in
    let exp_x = exp ctx x in
    let exp_neg_x = exp ctx (neg ctx x) in
    fdiv ctx (add ctx exp_x exp_neg_x) two_t

  let tanh ctx x =
    (* 2.0 * sigmoid(2.0 * x) - 1.0 *)
    let dt = dtype x in
    let one_t = full ctx dt (shape x) 1.0 in
    let two_t = full ctx dt (shape x) 2.0 in
    let sigmoid_arg = mul ctx two_t x in
    let sigmoid_val = sigmoid ctx sigmoid_arg in
    sub ctx (mul ctx two_t sigmoid_val) one_t

  let asinh ctx x =
    (* log(x + sqrt(x^2 + 1)) *)
    let dt = dtype x in
    let one_t = full ctx dt (shape x) 1.0 in
    let x_squared = square ctx x in
    let sqrt_term = sqrt ctx (add ctx x_squared one_t) in
    log ctx (add ctx x sqrt_term)

  let acosh ctx x =
    (* log(x + sqrt(x^2 - 1)) *)
    let dt = dtype x in
    let one_t = full ctx dt (shape x) 1.0 in
    let x_squared = square ctx x in
    let sqrt_term = sqrt ctx (sub ctx x_squared one_t) in
    log ctx (add ctx x sqrt_term)

  let atanh ctx x =
    (* log((1+x)/(1-x)) / 2 *)
    let dt = dtype x in
    let one_t = full ctx dt (shape x) 1.0 in
    let two_t = full ctx dt (shape x) 2.0 in
    let term_plus = add ctx one_t x in
    let term_minus = sub ctx one_t x in
    fdiv ctx (log ctx (fdiv ctx term_plus term_minus)) two_t

  (* Rounding, properties *)
  let trunc ctx x =
    (* Cast to int (truncates), then cast back to float *)
    let original_dt = dtype x in
    cast ctx (cast ctx x Dtype.int32) original_dt

  let ceil ctx x =
    (* (x > trunc(x)).where(trunc(x)+1, trunc(x)) *)
    let dt = dtype x in
    let one_t = full ctx dt (shape x) 1.0 in
    let trunc_x = trunc ctx x in
    let cond = cmpgt ctx x trunc_x in
    B.op_where ctx cond (add ctx trunc_x one_t) trunc_x

  let floor ctx x =
    (* (x < trunc(x)).where(trunc(x)-1, trunc(x)) *)
    let dt = dtype x in
    let one_t = full ctx dt (shape x) 1.0 in
    let trunc_x = trunc ctx x in
    let cond = cmplt ctx x trunc_x in
    B.op_where ctx cond (sub ctx trunc_x one_t) trunc_x

  (* Simplified round: round half away from zero. Python's `round` is half to
     even. *)
  let round ctx x =
    (* sign(x) * floor(abs(x) + 0.5) *)
    let dt = dtype x in
    let half_t = full ctx dt (shape x) 0.5 in
    let abs_x = abs ctx x in
    let floor_term = floor ctx (add ctx abs_x half_t) in
    mul ctx (sign ctx x) floor_term

  let isinf ctx x =
    let dt = dtype x in
    if not (Dtype.is_float dt) then zeros ctx Dtype.uint8 (shape x)
    else
      let pos_inf_const = B.op_const_scalar ctx Float.infinity dt in
      let neg_inf_const = B.op_const_scalar ctx Float.neg_infinity dt in
      let is_pos_inf = cmpeq ctx x (broadcast_to ctx pos_inf_const (shape x)) in
      let is_neg_inf = cmpeq ctx x (broadcast_to ctx neg_inf_const (shape x)) in
      logical_or ctx is_pos_inf is_neg_inf

  let isnan ctx x =
    let dt = dtype x in
    if not (Dtype.is_float dt) then zeros ctx Dtype.uint8 (shape x)
    else cmpne ctx x x

  let isfinite ctx x =
    let dt = dtype x in
    if not (Dtype.is_float dt) then ones ctx Dtype.uint8 (shape x)
    else logical_not ctx (logical_or ctx (isinf ctx x) (isnan ctx x))

  let lerp ctx start_tensor end_tensor weight =
    let end_minus_start = sub ctx end_tensor start_tensor in
    let weighted_diff = mul ctx end_minus_start weight in
    add ctx start_tensor weighted_diff

  (* Scalar version of lerp weight *)
  let lerp_scalar_weight ctx start_tensor end_tensor weight_val =
    let dt = dtype start_tensor in
    let weight_tensor = full ctx dt (shape start_tensor) weight_val in
    lerp ctx start_tensor end_tensor weight_tensor

  let lshift ctx x shift_val =
    let dt = dtype x in
    if not (Dtype.is_int dt) then
      invalid_arg
        ("lshift: unsupported dtype: " ^ Dtype.to_string dt
       ^ ". Expected integer type.");

    if shift_val < 0 then invalid_arg "lshift: shift_val must be non-negative";

    if shift_val = 0 then x
    else
      let factor_val = Dtype.power_of_two dt shift_val in
      let factor_tensor = B.op_const_scalar ctx factor_val dt in
      let factor_b = broadcast_to ctx factor_tensor (shape x) in
      B.op_mul ctx x factor_b

  let rshift ctx x shift_val =
    let dt = dtype x in
    if not (Dtype.is_int dt) then
      invalid_arg
        ("rshift: unsupported dtype: " ^ Dtype.to_string dt
       ^ ". Expected integer type.");

    if shift_val < 0 then invalid_arg "rshift: shift_val must be non-negative";

    if shift_val = 0 then x
    else
      let divisor_val = Dtype.power_of_two dt shift_val in
      let divisor_tensor = B.op_const_scalar ctx divisor_val dt in
      let divisor_b = broadcast_to ctx divisor_tensor (shape x) in
      B.op_idiv ctx x divisor_b

  (* Minimum using neg and maximum (already present from original file) *)
  let minimum ctx a b =
    let a_neg = neg ctx a in
    let b_neg = neg ctx b in
    let max_neg = maximum ctx a_neg b_neg in
    neg ctx max_neg

  let clamp ctx x ?min_val ?max_val () =
    let x_clamped_min =
      match min_val with
      | None -> x
      | Some min_v ->
          let min_t = full_like ctx x min_v in
          maximum ctx x min_t
    in
    match max_val with
    | None -> x_clamped_min
    | Some max_v ->
        let max_t = full_like ctx x_clamped_min max_v in
        (* Use x_clamped_min's dtype *)
        minimum ctx x_clamped_min max_t

  let clip = clamp

  (* ────────── ternary ops ────────── *)

  (* select between [if_true] and [if_false] based on [cond] *)
  let where ctx cond if_true if_false =
    let s_true = shape if_true in
    let s_false = shape if_false in
    let s_cond = shape cond in
    (* Broadcast all three to a common shape. Order matters for shape inference.
       First, find common shape for if_true and if_false. *)
    let target_data_shape = View.broadcast_shapes s_true s_false in
    (* Then, find common shape for that and cond. *)
    let final_target_shape = View.broadcast_shapes target_data_shape s_cond in

    let cond_b = broadcast_to ctx cond final_target_shape in
    let if_true_b = broadcast_to ctx if_true final_target_shape in
    let if_false_b = broadcast_to ctx if_false final_target_shape in
    B.op_where ctx cond_b if_true_b if_false_b

  (* ────────── reduction ops ────────── *)

  let sum ctx ?axes ?(keepdims = false) x : ('a, 'b) t =
    let current_shape = shape x in
    let rank = Array.length current_shape in
    let axes_to_reduce =
      match axes with
      | None -> Array.init rank Fun.id
      | Some ax_list ->
          Array.map (fun ax -> if ax < 0 then ax + rank else ax) ax_list
    in
    B.op_reduce_sum ctx x ~axes:axes_to_reduce ~keepdims

  let max_reduce ctx ?axes ?(keepdims = false) x : ('a, 'b) t =
    let rank = Array.length (shape x) in
    let axes_to_reduce =
      match axes with
      | None -> Array.init rank Fun.id
      | Some ax_list ->
          Array.map (fun ax -> if ax < 0 then ax + rank else ax) ax_list
    in
    B.op_reduce_max ctx x ~axes:axes_to_reduce ~keepdims

  (* Alias max to max_reduce to avoid conflict with element-wise maximum *)
  let reduce_max = max_reduce

  let prod ctx ?axes ?(keepdims = false) x : ('a, 'b) t =
    let rank = Array.length (shape x) in
    let axes_to_reduce =
      match axes with
      | None -> Array.init rank Fun.id
      | Some ax_list ->
          Array.map (fun ax -> if ax < 0 then ax + rank else ax) ax_list
    in
    B.op_reduce_prod ctx x ~axes:axes_to_reduce ~keepdims

  (* ────────── movement ops ────────── *)
  let permute ctx x axes_param =
    let rank = ndim x in
    let axes =
      Array.map (fun ax -> if ax < 0 then ax + rank else ax) axes_param
    in
    B.op_permute ctx x axes

  let pad ctx x padding_config fill_value =
    B.op_pad ctx x padding_config fill_value

  let shrink ctx x shrink_args = B.op_shrink ctx x shrink_args

  let flip ctx x (flip_axes_bools : bool array) =
    B.op_flip ctx x flip_axes_bools

  let contiguous ctx x = B.op_contiguous ctx x

  (* collapse dimensions between [start_dim] and [end_dim] *)
  let flatten ctx ?(start_dim = 0) ?(end_dim = -1) x =
    let sh = shape x in
    let r = Array.length sh in
    let s_orig = start_dim in
    let e_orig = end_dim in
    let s = if s_orig < 0 then s_orig + r else s_orig in
    let e = if e_orig < 0 then e_orig + r else e_orig in

    if
      not
        ((s >= 0 && s < r && e >= 0 && e < r)
        || (r = 0 && (s = 0 || s_orig = 0) && (e = -1 || e_orig = -1)))
    then
      invalid_arg
        (Printf.sprintf
           "flatten: start_dim %d or end_dim %d out of bounds for rank %d"
           start_dim end_dim r);
    if s > e then invalid_arg "flatten: start_dim must be <= end_dim";

    let new_shape_list =
      if r = 0 then [ 1 ] (* Flatten scalar to shape [1] *)
      else if s = 0 && e = r - 1 then [ View.prod sh ] (* Flatten all to 1D *)
      else
        let pre = Array.to_list (Array.sub sh 0 s) in
        let mid_slice = Array.sub sh s (e - s + 1) in
        let mid_prod =
          if Array.length mid_slice = 0 then 1 else View.prod mid_slice
        in
        let post = Array.to_list (Array.sub sh (e + 1) (r - (e + 1))) in
        pre @ [ mid_prod ] @ post
    in
    reshape ctx x (Array.of_list new_shape_list)

  (* drop axes of size 1; [axis] restricts the squeeze *)
  let squeeze ctx ?axis x =
    let sh = shape x in
    match axis with
    | None ->
        let new_shape_list = List.filter (( <> ) 1) (Array.to_list sh) in
        let new_shape = Array.of_list new_shape_list in
        if Array.length new_shape = 0 && Array.length sh > 0 then
          (* Squeezing all dims of a non-scalar results in a tensor of shape [1]
             (scalar-like but still a tensor view) *)
          reshape ctx x [||]
          (* Actually, should be scalar shape. If backend handles shape [| |] as
             scalar. *)
        else if Array.length new_shape = 0 && Array.length sh = 0 then x
          (* scalar to scalar *)
        else reshape ctx x new_shape
    | Some ax_val ->
        let r = Array.length sh in
        if r = 0 then x (* Cannot squeeze a scalar *)
        else
          let ax = if ax_val < 0 then ax_val + r else ax_val in
          if ax < 0 || ax >= r then invalid_arg "squeeze: axis out of bounds"
          else if sh.(ax) <> 1 then x
          else
            let sh_list = Array.to_list sh in
            let new_shape_list = List.filteri (fun i _ -> i <> ax) sh_list in
            let new_shape = Array.of_list new_shape_list in
            if Array.length new_shape = 0 && Array.length sh > 0 then
              reshape ctx x [||]
            else if Array.length new_shape = 0 && Array.length sh = 0 then x
            else reshape ctx x new_shape

  (* insert a size‑1 dimension at [axis] *)
  let unsqueeze ctx ?axis x =
    let sh = shape x in
    let r = Array.length sh in
    let ax_val =
      match axis with
      | None -> invalid_arg "unsqueeze: axis must be specified"
      | Some ax_v -> ax_v
    in
    let ax = if ax_val < 0 then ax_val + r + 1 else ax_val in

    if ax < 0 || ax > r then
      invalid_arg
        (Printf.sprintf "unsqueeze: axis %d out of bounds for rank %d tensor"
           ax_val r);

    let sh_list = Array.to_list sh in
    let rec insert_at_idx current_idx target_idx lst item =
      match lst with
      | [] ->
          if current_idx = target_idx then [ item ]
          else
            (* This case implies target_idx was > length of original list if we
               started at 0 *)
            [ item ]
            (* Insert at end if target_idx == current_idx ==
               len(original_list) *)
      | h :: t ->
          if current_idx = target_idx then item :: lst
          else h :: insert_at_idx (current_idx + 1) target_idx t item
    in
    let new_shape_list =
      if r = 0 && ax = 0 then [ 1 ] else insert_at_idx 0 ax sh_list 1
    in
    reshape ctx x (Array.of_list new_shape_list)

  (* swap two dimensions *)
  let transpose ctx ?(dim0 = -2) ?(dim1 = -1) x =
    let r = ndim x in
    if r < 2 then
      if r = 0 && (dim0 = 0 || dim0 = -1) && (dim1 = 0 || dim1 = -1) then x
        (* Scalar transpose is itself *)
      else if r = 1 && (dim0 = 0 || dim0 = -1) && (dim1 = 0 || dim1 = -1) then x
        (* 1D transpose is itself *)
      else if r < 2 then x (* General case for r < 2 *)
      else
        invalid_arg
          (Printf.sprintf
             "transpose: not enough dimensions (%d) for dim0=%d, dim1=%d" r dim0
             dim1)
    else
      let d0 = if dim0 < 0 then dim0 + r else dim0 in
      let d1 = if dim1 < 0 then dim1 + r else dim1 in
      if d0 < 0 || d0 >= r || d1 < 0 || d1 >= r then
        invalid_arg
          (Printf.sprintf "transpose: dims (%d, %d) out of bounds for rank %d"
             dim0 dim1 r);
      if d0 = d1 then x
      else
        let axes = Array.init r Fun.id in
        let temp = axes.(d0) in
        axes.(d0) <- axes.(d1);
        axes.(d1) <- temp;
        permute ctx x axes
end
