(* High-level tensor operations built on backend [B]. *)

[@@@ocaml.warning "-27"]

module Make (B : Backend_intf.S) = struct
  type ('a, 'b) t = ('a, 'b) B.t
  type context = B.context

  let create_context () = B.create_context ()
  let data t = B.data t
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

  let resolve_axis ?ndim_opt t (axis_opt : int option) =
    let ndim = match ndim_opt with Some n -> n | None -> ndim t in
    match axis_opt with
    | None -> Array.init ndim Fun.id (* all axes *)
    | Some a ->
        let resolved_a = if a < 0 then a + ndim else a in
        [| resolved_a |]

  let resolve_single_axis ?ndim_opt t axis : int =
    let ndim = match ndim_opt with Some n -> n | None -> ndim t in
    if axis < 0 then axis + ndim else axis

  let reshape ctx x shape_spec =
    let new_shape = View.resolve_neg_one_shape (shape x) shape_spec in
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

  let contiguous ctx x = B.op_contiguous ctx x
  let copy ctx t = B.op_copy ctx t

  let blit ctx src dst =
    if shape src <> shape dst then
      invalid_arg
        (Printf.sprintf
           "blit: tensors must have the same shape. src: %s, dst: %s"
           (View.pp_int_array (shape src))
           (View.pp_int_array (shape dst)));
    B.op_assign ctx dst src

  let fill ctx value t = failwith "todo: fill tensor with value"
  let create ctx dtype shape arr = failwith "todo: create from bigarray"
  let init ctx dtype shape f = failwith "todo: init with function"
  let scalar ctx dt value = B.op_const_scalar ctx value dt

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

  let scalar_like ctx x_ref value = scalar ctx (B.dtype x_ref) value

  let empty_like ctx x_ref =
    let self_dtype = B.dtype x_ref in
    let target_shape = shape x_ref in
    empty ctx self_dtype target_shape

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

  let add_s ctx tensor_a scalar_b_val =
    let scalar_b_tensor = scalar_like ctx tensor_a scalar_b_val in
    add ctx tensor_a scalar_b_tensor

  let radd_s ctx scalar_a_val tensor_b =
    let scalar_a_tensor = scalar_like ctx tensor_b scalar_a_val in
    add ctx scalar_a_tensor tensor_b

  let iadd ctx target_tensor value_tensor =
    let value_tensor_broadcasted =
      broadcast_to ctx value_tensor (shape target_tensor)
    in
    let result = B.op_add ctx target_tensor value_tensor_broadcasted in
    B.op_assign ctx target_tensor result;
    target_tensor

  let iadd_s ctx target_tensor scalar_val =
    let scalar_value_tensor = scalar_like ctx target_tensor scalar_val in
    let scalar_broadcasted =
      broadcast_to ctx scalar_value_tensor (shape target_tensor)
    in
    let result = B.op_add ctx target_tensor scalar_broadcasted in
    B.op_assign ctx target_tensor result;
    target_tensor

  let sub ctx a b =
    let a', b' = broadcasted ctx a b in
    let neg_b = B.op_neg ctx b' in
    B.op_add ctx a' neg_b

  let sub_s ctx tensor_a scalar_b_val =
    let scalar_b_tensor = scalar_like ctx tensor_a scalar_b_val in
    sub ctx tensor_a scalar_b_tensor

  let rsub_s ctx scalar_a_val tensor_b =
    let scalar_a_tensor = scalar_like ctx tensor_b scalar_a_val in
    sub ctx scalar_a_tensor tensor_b

  let isub ctx target_tensor value_tensor =
    let value_tensor_broadcasted =
      broadcast_to ctx value_tensor (shape target_tensor)
    in
    let neg_value_tensor = B.op_neg ctx value_tensor_broadcasted in
    let result = B.op_add ctx target_tensor neg_value_tensor in
    B.op_assign ctx target_tensor result;
    target_tensor

  let isub_s ctx target_tensor scalar_val =
    let scalar_value_tensor = scalar_like ctx target_tensor scalar_val in
    let scalar_broadcasted =
      broadcast_to ctx scalar_value_tensor (shape target_tensor)
    in
    let neg_scalar_broadcasted = B.op_neg ctx scalar_broadcasted in
    let result = B.op_add ctx target_tensor neg_scalar_broadcasted in
    B.op_assign ctx target_tensor result;
    target_tensor

  let mul ctx a b =
    let a', b' = broadcasted ctx a b in
    B.op_mul ctx a' b'

  let mul_s ctx tensor_a scalar_b_val =
    let scalar_b_tensor = scalar_like ctx tensor_a scalar_b_val in
    mul ctx tensor_a scalar_b_tensor

  let rmul_s ctx scalar_a_val tensor_b =
    let scalar_a_tensor = scalar_like ctx tensor_b scalar_a_val in
    mul ctx scalar_a_tensor tensor_b

  let imul ctx target_tensor value_tensor =
    let value_tensor_broadcasted =
      broadcast_to ctx value_tensor (shape target_tensor)
    in
    let result = B.op_mul ctx target_tensor value_tensor_broadcasted in
    B.op_assign ctx target_tensor result;
    target_tensor

  let imul_s ctx target_tensor scalar_val =
    let scalar_value_tensor = scalar_like ctx target_tensor scalar_val in
    let scalar_broadcasted =
      broadcast_to ctx scalar_value_tensor (shape target_tensor)
    in
    let result = B.op_mul ctx target_tensor scalar_broadcasted in
    B.op_assign ctx target_tensor result;
    target_tensor

  (* True division, result is float *)
  let div ctx a b =
    let target_dtype = Dtype.Float32 in
    (* Or a more sophisticated Dtype.least_upper_float logic *)
    let a_f = cast ctx a target_dtype in
    let b_f = cast ctx b target_dtype in
    let a_b, b_b = broadcasted ctx a_f b_f in
    B.op_fdiv ctx a_b b_b

  let div_s ctx tensor_a scalar_b_val =
    let scalar_b_tensor = scalar_like ctx tensor_a scalar_b_val in
    div ctx tensor_a scalar_b_tensor

  let rdiv_s ctx scalar_a_val tensor_b =
    let scalar_a_tensor = scalar_like ctx tensor_b scalar_a_val in
    div ctx scalar_a_tensor tensor_b

  (* Integer division, truncating (already present) *)
  let idiv (type a b) ctx (a : (a, b) t) (b : (a, b) t) : (a, b) t =
    if not (Dtype.is_int (dtype a)) then
      invalid_arg "idiv: operands must be integer types.";
    let a_broad, b_broad = broadcasted ctx a b in
    B.op_idiv ctx a_broad b_broad

  let idiv_s ctx tensor_a scalar_b_val =
    let scalar_b_tensor = scalar_like ctx tensor_a scalar_b_val in
    idiv ctx tensor_a scalar_b_tensor

  let ridiv_s ctx scalar_a_val tensor_b =
    let scalar_a_tensor = scalar_like ctx tensor_b scalar_a_val in
    idiv ctx scalar_a_tensor tensor_b

  let iidiv ctx target_tensor value_tensor =
    if not (Dtype.is_int (dtype target_tensor)) then
      invalid_arg "iidiv: target_tensor must be an integer type.";
    if not (Dtype.is_int (dtype value_tensor)) then
      invalid_arg "iidiv: value_tensor must be an integer type.";
    let value_tensor_broadcasted =
      broadcast_to ctx value_tensor (shape target_tensor)
    in
    let result = B.op_idiv ctx target_tensor value_tensor_broadcasted in
    B.op_assign ctx target_tensor result;
    target_tensor

  let iidiv_s ctx target_tensor scalar_val =
    if not (Dtype.is_int (dtype target_tensor)) then
      invalid_arg "iidiv_s: target_tensor must be an integer type.";
    let scalar_value_tensor = scalar_like ctx target_tensor scalar_val in
    let scalar_broadcasted =
      broadcast_to ctx scalar_value_tensor (shape target_tensor)
    in
    let result = B.op_idiv ctx target_tensor scalar_broadcasted in
    B.op_assign ctx target_tensor result;
    target_tensor

  (* Float division (already present) *)
  let fdiv ctx a b =
    let a_broad, b_broad = broadcasted ctx a b in
    B.op_fdiv ctx a_broad b_broad

  let fdiv_s ctx tensor_a scalar_b_val =
    let scalar_b_tensor = scalar_like ctx tensor_a scalar_b_val in
    fdiv ctx tensor_a scalar_b_tensor

  let rfdiv_s ctx scalar_a_val tensor_b =
    let scalar_a_tensor = scalar_like ctx tensor_b scalar_a_val in
    fdiv ctx scalar_a_tensor tensor_b

  let ifdiv ctx target_tensor value_tensor =
    if not (Dtype.is_float (dtype target_tensor)) then
      invalid_arg "ifdiv: target_tensor must be a float type.";
    (* Ensure value_tensor is also float and matches target_tensor's specific
       float type for B.op_fdiv *)
    let value_tensor_casted = cast ctx value_tensor (dtype target_tensor) in
    let value_tensor_broadcasted =
      broadcast_to ctx value_tensor_casted (shape target_tensor)
    in
    let result = B.op_fdiv ctx target_tensor value_tensor_broadcasted in
    B.op_assign ctx target_tensor result;
    target_tensor

  let ifdiv_s ctx target_tensor scalar_val =
    if not (Dtype.is_float (dtype target_tensor)) then
      invalid_arg "ifdiv_s: target_tensor must be a float type.";
    let scalar_value_tensor = scalar_like ctx target_tensor scalar_val in
    let scalar_broadcasted =
      broadcast_to ctx scalar_value_tensor (shape target_tensor)
    in
    let result = B.op_fdiv ctx target_tensor scalar_broadcasted in
    B.op_assign ctx target_tensor result;
    target_tensor

  let pow ctx a b =
    let a', b' = broadcasted ctx a b in
    B.op_pow ctx a' b'

  let pow_s ctx tensor_a scalar_b_val =
    let scalar_b_tensor = scalar_like ctx tensor_a scalar_b_val in
    pow ctx tensor_a scalar_b_tensor

  let rpow_s ctx scalar_a_val tensor_b =
    let scalar_a_tensor = scalar_like ctx tensor_b scalar_a_val in
    pow ctx scalar_a_tensor tensor_b

  let ipow ctx target_tensor value_tensor =
    let value_tensor_broadcasted =
      broadcast_to ctx value_tensor (shape target_tensor)
    in
    let result = B.op_pow ctx target_tensor value_tensor_broadcasted in
    B.op_assign ctx target_tensor result;
    target_tensor

  let ipow_s ctx target_tensor scalar_val =
    let scalar_value_tensor = scalar_like ctx target_tensor scalar_val in
    let scalar_broadcasted =
      broadcast_to ctx scalar_value_tensor (shape target_tensor)
    in
    let result = B.op_pow ctx target_tensor scalar_broadcasted in
    B.op_assign ctx target_tensor result;
    target_tensor

  let maximum ctx a b =
    let a', b' = broadcasted ctx a b in
    B.op_max ctx a' b'

  let maximum_s ctx tensor_a scalar_b_val =
    let scalar_b_tensor = scalar_like ctx tensor_a scalar_b_val in
    maximum ctx tensor_a scalar_b_tensor

  let rmaximum_s ctx scalar_a_val tensor_b =
    let scalar_a_tensor = scalar_like ctx tensor_b scalar_a_val in
    maximum ctx scalar_a_tensor tensor_b

  let imaximum ctx target_tensor value_tensor =
    let value_tensor_broadcasted =
      broadcast_to ctx value_tensor (shape target_tensor)
    in
    let result = B.op_max ctx target_tensor value_tensor_broadcasted in
    B.op_assign ctx target_tensor result;
    target_tensor

  let imaximum_s ctx target_tensor scalar_val =
    let scalar_value_tensor = scalar_like ctx target_tensor scalar_val in
    let scalar_broadcasted =
      broadcast_to ctx scalar_value_tensor (shape target_tensor)
    in
    let result = B.op_max ctx target_tensor scalar_broadcasted in
    B.op_assign ctx target_tensor result;
    target_tensor

  let minimum ctx a b =
    let a', b' = broadcasted ctx a b in
    let a_neg = B.op_neg ctx a' in
    let b_neg = B.op_neg ctx b' in
    let max_neg = B.op_max ctx a_neg b_neg in
    B.op_neg ctx max_neg

  let minimum_s ctx tensor_a scalar_b_val =
    let scalar_b_tensor = scalar_like ctx tensor_a scalar_b_val in
    minimum ctx tensor_a scalar_b_tensor

  let rminimum_s ctx scalar_a_val tensor_b =
    let scalar_a_tensor = scalar_like ctx tensor_b scalar_a_val in
    minimum ctx scalar_a_tensor tensor_b

  let iminimum ctx target_tensor value_tensor =
    let value_tensor_broadcasted =
      broadcast_to ctx value_tensor (shape target_tensor)
    in
    let target_neg = B.op_neg ctx target_tensor in
    let value_b_neg = B.op_neg ctx value_tensor_broadcasted in
    let max_of_negs = B.op_max ctx target_neg value_b_neg in
    let result = B.op_neg ctx max_of_negs in
    B.op_assign ctx target_tensor result;
    target_tensor

  let iminimum_s ctx target_tensor scalar_val =
    let scalar_value_tensor = scalar_like ctx target_tensor scalar_val in
    let scalar_broadcasted =
      broadcast_to ctx scalar_value_tensor (shape target_tensor)
    in
    let target_neg = B.op_neg ctx target_tensor in
    let scalar_b_neg = B.op_neg ctx scalar_broadcasted in
    let max_of_negs = B.op_max ctx target_neg scalar_b_neg in
    let result = B.op_neg ctx max_of_negs in
    B.op_assign ctx target_tensor result;
    target_tensor

  let mod_ ctx a b =
    let a', b' = broadcasted ctx a b in
    B.op_mod ctx a' b'

  let mod_s ctx tensor_a scalar_b_val =
    let scalar_b_tensor = scalar_like ctx tensor_a scalar_b_val in
    mod_ ctx tensor_a scalar_b_tensor

  let rmod_s ctx scalar_a_val tensor_b =
    let scalar_a_tensor = scalar_like ctx tensor_b scalar_a_val in
    mod_ ctx scalar_a_tensor tensor_b

  let imod ctx target_tensor value_tensor =
    if not (Dtype.is_int (dtype target_tensor)) then
      invalid_arg "imod: target_tensor must be an integer type.";
    let value_tensor_broadcasted =
      broadcast_to ctx value_tensor (shape target_tensor)
    in
    let result = B.op_mod ctx target_tensor value_tensor_broadcasted in
    B.op_assign ctx target_tensor result;
    target_tensor

  let imod_s ctx target_tensor scalar_val =
    if not (Dtype.is_int (dtype target_tensor)) then
      invalid_arg "imod_s: target_tensor must be an integer type.";
    let scalar_value_tensor = scalar_like ctx target_tensor scalar_val in
    let scalar_broadcasted =
      broadcast_to ctx scalar_value_tensor (shape target_tensor)
    in
    let result = B.op_mod ctx target_tensor scalar_broadcasted in
    B.op_assign ctx target_tensor result;
    target_tensor

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

  let less ctx a b = cmplt ctx a b

  let cmpne ctx a b =
    let a', b' = broadcasted ctx a b in
    B.op_cmpne ctx a' b'

  let not_equal ctx a b = cmpne ctx a b

  let cmpeq ctx a b =
    let ne_result = cmpne ctx a b in
    B.op_neg ctx ne_result

  let equal ctx a b = cmpeq ctx a b
  let cmpgt ctx a b = cmplt ctx b a
  let greater ctx a b = cmpgt ctx a b
  let cmple ctx a b = B.op_neg ctx (cmpgt ctx a b)
  let less_equal ctx a b = cmple ctx a b
  let cmpge ctx a b = B.op_neg ctx (cmplt ctx a b)
  let greater_equal ctx a b = cmpge ctx a b

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

  let logical_not ctx a = B.op_neg ctx a

  (* ────────── element-wise unary ops ────────── *)

  let neg ctx x = B.op_neg ctx x

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
  let poly_n_horner_coeffs_first ctx x_tensor coeffs =
    (* coeffs are [P_N, P_{N-1}, ..., P_0] for P_N x^N + ... + P_0 *)
    match coeffs with
    | [] -> invalid_arg "poly_n_horner_coeffs_first: empty coefficients list"
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
    let poly_val = poly_n_horner_coeffs_first ctx abs_x coeffs in
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

  let clamp ctx x ?min ?max () =
    let x_clamped_min =
      match min with
      | None -> x
      | Some min_v ->
          let min_t = full_like ctx x min_v in
          maximum ctx x min_t
    in
    match max with
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

  let sum ctx ?axes ?(keepdims = false) x =
    let current_shape = shape x in
    let rank = Array.length current_shape in
    let axes_to_reduce =
      match axes with
      | None -> Array.init rank Fun.id
      | Some ax_list ->
          Array.map (fun ax -> if ax < 0 then ax + rank else ax) ax_list
    in
    B.op_reduce_sum ctx x ~axes:axes_to_reduce ~keepdims

  let max ctx ?axes ?(keepdims = false) x =
    let rank = Array.length (shape x) in
    let axes_to_reduce =
      match axes with
      | None -> Array.init rank Fun.id
      | Some ax_list ->
          Array.map (fun ax -> if ax < 0 then ax + rank else ax) ax_list
    in
    B.op_reduce_max ctx x ~axes:axes_to_reduce ~keepdims

  let min ctx ?axes ?(keepdims = false) x = failwith "todo: min"

  let prod ctx ?axes ?(keepdims = false) x =
    let rank = Array.length (shape x) in
    let axes_to_reduce =
      match axes with
      | None -> Array.init rank Fun.id
      | Some ax_list ->
          Array.map (fun ax -> if ax < 0 then ax + rank else ax) ax_list
    in
    B.op_reduce_prod ctx x ~axes:axes_to_reduce ~keepdims

  let mean ctx ?axes ?(keepdims = false) x_orig =
    let x_dtype = B.dtype x_orig in
    let num_for_sum = sum ctx ?axes ~keepdims x_orig in

    let s_orig = shape x_orig in
    let r_orig = Array.length s_orig in
    let actual_axes_to_reduce =
      match axes with
      | None -> Array.init r_orig Fun.id
      | Some ax_list ->
          Array.map (fun ax -> if ax < 0 then ax + r_orig else ax) ax_list
    in
    let num_elements_in_reduced_dims =
      if Array.length actual_axes_to_reduce = 0 then 1
      else
        View.prod
          (Array.map (fun ax_idx -> s_orig.(ax_idx)) actual_axes_to_reduce)
    in
    let num_elements_divisor_float =
      float_of_int
        (if num_elements_in_reduced_dims = 0 then 1
         else num_elements_in_reduced_dims)
    in

    let divisor_val_ocaml =
      Dtype.float_to_dtype x_dtype num_elements_divisor_float
    in
    let divisor_scalar = scalar ctx x_dtype divisor_val_ocaml in
    let divisor_tensor = broadcast_to ctx divisor_scalar (shape num_for_sum) in

    B.op_fdiv ctx num_for_sum divisor_tensor

  let var ctx ?axes ?(keepdims = false) ?(correction = 1) x_orig =
    let x_dtype = B.dtype x_orig in
    let mean_x_keepdim_true = mean ctx ?axes ~keepdims:true x_orig in

    let diff = sub ctx x_orig mean_x_keepdim_true in
    let diff_sq = square ctx diff in
    let sum_diff_sq = sum ctx ?axes ~keepdims diff_sq in

    let s_orig = shape x_orig in
    let r_orig = Array.length s_orig in
    let actual_axes_to_reduce =
      match axes with
      | None -> Array.init r_orig Fun.id
      | Some ax_list ->
          Array.map (fun ax -> if ax < 0 then ax + r_orig else ax) ax_list
    in
    let num_elements_in_reduced_dims =
      if Array.length actual_axes_to_reduce = 0 then 1
      else
        View.prod
          (Array.map (fun ax_idx -> s_orig.(ax_idx)) actual_axes_to_reduce)
    in

    let n_corrected_val = num_elements_in_reduced_dims - correction in
    let n_corrected_float = float_of_int (Stdlib.max 0 n_corrected_val) in

    let divisor_val_ocaml = Dtype.float_to_dtype x_dtype n_corrected_float in
    let divisor_scalar = scalar ctx x_dtype divisor_val_ocaml in
    let divisor_tensor = broadcast_to ctx divisor_scalar (shape sum_diff_sq) in

    B.op_fdiv ctx sum_diff_sq divisor_tensor

  let std ctx ?axes ?(keepdims = false) ?(correction = 1) x_orig =
    let variance = var ctx ?axes ~keepdims ~correction x_orig in
    sqrt ctx variance

  (* ────────── movement ops ────────── *)

  let pad ctx x padding_config fill_value =
    B.op_pad ctx x padding_config fill_value

  let shrink ctx x shrink_args = B.op_shrink ctx x shrink_args

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

  let transpose ctx ?axes x =
    let r = ndim x in
    let resolved_axes =
      match axes with
      | None -> Array.init r (fun i -> r - 1 - i) (* Reverse dimensions *)
      | Some ax_arr ->
          if Array.length ax_arr <> r then
            invalid_arg
              (Printf.sprintf
                 "transpose: axes length %d does not match tensor rank %d"
                 (Array.length ax_arr) r);
          let seen = Array.make r false in
          Array.iter
            (fun ax_val ->
              let ax = if ax_val < 0 then ax_val + r else ax_val in
              if ax < 0 || ax >= r then
                invalid_arg
                  (Printf.sprintf "transpose: axis %d out of bounds for rank %d"
                     ax_val r);
              if seen.(ax) then
                invalid_arg
                  (Printf.sprintf "transpose: axis %d repeated" ax_val);
              seen.(ax) <- true)
            ax_arr;
          if not (Array.for_all Fun.id seen) then
            invalid_arg "transpose: axes do not form a permutation";
          Array.map
            (fun ax_val -> if ax_val < 0 then ax_val + r else ax_val)
            ax_arr
    in
    B.op_permute ctx x resolved_axes

  let flip ctx ?axes x =
    let r = ndim x in
    let flip_bools = Array.make r false in
    (match axes with
    | None -> Array.fill flip_bools 0 r true (* Flip all axes *)
    | Some ax_arr ->
        Array.iter
          (fun ax_val ->
            let ax = if ax_val < 0 then ax_val + r else ax_val in
            if ax < 0 || ax >= r then
              invalid_arg
                (Printf.sprintf "flip: axis %d out of bounds for rank %d" ax_val
                   r);
            flip_bools.(ax) <- true)
          ax_arr);
    B.op_flip ctx x flip_bools

  let moveaxis ctx src dst x =
    let r = ndim x in
    let norm_src = if src < 0 then src + r else src in
    let norm_dst = if dst < 0 then dst + r else dst in

    if norm_src < 0 || norm_src >= r then
      invalid_arg
        (Printf.sprintf "moveaxis: source axis %d out of bounds for rank %d" src
           r);
    (* Destination can be from 0 to r (inclusive conceptually, for insertion
       before/after existing axes) but for permutation, it must map to an
       existing slot 0 to r-1 *)
    if norm_dst < 0 || norm_dst >= r then
      (* If strictly moving to an existing slot index *)
      invalid_arg
        (Printf.sprintf
           "moveaxis: destination axis %d out of bounds for rank %d" dst r);

    if norm_src = norm_dst then x (* No change *)
    else
      let axes_list = Array.to_list (Array.init r Fun.id) in
      let item_to_move = List.nth axes_list norm_src in
      let list_without_item = List.filter (( <> ) item_to_move) axes_list in

      let rec insert_at idx item lst acc =
        match lst with
        | [] -> List.rev (item :: acc)
        | hd :: tl ->
            if idx = 0 then List.rev_append acc (item :: hd :: tl)
            else insert_at (idx - 1) item tl (hd :: acc)
      in
      let final_axes_list =
        insert_at norm_dst item_to_move list_without_item []
      in
      B.op_permute ctx x (Array.of_list final_axes_list)

  let swapaxes ctx axis1 axis2 x =
    let r = ndim x in
    let norm_axis1 = if axis1 < 0 then axis1 + r else axis1 in
    let norm_axis2 = if axis2 < 0 then axis2 + r else axis2 in

    if norm_axis1 < 0 || norm_axis1 >= r then
      invalid_arg
        (Printf.sprintf "swapaxes: axis1 %d out of bounds for rank %d" axis1 r);
    if norm_axis2 < 0 || norm_axis2 >= r then
      invalid_arg
        (Printf.sprintf "swapaxes: axis2 %d out of bounds for rank %d" axis2 r);

    if norm_axis1 = norm_axis2 then x (* No change *)
    else
      let axes = Array.init r Fun.id in
      let temp = axes.(norm_axis1) in
      axes.(norm_axis1) <- axes.(norm_axis2);
      axes.(norm_axis2) <- temp;
      B.op_permute ctx x axes

  let roll ctx ?axis shift x_orig =
    let x, ax_idx =
      match axis with
      | None ->
          let flat_x = flatten ctx x_orig in
          (* flatten handles rank 0 correctly for its own purpose *)
          (flat_x, 0)
      | Some specified_axis ->
          let r = ndim x_orig in
          let norm_axis =
            if specified_axis < 0 then specified_axis + r else specified_axis
          in
          if norm_axis < 0 || norm_axis >= r then
            invalid_arg
              (Printf.sprintf "roll: axis %d out of bounds for rank %d"
                 specified_axis r);
          (x_orig, norm_axis)
    in
    let current_shape = shape x in
    let r = ndim x in

    if r = 0 then x (* Cannot roll a scalar *)
    else
      let dim_size = current_shape.(ax_idx) in
      if dim_size = 0 then x (* Cannot roll an empty dimension *)
      else
        let s = shift mod dim_size in
        let actual_shift = if s < 0 then s + dim_size else s in

        if actual_shift = 0 then
          if axis = None then reshape ctx x (shape x_orig)
          else x_orig (* Reshape back if flattened and no-op roll *)
        else
          let ranges_part1 =
            Array.mapi
              (fun i cur_dim ->
                if i = ax_idx then (dim_size - actual_shift, cur_dim)
                else (0, cur_dim))
              current_shape
          in
          let ranges_part2 =
            Array.mapi
              (fun i cur_dim ->
                if i = ax_idx then (0, dim_size - actual_shift) else (0, cur_dim))
              current_shape
          in
          let part1 = shrink ctx x ranges_part1 in
          let part2 = shrink ctx x ranges_part2 in
          let rolled_x = B.op_cat ctx [ part1; part2 ] ax_idx in
          if axis = None then reshape ctx rolled_x (shape x_orig) else rolled_x

  let tile ctx reps orig_t =
    let t_shape = shape orig_t in
    let t_ndim = ndim orig_t in
    let reps_len = Array.length reps in

    if reps_len <> t_ndim then
      invalid_arg
        (Printf.sprintf "tile: length of reps %d must match tensor ndim %d"
           reps_len t_ndim);

    Array.iteri
      (fun i r ->
        if r < 0 then
          invalid_arg
            (Printf.sprintf "tile: rep count %d for axis %d cannot be negative"
               r i))
      reps;

    if Array.for_all (( = ) 1) reps then
      B.op_copy ctx orig_t (* optimization: no tiling needed *)
    else if Array.exists (( = ) 0) reps || Array.exists (( = ) 0) t_shape then
      (* If any rep is 0, or original shape has a 0, the tiled dimension becomes
         0 *)
      let tiled_shape = Array.mapi (fun i s_i -> s_i * reps.(i)) t_shape in
      empty ctx (dtype orig_t) tiled_shape
    else
      (* Reshape-expand-reshape trick *)
      (* Src shape for reshape: [1, dim0, 1, dim1, ...] *)
      let src_shape_expanded_list =
        List.concat_map (fun s_i -> [ 1; s_i ]) (Array.to_list t_shape)
      in
      (* Dst shape for expand: [rep0, dim0, rep1, dim1, ...] *)
      let list_concat_map2 f l1 l2 = List.concat (List.map2 f l1 l2) in
      let dst_shape_expanded_list =
        list_concat_map2
          (fun r_i s_i -> [ r_i; s_i ])
          (Array.to_list reps) (Array.to_list t_shape)
      in

      let temp_view_reshaped =
        reshape ctx orig_t (Array.of_list src_shape_expanded_list)
      in
      let temp_view_expanded =
        expand ctx temp_view_reshaped (Array.of_list dst_shape_expanded_list)
      in

      let final_tiled_shape =
        Array.mapi (fun i s_i -> s_i * reps.(i)) t_shape
      in
      let final_view = reshape ctx temp_view_expanded final_tiled_shape in
      B.op_copy ctx final_view

  let repeat ctx ?axis count orig_t =
    if count < 0 then invalid_arg "repeat: count must be non-negative";

    let t, ax_idx_eff =
      match axis with
      | None ->
          let flat_t = flatten ctx orig_t in
          (flat_t, 0)
      | Some specified_axis ->
          let r = ndim orig_t in
          let norm_axis =
            if specified_axis < 0 then specified_axis + r else specified_axis
          in
          if norm_axis < 0 || norm_axis >= r then
            invalid_arg
              (Printf.sprintf "repeat: axis %d out of bounds for rank %d"
                 specified_axis r);
          (orig_t, norm_axis)
    in

    let t_shape = shape t in
    let t_ndim = ndim t in

    if count = 0 then (
      let new_s = Array.copy t_shape in
      if t_ndim > 0 then new_s.(ax_idx_eff) <- 0;
      (* This check handles scalar case *)
      let final_shape_if_flattened = if axis = None then [| 0 |] else new_s in
      empty ctx (dtype orig_t) final_shape_if_flattened)
    else if count = 1 then B.op_copy ctx orig_t
    else if t_ndim = 0 then (* Repeating a scalar *)
      let scalar_val_arr = data orig_t in
      let scalar_val = Bigarray.Array1.get scalar_val_arr 0 in
      let repeated_scalar =
        init ctx (dtype orig_t) [| count |] (fun _ -> scalar_val)
      in
      if axis = None then repeated_scalar (* Already flat *)
      else reshape ctx repeated_scalar (shape orig_t)
    else
      let x_unsqueezed = unsqueeze ctx ~axis:(ax_idx_eff + 1) t in
      let expand_shape_arr = Array.copy (shape x_unsqueezed) in
      expand_shape_arr.(ax_idx_eff + 1) <- count;
      let x_expanded = expand ctx x_unsqueezed expand_shape_arr in

      let final_shape_arr = Array.copy t_shape in
      final_shape_arr.(ax_idx_eff) <- final_shape_arr.(ax_idx_eff) * count;

      let final_view = reshape ctx x_expanded final_shape_arr in
      let result_tensor = B.op_copy ctx final_view in

      if axis = None then result_tensor (* Already flat and repeated *)
      else
        reshape ctx result_tensor
          final_shape_arr (* Ensure original shape structure if not flattened *)

  (* *)

  let eye ctx ?m ?k dtype n =
    let rows = n in
    let cols = match m with Some v -> v | None -> n in
    let k_val = match k with Some v -> v | None -> 0 in

    let final_shape = [| rows; cols |] in

    (* Early exit if k is out of bounds such that no ones can be placed *)
    if rows <= 0 || cols <= 0 || k_val >= cols || k_val <= -rows then
      zeros ctx dtype final_shape
    else
      (* Create row indices: tensor([0, 1, ..., rows-1]) -> [[0], [1], ...,
         [rows-1]] -> broadcasted *)
      let r_arange =
        init ctx Dtype.int32 [| rows |] (fun i -> Int32.of_int i.(0))
      in
      let r_indices_col_vec = reshape ctx r_arange [| rows; 1 |] in
      let r_indices = broadcast_to ctx r_indices_col_vec final_shape in

      (* Create col indices: tensor([0, 1, ..., cols-1]) -> [[0, 1, ...,
         cols-1]] -> broadcasted *)
      let c_arange =
        init ctx Dtype.int32 [| cols |] (fun i -> Int32.of_int i.(0))
      in
      let c_indices_row_vec = reshape ctx c_arange [| 1; cols |] in
      let c_indices = broadcast_to ctx c_indices_row_vec final_shape in

      (* Condition: c_indices - r_indices = k_val *)
      let diff = sub ctx c_indices r_indices in
      let k_tensor_scalar = scalar ctx Dtype.int32 (Int32.of_int k_val) in
      let diag_mask = cmpeq ctx diff k_tensor_scalar in

      let zeros_tensor = zeros ctx dtype final_shape in
      let one_val = Dtype.one dtype in
      let ones_fill = full_like ctx zeros_tensor one_val in
      where ctx diag_mask ones_fill zeros_tensor

  let identity ctx dtype n = eye ctx ~m:n ~k:0 dtype n

  let arange (type a b) ctx (dtype : (a, b) Dtype.t) start stop step =
    if step = 0 then failwith "arange: step cannot be zero";
    let num_elements =
      if step > 0 then
        if start >= stop then 0
        else
          (stop - start + step - 1)
          / step (* Equivalent to ceil((stop-start)/step) for int math *)
      else if
        (* step < 0 *)
        start <= stop
      then 0
      else (start - stop + -step - 1) / -step
      (* Equivalent to ceil((start-stop)/(-step)) *)
    in
    if num_elements <= 0 then empty ctx dtype [| 0 |]
    else
      let f_init idx_arr : a =
        let i = idx_arr.(0) in
        match dtype with
        | Dtype.Float16 ->
            float_of_int start +. (float_of_int i *. float_of_int step)
        | Dtype.Float32 ->
            float_of_int start +. (float_of_int i *. float_of_int step)
        | Dtype.Float64 ->
            float_of_int start +. (float_of_int i *. float_of_int step)
        | Dtype.Int8 -> start + (i * step)
        | Dtype.UInt8 ->
            start + (i * step)
            (* Bigarray will handle unsigned conversion implicitly for 'int'
               OCaml type *)
        | Dtype.Int16 -> start + (i * step)
        | Dtype.UInt16 ->
            start + (i * step)
            (* Bigarray will handle unsigned conversion implicitly for 'int'
               OCaml type *)
        | Dtype.Int32 ->
            Int32.(add (of_int start) (mul (of_int i) (of_int step)))
        | Dtype.Int64 ->
            Int64.(add (of_int start) (mul (of_int i) (of_int step)))
        | Dtype.Int -> start + (i * step)
        | Dtype.NativeInt ->
            Nativeint.(add (of_int start) (mul (of_int i) (of_int step)))
        | Dtype.Complex32 ->
            {
              Complex.re =
                float_of_int start +. (float_of_int i *. float_of_int step);
              im = 0.;
            }
        | Dtype.Complex64 ->
            {
              Complex.re =
                float_of_int start +. (float_of_int i *. float_of_int step);
              im = 0.;
            }
      in
      init ctx dtype [| num_elements |] f_init

  let arange_f ctx dtype start_f stop_f step_f =
    if step_f = 0. then failwith "arange_f: step cannot be zero";
    let num_exact_steps = (stop_f -. start_f) /. step_f in
    let eps_factor = 1e-9 in
    (* Small factor to subtract before floor for robust exclusive bound *)
    let num_elements =
      (* Check if the range is non-positive or extremely small *)
      if
        (step_f > 0. && stop_f <= start_f +. (eps_factor *. Float.abs step_f))
        || (step_f < 0. && stop_f >= start_f +. (eps_factor *. Float.abs step_f))
        || (Float.abs num_exact_steps < eps_factor && num_exact_steps <= 0.)
      then 0
      else
        (* Apply epsilon correction for floor to ensure exclusive upper bound *)
        let corrected_num_steps =
          num_exact_steps -. Float.copy_sign eps_factor num_exact_steps
        in
        int_of_float (Float.floor corrected_num_steps +. 1.)
    in
    let num_elements = Stdlib.max 0 num_elements in
    (* Final guard, though prior logic should handle it *)

    if num_elements <= 0 then empty ctx dtype [| 0 |]
    else
      let f_init idx_arr =
        (* OCaml type 'a is float here *)
        start_f +. (float_of_int idx_arr.(0) *. step_f)
      in
      init ctx dtype [| num_elements |] f_init

  let linspace ctx dtype ?(endpoint = true) start_f stop_f count =
    if count < 0 then invalid_arg "linspace: count must be non-negative";
    if count = 0 then empty ctx dtype [| 0 |]
    else if count = 1 then
      full ctx dtype [| 1 |] (Dtype.float_to_dtype dtype start_f)
    else
      let div_factor = float_of_int (if endpoint then count - 1 else count) in
      let step_f = (stop_f -. start_f) /. div_factor in
      let f_init idx_arr =
        let i_f = float_of_int idx_arr.(0) in
        Dtype.float_to_dtype dtype (start_f +. (i_f *. step_f))
      in
      init ctx dtype [| count |] f_init

  let logspace ctx dtype ?(endpoint = true) ?(base = 10.0) start_exp_f
      stop_exp_f count =
    if count < 0 then invalid_arg "logspace: count must be non-negative";
    if count = 0 then empty ctx dtype [| 0 |]
    else
      (* The exponents should be generated with the same float precision as the
         final tensor type. *)
      let exponents_tensor =
        linspace ctx dtype ~endpoint start_exp_f stop_exp_f count
      in
      if base = Float.exp 1.0 then (* base is e *)
        exp ctx exponents_tensor
      else if base = 2.0 then exp2 ctx exponents_tensor
      else
        (* General case: base ** exponents = exp2(exponents * log2(base)) *)
        let log2_base = Stdlib.log base /. Stdlib.log 2.0 in
        let log2_base_tensor = scalar ctx dtype log2_base in
        (* Ensure log2_base_tensor is broadcastable with exponents_tensor *)
        let broadcasted_log2_base =
          broadcast_to ctx log2_base_tensor (shape exponents_tensor)
        in
        let scaled_exponents = mul ctx exponents_tensor broadcasted_log2_base in
        exp2 ctx scaled_exponents

  let geomspace ctx dtype ?(endpoint = true) start_val_f stop_val_f count =
    if start_val_f <= 0. || stop_val_f <= 0. then
      invalid_arg "geomspace: start and stop values must be positive";
    if count < 0 then invalid_arg "geomspace: count must be non-negative";
    if count = 0 then empty ctx dtype [| 0 |]
    else if count = 1 then
      full ctx dtype [| 1 |] start_val_f (* OCaml type 'a is float here *)
    else
      let log_start_f = Stdlib.log start_val_f in
      let log_stop_f = Stdlib.log stop_val_f in
      (* The log-points should be generated with the same float precision as the
         final tensor type. *)
      let log_points_tensor =
        linspace ctx dtype ~endpoint log_start_f log_stop_f count
      in
      exp ctx log_points_tensor

  (* *)

  let dot ctx x_tensor w_tensor =
    let ndim_x = ndim x_tensor in
    let ndim_w = ndim w_tensor in

    if not (ndim_x > 0 && ndim_w > 0) then
      invalid_arg "dot: both tensors must be at least 1D";

    let shape_x = shape x_tensor in
    let shape_w = shape w_tensor in

    (* Contraction axis for w_tensor: - If w is 1D, its only axis (index 0). -
       If w is >=2D, its second-to-last axis (index ndim_w - 2). This matches
       Python's w.shape[-min(w.ndim,2)] behavior. *)
    let axis_w_contract_idx = if ndim_w = 1 then 0 else ndim_w - 2 in

    (* Contracting dimension sizes must match. *)
    if shape_x.(ndim_x - 1) <> shape_w.(axis_w_contract_idx) then
      invalid_arg
        (Printf.sprintf
           "dot: shape mismatch on contracting dimension. x_shape: %s, \
            w_shape: %s. x_contract_dim_size: %d, w_contract_dim_size: %d"
           (View.pp_int_array shape_x)
           (View.pp_int_array shape_w)
           shape_x.(ndim_x - 1)
           shape_w.(axis_w_contract_idx));

    (* k_ones determines if an extra dimension of size 1 needs to be inserted
       for broadcasting the "matrix" parts. It's 1 if both tensors are >= 2D. *)
    let k_ones = Stdlib.min (Stdlib.min (ndim_x - 1) (ndim_w - 1)) 1 in

    let x_prepared =
      if k_ones = 0 then x_tensor (* No reshape if x or w is 1D *)
      else (* Both x and w are >= 2D *)
        let prefix_x = Array.sub shape_x 0 (ndim_x - 1) in
        let last_dim_x = shape_x.(ndim_x - 1) in
        (* Insert a '1' before the last dimension: e.g., (..., m, k) -> (..., m,
           1, k) *)
        let new_shape_x =
          Array.concat [ prefix_x; [| 1 |]; [| last_dim_x |] ]
        in
        reshape ctx x_tensor new_shape_x
    in

    let w_intermediate_prepared =
      if k_ones = 0 then w_tensor (* No reshape if x or w is 1D *)
      else (* Both x and w are >= 2D. ndim_w >= 2 is implied by k_ones = 1. *)
        let prefix_w = Array.sub shape_w 0 (ndim_w - 2) in
        let suffix_w = Array.sub shape_w (ndim_w - 2) 2 in
        (* Last two dims of original w *)
        (* Insert a '1' before the last two dimensions of original w: e.g.,
           (..., k, n) -> (..., 1, k, n) *)
        let new_shape_w_intermediate =
          Array.concat [ prefix_w; [| 1 |]; suffix_w ]
        in
        reshape ctx w_tensor new_shape_w_intermediate
    in

    let w_prepared =
      let rank_w_intermediate = ndim w_intermediate_prepared in
      if rank_w_intermediate < 2 then w_intermediate_prepared
        (* No transpose if less than 2D (e.g. if original w was 1D) *)
      else
        (* Transpose the (new) last two dimensions. E.g., (..., b, 1, k, n) ->
           (..., b, 1, n, k) Or if original w was 2D (k,n): (1,k,n) ->
           (1,n,k) *)
        let p = Array.init rank_w_intermediate Fun.id in
        let last = rank_w_intermediate - 1 in
        let second_last = rank_w_intermediate - 2 in
        p.(last) <- second_last;
        p.(second_last) <- last;
        transpose ctx ~axes:p w_intermediate_prepared
    in

    (* Element-wise multiplication. Broadcasting handles batch dimensions.
       Example: x_prepared(..., m, 1, k) and w_prepared(..., 1, n, k) broadcasts
       to (..., m, n, k) *)
    let multiplied = mul ctx x_prepared w_prepared in

    (* Sum over the last dimension (the contracting dimension k) *)
    let sum_axis_idx = ndim multiplied - 1 in
    (* The sum function handles accumulation dtype and potential cast back to
       ('a,'b) t *)
    sum ctx ~axes:[| sum_axis_idx |] multiplied

  let matmul ctx a_orig b_orig =
    let ndim_a_orig = ndim a_orig in
    let ndim_b_orig = ndim b_orig in

    if ndim_a_orig = 0 || ndim_b_orig = 0 then
      invalid_arg "matmul: inputs cannot be 0-D (scalars)";

    let a, b =
      if ndim_a_orig = 1 && ndim_b_orig = 1 then
        (* (k), (k) -> a becomes (1,k), b becomes (k,1) *)
        (unsqueeze ctx ~axis:0 a_orig, unsqueeze ctx ~axis:1 b_orig)
      else if ndim_a_orig = 1 then
        (* (k), (...,k,n) -> a becomes (1,k) *)
        (unsqueeze ctx ~axis:0 a_orig, b_orig)
      else if ndim_b_orig = 1 then
        (* (...,m,k), (k) -> b becomes (k,1) *)
        (a_orig, unsqueeze ctx ~axis:1 b_orig)
      else
        (* Both are >= 2D, no promotion needed for matmul semantics *)
        (a_orig, b_orig)
    in

    let result_intermediate = dot ctx a b in

    (* Squeeze the result if original inputs were 1D to match matmul
       semantics *)
    if ndim_a_orig = 1 && ndim_b_orig = 1 then
      (* Original (k) @ (k) -> result (1,1) from dot -> squeeze to scalar () *)
      squeeze ctx result_intermediate
    else if ndim_a_orig = 1 then
      (* Original (k) @ (...,k,n) -> result (...,1,n) from dot -> squeeze first matrix dim *)
      (* The '1' was prepended to a's matrix dimensions.
           If b was (k,n), dot result (1,n). Squeeze axis 0.
           If b was (B,k,n), dot result (B,1,n). Squeeze axis ndim-2.
        *)
      squeeze ctx ~axis:(ndim result_intermediate - 2) result_intermediate
    else if ndim_b_orig = 1 then
      (* Original (...,m,k) @ (k) -> result (...,m,1) from dot -> squeeze last
         matrix dim *)
      squeeze ctx ~axis:(ndim result_intermediate - 1) result_intermediate
    else
      (* Both original inputs were >= 2D, result from dot is already
         (...,m,n) *)
      result_intermediate

  (** Helper: flatten a list of arrays into a single array. *)
  let _flatten_list_of_arrays loa =
    Array.of_list (List.flatten (List.map Array.to_list loa))

  (** Check if any element of the array satisfies the predicate, providing the
      index. *)
  let array_existsi p arr =
    let n = Array.length arr in
    let rec check idx =
      if idx >= n then false
      else if p idx arr.(idx) then true
      else check (idx + 1)
    in
    check 0

  (** Integer ceiling division: (a + b - 1) / b for integers a, b where b > 0.
  *)
  let ceildiv a b =
    if b <= 0 then invalid_arg "_ceildiv: divisor b must be positive"
    else (a + b - 1) / b

  (** Full port of tinygrad's Tensor._pool method. This function extracts
      sliding local blocks (patches) from an input tensor. It is the core
      mechanism for operations like convolution and pooling.

      Args: ctx: The backend context. x_padded_input: The input tensor. It is
      assumed that the caller has already padded this tensor according to the
      desired convolution/pooling mode (e.g., 'Valid', 'Same', 'Full') and
      fill_value. k_s: int array, kernel spatial shape (e.g., [|kh; kw|]). s_s:
      int array, stride for each spatial dimension (e.g., [|sh; sw|]). d_s: int
      array, dilation for each spatial dimension (e.g., [|dh; dw|]).

      Returns: A tensor where spatial dimensions are transformed into patches.
      The shape is (prefix_dims..., output_spatial_dims...,
      kernel_spatial_dims...). For example, an input (N, C, IH, IW) with 2D
      pooling might result in (N, C, OH, OW, KH, KW). *)
  let _pool ctx x_padded_input ~k_s ~s_s ~d_s =
    let x_ndim = ndim x_padded_input in
    let num_spatial = Array.length k_s in

    if num_spatial = 0 then
      (* No spatial dimensions to pool over, return as is *)
      x_padded_input
    else if x_ndim < num_spatial then
      invalid_arg
        "_pool: input tensor ndim less than number of spatial kernel dimensions"
    else if
      not (Array.length s_s = num_spatial && Array.length d_s = num_spatial)
    then
      invalid_arg
        "_pool: stride and dilation arrays must match kernel spatiality"
    else
      let noop_rank = x_ndim - num_spatial in
      let prefix_shape = Array.sub (shape x_padded_input) 0 noop_rank in
      let spatial_shape_in =
        Array.sub (shape x_padded_input) noop_rank num_spatial
      in
      (* These are i_ in tinygrad *)

      (* Calculate output spatial dimensions (o_) based on the (potentially
         padded) input's spatial_shape_in *)
      let o_s =
        Array.init num_spatial (fun j ->
            let eff_kernel_span = (d_s.(j) * (k_s.(j) - 1)) + 1 in
            if spatial_shape_in.(j) < eff_kernel_span then 0
              (* Kernel too large for this input dim *)
            else ((spatial_shape_in.(j) - eff_kernel_span) / s_s.(j)) + 1)
      in

      (* If any output dimension is 0, it means no valid patches can be formed.
         The result should have 0 size along that output dimension, and
         subsequently, the total number of elements from patches will be 0. The
         final shape will be (prefix_shape..., o_s..., k_s...). If any o_s is 0,
         then prod(o_s) * prod(k_s) * prod(prefix_shape) is 0. We can construct
         an empty tensor with the correct final shape. *)
      if Array.exists (( = ) 0) o_s then
        let final_target_shape = Array.concat [ prefix_shape; o_s; k_s ] in
        empty ctx (dtype x_padded_input) final_target_shape
      else
        (* Condition for complex path (from tinygrad) *)
        let is_complex_path =
          array_existsi
            (fun j k_j ->
              k_j > s_s.(j) && s_s.(j) > 0
              (* Avoid div by zero if stride is 0 which is invalid anyway*))
            k_s
          || Array.exists (( <> ) 1) d_s
        in

        let result_tensor =
          if not is_complex_path then (
            (* Simple Path *)
            (* This path is taken when kernel_size <= stride for all spatial dims, and dilation = 1 for all. *)

            (* Stage 1: Pad x_padded_input further to align for striding. Target
               size for spatial dim j becomes o_s.(j) * s_s.(j). *)
            let pad1_config_spatial =
              Array.init num_spatial (fun spatial_idx ->
                  let pad_after =
                    Stdlib.max 0
                      ((o_s.(spatial_idx) * s_s.(spatial_idx))
                      - spatial_shape_in.(spatial_idx))
                  in
                  (0, pad_after))
            in
            let pad1_config_full =
              Array.concat [ Array.make noop_rank (0, 0); pad1_config_spatial ]
            in
            let x =
              pad ctx x_padded_input pad1_config_full
                (Dtype.zero (dtype x_padded_input))
            in

            (* Stage 2: Shrink to the total span that will be covered by
               strides. *)
            let shrink1_config_spatial =
              Array.init num_spatial (fun spatial_idx ->
                  (0, o_s.(spatial_idx) * s_s.(spatial_idx)))
            in
            let shrink1_config_full =
              Array.init (ndim x) (fun dim_idx ->
                  if dim_idx < noop_rank then (0, (shape x).(dim_idx))
                  else shrink1_config_spatial.(dim_idx - noop_rank))
            in
            let x = shrink ctx x shrink1_config_full in

            (* Stage 3: Reshape to separate output patch locations (o_s) and
               stride steps (s_s). *)
            let reshape1_shape_spatial_parts =
              List.flatten
                (Array.to_list
                   (Array.init num_spatial (fun j -> [ o_s.(j); s_s.(j) ])))
            in
            let reshape1_shape_full =
              Array.concat
                [ prefix_shape; Array.of_list reshape1_shape_spatial_parts ]
            in
            let x = reshape ctx x reshape1_shape_full in

            (* Stage 4: Shrink to take only kernel_size elements from each
               stride step. *)
            let shrink2_config_full = Array.copy (shape x) in
            (* Will be (prefix..., o_j, s_j, ...) *)
            for j = 0 to num_spatial - 1 do
              let stride_step_dim_idx = noop_rank + (j * 2) + 1 in
              (* Index of s_j dimension *)
              shrink2_config_full.(stride_step_dim_idx) <- k_s.(j)
              (* Shrink s_j to k_j *)
            done;
            let shrink2_config_pairs =
              Array.map (fun s -> (0, s)) shrink2_config_full
            in
            let x = shrink ctx x shrink2_config_pairs in

            (* Stage 5: Permute to (prefix_dims..., o1, o2, ..., k1, k2, ...) *)
            let perm_axes_list_prefix = List.init noop_rank Fun.id in
            let perm_axes_list_o =
              List.init num_spatial (fun j -> noop_rank + (j * 2))
            in
            let perm_axes_list_k =
              List.init num_spatial (fun j -> noop_rank + (j * 2) + 1)
            in
            let final_perm_axes =
              Array.of_list
                (List.concat
                   [ perm_axes_list_prefix; perm_axes_list_o; perm_axes_list_k ])
            in
            B.op_permute ctx x final_perm_axes)
          else
            (* Complex Path *)
            (* 1. f_s calculation (factor for repeat) *)
            let f_s =
              Array.init num_spatial (fun j ->
                  let oj, sj, ij, dj, kj =
                    (o_s.(j), s_s.(j), spatial_shape_in.(j), d_s.(j), k_s.(j))
                  in
                  let eff_kernel_span = (dj * (kj - 1)) + 1 in
                  if oj * sj > ij - eff_kernel_span + 1 then 2
                  else 1 (* Tinygrad: i - d*(k-1) *))
            in

            (* 2. Repeat input tensor (using `tile` function) *)
            let repeat_factors_spatial =
              Array.init num_spatial (fun j ->
                  let kj, ij, fj, dj =
                    (k_s.(j), spatial_shape_in.(j), f_s.(j), d_s.(j))
                  in
                  (* This (ij * fj + dj) term is directly from tinygrad. Its
                     exact role in achieving im2col is specific to tinygrad's
                     sequence of ops. It seems to ensure enough data is
                     available after repeat for subsequent shrink/reshape to
                     pick dilated elements. If dj is dilation, it's not a simple
                     size addition. If it's a small const for indexing, then its
                     value (often 1 in tinygrad examples if d=1) makes sense.
                     Assuming dj here is the dilation factor for now as per
                     tinygrad var names. *)
                  ceildiv (kj * ((ij * fj) + dj)) ij)
            in
            let repeat_factors_full =
              Array.concat [ Array.make noop_rank 1; repeat_factors_spatial ]
            in
            let x_repeated = tile ctx repeat_factors_full x_padded_input in

            (* 3. Handle dilation: Shrink and Reshape *)
            (* Target size for the dimension after repeat that will be reshaped for dilation handling *)
            let shrink_arg1_spatial_limits =
              Array.init num_spatial (fun j ->
                  let kj, ij, fj, dj =
                    (k_s.(j), spatial_shape_in.(j), f_s.(j), d_s.(j))
                  in
                  kj * ((ij * fj) + dj))
            in
            let shrink_arg1_full_pairs =
              Array.init (ndim x_repeated) (fun dim_idx ->
                  if dim_idx < noop_rank then (0, prefix_shape.(dim_idx))
                  else (0, shrink_arg1_spatial_limits.(dim_idx - noop_rank)))
            in
            let x_shrunk1 = shrink ctx x_repeated shrink_arg1_full_pairs in

            let reshape_arg1_spatial_parts_list =
              List.flatten
                (Array.to_list
                   (Array.init num_spatial (fun j ->
                        let kj, ij, fj, dj =
                          (k_s.(j), spatial_shape_in.(j), f_s.(j), d_s.(j))
                        in
                        [ kj; (ij * fj) + dj ]
                        (* This is the (k, inner_dim) part *))))
            in
            let reshape_arg1_full =
              Array.concat
                [ prefix_shape; Array.of_list reshape_arg1_spatial_parts_list ]
            in
            let x_reshaped1 = reshape ctx x_shrunk1 reshape_arg1_full in

            (* Shape: (prefix..., k_spatial0, inner_dim0, k_spatial1, inner_dim1, ...) *)
            (* where inner_dim_j = spatial_shape_in.(j) * f_s.(j) + d_s.(j) *)

            (* 4. Handle stride: Shrink and Reshape on inner_dims *)
            let current_shape_x_reshaped1 = shape x_reshaped1 in
            let shrink_arg2_config_pairs =
              Array.mapi (fun i s_dim -> (0, s_dim)) current_shape_x_reshaped1
            in
            let reshape_arg2_spatial_parts_list = ref [] in
            for j = 0 to num_spatial - 1 do
              let kj, oj, sj = (k_s.(j), o_s.(j), s_s.(j)) in
              let inner_dim_idx = noop_rank + (j * 2) + 1 in
              (* Index of inner_dim_j *)
              shrink_arg2_config_pairs.(inner_dim_idx) <- (0, oj * sj);
              reshape_arg2_spatial_parts_list :=
                List.append !reshape_arg2_spatial_parts_list [ kj; oj; sj ]
            done;
            let x_shrunk2 = shrink ctx x_reshaped1 shrink_arg2_config_pairs in

            let reshape_arg2_full =
              Array.concat
                [ prefix_shape; Array.of_list !reshape_arg2_spatial_parts_list ]
            in
            let x_reshaped2 = reshape ctx x_shrunk2 reshape_arg2_full in
            (* Shape: (prefix..., k_spatial0, o_spatial0, s_spatial0,
               k_spatial1, o_spatial1, s_spatial1, ...) *)

            (* 5. Final shrink to select one element per stride step
               (effectively making stride=1 for this new dimension) *)
            let current_shape_x_reshaped2 = shape x_reshaped2 in
            let shrink_arg3_config_pairs =
              Array.mapi (fun i s_dim -> (0, s_dim)) current_shape_x_reshaped2
            in
            let reshape_arg3_spatial_parts_list = ref [] in
            for j = 0 to num_spatial - 1 do
              let kj, oj = (k_s.(j), o_s.(j)) in
              let stride_dim_idx = noop_rank + (j * 3) + 2 in
              (* Index of s_spatial_j *)
              shrink_arg3_config_pairs.(stride_dim_idx) <- (0, 1);
              (* Select the first element of the stride group *)
              reshape_arg3_spatial_parts_list :=
                List.append !reshape_arg3_spatial_parts_list [ kj; oj ]
            done;
            let x_shrunk3 = shrink ctx x_reshaped2 shrink_arg3_config_pairs in

            let reshape_arg3_full =
              Array.concat
                [ prefix_shape; Array.of_list !reshape_arg3_spatial_parts_list ]
            in
            let x_reshaped3 = reshape ctx x_shrunk3 reshape_arg3_full in
            (* Shape: (prefix..., k_spatial0, o_spatial0, k_spatial1,
               o_spatial1, ...) *)

            (* 6. Permute to (prefix..., o_spatial..., k_spatial...) *)
            let perm_axes_list_prefix = List.init noop_rank Fun.id in
            let perm_axes_list_o =
              List.init num_spatial (fun j -> noop_rank + (j * 2) + 1)
            in
            (* o_j is at +1 index within each (k_j, o_j) pair *)
            let perm_axes_list_k =
              List.init num_spatial (fun j -> noop_rank + (j * 2))
            in
            (* k_j is at +0 index within each (k_j, o_j) pair *)
            let final_perm_axes =
              Array.of_list
                (List.concat
                   [ perm_axes_list_prefix; perm_axes_list_o; perm_axes_list_k ])
            in
            B.op_permute ctx x_reshaped3 final_perm_axes
        in
        result_tensor

  (** Calculate padding amounts for spatial dimensions based on convolution
      mode. This helper is used by callers of `_pool` (like `correlate2d`).

      Args: input_spatial_shape, shape of the input tensor's spatial dimensions
      (e.g., [|ih; iw|]). k_s, kernel spatial shape (e.g., [|kh; kw|]). s_s,
      stride for each spatial dimension (e.g., [|sh; sw|]). d_s, dilation for
      each spatial dimension (e.g., [|dh; dw|]). mode:
      [ `Full | `Valid | `Same ], the padding mode.

      Returns: An array of (pad_before, pad_after) tuples, one for each spatial
      dimension. *)
  let calculate_padding_for_mode input_spatial_shape ~k_s ~s_s ~d_s
      ~(mode : [ `Full | `Valid | `Same ]) =
    let num_spatial = Array.length input_spatial_shape in
    if
      not
        (Array.length k_s = num_spatial
        && Array.length s_s = num_spatial
        && Array.length d_s = num_spatial)
    then
      invalid_arg
        "calculate_padding_for_mode: shape/kernel/stride/dilation array length \
         mismatch";

    match mode with
    | `Valid -> Array.make num_spatial (0, 0)
    | `Full ->
        Array.init num_spatial (fun i ->
            (* Pad by (kernel_size - 1) * dilation on each side. This allows the
               kernel's first element to align with the first input element, and
               the kernel's last element to align with the last input
               element. *)
            let pad_each_side = d_s.(i) * (k_s.(i) - 1) in
            (pad_each_side, pad_each_side))
    | `Same ->
        Array.init num_spatial (fun i ->
            let is_d, ss_d, ks_d, ds_d =
              (input_spatial_shape.(i), s_s.(i), k_s.(i), d_s.(i))
            in
            (* Output size for 'SAME' padding is typically ceil(input_size /
               stride_size) *)
            let os_d = ceildiv is_d ss_d in
            (* Effective kernel size including dilation *)
            let eff_ks_d = (ds_d * (ks_d - 1)) + 1 in
            (* Calculate total padding needed *)
            let total_pad_d =
              Stdlib.max 0 (((os_d - 1) * ss_d) + eff_ks_d - is_d)
            in
            let pad_before = total_pad_d / 2 in
            let pad_after = total_pad_d - pad_before in
            (pad_before, pad_after))

  (** CorrelateND (cross-correlation) - Generic N-Dimensional version. x: input
      tensor (bs, cin_total, spatial_dims...) w: weight tensor (cout,
      cin_per_group, kernel_spatial_dims...) bias: optional bias tensor (cout)
      stride_s, dilation_s: arrays for spatial dimensions. padding_mode:
      [ `Full | `Valid | `Same ] fillvalue: optional scalar to fill padding.
      Defaults to 0 of x's dtype. num_spatial_dims: The number of spatial
      dimensions (e.g., 1 for 1D, 2 for 2D). *)
  let correlate_nd ctx ?(groups = 1) stride_s_arr
      ?(padding_mode : [ `Full | `Valid | `Same ] = `Valid) dilation_s_arr
      ?fillvalue num_spatial_dims x w ?bias () =
    if ndim w <> num_spatial_dims + 2 then
      invalid_arg
        (Printf.sprintf "correlate_nd: Weight tensor must be %dD"
           (num_spatial_dims + 2));
    if ndim x <> num_spatial_dims + 2 then
      invalid_arg
        (Printf.sprintf "correlate_nd: Input tensor must be %dD"
           (num_spatial_dims + 2));
    if Array.length stride_s_arr <> num_spatial_dims then
      invalid_arg
        "correlate_nd: stride_s_arr length mismatch with num_spatial_dims";
    if Array.length dilation_s_arr <> num_spatial_dims then
      invalid_arg
        "correlate_nd: dilation_s_arr length mismatch with num_spatial_dims";

    let bs = dim 0 x in
    let cin_total = dim 1 x in
    let input_spatial_shape_arr =
      Array.init num_spatial_dims (fun i -> dim (i + 2) x)
    in

    let cout = dim 0 w in
    let cin_per_group = dim 1 w in
    let kernel_spatial_shape_arr =
      Array.init num_spatial_dims (fun i -> dim (i + 2) w)
    in

    if cin_total <> groups * cin_per_group then
      invalid_arg
        (Printf.sprintf
           "Input channels %d not compatible with groups %d and weight \
            cin_per_group %d"
           cin_total groups cin_per_group);
    let rcout = cout / groups in
    if groups * rcout <> cout then
      invalid_arg
        (Printf.sprintf "cout %d not divisible by groups %d" cout groups);

    let actual_fillvalue =
      match fillvalue with Some v -> v | None -> Dtype.zero (dtype x)
    in

    let padding_config_pairs_arr =
      calculate_padding_for_mode input_spatial_shape_arr
        ~k_s:kernel_spatial_shape_arr ~s_s:stride_s_arr ~d_s:dilation_s_arr
        ~mode:padding_mode
    in

    let num_prefix_dims =
      2
      (* bs, cin_total *)
    in
    let op_pad_config_list_prefix =
      Array.to_list (Array.make num_prefix_dims (0, 0))
    in
    let op_pad_config_list_spatial = Array.to_list padding_config_pairs_arr in
    let op_pad_config_arr =
      Array.of_list (op_pad_config_list_prefix @ op_pad_config_list_spatial)
    in

    let x_padded = B.op_pad ctx x op_pad_config_arr actual_fillvalue in

    let pooled_x =
      _pool ctx x_padded ~k_s:kernel_spatial_shape_arr ~s_s:stride_s_arr
        ~d_s:dilation_s_arr
    in
    (* Expected pooled_x shape: (bs, cin_total, output_spatial_dims...,
       kernel_spatial_dims...) *)

    let output_spatial_shape_arr =
      Array.init num_spatial_dims (fun i ->
          (shape pooled_x).(num_prefix_dims + i))
    in

    (* Reshape pooled_x to (bs, groups, cin_per_group, 1 (for rcout broadcast),
       output_spatial..., kernel_spatial...) *)
    let shape_x_pre_expand_list =
      [ bs; groups; cin_per_group; 1 ] (* 1 is for rcout to be expanded *)
      @ Array.to_list output_spatial_shape_arr
      @ Array.to_list kernel_spatial_shape_arr
    in
    let pooled_x_reshaped =
      reshape ctx pooled_x (Array.of_list shape_x_pre_expand_list)
    in

    (* Expand for rcout *)
    let shape_x_expanded_list =
      [ bs; groups; cin_per_group; rcout ]
      @ Array.to_list output_spatial_shape_arr
      @ Array.to_list kernel_spatial_shape_arr
    in
    let pooled_x_expanded =
      expand ctx pooled_x_reshaped (Array.of_list shape_x_expanded_list)
    in

    (* Permute to (bs, groups, rcout, output_spatial..., cin_per_group,
       kernel_spatial...) *)
    let perm_axes_list_prefix_output =
      [ 0; 1; 3 ]
      (* bs, groups, rcout *)
    in
    let perm_axes_list_output_spatial =
      List.init num_spatial_dims (fun i -> 4 + i)
    in
    let perm_axes_list_cin_group = [ 2 ] in
    (* cin_per_group original index *)
    let perm_axes_list_kernel_spatial =
      List.init num_spatial_dims (fun i -> 4 + num_spatial_dims + i)
    in
    let perm_axes =
      Array.of_list
        (List.concat
           [
             perm_axes_list_prefix_output;
             perm_axes_list_output_spatial;
             perm_axes_list_cin_group;
             perm_axes_list_kernel_spatial;
           ])
    in
    let x_ready = B.op_permute ctx pooled_x_expanded perm_axes in

    (* Reshape w to (1 (bs broadcast), groups, rcout, 1s (output_spatial
       broadcast), cin_per_group, kernel_spatial...) *)
    let shape_w_broadcastable_list =
      [ 1; groups; rcout ]
      @ Array.to_list (Array.make num_spatial_dims 1)
      @ [ cin_per_group ]
      @ Array.to_list kernel_spatial_shape_arr
    in
    let w_broadcastable =
      reshape ctx w (Array.of_list shape_w_broadcastable_list)
    in

    let multiplied = mul ctx x_ready w_broadcastable in

    (* Sum over cin_per_group and kernel_spatial_dims. These are the last (1 +
       num_spatial_dims) dimensions after permute. Indices start from
       (ndim_multiplied - (1 + num_spatial_dims)). *)
    let ndim_multiplied = ndim multiplied in
    let num_reduce_dims = 1 + num_spatial_dims in
    let reduce_axes =
      Array.init num_reduce_dims (fun i ->
          ndim_multiplied - num_reduce_dims + i)
    in

    let summed = sum ctx multiplied ~axes:reduce_axes ~keepdims:true in

    let final_shape_list =
      [ bs; cout ] @ Array.to_list output_spatial_shape_arr
    in
    let result_reshaped = reshape ctx summed (Array.of_list final_shape_list) in

    match bias with
    | None -> result_reshaped
    | Some b ->
        let bias_reshape_target_list =
          [ 1; cout ] @ Array.to_list (Array.make num_spatial_dims 1)
        in
        let bias_reshaped =
          reshape ctx b (Array.of_list bias_reshape_target_list)
        in
        add ctx result_reshaped bias_reshaped

  (** Correlate1D (cross-correlation). x: input tensor (bs, cin_total, iw) w:
      weight tensor (cout, cin_per_group, kw) bias: optional bias tensor (cout)
      stride, dilation: integers for the spatial dimension. padding_mode:
      [ `Full | `Valid | `Same ] fillvalue: optional scalar to fill padding.
      Defaults to 0 of x's dtype. *)
  let correlate1d ctx ?groups ?(stride = 1) ?padding_mode ?(dilation = 1)
      ?fillvalue x w ?bias () =
    correlate_nd ctx ?groups [| stride |] ?padding_mode [| dilation |]
      ?fillvalue 1 x w ?bias ()

  (** Correlate2D (cross-correlation). x: input tensor (bs, cin_total, ih, iw)
      w: weight tensor (cout, cin_per_group, kh, kw) bias: optional bias tensor
      (cout) stride, dilation: (int*int) tuples for (h,w) spatial dimensions.
      padding_mode: [ `Full | `Valid | `Same ] fillvalue: optional scalar to
      fill padding. Defaults to 0 of x's dtype. *)
  let correlate2d ctx ?groups ?(stride = (1, 1)) ?padding_mode
      ?(dilation = (1, 1)) ?fillvalue x w ?bias () =
    correlate_nd ctx ?groups
      [| fst stride; snd stride |]
      ?padding_mode
      [| fst dilation; snd dilation |]
      ?fillvalue 2 x w ?bias ()

  (** ConvolveND - Generic N-Dimensional version. This flips the kernel
      (weights) along all its spatial dimensions then calls correlate_nd. *)
  let convolve_nd ctx ?groups stride_s_arr ?padding_mode dilation_s_arr
      ?fillvalue num_spatial_dims ?bias x w =
    let w_ndim = ndim w in
    if w_ndim < num_spatial_dims + 2 then
      invalid_arg
        (Printf.sprintf
           "convolve_nd: Weight tensor needs at least %d dims for spatial \
            flipping"
           (num_spatial_dims + 2));

    (* Flip all spatial dimensions of w: dims from 2 up to (2 + num_spatial_dims
       - 1) *)
    let flip_axes_bools = Array.make w_ndim false in
    for i = 0 to num_spatial_dims - 1 do
      flip_axes_bools.(2 + i) <- true
    done;

    let w_flipped = B.op_flip ctx w flip_axes_bools in
    correlate_nd ctx ?groups stride_s_arr ?padding_mode dilation_s_arr
      ?fillvalue num_spatial_dims x w_flipped ?bias ()

  (** Convolve1D. x: input tensor (bs, cin_total, iw) w: weight tensor (cout,
      cin_per_group, kw) *)
  let convolve1d ctx ?groups ?(stride = 1) ?padding_mode ?(dilation = 1)
      ?fillvalue ?bias x w =
    convolve_nd ctx ?groups [| stride |] ?padding_mode [| dilation |] ?fillvalue
      ?bias 1 x w

  (** Convolve2D. x: input tensor (bs, cin_total, ih, iw) w: weight tensor
      (cout, cin_per_group, kh, kw) *)
  let convolve2d ctx ?groups ?(stride = (1, 1)) ?padding_mode
      ?(dilation = (1, 1)) ?fillvalue x w ?bias () =
    convolve_nd ctx ?groups
      [| fst stride; snd stride |]
      ?padding_mode
      [| fst dilation; snd dilation |]
      ?fillvalue ?bias 2 x w

  (* *)

  let argmax ctx ?axis ?(keepdims = false) t =
    let t_ndim = ndim t in
    let reduction_axis =
      match axis with
      | None -> Array.init t_ndim Fun.id (* Flatten behavior: reduce all axes *)
      | Some ax -> [| resolve_single_axis ~ndim_opt:t_ndim t ax |]
    in
    let t_for_reduce = if axis = None then flatten ctx t else t in
    let current_axis_idx = if axis = None then 0 else reduction_axis.(0) in
    let axis_len = dim current_axis_idx t_for_reduce in

    if axis_len = 0 then (* Edge case: empty dimension *)
      let out_shape =
        if keepdims then
          shape
            t_for_reduce (* Or shape of t if axis was None and t was scalar *)
        else
          Array.of_list
            (List.filteri
               (fun i _ -> i <> current_axis_idx)
               (Array.to_list (shape t_for_reduce)))
      in
      if Array.length out_shape = 0 && numel t_for_reduce > 0 then
        scalar ctx Dtype.int32 0l (* scalar input *)
      else empty ctx Dtype.int32 out_shape
    else
      let max_vals = max ctx ~axes:reduction_axis ~keepdims:true t_for_reduce in
      let is_max_mask = equal ctx t_for_reduce max_vals in

      (* Create reversed arange: [axis_len-1, axis_len-2, ..., 0] *)
      (* Tinygrad uses arange(N, 0, -1) which is [N-1, ..., 0] for N elements *)
      let arange_vals = arange ctx Dtype.int32 (axis_len - 1) (-1) (-1) in

      (* Reshape arange_vals to be broadcastable for multiplication with
         is_max_mask *)
      let arange_shape = Array.make (ndim t_for_reduce) 1 in
      arange_shape.(current_axis_idx) <- axis_len;
      let arange_b = reshape ctx arange_vals arange_shape in
      let arange_bc = broadcast_to ctx arange_b (shape is_max_mask) in

      let masked_arange =
        mul ctx (cast ctx is_max_mask Dtype.int32) arange_bc
      in

      (* Get the max of these values (effectively the first index from the
         end) *)
      let max_indices_from_end =
        max ctx ~axes:reduction_axis ~keepdims:true masked_arange
      in

      (* Convert from "index from end" to "index from start" *)
      let axis_len_tensor =
        scalar ctx Dtype.int32 (Int32.of_int (axis_len - 1))
      in
      let axis_len_bc =
        broadcast_to ctx axis_len_tensor (shape max_indices_from_end)
      in
      let final_indices = sub ctx axis_len_bc max_indices_from_end in

      if keepdims then final_indices
      else
        let final_shape =
          if axis = None then [||] (* scalar output if all axes reduced *)
          else
            Array.of_list
              (List.filteri
                 (fun i _ -> i <> current_axis_idx)
                 (Array.to_list (shape t_for_reduce)))
        in
        reshape ctx final_indices final_shape

  let argmin (type a b) ctx ?axis ?(keepdims = false) (t : (a, b) t) :
      (int32, Dtype.int32_elt) t =
    (* For integers, -t might overflow. For floats, -t is fine. For unsigned,
       this is more complex. Tinygrad uses `_inverse` which is `-self` for
       floats, `~self` for ints, `logical_not` for bools. Let's assume a numeric
       type for simplicity here or require float. If dtype is integer, a robust
       way is: max_val - t *)
    let t_dtype = dtype t in
    let t_inverted =
      if Dtype.is_float t_dtype then neg ctx t
      else if Dtype.is_int t_dtype && not (Dtype.is_uint t_dtype) then neg ctx t
      else if Dtype.is_uint t_dtype then
        (* (max_val_for_dtype - t). This might need a cast if max_val is too
           large for 'a *)
        let max_val_specific : (a, b) t =
          match t_dtype with
          (* This is a bit of a hack; Dtype should provide this if general *)
          | Dtype.UInt8 -> scalar ctx Dtype.uint8 255
          | Dtype.UInt16 -> scalar ctx Dtype.uint16 65535
          (* Add other uint types as needed, or make Dtype.max_val more
             accessible *)
          | _ -> failwith "argmin: unsupported uint dtype for inversion"
        in
        let max_val_b = broadcast_to ctx max_val_specific (shape t) in
        sub ctx max_val_b t
      else (* Bool, etc. *)
        logical_not ctx t (* This will change argmin for bools compared to -t *)
    in
    argmax ctx ?axis ~keepdims t_inverted

  (* *)

  let pp_data (type a b) _ctx fmt (t : (a, b) t) =
    let open Format in
    let view = B.view t in
    let buffer = B.data t in
    let dtype = dtype t in
    let shape = view.shape in
    let ndim = Array.length shape in
    let sz = View.size view in

    let pp_element fmt (elt : a) =
      match dtype with
      | Float16 -> fprintf fmt "%g" elt
      | Float32 -> fprintf fmt "%g" elt
      | Float64 -> fprintf fmt "%g" elt
      | Int8 -> fprintf fmt "%d" elt
      | Int16 -> fprintf fmt "%d" elt
      | Int32 -> fprintf fmt "%ld" elt
      | Int64 -> fprintf fmt "%Ld" elt
      | UInt8 -> fprintf fmt "%d" elt
      | UInt16 -> fprintf fmt "%d" elt
      | Int -> fprintf fmt "%d" elt
      | NativeInt -> fprintf fmt "%nd" elt
      | Complex32 -> fprintf fmt "(%g+%gi)" elt.re elt.im
      | Complex64 -> fprintf fmt "(%g+%gi)" elt.re elt.im
    in

    if sz = 0 && ndim > 0 then fprintf fmt "[]"
    else if ndim = 0 then
      if sz > 0 then
        let value = Bigarray.Array1.unsafe_get buffer view.offset in
        pp_element fmt value
      else fprintf fmt "<empty scalar>"
    else
      let rec pp_slice fmt current_indices =
        let current_ndim = List.length current_indices in
        if current_ndim = ndim then
          let md_index = Array.of_list current_indices in
          let linear_offset =
            View.index_to_offset md_index view.strides + view.offset
          in
          if linear_offset < 0 || linear_offset >= Bigarray.Array1.dim buffer
          then
            fprintf fmt "<OOB:%d/%d>" linear_offset (Bigarray.Array1.dim buffer)
          else
            let value = Bigarray.Array1.unsafe_get buffer linear_offset in
            pp_element fmt value
        else
          let axis = current_ndim in
          let dim_size = shape.(axis) in
          fprintf fmt "[";
          if dim_size > 0 then (
            if axis < ndim - 1 then pp_open_vbox fmt 0 else pp_open_hbox fmt ();
            for i = 0 to dim_size - 1 do
              if i > 0 then (
                fprintf fmt ",";
                if axis = ndim - 1 then fprintf fmt " " else pp_print_cut fmt ());
              pp_slice fmt (current_indices @ [ i ])
            done;
            pp_close_box fmt ());
          fprintf fmt "]"
      in
      if sz > 0 then pp_slice fmt [] else fprintf fmt "[]"

  let data_to_string ctx t =
    let buf = Stdlib.Buffer.create 1024 in
    let fmt = Format.formatter_of_buffer buf in
    pp_data ctx fmt t;
    Format.pp_print_flush fmt ();
    Stdlib.Buffer.contents buf

  let print_data ctx t =
    pp_data ctx Format.std_formatter t;
    Format.pp_print_newline Format.std_formatter ();
    Format.pp_print_flush Format.std_formatter ()

  let pp_dtype context fmt dtype =
    Format.fprintf fmt "%s" (Dtype.to_string dtype)

  let shape_to_string _ctx shape =
    let shape_str =
      Array.map string_of_int shape |> Array.to_list |> String.concat "x"
    in
    Printf.sprintf "[%s]" shape_str

  let pp_shape context fmt shape =
    Format.fprintf fmt "%s" (shape_to_string context shape)

  let pp context fmt t =
    let open Format in
    let view = B.view t in

    fprintf fmt "@[<v 0>";
    fprintf fmt "Ndarray Info:@,";
    fprintf fmt "  Shape: %a@," (pp_shape context) view.shape;
    fprintf fmt "  Dtype: %a@," (pp_dtype context) (dtype t);
    fprintf fmt "  Strides: [%s]@,"
      (String.concat "; "
         (Array.to_list (Array.map string_of_int view.strides)));
    fprintf fmt "  Offset: %d@," view.offset;
    fprintf fmt "  Size: %d@," (View.size view)

  let print ctx t =
    pp ctx Format.std_formatter t;
    Format.pp_print_newline Format.std_formatter ();
    Format.pp_print_flush Format.std_formatter ()

  let to_string ctx t =
    let buf = Stdlib.Buffer.create 1024 in
    let fmt = Format.formatter_of_buffer buf in
    pp ctx fmt t;
    Format.pp_print_flush fmt ();
    Stdlib.Buffer.contents buf
end
