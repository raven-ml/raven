(* High-level tensor operations built on backend [B]. *)

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

  (** Integer ceiling division: (a + b - 1) / b for integers a, b where b > 0.
  *)
  let ceildiv a b =
    if b <= 0 then invalid_arg "ceildiv: divisor b must be positive"
    else (a + b - 1) / b

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

  let astype ctx dt x = cast ctx x dt

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

  let create ctx dtype shape arr =
    let n = Array.fold_left ( * ) 1 shape in
    if Array.length arr <> n then
      invalid_arg
        (Printf.sprintf "create: array size (%d) doesn't match shape (%d)"
           (Array.length arr) n);

    (* Create bigarray buffer with proper dtype *)
    let kind = Dtype.kind_of_dtype dtype in
    let bigarray = Bigarray.Array1.create kind Bigarray.c_layout n in

    (* Copy data from OCaml array to bigarray *)
    for i = 0 to n - 1 do
      Bigarray.Array1.unsafe_set bigarray i arr.(i)
    done;

    (* Create flat tensor and reshape if needed *)
    let tensor_1d = B.op_const_array ctx bigarray in
    if Array.length shape = 1 && shape.(0) = n then tensor_1d
    else B.op_reshape ctx tensor_1d shape

  let init ctx dtype shape f =
    let size = Array.fold_left ( * ) 1 shape in

    (* Helper to convert linear index to multi-dimensional indices *)
    let unravel_index idx shape =
      let ndim = Array.length shape in
      let indices = Array.make ndim 0 in
      let remaining = ref idx in
      for i = ndim - 1 downto 0 do
        let stride =
          Array.fold_left ( * ) 1 (Array.sub shape (i + 1) (ndim - i - 1))
        in
        indices.(i) <- !remaining / stride;
        remaining := !remaining mod stride
      done;
      indices
    in

    (* Create OCaml array with values from f *)
    let arr = Array.init size (fun i -> f (unravel_index i shape)) in

    (* Use create to handle the conversion *)
    create ctx dtype shape arr

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

  let fill ctx value t =
    let value_tensor = scalar_like ctx t value in
    let value_broadcasted = broadcast_to ctx value_tensor (shape t) in
    B.op_assign ctx t value_broadcasted;
    t

  (* *)

  let to_bigarray _ctx t =
    let array1 = data t in
    let ba = Bigarray.reshape (Bigarray.genarray_of_array1 array1) (shape t) in
    ba

  let of_bigarray ctx ba =
    let size = Array.fold_left ( * ) 1 (Bigarray.Genarray.dims ba) in
    let arr = Bigarray.reshape_1 ba size in
    let shape = Bigarray.Genarray.dims ba in
    let flat_tensor = B.op_const_array ctx arr in
    reshape ctx flat_tensor shape

  let to_array _ctx t =
    let ba = data t in
    let n = numel t in
    Array.init n (fun i -> Bigarray.Array1.get ba i)

  (* ────────── element-wise binary ops ────────── *)

  let binop ctx op a b =
    let a', b' = broadcasted ctx a b in
    op ctx a' b'

  let scalar_op ctx op tensor scalar_val =
    let scalar_tensor = scalar_like ctx tensor scalar_val in
    op ctx tensor scalar_tensor

  let inplace_op ctx op target value =
    let value_broadcasted = broadcast_to ctx value (shape target) in
    let result = op ctx target value_broadcasted in
    B.op_assign ctx target result;
    target

  let add ctx a b = binop ctx B.op_add a b
  let add_s ctx tensor scalar = scalar_op ctx add tensor scalar
  let iadd ctx target value = inplace_op ctx B.op_add target value

  let radd_s ctx scalar_a_val tensor_b =
    let scalar_a_tensor = scalar_like ctx tensor_b scalar_a_val in
    add ctx scalar_a_tensor tensor_b

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

  let invert ctx t = bitwise_not ctx t

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

  let min ctx ?axes ?(keepdims = false) x =
    let rank = Array.length (shape x) in
    let axes_to_reduce =
      match axes with
      | None -> Array.init rank Fun.id
      | Some ax_list ->
          Array.map (fun ax -> if ax < 0 then ax + rank else ax) ax_list
    in
    neg ctx (B.op_reduce_max ctx (neg ctx x) ~axes:axes_to_reduce ~keepdims)

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

  (* *)

  (* Check if all elements are true (non-zero) *)
  let all ctx ?axes ?(keepdims = false) x =
    let dt = dtype x in

    (* For boolean/uint8 tensors, we check if min == 1 For other numeric types,
       we check if min != 0 *)
    let min_val = min ctx ?axes ~keepdims x in

    if Dtype.eq dt Dtype.uint8 then
      (* For boolean tensors, all elements are true if min is 1 *)
      let one_val = Dtype.one Dtype.uint8 in
      let one_tensor = full_like ctx min_val one_val in
      cmpeq ctx min_val one_tensor
    else
      (* For numeric tensors, all elements are true if min is non-zero *)
      let zero_val = Dtype.zero dt in
      let zero_tensor = full_like ctx min_val zero_val in
      cmpne ctx min_val zero_tensor

  (* Check if any element is true (non-zero) *)
  let any ctx ?axes ?(keepdims = false) x =
    let dt = dtype x in

    (* For any type, we check if max != 0 *)
    let max_val = max ctx ?axes ~keepdims x in
    let zero_val = Dtype.zero dt in
    let zero_tensor = full_like ctx max_val zero_val in
    cmpne ctx max_val zero_tensor

  (* Check if two arrays are element-wise equal *)
  let array_equal ctx x y =
    (* First, check if we can broadcast the shapes *)
    let can_broadcast =
      try
        let _ = View.broadcast_shapes (shape x) (shape y) in
        true
      with _ -> false
    in

    if not can_broadcast then
      (* If shapes can't be broadcast, arrays are not equal Return a scalar
         False (0) *)
      zeros ctx Dtype.uint8 [||]
    else
      (* Check element-wise equality and then check if all are true *)
      let eq_result = equal ctx x y in
      all ctx eq_result (* Reduce over all axes to get scalar result *)

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

  let unflatten ctx t dim sizes =
    let dim = resolve_single_axis t dim in
    let current_shape = shape t in
    let dim_size = current_shape.(dim) in

    (* Handle -1 in sizes (infer dimension) *)
    let sizes = Array.copy sizes in
    let neg_one_count =
      Array.fold_left (fun acc s -> if s = -1 then acc + 1 else acc) 0 sizes
    in

    if neg_one_count > 1 then
      invalid_arg "unflatten: can only specify one unknown dimension";

    if neg_one_count = 1 then (
      let known_product =
        Array.fold_left (fun acc s -> if s = -1 then acc else acc * s) 1 sizes
      in
      if known_product = 0 || dim_size mod known_product <> 0 then
        invalid_arg
          (Printf.sprintf
             "unflatten: cannot infer dimension size for %d with known sizes \
              product %d"
             dim_size known_product);
      let inferred_size = dim_size / known_product in
      Array.iteri (fun i s -> if s = -1 then sizes.(i) <- inferred_size) sizes);

    (* Verify that product of sizes equals original dimension *)
    let sizes_product = Array.fold_left ( * ) 1 sizes in
    if sizes_product <> dim_size then
      invalid_arg
        (Printf.sprintf
           "unflatten: product of sizes %d does not match dimension size %d"
           sizes_product dim_size);

    (* Build new shape *)
    let new_shape =
      Array.concat
        [
          Array.sub current_shape 0 dim;
          sizes;
          Array.sub current_shape (dim + 1)
            (Array.length current_shape - dim - 1);
        ]
    in

    reshape ctx t new_shape

  let ravel ctx t = flatten ctx t

  module IntSet = Set.Make (Int)

  (* drop axes of size 1; [axes] restricts which axes to squeeze *)
  let squeeze ctx ?axes x =
    let sh = shape x in
    let r = Array.length sh in

    match axes with
    | None ->
        (* Squeeze all dimensions of size 1 *)
        let new_shape_list = List.filter (( <> ) 1) (Array.to_list sh) in
        let new_shape = Array.of_list new_shape_list in
        if Array.length new_shape = 0 && Array.length sh > 0 then
          reshape ctx x [||] (* Result is scalar *)
        else if Array.length new_shape = 0 && Array.length sh = 0 then x
          (* scalar to scalar *)
        else reshape ctx x new_shape
    | Some axes_arr ->
        if r = 0 then x (* Cannot squeeze a scalar *)
        else
          (* Normalize negative indices and validate *)
          let normalized_axes =
            Array.map (fun ax -> if ax < 0 then ax + r else ax) axes_arr
          in

          (* Check for duplicates *)
          let seen = Array.make r false in
          Array.iter
            (fun ax ->
              if ax < 0 || ax >= r then
                invalid_arg
                  (Printf.sprintf "squeeze: axis %d out of bounds for rank %d"
                     ax r);
              if seen.(ax) then
                invalid_arg (Printf.sprintf "squeeze: duplicate axis %d" ax);
              seen.(ax) <- true)
            normalized_axes;

          (* Check that all specified axes have size 1 *)
          Array.iter
            (fun ax ->
              if sh.(ax) <> 1 then
                invalid_arg
                  (Printf.sprintf
                     "squeeze: cannot squeeze axis %d of size %d (!= 1)" ax
                     sh.(ax)))
            normalized_axes;

          (* Build new shape by filtering out squeezed dimensions *)
          let axes_set =
            Array.fold_left
              (fun set ax -> IntSet.add ax set)
              IntSet.empty normalized_axes
          in

          let new_shape_list =
            List.filteri
              (fun i _ -> not (IntSet.mem i axes_set))
              (Array.to_list sh)
          in

          let new_shape = Array.of_list new_shape_list in

          if Array.length new_shape = 0 && Array.length sh > 0 then
            reshape ctx x [||] (* Result is scalar *)
          else if Array.length new_shape = 0 && Array.length sh = 0 then x
            (* scalar to scalar *)
          else reshape ctx x new_shape

  (* insert size-1 dimensions at specified axes *)
  let unsqueeze ctx ?axes x =
    let sh = shape x in
    let r = Array.length sh in

    let axes_arr =
      match axes with
      | None -> invalid_arg "unsqueeze: axes must be specified"
      | Some arr -> arr
    in

    if Array.length axes_arr = 0 then x (* No dimensions to add *)
    else
      let output_rank = r + Array.length axes_arr in

      (* Normalize negative indices (relative to output shape) *)
      let normalized_axes =
        Array.map (fun ax -> if ax < 0 then ax + output_rank else ax) axes_arr
      in

      (* Validate axes *)
      let seen = Array.make output_rank false in
      Array.iter
        (fun ax ->
          if ax < 0 || ax >= output_rank then
            invalid_arg
              (Printf.sprintf
                 "unsqueeze: axis %d out of bounds for output rank %d" ax
                 output_rank);
          if seen.(ax) then
            invalid_arg (Printf.sprintf "unsqueeze: duplicate axis %d" ax);
          seen.(ax) <- true)
        normalized_axes;

      (* Sort axes to process in order *)
      let sorted_axes = Array.copy normalized_axes in
      Array.sort compare sorted_axes;

      (* Build mapping from output position to input position *)
      let axes_set =
        Array.fold_left
          (fun set ax -> IntSet.add ax set)
          IntSet.empty normalized_axes
      in

      (* Create new shape *)
      let new_shape_list = ref [] in
      let input_idx = ref 0 in

      for output_idx = 0 to output_rank - 1 do
        if IntSet.mem output_idx axes_set then
          new_shape_list :=
            1 :: !new_shape_list (* Insert dimension of size 1 *)
        else if !input_idx < r then (
          new_shape_list := sh.(!input_idx) :: !new_shape_list;
          incr input_idx)
      done;

      let new_shape = Array.of_list (List.rev !new_shape_list) in
      reshape ctx x new_shape

  (* For backward compatibility, you might want to add these helper
     functions: *)

  (* squeeze a single axis *)
  let squeeze_axis ctx axis x = squeeze ctx ~axes:[| axis |] x

  (* unsqueeze a single axis *)
  let unsqueeze_axis ctx axis x = unsqueeze ctx ~axes:[| axis |] x

  (* expand_dims is an alias for unsqueeze *)
  let expand_dims ctx t axes = unsqueeze ctx ~axes t

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
      let x_unsqueezed = unsqueeze ctx ~axes:[| ax_idx_eff + 1 |] t in
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

  let concatenate ctx ?axis ts =
    match ts with
    | [] -> invalid_arg "concatenate: need at least one array"
    | [ t ] -> copy ctx t
    | _ ->
        let axis =
          match axis with
          | None ->
              (* Flatten all arrays first *)
              let flattened = List.map (flatten ctx) ts in
              B.op_cat ctx flattened 0
          | Some a ->
              let first_ndim = ndim (List.hd ts) in
              let axis =
                resolve_single_axis ~ndim_opt:first_ndim (List.hd ts) a
              in

              (* Check all arrays have same ndim *)
              if not (List.for_all (fun t -> ndim t = first_ndim) ts) then
                invalid_arg
                  "concatenate: all arrays must have same number of dimensions";

              (* Check shapes match except on concatenation axis *)
              let first_shape = shape (List.hd ts) in
              List.iter
                (fun t ->
                  let t_shape = shape t in
                  Array.iteri
                    (fun i s ->
                      if i <> axis && s <> first_shape.(i) then
                        invalid_arg
                          "concatenate: all arrays must have same shape except \
                           along concatenation axis")
                    t_shape)
                (List.tl ts);

              B.op_cat ctx ts axis
        in
        axis

  let stack ctx ?axis ts =
    match ts with
    | [] -> invalid_arg "stack: need at least one array"
    | _ ->
        let first_shape = shape (List.hd ts) in
        let first_ndim = Array.length first_shape in

        (* Check all arrays have same shape *)
        List.iter
          (fun t ->
            if shape t <> first_shape then
              invalid_arg "stack: all arrays must have same shape")
          (List.tl ts);

        (* Determine stacking axis *)
        let axis =
          match axis with
          | None -> 0
          | Some a ->
              let a = if a < 0 then a + first_ndim + 1 else a in
              if a < 0 || a > first_ndim then
                invalid_arg
                  (Printf.sprintf "stack: axis %d out of bounds for rank %d" a
                     first_ndim);
              a
        in

        (* Add new dimension to each array *)
        let expanded =
          List.map (fun t -> unsqueeze ctx ~axes:[| axis |] t) ts
        in

        (* Concatenate along the new axis *)
        B.op_cat ctx expanded axis

  let vstack ctx ts =
    match ts with
    | [] -> invalid_arg "vstack: need at least one array"
    | _ ->
        (* Make all arrays at least 2D *)
        let arrays_2d =
          List.map
            (fun t ->
              let nd = ndim t in
              if nd = 0 then reshape ctx t [| 1; 1 |]
              else if nd = 1 then reshape ctx t [| 1; numel t |]
              else t)
            ts
        in

        (* Concatenate along first axis *)
        concatenate ctx ~axis:0 arrays_2d

  let hstack ctx ts =
    match ts with
    | [] -> invalid_arg "hstack: need at least one array"
    | _ ->
        (* Handle different dimensions *)
        let all_1d = List.for_all (fun t -> ndim t <= 1) ts in

        if all_1d then
          (* For 1D arrays, concatenate along axis 0 *)
          let arrays_1d =
            List.map
              (fun t -> if ndim t = 0 then reshape ctx t [| 1 |] else t)
              ts
          in
          concatenate ctx ~axis:0 arrays_1d
        else
          (* Make all arrays at least 2D *)
          let arrays_2d =
            List.map
              (fun t ->
                let nd = ndim t in
                if nd = 0 then reshape ctx t [| 1; 1 |]
                else if nd = 1 then reshape ctx t [| numel t; 1 |]
                else t)
              ts
          in

          (* Concatenate along second axis *)
          concatenate ctx ~axis:1 arrays_2d

  let dstack ctx ts =
    match ts with
    | [] -> invalid_arg "dstack: need at least one array"
    | _ ->
        (* Make all arrays at least 3D *)
        let arrays_3d =
          List.map
            (fun t ->
              let s = shape t in
              let nd = Array.length s in
              if nd = 0 then reshape ctx t [| 1; 1; 1 |]
              else if nd = 1 then reshape ctx t [| s.(0); 1; 1 |]
              else if nd = 2 then reshape ctx t [| s.(0); s.(1); 1 |]
              else t)
            ts
        in

        (* Concatenate along third axis *)
        concatenate ctx ~axis:2 arrays_3d

  let broadcast_arrays ctx ts =
    match ts with
    | [] -> []
    | [ t ] -> [ t ]
    | _ ->
        (* Find broadcast shape *)
        let broadcast_shape =
          List.fold_left
            (fun acc_shape t -> View.broadcast_shapes acc_shape (shape t))
            (shape (List.hd ts))
            (List.tl ts)
        in

        (* Broadcast all arrays to common shape *)
        List.map (fun t -> broadcast_to ctx t broadcast_shape) ts

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
    if step = 0 then invalid_arg "arange: step cannot be zero";
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
    if step_f = 0. then invalid_arg "arange_f: step cannot be zero";
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

  (* Index type definition *)
  type index =
    | I of int (* single index *)
    | L of int list (* list of indices *)
    | R of int list (* index range *)

  (* Helper to normalize negative indices *)
  let normalize_index dim_size idx = if idx < 0 then dim_size + idx else idx

  (* Expand range specification according to Owl's conventions *)
  let expand_range_spec dim_size = function
    | [] -> (0, dim_size - 1, 1)
    | [ start ] ->
        let start' = normalize_index dim_size start in
        (start', start', 1)
    | [ start; stop ] ->
        let start' = normalize_index dim_size start in
        let stop' = normalize_index dim_size stop in
        if start' <= stop' then (start', stop', 1) else (start', stop', -1)
    | [ start; stop; step ] ->
        if step = 0 then invalid_arg "step cannot be zero"
        else
          let start' = normalize_index dim_size start in
          let stop' = normalize_index dim_size stop in
          (start', stop', step)
    | _ -> invalid_arg "range can have at most 3 elements"

  (* Convert index specification to list of indices *)
  let indices_of_spec dim_size = function
    | I idx ->
        let idx' = normalize_index dim_size idx in
        if idx' < 0 || idx' >= dim_size then
          invalid_arg (Printf.sprintf "index %d out of bounds" idx)
        else [ idx' ]
    | L indices ->
        List.map
          (fun idx ->
            let idx' = normalize_index dim_size idx in
            if idx' < 0 || idx' >= dim_size then
              invalid_arg (Printf.sprintf "index %d out of bounds" idx)
            else idx')
          indices
    | R range ->
        let start, stop, step = expand_range_spec dim_size range in
        let rec collect acc i =
          if step > 0 && i > stop then List.rev acc
          else if step < 0 && i < stop then List.rev acc
          else if i >= 0 && i < dim_size then collect (i :: acc) (i + step)
          else collect acc (i + step)
        in
        collect [] start

  (* Efficient get_slice that minimizes tensor operations *)
  let slice ctx slice_def x =
    let x_shape = shape x in
    let ndim = Array.length x_shape in

    (* Pad slice definition *)
    let full_slice =
      let n = List.length slice_def in
      if n > ndim then invalid_arg "too many indices"
      else slice_def @ List.init (ndim - n) (fun _ -> R [])
    in

    (* Analyze slice pattern *)
    let analyze_pattern slice =
      match slice with
      | [] -> `Empty
      | I _ :: rest when List.for_all (function I _ -> true | _ -> false) rest
        ->
          `AllSingles
      | _ ->
          (* Check if all are contiguous ranges *)
          let is_contiguous =
            List.for_all
              (function
                | R [] | R [ _ ] | R [ _; _ ] -> true
                | R [ s; e; 1 ] -> s <= e
                | R [ s; e; -1 ] -> s >= e
                | _ -> false)
              slice
          in
          if is_contiguous then `ContiguousRanges else `Mixed
    in

    match analyze_pattern full_slice with
    | `Empty -> x
    | `AllSingles ->
        (* Direct element access *)
        let indices =
          List.map
            (fun spec ->
              match spec with
              | I idx -> normalize_index x_shape.(0) idx
              | _ -> assert false)
            full_slice
        in
        let shrink_config =
          Array.of_list (List.mapi (fun _i idx -> (idx, idx + 1)) indices)
        in
        reshape ctx (shrink ctx x shrink_config) [||]
    | `ContiguousRanges ->
        (* Use shrink/flip operations only *)
        let rec apply_slices tensor dim = function
          | [] -> tensor
          | spec :: rest ->
              let dim_size = (shape tensor).(dim) in
              let tensor' =
                match spec with
                | R [] -> tensor (* Take all *)
                | R [ idx ] ->
                    let idx' = normalize_index dim_size idx in
                    let config =
                      Array.init (ndim - dim) (fun i ->
                          if i = 0 then (idx', idx' + 1)
                          else (0, (shape tensor).(dim + i)))
                    in
                    squeeze ctx ~axes:[| dim |] (shrink ctx tensor config)
                | R range ->
                    let start, stop, step = expand_range_spec dim_size range in
                    let s, e =
                      if step > 0 then (start, stop + 1) else (stop, start + 1)
                    in
                    let config =
                      Array.init (ndim - dim) (fun i ->
                          if i = 0 then (s, e) else (0, (shape tensor).(dim + i)))
                    in
                    let sliced = shrink ctx tensor config in
                    if step < 0 then flip ctx ~axes:[| dim |] sliced else sliced
                | _ -> assert false
              in
              apply_slices tensor' (dim + 1) rest
        in
        apply_slices x 0 full_slice
    | `Mixed ->
        (* Batch gather operations where possible *)
        let rec batch_process tensor processed_dims = function
          | [] -> tensor
          | specs ->
              (* Group consecutive gather operations *)
              let rec group_gathers acc current = function
                | [] -> List.rev (current :: acc)
                | ((I _ | L _) as spec) :: rest ->
                    group_gathers acc (spec :: current) rest
                | spec :: rest ->
                    group_gathers ((spec :: current) :: acc) [] rest
              in

              let groups = group_gathers [] [] specs in

              (* Process each group *)
              List.fold_left
                (fun tensor group ->
                  match group with
                  | [] -> tensor
                  | R spec :: rest ->
                      (* Single range - use shrink *)
                      let dim_size = (shape tensor).(0) in
                      let indices = indices_of_spec dim_size (R spec) in
                      let tensor' =
                        if List.length indices = dim_size then tensor
                        else if List.length indices = 1 then
                          squeeze ctx ~axes:[| 0 |]
                            (shrink ctx tensor
                               [| (List.hd indices, List.hd indices + 1) |])
                        else
                          (* Create index tensor and gather *)
                          let idx_tensor =
                            init ctx Dtype.int32
                              [| List.length indices |]
                              (fun arr ->
                                Int32.of_int (List.nth indices arr.(0)))
                          in
                          B.op_gather ctx tensor idx_tensor 0
                      in
                      batch_process tensor' (processed_dims + 1) rest
                  | group ->
                      (* Multiple gather operations - fall back to sequential processing *)
                      (* Note: A more sophisticated batched gather operation could be implemented
                         in the future to handle multiple dimensions at once *)
                      let indices_lists =
                        List.mapi
                          (fun i spec ->
                            let dim_idx = processed_dims + i in
                            indices_of_spec (shape tensor).(dim_idx) spec)
                          group
                      in

                      (* Process each gather operation sequentially *)
                      List.fold_left2
                        (fun t _spec indices ->
                          if List.length indices = 1 then
                            squeeze ctx ~axes:[| 0 |]
                              (shrink ctx t
                                 [| (List.hd indices, List.hd indices + 1) |])
                          else
                            let idx_tensor =
                              init ctx Dtype.int32
                                [| List.length indices |]
                                (fun arr ->
                                  Int32.of_int (List.nth indices arr.(0)))
                            in
                            B.op_gather ctx t idx_tensor 0)
                        tensor group indices_lists)
                tensor groups
        in
        batch_process x 0 full_slice

  (* Efficient set_slice using scatter operations *)
  let set_slice ctx slice_def x y =
    let x_shape = shape x in
    let y_shape = shape y in
    let ndim = Array.length x_shape in

    (* Pad slice definition *)
    let full_slice =
      let n = List.length slice_def in
      if n > ndim then invalid_arg "too many indices"
      else slice_def @ List.init (ndim - n) (fun _ -> R [])
    in

    (* Get indices for each dimension *)
    let indices_per_dim =
      List.mapi (fun i spec -> indices_of_spec x_shape.(i) spec) full_slice
    in

    (* Verify shape *)
    let expected_shape = Array.of_list (List.map List.length indices_per_dim) in
    if expected_shape <> y_shape then invalid_arg "shape mismatch";

    (* Check if we can use optimized paths *)
    let all_contiguous =
      List.for_all2
        (fun spec indices ->
          match spec with
          | R [ s; e ] | R [ s; e; 1 ] ->
              let s', e' =
                (normalize_index x_shape.(0) s, normalize_index x_shape.(0) e)
              in
              List.length indices = Stdlib.abs (e' - s') + 1
          | R [] -> List.length indices = x_shape.(0)
          | _ -> false)
        full_slice indices_per_dim
    in

    if all_contiguous then
      (* Can use direct blit with proper slicing *)
      let x_slice_config =
        List.mapi
          (fun i spec ->
            match spec with
            | R [] -> (0, x_shape.(i))
            | R [ s ] ->
                let s' = normalize_index x_shape.(i) s in
                (s', s' + 1)
            | R [ s; e ] | R [ s; e; 1 ] ->
                let s' = normalize_index x_shape.(i) s in
                let e' = normalize_index x_shape.(i) e in
                if s' <= e' then (s', e' + 1) else (e', s' + 1)
            | _ -> assert false)
          full_slice
      in

      let x_view = shrink ctx x (Array.of_list x_slice_config) in
      let y_reshaped = reshape ctx y (shape x_view) in
      blit ctx y_reshaped x_view
    else
      (* General case: build scatter indices *)
      let total_updates = Array.fold_left ( * ) 1 y_shape in

      (* Create flattened y *)
      let y_flat = reshape ctx y [| total_updates |] in

      (* Create index tensor for scatter *)
      let scatter_indices =
        init ctx Dtype.int32 [| total_updates |] (fun arr ->
            let linear_idx = arr.(0) in

            (* Convert to multi-dimensional position in y *)
            let temp = ref linear_idx in
            let y_pos = Array.make (Array.length y_shape) 0 in
            for i = Array.length y_shape - 1 downto 0 do
              y_pos.(i) <- !temp mod y_shape.(i);
              temp := !temp / y_shape.(i)
            done;

            (* Map to position in x *)
            let x_pos =
              Array.mapi
                (fun i y_idx -> List.nth (List.nth indices_per_dim i) y_idx)
                y_pos
            in

            (* Convert to linear index in x *)
            let x_linear = ref 0 in
            let stride = ref 1 in
            for i = ndim - 1 downto 0 do
              x_linear := !x_linear + (x_pos.(i) * !stride);
              stride := !stride * x_shape.(i)
            done;

            Int32.of_int !x_linear)
      in

      (* Flatten x, scatter, reshape back *)
      let x_flat = reshape ctx x [| Array.fold_left ( * ) 1 x_shape |] in
      let result_flat = B.op_scatter ctx x_flat scatter_indices y_flat 0 in
      let result = reshape ctx result_flat x_shape in
      blit ctx result x

  let slice_ranges ctx ?(steps = []) starts stops x =
    let n_dims = List.length starts in
    if List.length stops <> n_dims then
      invalid_arg "slice_ranges: starts and stops must have same length";
    if steps <> [] && List.length steps <> n_dims then
      invalid_arg
        "slice_ranges: steps must have same length as starts/stops if provided";

    let slice_def =
      List.mapi
        (fun i (start, stop) ->
          let step = if steps = [] then 1 else List.nth steps i in
          R [ start; stop; step ])
        (List.combine starts stops)
    in
    slice ctx slice_def x

  let set_slice_ranges ctx ?(steps = []) starts stops x y =
    let n_dims = List.length starts in
    if List.length stops <> n_dims then
      invalid_arg "set_slice_ranges: starts and stops must have same length";
    if steps <> [] && List.length steps <> n_dims then
      invalid_arg
        "set_slice_ranges: steps must have same length as starts/stops if \
         provided";

    let slice_def =
      List.mapi
        (fun i (start, stop) ->
          let step = if steps = [] then 1 else List.nth steps i in
          R [ start; stop; step ])
        (List.combine starts stops)
    in
    set_slice ctx slice_def x y

  (* Get a single element *)
  let get ctx indices x =
    slice ctx (List.map (fun i -> I i) indices) x |> fun t ->
    if numel t = 1 then reshape ctx t [||]
    else invalid_arg "get requires indices for all dimensions"

  (* Set a single element *)
  let set ctx indices x value =
    let value_tensor =
      if numel value = 1 then reshape ctx value [||]
      else invalid_arg "set requires scalar value"
    in
    set_slice ctx (List.map (fun i -> I i) indices) x value_tensor

  let unsafe_get_item ctx indices x =
    let scalar_tensor = get ctx indices x in
    let ba = data scalar_tensor in
    Bigarray.Array1.get ba 0

  let unsafe_set_item ctx indices x value =
    let scalar_tensor = scalar ctx (dtype x) value in
    set ctx indices x scalar_tensor

  let array_split ctx t ~axis sections =
    let ndim = ndim t in
    let axis = resolve_single_axis t axis in
    let axis_size = dim axis t in

    match sections with
    | `Indices indices ->
        (* Split at specific indices *)
        let indices = Array.of_list indices in
        let n_sections = Array.length indices + 1 in
        let splits = Array.make n_sections t in

        (* Add boundaries *)
        let boundaries = Array.make (n_sections + 1) 0 in
        boundaries.(0) <- 0;
        Array.iteri (fun i idx -> boundaries.(i + 1) <- idx) indices;
        boundaries.(n_sections) <- axis_size;

        (* Create slices *)
        for i = 0 to n_sections - 1 do
          let start = boundaries.(i) in
          let stop = boundaries.(i + 1) in

          if start < stop then
            let slice_spec =
              List.init ndim (fun j ->
                  if j = axis then R [ start; stop - 1 ] else R [])
            in
            splits.(i) <- slice ctx slice_spec t
          else
            (* Empty slice *)
            let empty_shape = Array.copy (shape t) in
            empty_shape.(axis) <- 0;
            splits.(i) <- empty ctx (dtype t) empty_shape
        done;
        Array.to_list splits
    | `Count n ->
        (* Split into n sections *)
        if n <= 0 then invalid_arg "array_split: sections must be positive";

        let base_size = axis_size / n in
        let remainder = axis_size mod n in

        (* Calculate section sizes *)
        let sizes = Array.make n base_size in
        for i = 0 to remainder - 1 do
          sizes.(i) <- sizes.(i) + 1
        done;

        (* Create slices *)
        let splits = Array.make n t in
        let start = ref 0 in

        for i = 0 to n - 1 do
          let size = sizes.(i) in
          let stop = !start + size in

          let slice_spec =
            List.init ndim (fun j ->
                if j = axis then R [ !start; stop - 1 ] else R [])
          in
          splits.(i) <- slice ctx slice_spec t;
          start := stop
        done;

        Array.to_list splits

  let split ctx t ~axis sections =
    let axis = resolve_single_axis t axis in
    let axis_size = dim axis t in

    if axis_size mod sections <> 0 then
      invalid_arg
        (Printf.sprintf
           "split: array of size %d cannot be evenly split into %d sections"
           axis_size sections);

    array_split ctx t ~axis (`Count sections)

  (* *)

  let rand ctx dtype ?(seed = 42) shape =
    if not (Dtype.is_float dtype) then
      invalid_arg "rand only supports float dtypes";

    (* Check shape is valid *)
    if Array.exists (fun x -> x < 0) shape then
      invalid_arg "shape dimensions must be non-negative";

    (* If shape has 0, return zeros *)
    let numel = View.prod shape in
    if numel = 0 then zeros ctx dtype shape
    else
      (* Generate random int32 values using threefry *)
      let num_pairs = ceildiv numel 2 in

      (* Create counter tensors for threefry - offset by seed *)
      let counts0 = arange ctx Dtype.int32 seed (seed + num_pairs) 1 in
      let counts1 =
        arange ctx Dtype.int32 (seed + num_pairs) (seed + (2 * num_pairs)) 1
      in

      (* Generate random bits using threefry *)
      let random_bits = B.op_threefry ctx counts0 counts1 in

      (* Flatten and take only what we need *)
      let bits_flat = flatten ctx random_bits in
      let bits_needed =
        if numel < size bits_flat then shrink ctx bits_flat [| (0, numel) |]
        else bits_flat
      in

      (* Convert to float64 for precision during normalization *)
      let bits_float64 = cast ctx bits_needed Dtype.float64 in

      (* Add 2^31 to shift from signed [-2^31, 2^31-1] to unsigned [0, 2^32-1]
         range *)
      let offset = scalar ctx Dtype.float64 2147483648.0 in
      (* 2^31 *)
      let shifted = add ctx bits_float64 offset in

      (* Normalize to [0, 1) by dividing by 2^32 *)
      let normalizer = scalar ctx Dtype.float64 4294967296.0 in
      (* 2^32 *)
      let normalized = div ctx shifted normalizer in

      (* Cast to target dtype *)
      let result = cast ctx normalized dtype in

      (* Reshape to final shape *)
      reshape ctx result shape

  let randn ctx dtype ?(seed = 42) shape =
    (* Check that dtype is float *)
    if not (Dtype.is_float dtype) then
      invalid_arg "randn only supports float dtypes";

    (* Check shape is valid *)
    if Array.exists (fun x -> x < 0) shape then
      invalid_arg "shape dimensions must be non-negative";

    (* If shape has 0, return zeros *)
    let numel = View.prod shape in
    if numel = 0 then zeros ctx dtype shape
    else
      (* Box-Muller transform: generate pairs of uniform random values *)
      (* We need 2 uniform values per output value *)
      let rand_shape = Array.concat [ [| 2 |]; shape ] in

      (* Generate uniform random values in (0, 1] - we use a different seed
         offset for u2 *)
      let u1 = rand ctx Dtype.float32 ~seed rand_shape in
      let u2 = rand ctx Dtype.float32 ~seed:(seed + numel) rand_shape in

      (* Split into the two components *)
      let u1_part = slice ctx [ I 0 ] u1 in
      let u2_part = slice ctx [ I 1 ] u2 in

      (* Box-Muller transform: z0 = cos(2π * u1) * sqrt(-2 * ln(u2)) We use u2
         for the log to avoid log(0) *)

      (* Compute 2π * u1 *)
      let two_pi = scalar ctx Dtype.float32 (2.0 *. Float.pi) in
      let angle = mul ctx u1_part two_pi in

      (* Compute cos(2π * u1) *)
      let cos_part = cos ctx angle in

      (* Compute sqrt(-2 * ln(u2)) *)
      (* First ensure u2 is not exactly 0 by using 1 - original_uniform *)
      let one = ones_like ctx u2_part in
      let u2_safe = sub ctx one u2_part in
      (* Now in [0, 1) *)

      (* Add small epsilon to avoid log(0) *)
      let eps = scalar ctx Dtype.float32 1e-7 in
      let u2_nonzero = maximum ctx u2_safe eps in

      let log_u2 = log ctx u2_nonzero in
      let neg_two = scalar ctx Dtype.float32 (-2.0) in
      let sqrt_arg = mul ctx neg_two log_u2 in
      let sqrt_part = sqrt ctx sqrt_arg in

      (* Combine: z0 = cos_part * sqrt_part *)
      let result_f32 = mul ctx cos_part sqrt_part in

      (* Cast to target dtype *)
      cast ctx result_f32 dtype

  let randint ctx dtype ?(seed = 42) ?(high = 10) shape low =
    (* Check that dtype is int *)
    if not (Dtype.is_int dtype) then
      invalid_arg "randint only supports integer dtypes";

    (* Check shape is valid *)
    if Array.exists (fun x -> x < 0) shape then
      invalid_arg "shape dimensions must be non-negative";

    (* Check range is valid *)
    if low >= high then
      invalid_arg
        (Printf.sprintf "low (%d) must be less than high (%d)" low high);

    (* If shape has 0, return zeros *)
    let numel = View.prod shape in
    if numel = 0 then zeros ctx dtype shape
    else
      (* Generate uniform random floats in [0, 1) *)
      let uniform = rand ctx Dtype.float32 ~seed shape in

      (* Scale to [0, high-low) *)
      let range = float_of_int (high - low) in
      let range_tensor = scalar ctx Dtype.float32 range in
      let scaled = mul ctx uniform range_tensor in

      (* Shift to [low, high) *)
      let low_tensor = scalar ctx Dtype.float32 (float_of_int low) in
      let shifted = add ctx scaled low_tensor in

      (* Floor to get integers (truncate towards negative infinity) *)
      let floored = floor ctx shifted in

      (* Cast to target integer dtype *)
      cast ctx floored dtype

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
        (unsqueeze ctx ~axes:[| 0 |] a_orig, unsqueeze ctx ~axes:[| 1 |] b_orig)
      else if ndim_a_orig = 1 then
        (* (k), (...,k,n) -> a becomes (1,k) *)
        (unsqueeze ctx ~axes:[| 0 |] a_orig, b_orig)
      else if ndim_b_orig = 1 then
        (* (...,m,k), (k) -> b becomes (k,1) *)
        (a_orig, unsqueeze ctx ~axes:[| 1 |] b_orig)
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
      squeeze ctx ~axes:[| ndim result_intermediate - 2 |] result_intermediate
    else if ndim_b_orig = 1 then
      (* Original (...,m,k) @ (k) -> result (...,m,1) from dot -> squeeze last
         matrix dim *)
      squeeze ctx ~axes:[| ndim result_intermediate - 1 |] result_intermediate
    else
      (* Both original inputs were >= 2D, result from dot is already
         (...,m,n) *)
      result_intermediate

  (* ────────── Winograd Convolution Support ────────── *)

  (* Winograd F(4x4, 3x3) transformation matrices *)
  let winograd_f4x4_3x3_g =
    [|
      [| 1.0; 0.0; 0.0 |];
      [| -2.0 /. 3.0; -1.0 /. 3.0; -1.0 /. 3.0 |];
      [| -2.0 /. 3.0; 1.0 /. 3.0; -1.0 /. 3.0 |];
      [| 1.0 /. 6.0; 1.0 /. 3.0; 2.0 /. 3.0 |];
      [| 1.0 /. 6.0; -1.0 /. 3.0; 2.0 /. 3.0 |];
      [| 0.0; 0.0; 1.0 |];
    |]

  let winograd_f4x4_3x3_bt =
    [|
      [| 4.0; 0.0; -5.0; 0.0; 1.0; 0.0 |];
      [| 0.0; -4.0; -4.0; 1.0; 1.0; 0.0 |];
      [| 0.0; 4.0; -4.0; -1.0; 1.0; 0.0 |];
      [| 0.0; -2.0; -1.0; 2.0; 1.0; 0.0 |];
      [| 0.0; 2.0; -1.0; -2.0; 1.0; 0.0 |];
      [| 0.0; 4.0; 0.0; -5.0; 0.0; 1.0 |];
    |]

  let winograd_f4x4_3x3_at =
    [|
      [| 1.0; 1.0; 1.0; 1.0; 1.0; 0.0 |];
      [| 0.0; 1.0; -1.0; 2.0; -2.0; 0.0 |];
      [| 0.0; 1.0; 1.0; 4.0; 4.0; 0.0 |];
      [| 0.0; 1.0; -1.0; 8.0; -8.0; 1.0 |];
    |]

  (* Helper to create tensor columns for Winograd transformation *)
  let get_winograd_matcols ctx mat dims base_shape device_dtype =
    List.init dims (fun dim ->
        List.init
          (Array.length mat.(0))
          (fun k ->
            let col_tensors =
              Array.to_list
                (Array.map
                   (fun row ->
                     let value = row.(k) in
                     let target_shape = Array.copy base_shape in
                     Array.set target_shape dim 1;
                     full ctx device_dtype target_shape value)
                   mat)
            in
            B.op_cat ctx col_tensors dim))

  (* Apply Winograd transformation matrix to tensor *)
  let apply_winograd_matrix ctx mat t dims =
    let t_shape = shape t in
    let device_dtype = dtype t in

    (* Reshape to add dimension for expansion *)
    let new_shape_1 =
      Array.concat
        [
          Array.sub t_shape 0 dims;
          Array.make dims 1;
          Array.sub t_shape dims (Array.length t_shape - dims);
        ]
    in
    let t_reshaped = reshape ctx t new_shape_1 in

    (* Expand to add output dimensions *)
    let expand_shape =
      Array.concat
        [
          Array.sub t_shape 0 dims;
          Array.make dims (Array.length mat);
          Array.sub t_shape dims (Array.length t_shape - dims);
        ]
    in
    let t_expanded = expand ctx t_reshaped expand_shape in

    (* Get matrix columns *)
    let matcols =
      get_winograd_matcols ctx mat dims
        (Array.sub expand_shape dims (Array.length expand_shape - dims))
        device_dtype
    in

    (* Generate all index combinations *)
    let rec cartesian_product lists =
      match lists with
      | [] -> [ [] ]
      | h :: t ->
          let rest = cartesian_product t in
          List.concat (List.map (fun x -> List.map (fun rs -> x :: rs) rest) h)
    in

    let mat_indices =
      cartesian_product
        (List.init dims (fun _ -> List.init (Array.length mat.(0)) Fun.id))
    in

    (* Compute the sum of products *)
    let terms =
      List.map
        (fun indices ->
          (* Extract the slice t_expanded[..., indices[0], indices[1], ...] *)
          let t_slice =
            let shrink_config =
              Array.mapi
                (fun i size ->
                  if i < dims then (0, size)
                  else if i < dims * 2 then
                    let idx = List.nth indices (i - dims) in
                    (idx, idx + 1)
                  else (0, size))
                (shape t_expanded)
            in
            let sliced = shrink ctx t_expanded shrink_config in

            (* Squeeze out the indexed dimensions *)
            let rec squeeze_dims tensor dim_offset = function
              | [] -> tensor
              | _ :: rest ->
                  let squeezed =
                    squeeze ctx ~axes:[| dims + dim_offset |] tensor
                  in
                  squeeze_dims squeezed dim_offset rest
            in
            squeeze_dims sliced 0 indices
          in

          (* Get the product of matrix columns for this index combination *)
          let col_prod =
            List.fold_left2
              (fun acc col idx -> mul ctx acc (List.nth col idx))
              (ones_like ctx t_slice) matcols indices
          in

          mul ctx col_prod t_slice)
        mat_indices
    in

    (* Sum all terms *)
    match terms with
    | [] -> zeros_like ctx t (* Should never happen *)
    | first :: rest -> List.fold_left (add ctx) first rest

  (* ────────── Optimized Pool Implementation ────────── *)

  let pool_simple_path ctx x ~noop_rank ~o_s ~s_s ~k_s ~prefix_shape =
    let num_spatial = Array.length k_s in

    (* Pad if needed for stride *)
    let pad_spatial =
      Array.init num_spatial (fun i ->
          let pad_after =
            Stdlib.max 0 ((o_s.(i) * s_s.(i)) - (shape x).(noop_rank + i))
          in
          (0, pad_after))
    in
    let pad_config =
      Array.concat [ Array.make noop_rank (0, 0); pad_spatial ]
    in
    let x = pad ctx x pad_config (Dtype.zero (dtype x)) in

    (* Shrink to exact needed size *)
    let shrink_config =
      Array.mapi
        (fun i size ->
          if i < noop_rank then (0, size)
          else (0, o_s.(i - noop_rank) * s_s.(i - noop_rank)))
        (shape x)
    in
    let x = shrink ctx x shrink_config in

    (* Reshape to separate output let stride dimensions *)
    let reshape_list = ref (Array.to_list prefix_shape) in
    for i = 0 to num_spatial - 1 do
      reshape_list := !reshape_list @ [ o_s.(i); s_s.(i) ]
    done;
    let x = reshape ctx x (Array.of_list !reshape_list) in

    (* Shrink stride dimensions to kernel size *)
    let shrink2_config =
      Array.mapi
        (fun i size ->
          if i < noop_rank then (0, size)
          else if (i - noop_rank) mod 2 = 1 then (0, k_s.((i - noop_rank) / 2))
          else (0, size))
        (shape x)
    in
    let x = shrink ctx x shrink2_config in

    (* Permute to final layout *)
    let perm =
      Array.of_list
        (List.init noop_rank Fun.id
        @ List.init num_spatial (fun i -> noop_rank + (i * 2))
        @ List.init num_spatial (fun i -> noop_rank + (i * 2) + 1))
    in
    B.op_permute ctx x perm

  let pool_dilated_path ctx x ~noop_rank ~o_s ~s_s ~k_s ~d_s ~prefix_shape
      ~spatial_shape_in =
    let num_spatial = Array.length k_s in

    (* Calculate expansion factors *)
    let f_s =
      Array.init num_spatial (fun j ->
          let oj, sj, ij, dj, kj =
            (o_s.(j), s_s.(j), spatial_shape_in.(j), d_s.(j), k_s.(j))
          in
          let eff_kernel_span = (dj * (kj - 1)) + 1 in
          if oj * sj > ij - eff_kernel_span + 1 then 2 else 1)
    in

    (* Calculate repeat factors *)
    let repeat_factors =
      Array.init num_spatial (fun j ->
          let kj, ij, fj, dj =
            (k_s.(j), spatial_shape_in.(j), f_s.(j), d_s.(j))
          in
          ceildiv (kj * ((ij * fj) + dj)) ij)
    in

    (* Tile the input *)
    let repeat_factors_full =
      Array.concat [ Array.make noop_rank 1; repeat_factors ]
    in
    let x = tile ctx repeat_factors_full x in

    (* First shrink *)
    let shrink1_limits =
      Array.init num_spatial (fun j ->
          let kj, ij, fj, dj =
            (k_s.(j), spatial_shape_in.(j), f_s.(j), d_s.(j))
          in
          kj * ((ij * fj) + dj))
    in
    let shrink1_config =
      Array.init (ndim x) (fun i ->
          if i < noop_rank then (0, prefix_shape.(i))
          else (0, shrink1_limits.(i - noop_rank)))
    in
    let x = shrink ctx x shrink1_config in

    (* First reshape to separate kernel and spatial+dilation dimensions *)
    let reshape1_list = ref (Array.to_list prefix_shape) in
    for j = 0 to num_spatial - 1 do
      let kj, ij, fj, dj = (k_s.(j), spatial_shape_in.(j), f_s.(j), d_s.(j)) in
      reshape1_list := !reshape1_list @ [ kj; (ij * fj) + dj ]
    done;
    let x = reshape ctx x (Array.of_list !reshape1_list) in

    (* Second shrink to output size *)
    let shrink2_config =
      Array.mapi
        (fun i size ->
          if i < noop_rank then (0, size)
          else if (i - noop_rank) mod 2 = 1 then
            (* This is an inner dimension *)
            let j = (i - noop_rank) / 2 in
            (0, o_s.(j) * s_s.(j))
          else (0, size))
        (shape x)
    in
    let x = shrink ctx x shrink2_config in

    (* Second reshape to separate output and stride dimensions *)
    let reshape2_list = ref (Array.to_list prefix_shape) in
    for j = 0 to num_spatial - 1 do
      reshape2_list := !reshape2_list @ [ k_s.(j); o_s.(j); s_s.(j) ]
    done;
    let x = reshape ctx x (Array.of_list !reshape2_list) in

    (* Third shrink to stride=1 (select every s_s-th element) *)
    let shrink3_config =
      Array.mapi
        (fun i size ->
          if i < noop_rank then (0, size)
          else if (i - noop_rank) mod 3 = 2 then
            (* This is a stride dimension *)
            (0, 1)
          else (0, size))
        (shape x)
    in
    let x = shrink ctx x shrink3_config in

    (* Third reshape to remove stride dimensions *)
    let reshape3_list = ref (Array.to_list prefix_shape) in
    for j = 0 to num_spatial - 1 do
      reshape3_list := !reshape3_list @ [ k_s.(j); o_s.(j) ]
    done;
    let x = reshape ctx x (Array.of_list !reshape3_list) in

    (* Final permutation to get (..., o_1, o_2, ..., k_1, k_2, ...) *)
    let perm =
      Array.of_list
        (List.init noop_rank Fun.id
        @ List.init num_spatial (fun j -> noop_rank + (j * 2) + 1)
        @
        (* output dims *)
        List.init num_spatial (fun j -> noop_rank + (j * 2)) (* kernel dims *))
    in
    B.op_permute ctx x perm

  let pool ctx x_padded_input ~k_s ~s_s ~d_s =
    let x_ndim = ndim x_padded_input in
    let num_spatial = Array.length k_s in

    if num_spatial = 0 then x_padded_input
    else if x_ndim < num_spatial then
      invalid_arg
        "pool: input tensor ndim less than number of spatial kernel dimensions"
    else
      let noop_rank = x_ndim - num_spatial in
      let prefix_shape = Array.sub (shape x_padded_input) 0 noop_rank in
      let spatial_shape_in =
        Array.sub (shape x_padded_input) noop_rank num_spatial
      in

      (* Calculate output shape *)
      let o_s =
        Array.init num_spatial (fun j ->
            let eff_kernel_span = (d_s.(j) * (k_s.(j) - 1)) + 1 in
            if spatial_shape_in.(j) < eff_kernel_span then 0
            else ((spatial_shape_in.(j) - eff_kernel_span) / s_s.(j)) + 1)
      in

      if Array.exists (( = ) 0) o_s then
        let final_target_shape = Array.concat [ prefix_shape; o_s; k_s ] in
        empty ctx (dtype x_padded_input) final_target_shape
      else
        (* Check if we can use simple path *)
        let use_simple_path =
          Array.for_all2 (fun k s -> k <= s) k_s s_s
          && Array.for_all (( = ) 1) d_s
        in

        if use_simple_path then
          pool_simple_path ctx x_padded_input ~noop_rank ~o_s ~s_s ~k_s
            ~prefix_shape
        else
          pool_dilated_path ctx x_padded_input ~noop_rank ~o_s ~s_s ~k_s ~d_s
            ~prefix_shape ~spatial_shape_in

  (* ────────── Optimized Convolution with Winograd ────────── *)

  let should_use_winograd ~kernel_size ~stride ~groups =
    groups = 1
    && Array.length kernel_size = 2
    && kernel_size.(0) = 3
    && kernel_size.(1) = 3
    && stride.(0) = 1
    && stride.(1) = 1

  let winograd_conv2d ctx x w =
    let bs, cin, h, w_dim = (dim 0 x, dim 1 x, dim 2 x, dim 3 x) in
    let cout, _, _kh, _kw = (dim 0 w, dim 1 w, dim 2 w, dim 3 w) in

    (* Transform weights: (cout, cin, 3, 3) -> (cout, cin, 6, 6) *)
    let w_transformed = apply_winograd_matrix ctx winograd_f4x4_3x3_g w 2 in

    (* Prepare input tiles *)
    let tile_h = (h + 3) / 4 in
    let tile_w = (w_dim + 3) / 4 in

    (* Pad input to multiple of 4 *)
    let pad_h = (tile_h * 4) - h in
    let pad_w = (tile_w * 4) - w_dim in
    let x_padded =
      pad ctx x
        [| (0, 0); (0, 0); (0, pad_h); (0, pad_w) |]
        (Dtype.zero (dtype x))
    in

    (* Extract 6x6 tiles with 4x4 stride *)
    let x_tiles =
      pool ctx x_padded ~k_s:[| 6; 6 |] ~s_s:[| 4; 4 |] ~d_s:[| 1; 1 |]
    in
    (* Shape: (bs, cin, tile_h, tile_w, 6, 6) *)

    (* Apply B^T transformation *)
    let x_transformed =
      apply_winograd_matrix ctx winograd_f4x4_3x3_bt x_tiles 2
    in

    (* Reshape for matmul: merge tile dimensions *)
    let x_reshaped =
      let s = shape x_transformed in
      reshape ctx x_transformed [| s.(0); s.(1); s.(2) * s.(3); s.(4); s.(5) |]
    in

    (* Prepare weights for broadcasting *)
    let w_reshaped =
      let s = shape w_transformed in
      reshape ctx w_transformed [| s.(0); s.(1); 1; s.(2); s.(3) |]
    in

    (* Element-wise multiplication in transform space *)
    let y_transformed =
      let x_exp = unsqueeze ctx ~axes:[| 1 |] x_reshaped in
      let w_exp =
        expand ctx w_reshaped [| 1; cout; cin; tile_h * tile_w; 6; 6 |]
      in
      let prod = mul ctx x_exp w_exp in
      sum ctx prod ~axes:[| 2 |]
      (* sum over cin *)
    in

    (* Apply A^T transformation *)
    let y_tiles =
      apply_winograd_matrix ctx winograd_f4x4_3x3_at y_transformed 2
    in

    (* Reshape back to image format *)
    (* y_tiles has shape (bs, cout, tile_h * tile_w, 4, 4) *)
    let y_reshaped =
      let s = shape y_tiles in
      reshape ctx y_tiles [| s.(0); s.(1); tile_h; tile_w; s.(3); s.(4) |]
    in

    (* Permute to group spatial dimensions: (bs, cout, tile_h, 4, tile_w, 4) *)
    let y_permuted = transpose ctx ~axes:[| 0; 1; 2; 4; 3; 5 |] y_reshaped in

    (* Merge tile dimensions with spatial dimensions *)
    let y_merged =
      reshape ctx y_permuted [| bs; cout; tile_h * 4; tile_w * 4 |]
    in

    (* Remove padding to get original output size *)
    shrink ctx y_merged [| (0, bs); (0, cout); (0, h); (0, w_dim) |]

  let calculate_padding_for_mode input_spatial_shape ~k_s ~s_s ~d_s
      ~(mode : [< `Full | `Valid | `Same ]) =
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
            let pad_each_side = d_s.(i) * (k_s.(i) - 1) in
            (pad_each_side, pad_each_side))
    | `Same ->
        Array.init num_spatial (fun i ->
            let is_d, ss_d, ks_d, ds_d =
              (input_spatial_shape.(i), s_s.(i), k_s.(i), d_s.(i))
            in
            let os_d = ceildiv is_d ss_d in
            let eff_ks_d = (ds_d * (ks_d - 1)) + 1 in
            let total_pad_d =
              Stdlib.max 0 (((os_d - 1) * ss_d) + eff_ks_d - is_d)
            in
            let pad_before = total_pad_d / 2 in
            let pad_after = total_pad_d - pad_before in
            (pad_before, pad_after))

  let correlate_nd_general ctx ~groups stride_s_arr ~padding_mode dilation_s_arr
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

    let num_prefix_dims = 2 in
    let op_pad_config_list_prefix =
      Array.to_list (Array.make num_prefix_dims (0, 0))
    in
    let op_pad_config_list_spatial = Array.to_list padding_config_pairs_arr in
    let op_pad_config_arr =
      Array.of_list (op_pad_config_list_prefix @ op_pad_config_list_spatial)
    in

    let x_padded = B.op_pad ctx x op_pad_config_arr actual_fillvalue in

    let pooled_x =
      pool ctx x_padded ~k_s:kernel_spatial_shape_arr ~s_s:stride_s_arr
        ~d_s:dilation_s_arr
    in

    let output_spatial_shape_arr =
      Array.init num_spatial_dims (fun i ->
          (shape pooled_x).(num_prefix_dims + i))
    in

    (* Reshape pooled_x to (bs, groups, cin_per_group, 1, output_spatial...,
       kernel_spatial...) *)
    let shape_x_pre_expand_list =
      [ bs; groups; cin_per_group; 1 ]
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
    let perm_axes_list_prefix_output = [ 0; 1; 3 ] in
    let perm_axes_list_output_spatial =
      List.init num_spatial_dims (fun i -> 4 + i)
    in
    let perm_axes_list_cin_group = [ 2 ] in
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

    (* Reshape w to (1, groups, rcout, 1s, cin_per_group, kernel_spatial...) *)
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

    (* Sum over cin_per_group and kernel_spatial_dims *)
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

  let correlate_nd ctx ?(groups = 1) stride_s_arr
      ?(padding_mode : [ `Full | `Valid | `Same ] = `Valid) dilation_s_arr
      ?fillvalue num_spatial_dims x w ?bias () =
    (* Check if we should use Winograd for 2D 3x3 convolutions *)
    if
      num_spatial_dims = 2
      && should_use_winograd
           ~kernel_size:(Array.sub (shape w) 2 2)
           ~stride:stride_s_arr ~groups
    then
      let result = winograd_conv2d ctx x w in
      match bias with
      | None -> result
      | Some b -> add ctx result (reshape ctx b [| 1; dim 0 b; 1; 1 |])
    else
      (* Original implementation *)
      correlate_nd_general ctx ~groups stride_s_arr ~padding_mode dilation_s_arr
        ?fillvalue num_spatial_dims x w ?bias ()

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

  (** Helper to resolve padding specification for pooling/convolution
      operations. Input `padding_spec` is user-facing. Output `(int*int) array`
      is for `B.op_pad`, (pad_before, pad_after) for each spatial dimension. *)
  let resolve_padding_for_ops padding_spec ~input_spatial_shape ~k_s ~s_s ~d_s =
    match padding_spec with
    | `Same | `Valid | `Full ->
        calculate_padding_for_mode input_spatial_shape ~k_s ~s_s ~d_s
          ~mode:padding_spec

  (** Helper to adjust padding for ceil_mode=true. Analogous to tinygrad's
      _apply_ceil_mode. Input `current_pads_pairs` is (pad_before, pad_after)
      for each spatial dim. Output is new (pad_before, pad_after) array for each
      spatial dim. *)
  let apply_ceil_mode ~current_pads_pairs ~input_spatial_shape ~k_s ~s_s ~d_s =
    let num_spatial_dims = Array.length k_s in
    let pads_adj = Array.copy current_pads_pairs in
    let o_s =
      Array.init num_spatial_dims (fun i ->
          let i_d = input_spatial_shape.(i) in
          let d_d = d_s.(i) in
          let k_d = k_s.(i) in
          let s_d = s_s.(i) in
          let p_b, p_a = current_pads_pairs.(i) in
          ceildiv (i_d + p_b + p_a - ((d_d * (k_d - 1)) + 1)) s_d + 1)
    in
    for i = 0 to num_spatial_dims - 1 do
      let o_d, i_d, s_d, k_d, d_d =
        (o_s.(i), input_spatial_shape.(i), s_s.(i), k_s.(i), d_s.(i))
      in
      let p_b, p_a = current_pads_pairs.(i) in
      let pad_needed_for_last_window_start =
        (s_d * (o_d - 1)) + ((d_d * (k_d - 1)) + 1) - (i_d + p_b + p_a)
      in
      let effective_pad_before_input_start =
        Stdlib.max 0 ((s_d * (o_d - 1)) - (p_b + i_d - 1))
      in
      (* Adjust pad_after (pads_adj.(i) |> snd) *)
      pads_adj.(i) <-
        ( fst pads_adj.(i),
          snd pads_adj.(i)
          + pad_needed_for_last_window_start - effective_pad_before_input_start
        )
    done;
    pads_adj

  let pool_setup ~num_spatial_dims ~kernel_size ?stride ?dilation ~padding_spec
      ~ceil_mode x_orig =
    let x_ndim = ndim x_orig in
    let input_spatial_shape =
      Array.sub (shape x_orig) (x_ndim - num_spatial_dims) num_spatial_dims
    in
    let s_s = Option.value stride ~default:kernel_size in
    let d_s = Option.value dilation ~default:(Array.make num_spatial_dims 1) in

    let reg_pads =
      resolve_padding_for_ops padding_spec ~input_spatial_shape ~k_s:kernel_size
        ~s_s ~d_s
    in
    let pads =
      if ceil_mode then
        apply_ceil_mode ~current_pads_pairs:reg_pads ~input_spatial_shape
          ~k_s:kernel_size ~s_s ~d_s
      else reg_pads
    in
    let full_pad_config =
      Array.concat [ Array.make (x_ndim - num_spatial_dims) (0, 0); pads ]
    in

    (input_spatial_shape, s_s, d_s, pads, reg_pads, full_pad_config)

  let avg_pool_nd ctx ~kernel_size ?stride ?dilation ~padding_spec ~ceil_mode
      ~count_include_pad ~num_spatial_dims x_orig =
    let x_ndim = ndim x_orig in

    (* Use pool_setup helper *)
    let ( _input_spatial_shape,
          s_s,
          d_s,
          current_pads_pairs,
          reg_pads_pairs,
          full_pad_config ) =
      pool_setup ~num_spatial_dims ~kernel_size ?stride ?dilation ~padding_spec
        ~ceil_mode x_orig
    in

    (* Always pad and pool *)
    let x_padded = pad ctx x_orig full_pad_config (Dtype.zero (dtype x_orig)) in
    let pooled_x = pool ctx x_padded ~k_s:kernel_size ~s_s ~d_s in

    let reduction_axes =
      Array.init num_spatial_dims (fun i ->
          ndim pooled_x - num_spatial_dims + i)
    in

    (* Compute sum *)
    let sum_pooled = sum ctx pooled_x ~axes:reduction_axes ~keepdims:false in

    (* Compute divisor based on mode *)
    if count_include_pad && not ceil_mode then
      (* Simple case: divide by kernel size *)
      let kernel_numel = View.prod kernel_size in
      div_s ctx sum_pooled (float_of_int kernel_numel)
    else
      (* Need to count valid elements *)
      let ones = ones_like ctx x_orig in
      let ones_padded =
        if ceil_mode && count_include_pad then
          (* Special padding for ceil_mode divisor calculation *)
          let reg_pad_config =
            Array.concat
              [ Array.make (x_ndim - num_spatial_dims) (0, 0); reg_pads_pairs ]
          in
          let ones_reg =
            pad ctx ones reg_pad_config (Dtype.zero (dtype ones))
          in
          let extra_pads =
            Array.map2
              (fun (cb, ca) (rb, ra) -> (cb - rb, ca - ra))
              current_pads_pairs reg_pads_pairs
          in
          let extra_pad_config =
            Array.concat
              [ Array.make (x_ndim - num_spatial_dims) (0, 0); extra_pads ]
          in
          pad ctx ones_reg extra_pad_config (Dtype.zero (dtype ones))
        else pad ctx ones full_pad_config (Dtype.zero (dtype ones))
      in
      let pooled_ones = pool ctx ones_padded ~k_s:kernel_size ~s_s ~d_s in
      let count = sum ctx pooled_ones ~axes:reduction_axes ~keepdims:false in
      div ctx sum_pooled count

  let max_pool_nd ctx ~kernel_size ?stride ?dilation ~padding_spec ~ceil_mode
      ~return_indices ~num_spatial_dims x_orig =
    let x_ndim = ndim x_orig in

    (* Use pool_setup helper *)
    let input_spatial_shape, s_s, d_s, current_pads_pairs, _, full_pad_config =
      pool_setup ~num_spatial_dims ~kernel_size ?stride ?dilation ~padding_spec
        ~ceil_mode x_orig
    in

    let reduction_axes =
      let pooled_ndim = x_ndim + num_spatial_dims in
      Array.init num_spatial_dims (fun i -> pooled_ndim - num_spatial_dims + i)
    in

    let fill_value = Dtype.min_val (dtype x_orig) in
    let x_padded = pad ctx x_orig full_pad_config fill_value in
    let pooled = pool ctx x_padded ~k_s:kernel_size ~s_s ~d_s in
    let max_values =
      B.op_reduce_max ctx pooled ~axes:reduction_axes ~keepdims:false
    in

    if not return_indices then (max_values, None)
    else
      let prod_spatial_size = View.prod input_spatial_shape in

      (* Create forward indices directly *)
      let indices_flat = arange ctx Dtype.int32 0 prod_spatial_size 1 in
      let indices_spatial = reshape ctx indices_flat input_spatial_shape in

      (* Pad indices with -1 (invalid index marker) *)
      let indices_padded =
        pad ctx indices_spatial current_pads_pairs (Int32.of_int (-1))
      in

      (* Broadcast and pool indices *)
      let shape_prefix_template =
        Array.sub (shape x_orig) 0 (x_ndim - num_spatial_dims)
      in
      let indices_broadcast =
        broadcast_to ctx indices_padded
          (Array.concat [ shape_prefix_template; shape indices_padded ])
      in
      let pooled_indices =
        pool ctx indices_broadcast ~k_s:kernel_size ~s_s ~d_s
      in

      (* Mask with max values *)
      let max_values_expanded =
        B.op_reduce_max ctx pooled ~axes:reduction_axes ~keepdims:true
      in
      let is_max = equal ctx pooled max_values_expanded in

      (* Select first occurrence of max (lowest index) *)
      let invalid_idx =
        scalar ctx Dtype.int32 (Int32.of_int prod_spatial_size)
      in
      let masked_indices =
        where ctx is_max pooled_indices
          (broadcast_to ctx invalid_idx (shape pooled_indices))
      in

      (* Get minimum valid index (first occurrence) *)
      let min_indices =
        min ctx masked_indices ~axes:reduction_axes ~keepdims:false
      in

      (* Filter out invalid indices *)
      let valid_mask = cmplt ctx min_indices invalid_idx in
      let final_indices =
        where ctx valid_mask min_indices (scalar ctx Dtype.int32 0l)
      in

      (max_values, Some final_indices)

  let avg_pool1d ctx x ~kernel_size ?stride ?dilation ?(padding_spec = `Valid)
      ?(ceil_mode = false) ?(count_include_pad = true) () =
    avg_pool_nd ctx x ~kernel_size:[| kernel_size |]
      ?stride:(Option.map (fun s -> [| s |]) stride)
      ?dilation:(Option.map (fun d -> [| d |]) dilation)
      ~padding_spec ~ceil_mode ~count_include_pad ~num_spatial_dims:1

  let avg_pool2d ctx x ~kernel_size ?stride ?dilation ?(padding_spec = `Valid)
      ?(ceil_mode = false) ?(count_include_pad = true) () =
    let ks_arr = [| fst kernel_size; snd kernel_size |] in
    let s_arr_opt = Option.map (fun s -> [| fst s; snd s |]) stride in
    let d_arr_opt = Option.map (fun d -> [| fst d; snd d |]) dilation in
    avg_pool_nd ctx x ~kernel_size:ks_arr ?stride:s_arr_opt ?dilation:d_arr_opt
      ~padding_spec ~ceil_mode ~count_include_pad ~num_spatial_dims:2

  let max_pool1d ctx x ~kernel_size ?stride ?dilation ?(padding_spec = `Valid)
      ?(ceil_mode = false) ?(return_indices = false) () =
    max_pool_nd ctx x ~kernel_size:[| kernel_size |]
      ?stride:(Option.map (fun s -> [| s |]) stride)
      ?dilation:(Option.map (fun d -> [| d |]) dilation)
      ~padding_spec ~ceil_mode ~return_indices ~num_spatial_dims:1

  let max_pool2d ctx x ~kernel_size ?stride ?dilation ?(padding_spec = `Valid)
      ?(ceil_mode = false) ?(return_indices = false) () =
    let ks_arr = [| fst kernel_size; snd kernel_size |] in
    let s_arr_opt = Option.map (fun s -> [| fst s; snd s |]) stride in
    let d_arr_opt = Option.map (fun d -> [| fst d; snd d |]) dilation in
    max_pool_nd ctx x ~kernel_size:ks_arr ?stride:s_arr_opt ?dilation:d_arr_opt
      ~padding_spec ~ceil_mode ~return_indices ~num_spatial_dims:2

  (** Helper for N-dim one-hot encoding. Creates a new last dimension for
      classes. *)
  let one_hot ctx index_tensor ~num_classes =
    let index_dt = dtype index_tensor in
    if not (Dtype.is_int index_dt || Dtype.is_uint index_dt) then
      invalid_arg "one_hot_nd: index_tensor must be an integer type";

    let index_expanded =
      unsqueeze ctx index_tensor ~axes:[| ndim index_tensor |]
    in
    (* Add new last dim *)

    let arange_t = arange ctx index_dt 0 num_classes 1 in
    (* Classes 0 to num_classes-1 *)

    (* Reshape arange to be (1, ..., 1, num_classes) to align with new last dim
       of index_expanded *)
    let ndim_expanded = ndim index_expanded in
    let shape_for_arange = Array.make ndim_expanded 1 in
    shape_for_arange.(ndim_expanded - 1) <- num_classes;
    let arange_b = reshape ctx arange_t shape_for_arange in

    cmpeq ctx index_expanded arange_b (* Broadcasts to one-hot mask *)

  (** Internal N-Dimensional max unpooling. *)
  let max_unpool_nd ctx ~kernel_size ?stride ?dilation ~padding_spec
      ?output_size_opt ~num_spatial_dims input_t indices_t =
    let bs = dim 0 input_t in
    let c = dim 1 input_t in
    let pooled_spatial_shape = Array.sub (shape input_t) 2 num_spatial_dims in

    let output_spatial_shape =
      match output_size_opt with
      | Some os_arr -> os_arr
      | None ->
          let s_s = Option.value stride ~default:kernel_size in
          let d_s =
            Option.value dilation ~default:(Array.make num_spatial_dims 1)
          in
          let pads_pairs =
            resolve_padding_for_ops padding_spec
              ~input_spatial_shape:pooled_spatial_shape
                (* Placeholder, see note in thought process *)
              ~k_s:kernel_size ~s_s ~d_s
          in
          Array.init num_spatial_dims (fun i ->
              let pooled_dim_size = pooled_spatial_shape.(i) in
              let k = kernel_size.(i) in
              let s = s_s.(i) in
              let d = d_s.(i) in
              let pb, pa = pads_pairs.(i) in
              ((pooled_dim_size - 1) * s) - pb - pa + ((d * (k - 1)) + 1))
    in
    let prod_output_spatial_size = View.prod output_spatial_shape in

    let one_hot_mask_for_indices =
      one_hot ctx indices_t ~num_classes:prod_output_spatial_size
    in

    let input_expanded = unsqueeze ctx input_t ~axes:[| ndim input_t |] in

    let multiplied = mul ctx one_hot_mask_for_indices input_expanded in

    let sum_axes = Array.init num_spatial_dims (fun i -> 2 + i) in
    let result_flat_spatial =
      sum ctx multiplied ~axes:sum_axes ~keepdims:false
    in

    let final_shape = Array.concat [ [| bs; c |]; output_spatial_shape ] in
    reshape ctx result_flat_spatial final_shape

  let max_unpool1d ctx input_t indices_t ~kernel_size ?stride ?dilation
      ?(padding_spec = `Valid) ?output_size_opt () =
    max_unpool_nd ctx input_t indices_t ~kernel_size:[| kernel_size |]
      ?stride:(Option.map (fun s -> [| s |]) stride)
      ?dilation:(Option.map (fun d -> [| d |]) dilation)
      ~padding_spec ?output_size_opt ~num_spatial_dims:1

  let max_unpool2d ctx input_t indices_t ~kernel_size ?stride ?dilation
      ?(padding_spec = `Valid) ?output_size_opt () =
    let ks_arr = [| fst kernel_size; snd kernel_size |] in
    let s_arr_opt = Option.map (fun s -> [| fst s; snd s |]) stride in
    let d_arr_opt = Option.map (fun d -> [| fst d; snd d |]) dilation in
    max_unpool_nd ctx input_t indices_t ~kernel_size:ks_arr ?stride:s_arr_opt
      ?dilation:d_arr_opt ~padding_spec ?output_size_opt ~num_spatial_dims:2

  (* *)

  let sort (type a b) ctx ?(descending = false) ?(axis = -1) (t : (a, b) t) =
    let axis = resolve_single_axis t axis in
    let orig_len = dim axis t in

    (* Handle edge case of empty or single element *)
    if orig_len <= 1 then
      let idx = arange ctx Dtype.int32 0 orig_len 1 in
      let idx_shape =
        Array.init (ndim t) (fun i -> if i = axis then orig_len else 1)
      in
      let idx = reshape ctx idx idx_shape |> fun x -> expand ctx x (shape t) in
      (t, idx)
    else
      (* Calculate number of stages for bitonic sort *)
      let n_stages =
        int_of_float (Float.ceil (Float.log2 (float_of_int orig_len)))
      in
      let padded_len = 1 lsl n_stages in

      (* Pad to power of 2 *)
      let fill_value =
        if descending then Dtype.min_val (dtype t)
        else
          (* Use a large value for ascending sort *)
          match dtype t with
          | dt when Dtype.is_float dt -> Dtype.float_to_dtype dt Float.infinity
          | Dtype.Int32 -> Int32.max_int
          | Dtype.Int64 -> Int64.max_int
          | dt -> Dtype.float_to_dtype dt 1e10 (* Fallback for other types *)
      in

      let pad_config =
        Array.init (ndim t) (fun i ->
            if i = axis then (0, padded_len - orig_len) else (0, 0))
      in

      let x = pad ctx t pad_config fill_value in

      (* Unflatten into binary tree structure *)
      let unflatten_sizes = Array.make n_stages 2 in
      let x = unflatten ctx x axis unflatten_sizes in

      (* Bitonic sort implementation *)
      let x = ref x in
      for stage = 1 to n_stages do
        (* Handle crossover for all stages except the last *)
        if stage <> n_stages then (
          let crossover_dim = axis + n_stages - stage - 1 in

          (* Split along crossover dimension *)
          let blue_slice =
            List.init (ndim !x) (fun i ->
                if i = crossover_dim then I 0 else R [])
          in
          let green_slice =
            List.init (ndim !x) (fun i ->
                if i = crossover_dim then I 1 else R [])
          in

          let blue_box = slice ctx blue_slice !x in
          let green_box = slice ctx green_slice !x in

          (* Flip green box dimensions *)
          let flip_axes =
            Array.to_list (Array.init (ndim !x) (fun i -> i))
            |> List.filter (fun i -> i > crossover_dim)
            |> Array.of_list
          in
          let green_box_flipped = flip ctx green_box ~axes:flip_axes in

          (* Reconstruct by stacking *)
          x := stack ctx ~axis:crossover_dim [ blue_box; green_box_flipped ];
          x := contiguous ctx !x);

        (* Compare and swap substages *)
        for substage = stage - 1 downto 0 do
          let partner_dim = axis + n_stages - substage - 1 in

          (* Split along partner dimension *)
          let top_slice =
            List.init (ndim !x) (fun i -> if i = partner_dim then I 0 else R [])
          in
          let bottom_slice =
            List.init (ndim !x) (fun i -> if i = partner_dim then I 1 else R [])
          in

          let x_top = slice ctx top_slice !x in
          let x_bottom = slice ctx bottom_slice !x in

          (* Compare and order *)
          let x_larger = maximum ctx x_top x_bottom in
          let x_smaller = minimum ctx x_top x_bottom in

          (* Stack based on sort order *)
          x :=
            if descending then
              stack ctx ~axis:partner_dim [ x_larger; x_smaller ]
            else stack ctx ~axis:partner_dim [ x_smaller; x_larger ];
          x := contiguous ctx !x
        done;

        (* Undo crossover if needed *)
        if stage <> n_stages then
          let crossover_dim = axis + n_stages - stage - 1 in

          let blue_slice =
            List.init (ndim !x) (fun i ->
                if i = crossover_dim then I 0 else R [])
          in
          let green_slice =
            List.init (ndim !x) (fun i ->
                if i = crossover_dim then I 1 else R [])
          in

          let blue_box = slice ctx blue_slice !x in
          let flipped_green_box = slice ctx green_slice !x in

          (* Unflip *)
          let flip_axes =
            Array.to_list (Array.init (ndim !x) (fun i -> i))
            |> List.filter (fun i -> i > crossover_dim)
            |> Array.of_list
          in
          let green_box = flip ctx ~axes:flip_axes flipped_green_box in

          x := stack ctx ~axis:crossover_dim [ blue_box; green_box ]
      done;

      (* Flatten back to original shape *)
      let x_sorted =
        flatten ctx ~start_dim:axis ~end_dim:(axis + n_stages - 1) !x
      in

      (* Remove padding *)
      let shrink_slice =
        List.init (ndim x_sorted) (fun i ->
            if i = axis then R [ 0; orig_len - 1 ] else R [])
      in
      let x_sorted = slice ctx shrink_slice x_sorted in

      (* Compute indices for stable sort *)
      (* Create index tensor *)
      let idx = arange ctx Dtype.int32 0 orig_len 1 in
      let idx_shape =
        Array.init (ndim t) (fun i -> if i = axis then orig_len else 1)
      in
      let idx = reshape ctx idx idx_shape |> fun x -> expand ctx x (shape t) in

      (* Compute counts for handling duplicates *)
      let compute_counts tensor =
        (* Count how many elements <= current index with same value *)
        let t_exp_new = unsqueeze ctx tensor ~axes:[| axis + 1 |] in
        let t_exp_orig = unsqueeze ctx tensor ~axes:[| axis |] in
        let idx_exp_new = unsqueeze ctx idx ~axes:[| axis + 1 |] in
        let idx_exp_orig = unsqueeze ctx idx ~axes:[| axis |] in

        let le_mask = less_equal ctx idx_exp_orig idx_exp_new in
        let eq_mask = equal ctx t_exp_orig t_exp_new in
        let mask = logical_and ctx le_mask eq_mask in
        sum ctx mask ~axes:[| axis + 1 |] ~keepdims:false
      in

      let count_orig = compute_counts t in
      let count_sorted = compute_counts x_sorted in

      (* Find where each original element ended up *)
      let self_exp = unsqueeze ctx t ~axes:[| axis + 1 |] in
      let sorted_exp = unsqueeze ctx x_sorted ~axes:[| axis |] in
      let count_orig_exp = unsqueeze ctx count_orig ~axes:[| axis + 1 |] in
      let count_sorted_exp = unsqueeze ctx count_sorted ~axes:[| axis |] in
      let idx_exp = unsqueeze ctx idx ~axes:[| axis + 1 |] in

      (* Match by value and count *)
      let value_match = equal ctx self_exp sorted_exp in
      let count_match = equal ctx count_orig_exp count_sorted_exp in
      let matches = logical_and ctx value_match count_match in

      (* Extract indices where matches occur *)
      let matches_int = cast ctx matches Dtype.int32 in
      let weighted_idx = mul ctx matches_int idx_exp in
      let final_idx = sum ctx weighted_idx ~axes:[| axis |] ~keepdims:false in

      (x_sorted, final_idx)

  let argsort ctx ?(descending = false) ?(axis = -1) t =
    let _, indices = sort ctx ~descending ~axis t in
    indices

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
    let t_dtype = dtype t in
    let t_inverted =
      if Dtype.is_float t_dtype then neg ctx t
      else if Dtype.is_int t_dtype && not (Dtype.is_uint t_dtype) then neg ctx t
      else if Dtype.is_uint t_dtype then
        let max_val_specific : (a, b) t =
          match t_dtype with
          | Dtype.UInt8 -> scalar ctx Dtype.uint8 255
          | Dtype.UInt16 -> scalar ctx Dtype.uint16 65535
          | _ -> failwith "argmin: unsupported uint dtype for inversion"
        in
        let max_val_b = broadcast_to ctx max_val_specific (shape t) in
        sub ctx max_val_b t
      else logical_not ctx t
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

  let pp_dtype _context fmt dtype =
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
