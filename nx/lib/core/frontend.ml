(* High-level tensor operations built on backend [B]. *)

module Make (B : Backend_intf.S) = struct
  module B = B

  (* ───── Core Types and Context ───── *)

  type ('a, 'b) t = ('a, 'b) B.t
  type context = B.context

  (* Concrete types for dtypes *)
  type float16_elt = Bigarray.float16_elt
  type float32_elt = Bigarray.float32_elt
  type float64_elt = Bigarray.float64_elt
  type int8_elt = Bigarray.int8_signed_elt
  type uint8_elt = Bigarray.int8_unsigned_elt
  type int16_elt = Bigarray.int16_signed_elt
  type uint16_elt = Bigarray.int16_unsigned_elt
  type int32_elt = Bigarray.int32_elt
  type int64_elt = Bigarray.int64_elt
  type int_elt = Bigarray.int_elt
  type nativeint_elt = Bigarray.nativeint_elt
  type complex32_elt = Bigarray.complex32_elt
  type complex64_elt = Bigarray.complex64_elt

  type ('a, 'b) dtype = ('a, 'b) Dtype.t =
    | Float16 : (float, float16_elt) dtype
    | Float32 : (float, float32_elt) dtype
    | Float64 : (float, float64_elt) dtype
    | Int8 : (int, int8_elt) dtype
    | UInt8 : (int, uint8_elt) dtype
    | Int16 : (int, int16_elt) dtype
    | UInt16 : (int, uint16_elt) dtype
    | Int32 : (int32, int32_elt) dtype
    | Int64 : (int64, int64_elt) dtype
    | Int : (int, int_elt) dtype
    | NativeInt : (nativeint, nativeint_elt) dtype
    | Complex32 : (Complex.t, complex32_elt) dtype
    | Complex64 : (Complex.t, complex64_elt) dtype

  type float16_t = (float, float16_elt) t
  type float32_t = (float, float32_elt) t
  type float64_t = (float, float64_elt) t
  type int8_t = (int, int8_elt) t
  type uint8_t = (int, uint8_elt) t
  type int16_t = (int, int16_elt) t
  type uint16_t = (int, uint16_elt) t
  type int32_t = (int32, int32_elt) t
  type int64_t = (int64, int64_elt) t
  type std_int_t = (int, int_elt) t
  type std_nativeint_t = (nativeint, nativeint_elt) t
  type complex32_t = (Complex.t, complex32_elt) t
  type complex64_t = (Complex.t, complex64_elt) t

  (* Constructor shortcuts *)
  let float16 = Float16
  let float32 = Float32
  let float64 = Float64
  let int8 = Int8
  let uint8 = UInt8
  let int16 = Int16
  let uint16 = UInt16
  let int32 = Int32
  let int64 = Int64
  let int = Int
  let nativeint = NativeInt
  let complex32 = Complex32
  let complex64 = Complex64

  (* Index type for slicing *)
  type index =
    | I of int (* single index *)
    | L of int list (* list of indices *)
    | R of int list (* index range *)

  (* ───── Basic Tensor Properties ───── *)

  let unsafe_data x = B.data x
  let shape x = View.shape (B.view x)
  let dtype x = B.dtype x

  let strides x =
    let elem_strides = View.strides (B.view x) in
    let itemsize = Dtype.itemsize (B.dtype x) in
    Array.map (fun s -> s * itemsize) elem_strides

  let stride i x =
    let elem_stride = View.stride i (B.view x) in
    let itemsize = Dtype.itemsize (B.dtype x) in
    elem_stride * itemsize

  let dims x = View.shape (B.view x)
  let dim i x = View.dim i (B.view x)
  let ndim x = View.ndim (B.view x)
  let itemsize x = Dtype.itemsize (B.dtype x)
  let size x = View.numel (B.view x)
  let numel x = size x
  let nbytes x = numel x * itemsize x
  let offset x = View.offset (B.view x)
  let is_c_contiguous x = View.is_c_contiguous (B.view x)

  (* ───── Internal Utilities ───── *)

  (* Create a power of 2 for integer shift operations *)
  let power_of_two : type a b. (a, b) Dtype.t -> int -> a =
   fun dtype shift_val ->
    if shift_val < 0 then
      Error.check_bounds ~op:"power_of_two" ~name:"shift_val" ~value:shift_val
        ~min:0 ();
    match dtype with
    | Int8 | UInt8 | Int16 | UInt16 | Int | NativeInt -> (
        let power = 1 lsl shift_val in
        match dtype with
        | Int8 -> power
        | UInt8 -> power land 0xFF
        | Int16 -> power
        | UInt16 -> power land 0xFFFF
        | Int -> power
        | NativeInt -> Nativeint.shift_left Nativeint.one shift_val
        | _ -> Error.failed ~op:"power_of_two" ~what:"unreachable code path" ())
    | Int32 -> Int32.shift_left Int32.one shift_val
    | Int64 -> Int64.shift_left Int64.one shift_val
    | _ ->
        Error.invalid ~op:"power_of_two"
          ~what:(Printf.sprintf "dtype %s" (Dtype.to_string dtype))
          ~reason:"not an integer type"
          ~hint:
            "use Int8, UInt8, Int16, UInt16, Int32, Int64, Int, or NativeInt"
          ()

  let array_prod arr = Array.fold_left ( * ) 1 arr

  (** Integer ceiling division: (a + b - 1) / b for integers a, b where b > 0.
  *)
  let ceildiv a b =
    Error.check_bounds ~op:"ceildiv" ~name:"divisor" ~value:b ~min:1 ();
    (a + b - 1) / b

  (* Type checking helpers *)
  let ensure_float_dtype fname x =
    if not (Dtype.is_float (dtype x)) then
      Error.invalid ~op:fname
        ~what:(Printf.sprintf "dtype %s" (Dtype.to_string (dtype x)))
        ~reason:"expected float type (Float16, Float32, or Float64)" ()

  let ensure_int_dtype fname x =
    if not (Dtype.is_int (dtype x)) then
      Error.invalid ~op:fname ~what:"dtype" ~reason:"must be an integer type" ()

  (* Helper to convert tuple to array *)
  let pair_to_array (a, b) = [| a; b |]

  let resolve_axis ?ndim_opt x (axis_opt : int option) =
    let ndim = match ndim_opt with Some n -> n | None -> ndim x in
    match axis_opt with
    | None -> Array.init ndim Fun.id (* all axes *)
    | Some a ->
        let resolved_a = if a < 0 then a + ndim else a in
        [| resolved_a |]

  let resolve_single_axis ?ndim_opt x axis : int =
    let ndim = match ndim_opt with Some n -> n | None -> ndim x in
    if axis < 0 then axis + ndim else axis

  let reshape shape_spec x =
    let new_shape = Shape.resolve_neg_one (shape x) shape_spec in
    if shape x = new_shape then x else B.op_reshape x new_shape

  (* reshape and expand [x] to [new_shape] following numpy-style rules *)
  let broadcast_to new_shape x =
    let current_shape = shape x in
    if current_shape = new_shape then x
    else
      let rank_current = Array.length current_shape in
      let rank_new = Array.length new_shape in
      if rank_current > rank_new then
        Error.cannot ~op:"broadcast_to" ~what:"broadcast"
          ~from:
            (Printf.sprintf "%s (rank %d)"
               (Shape.to_string current_shape)
               rank_current)
          ~to_:
            (Printf.sprintf "%s (rank %d)" (Shape.to_string new_shape) rank_new)
          ~reason:(Printf.sprintf "rank mismatch: %d>%d" rank_current rank_new)
          ~hint:"target shape must have at least as many dimensions as source"
          ()
      else
        let padded_shape =
          if rank_current < rank_new then
            Array.append (Array.make (rank_new - rank_current) 1) current_shape
          else current_shape
        in
        let compatible = ref true in
        let first_incompatible = ref None in
        for i = 0 to rank_new - 1 do
          if not (padded_shape.(i) = new_shape.(i) || padded_shape.(i) = 1) then (
            compatible := false;
            if !first_incompatible = None then first_incompatible := Some i)
        done;
        if not !compatible then
          match !first_incompatible with
          | Some _ ->
              Error.broadcast_incompatible ~op:"broadcast_to"
                ~shape1:current_shape ~shape2:new_shape ()
          | None -> assert false
        else
          let x_reshaped =
            if padded_shape <> current_shape then reshape padded_shape x else x
          in
          B.op_expand x_reshaped new_shape

  (* return [x] and [y] broadcasted to a common shape *)
  let broadcasted ?(reverse = false) x y =
    let a, b = if reverse then (y, x) else (x, y) in
    let broadcast_shape = Shape.broadcast (shape a) (shape b) in
    let a_broad = broadcast_to broadcast_shape a in
    let b_broad = broadcast_to broadcast_shape b in
    (a_broad, b_broad)

  (* like [broadcast_to] but [-1] keeps the original dimension *)
  let expand shape_spec x =
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
    broadcast_to new_shape x

  let cast (type a b c d) (dt : (c, d) Dtype.t) (x : (a, b) t) : (c, d) t =
    match Dtype.equal_witness (dtype x) dt with
    | Some Equal ->
        (* Here the compiler now *knows* that [x] has type [(c,d) t], so this
           type-safe “no-op” copy type-checks. *)
        B.op_copy x
    | None -> B.op_cast x dt

  let astype dt x = cast dt x

  (* ───── Tensor Creation ───── *)

  let contiguous x = B.op_contiguous x
  let copy x = B.op_copy x

  let blit src dst =
    if shape src <> shape dst then
      Error.shape_mismatch ~op:"blit" ~expected:(shape dst) ~actual:(shape src)
        ~hint:"source and destination must have identical shapes" ();
    B.op_assign dst src

  let create ctx dtype shape arr =
    let n = Array.fold_left ( * ) 1 shape in
    if Array.length arr <> n then
      Error.invalid ~op:"create" ~what:"array size"
        ~reason:
          (Printf.sprintf "got %d elements, expected %d" (Array.length arr) n)
        ();

    (* Create bigarray buffer with proper dtype *)
    let kind = Dtype.to_bigarray_kind dtype in
    let bigarray = Bigarray.Array1.create kind Bigarray.c_layout n in

    (* Copy data from OCaml array to bigarray *)
    for i = 0 to n - 1 do
      Bigarray.Array1.unsafe_set bigarray i arr.(i)
    done;

    (* Create flat tensor and reshape if needed *)
    let tensor_1d = B.op_const_array ctx bigarray in
    if Array.length shape = 1 && shape.(0) = n then tensor_1d
    else B.op_reshape tensor_1d shape

  let init ctx dtype shape f =
    let size = Array.fold_left ( * ) 1 shape in

    (* Helper to convert linear index to multi-dimensional indices *)
    let unravel_index idx shape =
      let ndim = Array.length shape in
      let indices = Array.make ndim 0 in
      let remaining = ref idx in
      for i = 0 to ndim - 1 do
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
    let numel = array_prod shape_arr in
    let buf = B.op_buffer ctx dtype numel in
    reshape shape_arr buf

  let full ctx dt target_shape fill_value =
    let scalar_tensor = B.op_const_scalar ctx fill_value dt in
    if Array.length target_shape = 0 then scalar_tensor
    else if array_prod target_shape = 0 then empty ctx dt target_shape
    else
      let rank = Array.length target_shape in
      let intermediate_shape = Array.make rank 1 in
      let reshaped_scalar = reshape intermediate_shape scalar_tensor in
      expand target_shape reshaped_scalar

  let zeros ctx dtype shape_arr =
    let zero_val = Dtype.zero dtype in
    let result = full ctx dtype shape_arr zero_val in
    (* Ensure the result is contiguous for write operations *)
    contiguous result

  let ones ctx dtype shape_arr =
    let one_val = Dtype.one dtype in
    let result = full ctx dtype shape_arr one_val in
    (* Ensure the result is contiguous for write operations *)
    contiguous result

  let scalar_like x_ref value = scalar (B.context x_ref) (B.dtype x_ref) value

  (* Generic _like helper *)
  let create_like x_ref fill_fn =
    let dtype = B.dtype x_ref in
    let shape = shape x_ref in
    fill_fn (B.context x_ref) dtype shape

  let empty_like x_ref = create_like x_ref empty

  let full_like x_ref fill_value =
    create_like x_ref (fun ctx dt sh -> full ctx dt sh fill_value)

  let zeros_like x = full_like x (Dtype.zero (B.dtype x))
  let ones_like x = full_like x (Dtype.one (B.dtype x))

  let fill value x =
    let value_tensor = scalar_like x value in
    let value_broadcasted = broadcast_to (shape x) value_tensor in
    B.op_assign x value_broadcasted;
    x

  (* ───── Tensor Conversion ───── *)

  let unsafe_to_bigarray x =
    let t_contiguous = contiguous x in
    let array1 = unsafe_data t_contiguous in
    let ba =
      Bigarray.reshape (Bigarray.genarray_of_array1 array1) (shape t_contiguous)
    in
    ba

  let of_bigarray ctx ba =
    let size = Array.fold_left ( * ) 1 (Bigarray.Genarray.dims ba) in
    let arr = Bigarray.reshape_1 ba size in
    let shape = Bigarray.Genarray.dims ba in
    let flat_xensor = B.op_const_array ctx arr in
    reshape shape flat_xensor

  let unsafe_to_array x =
    let t_contiguous = contiguous x in
    let ba = unsafe_data t_contiguous in
    let n = numel t_contiguous in
    Array.init n (fun i -> Bigarray.Array1.get ba i)

  (* ───── Element-wise Binary Operations ───── *)

  let binop op a b =
    let a', b' = broadcasted a b in
    op a' b'

  let scalar_op op tensor scalar_val =
    let scalar_tensor = scalar_like tensor scalar_val in
    op tensor scalar_tensor

  (* Generic scalar binary operation helper *)
  let scalar_binop op tensor scalar_val =
    let scalar_tensor = scalar_like tensor scalar_val in
    op tensor scalar_tensor

  (* Generic reverse scalar operation helper *)
  let reverse_scalar_op op scalar_val tensor =
    let scalar_tensor = scalar_like tensor scalar_val in
    op scalar_tensor tensor

  let inplace_op op target value =
    let value_broadcasted = broadcast_to (shape target) value in
    let result = op target value_broadcasted in
    B.op_assign target result;
    target

  (* Generic in-place scalar operation helper *)
  let inplace_scalar_op op target scalar_val =
    let scalar_tensor = scalar_like target scalar_val in
    inplace_op op target scalar_tensor

  let add a b = binop B.op_add a b
  let add_s tensor scalar = scalar_op add tensor scalar
  let iadd target value = inplace_op B.op_add target value
  let radd_s tensor value = reverse_scalar_op add tensor value
  let iadd_s tensor value = inplace_scalar_op B.op_add tensor value

  let sub a b =
    let a', b' = broadcasted a b in
    let neg_b = B.op_neg b' in
    B.op_add a' neg_b

  let sub_s tensor_a scalar_b_val = scalar_binop sub tensor_a scalar_b_val
  let rsub_s tensor value = reverse_scalar_op sub tensor value

  let isub target_tensor value_tensor =
    let value_tensor_broadcasted =
      broadcast_to (shape target_tensor) value_tensor
    in
    let neg_value_tensor = B.op_neg value_tensor_broadcasted in
    let result = B.op_add target_tensor neg_value_tensor in
    B.op_assign target_tensor result;
    target_tensor

  let isub_s target_tensor scalar_val =
    let scalar_tensor = scalar_like target_tensor scalar_val in
    let neg_scalar = B.op_neg scalar_tensor in
    inplace_op B.op_add target_tensor neg_scalar

  let mul a b =
    let a', b' = broadcasted a b in
    B.op_mul a' b'

  let mul_s tensor_a scalar_b_val = scalar_binop mul tensor_a scalar_b_val
  let rmul_s tensor value = reverse_scalar_op mul tensor value

  let imul target_tensor value_tensor =
    let value_tensor_broadcasted =
      broadcast_to (shape target_tensor) value_tensor
    in
    let result = B.op_mul target_tensor value_tensor_broadcasted in
    B.op_assign target_tensor result;
    target_tensor

  let imul_s tensor value = inplace_scalar_op B.op_mul tensor value

  let div a b =
    let dt = dtype a in
    let a_b, b_b = broadcasted a b in
    match dt with
    | dt when Dtype.is_float dt || Dtype.is_complex dt ->
        (* True division for float/complex *)
        B.op_fdiv a_b b_b
    | dt when Dtype.is_int dt || Dtype.is_uint dt ->
        (* Integer division for integers *)
        B.op_idiv a_b b_b
    | _ ->
        (* should not happen *)
        failwith "Unsupported dtype for division"

  let div_s tensor_a scalar_b_val = scalar_binop div tensor_a scalar_b_val
  let rdiv_s tensor value = reverse_scalar_op div tensor value

  let idiv target value =
    let value_broadcasted = broadcast_to (shape target) value in
    let dt = dtype target in
    let result =
      match dt with
      | dt when Dtype.is_float dt || Dtype.is_complex dt ->
          B.op_fdiv target value_broadcasted
      | dt when Dtype.is_int dt || Dtype.is_uint dt ->
          B.op_idiv target value_broadcasted
      | _ ->
          Error.invalid ~op:"idiv"
            ~what:("dtype " ^ Dtype.to_string dt)
            ~reason:"not supported" ()
    in
    B.op_assign target result;
    target

  let idiv_s target scalar_val =
    let scalar_tensor = scalar_like target scalar_val in
    idiv target scalar_tensor

  let pow a b =
    let a', b' = broadcasted a b in
    B.op_pow a' b'

  let pow_s tensor_a scalar_b_val = scalar_binop pow tensor_a scalar_b_val
  let rpow_s tensor value = reverse_scalar_op pow tensor value

  let ipow target_tensor value_tensor =
    let value_tensor_broadcasted =
      broadcast_to (shape target_tensor) value_tensor
    in
    let result = B.op_pow target_tensor value_tensor_broadcasted in
    B.op_assign target_tensor result;
    target_tensor

  let ipow_s tensor value = inplace_scalar_op B.op_pow tensor value

  let maximum a b =
    let a', b' = broadcasted a b in
    B.op_max a' b'

  let maximum_s tensor_a scalar_b_val =
    scalar_binop maximum tensor_a scalar_b_val

  let rmaximum_s tensor value = reverse_scalar_op maximum tensor value

  let imaximum target_tensor value_tensor =
    let value_tensor_broadcasted =
      broadcast_to (shape target_tensor) value_tensor
    in
    let result = B.op_max target_tensor value_tensor_broadcasted in
    B.op_assign target_tensor result;
    target_tensor

  let imaximum_s tensor value = inplace_scalar_op B.op_max tensor value

  let minimum a b =
    let a', b' = broadcasted a b in
    let a_neg = B.op_neg a' in
    let b_neg = B.op_neg b' in
    let max_neg = B.op_max a_neg b_neg in
    B.op_neg max_neg

  let minimum_s tensor_a scalar_b_val =
    scalar_binop minimum tensor_a scalar_b_val

  let rminimum_s tensor value = reverse_scalar_op minimum tensor value

  let iminimum target_tensor value_tensor =
    let value_tensor_broadcasted =
      broadcast_to (shape target_tensor) value_tensor
    in
    let target_neg = B.op_neg target_tensor in
    let value_b_neg = B.op_neg value_tensor_broadcasted in
    let max_of_negs = B.op_max target_neg value_b_neg in
    let result = B.op_neg max_of_negs in
    B.op_assign target_tensor result;
    target_tensor

  let iminimum_s target_tensor scalar_val =
    let scalar_value_tensor = scalar_like target_tensor scalar_val in
    let scalar_broadcasted =
      broadcast_to (shape target_tensor) scalar_value_tensor
    in
    let target_neg = B.op_neg target_tensor in
    let scalar_b_neg = B.op_neg scalar_broadcasted in
    let max_of_negs = B.op_max target_neg scalar_b_neg in
    let result = B.op_neg max_of_negs in
    B.op_assign target_tensor result;
    target_tensor

  let mod_ a b =
    let a', b' = broadcasted a b in
    B.op_mod a' b'

  let mod_s tensor_a scalar_b_val = scalar_binop mod_ tensor_a scalar_b_val
  let rmod_s tensor value = reverse_scalar_op mod_ tensor value
  let imod target value = inplace_op B.op_mod target value
  let imod_s tensor value = inplace_scalar_op B.op_mod tensor value

  let bitwise_xor a b =
    let a', b' = broadcasted a b in
    B.op_xor a' b'

  let bitwise_or a b =
    let a', b' = broadcasted a b in
    B.op_or a' b'

  let bitwise_and a b =
    let a', b' = broadcasted a b in
    B.op_and a' b'

  (* ───── Logical Operations ───── *)

  let logical_and a b =
    let a_b, b_b = broadcasted a b in
    B.op_and a_b b_b

  let logical_or a b =
    let a_b, b_b = broadcasted a b in
    B.op_or a_b b_b

  let logical_xor a b =
    let a_b, b_b = broadcasted a b in
    B.op_xor a_b b_b

  let logical_not (type a b) (a : (a, b) t) =
    (* For boolean tensors (uint8), logical not is 1 - x *)
    (* But sub doesn't support uint8, so we use XOR with 1 *)
    let dt = dtype a in
    match dt with
    | Dtype.UInt8 ->
        let one_val = Dtype.one dt in
        let one_tensor = full (B.context a) dt (shape a) one_val in
        B.op_xor a one_tensor
    | Dtype.Float16 | Dtype.Float32 | Dtype.Float64 | Dtype.Int32 | Dtype.Int64
    | Dtype.Int8 | Dtype.Int16 | Dtype.UInt16 | Dtype.Int | Dtype.NativeInt
    | Dtype.Complex32 | Dtype.Complex64 ->
        let one_val = Dtype.one dt in
        let one_tensor = full (B.context a) dt (shape a) one_val in
        sub one_tensor a

  (* ───── Comparison Operations ───── *)

  let cmplt a b =
    let a', b' = broadcasted a b in
    B.op_cmplt a' b'

  let less a b = cmplt a b

  let cmpne a b =
    let a', b' = broadcasted a b in
    B.op_cmpne a' b'

  let not_equal a b = cmpne a b

  let cmpeq a b =
    let ne_result = cmpne a b in
    logical_not ne_result

  let equal a b = cmpeq a b
  let cmpgt a b = cmplt b a
  let greater a b = cmpgt a b
  let cmple a b = logical_not (cmpgt a b)
  let less_equal a b = cmple a b
  let cmpge a b = logical_not (cmplt a b)
  let greater_equal a b = cmpge a b

  (* ───── Element-wise Unary Operations ───── *)

  let neg x = B.op_neg x

  let bitwise_not x =
    let dt = dtype x in
    let minus_one_val = Dtype.minus_one dt in
    let minus_one_tensor = B.op_const_scalar (B.context x) minus_one_val dt in
    let minus_one_b = broadcast_to (shape x) minus_one_tensor in
    B.op_xor x minus_one_b

  let invert x = bitwise_not x

  (* Math functions - assume float inputs as per B.op signatures *)
  let log2 x = B.op_log2 x
  let exp2 x = B.op_exp2 x
  let sin x = B.op_sin x
  let sqrt x = B.op_sqrt x
  let recip x = B.op_recip x

  let log x =
    let log2_x = log2 x in
    (* todo: remove float here, it prevents complex *)
    let ln_2_val = Stdlib.log 2.0 in
    let dt = dtype x in
    let ln_2_tensor = B.op_const_scalar (B.context x) ln_2_val dt in
    let ln_2_b = broadcast_to (shape log2_x) ln_2_tensor in
    B.op_mul log2_x ln_2_b

  let exp x =
    (* todo: remove float here, it prevents complex *)
    let one_over_ln_2_val = 1.0 /. Stdlib.log 2.0 in
    let dt = dtype x in
    let factor_tensor = B.op_const_scalar (B.context x) one_over_ln_2_val dt in
    let factor_b = broadcast_to (shape x) factor_tensor in
    let x_scaled = B.op_mul x factor_b in
    B.op_exp2 x_scaled

  let cos x =
    (* todo: remove float here, it prevents complex *)
    let pi_half_val = Stdlib.acos 0.0 in
    let dt = dtype x in
    let pi_half_tensor = B.op_const_scalar (B.context x) pi_half_val dt in
    let pi_half_b = broadcast_to (shape x) pi_half_tensor in
    let arg_to_sin = sub pi_half_b x in
    B.op_sin arg_to_sin

  let tan x =
    let sin_x = sin x in
    let cos_x = cos x in
    B.op_fdiv sin_x cos_x

  let square x = mul x x

  let abs x =
    let dt = dtype x in
    if Dtype.is_uint dt then x
    else
      let zero_val = Dtype.zero dt in
      let zero_tensor = B.op_const_scalar (B.context x) zero_val dt in
      let zero_b = broadcast_to (shape x) zero_tensor in
      let cond = cmplt x zero_b in
      (* x < 0 *)
      let neg_x = neg x in
      B.op_where cond neg_x x

  let sign x =
    let dt = dtype x in
    let zero_val = Dtype.zero dt in
    let one_val = Dtype.one dt in
    if Dtype.is_uint dt then full_like x one_val
    else
      let zero_x = full_like x zero_val in
      let one_x = full_like x one_val in
      let minus_one_val = Dtype.minus_one dt in
      let minus_one_x = full_like x minus_one_val in

      let is_positive = cmpgt x zero_x in
      let is_negative = cmplt x zero_x in

      let result = B.op_where is_positive one_x zero_x in
      B.op_where is_negative minus_one_x result

  (* Activations & related *)
  let relu x = maximum x (zeros_like x)
  (* equivalent to (x > 0).where(x, 0) *)

  let sigmoid x =
    (* 1 / (1 + exp(-x)) = 1 / (1 + (exp2(-x / log(2)))) *)
    let dt = dtype x in
    let neg_one_over_log2 =
      B.op_const_scalar (B.context x) (-1.0 /. Stdlib.log 2.0) dt
    in
    let one_x = ones_like x in
    let exp_term = exp2 (mul x neg_one_over_log2) in
    recip (add one_x exp_term)

  let rsqrt x = recip (sqrt x)

  (* More trig and hyperbolic, assuming float inputs *)
  let poly_n_horner_coeffs_first x_tensor coeffs =
    (* coeffs are [P_N, P_{N-1}, ..., P_0] for P_N x^N + ... + P_0 *)
    match coeffs with
    | [] ->
        Error.invalid ~op:"poly_n_horner_coeffs_first" ~what:"coefficients"
          ~reason:"list is empty" ()
    | p_n :: ps_from_n_minus_1_to_0 ->
        let dt = dtype x_tensor in
        let acc = full (B.context x_tensor) dt (shape x_tensor) p_n in
        (* Initialize with P_N *)
        List.fold_left
          (fun current_acc p_i_val ->
            let p_i_tensor =
              full (B.context x_tensor) dt (shape x_tensor) p_i_val
            in
            add (mul current_acc x_tensor) p_i_tensor)
          acc ps_from_n_minus_1_to_0

  let asin x =
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
    let pi_half_x = full (B.context x) dt (shape x) (Stdlib.Float.pi /. 2.0) in
    let one_x = full (B.context x) dt (shape x) 1.0 in
    let abs_x = abs x in
    let term_sqrt = sqrt (sub one_x abs_x) in
    let poly_val = poly_n_horner_coeffs_first abs_x coeffs in
    let val_before_sign = sub pi_half_x (mul term_sqrt poly_val) in
    mul (sign x) val_before_sign

  let acos x =
    let dt = dtype x in
    let pi_half_x = full (B.context x) dt (shape x) (Stdlib.Float.pi /. 2.0) in
    sub pi_half_x (asin x)

  let atan x =
    (* (self / (1 + self * self).sqrt()).asin() *)
    let dt = dtype x in
    let one_x = full (B.context x) dt (shape x) 1.0 in
    let x_squared = square x in
    let denominator = sqrt (add one_x x_squared) in
    asin (div x denominator)

  let sinh x =
    (* (exp(x) - exp(-x)) / 2 *)
    let dt = dtype x in
    let two_x = full (B.context x) dt (shape x) 2.0 in
    let exp_x = exp x in
    let exp_neg_x = exp (neg x) in
    div (sub exp_x exp_neg_x) two_x

  let cosh x =
    (* (exp(x) + exp(-x)) / 2 *)
    let dt = dtype x in
    let two_x = full (B.context x) dt (shape x) 2.0 in
    let exp_x = exp x in
    let exp_neg_x = exp (neg x) in
    div (add exp_x exp_neg_x) two_x

  let tanh x =
    (* 2.0 * sigmoid(2.0 * x) - 1.0 *)
    let dt = dtype x in
    let one_x = full (B.context x) dt (shape x) 1.0 in
    let two_x = full (B.context x) dt (shape x) 2.0 in
    let sigmoid_arg = mul two_x x in
    let sigmoid_val = sigmoid sigmoid_arg in
    sub (mul two_x sigmoid_val) one_x

  let asinh x =
    (* log(x + sqrt(x^2 + 1)) *)
    let dt = dtype x in
    let one_x = full (B.context x) dt (shape x) 1.0 in
    let x_squared = square x in
    let sqrt_term = sqrt (add x_squared one_x) in
    log (add x sqrt_term)

  let acosh x =
    (* log(x + sqrt(x^2 - 1)) *)
    let dt = dtype x in
    let one_x = full (B.context x) dt (shape x) 1.0 in
    let x_squared = square x in
    let sqrt_term = sqrt (sub x_squared one_x) in
    log (add x sqrt_term)

  let atanh x =
    (* log((1+x)/(1-x)) / 2 *)
    let dt = dtype x in
    let one_x = full (B.context x) dt (shape x) 1.0 in
    let two_x = full (B.context x) dt (shape x) 2.0 in
    let term_plus = add one_x x in
    let term_minus = sub one_x x in
    div (log (div term_plus term_minus)) two_x

  (* Rounding, properties *)
  let trunc x =
    (* Cast to int (truncates), then cast back to float *)
    let original_dt = dtype x in
    (* todo: wtf *)
    cast original_dt (cast Dtype.int32 x)

  let ceil x =
    (* (x > trunc(x)).where(trunc(x)+1, trunc(x)) *)
    let dt = dtype x in
    let one_x = full (B.context x) dt (shape x) 1.0 in
    let trunc_x = trunc x in
    let cond = cmpgt x trunc_x in
    B.op_where cond (add trunc_x one_x) trunc_x

  let floor x =
    (* (x < trunc(x)).where(trunc(x)-1, trunc(x)) *)
    let dt = dtype x in
    let one_x = full (B.context x) dt (shape x) 1.0 in
    let trunc_x = trunc x in
    let cond = cmplt x trunc_x in
    B.op_where cond (sub trunc_x one_x) trunc_x

  (* Simplified round: round half away from zero. Python's `round` is half to
     even. *)
  let round x =
    (* sign(x) * floor(abs(x) + 0.5) *)
    let dt = dtype x in
    let half_x = full (B.context x) dt (shape x) 0.5 in
    let abs_x = abs x in
    let floor_term = floor (add abs_x half_x) in
    mul (sign x) floor_term

  let isinf x =
    let dt = dtype x in
    if not (Dtype.is_float dt) then zeros (B.context x) Dtype.uint8 (shape x)
    else
      let pos_inf_const = B.op_const_scalar (B.context x) Float.infinity dt in
      let neg_inf_const =
        B.op_const_scalar (B.context x) Float.neg_infinity dt
      in
      let is_pos_inf = cmpeq x (broadcast_to (shape x) pos_inf_const) in
      let is_neg_inf = cmpeq x (broadcast_to (shape x) neg_inf_const) in
      logical_or is_pos_inf is_neg_inf

  let isnan x =
    let dt = dtype x in
    if not (Dtype.is_float dt) then zeros (B.context x) Dtype.uint8 (shape x)
    else cmpne x x

  let isfinite x =
    let dt = dtype x in
    if not (Dtype.is_float dt) then ones (B.context x) Dtype.uint8 (shape x)
    else logical_not (logical_or (isinf x) (isnan x))

  let lerp start_tensor end_tensor weight =
    let end_minus_start = sub end_tensor start_tensor in
    let weighted_diff = mul end_minus_start weight in
    add start_tensor weighted_diff

  (* Scalar version of lerp weight *)
  let lerp_scalar_weight start_tensor end_tensor weight_val =
    let dt = dtype start_tensor in
    let weight_tensor =
      full (B.context start_tensor) dt (shape start_tensor) weight_val
    in
    lerp start_tensor end_tensor weight_tensor

  let lshift x shift_val =
    let dt = dtype x in
    if not (Dtype.is_int dt) then
      Error.invalid ~op:"lshift"
        ~what:("dtype " ^ Dtype.to_string dt)
        ~reason:"expected integer type" ();

    if shift_val < 0 then
      Error.check_bounds ~op:"lshift" ~name:"shift_val" ~value:shift_val ~min:0
        ();

    if shift_val = 0 then x
    else
      let factor_val = power_of_two dt shift_val in
      let factor_tensor = B.op_const_scalar (B.context x) factor_val dt in
      let factor_b = broadcast_to (shape x) factor_tensor in
      B.op_mul x factor_b

  let rshift x shift_val =
    let dt = dtype x in
    if not (Dtype.is_int dt) then
      Error.invalid ~op:"rshift"
        ~what:("dtype " ^ Dtype.to_string dt)
        ~reason:"expected integer type" ();

    if shift_val < 0 then
      Error.check_bounds ~op:"rshift" ~name:"shift_val" ~value:shift_val ~min:0
        ();

    if shift_val = 0 then x
    else
      let divisor_val = power_of_two dt shift_val in
      let divisor_tensor = B.op_const_scalar (B.context x) divisor_val dt in
      let divisor_b = broadcast_to (shape x) divisor_tensor in
      B.op_idiv x divisor_b

  let clamp ?min ?max x =
    let x_clamped_min =
      match min with
      | None -> x
      | Some min_v ->
          let min_x = full_like x min_v in
          maximum x min_x
    in
    match max with
    | None -> x_clamped_min
    | Some max_v ->
        let max_x = full_like x_clamped_min max_v in
        (* Use x_clamped_min's dtype *)
        minimum x_clamped_min max_x

  let clip = clamp

  (* ───── Ternary Operations ───── *)

  (* select between [if_true] and [if_false] based on [cond] *)
  let where cond if_true if_false =
    let s_true = shape if_true in
    let s_false = shape if_false in
    let s_cond = shape cond in
    (* Broadcast all three to a common shape. Order matters for shape inference.
       First, find common shape for if_true and if_false. *)
    let target_data_shape = Shape.broadcast s_true s_false in
    (* Then, find common shape for that and cond. *)
    let final_target_shape = Shape.broadcast target_data_shape s_cond in

    let cond_b = broadcast_to final_target_shape cond in
    let if_true_b = broadcast_to final_target_shape if_true in
    let if_false_b = broadcast_to final_target_shape if_false in
    B.op_where cond_b if_true_b if_false_b

  (* ───── Binary Mathematical Functions ───── *)

  (* Two-argument arctangent: atan2(y, x) returns angle in [-π, π] *)
  let atan2 y x =
    let y', x' = broadcasted y x in
    (* Use the identity: atan2(y, x) = atan(y/x) with quadrant correction *)
    let dt = dtype y' in

    (* Constants *)
    let zero = zeros_like y' in
    let pi = full (B.context y') dt (shape y') Float.pi in
    let pi_half = full (B.context y') dt (shape y') (Float.pi /. 2.0) in
    let neg_pi_half = full (B.context y') dt (shape y') (-.Float.pi /. 2.0) in

    (* Conditions *)
    let x_pos = cmpgt x' zero in
    let x_zero = cmpeq x' zero in
    let y_pos = cmpgt y' zero in
    let y_zero = cmpeq y' zero in
    let y_neg = cmplt y' zero in

    (* Basic atan(y/x) for x > 0 *)
    let ratio = div y' x' in
    let base_angle = atan ratio in

    (* Build result by composing conditions *)
    (* Start with default case: when x > 0, use atan(y/x) *)
    let result_1 = where x_pos base_angle zero in

    (* When x < 0 and y >= 0: atan(y/x) + π *)
    let x_neg_y_nonneg = logical_and (logical_not x_pos) (logical_not y_neg) in
    let x_neg_y_nonneg = logical_and x_neg_y_nonneg (logical_not x_zero) in
    let result_2 = where x_neg_y_nonneg (add base_angle pi) result_1 in

    (* When x < 0 and y < 0: atan(y/x) - π *)
    let x_neg_y_neg = logical_and (logical_not x_pos) y_neg in
    let x_neg_y_neg = logical_and x_neg_y_neg (logical_not x_zero) in
    let result_3 = where x_neg_y_neg (sub base_angle pi) result_2 in

    (* When x = 0 and y > 0: π/2 *)
    let x_zero_y_pos = logical_and x_zero y_pos in
    let result_4 = where x_zero_y_pos pi_half result_3 in

    (* When x = 0 and y < 0: -π/2 *)
    let x_zero_y_neg = logical_and x_zero y_neg in
    let result_5 = where x_zero_y_neg neg_pi_half result_4 in

    (* When x = 0 and y = 0: 0 (by convention) *)
    let both_zero = logical_and x_zero y_zero in
    where both_zero zero result_5

  (* Hypotenuse: sqrt(x² + y²) with overflow protection *)
  let hypot x y =
    let x', y' = broadcasted x y in
    let x_abs = abs x' in
    let y_abs = abs y' in

    (* Use the numerically stable formula: max * sqrt(1 + (min/max)²) *)
    let max_val = maximum x_abs y_abs in
    let min_val = minimum x_abs y_abs in

    (* Handle the case where both are zero *)
    let both_zero =
      logical_and
        (cmpeq x_abs (zeros_like x_abs))
        (cmpeq y_abs (zeros_like y_abs))
    in

    (* Avoid division by zero *)
    let ratio = where both_zero (zeros_like min_val) (div min_val max_val) in
    let ratio_sq = square ratio in
    let one = ones_like ratio_sq in
    let sqrt_term = sqrt (add one ratio_sq) in

    let result = mul max_val sqrt_term in
    where both_zero (zeros_like result) result

  (* ───── Reduction Operations ───── *)

  (* Generic reduction helper *)
  let reduce_op backend_op ?axes ?(keepdims = false) x =
    let rank = Array.length (shape x) in
    let axes_to_reduce =
      match axes with
      | None -> Array.init rank Fun.id
      | Some ax_list ->
          Array.map (fun ax -> if ax < 0 then ax + rank else ax) ax_list
    in
    backend_op ~axes:axes_to_reduce ~keepdims x

  let sum ?axes ?(keepdims = false) x =
    reduce_op B.op_reduce_sum ?axes ~keepdims x

  let max ?axes ?(keepdims = false) x =
    reduce_op B.op_reduce_max ?axes ~keepdims x

  let min ?axes ?(keepdims = false) x =
    neg (reduce_op B.op_reduce_max ?axes ~keepdims (neg x))

  let prod ?axes ?(keepdims = false) x =
    reduce_op B.op_reduce_prod ?axes ~keepdims x

  let mean ?axes ?(keepdims = false) x =
    let x_dtype = B.dtype x in
    let num_for_sum = sum ?axes ~keepdims x in

    let s_orig = shape x in
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
        array_prod
          (Array.map (fun ax_idx -> s_orig.(ax_idx)) actual_axes_to_reduce)
    in
    let num_elements_divisor_float =
      float_of_int
        (if num_elements_in_reduced_dims = 0 then 1
         else num_elements_in_reduced_dims)
    in

    let divisor_val_ocaml = Dtype.of_float x_dtype num_elements_divisor_float in
    let divisor_scalar = scalar (B.context x) x_dtype divisor_val_ocaml in
    let divisor_tensor = broadcast_to (shape num_for_sum) divisor_scalar in

    B.op_fdiv num_for_sum divisor_tensor

  let var ?axes ?(keepdims = false) ?(ddof = 0) x =
    let x_dtype = B.dtype x in
    let mean_x_keepdim_true = mean ?axes ~keepdims:true x in

    let diff = sub x mean_x_keepdim_true in
    let diff_sq = square diff in
    let sum_diff_sq = sum ?axes ~keepdims diff_sq in

    let s_orig = shape x in
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
        array_prod
          (Array.map (fun ax_idx -> s_orig.(ax_idx)) actual_axes_to_reduce)
    in

    let n_corrected_val = num_elements_in_reduced_dims - ddof in
    let n_corrected_float = float_of_int (Stdlib.max 0 n_corrected_val) in

    let divisor_val_ocaml = Dtype.of_float x_dtype n_corrected_float in
    let divisor_scalar = scalar (B.context x) x_dtype divisor_val_ocaml in
    let divisor_tensor = broadcast_to (shape sum_diff_sq) divisor_scalar in

    B.op_fdiv sum_diff_sq divisor_tensor

  let std ?axes ?(keepdims = false) ?(ddof = 0) x =
    let variance = var ?axes ~keepdims ~ddof x in
    sqrt variance

  (* Check if all elements are true (non-zero) *)
  let all ?axes ?(keepdims = false) x =
    let dt = dtype x in

    if Dtype.equal dt Dtype.uint8 then
      (* For boolean/uint8 tensors, we can use prod since 1*1*...*1 = 1 and
         1*1*...*0*...*1 = 0 *)
      let prod_val = prod ?axes ~keepdims x in
      (* Check if product is non-zero (which means all elements were 1) *)
      let zero_val = Dtype.zero dt in
      let zero_tensor = full_like prod_val zero_val in
      cmpne prod_val zero_tensor
    else
      (* For other numeric types, we check if min != 0 *)
      let min_val = min ?axes ~keepdims x in
      let zero_val = Dtype.zero dt in
      let zero_tensor = full_like min_val zero_val in
      cmpne min_val zero_tensor

  (* Check if any element is true (non-zero) *)
  let any ?axes ?(keepdims = false) x =
    let dt = dtype x in

    (* For any type, we check if max != 0 *)
    let max_val = max ?axes ~keepdims x in
    let zero_val = Dtype.zero dt in
    let zero_tensor = full_like max_val zero_val in
    cmpne max_val zero_tensor

  (* Check if two arrays are element-wise equal *)
  let array_equal x y =
    (* First, check if we can broadcast the shapes *)
    let can_broadcast =
      try
        let _ = Shape.broadcast (shape x) (shape y) in
        true
      with _ -> false
    in

    if not can_broadcast then
      (* If shapes can't be broadcast, arrays are not equal Return a scalar
         False (0) *)
      zeros (B.context x) Dtype.uint8 [||]
    else
      (* Check element-wise equality and then check if all are true *)
      let eq_result = equal x y in
      all eq_result (* Reduce over all axes to get scalar result *)

  (* ───── Shape Manipulation ───── *)

  let pad padding_config fill_value x = B.op_pad x padding_config fill_value
  let shrink shrink_args x = B.op_shrink x shrink_args

  (* collapse dimensions between [start_dim] and [end_dim] *)
  let flatten ?(start_dim = 0) ?(end_dim = -1) x =
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
      Error.invalid ~op:"flatten"
        ~what:(Printf.sprintf "start_dim %d or end_dim %d" start_dim end_dim)
        ~reason:(Printf.sprintf "out of bounds for rank %d" r)
        ();
    if s > e then
      Error.invalid ~op:"flatten" ~what:"dimensions"
        ~reason:"start_dim must be <= end_dim" ();

    let new_shape_list =
      if r = 0 then [ 1 ] (* Flatten scalar to shape [1] *)
      else if s = 0 && e = r - 1 then [ array_prod sh ] (* Flatten all to 1D *)
      else
        let pre = Array.to_list (Array.sub sh 0 s) in
        let mid_slice = Array.sub sh s (e - s + 1) in
        let mid_prod =
          if Array.length mid_slice = 0 then 1 else array_prod mid_slice
        in
        let post = Array.to_list (Array.sub sh (e + 1) (r - (e + 1))) in
        pre @ [ mid_prod ] @ post
    in
    reshape (Array.of_list new_shape_list) x

  let unflatten dim sizes x =
    let dim = resolve_single_axis x dim in
    let current_shape = shape x in
    let dim_size = current_shape.(dim) in

    (* Handle -1 in sizes (infer dimension) *)
    let sizes = Array.copy sizes in
    let neg_one_count =
      Array.fold_left (fun acc s -> if s = -1 then acc + 1 else acc) 0 sizes
    in

    if neg_one_count > 1 then
      Error.invalid ~op:"unflatten" ~what:"sizes"
        ~reason:"can only specify one unknown dimension (using -1)" ();

    if neg_one_count = 1 then (
      let known_product =
        Array.fold_left (fun acc s -> if s = -1 then acc else acc * s) 1 sizes
      in
      if known_product = 0 || dim_size mod known_product <> 0 then
        Error.cannot ~op:"unflatten" ~what:"infer dimension"
          ~from:(Printf.sprintf "total size %d" dim_size)
          ~to_:(Printf.sprintf "known product %d" known_product)
          ~reason:
            (Printf.sprintf "%d not divisible by %d" dim_size known_product)
          ~hint:"ensure total size is divisible by product of known dimensions"
          ();
      let inferred_size = dim_size / known_product in
      Array.iteri (fun i s -> if s = -1 then sizes.(i) <- inferred_size) sizes);

    (* Verify that product of sizes equals original dimension *)
    let sizes_product = Array.fold_left ( * ) 1 sizes in
    if sizes_product <> dim_size then
      Error.invalid ~op:"unflatten" ~what:"sizes"
        ~reason:
          (Printf.sprintf "product %d does not match dimension size %d"
             sizes_product dim_size)
        ();

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

    reshape new_shape x

  let ravel x = flatten x

  module IntSet = Set.Make (Int)

  (* drop axes of size 1; [axes] restricts which axes to squeeze *)
  let squeeze ?axes x =
    let sh = shape x in
    let r = Array.length sh in

    match axes with
    | None ->
        (* Squeeze all dimensions of size 1 *)
        let new_shape_list = List.filter (( <> ) 1) (Array.to_list sh) in
        let new_shape = Array.of_list new_shape_list in
        if Array.length new_shape = 0 && Array.length sh > 0 then
          reshape [||] x (* Result is scalar *)
        else if Array.length new_shape = 0 && Array.length sh = 0 then x
          (* scalar to scalar *)
        else reshape new_shape x
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
                Error.axis_out_of_bounds ~op:"squeeze" ~axis:ax ~ndim:r ();
              if seen.(ax) then
                Error.invalid ~op:"squeeze"
                  ~what:(Printf.sprintf "axis %d" ax)
                  ~reason:"duplicate axis" ();
              seen.(ax) <- true)
            normalized_axes;

          (* Check that all specified axes have size 1 *)
          Array.iter
            (fun ax ->
              if sh.(ax) <> 1 then
                Error.cannot ~op:"squeeze" ~what:"remove dimension"
                  ~from:(Printf.sprintf "axis %d (size %d)" ax sh.(ax))
                  ~to_:"squeezed"
                  ~reason:(Printf.sprintf "size %d≠1" sh.(ax))
                  ())
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
            reshape [||] x (* Result is scalar *)
          else if Array.length new_shape = 0 && Array.length sh = 0 then x
            (* scalar to scalar *)
          else reshape new_shape x

  (* insert size-1 dimensions at specified axes *)
  let unsqueeze ?axes x =
    let sh = shape x in
    let r = Array.length sh in

    let axes_arr =
      match axes with
      | None ->
          Error.invalid ~op:"unsqueeze" ~what:"axes" ~reason:"must be specified"
            ()
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
            Error.invalid ~op:"unsqueeze"
              ~what:(Printf.sprintf "axis %d" ax)
              ~reason:
                (Printf.sprintf "out of bounds for output rank %d" output_rank)
              ~hint:
                (Printf.sprintf "valid range is [%d, %d)" (-output_rank)
                   output_rank)
              ();
          if seen.(ax) then
            Error.invalid ~op:"unsqueeze"
              ~what:(Printf.sprintf "axis %d" ax)
              ~reason:"duplicate axis" ();
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
      reshape new_shape x

  (* For backward compatibility, you might want to add these helper
     functions: *)

  (* squeeze a single axis *)
  let squeeze_axis axis x = squeeze ~axes:[| axis |] x

  (* unsqueeze a single axis *)
  let unsqueeze_axis axis x = unsqueeze ~axes:[| axis |] x

  (* expand_dims is an alias for unsqueeze *)
  let expand_dims axes x = unsqueeze ~axes x

  let transpose ?axes x =
    let r = ndim x in
    let resolved_axes =
      match axes with
      | None -> Array.init r (fun i -> r - 1 - i) (* Reverse dimensions *)
      | Some ax_arr ->
          if Array.length ax_arr <> r then
            Error.invalid ~op:"transpose"
              ~what:(Printf.sprintf "axes (length %d)" (Array.length ax_arr))
              ~reason:
                (Printf.sprintf "expected rank %d, got %d" r
                   (Array.length ax_arr))
              ~hint:"provide exactly one axis per dimension" ();
          let seen = Array.make r false in
          Array.iter
            (fun ax_val ->
              let ax = if ax_val < 0 then ax_val + r else ax_val in
              if ax < 0 || ax >= r then
                Error.axis_out_of_bounds ~op:"transpose" ~axis:ax_val ~ndim:r ();
              if seen.(ax) then
                Error.invalid ~op:"transpose"
                  ~what:(Printf.sprintf "axis %d" ax_val)
                  ~reason:"repeated" ();
              seen.(ax) <- true)
            ax_arr;
          if not (Array.for_all Fun.id seen) then
            Error.invalid ~op:"transpose" ~what:"axes"
              ~reason:"do not form a permutation" ();
          Array.map
            (fun ax_val -> if ax_val < 0 then ax_val + r else ax_val)
            ax_arr
    in
    B.op_permute x resolved_axes

  let flip ?axes x =
    let r = ndim x in
    let flip_bools = Array.make r false in
    (match axes with
    | None -> Array.fill flip_bools 0 r true (* Flip all axes *)
    | Some ax_arr ->
        Array.iter
          (fun ax_val ->
            let ax = if ax_val < 0 then ax_val + r else ax_val in
            if ax < 0 || ax >= r then
              Error.axis_out_of_bounds ~op:"flip" ~axis:ax_val ~ndim:r ();
            flip_bools.(ax) <- true)
          ax_arr);
    B.op_flip x flip_bools

  let moveaxis src dst x =
    let r = ndim x in
    let norm_src = if src < 0 then src + r else src in
    let norm_dst = if dst < 0 then dst + r else dst in

    if norm_src < 0 || norm_src >= r || norm_dst < 0 || norm_dst >= r then
      Error.invalid ~op:"moveaxis"
        ~what:(Printf.sprintf "source %d or destination %d" src dst)
        ~reason:
          (Format.asprintf "out of bounds for shape %a" Shape.pp (shape x))
        ();

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
      B.op_permute x (Array.of_list final_axes_list)

  let swapaxes axis1 axis2 x =
    let r = ndim x in
    let norm_axis1 = if axis1 < 0 then axis1 + r else axis1 in
    let norm_axis2 = if axis2 < 0 then axis2 + r else axis2 in

    if norm_axis1 < 0 || norm_axis1 >= r || norm_axis2 < 0 || norm_axis2 >= r
    then
      Error.invalid ~op:"swapaxes"
        ~what:(Printf.sprintf "axes (%d, %d)" axis1 axis2)
        ~reason:
          (Format.asprintf "out of bounds for shape %a" Shape.pp (shape x))
        ();

    if norm_axis1 = norm_axis2 then x (* No change *)
    else
      let axes = Array.init r Fun.id in
      let temp = axes.(norm_axis1) in
      axes.(norm_axis1) <- axes.(norm_axis2);
      axes.(norm_axis2) <- temp;
      B.op_permute x axes

  let roll ?axis shift x =
    let original_shape = shape x in
    let x, ax_idx =
      match axis with
      | None ->
          let flat_x = flatten x in
          (* flatten handles rank 0 correctly for its own purpose *)
          (flat_x, 0)
      | Some specified_axis ->
          let r = ndim x in
          let norm_axis =
            if specified_axis < 0 then specified_axis + r else specified_axis
          in
          if norm_axis < 0 || norm_axis >= r then
            Error.axis_out_of_bounds ~op:"roll" ~axis:specified_axis ~ndim:r ();
          (x, norm_axis)
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
          if axis = None then reshape (shape x) x
          else x (* Reshape back if flattened and no-op roll *)
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
          let part1 = shrink ranges_part1 x in
          let part2 = shrink ranges_part2 x in
          let rolled_x = B.op_cat [ part1; part2 ] ax_idx in
          if axis = None then reshape original_shape rolled_x else rolled_x

  let tile reps x =
    let t_shape = shape x in
    let t_ndim = ndim x in
    let reps_len = Array.length reps in

    if reps_len < t_ndim then
      Error.invalid ~op:"tile" ~what:"reps length"
        ~reason:"must be >= tensor rank" ();

    (* If reps has more dimensions than x, prepend 1s to x's shape *)
    let x_promoted, promoted_shape =
      if reps_len > t_ndim then (
        let new_shape = Array.make reps_len 1 in
        Array.blit t_shape 0 new_shape (reps_len - t_ndim) t_ndim;
        (reshape new_shape x, new_shape))
      else (x, t_shape)
    in

    Array.iteri
      (fun i r ->
        if r < 0 then
          Error.invalid ~op:"tile"
            ~what:(Printf.sprintf "reps[%d]" i)
            ~reason:(Printf.sprintf "negative (%d<0)" r)
            ~hint:"use positive integers (or 0 for empty result)" ())
      reps;

    if Array.for_all (( = ) 1) reps then
      B.op_copy x_promoted (* optimization: no tiling needed *)
    else if Array.exists (( = ) 0) reps || Array.exists (( = ) 0) promoted_shape
    then
      (* If any rep is 0, or original shape has a 0, the tiled dimension becomes
         0 *)
      let tiled_shape =
        Array.mapi (fun i s_i -> s_i * reps.(i)) promoted_shape
      in
      empty (B.context x) (dtype x) tiled_shape
    else
      (* Tile using concatenation along each axis *)
      let rec tile_axis curr_x axis =
        if axis >= reps_len then curr_x
        else if reps.(axis) = 1 then tile_axis curr_x (axis + 1)
        else
          (* Concatenate reps.(axis) copies along this axis *)
          let copies = List.init reps.(axis) (fun _ -> curr_x) in
          let concatenated = B.op_cat copies axis in
          tile_axis concatenated (axis + 1)
      in
      tile_axis x_promoted 0

  let repeat ?axis count x =
    if count < 0 then
      Error.check_bounds ~op:"repeat" ~name:"count" ~value:count ~min:0 ();

    let x, ax_idx_eff =
      match axis with
      | None ->
          let flat_x = flatten x in
          (flat_x, 0)
      | Some specified_axis ->
          let r = ndim x in
          let norm_axis =
            if specified_axis < 0 then specified_axis + r else specified_axis
          in
          if norm_axis < 0 || norm_axis >= r then
            Error.axis_out_of_bounds ~op:"repeat" ~axis:specified_axis ~ndim:r
              ();
          (x, norm_axis)
    in

    let t_shape = shape x in
    let t_ndim = ndim x in

    if count = 0 then (
      let new_s = Array.copy t_shape in
      if t_ndim > 0 then new_s.(ax_idx_eff) <- 0;
      let final_shape_if_flattened = if axis = None then [| 0 |] else new_s in
      empty (B.context x) (dtype x) final_shape_if_flattened)
    else if count = 1 then B.op_copy x
    else if t_ndim = 0 then
      let scalar_reshaped = reshape [| 1 |] x in
      let repeated = expand [| count |] scalar_reshaped in

      if axis = None then repeated else reshape (shape x) repeated
    else
      (* Repeat using concatenation of individual elements *)
      let axis_size = t_shape.(ax_idx_eff) in
      let slices = ref [] in

      (* Extract each element along the axis and repeat it *)
      for i = axis_size - 1 downto 0 do
        (* Get slice at position i *)
        let slice_ranges =
          Array.init t_ndim (fun dim ->
              if dim = ax_idx_eff then (i, i + 1) else (0, t_shape.(dim)))
        in
        let slice_view = B.op_shrink x slice_ranges in

        (* Repeat this slice count times *)
        for _ = 1 to count do
          slices := slice_view :: !slices
        done
      done;

      (* Concatenate all slices *)
      let result = B.op_cat !slices ax_idx_eff in

      if axis = None then result else result

  let concatenate ?axis ts =
    match ts with
    | [] ->
        Error.invalid ~op:"concatenate" ~what:"tensor list" ~reason:"empty"
          ~hint:"provide at least one tensor" ()
    | [ x ] -> copy x
    | _ ->
        let axis =
          match axis with
          | None ->
              (* Check all arrays have same dtype *)
              let first_dtype = dtype (List.hd ts) in
              List.iter
                (fun x ->
                  let x_dtype = dtype x in
                  if not (Dtype.equal first_dtype x_dtype) then
                    Error.dtype_mismatch ~op:"concatenate"
                      ~expected:(Dtype.to_string first_dtype)
                      ~actual:(Dtype.to_string x_dtype) ())
                (List.tl ts);

              (* Flatten all arrays first *)
              let flattened = List.map flatten ts in
              B.op_cat flattened 0
          | Some a ->
              let first = List.hd ts in
              let first_ndim = ndim first in
              let axis = resolve_single_axis ~ndim_opt:first_ndim first a in

              (* Check all arrays have same dtype *)
              let first_dtype = dtype first in
              List.iter
                (fun x ->
                  let x_dtype = dtype x in
                  if not (Dtype.equal first_dtype x_dtype) then
                    Error.dtype_mismatch ~op:"concatenate"
                      ~expected:(Dtype.to_string first_dtype)
                      ~actual:(Dtype.to_string x_dtype) ())
                (List.tl ts);

              (* Check all arrays have same ndim *)
              if not (List.for_all (fun x -> ndim x = first_ndim) ts) then
                Error.invalid ~op:"concatenate" ~what:"arrays"
                  ~reason:"must have same number of dimensions" ();

              (* Check shapes match except on concatenation axis *)
              let first_shape = shape (List.hd ts) in
              List.iter
                (fun x ->
                  let t_shape = shape x in
                  Array.iteri
                    (fun i s ->
                      if i <> axis && s <> first_shape.(i) then
                        Error.invalid ~op:"concatenate"
                          ~what:(Printf.sprintf "dimension %d" i)
                          ~reason:
                            (Printf.sprintf "size %d≠%d" s first_shape.(i))
                          ())
                    t_shape)
                (List.tl ts);

              B.op_cat ts axis
        in
        axis

  let stack ?axis ts =
    match ts with
    | [] -> Error.empty_input ~op:"stack" ~what:"tensor list"
    | _ ->
        let first_shape = shape (List.hd ts) in
        let first_ndim = Array.length first_shape in

        (* Determine stacking axis *)
        let axis =
          match axis with
          | None -> 0
          | Some a ->
              let a = if a < 0 then a + first_ndim + 1 else a in
              if a < 0 || a > first_ndim then
                Error.axis_out_of_bounds ~op:"stack" ~axis:a ~ndim:first_ndim ();
              a
        in

        (* Add new dimension to each array *)
        let expanded = List.map (fun x -> unsqueeze ~axes:[| axis |] x) ts in

        (* Concatenate along the new axis *)
        concatenate ~axis expanded

  (* Helper to ensure arrays have at least n dimensions *)
  let ensure_ndim n x =
    let s = shape x in
    let nd = Array.length s in
    if nd >= n then x
    else
      let new_shape = Array.make n 1 in
      Array.blit s 0 new_shape 0 nd;
      reshape new_shape x

  let vstack ts =
    match ts with
    | [] -> Error.empty_input ~op:"vstack" ~what:"tensor list"
    | _ ->
        (* Make all arrays at least 2D *)
        let arrays_2d =
          List.map
            (fun x ->
              let nd = ndim x in
              if nd = 0 then reshape [| 1; 1 |] x
              else if nd = 1 then reshape [| 1; numel x |] x
              else x)
            ts
        in
        (* Concatenate along first axis *)
        concatenate ~axis:0 arrays_2d

  let hstack ts =
    match ts with
    | [] -> Error.empty_input ~op:"hstack" ~what:"tensor list"
    | _ ->
        (* Handle different dimensions *)
        let all_1d = List.for_all (fun x -> ndim x <= 1) ts in

        if all_1d then
          (* For 1D arrays, concatenate along axis 0 *)
          let arrays_1d =
            List.map (fun x -> if ndim x = 0 then reshape [| 1 |] x else x) ts
          in
          concatenate ~axis:0 arrays_1d
        else
          (* Make all arrays at least 2D *)
          let arrays_2d =
            List.map
              (fun x ->
                let nd = ndim x in
                if nd = 0 then reshape [| 1; 1 |] x
                else if nd = 1 then reshape [| numel x; 1 |] x
                else x)
              ts
          in

          (* Concatenate along second axis *)
          concatenate ~axis:1 arrays_2d

  let dstack ts =
    match ts with
    | [] -> Error.empty_input ~op:"dstack" ~what:"tensor list"
    | _ ->
        (* Make all arrays at least 3D *)
        let arrays_3d =
          List.map
            (fun x ->
              let s = shape x in
              let nd = Array.length s in
              if nd = 0 then reshape [| 1; 1; 1 |] x
              else if nd = 1 then reshape [| 1; s.(0); 1 |] x
              else if nd = 2 then reshape [| s.(0); s.(1); 1 |] x
              else x)
            ts
        in

        (* Concatenate along third axis *)
        concatenate ~axis:2 arrays_3d

  let broadcast_arrays ts =
    match ts with
    | [] -> []
    | [ x ] -> [ x ]
    | _ ->
        (* Find broadcast shape *)
        let broadcast_shape =
          List.fold_left
            (fun acc_shape x -> Shape.broadcast acc_shape (shape x))
            (shape (List.hd ts))
            (List.tl ts)
        in

        (* Broadcast all arrays to common shape *)
        List.map (fun x -> broadcast_to broadcast_shape x) ts

  let eye ctx ?m ?k dtype n =
    let rows = match m with Some v -> v | None -> n in
    let cols = n in
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
      let r_indices_col_vec = reshape [| rows; 1 |] r_arange in
      let r_indices = broadcast_to final_shape r_indices_col_vec in

      (* Create col indices: tensor([0, 1, ..., cols-1]) -> [[0, 1, ...,
         cols-1]] -> broadcasted *)
      let c_arange =
        init ctx Dtype.int32 [| cols |] (fun i -> Int32.of_int i.(0))
      in
      let c_indices_row_vec = reshape [| 1; cols |] c_arange in
      let c_indices = broadcast_to final_shape c_indices_row_vec in

      (* Condition: c_indices - r_indices = k_val *)
      let diff = sub c_indices r_indices in
      let k_tensor_scalar = scalar ctx Dtype.int32 (Int32.of_int k_val) in
      (* Create the diagonal mask *)
      let diag_mask = cmpeq diff k_tensor_scalar in

      let zeros_tensor = zeros ctx dtype final_shape in
      let one_val = Dtype.one dtype in
      let ones_fill = full_like zeros_tensor one_val in
      where diag_mask ones_fill zeros_tensor

  let identity ctx dtype n = eye ctx ~m:n ~k:0 dtype n

  let arange (type a b) ctx (dtype : (a, b) Dtype.t) start stop step =
    if start >= stop && step > 0 then
      Error.invalid ~op:"arange"
        ~what:(Printf.sprintf "range [%d, %d)" start stop)
        ~reason:(Printf.sprintf "empty with step=%d" step)
        ~hint:
          "ensure start < stop for positive step, or start > stop for negative \
           step"
        ();
    if step = 0 then
      Error.invalid ~op:"arange" ~what:"step" ~reason:"cannot be zero" ();
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
    if step_f = 0. then
      Error.invalid ~op:"arange_f" ~what:"step" ~reason:"cannot be zero" ();
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
    if count < 0 then
      Error.invalid ~op:"linspace"
        ~what:(Printf.sprintf "count %d" count)
        ~reason:"negative count" ~hint:"use count >= 0" ();
    if count = 0 then empty ctx dtype [| 0 |]
    else if count = 1 then full ctx dtype [| 1 |] (Dtype.of_float dtype start_f)
    else
      let div_factor = float_of_int (if endpoint then count - 1 else count) in
      let step_f = (stop_f -. start_f) /. div_factor in
      let f_init idx_arr =
        let i_f = float_of_int idx_arr.(0) in
        Dtype.of_float dtype (start_f +. (i_f *. step_f))
      in
      init ctx dtype [| count |] f_init

  let logspace ctx dtype ?(endpoint = true) ?(base = 10.0) start_exp_f
      stop_exp_f count =
    if count < 0 then
      Error.check_bounds ~op:"logspace" ~name:"count" ~value:count ~min:0 ();
    if count = 0 then empty ctx dtype [| 0 |]
    else
      (* The exponents should be generated with the same float precision as the
         final tensor type. *)
      let exponents_tensor =
        linspace ctx dtype ~endpoint start_exp_f stop_exp_f count
      in
      if base = Float.exp 1.0 then (* base is e *)
        exp exponents_tensor
      else if base = 2.0 then exp2 exponents_tensor
      else
        (* General case: base ** exponents = exp2(exponents * log2(base)) *)
        let log2_base = Stdlib.log base /. Stdlib.log 2.0 in
        let log2_base_tensor = scalar ctx dtype log2_base in
        (* Ensure log2_base_tensor is broadcastable with exponents_tensor *)
        let broadcasted_log2_base =
          broadcast_to (shape exponents_tensor) log2_base_tensor
        in
        let scaled_exponents = mul exponents_tensor broadcasted_log2_base in
        exp2 scaled_exponents

  let geomspace ctx dtype ?(endpoint = true) start_val_f stop_val_f count =
    if start_val_f <= 0. || stop_val_f <= 0. then
      Error.invalid ~op:"geomspace"
        ~what:
          (if start_val_f <= 0. then Printf.sprintf "start %g" start_val_f
           else Printf.sprintf "stop %g" stop_val_f)
        ~reason:"must be positive (>0)"
        ~hint:"geomspace requires positive values for logarithmic spacing" ();
    if count < 0 then
      Error.check_bounds ~op:"geomspace" ~name:"count" ~value:count ~min:0 ();
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
      exp log_points_tensor

  (* ───── Indexing and Slicing ───── *)

  (* Helper to normalize negative indices *)
  let normalize_index dim_size idx = if idx < 0 then dim_size + idx else idx

  (* Expand range specification with Python/NumPy-style exclusive stop *)
  let expand_range_spec dim_size = function
    | [] -> (0, dim_size - 1, 1)
    | [ start ] ->
        let start' = normalize_index dim_size start in
        (start', start', 1)
    | [ start; stop ] ->
        let start' = normalize_index dim_size start in
        let stop' = normalize_index dim_size stop in
        (* Make stop exclusive by subtracting 1 *)
        (* Always use step 1 when not specified - empty range if start >= stop *)
        (start', stop' - 1, 1)
    | [ start; stop; step ] ->
        if step = 0 then
          Error.invalid ~op:"slice" ~what:"step" ~reason:"cannot be zero"
            ~hint:
              "use positive step for forward slicing or negative for reverse"
            ()
        else
          let start' = normalize_index dim_size start in
          let stop' = normalize_index dim_size stop in
          (* Make stop exclusive based on step direction *)
          if step > 0 then (start', stop' - 1, step)
          else (start', stop' + 1, step)
    | _ ->
        Error.invalid ~op:"slice" ~what:"range"
          ~reason:"can have at most 3 elements" ()

  (* Convert index specification to list of indices *)
  let indices_of_spec dim_size = function
    | I idx ->
        let idx' = normalize_index dim_size idx in
        if idx' < 0 || idx' >= dim_size then
          Error.invalid ~op:"slice"
            ~what:(Printf.sprintf "index %d" idx)
            ~reason:
              (Printf.sprintf "out of bounds [%d, %d)"
                 (if idx < 0 then -dim_size else 0)
                 dim_size)
            ()
        else [ idx' ]
    | L indices ->
        List.map
          (fun idx ->
            let idx' = normalize_index dim_size idx in
            if idx' < 0 || idx' >= dim_size then
              Error.invalid ~op:"slice"
                ~what:(Printf.sprintf "index %d" idx)
                ~reason:
                  (Printf.sprintf "out of bounds [%d, %d)"
                     (if idx < 0 then -dim_size else 0)
                     dim_size)
                ()
            else idx')
          indices
    | R range ->
        let start, stop, step = expand_range_spec dim_size range in
        let rec collect acc i =
          if step > 0 then
            if i > stop then List.rev acc
            else if i >= dim_size then List.rev acc (* Out of bounds, stop *)
            else if i < 0 then collect acc (i + step)
              (* Skip negative, may become valid *)
            else collect (i :: acc) (i + step)
          else if
            (* step < 0 *)
            i < stop
          then List.rev acc
          else if i < 0 then List.rev acc (* Out of bounds, stop *)
          else if i >= dim_size then collect acc (i + step)
            (* Skip too large, may become valid *)
          else collect (i :: acc) (i + step)
        in
        collect [] start

  (* Efficient get_slice that minimizes tensor operations *)
  let slice slice_def x =
    let x_shape = shape x in
    let ndim = Array.length x_shape in

    (* Pad slice definition *)
    let full_slice =
      let n = List.length slice_def in
      if n > ndim then
        Error.invalid ~op:"slice" ~what:"indices"
          ~reason:(Printf.sprintf "too many (%d > %d)" n ndim)
          ()
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
          let is_c_contiguous =
            List.for_all
              (function
                | I _ -> true (* Single index is a contiguous range of size 1 *)
                | R [] | R [ _ ] | R [ _; _ ] -> true
                | R [ s; e; 1 ] -> s <= e
                | R [ s; e; -1 ] -> s >= e
                | _ -> false)
              slice
          in
          if is_c_contiguous then `ContiguousRanges else `Mixed
    in

    match analyze_pattern full_slice with
    | `Empty -> x
    | `AllSingles ->
        (* Direct element access *)
        let indices =
          List.mapi
            (fun i spec ->
              match spec with
              | I idx -> normalize_index x_shape.(i) idx
              | _ -> assert false)
            full_slice
        in
        let shrink_config =
          Array.of_list (List.mapi (fun _i idx -> (idx, idx + 1)) indices)
        in
        reshape [||] (shrink shrink_config x)
    | `ContiguousRanges ->
        (* Use shrink/flip operations only *)
        let rec apply_slices tensor dim = function
          | [] -> tensor
          | spec :: rest ->
              let tensor_ndim = Array.length (shape tensor) in
              if dim >= tensor_ndim then tensor
                (* No more dimensions to process *)
              else
                let dim_size = (shape tensor).(dim) in
                let tensor', next_dim =
                  match spec with
                  | I idx ->
                      (* Single index - shrink and squeeze that dimension *)
                      let idx' = normalize_index dim_size idx in
                      let config =
                        Array.init tensor_ndim (fun i ->
                            if i = dim then (idx', idx' + 1)
                            else (0, (shape tensor).(i)))
                      in
                      (squeeze ~axes:[| dim |] (shrink config tensor), dim)
                      (* Don't increment dim since we removed a dimension *)
                  | R [] -> (tensor, dim + 1) (* Take all *)
                  | R [ idx ] ->
                      let idx' = normalize_index dim_size idx in
                      let config =
                        Array.init tensor_ndim (fun i ->
                            if i = dim then (idx', idx' + 1)
                            else (0, (shape tensor).(i)))
                      in
                      (squeeze ~axes:[| dim |] (shrink config tensor), dim)
                      (* Don't increment dim since we removed a dimension *)
                  | R range ->
                      let start, stop, step =
                        expand_range_spec dim_size range
                      in
                      (* Check if the range is empty *)
                      let is_empty =
                        if step > 0 then start > stop else start < stop
                      in
                      if is_empty then (
                        (* Create empty tensor with 0 size in this dimension *)
                        let new_shape = Array.copy (shape tensor) in
                        new_shape.(dim) <- 0;
                        ( empty (B.context tensor) (dtype tensor) new_shape,
                          dim + 1 ))
                      else
                        let s, e =
                          (* expand_range_spec already returns exclusive stop *)
                          if step > 0 then (start, stop + 1)
                          else (stop, start + 1)
                        in
                        (* Clamp bounds to valid range *)
                        let s_clamped = Int.max 0 (Int.min s dim_size) in
                        let e_clamped = Int.max 0 (Int.min e dim_size) in
                        let config =
                          Array.init tensor_ndim (fun i ->
                              if i = dim then (s_clamped, e_clamped)
                              else (0, (shape tensor).(i)))
                        in
                        let sliced = shrink config tensor in
                        ( (if step < 0 then flip ~axes:[| dim |] sliced
                           else sliced),
                          dim + 1 )
                  | _ -> assert false
                in
                apply_slices tensor' next_dim rest
        in
        apply_slices x 0 full_slice
    | `Mixed ->
        (* Batch gather operations where possible *)
        let batch_process tensor processed_dims = function
          | [] -> tensor
          | specs ->
              (* Group consecutive gather operations *)
              let rec group_gathers acc current = function
                | [] ->
                    if current = [] then List.rev acc
                    else List.rev (List.rev current :: acc)
                | ((I _ | L _) as spec) :: rest ->
                    group_gathers acc (spec :: current) rest
                | spec :: rest ->
                    let acc' =
                      if current = [] then acc else List.rev current :: acc
                    in
                    group_gathers ([ spec ] :: acc') [] rest
              in

              let groups = group_gathers [] [] specs in

              (* Process each group *)
              let _, result =
                List.fold_left
                  (fun (current_dim, tensor) group ->
                    match group with
                    | [] -> (current_dim, tensor)
                    | R spec :: rest ->
                        (* Single range - use shrink *)
                        let dim_size = (shape tensor).(current_dim) in
                        let indices = indices_of_spec dim_size (R spec) in
                        let tensor' =
                          if List.length indices = dim_size then tensor
                          else if List.length indices = 0 then (
                            (* Empty slice - create tensor with 0 size in this
                               dimension *)
                            let new_shape = Array.copy (shape tensor) in
                            new_shape.(current_dim) <- 0;
                            empty (B.context tensor) (dtype tensor) new_shape)
                          else if List.length indices = 1 then
                            squeeze ~axes:[| current_dim |]
                              (shrink
                                 (Array.init
                                    (Array.length (shape tensor))
                                    (fun i ->
                                      if i = current_dim then
                                        (List.hd indices, List.hd indices + 1)
                                      else (0, (shape tensor).(i))))
                                 tensor)
                          else
                            (* Create index tensor and gather *)
                            (* For gather to work with multi-dimensional tensors, we need to
                             create an index tensor that matches the rank of the data tensor *)
                            let tensor_shape = shape tensor in
                            let idx_shape = Array.copy tensor_shape in
                            idx_shape.(current_dim) <- List.length indices;

                            let idx_tensor =
                              init (B.context x) Dtype.int32 idx_shape
                                (fun arr ->
                                  (* For the gather dimension, use the indices list *)
                                  (* For other dimensions, use the identity mapping *)
                                  if arr.(current_dim) < List.length indices
                                  then
                                    Int32.of_int
                                      (List.nth indices arr.(current_dim))
                                  else Int32.of_int arr.(current_dim))
                            in
                            B.op_gather tensor idx_tensor current_dim
                        in
                        (* R spec processes one dimension, plus any rest
                           specs *)
                        (current_dim + 1 + List.length rest, tensor')
                    | group ->
                        (* Multiple gather operations - fall back to sequential processing *)
                        (* Note: A more sophisticated batched gather operation could be implemented
                         in the future to handle multiple dimensions at once *)
                        let indices_lists =
                          List.mapi
                            (fun i spec ->
                              let dim_idx = current_dim + i in
                              indices_of_spec (shape tensor).(dim_idx) spec)
                            group
                        in

                        (* Process each gather operation sequentially *)
                        let tensor' =
                          List.fold_left2
                            (fun x _spec indices ->
                              if List.length indices = 1 then
                                squeeze ~axes:[| 0 |]
                                  (shrink
                                     [|
                                       (List.hd indices, List.hd indices + 1);
                                     |]
                                     x)
                              else
                                let idx_tensor =
                                  init (B.context x) Dtype.int32
                                    [| List.length indices |]
                                    (fun arr ->
                                      Int32.of_int (List.nth indices arr.(0)))
                                in
                                B.op_gather x idx_tensor 0)
                            tensor group indices_lists
                        in
                        (* This group processes as many dimensions as it has
                           specs *)
                        (current_dim + List.length group, tensor'))
                  (processed_dims, tensor) groups
              in
              result
        in
        batch_process x 0 full_slice

  (* Efficient set_slice using scatter operations *)
  let set_slice slice_def x y =
    let x_shape = shape x in
    let y_shape = shape y in
    let ndim = Array.length x_shape in

    (* Pad slice definition *)
    let full_slice =
      let n = List.length slice_def in
      if n > ndim then
        Error.invalid ~op:"slice" ~what:"indices"
          ~reason:(Printf.sprintf "too many (%d > %d)" n ndim)
          ()
      else slice_def @ List.init (ndim - n) (fun _ -> R [])
    in

    (* Get indices for each dimension *)
    let indices_per_dim =
      List.mapi (fun i spec -> indices_of_spec x_shape.(i) spec) full_slice
    in

    (* Check if this is scalar setting (all indices provided and all are
       single) *)
    let is_scalar_setting =
      List.length slice_def = ndim
      && List.for_all (function I _ -> true | _ -> false) slice_def
    in

    if is_scalar_setting then (
      (* Special case for scalar setting *)
      let indices =
        List.mapi
          (fun i spec ->
            match spec with
            | I idx -> normalize_index x_shape.(i) idx
            | _ -> assert false)
          slice_def
      in

      (* Verify y is scalar *)
      if y_shape <> [||] then
        Error.cannot ~op:"set_slice" ~what:"assign"
          ~from:(Shape.to_string y_shape) ~to_:"scalar position"
          ~reason:"value must be scalar (shape [])"
          ~hint:"use a scalar tensor or reshape to []" ();

      (* For JIT compatibility, use scatter instead of direct assignment *)
      (* Create index tensor with single element *)
      let linear_idx = ref 0 in
      let stride = ref 1 in
      for i = ndim - 1 downto 0 do
        let idx = if i < List.length indices then List.nth indices i else 0 in
        linear_idx := !linear_idx + (idx * !stride);
        stride := !stride * x_shape.(i)
      done;

      let scatter_indices =
        init (B.context x) Dtype.int32 [| 1 |] (fun _ ->
            Int32.of_int !linear_idx)
      in

      (* Flatten x and y for scatter *)
      let x_flat = reshape [| Array.fold_left ( * ) 1 x_shape |] x in
      let y_flat = reshape [| 1 |] y in

      (* Scatter and reshape back *)
      let result_flat = B.op_scatter x_flat scatter_indices y_flat 0 in
      let result = reshape x_shape result_flat in
      blit result x)
    else
      (* Verify shape for non-scalar case *)
      (* Compute expected shape, excluding dimensions with single indices (which get squeezed) *)
      let expected_shape_list =
        List.mapi
          (fun i indices ->
            match List.nth full_slice i with
            | I _ -> None (* Single index - dimension will be squeezed *)
            | _ -> Some (List.length indices))
          indices_per_dim
        |> List.filter_map (fun x -> x)
      in
      let expected_shape = Array.of_list expected_shape_list in
      if expected_shape <> y_shape then
        Error.shape_mismatch ~op:"set_slice" ~expected:expected_shape
          ~actual:y_shape ();

      (* Check if we can use optimized paths *)
      let all_contiguous =
        List.for_all2
          (fun (i, spec) indices ->
            match spec with
            | I idx ->
                List.length indices = 1
                && List.hd indices = normalize_index x_shape.(i) idx
            | R [ s; e ] | R [ s; e; 1 ] ->
                let s', e' =
                  (normalize_index x_shape.(i) s, normalize_index x_shape.(i) e)
                in
                List.length indices = Stdlib.abs (e' - s') + 1
            | R [] -> List.length indices = x_shape.(i)
            | _ -> false)
          (List.mapi (fun i spec -> (i, spec)) full_slice)
          indices_per_dim
      in

      if all_contiguous then
        (* Can use direct blit with proper slicing *)
        let x_slice_config =
          List.mapi
            (fun i spec ->
              match spec with
              | I idx ->
                  let idx' = normalize_index x_shape.(i) idx in
                  (idx', idx' + 1)
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

        let x_view = shrink (Array.of_list x_slice_config) x in
        (* If shapes are compatible for broadcasting, expand y to match
           x_view *)
        let y_for_blit =
          if shape x_view = y_shape then y
          else
            (* Try to broadcast y to x_view shape *)
            try broadcast_to (shape x_view) y
            with _ ->
              Error.broadcast_incompatible ~op:"set_slice" ~shape1:y_shape
                ~shape2:(shape x_view) ()
        in
        blit y_for_blit x_view
      else
        (* General case: build scatter indices *)
        let total_updates = Array.fold_left ( * ) 1 y_shape in

        (* Create flattened y *)
        let y_flat = reshape [| total_updates |] y in

        (* Create index tensor for scatter *)
        let scatter_indices =
          init (B.context x) Dtype.int32 [| total_updates |] (fun arr ->
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
        let x_flat = reshape [| Array.fold_left ( * ) 1 x_shape |] x in
        let result_flat = B.op_scatter x_flat scatter_indices y_flat 0 in
        let result = reshape x_shape result_flat in
        blit result x

  let slice_ranges ?(steps = []) starts stops x =
    let n_dims = List.length starts in
    if List.length stops <> n_dims then
      Error.invalid ~op:"slice_ranges" ~what:"dimensions"
        ~reason:"starts and stops must have same length" ();
    if steps <> [] && List.length steps <> n_dims then
      Error.invalid ~op:"slice_ranges" ~what:"steps"
        ~reason:"must have same length as starts/stops if provided" ();

    let slice_def =
      List.mapi
        (fun i (start, stop) ->
          let step = if steps = [] then 1 else List.nth steps i in
          R [ start; stop; step ])
        (List.combine starts stops)
    in
    slice slice_def x

  let set_slice_ranges ?(steps = []) starts stops x y =
    let n_dims = List.length starts in
    if List.length stops <> n_dims then
      Error.invalid ~op:"set_slice_ranges" ~what:"dimensions"
        ~reason:"starts and stops must have same length" ();
    if steps <> [] && List.length steps <> n_dims then
      Error.invalid ~op:"set_slice_ranges" ~what:"steps"
        ~reason:"must have same length as starts/stops if provided" ();

    let slice_def =
      List.mapi
        (fun i (start, stop) ->
          let step = if steps = [] then 1 else List.nth steps i in
          R [ start; stop; step ])
        (List.combine starts stops)
    in
    set_slice slice_def x y

  (* Get a single element or sub-tensor *)
  let get indices x =
    let x_shape = shape x in
    (* Check bounds for each index *)
    List.iteri
      (fun dim idx ->
        if dim >= Array.length x_shape then
          Error.invalid ~op:"get" ~what:"indices"
            ~reason:(Format.asprintf "too many for shape %a" Shape.pp x_shape)
            ()
        else if idx < 0 || idx >= x_shape.(dim) then
          Error.invalid ~op:"get"
            ~what:
              (Printf.sprintf "index [%s]"
                 (String.concat "," (List.map string_of_int indices)))
            ~reason:
              (Printf.sprintf "out of bounds for shape %s"
                 (Shape.to_string x_shape))
            ~hint:
              (Printf.sprintf "index %d at dim %d: %d ∉ [0, %d)" dim dim idx
                 x_shape.(dim))
            ())
      indices;

    slice (List.map (fun i -> I i) indices) x

  (* Set a single element or sub-tensor *)
  let set indices x value =
    let x_shape = shape x in
    (* Check bounds for each index *)
    List.iteri
      (fun dim idx ->
        if dim >= Array.length x_shape then
          Error.invalid ~op:"set" ~what:"indices"
            ~reason:(Format.asprintf "too many for shape %a" Shape.pp x_shape)
            ()
        else if idx < 0 || idx >= x_shape.(dim) then
          Error.invalid ~op:"set"
            ~what:(Printf.sprintf "index %d at dimension %d" idx dim)
            ~reason:
              (Format.asprintf "out of bounds for shape %a" Shape.pp x_shape)
            ~hint:
              (Printf.sprintf "index %d at dim %d: %d ∉ [0, %d)" dim dim idx
                 x_shape.(dim))
            ())
      indices;

    set_slice (List.map (fun i -> I i) indices) x value

  let unsafe_get indices x =
    let scalar_tensor = get indices x in
    let ba = unsafe_data scalar_tensor in
    (* For a scalar tensor (result of get), the data is already at index 0 *)
    if List.length indices = 0 && numel scalar_tensor = 1 then
      Bigarray.Array1.get ba 0
    else
      let view_offset = offset scalar_tensor in
      Bigarray.Array1.get ba view_offset

  let unsafe_set indices value x =
    let scalar_tensor = scalar (B.context x) (dtype x) value in
    set indices x scalar_tensor

  let array_split ~axis sections x =
    let ndim = ndim x in
    let axis = resolve_single_axis x axis in
    let axis_size = dim axis x in

    match sections with
    | `Indices indices ->
        (* Split at specific indices *)
        let indices = Array.of_list indices in
        let n_sections = Array.length indices + 1 in
        let splits = Array.make n_sections x in

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
                  if j = axis then R [ start; stop ] else R [])
            in
            splits.(i) <- slice slice_spec x
          else
            (* Empty slice *)
            let empty_shape = Array.copy (shape x) in
            empty_shape.(axis) <- 0;
            splits.(i) <- empty (B.context x) (dtype x) empty_shape
        done;
        Array.to_list splits
    | `Count n ->
        (* Split into n sections *)
        if n <= 0 then
          Error.check_bounds ~op:"array_split" ~name:"sections" ~value:n ~min:1
            ();

        let base_size = axis_size / n in
        let remainder = axis_size mod n in

        (* Calculate section sizes *)
        let sizes = Array.make n base_size in
        for i = 0 to remainder - 1 do
          sizes.(i) <- sizes.(i) + 1
        done;

        (* Create slices *)
        let splits = Array.make n x in
        let start = ref 0 in

        for i = 0 to n - 1 do
          let size = sizes.(i) in
          let stop = !start + size in

          let slice_spec =
            List.init ndim (fun j ->
                if j = axis then R [ !start; stop ] else R [])
          in
          splits.(i) <- slice slice_spec x;
          start := stop
        done;

        Array.to_list splits

  let split ~axis sections x =
    let axis = resolve_single_axis x axis in
    let axis_size = dim axis x in

    if axis_size mod sections <> 0 then
      Error.cannot ~op:"split" ~what:"divide evenly"
        ~from:(Printf.sprintf "axis %d (size %d)" axis axis_size)
        ~to_:(Printf.sprintf "%d sections" sections)
        ~reason:
          (Printf.sprintf "%d %% %d = %d" axis_size sections
             (axis_size mod sections))
        ~hint:"use array_split for uneven division" ();

    array_split ~axis (`Count sections) x

  (* ───── Random Number Generation ───── *)

  (* Validate parameters for random functions *)
  let validate_random_params fname dtype shape =
    if not (Dtype.is_float dtype) then
      Error.invalid ~op:fname
        ~what:(Printf.sprintf "dtype %s" (Dtype.to_string dtype))
        ~reason:"not a float type"
        ~hint:"rand/randn only support Float16, Float32, Float64" ();
    if Array.exists (fun x -> x < 0) shape then
      Error.invalid_shape ~op:fname ~shape
        ~reason:"dimensions must be non-negative" ()

  let rand ctx dtype ?(seed = 42) shape =
    validate_random_params "rand" dtype shape;

    (* If shape has 0, return zeros *)
    let numel = array_prod shape in
    if numel = 0 then zeros ctx dtype shape
    else
      (* Generate random int32 values using threefry *)
      (* Note: Current threefry implementation only returns one value per input,
         not two as originally expected *)
      let num_values = numel in

      (* Create counter tensors for threefry - offset by seed *)
      let counts0 = arange ctx Dtype.int32 seed (seed + num_values) 1 in
      let counts1 =
        arange ctx Dtype.int32 (seed + num_values) (seed + (2 * num_values)) 1
      in

      (* Generate random bits using threefry *)
      let random_bits = B.op_threefry counts0 counts1 in

      (* Flatten and take only what we need *)
      let bits_flat = flatten random_bits in
      let bits_needed =
        if numel < size bits_flat then shrink [| (0, numel) |] bits_flat
        else bits_flat
      in

      (* Convert to float64 for precision during normalization *)
      let bits_float64 = cast Dtype.float64 bits_needed in

      (* Add 2^31 to shift from signed [-2^31, 2^31-1] to unsigned [0, 2^32-1]
         range *)
      let offset = scalar ctx Dtype.float64 2147483648.0 in
      (* 2^31 *)
      let shifted = add bits_float64 offset in

      (* Normalize to [0, 1) by dividing by 2^32 *)
      let normalizer = scalar ctx Dtype.float64 4294967296.0 in
      (* 2^32 *)
      let normalized = div shifted normalizer in

      (* Cast to target dtype *)
      let result = cast dtype normalized in

      (* Reshape to final shape *)
      reshape shape result

  let randn ctx dtype ?(seed = 42) shape =
    validate_random_params "randn" dtype shape;

    (* If shape has 0, return zeros *)
    let numel = array_prod shape in
    if numel = 0 then zeros ctx dtype shape
    else
      (* Box-Muller transform: generate pairs of uniform random values *)
      (* Generate two sets of uniform random values *)
      let u1 = rand ctx Dtype.float32 ~seed shape in
      let u2 = rand ctx Dtype.float32 ~seed:(seed + numel) shape in

      (* Box-Muller transform: z0 = cos(2π * u1) * sqrt(-2 * ln(u2)) We use u2
         for the log to avoid log(0) *)

      (* Compute 2π * u1 *)
      let two_pi = scalar ctx Dtype.float32 (2.0 *. Float.pi) in
      let angle = mul u1 two_pi in

      (* Compute cos(2π * u1) *)
      let cos_part = cos angle in

      (* Compute sqrt(-2 * ln(u2)) *)
      (* First ensure u2 is not exactly 0 by using 1 - original_uniform *)
      let one = ones_like u2 in
      let u2_safe = sub one u2 in
      (* Now in [0, 1) *)

      (* Add small epsilon to avoid log(0) *)
      let eps = scalar ctx Dtype.float32 1e-7 in
      let u2_nonzero = maximum u2_safe eps in

      let log_u2 = log u2_nonzero in
      let neg_two = scalar ctx Dtype.float32 (-2.0) in
      let sqrt_arg = mul neg_two log_u2 in
      let sqrt_part = sqrt sqrt_arg in

      (* Combine: z0 = cos_part * sqrt_part *)
      let result_f32 = mul cos_part sqrt_part in

      (* Cast to target dtype *)
      cast dtype result_f32

  let randint ctx dtype ?(seed = 42) ?(high = 10) shape low =
    if not (Dtype.is_int dtype) then
      Error.invalid ~op:"randint" ~what:"dtype"
        ~reason:"only integer dtypes supported" ();
    if Array.exists (fun x -> x < 0) shape then
      Error.invalid_shape ~op:"randint" ~shape
        ~reason:"dimensions must be non-negative" ();

    (* Check range is valid *)
    if low >= high then
      Error.invalid ~op:"randint" ~what:"range"
        ~reason:(Printf.sprintf "low=%d ≥ high=%d" low high)
        ();

    (* If shape has 0, return zeros *)
    let numel = array_prod shape in
    if numel = 0 then zeros ctx dtype shape
    else
      (* Generate uniform random floats in [0, 1) *)
      let uniform = rand ctx Dtype.float32 ~seed shape in

      (* Scale to [0, high-low) *)
      let range = float_of_int (high - low) in
      let range_tensor = scalar ctx Dtype.float32 range in
      let scaled = mul uniform range_tensor in

      (* Shift to [low, high) *)
      let low_tensor = scalar ctx Dtype.float32 (float_of_int low) in
      let shifted = add scaled low_tensor in

      (* Floor to get integers (truncate towards negative infinity) *)
      let floored = floor shifted in

      (* Cast to target integer dtype *)
      cast dtype floored

  (* ───── Sorting and Searching ───── *)

  let sort (type a b) ?(descending = false) ?(axis = -1) (x : (a, b) t) =
    let axis = resolve_single_axis x axis in
    let ndim_x = ndim x in
    if axis < 0 || axis >= ndim_x then
      Error.axis_out_of_bounds ~op:"sort" ~axis ~ndim:ndim_x ();
    let orig_len = dim axis x in

    (* Handle edge case of empty or single element *)
    if orig_len <= 1 then
      let idx = arange (B.context x) Dtype.int32 0 orig_len 1 in
      let idx_shape =
        Array.init (ndim x) (fun i -> if i = axis then orig_len else 1)
      in
      let idx = reshape idx_shape idx in
      (x, idx)
    else
      (* Calculate number of stages for bitonic sort *)
      let n_stages =
        int_of_float (Float.ceil (Float.log2 (float_of_int orig_len)))
      in
      let padded_len = 1 lsl n_stages in

      (* Pad to power of 2 *)
      let fill_value =
        if descending then Dtype.min_value (dtype x)
        else
          (* Use a large value for ascending sort *)
          match dtype x with
          | dt when Dtype.is_float dt -> Dtype.of_float dt Float.infinity
          | Dtype.Int32 -> Int32.max_int
          | Dtype.Int64 -> Int64.max_int
          | dt -> Dtype.of_float dt 1e10 (* Fallback for other types *)
      in

      (* Handle NaN values by replacing them with infinity for sorting *)
      let x_for_sort =
        if Dtype.is_float (dtype x) then
          (* Replace NaN with inf (for ascending) or -inf (for descending) *)
          let is_nan = isnan x in
          let inf_val =
            if descending then
              full_like x (Dtype.of_float (dtype x) Float.neg_infinity)
            else full_like x (Dtype.of_float (dtype x) Float.infinity)
          in
          where is_nan inf_val x
        else x
      in

      let pad_config =
        Array.init (ndim x) (fun i ->
            if i = axis then (0, padded_len - orig_len) else (0, 0))
      in

      let x_pad = pad pad_config fill_value x_for_sort in

      (* Unflatten into binary tree structure *)
      let unflatten_sizes = Array.make n_stages 2 in
      let x_unflatten = unflatten axis unflatten_sizes x_pad in

      (* Bitonic sort implementation *)
      let x_ref = ref x_unflatten in
      for stage = 1 to n_stages do
        (* Handle crossover for all stages except the last *)
        (if stage <> n_stages then
           let crossover_dim = axis + n_stages - stage - 1 in

           (* Split along crossover dimension *)
           let boxes = split ~axis:crossover_dim 2 !x_ref in
           let blue_box = List.nth boxes 0 in
           let green_box = List.nth boxes 1 in

           (* Flip green box dimensions *)
           (* Tinygrad: flip_dims = tuple(-i for i in range(1, stage+1+(self.ndim-dim))) *)
           (* This means flip the last (stage + 1 + (ndim - axis)) dimensions *)
           let n_dims_to_flip = stage + 1 + (ndim x - axis) in
           let flip_axes =
             Array.init n_dims_to_flip (fun i -> ndim green_box - 1 - i)
             |> Array.to_list
             |> List.filter (fun i -> i >= 0 && i < ndim green_box)
             |> Array.of_list
           in
           let green_box_flipped = flip green_box ~axes:flip_axes in

           (* Reconstruct by concatenating *)
           x_ref :=
             concatenate ~axis:crossover_dim [ blue_box; green_box_flipped ]);

        (* Compare and swap substages *)
        for substage = stage - 1 downto 0 do
          let partner_dim = axis + n_stages - substage - 1 in

          (* Split along partner dimension *)
          let parts = split ~axis:partner_dim 2 !x_ref in
          let x_top = List.nth parts 0 in
          let x_bottom = List.nth parts 1 in

          (* Compare and order *)
          let x_larger = maximum x_top x_bottom in
          let x_smaller = minimum x_top x_bottom in

          (* Concatenate based on sort order *)
          x_ref :=
            if descending then
              concatenate ~axis:partner_dim [ x_larger; x_smaller ]
            else concatenate ~axis:partner_dim [ x_smaller; x_larger ]
        done;

        (* Undo crossover if needed *)
        if stage <> n_stages then
          let crossover_dim = axis + n_stages - stage - 1 in

          (* Split to undo crossover *)
          let boxes = split ~axis:crossover_dim 2 !x_ref in
          let blue_box = List.nth boxes 0 in
          let flipped_green_box = List.nth boxes 1 in

          (* Unflip - use the same flip_axes as before *)
          let n_dims_to_flip = stage + 1 + (ndim x - axis) in
          let flip_axes =
            Array.init n_dims_to_flip (fun i -> ndim flipped_green_box - 1 - i)
            |> Array.to_list
            |> List.filter (fun i -> i >= 0 && i < ndim flipped_green_box)
            |> Array.of_list
          in
          let green_box = flip ~axes:flip_axes flipped_green_box in

          x_ref := concatenate ~axis:crossover_dim [ blue_box; green_box ]
      done;

      (* Flatten back to original shape *)
      let x_sorted =
        flatten ~start_dim:axis ~end_dim:(axis + n_stages - 1) !x_ref
      in

      (* Remove padding *)
      let shrink_slice =
        List.init (ndim x_sorted) (fun i ->
            if i = axis then R [ 0; orig_len ] else R [])
      in
      let x_sorted = slice shrink_slice x_sorted in

      (* Compute indices for stable sort *)
      (* Create index tensor *)
      let idx = arange (B.context x) Dtype.int32 0 orig_len 1 in
      let idx_shape =
        Array.init (ndim x) (fun i -> if i = axis then orig_len else 1)
      in
      let idx = reshape idx_shape idx in
      let idx = expand (shape x_sorted) idx in

      (* Compute counts for handling duplicates *)
      let compute_counts tensor =
        (* Count how many elements <= current index with same value *)
        let t_exp_new = unsqueeze tensor ~axes:[| axis + 1 |] in
        let t_exp_orig = unsqueeze tensor ~axes:[| axis |] in
        let idx_exp_new = unsqueeze idx ~axes:[| axis + 1 |] in
        let idx_exp_orig = unsqueeze idx ~axes:[| axis |] in

        let le_mask = less_equal idx_exp_orig idx_exp_new in
        let eq_mask = equal t_exp_orig t_exp_new in
        let mask = logical_and le_mask eq_mask in
        sum mask ~axes:[| axis + 1 |] ~keepdims:false
      in

      let count_orig = compute_counts x_for_sort in
      let count_sorted = compute_counts x_sorted in

      (* Find where each original element ended up *)
      let self_exp = unsqueeze ~axes:[| axis + 1 |] x_for_sort in
      let sorted_exp = unsqueeze x_sorted ~axes:[| axis |] in
      let count_orig_exp = unsqueeze count_orig ~axes:[| axis + 1 |] in
      let count_sorted_exp = unsqueeze count_sorted ~axes:[| axis |] in
      let idx_exp = unsqueeze idx ~axes:[| axis + 1 |] in

      (* Match by value and count *)
      let value_match = equal self_exp sorted_exp in
      let count_match = equal count_orig_exp count_sorted_exp in
      let matches = logical_and value_match count_match in

      (* Extract indices where matches occur *)
      let matches_int = cast Dtype.int32 matches in
      let weighted_idx = mul matches_int idx_exp in
      let final_idx = sum weighted_idx ~axes:[| axis |] ~keepdims:false in

      (* Restore original NaN values in sorted output *)
      let x_sorted_final =
        if Dtype.is_float (dtype x) then
          (* Where x_sorted is inf and original x had NaN, restore NaN *)
          let nan_val =
            full_like x_sorted (Dtype.of_float (dtype x) Float.nan)
          in
          let is_inf =
            if descending then
              equal x_sorted
                (full_like x_sorted
                   (Dtype.of_float (dtype x) Float.neg_infinity))
            else
              equal x_sorted
                (full_like x_sorted (Dtype.of_float (dtype x) Float.infinity))
          in
          where is_inf nan_val x_sorted
        else x_sorted
      in

      (x_sorted_final, final_idx)

  let argsort ?(descending = false) ?(axis = -1) x =
    let _, indices = sort ~descending ~axis x in
    indices

  let argmax ?axis ?(keepdims = false) x =
    let t_ndim = ndim x in
    let reduction_axis =
      match axis with
      | None -> Array.init t_ndim Fun.id (* Flatten behavior: reduce all axes *)
      | Some ax -> [| resolve_single_axis ~ndim_opt:t_ndim x ax |]
    in
    let t_for_reduce = if axis = None then flatten x else x in
    let current_axis_idx = if axis = None then 0 else reduction_axis.(0) in
    let axis_len = dim current_axis_idx t_for_reduce in

    if axis_len = 0 then (* Edge case: empty dimension *)
      let out_shape =
        if keepdims then
          shape
            t_for_reduce (* Or shape of x if axis was None and x was scalar *)
        else
          Array.of_list
            (List.filteri
               (fun i _ -> i <> current_axis_idx)
               (Array.to_list (shape t_for_reduce)))
      in
      if Array.length out_shape = 0 && numel t_for_reduce > 0 then
        scalar (B.context x) Dtype.int32 0l (* scalar input *)
      else empty (B.context x) Dtype.int32 out_shape
    else
      let max_vals = max ~axes:reduction_axis ~keepdims:true t_for_reduce in
      let is_max_mask = equal t_for_reduce max_vals in

      (* Create reversed arange: [axis_len-1, axis_len-2, ..., 0] *)
      (* Tinygrad uses arange(N, 0, -1) which is [N-1, ..., 0] for N elements *)
      let arange_vals =
        arange (B.context x) Dtype.int32 (axis_len - 1) (-1) (-1)
      in

      (* Reshape arange_vals to be broadcastable for multiplication with
         is_max_mask *)
      let arange_shape = Array.make (ndim t_for_reduce) 1 in
      arange_shape.(current_axis_idx) <- axis_len;
      let arange_b = reshape arange_shape arange_vals in
      let arange_bc = broadcast_to (shape is_max_mask) arange_b in

      let masked_arange = mul (cast Dtype.int32 is_max_mask) arange_bc in

      (* Get the max of these values (effectively the first index from the
         end) *)
      let max_indices_from_end =
        max ~axes:reduction_axis ~keepdims:true masked_arange
      in

      (* Convert from "index from end" to "index from start" *)
      let axis_len_tensor =
        scalar (B.context x) Dtype.int32 (Int32.of_int (axis_len - 1))
      in
      let axis_len_bc =
        broadcast_to (shape max_indices_from_end) axis_len_tensor
      in
      let final_indices = sub axis_len_bc max_indices_from_end in

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
        reshape final_shape final_indices

  let argmin (type a b) ?axis ?(keepdims = false) (x : (a, b) t) :
      (int32, Dtype.int32_elt) t =
    let t_dtype = dtype x in
    let t_inverted =
      if Dtype.is_float t_dtype then neg x
      else if Dtype.is_int t_dtype && not (Dtype.is_uint t_dtype) then neg x
      else if Dtype.is_uint t_dtype then
        let max_val_specific : (a, b) t =
          match t_dtype with
          | Dtype.UInt8 -> scalar (B.context x) Dtype.uint8 255
          | Dtype.UInt16 -> scalar (B.context x) Dtype.uint16 65535
          | _ ->
              Error.failed ~op:"argmin"
                ~what:"unsupported uint dtype for inversion" ()
        in
        let max_val_b = broadcast_to (shape x) max_val_specific in
        sub max_val_b x
      else Error.failed ~op:"argmin" ~what:"unsupported dtype" ()
    in
    argmax ?axis ~keepdims t_inverted

  (* ───── Linear Algebra ───── *)

  let dot x_tensor w_tensor =
    let ndim_x = ndim x_tensor in
    let ndim_w = ndim w_tensor in

    if not (ndim_x > 0 && ndim_w > 0) then
      Error.invalid ~op:"dot" ~what:"tensors" ~reason:"both must be at least 1D"
        ();

    (* Handle special cases for 1D tensors *)
    match (ndim_x, ndim_w) with
    | 1, 1 ->
        (* 1D x 1D -> scalar (sum of element-wise product) *)
        let product = mul x_tensor w_tensor in
        sum product
    | 1, _ ->
        (* 1D x ND -> contract on first axis of w *)
        (* Reshape x to (1, k) and use matmul *)
        let x_2d = unsqueeze ~axes:[| 0 |] x_tensor in
        let result = B.op_matmul x_2d w_tensor in
        (* Result has shape (..., 1, n) -> squeeze the 1 *)
        squeeze ~axes:[| ndim result - 2 |] result
    | _, 1 ->
        (* ND x 1D -> contract on last axis of x *)
        (* Reshape w to (k, 1) and use matmul *)
        let w_2d = unsqueeze ~axes:[| 1 |] w_tensor in
        let result = B.op_matmul x_tensor w_2d in
        (* Result has shape (..., m, 1) -> squeeze the 1 *)
        squeeze ~axes:[| ndim result - 1 |] result
    | _ ->
        (* ND x ND -> use matmul directly *)
        B.op_matmul x_tensor w_tensor

  let matmul a_orig b_orig =
    let ndim_a_orig = ndim a_orig in
    let ndim_b_orig = ndim b_orig in

    if ndim_a_orig = 0 || ndim_b_orig = 0 then
      Error.invalid ~op:"matmul" ~what:"inputs"
        ~reason:"cannot be 0-D (scalars)" ();

    let a, b =
      if ndim_a_orig = 1 && ndim_b_orig = 1 then
        (* (k), (k) -> a becomes (1,k), b becomes (k,1) *)
        (unsqueeze ~axes:[| 0 |] a_orig, unsqueeze ~axes:[| 1 |] b_orig)
      else if ndim_a_orig = 1 then
        (* (k), (...,k,n) -> a becomes (1,k) *)
        (unsqueeze ~axes:[| 0 |] a_orig, b_orig)
      else if ndim_b_orig = 1 then
        (* (...,m,k), (k) -> b becomes (k,1) *)
        (a_orig, unsqueeze ~axes:[| 1 |] b_orig)
      else
        (* Both are >= 2D, no promotion needed for matmul semantics *)
        (a_orig, b_orig)
    in

    let result_intermediate = B.op_matmul a b in

    (* Squeeze the result if original inputs were 1D to match matmul
       semantics *)
    if ndim_a_orig = 1 && ndim_b_orig = 1 then
      (* Original (k) @ (k) -> result (1,1) from backend -> squeeze to scalar () *)
      squeeze result_intermediate
    else if ndim_a_orig = 1 then
      (* Original (k) @ (...,k,n) -> result (...,1,n) from backend -> squeeze first matrix dim *)
      squeeze ~axes:[| ndim result_intermediate - 2 |] result_intermediate
    else if ndim_b_orig = 1 then
      (* Original (...,m,k) @ (k) -> result (...,m,1) from backend -> squeeze last
         matrix dim *)
      squeeze ~axes:[| ndim result_intermediate - 1 |] result_intermediate
    else
      (* Both original inputs were >= 2D, result from backend is already
         (...,m,n) *)
      result_intermediate

  (* ───── Neural Network Operations ───── *)

  (* Activations *)

  (* ReLU6: min(max(x, 0), 6) *)
  let relu6 x =
    let zero = scalar_like x 0.0 in
    let six = scalar_like x 6.0 in
    let max_x = maximum x zero in
    minimum max_x six

  (* Hard Sigmoid: relu6(x + 3) / 6 *)
  let hard_sigmoid ?(alpha = 1.0 /. 6.0) ?(beta = 0.5) x =
    let dt = dtype x in
    let alpha_x = B.op_const_scalar (B.context x) alpha dt in
    let beta_x = B.op_const_scalar (B.context x) beta dt in
    let one_x = B.op_const_scalar (B.context x) 1.0 dt in

    let term1_arg = add (mul alpha_x x) beta_x in
    let term1 = relu term1_arg in

    let term2_arg = sub term1_arg one_x in
    let term2 = relu term2_arg in
    sub term1 term2

  (* Softplus: log(1 + exp(x)) *)
  let softplus x =
    let one = scalar_like x 1. in
    let exp_x = exp x in
    let sum = add one exp_x in
    log sum

  (* SiLU (Swish): x * sigmoid(x) *)
  let silu x =
    let sig_x = sigmoid x in
    mul x sig_x

  (* Hard SiLU: x * hard_sigmoid(x) *)
  let hard_silu x =
    let y = hard_sigmoid x in
    mul x y

  (* Log-Sigmoid: log(sigmoid(x)) *)
  let log_sigmoid x =
    let sig_x = sigmoid x in
    log sig_x

  (* Leaky ReLU: max(x, negative_slope * x) *)
  let leaky_relu ?(negative_slope = 0.01) x =
    let slope = scalar_like x negative_slope in
    let slope_x = mul slope x in
    maximum x slope_x

  (* Hard Tanh: max(-1, min(1, x)) *)
  let hard_tanh x =
    let one = scalar_like x 1. in
    let neg_one = scalar_like x (-1.0) in
    let min_x = minimum x one in
    maximum neg_one min_x

  (* Exponential Linear Unit (ELU): alpha * (exp(x) - 1) if x < 0, else x *)
  let elu ?(alpha = 1.0) x =
    let zero = scalar_like x 0.0 in
    let one = scalar_like x 1. in
    let alpha_scalar = scalar_like x alpha in
    let exp_x = exp x in
    let exp_minus_one = sub exp_x one in
    let min_part = minimum zero exp_minus_one in
    let alpha_min = mul alpha_scalar min_part in
    let max_x = maximum x zero in
    add max_x alpha_min

  (* Scaled Exponential Linear Unit (SELU): lambda * elu(x) if x < 0, else
     lambda * x *)
  let selu x =
    let alpha = 1.6732632423543772848170429916717 in
    let lambda = 1.0507009873554804934193349852946 in
    let elu_x = elu ~alpha x in
    let lambda_scalar = scalar_like x lambda in
    mul lambda_scalar elu_x

  (* Softmax: exp(x - max(x)) / sum(exp(x - max(x))) along specified axes *)
  let softmax ?(axes = [| -1 |]) x =
    let ndim = Array.length (shape x) in
    let axes = Array.map (fun ax -> if ax < 0 then ndim + ax else ax) axes in
    let max_x = max x ~axes ~keepdims:true in
    let x_shifted = sub x max_x in
    let exp_x = exp x_shifted in
    let sum_exp = sum exp_x ~axes ~keepdims:true in
    div exp_x sum_exp

  (* Approximated Gaussian Error Linear Unit: 0.5 * x * (1 + tanh(x *
     0.7978845608 * (1 + 0.044715 * x * x))) *)
  let gelu_approx x =
    let one = scalar_like x 1.0 in
    let half = scalar_like x 0.5 in
    let sqrt2_pi = scalar_like x 0.7978845608 in
    let coeff = scalar_like x 0.044715 in
    let x2 = mul x x in
    let inner = add one (mul coeff x2) in
    let arg = mul (mul x sqrt2_pi) inner in
    let y = tanh arg in
    mul half (mul x (add one y))

  (* Soft-sign: x / (|x| + 1)*)
  let softsign x =
    let one = scalar_like x 1.0 in
    let abs_x = maximum x (neg x) in
    div x (add one abs_x)

  (* Mish: x * tanh(softplus(x)) *)
  let mish x =
    let arg = softplus x in
    let y = tanh arg in
    mul x y








  let calculate_padding_for_mode input_spatial_shape ~k_s ~s_s ~d_s
      ~(mode : [< `Full | `Valid | `Same ])
      ~(op_type : [ `Convolution | `Correlation ]) =
    let num_spatial = Array.length input_spatial_shape in
    if
      not
        (Array.length k_s = num_spatial
        && Array.length s_s = num_spatial
        && Array.length d_s = num_spatial)
    then
      Error.invalid ~op:"calculate_padding_for_mode" ~what:"array lengths"
        ~reason:"shape/kernel/stride/dilation must have same length" ();

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
            (* For even kernels with odd total padding, convolution and
               correlation pad differently to match NumPy/SciPy behavior *)
            let pad_before, pad_after =
              if
                ks_d mod 2 = 0
                && total_pad_d mod 2 = 1
                && op_type = `Convolution
              then
                (* Convolution: pad more on top/left (before) *)
                ((total_pad_d / 2) + 1, total_pad_d / 2)
              else
                (* Correlation: pad more on bottom/right (after) - default
                   behavior *)
                (total_pad_d / 2, total_pad_d - (total_pad_d / 2))
            in
            (pad_before, pad_after))

  let correlate_nd_general ~groups stride_s_arr ~padding_mode dilation_s_arr
      ?fillvalue:_ num_spatial_dims ?bias ~op_type x w =
    if ndim w <> num_spatial_dims + 2 then
      Error.invalid ~op:"correlate_nd" ~what:"weight tensor"
        ~reason:(Printf.sprintf "must be %dD" (num_spatial_dims + 2))
        ();
    if ndim x <> num_spatial_dims + 2 then
      Error.invalid ~op:"correlate_nd" ~what:"input tensor"
        ~reason:(Printf.sprintf "must be %dD" (num_spatial_dims + 2))
        ();
    if Array.length stride_s_arr <> num_spatial_dims then
      Error.invalid ~op:"correlate_nd" ~what:"stride_s_arr length"
        ~reason:"mismatch with num_spatial_dims" ();
    if Array.length dilation_s_arr <> num_spatial_dims then
      Error.invalid ~op:"correlate_nd" ~what:"dilation_s_arr length"
        ~reason:"mismatch with num_spatial_dims" ();

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
    
    (* Handle empty input - if any spatial dimension is 0, return empty output *)
    if Array.exists (fun d -> d = 0) input_spatial_shape_arr then
      let empty_spatial = Array.make num_spatial_dims 0 in
      let empty_shape = Array.concat [ [| bs; cout |]; empty_spatial ] in
      empty (B.context x) (dtype x) empty_shape
    else begin

    if cin_total <> groups * cin_per_group then
      Error.invalid ~op:"correlate_nd"
        ~what:(Printf.sprintf "channel configuration")
        ~reason:(Printf.sprintf "%d ≠ %d×%d" cin_total groups cin_per_group)
        ~hint:
          (Printf.sprintf
             "expected %d channels for %d groups with %d channels each"
             (groups * cin_per_group) groups cin_per_group)
        ();
    let rcout = cout / groups in
    if groups * rcout <> cout then
      Error.invalid ~op:"correlate_nd"
        ~what:(Printf.sprintf "cout %d" cout)
        ~reason:(Printf.sprintf "%d %% %d ≠ 0" cout groups)
        ~hint:
          (Printf.sprintf
             "expected %d channels for %d groups with %d channels each" cout
             groups rcout)
        ();

    let padding_config_pairs_arr =
      calculate_padding_for_mode input_spatial_shape_arr
        ~k_s:kernel_spatial_shape_arr ~s_s:stride_s_arr ~d_s:dilation_s_arr
        ~mode:padding_mode ~op_type
    in

    (* Use unfold (im2col) for convolution *)
    let x_col = B.op_unfold x ~kernel_size:kernel_spatial_shape_arr 
                           ~stride:stride_s_arr ~dilation:dilation_s_arr 
                           ~padding:padding_config_pairs_arr in
    (* x_col shape: [bs, cin_total * kernel_elements, num_blocks] *)
    
    (* Calculate output spatial shape from the unfold result *)
    let x_col_shape = shape x_col in
    let num_blocks = x_col_shape.(2) in
    
    (* Calculate output spatial shape based on actual num_blocks *)
    let output_spatial_shape_arr = 
      if num_spatial_dims = 1 then
        [| num_blocks |]
      else if num_spatial_dims = 2 then
        (* For 2D, we need to figure out height and width from num_blocks *)
        let padded_h = input_spatial_shape_arr.(0) + (fst padding_config_pairs_arr.(0)) + (snd padding_config_pairs_arr.(0)) in
        let padded_w = input_spatial_shape_arr.(1) + (fst padding_config_pairs_arr.(1)) + (snd padding_config_pairs_arr.(1)) in
        let effective_kh = (kernel_spatial_shape_arr.(0) - 1) * dilation_s_arr.(0) + 1 in
        let effective_kw = (kernel_spatial_shape_arr.(1) - 1) * dilation_s_arr.(1) + 1 in
        let out_h = (padded_h - effective_kh) / stride_s_arr.(0) + 1 in
        let out_w = (padded_w - effective_kw) / stride_s_arr.(1) + 1 in
        [| out_h; out_w |]
      else
        (* For higher dimensions, calculate normally *)
        let padded_spatial = Array.init num_spatial_dims (fun i ->
          input_spatial_shape_arr.(i) + (fst padding_config_pairs_arr.(i)) + (snd padding_config_pairs_arr.(i))
        ) in
        Array.init num_spatial_dims (fun i ->
          let effective_kernel = (kernel_spatial_shape_arr.(i) - 1) * dilation_s_arr.(i) + 1 in
          (padded_spatial.(i) - effective_kernel) / stride_s_arr.(i) + 1
        )
    in
    
    (* Reshape weights for matrix multiplication *)
    let kernel_elements = Array.fold_left ( * ) 1 kernel_spatial_shape_arr in
    
    let result = if groups = 1 then
      (* Standard convolution: use direct matmul *)
      (* x_col: [bs, cin * kernel_elements, num_blocks] *)
      (* w: [cout, cin, *kernel] *)
      
      (* Reshape weights to [cout, cin * kernel_elements] *)
      let w_cont = B.op_contiguous w in
      let w_reshaped = reshape [| cout; cin_total * kernel_elements |] w_cont in
      
      (* Use matmul: w_reshaped @ x_col = [cout, cin*k] @ [cin*k, num_blocks] -> [cout, num_blocks] *)
      (* But we need to handle batch dimension, so use batched approach *)
      
      (* Add singleton dimension to weights for broadcasting *)
      let w_for_matmul = unsqueeze ~axes:[| 0 |] w_reshaped in (* [1, cout, cin*k] *)
      
      (* Transpose x_col for matmul: need [bs, cin*k, num_blocks] *)
      (* x_col is already [bs, cin*k, num_blocks] *)
      
      (* Use matmul with broadcasting: [1, cout, cin*k] @ [bs, cin*k, num_blocks] -> [bs, cout, num_blocks] *)
      let result = matmul w_for_matmul x_col in
      
      (* Remove the singleton batch dimension from result if it was added *)
      (* Actually matmul should broadcast correctly, so result is [bs, cout, num_blocks] *)
      result
    else
      (* Grouped convolution *)
      (* x_col: [bs, cin_total * kernel_elements, num_blocks] *)
      
      (* Reshape input to separate groups *)
      let x_col_grouped = reshape [| bs; groups; cin_per_group * kernel_elements; num_blocks |] x_col in
      
      (* Reshape weights to [groups, rcout, cin_per_group * kernel_elements] *)
      let w_cont = B.op_contiguous w in
      let w_grouped = reshape [| groups; rcout; cin_per_group * kernel_elements |] w_cont in
      
      (* Process groups using batched matmul *)
      (* Combine batch and group dimensions *)
      let x_col_batched = reshape [| bs * groups; cin_per_group * kernel_elements; num_blocks |] x_col_grouped in
      let w_batched = reshape [| groups; rcout; cin_per_group * kernel_elements |] w_grouped in
      
      (* Expand weights to match batch*groups dimension *)
      let w_expanded = unsqueeze ~axes:[| 0 |] w_batched in
      let w_expanded = expand [| bs; groups; rcout; cin_per_group * kernel_elements |] w_expanded in
      let w_expanded = reshape [| bs * groups; rcout; cin_per_group * kernel_elements |] w_expanded in
      
      (* Matmul: [bs*groups, rcout, cin_per_group*k] @ [bs*groups, cin_per_group*k, num_blocks] *)
      let result_batched = matmul w_expanded x_col_batched in
      
      (* Reshape back to separate batch and groups *)
      let result_grouped = reshape [| bs; groups; rcout; num_blocks |] result_batched in
      
      (* Merge groups and rcout to get final output channels *)
      reshape [| bs; cout; num_blocks |] result_grouped
    in
    
    (* Reshape result from [bs, cout, num_blocks] to [bs, cout, *output_spatial] *)
    let final_shape = Array.concat [ [| bs; cout |]; output_spatial_shape_arr ] in
    let result_reshaped = reshape final_shape result in

    match bias with
    | None -> result_reshaped
    | Some b ->
        let bias_shape =
          Array.concat [ [| 1; cout |]; Array.make num_spatial_dims 1 ]
        in
        let bias_reshaped = reshape bias_shape b in
        add result_reshaped bias_reshaped
    
    end (* Close the begin block for non-empty input case *)

  let correlate_nd ?(groups = 1) stride_s_arr
      ?(padding_mode : [ `Full | `Valid | `Same ] = `Valid) dilation_s_arr
      ?fillvalue num_spatial_dims ?bias x w =
    correlate_nd_general ~groups stride_s_arr ~padding_mode dilation_s_arr
      ?fillvalue num_spatial_dims ?bias ~op_type:`Correlation x w

  (** Correlate1D (cross-correlation). x: input tensor (bs, cin_total, iw) w:
      weight tensor (cout, cin_per_group, kw) bias: optional bias tensor (cout)
      stride, dilation: integers for the spatial dimension. padding_mode:
      [ `Full | `Valid | `Same ] fillvalue: optional scalar to fill padding.
      Defaults to 0 of x's dtype. *)
  let correlate1d ?groups ?(stride = 1) ?padding_mode ?(dilation = 1) ?fillvalue
      ?bias x w =
    correlate_nd ?groups [| stride |] ?padding_mode [| dilation |] ?fillvalue 1
      ?bias x w

  (** Correlate2D (cross-correlation). x: input tensor (bs, cin_total, ih, iw)
      w: weight tensor (cout, cin_per_group, kh, kw) bias: optional bias tensor
      (cout) stride, dilation: (int*int) tuples for (h,w) spatial dimensions.
      padding_mode: [ `Full | `Valid | `Same ] fillvalue: optional scalar to
      fill padding. Defaults to 0 of x's dtype. *)
  let correlate2d ?groups ?(stride = (1, 1)) ?padding_mode ?(dilation = (1, 1))
      ?fillvalue ?bias x w =
    correlate_nd ?groups (pair_to_array stride) ?padding_mode
      (pair_to_array dilation) ?fillvalue 2 ?bias x w

  (** ConvolveND - Generic N-Dimensional version. This flips the kernel
      (weights) along all its spatial dimensions then calls correlate_nd. *)
  let convolve_nd ?groups stride_s_arr ?padding_mode dilation_s_arr ?fillvalue
      num_spatial_dims ?bias x w =
    let w_ndim = ndim w in
    if w_ndim < num_spatial_dims + 2 then
      Error.invalid ~op:"convolve_nd" ~what:"weight tensor"
        ~reason:
          (Printf.sprintf "needs at least %d dims for spatial flipping"
             (num_spatial_dims + 2))
        ();

    (* Flip all spatial dimensions of w: dims from 2 up to (2 + num_spatial_dims
       - 1) *)
    let flip_axes_bools = Array.make w_ndim false in
    for i = 0 to num_spatial_dims - 1 do
      flip_axes_bools.(2 + i) <- true
    done;

    let w_flipped = B.op_flip w flip_axes_bools in
    (* Call correlate_nd_general directly with Convolution op_type *)
    let groups = Option.value groups ~default:1 in
    let padding_mode = Option.value padding_mode ~default:`Valid in
    correlate_nd_general ~groups stride_s_arr ~padding_mode dilation_s_arr
      ?fillvalue num_spatial_dims ?bias ~op_type:`Convolution x w_flipped

  (** Convolve1D. x: input tensor (bs, cin_total, iw) w: weight tensor (cout,
      cin_per_group, kw) *)
  let convolve1d ?groups ?(stride = 1) ?padding_mode ?(dilation = 1) ?fillvalue
      ?bias x w =
    convolve_nd ?groups [| stride |] ?padding_mode [| dilation |] ?fillvalue 1
      ?bias x w

  (** Convolve2D. x: input tensor (bs, cin_total, ih, iw) w: weight tensor
      (cout, cin_per_group, kh, kw) *)
  let convolve2d ?groups ?(stride = (1, 1)) ?padding_mode ?(dilation = (1, 1))
      ?fillvalue ?bias x w =
    convolve_nd ?groups (pair_to_array stride) ?padding_mode
      (pair_to_array dilation) ?fillvalue 2 ?bias x w

  (** Helper to resolve padding specification for pooling/convolution
      operations. Input `padding_spec` is user-facing. Output `(int*int) array`
      is for `B.op_pad`, (pad_before, pad_after) for each spatial dimension. *)
  let resolve_padding_for_ops padding_spec ~input_spatial_shape ~k_s ~s_s ~d_s
      ~op_type =
    match padding_spec with
    | `Same | `Valid | `Full ->
        calculate_padding_for_mode input_spatial_shape ~k_s ~s_s ~d_s
          ~mode:padding_spec ~op_type

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
      ~ceil_mode x =
    let x_ndim = ndim x in
    let input_spatial_shape =
      Array.sub (shape x) (x_ndim - num_spatial_dims) num_spatial_dims
    in
    let s_s = Option.value stride ~default:kernel_size in
    let d_s = Option.value dilation ~default:(Array.make num_spatial_dims 1) in

    let reg_pads =
      resolve_padding_for_ops padding_spec ~input_spatial_shape ~k_s:kernel_size
        ~s_s ~d_s ~op_type:`Convolution
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

  let avg_pool_nd ~kernel_size ?stride ?dilation ~padding_spec ~ceil_mode
      ~count_include_pad ~num_spatial_dims x =
    let x_ndim = ndim x in

    (* Use pool_setup helper *)
    let ( _input_spatial_shape,
          s_s,
          d_s,
          _current_pads_pairs,
          _reg_pads_pairs,
          full_pad_config ) =
      pool_setup ~num_spatial_dims ~kernel_size ?stride ?dilation ~padding_spec
        ~ceil_mode x
    in

    (* Use unfold for pooling *)
    let padding_pairs = Array.sub full_pad_config (x_ndim - num_spatial_dims) num_spatial_dims in
    let x_unfolded = B.op_unfold x ~kernel_size ~stride:s_s ~dilation:d_s ~padding:padding_pairs in
    (* x_unfolded shape: [batch..., channels * kernel_elements, num_blocks] *)
    
    (* Calculate the output shape *)
    let prefix_shape = Array.sub (shape x) 0 (x_ndim - num_spatial_dims) in
    let padded_spatial = Array.init num_spatial_dims (fun i ->
      (shape x).(x_ndim - num_spatial_dims + i) + (fst padding_pairs.(i)) + (snd padding_pairs.(i))
    ) in
    let output_spatial = Array.init num_spatial_dims (fun i ->
      let effective_kernel = (kernel_size.(i) - 1) * d_s.(i) + 1 in
      (padded_spatial.(i) - effective_kernel) / s_s.(i) + 1
    ) in
    
    let channels = if x_ndim - num_spatial_dims >= 1 then 
      let ch_prod = ref 1 in
      for i = 1 to x_ndim - num_spatial_dims - 1 do
        ch_prod := !ch_prod * (shape x).(i)
      done;
      !ch_prod
    else 1 in
    
    let kernel_elements = array_prod kernel_size in
    
    (* Reshape to separate channels and kernel elements *)
    let num_blocks = (shape x_unfolded).(Array.length (shape x_unfolded) - 1) in
    let x_reshaped = reshape (Array.concat [prefix_shape; [| channels; kernel_elements; num_blocks |]]) x_unfolded in
    
    (* Compute sum over kernel elements *)
    let sum_pooled = sum x_reshaped ~axes:[| Array.length (shape x_reshaped) - 2 |] in
    
    (* Reshape back to original layout *)
    let result_shape = Array.concat [prefix_shape; output_spatial] in
    let sum_reshaped = reshape result_shape sum_pooled in
    
    (* Compute divisor based on mode *)
    if count_include_pad && not ceil_mode then
      (* Simple case: divide by kernel size *)
      let kernel_numel = float_of_int kernel_elements in
      div_s sum_reshaped kernel_numel
    else
      (* Need to count valid elements - use same unfold approach on ones *)
      let ones = ones_like x in
      let ones_unfolded = B.op_unfold ones ~kernel_size ~stride:s_s ~dilation:d_s ~padding:padding_pairs in
      let ones_reshaped = reshape (Array.concat [prefix_shape; [| channels; kernel_elements; num_blocks |]]) ones_unfolded in
      let count = sum ones_reshaped ~axes:[| Array.length (shape ones_reshaped) - 2 |] in
      let count_reshaped = reshape result_shape count in
      div sum_reshaped count_reshaped

  let max_pool_nd ~kernel_size ?stride ?dilation ~padding_spec ~ceil_mode
      ~return_indices ~num_spatial_dims x =
    let x_ndim = ndim x in

    (* Use pool_setup helper *)
    let _input_spatial_shape, s_s, d_s, _current_pads_pairs, _, full_pad_config =
      pool_setup ~num_spatial_dims ~kernel_size ?stride ?dilation ~padding_spec
        ~ceil_mode x
    in

    (* Use unfold for pooling *)
    let padding_pairs = Array.sub full_pad_config (x_ndim - num_spatial_dims) num_spatial_dims in
    let x_unfolded = B.op_unfold x ~kernel_size ~stride:s_s ~dilation:d_s ~padding:padding_pairs in
    (* x_unfolded shape: [batch..., channels * kernel_elements, num_blocks] *)
    
    (* Calculate the output shape *)
    let prefix_shape = Array.sub (shape x) 0 (x_ndim - num_spatial_dims) in
    let padded_spatial = Array.init num_spatial_dims (fun i ->
      (shape x).(x_ndim - num_spatial_dims + i) + (fst padding_pairs.(i)) + (snd padding_pairs.(i))
    ) in
    let output_spatial = Array.init num_spatial_dims (fun i ->
      let effective_kernel = (kernel_size.(i) - 1) * d_s.(i) + 1 in
      (padded_spatial.(i) - effective_kernel) / s_s.(i) + 1
    ) in
    
    let channels = if x_ndim - num_spatial_dims >= 1 then 
      let ch_prod = ref 1 in
      for i = 1 to x_ndim - num_spatial_dims - 1 do
        ch_prod := !ch_prod * (shape x).(i)
      done;
      !ch_prod
    else 1 in
    
    let kernel_elements = array_prod kernel_size in
    
    (* Reshape to separate channels and kernel elements *)
    let num_blocks = (shape x_unfolded).(Array.length (shape x_unfolded) - 1) in
    let x_reshaped = reshape (Array.concat [prefix_shape; [| channels; kernel_elements; num_blocks |]]) x_unfolded in
    
    (* Compute max over kernel elements *)
    let max_pooled = B.op_reduce_max x_reshaped ~axes:[| Array.length (shape x_reshaped) - 2 |] ~keepdims:false in
    
    (* Reshape back to original layout *)
    let result_shape = Array.concat [prefix_shape; output_spatial] in
    let max_values = reshape result_shape max_pooled in

    if not return_indices then (max_values, None)
    else
      (* For indices, we need to track which element in the kernel was the max *)
      (* Create indices for each kernel position *)
      let kernel_indices = arange (B.context x) Dtype.int32 0 kernel_elements 1 in
      let kernel_indices_reshaped = reshape [| 1; 1; kernel_elements; 1 |] kernel_indices in
      let kernel_indices_broadcast = broadcast_to (shape x_reshaped) kernel_indices_reshaped in
      
      (* Find which kernel position has the max value *)
      let max_expanded = unsqueeze ~axes:[| Array.length (shape x_reshaped) - 2 |] max_pooled in
      let max_broadcast = broadcast_to (shape x_reshaped) max_expanded in
      let is_max = equal x_reshaped max_broadcast in
      
      (* Use where to select indices, with a large value for non-max positions *)
      let large_val = scalar (B.context x) Dtype.int32 (Int32.of_int kernel_elements) in
      let masked_indices = where is_max kernel_indices_broadcast (broadcast_to (shape kernel_indices_broadcast) large_val) in
      
      (* Get the minimum index (first occurrence of max) *)
      let kernel_idx = min masked_indices ~axes:[| Array.length (shape masked_indices) - 2 |] ~keepdims:false in
      
      (* Convert kernel index to spatial index *)
      (* For now, return the kernel indices directly - converting to spatial indices would require 
         more complex index arithmetic based on the spatial layout *)
      let final_indices = reshape result_shape kernel_idx in
      
      (max_values, Some final_indices)

  let avg_pool1d ~kernel_size ?stride ?dilation ?(padding_spec = `Valid)
      ?(ceil_mode = false) ?(count_include_pad = true) x =
    avg_pool_nd x ~kernel_size:[| kernel_size |]
      ?stride:(Option.map (fun s -> [| s |]) stride)
      ?dilation:(Option.map (fun d -> [| d |]) dilation)
      ~padding_spec ~ceil_mode ~count_include_pad ~num_spatial_dims:1

  let avg_pool2d ~kernel_size ?stride ?dilation ?(padding_spec = `Valid)
      ?(ceil_mode = false) ?(count_include_pad = true) x =
    avg_pool_nd x
      ~kernel_size:(pair_to_array kernel_size)
      ?stride:(Option.map pair_to_array stride)
      ?dilation:(Option.map pair_to_array dilation)
      ~padding_spec ~ceil_mode ~count_include_pad ~num_spatial_dims:2

  let max_pool1d ~kernel_size ?stride ?dilation ?(padding_spec = `Valid)
      ?(ceil_mode = false) ?(return_indices = false) x =
    max_pool_nd x ~kernel_size:[| kernel_size |]
      ?stride:(Option.map (fun s -> [| s |]) stride)
      ?dilation:(Option.map (fun d -> [| d |]) dilation)
      ~padding_spec ~ceil_mode ~return_indices ~num_spatial_dims:1

  let max_pool2d ~kernel_size ?stride ?dilation ?(padding_spec = `Valid)
      ?(ceil_mode = false) ?(return_indices = false) x =
    max_pool_nd x
      ~kernel_size:(pair_to_array kernel_size)
      ?stride:(Option.map pair_to_array stride)
      ?dilation:(Option.map pair_to_array dilation)
      ~padding_spec ~ceil_mode ~return_indices ~num_spatial_dims:2

  (** Helper for N-dim one-hot encoding. Creates a new last dimension for
      classes. *)
  let one_hot ~num_classes index_tensor =
    let index_dt = dtype index_tensor in
    if not (Dtype.is_int index_dt || Dtype.is_uint index_dt) then
      Error.invalid ~op:"one_hot"
        ~what:(Printf.sprintf "dtype %s" (Dtype.to_string index_dt))
        ~reason:"indices must be integer type" ();

    let index_expanded = unsqueeze index_tensor ~axes:[| ndim index_tensor |] in
    (* Add new last dim *)

    let arange_x = arange (B.context index_tensor) index_dt 0 num_classes 1 in
    (* Classes 0 to num_classes-1 *)

    (* Reshape arange to be (1, ..., 1, num_classes) to align with new last dim
       of index_expanded *)
    let ndim_expanded = ndim index_expanded in
    let shape_for_arange = Array.make ndim_expanded 1 in
    shape_for_arange.(ndim_expanded - 1) <- num_classes;
    let arange_b = reshape shape_for_arange arange_x in

    cmpeq index_expanded arange_b (* Broadcasts to one-hot mask *)

  (** Internal N-Dimensional max unpooling. *)
  let max_unpool_nd ~kernel_size ?stride ?dilation ~padding_spec
      ?output_size_opt ~num_spatial_dims input_x indices_x =
    let bs = dim 0 input_x in
    let c = dim 1 input_x in
    let pooled_spatial_shape = Array.sub (shape input_x) 2 num_spatial_dims in

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
              ~k_s:kernel_size ~s_s ~d_s ~op_type:`Convolution
          in
          Array.init num_spatial_dims (fun i ->
              let pooled_dim_size = pooled_spatial_shape.(i) in
              let k = kernel_size.(i) in
              let s = s_s.(i) in
              let d = d_s.(i) in
              let pb, pa = pads_pairs.(i) in
              ((pooled_dim_size - 1) * s) - pb - pa + ((d * (k - 1)) + 1))
    in
    let prod_output_spatial_size = array_prod output_spatial_shape in

    let one_hot_mask_for_indices =
      one_hot indices_x ~num_classes:prod_output_spatial_size
    in

    let input_expanded = unsqueeze input_x ~axes:[| ndim input_x |] in

    let multiplied = mul one_hot_mask_for_indices input_expanded in

    let sum_axes = Array.init num_spatial_dims (fun i -> 2 + i) in
    let result_flat_spatial = sum multiplied ~axes:sum_axes ~keepdims:false in

    let final_shape = Array.concat [ [| bs; c |]; output_spatial_shape ] in
    reshape final_shape result_flat_spatial

  let max_unpool1d input_x indices_x ~kernel_size ?stride ?dilation
      ?(padding_spec = `Valid) ?output_size_opt () =
    max_unpool_nd input_x indices_x ~kernel_size:[| kernel_size |]
      ?stride:(Option.map (fun s -> [| s |]) stride)
      ?dilation:(Option.map (fun d -> [| d |]) dilation)
      ~padding_spec ?output_size_opt ~num_spatial_dims:1

  let max_unpool2d input_x indices_x ~kernel_size ?stride ?dilation
      ?(padding_spec = `Valid) ?output_size_opt () =
    max_unpool_nd input_x indices_x
      ~kernel_size:(pair_to_array kernel_size)
      ?stride:(Option.map pair_to_array stride)
      ?dilation:(Option.map pair_to_array dilation)
      ~padding_spec ?output_size_opt ~num_spatial_dims:2

  (* ───── Display and Formatting ───── *)

  let pp_data (type a b) fmt (x : (a, b) t) =
    let open Format in
    let view = B.view x in
    let buffer = B.data x in
    let dtype = dtype x in
    let shape = View.shape view in
    let ndim = Array.length shape in
    let sz = View.numel view in

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
        let value = Bigarray.Array1.unsafe_get buffer (View.offset view) in
        pp_element fmt value
      else fprintf fmt "<empty scalar>"
    else
      let rec pp_slice fmt current_indices =
        let current_ndim = List.length current_indices in
        if current_ndim = ndim then
          let md_index = Array.of_list current_indices in
          let linear_offset =
            Shape.ravel_index md_index (View.strides view) + View.offset view
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

  (* Helper for formatter-based string conversion *)
  let format_to_string pp x =
    let buf = Stdlib.Buffer.create 1024 in
    let fmt = Format.formatter_of_buffer buf in
    pp fmt x;
    Format.pp_print_flush fmt ();
    Stdlib.Buffer.contents buf

  (* Helper for printing to stdout *)
  let print_with_formatter pp x =
    pp Format.std_formatter x;
    Format.pp_print_newline Format.std_formatter ();
    Format.pp_print_flush Format.std_formatter ()

  let data_to_string x = format_to_string pp_data x
  let print_data x = print_with_formatter pp_data x
  let pp_dtype fmt dtype = Format.fprintf fmt "%s" (Dtype.to_string dtype)
  let dtype_to_string dtype = Dtype.to_string dtype

  let shape_to_string shape =
    let shape_str =
      Array.map string_of_int shape |> Array.to_list |> String.concat "x"
    in
    Printf.sprintf "[%s]" shape_str

  let pp_shape fmt shape = Format.fprintf fmt "%s" (shape_to_string shape)

  let pp fmt x =
    let open Format in
    let view = B.view x in

    fprintf fmt "@[<v 0>";
    fprintf fmt "Nx Info:@,";
    fprintf fmt "  Shape: %a@," pp_shape (View.shape view);
    fprintf fmt "  Dtype: %a@," pp_dtype (dtype x);
    fprintf fmt "  Strides: [%s]@,"
      (String.concat "; "
         (Array.to_list (Array.map string_of_int (View.strides view))));
    fprintf fmt "  Offset: %d@," (View.offset view);
    fprintf fmt "  Size: %d@," (View.numel view);
    fprintf fmt "  Data: %a@," pp_data x

  let print x = print_with_formatter pp x
  let to_string x = format_to_string pp x

  (* ───── Higher-order functions ───── *)

  (* Map a function over all elements of a tensor *)
  let unsafe_map f x =
    let dt = dtype x in
    let sh = shape x in
    let result = empty (B.context x) dt sh in
    let data_src = unsafe_data (contiguous x) in
    let data_dst = unsafe_data result in
    let sz = size x in
    for i = 0 to sz - 1 do
      let v = Bigarray.Array1.unsafe_get data_src i in
      let v' = f v in
      Bigarray.Array1.unsafe_set data_dst i v'
    done;
    result

  (* Iterate a function over all elements of a tensor for side effects *)
  let unsafe_iter f x =
    let data_src = unsafe_data (contiguous x) in
    let sz = size x in
    for i = 0 to sz - 1 do
      let v = Bigarray.Array1.unsafe_get data_src i in
      f v
    done

  (* Fold a function over all elements of a tensor *)
  let unsafe_fold f init x =
    let data_src = unsafe_data (contiguous x) in
    let sz = size x in
    let acc = ref init in
    for i = 0 to sz - 1 do
      let v = Bigarray.Array1.unsafe_get data_src i in
      acc := f !acc v
    done;
    !acc

  (* Safe versions using backend operations - JAX semantics *)

  let map f x =
    let dt = dtype x in
    let sh = shape x in
    let result = empty (B.context x) dt sh in

    (* Process each element *)
    let total_size = size x in
    for i = 0 to total_size - 1 do
      let idx = Shape.unravel_index i sh |> Array.to_list in
      let v = get idx x in
      let v' = f v in
      set idx result v'
    done;
    result

  let iter f x =
    let sh = shape x in

    (* Process each element *)
    let total_size = size x in
    for i = 0 to total_size - 1 do
      let idx = Shape.unravel_index i sh |> Array.to_list in
      let v = get idx x in
      f v
    done

  let fold f init x =
    let sh = shape x in

    (* Process each element *)
    let total_size = size x in
    let acc = ref init in
    for i = 0 to total_size - 1 do
      let idx = Shape.unravel_index i sh |> Array.to_list in
      let v = get idx x in
      acc := f !acc v
    done;
    !acc
end
