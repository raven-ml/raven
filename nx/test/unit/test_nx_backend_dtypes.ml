(* Comprehensive data type coverage tests for all backend operations *)

open Alcotest
open Nx_core

module Make (Backend : Backend_intf.S) = struct
  (* Helper to create small test tensors *)
  let small_shape = [| 2; 3 |]
  let scalar_shape = [||]

  (* Helper to create tensors from arrays *)
  let create_tensor ctx dtype shape values =
    let size = Array.fold_left ( * ) 1 shape in
    let ba =
      Bigarray_ext.(
        Array1.create (Dtype.to_bigarray_ext_kind dtype) c_layout size)
    in
    Array.iteri (fun i v -> Bigarray_ext.Array1.set ba i v) values;
    let t = Backend.op_const_array ctx ba in
    if shape = [||] then t
    else Backend.op_reshape t (Symbolic_shape.of_ints shape)

  (* Helper to get shape from tensor *)
  let get_shape t =
    let view = Backend.view t in
    let symbolic_shape = Lazy_view.shape view in
    match Symbolic_shape.eval symbolic_shape with
    | Some shape -> shape
    | None -> failwith "Cannot evaluate symbolic shape"

  (* Helper to check if backend is Metal - check module name *)
  let is_metal_backend backend_name =
    String.lowercase_ascii backend_name = "metal"

  (* Helper to check if a dtype matches any in a packed list *)
  let dtype_in_list (type a b) (dtype : (a, b) Dtype.t)
      (packed_list : Dtype.packed list) : bool =
    List.exists (fun (Dtype.Pack dt) -> Dtype.equal dtype dt) packed_list

  (* Common dtype groups *)
  let complex_dtypes =
    [
      Dtype.Pack Dtype.Complex32;
      Dtype.Pack Dtype.Complex64;
      Dtype.Pack Dtype.Complex16;
    ]

  let float_dtypes =
    [
      Dtype.Pack Dtype.Float16;
      Dtype.Pack Dtype.Float32;
      Dtype.Pack Dtype.Float64;
      Dtype.Pack Dtype.BFloat16;
      Dtype.Pack Dtype.Float8_e4m3;
      Dtype.Pack Dtype.Float8_e5m2;
    ]

  let int_dtypes =
    [
      Dtype.Pack Dtype.Int8;
      Dtype.Pack Dtype.UInt8;
      Dtype.Pack Dtype.Int16;
      Dtype.Pack Dtype.UInt16;
      Dtype.Pack Dtype.Int32;
      Dtype.Pack Dtype.Int64;
      Dtype.Pack Dtype.Int;
      Dtype.Pack Dtype.NativeInt;
      Dtype.Pack Dtype.Int4;
      Dtype.Pack Dtype.UInt4;
      Dtype.Pack Dtype.QInt8;
      Dtype.Pack Dtype.QUInt8;
    ]

  let bool_dtype = [ Dtype.Pack Dtype.Bool ]

  (* Helper to check if dtype is supported on current backend *)
  let is_dtype_supported (type a b) backend_name (dtype : (a, b) Dtype.t) =
    if is_metal_backend backend_name then
      match dtype with
      (* Metal hardware supports these types *)
      | Dtype.Float16 -> true
      | Dtype.Float32 -> true
      | Dtype.Int8 -> true
      | Dtype.UInt8 -> true
      | Dtype.Int16 -> true
      | Dtype.UInt16 -> true
      | Dtype.Int32 -> true
      | Dtype.Int64 -> true
      | Dtype.BFloat16 -> true
      | Dtype.Bool -> true
      (* Metal does NOT support these types *)
      | Dtype.Float64 -> false (* no double precision *)
      | Dtype.Int -> false (* platform-specific sizes *)
      | Dtype.NativeInt -> false (* platform-specific sizes *)
      | Dtype.Complex32 -> false (* no complex *)
      | Dtype.Complex64 -> false (* no complex *)
      | Dtype.Complex16 -> false (* no complex *)
      | Dtype.Int4 -> false (* no 4-bit types *)
      | Dtype.UInt4 -> false (* no 4-bit types *)
      | Dtype.Float8_e4m3 -> false (* no 8-bit floats *)
      | Dtype.Float8_e5m2 -> false (* no 8-bit floats *)
      | Dtype.QInt8 -> false (* no quantized types *)
      | Dtype.QUInt8 -> false (* no quantized types *)
    else true

  (* Test all dtypes *)
  let all_dtypes = Dtype.all_dtypes

  (* Get appropriate test values for dtype *)
  let test_values : type a b. (a, b) Dtype.t -> a array = function
    | Dtype.Float16 -> [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
    | Dtype.Float32 -> [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
    | Dtype.Float64 -> [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
    | Dtype.BFloat16 -> [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
    | Dtype.Float8_e4m3 -> [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
    | Dtype.Float8_e5m2 -> [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
    | Dtype.Int8 -> [| 1; 2; 3; 4; 5; 6 |]
    | Dtype.UInt8 -> [| 1; 2; 3; 4; 5; 6 |]
    | Dtype.Int16 -> [| 1; 2; 3; 4; 5; 6 |]
    | Dtype.UInt16 -> [| 1; 2; 3; 4; 5; 6 |]
    | Dtype.Int -> [| 1; 2; 3; 4; 5; 6 |]
    | Dtype.UInt4 -> [| 1; 2; 3; 4; 5; 6 |]
    | Dtype.QInt8 -> [| 1; 2; 3; 4; 5; 6 |]
    | Dtype.QUInt8 -> [| 1; 2; 3; 4; 5; 6 |]
    | Dtype.Int4 -> [| -3; -1; 0; 1; 2; 3 |]
    | Dtype.Int32 -> [| 1l; 2l; 3l; 4l; 5l; 6l |]
    | Dtype.Int64 -> [| 1L; 2L; 3L; 4L; 5L; 6L |]
    | Dtype.NativeInt -> [| 1n; 2n; 3n; 4n; 5n; 6n |]
    | Dtype.Complex32 ->
        [|
          Complex.{ re = 1.0; im = 0.0 };
          Complex.{ re = 2.0; im = 1.0 };
          Complex.{ re = 3.0; im = 0.0 };
          Complex.{ re = 4.0; im = -1.0 };
          Complex.{ re = 5.0; im = 0.0 };
          Complex.{ re = 6.0; im = 2.0 };
        |]
    | Dtype.Complex64 ->
        [|
          Complex.{ re = 1.0; im = 0.0 };
          Complex.{ re = 2.0; im = 1.0 };
          Complex.{ re = 3.0; im = 0.0 };
          Complex.{ re = 4.0; im = -1.0 };
          Complex.{ re = 5.0; im = 0.0 };
          Complex.{ re = 6.0; im = 2.0 };
        |]
    | Dtype.Complex16 ->
        [|
          Complex.{ re = 1.0; im = 0.0 };
          Complex.{ re = 2.0; im = 1.0 };
          Complex.{ re = 3.0; im = 0.0 };
          Complex.{ re = 4.0; im = -1.0 };
          Complex.{ re = 5.0; im = 0.0 };
          Complex.{ re = 6.0; im = 2.0 };
        |]
    | Dtype.Bool -> [| true; false; true; false; true; false |]

  let get_one : type a b. (a, b) Dtype.t -> a = fun dtype -> Dtype.one dtype
  let get_zero : type a b. (a, b) Dtype.t -> a = fun dtype -> Dtype.zero dtype

  (* Binary operations tests *)
  let test_binary_op backend_name name op_fn dtype ctx () =
    if not (is_dtype_supported backend_name dtype) then skip ()
    else
      let values = test_values dtype in
      let a = create_tensor ctx dtype small_shape values in
      let b = create_tensor ctx dtype small_shape values in
      let result = op_fn a b in
      (* Just check shape is preserved *)
      check (array int) (name ^ " shape") small_shape (get_shape result)

  let test_add backend_name dtype =
    test_binary_op backend_name "add" Backend.op_add dtype

  let test_mul backend_name dtype =
    test_binary_op backend_name "mul" Backend.op_mul dtype

  let test_sub backend_name dtype ctx () =
    if not (is_dtype_supported backend_name dtype) then skip ()
    else
      let values = test_values dtype in
      let a = create_tensor ctx dtype small_shape values in
      let b = create_tensor ctx dtype small_shape values in
      (* sub = a + (-b) *)
      let neg_b = Backend.op_neg b in
      let result = Backend.op_add a neg_b in
      check (array int) "sub shape" small_shape (get_shape result)

  let test_div backend_name dtype ctx () =
    if not (is_dtype_supported backend_name dtype) then skip ()
    else if dtype_in_list dtype bool_dtype then
      skip () (* Division not meaningful for bool *)
    else if dtype_in_list dtype int_dtypes then
      let values = test_values dtype in
      let a = create_tensor ctx dtype small_shape values in
      let one = get_one dtype in
      let b = create_tensor ctx dtype small_shape (Array.make 6 one) in
      let result = Backend.op_idiv a b in
      check (array int) "idiv shape" small_shape (get_shape result)
    else
      let values = test_values dtype in
      let a = create_tensor ctx dtype small_shape values in
      let one = get_one dtype in
      let b = create_tensor ctx dtype small_shape (Array.make 6 one) in
      let result = Backend.op_fdiv a b in
      check (array int) "fdiv shape" small_shape (get_shape result)

  let test_pow backend_name dtype ctx () =
    if not (is_dtype_supported backend_name dtype) then skip ()
    else if dtype_in_list dtype (bool_dtype @ complex_dtypes) then skip ()
    else if is_metal_backend backend_name && dtype_in_list dtype int_dtypes then
      skip () (* Metal only has pow for float *)
    else
      let values = test_values dtype in
      let a = create_tensor ctx dtype small_shape values in
      let two = Dtype.two dtype in
      let b = create_tensor ctx dtype small_shape (Array.make 6 two) in
      let result = Backend.op_pow a b in
      check (array int) "pow shape" small_shape (get_shape result)

  let test_max backend_name dtype ctx () =
    if not (is_dtype_supported backend_name dtype) then skip ()
    else if dtype_in_list dtype complex_dtypes then skip ()
    else
      let values = test_values dtype in
      let a = create_tensor ctx dtype small_shape values in
      let b = create_tensor ctx dtype small_shape values in
      let result = Backend.op_max a b in
      check (array int) "max shape" small_shape (get_shape result)

  let test_mod backend_name dtype ctx () =
    if not (is_dtype_supported backend_name dtype) then skip ()
    else if dtype_in_list dtype (float_dtypes @ complex_dtypes @ bool_dtype)
    then skip ()
    else
      let values = test_values dtype in
      let a = create_tensor ctx dtype small_shape values in
      let two = Dtype.two dtype in
      let b = create_tensor ctx dtype small_shape (Array.make 6 two) in
      let result = Backend.op_mod a b in
      check (array int) "mod shape" small_shape (get_shape result)

  (* Comparison operations *)
  let test_cmplt backend_name dtype ctx () =
    if not (is_dtype_supported backend_name dtype) then skip ()
    else if dtype_in_list dtype complex_dtypes then skip ()
    else
      let values = test_values dtype in
      let a = create_tensor ctx dtype small_shape values in
      let b = create_tensor ctx dtype small_shape values in
      let result = Backend.op_cmplt a b in
      check (array int) "cmplt shape" small_shape (get_shape result)

  let test_cmpne backend_name dtype ctx () =
    if not (is_dtype_supported backend_name dtype) then skip ()
    else
      let values = test_values dtype in
      let a = create_tensor ctx dtype small_shape values in
      let b = create_tensor ctx dtype small_shape values in
      let result = Backend.op_cmpne a b in
      check (array int) "cmpne shape" small_shape (get_shape result)

  (* Bitwise operations *)
  let test_bitwise_op backend_name name op_fn dtype ctx () =
    if not (is_dtype_supported backend_name dtype) then skip ()
    else if dtype_in_list dtype (float_dtypes @ complex_dtypes) then skip ()
    else
      let values = test_values dtype in
      let a = create_tensor ctx dtype small_shape values in
      let b = create_tensor ctx dtype small_shape values in
      let result = op_fn a b in
      check (array int) (name ^ " shape") small_shape (get_shape result)

  let test_xor backend_name dtype =
    test_bitwise_op backend_name "xor" Backend.op_xor dtype

  let test_or backend_name dtype =
    test_bitwise_op backend_name "or" Backend.op_or dtype

  let test_and backend_name dtype =
    test_bitwise_op backend_name "and" Backend.op_and dtype

  (* Unary operations *)
  let test_neg backend_name dtype ctx () =
    if not (is_dtype_supported backend_name dtype) then skip ()
    else
      let values = test_values dtype in
      let a = create_tensor ctx dtype small_shape values in
      let result = Backend.op_neg a in
      check (array int) "neg shape" small_shape (get_shape result)

  let test_log2 backend_name dtype ctx () =
    if not (is_dtype_supported backend_name dtype) then skip ()
    else if dtype_in_list dtype (int_dtypes @ bool_dtype @ complex_dtypes) then
      skip ()
    else
      let one = get_one dtype in
      let a = create_tensor ctx dtype small_shape (Array.make 6 one) in
      let result = Backend.op_log2 a in
      check (array int) "log2 shape" small_shape (get_shape result)

  let test_exp2 backend_name dtype ctx () =
    if not (is_dtype_supported backend_name dtype) then skip ()
    else if dtype_in_list dtype (int_dtypes @ bool_dtype @ complex_dtypes) then
      skip ()
    else
      let one = get_one dtype in
      let a = create_tensor ctx dtype small_shape (Array.make 6 one) in
      let result = Backend.op_exp2 a in
      check (array int) "exp2 shape" small_shape (get_shape result)

  let test_sin backend_name dtype ctx () =
    if not (is_dtype_supported backend_name dtype) then skip ()
    else if dtype_in_list dtype (int_dtypes @ bool_dtype @ complex_dtypes) then
      skip ()
    else
      let values = test_values dtype in
      let a = create_tensor ctx dtype small_shape values in
      let result = Backend.op_sin a in
      check (array int) "sin shape" small_shape (get_shape result)

  let test_sqrt backend_name dtype ctx () =
    if not (is_dtype_supported backend_name dtype) then skip ()
    else if dtype_in_list dtype (int_dtypes @ bool_dtype) then skip ()
    else
      let values = test_values dtype in
      let a = create_tensor ctx dtype small_shape values in
      let result = Backend.op_sqrt a in
      check (array int) "sqrt shape" small_shape (get_shape result)

  let test_recip backend_name dtype ctx () =
    if not (is_dtype_supported backend_name dtype) then skip ()
    else if dtype_in_list dtype (int_dtypes @ bool_dtype) then skip ()
    else
      let one = get_one dtype in
      let a = create_tensor ctx dtype small_shape (Array.make 6 one) in
      let result = Backend.op_recip a in
      check (array int) "recip shape" small_shape (get_shape result)

  (* Reduction operations *)
  let test_reduce_sum backend_name dtype ctx () =
    if not (is_dtype_supported backend_name dtype) then skip ()
    else
      let values = test_values dtype in
      let a = create_tensor ctx dtype small_shape values in
      let result = Backend.op_reduce_sum ~axes:[| 0 |] ~keepdims:false a in
      check (array int) "reduce_sum shape" [| 3 |] (get_shape result)

  let test_reduce_max backend_name dtype ctx () =
    if not (is_dtype_supported backend_name dtype) then skip ()
    else if dtype_in_list dtype complex_dtypes then skip ()
    else
      let values = test_values dtype in
      let a = create_tensor ctx dtype small_shape values in
      let result = Backend.op_reduce_max ~axes:[| 0 |] ~keepdims:false a in
      check (array int) "reduce_max shape" [| 3 |] (get_shape result)

  let test_reduce_prod backend_name dtype ctx () =
    if not (is_dtype_supported backend_name dtype) then skip ()
    else
      let values = test_values dtype in
      let a = create_tensor ctx dtype small_shape values in
      let result = Backend.op_reduce_prod ~axes:[| 0 |] ~keepdims:false a in
      check (array int) "reduce_prod shape" [| 3 |] (get_shape result)

  (* Movement operations *)
  let test_expand backend_name dtype ctx () =
    if not (is_dtype_supported backend_name dtype) then skip ()
    else
      let a = Backend.op_const_scalar ctx (get_one dtype) dtype in
      let result = Backend.op_expand a (Symbolic_shape.of_ints [| 2; 3 |]) in
      check (array int) "expand shape" [| 2; 3 |] (get_shape result)

  let test_reshape backend_name dtype ctx () =
    if not (is_dtype_supported backend_name dtype) then skip ()
    else
      let values = test_values dtype in
      let a = create_tensor ctx dtype small_shape values in
      let result = Backend.op_reshape a (Symbolic_shape.of_ints [| 3; 2 |]) in
      check (array int) "reshape shape" [| 3; 2 |] (get_shape result)

  let test_permute backend_name dtype ctx () =
    if not (is_dtype_supported backend_name dtype) then skip ()
    else
      let values = test_values dtype in
      let a = create_tensor ctx dtype small_shape values in
      let result = Backend.op_permute a [| 1; 0 |] in
      check (array int) "permute shape" [| 3; 2 |] (get_shape result)

  let test_pad backend_name dtype ctx () =
    if not (is_dtype_supported backend_name dtype) then skip ()
    else
      let values = test_values dtype in
      let a = create_tensor ctx dtype [| 2; 2 |] (Array.sub values 0 4) in
      let result = Backend.op_pad a [| (1, 1); (0, 0) |] (get_zero dtype) in
      check (array int) "pad shape" [| 4; 2 |] (get_shape result)

  let test_shrink backend_name dtype ctx () =
    if not (is_dtype_supported backend_name dtype) then skip ()
    else
      let values = test_values dtype in
      let a = create_tensor ctx dtype small_shape values in
      let result = Backend.op_shrink a [| (0, 2); (0, 2) |] in
      check (array int) "shrink shape" [| 2; 2 |] (get_shape result)

  let test_flip backend_name dtype ctx () =
    if not (is_dtype_supported backend_name dtype) then skip ()
    else
      let values = test_values dtype in
      let a = create_tensor ctx dtype small_shape values in
      let result = Backend.op_flip a [| true; false |] in
      check (array int) "flip shape" small_shape (get_shape result)

  let test_cat backend_name dtype ctx () =
    if not (is_dtype_supported backend_name dtype) then skip ()
    else
      let values = test_values dtype in
      let a = create_tensor ctx dtype [| 2; 2 |] (Array.sub values 0 4) in
      let b = create_tensor ctx dtype [| 2; 2 |] (Array.sub values 2 4) in
      let result = Backend.op_cat [ a; b ] 0 in
      check (array int) "cat shape" [| 4; 2 |] (get_shape result)

  (* Other operations *)
  let test_cast backend_name dtype ctx () =
    if not (is_dtype_supported backend_name dtype) then skip ()
    else
      let values = test_values dtype in
      let a = create_tensor ctx dtype small_shape values in
      (* Cast to float32 and back *)
      let casted = Backend.op_cast a Dtype.float32 in
      let back = Backend.op_cast casted dtype in
      check (array int) "cast shape" small_shape (get_shape back)

  let test_contiguous backend_name dtype ctx () =
    if not (is_dtype_supported backend_name dtype) then skip ()
    else
      let values = test_values dtype in
      let a = create_tensor ctx dtype small_shape values in
      let transposed = Backend.op_permute a [| 1; 0 |] in
      let result = Backend.op_contiguous transposed in
      check (array int) "contiguous shape" [| 3; 2 |] (get_shape result)

  let test_copy backend_name dtype ctx () =
    if not (is_dtype_supported backend_name dtype) then skip ()
    else
      let values = test_values dtype in
      let a = create_tensor ctx dtype small_shape values in
      let result = Backend.op_copy a in
      check (array int) "copy shape" small_shape (get_shape result)

  let test_where backend_name dtype ctx () =
    if not (is_dtype_supported backend_name dtype) then skip ()
    else
      let values = test_values dtype in
      let cond =
        create_tensor ctx Dtype.uint8 small_shape [| 1; 0; 1; 0; 1; 0 |]
      in
      let a = create_tensor ctx dtype small_shape values in
      let b = create_tensor ctx dtype small_shape values in
      let result = Backend.op_where cond a b in
      check (array int) "where shape" small_shape (get_shape result)

  (* Generate test suite for a specific dtype *)
  let dtype_suite backend_name dtype ctx =
    let dtype_str = Dtype.to_string dtype in
    [
      (* Binary ops *)
      test_case (dtype_str ^ " add") `Quick (test_add backend_name dtype ctx);
      test_case (dtype_str ^ " mul") `Quick (test_mul backend_name dtype ctx);
      test_case (dtype_str ^ " sub") `Quick (test_sub backend_name dtype ctx);
      test_case (dtype_str ^ " div") `Quick (test_div backend_name dtype ctx);
      test_case (dtype_str ^ " pow") `Quick (test_pow backend_name dtype ctx);
      test_case (dtype_str ^ " max") `Quick (test_max backend_name dtype ctx);
      test_case (dtype_str ^ " mod") `Quick (test_mod backend_name dtype ctx);
      (* Comparison ops *)
      test_case (dtype_str ^ " cmplt") `Quick
        (test_cmplt backend_name dtype ctx);
      test_case (dtype_str ^ " cmpne") `Quick
        (test_cmpne backend_name dtype ctx);
      (* Bitwise ops *)
      test_case (dtype_str ^ " xor") `Quick (test_xor backend_name dtype ctx);
      test_case (dtype_str ^ " or") `Quick (test_or backend_name dtype ctx);
      test_case (dtype_str ^ " and") `Quick (test_and backend_name dtype ctx);
      (* Unary ops *)
      test_case (dtype_str ^ " neg") `Quick (test_neg backend_name dtype ctx);
      test_case (dtype_str ^ " log2") `Quick (test_log2 backend_name dtype ctx);
      test_case (dtype_str ^ " exp2") `Quick (test_exp2 backend_name dtype ctx);
      test_case (dtype_str ^ " sin") `Quick (test_sin backend_name dtype ctx);
      test_case (dtype_str ^ " sqrt") `Quick (test_sqrt backend_name dtype ctx);
      test_case (dtype_str ^ " recip") `Quick
        (test_recip backend_name dtype ctx);
      (* Reduction ops *)
      test_case
        (dtype_str ^ " reduce_sum")
        `Quick
        (test_reduce_sum backend_name dtype ctx);
      test_case
        (dtype_str ^ " reduce_max")
        `Quick
        (test_reduce_max backend_name dtype ctx);
      test_case
        (dtype_str ^ " reduce_prod")
        `Quick
        (test_reduce_prod backend_name dtype ctx);
      (* Movement ops *)
      test_case (dtype_str ^ " expand") `Quick
        (test_expand backend_name dtype ctx);
      test_case (dtype_str ^ " reshape") `Quick
        (test_reshape backend_name dtype ctx);
      test_case (dtype_str ^ " permute") `Quick
        (test_permute backend_name dtype ctx);
      test_case (dtype_str ^ " pad") `Quick (test_pad backend_name dtype ctx);
      test_case (dtype_str ^ " shrink") `Quick
        (test_shrink backend_name dtype ctx);
      test_case (dtype_str ^ " flip") `Quick (test_flip backend_name dtype ctx);
      test_case (dtype_str ^ " cat") `Quick (test_cat backend_name dtype ctx);
      (* Other ops *)
      test_case (dtype_str ^ " cast") `Quick (test_cast backend_name dtype ctx);
      test_case
        (dtype_str ^ " contiguous")
        `Quick
        (test_contiguous backend_name dtype ctx);
      test_case (dtype_str ^ " copy") `Quick (test_copy backend_name dtype ctx);
      test_case (dtype_str ^ " where") `Quick
        (test_where backend_name dtype ctx);
    ]

  let suite backend_name ctx =
    List.map
      (fun (Dtype.Pack dtype) ->
        let dtype_str = Dtype.to_string dtype in
        ( "Backend Dtype :: " ^ backend_name ^ " " ^ dtype_str,
          dtype_suite backend_name dtype ctx ))
      all_dtypes
end
