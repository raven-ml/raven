(* Test for BLAS backend *)
module B = Nx_cblas
open Nx_core

let test_scalar_ops () =
  let ctx = () in

  (* Test scalar operations *)
  let a = B.op_const_scalar ctx 2.0 Float32 in
  let b = B.op_const_scalar ctx 3.0 Float32 in

  let c = B.op_add a b in
  let c_data = B.data c in
  Alcotest.(check (float 0.001)) "2.0 + 3.0 = 5.0" 5.0 c_data.{0}

let test_vector_ops () =
  let ctx = () in

  (* Test vector operations *)
  let arr1 =
    Bigarray.Array1.of_array Float32 Bigarray.c_layout [| 1.0; 2.0; 3.0; 4.0 |]
  in
  let arr2 =
    Bigarray.Array1.of_array Float32 Bigarray.c_layout [| 5.0; 6.0; 7.0; 8.0 |]
  in

  let x = B.op_const_array ctx arr1 in
  let y = B.op_const_array ctx arr2 in

  (* Test addition *)
  let z_add = B.op_add x y in
  let z_add_data = B.data z_add in
  Alcotest.(check (float 0.001)) "add[0]" 6.0 z_add_data.{0};
  Alcotest.(check (float 0.001)) "add[1]" 8.0 z_add_data.{1};
  Alcotest.(check (float 0.001)) "add[2]" 10.0 z_add_data.{2};
  Alcotest.(check (float 0.001)) "add[3]" 12.0 z_add_data.{3};

  (* Test subtraction *)
  let z_sub = B.op_sub x y in
  let z_sub_data = B.data z_sub in
  Alcotest.(check (float 0.001)) "sub[0]" (-4.0) z_sub_data.{0};
  Alcotest.(check (float 0.001)) "sub[1]" (-4.0) z_sub_data.{1};
  Alcotest.(check (float 0.001)) "sub[2]" (-4.0) z_sub_data.{2};
  Alcotest.(check (float 0.001)) "sub[3]" (-4.0) z_sub_data.{3};

  (* Test element-wise multiplication *)
  let z_mul = B.op_mul x y in
  let z_mul_data = B.data z_mul in
  Alcotest.(check (float 0.001)) "mul[0]" 5.0 z_mul_data.{0};
  Alcotest.(check (float 0.001)) "mul[1]" 12.0 z_mul_data.{1};
  Alcotest.(check (float 0.001)) "mul[2]" 21.0 z_mul_data.{2};
  Alcotest.(check (float 0.001)) "mul[3]" 32.0 z_mul_data.{3};

  (* Test negation *)
  let z_neg = B.op_neg x in
  let z_neg_data = B.data z_neg in
  Alcotest.(check (float 0.001)) "neg[0]" (-1.0) z_neg_data.{0};
  Alcotest.(check (float 0.001)) "neg[1]" (-2.0) z_neg_data.{1};
  Alcotest.(check (float 0.001)) "neg[2]" (-3.0) z_neg_data.{2};
  Alcotest.(check (float 0.001)) "neg[3]" (-4.0) z_neg_data.{3}

let test_matrix_mul () =
  let ctx = () in

  (* Test matrix multiplication *)
  (* Now using C-contiguous layout, so data is in row-major order *)
  let mat1 =
    Bigarray.Array1.of_array Float32 Bigarray.c_layout [| 1.0; 2.0; 3.0; 4.0 |]
  in
  let mat2 =
    Bigarray.Array1.of_array Float32 Bigarray.c_layout [| 5.0; 6.0; 7.0; 8.0 |]
  in

  let m1 = B.op_const_array ctx mat1 in
  let m2 = B.op_const_array ctx mat2 in

  (* Reshape to 2x2 matrices *)
  let m1_2x2 = B.op_reshape m1 [| 2; 2 |] in
  let m2_2x2 = B.op_reshape m2 [| 2; 2 |] in

  (* Matrix multiplication *)
  let m_result = B.op_mul m1_2x2 m2_2x2 in
  let m_result_data = B.data m_result in

  (* Expected: [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]] *)
  Alcotest.(check (float 0.001)) "matmul[0,0]" 19.0 m_result_data.{0};
  Alcotest.(check (float 0.001)) "matmul[0,1]" 22.0 m_result_data.{1};
  Alcotest.(check (float 0.001)) "matmul[1,0]" 43.0 m_result_data.{2};
  Alcotest.(check (float 0.001)) "matmul[1,1]" 50.0 m_result_data.{3}

let test_reduction () =
  let ctx = () in

  let arr =
    Bigarray.Array1.of_array Float32 Bigarray.c_layout [| 1.0; 2.0; 3.0; 4.0 |]
  in
  let x = B.op_const_array ctx arr in

  (* Test sum reduction *)
  let sum = B.op_reduce_sum ~axes:None ~keepdims:false x in
  let sum_data = B.data sum in
  Alcotest.(check (float 0.001)) "sum" 10.0 sum_data.{0}

let test_batched_matmul () =
  let ctx = () in

  (* Test 3D batched matrix multiplication *)
  (* Create two batches of 2x2 matrices *)
  (* Batch 0: [[1,2],[3,4]] and [[5,6],[7,8]] *)
  (* Batch 1: [[9,10],[11,12]] and [[13,14],[15,16]] *)

  (* In row-major order, shape [2,2,2] means [batch, rows, cols] *)
  let mat1 =
    Bigarray.Array1.of_array Float32 Bigarray.c_layout
      [|
        1.0;
        2.0;
        3.0;
        4.0;
        (* batch 0: [[1,2],[3,4]] *)
        9.0;
        10.0;
        11.0;
        12.0 (* batch 1: [[9,10],[11,12]] *);
      |]
  in
  let mat2 =
    Bigarray.Array1.of_array Float32 Bigarray.c_layout
      [|
        5.0;
        6.0;
        7.0;
        8.0;
        (* batch 0: [[5,6],[7,8]] *)
        13.0;
        14.0;
        15.0;
        16.0 (* batch 1: [[13,14],[15,16]] *);
      |]
  in

  let m1 = B.op_const_array ctx mat1 in
  let m2 = B.op_const_array ctx mat2 in

  (* Reshape to [2, 2, 2] - 2 batches of 2x2 matrices *)
  let m1_3d = B.op_reshape m1 [| 2; 2; 2 |] in
  let m2_3d = B.op_reshape m2 [| 2; 2; 2 |] in

  (* Batched matrix multiplication *)
  let m_result = B.op_mul m1_3d m2_3d in
  let m_result_data = B.data m_result in

  (* Expected results: Batch 0: [[1,2],[3,4]] @ [[5,6],[7,8]] =
     [[19,22],[43,50]] Batch 1: [[9,10],[11,12]] @ [[13,14],[15,16]] =
     [[267,286],[323,346]] *)

  (* Batch 0 results *)
  Alcotest.(check (float 0.001)) "batch0[0,0]" 19.0 m_result_data.{0};
  Alcotest.(check (float 0.001)) "batch0[0,1]" 22.0 m_result_data.{1};
  Alcotest.(check (float 0.001)) "batch0[1,0]" 43.0 m_result_data.{2};
  Alcotest.(check (float 0.001)) "batch0[1,1]" 50.0 m_result_data.{3};

  (* Batch 1 results *)
  Alcotest.(check (float 0.001)) "batch1[0,0]" 267.0 m_result_data.{4};
  Alcotest.(check (float 0.001)) "batch1[0,1]" 286.0 m_result_data.{5};
  Alcotest.(check (float 0.001)) "batch1[1,0]" 323.0 m_result_data.{6};
  Alcotest.(check (float 0.001)) "batch1[1,1]" 346.0 m_result_data.{7}

let test_float64_ops () =
  let ctx = () in

  (* Test Float64 operations *)
  let arr1 =
    Bigarray.Array1.of_array Float64 Bigarray.c_layout [| 1.0; 2.0; 3.0; 4.0 |]
  in
  let arr2 =
    Bigarray.Array1.of_array Float64 Bigarray.c_layout [| 5.0; 6.0; 7.0; 8.0 |]
  in

  let x = B.op_const_array ctx arr1 in
  let y = B.op_const_array ctx arr2 in

  (* Test addition *)
  let z_add = B.op_add x y in
  let z_add_data = B.data z_add in
  Alcotest.(check (float 0.001)) "add[0]" 6.0 z_add_data.{0};
  Alcotest.(check (float 0.001)) "add[3]" 12.0 z_add_data.{3};

  (* Test matrix multiplication with Float64 *)
  let m1 = B.op_reshape x [| 2; 2 |] in
  let m2 = B.op_reshape y [| 2; 2 |] in
  let m_result = B.op_mul m1 m2 in
  let m_result_data = B.data m_result in

  Alcotest.(check (float 0.001)) "matmul[0,0]" 19.0 m_result_data.{0};
  Alcotest.(check (float 0.001)) "matmul[1,1]" 50.0 m_result_data.{3}

let test_contiguous_copy () =
  let ctx = () in

  (* Test making non-contiguous tensor contiguous *)
  let arr =
    Bigarray.Array1.of_array Float32 Bigarray.c_layout
      [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
  in
  let x = B.op_const_array ctx arr in
  let x_2x3 = B.op_reshape x [| 2; 3 |] in

  (* Create a non-contiguous view by transposing *)
  let x_transposed = B.op_permute x_2x3 [| 1; 0 |] in

  (* Check that it's not contiguous *)
  let view = B.view x_transposed in
  Alcotest.(check bool)
    "transposed is not contiguous" false
    (View.is_c_contiguous view);

  (* Make it contiguous *)
  let x_contig = B.op_contiguous x_transposed in
  let view_contig = B.view x_contig in
  Alcotest.(check bool)
    "contiguous copy is contiguous" true
    (View.is_c_contiguous view_contig);

  (* Check the data is correct (transposed) *)
  let data = B.data x_contig in
  (* Original: [[1,2,3],[4,5,6]] Transposed: [[1,4],[2,5],[3,6]] In row-major:
     [1,4,2,5,3,6] *)
  Alcotest.(check (float 0.001)) "contig[0]" 1.0 data.{0};
  Alcotest.(check (float 0.001)) "contig[1]" 4.0 data.{1};
  Alcotest.(check (float 0.001)) "contig[2]" 2.0 data.{2};
  Alcotest.(check (float 0.001)) "contig[3]" 5.0 data.{3}

let test_view_operations () =
  let ctx = () in

  (* Test expand operation *)
  let arr =
    Bigarray.Array1.of_array Float32 Bigarray.c_layout [| 1.0; 2.0; 3.0 |]
  in
  let x = B.op_const_array ctx arr in
  let x_1x3 = B.op_reshape x [| 1; 3 |] in
  let x_expanded = B.op_expand x_1x3 [| 2; 3 |] in

  let view = B.view x_expanded in
  let shape = View.shape view in
  Alcotest.(check (array int)) "expanded shape" [| 2; 3 |] shape;

  (* Test shrink (slicing) *)
  let x_sliced = B.op_shrink x_expanded [| (0, 1); (1, 3) |] in
  let view_sliced = B.view x_sliced in
  let shape_sliced = View.shape view_sliced in
  Alcotest.(check (array int)) "sliced shape" [| 1; 2 |] shape_sliced

let test_arithmetic_ops () =
  let ctx = () in

  (* Test division *)
  let arr1 =
    Bigarray.Array1.of_array Float32 Bigarray.c_layout
      [| 10.0; 20.0; 30.0; 40.0 |]
  in
  let arr2 =
    Bigarray.Array1.of_array Float32 Bigarray.c_layout [| 2.0; 4.0; 5.0; 8.0 |]
  in

  let x = B.op_const_array ctx arr1 in
  let y = B.op_const_array ctx arr2 in

  let z_div = B.op_fdiv x y in
  let z_div_data = B.data z_div in
  Alcotest.(check (float 0.001)) "div[0]" 5.0 z_div_data.{0};
  Alcotest.(check (float 0.001)) "div[1]" 5.0 z_div_data.{1};
  Alcotest.(check (float 0.001)) "div[2]" 6.0 z_div_data.{2};
  Alcotest.(check (float 0.001)) "div[3]" 5.0 z_div_data.{3};

  (* Test sqrt *)
  let arr_sqrt =
    Bigarray.Array1.of_array Float32 Bigarray.c_layout
      [| 4.0; 9.0; 16.0; 25.0 |]
  in
  let x_sqrt = B.op_const_array ctx arr_sqrt in
  let z_sqrt = B.op_sqrt x_sqrt in
  let z_sqrt_data = B.data z_sqrt in
  Alcotest.(check (float 0.001)) "sqrt[0]" 2.0 z_sqrt_data.{0};
  Alcotest.(check (float 0.001)) "sqrt[1]" 3.0 z_sqrt_data.{1};
  Alcotest.(check (float 0.001)) "sqrt[2]" 4.0 z_sqrt_data.{2};
  Alcotest.(check (float 0.001)) "sqrt[3]" 5.0 z_sqrt_data.{3}

let test_math_ops () =
  let ctx = () in

  (* Test sin *)
  let pi = 3.14159265359 in
  let arr =
    Bigarray.Array1.of_array Float32 Bigarray.c_layout
      [| 0.0; pi /. 2.0; pi; 3.0 *. pi /. 2.0 |]
  in
  let x = B.op_const_array ctx arr in
  let z_sin = B.op_sin x in
  let z_sin_data = B.data z_sin in
  Alcotest.(check (float 0.001)) "sin[0]" 0.0 z_sin_data.{0};
  Alcotest.(check (float 0.001)) "sin[pi/2]" 1.0 z_sin_data.{1};
  Alcotest.(check (float 0.001)) "sin[pi]" 0.0 z_sin_data.{2};
  Alcotest.(check (float 0.001)) "sin[3pi/2]" (-1.0) z_sin_data.{3};

  (* Test recip *)
  let arr_recip =
    Bigarray.Array1.of_array Float32 Bigarray.c_layout [| 2.0; 4.0; 0.5; 0.25 |]
  in
  let x_recip = B.op_const_array ctx arr_recip in
  let z_recip = B.op_recip x_recip in
  let z_recip_data = B.data z_recip in
  Alcotest.(check (float 0.001)) "recip[0]" 0.5 z_recip_data.{0};
  Alcotest.(check (float 0.001)) "recip[1]" 0.25 z_recip_data.{1};
  Alcotest.(check (float 0.001)) "recip[2]" 2.0 z_recip_data.{2};
  Alcotest.(check (float 0.001)) "recip[3]" 4.0 z_recip_data.{3};

  (* Test log2 and exp2 *)
  let arr_log =
    Bigarray.Array1.of_array Float32 Bigarray.c_layout [| 1.0; 2.0; 4.0; 8.0 |]
  in
  let x_log = B.op_const_array ctx arr_log in
  let z_log2 = B.op_log2 x_log in
  let z_log2_data = B.data z_log2 in
  Alcotest.(check (float 0.001)) "log2[1]" 0.0 z_log2_data.{0};
  Alcotest.(check (float 0.001)) "log2[2]" 1.0 z_log2_data.{1};
  Alcotest.(check (float 0.001)) "log2[4]" 2.0 z_log2_data.{2};
  Alcotest.(check (float 0.001)) "log2[8]" 3.0 z_log2_data.{3};

  let arr_exp =
    Bigarray.Array1.of_array Float32 Bigarray.c_layout [| 0.0; 1.0; 2.0; 3.0 |]
  in
  let x_exp = B.op_const_array ctx arr_exp in
  let z_exp2 = B.op_exp2 x_exp in
  let z_exp2_data = B.data z_exp2 in
  Alcotest.(check (float 0.001)) "exp2[0]" 1.0 z_exp2_data.{0};
  Alcotest.(check (float 0.001)) "exp2[1]" 2.0 z_exp2_data.{1};
  Alcotest.(check (float 0.001)) "exp2[2]" 4.0 z_exp2_data.{2};
  Alcotest.(check (float 0.001)) "exp2[3]" 8.0 z_exp2_data.{3}

let test_non_contiguous_ops () =
  let ctx = () in

  (* Create a 2x3 matrix and transpose it to get non-contiguous view *)
  let arr =
    Bigarray.Array1.of_array Float32 Bigarray.c_layout
      [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
  in
  let x = B.op_const_array ctx arr in
  let x_2x3 = B.op_reshape x [| 2; 3 |] in

  (* Transpose to get non-contiguous view *)
  let x_transposed = B.op_permute x_2x3 [| 1; 0 |] in

  (* Create another non-contiguous array *)
  let arr2 =
    Bigarray.Array1.of_array Float32 Bigarray.c_layout
      [| 10.0; 20.0; 30.0; 40.0; 50.0; 60.0 |]
  in
  let y = B.op_const_array ctx arr2 in
  let y_2x3 = B.op_reshape y [| 2; 3 |] in
  let y_transposed = B.op_permute y_2x3 [| 1; 0 |] in

  (* Test addition on non-contiguous arrays *)
  let z_add = B.op_add x_transposed y_transposed in
  let z_add_contig = B.op_contiguous z_add in
  let z_add_data = B.data z_add_contig in

  (* Expected: x_transposed is [[1,4],[2,5],[3,6]] y_transposed is
     [[10,40],[20,50],[30,60]] Result should be [[11,44],[22,55],[33,66]] In
     row-major: [11,44,22,55,33,66] *)
  Alcotest.(check (float 0.001)) "non-contig add[0,0]" 11.0 z_add_data.{0};
  Alcotest.(check (float 0.001)) "non-contig add[0,1]" 44.0 z_add_data.{1};
  Alcotest.(check (float 0.001)) "non-contig add[1,0]" 22.0 z_add_data.{2};
  Alcotest.(check (float 0.001)) "non-contig add[1,1]" 55.0 z_add_data.{3};

  (* Test subtraction on non-contiguous arrays *)
  let z_sub = B.op_sub x_transposed y_transposed in
  let z_sub_contig = B.op_contiguous z_sub in
  let z_sub_data = B.data z_sub_contig in

  (* Expected: [[1-10,4-40],[2-20,5-50],[3-30,6-60]] =
     [[-9,-36],[-18,-45],[-27,-54]] *)
  Alcotest.(check (float 0.001)) "non-contig sub[0,0]" (-9.0) z_sub_data.{0};
  Alcotest.(check (float 0.001)) "non-contig sub[0,1]" (-36.0) z_sub_data.{1};

  (* Test multiplication on non-contiguous arrays *)
  let z_mul = B.op_mul x_transposed y_transposed in
  let z_mul_contig = B.op_contiguous z_mul in
  let z_mul_data = B.data z_mul_contig in

  (* Expected: [[1*10,4*40],[2*20,5*50],[3*30,6*60]] =
     [[10,160],[40,250],[90,360]] *)
  Alcotest.(check (float 0.001)) "non-contig mul[0,0]" 10.0 z_mul_data.{0};
  Alcotest.(check (float 0.001)) "non-contig mul[0,1]" 160.0 z_mul_data.{1};
  Alcotest.(check (float 0.001)) "non-contig mul[1,0]" 40.0 z_mul_data.{2};
  Alcotest.(check (float 0.001)) "non-contig mul[1,1]" 250.0 z_mul_data.{3}

let test_pow_and_fdiv () =
  let ctx = () in

  (* Test power operation *)
  let arr1 =
    Bigarray.Array1.of_array Float32 Bigarray.c_layout [| 2.0; 3.0; 4.0; 5.0 |]
  in
  let arr2 =
    Bigarray.Array1.of_array Float32 Bigarray.c_layout [| 2.0; 2.0; 3.0; 0.0 |]
  in

  let x = B.op_const_array ctx arr1 in
  let y = B.op_const_array ctx arr2 in

  let z_pow = B.op_pow x y in
  let z_pow_data = B.data z_pow in
  Alcotest.(check (float 0.001)) "pow[0]" 4.0 z_pow_data.{0};
  (* 2^2 *)
  Alcotest.(check (float 0.001)) "pow[1]" 9.0 z_pow_data.{1};
  (* 3^2 *)
  Alcotest.(check (float 0.001)) "pow[2]" 64.0 z_pow_data.{2};
  (* 4^3 *)
  Alcotest.(check (float 0.001)) "pow[3]" 1.0 z_pow_data.{3};

  (* 5^0 *)

  (* Test division with new implementation *)
  let arr3 =
    Bigarray.Array1.of_array Float32 Bigarray.c_layout
      [| 10.0; 20.0; 30.0; 40.0 |]
  in
  let arr4 =
    Bigarray.Array1.of_array Float32 Bigarray.c_layout [| 2.0; 4.0; 5.0; 8.0 |]
  in

  let x_div = B.op_const_array ctx arr3 in
  let y_div = B.op_const_array ctx arr4 in

  let z_div = B.op_fdiv x_div y_div in
  let z_div_data = B.data z_div in
  Alcotest.(check (float 0.001)) "fdiv[0]" 5.0 z_div_data.{0};
  Alcotest.(check (float 0.001)) "fdiv[1]" 5.0 z_div_data.{1};
  Alcotest.(check (float 0.001)) "fdiv[2]" 6.0 z_div_data.{2};
  Alcotest.(check (float 0.001)) "fdiv[3]" 5.0 z_div_data.{3}

let test_comparison_ops () =
  let ctx = () in

  let arr1 =
    Bigarray.Array1.of_array Float32 Bigarray.c_layout [| 1.0; 2.0; 3.0; 4.0 |]
  in
  let arr2 =
    Bigarray.Array1.of_array Float32 Bigarray.c_layout [| 2.0; 2.0; 1.0; 5.0 |]
  in

  let x = B.op_const_array ctx arr1 in
  let y = B.op_const_array ctx arr2 in

  (* Test less than *)
  let z_lt = B.op_cmplt x y in
  let z_lt_data = B.data z_lt in
  Alcotest.(check int) "lt[0]" 1 z_lt_data.{0};
  (* 1 < 2 *)
  Alcotest.(check int) "lt[1]" 0 z_lt_data.{1};
  (* 2 < 2 *)
  Alcotest.(check int) "lt[2]" 0 z_lt_data.{2};
  (* 3 < 1 *)
  Alcotest.(check int) "lt[3]" 1 z_lt_data.{3};

  (* 4 < 5 *)

  (* Test not equal *)
  let z_ne = B.op_cmpne x y in
  let z_ne_data = B.data z_ne in
  Alcotest.(check int) "ne[0]" 1 z_ne_data.{0};
  (* 1 != 2 *)
  Alcotest.(check int) "ne[1]" 0 z_ne_data.{1};
  (* 2 != 2 *)
  Alcotest.(check int) "ne[2]" 1 z_ne_data.{2};
  (* 3 != 1 *)
  Alcotest.(check int) "ne[3]" 1 z_ne_data.{3};

  (* 4 != 5 *)

  (* Test max *)
  let z_max = B.op_max x y in
  let z_max_data = B.data z_max in
  Alcotest.(check (float 0.001)) "max[0]" 2.0 z_max_data.{0};
  Alcotest.(check (float 0.001)) "max[1]" 2.0 z_max_data.{1};
  Alcotest.(check (float 0.001)) "max[2]" 3.0 z_max_data.{2};
  Alcotest.(check (float 0.001)) "max[3]" 5.0 z_max_data.{3}

let test_mod_op () =
  let ctx = () in

  let arr1 =
    Bigarray.Array1.of_array Float32 Bigarray.c_layout [| 7.5; 8.0; 9.3; 10.0 |]
  in
  let arr2 =
    Bigarray.Array1.of_array Float32 Bigarray.c_layout [| 2.0; 3.0; 4.0; 3.0 |]
  in

  let x = B.op_const_array ctx arr1 in
  let y = B.op_const_array ctx arr2 in

  (* Test modulo *)
  let z_mod = B.op_mod x y in
  let z_mod_data = B.data z_mod in
  Alcotest.(check (float 0.001)) "mod[0]" 1.5 z_mod_data.{0};
  (* 7.5 % 2.0 = 1.5 *)
  Alcotest.(check (float 0.001)) "mod[1]" 2.0 z_mod_data.{1};
  (* 8.0 % 3.0 = 2.0 *)
  Alcotest.(check (float 0.001)) "mod[2]" 1.3 z_mod_data.{2};
  (* 9.3 % 4.0 = 1.3 *)
  Alcotest.(check (float 0.001)) "mod[3]" 1.0 z_mod_data.{3}
(* 10.0 % 3.0 = 1.0 *)

(* Helper to create int32 array *)
let int32_array vals =
  Bigarray.Array1.of_array Int32 Bigarray.c_layout (Array.map Int32.of_int vals)

(* Helper to create int64 array *)
let int64_array vals =
  Bigarray.Array1.of_array Int64 Bigarray.c_layout (Array.map Int64.of_int vals)

(* Helper to create uint8 array *)
let uint8_array vals =
  Bigarray.Array1.of_array Int8_unsigned Bigarray.c_layout vals

let test_idiv () =
  let ctx = () in

  (* Test int32 division *)
  let arr1 = int32_array [| 10; 20; 30; 45 |] in
  let arr2 = int32_array [| 3; 4; 7; 9 |] in
  
  let x = B.op_const_array ctx arr1 in
  let y = B.op_const_array ctx arr2 in
  
  let z = B.op_idiv x y in
  let z_data = B.data z in
  
  Alcotest.(check int32) "idiv[0]" 3l z_data.{0};  (* 10 / 3 = 3 *)
  Alcotest.(check int32) "idiv[1]" 5l z_data.{1};  (* 20 / 4 = 5 *)
  Alcotest.(check int32) "idiv[2]" 4l z_data.{2};  (* 30 / 7 = 4 *)
  Alcotest.(check int32) "idiv[3]" 5l z_data.{3};  (* 45 / 9 = 5 *)
  
  (* Test int64 division *)
  let arr1_64 = int64_array [| 100; 200; 300; 450 |] in
  let arr2_64 = int64_array [| 3; 4; 7; 9 |] in
  
  let x64 = B.op_const_array ctx arr1_64 in
  let y64 = B.op_const_array ctx arr2_64 in
  
  let z64 = B.op_idiv x64 y64 in
  let z64_data = B.data z64 in
  
  Alcotest.(check int64) "idiv64[0]" 33L z64_data.{0};  (* 100 / 3 = 33 *)
  Alcotest.(check int64) "idiv64[1]" 50L z64_data.{1};  (* 200 / 4 = 50 *)
  Alcotest.(check int64) "idiv64[2]" 42L z64_data.{2};  (* 300 / 7 = 42 *)
  Alcotest.(check int64) "idiv64[3]" 50L z64_data.{3}   (* 450 / 9 = 50 *)

let test_bitwise_ops () =
  let ctx = () in

  (* Test int32 bitwise operations *)
  let arr1 = int32_array [| 0b1100; 0b1010; 0b1111; 0b0001 |] in
  let arr2 = int32_array [| 0b1010; 0b0110; 0b0101; 0b0001 |] in
  
  let x = B.op_const_array ctx arr1 in
  let y = B.op_const_array ctx arr2 in
  
  (* Test XOR *)
  let z_xor = B.op_xor x y in
  let z_xor_data = B.data z_xor in
  Alcotest.(check int32) "xor[0]" 0b0110l z_xor_data.{0};  (* 1100 ^ 1010 = 0110 *)
  Alcotest.(check int32) "xor[1]" 0b1100l z_xor_data.{1};  (* 1010 ^ 0110 = 1100 *)
  Alcotest.(check int32) "xor[2]" 0b1010l z_xor_data.{2};  (* 1111 ^ 0101 = 1010 *)
  Alcotest.(check int32) "xor[3]" 0b0000l z_xor_data.{3};  (* 0001 ^ 0001 = 0000 *)
  
  (* Test OR *)
  let z_or = B.op_or x y in
  let z_or_data = B.data z_or in
  Alcotest.(check int32) "or[0]" 0b1110l z_or_data.{0};  (* 1100 | 1010 = 1110 *)
  Alcotest.(check int32) "or[1]" 0b1110l z_or_data.{1};  (* 1010 | 0110 = 1110 *)
  Alcotest.(check int32) "or[2]" 0b1111l z_or_data.{2};  (* 1111 | 0101 = 1111 *)
  Alcotest.(check int32) "or[3]" 0b0001l z_or_data.{3};  (* 0001 | 0001 = 0001 *)
  
  (* Test AND *)
  let z_and = B.op_and x y in
  let z_and_data = B.data z_and in
  Alcotest.(check int32) "and[0]" 0b1000l z_and_data.{0};  (* 1100 & 1010 = 1000 *)
  Alcotest.(check int32) "and[1]" 0b0010l z_and_data.{1};  (* 1010 & 0110 = 0010 *)
  Alcotest.(check int32) "and[2]" 0b0101l z_and_data.{2};  (* 1111 & 0101 = 0101 *)
  Alcotest.(check int32) "and[3]" 0b0001l z_and_data.{3};  (* 0001 & 0001 = 0001 *)
  
  (* Test uint8 bitwise operations *)
  let arr1_u8 = uint8_array [| 0xff; 0xaa; 0x55; 0x0f |] in
  let arr2_u8 = uint8_array [| 0xf0; 0x55; 0xaa; 0xf0 |] in
  
  let x_u8 = B.op_const_array ctx arr1_u8 in
  let y_u8 = B.op_const_array ctx arr2_u8 in
  
  let z_xor_u8 = B.op_xor x_u8 y_u8 in
  let z_xor_u8_data = B.data z_xor_u8 in
  Alcotest.(check int) "xor_u8[0]" 0x0f z_xor_u8_data.{0};  (* 0xff ^ 0xf0 = 0x0f *)
  Alcotest.(check int) "xor_u8[1]" 0xff z_xor_u8_data.{1}   (* 0xaa ^ 0x55 = 0xff *)

let test_where () =
  let ctx = () in

  (* Create condition array (0 = false, non-zero = true) *)
  let cond_arr = uint8_array [| 1; 0; 1; 0; 1; 1; 0; 0 |] in
  let x_arr = Bigarray.Array1.of_array Float32 Bigarray.c_layout 
    [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0 |] in
  let y_arr = Bigarray.Array1.of_array Float32 Bigarray.c_layout 
    [| 10.0; 20.0; 30.0; 40.0; 50.0; 60.0; 70.0; 80.0 |] in
  
  let cond = B.op_const_array ctx cond_arr in
  let x = B.op_const_array ctx x_arr in
  let y = B.op_const_array ctx y_arr in
  
  let z = B.op_where cond x y in
  let z_data = B.data z in
  
  (* Expected: where cond is true (non-zero), take x, else take y *)
  Alcotest.(check (float 0.001)) "where[0]" 1.0 z_data.{0};   (* cond=1, take x *)
  Alcotest.(check (float 0.001)) "where[1]" 20.0 z_data.{1};  (* cond=0, take y *)
  Alcotest.(check (float 0.001)) "where[2]" 3.0 z_data.{2};   (* cond=1, take x *)
  Alcotest.(check (float 0.001)) "where[3]" 40.0 z_data.{3};  (* cond=0, take y *)
  Alcotest.(check (float 0.001)) "where[4]" 5.0 z_data.{4};   (* cond=1, take x *)
  Alcotest.(check (float 0.001)) "where[5]" 6.0 z_data.{5};   (* cond=1, take x *)
  Alcotest.(check (float 0.001)) "where[6]" 70.0 z_data.{6};  (* cond=0, take y *)
  Alcotest.(check (float 0.001)) "where[7]" 80.0 z_data.{7};  (* cond=0, take y *)
  
  (* Test with float64 *)
  let x_arr_f64 = Bigarray.Array1.of_array Float64 Bigarray.c_layout 
    [| 1.0; 2.0; 3.0; 4.0 |] in
  let y_arr_f64 = Bigarray.Array1.of_array Float64 Bigarray.c_layout 
    [| 10.0; 20.0; 30.0; 40.0 |] in
  let cond_arr_2 = uint8_array [| 0; 1; 0; 1 |] in
  
  let cond2 = B.op_const_array ctx cond_arr_2 in
  let x_f64 = B.op_const_array ctx x_arr_f64 in
  let y_f64 = B.op_const_array ctx y_arr_f64 in
  
  let z_f64 = B.op_where cond2 x_f64 y_f64 in
  let z_f64_data = B.data z_f64 in
  
  Alcotest.(check (float 0.001)) "where_f64[0]" 10.0 z_f64_data.{0};
  Alcotest.(check (float 0.001)) "where_f64[1]" 2.0 z_f64_data.{1};
  Alcotest.(check (float 0.001)) "where_f64[2]" 30.0 z_f64_data.{2};
  Alcotest.(check (float 0.001)) "where_f64[3]" 4.0 z_f64_data.{3}

let test_reduce_max () =
  let ctx = () in

  (* Test reduce_max *)
  let arr = Bigarray.Array1.of_array Float32 Bigarray.c_layout 
    [| 3.0; 7.0; 2.0; 9.0; 1.0; 5.0; 8.0; 4.0 |] in
  let x = B.op_const_array ctx arr in
  
  let max_val = B.op_reduce_max ~axes:None ~keepdims:false x in
  let max_data = B.data max_val in
  Alcotest.(check (float 0.001)) "reduce_max" 9.0 max_data.{0};
  
  (* Test with 2D array *)
  let x_2d = B.op_reshape x [| 2; 4 |] in
  let max_2d = B.op_reduce_max ~axes:None ~keepdims:false x_2d in
  let max_2d_data = B.data max_2d in
  Alcotest.(check (float 0.001)) "reduce_max_2d" 9.0 max_2d_data.{0};
  
  (* Test with keepdims *)
  let max_keepdims = B.op_reduce_max ~axes:None ~keepdims:true x_2d in
  let view = B.view max_keepdims in
  let shape = View.shape view in
  Alcotest.(check (array int)) "reduce_max keepdims shape" [| 1; 1 |] shape;
  
  (* Test float64 *)
  let arr_f64 = Bigarray.Array1.of_array Float64 Bigarray.c_layout 
    [| 1.5; 3.7; 2.2; 3.6 |] in
  let x_f64 = B.op_const_array ctx arr_f64 in
  let max_f64 = B.op_reduce_max ~axes:None ~keepdims:false x_f64 in
  let max_f64_data = B.data max_f64 in
  Alcotest.(check (float 0.001)) "reduce_max_f64" 3.7 max_f64_data.{0}

let test_reduce_prod () =
  let ctx = () in

  (* Test reduce_prod *)
  let arr = Bigarray.Array1.of_array Float32 Bigarray.c_layout 
    [| 2.0; 3.0; 4.0; 5.0 |] in
  let x = B.op_const_array ctx arr in
  
  let prod_val = B.op_reduce_prod ~axes:None ~keepdims:false x in
  let prod_data = B.data prod_val in
  Alcotest.(check (float 0.001)) "reduce_prod" 120.0 prod_data.{0};  (* 2*3*4*5 = 120 *)
  
  (* Test with 2D array *)
  let x_2d = B.op_reshape x [| 2; 2 |] in
  let prod_2d = B.op_reduce_prod ~axes:None ~keepdims:false x_2d in
  let prod_2d_data = B.data prod_2d in
  Alcotest.(check (float 0.001)) "reduce_prod_2d" 120.0 prod_2d_data.{0};
  
  (* Test with keepdims *)
  let prod_keepdims = B.op_reduce_prod ~axes:None ~keepdims:true x_2d in
  let view = B.view prod_keepdims in
  let shape = View.shape view in
  Alcotest.(check (array int)) "reduce_prod keepdims shape" [| 1; 1 |] shape;
  
  (* Test float64 *)
  let arr_f64 = Bigarray.Array1.of_array Float64 Bigarray.c_layout 
    [| 2.0; 3.0; 4.0 |] in
  let x_f64 = B.op_const_array ctx arr_f64 in
  let prod_f64 = B.op_reduce_prod ~axes:None ~keepdims:false x_f64 in
  let prod_f64_data = B.data prod_f64 in
  Alcotest.(check (float 0.001)) "reduce_prod_f64" 24.0 prod_f64_data.{0}  (* 2*3*4 = 24 *)

let test_pad () =
  let ctx = () in

  (* Test 1D padding *)
  let arr = Bigarray.Array1.of_array Float32 Bigarray.c_layout [| 1.0; 2.0; 3.0 |] in
  let x = B.op_const_array ctx arr in
  
  (* Pad with 1 element before and 2 elements after *)
  let pad_config = [| (1, 2) |] in
  let padded = B.op_pad x pad_config 0.0 in
  let padded_data = B.data padded in
  
  (* Expected: [0.0, 1.0, 2.0, 3.0, 0.0, 0.0] *)
  Alcotest.(check (float 0.001)) "pad_1d[0]" 0.0 padded_data.{0};
  Alcotest.(check (float 0.001)) "pad_1d[1]" 1.0 padded_data.{1};
  Alcotest.(check (float 0.001)) "pad_1d[2]" 2.0 padded_data.{2};
  Alcotest.(check (float 0.001)) "pad_1d[3]" 3.0 padded_data.{3};
  Alcotest.(check (float 0.001)) "pad_1d[4]" 0.0 padded_data.{4};
  Alcotest.(check (float 0.001)) "pad_1d[5]" 0.0 padded_data.{5};
  
  (* Test 2D padding *)
  let arr_2d = Bigarray.Array1.of_array Float32 Bigarray.c_layout 
    [| 1.0; 2.0; 3.0; 4.0 |] in
  let x_2d = B.op_const_array ctx arr_2d in
  let x_2d = B.op_reshape x_2d [| 2; 2 |] in
  
  (* Pad with 1 element on all sides *)
  let pad_config_2d = [| (1, 1); (1, 1) |] in
  let padded_2d = B.op_pad x_2d pad_config_2d (-1.0) in
  let padded_2d_data = B.data padded_2d in
  
  (* Expected 4x4 matrix:
     -1 -1 -1 -1
     -1  1  2 -1
     -1  3  4 -1
     -1 -1 -1 -1
  *)
  let view = B.view padded_2d in
  let shape = View.shape view in
  Alcotest.(check (array int)) "pad_2d shape" [| 4; 4 |] shape;
  
  (* Check corners *)
  Alcotest.(check (float 0.001)) "pad_2d[0,0]" (-1.0) padded_2d_data.{0};
  Alcotest.(check (float 0.001)) "pad_2d[0,3]" (-1.0) padded_2d_data.{3};
  Alcotest.(check (float 0.001)) "pad_2d[3,0]" (-1.0) padded_2d_data.{12};
  Alcotest.(check (float 0.001)) "pad_2d[3,3]" (-1.0) padded_2d_data.{15};
  
  (* Check original data *)
  Alcotest.(check (float 0.001)) "pad_2d[1,1]" 1.0 padded_2d_data.{5};
  Alcotest.(check (float 0.001)) "pad_2d[1,2]" 2.0 padded_2d_data.{6};
  Alcotest.(check (float 0.001)) "pad_2d[2,1]" 3.0 padded_2d_data.{9};
  Alcotest.(check (float 0.001)) "pad_2d[2,2]" 4.0 padded_2d_data.{10}

let test_cast () =
  let ctx = () in

  (* Test float32 to float64 *)
  let arr_f32 = Bigarray.Array1.of_array Float32 Bigarray.c_layout 
    [| 1.5; 2.7; 3.14159; 4.0 |] in
  let x_f32 = B.op_const_array ctx arr_f32 in
  
  let x_f64 = B.op_cast x_f32 Dtype.Float64 in
  let x_f64_data = B.data x_f64 in
  
  (* Check dtype changed *)
  Alcotest.(check bool) "cast to float64 dtype" true (B.dtype x_f64 = Dtype.Float64);
  
  (* Check values preserved *)
  Alcotest.(check (float 0.0001)) "cast f32->f64[0]" 1.5 x_f64_data.{0};
  Alcotest.(check (float 0.0001)) "cast f32->f64[1]" 2.7 x_f64_data.{1};
  Alcotest.(check (float 0.0001)) "cast f32->f64[2]" 3.14159 x_f64_data.{2};
  Alcotest.(check (float 0.0001)) "cast f32->f64[3]" 4.0 x_f64_data.{3};
  
  (* Test float64 to float32 *)
  let arr_f64 = Bigarray.Array1.of_array Float64 Bigarray.c_layout 
    [| 1.123456789; 2.987654321; 3.0; 4.5 |] in
  let y_f64 = B.op_const_array ctx arr_f64 in
  
  let y_f32 = B.op_cast y_f64 Dtype.Float32 in
  let y_f32_data = B.data y_f32 in
  
  (* Check dtype changed *)
  Alcotest.(check bool) "cast to float32 dtype" true (B.dtype y_f32 = Dtype.Float32);
  
  (* Check values (with some precision loss expected) *)
  Alcotest.(check (float 0.0001)) "cast f64->f32[0]" 1.123457 y_f32_data.{0};
  Alcotest.(check (float 0.0001)) "cast f64->f32[1]" 2.987654 y_f32_data.{1};
  Alcotest.(check (float 0.0001)) "cast f64->f32[2]" 3.0 y_f32_data.{2};
  Alcotest.(check (float 0.0001)) "cast f64->f32[3]" 4.5 y_f32_data.{3}

let test_cat () =
  let ctx = () in

  (* Test 1D concatenation *)
  let arr1 = Bigarray.Array1.of_array Float32 Bigarray.c_layout [| 1.0; 2.0 |] in
  let arr2 = Bigarray.Array1.of_array Float32 Bigarray.c_layout [| 3.0; 4.0; 5.0 |] in
  let arr3 = Bigarray.Array1.of_array Float32 Bigarray.c_layout [| 6.0 |] in
  
  let x1 = B.op_const_array ctx arr1 in
  let x2 = B.op_const_array ctx arr2 in
  let x3 = B.op_const_array ctx arr3 in
  
  let result = B.op_cat [x1; x2; x3] 0 in
  let result_data = B.data result in
  
  (* Expected: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] *)
  Alcotest.(check (float 0.001)) "cat_1d[0]" 1.0 result_data.{0};
  Alcotest.(check (float 0.001)) "cat_1d[1]" 2.0 result_data.{1};
  Alcotest.(check (float 0.001)) "cat_1d[2]" 3.0 result_data.{2};
  Alcotest.(check (float 0.001)) "cat_1d[3]" 4.0 result_data.{3};
  Alcotest.(check (float 0.001)) "cat_1d[4]" 5.0 result_data.{4};
  Alcotest.(check (float 0.001)) "cat_1d[5]" 6.0 result_data.{5};
  
  (* Test 2D concatenation along axis 0 *)
  let mat1 = Bigarray.Array1.of_array Float32 Bigarray.c_layout 
    [| 1.0; 2.0; 3.0; 4.0 |] in
  let mat2 = Bigarray.Array1.of_array Float32 Bigarray.c_layout 
    [| 5.0; 6.0; 7.0; 8.0 |] in
  
  let m1 = B.op_const_array ctx mat1 |> fun x -> B.op_reshape x [| 2; 2 |] in
  let m2 = B.op_const_array ctx mat2 |> fun x -> B.op_reshape x [| 2; 2 |] in
  
  let result_2d = B.op_cat [m1; m2] 0 in
  let result_2d_data = B.data result_2d in
  let view = B.view result_2d in
  let shape = View.shape view in
  
  Alcotest.(check (array int)) "cat_2d axis0 shape" [| 4; 2 |] shape;
  
  (* Expected:
     1 2
     3 4
     5 6
     7 8
  *)
  Alcotest.(check (float 0.001)) "cat_2d_axis0[0,0]" 1.0 result_2d_data.{0};
  Alcotest.(check (float 0.001)) "cat_2d_axis0[0,1]" 2.0 result_2d_data.{1};
  Alcotest.(check (float 0.001)) "cat_2d_axis0[1,0]" 3.0 result_2d_data.{2};
  Alcotest.(check (float 0.001)) "cat_2d_axis0[1,1]" 4.0 result_2d_data.{3};
  Alcotest.(check (float 0.001)) "cat_2d_axis0[2,0]" 5.0 result_2d_data.{4};
  Alcotest.(check (float 0.001)) "cat_2d_axis0[2,1]" 6.0 result_2d_data.{5};
  
  (* Test 2D concatenation along axis 1 *)
  let result_2d_axis1 = B.op_cat [m1; m2] 1 in
  let result_2d_axis1_data = B.data result_2d_axis1 in
  let view1 = B.view result_2d_axis1 in
  let shape1 = View.shape view1 in
  
  Alcotest.(check (array int)) "cat_2d axis1 shape" [| 2; 4 |] shape1;
  
  (* Expected:
     1 2 5 6
     3 4 7 8
  *)
  Alcotest.(check (float 0.001)) "cat_2d_axis1[0,0]" 1.0 result_2d_axis1_data.{0};
  Alcotest.(check (float 0.001)) "cat_2d_axis1[0,1]" 2.0 result_2d_axis1_data.{1};
  Alcotest.(check (float 0.001)) "cat_2d_axis1[0,2]" 5.0 result_2d_axis1_data.{2};
  Alcotest.(check (float 0.001)) "cat_2d_axis1[0,3]" 6.0 result_2d_axis1_data.{3};
  Alcotest.(check (float 0.001)) "cat_2d_axis1[1,0]" 3.0 result_2d_axis1_data.{4};
  Alcotest.(check (float 0.001)) "cat_2d_axis1[1,1]" 4.0 result_2d_axis1_data.{5};
  Alcotest.(check (float 0.001)) "cat_2d_axis1[1,2]" 7.0 result_2d_axis1_data.{6};
  Alcotest.(check (float 0.001)) "cat_2d_axis1[1,3]" 8.0 result_2d_axis1_data.{7}

let test_assign () =
  let ctx = () in

  (* Test 1D assignment *)
  let arr_dst = Bigarray.Array1.of_array Float32 Bigarray.c_layout 
    [| 1.0; 2.0; 3.0; 4.0; 5.0 |] in
  let arr_src = Bigarray.Array1.of_array Float32 Bigarray.c_layout 
    [| 10.0; 20.0; 30.0; 40.0; 50.0 |] in
  
  let dst = B.op_const_array ctx arr_dst in
  let src = B.op_const_array ctx arr_src in
  
  B.op_assign dst src;
  
  let dst_data = B.data dst in
  
  (* Check dst was modified *)
  Alcotest.(check (float 0.001)) "assign[0]" 10.0 dst_data.{0};
  Alcotest.(check (float 0.001)) "assign[1]" 20.0 dst_data.{1};
  Alcotest.(check (float 0.001)) "assign[2]" 30.0 dst_data.{2};
  Alcotest.(check (float 0.001)) "assign[3]" 40.0 dst_data.{3};
  Alcotest.(check (float 0.001)) "assign[4]" 50.0 dst_data.{4}

let test_threefry () =
  let ctx = () in

  (* Create key and counter arrays *)
  let key_arr = int32_array [| 42; 123 |] in
  let counter_arr = int32_array [| 0; 1; 2; 3 |] in
  
  let key = B.op_const_array ctx key_arr in
  let counter = B.op_const_array ctx counter_arr in
  
  let random = B.op_threefry key counter in
  let random_data = B.data random in
  
  (* Check that we got int32 output *)
  Alcotest.(check bool) "threefry dtype" true (B.dtype random = Dtype.Int32);
  
  (* Check shape matches counter *)
  let view = B.view random in
  let shape = View.shape view in
  Alcotest.(check (array int)) "threefry shape" [| 4 |] shape;
  
  (* Values should be pseudo-random but deterministic *)
  (* Just check they're not all the same or all zero *)
  let all_same = ref true in
  let all_zero = ref true in
  for i = 1 to 3 do
    if random_data.{i} <> random_data.{0} then all_same := false;
    if random_data.{i} <> 0l then all_zero := false
  done;
  Alcotest.(check bool) "threefry not all same" false !all_same;
  Alcotest.(check bool) "threefry not all zero" false !all_zero;
  
  (* Test with different counter values - should produce different results *)
  let counter2_arr = int32_array [| 4; 5; 6; 7 |] in
  let counter2 = B.op_const_array ctx counter2_arr in
  let random2 = B.op_threefry key counter2 in
  let random2_data = B.data random2 in
  
  (* Results should be different *)
  let any_different = ref false in
  for i = 0 to 3 do
    if random_data.{i} <> random2_data.{i} then any_different := true
  done;
  Alcotest.(check bool) "threefry different counters" true !any_different

let test_gather () =
  let ctx = () in

  (* Test 1D gather *)
  let data_arr = Bigarray.Array1.of_array Float32 Bigarray.c_layout 
    [| 10.0; 20.0; 30.0; 40.0; 50.0 |] in
  let indices_arr = int32_array [| 0; 2; 4; 1; 3 |] in
  
  let data = B.op_const_array ctx data_arr in
  let indices = B.op_const_array ctx indices_arr in
  
  let gathered = B.op_gather data indices 0 in
  let gathered_data = B.data gathered in
  
  (* Expected: [10.0, 30.0, 50.0, 20.0, 40.0] *)
  Alcotest.(check (float 0.001)) "gather_1d[0]" 10.0 gathered_data.{0};
  Alcotest.(check (float 0.001)) "gather_1d[1]" 30.0 gathered_data.{1};
  Alcotest.(check (float 0.001)) "gather_1d[2]" 50.0 gathered_data.{2};
  Alcotest.(check (float 0.001)) "gather_1d[3]" 20.0 gathered_data.{3};
  Alcotest.(check (float 0.001)) "gather_1d[4]" 40.0 gathered_data.{4};
  
  (* Test 2D gather along axis 0 *)
  let data_2d_arr = Bigarray.Array1.of_array Float32 Bigarray.c_layout 
    [| 1.0; 2.0; 3.0;    (* row 0 *)
       4.0; 5.0; 6.0;    (* row 1 *)
       7.0; 8.0; 9.0 |]  (* row 2 *)
  in
  let data_2d = B.op_const_array ctx data_2d_arr |> fun x -> B.op_reshape x [| 3; 3 |] in
  
  (* Gather rows 2, 0, 1 *)
  let indices_2d_arr = int32_array [| 2; 0; 1 |] in
  let indices_2d = B.op_const_array ctx indices_2d_arr |> fun x -> B.op_reshape x [| 3; 1 |] in
  
  let gathered_2d = B.op_gather data_2d indices_2d 0 in
  let gathered_2d_data = B.data gathered_2d in
  
  (* Expected shape: [3, 1] (indices shape) *)
  let view = B.view gathered_2d in
  let shape = View.shape view in
  Alcotest.(check (array int)) "gather_2d shape" [| 3; 1 |] shape;
  
  (* Expected values: [[7], [1], [4]] *)
  Alcotest.(check (float 0.001)) "gather_2d[0,0]" 7.0 gathered_2d_data.{0};
  Alcotest.(check (float 0.001)) "gather_2d[1,0]" 1.0 gathered_2d_data.{1};
  Alcotest.(check (float 0.001)) "gather_2d[2,0]" 4.0 gathered_2d_data.{2};
  
  (* Test float64 *)
  let data_f64_arr = Bigarray.Array1.of_array Float64 Bigarray.c_layout 
    [| 1.5; 2.5; 3.5; 4.5 |] in
  let data_f64 = B.op_const_array ctx data_f64_arr in
  let indices_f64 = B.op_const_array ctx (int32_array [| 3; 1; 0; 2 |]) in
  
  let gathered_f64 = B.op_gather data_f64 indices_f64 0 in
  let gathered_f64_data = B.data gathered_f64 in
  
  Alcotest.(check (float 0.001)) "gather_f64[0]" 4.5 gathered_f64_data.{0};
  Alcotest.(check (float 0.001)) "gather_f64[1]" 2.5 gathered_f64_data.{1};
  Alcotest.(check (float 0.001)) "gather_f64[2]" 1.5 gathered_f64_data.{2};
  Alcotest.(check (float 0.001)) "gather_f64[3]" 3.5 gathered_f64_data.{3}

let test_scatter () =
  let ctx = () in

  (* Test 1D scatter *)
  let template_arr = Bigarray.Array1.of_array Float32 Bigarray.c_layout 
    [| 0.0; 0.0; 0.0; 0.0; 0.0 |] in
  let indices_arr = int32_array [| 1; 3; 0; 4 |] in
  let updates_arr = Bigarray.Array1.of_array Float32 Bigarray.c_layout 
    [| 10.0; 20.0; 30.0; 40.0 |] in
  
  let template = B.op_const_array ctx template_arr in
  let indices = B.op_const_array ctx indices_arr in
  let updates = B.op_const_array ctx updates_arr in
  
  let scattered = B.op_scatter template indices updates 0 in
  let scattered_data = B.data scattered in
  
  (* Expected: [30.0, 10.0, 0.0, 20.0, 40.0] *)
  Alcotest.(check (float 0.001)) "scatter_1d[0]" 30.0 scattered_data.{0};
  Alcotest.(check (float 0.001)) "scatter_1d[1]" 10.0 scattered_data.{1};
  Alcotest.(check (float 0.001)) "scatter_1d[2]" 0.0 scattered_data.{2};
  Alcotest.(check (float 0.001)) "scatter_1d[3]" 20.0 scattered_data.{3};
  Alcotest.(check (float 0.001)) "scatter_1d[4]" 40.0 scattered_data.{4};
  
  (* Test with non-zero template *)
  let template2_arr = Bigarray.Array1.of_array Float32 Bigarray.c_layout 
    [| 1.0; 2.0; 3.0; 4.0 |] in
  let template2 = B.op_const_array ctx template2_arr in
  let indices2 = B.op_const_array ctx (int32_array [| 0; 2 |]) in
  let updates2 = B.op_const_array ctx 
    (Bigarray.Array1.of_array Float32 Bigarray.c_layout [| 100.0; 200.0 |]) in
  
  let scattered2 = B.op_scatter template2 indices2 updates2 0 in
  let scattered2_data = B.data scattered2 in
  
  (* Expected: [100.0, 2.0, 200.0, 4.0] *)
  Alcotest.(check (float 0.001)) "scatter_template[0]" 100.0 scattered2_data.{0};
  Alcotest.(check (float 0.001)) "scatter_template[1]" 2.0 scattered2_data.{1};
  Alcotest.(check (float 0.001)) "scatter_template[2]" 200.0 scattered2_data.{2};
  Alcotest.(check (float 0.001)) "scatter_template[3]" 4.0 scattered2_data.{3};
  
  (* Test float64 *)
  let template_f64 = B.op_const_array ctx 
    (Bigarray.Array1.of_array Float64 Bigarray.c_layout [| 0.0; 0.0; 0.0 |]) in
  let indices_f64 = B.op_const_array ctx (int32_array [| 2; 0 |]) in
  let updates_f64 = B.op_const_array ctx 
    (Bigarray.Array1.of_array Float64 Bigarray.c_layout [| 5.5; 7.7 |]) in
  
  let scattered_f64 = B.op_scatter template_f64 indices_f64 updates_f64 0 in
  let scattered_f64_data = B.data scattered_f64 in
  
  Alcotest.(check (float 0.001)) "scatter_f64[0]" 7.7 scattered_f64_data.{0};
  Alcotest.(check (float 0.001)) "scatter_f64[1]" 0.0 scattered_f64_data.{1};
  Alcotest.(check (float 0.001)) "scatter_f64[2]" 5.5 scattered_f64_data.{2}

let () =
  let open Alcotest in
  run "BLAS Backend"
    [
      ("scalar_ops", [ test_case "scalar addition" `Quick test_scalar_ops ]);
      ( "vector_ops",
        [
          test_case "vector operations" `Quick test_vector_ops;
          test_case "float64 operations" `Quick test_float64_ops;
        ] );
      ( "matrix_ops",
        [
          test_case "matrix multiplication" `Quick test_matrix_mul;
          test_case "batched matrix multiplication" `Quick test_batched_matmul;
        ] );
      ("reduction_ops", [ test_case "sum reduction" `Quick test_reduction ]);
      ( "view_ops",
        [
          test_case "contiguous copy" `Quick test_contiguous_copy;
          test_case "view operations" `Quick test_view_operations;
        ] );
      ( "arithmetic_ops",
        [
          test_case "division and sqrt" `Quick test_arithmetic_ops;
          test_case "power and fdiv" `Quick test_pow_and_fdiv;
          test_case "modulo" `Quick test_mod_op;
        ] );
      ("math_ops", [ test_case "sin, recip, log2, exp2" `Quick test_math_ops ]);
      ( "comparison_ops",
        [ test_case "cmplt, cmpne, max" `Quick test_comparison_ops ] );
      ( "strided_ops",
        [ test_case "non-contiguous operations" `Quick test_non_contiguous_ops ]
      );
      (* New operation tests *)
      ("idiv", [ test_case "integer division" `Quick test_idiv ]);
      ("bitwise", [ test_case "bitwise operations" `Quick test_bitwise_ops ]);
      ("where", [ test_case "where operation" `Quick test_where ]);
      ("reduce_max", [ test_case "reduce max" `Quick test_reduce_max ]);
      ("reduce_prod", [ test_case "reduce product" `Quick test_reduce_prod ]);
      ("pad", [ test_case "pad operation" `Quick test_pad ]);
      ("cast", [ test_case "type casting" `Quick test_cast ]);
      ("cat", [ test_case "concatenation" `Quick test_cat ]);
      ("assign", [ test_case "assignment" `Quick test_assign ]);
      ("threefry", [ test_case "random generation" `Quick test_threefry ]);
      ("gather", [ test_case "gather operation" `Quick test_gather ]);
      ("scatter", [ test_case "scatter operation" `Quick test_scatter ]);
    ]
