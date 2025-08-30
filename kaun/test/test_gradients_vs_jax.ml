(** Test gradient computation against JAX/Flax reference implementation *)

open Rune

let check_gradient_match ~eps name expected_grad computed_grad =
  let expected_arr = to_array expected_grad in
  let computed_arr = to_array computed_grad in
  let max_diff = ref 0. in
  let max_rel_diff = ref 0. in
  
  Array.iteri (fun i exp_val ->
    let comp_val = computed_arr.(i) in
    let diff = Float.abs (exp_val -. comp_val) in
    let rel_diff = 
      if Float.abs exp_val > 1e-10 then 
        diff /. (Float.abs exp_val)
      else 
        diff
    in
    max_diff := Float.max !max_diff diff;
    max_rel_diff := Float.max !max_rel_diff rel_diff
  ) expected_arr;
  
  Alcotest.(check (float eps)) 
    (Printf.sprintf "%s max absolute difference" name) 
    0. !max_diff;
  
  if !max_diff > eps then
    Printf.printf "  FAIL: %s - max diff: %.6e, max rel diff: %.6e\n" 
      name !max_diff !max_rel_diff

(* Test 1: Simple matmul gradient *)
let test_matmul_gradient () =
  (* Test case: y = x @ w, loss = mean(y) *)
  
  (* Input: [2, 3] *)
  let x = create c float32 [|2; 3|] [|
    1.0; 2.0; 3.0;
    4.0; 5.0; 6.0;
  |] in
  
  (* Weight: [3, 4] *)
  let w = create c float32 [|3; 4|] [|
    0.1; 0.2; 0.3; 0.4;
    0.5; 0.6; 0.7; 0.8;
    0.9; 1.0; 1.1; 1.2;
  |] in
  
  (* Compute gradient *)
  let grad_w = Kaun.grad (fun w_param ->
    match w_param with
    | Kaun.Ptree.Tensor w_t ->
        let y = matmul x w_t in
        mean y
    | _ -> failwith "Expected tensor"
  ) (Kaun.Ptree.Tensor w) in
  
  let computed_grad = match grad_w with
    | Kaun.Ptree.Tensor t -> t
    | _ -> failwith "Expected tensor gradient"
  in
  
  (* Expected gradient from JAX *)
  let expected_grad = create c float32 [|3; 4|] [|
    0.625000; 0.625000; 0.625000; 0.625000;
    0.875000; 0.875000; 0.875000; 0.875000;
    1.125000; 1.125000; 1.125000; 1.125000;
  |] in
  
  check_gradient_match ~eps:1e-6 "matmul gradient" expected_grad computed_grad

(* Test 2: Add with broadcasting *)
let test_add_broadcast_gradient () =
  (* Test case: y = x + b (broadcast), loss = mean(y) *)
  
  let x = create c float32 [|2; 3|] [|
    1.0; 2.0; 3.0;
    4.0; 5.0; 6.0;
  |] in
  
  let b = create c float32 [|3|] [|0.1; 0.2; 0.3|] in
  
  (* Compute gradient w.r.t bias *)
  let grad_b = Kaun.grad (fun b_param ->
    match b_param with
    | Kaun.Ptree.Tensor b_t ->
        let b_expanded = reshape [|1; 3|] b_t in
        let y = add x b_expanded in
        mean y
    | _ -> failwith "Expected tensor"
  ) (Kaun.Ptree.Tensor b) in
  
  let computed_grad = match grad_b with
    | Kaun.Ptree.Tensor t -> t
    | _ -> failwith "Expected tensor gradient"
  in
  
  (* Expected gradient from JAX *)
  let expected_grad = create c float32 [|3|] [|
    0.333333; 0.333333; 0.333333;
  |] in
  
  check_gradient_match ~eps:1e-6 "add broadcast gradient" expected_grad computed_grad

(* Test 3: ReLU activation gradient *)
let test_relu_gradient () =
  let x = create c float32 [|2; 3|] [|
    -1.0; 0.0; 1.0;
    -2.0; 2.0; 3.0;
  |] in
  
  let grad_x = Kaun.grad (fun x_param ->
    match x_param with
    | Kaun.Ptree.Tensor x_t ->
        let y = relu x_t in
        mean y
    | _ -> failwith "Expected tensor"
  ) (Kaun.Ptree.Tensor x) in
  
  let computed_grad = match grad_x with
    | Kaun.Ptree.Tensor t -> t
    | _ -> failwith "Expected tensor gradient"
  in
  
  (* Expected gradient from JAX *)
  let expected_grad = create c float32 [|2; 3|] [|
    0.000000; 0.000000; 0.166667;
    0.000000; 0.166667; 0.166667;
  |] in
  
  check_gradient_match ~eps:1e-6 "relu gradient" expected_grad computed_grad

(* Test 4: GELU activation gradient *)
let test_gelu_gradient () =
  let x = create c float32 [|2; 2|] [|
    -1.0; 0.0;
    1.0; 2.0;
  |] in
  
  let grad_x = Kaun.grad (fun x_param ->
    match x_param with
    | Kaun.Ptree.Tensor x_t ->
        let y = gelu x_t in
        mean y
    | _ -> failwith "Expected tensor"
  ) (Kaun.Ptree.Tensor x) in
  
  let computed_grad = match grad_x with
    | Kaun.Ptree.Tensor t -> t
    | _ -> failwith "Expected tensor gradient"
  in
  
  (* Expected gradient from JAX *)
  let expected_grad = create c float32 [|2; 2|] [|
    -0.020829; 0.125000;
    0.270829; 0.271308;
  |] in
  
  check_gradient_match ~eps:1e-5 "gelu gradient" expected_grad computed_grad

(* Test 5: Simple Linear layer (matmul + add) *)
let test_linear_gradient () =
  let x = create c float32 [|2; 3|] [|
    1.0; 2.0; 3.0;
    4.0; 5.0; 6.0;
  |] in
  
  let w = create c float32 [|3; 4|] [|
    0.1; 0.2; 0.3; 0.4;
    0.5; 0.6; 0.7; 0.8;
    0.9; 1.0; 1.1; 1.2;
  |] in
  
  let b = create c float32 [|4|] [|0.01; 0.02; 0.03; 0.04|] in
  
  (* Compute gradients for linear: y = x @ w + b *)
  let params = Kaun.Ptree.record_of [
    ("weight", Kaun.Ptree.Tensor w);
    ("bias", Kaun.Ptree.Tensor b);
  ] in
  
  let grads = Kaun.grad (fun params ->
    match params with
    | Kaun.Ptree.Record fields ->
        let w_t = match Kaun.Ptree.Record.find "weight" fields with
          | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected weight tensor"
        in
        let b_t = match Kaun.Ptree.Record.find "bias" fields with
          | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected bias tensor"
        in
        let y = matmul x w_t in
        let y = add y (reshape [|1; 4|] b_t) in
        mean y
    | _ -> failwith "Expected record"
  ) params in
  
  let grad_w = match Kaun.Ptree.get_by_path "weight" grads with
    | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected weight gradient"
  in
  let grad_b = match Kaun.Ptree.get_by_path "bias" grads with
    | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected bias gradient"
  in
  
  (* Expected gradients from JAX *)
  let expected_grad_w = create c float32 [|3; 4|] [|
    0.625000; 0.625000; 0.625000; 0.625000;
    0.875000; 0.875000; 0.875000; 0.875000;
    1.125000; 1.125000; 1.125000; 1.125000;
  |] in
  
  let expected_grad_b = create c float32 [|4|] [|
    0.250000; 0.250000; 0.250000; 0.250000;
  |] in
  
  check_gradient_match ~eps:1e-6 "linear weight gradient" expected_grad_w grad_w;
  check_gradient_match ~eps:1e-6 "linear bias gradient" expected_grad_b grad_b

(* Test 6: Two-layer MLP *)
let test_mlp_gradient () =
  (* Test case: two-layer MLP with ReLU activation *)
  
  let x = create c float32 [|2; 2|] [|
    1.0; 2.0;
    3.0; 4.0;
  |] in
  
  let w1 = create c float32 [|2; 3|] [|
    0.1; 0.2; 0.3;
    0.4; 0.5; 0.6;
  |] in
  
  let b1 = create c float32 [|3|] [|0.01; 0.02; 0.03|] in
  
  let w2 = create c float32 [|3; 2|] [|
    0.7; 0.8;
    0.9; 1.0;
    1.1; 1.2;
  |] in
  
  let b2 = create c float32 [|2|] [|0.04; 0.05|] in
  
  (* Compute gradients for MLP *)
  let params = Kaun.Ptree.record_of [
    ("w1", Kaun.Ptree.Tensor w1);
    ("b1", Kaun.Ptree.Tensor b1);
    ("w2", Kaun.Ptree.Tensor w2);
    ("b2", Kaun.Ptree.Tensor b2);
  ] in
  
  let grads = Kaun.grad (fun params ->
    match params with
    | Kaun.Ptree.Record fields ->
        let w1_t = match Kaun.Ptree.Record.find "w1" fields with
          | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected w1 tensor"
        in
        let b1_t = match Kaun.Ptree.Record.find "b1" fields with
          | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected b1 tensor"
        in
        let w2_t = match Kaun.Ptree.Record.find "w2" fields with
          | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected w2 tensor"
        in
        let b2_t = match Kaun.Ptree.Record.find "b2" fields with
          | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected b2 tensor"
        in
        (* First layer *)
        let h = matmul x w1_t in
        let h = add h (reshape [|1; 3|] b1_t) in
        let h = relu h in
        (* Second layer *)
        let y = matmul h w2_t in
        let y = add y (reshape [|1; 2|] b2_t) in
        mean y
    | _ -> failwith "Expected record"
  ) params in
  
  let grad_w1 = match Kaun.Ptree.get_by_path "w1" grads with
    | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected w1 gradient"
  in
  let grad_b1 = match Kaun.Ptree.get_by_path "b1" grads with
    | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected b1 gradient"
  in
  let grad_w2 = match Kaun.Ptree.get_by_path "w2" grads with
    | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected w2 gradient"
  in
  let grad_b2 = match Kaun.Ptree.get_by_path "b2" grads with
    | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected b2 gradient"
  in
  
  (* Expected gradients from JAX *)
  let expected_grad_w1 = create c float32 [|2; 3|] [|
    1.500000; 1.900000; 2.300000;
    2.250000; 2.850000; 3.450000;
  |] in
  
  let expected_grad_b1 = create c float32 [|3|] [|
    0.750000; 0.950000; 1.150000;
  |] in
  
  let expected_grad_w2 = create c float32 [|3; 2|] [|
    0.705000; 0.705000;
    0.960000; 0.960000;
    1.215000; 1.215000;
  |] in
  
  let expected_grad_b2 = create c float32 [|2|] [|
    0.500000; 0.500000;
  |] in
  
  check_gradient_match ~eps:1e-5 "mlp w1 gradient" expected_grad_w1 grad_w1;
  check_gradient_match ~eps:1e-5 "mlp b1 gradient" expected_grad_b1 grad_b1;
  check_gradient_match ~eps:1e-5 "mlp w2 gradient" expected_grad_w2 grad_w2;
  check_gradient_match ~eps:1e-5 "mlp b2 gradient" expected_grad_b2 grad_b2

(* Test 7: Reduction operations *)
let test_reduction_gradients () =
  let x = create c float32 [|2; 3|] [|
    1.0; 2.0; 3.0;
    4.0; 5.0; 6.0;
  |] in
  
  (* Test sum with axis=0 *)
  let grad_sum_axis0 = Kaun.grad (fun x_param ->
    match x_param with
    | Kaun.Ptree.Tensor x_t ->
        let y = sum x_t ~axes:[|0|] in
        sum y
    | _ -> failwith "Expected tensor"
  ) (Kaun.Ptree.Tensor x) in
  
  let computed_grad_axis0 = match grad_sum_axis0 with
    | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected tensor gradient"
  in
  
  let expected_grad_axis0 = create c float32 [|2; 3|] [|
    1.000000; 1.000000; 1.000000;
    1.000000; 1.000000; 1.000000;
  |] in
  
  check_gradient_match ~eps:1e-6 "sum axis=0 gradient" expected_grad_axis0 computed_grad_axis0;
  
  (* Test sum with axis=1 *)
  let grad_sum_axis1 = Kaun.grad (fun x_param ->
    match x_param with
    | Kaun.Ptree.Tensor x_t ->
        let y = sum x_t ~axes:[|1|] in
        sum y
    | _ -> failwith "Expected tensor"
  ) (Kaun.Ptree.Tensor x) in
  
  let computed_grad_axis1 = match grad_sum_axis1 with
    | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected tensor gradient"
  in
  
  let expected_grad_axis1 = create c float32 [|2; 3|] [|
    1.000000; 1.000000; 1.000000;
    1.000000; 1.000000; 1.000000;
  |] in
  
  check_gradient_match ~eps:1e-6 "sum axis=1 gradient" expected_grad_axis1 computed_grad_axis1;
  
  (* Test mean with keepdims *)
  let grad_mean_keepdims = Kaun.grad (fun x_param ->
    match x_param with
    | Kaun.Ptree.Tensor x_t ->
        let y = mean x_t ~axes:[|1|] ~keepdims:true in
        sum y
    | _ -> failwith "Expected tensor"
  ) (Kaun.Ptree.Tensor x) in
  
  let computed_grad_keepdims = match grad_mean_keepdims with
    | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected tensor gradient"
  in
  
  let expected_grad_keepdims = create c float32 [|2; 3|] [|
    0.333333; 0.333333; 0.333333;
    0.333333; 0.333333; 0.333333;
  |] in
  
  check_gradient_match ~eps:1e-6 "mean keepdims gradient" expected_grad_keepdims computed_grad_keepdims

(* Test 8: More activation functions *)
let test_activation_gradients () =
  let x = create c float32 [|2; 3|] [|
    -2.0; -1.0; 0.0;
    1.0; 2.0; 3.0;
  |] in
  
  (* Test sigmoid *)
  let grad_sigmoid = Kaun.grad (fun x_param ->
    match x_param with
    | Kaun.Ptree.Tensor x_t ->
        let y = sigmoid x_t in
        mean y
    | _ -> failwith "Expected tensor"
  ) (Kaun.Ptree.Tensor x) in
  
  let computed_grad_sigmoid = match grad_sigmoid with
    | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected tensor gradient"
  in
  
  let expected_grad_sigmoid = create c float32 [|2; 3|] [|
    0.017499; 0.032769; 0.041667;
    0.032769; 0.017499; 0.007529;
  |] in
  
  check_gradient_match ~eps:1e-5 "sigmoid gradient" expected_grad_sigmoid computed_grad_sigmoid;
  
  (* Test tanh *)
  let grad_tanh = Kaun.grad (fun x_param ->
    match x_param with
    | Kaun.Ptree.Tensor x_t ->
        let y = tanh x_t in
        mean y
    | _ -> failwith "Expected tensor"
  ) (Kaun.Ptree.Tensor x) in
  
  let computed_grad_tanh = match grad_tanh with
    | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected tensor gradient"
  in
  
  let expected_grad_tanh = create c float32 [|2; 3|] [|
    0.011775; 0.069996; 0.166667;
    0.069996; 0.011775; 0.001644;
  |] in
  
  check_gradient_match ~eps:1e-5 "tanh gradient" expected_grad_tanh computed_grad_tanh

(* Test 9: Softmax *)
let test_softmax_gradient () =
  let x = create c float32 [|2; 3|] [|
    1.0; 2.0; 3.0;
    1.0; 3.0; 2.0;
  |] in
  
  let grad_softmax = Kaun.grad (fun x_param ->
    match x_param with
    | Kaun.Ptree.Tensor x_t ->
        let y = softmax x_t ~axes:[|-1|] in
        mean y
    | _ -> failwith "Expected tensor"
  ) (Kaun.Ptree.Tensor x) in
  
  let computed_grad = match grad_softmax with
    | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected tensor gradient"
  in
  
  (* Note: Softmax gradients are very small, essentially 0 *)
  let expected_grad = create c float32 [|2; 3|] [|
    0.000000; 0.000000; 0.000000;
    0.000000; 0.000000; 0.000000;
  |] in
  
  check_gradient_match ~eps:1e-6 "softmax gradient" expected_grad computed_grad

(* Test 10: Transpose and reshape *)
let test_transpose_reshape_gradients () =
  let x = create c float32 [|2; 3|] [|
    1.0; 2.0; 3.0;
    4.0; 5.0; 6.0;
  |] in
  
  (* Test transpose *)
  let grad_transpose = Kaun.grad (fun x_param ->
    match x_param with
    | Kaun.Ptree.Tensor x_t ->
        let y = transpose x_t in
        mean y
    | _ -> failwith "Expected tensor"
  ) (Kaun.Ptree.Tensor x) in
  
  let computed_grad_transpose = match grad_transpose with
    | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected tensor gradient"
  in
  
  let expected_grad_transpose = create c float32 [|2; 3|] [|
    0.166667; 0.166667; 0.166667;
    0.166667; 0.166667; 0.166667;
  |] in
  
  check_gradient_match ~eps:1e-6 "transpose gradient" expected_grad_transpose computed_grad_transpose;
  
  (* Test reshape *)
  let grad_reshape = Kaun.grad (fun x_param ->
    match x_param with
    | Kaun.Ptree.Tensor x_t ->
        let y = reshape [|3; 2|] x_t in
        mean y
    | _ -> failwith "Expected tensor"
  ) (Kaun.Ptree.Tensor x) in
  
  let computed_grad_reshape = match grad_reshape with
    | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected tensor gradient"
  in
  
  let expected_grad_reshape = create c float32 [|2; 3|] [|
    0.166667; 0.166667; 0.166667;
    0.166667; 0.166667; 0.166667;
  |] in
  
  check_gradient_match ~eps:1e-6 "reshape gradient" expected_grad_reshape computed_grad_reshape

(* Test 11: Element-wise operations *)
let test_elementwise_gradients () =
  let x = create c float32 [|2; 2|] [|
    1.0; 2.0;
    3.0; 4.0;
  |] in
  
  (* Test exp *)
  let grad_exp = Kaun.grad (fun x_param ->
    match x_param with
    | Kaun.Ptree.Tensor x_t ->
        let y = exp x_t in
        mean y
    | _ -> failwith "Expected tensor"
  ) (Kaun.Ptree.Tensor x) in
  
  let computed_grad_exp = match grad_exp with
    | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected tensor gradient"
  in
  
  (* Gradient of exp is exp(x) * grad_output / n *)
  let expected_grad_exp = create c float32 [|2; 2|] [|
    0.679570; 1.847264;  (* exp(1)/4, exp(2)/4 *)
    5.021384; 13.649537; (* exp(3)/4, exp(4)/4 *)
  |] in
  
  check_gradient_match ~eps:1e-4 "exp gradient" expected_grad_exp computed_grad_exp;
  
  (* Test log *)
  let grad_log = Kaun.grad (fun x_param ->
    match x_param with
    | Kaun.Ptree.Tensor x_t ->
        let y = log x_t in
        mean y
    | _ -> failwith "Expected tensor"
  ) (Kaun.Ptree.Tensor x) in
  
  let computed_grad_log = match grad_log with
    | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected tensor gradient"
  in
  
  (* Gradient of log is 1/x * grad_output / n *)
  let expected_grad_log = create c float32 [|2; 2|] [|
    0.250000; 0.125000;  (* 1/(1*4), 1/(2*4) *)
    0.083333; 0.062500;  (* 1/(3*4), 1/(4*4) *)
  |] in
  
  check_gradient_match ~eps:1e-6 "log gradient" expected_grad_log computed_grad_log;
  
  (* Test sqrt *)
  let grad_sqrt = Kaun.grad (fun x_param ->
    match x_param with
    | Kaun.Ptree.Tensor x_t ->
        let y = sqrt x_t in
        mean y
    | _ -> failwith "Expected tensor"
  ) (Kaun.Ptree.Tensor x) in
  
  let computed_grad_sqrt = match grad_sqrt with
    | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected tensor gradient"
  in
  
  (* Gradient of sqrt is 0.5/sqrt(x) * grad_output / n *)
  let expected_grad_sqrt = create c float32 [|2; 2|] [|
    0.125000; 0.088388;  (* 0.5/(sqrt(1)*4), 0.5/(sqrt(2)*4) *)
    0.072169; 0.062500;  (* 0.5/(sqrt(3)*4), 0.5/(sqrt(4)*4) *)
  |] in
  
  check_gradient_match ~eps:1e-5 "sqrt gradient" expected_grad_sqrt computed_grad_sqrt

(* Test 12: Concatenation *)
let test_concat_gradient () =
  let x1 = create c float32 [|2; 2|] [|
    1.0; 2.0;
    3.0; 4.0;
  |] in
  
  let x2 = create c float32 [|2; 2|] [|
    5.0; 6.0;
    7.0; 8.0;
  |] in
  
  (* Test concatenation along axis 1 *)
  let params = Kaun.Ptree.record_of [
    ("x1", Kaun.Ptree.Tensor x1);
    ("x2", Kaun.Ptree.Tensor x2);
  ] in
  
  let grads = Kaun.grad (fun params ->
    match params with
    | Kaun.Ptree.Record fields ->
        let x1_t = match Kaun.Ptree.Record.find "x1" fields with
          | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected x1 tensor"
        in
        let x2_t = match Kaun.Ptree.Record.find "x2" fields with
          | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected x2 tensor"
        in
        let y = concatenate [x1_t; x2_t] ~axis:1 in
        mean y
    | _ -> failwith "Expected record"
  ) params in
  
  let grad_x1 = match Kaun.Ptree.get_by_path "x1" grads with
    | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected x1 gradient"
  in
  let grad_x2 = match Kaun.Ptree.get_by_path "x2" grads with
    | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected x2 gradient"
  in
  
  (* Gradients should be uniform 1/8 for all elements *)
  let expected_grad = create c float32 [|2; 2|] [|
    0.125000; 0.125000;
    0.125000; 0.125000;
  |] in
  
  check_gradient_match ~eps:1e-6 "concat x1 gradient" expected_grad grad_x1;
  check_gradient_match ~eps:1e-6 "concat x2 gradient" expected_grad grad_x2

(* Test 13: Scaled Dot-Product Attention *)
let test_attention_gradient () =
  (* Small example: batch=1, seq_len=3, d_k=4 *)
  let q = create c float32 [|1; 3; 4|] [|
    0.1; 0.2; 0.3; 0.4;
    0.5; 0.6; 0.7; 0.8;
    0.9; 1.0; 1.1; 1.2;
  |] in
  
  let k = create c float32 [|1; 3; 4|] [|
    0.2; 0.3; 0.4; 0.5;
    0.6; 0.7; 0.8; 0.9;
    1.0; 1.1; 1.2; 1.3;
  |] in
  
  let v = create c float32 [|1; 3; 4|] [|
    0.1; 0.2; 0.3; 0.4;
    0.5; 0.6; 0.7; 0.8;
    0.9; 1.0; 1.1; 1.2;
  |] in
  
  let params = Kaun.Ptree.record_of [
    ("q", Kaun.Ptree.Tensor q);
    ("k", Kaun.Ptree.Tensor k);
    ("v", Kaun.Ptree.Tensor v);
  ] in
  
  let grads = Kaun.grad (fun params ->
    match params with
    | Kaun.Ptree.Record fields ->
        let q_t = match Kaun.Ptree.Record.find "q" fields with
          | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected q tensor"
        in
        let k_t = match Kaun.Ptree.Record.find "k" fields with
          | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected k tensor"
        in
        let v_t = match Kaun.Ptree.Record.find "v" fields with
          | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected v tensor"
        in
        (* Scaled dot-product attention *)
        let shape_q = shape q_t in
        let d_k = Float.of_int shape_q.(Array.length shape_q - 1) in
        let scale = scalar_like q_t (1.0 /. Float.sqrt d_k) in
        let scores = matmul q_t (transpose ~axes:[|0; 2; 1|] k_t) in
        let scores = mul scores scale in
        let attention_weights = softmax scores ~axes:[|-1|] in
        let output = matmul attention_weights v_t in
        mean output
    | _ -> failwith "Expected record"
  ) params in
  
  let grad_q = match Kaun.Ptree.get_by_path "q" grads with
    | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected q gradient"
  in
  let grad_k = match Kaun.Ptree.get_by_path "k" grads with
    | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected k gradient"
  in
  let grad_v = match Kaun.Ptree.get_by_path "v" grads with
    | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected v gradient"
  in
  
  (* Expected gradients from JAX - reshape from [1,3,4] to [3,4] for comparison *)
  let expected_grad_q = create c float32 [|3; 4|] [|
    0.017427; 0.017427; 0.017427; 0.017427;
    0.015590; 0.015590; 0.015590; 0.015590;
    0.012809; 0.012809; 0.012809; 0.012809;
  |] in
  
  let expected_grad_k = create c float32 [|3; 4|] [|
    -0.020475; -0.025273; -0.030071; -0.034870;
    -0.011716; -0.013577; -0.015437; -0.017297;
    0.032191; 0.038850; 0.045508; 0.052167;
  |] in
  
  let expected_grad_v = create c float32 [|3; 4|] [|
    0.047161; 0.047161; 0.047161; 0.047161;
    0.075078; 0.075078; 0.075078; 0.075078;
    0.127761; 0.127761; 0.127761; 0.127761;
  |] in
  
  (* Reshape gradients from [1,3,4] to [3,4] for comparison *)
  let grad_q_reshaped = reshape [|3; 4|] grad_q in
  let grad_k_reshaped = reshape [|3; 4|] grad_k in
  let grad_v_reshaped = reshape [|3; 4|] grad_v in
  
  check_gradient_match ~eps:1e-6 "attention q gradient" expected_grad_q grad_q_reshaped;
  check_gradient_match ~eps:1e-6 "attention k gradient" expected_grad_k grad_k_reshaped;
  check_gradient_match ~eps:1e-6 "attention v gradient" expected_grad_v grad_v_reshaped

(* Test 14: Loss Functions *)
let test_loss_functions () =
  let predictions = create c float32 [|2; 3|] [|
    0.7; 0.2; 0.1;
    0.1; 0.8; 0.1;
  |] in
  
  let targets = create c float32 [|2; 3|] [|
    1.0; 0.0; 0.0;
    0.0; 1.0; 0.0;
  |] in
  
  (* Test MSE gradient *)
  let grad_mse = Kaun.grad (fun pred ->
    match pred with
    | Kaun.Ptree.Tensor pred_t ->
        let diff = sub pred_t targets in
        let squared = mul diff diff in
        mean squared
    | _ -> failwith "Expected tensor"
  ) (Kaun.Ptree.Tensor predictions) in
  
  let computed_grad_mse = match grad_mse with
    | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected tensor gradient"
  in
  
  let expected_grad_mse = create c float32 [|2; 3|] [|
    -0.100000; 0.066667; 0.033333;
    0.033333; -0.066667; 0.033333;
  |] in
  
  check_gradient_match ~eps:1e-5 "MSE gradient" expected_grad_mse computed_grad_mse;
  
  (* Test MAE gradient *)
  let grad_mae = Kaun.grad (fun pred ->
    match pred with
    | Kaun.Ptree.Tensor pred_t ->
        let diff = sub pred_t targets in
        let abs_diff = abs diff in
        mean abs_diff
    | _ -> failwith "Expected tensor"
  ) (Kaun.Ptree.Tensor predictions) in
  
  let computed_grad_mae = match grad_mae with
    | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected tensor gradient"
  in
  
  let expected_grad_mae = create c float32 [|2; 3|] [|
    -0.166667; 0.166667; 0.166667;
    0.166667; -0.166667; 0.166667;
  |] in
  
  check_gradient_match ~eps:1e-5 "MAE gradient" expected_grad_mae computed_grad_mae

(* Test 15: Cross-Entropy Loss *)
let test_cross_entropy_gradient () =
  let logits = create c float32 [|2; 3|] [|
    2.0; 1.0; 0.1;
    0.1; 2.5; 0.3;
  |] in
  
  let grad_ce = Kaun.grad (fun logits_param ->
    match logits_param with
    | Kaun.Ptree.Tensor logits_t ->
        (* Compute log_softmax *)
        let max_logits = max logits_t ~axes:[|-1|] ~keepdims:true in
        let shifted = sub logits_t max_logits in
        let exp_shifted = exp shifted in
        let sum_exp = sum exp_shifted ~axes:[|-1|] ~keepdims:true in
        let log_sum_exp = log sum_exp in
        let log_probs = sub shifted log_sum_exp in
        
        (* Create one-hot labels - simplified for this test *)
        let one_hot = create c float32 [|2; 3|] [|
          1.0; 0.0; 0.0;  (* label 0 *)
          0.0; 1.0; 0.0;  (* label 1 *)
        |] in
        
        (* Compute cross-entropy loss *)
        let ce = mul one_hot log_probs in
        let ce_sum = sum ce ~axes:[|-1|] in
        neg (mean ce_sum)
    | _ -> failwith "Expected tensor"
  ) (Kaun.Ptree.Tensor logits) in
  
  let computed_grad = match grad_ce with
    | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected tensor gradient"
  in
  
  let expected_grad = create c float32 [|2; 3|] [|
    -0.170499; 0.121216; 0.049283;
    0.037751; -0.083861; 0.046110;
  |] in
  
  check_gradient_match ~eps:1e-4 "cross-entropy gradient" expected_grad computed_grad

(* Test 16: BatchNorm *)
let test_batchnorm_gradient () =
  let x = create c float32 [|3; 3|] [|
    1.0; 2.0; 3.0;
    4.0; 5.0; 6.0;
    7.0; 8.0; 9.0;
  |] in
  
  let gamma = create c float32 [|3|] [|1.0; 1.0; 1.0|] in
  let beta = create c float32 [|3|] [|0.0; 0.0; 0.0|] in
  
  let params = Kaun.Ptree.record_of [
    ("x", Kaun.Ptree.Tensor x);
    ("gamma", Kaun.Ptree.Tensor gamma);
    ("beta", Kaun.Ptree.Tensor beta);
  ] in
  
  let grads = Kaun.grad (fun params ->
    match params with
    | Kaun.Ptree.Record fields ->
        let x_t = match Kaun.Ptree.Record.find "x" fields with
          | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected x tensor"
        in
        let gamma_t = match Kaun.Ptree.Record.find "gamma" fields with
          | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected gamma tensor"
        in
        let beta_t = match Kaun.Ptree.Record.find "beta" fields with
          | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected beta tensor"
        in
        (* BatchNorm computation *)
        let eps = scalar_like x_t 1e-5 in
        let mean_val = mean x_t ~axes:[|0|] ~keepdims:true in
        let x_centered = sub x_t mean_val in
        let var = mean (mul x_centered x_centered) ~axes:[|0|] ~keepdims:true in
        let std = sqrt (add var eps) in
        let x_normalized = div x_centered std in
        let gamma_expanded = reshape [|1; 3|] gamma_t in
        let beta_expanded = reshape [|1; 3|] beta_t in
        let y = add (mul gamma_expanded x_normalized) beta_expanded in
        mean y
    | _ -> failwith "Expected record"
  ) params in
  
  let grad_beta = match Kaun.Ptree.get_by_path "beta" grads with
    | Kaun.Ptree.Tensor t -> t | _ -> failwith "Expected beta gradient"
  in
  
  (* For BatchNorm, beta gradient is simpler to verify *)
  let expected_grad_beta = create c float32 [|3|] [|
    0.333333; 0.333333; 0.333333;
  |] in
  
  check_gradient_match ~eps:1e-5 "batchnorm beta gradient" expected_grad_beta grad_beta

(* Test suite *)
let gradient_tests = [
  "matmul gradient", `Quick, test_matmul_gradient;
  "add broadcast gradient", `Quick, test_add_broadcast_gradient;
  "relu gradient", `Quick, test_relu_gradient;
  "gelu gradient", `Quick, test_gelu_gradient;
  "linear gradient", `Quick, test_linear_gradient;
  "mlp gradient", `Quick, test_mlp_gradient;
  "reduction gradients", `Quick, test_reduction_gradients;
  "activation gradients", `Quick, test_activation_gradients;
  "softmax gradient", `Quick, test_softmax_gradient;
  "transpose/reshape gradients", `Quick, test_transpose_reshape_gradients;
  "elementwise gradients", `Quick, test_elementwise_gradients;
  "concat gradient", `Quick, test_concat_gradient;
  "attention gradient", `Quick, test_attention_gradient;
  "loss functions", `Quick, test_loss_functions;
  "cross-entropy gradient", `Quick, test_cross_entropy_gradient;
  "batchnorm gradient", `Quick, test_batchnorm_gradient;
]

let () =
  Alcotest.run "Gradient vs JAX" [
    "basic gradients", gradient_tests;
  ]