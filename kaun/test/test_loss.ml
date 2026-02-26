(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module Loss = Kaun.Loss

let flatten_f32 t =
  Rune.to_array (Rune.reshape [| -1 |] (Rune.cast Rune.float32 t))

let tensor_close ~eps ~expected ~actual =
  let xs = flatten_f32 expected in
  let ys = flatten_f32 actual in
  let nx = Array.length xs in
  let ny = Array.length ys in
  if nx <> ny then false
  else
    let ok = ref true in
    for i = 0 to nx - 1 do
      if abs_float (xs.(i) -. ys.(i)) > eps then ok := false
    done;
    !ok

let test_cross_entropy_known_value () =
  let logits = Rune.create Rune.float32 [| 1; 3 |] [| 2.0; 0.0; -2.0 |] in
  let labels = Rune.create Rune.float32 [| 1; 3 |] [| 1.0; 0.0; 0.0 |] in
  let expected = log (1.0 +. exp (-2.0) +. exp (-4.0)) in
  let actual = Loss.cross_entropy logits labels |> Rune.item [] in
  equal ~msg:"cross_entropy known value" (float 1e-6) expected actual

let test_sparse_matches_dense_2d () =
  let logits =
    Rune.create Rune.float32 [| 3; 4 |]
      [| 2.0; 0.1; -1.0; 0.3; 0.2; 1.7; -0.4; 0.9; -0.1; 0.8; 1.4; -2.0 |]
  in
  let indices = Rune.create Rune.int32 [| 3 |] [| 0l; 1l; 2l |] in
  let one_hot = Rune.cast Rune.float32 (Rune.one_hot ~num_classes:4 indices) in
  let dense = Loss.cross_entropy logits one_hot in
  let sparse = Loss.cross_entropy_sparse logits indices in
  equal ~msg:"sparse = dense (2d)" (float 1e-6) (Rune.item [] dense)
    (Rune.item [] sparse)

let test_sparse_matches_dense_nd () =
  let logits =
    Rune.create Rune.float32 [| 2; 2; 3 |]
      [| 0.7; -1.2; 0.5; 1.1; 0.2; -0.3; -0.8; 0.9; 0.4; 0.6; -0.5; 1.7 |]
  in
  let indices = Rune.create Rune.int32 [| 2; 2 |] [| 0l; 2l; 1l; 2l |] in
  let one_hot = Rune.cast Rune.float32 (Rune.one_hot ~num_classes:3 indices) in
  let dense = Loss.cross_entropy logits one_hot in
  let sparse = Loss.cross_entropy_sparse logits indices in
  equal ~msg:"sparse = dense (nd)" (float 1e-6) (Rune.item [] dense)
    (Rune.item [] sparse)

let test_cross_entropy_rejects_invalid_shapes () =
  raises_invalid_arg "Loss.cross_entropy: logits must have rank >= 1" (fun () ->
      let logits = Rune.scalar Rune.float32 0.0 in
      let labels = Rune.scalar Rune.float32 1.0 in
      ignore (Loss.cross_entropy logits labels));
  raises_invalid_arg
    "Loss.cross_entropy: labels rank mismatch (got 1, expected 2)" (fun () ->
      let logits = Rune.zeros Rune.float32 [| 2; 3 |] in
      let labels = Rune.zeros Rune.float32 [| 2 |] in
      ignore (Loss.cross_entropy logits labels));
  raises_invalid_arg
    "Loss.cross_entropy: labels shape mismatch at axis 0 (got 4, expected 2)"
    (fun () ->
      let logits = Rune.zeros Rune.float32 [| 2; 3 |] in
      let labels = Rune.zeros Rune.float32 [| 4; 3 |] in
      ignore (Loss.cross_entropy logits labels));
  raises_invalid_arg
    "Loss.cross_entropy: logits class dimension must be positive (got 0)"
    (fun () ->
      let logits = Rune.zeros Rune.float32 [| 2; 0 |] in
      let labels = Rune.zeros Rune.float32 [| 2; 0 |] in
      ignore (Loss.cross_entropy logits labels))

let test_sparse_rejects_non_integer_labels () =
  let logits = Rune.zeros Rune.float32 [| 2; 3 |] in
  let bad = Rune.zeros Rune.float32 [| 2 |] in
  let msg =
    Printf.sprintf "Loss.cross_entropy_sparse: expected integer labels, got %s"
      (Nx_core.Dtype.to_string Rune.float32)
  in
  raises_invalid_arg msg (fun () ->
      ignore (Loss.cross_entropy_sparse logits bad))

let test_sparse_rejects_shape_mismatch () =
  let logits_2d = Rune.zeros Rune.float32 [| 2; 3 |] in
  let bad_rank = Rune.zeros Rune.int32 [| 2; 1 |] in
  raises_invalid_arg
    "Loss.cross_entropy_sparse: labels rank mismatch (got 2, expected 1)"
    (fun () -> ignore (Loss.cross_entropy_sparse logits_2d bad_rank));
  let logits_3d = Rune.zeros Rune.float32 [| 2; 3; 4 |] in
  let bad_shape = Rune.zeros Rune.int32 [| 2; 5 |] in
  raises_invalid_arg
    "Loss.cross_entropy_sparse: labels shape mismatch at axis 1 (got 5, \
     expected 3)" (fun () ->
      ignore (Loss.cross_entropy_sparse logits_3d bad_shape))

let test_sparse_rejects_invalid_logits_shape () =
  raises_invalid_arg "Loss.cross_entropy_sparse: logits must have rank >= 1"
    (fun () ->
      let logits = Rune.scalar Rune.float32 0.0 in
      let labels = Rune.scalar Rune.int32 0l in
      ignore (Loss.cross_entropy_sparse logits labels));
  raises_invalid_arg
    "Loss.cross_entropy_sparse: logits class dimension must be positive (got 0)"
    (fun () ->
      let logits = Rune.zeros Rune.float32 [| 2; 0 |] in
      let labels = Rune.zeros Rune.int32 [| 2 |] in
      ignore (Loss.cross_entropy_sparse logits labels))

let test_binary_cross_entropy_logits_stable () =
  let logits =
    Rune.create Rune.float32 [| 5 |] [| 1000.0; -1000.0; 0.0; 50.0; -50.0 |]
  in
  let labels = Rune.create Rune.float32 [| 5 |] [| 1.0; 0.0; 1.0; 0.0; 1.0 |] in
  let loss = Loss.binary_cross_entropy logits labels |> Rune.item [] in
  let xs = [| 1000.0; -1000.0; 0.0; 50.0; -50.0 |] in
  let ys = [| 1.0; 0.0; 1.0; 0.0; 1.0 |] in
  let expected = ref 0.0 in
  for i = 0 to Array.length xs - 1 do
    let x = xs.(i) in
    let y = ys.(i) in
    let v = max x 0.0 -. (x *. y) +. log1p (exp (-.abs_float x)) in
    expected := !expected +. v
  done;
  expected := !expected /. float_of_int (Array.length xs);
  equal ~msg:"binary_cross_entropy stable value" (float 1e-6) !expected loss;
  equal ~msg:"binary_cross_entropy finite" bool true
    (match classify_float loss with FP_nan | FP_infinite -> false | _ -> true)

let test_binary_cross_entropy_rejects_invalid_shapes () =
  raises_invalid_arg
    "Loss.binary_cross_entropy: labels rank mismatch (got 1, expected 2)"
    (fun () ->
      let logits = Rune.zeros Rune.float32 [| 2; 1 |] in
      let labels = Rune.zeros Rune.float32 [| 2 |] in
      ignore (Loss.binary_cross_entropy logits labels));
  raises_invalid_arg
    "Loss.binary_cross_entropy: labels shape mismatch at axis 0 (got 3, \
     expected 2)" (fun () ->
      let logits = Rune.zeros Rune.float32 [| 2; 1 |] in
      let labels = Rune.zeros Rune.float32 [| 3; 1 |] in
      ignore (Loss.binary_cross_entropy logits labels))

let test_mse_gradient_exact () =
  let predictions =
    Rune.create Rune.float32 [| 2; 2 |] [| 0.5; -1.0; 2.0; 3.0 |]
  in
  let targets = Rune.create Rune.float32 [| 2; 2 |] [| 0.0; 1.0; 1.0; 2.0 |] in
  let grad = Rune.grad (fun p -> Loss.mse p targets) predictions in
  let expected =
    Rune.create Rune.float32 [| 2; 2 |]
      [|
        2.0 *. (0.5 -. 0.0) /. 4.0;
        2.0 *. (-1.0 -. 1.0) /. 4.0;
        2.0 *. (2.0 -. 1.0) /. 4.0;
        2.0 *. (3.0 -. 2.0) /. 4.0;
      |]
  in
  equal ~msg:"mse grad exact" bool true
    (tensor_close ~eps:1e-6 ~expected ~actual:grad)

let test_cross_entropy_sparse_dense_gradient_match () =
  let logits =
    Rune.create Rune.float32 [| 2; 3 |] [| 2.0; 1.0; 0.5; -1.0; 0.2; 0.0 |]
  in
  let indices = Rune.create Rune.int32 [| 2 |] [| 0l; 2l |] in
  let one_hot = Rune.cast Rune.float32 (Rune.one_hot ~num_classes:3 indices) in
  let dense_grad = Rune.grad (fun x -> Loss.cross_entropy x one_hot) logits in
  let sparse_grad =
    Rune.grad (fun x -> Loss.cross_entropy_sparse x indices) logits
  in
  equal ~msg:"cross_entropy sparse grad = dense grad" bool true
    (tensor_close ~eps:1e-6 ~expected:dense_grad ~actual:sparse_grad)

let test_regression_values () =
  let predictions = Rune.create Rune.float32 [| 3 |] [| 1.0; 4.0; 3.0 |] in
  let targets = Rune.create Rune.float32 [| 3 |] [| 1.0; 1.0; 2.0 |] in
  equal ~msg:"mse value" (float 1e-6) (10.0 /. 3.0)
    (Rune.item [] (Loss.mse predictions targets));
  equal ~msg:"mae value" (float 1e-6) (4.0 /. 3.0)
    (Rune.item [] (Loss.mae predictions targets))

let test_regression_broadcasting () =
  let predictions =
    Rune.create Rune.float32 [| 2; 3 |] [| 0.0; 1.0; 2.0; 3.0; 4.0; 5.0 |]
  in
  let targets = Rune.create Rune.float32 [| 1; 3 |] [| 1.0; 1.0; 1.0 |] in
  equal ~msg:"mse broadcast" (float 1e-6) (31.0 /. 6.0)
    (Rune.item [] (Loss.mse predictions targets));
  equal ~msg:"mae broadcast" (float 1e-6) (11.0 /. 6.0)
    (Rune.item [] (Loss.mae predictions targets))

let test_mae_gradient_exact () =
  let predictions =
    Rune.create Rune.float32 [| 2; 2 |] [| 2.0; -1.0; 4.0; 0.0 |]
  in
  let targets = Rune.create Rune.float32 [| 2; 2 |] [| 1.0; 1.0; 2.0; -3.0 |] in
  let grad = Rune.grad (fun p -> Loss.mae p targets) predictions in
  let expected =
    Rune.create Rune.float32 [| 2; 2 |] [| 0.25; -0.25; 0.25; 0.25 |]
  in
  equal ~msg:"mae grad exact" bool true
    (tensor_close ~eps:1e-6 ~expected ~actual:grad)

let () =
  run "Kaun.Loss"
    [
      group "cross-entropy"
        [
          test "cross entropy known value" test_cross_entropy_known_value;
          test "cross entropy rejects invalid shapes"
            test_cross_entropy_rejects_invalid_shapes;
          test "sparse matches dense (2d)" test_sparse_matches_dense_2d;
          test "sparse matches dense (nd)" test_sparse_matches_dense_nd;
          test "sparse rejects non-integer labels"
            test_sparse_rejects_non_integer_labels;
          test "sparse rejects shape mismatch"
            test_sparse_rejects_shape_mismatch;
          test "sparse rejects invalid logits shape"
            test_sparse_rejects_invalid_logits_shape;
          test "sparse/dense gradients match"
            test_cross_entropy_sparse_dense_gradient_match;
        ];
      group "binary"
        [
          test "binary cross entropy logits stable"
            test_binary_cross_entropy_logits_stable;
          test "binary cross entropy rejects invalid shapes"
            test_binary_cross_entropy_rejects_invalid_shapes;
        ];
      group "regression"
        [
          test "mse value + mae value" test_regression_values;
          test "mse/mae broadcasting" test_regression_broadcasting;
          test "mse gradient exact" test_mse_gradient_exact;
          test "mae gradient exact" test_mae_gradient_exact;
        ];
    ]
