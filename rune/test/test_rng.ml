open Rune
module A = Alcotest

let test_key_creation () =
  let key1 = Rng.key 42 in
  let key2 = Rng.key 42 in
  let key3 = Rng.key 43 in
  A.check A.int "same seed produces same key" (Rng.to_int key1)
    (Rng.to_int key2);
  A.check A.bool "different seeds produce different keys" true
    (Rng.to_int key1 <> Rng.to_int key3)

let test_key_splitting () =
  let key = Rng.key 42 in
  let keys = Rng.split key in
  A.check A.int "default split produces 2 keys" 2 (Array.length keys);

  let keys3 = Rng.split ~n:3 key in
  A.check A.int "split with n=3 produces 3 keys" 3 (Array.length keys3);

  (* Check keys are different *)
  A.check A.bool "split keys are different" true
    (Rng.to_int keys.(0) <> Rng.to_int keys.(1));

  (* Check deterministic *)
  let keys2 = Rng.split key in
  A.check A.int "split is deterministic"
    (Rng.to_int keys.(0))
    (Rng.to_int keys2.(0))

let test_fold_in () =
  let key = Rng.key 42 in
  let key1 = Rng.fold_in key 1 in
  let key2 = Rng.fold_in key 2 in
  let key1_again = Rng.fold_in key 1 in

  A.check A.bool "fold_in with different data produces different keys" true
    (Rng.to_int key1 <> Rng.to_int key2);
  A.check A.int "fold_in is deterministic" (Rng.to_int key1)
    (Rng.to_int key1_again)

let test_uniform () =
  let key = Rng.key 42 in
  let shape = [| 3; 4 |] in
  let t = Rng.uniform key Float32 shape in

  A.check (A.array A.int) "uniform produces correct shape" shape (Rune.shape t);

  (* Check values are in [0, 1) *)
  let values = Rune.to_array (Rune.reshape [| 12 |] t) in
  Array.iter
    (fun v ->
      A.check A.bool "uniform values in [0, 1)" true (v >= 0. && v < 1.))
    values;

  (* Check deterministic *)
  let t2 = Rng.uniform key Float32 shape in
  let is_equal = Rune.all (Rune.equal t t2) in
  let is_equal_val = Rune.to_array is_equal in
  A.check A.bool "uniform is deterministic" true (is_equal_val.(0) > 0)

let test_normal () =
  let key = Rng.key 42 in
  let shape = [| 100 |] in
  let t = Rng.normal key Float32 shape in

  A.check (A.array A.int) "normal produces correct shape" shape (Rune.shape t);

  (* Check roughly normal distribution (mean ~0, std ~1) *)
  let values = Rune.to_array t in
  let mean =
    Array.fold_left ( +. ) 0. values /. float_of_int (Array.length values)
  in
  let variance =
    Array.fold_left (fun acc v -> acc +. ((v -. mean) ** 2.)) 0. values
    /. float_of_int (Array.length values)
  in
  let std = Stdlib.sqrt variance in

  A.check (A.float 0.2) "normal mean ~0" 0. mean;
  A.check (A.float 0.3) "normal std ~1" 1. std

let test_randint () =
  let key = Rng.key 42 in
  let shape = [| 10 |] in
  let t = Rng.randint key ~min:5 ~max:15 shape in

  A.check (A.array A.int) "randint produces correct shape" shape (Rune.shape t);

  (* Check values are in [min, max) *)
  let values = Rune.to_array t in
  Array.iter
    (fun v ->
      let v = Int32.to_int v in
      A.check A.bool "randint values in [5, 15)" true (v >= 5 && v < 15))
    values

let test_bernoulli () =
  let key = Rng.key 42 in
  let shape = [| 1000 |] in
  let p = 0.3 in
  let t = Rng.bernoulli key ~p shape in

  A.check (A.array A.int) "bernoulli produces correct shape" shape
    (Rune.shape t);

  (* Check proportion roughly matches p *)
  let values = Rune.to_array t in
  let ones =
    Array.fold_left (fun acc v -> acc + if v > 0 then 1 else 0) 0 values
  in
  let prop = float_of_int ones /. float_of_int (Array.length values) in
  A.check (A.float 0.05) "bernoulli proportion ~p" p prop

(* TODO: Enable when argsort works with lazy views let test_permutation () = let
   Rng.permutation key n in

   A.check (A.array A.int) "permutation has correct shape" [| n |] (Rune.shape
   perm);

   (* Check it's a valid permutation *) let values = Rune.to_array perm |>
   Array.map Int32.to_int in let sorted = Array.copy values in Array.sort
   compare sorted; Array.iteri (fun i v -> A.check A.int (Printf.sprintf
   "permutation contains %d" i) i v) sorted *)

(* TODO: Enable when argsort works with lazy views let test_shuffle () = let key
   let x = Rune.reshape [| 10; 1 |] x in let shuffled = Rng.shuffle key x in

   A.check (A.array A.int) "shuffle preserves shape" (Rune.shape x) (Rune.shape
   shuffled);

   (* Check all elements are still there *) let original = Rune.to_array
   (Rune.reshape [| 10 |] x) in let shuffled_vals = Rune.to_array (Rune.reshape
   [| 10 |] shuffled) in let sorted_orig = Array.copy original in let
   sorted_shuf = Array.copy shuffled_vals in Array.sort compare sorted_orig;
   Array.sort compare sorted_shuf; A.check (A.array (A.float 0.001)) "shuffle
   preserves elements" sorted_orig sorted_shuf;

   (* Check it's actually shuffled *) let is_equal = Rune.all (Rune.array_equal
   x shuffled) in let is_equal_val = Rune.to_array is_equal in let is_different
   = is_equal_val.(0) = 0 in A.check A.bool "shuffle changes order" true
   is_different *)

let test_truncated_normal () =
  let key = Rng.key 42 in
  let shape = [| 100 |] in
  let lower = -1.5 in
  let upper = 2.0 in
  let t = Rng.truncated_normal key Float32 ~lower ~upper shape in

  A.check (A.array A.int) "truncated_normal produces correct shape" shape
    (Rune.shape t);

  (* Check all values are within bounds *)
  let values = Rune.to_array t in
  Array.iter
    (fun v ->
      A.check A.bool
        (Printf.sprintf "truncated_normal values in [%.1f, %.1f]: %.3f" lower
           upper v)
        true
        (v >= lower && v <= upper))
    values

let test_categorical () =
  let key = Rng.key 42 in

  (* Test with simple 1D logits: [0.0, 1.0, 2.0] *)
  (* Expected probabilities after softmax: [0.090, 0.245, 0.665] approximately *)
  let logits = Rune.create Float32 [| 3 |] [| 0.0; 1.0; 2.0 |] in
  let samples = Rng.categorical key logits in

  (* Check output shape *)
  let output_shape = Rune.shape samples in
  A.check (A.array A.int) "categorical produces correct shape" [||] output_shape;

  (* Check that output is a scalar int32 *)
  let sample_val = Rune.to_array samples in
  A.check A.int "categorical produces single value" 1 (Array.length sample_val);

  (* Check value is in valid range [0, 2] *)
  let sample_idx = Int32.to_int sample_val.(0) in
  A.check A.bool "categorical value in valid range" true
    (sample_idx >= 0 && sample_idx <= 2);

  (* Test determinism *)
  let samples2 = Rng.categorical key logits in
  let is_equal = Rune.all (Rune.equal samples samples2) in
  let is_equal_val = Rune.to_array is_equal in
  A.check A.bool "categorical is deterministic" true (is_equal_val.(0) > 0);

  (* Test with Float64 *)
  let logits64 = Rune.create Float64 [| 3 |] [| 0.0; 1.0; 2.0 |] in
  let samples64 = Rng.categorical key logits64 in
  let is_equal64 = Rune.all (Rune.equal samples samples64) in
  let is_equal_val64 = Rune.to_array is_equal64 in
  A.check A.bool "categorical is type agnostic" true (is_equal_val64.(0) > 0)

let test_categorical_2d () =
  let key = Rng.key 42 in

  (* Test with 2D logits: [[0.0, 1.0], [2.0, 0.0]] *)
  (* Expected probabilities after softmax: [[0.269, 0.731], [0.881, 0.119]] approximately *)
  let logits = Rune.create Float32 [| 2; 2 |] [| 0.0; 1.0; 2.0; 0.0 |] in
  let samples = Rng.categorical key logits in

  (* Check output shape (should be [2] - one sample per row) *)
  let output_shape = Rune.shape samples in
  A.check (A.array A.int) "categorical 2D produces correct shape" [| 2 |]
    output_shape;

  (* Check values are in valid range [0, 1] for each row *)
  let sample_vals = Rune.to_array samples in
  A.check A.int "categorical 2D produces 2 values" 2 (Array.length sample_vals);

  Array.iter
    (fun v ->
      let idx = Int32.to_int v in
      A.check A.bool "categorical 2D value in valid range" true
        (idx >= 0 && idx <= 1))
    sample_vals

let test_categorical_axis_handling () =
  let key = Rng.key 42 in

  (* 2D logits: shape [2; 3] Row 0 → [0.0, 1.0, 2.0] Row 1 → [2.0, 0.5, -1.0]
     This ensures all probabilities differ. *)
  let logits =
    Rune.create Float32 [| 2; 3 |] [| 0.0; 1.0; 2.0; 2.0; 0.5; -1.0 |]
  in

  (* axis=1 → sample across columns for each row → shape [2] *)
  let samples_axis_1 = Rng.categorical key ~axis:1 logits in

  (* axis=-1 → equivalent to axis=1 → shape [2] *)
  let samples_axis_neg_1 = Rng.categorical key ~axis:(-1) logits in

  (* axis=0 → sample across rows for each column → shape [3] *)
  let samples_axis_0 = Rng.categorical key ~axis:0 logits in

  (* Check shape for axis=1 *)
  let shape_axis_1 = Rune.shape samples_axis_1 in
  A.check (A.array A.int) "categorical axis=1 produces correct shape" [| 2 |]
    shape_axis_1;

  (* Check shape for axis=-1 (should match axis=1) *)
  let shape_axis_neg_1 = Rune.shape samples_axis_neg_1 in
  A.check (A.array A.int) "categorical axis=-1 matches axis=1 shape" [| 2 |]
    shape_axis_neg_1;

  (* Check shape for axis=0 *)
  let shape_axis_0 = Rune.shape samples_axis_0 in
  A.check (A.array A.int) "categorical axis=0 produces correct shape" [| 3 |]
    shape_axis_0;

  (* Check that axis=1 and axis=-1 give identical results *)
  let is_equal = Rune.all (Rune.equal samples_axis_1 samples_axis_neg_1) in
  let is_equal_val = Rune.to_array is_equal in
  A.check A.bool "categorical axis=-1 behaves like axis=1" true
    (is_equal_val.(0) > 0);

  (* Sanity check: ensure sampled indices are in valid range *)
  let vals_axis_0 = Rune.to_array samples_axis_0 in
  Array.iter
    (fun i ->
      A.check A.bool "axis=0 value in valid range" true
        (Int32.to_int i >= 0 && Int32.to_int i < 2))
    vals_axis_0;

  let vals_axis_1 = Rune.to_array samples_axis_1 in
  Array.iter
    (fun i ->
      A.check A.bool "axis=1 value in valid range" true
        (Int32.to_int i >= 0 && Int32.to_int i < 3))
    vals_axis_1

let test_categorical_shape_prefix_axis () =
  let key = Rng.key 314 in
  let logits =
    Rune.create Float64 [| 2; 3; 4 |]
      [|
        0.0;
        0.5;
        1.0;
        1.5;
        2.0;
        2.5;
        3.0;
        -0.5;
        0.25;
        1.25;
        -1.0;
        0.75;
        -0.25;
        0.4;
        1.8;
        -1.5;
        0.2;
        1.1;
        0.3;
        -0.8;
        0.6;
        1.4;
        -0.2;
        0.9;
      |]
  in

  let prefix_shape = [| 5; 6 |] in
  let samples = Rng.categorical key ~shape:prefix_shape ~axis:(-2) logits in

  let expected_shape = [| 5; 6; 2; 4 |] in
  A.check (A.array A.int) "categorical shape prefix keeps axis semantics"
    expected_shape (Rune.shape samples);

  let values = Rune.to_array samples |> Array.map Int32.to_int in
  Array.iter
    (fun v ->
      A.check A.bool "categorical indices within axis range" true
        (v >= 0 && v < 3))
    values

let test_categorical_distribution () =
  let key = Rng.key 123 in
  let logits = Rune.create Rune.Float32 [| 3 |] [| 0.0; 1.0; 2.0 |] in

  let n_samples = 20000 in
  let inds = Rng.categorical key ~shape:[| n_samples |] logits in

  A.check (A.array A.int) "categorical produces correct shape" [| n_samples |]
    (Rune.shape inds);

  let values = Rune.to_array inds |> Array.map Int32.to_int in

  (* Histogram counts *)
  let n_classes = 3 in
  let counts = Array.make n_classes 0 in
  Array.iter (fun v -> counts.(v) <- counts.(v) + 1) values;

  (* Compute softmax probabilities from logits_arr *)
  let logits_arr = [| 0.0; 1.0; 2.0 |] in
  let max_logit =
    Array.fold_left
      (fun acc x -> if x > acc then x else acc)
      neg_infinity logits_arr
  in
  let exps = Array.map (fun x -> Stdlib.exp (x -. max_logit)) logits_arr in
  let sum_exps = Array.fold_left ( +. ) 0. exps in
  let probs = Array.map (fun e -> e /. sum_exps) exps in

  (* Check each bucket is within a reasonable statistical tolerance *)
  Array.iteri
    (fun i p ->
      let prop = float_of_int counts.(i) /. float_of_int n_samples in
      let se = Stdlib.sqrt (p *. (1. -. p) /. float_of_int n_samples) in
      let tol = Stdlib.max (4. *. se) 0.01 in
      A.check (A.float tol)
        (Printf.sprintf "categorical bucket %d ~ p" i)
        p prop)
    probs

let test_rng_with_autodiff () =
  let key = Rng.key 42 in

  (* Create a simple function that uses RNG *)
  let f x =
    let noise = Rng.normal key Float32 (Rune.shape x) in
    Rune.add x noise
  in

  (* Test gradient computation *)
  let x = Rune.ones Float32 [| 3; 3 |] in
  let grad_fn = Rune.grad f in
  let dx = grad_fn x in

  (* Gradient of x should be ones (since we just add noise) *)
  let expected = Rune.ones Float32 [| 3; 3 |] in
  let is_equal = Rune.all (Rune.equal dx expected) in
  let is_equal_val = Rune.to_array is_equal in
  A.check A.bool "RNG works with autodiff" true (is_equal_val.(0) > 0)

let () =
  A.run "Rune.Rng"
    [
      ( "key",
        [
          A.test_case "creation" `Quick test_key_creation;
          A.test_case "splitting" `Quick test_key_splitting;
          A.test_case "fold_in" `Quick test_fold_in;
        ] );
      ( "sampling",
        [
          A.test_case "uniform" `Quick test_uniform;
          A.test_case "normal" `Quick test_normal;
          A.test_case "randint" `Quick test_randint;
          A.test_case "bernoulli" `Quick test_bernoulli;
          (* TODO: Enable when argsort works with lazy views *)
          (* A.test_case "permutation" `Quick test_permutation;
          A.test_case "shuffle" `Quick test_shuffle; *)
          A.test_case "truncated_normal" `Quick test_truncated_normal;
          A.test_case "categorical" `Quick test_categorical;
          A.test_case "categorical_2d" `Quick test_categorical_2d;
          A.test_case "categorical_axis_handling" `Quick
            test_categorical_axis_handling;
          A.test_case "categorical_shape_prefix_axis" `Quick
            test_categorical_shape_prefix_axis;
          A.test_case "categorical_distribution" `Quick
            test_categorical_distribution;
        ] );
      ("autodiff", [ A.test_case "rng_with_grad" `Quick test_rng_with_autodiff ]);
    ]
