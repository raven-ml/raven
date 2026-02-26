(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx
open Windtrap

let test_key_creation () =
  let key1 = Rng.key 42 in
  let key2 = Rng.key 42 in
  let key3 = Rng.key 43 in
  equal ~msg:"same seed produces same key" int (Rng.to_int key1)
    (Rng.to_int key2);
  equal ~msg:"different seeds produce different keys" bool true
    (Rng.to_int key1 <> Rng.to_int key3)

let test_key_splitting () =
  let key = Rng.key 42 in
  let keys = Rng.split key in
  equal ~msg:"default split produces 2 keys" int 2 (Array.length keys);

  let keys3 = Rng.split ~n:3 key in
  equal ~msg:"split with n=3 produces 3 keys" int 3 (Array.length keys3);

  (* Check keys are different *)
  equal ~msg:"split keys are different" bool true
    (Rng.to_int keys.(0) <> Rng.to_int keys.(1));

  (* Check deterministic *)
  let keys2 = Rng.split key in
  equal ~msg:"split is deterministic" int
    (Rng.to_int keys.(0))
    (Rng.to_int keys2.(0))

let test_fold_in () =
  let key = Rng.key 42 in
  let key1 = Rng.fold_in key 1 in
  let key2 = Rng.fold_in key 2 in
  let key1_again = Rng.fold_in key 1 in

  equal ~msg:"fold_in with different data produces different keys" bool true
    (Rng.to_int key1 <> Rng.to_int key2);
  equal ~msg:"fold_in is deterministic" int (Rng.to_int key1)
    (Rng.to_int key1_again)

let test_rand () =
  let shape = [| 3; 4 |] in
  let t = Rng.run ~seed:42 (fun () -> rand float32 shape) in

  equal ~msg:"rand produces correct shape" (array int) shape (Nx.shape t);

  (* Check values are in [0, 1) *)
  let values = Nx.to_array (Nx.reshape [| 12 |] t) in
  Array.iter
    (fun v -> equal ~msg:"rand values in [0, 1)" bool true (v >= 0. && v < 1.))
    values;

  (* Check deterministic *)
  let t2 = Rng.run ~seed:42 (fun () -> rand float32 shape) in
  let is_equal = Nx.all (Nx.equal t t2) in
  let is_equal_val = Nx.to_array is_equal in
  equal ~msg:"rand is deterministic" bool true is_equal_val.(0)

let test_randn () =
  let shape = [| 100 |] in
  let t = Rng.run ~seed:42 (fun () -> randn float32 shape) in

  equal ~msg:"randn produces correct shape" (array int) shape (Nx.shape t);

  (* Check roughly normal distribution (mean ~0, std ~1) *)
  let values = Nx.to_array t in
  let mean =
    Array.fold_left ( +. ) 0. values /. float_of_int (Array.length values)
  in
  let variance =
    Array.fold_left (fun acc v -> acc +. ((v -. mean) ** 2.)) 0. values
    /. float_of_int (Array.length values)
  in
  let std = Stdlib.sqrt variance in

  equal ~msg:"randn mean ~0" (float 0.2) 0. mean;
  equal ~msg:"randn std ~1" (float 0.3) 1. std

let test_randint () =
  let shape = [| 10 |] in
  let t = Rng.run ~seed:42 (fun () -> randint Nx.int32 ~high:15 shape 5) in

  equal ~msg:"randint produces correct shape" (array int) shape (Nx.shape t);

  (* Check values are in [min, max) *)
  let values = Nx.to_array t in
  Array.iter
    (fun v ->
      let v = Int32.to_int v in
      equal ~msg:"randint values in [5, 15)" bool true (v >= 5 && v < 15))
    values

let test_bernoulli () =
  let shape = [| 1000 |] in
  let p = 0.3 in
  let t = Rng.run ~seed:42 (fun () -> Rng.bernoulli ~p shape) in

  equal ~msg:"bernoulli produces correct shape" (array int) shape (Nx.shape t);
  let t_int = astype uint8 t in
  (* Check proportion roughly matches p *)
  let values = Nx.to_array t_int in
  let ones =
    Array.fold_left (fun acc v -> acc + if v > 0 then 1 else 0) 0 values
  in
  let prop = float_of_int ones /. float_of_int (Array.length values) in
  equal ~msg:"bernoulli proportion ~p" (float 0.05) p prop

let test_shuffle_preserves_shape () =
  let shape = [| 6; 4 |] in
  let data =
    Array.init (shape.(0) * shape.(1)) (fun i -> float_of_int (i + 1))
  in
  let x = Nx.create float32 shape data in
  let shuffled = Rng.run ~seed:7 (fun () -> Rng.shuffle x) in

  equal ~msg:"shuffle preserves leading axis" (array int) shape
    (Nx.shape shuffled);

  let flatten t =
    let dims = Nx.shape t in
    let total = Array.fold_left ( * ) 1 dims in
    let reshaped = Nx.reshape [| total |] t in
    Nx.to_array reshaped
  in
  let orig_flat = flatten x in
  let shuffled_flat = flatten shuffled in

  let sorted_orig = Array.copy orig_flat in
  let sorted_shuffled = Array.copy shuffled_flat in
  Array.sort compare sorted_orig;
  Array.sort compare sorted_shuffled;
  equal ~msg:"shuffle preserves multiset"
    (array (float 0.0))
    sorted_orig sorted_shuffled;

  let shuffled_again = Rng.run ~seed:7 (fun () -> Rng.shuffle x) in
  let equality = Nx.equal shuffled shuffled_again |> Nx.all |> Nx.to_array in
  equal ~msg:"shuffle deterministic with same seed" bool true equality.(0)

let test_truncated_normal () =
  let shape = [| 100 |] in
  let lower = -1.5 in
  let upper = 2.0 in
  let t =
    Rng.run ~seed:42 (fun () ->
        Rng.truncated_normal float32 ~lower ~upper shape)
  in

  equal ~msg:"truncated_normal produces correct shape" (array int) shape
    (Nx.shape t);

  (* Check all values are within bounds *)
  let values = Nx.to_array t in
  Array.iter
    (fun v ->
      equal
        ~msg:
          (Printf.sprintf "truncated_normal values in [%.1f, %.1f]: %.3f" lower
             upper v)
        bool true
        (v >= lower && v <= upper))
    values

let test_truncated_normal_distribution () =
  let shape = [| 20_000 |] in
  let lower = -0.75 in
  let upper = 1.25 in
  let samples =
    Rng.run ~seed:123 (fun () ->
        Rng.truncated_normal float32 ~lower ~upper shape)
  in

  equal ~msg:"truncated_normal produces correct shape" (array int) shape
    (Nx.shape samples);

  let values = Nx.to_array samples in
  let total = Array.length values in
  let boundary_hits =
    Array.fold_left
      (fun acc v ->
        if Float.abs (v -. lower) < 1e-6 || Float.abs (v -. upper) < 1e-6 then
          acc + 1
        else acc)
      0 values
  in

  equal
    ~msg:
      (Printf.sprintf
         "truncated normal rarely clips to bounds (%d / %d clipped)"
         boundary_hits total)
    bool true
    (boundary_hits < total / 1000);

  let mean = Array.fold_left ( +. ) 0. values /. float_of_int total in
  equal ~msg:"truncated normal mean lies within interval" bool true
    (mean > lower && mean < upper)

let test_categorical () =
  (* Test with simple 1D logits: [0.0, 1.0, 2.0] *)
  (* Expected probabilities after softmax: [0.090, 0.245, 0.665] approximately *)
  let logits = Nx.create float32 [| 3 |] [| 0.0; 1.0; 2.0 |] in
  let samples = Rng.run ~seed:42 (fun () -> Rng.categorical logits) in

  (* Check output shape *)
  let output_shape = Nx.shape samples in
  equal ~msg:"categorical produces correct shape" (array int) [||] output_shape;

  (* Check that output is a scalar int32 *)
  let sample_val = Nx.to_array samples in
  equal ~msg:"categorical produces single value" int 1 (Array.length sample_val);

  (* Check value is in valid range [0, 2] *)
  let sample_idx = Int32.to_int sample_val.(0) in
  equal ~msg:"categorical value in valid range" bool true
    (sample_idx >= 0 && sample_idx <= 2);

  (* Test determinism *)
  let samples2 = Rng.run ~seed:42 (fun () -> Rng.categorical logits) in
  let is_equal = Nx.all (Nx.equal samples samples2) in
  let is_equal_val = Nx.to_array is_equal in
  equal ~msg:"categorical is deterministic" bool true is_equal_val.(0);

  (* Test with Float64 *)
  let logits64 = Nx.create float64 [| 3 |] [| 0.0; 1.0; 2.0 |] in
  let samples64 = Rng.run ~seed:42 (fun () -> Rng.categorical logits64) in
  let is_equal64 = Nx.all (Nx.equal samples samples64) in
  let is_equal_val64 = Nx.to_array is_equal64 in
  equal ~msg:"categorical is type agnostic" bool true is_equal_val64.(0)

let test_categorical_2d () =
  (* Test with 2D logits: [[0.0, 1.0], [2.0, 0.0]] *)
  (* Expected probabilities after softmax: [[0.269, 0.731], [0.881, 0.119]] approximately *)
  let logits = Nx.create float32 [| 2; 2 |] [| 0.0; 1.0; 2.0; 0.0 |] in
  let samples = Rng.run ~seed:42 (fun () -> Rng.categorical logits) in

  (* Check output shape (should be [2] - one sample per row) *)
  let output_shape = Nx.shape samples in
  equal ~msg:"categorical 2D produces correct shape" (array int) [| 2 |]
    output_shape;

  (* Check values are in valid range [0, 1] for each row *)
  let sample_vals = Nx.to_array samples in
  equal ~msg:"categorical 2D produces 2 values" int 2 (Array.length sample_vals);

  Array.iter
    (fun v ->
      let idx = Int32.to_int v in
      equal ~msg:"categorical 2D value in valid range" bool true
        (idx >= 0 && idx <= 1))
    sample_vals

let test_categorical_axis_handling () =
  (* 2D logits: shape [2; 3] Row 0 -> [0.0, 1.0, 2.0] Row 1 -> [2.0, 0.5, -1.0]
     This ensures all probabilities differ. *)
  let logits =
    Nx.create float32 [| 2; 3 |] [| 0.0; 1.0; 2.0; 2.0; 0.5; -1.0 |]
  in

  (* axis=1 -> sample across columns for each row -> shape [2] *)
  let samples_axis_1 =
    Rng.run ~seed:42 (fun () -> Rng.categorical ~axis:1 logits)
  in

  (* axis=-1 -> equivalent to axis=1 -> shape [2] *)
  let samples_axis_neg_1 =
    Rng.run ~seed:42 (fun () -> Rng.categorical ~axis:(-1) logits)
  in

  (* axis=0 -> sample across rows for each column -> shape [3] *)
  let samples_axis_0 =
    Rng.run ~seed:42 (fun () -> Rng.categorical ~axis:0 logits)
  in

  (* Check shape for axis=1 *)
  let shape_axis_1 = Nx.shape samples_axis_1 in
  equal ~msg:"categorical axis=1 produces correct shape" (array int) [| 2 |]
    shape_axis_1;

  (* Check shape for axis=-1 (should match axis=1) *)
  let shape_axis_neg_1 = Nx.shape samples_axis_neg_1 in
  equal ~msg:"categorical axis=-1 matches axis=1 shape" (array int) [| 2 |]
    shape_axis_neg_1;

  (* Check shape for axis=0 *)
  let shape_axis_0 = Nx.shape samples_axis_0 in
  equal ~msg:"categorical axis=0 produces correct shape" (array int) [| 3 |]
    shape_axis_0;

  (* Check that axis=1 and axis=-1 give identical results *)
  let is_equal = Nx.all (Nx.equal samples_axis_1 samples_axis_neg_1) in
  let is_equal_val = Nx.to_array is_equal in
  equal ~msg:"categorical axis=-1 behaves like axis=1" bool true
    is_equal_val.(0);

  (* Sanity check: ensure sampled indices are in valid range *)
  let vals_axis_0 = Nx.to_array samples_axis_0 in
  Array.iter
    (fun i ->
      equal ~msg:"axis=0 value in valid range" bool true
        (Int32.to_int i >= 0 && Int32.to_int i < 2))
    vals_axis_0;

  let vals_axis_1 = Nx.to_array samples_axis_1 in
  Array.iter
    (fun i ->
      equal ~msg:"axis=1 value in valid range" bool true
        (Int32.to_int i >= 0 && Int32.to_int i < 3))
    vals_axis_1

let test_categorical_shape_prefix_axis () =
  let logits =
    Nx.create float64 [| 2; 3; 4 |]
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
  let samples =
    Rng.run ~seed:314 (fun () ->
        Rng.categorical ~shape:prefix_shape ~axis:(-2) logits)
  in

  let expected_shape = [| 5; 6; 2; 4 |] in
  equal ~msg:"categorical shape prefix keeps axis semantics" (array int)
    expected_shape (Nx.shape samples);

  let values = Nx.to_array samples |> Array.map Int32.to_int in
  Array.iter
    (fun v ->
      equal ~msg:"categorical indices within axis range" bool true
        (v >= 0 && v < 3))
    values

let test_categorical_distribution () =
  let logits = Nx.create float32 [| 3 |] [| 0.0; 1.0; 2.0 |] in

  let n_samples = 20000 in
  let inds =
    Rng.run ~seed:123 (fun () -> Rng.categorical ~shape:[| n_samples |] logits)
  in

  equal ~msg:"categorical produces correct shape" (array int) [| n_samples |]
    (Nx.shape inds);

  let values = Nx.to_array inds |> Array.map Int32.to_int in

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
      equal
        ~msg:(Printf.sprintf "categorical bucket %d ~ p" i)
        (float tol) p prop)
    probs

let () =
  run "Nx.Rng"
    [
      group "key"
        [
          test "creation" test_key_creation;
          test "splitting" test_key_splitting;
          test "fold_in" test_fold_in;
        ];
      group "sampling"
        [
          test "rand" test_rand;
          test "randn" test_randn;
          test "randint" test_randint;
          test "bernoulli" test_bernoulli;
          test "shuffle_preserves_shape" test_shuffle_preserves_shape;
          test "truncated_normal" test_truncated_normal;
          test "truncated_normal_distribution"
            test_truncated_normal_distribution;
          test "categorical" test_categorical;
          test "categorical_2d" test_categorical_2d;
          test "categorical_axis_handling" test_categorical_axis_handling;
          test "categorical_shape_prefix_axis"
            test_categorical_shape_prefix_axis;
          test "categorical_distribution" test_categorical_distribution;
        ];
    ]
