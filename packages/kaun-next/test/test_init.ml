(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module Init = Kaun_next.Init

let flatten_f32 t = Nx.to_array (Nx.reshape [| -1 |] (Nx.cast Nx.float32 t))
let tensor_all pred t = Array.for_all pred (flatten_f32 t)

let tensor_stats t =
  let a = flatten_f32 t in
  let n = float_of_int (Array.length a) in
  let mean = Array.fold_left ( +. ) 0.0 a /. n in
  let sq = Array.fold_left (fun acc x -> acc +. ((x -. mean) ** 2.0)) 0.0 a in
  (mean, sq /. n)

(* Constant *)

let test_constants () =
  let shape = [| 11; 13 |] in
  let zeros = Init.zeros ~fan_in:(-1) ~fan_out:0 Nx.float32 shape in
  is_true ~msg:"zeros" (tensor_all (fun x -> x = 0.0) zeros);
  let ones = Init.ones ~fan_in:(-1) ~fan_out:0 Nx.float32 shape in
  is_true ~msg:"ones" (tensor_all (fun x -> x = 1.0) ones);
  let c = Init.constant 3.5 ~fan_in:(-1) ~fan_out:0 Nx.float32 shape in
  is_true ~msg:"constant" (tensor_all (fun x -> x = 3.5) c)

(* Random *)

let test_uniform_range_and_mean () =
  Nx.Rng.run ~seed:1 @@ fun () ->
  let scale = 0.25 in
  let t = Init.uniform ~scale ~fan_in:0 ~fan_out:0 Nx.float32 [| 120_000 |] in
  is_true ~msg:"uniform range" (tensor_all (fun x -> x >= 0.0 && x < scale) t);
  let mean, _ = tensor_stats t in
  equal ~msg:"uniform mean" (float 8e-3) (scale /. 2.0) mean

let test_normal_mean_and_variance () =
  Nx.Rng.run ~seed:2 @@ fun () ->
  let stddev = 0.2 in
  let t = Init.normal ~stddev ~fan_in:0 ~fan_out:0 Nx.float32 [| 140_000 |] in
  let mean, variance = tensor_stats t in
  equal ~msg:"normal mean" (float 6e-3) 0.0 mean;
  equal ~msg:"normal variance" (float 8e-3) (stddev *. stddev) variance

let test_deterministic_same_seed () =
  let shape = [| 64; 64 |] in
  let draw seed =
    Nx.Rng.run ~seed @@ fun () ->
    flatten_f32 (Init.he_uniform ~fan_in:64 ~fan_out:64 Nx.float32 shape)
  in
  is_true ~msg:"same seed, same tensor" (draw 12 = draw 12);
  is_true ~msg:"different seed, different tensor" (draw 12 <> draw 13)

(* Variance scaling families *)

let test_glorot_uniform_bounds () =
  Nx.Rng.run ~seed:3 @@ fun () ->
  let fan_in = 64 and fan_out = 32 in
  let limit = sqrt (6.0 /. float_of_int (fan_in + fan_out)) in
  let t = Init.glorot_uniform ~fan_in ~fan_out Nx.float32 [| 64; 32 |] in
  is_true ~msg:"glorot_uniform bounds"
    (tensor_all (fun x -> x >= -.limit && x <= limit) t)

let test_glorot_normal_variance () =
  Nx.Rng.run ~seed:4 @@ fun () ->
  let fan_in = 960 and fan_out = 480 in
  let expected = 2.0 /. float_of_int (fan_in + fan_out) in
  let t = Init.glorot_normal ~fan_in ~fan_out Nx.float32 [| 960; 480 |] in
  let _, variance = tensor_stats t in
  equal ~msg:"glorot_normal variance" (float 3e-4) expected variance

let test_he_uniform_bounds () =
  Nx.Rng.run ~seed:5 @@ fun () ->
  let fan_in = 128 in
  let limit = sqrt (6.0 /. float_of_int fan_in) in
  let t = Init.he_uniform ~fan_in ~fan_out:64 Nx.float32 [| 128; 64 |] in
  is_true ~msg:"he_uniform bounds"
    (tensor_all (fun x -> x >= -.limit && x <= limit) t)

let test_he_normal_variance () =
  Nx.Rng.run ~seed:6 @@ fun () ->
  let fan_in = 256 in
  let expected = 2.0 /. float_of_int fan_in in
  let t = Init.he_normal ~fan_in ~fan_out:64 Nx.float32 [| 256; 64 |] in
  let _, variance = tensor_stats t in
  equal ~msg:"he_normal variance" (float 2e-3) expected variance

let test_lecun_uniform_bounds () =
  Nx.Rng.run ~seed:7 @@ fun () ->
  let fan_in = 128 in
  let limit = sqrt (3.0 /. float_of_int fan_in) in
  let t = Init.lecun_uniform ~fan_in ~fan_out:32 Nx.float32 [| 128; 32 |] in
  is_true ~msg:"lecun_uniform bounds"
    (tensor_all (fun x -> x >= -.limit && x <= limit) t)

let test_lecun_normal_variance () =
  Nx.Rng.run ~seed:8 @@ fun () ->
  let fan_in = 128 in
  let expected = 1.0 /. float_of_int fan_in in
  let t = Init.lecun_normal ~fan_in ~fan_out:16 Nx.float32 [| 128; 16 |] in
  let _, variance = tensor_stats t in
  equal ~msg:"lecun_normal variance" (float 1.5e-3) expected variance

let test_variance_scaling_fan_out_mode () =
  Nx.Rng.run ~seed:9 @@ fun () ->
  let fan_out = 40 in
  let scale = 1.7 in
  let limit = sqrt (3.0 *. scale /. float_of_int fan_out) in
  let init =
    Init.variance_scaling ~scale ~mode:`Fan_out ~distribution:`Uniform
  in
  let t = init ~fan_in:9 ~fan_out Nx.float32 [| 2; 9; 4 |] in
  is_true ~msg:"fan_out mode bounds"
    (tensor_all (fun x -> x >= -.limit && x <= limit) t)

let test_variance_follows_fans_not_shape () =
  Nx.Rng.run ~seed:10 @@ fun () ->
  (* The shape is unrelated to the fans; only the fans set the variance. *)
  let t = Init.he_normal ~fan_in:8 ~fan_out:1 Nx.float32 [| 400; 400 |] in
  let _, variance = tensor_stats t in
  equal ~msg:"variance follows fan_in" (float 1e-2) (2.0 /. 8.0) variance

(* Shape and dtype *)

let test_requested_shape () =
  Nx.Rng.run ~seed:11 @@ fun () ->
  let shape = [| 3; 4; 5 |] in
  let t = Init.glorot_uniform ~fan_in:4 ~fan_out:5 Nx.float32 shape in
  equal ~msg:"shape" (array int) shape (Nx.shape t)

let test_float64_dtype () =
  Nx.Rng.run ~seed:12 @@ fun () ->
  (* One polymorphic initializer value serves several float dtypes. *)
  let init = Init.lecun_normal in
  let t32 = init ~fan_in:64 ~fan_out:64 Nx.float32 [| 64; 64 |] in
  let t64 = init ~fan_in:64 ~fan_out:64 Nx.float64 [| 64; 64 |] in
  is_true ~msg:"float32 dtype" (Nx.dtype t32 = Nx.float32);
  is_true ~msg:"float64 dtype" (Nx.dtype t64 = Nx.float64);
  let _, variance = tensor_stats t64 in
  equal ~msg:"float64 variance" (float 4e-3) (1.0 /. 64.0) variance

(* Validation *)

let test_negative_scale_rejected () =
  raises_invalid_arg "scale must be >= 0, got -1" (fun () ->
      Init.uniform ~scale:(-1.0));
  raises_invalid_arg "stddev must be >= 0, got -0.1" (fun () ->
      Init.normal ~stddev:(-0.1));
  raises_invalid_arg "scale must be >= 0, got -1" (fun () ->
      Init.variance_scaling ~scale:(-1.0) ~mode:`Fan_in ~distribution:`Uniform)

let test_non_positive_fan_rejected () =
  raises_invalid_arg "fans must be positive, got fan_in=0 fan_out=4" (fun () ->
      Init.he_normal ~fan_in:0 ~fan_out:4 Nx.float32 [| 4; 4 |]);
  raises_invalid_arg "fans must be positive, got fan_in=3 fan_out=-1" (fun () ->
      Init.variance_scaling ~scale:1.0 ~mode:`Fan_in ~distribution:`Normal
        ~fan_in:3 ~fan_out:(-1) Nx.float32 [| 3 |])

let () =
  run "kaun-next init"
    [
      group "constant"
        [ test "zeros, ones, constant fill and ignore fans" test_constants ];
      group "random"
        [
          test "uniform stays in range with the right mean"
            test_uniform_range_and_mean;
          test "normal matches mean and variance" test_normal_mean_and_variance;
          test "same seed reproduces the draw" test_deterministic_same_seed;
        ];
      group "variance scaling"
        [
          test "glorot uniform respects its limit" test_glorot_uniform_bounds;
          test "glorot normal hits fan-average variance"
            test_glorot_normal_variance;
          test "he uniform respects its limit" test_he_uniform_bounds;
          test "he normal hits fan-in variance" test_he_normal_variance;
          test "lecun uniform respects its limit" test_lecun_uniform_bounds;
          test "lecun normal hits fan-in variance" test_lecun_normal_variance;
          test "fan-out mode scales by fan_out"
            test_variance_scaling_fan_out_mode;
          test "variance follows fans, not shape"
            test_variance_follows_fans_not_shape;
        ];
      group "shape and dtype"
        [
          test "produces the requested shape" test_requested_shape;
          test "one initializer serves float32 and float64" test_float64_dtype;
        ];
      group "validation"
        [
          test "negative scale or stddev is rejected"
            test_negative_scale_rejected;
          test "non-positive fans are rejected" test_non_positive_fan_rejected;
        ];
    ]
