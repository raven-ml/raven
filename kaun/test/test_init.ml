(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module Init = Kaun.Init

let string_contains s sub =
  let slen = String.length s in
  let sub_len = String.length sub in
  let rec loop i =
    if i + sub_len > slen then false
    else if String.sub s i sub_len = sub then true
    else loop (i + 1)
  in
  if sub_len = 0 then true else loop 0

let raises_invalid_arg_contains needle f =
  raises_match
    (fun exn ->
      match exn with
      | Invalid_argument msg -> string_contains msg needle
      | _ -> false)
    f

let flatten_f32 t =
  Rune.to_array (Rune.reshape [| -1 |] (Rune.cast Rune.float32 t))

let tensor_all pred t =
  let a = flatten_f32 t in
  Array.for_all pred a

let tensor_stats t =
  let a = flatten_f32 t in
  let n = Array.length a in
  let sum = ref 0.0 in
  for i = 0 to n - 1 do
    sum := !sum +. a.(i)
  done;
  let mean = !sum /. float_of_int n in
  let sq = ref 0.0 in
  for i = 0 to n - 1 do
    let d = a.(i) -. mean in
    sq := !sq +. (d *. d)
  done;
  let variance = !sq /. float_of_int n in
  (mean, variance)

let compute_fans shape ~in_axis ~out_axis =
  let rank = Array.length shape in
  if rank = 0 then (1, 1)
  else if rank = 1 then (shape.(0), shape.(0))
  else
    let normalize_axis axis = if axis < 0 then rank + axis else axis in
    let in_axis = normalize_axis in_axis in
    let out_axis = normalize_axis out_axis in
    let fan_in = shape.(in_axis) in
    let fan_out = shape.(out_axis) in
    let receptive = ref 1 in
    for i = 0 to rank - 1 do
      if i <> in_axis && i <> out_axis then receptive := !receptive * shape.(i)
    done;
    (fan_in * !receptive, fan_out * !receptive)

let expected_variance ~scale ~mode ~fan_in ~fan_out =
  let n =
    match mode with
    | `Fan_in -> float_of_int fan_in
    | `Fan_out -> float_of_int fan_out
    | `Fan_avg -> float_of_int (fan_in + fan_out) /. 2.0
  in
  scale /. n

let uniform_limit variance = sqrt (3.0 *. variance)

let test_constants () =
  let key = Rune.Rng.key 0 in
  let shape = [| 11; 13 |] in
  let zeros = Init.zeros.f key shape Rune.float32 in
  equal ~msg:"zeros" bool true (tensor_all (fun x -> x = 0.0) zeros);
  let ones = Init.ones.f key shape Rune.float32 in
  equal ~msg:"ones" bool true (tensor_all (fun x -> x = 1.0) ones);
  let c = (Init.constant 3.5).f key shape Rune.float32 in
  equal ~msg:"constant" bool true (tensor_all (fun x -> x = 3.5) c)

let test_uniform_range_and_mean () =
  let key = Rune.Rng.key 1 in
  let scale = 0.25 in
  let t = (Init.uniform ~scale ()).f key [| 120_000 |] Rune.float32 in
  equal ~msg:"uniform range" bool true
    (tensor_all (fun x -> x >= 0.0 && x < scale) t);
  let mean, _ = tensor_stats t in
  equal ~msg:"uniform mean" (float 8e-3) (scale /. 2.0) mean

let test_normal_mean_and_variance () =
  let key = Rune.Rng.key 2 in
  let stddev = 0.2 in
  let t = (Init.normal ~stddev ()).f key [| 140_000 |] Rune.float32 in
  let mean, variance = tensor_stats t in
  equal ~msg:"normal mean" (float 6e-3) 0.0 mean;
  equal ~msg:"normal variance" (float 8e-3) (stddev *. stddev) variance

let test_glorot_uniform_bounds () =
  let key = Rune.Rng.key 3 in
  let shape = [| 64; 32 |] in
  let fan_in, fan_out = compute_fans shape ~in_axis:(-2) ~out_axis:(-1) in
  let variance = expected_variance ~scale:1.0 ~mode:`Fan_avg ~fan_in ~fan_out in
  let limit = uniform_limit variance in
  let t = (Init.glorot_uniform ()).f key shape Rune.float32 in
  equal ~msg:"glorot_uniform bounds" bool true
    (tensor_all (fun x -> x >= -.limit && x <= limit) t)

let test_glorot_normal_variance () =
  let key = Rune.Rng.key 4 in
  let shape = [| 960; 480 |] in
  let fan_in, fan_out = compute_fans shape ~in_axis:(-2) ~out_axis:(-1) in
  let expected = expected_variance ~scale:1.0 ~mode:`Fan_avg ~fan_in ~fan_out in
  let t = (Init.glorot_normal ()).f key shape Rune.float32 in
  let _, variance = tensor_stats t in
  equal ~msg:"glorot_normal variance" (float 3e-4) expected variance

let test_he_uniform_bounds () =
  let key = Rune.Rng.key 5 in
  let shape = [| 128; 64 |] in
  let fan_in, fan_out = compute_fans shape ~in_axis:(-2) ~out_axis:(-1) in
  let variance = expected_variance ~scale:2.0 ~mode:`Fan_in ~fan_in ~fan_out in
  let limit = uniform_limit variance in
  let t = (Init.he_uniform ()).f key shape Rune.float32 in
  equal ~msg:"he_uniform bounds" bool true
    (tensor_all (fun x -> x >= -.limit && x <= limit) t)

let test_he_normal_variance () =
  let key = Rune.Rng.key 6 in
  let shape = [| 256; 64 |] in
  let fan_in, fan_out = compute_fans shape ~in_axis:(-2) ~out_axis:(-1) in
  let expected = expected_variance ~scale:2.0 ~mode:`Fan_in ~fan_in ~fan_out in
  let t = (Init.he_normal ()).f key shape Rune.float32 in
  let _, variance = tensor_stats t in
  equal ~msg:"he_normal variance" (float 2e-3) expected variance

let test_lecun_uniform_bounds () =
  let key = Rune.Rng.key 7 in
  let shape = [| 128; 32 |] in
  let fan_in, fan_out = compute_fans shape ~in_axis:(-2) ~out_axis:(-1) in
  let variance = expected_variance ~scale:1.0 ~mode:`Fan_in ~fan_in ~fan_out in
  let limit = uniform_limit variance in
  let t = (Init.lecun_uniform ()).f key shape Rune.float32 in
  equal ~msg:"lecun_uniform bounds" bool true
    (tensor_all (fun x -> x >= -.limit && x <= limit) t)

let test_lecun_normal_variance () =
  let key = Rune.Rng.key 8 in
  let shape = [| 128; 16 |] in
  let fan_in, fan_out = compute_fans shape ~in_axis:(-2) ~out_axis:(-1) in
  let expected = expected_variance ~scale:1.0 ~mode:`Fan_in ~fan_in ~fan_out in
  let t = (Init.lecun_normal ()).f key shape Rune.float32 in
  let _, variance = tensor_stats t in
  equal ~msg:"lecun_normal variance" (float 1.5e-3) expected variance

let test_variance_scaling_axis_override () =
  let key = Rune.Rng.key 9 in
  let shape = [| 2; 9; 4 |] in
  let in_axis = 2 in
  let out_axis = 0 in
  let fan_in, fan_out = compute_fans shape ~in_axis ~out_axis in
  let variance = expected_variance ~scale:1.7 ~mode:`Fan_out ~fan_in ~fan_out in
  let limit = uniform_limit variance in
  let init =
    Init.variance_scaling ~scale:1.7 ~mode:`Fan_out ~distribution:`Uniform
      ~in_axis ~out_axis ()
  in
  let t = init.f key shape Rune.float32 in
  equal ~msg:"variance_scaling axis override" bool true
    (tensor_all (fun x -> x >= -.limit && x <= limit) t)

let test_validation_errors () =
  raises_invalid_arg_contains "scale" (fun () ->
      ignore (Init.uniform ~scale:(-1.0) ()));
  raises_invalid_arg_contains "stddev" (fun () ->
      ignore (Init.normal ~stddev:(-0.1) ()));
  raises_invalid_arg_contains "scale" (fun () ->
      ignore
        (Init.variance_scaling ~scale:(-1.0) ~mode:`Fan_in
           ~distribution:`Uniform ()));
  let init =
    Init.variance_scaling ~scale:1.0 ~mode:`Fan_avg ~distribution:`Uniform
      ~in_axis:9 ()
  in
  raises_invalid_arg_contains "invalid in axis" (fun () ->
      ignore (init.f (Rune.Rng.key 10) [| 3; 4 |] Rune.float32));
  let zero_fan =
    Init.variance_scaling ~scale:1.0 ~mode:`Fan_in ~distribution:`Uniform ()
  in
  raises_invalid_arg_contains "non-positive fan" (fun () ->
      ignore (zero_fan.f (Rune.Rng.key 11) [| 0; 4 |] Rune.float32))

let test_deterministic_same_key () =
  let key = Rune.Rng.key 12 in
  let init = Init.he_uniform () in
  let shape = [| 64; 64 |] in
  let t0 = init.f key shape Rune.float32 |> flatten_f32 in
  let t1 = init.f key shape Rune.float32 |> flatten_f32 in
  equal ~msg:"same key deterministic" bool true (t0 = t1)

let () =
  run "Kaun.Init"
    [
      group "constant" [ test "zeros ones constant" test_constants ];
      group "random"
        [
          test "uniform range and mean" test_uniform_range_and_mean;
          test "normal mean and variance" test_normal_mean_and_variance;
          test "deterministic same key" test_deterministic_same_key;
        ];
      group "variance scaling families"
        [
          test "glorot uniform bounds" test_glorot_uniform_bounds;
          test "glorot normal variance" test_glorot_normal_variance;
          test "he uniform bounds" test_he_uniform_bounds;
          test "he normal variance" test_he_normal_variance;
          test "lecun uniform bounds" test_lecun_uniform_bounds;
          test "lecun normal variance" test_lecun_normal_variance;
          test "variance scaling axis override"
            test_variance_scaling_axis_override;
        ];
      group "validation" [ test "invalid arguments" test_validation_errors ];
    ]
