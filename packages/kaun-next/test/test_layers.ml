(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Kaun_next

(* Float64 instances for gradient checking; the layer traversals are
   dtype-generic, so each instance is just a type pin. *)

module Linear64 = struct
  type t = Nx.float64_elt Linear.params

  let map = Linear.map
  let map2 = Linear.map2
  let iter = Linear.iter
end

module Embedding64 = struct
  type t = Nx.float64_elt Embedding.params

  let map = Embedding.map
  let map2 = Embedding.map2
  let iter = Embedding.iter
end

module Layer_norm64 = struct
  type t = Nx.float64_elt Layer_norm.params

  let map = Layer_norm.map
  let map2 = Layer_norm.map2
  let iter = Layer_norm.iter
end

let grads_ok = function Ok () -> () | Error m -> fail m
let shape_is ?msg expected t = equal ?msg (array int) expected (Nx.shape t)

let values_are ?msg ~tol expected t =
  equal ?msg (array (float tol)) expected (Nx.to_array t)

(* Linear *)

let test_linear_init_shapes () =
  Nx.Rng.run ~seed:1 @@ fun () ->
  let p = Linear.init ~inputs:3 ~outputs:5 in
  shape_is ~msg:"w shape" [| 3; 5 |] p.Linear.w;
  match p.Linear.b with
  | None -> fail "init should create a bias"
  | Some b -> shape_is ~msg:"b shape" [| 5 |] b

let test_linear_apply_affine () =
  let p =
    {
      Linear.w = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |];
      b = Some (Nx.create Nx.float32 [| 2 |] [| 10.; 20. |]);
    }
  in
  let x = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 1.; 0.; 2. |] in
  values_are ~msg:"x @ w + b" ~tol:1e-6 [| 14.; 26.; 16.; 28. |]
    (Linear.apply p x)

let test_linear_no_bias () =
  Nx.Rng.run ~seed:2 @@ fun () ->
  let p = Linear.make ~bias:false ~inputs:4 ~outputs:3 Nx.float32 in
  is_true ~msg:"no bias parameter" (p.Linear.b = None);
  let x = Nx.create Nx.float32 [| 2; 4 |] (Array.init 8 float_of_int) in
  values_are ~msg:"apply is the plain matmul" ~tol:1e-6
    (Nx.to_array (Nx.matmul x p.Linear.w))
    (Linear.apply p x)

let test_linear_batched_apply () =
  Nx.Rng.run ~seed:3 @@ fun () ->
  let p = Linear.init ~inputs:4 ~outputs:2 in
  let x = Nx.zeros Nx.float32 [| 2; 3; 4 |] in
  shape_is ~msg:"leading axes are batch axes" [| 2; 3; 2 |] (Linear.apply p x)

let test_linear_custom_inits () =
  let p =
    Linear.make ~w_init:(Init.constant 2.0) ~bias_init:(Init.constant 1.0)
      ~inputs:2 ~outputs:3 Nx.float32
  in
  values_are ~msg:"w_init" ~tol:0.0 (Array.make 6 2.0) p.Linear.w;
  match p.Linear.b with
  | None -> fail "make should create a bias by default"
  | Some b -> values_are ~msg:"bias_init" ~tol:0.0 (Array.make 3 1.0) b

let test_linear_names () =
  Nx.Rng.run ~seed:4 @@ fun () ->
  let with_bias = Linear.init ~inputs:2 ~outputs:2 in
  equal ~msg:"with bias" (list string) [ "w"; "b" ] (Linear.names with_bias);
  let no_bias = Linear.make ~bias:false ~inputs:2 ~outputs:2 Nx.float32 in
  equal ~msg:"without bias" (list string) [ "w" ] (Linear.names no_bias)

let test_linear_gradients () =
  Nx.Rng.run ~seed:5 @@ fun () ->
  let x = Nx.randn Nx.float64 [| 4; 3 |] in
  let loss p =
    let y = Linear.apply p x in
    Nx.sum (Nx.mul y y)
  in
  let p = Linear.make ~inputs:3 ~outputs:2 Nx.float64 in
  grads_ok (Rune_next.check_grads (module Linear64) loss p);
  let no_bias = Linear.make ~bias:false ~inputs:3 ~outputs:2 Nx.float64 in
  grads_ok (Rune_next.check_grads (module Linear64) loss no_bias)

let test_linear_map2_bias_mismatch () =
  Nx.Rng.run ~seed:6 @@ fun () ->
  let p = Linear.init ~inputs:2 ~outputs:2 in
  let q = Linear.make ~bias:false ~inputs:2 ~outputs:2 Nx.float32 in
  raises_invalid_arg "Linear.map2: bias mismatch" (fun () ->
      Linear.map2 (fun a _ -> a) p q)

let test_linear_rejects_bad_geometry () =
  raises_invalid_arg
    "Linear.make: inputs and outputs must be positive, got inputs=0 outputs=4"
    (fun () -> Linear.make ~inputs:0 ~outputs:4 Nx.float32);
  raises_invalid_arg
    "Linear.make: inputs and outputs must be positive, got inputs=3 outputs=-1"
    (fun () -> Linear.make ~inputs:3 ~outputs:(-1) Nx.float32)

(* Embedding *)

let embedding_4x3 () =
  (* Row i is [3i; 3i + 1; 3i + 2]. *)
  {
    Embedding.table =
      Nx.create Nx.float32 [| 4; 3 |] (Array.init 12 float_of_int);
  }

let test_embedding_init_shape () =
  Nx.Rng.run ~seed:7 @@ fun () ->
  let p = Embedding.init ~vocab:7 ~dim:4 in
  shape_is ~msg:"table shape" [| 7; 4 |] p.Embedding.table;
  equal ~msg:"names" (list string) [ "table" ] (Embedding.names p)

let test_embedding_gathers_rows () =
  let p = embedding_4x3 () in
  let ids = Nx.create Nx.int32 [| 2 |] [| 1l; 3l |] in
  values_are ~msg:"rows 1 and 3" ~tol:0.0
    [| 3.; 4.; 5.; 9.; 10.; 11. |]
    (Embedding.apply p ids)

let test_embedding_output_shape () =
  let p = embedding_4x3 () in
  let ids = Nx.create Nx.int32 [| 2; 3 |] [| 0l; 1l; 2l; 3l; 0l; 1l |] in
  shape_is ~msg:"ids shape plus dim" [| 2; 3; 3 |] (Embedding.apply p ids)

let test_embedding_scalar_id () =
  let p = embedding_4x3 () in
  let id = Nx.create Nx.int32 [||] [| 2l |] in
  let row = Embedding.apply p id in
  shape_is ~msg:"a single row" [| 3 |] row;
  values_are ~msg:"row 2" ~tol:0.0 [| 6.; 7.; 8. |] row

let test_embedding_duplicate_id_gradient () =
  Nx.Rng.run ~seed:8 @@ fun () ->
  let p = Embedding.make ~vocab:3 ~dim:2 Nx.float64 in
  let ids = Nx.create Nx.int32 [| 3 |] [| 0l; 2l; 0l |] in
  let loss p = Nx.sum (Embedding.apply p ids) in
  let g = Rune_next.grad (module Embedding64) loss p in
  (* Row 0 is gathered twice, row 1 never, row 2 once. *)
  values_are ~msg:"gradient counts occurrences" ~tol:1e-12
    [| 2.; 2.; 0.; 0.; 1.; 1. |]
    g.Embedding.table

let test_embedding_gradients () =
  Nx.Rng.run ~seed:9 @@ fun () ->
  let p = Embedding.make ~vocab:5 ~dim:3 Nx.float64 in
  let ids = Nx.create Nx.int32 [| 2; 2 |] [| 0l; 3l; 3l; 1l |] in
  let loss p =
    let y = Embedding.apply p ids in
    Nx.sum (Nx.mul y y)
  in
  grads_ok (Rune_next.check_grads (module Embedding64) loss p)

let test_embedding_rejects_out_of_bounds () =
  let p = embedding_4x3 () in
  raises_match ~msg:"id 4 is out of bounds for vocab 4"
    (function Failure _ -> true | _ -> false)
    (fun () -> Embedding.apply p (Nx.create Nx.int32 [| 1 |] [| 4l |]))

let test_embedding_rejects_bad_geometry () =
  raises_invalid_arg
    "Embedding.make: vocab and dim must be positive, got vocab=0 dim=3"
    (fun () -> Embedding.make ~vocab:0 ~dim:3 Nx.float32)

(* Layer norm *)

let test_layer_norm_init_shapes () =
  let p = Layer_norm.init ~dim:6 in
  shape_is ~msg:"gamma shape" [| 6 |] p.Layer_norm.gamma;
  shape_is ~msg:"beta shape" [| 6 |] p.Layer_norm.beta;
  values_are ~msg:"gamma is ones" ~tol:0.0 (Array.make 6 1.0) p.Layer_norm.gamma;
  values_are ~msg:"beta is zeros" ~tol:0.0 (Array.make 6 0.0) p.Layer_norm.beta;
  equal ~msg:"names" (list string) [ "gamma"; "beta" ] (Layer_norm.names p)

let test_layer_norm_analytic () =
  (* Per row: mean 1, variance 1, so x normalizes to [-1; 1] (up to eps), then
     gamma scales and beta shifts. Rows normalize independently. *)
  let p =
    {
      Layer_norm.gamma = Nx.create Nx.float32 [| 2 |] [| 2.; 3. |];
      beta = Nx.create Nx.float32 [| 2 |] [| 1.; -1. |];
    }
  in
  let x = Nx.create Nx.float32 [| 2; 2 |] [| 0.; 2.; 10.; 30. |] in
  values_are ~msg:"gamma * xhat + beta" ~tol:1e-4 [| -1.; 2.; -1.; 2. |]
    (Layer_norm.apply p x)

let test_layer_norm_standardizes () =
  Nx.Rng.run ~seed:10 @@ fun () ->
  let x = Nx.add_s (Nx.mul_s (Nx.randn Nx.float32 [| 3; 16 |]) 3.0) 7.0 in
  let y = Layer_norm.apply (Layer_norm.init ~dim:16) x in
  shape_is ~msg:"shape preserved" [| 3; 16 |] y;
  let row_means = Nx.to_array (Nx.mean ~axes:[ 1 ] y) in
  let row_vars = Nx.to_array (Nx.mean ~axes:[ 1 ] (Nx.mul y y)) in
  Array.iteri
    (fun i m -> equal ~msg:(Printf.sprintf "row %d mean" i) (float 1e-4) 0.0 m)
    row_means;
  Array.iteri
    (fun i v ->
      equal ~msg:(Printf.sprintf "row %d variance" i) (float 1e-3) 1.0 v)
    row_vars

let test_layer_norm_constant_input () =
  let p = Layer_norm.init ~dim:4 in
  let x = Nx.full Nx.float32 [| 2; 4 |] 5.0 in
  values_are ~msg:"constant vectors map to beta" ~tol:0.0 (Array.make 8 0.0)
    (Layer_norm.apply p x)

let test_layer_norm_eps () =
  (* With eps = 3, the row [0; 2] has variance 1 and normalizes by sqrt (1 + 3)
     = 2. *)
  let p = Layer_norm.init ~dim:2 in
  let x = Nx.create Nx.float32 [| 1; 2 |] [| 0.; 2. |] in
  values_are ~msg:"eps enters the denominator" ~tol:1e-6 [| -0.5; 0.5 |]
    (Layer_norm.apply ~eps:3.0 p x)

let test_layer_norm_gradients () =
  Nx.Rng.run ~seed:11 @@ fun () ->
  let p =
    {
      Layer_norm.gamma = Nx.randn Nx.float64 [| 4 |];
      beta = Nx.randn Nx.float64 [| 4 |];
    }
  in
  let x = Nx.randn Nx.float64 [| 3; 4 |] in
  let loss p =
    let y = Layer_norm.apply p x in
    Nx.sum (Nx.mul y y)
  in
  grads_ok (Rune_next.check_grads (module Layer_norm64) loss p)

let test_layer_norm_rejects_bad_input () =
  let p = Layer_norm.init ~dim:4 in
  raises_invalid_arg
    "Layer_norm.apply: last axis has size 3 but the layer normalizes 4 features"
    (fun () -> Layer_norm.apply p (Nx.zeros Nx.float32 [| 2; 3 |]));
  raises_invalid_arg "Layer_norm.apply: eps must be >= 0, got -1" (fun () ->
      Layer_norm.apply ~eps:(-1.0) p (Nx.zeros Nx.float32 [| 2; 4 |]));
  raises_invalid_arg "Layer_norm.make: dim must be positive, got 0" (fun () ->
      Layer_norm.make ~dim:0 Nx.float32)

let () =
  run "kaun-next layers"
    [
      group "linear"
        [
          test "init produces the documented shapes" test_linear_init_shapes;
          test "apply matches the affine map" test_linear_apply_affine;
          test "bias:false drops the bias parameter" test_linear_no_bias;
          test "leading axes are batch axes" test_linear_batched_apply;
          test "make respects w_init and bias_init" test_linear_custom_inits;
          test "names follow traversal order" test_linear_names;
          test "gradients agree with finite differences" test_linear_gradients;
          test "map2 rejects a bias mismatch" test_linear_map2_bias_mismatch;
          test "make rejects non-positive geometry"
            test_linear_rejects_bad_geometry;
        ];
      group "embedding"
        [
          test "init produces the documented shape" test_embedding_init_shape;
          test "apply gathers the indexed rows" test_embedding_gathers_rows;
          test "output shape is the ids shape plus dim"
            test_embedding_output_shape;
          test "a scalar id yields a single row" test_embedding_scalar_id;
          test "gradient counts duplicate ids"
            test_embedding_duplicate_id_gradient;
          test "gradients agree with finite differences"
            test_embedding_gradients;
          test "out-of-bounds ids are rejected"
            test_embedding_rejects_out_of_bounds;
          test "make rejects non-positive geometry"
            test_embedding_rejects_bad_geometry;
        ];
      group "layer norm"
        [
          test "init is the identity normalization" test_layer_norm_init_shapes;
          test "matches the analytic normalization" test_layer_norm_analytic;
          test "standardizes each vector of a batch"
            test_layer_norm_standardizes;
          test "constant vectors map to beta" test_layer_norm_constant_input;
          test "eps enters the denominator" test_layer_norm_eps;
          test "gradients agree with finite differences"
            test_layer_norm_gradients;
          test "invalid inputs are rejected" test_layer_norm_rejects_bad_input;
        ];
    ]
