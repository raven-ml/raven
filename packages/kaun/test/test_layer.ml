(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module Layer = Kaun.Layer
module Ptree = Kaun.Ptree

let flatten_f32 t = Nx.to_array (Nx.reshape [| -1 |] (Nx.cast Nx.float32 t))

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

let apply_out (type a in_elt) (m : (a, float) Layer.t) vars ~training
    (x : (a, in_elt) Nx.t) =
  let y, _ = Layer.apply m vars ~training x in
  y

(* Linear *)

let test_linear_shapes () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let m = Layer.linear ~in_features:4 ~out_features:3 () in
  let vars = Layer.init m ~dtype:Nx.float32 in
  let fields = Ptree.Dict.fields_exn (Layer.params vars) in
  let w = Ptree.Dict.get_tensor_exn fields ~name:"weight" Nx.float32 in
  let b = Ptree.Dict.get_tensor_exn fields ~name:"bias" Nx.float32 in
  equal ~msg:"weight shape" (list int) [ 4; 3 ] (Array.to_list (Nx.shape w));
  equal ~msg:"bias shape" (list int) [ 3 ] (Array.to_list (Nx.shape b))

let test_linear_forward () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let m = Layer.linear ~in_features:2 ~out_features:3 () in
  let vars = Layer.init m ~dtype:Nx.float32 in
  let x = Nx.ones Nx.float32 [| 1; 2 |] in
  let y = apply_out m vars ~training:false x in
  equal ~msg:"output shape" (list int) [ 1; 3 ] (Array.to_list (Nx.shape y))

let test_linear_manual_params () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let m = Layer.linear ~in_features:2 ~out_features:2 () in
  let w = Nx.create Nx.float32 [| 2; 2 |] [| 1.0; 0.0; 0.0; 1.0 |] in
  let b = Nx.create Nx.float32 [| 2 |] [| 0.5; -0.5 |] in
  let params =
    Ptree.dict [ ("weight", Ptree.tensor w); ("bias", Ptree.tensor b) ]
  in
  let vars =
    Layer.init m ~dtype:Nx.float32 |> fun vars -> Layer.with_params vars params
  in
  let x = Nx.create Nx.float32 [| 1; 2 |] [| 3.0; 4.0 |] in
  let y = apply_out m vars ~training:false x in
  let expected = Nx.create Nx.float32 [| 1; 2 |] [| 3.5; 3.5 |] in
  equal ~msg:"linear identity + bias" bool true
    (tensor_close ~eps:1e-6 ~expected ~actual:y)

(* Normalization *)

let test_layer_norm_shapes () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let m = Layer.layer_norm ~dim:8 () in
  let vars = Layer.init m ~dtype:Nx.float32 in
  let fields = Ptree.Dict.fields_exn (Layer.params vars) in
  let gamma = Ptree.Dict.get_tensor_exn fields ~name:"gamma" Nx.float32 in
  let beta = Ptree.Dict.get_tensor_exn fields ~name:"beta" Nx.float32 in
  equal ~msg:"gamma shape" (list int) [ 8 ] (Array.to_list (Nx.shape gamma));
  equal ~msg:"beta shape" (list int) [ 8 ] (Array.to_list (Nx.shape beta));
  equal ~msg:"gamma values" bool true
    (Array.for_all (fun x -> x = 1.0) (flatten_f32 gamma));
  equal ~msg:"beta values" bool true
    (Array.for_all (fun x -> x = 0.0) (flatten_f32 beta))

let test_layer_norm_forward () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let m = Layer.layer_norm ~dim:4 () in
  let vars = Layer.init m ~dtype:Nx.float32 in
  let x =
    Nx.create Nx.float32 [| 2; 4 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0 |]
  in
  let y = apply_out m vars ~training:false x in
  equal ~msg:"output shape" (list int) [ 2; 4 ] (Array.to_list (Nx.shape y))

let test_rms_norm_shapes () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let m = Layer.rms_norm ~dim:6 () in
  let vars = Layer.init m ~dtype:Nx.float32 in
  let fields = Ptree.Dict.fields_exn (Layer.params vars) in
  let scale = Ptree.Dict.get_tensor_exn fields ~name:"scale" Nx.float32 in
  equal ~msg:"scale shape" (list int) [ 6 ] (Array.to_list (Nx.shape scale));
  equal ~msg:"scale values" bool true
    (Array.for_all (fun x -> x = 1.0) (flatten_f32 scale))

let test_batch_norm_shapes () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let m = Layer.batch_norm ~num_features:3 () in
  let vars = Layer.init m ~dtype:Nx.float32 in
  let param_fields = Ptree.Dict.fields_exn (Layer.params vars) in
  let state_fields = Ptree.Dict.fields_exn (Layer.state vars) in
  let scale = Ptree.Dict.get_tensor_exn param_fields ~name:"scale" Nx.float32 in
  let bias = Ptree.Dict.get_tensor_exn param_fields ~name:"bias" Nx.float32 in
  let running_mean =
    Ptree.Dict.get_tensor_exn state_fields ~name:"running_mean" Nx.float32
  in
  let running_var =
    Ptree.Dict.get_tensor_exn state_fields ~name:"running_var" Nx.float32
  in
  equal ~msg:"scale shape" (list int) [ 3 ] (Array.to_list (Nx.shape scale));
  equal ~msg:"bias shape" (list int) [ 3 ] (Array.to_list (Nx.shape bias));
  equal ~msg:"running_mean shape" (list int) [ 3 ]
    (Array.to_list (Nx.shape running_mean));
  equal ~msg:"running_var shape" (list int) [ 3 ]
    (Array.to_list (Nx.shape running_var))

let test_batch_norm_rank3_axes () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let m = Layer.batch_norm ~num_features:3 () in
  let vars = Layer.init m ~dtype:Nx.float32 in
  let param_fields = Ptree.Dict.fields_exn (Layer.params vars) in
  let scale = Ptree.Dict.get_tensor_exn param_fields ~name:"scale" Nx.float32 in
  let bias = Ptree.Dict.get_tensor_exn param_fields ~name:"bias" Nx.float32 in
  let x =
    Nx.create Nx.float32 [| 2; 3; 4 |]
      [|
        1.0;
        2.0;
        3.0;
        4.0;
        5.0;
        6.0;
        7.0;
        8.0;
        9.0;
        10.0;
        11.0;
        12.0;
        2.0;
        4.0;
        6.0;
        8.0;
        1.0;
        3.0;
        5.0;
        7.0;
        0.5;
        1.5;
        2.5;
        3.5;
      |]
  in
  let y, _ = Layer.apply m vars ~training:true x in
  let expected = Kaun.Fn.batch_norm ~axes:[ 0; 2 ] ~scale ~bias x in
  equal ~msg:"batch_norm rank3 uses [0;2] axes" bool true
    (tensor_close ~eps:1e-6 ~expected ~actual:y)

let test_batch_norm_running_stats_eval () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let m = Layer.batch_norm ~num_features:2 () in
  let vars0 = Layer.init m ~dtype:Nx.float32 in
  let x_train = Nx.create Nx.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let _y_train, vars1 = Layer.apply m vars0 ~training:true x_train in
  let param_fields = Ptree.Dict.fields_exn (Layer.params vars1) in
  let state_fields = Ptree.Dict.fields_exn (Layer.state vars1) in
  let scale = Ptree.Dict.get_tensor_exn param_fields ~name:"scale" Nx.float32 in
  let bias = Ptree.Dict.get_tensor_exn param_fields ~name:"bias" Nx.float32 in
  let running_mean =
    Ptree.Dict.get_tensor_exn state_fields ~name:"running_mean" Nx.float32
  in
  let running_var =
    Ptree.Dict.get_tensor_exn state_fields ~name:"running_var" Nx.float32
  in
  let x_eval = Nx.create Nx.float32 [| 2; 2 |] [| 10.0; 20.0; 30.0; 40.0 |] in
  let y_eval, vars2 = Layer.apply m vars1 ~training:false x_eval in
  let expected =
    Nx.standardize ~axes:[ 0 ] ~mean:running_mean ~variance:running_var x_eval
    |> fun z ->
    Nx.add (Nx.mul z (Nx.reshape [| 1; 2 |] scale)) (Nx.reshape [| 1; 2 |] bias)
  in
  equal ~msg:"batch_norm eval uses running stats" bool true
    (tensor_close ~eps:1e-6 ~expected ~actual:y_eval);
  let state_fields2 = Ptree.Dict.fields_exn (Layer.state vars2) in
  let running_mean2 =
    Ptree.Dict.get_tensor_exn state_fields2 ~name:"running_mean" Nx.float32
  in
  let running_var2 =
    Ptree.Dict.get_tensor_exn state_fields2 ~name:"running_var" Nx.float32
  in
  equal ~msg:"batch_norm eval keeps running_mean" bool true
    (tensor_close ~eps:1e-6 ~expected:running_mean ~actual:running_mean2);
  equal ~msg:"batch_norm eval keeps running_var" bool true
    (tensor_close ~eps:1e-6 ~expected:running_var ~actual:running_var2)

let test_batch_norm_eval_affine_rank3 () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let m = Layer.batch_norm ~num_features:3 () in
  let scale = Nx.create Nx.float32 [| 3 |] [| 2.0; 3.0; 4.0 |] in
  let bias = Nx.create Nx.float32 [| 3 |] [| 10.0; 20.0; 30.0 |] in
  let running_mean = Nx.create Nx.float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let running_var = Nx.create Nx.float32 [| 3 |] [| 4.0; 9.0; 16.0 |] in
  let vars =
    Layer.init m ~dtype:Nx.float32 |> fun vars ->
    Layer.with_params vars
      (Ptree.dict
         [ ("scale", Ptree.tensor scale); ("bias", Ptree.tensor bias) ])
    |> fun vars ->
    Layer.with_state vars
      (Ptree.dict
         [
           ("running_mean", Ptree.tensor running_mean);
           ("running_var", Ptree.tensor running_var);
         ])
  in
  let x =
    Nx.create Nx.float32 [| 1; 3; 2 |] [| 1.0; 5.0; 2.0; 8.0; 3.0; 11.0 |]
  in
  let y, _ = Layer.apply m vars ~training:false x in
  let expected =
    Nx.standardize ~axes:[ 0; 2 ] ~mean:running_mean ~variance:running_var x
    |> fun z ->
    Nx.add
      (Nx.mul z (Nx.reshape [| 1; 3; 1 |] scale))
      (Nx.reshape [| 1; 3; 1 |] bias)
  in
  equal ~msg:"batch_norm eval rank3 applies affine" bool true
    (tensor_close ~eps:1e-6 ~expected ~actual:y)

(* Embedding *)

let test_embedding_shapes () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let m = Layer.embedding ~vocab_size:100 ~embed_dim:16 () in
  let vars = Layer.init m ~dtype:Nx.float32 in
  let fields = Ptree.Dict.fields_exn (Layer.params vars) in
  let emb = Ptree.Dict.get_tensor_exn fields ~name:"embedding" Nx.float32 in
  equal ~msg:"embedding shape" (list int) [ 100; 16 ]
    (Array.to_list (Nx.shape emb))

let test_embedding_forward () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let m = Layer.embedding ~vocab_size:10 ~embed_dim:4 ~scale:false () in
  let vars = Layer.init m ~dtype:Nx.float32 in
  let indices = Nx.create Nx.int32 [| 3 |] [| 0l; 5l; 2l |] in
  let y = apply_out m vars ~training:false indices in
  equal ~msg:"embedding output shape" (list int) [ 3; 4 ]
    (Array.to_list (Nx.shape y))

let test_compose_embedding_linear () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let emb = Layer.embedding ~vocab_size:10 ~embed_dim:4 ~scale:false () in
  let proj = Layer.linear ~in_features:4 ~out_features:2 () in
  let m = Layer.compose emb proj in
  let vars = Layer.init m ~dtype:Nx.float32 in
  let indices = Nx.create Nx.int32 [| 3 |] [| 0l; 5l; 2l |] in
  let y, _ = Layer.apply m vars ~training:false indices in
  equal ~msg:"compose embedding+linear output shape" (list int) [ 3; 2 ]
    (Array.to_list (Nx.shape y))

(* Activations *)

let test_relu () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let m = Layer.relu () in
  let vars = Layer.init m ~dtype:Nx.float32 in
  let x = Nx.create Nx.float32 [| 4 |] [| -2.0; -0.5; 0.0; 3.0 |] in
  let y = apply_out m vars ~training:false x in
  let expected = Nx.create Nx.float32 [| 4 |] [| 0.0; 0.0; 0.0; 3.0 |] in
  equal ~msg:"relu" bool true (tensor_close ~eps:1e-6 ~expected ~actual:y)

let test_activation_no_params () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let activations =
    [
      Layer.relu ();
      Layer.gelu ();
      Layer.silu ();
      Layer.tanh ();
      Layer.sigmoid ();
    ]
  in
  let assert_no_params (m : (float, float) Layer.t) =
    let vars = Layer.init m ~dtype:Nx.float32 in
    match (Layer.params vars, Layer.state vars) with
    | Ptree.List [], Ptree.List [] -> ()
    | _ -> fail "expected empty params and state"
  in
  List.iter (fun m -> assert_no_params m) activations

(* Dropout *)

let test_dropout_eval_identity () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let m = Layer.dropout ~rate:0.99 () in
  let vars = Layer.init m ~dtype:Nx.float32 in
  let x = Nx.ones Nx.float32 [| 10 |] in
  let y = apply_out m vars ~training:false x in
  equal ~msg:"dropout eval = identity" bool true
    (tensor_close ~eps:1e-6 ~expected:x ~actual:y)

let test_dropout_training () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let m = Layer.dropout ~rate:0.5 () in
  let vars = Layer.init m ~dtype:Nx.float32 in
  let x = Nx.ones Nx.float32 [| 10 |] in
  let y = apply_out m vars ~training:true x in
  equal ~msg:"dropout training shape" (list int) [ 10 ]
    (Array.to_list (Nx.shape y))

let test_dropout_rate_bounds () =
  raises_match
    (fun exn -> match exn with Invalid_argument _ -> true | _ -> false)
    (fun () -> ignore (Layer.dropout ~rate:(-0.1) ()));
  raises_match
    (fun exn -> match exn with Invalid_argument _ -> true | _ -> false)
    (fun () -> ignore (Layer.dropout ~rate:1.0 ()))

(* Flatten *)

let test_flatten_forward () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let m = Layer.flatten () in
  let vars = Layer.init m ~dtype:Nx.float32 in
  let x = Nx.ones Nx.float32 [| 2; 3; 4 |] in
  let y = apply_out m vars ~training:false x in
  equal ~msg:"flatten shape" (list int) [ 2; 12 ] (Array.to_list (Nx.shape y))

(* Sequential *)

let test_sequential_init_structure () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let m =
    Layer.sequential
      [
        Layer.linear ~in_features:4 ~out_features:3 ();
        Layer.relu ();
        Layer.linear ~in_features:3 ~out_features:2 ();
      ]
  in
  let vars = Layer.init m ~dtype:Nx.float32 in
  let param_items = Ptree.List.items_exn (Layer.params vars) in
  let state_items = Ptree.List.items_exn (Layer.state vars) in
  equal ~msg:"sequential params length" int 3 (List.length param_items);
  equal ~msg:"sequential state length" int 3 (List.length state_items);
  let f0 = Ptree.Dict.fields_exn (List.nth param_items 0) in
  let w0 = Ptree.Dict.get_tensor_exn f0 ~name:"weight" Nx.float32 in
  equal ~msg:"layer0 weight shape" (list int) [ 4; 3 ]
    (Array.to_list (Nx.shape w0));
  (match List.nth param_items 1 with
  | Ptree.List [] -> ()
  | _ -> fail "relu should have no params");
  let f2 = Ptree.Dict.fields_exn (List.nth param_items 2) in
  let w2 = Ptree.Dict.get_tensor_exn f2 ~name:"weight" Nx.float32 in
  equal ~msg:"layer2 weight shape" (list int) [ 3; 2 ]
    (Array.to_list (Nx.shape w2))

let test_sequential_forward () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let m =
    Layer.sequential
      [
        Layer.linear ~in_features:4 ~out_features:3 ();
        Layer.relu ();
        Layer.linear ~in_features:3 ~out_features:2 ();
      ]
  in
  let vars = Layer.init m ~dtype:Nx.float32 in
  let x = Nx.ones Nx.float32 [| 5; 4 |] in
  let y = apply_out m vars ~training:false x in
  equal ~msg:"sequential output shape" (list int) [ 5; 2 ]
    (Array.to_list (Nx.shape y))

(* Convolution *)

let test_conv1d_shapes () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let m = Layer.conv1d ~in_channels:3 ~out_channels:8 () in
  let vars = Layer.init m ~dtype:Nx.float32 in
  let fields = Ptree.Dict.fields_exn (Layer.params vars) in
  let w = Ptree.Dict.get_tensor_exn fields ~name:"weight" Nx.float32 in
  let b = Ptree.Dict.get_tensor_exn fields ~name:"bias" Nx.float32 in
  equal ~msg:"weight shape" (list int) [ 8; 3; 3 ] (Array.to_list (Nx.shape w));
  equal ~msg:"bias shape" (list int) [ 8 ] (Array.to_list (Nx.shape b))

let test_conv1d_forward () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let m = Layer.conv1d ~in_channels:2 ~out_channels:4 ~kernel_size:3 () in
  let vars = Layer.init m ~dtype:Nx.float32 in
  let x = Nx.ones Nx.float32 [| 1; 2; 10 |] in
  let y = apply_out m vars ~training:false x in
  let shape = Nx.shape y in
  equal ~msg:"conv1d output batch" int 1 shape.(0);
  equal ~msg:"conv1d output channels" int 4 shape.(1);
  equal ~msg:"conv1d output length" int 10 shape.(2)

let test_conv2d_shapes () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let m = Layer.conv2d ~in_channels:3 ~out_channels:16 () in
  let vars = Layer.init m ~dtype:Nx.float32 in
  let fields = Ptree.Dict.fields_exn (Layer.params vars) in
  let w = Ptree.Dict.get_tensor_exn fields ~name:"weight" Nx.float32 in
  let b = Ptree.Dict.get_tensor_exn fields ~name:"bias" Nx.float32 in
  equal ~msg:"weight shape" (list int) [ 16; 3; 3; 3 ]
    (Array.to_list (Nx.shape w));
  equal ~msg:"bias shape" (list int) [ 16 ] (Array.to_list (Nx.shape b))

let test_conv2d_forward () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let m = Layer.conv2d ~in_channels:1 ~out_channels:4 ~kernel_size:(3, 3) () in
  let vars = Layer.init m ~dtype:Nx.float32 in
  let x = Nx.ones Nx.float32 [| 1; 1; 8; 8 |] in
  let y = apply_out m vars ~training:false x in
  equal ~msg:"conv2d output shape" (list int) [ 1; 4; 8; 8 ]
    (Array.to_list (Nx.shape y))

(* Pooling *)

let test_max_pool2d () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let m = Layer.max_pool2d ~kernel_size:(2, 2) () in
  let vars = Layer.init m ~dtype:Nx.float32 in
  let x = Nx.ones Nx.float32 [| 1; 1; 4; 4 |] in
  let y = apply_out m vars ~training:false x in
  equal ~msg:"max_pool2d shape" (list int) [ 1; 1; 2; 2 ]
    (Array.to_list (Nx.shape y))

let test_avg_pool2d () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let m = Layer.avg_pool2d ~kernel_size:(2, 2) () in
  let vars = Layer.init m ~dtype:Nx.float32 in
  let x = Nx.ones Nx.float32 [| 1; 1; 6; 6 |] in
  let y = apply_out m vars ~training:false x in
  equal ~msg:"avg_pool2d shape" (list int) [ 1; 1; 3; 3 ]
    (Array.to_list (Nx.shape y))

(* Parameter count *)

let test_param_count () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let m = Layer.linear ~in_features:10 ~out_features:5 () in
  let vars = Layer.init m ~dtype:Nx.float32 in
  equal ~msg:"linear param count" int 55
    (Ptree.count_parameters (Layer.params vars))

let () =
  run "Kaun.Layer"
    [
      group "linear"
        [
          test "shapes" test_linear_shapes;
          test "forward" test_linear_forward;
          test "manual params" test_linear_manual_params;
          test "param count" test_param_count;
        ];
      group "normalization"
        [
          test "layer_norm shapes" test_layer_norm_shapes;
          test "layer_norm forward" test_layer_norm_forward;
          test "rms_norm shapes" test_rms_norm_shapes;
          test "batch_norm shapes" test_batch_norm_shapes;
          test "batch_norm rank3 axes" test_batch_norm_rank3_axes;
          test "batch_norm running stats eval"
            test_batch_norm_running_stats_eval;
          test "batch_norm eval affine rank3" test_batch_norm_eval_affine_rank3;
        ];
      group "embedding"
        [
          test "shapes" test_embedding_shapes;
          test "forward" test_embedding_forward;
          test "compose embedding+linear" test_compose_embedding_linear;
        ];
      group "activation"
        [ test "relu" test_relu; test "no params" test_activation_no_params ];
      group "regularization"
        [
          test "dropout eval identity" test_dropout_eval_identity;
          test "dropout training" test_dropout_training;
          test "dropout rate bounds" test_dropout_rate_bounds;
        ];
      group "conv"
        [
          test "conv1d shapes" test_conv1d_shapes;
          test "conv1d forward" test_conv1d_forward;
          test "conv2d shapes" test_conv2d_shapes;
          test "conv2d forward" test_conv2d_forward;
        ];
      group "pooling"
        [ test "max_pool2d" test_max_pool2d; test "avg_pool2d" test_avg_pool2d ];
      group "reshape" [ test "flatten" test_flatten_forward ];
      group "sequential"
        [
          test "init structure" test_sequential_init_structure;
          test "forward" test_sequential_forward;
        ];
    ]
