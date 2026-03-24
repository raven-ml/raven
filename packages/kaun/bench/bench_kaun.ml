(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Fn = Kaun.Fn
module Layer = Kaun.Layer
module Loss = Kaun.Loss
module Attention = Kaun.Attention

let batch = 32
let seq_len = 128
let dim = 256
let num_heads = 8
let num_classes = 10

let normalization_benchmarks () =
  let x = Nx.rand Nx.float32 [| batch; seq_len; dim |] in
  let gamma = Nx.ones Nx.float32 [| dim |] in
  let beta = Nx.zeros Nx.float32 [| dim |] in
  let x_4d = Nx.rand Nx.float32 [| batch; dim; 8; 8 |] in
  let bn4_scale = Nx.ones Nx.float32 [| dim |] in
  let bn4_bias = Nx.zeros Nx.float32 [| dim |] in
  let x_2d = Nx.rand Nx.float32 [| batch; dim |] in
  let bn2_scale = Nx.ones Nx.float32 [| dim |] in
  let bn2_bias = Nx.zeros Nx.float32 [| dim |] in
  [
    Thumper.bench "layer_norm [32;128;256]" (fun () ->
        Fn.layer_norm ~gamma ~beta x);
    Thumper.bench "rms_norm [32;128;256]" (fun () -> Fn.rms_norm x);
    Thumper.bench "batch_norm [32;256;8;8]" (fun () ->
        Fn.batch_norm ~scale:bn4_scale ~bias:bn4_bias x_4d);
    Thumper.bench "batch_norm [32;256]" (fun () ->
        Fn.batch_norm ~scale:bn2_scale ~bias:bn2_bias x_2d);
  ]

let attention_benchmarks () =
  let head_dim = dim / num_heads in
  let q = Nx.rand Nx.float32 [| batch; num_heads; seq_len; head_dim |] in
  let k = Nx.rand Nx.float32 [| batch; num_heads; seq_len; head_dim |] in
  let v = Nx.rand Nx.float32 [| batch; num_heads; seq_len; head_dim |] in
  let x = Nx.rand Nx.float32 [| batch; seq_len; head_dim |] in
  [
    Thumper.bench "dot_product_attention [32;8;128;32]" (fun () ->
        Fn.dot_product_attention q k v);
    Thumper.bench "dot_product_attention causal [32;8;128;32]" (fun () ->
        Fn.dot_product_attention ~is_causal:true q k v);
    Thumper.bench "rope [32;128;32]" (fun () -> Attention.rope x);
  ]

let loss_benchmarks () =
  let logits = Nx.rand Nx.float32 [| batch; num_classes |] in
  let labels_onehot = Nx.zeros Nx.float32 [| batch; num_classes |] in
  let targets = Nx.rand Nx.float32 [| batch; num_classes |] in
  let predictions = Nx.rand Nx.float32 [| batch; num_classes |] in
  let binary_logits = Nx.rand Nx.float32 [| batch |] in
  let binary_labels = Nx.zeros Nx.float32 [| batch |] in
  [
    Thumper.bench "cross_entropy [32;10]" (fun () ->
        Loss.cross_entropy logits labels_onehot);
    Thumper.bench "binary_cross_entropy [32]" (fun () ->
        Loss.binary_cross_entropy binary_logits binary_labels);
    Thumper.bench "mse [32;10]" (fun () -> Loss.mse predictions targets);
    Thumper.bench "mae [32;10]" (fun () -> Loss.mae predictions targets);
  ]

let conv_benchmarks () =
  let x1d = Nx.rand Nx.float32 [| batch; 64; 128 |] in
  let w1d = Nx.rand Nx.float32 [| 128; 64; 3 |] in
  let x2d = Nx.rand Nx.float32 [| batch; 64; 32; 32 |] in
  let w2d = Nx.rand Nx.float32 [| 128; 64; 3; 3 |] in
  [
    Thumper.bench "conv1d [32;64;128] k=3" (fun () -> Fn.conv1d x1d w1d);
    Thumper.bench "conv1d same [32;64;128] k=3" (fun () ->
        Fn.conv1d ~padding:`Same x1d w1d);
    Thumper.bench "conv2d [32;64;32;32] k=3x3" (fun () -> Fn.conv2d x2d w2d);
    Thumper.bench "conv2d same [32;64;32;32] k=3x3" (fun () ->
        Fn.conv2d ~padding:`Same x2d w2d);
  ]

let pooling_benchmarks () =
  let x2d = Nx.rand Nx.float32 [| batch; 64; 32; 32 |] in
  [
    Thumper.bench "max_pool2d [32;64;32;32] k=2x2" (fun () ->
        Fn.max_pool2d ~kernel_size:(2, 2) x2d);
    Thumper.bench "avg_pool2d [32;64;32;32] k=2x2" (fun () ->
        Fn.avg_pool2d ~kernel_size:(2, 2) x2d);
  ]

let layer_benchmarks () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let linear_layer = Layer.linear ~in_features:dim ~out_features:dim () in
  let linear_vars = Layer.init linear_layer ~dtype:Nx.float32 in
  let ln_layer = Layer.layer_norm ~dim () in
  let ln_vars = Layer.init ln_layer ~dtype:Nx.float32 in
  let mha_layer = Attention.multi_head_attention ~embed_dim:dim ~num_heads () in
  let mha_vars = Layer.init mha_layer ~dtype:Nx.float32 in
  let x = Nx.rand Nx.float32 [| batch; seq_len; dim |] in
  [
    Thumper.bench "Layer.linear [32;128;256]->[32;128;256]" (fun () ->
        Layer.apply linear_layer linear_vars ~training:false x);
    Thumper.bench "Layer.layer_norm [32;128;256]" (fun () ->
        Layer.apply ln_layer ln_vars ~training:false x);
    Thumper.bench "Layer.multi_head_attention [32;128;256] h=8" (fun () ->
        Layer.apply mha_layer mha_vars ~training:false x);
  ]

let embedding_benchmarks () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let vocab_size = 32000 in
  let embed_dim = dim in
  let table = Nx.rand Nx.float32 [| vocab_size; embed_dim |] in
  let indices =
    Nx.create Nx.int32 [| batch; seq_len |]
      (Array.init (batch * seq_len) (fun i -> Int32.of_int (i mod vocab_size)))
  in
  [
    Thumper.bench "embedding [32;128] vocab=32000 dim=256" (fun () ->
        Fn.embedding ~embedding:table indices);
  ]

let build_benchmarks () =
  [
    Thumper.group "Normalization" (normalization_benchmarks ());
    Thumper.group "Attention" (attention_benchmarks ());
    Thumper.group "Loss" (loss_benchmarks ());
    Thumper.group "Convolution" (conv_benchmarks ());
    Thumper.group "Pooling" (pooling_benchmarks ());
    Thumper.group "Layer" (layer_benchmarks ());
    Thumper.group "Embedding" (embedding_benchmarks ());
  ]

let () =
  let benchmarks = build_benchmarks () in
  Thumper.run "kaun" benchmarks
