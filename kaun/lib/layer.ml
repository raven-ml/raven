open! Module

let conv2d ~in_channels ~out_channels ?(kernel_size = (3, 3)) () =
  let kh, kw = kernel_size in
  {
    init =
      (fun (type l d) ~rngs (x : (l, d) tensor) ->
        Rune.debug_with_context
          (Printf.sprintf "conv2d_%dx%d_%dx%d_init" in_channels out_channels kh
             kw) (fun () ->
            let rngs_split = Rune.Rng.split rngs in
            let rng1 = rngs_split.(0) in
            let dev = Rune.device x in
            let dtype = Rune.dtype x in
            let fan_in = in_channels * kh * kw in
            let fan_out = out_channels * kh * kw in
            let limit = sqrt (6.0 /. float_of_int (fan_in + fan_out)) in
            let weight_shape = [| out_channels; in_channels; kh; kw |] in
            let w = Rune.Rng.uniform rng1 dev dtype weight_shape in
            let w =
              Rune.sub
                (Rune.mul w (Rune.scalar dev dtype (2.0 *. limit)))
                (Rune.scalar dev dtype limit)
            in
            let b = Rune.zeros dev dtype [| out_channels |] in
            Ptree.record_of [ ("weight", Tensor w); ("bias", Tensor b) ]));
    apply =
      (fun (type l d)
        (params : (l, d) Ptree.t)
        ~training:_
        ?rngs:_
        (x : (l, d) tensor)
      ->
        match params with
        | Record fields ->
            (* Handle fields in any order *)
            let w =
              match Ptree.Record.find_opt "weight" fields with
              | Some (Tensor t) -> t
              | _ -> failwith "conv2d: missing or invalid weight parameter"
            in
            let b =
              match Ptree.Record.find_opt "bias" fields with
              | Some (Tensor t) -> t
              | _ -> failwith "conv2d: missing or invalid bias parameter"
            in
            Rune.debug_with_context
              (Printf.sprintf "conv2d_%dx%d_%dx%d" in_channels out_channels kh
                 kw) (fun () ->
                let conv =
                  Rune.convolve2d x w ~stride:(1, 1) ~padding_mode:`Same
                in
                let b_reshaped = Rune.reshape [| 1; out_channels; 1; 1 |] b in
                Rune.add conv b_reshaped)
        | _ -> failwith "conv2d: invalid params structure");
  }

let linear ~in_features ~out_features ?weight_init ?bias_init () =
  {
    init =
      (fun (type l d) ~rngs (x : (l, d) tensor) ->
        Rune.debug_with_context
          (Printf.sprintf "linear_%dx%d_init" in_features out_features)
          (fun () ->
            (* Get initializers with correct types in this scope *)
            let weight_init_f =
              match weight_init with
              | Some init -> init.Initializers.f
              | None -> (Initializers.glorot_uniform ()).f
            in
            let bias_init_f =
              match bias_init with
              | Some init -> init.Initializers.f
              | None -> (Initializers.zeros ()).f
            in
            (* Use the rngs passed during initialization *)
            let rngs_split = Rune.Rng.split rngs in
            let rng1 = rngs_split.(0) in
            let rng2 = rngs_split.(1) in
            let dev = Rune.device x in
            let dtype = Rune.dtype x in

            let w =
              weight_init_f (Rune.Rng.to_int rng1)
                [| in_features; out_features |]
                dev dtype
            in
            let b =
              bias_init_f (Rune.Rng.to_int rng2) [| out_features |] dev dtype
            in
            Ptree.record_of [ ("weight", Tensor w); ("bias", Tensor b) ]));
    apply =
      (fun (type l d)
        (params : (l, d) Ptree.t)
        ~training:_
        ?rngs:_
        (x : (l, d) tensor)
      ->
        Rune.debug_with_context
          (Printf.sprintf "linear_%dx%d" in_features out_features) (fun () ->
            match params with
            | Record fields ->
                let w =
                  match Ptree.Record.find_opt "weight" fields with
                  | Some (Tensor t) -> t
                  | _ -> failwith "linear: missing or invalid weight parameter"
                in
                let b =
                  match Ptree.Record.find_opt "bias" fields with
                  | Some (Tensor t) -> t
                  | _ -> failwith "linear: missing or invalid bias parameter"
                in
                let z = Rune.matmul x w in
                Rune.add z b
            | _ -> failwith "linear: invalid params structure"));
  }

let dropout ~rate () =
  {
    init = (fun ~rngs:_ _x -> List []);
    (* No params needed *)
    apply =
      (fun _params ~training ?rngs x ->
        if training && rate > 0.0 then
          match rngs with
          | Some rng ->
              (* Generate dropout mask *)
              let key = (Rune.Rng.split rng).(0) in
              let dev = Rune.device x in
              let dtype = Rune.dtype x in
              let shape = Rune.shape x in
              let mask = Rune.Rng.uniform key dev dtype shape in
              let keep_prob = 1.0 -. rate in
              let threshold = Rune.scalar dev dtype keep_prob in
              let binary_mask = Rune.less mask threshold in
              let binary_mask_float = Rune.cast dtype binary_mask in
              let scale = Rune.scalar dev dtype (1.0 /. keep_prob) in
              (* Apply mask and scale *)
              Rune.mul x (Rune.mul binary_mask_float scale)
          | None -> failwith "dropout requires RNG during training"
        else x);
  }

let batch_norm ~num_features () =
  {
    init =
      (fun ~rngs x ->
        Rune.debug_with_context
          (Printf.sprintf "batch_norm_%d_init" num_features) (fun () ->
            let _rngs_split = Rune.Rng.split rngs in
            let dev = Rune.device x in
            let dtype = Rune.dtype x in
            (* Initialize scale (gamma) to 1 and bias (beta) to 0 *)
            let scale = Rune.ones dev dtype [| num_features |] in
            let bias = Rune.zeros dev dtype [| num_features |] in
            (* We don't track running stats in this simple implementation *)
            Ptree.record_of [ ("scale", Tensor scale); ("bias", Tensor bias) ]));
    apply =
      (fun params ~training:_ ?rngs:_ x ->
        match params with
        | Record fields ->
            (* Handle fields in any order *)
            let scale =
              match Ptree.Record.find_opt "scale" fields with
              | Some (Tensor t) -> t
              | _ -> failwith "batch_norm: missing or invalid scale parameter"
            in
            let bias =
              match Ptree.Record.find_opt "bias" fields with
              | Some (Tensor t) -> t
              | _ -> failwith "batch_norm: missing or invalid bias parameter"
            in
            (* Compute batch statistics *)
            let axes =
              match Array.length (Rune.shape x) with
              | 2 -> [| 0 |] (* (batch, features) *)
              | 4 -> [| 0; 2; 3 |] (* (batch, channels, height, width) *)
              | _ -> [| 0 |]
              (* Default to first axis *)
            in
            Rune.debug_with_context
              (Printf.sprintf "batch_norm_%d_apply" num_features) (fun () ->
                (* Compute mean and variance *)
                let mean = Rune.mean x ~axes ~keepdims:true in
                let variance = Rune.var x ~axes ~keepdims:true in
                let eps = 1e-5 in
                let dtype = Rune.dtype x in
                let dev = Rune.device x in
                let epsilon = Rune.scalar dev dtype eps in
                (* Normalize *)
                let x_normalized =
                  Rune.div (Rune.sub x mean)
                    (Rune.sqrt (Rune.add variance epsilon))
                in
                (* Scale and shift *)
                let scale_shape =
                  match Array.length (Rune.shape x) with
                  | 2 -> [| 1; num_features |]
                  | 4 -> [| 1; num_features; 1; 1 |]
                  | _ -> [| 1; num_features |]
                in
                let scale_reshaped = Rune.reshape scale_shape scale in
                let bias_reshaped = Rune.reshape scale_shape bias in
                Rune.add (Rune.mul x_normalized scale_reshaped) bias_reshaped)
        | _ -> failwith "batch_norm: invalid params structure");
  }

let max_pool2d ~kernel_size ?stride () =
  let stride = match stride with Some s -> s | None -> kernel_size in
  {
    init = (fun ~rngs:_ _x -> List []);
    apply =
      (fun _params ~training:_ ?rngs:_ x ->
        let pooled, _ = Rune.max_pool2d x ~kernel_size ~stride in
        pooled);
  }

let avg_pool2d ~kernel_size ?stride () =
  let stride = match stride with Some s -> s | None -> kernel_size in
  {
    init = (fun ~rngs:_ _x -> List []);
    apply =
      (fun _params ~training:_ ?rngs:_ x ->
        Rune.avg_pool2d x ~kernel_size ~stride);
  }

let flatten () =
  {
    init = (fun ~rngs:_ _x -> List []);
    apply =
      (fun _params ~training:_ ?rngs:_ x ->
        let shape = Rune.shape x in
        let batch_size = shape.(0) in
        let flat_size =
          Array.fold_left ( * ) 1 (Array.sub shape 1 (Array.length shape - 1))
        in
        (* Ensure tensor is contiguous before reshaping *)
        let x = if Rune.is_c_contiguous x then x else Rune.contiguous x in
        Rune.reshape [| batch_size; flat_size |] x);
  }

let relu () =
  {
    init = (fun ~rngs:_ _x -> List []);
    apply = (fun _params ~training:_ ?rngs:_ x -> Rune.relu x);
  }

let sigmoid () =
  {
    init = (fun ~rngs:_ _x -> List []);
    apply = (fun _params ~training:_ ?rngs:_ x -> Rune.sigmoid x);
  }

let tanh () =
  {
    init = (fun ~rngs:_ _x -> List []);
    apply = (fun _params ~training:_ ?rngs:_ x -> Rune.tanh x);
  }

let sequential models =
  {
    init =
      (fun ~rngs x ->
        let rec init_layers models x acc rngs_current layer_idx =
          match models with
          | [] -> Ptree.List (List.rev acc)
          | m :: rest ->
              let rngs_split = Rune.Rng.split rngs_current in
              let rngs_layer = rngs_split.(0) in
              let rngs_rest = rngs_split.(1) in
              let params = m.init ~rngs:rngs_layer x in
              let x' = m.apply params ~training:false x in
              init_layers rest x' (params :: acc) rngs_rest (layer_idx + 1)
        in
        init_layers models x [] rngs 1);
    apply =
      (fun params ~training ?rngs:_ x ->
        match params with
        | List param_list ->
            let rec apply_layers models params x layer_idx =
              match (models, params) with
              | [], [] -> x
              | m :: ms, p :: ps ->
                  let x' = m.apply p ~training x in
                  apply_layers ms ps x' (layer_idx + 1)
              | _ -> failwith "sequential: mismatched models and params"
            in
            apply_layers models param_list x 1
        | _ -> failwith "sequential: invalid params structure");
  }

let einsum ~einsum_str ~shape ?kernel_init () =
  {
    init =
      (fun (type l d) ~rngs (_x : (l, d) tensor) ->
        let kernel_init_f =
          match kernel_init with
          | Some init -> init.Initializers.f
          | None -> (Initializers.glorot_uniform ()).f
        in
        let dev = Rune.device _x in
        let dtype = Rune.dtype _x in
        let key = (Rune.Rng.split rngs).(0) in
        let w = kernel_init_f (Rune.Rng.to_int key) shape dev dtype in
        Ptree.Tensor w);
    apply =
      (fun params ~training:_ ?rngs:_ x ->
        match params with
        | Tensor w ->
            (* Use Rune.einsum which exists *)
            Rune.einsum einsum_str [| x; w |]
        | _ -> failwith "einsum: invalid params");
  }

let rms_norm ~dim ?(eps = 1e-6) ?scale_init () =
  {
    init =
      (fun (type l d) ~rngs (x : (l, d) tensor) ->
        let scale_init_f =
          match scale_init with
          | Some init -> init.Initializers.f
          | None -> (Initializers.ones ()).f
        in
        let dev = Rune.device x in
        let dtype = Rune.dtype x in
        let key = (Rune.Rng.split rngs).(0) in
        let scale = scale_init_f (Rune.Rng.to_int key) [| dim |] dev dtype in
        Ptree.Tensor scale);
    apply =
      (fun params ~training:_ ?rngs:_ x ->
        match params with
        | Tensor scale ->
            let var = Rune.mean (Rune.square x) ~axes:[| -1 |] ~keepdims:true in
            let normed =
              Rune.mul x
                (Rune.rsqrt
                   (Rune.add var
                      (Rune.scalar (Rune.device x) (Rune.dtype x) eps)))
            in
            let scale_expanded =
              let x_dims = Array.length (Rune.shape x) in
              let scale_shape = Array.make x_dims 1 in
              scale_shape.(x_dims - 1) <- dim;
              Rune.reshape scale_shape scale
            in
            Rune.mul normed
              (Rune.add
                 (Rune.scalar (Rune.device x) (Rune.dtype x) 1.0)
                 scale_expanded)
        | _ -> failwith "rms_norm: invalid params");
  }

let layer_norm ~dim ?(eps = 1e-5) ?(elementwise_affine = true) () =
  {
    init =
      (fun ~rngs:_ x ->
        if elementwise_affine then
          let dev = Rune.device x in
          let dtype = Rune.dtype x in
          let gamma = Rune.ones dev dtype [| dim |] in
          let beta = Rune.zeros dev dtype [| dim |] in
          Ptree.record_of [ ("gamma", Tensor gamma); ("beta", Tensor beta) ]
        else List []);
    apply =
      (fun params ~training:_ ?rngs:_ x ->
        let mean = Rune.mean x ~axes:[| -1 |] ~keepdims:true in
        let var = Rune.var x ~axes:[| -1 |] ~keepdims:true in
        let eps_scalar = Rune.scalar (Rune.device x) (Rune.dtype x) eps in
        let normalized =
          Rune.div (Rune.sub x mean) (Rune.sqrt (Rune.add var eps_scalar))
        in
        if elementwise_affine then
          match params with
          | Record fields ->
              let gamma =
                match Ptree.Record.find_opt "gamma" fields with
                | Some (Tensor t) -> t
                | _ -> failwith "layer_norm: missing gamma"
              in
              let beta =
                match Ptree.Record.find_opt "beta" fields with
                | Some (Tensor t) -> t
                | _ -> failwith "layer_norm: missing beta"
              in
              let x_dims = Array.length (Rune.shape x) in
              let reshape_dims = Array.make x_dims 1 in
              reshape_dims.(x_dims - 1) <- dim;
              let gamma_reshaped = Rune.reshape reshape_dims gamma in
              let beta_reshaped = Rune.reshape reshape_dims beta in
              Rune.add (Rune.mul normalized gamma_reshaped) beta_reshaped
          | _ -> failwith "layer_norm: invalid params"
        else normalized);
  }

let embedding ~vocab_size ~embed_dim ?(scale = true) ?embedding_init () =
  {
    init =
      (fun (type l d) ~rngs (_x : (l, d) tensor) ->
        let embedding_init_f =
          match embedding_init with
          | Some init -> init.Initializers.f
          | None -> (Initializers.normal_range ~mean:0.0 ~stddev:0.02 ()).f
        in
        let dev = Rune.device _x in
        let dtype = Rune.dtype _x in
        let key = (Rune.Rng.split rngs).(0) in
        let embedding =
          embedding_init_f (Rune.Rng.to_int key)
            [| vocab_size; embed_dim |]
            dev dtype
        in
        Ptree.Tensor embedding);
    apply =
      (fun params ~training:_ ?rngs:_ x ->
        match params with
        | Tensor embedding ->
            (* For now, implement a simple embedding lookup *)
            (* This is a placeholder - proper implementation would use gather op *)
            (* We'll just return a dummy tensor of the right shape for now *)
            let batch_shape = Rune.shape x in
            let output_shape = Array.append batch_shape [| embed_dim |] in
            let dev = Rune.device x in
            let dtype = Rune.dtype embedding in

            (* Create output tensor - this should be replaced with proper
               gather *)
            let embedded = Rune.zeros dev dtype output_shape in
            if scale then
              let scale_factor = sqrt (float_of_int embed_dim) in
              Rune.mul embedded
                (Rune.scalar (Rune.device embedded) (Rune.dtype embedded)
                   scale_factor)
            else embedded
        | _ -> failwith "embedding: invalid params");
  }

let gelu () =
  {
    init = (fun ~rngs:_ _x -> List []);
    apply =
      (fun _params ~training:_ ?rngs:_ x ->
        (* GELU(x) = x * Φ(x) where Φ is the CDF of standard normal distribution *)
        (* Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3))) *)
        let sqrt_2_over_pi = 0.7978845608 in
        let x_cubed = Rune.mul x (Rune.mul x x) in
        let inner =
          Rune.add x
            (Rune.mul
               (Rune.scalar (Rune.device x) (Rune.dtype x) 0.044715)
               x_cubed)
        in
        let tanh_arg =
          Rune.mul
            (Rune.scalar (Rune.device x) (Rune.dtype x) sqrt_2_over_pi)
            inner
        in
        Rune.mul x
          (Rune.mul
             (Rune.scalar (Rune.device x) (Rune.dtype x) 0.5)
             (Rune.add
                (Rune.scalar (Rune.device x) (Rune.dtype x) 1.0)
                (Rune.tanh tanh_arg))));
  }

let swish () =
  {
    init = (fun ~rngs:_ _x -> List []);
    apply =
      (fun _params ~training:_ ?rngs:_ x ->
        (* Swish(x) = x * sigmoid(x) *)
        Rune.mul x (Rune.sigmoid x));
  }

let multi_head_attention ~embed_dim ~num_heads ?num_kv_heads:_ ?head_dim:_
    ?dropout:_ ?use_qk_norm:_ ?attn_logits_soft_cap:_ ?query_pre_attn_scalar:_
    () =
  let _ = embed_dim in
  let _ = num_heads in
  {
    init = (fun ~rngs:_ _x -> List []);
    apply =
      (fun _params ~training:_ ?rngs:_ _x ->
        failwith
          "multi_head_attention: not yet fully implemented - needs attention \
           mechanism");
  }

let rope_embedding ~dim ?max_seq_len:_ ?base_frequency:_ ?scale_factor:_ () =
  let _ = dim in
  {
    init = (fun ~rngs:_ _x -> List []);
    apply =
      (fun _params ~training:_ ?rngs:_ _x ->
        failwith
          "rope_embedding: not yet implemented - needs complex number support");
  }

let sinusoidal_pos_embedding ~max_len ~embed_dim () =
  let _ = max_len in
  let _ = embed_dim in
  {
    init = (fun ~rngs:_ _x -> List []);
    apply =
      (fun _params ~training:_ ?rngs:_ _x ->
        failwith
          "sinusoidal_pos_embedding: not yet implemented - needs sin/cos \
           filling");
  }
