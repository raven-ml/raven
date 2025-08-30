type ('layout, 'dev) tensor = (float, 'layout, 'dev) Rune.t

type module_ = {
  init :
    'layout 'dev.
    rngs:Rune.Rng.key ->
    device:'dev Rune.device ->
    dtype:(float, 'layout) Rune.dtype ->
    ('layout, 'dev) Ptree.t;
  apply :
    'layout 'dev.
    ('layout, 'dev) Ptree.t ->
    training:bool ->
    ?rngs:Rune.Rng.key ->
    ('layout, 'dev) tensor ->
    ('layout, 'dev) tensor;
}

let relu () =
  {
    init = (fun ~rngs:_ ~device:_ ~dtype:_ -> List []);
    apply = (fun _params ~training:_ ?rngs:_ x -> Rune.relu x);
  }

let sigmoid () =
  {
    init = (fun ~rngs:_ ~device:_ ~dtype:_ -> List []);
    apply = (fun _params ~training:_ ?rngs:_ x -> Rune.sigmoid x);
  }

let tanh () =
  {
    init = (fun ~rngs:_ ~device:_ ~dtype:_ -> List []);
    apply = (fun _params ~training:_ ?rngs:_ x -> Rune.tanh x);
  }

let gelu () =
  {
    init = (fun ~rngs:_ ~device:_ ~dtype:_ -> List []);
    apply = (fun _params ~training:_ ?rngs:_ x -> Activations.gelu x);
  }

let swish () =
  {
    init = (fun ~rngs:_ ~device:_ ~dtype:_ -> List []);
    apply = (fun _params ~training:_ ?rngs:_ x -> Activations.swish x);
  }

let conv2d ~in_channels ~out_channels ?(kernel_size = (3, 3)) () =
  let kh, kw = kernel_size in
  {
    init =
      (fun (type l d)
        ~rngs
        ~device:(dev : d Rune.device)
        ~(dtype : (float, l) Rune.dtype)
      ->
        Rune.debug_with_context
          (Printf.sprintf "conv2d_%dx%d_%dx%d_init" in_channels out_channels kh
             kw) (fun () ->
            let rngs_split = Rune.Rng.split rngs in
            let rng1 = rngs_split.(0) in
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
      (fun ~rngs ~device ~dtype ->
        Rune.debug_with_context
          (Printf.sprintf "linear_%dx%d_init" in_features out_features)
          (fun () ->
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
            let rngs_split = Rune.Rng.split rngs in
            let rng1 = rngs_split.(0) in
            let rng2 = rngs_split.(1) in
            let w =
              weight_init_f (Rune.Rng.to_int rng1)
                [| in_features; out_features |]
                device dtype
            in
            let b =
              bias_init_f (Rune.Rng.to_int rng2) [| out_features |] device dtype
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
    init = (fun ~rngs:_ ~device:_ ~dtype:_ -> List []);
    apply =
      (fun _params ~training ?rngs x ->
        if training then Ops.dropout ~rate ?rngs x else x);
  }

(* alias for internal use *)
let dropout_layer = dropout

let batch_norm ~num_features () =
  {
    init =
      (fun ~rngs ~device ~dtype ->
        Rune.debug_with_context
          (Printf.sprintf "batch_norm_%d_init" num_features) (fun () ->
            let _rngs_split = Rune.Rng.split rngs in
            let scale = Rune.ones device dtype [| num_features |] in
            let bias = Rune.zeros device dtype [| num_features |] in
            Ptree.record_of [ ("scale", Tensor scale); ("bias", Tensor bias) ]));
    apply =
      (fun params ~training:_ ?rngs:_ x ->
        match params with
        | Record fields ->
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
            Rune.debug_with_context
              (Printf.sprintf "batch_norm_%d_apply" num_features) (fun () ->
                Ops.batch_norm ~scale ~bias ~num_features x)
        | _ -> failwith "batch_norm: invalid params structure");
  }

let max_pool2d ~kernel_size ?stride () =
  let stride = match stride with Some s -> s | None -> kernel_size in
  {
    init = (fun ~rngs:_ ~device:_ ~dtype:_ -> List []);
    apply =
      (fun _params ~training:_ ?rngs:_ x ->
        let pooled, _ = Rune.max_pool2d x ~kernel_size ~stride in
        pooled);
  }

let avg_pool2d ~kernel_size ?stride () =
  let stride = match stride with Some s -> s | None -> kernel_size in
  {
    init = (fun ~rngs:_ ~device:_ ~dtype:_ -> List []);
    apply =
      (fun _params ~training:_ ?rngs:_ x ->
        Rune.avg_pool2d x ~kernel_size ~stride);
  }

let flatten () =
  {
    init = (fun ~rngs:_ ~device:_ ~dtype:_ -> List []);
    apply =
      (fun _params ~training:_ ?rngs:_ x ->
        let shape = Rune.shape x in
        let batch_size = shape.(0) in
        let flat_size =
          Array.fold_left ( * ) 1 (Array.sub shape 1 (Array.length shape - 1))
        in
        let x = if Rune.is_c_contiguous x then x else Rune.contiguous x in
        Rune.reshape [| batch_size; flat_size |] x);
  }

let sequential models =
  {
    init =
      (fun ~rngs ~device ~dtype ->
        let rec init_layers models acc rngs_current =
          match models with
          | [] -> Ptree.List (List.rev acc)
          | m :: rest ->
              let rngs_split = Rune.Rng.split rngs_current in
              let rngs_layer = rngs_split.(0) in
              let rngs_rest = rngs_split.(1) in
              let params = m.init ~rngs:rngs_layer ~device ~dtype in
              init_layers rest (params :: acc) rngs_rest
        in
        init_layers models [] rngs);
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
      (fun ~rngs ~device ~dtype ->
        let kernel_init_f =
          match kernel_init with
          | Some init -> init.Initializers.f
          | None -> (Initializers.glorot_uniform ()).f
        in
        let key = (Rune.Rng.split rngs).(0) in
        let w = kernel_init_f (Rune.Rng.to_int key) shape device dtype in
        Ptree.Tensor w);
    apply =
      (fun params ~training:_ ?rngs:_ x ->
        match params with
        | Tensor w -> Rune.einsum einsum_str [| x; w |]
        | _ -> failwith "einsum: invalid params");
  }

let rms_norm ~dim ?(eps = 1e-6) ?scale_init () =
  {
    init =
      (fun ~rngs ~device ~dtype ->
        let scale_init_f =
          match scale_init with
          | Some init -> init.Initializers.f
          | None -> (Initializers.ones ()).f
        in
        let key = (Rune.Rng.split rngs).(0) in
        let scale = scale_init_f (Rune.Rng.to_int key) [| dim |] device dtype in
        Ptree.Tensor scale);
    apply =
      (fun params ~training:_ ?rngs:_ x ->
        match params with
        | Tensor scale -> Ops.rms_norm ~scale ~dim ~eps x
        | _ -> failwith "rms_norm: invalid params");
  }

let layer_norm ~dim ?(eps = 1e-5) ?(elementwise_affine = true) () =
  {
    init =
      (fun ~rngs:_ ~device ~dtype ->
        if elementwise_affine then
          let gamma = Rune.ones device dtype [| dim |] in
          let beta = Rune.zeros device dtype [| dim |] in
          Ptree.record_of [ ("gamma", Tensor gamma); ("beta", Tensor beta) ]
        else List []);
    apply =
      (fun params ~training:_ ?rngs:_ x ->
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
              Ops.layer_norm ~gamma ~beta ~dim ~eps ~elementwise_affine:true x
          | _ -> failwith "layer_norm: invalid params"
        else Ops.layer_norm ~dim ~eps ~elementwise_affine:false x);
  }

let embedding ~vocab_size ~embed_dim ?(scale = true) ?embedding_init () =
  {
    init =
      (fun ~rngs ~device ~dtype ->
        let embedding_init_f =
          match embedding_init with
          | Some init -> init.Initializers.f
          | None -> (Initializers.normal_range ~mean:0.0 ~stddev:0.02 ()).f
        in
        let key = (Rune.Rng.split rngs).(0) in
        let embedding =
          embedding_init_f (Rune.Rng.to_int key)
            [| vocab_size; embed_dim |]
            device dtype
        in
        Ptree.Tensor embedding);
    apply =
      (fun params ~training:_ ?rngs:_ x ->
        match params with
        | Tensor embedding -> 
            (* Cast input to int32 for embedding lookup *)
            let indices = Rune.cast Rune.int32 x in
            Ops.embedding ~embedding ~embed_dim ~scale indices
        | _ -> failwith "embedding: invalid params");
  }

let multi_head_attention ~embed_dim ~num_heads ?(num_kv_heads = num_heads)
    ?head_dim ?(dropout = 0.0) ?(use_qk_norm = false) ?attn_logits_soft_cap
    ?query_pre_attn_scalar () =
  let head_dim = Option.value head_dim ~default:(embed_dim / num_heads) in
  assert (head_dim * num_heads = embed_dim);
  {
    init =
      (fun ~rngs ~device ~dtype ->
        let num_keys = if use_qk_norm then 6 else 4 in
        let keys = Rune.Rng.split ~n:num_keys rngs in
        let init_fn = (Initializers.glorot_uniform ()).f in
        let q_proj =
          init_fn
            (Rune.Rng.to_int keys.(0))
            [| embed_dim; num_heads * head_dim |]
            device dtype
        in
        let k_proj =
          init_fn
            (Rune.Rng.to_int keys.(1))
            [| embed_dim; num_kv_heads * head_dim |]
            device dtype
        in
        let v_proj =
          init_fn
            (Rune.Rng.to_int keys.(2))
            [| embed_dim; num_kv_heads * head_dim |]
            device dtype
        in
        let out_proj =
          init_fn
            (Rune.Rng.to_int keys.(3))
            [| num_heads * head_dim; embed_dim |]
            device dtype
        in
        let params_list =
          [
            ("q_proj", Ptree.Tensor q_proj);
            ("k_proj", Ptree.Tensor k_proj);
            ("v_proj", Ptree.Tensor v_proj);
            ("out_proj", Ptree.Tensor out_proj);
          ]
        in
        (* Add QK normalization parameters if enabled *)
        let params_list =
          if use_qk_norm then
            let q_norm_scale = Rune.ones device dtype [| head_dim |] in
            let k_norm_scale = Rune.ones device dtype [| head_dim |] in
            params_list
            @ [
                ("q_norm_scale", Ptree.Tensor q_norm_scale);
                ("k_norm_scale", Ptree.Tensor k_norm_scale);
              ]
          else params_list
        in
        Ptree.record_of params_list);
    apply =
      (fun params ~training ?rngs x ->
        (* TODO: Support attention masks properly The current module interface
           only accepts a single tensor x. We need to either:

           * Accept a record/tuple type that includes both input and mask

           * Use a context/state mechanism to pass masks through layers

           * Create a specialized attention module type with richer interface

           For now, attention_mask is always None. *)
        let query, key, value, attention_mask =
          match x with
          | x when Rune.ndim x = 3 ->
              (* Self-attention: query = key = value = x *)
              (x, None, None, None)
          | _ ->
              (* For now, assume self-attention *)
              (x, None, None, None)
        in
        match params with
        | Record fields ->
            let get_weight name =
              match Ptree.Record.find_opt name fields with
              | Some (Tensor t) -> t
              | _ -> failwith ("multi_head_attention: missing " ^ name)
            in
            let q_proj = get_weight "q_proj" in
            let k_proj = get_weight "k_proj" in
            let v_proj = get_weight "v_proj" in
            let out_proj = get_weight "out_proj" in

            (* Apply query pre-attention scalar if specified *)
            let scale =
              match query_pre_attn_scalar with
              | Some s -> s
              | None -> 1.0 /. sqrt (float_of_int head_dim)
            in

            (* Apply dropout only during training *)
            let effective_dropout = if training then dropout else 0.0 in

            (* Pass RNGs only when dropout > 0 and training *)
            let rngs_for_dropout =
              if training && dropout > 0.0 then rngs else None
            in

            let output, _attn_weights_opt =
              Ops.multi_head_attention ~q_proj_w:q_proj ~k_proj_w:k_proj
                ~v_proj_w:v_proj ~out_proj_w:out_proj ?q_bias:None ?k_bias:None
                ?v_bias:None ?out_bias:None ?k_bias_kv:None ?v_bias_kv:None
                ~query ?key ?value ?attention_mask ?is_causal:None
                ?rngs:rngs_for_dropout ~embed_dim ~num_heads ~num_kv_heads
                ~head_dim ~dropout:effective_dropout ~bias:false
                ~add_bias_kv:false ~scale ()
            in

            (* Apply attention logits soft cap if specified *)
            let output =
              match attn_logits_soft_cap with
              | Some cap ->
                  (* Soft capping: tanh(logits / cap) * cap *)
                  let scaled =
                    Rune.div output
                      (Rune.scalar (Rune.device output) (Rune.dtype output) cap)
                  in
                  let capped = Rune.tanh scaled in
                  Rune.mul capped
                    (Rune.scalar (Rune.device output) (Rune.dtype output) cap)
              | None -> output
            in
            output
        | _ -> failwith "multi_head_attention: invalid params");
  }

let mlp ~in_features ~hidden_features ~out_features ?(activation = `gelu)
    ?(dropout = 0.0) () =
  let act =
    match activation with
    | `relu -> relu ()
    | `gelu -> gelu ()
    | `swish -> swish ()
  in
  sequential
    [
      linear ~in_features ~out_features:hidden_features ();
      act;
      dropout_layer ~rate:dropout ();
      linear ~in_features:hidden_features ~out_features ();
      dropout_layer ~rate:dropout ();
    ]

let transformer_encoder_layer ~hidden_size ~num_attention_heads
    ~intermediate_size ?(hidden_dropout_prob = 0.1)
    ?(attention_probs_dropout_prob = 0.1) ?(layer_norm_eps = 1e-12)
    ?(hidden_act = `gelu) ?(use_bias = true) () =
  {
    init =
      (fun ~rngs ~device ~dtype ->
        let keys = Rune.Rng.split ~n:10 rngs in
        let init_fn = (Initializers.glorot_uniform ()).f in

        (* Attention weights *)
        let q_weight =
          init_fn
            (Rune.Rng.to_int keys.(0))
            [| hidden_size; hidden_size |]
            device dtype
        in
        let k_weight =
          init_fn
            (Rune.Rng.to_int keys.(1))
            [| hidden_size; hidden_size |]
            device dtype
        in
        let v_weight =
          init_fn
            (Rune.Rng.to_int keys.(2))
            [| hidden_size; hidden_size |]
            device dtype
        in
        let attn_out_weight =
          init_fn
            (Rune.Rng.to_int keys.(3))
            [| hidden_size; hidden_size |]
            device dtype
        in

        (* FFN weights *)
        let inter_weight =
          init_fn
            (Rune.Rng.to_int keys.(4))
            [| hidden_size; intermediate_size |]
            device dtype
        in
        let out_weight =
          init_fn
            (Rune.Rng.to_int keys.(5))
            [| intermediate_size; hidden_size |]
            device dtype
        in

        (* Biases (if enabled) *)
        let bias_params =
          if use_bias then
            let zero_init = (Initializers.zeros ()).f in
            [
              ( "q_bias",
                Ptree.Tensor (zero_init 0 [| hidden_size |] device dtype) );
              ( "k_bias",
                Ptree.Tensor (zero_init 0 [| hidden_size |] device dtype) );
              ( "v_bias",
                Ptree.Tensor (zero_init 0 [| hidden_size |] device dtype) );
              ( "attn_out_bias",
                Ptree.Tensor (zero_init 0 [| hidden_size |] device dtype) );
              ( "inter_bias",
                Ptree.Tensor (zero_init 0 [| intermediate_size |] device dtype)
              );
              ( "out_bias",
                Ptree.Tensor (zero_init 0 [| hidden_size |] device dtype) );
            ]
          else []
        in

        (* Layer norm parameters *)
        let attn_gamma = Rune.ones device dtype [| hidden_size |] in
        let attn_beta = Rune.zeros device dtype [| hidden_size |] in
        let ffn_gamma = Rune.ones device dtype [| hidden_size |] in
        let ffn_beta = Rune.zeros device dtype [| hidden_size |] in

        Ptree.record_of
          ([
             ("q_weight", Ptree.Tensor q_weight);
             ("k_weight", Ptree.Tensor k_weight);
             ("v_weight", Ptree.Tensor v_weight);
             ("attn_out_weight", Ptree.Tensor attn_out_weight);
             ("inter_weight", Ptree.Tensor inter_weight);
             ("out_weight", Ptree.Tensor out_weight);
             ("attn_gamma", Ptree.Tensor attn_gamma);
             ("attn_beta", Ptree.Tensor attn_beta);
             ("ffn_gamma", Ptree.Tensor ffn_gamma);
             ("ffn_beta", Ptree.Tensor ffn_beta);
           ]
          @ bias_params));
    apply =
      (fun params ~training ?rngs hidden_states ->
        match params with
        | Ptree.Record fields ->
            let get_weight name =
              match Ptree.Record.find_opt name fields with
              | Some (Ptree.Tensor t) -> t
              | _ -> failwith ("transformer_encoder_layer: missing " ^ name)
            in
            let get_bias_opt name =
              match Ptree.Record.find_opt name fields with
              | Some (Ptree.Tensor t) -> Some t
              | _ -> None
            in

            Ops.transformer_encoder_layer ~q_weight:(get_weight "q_weight")
              ~k_weight:(get_weight "k_weight")
              ~v_weight:(get_weight "v_weight")
              ~attn_out_weight:(get_weight "attn_out_weight")
              ~inter_weight:(get_weight "inter_weight")
              ~out_weight:(get_weight "out_weight")
              ?q_bias:(get_bias_opt "q_bias") ?k_bias:(get_bias_opt "k_bias")
              ?v_bias:(get_bias_opt "v_bias")
              ?attn_out_bias:(get_bias_opt "attn_out_bias")
              ?inter_bias:(get_bias_opt "inter_bias")
              ?out_bias:(get_bias_opt "out_bias")
              ~attn_gamma:(get_weight "attn_gamma")
              ~attn_beta:(get_weight "attn_beta")
              ~ffn_gamma:(get_weight "ffn_gamma")
              ~ffn_beta:(get_weight "ffn_beta") ~hidden_states ~training ?rngs
              ~hidden_size ~num_attention_heads ~intermediate_size
              ~hidden_dropout_prob ~attention_probs_dropout_prob ~layer_norm_eps
              ~hidden_act ~use_bias ()
        | _ -> failwith "transformer_encoder_layer: invalid params");
  }

let transformer_encoder ~num_layers ~hidden_size ~num_attention_heads
    ~intermediate_size ?(hidden_dropout_prob = 0.1)
    ?(attention_probs_dropout_prob = 0.1) ?(layer_norm_eps = 1e-12)
    ?(hidden_act = `gelu) ?(use_bias = true) () =
  let layers =
    List.init num_layers (fun _ ->
        transformer_encoder_layer ~hidden_size ~num_attention_heads
          ~intermediate_size ~hidden_dropout_prob ~attention_probs_dropout_prob
          ~layer_norm_eps ~hidden_act ~use_bias ())
  in
  sequential layers
