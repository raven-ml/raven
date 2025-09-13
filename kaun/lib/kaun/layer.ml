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

let conv1d ~in_channels ~out_channels ?(kernel_size = 3) ?(stride = 1)
    ?(dilation = 1) ?(padding = `Same) () =
  {
    init =
      (fun (type l d)
        ~rngs
        ~device:(dev : d Rune.device)
        ~(dtype : (float, l) Rune.dtype)
      ->
        Rune.debug_with_context
          (Printf.sprintf "conv1d_%dx%d_%d_init" in_channels out_channels
             kernel_size) (fun () ->
            let rngs_split = Rune.Rng.split rngs in
            let rng1 = rngs_split.(0) in
            let fan_in = in_channels * kernel_size in
            let fan_out = out_channels * kernel_size in
            let limit = sqrt (6.0 /. float_of_int (fan_in + fan_out)) in
            let weight_shape = [| out_channels; in_channels; kernel_size |] in
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
              | _ -> failwith "conv1d: missing or invalid weight parameter"
            in
            let b =
              match Ptree.Record.find_opt "bias" fields with
              | Some (Tensor t) -> t
              | _ -> failwith "conv1d: missing or invalid bias parameter"
            in
            Rune.debug_with_context
              (Printf.sprintf "conv1d_%dx%d_%d" in_channels out_channels
                 kernel_size) (fun () ->
                let x =
                  match padding with
                  | `Same -> x
                  | `Valid -> x
                  | `Causal ->
                      let pad_left = (kernel_size - 1) * dilation in
                      let pad_cfg = [| (0, 0); (0, 0); (pad_left, 0) |] in
                      Rune.pad pad_cfg 0.0 x
                in
                let padding_mode =
                  match padding with
                  | `Same -> `Same
                  | `Valid -> `Valid
                  | `Causal -> `Valid
                in
                let conv =
                  Rune.convolve1d x w ~stride ~dilation ~padding_mode
                in
                let b_reshaped = Rune.reshape [| 1; out_channels; 1 |] b in
                Rune.add conv b_reshaped)
        | _ -> failwith "conv1d: invalid params structure");
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

(* Recurrent layers *)

let rnn ~input_size ~hidden_size ?(return_sequences = false)
    ?(learned_init = false) () =
  {
    init =
      (fun ~rngs ~device ~dtype ->
        let glorot = (Initializers.glorot_uniform ()).f in
        let keys = Rune.Rng.split ~n:2 rngs in
        let w_xh =
          glorot
            (Rune.Rng.to_int keys.(0))
            [| input_size; hidden_size |]
            device dtype
        in
        let w_hh =
          glorot
            (Rune.Rng.to_int keys.(1))
            [| hidden_size; hidden_size |]
            device dtype
        in
        let b = Rune.zeros device dtype [| hidden_size |] in
        let base =
          [
            ("w_xh", Ptree.Tensor w_xh);
            ("w_hh", Ptree.Tensor w_hh);
            ("b", Ptree.Tensor b);
          ]
        in
        let base =
          if learned_init then
            let h0 = Rune.zeros device dtype [| hidden_size |] in
            base @ [ ("h0", Ptree.Tensor h0) ]
          else base
        in
        Ptree.record_of base);
    apply =
      (fun params ~training:_ ?rngs:_ x ->
        match params with
        | Record fields ->
            let get name =
              match Ptree.Record.find_opt name fields with
              | Some (Tensor t) -> t
              | _ -> failwith ("rnn: missing " ^ name)
            in
            let w_xh = get "w_xh" and w_hh = get "w_hh" and b = get "b" in
            let dev = Rune.device x and dt = Rune.dtype x in
            let batch, seq_len, _ =
              match Rune.shape x with
              | [| b; s; i |] -> (b, s, i)
              | _ -> failwith "rnn: expected [b; s; i]"
            in
            let h_init =
              match Ptree.Record.find_opt "h0" fields with
              | Some (Tensor h0) ->
                  Rune.reshape [| 1; hidden_size |] h0
                  |> Rune.expand [| batch; hidden_size |]
              | _ -> Rune.zeros dev dt [| batch; hidden_size |]
            in
            let h = ref h_init in
            let outputs =
              Array.make seq_len (Rune.zeros dev dt [| batch; hidden_size |])
            in
            for t = 0 to seq_len - 1 do
              let xt = Rune.slice [ Rune.A; Rune.I t; Rune.A ] x in
              let pre =
                Rune.add (Rune.matmul xt w_xh)
                  (Rune.add (Rune.matmul !h w_hh)
                     (Rune.reshape [| 1; hidden_size |] b))
              in
              h := Rune.tanh pre
            done;
            if return_sequences then (
              (* Fill outputs in second loop to keep simple shape reuse *)
              let h2 = ref h_init in
              for t = 0 to seq_len - 1 do
                let xt = Rune.slice [ Rune.A; Rune.I t; Rune.A ] x in
                let pre =
                  Rune.add (Rune.matmul xt w_xh)
                    (Rune.add (Rune.matmul !h2 w_hh)
                       (Rune.reshape [| 1; hidden_size |] b))
                in
                h2 := Rune.tanh pre;
                outputs.(t) <- !h2
              done;
              Rune.stack ~axis:1 (Array.to_list outputs))
            else !h
        | _ -> failwith "rnn: invalid params");
  }

let gru ~input_size ~hidden_size ?(return_sequences = false)
    ?(learned_init = false) () =
  {
    init =
      (fun ~rngs ~device ~dtype ->
        let glorot = (Initializers.glorot_uniform ()).f in
        let keys = Rune.Rng.split ~n:2 rngs in
        let w_ih =
          glorot
            (Rune.Rng.to_int keys.(0))
            [| input_size; hidden_size * 3 |]
            device dtype
        in
        let w_hh =
          glorot
            (Rune.Rng.to_int keys.(1))
            [| hidden_size; hidden_size * 3 |]
            device dtype
        in
        let b = Rune.zeros device dtype [| hidden_size * 3 |] in
        let base =
          [
            ("w_ih", Ptree.Tensor w_ih);
            ("w_hh", Ptree.Tensor w_hh);
            ("b", Ptree.Tensor b);
          ]
        in
        let base =
          if learned_init then
            let h0 = Rune.zeros device dtype [| hidden_size |] in
            base @ [ ("h0", Ptree.Tensor h0) ]
          else base
        in
        Ptree.record_of base);
    apply =
      (fun params ~training:_ ?rngs:_ x ->
        match params with
        | Record fields ->
            let get name =
              match Ptree.Record.find_opt name fields with
              | Some (Tensor t) -> t
              | _ -> failwith ("gru: missing " ^ name)
            in
            let w_ih = get "w_ih" and w_hh = get "w_hh" and b = get "b" in
            let dev = Rune.device x and dt = Rune.dtype x in
            let batch, seq_len, _ =
              match Rune.shape x with
              | [| b; s; i |] -> (b, s, i)
              | _ -> failwith "gru: expected [b; s; i]"
            in
            let h_init =
              match Ptree.Record.find_opt "h0" fields with
              | Some (Tensor h0) ->
                  Rune.reshape [| 1; hidden_size |] h0
                  |> Rune.expand [| batch; hidden_size |]
              | _ -> Rune.zeros dev dt [| batch; hidden_size |]
            in
            let h = ref h_init in
            let outputs =
              Array.make seq_len (Rune.zeros dev dt [| batch; hidden_size |])
            in
            for t = 0 to seq_len - 1 do
              let xt = Rune.slice [ Rune.A; Rune.I t; Rune.A ] x in
              let gates =
                Rune.add (Rune.matmul xt w_ih)
                  (Rune.add (Rune.matmul !h w_hh)
                     (Rune.reshape [| 1; hidden_size * 3 |] b))
              in
              let z =
                Rune.sigmoid
                  (Rune.slice [ Rune.A; Rune.R (0, hidden_size) ] gates)
              in
              let r =
                Rune.sigmoid
                  (Rune.slice
                     [ Rune.A; Rune.R (hidden_size, hidden_size * 2) ]
                     gates)
              in
              let n =
                Rune.tanh
                  (Rune.add
                     (Rune.slice
                        [ Rune.A; Rune.R (hidden_size * 2, hidden_size * 3) ]
                        gates)
                     (Rune.matmul (Rune.mul r !h)
                        (Rune.slice [ Rune.A; Rune.R (0, hidden_size) ] w_hh)))
              in
              h :=
                Rune.add
                  (Rune.mul (Rune.sub (Rune.scalar dev dt 1.0) z) n)
                  (Rune.mul z !h)
            done;
            if return_sequences then (
              let h2 = ref h_init in
              for t = 0 to seq_len - 1 do
                let xt = Rune.slice [ Rune.A; Rune.I t; Rune.A ] x in
                let gates =
                  Rune.add (Rune.matmul xt w_ih)
                    (Rune.add (Rune.matmul !h2 w_hh)
                       (Rune.reshape [| 1; hidden_size * 3 |] b))
                in
                let z =
                  Rune.sigmoid
                    (Rune.slice [ Rune.A; Rune.R (0, hidden_size) ] gates)
                in
                let r =
                  Rune.sigmoid
                    (Rune.slice
                       [ Rune.A; Rune.R (hidden_size, hidden_size * 2) ]
                       gates)
                in
                let n =
                  Rune.tanh
                    (Rune.add
                       (Rune.slice
                          [ Rune.A; Rune.R (hidden_size * 2, hidden_size * 3) ]
                          gates)
                       (Rune.matmul (Rune.mul r !h2)
                          (Rune.slice [ Rune.A; Rune.R (0, hidden_size) ] w_hh)))
                in
                h2 :=
                  Rune.add
                    (Rune.mul (Rune.sub (Rune.scalar dev dt 1.0) z) n)
                    (Rune.mul z !h2);
                outputs.(t) <- !h2
              done;
              Rune.stack ~axis:1 (Array.to_list outputs))
            else !h
        | _ -> failwith "gru: invalid params");
  }

let lstm ~input_size ~hidden_size ?(return_sequences = false)
    ?(learned_init = false) () =
  {
    init =
      (fun ~rngs ~device ~dtype ->
        let glorot = (Initializers.glorot_uniform ()).f in
        let keys = Rune.Rng.split ~n:2 rngs in
        let w_ih =
          glorot
            (Rune.Rng.to_int keys.(0))
            [| input_size; hidden_size * 4 |]
            device dtype
        in
        let w_hh =
          glorot
            (Rune.Rng.to_int keys.(1))
            [| hidden_size; hidden_size * 4 |]
            device dtype
        in
        let b = Rune.zeros device dtype [| hidden_size * 4 |] in
        let base =
          [
            ("w_ih", Ptree.Tensor w_ih);
            ("w_hh", Ptree.Tensor w_hh);
            ("b", Ptree.Tensor b);
          ]
        in
        let base =
          if learned_init then
            let h0 = Rune.zeros device dtype [| hidden_size |] in
            let c0 = Rune.zeros device dtype [| hidden_size |] in
            base @ [ ("h0", Ptree.Tensor h0); ("c0", Ptree.Tensor c0) ]
          else base
        in
        Ptree.record_of base);
    apply =
      (fun params ~training:_ ?rngs:_ x ->
        match params with
        | Record fields ->
            let get name =
              match Ptree.Record.find_opt name fields with
              | Some (Tensor t) -> t
              | _ -> failwith ("lstm: missing " ^ name)
            in
            let w_ih = get "w_ih" and w_hh = get "w_hh" and b = get "b" in
            let dev = Rune.device x and dt = Rune.dtype x in
            let batch, seq_len, _ =
              match Rune.shape x with
              | [| b; s; i |] -> (b, s, i)
              | _ -> failwith "lstm: expected [b; s; i]"
            in
            let h_init =
              match Ptree.Record.find_opt "h0" fields with
              | Some (Tensor h0) ->
                  Rune.reshape [| 1; hidden_size |] h0
                  |> Rune.expand [| batch; hidden_size |]
              | _ -> Rune.zeros dev dt [| batch; hidden_size |]
            in
            let c_init =
              match Ptree.Record.find_opt "c0" fields with
              | Some (Tensor c0) ->
                  Rune.reshape [| 1; hidden_size |] c0
                  |> Rune.expand [| batch; hidden_size |]
              | _ -> Rune.zeros dev dt [| batch; hidden_size |]
            in
            let h = ref h_init in
            let c = ref c_init in
            let outputs =
              Array.make seq_len (Rune.zeros dev dt [| batch; hidden_size |])
            in
            for t = 0 to seq_len - 1 do
              let xt = Rune.slice [ Rune.A; Rune.I t; Rune.A ] x in
              let gates =
                Rune.add (Rune.matmul xt w_ih)
                  (Rune.add (Rune.matmul !h w_hh)
                     (Rune.reshape [| 1; hidden_size * 4 |] b))
              in
              let i =
                Rune.sigmoid
                  (Rune.slice [ Rune.A; Rune.R (0, hidden_size) ] gates)
              in
              let f =
                Rune.sigmoid
                  (Rune.slice
                     [ Rune.A; Rune.R (hidden_size, hidden_size * 2) ]
                     gates)
              in
              let g =
                Rune.tanh
                  (Rune.slice
                     [ Rune.A; Rune.R (hidden_size * 2, hidden_size * 3) ]
                     gates)
              in
              let o =
                Rune.sigmoid
                  (Rune.slice
                     [ Rune.A; Rune.R (hidden_size * 3, hidden_size * 4) ]
                     gates)
              in
              c := Rune.add (Rune.mul f !c) (Rune.mul i g);
              h := Rune.mul o (Rune.tanh !c)
            done;
            if return_sequences then (
              let h2 = ref h_init in
              let c2 = ref c_init in
              for t = 0 to seq_len - 1 do
                let xt = Rune.slice [ Rune.A; Rune.I t; Rune.A ] x in
                let gates =
                  Rune.add (Rune.matmul xt w_ih)
                    (Rune.add (Rune.matmul !h2 w_hh)
                       (Rune.reshape [| 1; hidden_size * 4 |] b))
                in
                let i =
                  Rune.sigmoid
                    (Rune.slice [ Rune.A; Rune.R (0, hidden_size) ] gates)
                in
                let f =
                  Rune.sigmoid
                    (Rune.slice
                       [ Rune.A; Rune.R (hidden_size, hidden_size * 2) ]
                       gates)
                in
                let g =
                  Rune.tanh
                    (Rune.slice
                       [ Rune.A; Rune.R (hidden_size * 2, hidden_size * 3) ]
                       gates)
                in
                let o =
                  Rune.sigmoid
                    (Rune.slice
                       [ Rune.A; Rune.R (hidden_size * 3, hidden_size * 4) ]
                       gates)
                in
                c2 := Rune.add (Rune.mul f !c2) (Rune.mul i g);
                h2 := Rune.mul o (Rune.tanh !c2);
                outputs.(t) <- !h2
              done;
              Rune.stack ~axis:1 (Array.to_list outputs))
            else !h
        | _ -> failwith "lstm: invalid params");
  }

let positional_embedding_learned ~max_len ~embed_dim () =
  {
    init =
      (fun ~rngs ~device ~dtype ->
        let initf = (Initializers.normal_range ~mean:0.0 ~stddev:0.02 ()).f in
        let key = (Rune.Rng.split rngs).(0) in
        let table =
          initf (Rune.Rng.to_int key) [| max_len; embed_dim |] device dtype
        in
        Ptree.Tensor table);
    apply =
      (fun params ~training:_ ?rngs:_ x ->
        match params with
        | Tensor table ->
            let b, s, _ =
              match Rune.shape x with
              | [| b; s; e |] -> (b, s, e)
              | _ -> failwith "positional_embedding: expected [b; s; e]"
            in
            let dev = Rune.device x in
            let pos = Rune.arange dev Rune.int32 0 s 1 in
            let pos = Rune.reshape [| 1; s |] pos |> Rune.expand [| b; s |] in
            let pos_e =
              Ops.embedding ~embedding:table ~embed_dim ~scale:false pos
            in
            Rune.add x pos_e
        | _ -> failwith "positional_embedding: invalid params");
  }

let positional_encoding_sinusoidal_table ~max_len ~embed_dim ~device ~dtype =
  let dt = dtype in
  let d2 = embed_dim / 2 in
  let position =
    Rune.arange device Rune.int32 0 max_len 1
    |> Rune.cast dt
    |> Rune.reshape [| max_len; 1 |]
  in
  let j =
    Rune.arange device Rune.int32 0 d2 1
    |> Rune.cast dt
    |> Rune.reshape [| 1; d2 |]
  in
  let exponent =
    Rune.div
      (Rune.mul (Rune.scalar device dt 2.0) j)
      (Rune.scalar device dt (float_of_int embed_dim))
  in
  let angle_rate = Rune.pow (Rune.scalar device dt 10000.0) exponent in
  let angle = Rune.div position angle_rate in
  let sin_term = Rune.sin angle in
  let cos_term = Rune.cos angle in
  let sin_e = Rune.expand_dims [| 2 |] sin_term in
  let cos_e = Rune.expand_dims [| 2 |] cos_term in
  let stacked = Rune.stack ~axis:2 [ sin_e; cos_e ] in
  (* [L; d2; 2] *)
  Rune.reshape [| max_len; d2 * 2 |] stacked

let transformer_decoder_block ~embed_dim ~num_heads ~mlp_hidden ?(dropout = 0.0)
    () =
  let attn = multi_head_attention ~embed_dim ~num_heads () in
  let ln1 = layer_norm ~dim:embed_dim () in
  let ln2 = layer_norm ~dim:embed_dim () in
  let ff =
    sequential
      [
        linear ~in_features:embed_dim ~out_features:mlp_hidden ();
        gelu ();
        linear ~in_features:mlp_hidden ~out_features:embed_dim ();
      ]
  in
  {
    init =
      (fun ~rngs ~device ~dtype ->
        let ks = Rune.Rng.split ~n:4 rngs in
        Ptree.record_of
          [
            ("attn", attn.init ~rngs:ks.(0) ~device ~dtype);
            ("ln1", ln1.init ~rngs:ks.(1) ~device ~dtype);
            ("ln2", ln2.init ~rngs:ks.(2) ~device ~dtype);
            ("ff", ff.init ~rngs:ks.(3) ~device ~dtype);
          ]);
    apply =
      (fun params ~training ?rngs x ->
        match params with
        | Record fields ->
            let get name =
              match Ptree.Record.find_opt name fields with
              | Some p -> p
              | None -> failwith ("decoder_block: missing " ^ name)
            in
            let p_attn = get "attn"
            and p_ln1 = get "ln1"
            and p_ln2 = get "ln2"
            and p_ff = get "ff" in
            let x_norm = ln1.apply p_ln1 ~training ?rngs x in
            (* Extract weights from attn params to call Ops with is_causal *)
            let attn_out =
              match p_attn with
              | Record f ->
                  let getw n =
                    match Ptree.Record.find_opt n f with
                    | Some (Tensor t) -> t
                    | _ -> failwith ("attn param " ^ n)
                  in
                  let q = getw "q_proj"
                  and k = getw "k_proj"
                  and v = getw "v_proj"
                  and o = getw "out_proj" in
                  let head_dim = embed_dim / num_heads in
                  let effective_dropout = if training then dropout else 0.0 in
                  let out, _ =
                    Ops.multi_head_attention ~q_proj_w:q ~k_proj_w:k ~v_proj_w:v
                      ~out_proj_w:o ~query:x_norm ~is_causal:true ~embed_dim
                      ~num_heads ~num_kv_heads:num_heads ~head_dim
                      ~dropout:effective_dropout ?rngs ~bias:false
                      ~add_bias_kv:false ~scale:1.0 ()
                  in
                  out
              | _ -> failwith "attn params"
            in
            let x = Rune.add x attn_out in
            let x2 = ln2.apply p_ln2 ~training ?rngs x in
            let ff_out = ff.apply p_ff ~training ?rngs x2 in
            Rune.add x ff_out
        | _ -> failwith "decoder_block: invalid params");
  }

let transformer_decoder ~num_layers ~embed_dim ~num_heads ~mlp_hidden
    ?(dropout = 0.0) () =
  let layers =
    List.init num_layers (fun _ ->
        transformer_decoder_block ~embed_dim ~num_heads ~mlp_hidden ~dropout ())
  in
  sequential layers
