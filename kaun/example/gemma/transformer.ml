open Kaun

(* Feed-forward network with gated activation *)
let feed_forward_network ~embed_dim ~hidden_dim () =
  Model
    {
      init =
        (fun ~rngs x ->
          let dev = Rune.device x in
          let dtype = Rune.dtype x in
          let rngs_split = Rune.Rng.split rngs in
          let rng1 = rngs_split.(0) in
          let rng2 = rngs_split.(1) in
          let rngs_split2 = Rune.Rng.split rng2 in
          let rng3 = rngs_split2.(0) in

          let init = Initializer.glorot_uniform () in

          (* Gate and up projections *)
          let gate_proj =
            Initializer.apply init
              (Rune.Rng.to_int (Rune.Rng.split rng1).(0))
              [| embed_dim; hidden_dim |]
              dev dtype
          in
          let up_proj =
            Initializer.apply init
              (Rune.Rng.to_int (Rune.Rng.split rng2).(0))
              [| embed_dim; hidden_dim |]
              dev dtype
          in
          let down_proj =
            Initializer.apply init
              (Rune.Rng.to_int (Rune.Rng.split rng3).(0))
              [| hidden_dim; embed_dim |]
              dev dtype
          in

          Record
            [
              ("gate_proj", Tensor gate_proj);
              ("up_proj", Tensor up_proj);
              ("down_proj", Tensor down_proj);
            ]);
      apply =
        (fun params ~training:_ ?rngs:_ x ->
          match params with
          | Record fields ->
              let get_tensor name =
                match List.assoc name fields with
                | Tensor t -> t
                | _ -> failwith (Printf.sprintf "Expected tensor for %s" name)
              in

              let gate_proj = get_tensor "gate_proj" in
              let up_proj = get_tensor "up_proj" in
              let down_proj = get_tensor "down_proj" in

              (* Gated activation: down_proj(gelu(gate_proj(x)) * up_proj(x)) *)
              let gate = Rune.matmul x gate_proj in
              let up = Rune.matmul x up_proj in

              (* GELU activation on gate *)
              let sqrt_2_over_pi = 0.7978845608 in
              let gate_cubed = Rune.mul gate (Rune.mul gate gate) in
              let inner =
                Rune.add gate
                  (Rune.mul
                     (Rune.scalar (Rune.device gate) (Rune.dtype gate) 0.044715)
                     gate_cubed)
              in
              let tanh_arg =
                Rune.mul
                  (Rune.scalar (Rune.device gate) (Rune.dtype gate)
                     sqrt_2_over_pi)
                  inner
              in
              let gate_activated =
                Rune.mul gate
                  (Rune.mul
                     (Rune.scalar (Rune.device gate) (Rune.dtype gate) 0.5)
                     (Rune.add
                        (Rune.scalar (Rune.device gate) (Rune.dtype gate) 1.0)
                        (Rune.tanh tanh_arg)))
              in

              (* Element-wise multiplication and down projection *)
              let hidden = Rune.mul gate_activated up in
              Rune.matmul hidden down_proj
          | _ -> failwith "feed_forward_network: invalid params");
    }

(* Transformer block *)
let transformer_block ~config ~layer_idx () =
  let attention_type = config.Config.attention_types.(layer_idx) in
  let query_pre_attn_scalar =
    match config.Config.query_pre_attn_norm with
    | Config.By_one_over_sqrt_head_dim ->
        1.0 /. sqrt (float_of_int config.Config.head_dim)
    | Config.By_embed_dim_div_num_heads ->
        float_of_int (config.Config.embed_dim / config.Config.num_heads)
    | Config.By_one_over_sqrt_embed_dim_div_num_heads ->
        1.0
        /. sqrt
             (float_of_int (config.Config.embed_dim / config.Config.num_heads))
  in

  let sliding_window =
    match attention_type with
    | Config.Local_sliding -> config.Config.sliding_window_size
    | Config.Global -> None
  in

  Model
    {
      init =
        (fun ~rngs x ->
          let rngs_split = Rune.Rng.split rngs in
          let rng1 = rngs_split.(0) in
          let rng2 = rngs_split.(1) in
          let rngs_split2 = Rune.Rng.split rng2 in
          let rng3 = rngs_split2.(0) in
          let rng4 = rngs_split2.(1) in
          let rngs_split3 = Rune.Rng.split rng4 in
          let rng5 = rngs_split3.(0) in
          let rng6 = rngs_split3.(1) in
          let rngs_split4 = Rune.Rng.split rng6 in
          let rng7 = rngs_split4.(0) in
          let rng8 = rngs_split4.(1) in

          (* Pre-attention norm *)
          let pre_attn_norm =
            Layer.rms_norm ~dim:config.Config.embed_dim
              ~eps:config.Config.rms_norm_eps ()
          in
          let pre_attn_norm_params = init pre_attn_norm ~rngs:rng1 x in

          (* Attention *)
          let attention =
            Attention.multi_head_attention_with_rope
              ~embed_dim:config.Config.embed_dim
              ~num_heads:config.Config.num_heads
              ~num_kv_heads:config.Config.num_kv_heads
              ~head_dim:config.Config.head_dim ~use_qk_norm:false
              ~attn_logits_soft_cap:config.Config.attn_logits_soft_cap
              ~query_pre_attn_scalar ~sliding_window_size:sliding_window ()
          in
          let attention_params = init attention ~rngs:rng3 x in

          (* Post-attention norm (if enabled) *)
          let post_attn_norm_params =
            if config.Config.use_post_attn_norm then
              let norm =
                Layer.rms_norm ~dim:config.Config.embed_dim
                  ~eps:config.Config.rms_norm_eps ()
              in
              init norm ~rngs:rng5 x
            else List []
          in

          (* Pre-FFN norm *)
          let pre_ffn_norm =
            Layer.rms_norm ~dim:config.Config.embed_dim
              ~eps:config.Config.rms_norm_eps ()
          in
          let pre_ffn_norm_params = init pre_ffn_norm ~rngs:rng7 x in

          (* Feed-forward network *)
          let ffn =
            feed_forward_network ~embed_dim:config.Config.embed_dim
              ~hidden_dim:config.Config.hidden_dim ()
          in
          let ffn_params = init ffn ~rngs:rng8 x in

          (* Post-FFN norm (if enabled) *)
          let post_ffn_norm_params =
            if config.Config.use_post_ffw_norm then
              let norm =
                Layer.rms_norm ~dim:config.Config.embed_dim
                  ~eps:config.Config.rms_norm_eps ()
              in
              init norm ~rngs:rng2 x
            else List []
          in

          Record
            [
              ("pre_attn_norm", pre_attn_norm_params);
              ("attention", attention_params);
              ("post_attn_norm", post_attn_norm_params);
              ("pre_ffn_norm", pre_ffn_norm_params);
              ("ffn", ffn_params);
              ("post_ffn_norm", post_ffn_norm_params);
            ]);
      apply =
        (fun params ~training ?rngs x ->
          match params with
          | Record fields ->
              (* Helper to apply a layer *)
              let apply_layer name model x =
                let params = List.assoc name fields in
                apply model params ~training ?rngs x
              in

              (* Pre-attention norm *)
              let pre_attn_norm =
                Layer.rms_norm ~dim:config.Config.embed_dim
                  ~eps:config.Config.rms_norm_eps ()
              in
              let normed = apply_layer "pre_attn_norm" pre_attn_norm x in

              (* Attention with residual *)
              let attention =
                Attention.multi_head_attention_with_rope
                  ~embed_dim:config.Config.embed_dim
                  ~num_heads:config.Config.num_heads
                  ~num_kv_heads:config.Config.num_kv_heads
                  ~head_dim:config.Config.head_dim ~use_qk_norm:false
                  ~attn_logits_soft_cap:config.Config.attn_logits_soft_cap
                  ~query_pre_attn_scalar ~sliding_window_size:sliding_window ()
              in
              let attn_out = apply_layer "attention" attention normed in
              let x = Rune.add x attn_out in

              (* Post-attention norm if enabled *)
              let x =
                if config.Config.use_post_attn_norm then
                  let post_attn_norm =
                    Layer.rms_norm ~dim:config.Config.embed_dim
                      ~eps:config.Config.rms_norm_eps ()
                  in
                  apply_layer "post_attn_norm" post_attn_norm x
                else x
              in

              (* Pre-FFN norm *)
              let pre_ffn_norm =
                Layer.rms_norm ~dim:config.Config.embed_dim
                  ~eps:config.Config.rms_norm_eps ()
              in
              let normed = apply_layer "pre_ffn_norm" pre_ffn_norm x in

              (* FFN with residual *)
              let ffn =
                feed_forward_network ~embed_dim:config.Config.embed_dim
                  ~hidden_dim:config.Config.hidden_dim ()
              in
              let ffn_out = apply_layer "ffn" ffn normed in
              let x = Rune.add x ffn_out in

              (* Post-FFN norm if enabled *)
              if config.Config.use_post_ffw_norm then
                let post_ffn_norm =
                  Layer.rms_norm ~dim:config.Config.embed_dim
                    ~eps:config.Config.rms_norm_eps ()
                in
                apply_layer "post_ffn_norm" post_ffn_norm x
              else x
          | _ -> failwith "transformer_block: invalid params");
    }

(* Complete Gemma model *)
let create_gemma_model config =
  (* RoPE embeddings will be created inside attention blocks *)
  Model
    {
      init =
        (fun ~rngs x ->
          let dev = Rune.device x in
          let dtype_float = Rune.dtype x in

          (* Split RNGs for each component *)
          let rngs_split = Rune.Rng.split rngs in
          let rng_embed = rngs_split.(0) in
          let rng_rest = rngs_split.(1) in
          let rngs_split2 = Rune.Rng.split rng_rest in
          let rng_blocks = rngs_split2.(0) in
          let rng_final = rngs_split2.(1) in
          let rngs_split3 = Rune.Rng.split rng_final in
          let rng_norm = rngs_split3.(0) in
          let rng_lm_head = rngs_split3.(1) in

          (* Token embeddings *)
          let embed_init = Initializer.normal ~mean:0.0 ~std:0.02 in
          let embeddings =
            Initializer.apply embed_init
              (Rune.Rng.to_int (Rune.Rng.split rng_embed).(0))
              [| config.Config.vocab_size; config.Config.embed_dim |]
              dev dtype_float
          in

          (* Transformer blocks *)
          let blocks_params =
            let rec init_blocks idx rngs acc =
              if idx >= config.Config.num_layers then List.rev acc
              else
                let rngs_split_block = Rune.Rng.split rngs in
                let rng_block = rngs_split_block.(0) in
                let rng_next = rngs_split_block.(1) in
                let block = transformer_block ~config ~layer_idx:idx () in
                (* Create dummy input for block initialization *)
                let dummy_input =
                  Rune.zeros dev dtype_float [| 1; 1; config.Config.embed_dim |]
                in
                let block_params = init block ~rngs:rng_block dummy_input in
                init_blocks (idx + 1) rng_next (block_params :: acc)
            in
            init_blocks 0 rng_blocks []
          in

          (* Final RMS norm *)
          let final_norm =
            Layer.rms_norm ~dim:config.Config.embed_dim
              ~eps:config.Config.rms_norm_eps ()
          in
          let dummy_input =
            Rune.zeros dev dtype_float [| 1; 1; config.Config.embed_dim |]
          in
          let final_norm_params = init final_norm ~rngs:rng_norm dummy_input in

          (* Language model head *)
          let lm_head_init =
            Initializer.normal ~mean:0.0
              ~std:(0.02 /. sqrt (float_of_int config.Config.num_layers))
          in
          let lm_head =
            Initializer.apply lm_head_init
              (Rune.Rng.to_int (Rune.Rng.split rng_lm_head).(0))
              [| config.Config.embed_dim; config.Config.vocab_size |]
              dev dtype_float
          in

          Record
            [
              ("embeddings", Tensor embeddings);
              ("blocks", List blocks_params);
              ("final_norm", final_norm_params);
              ("lm_head", Tensor lm_head);
            ]);
      apply =
        (fun params ~training ?rngs x ->
          match params with
          | Record fields -> (
              (* Get embeddings *)
              let embeddings =
                match List.assoc "embeddings" fields with
                | Tensor t -> t
                | _ -> failwith "Expected tensor for embeddings"
              in

              (* Token embedding lookup using gather *)
              let x = Ops.gather embeddings ~indices:x ~axis:0 in

              (* Scale embeddings *)
              let scale = sqrt (float_of_int config.Config.embed_dim) in
              let x =
                Rune.mul x (Rune.scalar (Rune.device x) (Rune.dtype x) scale)
              in

              (* Apply transformer blocks *)
              let blocks_params =
                match List.assoc "blocks" fields with
                | List l -> l
                | _ -> failwith "Expected list for blocks"
              in
              let x =
                List.fold_left2
                  (fun x block_params idx ->
                    let block = transformer_block ~config ~layer_idx:idx () in
                    apply block block_params ~training ?rngs x)
                  x blocks_params
                  (List.init config.Config.num_layers (fun i -> i))
              in

              (* Final norm *)
              let final_norm =
                Layer.rms_norm ~dim:config.Config.embed_dim
                  ~eps:config.Config.rms_norm_eps ()
              in
              let final_norm_params = List.assoc "final_norm" fields in
              let x = apply final_norm final_norm_params ~training ?rngs x in

              (* Language model head *)
              let lm_head =
                match List.assoc "lm_head" fields with
                | Tensor t -> t
                | _ -> failwith "Expected tensor for lm_head"
              in
              let logits = Rune.matmul x lm_head in

              (* Apply final logit soft capping if specified *)
              match config.Config.final_logit_softcap with
              | Some cap ->
                  let cap_scalar =
                    Rune.scalar (Rune.device logits) (Rune.dtype logits) cap
                  in
                  Rune.mul (Rune.tanh (Rune.div logits cap_scalar)) cap_scalar
              | None -> logits)
          | _ -> failwith "gemma_model: invalid params");
    }
