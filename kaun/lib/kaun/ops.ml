open Rune

let scaled_dot_product_attention ?attention_mask ?(dropout = 0.0) ?is_causal
    ?scale ?rngs q k v =
  let device = device q in
  let dtype = dtype q in
  let shape_q = shape q in
  let shape_k = shape k in

  (* Determine dimensions *)
  let _num_heads = shape_q.(1) in
  let seq_len_q = shape_q.(2) in
  let seq_len_k = shape_k.(2) in
  let head_dim = shape_q.(3) in

  (* Compute scale factor if not provided *)
  let scale =
    match scale with
    | Some s -> s
    | None -> 1.0 /. Stdlib.sqrt (float_of_int head_dim)
  in

  (* Compute attention scores: Q @ K^T *)
  let k_transposed = transpose ~axes:[| 0; 1; 3; 2 |] k in
  let scores = matmul q k_transposed in

  (* Apply scaling *)
  let scores = mul_s scores scale in

  (* Apply causal mask if requested *)
  let scores =
    if Option.value is_causal ~default:false then (
      if seq_len_q <> seq_len_k then
        failwith
          "scaled_dot_product_attention: causal masking requires seq_len_q == \
           seq_len_k";
      (* Create lower triangular mask using tril *)
      let mask_shape = [| 1; 1; seq_len_q; seq_len_k |] in
      let ones_matrix = ones device dtype [| seq_len_q; seq_len_k |] in
      let causal_mask = tril ones_matrix in
      let causal_mask = reshape mask_shape causal_mask in
      let causal_mask = cast uint8 causal_mask in
      let neg_inf = scalar device dtype (-1e9) in
      where causal_mask scores neg_inf)
    else scores
  in

  (* Apply additional mask if provided *)
  let scores =
    match attention_mask with
    | Some m ->
        let neg_inf = scalar device dtype (-1e9) in
        where m scores neg_inf
    | None -> scores
  in

  (* Apply softmax *)
  let attn_weights = softmax ~axes:[| 3 |] scores in

  (* Apply dropout if specified *)
  let attn_weights =
    if dropout > 0.0 then
      let keep_prob = 1.0 -. dropout in
      let dropout_mask =
        match rngs with
        | Some rng ->
            let seed = Rune.Rng.to_int rng in
            let rand_fn = Rune.rand ~seed in
            rand_fn device dtype (shape attn_weights)
        | None -> rand device dtype (shape attn_weights)
      in
      let threshold = scalar device dtype dropout in
      let mask = greater dropout_mask threshold in
      let mask_float = cast dtype mask in
      mul attn_weights (mul_s mask_float (1.0 /. keep_prob))
    else attn_weights
  in

  (* Apply attention to values: attention @ V *)
  let output = matmul attn_weights v in

  (* Return output and optionally attention weights *)
  (output, attn_weights)

let multi_head_attention ~q_proj_w ~k_proj_w ~v_proj_w ~out_proj_w ?q_bias
    ?k_bias ?v_bias ?out_bias ?k_bias_kv ?v_bias_kv ~query ?key ?value
    ?attention_mask ?is_causal ?rngs ~embed_dim ~num_heads ~num_kv_heads
    ~head_dim ~dropout ~bias ~add_bias_kv ~scale () =
  let key = Option.value key ~default:query in
  let value = Option.value value ~default:query in
  let kv_dim = num_kv_heads * head_dim in

  (* Determine shape *)
  let shape_q = shape query in
  let batch_size = shape_q.(0) in
  let tgt_seq_len = shape_q.(1) in

  (* Project Q, K, V *)
  let q = matmul query q_proj_w in
  let q =
    if bias then Option.fold ~none:q ~some:(fun b -> add q b) q_bias else q
  in

  let k = matmul key k_proj_w in
  let k =
    if bias then Option.fold ~none:k ~some:(fun b -> add k b) k_bias else k
  in

  let v = matmul value v_proj_w in
  let v =
    if bias then Option.fold ~none:v ~some:(fun b -> add v b) v_bias else v
  in

  (* Add bias_kv if present (for cross-attention) *)
  let k =
    if add_bias_kv then
      Option.fold ~none:k
        ~some:(fun bias ->
          let bias = broadcast_to [| batch_size; 1; kv_dim |] bias in
          concatenate ~axis:1 [ k; bias ])
        k_bias_kv
    else k
  in

  let v =
    if add_bias_kv then
      Option.fold ~none:v
        ~some:(fun bias ->
          let bias = broadcast_to [| batch_size; 1; kv_dim |] bias in
          concatenate ~axis:1 [ v; bias ])
        v_bias_kv
    else v
  in

  (* Reshape for multi-head attention: [batch, seq, embed] -> [batch, heads,
     seq, head_dim] *)
  let reshape_for_attention x num_heads_x =
    let x_shape = shape x in
    let seq_len = x_shape.(1) in
    let head_dim_x = x_shape.(2) / num_heads_x in
    let x = reshape [| batch_size; seq_len; num_heads_x; head_dim_x |] x in
    transpose ~axes:[| 0; 2; 1; 3 |] x
  in

  let q = reshape_for_attention q num_heads in
  let k = reshape_for_attention k num_kv_heads in
  let v = reshape_for_attention v num_kv_heads in

  (* Handle grouped-query attention if num_kv_heads < num_heads *)
  let k, v =
    if num_kv_heads < num_heads then
      let repeat_factor = num_heads / num_kv_heads in
      let repeat_kv tensor =
        let shape_t = shape tensor in
        (* [batch, heads_kv, seq, head_dim] *)
        let expanded = expand_dims [| 2 |] tensor in
        (* [batch, heads_kv, 1, seq, head_dim] *)
        let repeated =
          broadcast_to
            [|
              shape_t.(0); shape_t.(1); repeat_factor; shape_t.(2); shape_t.(3);
            |]
            expanded
        in
        reshape [| shape_t.(0); num_heads; shape_t.(2); shape_t.(3) |] repeated
      in
      (repeat_kv k, repeat_kv v)
    else (k, v)
  in

  (* Compute scaled dot-product attention *)
  let attn_output, attn_weights_opt =
    scaled_dot_product_attention ?attention_mask ~dropout ?is_causal ~scale
      ?rngs q k v
  in

  (* Transpose and reshape back: [batch, heads, seq, head_dim] -> [batch, seq,
     embed] *)
  let attn_output = transpose ~axes:[| 0; 2; 1; 3 |] attn_output in
  let attn_output = contiguous attn_output in
  let attn_output =
    reshape [| batch_size; tgt_seq_len; embed_dim |] attn_output
  in

  (* Apply output projection *)
  let output = matmul attn_output out_proj_w in
  let output =
    if bias then Option.fold ~none:output ~some:(fun b -> add output b) out_bias
    else output
  in

  (output, attn_weights_opt)

let dropout ~rate ?rngs x =
  if rate > 0.0 then
    match rngs with
    | Some rng ->
        let key = (Rng.split rng).(0) in
        let dev = device x in
        let dtype_x = dtype x in
        let shape_x = shape x in
        let mask = Rng.uniform key dev dtype_x shape_x in
        let keep_prob = 1.0 -. rate in
        let threshold = scalar dev dtype_x keep_prob in
        let binary_mask = less mask threshold in
        let binary_mask_float = cast dtype_x binary_mask in
        let scale = scalar dev dtype_x (1.0 /. keep_prob) in
        mul x (mul binary_mask_float scale)
    | None -> failwith "dropout requires RNG if rate > 0.0"
  else x

let batch_norm ~scale ~bias ~num_features x =
  let axes =
    match Array.length (shape x) with
    | 2 -> [| 0 |] (* (batch, features) *)
    | 4 -> [| 0; 2; 3 |] (* (batch, channels, height, width) *)
    | _ -> [| 0 |]
    (* Default to first axis *)
  in
  let mean = mean x ~axes ~keepdims:true in
  let variance = var x ~axes ~keepdims:true in
  let eps = 1e-5 in
  let dtype_x = dtype x in
  let dev = device x in
  let epsilon = scalar dev dtype_x eps in
  let x_normalized = div (sub x mean) (sqrt (add variance epsilon)) in
  let scale_shape =
    match Array.length (shape x) with
    | 2 -> [| 1; num_features |]
    | 4 -> [| 1; num_features; 1; 1 |]
    | _ -> [| 1; num_features |]
  in
  let scale_reshaped = reshape scale_shape scale in
  let bias_reshaped = reshape scale_shape bias in
  add (mul x_normalized scale_reshaped) bias_reshaped

let rms_norm ~scale ~dim ?(eps = 1e-6) x =
  let var = mean (square x) ~axes:[| -1 |] ~keepdims:true in
  let normed = mul x (rsqrt (add var (scalar (device x) (dtype x) eps))) in
  let scale_expanded =
    let x_dims = Array.length (shape x) in
    let scale_shape = Array.make x_dims 1 in
    scale_shape.(x_dims - 1) <- dim;
    reshape scale_shape scale
  in
  mul normed (add (scalar (device x) (dtype x) 1.0) scale_expanded)

let layer_norm ?gamma ?beta ~dim ?(eps = 1e-5) ~elementwise_affine x =
  let mean = mean x ~axes:[| -1 |] ~keepdims:true in
  let var = var x ~axes:[| -1 |] ~keepdims:true in
  let eps_scalar = scalar (device x) (dtype x) eps in
  let normalized = div (sub x mean) (sqrt (add var eps_scalar)) in
  if elementwise_affine then
    match (gamma, beta) with
    | Some g, Some b ->
        let x_dims = Array.length (shape x) in
        let reshape_dims = Array.make x_dims 1 in
        reshape_dims.(x_dims - 1) <- dim;
        let gamma_reshaped = reshape reshape_dims g in
        let beta_reshaped = reshape reshape_dims b in
        add (mul normalized gamma_reshaped) beta_reshaped
    | _ ->
        failwith
          "layer_norm: gamma and beta required if elementwise_affine=true"
  else normalized

let embedding ~embedding ~embed_dim ?(scale = true) x =
  (* Use Rune.take for differentiable embedding lookup *)
  (* x must be an int32 tensor of indices *)
  let input_shape = shape x in
  let flat_indices = reshape [| -1 |] x in

  (* Use take to gather embeddings at the specified indices *)
  let gathered = take ~axis:0 flat_indices embedding in

  (* Reshape back to original shape plus embedding dimension *)
  let output_shape = Array.append input_shape [| embed_dim |] in
  let embedded = reshape output_shape gathered in

  if scale then
    let device_emb = device embedding in
    let dtype_emb = dtype embedding in
    let scale_factor = Stdlib.sqrt (float_of_int embed_dim) in
    mul embedded (scalar device_emb dtype_emb scale_factor)
  else embedded

let transformer_encoder_layer ~q_weight ~k_weight ~v_weight ~attn_out_weight
    ~inter_weight ~out_weight ?q_bias ?k_bias ?v_bias ?attn_out_bias ?inter_bias
    ?out_bias ~attn_gamma ~attn_beta ~ffn_gamma ~ffn_beta ~hidden_states
    ~training ?rngs ~hidden_size ~num_attention_heads ~intermediate_size
    ~hidden_dropout_prob ~attention_probs_dropout_prob ~layer_norm_eps
    ~hidden_act ~use_bias () =
  (* Self-Attention *)
  let attn_output =
    (* Project to Q, K, V - weights should be [hidden_size, hidden_size] *)
    let q = matmul hidden_states q_weight in
    let q = if use_bias then Option.fold ~none:q ~some:(add q) q_bias else q in

    let k = matmul hidden_states k_weight in
    let k = if use_bias then Option.fold ~none:k ~some:(add k) k_bias else k in

    let v = matmul hidden_states v_weight in
    let v = if use_bias then Option.fold ~none:v ~some:(add v) v_bias else v in

    (* Multi-head attention reshape and compute *)
    let batch_size, seq_len, inferred_hidden_dim =
      match shape hidden_states with
      | [| b; s; h |] -> (b, s, h)
      | _ -> failwith "Invalid hidden states shape"
    in

    (* Validate hidden_size matches the actual dimension *)
    if inferred_hidden_dim <> hidden_size then
      failwith
        (Printf.sprintf
           "transformer_encoder_layer: hidden_size mismatch (expected %d, got \
            %d)"
           hidden_size inferred_hidden_dim);

    let num_heads = num_attention_heads in
    let head_dim = hidden_size / num_heads in

    let reshape_for_attention x =
      let x = contiguous x in
      let x = reshape [| batch_size; seq_len; num_heads; head_dim |] x in
      transpose x ~axes:[| 0; 2; 1; 3 |]
    in

    let q = reshape_for_attention q in
    let k = reshape_for_attention k in
    let v = reshape_for_attention v in

    (* Scaled dot-product attention *)
    let scale = 1.0 /. Float.sqrt (float_of_int head_dim) in
    let scores = matmul q (transpose k ~axes:[| 0; 1; 3; 2 |]) in
    let scores = mul_s scores scale in
    let attn_weights = softmax scores ~axes:[| 3 |] in

    (* Apply attention dropout if training *)
    let attn_weights =
      if training && attention_probs_dropout_prob > 0.0 then
        dropout ~rate:attention_probs_dropout_prob ?rngs attn_weights
      else attn_weights
    in

    (* Apply attention to values *)
    let context = matmul attn_weights v in
    let context = transpose context ~axes:[| 0; 2; 1; 3 |] in
    let context = contiguous context in
    let context = reshape [| batch_size; seq_len; hidden_size |] context in

    (* Output projection *)
    let output = matmul context attn_out_weight in
    let output =
      if use_bias then Option.fold ~none:output ~some:(add output) attn_out_bias
      else output
    in

    (* Apply hidden dropout if training *)
    if training && hidden_dropout_prob > 0.0 then
      dropout ~rate:hidden_dropout_prob ?rngs output
    else output
  in

  (* Add & Norm after attention *)
  let hidden_states = add hidden_states attn_output in
  let hidden_states =
    layer_norm ~gamma:attn_gamma ~beta:attn_beta ~dim:hidden_size
      ~eps:layer_norm_eps ~elementwise_affine:true hidden_states
  in

  (* Feed-forward network *)
  let ffn_output =
    (* Intermediate linear *)
    let intermediate = matmul hidden_states inter_weight in
    let intermediate =
      if use_bias then
        Option.fold ~none:intermediate ~some:(add intermediate) inter_bias
      else intermediate
    in

    (* Validate intermediate dimensions *)
    let inter_shape = shape intermediate in
    if inter_shape.(2) <> intermediate_size then
      failwith
        (Printf.sprintf
           "transformer_encoder_layer: intermediate_size mismatch (expected \
            %d, got %d)"
           intermediate_size inter_shape.(2));

    (* Apply activation function *)
    let activated =
      match hidden_act with
      | `gelu | `gelu_new -> Activations.gelu intermediate
      | `relu -> Activations.relu intermediate
      | `swish -> Activations.swish intermediate
    in

    (* Output linear *)
    let output = matmul activated out_weight in
    let output =
      if use_bias then Option.fold ~none:output ~some:(add output) out_bias
      else output
    in

    (* Apply hidden dropout if training *)
    if training && hidden_dropout_prob > 0.0 then
      dropout ~rate:hidden_dropout_prob ?rngs output
    else output
  in

  (* Add & Norm after FFN *)
  let hidden_states = add hidden_states ffn_output in
  layer_norm ~gamma:ffn_gamma ~beta:ffn_beta ~dim:hidden_size
    ~eps:layer_norm_eps ~elementwise_affine:true hidden_states
