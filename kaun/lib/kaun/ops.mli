val scaled_dot_product_attention :
  ?attention_mask:(int, Rune.uint8_elt) Rune.t ->
  ?dropout:float ->
  ?is_causal:bool ->
  ?scale:float ->
  ?rngs:Rune.Rng.key ->
  (float, 'a) Rune.t ->
  (float, 'a) Rune.t ->
  (float, 'a) Rune.t ->
  (float, 'a) Rune.t * (float, 'a) Rune.t

val multi_head_attention :
  q_proj_w:(float, 'a) Rune.t ->
  k_proj_w:(float, 'a) Rune.t ->
  v_proj_w:(float, 'a) Rune.t ->
  out_proj_w:(float, 'a) Rune.t ->
  ?q_bias:(float, 'a) Rune.t ->
  ?k_bias:(float, 'a) Rune.t ->
  ?v_bias:(float, 'a) Rune.t ->
  ?out_bias:(float, 'a) Rune.t ->
  ?k_bias_kv:(float, 'a) Rune.t ->
  ?v_bias_kv:(float, 'a) Rune.t ->
  query:(float, 'a) Rune.t ->
  ?key:(float, 'a) Rune.t ->
  ?value:(float, 'a) Rune.t ->
  ?attention_mask:(int, Rune.uint8_elt) Rune.t ->
  ?is_causal:bool ->
  ?rngs:Rune.Rng.key ->
  embed_dim:int ->
  num_heads:int ->
  num_kv_heads:int ->
  head_dim:int ->
  dropout:float ->
  bias:bool ->
  add_bias_kv:bool ->
  scale:float ->
  unit ->
  (float, 'a) Rune.t * (float, 'a) Rune.t

val dropout :
  rate:float -> ?rngs:Rune.Rng.key -> (float, 'a) Rune.t -> (float, 'a) Rune.t

val batch_norm :
  scale:(float, 'a) Rune.t ->
  bias:(float, 'a) Rune.t ->
  num_features:int ->
  (float, 'a) Rune.t ->
  (float, 'a) Rune.t

val rms_norm :
  scale:(float, 'a) Rune.t ->
  dim:int ->
  ?eps:float ->
  (float, 'a) Rune.t ->
  (float, 'a) Rune.t

val layer_norm :
  ?gamma:(float, 'a) Rune.t ->
  ?beta:(float, 'a) Rune.t ->
  dim:int ->
  ?eps:float ->
  elementwise_affine:bool ->
  (float, 'a) Rune.t ->
  (float, 'a) Rune.t

val embedding :
  embedding:(float, 'a) Rune.t ->
  embed_dim:int ->
  ?scale:bool ->
  (int32, Rune.int32_elt) Rune.t ->
  (float, 'a) Rune.t

val transformer_encoder_layer :
  q_weight:(float, 'a) Rune.t ->
  k_weight:(float, 'a) Rune.t ->
  v_weight:(float, 'a) Rune.t ->
  attn_out_weight:(float, 'a) Rune.t ->
  inter_weight:(float, 'a) Rune.t ->
  out_weight:(float, 'a) Rune.t ->
  ?q_bias:(float, 'a) Rune.t ->
  ?k_bias:(float, 'a) Rune.t ->
  ?v_bias:(float, 'a) Rune.t ->
  ?attn_out_bias:(float, 'a) Rune.t ->
  ?inter_bias:(float, 'a) Rune.t ->
  ?out_bias:(float, 'a) Rune.t ->
  attn_gamma:(float, 'a) Rune.t ->
  attn_beta:(float, 'a) Rune.t ->
  ffn_gamma:(float, 'a) Rune.t ->
  ffn_beta:(float, 'a) Rune.t ->
  hidden_states:(float, 'a) Rune.t ->
  training:bool ->
  ?rngs:Rune.Rng.key ->
  hidden_size:int ->
  num_attention_heads:int ->
  intermediate_size:int ->
  hidden_dropout_prob:float ->
  attention_probs_dropout_prob:float ->
  layer_norm_eps:float ->
  hidden_act:[< `gelu | `gelu_new | `relu | `swish ] ->
  use_bias:bool ->
  unit ->
  (float, 'a) Rune.t
