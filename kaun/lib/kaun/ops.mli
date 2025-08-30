val scaled_dot_product_attention :
  ?attention_mask:(int, Rune.uint8_elt, 'a) Rune.t ->
  ?dropout:float ->
  ?is_causal:bool ->
  ?scale:float ->
  ?rngs:Rune.Rng.key ->
  (float, 'b, 'a) Rune.t ->
  (float, 'b, 'a) Rune.t ->
  (float, 'b, 'a) Rune.t ->
  (float, 'b, 'a) Rune.t * (float, 'b, 'a) Rune.t

val multi_head_attention :
  q_proj_w:(float, 'a, 'b) Rune.t ->
  k_proj_w:(float, 'a, 'b) Rune.t ->
  v_proj_w:(float, 'a, 'b) Rune.t ->
  out_proj_w:(float, 'a, 'b) Rune.t ->
  ?q_bias:(float, 'a, 'b) Rune.t ->
  ?k_bias:(float, 'a, 'b) Rune.t ->
  ?v_bias:(float, 'a, 'b) Rune.t ->
  ?out_bias:(float, 'a, 'b) Rune.t ->
  ?k_bias_kv:(float, 'a, 'b) Rune.t ->
  ?v_bias_kv:(float, 'a, 'b) Rune.t ->
  query:(float, 'a, 'b) Rune.t ->
  ?key:(float, 'a, 'b) Rune.t ->
  ?value:(float, 'a, 'b) Rune.t ->
  ?attention_mask:(int, Rune.uint8_elt, 'b) Rune.t ->
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
  (float, 'a, 'b) Rune.t * (float, 'a, 'b) Rune.t

val dropout :
  rate:float ->
  ?rngs:Rune.Rng.key ->
  (float, 'a, 'b) Rune.t ->
  (float, 'a, 'b) Rune.t

val batch_norm :
  scale:(float, 'a, 'b) Rune.t ->
  bias:(float, 'a, 'b) Rune.t ->
  num_features:int ->
  (float, 'a, 'b) Rune.t ->
  (float, 'a, 'b) Rune.t

val rms_norm :
  scale:(float, 'a, 'b) Rune.t ->
  dim:int ->
  ?eps:float ->
  (float, 'a, 'b) Rune.t ->
  (float, 'a, 'b) Rune.t

val layer_norm :
  ?gamma:(float, 'a, 'b) Rune.t ->
  ?beta:(float, 'a, 'b) Rune.t ->
  dim:int ->
  ?eps:float ->
  elementwise_affine:bool ->
  (float, 'a, 'b) Rune.t ->
  (float, 'a, 'b) Rune.t

val embedding :
  embedding:(float, 'a, 'b) Rune.t ->
  embed_dim:int ->
  ?scale:bool ->
  (int32, Rune.int32_elt, 'b) Rune.t ->
  (float, 'a, 'b) Rune.t

val transformer_encoder_layer :
  q_weight:(float, 'a, 'b) Rune.t ->
  k_weight:(float, 'a, 'b) Rune.t ->
  v_weight:(float, 'a, 'b) Rune.t ->
  attn_out_weight:(float, 'a, 'b) Rune.t ->
  inter_weight:(float, 'a, 'b) Rune.t ->
  out_weight:(float, 'a, 'b) Rune.t ->
  ?q_bias:(float, 'a, 'b) Rune.t ->
  ?k_bias:(float, 'a, 'b) Rune.t ->
  ?v_bias:(float, 'a, 'b) Rune.t ->
  ?attn_out_bias:(float, 'a, 'b) Rune.t ->
  ?inter_bias:(float, 'a, 'b) Rune.t ->
  ?out_bias:(float, 'a, 'b) Rune.t ->
  attn_gamma:(float, 'a, 'b) Rune.t ->
  attn_beta:(float, 'a, 'b) Rune.t ->
  ffn_gamma:(float, 'a, 'b) Rune.t ->
  ffn_beta:(float, 'a, 'b) Rune.t ->
  hidden_states:(float, 'a, 'b) Rune.t ->
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
  (float, 'a, 'b) Rune.t
