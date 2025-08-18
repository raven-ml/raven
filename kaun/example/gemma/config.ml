(* Gemma 3 model configurations *)

type attention_type = Global | Local_sliding

type query_pre_attention_norm =
  | By_one_over_sqrt_head_dim
  | By_embed_dim_div_num_heads
  | By_one_over_sqrt_embed_dim_div_num_heads

type t = {
  (* Model architecture *)
  num_layers : int;
  vocab_size : int;
  embed_dim : int;
  hidden_dim : int;
  num_heads : int;
  head_dim : int;
  num_kv_heads : int;
  (* Attention configuration *)
  attention_types : attention_type array;
  query_pre_attn_norm : query_pre_attention_norm;
  attn_logits_soft_cap : float option;
  sliding_window_size : int option;
  (* Normalization *)
  use_post_attn_norm : bool;
  use_post_ffw_norm : bool;
  rms_norm_eps : float;
  (* Output *)
  final_logit_softcap : float option;
  (* Training *)
  max_seq_len : int;
  dropout : float;
  (* RoPE *)
  rope_base : float;
  rope_scale : float;
}

(* Helper to create attention pattern for Gemma 3 *)
let make_gemma3_attention_pattern num_layers =
  (* Gemma 3 uses pattern: 5 local sliding, 1 global, repeat *)
  let pattern =
    [|
      Local_sliding;
      Local_sliding;
      Local_sliding;
      Local_sliding;
      Local_sliding;
      Global;
    |]
  in
  let pattern_len = Array.length pattern in
  Array.init num_layers (fun i -> pattern.(i mod pattern_len))

(* Gemma 3 model configurations *)
let gemma3_1b =
  {
    num_layers = 26;
    vocab_size = 256128;
    embed_dim = 2304;
    hidden_dim = 4864;
    num_heads = 18;
    head_dim = 128;
    num_kv_heads = 2;
    attention_types = make_gemma3_attention_pattern 26;
    query_pre_attn_norm = By_one_over_sqrt_head_dim;
    attn_logits_soft_cap = Some 50.0;
    sliding_window_size = Some 4096;
    use_post_attn_norm = true;
    use_post_ffw_norm = true;
    rms_norm_eps = 1e-6;
    final_logit_softcap = Some 30.0;
    max_seq_len = 8192;
    dropout = 0.0;
    rope_base = 10000.0;
    rope_scale = 1.0;
  }

let gemma3_4b =
  {
    num_layers = 34;
    vocab_size = 256128;
    embed_dim = 3072;
    hidden_dim = 8192;
    num_heads = 24;
    head_dim = 128;
    num_kv_heads = 4;
    attention_types = make_gemma3_attention_pattern 34;
    query_pre_attn_norm = By_one_over_sqrt_head_dim;
    attn_logits_soft_cap = Some 50.0;
    sliding_window_size = Some 4096;
    use_post_attn_norm = true;
    use_post_ffw_norm = true;
    rms_norm_eps = 1e-6;
    final_logit_softcap = Some 30.0;
    max_seq_len = 8192;
    dropout = 0.0;
    rope_base = 10000.0;
    rope_scale = 1.0;
  }

let gemma3_12b =
  {
    num_layers = 48;
    vocab_size = 256128;
    embed_dim = 4608;
    hidden_dim = 14336;
    num_heads = 36;
    head_dim = 128;
    num_kv_heads = 4;
    attention_types = make_gemma3_attention_pattern 48;
    query_pre_attn_norm = By_one_over_sqrt_head_dim;
    attn_logits_soft_cap = Some 50.0;
    sliding_window_size = Some 4096;
    use_post_attn_norm = true;
    use_post_ffw_norm = true;
    rms_norm_eps = 1e-6;
    final_logit_softcap = Some 30.0;
    max_seq_len = 8192;
    dropout = 0.0;
    rope_base = 10000.0;
    rope_scale = 1.0;
  }

let gemma3_27b =
  {
    num_layers = 62;
    vocab_size = 256128;
    embed_dim = 5120;
    hidden_dim = 16384;
    num_heads = 40;
    head_dim = 128;
    num_kv_heads = 8;
    attention_types = make_gemma3_attention_pattern 62;
    query_pre_attn_norm = By_one_over_sqrt_head_dim;
    attn_logits_soft_cap = Some 50.0;
    sliding_window_size = Some 4096;
    use_post_attn_norm = true;
    use_post_ffw_norm = true;
    rms_norm_eps = 1e-6;
    final_logit_softcap = Some 30.0;
    max_seq_len = 8192;
    dropout = 0.0;
    rope_base = 10000.0;
    rope_scale = 1.0;
  }
