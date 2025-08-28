type t = {
  vocab_size : int;
  n_positions : int; (* max sequence length *)
  n_embd : int; (* embedding dimension *)
  n_layer : int; (* number of transformer blocks *)
  n_head : int; (* number of attention heads *)
  n_inner : int option; (* hidden dimension of FFN, None means 4 * n_embd *)
  activation_function : string; (* activation function type *)
  resid_pdrop : float; (* residual dropout probability *)
  embd_pdrop : float; (* embedding dropout probability *)
  attn_pdrop : float; (* attention dropout probability *)
  layer_norm_epsilon : float; (* epsilon for layer normalization *)
  scale_attn_weights : bool; (* scale attention weights by 1/sqrt(n_embd) *)
  use_cache : bool; (* whether to use key-value cache *)
  bos_token_id : int;
  eos_token_id : int;
}

let gpt2_small =
  {
    vocab_size = 50257;
    n_positions = 1024;
    n_embd = 768;
    n_layer = 12;
    n_head = 12;
    n_inner = None;
    activation_function = "gelu";
    resid_pdrop = 0.1;
    embd_pdrop = 0.1;
    attn_pdrop = 0.1;
    layer_norm_epsilon = 1e-5;
    scale_attn_weights = true;
    use_cache = true;
    bos_token_id = 50256;
    eos_token_id = 50256;
  }

let gpt2_medium =
  {
    vocab_size = 50257;
    n_positions = 1024;
    n_embd = 1024;
    n_layer = 24;
    n_head = 16;
    n_inner = None;
    activation_function = "gelu";
    resid_pdrop = 0.1;
    embd_pdrop = 0.1;
    attn_pdrop = 0.1;
    layer_norm_epsilon = 1e-5;
    scale_attn_weights = true;
    use_cache = true;
    bos_token_id = 50256;
    eos_token_id = 50256;
  }

let gpt2_large =
  {
    vocab_size = 50257;
    n_positions = 1024;
    n_embd = 1280;
    n_layer = 36;
    n_head = 20;
    n_inner = None;
    activation_function = "gelu";
    resid_pdrop = 0.1;
    embd_pdrop = 0.1;
    attn_pdrop = 0.1;
    layer_norm_epsilon = 1e-5;
    scale_attn_weights = true;
    use_cache = true;
    bos_token_id = 50256;
    eos_token_id = 50256;
  }

let gpt2_xl =
  {
    vocab_size = 50257;
    n_positions = 1024;
    n_embd = 1600;
    n_layer = 48;
    n_head = 25;
    n_inner = None;
    activation_function = "gelu";
    resid_pdrop = 0.1;
    embd_pdrop = 0.1;
    attn_pdrop = 0.1;
    layer_norm_epsilon = 1e-5;
    scale_attn_weights = true;
    use_cache = true;
    bos_token_id = 50256;
    eos_token_id = 50256;
  }
