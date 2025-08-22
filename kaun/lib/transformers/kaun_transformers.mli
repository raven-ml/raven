(** Transformer building blocks for Kaun

    This module provides optimized implementations of transformer components
    including attention mechanisms, positional encodings, and normalization
    layers. *)

(** {1 Attention Mechanisms} *)

val scaled_dot_product_attention :
  ?mask:(int, Rune.uint8_elt, 'dev) Rune.t ->
  ?dropout:float ->
  ?is_causal:bool ->
  ?scale:float ->
  (float, 'b, 'dev) Rune.t ->
  (float, 'b, 'dev) Rune.t ->
  (float, 'b, 'dev) Rune.t ->
  (float, 'b, 'dev) Rune.t
(** [scaled_dot_product_attention ?mask ?dropout ?is_causal ?scale q k v]
    computes scaled dot-product attention.

    Arguments:
    - [q]: Query tensor of shape [batch, seq_len, num_heads, head_dim] or
      [batch, num_heads, seq_len, head_dim]
    - [k]: Key tensor with same shape as [q]
    - [v]: Value tensor with same shape as [q]
    - [mask]: Optional attention mask
    - [dropout]: Dropout probability (default: 0.0)
    - [is_causal]: Use causal mask (default: false)
    - [scale]: Scaling factor (default: 1/sqrt(head_dim))

    Returns attention output with same shape as input.

    This implements the attention mechanism:
    {[
      Attention (Q, K, V) = softmax (QK ^ (T / sqrt d_k)) V
    ]} *)

(** {2 Rotary Position Embeddings} *)

module Rope : sig
  type t
  (** Rotary position embedding frequencies *)

  val make : ?base:float -> dim:int -> max_seq_len:int -> unit -> t
  (** [make ?base ~dim ~max_seq_len ()] creates RoPE frequencies.

      Arguments:
      - [base]: Base for the geometric progression (default: 10000.0)
      - [dim]: Dimension of the model (must be even)
      - [max_seq_len]: Maximum sequence length

      Returns precomputed cos and sin frequencies for RoPE. *)

  val apply :
    t ->
    ?seq_len:int ->
    (float, 'b, 'dev) Rune.t ->
    (float, 'b, 'dev) Rune.t ->
    (float, 'b, 'dev) Rune.t * (float, 'b, 'dev) Rune.t
  (** [apply rope ?seq_len q k] applies rotary embeddings to query and key
      tensors.

      Arguments:
      - [rope]: Precomputed RoPE frequencies
      - [seq_len]: Sequence length to use (default: shape of q/k)
      - [q]: Query tensor [batch, num_heads, seq_len, head_dim]
      - [k]: Key tensor [batch, num_heads, seq_len, head_dim]

      Returns (q', k') with rotary embeddings applied. *)
end
