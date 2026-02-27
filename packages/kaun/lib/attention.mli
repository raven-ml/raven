(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Multi-head self-attention.

    Provides scaled dot-product attention with support for grouped query
    attention (GQA), causal masking, rotary position embeddings (RoPE), and
    dropout. *)

(** {1:rope Rotary Position Embeddings} *)

val rope : ?theta:float -> ?seq_dim:int -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [rope ?theta ?seq_dim x] applies rotary position embeddings to [x].

    [x] may have any rank [>= 2], with shape [[d0; ...; dn-1]] where:
    - [head_dim = dn-1] (last axis).
    - [seq_len] is on axis [seq_dim].

    [theta] defaults to [10000.0]. [seq_dim] defaults to [-2] (second-to-last
    axis). Negative [seq_dim] values are interpreted relative to rank.

    [head_dim] must be even.

    Raises [Invalid_argument] if [x] has rank < 2, if [seq_dim] is out of
    bounds, if [seq_dim] designates the last axis, or if [head_dim] is odd. *)

(** {1:mha Multi-Head Attention} *)

val attention_mask_key : string
(** [attention_mask_key] is ["attention_mask"]. The well-known {!Context} key
    that {!multi_head_attention} reads during the forward pass. *)

val multi_head_attention :
  embed_dim:int ->
  num_heads:int ->
  ?num_kv_heads:int ->
  ?dropout:float ->
  ?is_causal:bool ->
  ?rope:bool ->
  ?rope_theta:float ->
  unit ->
  (float, float) Layer.t
(** [multi_head_attention ~embed_dim ~num_heads ()] is a multi-head
    self-attention layer.

    Input shape: [[batch; seq_len; embed_dim]]. Output shape:
    [[batch; seq_len; embed_dim]].

    [num_kv_heads] defaults to [num_heads] (standard MHA). When
    [num_kv_heads < num_heads], grouped query attention (GQA) is used.
    [num_heads] must be divisible by [num_kv_heads].

    [dropout] defaults to [0.0]. When positive, dropout is applied during
    training using keys from the implicit RNG scope.

    [is_causal] defaults to [false]. When [true], a causal mask prevents
    attending to future positions.

    [rope] defaults to [false]. When [true], rotary position embeddings are
    applied to Q and K before the attention computation. [rope_theta] defaults
    to [10000.0].

    When [ctx] contains {!attention_mask_key} (a bool or int32 tensor of shape
    [[batch; seq_k]]), it is applied as a padding mask. [true] / nonzero keeps
    the position, [false] / [0] masks it.

    Parameters:
    - [q_proj] ([[embed_dim; num_heads * head_dim]])
    - [k_proj] ([[embed_dim; num_kv_heads * head_dim]])
    - [v_proj] ([[embed_dim; num_kv_heads * head_dim]])
    - [out_proj] ([[num_heads * head_dim; embed_dim]])

    Raises [Invalid_argument] if [embed_dim] is not divisible by [num_heads], or
    if [num_heads] is not divisible by [num_kv_heads]. *)
