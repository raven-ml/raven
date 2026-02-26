(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Functional neural network operations.

    Stateless building blocks for neural networks: normalization, attention,
    embedding lookup, and regularization. All functions are differentiable
    through Rune's autodiff. *)

(** {1:norm Normalization} *)

val batch_norm :
  ?axes:int list ->
  ?epsilon:float ->
  scale:(float, 'b) Rune.t ->
  bias:(float, 'b) Rune.t ->
  (float, 'b) Rune.t ->
  (float, 'b) Rune.t
(** [batch_norm ?axes ?epsilon ~scale ~bias x] normalizes [x] along [axes], then
    applies learnable [scale] and [bias].

    [axes] defaults to [[0]] for 2D and [[0; 2; 3]] for 4D input. [epsilon]
    defaults to [1e-5].

    [scale] and [bias] must broadcast across the normalized axes.

    Raises [Invalid_argument] if [axes] is empty or out of bounds, or if
    [scale]/[bias] shapes are incompatible. *)

val layer_norm :
  ?axes:int list ->
  ?epsilon:float ->
  ?gamma:(float, 'b) Rune.t ->
  ?beta:(float, 'b) Rune.t ->
  (float, 'b) Rune.t ->
  (float, 'b) Rune.t
(** [layer_norm ?axes ?epsilon ?gamma ?beta x] subtracts the mean and divides by
    the standard deviation along [axes], optionally scaling by [gamma] and
    shifting by [beta].

    [axes] defaults to [[-1]]. [epsilon] defaults to [1e-5].

    Raises [Invalid_argument] if [axes] is out of bounds, or if [gamma]/[beta]
    shapes are incompatible. *)

val rms_norm :
  ?axes:int list ->
  ?epsilon:float ->
  ?gamma:(float, 'b) Rune.t ->
  (float, 'b) Rune.t ->
  (float, 'b) Rune.t
(** [rms_norm ?axes ?epsilon ?gamma x] normalizes [x] by the root mean square
    along [axes], optionally scaling by [gamma].

    [axes] defaults to [[-1]]. [epsilon] defaults to [1e-5].

    Raises [Invalid_argument] if [axes] is empty or out of bounds, or if [gamma]
    shape is incompatible. *)

(** {1:embedding Embedding} *)

val embedding :
  ?scale:bool ->
  embedding:(float, 'b) Rune.t ->
  (int32, Rune.int32_elt) Rune.t ->
  (float, 'b) Rune.t
(** [embedding ?scale ~embedding indices] gathers rows of [embedding] at
    positions given by [indices].

    [embedding] must have shape [[vocab_size; embed_dim]]. The result has shape
    [[*indices_shape; embed_dim]].

    When [scale] is [true] (the default), the result is multiplied by
    [sqrt(embed_dim)].

    Raises [Invalid_argument] if [embedding] is not rank 2 or if [vocab_size] is
    not positive. *)

(** {1:dropout Dropout} *)

val dropout :
  key:Rune.Rng.key -> rate:float -> (float, 'b) Rune.t -> (float, 'b) Rune.t
(** [dropout ~key ~rate x] randomly zeroes elements of [x] with probability
    [rate] and scales the remaining values by [1 / (1 - rate)].

    [rate] must satisfy [0.0 <= rate < 1.0]. When [rate] is [0.0], [x] is
    returned unchanged.

    Raises [Invalid_argument] if [rate] is out of range or [x] is not floating
    point. *)

(** {1:attention Attention} *)

val dot_product_attention :
  ?attention_mask:(bool, Rune.bool_elt) Rune.t ->
  ?scale:float ->
  ?dropout_rate:float ->
  ?dropout_key:Rune.Rng.key ->
  ?is_causal:bool ->
  (float, 'b) Rune.t ->
  (float, 'b) Rune.t ->
  (float, 'b) Rune.t ->
  (float, 'b) Rune.t
(** [dot_product_attention ?attention_mask ?scale ?dropout_rate ?dropout_key
     ?is_causal q k v] is scaled dot-product attention.

    [q], [k], [v] must have matching rank (>= 2) and the last dimension of [q]
    and [k] must agree.

    [scale] defaults to [1 / sqrt(depth)]. [is_causal] defaults to [false]; when
    [true], a lower-triangular mask is applied (requires
    [seq_len_q = seq_len_k]).

    [attention_mask], when provided, broadcasts to the attention score shape:
    [true] keeps scores, [false] sets them to negative infinity.

    Raises [Invalid_argument] if ranks, shapes, or dtypes are incompatible, or
    if [dropout_rate] is set without [dropout_key]. *)
