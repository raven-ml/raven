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
  scale:(float, 'b) Nx.t ->
  bias:(float, 'b) Nx.t ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t
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
  ?gamma:(float, 'b) Nx.t ->
  ?beta:(float, 'b) Nx.t ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t
(** [layer_norm ?axes ?epsilon ?gamma ?beta x] subtracts the mean and divides by
    the standard deviation along [axes], optionally scaling by [gamma] and
    shifting by [beta].

    [axes] defaults to [[-1]]. [epsilon] defaults to [1e-5].

    Raises [Invalid_argument] if [axes] is out of bounds, or if [gamma]/[beta]
    shapes are incompatible. *)

val rms_norm :
  ?axes:int list ->
  ?epsilon:float ->
  ?gamma:(float, 'b) Nx.t ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t
(** [rms_norm ?axes ?epsilon ?gamma x] normalizes [x] by the root mean square
    along [axes], optionally scaling by [gamma].

    [axes] defaults to [[-1]]. [epsilon] defaults to [1e-5].

    Raises [Invalid_argument] if [axes] is empty or out of bounds, or if [gamma]
    shape is incompatible. *)

(** {1:embedding Embedding} *)

val embedding :
  ?scale:bool ->
  embedding:(float, 'b) Nx.t ->
  (int32, Nx.int32_elt) Nx.t ->
  (float, 'b) Nx.t
(** [embedding ?scale ~embedding indices] gathers rows of [embedding] at
    positions given by [indices].

    [embedding] must have shape [[vocab_size; embed_dim]]. The result has shape
    [[*indices_shape; embed_dim]].

    When [scale] is [true] (the default), the result is multiplied by
    [sqrt(embed_dim)].

    Raises [Invalid_argument] if [embedding] is not rank 2 or if [vocab_size] is
    not positive. *)

(** {1:dropout Dropout} *)

val dropout : rate:float -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [dropout ~rate x] randomly zeroes elements of [x] with probability [rate]
    and scales the remaining values by [1 / (1 - rate)].

    [rate] must satisfy [0.0 <= rate < 1.0]. When [rate] is [0.0], [x] is
    returned unchanged.

    Random keys are drawn from the implicit RNG scope.

    Raises [Invalid_argument] if [rate] is out of range or [x] is not floating
    point. *)

(** {1:attention Attention} *)

val dot_product_attention :
  ?attention_mask:(bool, Nx.bool_elt) Nx.t ->
  ?scale:float ->
  ?dropout_rate:float ->
  ?is_causal:bool ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t
(** [dot_product_attention ?attention_mask ?scale ?dropout_rate ?is_causal q k
     v] is scaled dot-product attention.

    [q], [k], [v] must have matching rank (>= 2) and the last dimension of [q]
    and [k] must agree.

    [scale] defaults to [1 / sqrt(depth)]. [is_causal] defaults to [false]; when
    [true], a lower-triangular mask is applied (requires
    [seq_len_q = seq_len_k]).

    [attention_mask], when provided, broadcasts to the attention score shape:
    [true] keeps scores, [false] sets them to negative infinity.

    When [dropout_rate] is set, dropout is applied to attention weights using
    keys from the implicit RNG scope.

    Raises [Invalid_argument] if ranks, shapes, or dtypes are incompatible. *)

(** {1:conv Convolution} *)

val conv1d :
  ?groups:int ->
  ?stride:int ->
  ?dilation:int ->
  ?padding:[ `Same | `Valid ] ->
  ?bias:(float, 'b) Nx.t ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t
(** [conv1d ?groups ?stride ?dilation ?padding ?bias x w] computes 1D
    convolution.

    [x]: [(N, C_in, L)]. [w]: [(C_out, C_in/groups, K)].

    [groups] defaults to [1]. [stride] and [dilation] default to [1]. [padding]
    defaults to [`Valid].

    Raises [Invalid_argument] if input/weight shapes are incompatible or channel
    counts do not match [groups]. *)

val conv2d :
  ?groups:int ->
  ?stride:int * int ->
  ?dilation:int * int ->
  ?padding:[ `Same | `Valid ] ->
  ?bias:(float, 'b) Nx.t ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t
(** [conv2d ?groups ?stride ?dilation ?padding ?bias x w] computes 2D
    convolution.

    [x]: [(N, C_in, H, W)]. [w]: [(C_out, C_in/groups, kH, kW)].

    [groups] defaults to [1]. [stride] and [dilation] default to [(1, 1)].
    [padding] defaults to [`Valid].

    Raises [Invalid_argument] if input/weight shapes are incompatible or channel
    counts do not match [groups]. *)

(** {1:pool Pooling} *)

val max_pool1d :
  kernel_size:int ->
  ?stride:int ->
  ?dilation:int ->
  ?padding:[ `Same | `Valid ] ->
  ?ceil_mode:bool ->
  ('a, 'b) Nx.t ->
  ('a, 'b) Nx.t
(** [max_pool1d ~kernel_size ?stride ?dilation ?padding ?ceil_mode x] applies 1D
    max pooling.

    [x]: [(N, C, L)]. [stride] defaults to [1]. [dilation] defaults to [1].
    [padding] defaults to [`Valid]. [ceil_mode] defaults to [false]. *)

val max_pool2d :
  kernel_size:int * int ->
  ?stride:int * int ->
  ?dilation:int * int ->
  ?padding:[ `Same | `Valid ] ->
  ?ceil_mode:bool ->
  ('a, 'b) Nx.t ->
  ('a, 'b) Nx.t
(** [max_pool2d ~kernel_size ?stride ?dilation ?padding ?ceil_mode x] applies 2D
    max pooling.

    [x]: [(N, C, H, W)]. [stride] defaults to [(1, 1)]. [dilation] defaults to
    [(1, 1)]. [padding] defaults to [`Valid]. [ceil_mode] defaults to [false].
*)

val avg_pool1d :
  kernel_size:int ->
  ?stride:int ->
  ?dilation:int ->
  ?padding:[ `Same | `Valid ] ->
  ?ceil_mode:bool ->
  ?count_include_pad:bool ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t
(** [avg_pool1d ~kernel_size ?stride ?dilation ?padding ?ceil_mode
     ?count_include_pad x] applies 1D average pooling.

    [x]: [(N, C, L)]. [stride] defaults to [1]. [dilation] defaults to [1].
    [padding] defaults to [`Valid]. [ceil_mode] defaults to [false].
    [count_include_pad] defaults to [true]. *)

val avg_pool2d :
  kernel_size:int * int ->
  ?stride:int * int ->
  ?dilation:int * int ->
  ?padding:[ `Same | `Valid ] ->
  ?ceil_mode:bool ->
  ?count_include_pad:bool ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t
(** [avg_pool2d ~kernel_size ?stride ?dilation ?padding ?ceil_mode
     ?count_include_pad x] applies 2D average pooling.

    [x]: [(N, C, H, W)]. [stride] defaults to [(1, 1)]. [dilation] defaults to
    [(1, 1)]. [padding] defaults to [`Valid]. [ceil_mode] defaults to [false].
    [count_include_pad] defaults to [true]. *)
