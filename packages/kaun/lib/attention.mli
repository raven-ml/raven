(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Multi-head self-attention layers (Vaswani et al., 2017).

    An attention layer is a plain record of four {!Linear} projections: query,
    key, value and output, each mapping [embed_dim] to [embed_dim] features.
    Construct one with {!init} or {!make} and transform sequences with {!apply},
    which splits the projections into heads, runs
    {!scaled_dot_product_attention} on each head and merges the results through
    the output projection. Like the other layers, it composes into models
    through record nesting; {!map}, {!map2}, {!iter} and {!names} supply the
    {!Nx.Ptree.S} and checkpoint plumbing.

    The head count is not a parameter: the projections are
    [embed_dim × embed_dim] whatever the head count, so [num_heads] is an
    argument of {!apply}, like [eps] of {!Layer_norm.apply}.

    {!scaled_dot_product_attention} is the pure core — no parameters, no head
    bookkeeping. Use it directly for cross-attention, externally projected
    queries and keys, or custom masking; use {!apply} for the standard
    self-attention block. *)

(** {1:types Types} *)

type 'b params = {
  q : 'b Linear.params;
  k : 'b Linear.params;
  v : 'b Linear.params;
  out : 'b Linear.params;
}
(** The type for attention parameters with float dtype layout ['b]: the query,
    key, value and output projections, each [embed_dim] to [embed_dim] features.
*)

type t = Nx.float32_elt params
(** The type for single-precision attention layers, the common case. *)

(** {1:constructors Constructors} *)

val make :
  ?w_init:'b Init.t ->
  ?bias_init:'b Init.t ->
  ?bias:bool ->
  embed_dim:int ->
  (float, 'b) Nx.dtype ->
  'b params
(** [make ~embed_dim dtype] is a fresh layer attending over [embed_dim]
    features: four {!Linear.make} projections with [inputs] and [outputs] both
    [embed_dim]. [w_init], [bias_init] and [bias] are passed to every projection
    and have the defaults of {!Linear.make}.

    Random initializers draw from the implicit RNG scope (see {!Nx.Rng}).

    Raises [Invalid_argument] if [embed_dim] is not positive. *)

val init : embed_dim:int -> t
(** [init ~embed_dim] is [make ~embed_dim Nx.float32]: Glorot-uniform weights,
    zero biases. *)

(** {1:applying Applying} *)

val apply :
  ?num_heads:int ->
  ?causal:bool ->
  'b params ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t
(** [apply p x] is multi-head self-attention over [x], with:

    - [num_heads], the number of attention heads. Defaults to [1]. It must
      divide the embedding dimension; each head attends over
      [embed_dim / num_heads] features of the projected sequences.
    - [causal], whether each position sees only itself and earlier positions.
      Defaults to [false]. With [causal = true] the attention weights of
      position [i] are zero on every position [j > i], so future positions
      cannot influence the output at [i].

    [x]'s last axis must have size [embed_dim] and its second-to-last axis is
    the sequence; earlier axes are batch axes. The result has [x]'s shape.
    Semantically, [apply] projects [x] with [p.q], [p.k] and [p.v], splits each
    projection into [num_heads] heads, runs {!scaled_dot_product_attention} per
    head, concatenates the heads back and projects with [p.out]. Differentiable
    through Rune.

    Raises [Invalid_argument] if [x] has fewer than 2 axes, [x]'s last axis does
    not have size [embed_dim], [num_heads] is not positive, or [num_heads] does
    not divide [embed_dim]. *)

(** {1:cache Decoding with a key-value cache}

    Autoregressive decoding runs the same causal self-attention one query at a
    time: the keys and values of earlier positions never change, so they are
    computed once and cached. A {!Cache.t} holds them in fixed-shape tensors of
    [len] slots and the current position enters {!apply_cached} as a one-element
    tensor, so shapes are independent of the position: a decode step compiled
    once (with {!Rune.jit}) serves the whole generation loop.

    The cache is functional: {!apply_cached} returns the updated cache and never
    mutates its argument. Thread it through the decode loop like any other
    state. *)

(** Key-value caches. *)
module Cache : sig
  type 'b t = { keys : (float, 'b) Nx.t; values : (float, 'b) Nx.t }
  (** The type for key-value caches with float dtype layout ['b]: the projected
      keys and values of the positions seen so far, each of shape
      [[| batch; num_heads; len; head_dim |]]. Slots at positions not yet seen
      hold zeros and are never attended to. *)

  val make :
    ?batch:int ->
    num_heads:int ->
    head_dim:int ->
    len:int ->
    (float, 'b) Nx.dtype ->
    'b t
  (** [make ~num_heads ~head_dim ~len dtype] is an empty cache of [len] slots:
      zero tensors of shape [[| batch; num_heads; len; head_dim |]]. [batch]
      defaults to [1]. [len] bounds the total sequence length (prompt plus
      generated positions).

      Raises [Invalid_argument] if any dimension is not positive. *)

  val map : ('a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t) -> 'b t -> 'b t
  (** [map f c] is [c] with [f] applied to [c.keys] and [c.values], in that
      order. With {!map2} and {!iter} it satisfies the {!Nx.Ptree.S} contract at
      any fixed ['b], so caches can be leaves of a jitted step's parameter tree.
  *)

  val map2 :
    ('a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t -> ('a, 'c) Nx.t) ->
    'b t ->
    'b t ->
    'b t
  (** [map2 f c c'] combines [c] and [c'] leafwise with [f]. *)

  val iter : ('a 'c. ('a, 'c) Nx.t -> unit) -> 'b t -> unit
  (** [iter f c] applies [f] to [c.keys] and [c.values], in that order. *)

  val astype : (float, 'c) Nx.dtype -> 'b t -> 'c t
  (** [astype dt c] is [c] with [c.keys] and [c.values] cast to [dt].
      Differentiable through Rune: gradients flow back at each original leaf's
      dtype, so an astype of float32 parameters inside a loss function yields
      float32 gradients. *)
end

val apply_cached :
  ?num_heads:int ->
  pos:(int32, Nx.int32_elt) Nx.t ->
  cache:'b Cache.t ->
  'b params ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t * 'b Cache.t
(** [apply_cached ~pos ~cache p x] is causal multi-head self-attention of [x]
    over the cached sequence: the result, of [x]'s shape, and the cache with
    [x]'s keys and values written at slots [pos] to [pos + seq - 1].

    [x] has shape [[| batch; seq; embed |]] — the positions [pos] to
    [pos + seq - 1] of the sequence — and [pos] is a one-element int32 tensor.
    The query at input position [i] attends to cache slots [j <= pos + i], so a
    full-prompt prefill at [pos = 0] matches [apply ~causal:true] and a
    single-token step ([seq = 1]) attends to every position seen so far.
    [num_heads] is as in {!apply}.

    Slots are addressed with tensor arithmetic on [pos] (a gather and a position
    mask), never its value, so the step traces once under {!Rune.jit} whatever
    the position. Differentiable through Rune.

    The caller steps [pos] itself and must keep [pos + seq <= len]: writes past
    the last slot are silently dropped.

    Raises [Invalid_argument] if [x] is not of rank 3 or its last axis is not
    [embed], [num_heads] is invalid, the cache's batch, heads or head dimension
    disagree with [x] and [num_heads], [seq] exceeds the cache length, or [pos]
    has more than one element. *)

(** {1:core The attention core} *)

val scaled_dot_product_attention :
  ?mask:(bool, Nx.bool_elt) Nx.t ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t
(** [scaled_dot_product_attention q k v] is [softmax (q @ kᵀ / sqrt d) @ v]:
    each of the [n] query rows takes a weighted average of the [m] value rows,
    weighted by the softmax of its scaled dot products with the key rows.

    [q] has shape [[| ...; n; d |]], [k] shape [[| ...; m; d |]] and [v] shape
    [[| ...; m; dv |]]; the result has shape [[| ...; n; dv |]]. Leading axes
    are batch axes and broadcast, so stacked attention heads are just a batch
    axis. Differentiable through Rune.

    [mask], when given, must broadcast to [[| ...; n; m |]]: weights are
    computed only where it is [true], and are exactly [0] where it is [false]
    (masked scores are set to negative infinity before the softmax). Every query
    row must keep at least one unmasked key, otherwise its output is [nan].

    For half and quarter precision inputs (float16, bfloat16, float8) the
    scores, masking and softmax are computed in a float32 island: [q] and [k]
    are upcast, the probabilities are cast back to the input dtype, and the
    value matmul runs at the input dtype. Float32 and float64 inputs use their
    own dtype throughout, exactly as if the island were absent. {!apply} and
    {!apply_cached} inherit this contract.

    Raises [Invalid_argument] if [q], [k] or [v] has fewer than 2 axes, [q] and
    [k] differ in their last axis, or [k] and [v] differ in their second-to-last
    axis. *)

(** {1:traversals Traversals}

    Plain traversals over the parameter leaves, in the order [q], [k], [v],
    [out], each traversed as by {!Linear}. They satisfy the {!Nx.Ptree.S}
    contract at any fixed ['b]. *)

val map : ('a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t) -> 'b params -> 'b params
(** [map f p] is [p] with [f] applied to every parameter leaf. *)

val map2 :
  ('a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t -> ('a, 'c) Nx.t) ->
  'b params ->
  'b params ->
  'b params
(** [map2 f p p'] combines [p] and [p'] leafwise with [f].

    Raises [Invalid_argument] if a projection of [p] has a bias and the
    corresponding projection of [p'] does not (see {!Linear.map2}). *)

val iter : ('a 'c. ('a, 'c) Nx.t -> unit) -> 'b params -> unit
(** [iter f p] applies [f] to every parameter leaf of [p]. *)

val astype : (float, 'c) Nx.dtype -> 'b params -> 'c params
(** [astype dt p] is [p] with every parameter leaf cast to [dt]. Differentiable
    through Rune: gradients flow back at each original leaf's dtype, so an
    astype of float32 parameters inside a loss function yields float32
    gradients. *)

val names : 'b params -> string list
(** [names p] is the checkpoint name of each parameter leaf of [p], in traversal
    order: each projection's {!Linear.names} prefixed with ["q."], ["k."],
    ["v."] and ["out."] (e.g. [["q.w"; "q.b"; ...; "out.b"]]). See
    {!Checkpoint.Named}. *)
