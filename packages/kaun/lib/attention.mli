(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Multi-head self-attention layers (Vaswani et al., 2017).

    An attention layer is a plain record of four {!Linear} projections: query,
    key, value and output. Construct one with {!init} or {!make}. {!apply} is
    attention of a sequence over itself; {!cached} is attention of new tokens
    over a key-value cache, for decoding. Both split the projections into heads,
    run {!scaled_dot_product_attention} and merge through the output projection.
    Like the other layers, an attention layer composes into models through
    record nesting; the traversals supply the {!Nx.Ptree.Uniform} and checkpoint
    plumbing.

    Head counts are not parameters. [head_dim], the one integer shared by the
    queries, the keys, the cache and the rotary embedding, is an argument of
    {!apply} and {!cached}, and both head counts are read from the projection
    widths: [heads] is the query width over [head_dim] and [kv_heads] the key
    width over [head_dim]. When [kv_heads] is smaller, each key-value head
    serves [heads / kv_heads] query heads (grouped-query attention); the keys
    broadcast over the group, so none is repeated.

    {!scaled_dot_product_attention} is the pure core, with no parameters and no
    head bookkeeping. Use it directly for cross-attention, externally projected
    queries and keys, or custom masking. *)

(** {1:types Types} *)

type 'a t = {
  q : 'a Linear.t;
  k : 'a Linear.t;
  v : 'a Linear.t;
  out : 'a Linear.t;
}
(** The type for attention parameters over payload ['a]: the query, key, value
    and output projections. *)

(** {1:constructors Constructors} *)

val make :
  ?w_init:'b Init.t ->
  ?bias_init:'b Init.t ->
  ?bias:bool ->
  ?q_dim:int ->
  ?kv_dim:int ->
  embed_dim:int ->
  (float, 'b) Nx.dtype ->
  (float, 'b) Nx.t t
(** [make ~embed_dim dtype] is a fresh layer attending over [embed_dim]
    features: [q] maps [embed_dim] to [q_dim], [k] and [v] map [embed_dim] to
    [kv_dim], and [out] maps [q_dim] back to [embed_dim]. [q_dim] and [kv_dim]
    default to [embed_dim]; a smaller [kv_dim] is grouped-query attention.
    [w_init], [bias_init] and [bias] are passed to every projection and have the
    defaults of {!Linear.make}.

    Random initializers draw from the implicit RNG scope (see {!Nx.Rng}).

    Raises [Invalid_argument] if a dimension is not positive. *)

val init : embed_dim:int -> Nx.float32_t t
(** [init ~embed_dim] is [make ~embed_dim Nx.float32]: Glorot-uniform weights,
    zero biases. *)

(** {1:applying Attention of a sequence over itself} *)

val apply :
  head_dim:int ->
  ?mask:(bool, Nx.bool_elt) Nx.t ->
  ?rope:Rope.t ->
  (float, 'b) Nx.t t ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t
(** [apply ~head_dim p x] is multi-head self-attention over [x], with:

    - [head_dim], the feature width of one head. It must divide the projection
      widths.
    - [mask], which keys each query may see, of shape [[| seq; seq |]] or
      [[| batch; seq; seq |]]: weights are computed only where it is [true]. The
      layer inserts the head axes. A query that sees no key yields the output
      projection of zero. Without a mask every token sees every token.
    - [rope], a rotary schedule: the query and key heads are rotated at
      positions [0] to [seq - 1] before the scores.

    [x]'s last axis must have size [embed_dim] and its second-to-last axis is
    the sequence; earlier axes are batch axes, folded into one for [mask]. The
    result has [x]'s shape. Differentiable through Rune.

    Raises [Invalid_argument] if [x] has fewer than 2 axes, [x]'s last axis does
    not have size [embed_dim], [head_dim] does not divide the projection widths,
    [kv_heads] does not divide [heads], or [mask] has another shape. *)

val causal_mask :
  seq:int -> ?valid:(bool, Nx.bool_elt) Nx.t -> unit -> (bool, Nx.bool_elt) Nx.t
(** [causal_mask ~seq ()] is the mask of shape [[| seq; seq |]] under which
    query [i] sees keys [j <= i]. With [valid], of shape [[| batch; seq |]] and
    [true] at real tokens, the result has shape [[| batch; seq; seq |]] and
    hides padded keys as well.

    Raises [Invalid_argument] if [seq] is not positive or [valid] has another
    shape. *)

(** {1:cache Decoding with a key-value cache}

    Autoregressive decoding runs the same causal self-attention on a few new
    tokens at a time: the keys and values of earlier positions never change, so
    they are computed once and cached. A {!Cache.t} is a pool of {e slots}, each
    holding one token's key and value, with no batch axis; a {!Cache_index.t}
    says where the call's tokens sit in it.

    Everything is functional: {!cached} returns the written cache and never
    mutates its argument. Thread the cache through the decode loop like any
    other state; under {!Rune.jit} with [~donate:true] the write happens in the
    cache's own storage.

    The addressing laws are {!Cache_index}'s. The layer adds three:

    + {b Chunking is invariant.} A prompt fed whole, in chunks, token by token,
      or as one-token lanes of one sequence gives the same outputs and the same
      written slots, up to floating-point reassociation.
    + {b The whole-sequence pass is the cached pass.} Over {!Cache_index.whole},
      {!cached} is causal attention of the tokens over themselves and returns
      its cache as given, at the cost of {!apply}; over {!Cache_index.rows} and
      a fresh cache it gives the same outputs. Kaun owes that agreement for this
      layer. A model defines its training forward pass as {!cached} over
      {!Cache_index.whole}, so it has one implementation and none to keep equal.
    + {b One tensor per cache leaf.} Two sequences share a prefix by naming the
      same slots in an index's table, never by two leaves holding one tensor:
      storage reuse under [~donate:true] needs each donated tensor to seed one
      leaf. *)

(** Key-value caches. *)
module Cache : sig
  type 'a t = { keys : 'a; values : 'a }
  (** The type for key-value caches over payload ['a]. At tensor payloads,
      [keys] and [values] each have shape [[| slots + 1; kv_heads; head_dim |]]:
      slot [s] holds the key and value of whatever token a cache index stored
      there, and the last row is the scratch row (see {!Cache_index}). *)

  val make :
    slots:int ->
    kv_heads:int ->
    head_dim:int ->
    (float, 'b) Nx.dtype ->
    (float, 'b) Nx.t t
  (** [make ~slots ~kv_heads ~head_dim dtype] is a cache of [slots] slots
      holding zeros. Use the parameters' dtype. [slots] may be [0]: the state a
      {!Cache_index.whole} call is given.

      Raises [Invalid_argument] if [slots] is negative or another dimension is
      not positive. *)

  val map : ('a -> 'b) -> 'a t -> 'b t
  (** [map f c] is [c] with [f] applied to [c.keys] and [c.values], in that
      order. The traversals satisfy the {!Nx.Ptree.Uniform} contract. *)

  val map2 : ('a -> 'b -> 'c) -> 'a t -> 'b t -> 'c t
  (** [map2 f c c'] combines [c] and [c'] leafwise with [f]. *)

  val iter : ('a -> unit) -> 'a t -> unit
  (** [iter f c] applies [f] to [c.keys] and [c.values], in that order. *)

  val fold : (string -> 'acc -> 'a -> 'acc) -> 'acc -> 'a t -> 'acc
  (** [fold f acc c] reduces [c] leafwise; leaf paths are ["keys"] and
      ["values"]. *)

  val fold2 :
    (string -> 'acc -> 'a -> 'b -> 'acc) -> 'acc -> 'a t -> 'b t -> 'acc
  (** [fold2 f acc c c'] is like {!fold} across two caches. *)

  val names : 'a t -> string t
  (** [names c] is [{ keys = "keys"; values = "values" }]. *)

  val extend :
    Cache_index.t ->
    (float, 'b) Nx.t t ->
    (float, 'b) Nx.t ->
    (float, 'b) Nx.t ->
    (float, 'b) Nx.t * (float, 'b) Nx.t * (float, 'b) Nx.t t
  (** [extend index cache k v] is [(k', v', cache')]: [cache] with the call's
      keys [k] and values [v] stored, both of shape
      [[| batch; kv_heads; seq; head_dim |]], and what the call's tokens attend
      over, of shape [[| batch; kv_heads; context; head_dim |]]. It is
      {!Cache_index.extend} on each pool, which holds tokens before heads, with
      the two axes exchanged on the way in and out. On a whole index [k'] is
      [k], [v'] is [v] and [cache'] is [cache].

      Raises [Invalid_argument] if [k] or [v] does not have that shape for
      [index] and [cache]. *)

  module List : Nx.Ptree.Uniform with type 'a t = 'a t list
  (** One cache per block, in block order: the carried state of a decoder whose
      only state is its attention caches. Leaf paths are ["0.keys"],
      ["0.values"], ["1.keys"], ... *)
end

val cached :
  head_dim:int ->
  ?rope:Rope.t ->
  (float, 'b) Nx.t t ->
  (float, 'b) Nx.t Cache.t ->
  Cache_index.t ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t * (float, 'b) Nx.t Cache.t
(** [cached ~head_dim p cache index x] is causal self-attention of the tokens
    [x], of shape [[| batch; seq; embed_dim |]], over their sequences, and
    [cache] with their keys and values written. The result has [x]'s shape.

    A token sees the positions of its sequence at or before its own, those of
    this call included, and under the index's {!Cache_index.window} only the
    last of them: a prompt fed whole, in chunks, or token by token gives the
    same outputs up to floating-point reassociation. With [rope], queries and
    keys are rotated at the index's positions before the keys are stored. A
    padded token's output is the output projection of zero.

    The layer extends each leaf with {!Cache_index.extend} and attends once over
    what that returns, under {!Cache_index.mask}. On a whole index the tokens
    attend over themselves and [cache] is returned as it is, at the cost of
    {!apply}: a model's whole-sequence forward pass is this function over
    {!Cache_index.whole}. Otherwise each leaf costs one scatter of the call's
    tokens and one gather of its context; both trace once under {!Rune.jit}
    whatever the index holds. With [~donate:true] on a device the step's cost
    does not depend on the size of the cache. Differentiable through Rune.

    Raises [Invalid_argument] if [x] does not have shape
    [[| batch; seq; embed_dim |]] with the index's batch and seq, [head_dim]
    does not divide the projection widths, [kv_heads] does not divide [heads],
    or the cache does not have shape [[| _; kv_heads; head_dim |]]. *)

(** {1:pieces The pieces of a layer}

    {!apply} and {!cached} are compositions of three functions with
    {!Rope.apply}, {!Cache.extend} and {!Cache_index.mask}. A model whose
    attention differs (sinks, another score scale, a normalisation of queries
    and keys, its own cache record) composes them itself. This is {!cached}:

    {[
    let cached ~head_dim ~rope p cache index x =
      let pos = Cache_index.positions index in
      let q = Rope.apply rope ~pos (Attention.split ~head_dim p.q x) in
      let k = Rope.apply rope ~pos (Attention.split ~head_dim p.k x) in
      let v = Attention.split ~head_dim p.v x in
      let k, v, cache = Attention.Cache.extend index cache k v in
      let mask = Cache_index.mask index in
      (Attention.merge p.out (Attention.attend ~mask q k v), cache)
    ]}

    Between the pieces a tensor has shape [[| batch; heads; seq; head_dim |]],
    which is what {!Rope.apply} takes. *)

val split :
  head_dim:int ->
  (float, 'b) Nx.t Linear.t ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t
(** [split ~head_dim l x] projects [x], of shape [[| batch; seq; embed |]],
    through [l] and splits the result into heads: shape
    [[| batch; heads; seq; head_dim |]], where [heads] is [l]'s output width
    divided by [head_dim]. A query projection and a key-value projection of
    different widths give different head counts, which {!attend} pairs.

    Raises [Invalid_argument] if [x] does not have three axes or [head_dim] is
    not positive or does not divide [l]'s output width. *)

val attend :
  ?mask:(bool, Nx.bool_elt) Nx.t ->
  ?scale:float ->
  ?sinks:(float, 'b) Nx.t ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t
(** [attend q k v] is grouped-query attention of [q], of shape
    [[| batch; heads; n; d |]], over [k] and [v], of shape
    [[| batch; kv_heads; m; d |]] and [[| batch; kv_heads; m; dv |]]: the result
    has shape [[| batch; heads; n; dv |]]. [kv_heads] divides [heads], and
    key-value head [h] serves query heads [h * groups] to
    [h * groups + groups - 1], where [groups] is [heads / kv_heads]. No key is
    repeated: the queries gain a group axis the keys broadcast over. With:

    - [mask], of shape [[| n; m |]] or [[| batch; n; m |]], shared by the heads.
    - [scale], as for {!scaled_dot_product_attention}.
    - [sinks], of shape [[| heads |]]: one attention-sink logit per query head,
      as for {!scaled_dot_product_attention}, which is given them reshaped to
      [[| kv_heads; groups; 1 |]].

    It is {!scaled_dot_product_attention} over the grouped shapes and inherits
    its totality and its float32 island.

    Raises [Invalid_argument] if a shape is not as above or [kv_heads] does not
    divide [heads]. *)

val merge : (float, 'b) Nx.t Linear.t -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [merge l y] concatenates the heads of [y], of shape
    [[| batch; heads; seq; d |]], and projects them through [l]: shape
    [[| batch; seq; embed |]]. It undoes {!split}.

    Raises [Invalid_argument] if [y] does not have four axes or [heads * d] is
    not [l]'s input width. *)

(** {1:core The attention core} *)

val scaled_dot_product_attention :
  ?mask:(bool, Nx.bool_elt) Nx.t ->
  ?scale:float ->
  ?sinks:(float, 'b) Nx.t ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t
(** [scaled_dot_product_attention q k v] is [softmax (q @ kᵀ * scale) @ v]: each
    of the [n] query rows takes a weighted average of the [m] value rows,
    weighted by the softmax of its scaled dot products with the key rows.
    [scale] defaults to [1 / sqrt d]; a model whose scores have another
    temperature passes its own (see {!Rope.yarn}).

    [q] has shape [[| ...; n; d |]], [k] shape [[| ...; m; d |]] and [v] shape
    [[| ...; m; dv |]]; the result has shape [[| ...; n; dv |]]. Leading axes
    are batch axes and broadcast, so stacked attention heads are just a batch
    axis. Differentiable through Rune.

    [mask], when given, must broadcast to [[| ...; n; m |]]: weights are
    computed only where it is [true], and are exactly [0] where it is [false]
    (masked scores are set to negative infinity before the softmax). The
    function is total: a query row whose mask hides every key has zero weights,
    so its output is zero over finite values, and its gradients are zero.

    [sinks], when given, are attention sinks (Xiao et al., 2023, as gpt-oss uses
    them): learned logits that take part of a query's weight and carry no value.
    [sinks] must broadcast to [[| ...; n |]], the scores without their last
    axis, so each query row has one: for [q] of shape
    [[| batch; heads; n; d |]], one sink per head is a tensor of shape
    [[| heads; 1 |]]. A query's sink joins its softmax as one more key whose
    value is zero: the scores and the sink are normalised together, the sink is
    a raw logit that [scale] does not multiply, and it has no column in the
    weights, which then sum to less than [1]. A query whose mask hides every key
    puts all its weight on its sink: its output and its gradients are zero.
    Differentiable through Rune in [sinks].

    Without [scale] and [sinks] the computation is exactly the one above.

    For half and quarter precision inputs (float16, bfloat16, float8) the
    scores, masking, sinks and softmax are computed in a float32 island: [q],
    [k] and [sinks] are upcast, the probabilities are cast back to the input
    dtype, and the value matmul runs at the input dtype. Float32 and float64
    inputs use their own dtype throughout, exactly as if the island were absent.
    {!apply} and {!cached} inherit this contract.

    Raises [Invalid_argument] if [q], [k] or [v] has fewer than 2 axes, [q] and
    [k] differ in their last axis, [k] and [v] differ in their second-to-last
    axis, or [sinks] does not broadcast to the scores without their last axis.
*)

(** {1:traversals Traversals}

    Payload traversals in the order [q], [k], [v], [out], each traversed as by
    {!Linear}, satisfying the {!Nx.Ptree.Uniform} contract. Leaf paths are the
    projections' paths prefixed with the field name (["q.w"], ["q.b"], ...,
    ["out.b"]). *)

val map : ('a -> 'b) -> 'a t -> 'b t
(** [map f p] is [p] with [f] applied to every payload leaf. [map (Nx.cast dt)]
    converts a layer's precision; the cast is differentiable through Rune. *)

val map2 : ('a -> 'b -> 'c) -> 'a t -> 'b t -> 'c t
(** [map2 f p p'] combines [p] and [p'] leafwise with [f].

    Raises [Invalid_argument] if a projection of [p] has a bias and the
    corresponding projection of [p'] does not (see {!Linear.map2}). *)

val iter : ('a -> unit) -> 'a t -> unit
(** [iter f p] applies [f] to every payload leaf of [p]. *)

val fold : (string -> 'acc -> 'a -> 'acc) -> 'acc -> 'a t -> 'acc
(** [fold f acc p] reduces [p] leafwise, threading each leaf's path. *)

val fold2 : (string -> 'acc -> 'a -> 'b -> 'acc) -> 'acc -> 'a t -> 'b t -> 'acc
(** [fold2 f acc p p'] is like {!fold} across two structurally equal layers.

    Raises [Invalid_argument] if a projection of [p] has a bias and the
    corresponding projection of [p'] does not. *)

val names : 'a t -> string t
(** [names p] is [p] with every payload replaced by its path. *)
