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

    {!scaled_dot_product_attention} is the pure core — no parameters, no head
    bookkeeping. Use it directly for cross-attention, externally projected
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
    they are computed once and cached. Three values describe a call.

    A {!Cache.t} is a pool of {e slots}, each holding one token's key and value.
    It has no batch axis: which sequence owns a slot is not the cache's
    business.

    A {!Span.t} says where the call's tokens sit: [pos], the position of each
    token in its sequence, and [slots], the slot holding each position of each
    row's sequence. One contiguous run per row is what {!Span.rows} builds;
    paged allocation, a prefix shared by two rows, and a forked beam are other
    values of [slots], and the layer is the same for all of them.

    A {!type-route} is a span resolved against a cache size: a model computes it
    once per call with {!val-route} and hands it to every block.

    Everything is functional: {!cached} returns the written cache and never
    mutates its argument. Thread the cache through the decode loop like any
    other state; under {!Rune.jit} with [~donate:true] the write happens in the
    cache's own storage.

    {b An address outside its range addresses nothing.} A token whose position
    is below [0] or not below [context] is padding: it writes nothing, it is
    rotated and masked as position [0], and its output is unspecified. A column
    whose slot is outside the cache is unallocated: nothing is written to it and
    it reads as zero. [-1] is the conventional value for both. Positions and
    slots are addresses with a no-address value, so unlike tensor indices (see
    {!Nx.take}) they are never out of range, and an eager run and a compiled run
    agree on every input. *)

(** Where a call's tokens sit. *)
module Span : sig
  type t = private { pos : Nx.int32_t; slots : Nx.int32_t }
  (** The type for spans. [pos] has shape [[| batch; seq |]]: [pos.(b).(i)] is
      the position of token [i] of row [b] in its sequence, or [-1] for padding.
      [slots] has shape [[| batch; context |]]: [slots.(b).(j)] is the cache
      slot holding position [j] of row [b]'s sequence, or [-1] when none is
      allocated. Column [j] is position [j]: the key a row stores at a slot was
      rotated for that position, so two rows may share a slot only at the same
      position. *)

  val make : pos:Nx.int32_t -> slots:Nx.int32_t -> t
  (** [make ~pos ~slots] is the span of those tensors.

      Raises [Invalid_argument] unless both have rank 2, the same batch, and no
      empty axis. *)

  val rows : context:int -> int array -> t
  (** [rows ~context lens] gives each row its own run of [context] slots — row
      [b] owns slots [b * context] to [b * context + context - 1] — and places
      its [lens.(b)] tokens at positions [0] to [lens.(b) - 1]. Rows are padded
      on the left to the longest, so [seq] is the largest length and the last
      column is every row's last token; pad the token ids the same way. The
      matching cache has [Array.length lens * context] slots.

      Raises [Invalid_argument] if there is no row, [context] is not positive,
      or a length is negative or exceeds [context]. *)

  val advance : t -> t
  (** [advance s] is [s] with [pos] replaced by one past each row's greatest
      position, of shape [[| batch; 1 |]]: the span of the next token of every
      row. A row of padding advances to position [0]. *)

  val positions : t -> Nx.int32_t
  (** [positions s] is [s.pos] with padding replaced by [0]: every entry is at
      least [0] and below [context]. A model that indexes a table by position
      (learned position embeddings) indexes it with this. *)

  val map : (Nx.int32_t -> Nx.int32_t) -> t -> t
  (** [map f s] applies [f] to [pos] then [slots], for the traversals of a
      jitted step's state. *)

  val map2 : (Nx.int32_t -> Nx.int32_t -> Nx.int32_t) -> t -> t -> t
  (** [map2 f s s'] combines [s] and [s'] leafwise with [f]. *)

  val iter : (Nx.int32_t -> unit) -> t -> unit
  (** [iter f s] applies [f] to [pos] then [slots]. *)
end

(** Key-value caches. *)
module Cache : sig
  type 'a t = { keys : 'a; values : 'a }
  (** The type for key-value caches over payload ['a]. At tensor payloads,
      [keys] and [values] each have shape [[| slots; kv_heads; head_dim |]].
      Slot [s] holds the projected key and value of whatever token a span
      assigned to it; unwritten slots hold zeros. *)

  val make :
    slots:int ->
    kv_heads:int ->
    head_dim:int ->
    (float, 'b) Nx.dtype ->
    (float, 'b) Nx.t t
  (** [make ~slots ~kv_heads ~head_dim dtype] is an empty cache of [slots]
      slots. Use the parameters' dtype.

      Raises [Invalid_argument] if any dimension is not positive. *)

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

  module List : Nx.Ptree.Uniform with type 'a t = 'a t list
  (** One cache per block, in block order: the carried state of a decoder whose
      only state is its attention caches. Leaf paths are ["0.keys"],
      ["0.values"], ["1.keys"], ... *)
end

type route
(** The type for a span resolved against a cache of a given size: which token
    writes each slot, which slot each column reads, which columns are live, and
    the causal mask. It is derived inside the step from the span and is the same
    for every block of a model. *)

val route : slots:int -> Span.t -> route
(** [route ~slots span] resolves [span] against caches of [slots] slots. Token
    [(b, i)] writes slot [span.slots.(b).(span.pos.(b).(i))]; when several
    tokens of the call aim at one slot, the last in row-major order wins, as in
    {!Nx.scatter}. The cost is [slots * batch * seq] int32 comparisons, once per
    call.

    Raises [Invalid_argument] if [slots] is not positive. *)

val cached :
  head_dim:int ->
  ?rope:Rope.t ->
  (float, 'b) Nx.t t ->
  (float, 'b) Nx.t Cache.t ->
  route ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t * (float, 'b) Nx.t Cache.t
(** [cached ~head_dim p cache route x] is causal self-attention of the tokens
    [x], of shape [[| batch; seq; embed_dim |]], over the cached sequences, and
    [cache] with their keys and values written. The result has [x]'s shape.

    Query [(b, i)] sees the columns [j <= pos.(b).(i)] of its row, which include
    the tokens of this call at or before it: a prompt fed whole, in chunks, or
    token by token gives the same outputs up to floating-point reassociation.
    With [rope], queries and keys are rotated at the span's positions before the
    keys are stored. Columns no query of a row may see are read as zero, so they
    contribute exactly zero to the row's outputs, whatever the slot holds.

    The write is one select over the cache and the read one gather of
    [batch * context] slots; both trace once under {!Rune.jit} whatever the
    positions and slots, which enter as tensors. Differentiable through Rune.

    Raises [Invalid_argument] if [x] does not have shape
    [[| batch; seq; embed_dim |]] with the route's batch and seq, [head_dim]
    does not divide the projection widths, [kv_heads] does not divide [heads],
    or the cache does not have shape [[| slots; kv_heads; head_dim |]]. *)

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
    (masked scores are set to negative infinity before the softmax). The
    function is total: a query row whose mask hides every key has zero weights,
    so its output is zero over finite values, and its gradients are zero.

    For half and quarter precision inputs (float16, bfloat16, float8) the
    scores, masking and softmax are computed in a float32 island: [q] and [k]
    are upcast, the probabilities are cast back to the input dtype, and the
    value matmul runs at the input dtype. Float32 and float64 inputs use their
    own dtype throughout, exactly as if the island were absent. {!apply} and
    {!cached} inherit this contract.

    Raises [Invalid_argument] if [q], [k] or [v] has fewer than 2 axes, [q] and
    [k] differ in their last axis, or [k] and [v] differ in their second-to-last
    axis. *)

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
