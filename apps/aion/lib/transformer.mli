(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** AION-1's transformer: embeddings of tokens, and predictions of tokens.

    The transformer is a 4M encoder-decoder (Mizrahi et al., "4M: Massively
    Multimodal Masked Modeling", NeurIPS 2023). It reads the tokens of some
    modalities of a batch of objects ({!Modality}) and predicts the tokens of
    others.

    The {e encoder} reads one row of tokens per object. Each token is the sum of
    its token embedding, a row of its modality's table, and its
    {e position embedding}: its modality's encoder embedding of the token's
    place within the modality (a fixed sine-cosine table in AION-1's weights),
    plus the modality's learned {e modality embedding}. An input's tokens may
    be partly {e absent}; the row holds the present tokens first, in the order
    of the inputs and of their tokens, then the absent ones, zeroed, and the
    object's {e length} counts the present ones. Every attention over this row sees only
    the keys before the object's length. The encoder's blocks, then a layer
    norm, give each token a vector, and the {e context} is a linear projection
    of it (with a bias) plus the token's position embedding again.

    The {e decoder} reads one row per object as well, one token for each
    predicted token of the {e targets}, the modalities predicted, at all of a
    target's positions or at chosen ones: each is a learned {e mask token} plus
    the predicted token's position embedding, by the modality's decoder table
    (in AION-1's weights equal to the encoder's). A decoder token attends to the
    tokens of its own target, and to the context of its object's present tokens.
    Each target's projection of the decoder's output gives its {e logits}, one
    score per token of its vocabulary.

    Every block is pre-norm, with residual connections. An encoder block is
    self-attention then a SwiGLU feed-forward; a decoder block is
    self-attention, cross-attention to the context, then the feed-forward. The
    feed-forward is [w2 (silu (w1 x) · w3 x)]. In each attention, a layer norm
    over each head's queries and keys follows their projections. Every layer
    norm has a scale and a shift, and epsilon [1e-6]. No linear layer has a bias
    but the context projection.

    The weights fix the width, the number of blocks and of heads, and the
    modalities. AION-1's base weights ({!Hub.weights}) have width 768, 12
    encoder and 12 decoder blocks of 12 heads and a feed-forward of 2048, and
    every modality but {!Modality.Catalog}, whose model tokens no codec makes.

    A transformer is a structure ({!walk}) of weights on a device set:
    [Nx.Ptree.place (module Transformer)] moves it and
    [Nx.Ptree.cast (module Transformer)] gives it another float dtype, as
    float64 for a reference computation. It computes with tokens of its set at
    its weights' float dtype, float32 as AION-1 stores them.

    Compiled by [Rune.jit] with the transformer as an argument, as in
    [Rune.jit Nx.Ptree.(instantiate (module Transformer)
     @-> list (v Tokens.walk)
     @-> returns (instantiate (module Transformer.Encoding)))
     Transformer.encode], it computes to the accuracy [Rune.jit] states: its
    sums may combine their terms in another order than eager ones, so its
    results differ in their last bits.
    A program is traced per list of modalities and per shape, dtype and
    placement of the arguments, the transformer's included. *)

(** {1:transformers Transformers} *)

type ('a, 'd) t
(** The type for transformers whose weights are ['a] on the device set ['d]:
    [((float, 'b, 'd) Nx.t, 'd) t] computes at the float dtype ['b]. *)

val walk : ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> ('a, 'd) t -> ('b, 'e) t
(** [walk c t] walks [t]'s weights and reports its modalities as cases and the
    number of its blocks. *)

val of_archive : 'd Nx_io.Archive.t -> ('d Nx.float32_t, 'd) t
(** [of_archive a] is the transformer whose weights are [a]'s, named as in
    AION-1's file [model.safetensors]. Its modalities are the modalities of
    {!Modality.all}, the catalog excepted, whose encoder token embedding [a]
    holds, in {!Modality.all}'s order. Each modality's tables must have the
    {!Modality.tokens} and {!Modality.vocab} of that modality.

    The encoder reads each modality's encoder position embedding and the
    decoder its decoder one. The decoder's token embeddings and the catalog's
    tensors are not read. Each modality's embedding is stored once, under the
    decoder's name. AION-1's transformer is [of_archive (Hub.weights Base)].

    Raises [Failure] if a weight is missing, is not float32 or has another
    shape, if [a] holds no modality, or if [a] holds a tensor the transformer
    does not read. *)

type arch = {
  width : int;  (** The width of every token's vector: 768. *)
  heads : int;  (** The heads of each attention: 12. *)
  blocks : int * int;  (** The encoder's and the decoder's blocks: 12 each. *)
  hidden : int;  (** The feed-forward's width: 2048. *)
  modalities : Modality.t list;  (** The modalities read and predicted. *)
}
(** The type for transformer architectures. *)

val init : key:'d Nx.Rng.key -> arch -> ('d Nx.float32_t, 'd) t
(** [init ~key a] is a transformer of architecture [a] at [key]'s set, with
    kaun's initialisers, each weight drawn from a key folded from [key]:
    Glorot-uniform linear layers, the context projection's bias zero, identity
    layer norms, token embeddings of {!Kaun.Embedding.init}, and position,
    modality and mask embeddings of standard deviation [0.02].

    Raises [Invalid_argument] if [width], [heads] or [hidden] is not positive,
    if a count of [blocks] is negative, if [heads] does not divide [width], or
    if [a] has no modality or repeats one. *)

val modalities : ('a, 'd) t -> Modality.t list
(** [modalities t] is the modalities [t] reads and predicts. *)

val width : ((float, 'b, 'd) Nx.t, 'd) t -> int
(** [width t] is the width of [t]'s token vectors ({!arch}'s [width]), the
    last dimension of an {!embedding}. *)

(** {1:computing Computing}

    The encoding and the logits of a batch of [b] objects are functions of the
    inputs, each a modality's {!Tokens.t}, given in an order: an input's tokens
    follow the tokens of the inputs before it. {!encode} computes the context,
    {!decode} the logits from a context, and {!forward} both, once each.

    Each call checks its arguments, then computes. It checks shapes at the
    call, and the values of the tokens, their presence, the lengths and the
    positions as {!Nx.check} does: at the call on the host, eagerly or compiled,
    and at a later wait on another set.

    A call raises [Invalid_argument] if it has no input or no target, if two
    inputs or two targets are one modality, if [t] lacks an input's or a
    target's modality, if inputs have different numbers of objects, if a present
    token is outside its modality's vocabulary ([\[0, ]{!Modality.vocab}[)]), or
    if an object has no present token. *)

type ('a, 'd) encoding = {
  context : 'a;
      (** The context of each object's row of tokens, [[|b; n; width|]], [n] the
          inputs' tokens in all. Rows past an object's length are computed from
          the zeroed absent tokens. *)
  lengths : 'd Nx.int64_t;  (** Each object's length, [[|b|]], in [[1, n]]. *)
}
(** The type for encodings, whose context is ['a]. *)

module Encoding : Nx.Ptree.S with type ('a, 'd) t = ('a, 'd) encoding
(** Encodings as a structure: its walk walks the context at the parameter's
    position and the lengths as a fixed array. *)

type 'd target = {
  modality : Modality.t;
  positions : 'd Nx.int64_t option;
      (** [[|b; k|]], the [k] positions of the modality's tokens predicted for
          each object, in [\[0, n)] and distinct within a row, [n] the
          modality's {!Modality.tokens}; [None] for all [n] in order. *)
}
(** The type for the tokens of one modality predicted for each object. The
    decoder holds one token for each predicted position, and each attends to the
    tokens of its own target, so a prediction depends on which positions its
    target holds: the logits of a target of a few positions differ from the same
    rows of a target of all of them. *)

val walk_target :
  ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> 'd target -> 'e target
(** [walk_target c t] reports [t]'s modality and whether its positions are
    given, and walks them as a fixed array. *)

val encode :
  ((float, 'b, 'd) Nx.t, 'd) t ->
  'd Tokens.t list ->
  ((float, 'b, 'd) Nx.t, 'd) encoding
(** [encode t inputs] is the encoder's context of [inputs].

    Raises [Invalid_argument] as the section states, or if an input's [tokens]
    or [present] is not of shape [[|b; n|]], [n] its modality's
    {!Modality.tokens}. *)

val embedding : ((float, 'b, 'd) Nx.t, 'd) encoding -> (float, 'b, 'd) Nx.t
(** [embedding e] is each object's embedding, [[|b; width|]]: the mean of its
    context over its present tokens, the rows before its length (AION-1,
    arXiv:2510.17960, §7.2, Eq. 7). The rows past the length, which the other
    objects' tokens put in the batch, do not enter it, so an object's embedding
    does not depend on the objects encoded with it. *)

val decode :
  ((float, 'b, 'd) Nx.t, 'd) t ->
  ((float, 'b, 'd) Nx.t, 'd) encoding ->
  'd target list ->
  (float, 'b, 'd) Nx.t list
(** [decode t e targets] is the logits of each target of [targets],
    [[|b; k; v|]], [k] the target's number of positions and [v] its
    {!Modality.vocab}, decoded from the context [e]: row [j] of object [i]
    scores the token at the target's [j]th position of that object. A target's
    logits do not depend on the other targets.

    Raises [Invalid_argument] as the section states, if [e] is not an encoding
    by [t]'s width (a [context] of shape [[|b; n; width|]] and [lengths] of
    shape [[|b|]] with each length in [[1, n]]), or if a target's [positions]
    are not of shape [[|b; k|]] with [k >= 1], lie outside [\[0, n)] or repeat
    within a row. *)

val forward :
  ((float, 'b, 'd) Nx.t, 'd) t ->
  'd Tokens.t list ->
  'd target list ->
  (float, 'b, 'd) Nx.t list
(** [forward t inputs targets] is [decode t (encode t inputs) targets].

    Raises as {!encode} and {!decode}. *)
