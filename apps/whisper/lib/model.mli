(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Whisper's model: an audio encoder and a text decoder (Radford et al.,
    {e Robust Speech Recognition via Large-Scale Weak Supervision}, 2022, §2.3).

    The {e encoder} reads a log-mel spectrogram of [n_mels] bands over
    [2 * n_audio_ctx] frames, laid out channels last, and gives [n_audio_ctx]
    {e audio features}: two convolutions, the second of stride 2, each followed
    by GELU, a stored table of positions added, then transformer blocks and a
    layer norm. The {e decoder} reads token ids and attends to their
    predecessors and to the audio features, and gives a logit per token of the
    vocabulary at each position. Its blocks attend through a key-value cache, so
    a transcript is decoded one token at a time at the cost of that token.

    A model is a record of {!Kaun} layers over a payload ['a]: tensors of one
    float dtype on the set ['d]. The released checkpoints store their weights at
    float16 and Whisper computes in float32, so a model is usually loaded at
    float32, where each weight is its float16 value exactly. Every block is
    pre-norm: its sublayers add their output to the residual stream, each
    reading it through its own layer norm. *)

(** {1:dims Dimensions} *)

type dims = {
  n_mels : int;  (** Mel bands of the input: 80, or 128 for large-v3. *)
  n_audio_ctx : int;  (** Audio features: 1500, for 3000 frames. *)
  n_audio_state : int;  (** The encoder's width. *)
  n_audio_head : int;  (** The encoder's attention heads. *)
  n_audio_layer : int;  (** The encoder's blocks. *)
  n_vocab : int;  (** Tokens, the tokenizer's {!Tokenizer.vocab_size}. *)
  n_text_ctx : int;  (** Positions of the decoder: 448. *)
  n_text_state : int;  (** The decoder's width. *)
  n_text_head : int;  (** The decoder's attention heads. *)
  n_text_layer : int;  (** The decoder's blocks. *)
}
(** The type for a model's dimensions, as a checkpoint's [dims] holds them. *)

val multilingual : dims -> bool
(** [multilingual dims] is [true] iff the model transcribes other languages than
    English: its vocabulary has 51865 tokens or more. *)

val languages : dims -> int
(** [languages dims] is the number of languages of the tokenizer
    ({!Tokenizer.of_ranks}) of a released model of [dims]: 99 for the
    English-only models and the multilingual ones before large-v3, 100 from it
    on. *)

(** {1:params Parameters} *)

type ('a, 'd) encoder_block = {
  attn_ln : ('a, 'd) Kaun.Layer_norm.t;
  attn : ('a, 'd) Kaun.Attention.t;  (** Its key has no bias. *)
  mlp_ln : ('a, 'd) Kaun.Layer_norm.t;
  mlp_up : ('a, 'd) Kaun.Linear.t;  (** Width to four times the width. *)
  mlp_down : ('a, 'd) Kaun.Linear.t;
}
(** The type for the encoder's blocks: self-attention, then an MLP with GELU
    between its two projections. *)

type ('a, 'd) decoder_block = {
  attn_ln : ('a, 'd) Kaun.Layer_norm.t;
  attn : ('a, 'd) Kaun.Attention.t;
  cross_attn_ln : ('a, 'd) Kaun.Layer_norm.t;
  cross_attn : ('a, 'd) Kaun.Attention.t;
      (** Queries from the text, keys and values from the audio features. *)
  mlp_ln : ('a, 'd) Kaun.Layer_norm.t;
  mlp_up : ('a, 'd) Kaun.Linear.t;
  mlp_down : ('a, 'd) Kaun.Linear.t;
}
(** The type for the decoder's blocks: causal self-attention, attention to the
    audio features, then an MLP as in {!encoder_block}. Every key projection has
    no bias. *)

type ('a, 'd) encoder = {
  conv1 : ('a, 'd) Kaun.Conv.t;  (** [n_mels] to the width, extent 3. *)
  conv2 : ('a, 'd) Kaun.Conv.t;  (** Width to width, extent 3, stride 2. *)
  positions : 'a;
      (** [[| n_audio_ctx; n_audio_state |]], added to the convolutions' output.
          The checkpoint's table, sinusoids rounded to float16. *)
  blocks : ('a, 'd) encoder_block list;
  ln_post : ('a, 'd) Kaun.Layer_norm.t;
}
(** The type for the encoder's parameters. *)

type ('a, 'd) decoder = {
  tokens : ('a, 'd) Kaun.Embedding.t;
      (** [[| n_vocab; n_text_state |]], also the output projection. *)
  positions : ('a, 'd) Kaun.Embedding.t;
      (** [[| n_text_ctx; n_text_state |]]. *)
  blocks : ('a, 'd) decoder_block list;
  ln : ('a, 'd) Kaun.Layer_norm.t;
}
(** The type for the decoder's parameters. *)

type ('a, 'd) t = {
  dims : dims;
  encoder : ('a, 'd) encoder;
  decoder : ('a, 'd) decoder;
}
(** The type for models over payload ['a]. *)

module Params : Nx.Ptree.S with type ('a, 'd) t = ('a, 'd) t
(** The models' structure: [Nx.Ptree.instantiate (module Params)] is what
    {!Rune.val-jit}'s signatures, {!Nx.Ptree.cast}, {!Nx.Ptree.place} and
    {!Nx_io.Archive.of_value} take. Leaves are at [encoder.conv1.w], ...,
    [encoder.positions], [encoder.blocks.0.attn_ln.gamma], ...,
    [decoder.tokens.table], ..., [decoder.ln.beta]. The dimensions are reported
    as integers at [dims.n_mels], ..., so a compiled function keeps one program
    per model's dimensions. *)

(** {1:encoder Encoder} *)

val encode :
  ((float, 'b, 'd) Nx.t, 'd) t -> (float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t
(** [encode m mel] is the audio features of the log-mel spectrograms [mel], of
    shape [[| n; 2 * n_audio_ctx; n_mels |]]: shape
    [[| n; n_audio_ctx; n_audio_state |]]. Each convolution keeps its input's
    extent under a padding of one frame each side, the second halving it by its
    stride. Every attention head scores a query against a key as their dot
    product over the square root of the head's width.

    Raises [Invalid_argument] if [mel] does not have that shape. *)

(** {1:decoder Decoder}

    The decoder reads the ids of a transcript's tokens and gives, at each
    position, a residual stream, whose {!logits} are those of the token that
    follows. Each block attends to the tokens before its own through a key-value
    cache ({!Kaun.Attention.cached}), and to the audio features through their
    keys and values, computed once per audio by {!val-audio}. A
    {!Kaun.Cache_index.t} says where the call's tokens sit: decoding feeds the
    start sequence at once over [Kaun.Cache_index.rows ~context:n_text_ctx],
    then one token a call over {!Kaun.Cache_index.advance} of the index before.
    A token's position, read from the index, selects its learned position
    embedding. *)

type 'a audio = { keys : 'a; values : 'a }
(** The type for the audio features as one decoder block's cross-attention reads
    them: [keys] and [values] each of shape
    [[| n; n_text_head; n_audio_ctx; n_text_state / n_text_head |]]. *)

module Audio : Nx.Ptree.S with type ('a, 'd) t = 'a audio
(** The structure of {!type-audio}, at the paths [keys] and [values]. A
    decoder's audio is [Nx.Ptree.list (Nx.Ptree.instantiate (module Audio))],
    and its decoding state
    [Nx.Ptree.list (Nx.Ptree.instantiate (module Kaun.Attention.Cache))]: with
    {!Params} and {!Kaun.Cache_index.walk}, what a signature of {!decode} for
    {!Rune.val-jit} takes. *)

val audio :
  ((float, 'b, 'd) Nx.t, 'd) t ->
  (float, 'b, 'd) Nx.t ->
  (float, 'b, 'd) Nx.t audio list
(** [audio m features] is the keys and values of the audio [features], of shape
    [[| n; n_audio_ctx; n_audio_state |]], one {!type-audio} per decoder block
    in block order.

    Raises [Invalid_argument] if [features] does not have that shape. *)

val cache :
  ((float, 'b, 'd) Nx.t, 'd) t ->
  slots:int ->
  'd Nx.Placement.t ->
  ((float, 'b, 'd) Nx.t, 'd) Kaun.Attention.Cache.t list
(** [cache m ~slots at] is an empty decoding state at [at], at [m]'s dtype: one
    key-value cache of [slots] slots per decoder block, in block order.
    [Kaun.Cache_index.rows ~context lens] needs [Array.length lens * context] of
    them, and {!Kaun.Cache_index.whole} none.

    Raises [Invalid_argument] if [slots] is negative. *)

val decode :
  ((float, 'b, 'd) Nx.t, 'd) t ->
  (float, 'b, 'd) Nx.t audio list ->
  ((float, 'b, 'd) Nx.t, 'd) Kaun.Attention.Cache.t list ->
  'd Kaun.Cache_index.t ->
  'd Nx.int64_t ->
  (float, 'b, 'd) Nx.t * ((float, 'b, 'd) Nx.t, 'd) Kaun.Attention.Cache.t list
(** [decode m audio caches index ids] is the residual stream after the last
    block of the tokens [ids] of shape [[| n; seq |]], which sit where [index]
    says, and [caches] with their keys and values written: shape
    [[| n; seq; n_text_state |]]. A token sees the tokens of its sequence at or
    before its own position, those of this call included, and every audio
    feature of its lane's audio. Select the positions to read, then take their
    {!logits}.

    [audio] holds [a] audios' keys and values for the index's [n] lanes, [a]
    dividing [n]: lane [l] reads audio [l / (n / a)], so a group of lanes
    decoding one audio, as the beams of a beam search do, shares its keys and
    values, which are not repeated. The result is that of [audio] with each
    audio repeated for each lane of its group, bit for bit when computed
    eagerly; under [Rune.jit] the two are two programs, which may round
    differently.

    Raises [Invalid_argument] if [audio] or [caches] does not have one entry per
    decoder block, the index's context exceeds [n_text_ctx], [audio]'s lanes do
    not divide the index's, or as {!Kaun.Attention.cached} does. *)

val logits :
  ((float, 'b, 'd) Nx.t, 'd) t -> (float, 'b, 'd) Nx.t -> 'd Nx.float32_t
(** [logits m h] is the logits, at float32, of the residual stream [h] that
    {!decode} gives, of shape [[| ...; n_text_state |]]: shape
    [[| ...; n_vocab |]], the logits of the token that follows each position.
    They are the last layer norm's output times the token embeddings' transpose,
    computed at [m]'s dtype and then converted.

    Raises [Invalid_argument] if [h]'s last axis is not [n_text_state]. *)

(** {1:checkpoints Checkpoints}

    A released checkpoint is a PyTorch file of the dict
    [{"dims": dims, "model_state_dict": state}], whose state names each weight
    by its module's path, as [encoder.blocks.0.attn.query.weight]. A linear
    layer's weight is stored [[| outputs; inputs |]] and a convolution's
    [[| out; in; k |]]; a model holds views of them in kaun's layouts. *)

val of_checkpoint :
  (float, 'b) Nx.dtype ->
  Nx_io.Pytorch.value ->
  ((float, 'b, Nx.host) Nx.t, Nx.host) t
(** [of_checkpoint dt c] is the model of the checkpoint [c] at [dt]. At the
    stored dtype the leaves are the file's tensors; at another each is cast once
    ({!Nx_io.Archive.float}). The model's views of the file's tensors copy
    nothing.

    Raises [Failure], naming the entry, if [c] is not such a dict, a dimension
    is missing or not an integer, or a weight is missing, has another shape than
    the dimensions give it, or is not a float. *)

val load :
  (float, 'b) Nx.dtype -> string -> ((float, 'b, Nx.host) Nx.t, Nx.host) t
(** [load dt path] is [of_checkpoint dt (Nx_io.load_pytorch path)].

    Raises [Failure] as {!Nx_io.load_pytorch} and {!of_checkpoint} do. *)

(** {1:releases Releases} *)

(** The type for the released checkpoints. *)
type release =
  | Tiny_en
  | Tiny
  | Base_en
  | Base
  | Small_en
  | Small
  | Medium_en
  | Medium
  | Large_v1
  | Large_v2
  | Large_v3
  | Large_v3_turbo

val release_of_string : string -> release option
(** [release_of_string s] is the release upstream names [s]: ["tiny.en"],
    ["tiny"], ..., ["large-v3"], ["large-v3-turbo"], and the aliases ["large"]
    for large-v3 and ["turbo"] for large-v3-turbo. *)

val alignment_heads : release -> (int * int) list
(** [alignment_heads r] is the [(layer, head)] pairs, in increasing order, of
    the decoder's cross-attention heads whose weights follow the alignment of
    text to audio in the release [r]: the heads word timestamps read. *)
