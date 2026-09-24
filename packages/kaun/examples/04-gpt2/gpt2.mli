(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** GPT-2 (Radford et al., 2019) from kaun layers.

    The model is a plain record of {!Kaun} layers; {!hidden}, {!cached} and
    {!logits} are its forward passes and {!Params} its structure. {!of_hf}
    builds the parameters from the entries of the HuggingFace checkpoint, whose
    [h.{i}.attn.c_attn] fuses the query, key and value projections, and
    {!from_pretrained} downloads the checkpoint first. *)

type config = {
  vocab_size : int;
  n_positions : int;  (** Maximum sequence length. *)
  n_embd : int;
  n_layer : int;
  n_head : int;
  n_inner : int;  (** MLP hidden width, [4 * n_embd] in the released models. *)
  layer_norm_eps : float;
}
(** The type for GPT-2 hyperparameters, as in HuggingFace's [config.json]. *)

type 'a block = {
  ln1 : 'a Kaun.Layer_norm.t;
  attn : 'a Kaun.Attention.t;
  ln2 : 'a Kaun.Layer_norm.t;
  fc : 'a Kaun.Linear.t;  (** MLP up projection, [n_embd → n_inner]. *)
  proj : 'a Kaun.Linear.t;  (** MLP down projection, [n_inner → n_embd]. *)
}
(** The type for one pre-norm transformer block over payload ['a]. *)

type 'a params = {
  wte : 'a Kaun.Embedding.t;  (** Token embeddings, also the tied LM head. *)
  wpe : 'a Kaun.Embedding.t;  (** Learned position embeddings. *)
  blocks : 'a block list;
  ln_f : 'a Kaun.Layer_norm.t;
}
(** The type for GPT-2 parameters over payload ['a]. *)

type t = Nx.float32_t params
(** The type for single-precision GPT-2 parameters, the checkpoint dtype. *)

module Params : Nx.Ptree.S with type 'a t = 'a params
(** The parameters' structure: [Nx.Ptree.instantiate (module Params)] is what
    the transformations, the optimisers and {!Kaun.Checkpoint.of_value} take.
    Leaves are at [wte.table], [wpe.table], [blocks.0.ln1.gamma],
    [blocks.0.attn.q.w], ..., [ln_f.beta], the names a checkpoint gives them.

    [Nx.Ptree.cast (module Params) dt p] converts precision: for half precision
    inference, cast a float32 checkpoint once; for mixed-precision training,
    cast inside the loss function so the float32 parameters receive float32
    gradients. The kaun layers keep their attention-score and layer-norm
    statistics in float32 islands whatever [dt]. *)

val make : config -> t
(** [make cfg] is a zero-initialized model, the [~like] template for
    {!Kaun.Checkpoint.to_value}. *)

(** {1:placement Placement}

    {!of_hf} and {!cache} place each leaf they build with
    [placement role ~axis]: [role] is the cut a tensor-parallel placement makes
    in the leaf and [axis] the axis of the leaf that cut runs along, [0] for
    [Whole]. One device ignores both: [fun _ ~axis:_ -> p]. *)

(** The type for the cuts of a tensor-parallel placement. *)
type role =
  | Whole
      (** Kept whole: embeddings, layer norms, a [Row] projection's bias. *)
  | Column
      (** Cut along its outputs: the query, key and value projections and the
          MLP's up projection, biases included. *)
  | Row
      (** Cut along its inputs: the attention's output projection and the MLP's
          down projection. *)
  | Kv_heads  (** Cut along its heads: a cache pool. *)

(** {1:forward Forward passes}

    One model, one fold over the blocks: {!cached} is the residual stream of
    tokens that sit where a cache index says and attend through key-value
    caches, {!hidden} is [cached] over {!Kaun.Cache_index.whole}, which reads
    and keeps nothing, and {!logits} is the head applied to either. *)

val cache :
  ?placement:(role -> axis:int -> Nx.Placement.t) ->
  config ->
  slots:int ->
  (float, 'b) Nx.dtype ->
  (float, 'b) Nx.t Kaun.Attention.Cache.t list
(** [cache cfg ~slots dtype] is an empty decoding state: one key-value cache per
    block, in block order, each of [slots] slots. Its structure is
    [Nx.Ptree.list (Nx.Ptree.instantiate (module Kaun.Attention.Cache))].
    [Kaun.Cache_index.rows ~context lens] needs [Array.length lens * context] of
    them. [dtype] is the parameters' dtype.

    With [placement], each pool is placed with [placement Kv_heads ~axis:1], so
    a compiled step finds the caches where the model is from its first call. *)

val cached :
  config ->
  ?dropout:float * Nx.Rng.key ->
  (float, 'b) Nx.t params ->
  (float, 'b) Nx.t Kaun.Attention.Cache.t list ->
  Kaun.Cache_index.t ->
  (int32, Nx.int32_elt) Nx.t ->
  (float, 'b) Nx.t * (float, 'b) Nx.t Kaun.Attention.Cache.t list
(** [cached cfg p caches index ids] is the residual stream of the tokens [ids] —
    shape [[| batch; seq; n_embd |]] — which sit where [index] says and attend
    through [caches], and the caches with their keys and values written. A
    whole-prompt call prefills the caches; a single-token call advances decoding
    by one step. The index's positions and slots are tensors, so both trace
    under {!Rune.jit} and one compiled single-token step serves the whole decode
    loop. See {!Kaun.Attention.cached} and {!Kaun.Cache_index}.

    [?dropout:(rate, key)] enables training-time dropout at the canonical GPT-2
    sites — the embedding sum and each block's post-attention and post-MLP
    projections — with masks applied at the activations' dtype and derived from
    [key] by {!Nx.Rng.fold_in}, one subkey per site. The same key gives the same
    masks; derive a fresh key per training step ({!Nx.Rng.fold_in} a step
    counter into a root key), and under {!Rune.jit} pass it as an input leaf of
    the step. Inference (the default) applies no dropout.

    Raises [Invalid_argument] if the index's context exceeds [cfg.n_positions],
    or on the geometry errors of {!Kaun.Attention.cached}. *)

val hidden :
  config ->
  ?dropout:float * Nx.Rng.key ->
  (float, 'b) Nx.t params ->
  (int32, Nx.int32_elt) Nx.t ->
  (float, 'b) Nx.t
(** [hidden cfg ?dropout p ids] is the residual stream after the last block for
    the [[| batch; seq |]] id tensor [ids], of shape [[| batch; seq; n_embd |]],
    at the parameters' dtype. Every token attends to the tokens before it.

    Raises [Invalid_argument] if [ids] has more than [cfg.n_positions]
    positions. *)

val logits :
  config -> (float, 'b) Nx.t params -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [logits cfg p h] is the final layer norm and the language-model head applied
    to a residual stream: [[| ...; n_embd |]] to [[| ...; vocab_size |]]. The
    head is tied to [p.wte]. Both act per position, so select the positions of
    interest first: decoding wants the last one only. *)

val config_of_json : Jsont.json -> config
(** [config_of_json json] reads HuggingFace's [config.json].

    Raises [Failure] on a missing field. *)

val of_hf :
  ?placement:(role -> axis:int -> Nx.Placement.t) ->
  config ->
  (float, 'b) Nx.dtype ->
  Kaun.Checkpoint.t ->
  (float, 'b) Nx.t params
(** [of_hf cfg dt ckpt] is the model of the HuggingFace GPT-2 checkpoint [ckpt],
    at [dt]. Each entry is read by its name in the file with the shape [cfg]
    gives it. The file's weights are already [inputs × outputs]; each block's
    fused [c_attn] weight and bias are cut into the [q], [k] and [v] projections
    with [Nx.split], which copies nothing. At the file's own dtype the leaves
    are the file's entries; at another one each leaf is cast. Entries the model
    does not use (attention mask buffers) are never read.

    With [placement], each leaf is placed with [Nx.place (placement role ~axis)]
    as it is built (see {!role}), so a function compiled where the model is that
    captures it uploads nothing.

    Raises [Invalid_argument], naming the entry, if one is missing, has another
    shape than [cfg] says, or is not a floating-point entry. *)

val from_file :
  ?placement:(role -> axis:int -> Nx.Placement.t) ->
  config ->
  (float, 'b) Nx.dtype ->
  string ->
  (float, 'b) Nx.t params
(** [from_file cfg dt path] is [of_hf cfg dt (Checkpoint.load path)]: the
    parameters of a local HuggingFace-layout safetensors file.

    Raises [Failure] on I/O or format errors, [Invalid_argument] as {!of_hf}. *)

type dtype =
  | Dtype : (float, 'b) Nx.dtype -> dtype
      (** A floating-point dtype chosen at run time. *)

val dtype_of_string : string -> dtype
(** [dtype_of_string s] is the dtype named ["float32"], ["float16"] or
    ["bfloat16"]. Raises [Failure] on another name. *)

val stored_dtype : Kaun.Checkpoint.t -> dtype
(** [stored_dtype ckpt] is the dtype [ckpt] stores its token table at, the dtype
    at which {!of_hf} casts nothing. *)

val from_pretrained :
  ?placement:(role -> axis:int -> Nx.Placement.t) ->
  ?repo_id:string ->
  (float, 'b) Nx.dtype ->
  config * (float, 'b) Nx.t params
(** [from_pretrained dt] downloads [repo_id] (defaults to ["gpt2"]) from the
    HuggingFace Hub, config and weights, and is the parsed configuration with
    the pretrained parameters at [dt].

    Raises [Failure] on download or parse errors. *)
