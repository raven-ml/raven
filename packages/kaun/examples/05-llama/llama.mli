(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Llama (Touvron et al., 2023; Llama 3, 2024) from kaun layers.

    A pre-norm decoder: RMS normalization, grouped-query attention with rotary
    positions, a SwiGLU feed-forward. The model is a plain record of {!Kaun}
    layers written on the decode contract: {!hidden}, {!cached} and {!logits}
    are its forward passes and {!Params} its structure. {!of_hf} adapts the
    HuggingFace weights — [torch.nn.Linear] orientation and naming — onto
    {!Params}' names, {!from_pretrained} runs the whole pipeline, and {!of_gguf}
    reads a GGUF checkpoint, quantised or not. *)

type 'd config = {
  vocab_size : int;
  dim : int;
  n_layers : int;
  n_heads : int;
  n_kv_heads : int;  (** Key-value heads; divides [n_heads]. *)
  head_dim : int;
  hidden_dim : int;  (** SwiGLU inner width. *)
  norm_eps : float;
  rope : 'd Kaun.Rope.t;  (** The rotary schedule, where the model reads it. *)
  tied : bool;  (** Whether the head is the token table (Llama 3.2 1B, 3B). *)
}
(** The type for Llama hyperparameters, with the rotary schedule on the set
    ['d]. *)

val place_config : 'e Nx.Placement.t -> 'd config -> 'e config
(** [place_config at cfg] is [cfg] with its rotary schedule at [at]
    ({!Kaun.Rope.place}). {!config_of_json} makes it on the host; a model on
    other devices places it once, where its parameters lie. *)

type ('a, 'd) block = {
  attn_norm : ('a, 'd) Kaun.Rms_norm.t;
  attn : ('a, 'd) Kaun.Attention.t;
  ffn_norm : ('a, 'd) Kaun.Rms_norm.t;
  gate : ('a, 'd) Kaun.Linear.t;
  up : ('a, 'd) Kaun.Linear.t;
  down : ('a, 'd) Kaun.Linear.t;
}
(** The type for one block over payload ['a]. No projection has a bias. *)

type ('a, 'd) params = {
  tok : ('a, 'd) Kaun.Embedding.t;
  blocks : ('a, 'd) block list;
  norm : ('a, 'd) Kaun.Rms_norm.t;
  head : ('a, 'd) Kaun.Linear.t option;  (** [None]: tied to [tok]. *)
}
(** The type for Llama parameters over payload ['a]. *)

type 'd t = ('d Nx.float32_t, 'd) params
(** The type for single-precision Llama parameters on the set ['d]. *)

module Params : Nx.Ptree.S with type ('a, 'd) t = ('a, 'd) params
(** The parameters' structure: [Nx.Ptree.instantiate (module Params)] is what
    the transformations, the optimisers and {!Nx_io.Archive.of_value} take.
    Leaves are at [tok.table], [blocks.0.attn.q.w], [norm.gamma], [head.w], ...,
    their names in a saved file. [Nx.Ptree.cast (module Params) dt p] converts
    precision; the layers compute their attention scores and norms at
    float32 or wider whatever [dt]. *)

val make : key:'d Nx.Rng.key -> 'e config -> 'd t
(** [make ~key cfg] is a zero-initialized float32 model where [key] lies: the
    starting point of training from scratch, and the [~like] value that
    {!Nx_io.Archive.to_value} needs to read back parameters this library saved.
*)

(** {1:placement Placement}

    {!of_hf} and {!cache} make the arrays a model starts from, the file's
    entries and empty pools, and place each leaf they build with
    [placement role ~axis]: [role] is the cut a tensor-parallel placement makes
    in the leaf and [axis] the axis of the leaf that cut runs along, [0] for
    [Whole]. One device ignores both: [fun _ ~axis:_ -> p], and the host is
    [fun _ ~axis:_ -> Nx.Host.on]. A compiled step over several devices
    compiles as over one ({!Rune.section-compiled}). *)

(** The type for the cuts of a tensor-parallel placement. *)
type role =
  | Whole  (** Kept whole: the token table and the norms. *)
  | Column
      (** Cut along its outputs: the query, key and value projections, the gate
          and up projections, and an untied head. *)
  | Row
      (** Cut along its inputs: the attention's output projection and the down
          projection. *)
  | Kv_heads  (** Cut along its heads: a cache pool. *)

(** {1:forward Forward passes}

    One fold over the blocks: [hidden cfg p ids] is
    [fst (cached cfg p caches index ids)] over {!Kaun.Cache_index.whole}, which
    reads and keeps nothing, so training never writes or reads a cache. *)

val hidden :
  'd config ->
  ((float, 'b, 'd) Nx.t, 'd) params ->
  'd Nx.int64_t ->
  (float, 'b, 'd) Nx.t
(** [hidden cfg p ids] is the residual stream after the last block for the
    [[| batch; seq |]] id tensor [ids], of shape [[| batch; seq; dim |]]. Every
    token attends to the tokens before it. *)

val cache :
  placement:(role -> axis:int -> 'd Nx.Placement.t) ->
  'e config ->
  slots:int ->
  (float, 'b) Nx.dtype ->
  ((float, 'b, 'd) Nx.t, 'd) Kaun.Attention.Cache.t list
(** [cache ~placement cfg ~slots dtype] is an empty decoding state: one
    key-value cache per block, in block order, each of [slots] slots, at the
    parameters' dtype. Its structure is
    [Nx.Ptree.list (Nx.Ptree.instantiate (module Kaun.Attention.Cache))].

    Each pool is made at [placement Kv_heads ~axis:1], so
    a compiled step finds the caches where the model is from its first call, on
    a set that computes under [Rune.jit] alone too. *)

val cached :
  'd config ->
  ((float, 'b, 'd) Nx.t, 'd) params ->
  ((float, 'b, 'd) Nx.t, 'd) Kaun.Attention.Cache.t list ->
  'd Kaun.Cache_index.t ->
  'd Nx.int64_t ->
  (float, 'b, 'd) Nx.t * ((float, 'b, 'd) Nx.t, 'd) Kaun.Attention.Cache.t list
(** [cached cfg p caches index ids] is the residual stream of the tokens [ids],
    which sit where [index] says and attend through [caches], and the caches
    with their keys and values written. See {!Kaun.Attention.cached}. *)

val logits :
  'e config ->
  ((float, 'b, 'd) Nx.t, 'd) params ->
  (float, 'b, 'd) Nx.t ->
  (float, 'b, 'd) Nx.t
(** [logits cfg p h] is the final norm and the language-model head applied to a
    residual stream, per position. Select positions first. *)

(** {1:loading Pretrained weights} *)

val config_of_json : Jsont.json -> Nx.host config
(** [config_of_json json] reads HuggingFace's [config.json], including the
    ["llama3"] rotary scaling.

    Raises [Failure] on a missing field or another rotary scaling type. *)

val of_hf :
  placement:(role -> axis:int -> 'd Nx.Placement.t) ->
  'c config ->
  (float, 'b) Nx.dtype ->
  'e Nx_io.Archive.t ->
  ((float, 'b, 'd) Nx.t, 'd) params
(** [of_hf ~placement cfg dt weights] is the model of the HuggingFace Llama
    [weights], at [dt]. Each entry is read by its name in the file with the
    shape [cfg] gives it, and every projection is transposed to
    [inputs × outputs], a view. At the file's own dtype nothing is copied; at
    another one each leaf is cast.

    Each leaf is placed with [Nx.place (placement role ~axis)] as it is built
    (see {!role}), before the next is read, so at most one leaf's cast is alive
    on the host and a function compiled where the model is that captures it
    uploads nothing.

    Raises [Failure], naming the entry, if one is missing, has another shape
    than [cfg] says, or is not a float16, bfloat16, float32 or float64 entry. *)

val from_file :
  placement:(role -> axis:int -> 'd Nx.Placement.t) ->
  'c config ->
  (float, 'b) Nx.dtype ->
  string ->
  ((float, 'b, 'd) Nx.t, 'd) params
(** [from_file ~placement cfg dt path] is {!of_hf} on the safetensors file
    [path]. *)

(** {1:gguf GGUF checkpoints}

    A GGUF file ({!Nx_io.Gguf}) of the [llama] architecture, as llama.cpp's
    converter writes it: the configuration in its metadata, under the
    architecture's name, and the weights as [token_embd], [blk.{i}.attn_q], ...,
    [output_norm] and, untied, [output], each in [[| outputs; inputs |]] order.
    The converter reorders each head's query and key rows so that the rotation
    pairs adjacent features; the import restores the order the rotary schedule
    pairs ({!Kaun.Rope}).

    A block-format weight is decoded once, as it is read, by
    {!Nx_quant.dequant}: the model holds floats at the dtype it is read at, and
    its forward passes are those of a float model with the file's decoded
    values. *)

val config_of_gguf : Nx_io.Gguf.t -> Nx.host config
(** [config_of_gguf g] reads [g]'s [llama.*] metadata. The vocabulary is
    [llama.vocab_size], or else [token_embd]'s rows; the head is tied when [g]
    has no [output.weight]. The rotary schedule is the standard one at
    [llama.rope.freq_base], each pair's frequency divided by its entry of
    [rope_freqs.weight] where [g] has one, as Llama 3 files store their
    schedule.

    Raises [Failure] if [general.architecture] is not ["llama"], a required key
    is missing or of another type, [llama.rope.dimension_count] is not the head
    width, or [llama.rope.scaling.type] is present and not ["none"]. *)

val of_gguf :
  placement:(role -> axis:int -> 'd Nx.Placement.t) ->
  'c config ->
  (float, 'b) Nx.dtype ->
  Nx_io.Gguf.t ->
  ((float, 'b, 'd) Nx.t, 'd) params
(** [of_gguf ~placement cfg dt g] is the model of [g]'s weights at [dt]. An
    entry of a float type is cast to [dt] as {!Nx_io.Archive.float} casts; a
    Q8_0, Q4_K or Q6_K entry is [Nx_quant.dequant dt] of its blocks, each value
    rounded once to [dt]. Each leaf is placed as {!of_hf} places it.

    Raises [Failure], naming the entry, if one is missing, has another logical
    shape than [cfg] says, or is stored as another type. *)

val from_gguf :
  placement:(role -> axis:int -> 'd Nx.Placement.t) ->
  (float, 'b) Nx.dtype ->
  string ->
  Nx.host config * ((float, 'b, 'd) Nx.t, 'd) params
(** [from_gguf ~placement dt path] is the configuration and the parameters at
    [dt] of the GGUF file [path] ({!Nx_io.load_gguf}). *)

type dtype =
  | Dtype : (float, 'b) Nx.dtype -> dtype
      (** A floating-point dtype chosen at run time. *)

val dtype_of_string : string -> dtype
(** [dtype_of_string s] is the dtype named ["float32"], ["float16"] or
    ["bfloat16"]. Raises [Failure] on another name. *)

val stored_dtype : 'd Nx_io.Archive.t -> dtype
(** [stored_dtype weights] is the dtype [weights] stores its embedding table at,
    the dtype at which {!of_hf} casts nothing.

    Raises [Failure] if the table is missing or is not a float16, bfloat16 or
    float32 entry. *)

val default_repo : string
(** An ungated HuggingFace repository of Llama 3.2 1B whose weight file is
    byte-identical to Meta's gated one. *)

val from_pretrained :
  placement:(role -> axis:int -> 'd Nx.Placement.t) ->
  ?repo_id:string ->
  (float, 'b) Nx.dtype ->
  Nx.host config * ((float, 'b, 'd) Nx.t, 'd) params
(** [from_pretrained ~placement dt] downloads {!default_repo} (about 2.5 GB,
    cached afterwards), or [repo_id], and is its configuration and its
    parameters at [dt]. *)
