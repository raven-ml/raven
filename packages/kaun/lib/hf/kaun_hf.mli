(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** HuggingFace Hub integration.

    Fetches files from {{:https://huggingface.co}HuggingFace Hub} repositories
    into a local cache and loads safetensors checkpoints — single-file or
    sharded — as {!Kaun.Checkpoint.t} values. Since Hub checkpoints name and lay
    out tensors by the exporting framework's conventions, {!rename},
    {!transpose} and {!split} adapt them to a model's own scheme; typed
    parameters then come out through {!Kaun.Checkpoint.to_params}:

    {[
    let params =
      Kaun_hf.load_checkpoint "gpt2"
      |> remap_gpt2 (* rename entries, split fused projections, ... *)
      |> Checkpoint.to_params (module Model) ~like:template ~cast:true
    ]}

    Downloading requires [curl] on the [PATH]. Fetched files are cached under
    {!cache_path} and served from the cache on subsequent calls, so only the
    first access of a given file touches the network. *)

(** {1:fetching Fetching files} *)

val download_file :
  ?token:string ->
  ?cache_dir:string ->
  ?offline:bool ->
  ?revision:string ->
  file:string ->
  string ->
  string
(** [download_file ~file repo_id] is the local path to [file] from the Hub
    repository [repo_id] (e.g. ["gpt2"] or ["openai-community/gpt2"]), that is
    [cache_path ~file repo_id]. The file is downloaded on first access and
    served from the cache afterwards, with:

    - [token], a HuggingFace API token sent as a bearer token, for private
      repositories. Defaults to the value of [HF_TOKEN], if set.
    - [cache_dir], the cache root. Defaults as in {!cache_path}.
    - [offline], whether the network must not be touched. Defaults to [false].
      Cached files are returned either way; when [true] a missing file raises
      instead of downloading.
    - [revision], the branch name, tag, or commit hash to fetch from. Defaults
      to ["main"].

    Raises [Failure] if the download fails or if [offline] is [true] and the
    file is not cached. *)

(** {1:loading Loading models} *)

val load_config :
  ?token:string ->
  ?cache_dir:string ->
  ?offline:bool ->
  ?revision:string ->
  string ->
  Jsont.json
(** [load_config repo_id] is the parsed [config.json] of [repo_id]. Optional
    arguments are those of {!download_file}.

    Raises [Failure] on download or JSON parse errors. *)

val load_checkpoint :
  ?token:string ->
  ?cache_dir:string ->
  ?offline:bool ->
  ?revision:string ->
  string ->
  Kaun.Checkpoint.t
(** [load_checkpoint repo_id] is [repo_id]'s safetensors checkpoint. When the
    repository has a [model.safetensors.index.json] index, all shards it
    references are fetched and their entries merged; otherwise the single
    [model.safetensors] file is fetched. Entry names are the raw safetensors
    keys (e.g. ["h.0.attn.c_attn.weight"]); adapt them with {!rename},
    {!transpose} and {!split}. Optional arguments are those of {!download_file}.

    Raises [Failure] if the repository has neither an index nor a
    [model.safetensors] file, if an indexed tensor is missing from its shard, or
    on download or parse errors. *)

(** {1:adapting Adapting foreign checkpoints}

    Checkpoint-to-checkpoint transformations for mapping a foreign checkpoint
    onto a model's own entry names and tensor layouts. Compose them with [(|>)];
    entries left over after adaptation are harmless, since
    {!Kaun.Checkpoint.to_params} ignores entries its template does not name. *)

val rename : (string -> string) -> Kaun.Checkpoint.t -> Kaun.Checkpoint.t
(** [rename f t] is [t] with every entry name [n] replaced by [f n]. Names the
    model does not care about can be left alone by returning them unchanged.

    Raises [Invalid_argument] if the new names are not distinct and non-empty.
*)

val transpose : string -> Kaun.Checkpoint.t -> Kaun.Checkpoint.t
(** [transpose name t] is [t] with the last two axes of entry [name] swapped.
    Use it on weights stored with the opposite orientation, such as
    [torch.nn.Linear]'s [outputs × inputs] weights when the model expects
    [inputs × outputs].

    Raises [Invalid_argument] if [name] has no entry or its entry has fewer than
    2 axes. *)

val split :
  ?axis:int ->
  string ->
  into:string list ->
  Kaun.Checkpoint.t ->
  Kaun.Checkpoint.t
(** [split name ~into t] is [t] with entry [name] replaced by [List.length into]
    entries, the equal sections of the original tensor along [axis], in order.
    Use it on fused projections, such as GPT-2's [c_attn] weight whose columns
    are the concatenated query, key and value projections. [axis] defaults to
    [-1], the last axis; negative values count from the last axis.

    Raises [Invalid_argument] if [name] has no entry, [into] is empty, [axis] is
    out of bounds, the size of [axis] is not a multiple of [List.length into],
    or the resulting names are not distinct and non-empty. *)

(** {1:cache The cache} *)

val cache_path :
  ?cache_dir:string -> ?revision:string -> file:string -> string -> string
(** [cache_path ~file repo_id] is the local path where {!download_file} caches
    [file] from [repo_id]: [cache_dir/repo/revision/file], where [repo] is
    [repo_id] with ["/"] replaced by ["-"]. [revision] defaults to ["main"].
    [cache_dir] defaults to [$RAVEN_CACHE_ROOT/huggingface] when
    [RAVEN_CACHE_ROOT] is set, and [$XDG_CACHE_HOME/raven/huggingface] otherwise
    ([XDG_CACHE_HOME] itself defaulting to [$HOME/.cache]). The path is
    computed, not touched: the file may not exist. *)

val clear_cache : ?cache_dir:string -> ?repo_id:string -> unit -> unit
(** [clear_cache ()] removes every cached file under [cache_dir] (defaulting as
    in {!cache_path}). When [repo_id] is given, only that repository's files are
    removed. *)
