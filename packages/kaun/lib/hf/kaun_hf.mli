(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** HuggingFace Hub integration.

    Fetches files from {{:https://huggingface.co}HuggingFace Hub} repositories
    into a local cache and loads safetensors checkpoints, single-file or
    sharded, as {!Kaun.Checkpoint.t} values. A Hub checkpoint names and lays out
    its tensors by the conventions of the framework that exported it, so a
    model's importer asks for each entry by that name and reshapes it with nx
    (see {!Kaun.Checkpoint.to_float}):

    {[
    let ckpt = Kaun_hf.load_checkpoint "gpt2" in
    let table =
      Checkpoint.to_float ~shape:[| 50257; 768 |] Nx.float32 "wte.weight" ckpt
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
    served from the cache afterwards. A download is written to a temporary file
    beside the cache path and renamed once complete, so the cache path never
    holds a partial file, and two processes that fetch the same file each write
    their own. Optional arguments:

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
    keys (e.g. ["h.0.attn.c_attn.weight"]). Loading reads headers only; the
    entries are views of the cached files (see {!Kaun.Checkpoint.load}).
    Optional arguments are those of {!download_file}.

    Raises [Failure] if the repository has neither an index nor a
    [model.safetensors] file, if an indexed tensor is missing from its shard, or
    on download or parse errors. *)

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
    removed. If a file cannot be removed, which happens on platforms that lock a
    file while tensors loaded from it are alive, a major collection runs and the
    removal is retried once.

    Raises [Sys_error] or [Unix.Unix_error] if a file still cannot be removed.
*)
