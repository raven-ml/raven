(** Kaun_huggingface: HuggingFace model hub integration for Kaun.

    Provides seamless loading of pretrained models from HuggingFace, with
    automatic downloading, caching, and type-safe model definitions. *)

(** {1 Types} *)

type model_id = string
(** HuggingFace model identifier (e.g., "gpt2", "bert-base-uncased") *)

type revision =
  | Latest
  | Tag of string
  | Commit of string  (** Model revision specification *)

type cache_dir = string
(** Local cache directory for downloaded models *)

type download_progress = {
  downloaded_bytes : int;
  total_bytes : int option;
  rate : float; (* bytes/sec *)
}

type 'a download_result = Cached of 'a | Downloaded of 'a * download_progress

(** {1 Configuration} *)

module Config : sig
  type t = {
    cache_dir : cache_dir;
        (** Directory to cache downloaded models. Default:
            ~/.cache/kaun/huggingface *)
    token : string option;
        (** Optional HuggingFace API token for private models *)
    offline_mode : bool;  (** If true, only use cached models, don't download *)
    force_download : bool;  (** If true, re-download even if cached *)
    show_progress : bool;  (** If true, show download progress *)
  }

  val default : t
  (** Default configuration with sensible defaults *)

  val from_env : unit -> t
  (** Load configuration from environment variables:
      - KAUN_HF_CACHE_DIR
      - KAUN_HF_TOKEN
      - KAUN_HF_OFFLINE_MODE *)
end

(** {1 Model Registry} *)

module Registry : sig
  (** Registry of known model architectures with their loading functions *)

  type ('params, 'a, 'dev) model_spec = {
    architecture : string;  (** Architecture name (e.g., "GPT2", "BERT") *)
    config_file : string;  (** Config filename (e.g., "config.json") *)
    weight_files : string list;
        (** Weight filenames to try (e.g.,
            ["model.safetensors", "pytorch_model.bin"]) *)
    load_config : Yojson.Safe.t -> 'params;
        (** Parse config JSON into model parameters *)
    build_params : dtype:(float, 'a) Rune.dtype -> 'params -> Kaun.params;
        (** Build parameter tree from config *)
  }

  val register : string -> ('params, 'a, 'dev) model_spec -> unit
  (** Register a model architecture *)

  val get : string -> ('params, 'a, 'dev) model_spec option
  (** Get a registered model spec *)
end

(** {1 Core Loading Functions} *)

val download_file :
  ?config:Config.t ->
  ?revision:revision ->
  model_id:model_id ->
  filename:string ->
  unit ->
  string download_result
(** [download_file ~model_id ~filename ()] downloads a single file from
    HuggingFace. Returns the local path to the file (either cached or newly
    downloaded). *)

val load_safetensors :
  ?config:Config.t ->
  ?revision:revision ->
  model_id:model_id ->
  unit ->
  Kaun.params download_result
(** [load_safetensors ~model_id ~dtype ()] downloads and loads safetensors
    weights. Automatically tries common filenames like "model.safetensors". *)

val load_config :
  ?config:Config.t ->
  ?revision:revision ->
  model_id:model_id ->
  unit ->
  Yojson.Safe.t download_result
(** [load_config ~model_id ()] downloads and parses the model's config.json *)

(** {1 High-level Model Loading} *)

val from_pretrained :
  ?config:Config.t ->
  ?revision:revision ->
  model_id:model_id ->
  unit ->
  Kaun.params
(** [from_pretrained ~model_id ~dtype ()] loads a complete model.

    This is the main entry point for loading models. It: 1. Downloads the model
    configuration 2. Downloads the model weights 3. Loads and returns the
    parameter tree

    @raise Failure if model architecture is unknown or download fails

    Example:
    {[
      let gpt2_params =
        Kaun_huggingface.from_pretrained ~model_id:"gpt2" ~dtype:Rune.Float32 ()
    ]} *)

(** {1 Utilities} *)

val list_cached_models : ?config:Config.t -> unit -> model_id list
(** List all models in the local cache *)

val clear_cache : ?config:Config.t -> ?model_id:model_id -> unit -> unit
(** Clear the cache for a specific model or all models *)

val get_model_info : model_id -> (Yojson.Safe.t, string) result
(** Fetch model card/info from HuggingFace API *)
