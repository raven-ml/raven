(** Cache directory utilities for the Raven ecosystem. *)

val get_root : ?getenv:(string -> string option) -> unit -> string
(** [get_root ?getenv ()] returns the base cache directory for Raven.

    The cache directory is resolved using the following priority order: 1.
    [RAVEN_CACHE_ROOT] environment variable (highest priority; absolute cache
    root) 2. [XDG_CACHE_HOME] environment variable (if RAVEN_CACHE_ROOT not set)
    3. [$HOME/.cache] (fallback, default behavior)

    The resolved path will be [RAVEN_CACHE_ROOT] or
    "[XDG_CACHE_HOME or HOME/.cache]/raven".

    @param getenv
      optional environment lookup function (defaults to [Sys.getenv_opt]) to
      facilitate testing.

    {2 Environment Variables}
    - [RAVEN_CACHE_ROOT]: Custom cache directory root (overrides all other
      settings)
    - [XDG_CACHE_HOME]: XDG Base Directory cache location (standard on
      Linux/Unix)
    - [HOME]: User home directory (used for fallback cache location) *)

val get_path_in_cache :
  ?getenv:(string -> string option) -> scope:string list -> string -> string
(** [get_path_in_cache ?getenv ~scope name] returns the cache directory path for
    a specific component.

    {2 Parameters}
    - [scope]: list of directory names forming the scope (e.g. [["datasets"]],
      [["models"; "bert"]])
    - [name]: the specific name within that scope (e.g. "iris", "gpt2")

    {2 Returns}
    - the cache directory path, including trailing slash.

    @param getenv
      optional environment lookup function (defaults to [Sys.getenv_opt]) to
      facilitate testing.

    {2 Examples}

    Getting cache directory for the iris dataset:
    {[
      let cache_dir = Nx_core.Cache_dir.get_path_in_cache ~scope:["datasets"] "iris" in
      (* With default environment: ~/.cache/raven/datasets/iris/ *)
    ]}

    Getting cache directory with custom root:
    {[
      let getenv var =
        if var = "RAVEN_CACHE_ROOT" then Some "/tmp/my-cache" else None
      in
      let cache_dir =
        Nx_core.Cache_dir.get_path_in_cache ~getenv ~scope:["models"] "bert-base-uncased"
      in
      (* Result: /tmp/my-cache/models/bert-base-uncased/ *)
    ]} *)
