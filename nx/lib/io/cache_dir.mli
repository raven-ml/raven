(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Cache directory path resolution for Raven components. *)

(** {1:root Root directory} *)

val get_root : ?getenv:(string -> string option) -> unit -> string
(** [get_root ?getenv ()] is Raven's cache root directory.

    Resolution order is:
    - [RAVEN_CACHE_ROOT] if set and non-empty.
    - [XDG_CACHE_HOME/raven] via XDG defaults.

    [getenv] defaults to [Sys.getenv_opt]. *)

(** {1:paths Scoped paths} *)

val get_path_in_cache :
  ?getenv:(string -> string option) -> scope:string list -> string -> string
(** [get_path_in_cache ?getenv ~scope name] is the cache directory path for
    [scope @ [name]].

    The returned path always ends with a directory separator.

    [getenv] defaults to [Sys.getenv_opt]. *)
