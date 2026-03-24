(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Simple file-based disk cache.

    Stores marshalled OCaml values keyed by [(table, key)] pairs. Each entry
    is a separate file under the platform cache directory. Values are
    automatically invalidated when the cache version changes.

    Uses Marshal for serialization and individual files per key. *)

val get : table:string -> key:string -> 'a option
(** [get ~table ~key] retrieves a cached value, or [None] if the key is
    absent or the cache file is corrupt/stale. *)

val put : table:string -> key:string -> 'a -> unit
(** [put ~table ~key value] stores [value] in the cache. Creates the cache
    directory if needed. *)
