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
    directory if needed.

    The entry is written to a temporary file and atomically renamed into
    place, so concurrent readers and writers of the same key always observe
    a complete entry (the previous one or the new one), never a torn
    write. Failures are silently ignored: the cache is best-effort. *)
