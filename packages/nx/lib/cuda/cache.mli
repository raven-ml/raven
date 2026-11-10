(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What a device keeps from call to call: entries by key, newest first.

    Domains share a cache. A key's entry is found by structural equality.
    Two domains that make an entry for one key concurrently each keep
    theirs, and {!find} answers the newer. *)

type ('k, 'v) t
(** The type for caches of entries ['v] by keys ['k]. *)

val make : unit -> ('k, 'v) t
(** [make ()] is a cache of no entry. *)

val find : ('k, 'v) t -> 'k -> 'v option
(** [find c k] is [k]'s newest entry in [c]. *)

val find_by : ('k -> 'a -> bool) -> ('k, 'v) t -> 'a -> 'v option
(** [find_by f c x] is the newest entry of [c] whose key [k] has [f k x]. *)

val keep :
  ?count:int -> ?bytes:int * ('v -> int) -> ('k, 'v) t -> 'k -> 'v -> 'v
(** [keep ~count ~bytes:(most, size) c k v] is [v], kept in [c] as [k]'s
    entry, with the newest entries before it, at most [count] in all whose
    [size]s sum to at most [most]. A [v] whose [size] exceeds [most] is not
    kept, and [c] is unchanged. [count] defaults to [max_int] and must be
    positive; without [bytes], sizes are unbounded. *)
