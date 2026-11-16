(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Entries by key in cells, whose bytes together stay within a budget.

    A device keeps its submissions so, a cell for each first kernel: a
    lookup reads one cell's few keys, and the budget bounds the memory of
    all of them. Every {!find_by} that answers an entry and every {!keep}
    uses it, and a {!keep} past the budget drops the entries least recently
    used, of any cell.

    Domains share a value. A find reads the cells as some keep left them,
    and a keep replaces them in one step, so a keep's drops are those of
    the cells as it found them: a find that runs while a keep does counts
    as used before it or after it. Two keeps of one key, concurrent or not,
    each keep an entry, and a find answers the newer. *)

type ('k, 'v) t
(** The type for kept entries ['v] by keys ['k]. *)

val make : cells:int -> count:int -> ('k, 'v) t
(** [make ~cells ~count] keeps no entry, in [cells] cells, at most [count]
    entries in each.

    Raises [Invalid_argument] if [cells] is negative or [count] is not
    positive. *)

val find_by : ('k -> 'a -> bool) -> ('k, 'v) t -> int -> 'a -> 'v option
(** [find_by f t i x] is the newest entry of cell [i] whose key [k] has
    [f k x], which it uses.

    Raises [Invalid_argument] if [t] has no cell [i]. *)

val keep : ('k, 'v) t -> int -> most:int -> bytes:int -> 'k -> 'v -> unit
(** [keep t i ~most ~bytes k v] keeps [v], of [bytes] bytes, as [k]'s entry
    in cell [i], unless [bytes] exceeds [most], which leaves [t] unchanged.
    Before it, it drops cell [i]'s least recently used entry where the cell
    holds [count] entries, then the least recently used entries of every
    cell while the kept bytes and [bytes] exceed [most].

    Raises [Invalid_argument] if [t] has no cell [i] or [bytes] is
    negative. *)

val clear : ('k, 'v) t -> unit
(** [clear t] drops every entry of [t]. *)

val count : ('k, 'v) t -> int
(** [count t] is the number of entries [t] keeps. *)

val bytes : ('k, 'v) t -> int
(** [bytes t] is the sum of the bytes of the entries [t] keeps. *)
