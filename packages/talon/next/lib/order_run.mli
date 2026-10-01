(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Ordering batches.

    The lowerings of [sort], of a slice from the start over a [sort], and of the
    check of a source's stated order. Each reads its keys' words from
    {!Key.order}, so every order of talon is {!Type.compare_value}'s. *)

val sort : Order.t list -> Table.t -> Table.t
(** [sort ks b] is the one-batch table [b]'s rows ordered by [ks], stably: the
    permutation {!Nx.lexsort} gives, applied to every column with
    {!Column.permute}. *)

val top_k : offset:int -> length:int -> Order.t list -> Table.t -> Table.t
(** [top_k ~offset ~length ks b] is [sort ks b]'s rows [offset] to
    [offset + length - 1] that it has, [offset >= 0]. Past [offset + length]
    rows, only the rows whose first key word is at most the [offset + length]th
    smallest are sorted: {!Nx.top_k} finds that word in passes over the rows,
    and ties on it make the rows sorted more. *)

val unordered : Order.t list -> Table.t option -> Table.t -> int option
(** [unordered ks prev b] is the first row of the one-batch table [b] that
    orders before the row before it by [ks]; the row before [b]'s first is the
    last of [prev], a one-batch table with rows, if given. *)
