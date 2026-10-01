(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Keys: columns as rows of words.

    A key is a row of [uint64] words that nx's [unique], [lexsort] and
    [searchsorted] compare. Every operation of talon that compares values
    (grouping, sorting, joining, partitioning, [n_unique], [is_in], [min] and
    [max], every comparison of [Expr], and [Talon_next.equal]) reads its words
    here, so key identity and the total order have one implementation.

    {b One word per value.} A column's value word, zero under a null, is:
    - for a number, a boolean, a temporal value, a decimal or a categorical
      code: [Nx.order_key uint64] of its value, a float with [-0.] made [0.]
      first, so that every NaN is one word, the greatest;
    - for a compound value (text, a byte string, a tensor, a record, a list):
      the code of its row of element words, [Nx_ragged.ids] of the rows for
      identity and [Nx_ragged.rank] for order. Text's elements are its bytes; a
      tensor's are its elements' words, row-major; a record's are each field's
      null flag and value word; a list's, each element's;
    - for an extension: its storage's.

    A key of [k] columns therefore has at most [2k] words: a null flag word for
    each column with a null, and its value word.

    {b Codes are relative.} Ids and ranks number the rows of the column they are
    computed over: the words of two compound columns compare only when computed
    over their concatenation ({!Column.concat}). Fixed-width words are absolute.
*)

type use =
  | Identity  (** Words equal iff the values are the same key. *)
  | Order  (** Words ordered as talon's total order orders the values. *)

val value : use -> Column.t -> Nx.uint64_t
(** [value u c] is the value word of each row of [c], zero under a null. Its
    unsigned order is talon's total order for [Order]. For a fixed-width column
    the two uses give the same words. *)

val identity : Column.t list -> Nx.uint64_t
(** [identity cs] is the [[n; w]] matrix of the keys of the rows of [cs],
    columns of [n] rows: two rows are equal iff their values are the same keys
    in every column, by key identity ({!Type.compare_value}, null being one more
    key, every NaN one key and [-0.] the key of [0.]).

    Raises [Invalid_argument] if [cs] is empty. *)

val order : (Column.t * Order.t) list -> Nx.uint64_t
(** [order ks] is the [[n; w]] matrix whose rows, compared as unsigned words in
    lexicographic order, order as the keys [ks] order their columns: each column
    by talon's total order, its value word complemented for a descending key,
    after its null flag, [1] for a null (nulls last) or [0] with
    [Order.nulls_first]. Only each key's direction and null placement are read,
    not its name.

    Raises [Invalid_argument] if [ks] is empty. *)
