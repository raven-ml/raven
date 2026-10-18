(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Ragged arrays.

    A ragged array is a sequence of rows of varying lengths: an array of
    {e values} cut along axis 0 into consecutive runs by int64 {e offsets}. Row
    [r] is [values] from [offsets.{r}] to [offsets.{r + 1}] along axis 0. A
    row's cells have the shape of [values] without its first axis: bytes for a
    string, token ids for a tokenized text, vectors for a sequence. Offsets and
    values lie on one device set, and each function computes there.

    Offsets need not start at [0], and values may hold rows before the first
    offset and after the last. This is Arrow's large list and large string
    layout, slices included, so {!sub} is O(1) and Arrow's buffers map without a
    copy (Arrow's int32 offsets widen once, in O(rows)).

    {b Checks and reads.} {!v} and {!of_lengths} state their invariant as a
    check ({!Nx.check}), which raises at the call on the host and at the next
    wait on the set elsewhere. {!take}, {!concat}, {!ids} and {!rank} read on
    the host the lengths that are data and size their work, and {!to_strings}
    reads the bytes it returns; each states its reads as a cost. A read waits
    for the set, and of a traced value raises [Invalid_argument] naming the
    function. Every other function reads nothing, so it takes traced values. *)

type ('v, 's, 'd) t
(** The type for ragged arrays of values of type ['v] stored as ['s] on the set
    ['d]. Its offsets are 1-D with at least one entry, never decrease, start at
    [0] or later and end at [dim 0 values] or earlier. *)

(** {1:make Ragged arrays} *)

val v : offsets:'d Nx.int64_t -> ('v, 's, 'd) Nx.t -> ('v, 's, 'd) t
(** [v ~offsets values] is the ragged array of [values] cut at [offsets].

    Raises [Invalid_argument] if [values] is 0-D, or if [offsets] is not 1-D
    with an entry; and, as a check, if [offsets] starts below [0], decreases, or
    ends past [dim 0 values]. *)

val of_lengths : 'd Nx.int64_t -> ('v, 's, 'd) Nx.t -> ('v, 's, 'd) t
(** [of_lengths lengths values] is the ragged array whose rows have [lengths],
    one after the other from the first row of [values]: its offsets are [0]
    followed by the running sum of [lengths].

    Raises [Invalid_argument] if [values] is 0-D, or if [lengths] is not 1-D;
    and, as a check, if [lengths] holds a negative length or sums past
    [dim 0 values] or [int64]'s range. *)

val of_ids :
  segments:int -> 'd Nx.int64_t -> ('v, 's, 'd) Nx.t -> ('v, 's, 'd) t
(** [of_ids ~segments ids x] groups the rows of [x] by [ids]: row [s] of the
    result holds the rows [i] of [x] with [ids.{i} = s], in their order in [x],
    for each [s] in \[[0], [segments]). A row whose id is outside that range is
    dropped: it is kept in the values, after the last offset. The result's
    values are all of [x]'s rows, permuted.

    Raises [Invalid_argument] if [segments] is negative, [x] is 0-D, or [ids] is
    not 1-D with [dim 0 x] entries. *)

val of_strings : string array -> (int, Nx.int8_unsigned_elt, Nx.host) t
(** [of_strings ss] is the ragged array on the host whose row [i] holds the
    bytes of [ss.(i)] as [uint8] values, from offset [0]. Every byte is kept as
    it is, NUL and bytes above 127 included: no text encoding is assumed. *)

(** {1:observe Observing} *)

val offsets : ('v, 's, 'd) t -> 'd Nx.int64_t
(** [offsets r] is [r]'s offsets, [length r + 1] of them. *)

val values : ('v, 's, 'd) t -> ('v, 's, 'd) Nx.t
(** [values r] is [r]'s values, including any rows outside its offsets. *)

val length : ('v, 's, 'd) t -> int
(** [length r] is the number of rows of [r]. *)

val lengths : ('v, 's, 'd) t -> 'd Nx.int64_t
(** [lengths r] is the length of each row of [r]. *)

val to_strings : (int, Nx.int8_unsigned_elt, 'd) t -> string array
(** [to_strings r] is the bytes of each row of [r] as a string:
    [to_strings (of_strings ss)] is [ss].

    Cost: two reads, of [r]'s offsets and then of the values from the first
    offset to the last.

    Raises [Invalid_argument] if [r]'s values are not 1-D. *)

(** {1:transform Transforming} *)

val sub : ('v, 's, 'd) t -> offset:int -> length:int -> ('v, 's, 'd) t
(** [sub r ~offset ~length] is rows [offset] to [offset + length - 1] of [r],
    over [r]'s values, in O(1).

    Raises [Invalid_argument] if [offset] or [length] is negative, or if
    [offset + length > length r]. *)

val take : indices:'d Nx.int64_t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [take ~indices r] is the ragged array whose row [j] is row [indices.{j}] of
    [r]. An index outside \[[0], [length r]) reads an empty row. The result's
    values are exactly its rows', from offset [0].

    Cost: one read, of the result's length.

    Raises [Invalid_argument] if [indices] is not 1-D, or if the rows' lengths
    sum past [int64]'s range. *)

val concat : ('v, 's, 'd) t list -> ('v, 's, 'd) t
(** [concat rs] is the rows of [rs] one after the other. A single ragged array
    is returned as it is; otherwise the result's values are exactly its rows',
    from offset [0].

    Cost: one read, of the first and last offsets of [rs].

    Raises [Invalid_argument] if [rs] is empty, or if the cells of [rs] differ
    in shape. *)

val map :
  (('v, 's, 'd) Nx.t -> ('w, 'r, 'd) Nx.t) -> ('v, 's, 'd) t -> ('w, 'r, 'd) t
(** [map f r] is [r] with values [f (values r)] and [r]'s offsets.

    Raises [Invalid_argument] if [f]'s result is 0-D or does not keep the extent
    of axis 0. *)

(** {1:summarize Summarizing rows}

    Rows compare element by element in nx's sort order ({!Nx.section-order}):
    every NaN equals every other, and [-0.] orders before [0.] and differs from
    it. A row orders before every row it is a prefix of, so byte strings order
    as [memcmp] orders them. Complex values have no order. *)

val quantile : float array -> (float, 's, 'd) t -> (float, 's, 'd) Nx.t
(** [quantile qs r] is [Nx.quantile qs] of each row of [r], over the row's
    elements: entry [(i, j)] is quantile [qs.(i)] of row [j], of shape
    [[|Array.length qs; length r|]]. An empty row's quantiles are NaN as the
    dtype stores it ({!Nx.Dtype.of_float}), [+0.] in float4_e2m1fn. One sort of
    the values by row, then value, serves every row and probability.

    Raises [Invalid_argument] if a probability is outside \[[0], [1]\] or NaN.
*)

val ids : ('v, 's, 'd) t -> 'd Nx.int64_t
(** [ids r] numbers the rows of [r] in order of first appearance: equal rows
    have one id, and the first row of each new value takes the next. Ids are
    relative to [r]: the ids of two ragged arrays compare only when computed
    over their {!concat}.

    Cost: {!rank}'s, and one sort of an integer per row ({!Nx.unique}).

    Raises [Invalid_argument] if [r]'s values are complex. *)

val rank : ('v, 's, 'd) t -> 'd Nx.int64_t
(** [rank r] is the dense rank of each row of [r]: the number of distinct rows
    that order before it.

    Cost: rows are told apart in rounds of up to 64 bytes of their elements'
    keys ({!Nx.order_key}, the narrowest as wide as the element), [⌈n w / 64⌉]
    rounds at most for a longest row of [n] keys of [w] bytes. A round
    {!Nx.lexsort}s the rows not yet told apart from every other by up to ten
    columns, one sort per column: their class, their next keys in up to eight
    words, and how many keys each holds in the round. One read of the rows'
    lengths comes before the rounds, and one of the rows left after each round
    but the last.

    Raises [Invalid_argument] if [r]'s values are complex. *)
