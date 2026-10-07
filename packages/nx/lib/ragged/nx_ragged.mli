(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Ragged arrays.

    A ragged array is a sequence of rows of varying lengths: a tensor of
    {e values} cut along axis 0 into consecutive runs by int64 {e offsets}. Row
    [r] is [values] from [offsets.{r}] to [offsets.{r + 1}] along axis 0. A
    row's cells have the shape of [values] without its first axis: bytes for a
    string, token ids for a tokenized text, vectors for a sequence.

    Offsets need not start at [0], and values may hold rows before the first
    offset and after the last. This is Arrow's large list and large string
    layout, slices included, so {!sub} is O(1) and Arrow's buffers map without a
    copy (Arrow's int32 offsets widen once, in O(rows)).

    The library [nx.ragged] is built from nx's operations. Offsets and values
    are placed as any operands are, and values differentiate through {!take} and
    {!of_ids}.

    {b Reads.} {!v}, {!of_lengths}, {!to_strings}, {!take}, {!concat}, {!ids}
    and {!rank} read values: [v] and [of_lengths] whether their invariant holds,
    [to_strings] its offsets and then its bytes, [take] the length of its
    result, [concat] the bounds of its operands, and [ids] and [rank] what each
    of their rounds needs. Each reads once, [to_strings] twice, and [ids] and
    [rank] once per round. Under a transformation that traces, such as a
    compiled function or a mapped one, a read raises naming the function. Every
    other operation reads nothing. *)

type ('a, 'b) t
(** The type for ragged arrays of values of type ['a] stored as ['b]. Its
    offsets are 1-D with at least one entry, never decrease, start at [0] or
    later and end at [dim 0 values] or earlier. *)

(** {1:make Ragged arrays} *)

val v : offsets:Nx.int64_t -> ('a, 'b) Nx.t -> ('a, 'b) t
(** [v ~offsets values] is the ragged array of [values] cut at [offsets].

    Raises [Invalid_argument] if [values] is a scalar, or if [offsets] is not
    1-D with an entry, starts below [0], decreases, or ends past [dim 0 values].
*)

val of_lengths : Nx.int64_t -> ('a, 'b) Nx.t -> ('a, 'b) t
(** [of_lengths lengths values] is the ragged array whose rows have [lengths],
    one after the other from the first row of [values]: its offsets are [0]
    followed by the running sum of [lengths].

    Raises [Invalid_argument] if [values] is a scalar, or if [lengths] is not
    1-D, holds a negative length, or sums past [dim 0 values]. *)

val of_ids : segments:int -> Nx.int64_t -> ('a, 'b) Nx.t -> ('a, 'b) t
(** [of_ids ~segments ids x] groups the rows of [x] by [ids]: row [s] of the
    result holds the rows [i] of [x] with [ids.{i} = s], in their order in [x],
    for each [s] in \[[0], [segments]). A row whose id is outside that range is
    dropped: it is kept in the values, after the last offset. The result's
    values are all of [x]'s rows, permuted.

    It reads nothing, so it compiles.

    Raises [Invalid_argument] if [segments] is negative, [x] is a scalar, or
    [ids] is not 1-D with [dim 0 x] entries. *)

val of_strings : string array -> (int, Nx.uint8_elt) t
(** [of_strings ss] is the ragged array whose row [i] holds the bytes of
    [ss.(i)] as [uint8] values, from offset [0]. Every byte is kept as it is,
    NUL and bytes above 127 included: no text encoding is assumed. *)

(** {1:observe Observing} *)

val offsets : ('a, 'b) t -> Nx.int64_t
(** [offsets r] is [r]'s offsets, [length r + 1] of them. *)

val values : ('a, 'b) t -> ('a, 'b) Nx.t
(** [values r] is [r]'s values, including any rows outside its offsets. *)

val length : ('a, 'b) t -> int
(** [length r] is the number of rows of [r]. *)

val lengths : ('a, 'b) t -> Nx.int64_t
(** [lengths r] is the length of each row of [r]. *)

val to_strings : (int, Nx.uint8_elt) t -> string array
(** [to_strings r] is the bytes of each row of [r] as a string:
    [to_strings (of_strings ss)] is [ss].

    It reads [r]'s offsets, then the values from the first offset to the last.

    Raises [Invalid_argument] if [r]'s values are not 1-D. *)

(** {1:transform Transforming} *)

val sub : ('a, 'b) t -> offset:int -> length:int -> ('a, 'b) t
(** [sub r ~offset ~length] is rows [offset] to [offset + length - 1] of [r],
    over [r]'s values, in O(1).

    Raises [Invalid_argument] if [offset] or [length] is negative, or if
    [offset + length > length r]. *)

val take : indices:Nx.int64_t -> ('a, 'b) t -> ('a, 'b) t
(** [take ~indices r] is the ragged array whose row [j] is row [indices.{j}] of
    [r]. An index outside \[[0], [length r]) reads an empty row. The result's
    values are exactly its rows', from offset [0].

    It reads the length of its result once.

    Raises [Invalid_argument] if [indices] is not 1-D, or if the rows' lengths
    sum past [int64]'s range. *)

val concat : ('a, 'b) t list -> ('a, 'b) t
(** [concat rs] is the rows of [rs] one after the other. A single ragged array
    is returned as it is; otherwise the result's values are exactly its rows',
    from offset [0].

    It reads the first and last offsets of [rs] once.

    Raises [Invalid_argument] if [rs] is empty, or if the cells of [rs] differ
    in dtype or shape. *)

val map : (('a, 'b) Nx.t -> ('c, 'd) Nx.t) -> ('a, 'b) t -> ('c, 'd) t
(** [map f r] is [r] with values [f (values r)] and [r]'s offsets.

    Raises [Invalid_argument] if [f]'s result is a scalar or does not keep the
    extent of axis 0. *)

(** {1:summarize Summarizing rows} *)

val quantile : float array -> (float, 'b) t -> (float, 'b) Nx.t
(** [quantile qs r] is [Nx.quantile qs] of each row of [r], over the row's
    elements: entry [(i, j)] is quantile [qs.(i)] of row [j], of shape
    [[|Array.length qs; length r|]]. An empty row's quantiles are NaN.

    One sort of the values by row, then value, serves every row and probability.
    It reads nothing, so it compiles.

    Raises [Invalid_argument] if a probability is outside \[[0], [1]\] or NaN.
*)

val ids : ('a, 'b) t -> Nx.int64_t
(** [ids r] numbers the rows of [r] in order of first appearance: equal rows
    have one id, and the first row of each new value takes the next. Rows are
    equal when their elements are, in the sort order of [Nx.sort]: every NaN
    equals every other, and [-0.] differs from [0.].

    Ids are relative to [r]: the ids of two ragged arrays compare only when
    computed over their {!concat}.

    Raises [Invalid_argument] if [r]'s values are complex. *)

val rank : ('a, 'b) t -> Nx.int64_t
(** [rank r] is the dense rank of each row of [r]: the number of distinct rows
    that order before it. Rows order element by element in the sort order of
    [Nx.sort], a row before every row it is a prefix of, so byte strings order
    as [memcmp] orders them.

    Raises [Invalid_argument] if [r]'s values are complex. *)
