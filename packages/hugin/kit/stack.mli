(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Stacks.

    A stack lays lengths end to end. Each length belongs to a {e column}, the x
    value of a bar or of the vertices of stacked areas, and to a {e series}, the
    category or layer it is part of. Columns and series are numbered from [0]:
    columns in the order of their x values, series in the order of their
    categories.

    Within a column, the lengths that are not negative are laid upward from [0.]
    and the negative ones downward from [0.], in increasing series number, so
    positive and negative lengths never cancel. The {e extent} of a column then
    runs from the end of its negative pile, [lo], to the end of its positive
    pile, [hi], with [lo <= 0. <= hi]. An {{!type-offset}offset} then moves or
    scales each column.

    {!intervals} gives each length the interval it covers. A figure draws a
    stacked bar from the start of its interval to the end, and a stacked area
    between the starts and the ends of a series. A length that is [nan] or
    infinite is {e missing}: it covers no interval, and its column is laid as if
    it were absent. *)

(** {1:stacks Stacks} *)

type offset =
  [ `Zero  (** Each column's piles start at [0.]. *)
  | `Expand
    (** Each column is scaled and moved so that its extent is \[[0];[1]\]:
        lengths become fractions of their column's extent. A column whose extent
        is [0.] to [0.] is left as it is. *)
  | `Center
    (** Each column is moved by [-. (lo +. hi) /. 2.], which centres its extent
        on [0.]. *) ]
(** The type for stack offsets. *)

val intervals :
  ?offset:offset ->
  columns:int array ->
  series:int array ->
  float array ->
  float array * float array
(** [intervals ~offset ~columns ~series lengths] is [(starts, ends)], where the
    length [lengths.(r)], in column [columns.(r)] and series [series.(r)],
    covers the interval from [starts.(r)] to [ends.(r)]. In its pile, its start
    is the end of the length laid before it, or [0.] for the first, and its end
    is its start plus itself, both then moved and scaled by [offset]. A pile
    lays its lengths by increasing series number, and the lengths of one series
    by increasing [r]. A missing length has [nan] as its start and end. Piles
    whose sums or extent exceed [max_float] give non-finite intervals. [offset]
    defaults to [`Zero].

    Raises [Invalid_argument] if [columns], [series] and [lengths] differ in
    length or if a column or series number is negative. *)
