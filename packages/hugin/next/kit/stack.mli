(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Stacks.

    A stack lays lengths end to end. Each length belongs to a {e column}, the x
    value of a bar or of the vertices of stacked areas, and to a {e series}, the
    category or layer it is part of. Columns and series are numbered from [0]:
    columns in the order of their x values, series in the order of their
    categories. A stack has the columns from [0] to the greatest column number
    it is given, and the series likewise; a number given no length is an empty
    column or series. Consecutive column numbers are neighbours, so [`Wiggle]
    stacks one sequence of columns per call: a figure stacks each panel on its
    own.

    Within a column, the lengths that are not negative are laid upward from [0.]
    and the negative ones downward from [0.], in the order of their series
    ({!type-order}), so positive and negative lengths never cancel. The
    {e extent} of a column then runs from the end of its negative pile, [lo], to
    the end of its positive pile, [hi], with [lo <= 0. <= hi]. An
    {{!type-offset}offset} then moves or scales each column.

    {!intervals} gives each length the interval it covers. A figure draws a
    stacked bar from the start of its interval to the end, and a stacked area
    between the starts and the ends of a series. A length that is [nan] or
    infinite is {e missing}: it covers no interval, and its column is laid as if
    it were absent.

    Reference: Lee Byron and Martin Wattenberg.
    {e Stacked graphs: geometry and aesthetics}. IEEE Transactions on
    Visualization and Computer Graphics 14(6), 2008. *)

(** {1:stacks Stacks} *)

type offset =
  [ `Zero  (** Each column's piles start at [0.]. *)
  | `Expand
    (** Each column is scaled and moved so that its extent is \[[0];[1]\]:
        lengths become fractions of their column's extent. A column whose extent
        is [0.] to [0.] is left as it is. *)
  | `Center
    (** Each column is moved by [-. (lo +. hi) /. 2.], which centres its extent
        on [0.]. *)
  | `Wiggle
    (** Each column [c] is moved by [g c], chosen so that series change as
        little as they can from column to column. With [f i c] the sum of the
        lengths of series [i] in column [c], missing and absent ones counting
        [0.]:
        - [g 0] is [0.];
        - [g c] is [g (c - 1) -. a /. b], or [g (c - 1)] if [b] is [0.], where
          [b] is the sum of the weights [w i = |f i c|] over the series and [a]
          the sum of [w i × (Σ d j + d i / 2)], [j] running over the series
          before [i] in the order, whatever the signs of their lengths, and
          [d i] being [f i c -. f i (c - 1)];
        - every [g c] then gains one constant, which centres on [0.] the extent
          of the whole stack, from the least [lo +. g c] to the greatest
          [hi +. g c] over the columns that hold a length not missing.

        For lengths that are not negative, [g] is the baseline of Byron and
        Wattenberg that minimises the layers' change from column to column
        weighted by their thickness. A streamgraph is [`Wiggle] with the order
        [`Inside_out]. *) ]
(** The type for stack offsets. *)

type order =
  [ `Given  (** Series [0] first, then series [1], and so on. *)
  | `Reverse  (** The last series first, then the one before, and so on. *)
  | `Ascending
    (** By increasing sum of a series' lengths, missing ones counting [0.]. *)
  | `Descending  (** By decreasing sum of a series' lengths. *)
  | `Appearance
    (** By increasing {e peak}, the first column in which the sum of a series'
        lengths is greatest, missing and absent ones counting [0.]. *)
  | `Inside_out
    (** The series in [`Appearance] order are dealt in turn to two groups, each
        to the group whose sum of lengths so far is less, the lower group on
        ties. The lower group is laid first, from its last series to its first,
        then the upper group from its first to its last, so the series that peak
        early sit in the middle of the stack. *) ]
(** The type for stacking orders: the order in which the series of a column are
    laid from [0.]. Series that an order ranks equal keep their numbering order.
*)

val intervals :
  ?offset:offset ->
  ?order:order ->
  columns:int array ->
  series:int array ->
  float array ->
  float array * float array
(** [intervals ~offset ~order ~columns ~series lengths] is [(starts, ends)],
    where the length [lengths.(r)], in column [columns.(r)] and series
    [series.(r)], covers the interval from [starts.(r)] to [ends.(r)]. In its
    pile, its start is the end of the length laid before it, or [0.] for the
    first, and its end is its start plus itself, both then moved and scaled by
    [offset]. A pile lays its lengths in the order of their series under
    [order], and the lengths of one series in increasing [r]. A missing length
    has [nan] as its start and end. Piles whose sums or extent exceed
    [max_float] give non-finite intervals. [offset] defaults to [`Zero] and
    [order] to [`Given].

    Raises [Invalid_argument] if [columns], [series] and [lengths] differ in
    length or if a column or series number is negative. *)
