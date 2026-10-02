(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Axis ticks.

    The ticks of an axis, a colour bar or a facet header are guide values of the
    scale it shows, at their normalised positions, each with a label; minor
    ticks between them; and possibly a {e note} that every label is read
    against, such as a shared power of ten. {!choose} picks the ticks of an axis
    of a given length so that their labels read well, measuring the labels of
    every candidate with a function the caller gives, since the kit measures no
    text. {!of_values} labels guide values the caller picked.

    Labels are a function of the values they label, the scale, and the notation
    and locale asked for ({!section-labels}), so the ticks {!choose} picks and
    the same values given to {!of_values} have the same labels.

    Reference: Justin Talbot, Sharon Lin and Pat Hanrahan.
    {e An Extension of Wilkinson's Algorithm for Positioning Tick Labels on
       Axes}. IEEE Transactions on Visualization and Computer Graphics 16(6),
    2010. *)

(** {1:ticks Ticks} *)

type tick = {
  position : float;  (** The tick's value normalised by its scale. *)
  label : string;  (** The tick's label ({!section-labels}). *)
  context : string option;
      (** On a temporal scale, the second line of the tick's label, naming the
          coarser part of its instant: set on the first tick and on each tick
          whose context differs from the previous tick's, and [None] on the
          others and on every other scale ({!section-time}). *)
}
(** The type for labelled ticks. *)

type t = private {
  major : tick list;
      (** The labelled ticks, in increasing order of their positions. *)
  minor : float list;
      (** The positions of the unlabelled ticks, increasing ({!section-minor}).
      *)
  note : string option;
      (** What every label leaves out, written once for the axis: a power of ten
          the labels are multiplied by, a value added to them, or both
          ({!section-quantities}). *)
}
(** The type for the ticks of an axis. Every position is in \[[0];[1]\], and no
    minor position is a major one. *)

(** {1:making Making ticks} *)

val choose :
  ?locale:Locale.t ->
  ?notation:Number.notation ->
  length:float ->
  measure:(string -> float) ->
  'd Scale.t ->
  t
(** [choose ~locale ~notation ~length ~measure s] is the most legible ticks of
    [s] for an axis [length] long on which the normalised values [0] and [1] lie
    at the ends, where [measure l] is the extent that label [l] needs along the
    axis, in the unit of [length], including the clearance it wants from its
    neighbours. The major ticks are the values of the best
    {{!section-candidates}candidate}, labelled as {!of_values} labels them with
    [locale] and [notation], and the minor ticks are the candidate's
    ({!section-minor}).

    Every major tick lies in the domain of [s]: [choose] never widens a domain.
    A {{!Scale.section-continuous}constant} domain has one tick, and a band
    scale without categories none, nor a scale that normalises every value of
    its domain to [nan], such as an unfitted logit ({!Scale.custom}).

    [measure] must be pure. [choose] calls it at most once per distinct string,
    and takes the extent of a temporal tick as the greater of the extents of its
    label and its context.

    Raises [Invalid_argument] if [length] is not finite and positive, [measure]
    returns a value that is not finite and positive, or [notation] is given and
    [s] is not quantitative. *)

val of_values :
  ?locale:Locale.t -> ?notation:Number.notation -> 'd Scale.t -> 'd array -> t
(** [of_values ~locale ~notation s vs] is the ticks at the values of [vs] that
    lie in the domain of [s] and are not missing for it, each once, labelled
    with the strings of [locale], which defaults to {!Locale.default}, and in
    [notation] if given ({!section-labels}). It has no minor ticks.

    Raises [Invalid_argument] if [notation] is given and [s] is not
    quantitative. *)

(** {2:candidates Candidates}

    A candidate is a set of guide values inside the domain, drawn from one of
    these families, each with a skip [j >= 1] that keeps every [j]th value, from
    an offset [r] in \[[0];[j - 1]\]:
    - {b Decimal steps}, on linear, pow, custom, symlog and log scales: the
      multiples inside the domain of a step [q × 10^z], for [q] in [1], [5],
      [2], [2.5], [4] and [3], in this order of preference, and an integer [z].
      A multiple is the float nearest its integer index times the step, as nice
      domains compute them ({!Scale.section-nice}), so [3 × 0.1] is [0.3]. Steps
      below a 32nd of the distance from each end of the domain to the next float
      towards zero are left out. A step finer than the floats gives ticks at the
      floats its multiples round to, labelled by the decimals of those floats:
      on \[[1e17];[1e17 + 32]\], whose floats are 16 apart, the step [10] gives
      ticks at the positions [0], [0.5] and [1] that read [0], [20] and [30]
      against the note [+10¹⁷].
    - {b Powers}, on log scales of base [b]: the integer powers [b^i] inside the
      domain; for [j = 1] also the powers with their multiples by [2] and [5],
      in base [10], and with their multiples by every integer from [2] to
      [b - 1], in an {{!Scale.log}integer base}. In base [10], [10^i] is the
      float nearest it and [n × 10^i] a decimal multiple; in another base [b^i]
      is [Float.pow b (float i)] and [n × b^i] is
      [float n *. Float.pow b (float i)].
    - {b Signed powers}, on symlog scales of constant [c]: [0.] if it is inside
      the domain, and the powers of ten and their negations of magnitude at
      least [c] inside the domain.
    - {b Calendar intervals}, on temporal scales: the boundaries inside the
      domain, at the scale's offset, of one of the
      {{!Scale.section-time_intervals}time intervals}.
    - {b Strides}, on band scales: every [k]th category from the first, for [k]
      in [1], [2], [5], [10], [20], [50], ….

    A candidate is discarded if two adjacent labels, centred on their ticks,
    overlap: [(p_(k+1) - p_k) × length < (e_k + e_(k+1)) / 2] for their
    positions [p_k] and extents [e_k]. A candidate of one tick never overlaps,
    and every family has one on a domain that is not constant and whose values
    normalise, so a candidate is always chosen.

    On a band scale the candidate is the stride of the least [k] that gives at
    most a hundred ticks and whose labels do not overlap: a reader cannot place
    a category between two labels. Otherwise the candidate chosen balances, as
    Talbot, Lin and Hanrahan score it, its {e simplicity} (a preferred step, a
    small skip, and [0.] among the ticks, [1.] on a log scale), its {e coverage}
    of the domain by the span of its ticks, and its {e density}, near the number
    of ticks at which labels fill half the axis, but no more than a hundred,
    which a reader cannot take in on one axis. Those labels are the labels of
    about ten values of the scale's family inside the domain:
    - on linear, pow and custom scales, the multiples of the decimal step for a
      tenth of the domain's length, the step its nice domain rounds to
      ({!Scale.section-nice});
    - on log scales of base [b], in an {{!Scale.log}integer base} over fewer
      than ten powers, the powers with their multiples by [1] to [b - 1];
      otherwise the powers whose exponents are multiples of the decimal step for
      a tenth of the span of the exponents of the domain's ends, a step of at
      least [1]; and the multiples of the first rule if these are fewer than
      five;
    - on symlog scales of constant [c], [0.] if it is inside the domain, and the
      powers of ten and their negations of magnitude at least [c] whose
      exponents are multiples of the least [k] in [1], [2], [5], [10], … that
      gives at most ten values or, if none does, of the [k] that gives the
      fewest; and the multiples of the first rule if these are fewer than two;
    - on temporal scales, the boundaries of the time interval for a tenth of the
      domain's length, or of the next finer one that has one in the domain, the
      interval its nice domain rounds to.

    The search makes candidates of a number of ticks bounded whatever [length].
*)

(** {1:labels Labels}

    Labels depend only on the values they label, the scale, the notation and the
    locale. The values of linear, pow and custom scales are labelled as
    {e quantities}. So are those of log scales, unless every value is an integer
    power of the base or, in an {{!Scale.log}integer base} [b], such a power
    times an integer from [2] to [b - 1], and those of symlog scales, unless
    every value is [0.], a power of ten or its negation, each computed as the
    powers family computes them ({!section-candidates}): such values are
    labelled as {e logarithms}. Instants and categories have rules of their own.

    {2:quantities Quantities}

    Quantities are labelled together, each value read as the shortest decimal
    that rounds to it. Let [10^p] be the greatest power of ten of which every
    value is a multiple ([p = 0] if every value is [0.]), and [e] the exponent
    of the largest magnitude, [10^e <= |x| < 10^(e+1)] ([e = 0] if every value
    is [0.]). Without [notation]:
    + {b Offset.} If there are at least two values, all positive or all
      negative, and [e - p + 1 > 7], so that their labels would need more than
      seven significant digits, let the {e offset} [o] be, of the multiples of
      [10^q] that lie between zero and every value, the one farthest from zero,
      for the least [q] such that [10^q] exceeds the difference of the greatest
      and least values. If [o ≠ 0], the labels show each value minus [o], the
      difference of decimals taken exactly, and the rules below then apply to
      the differences, [e] being theirs. Values from [1000000.1] to [1000000.5]
      read [0.1] to [0.5] with the note [+10⁶].
    + {b Factor.} If [e < -4] or [e > 5], the labels show each value divided by
      [10^k] for [k = 3 ⌊e / 3⌋]: values from [0] to [2000000] by [500000] read
      [0.0], [0.5], …, [2.0] with the note [×10⁶].
    + {b Digits.} Labels are in {!Number.Plain} notation with [max 0 (k - p)]
      decimals ([k = 0] without a factor), grouped if the largest magnitude
      written is at least [10^4], so that labels have the same decimals and
      distinct values distinct labels: values [0.25] apart read [0.00], [0.25],
      [0.50].

    The note is [×10] and [k] in superscript digits if there is a factor,
    followed, after a space if there is a factor, by the offset if there is one:
    [+] or the locale's minus sign, then [|o|] with the fewest digits that write
    it exactly, in {!Number.Plain} notation grouped if [10^-4 <= |o| < 10^6] and
    in {!Number.Exponent} notation otherwise. So [×10⁶], [+12,345], [+10⁶] and
    [×10⁻⁶ +2.5×10⁹] are notes. Without a factor or an offset, the note is
    [None].

    With [notation] there is no note. {!Number.Plain} and {!Number.Percent}
    labels share [max 0 (-p)] decimals ([max 0 (-p - 2)] for percentages) and
    are grouped as above. {!Number.Exponent} and {!Number.Si} labels each have
    the decimals that [10^p] needs at their own exponent or prefix, and are
    trimmed: values from [0] to [1500] by [500] read [0], [500], [1k], [1.5k].

    {2:logarithms Logarithms}

    Logarithms are written each alone, trimmed, with the fewest digits that read
    back as the value. In base [10] and on symlog scales they are in
    {!Number.Plain} notation, grouped as quantities are
    ({{!section-quantities}digits}), if every nonzero one has a magnitude in
    \[[10^-3];[10^4]\], and in {!Number.Exponent} notation otherwise: [0.01],
    [0.1], [1], [10], or [10⁻⁵], [2×10⁻⁵], [10⁻⁴]. In another base [b], a power
    [b^i] is written as the base followed by [i] in superscript digits, and a
    multiple [n b^i] as [n], [×], then the power: [2⁵], [3×16²], [e²]. The base
    is written in {!Number.Plain} notation with the fewest digits that read back
    as it, or [e] if it is [Float.exp 1.]. With [notation], every label is in
    that notation, alone, trimmed, with the fewest digits that read back as its
    value.

    {2:time Time}

    The values of a temporal scale are labelled by the coarsest of these units
    whose starts, in local time at the scale's offset, include every value; a
    label and its context are then:
    - years: the year, [2026], and no context;
    - months: [Mar], and the year, [2026];
    - days: the day of the month, [3], and the month and year, [Mar 2026];
    - minutes: [14:05], and the day and month, [3 Mar];
    - seconds: [:09], and the time to the minute, [14:05];
    - otherwise, the fraction of the second: the locale's decimal separator
      followed by the fewest digits, at least one, that write the fraction of
      every value exactly, [.25] for values a quarter of a second apart, and the
      time to the second, [14:05:09].

    Times are on a 24-hour clock with two digits per field, months are named by
    the locale, and years are written in decimal without grouping, with the
    locale's minus sign before a negative year.

    {2:categories Categories}

    A category is labelled by its label, or by the text of its integer.

    {1:minor Minor ticks}

    The minor ticks of {!choose} lie inside the domain, never at a major tick,
    and continue beyond the outer major ticks to the ends of the domain. By the
    family of the chosen candidate, they are:
    - decimal steps: the multiples the skip leaves out if [j > 1]; otherwise the
      step divided into [5] parts for [q] in [1], [2.5] and [5], into [4] parts
      for [q] in [2] and [4], and into [3] parts for [q = 3];
    - powers: the powers the skip leaves out if [j > 1]; otherwise, in an
      {{!Scale.log}integer base} [b], the multiples by [2] to [b - 1] of the
      powers that are not ticks;
    - signed powers: the powers the skip leaves out;
    - calendar intervals: the boundaries the skip leaves out if [j > 1];
      otherwise the boundaries of the finest time interval, if any, whose
      boundaries include the ticks and that divides the span between two ticks
      into at most seven parts: years into quarters, a day into six hours, an
      hour into quarter hours, a month into nothing;
    - strides: none. *)

(** {1:comparing Comparing and formatting} *)

val equal : t -> t -> bool
(** [equal t t'] is [true] iff [t] and [t'] have equal major ticks, minor
    positions and notes, positions compared by [Float.equal]. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf t] formats the positions, labels, contexts and note of [t] for
    debugging and tests. *)
