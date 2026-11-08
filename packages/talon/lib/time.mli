(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Civil time values: stamps, spans and dates.

    These are the OCaml values that temporal cells read as: a datetime cell
    reads as a {!stamp}, a duration or a time of day as a {!span}, and a date as
    a {!date}. All three are exact integers. Time here is civil (POSIX) time:
    every day lasts 86,400 seconds and there are no leap seconds. Timescales
    such as TAI are extension types.

    Conversions name their unit, as in [to_ms] and [Span.of_us]. Converting to a
    coarser unit is total and rounds toward negative infinity. Converting from a
    unit coarser than nanoseconds can overflow: the [of_*] functions then return
    [None], and the {!Span} constructors raise [Invalid_argument]. *)

(** {1:stamps Instants and wall clocks}

    A {e stamp} is a signed count of nanoseconds since 1970-01-01 00:00:00 on a
    clock, from -2{^ 63} to 2{^ 63} - 1, about 1677-09-21 to 2262-04-11. A
    stamp's type states which of two clocks it counts. An {e instant} counts
    UTC. It is a point on the global timeline, the same reading for everyone. A
    {e wall clock} counts a local calendar and clock face, a reading a person
    saw, such as a logger's [02:30]. A wall clock takes an instant only once a
    zone says where the clock hung, and two sites' wall clocks do not compare
    as moments.

    The two meet only through a zone, in [Talon.Expr.Temporal]: [wall] reads an
    instant on a zone's clock, and [localize] finds a wall clock's instant under
    stated policies, because a zone's clock can skip or repeat a reading. A
    stamp of the wrong clock is a type error, so a literal, a column handle or a
    function cannot mix the two.

    {b Composing with ptime.} An instant becomes a [Ptime.t] over {!to_s} and
    {!to_ns}; a wall clock, read as UTC, the same way:
    {[
    let to_ptime t =
      let s = Time.to_s t in
      let ps = Int64.(mul 1000L (sub (Time.to_ns t) (mul s 1_000_000_000L))) in
      Ptime.of_span Ptime.Span.(add (of_int_s (Int64.to_int s)) (v (0, ps)))
    ]} *)

type +!'c stamp
(** The type for stamps of the clock ['c]. *)

type utc
(** The index of the UTC clock. *)

type local
(** The index of a wall clock. *)

type instant = utc stamp
(** The type for instants. *)

type wall = local stamp
(** The type for wall clocks. *)

(** The type for the witness of a stamp's clock, which a column type holds. *)
type _ clock =
  | Utc : utc clock  (** The UTC clock of an instant. *)
  | Local : local clock  (** The clock of a wall clock. *)

val of_ns : int64 -> 'c stamp
(** [of_ns n] is the stamp [n] nanoseconds after 1970-01-01 00:00:00 of its
    clock. *)

val of_us : int64 -> 'c stamp option
(** [of_us n] is the stamp [n] microseconds after 1970-01-01 00:00:00 of its
    clock, or [None] if it is out of range. *)

val of_ms : int64 -> 'c stamp option
(** [of_ms n] is like {!of_us} in milliseconds. *)

val of_s : int64 -> 'c stamp option
(** [of_s n] is like {!of_us} in seconds. *)

val to_ns : 'c stamp -> int64
(** [to_ns t] is the number of nanoseconds from 1970-01-01 00:00:00 of [t]'s
    clock to [t]. *)

val to_us : 'c stamp -> int64
(** [to_us t] is the number of whole microseconds from 1970-01-01 00:00:00 of
    [t]'s clock to [t], rounded toward negative infinity: the stamp one
    nanosecond before the epoch gives [-1L]. [of_us (to_us t)] is [t] whenever
    [t] is a whole microsecond. *)

val to_ms : 'c stamp -> int64
(** [to_ms t] is like {!to_us} in milliseconds. *)

val to_s : 'c stamp -> int64
(** [to_s t] is like {!to_us} in seconds. *)

val equal : 'c stamp -> 'c stamp -> bool
(** [equal t0 t1] is [true] iff [t0] and [t1] are the same reading of one clock.
*)

val compare : 'c stamp -> 'c stamp -> int
(** [compare t0 t1] orders stamps of one clock chronologically. It is compatible
    with {!equal}. *)

val pp : Format.formatter -> 'c stamp -> unit
(** [pp ppf t] formats [t] as its civil date and time of day in ISO 8601's
    extended format, [YYYY-MM-DDThh:mm:ss], followed by a point and 3, 6 or 9
    digits, the fewest that are exact, when [t] is not a whole second:
    [2024-03-15T09:30:00], [1969-12-31T23:59:59.999999999]. It writes no zone
    designator, since a stamp does not hold its clock at run time. *)

(** {1:spans Spans} *)

type span
(** The type for spans: a signed count of nanoseconds, from -2{^ 63} to 2{^ 63}
    \- 1, about 292 years either way. A span lasts the same everywhere; calendar
    lengths such as a month are {!step}s. *)

(** Spans. *)
module Span : sig
  type t = span
  (** The type for spans. *)

  (** {1:constructors Constructors} *)

  val ns : int -> span
  (** [ns n] is [n] nanoseconds. *)

  val us : int -> span
  (** [us n] is [n] microseconds.

      Raises [Invalid_argument] if the span is out of range. *)

  val ms : int -> span
  (** [ms n] is like {!us} in milliseconds. *)

  val s : int -> span
  (** [s n] is like {!us} in seconds. *)

  val minutes : int -> span
  (** [minutes n] is like {!us} in minutes. *)

  val hours : int -> span
  (** [hours n] is like {!us} in hours. *)

  val days : int -> span
  (** [days n] is like {!us} in days of 86,400 seconds. *)

  (** {1:converting Converting} *)

  val of_ns : int64 -> span
  (** [of_ns n] is [n] nanoseconds. *)

  val of_us : int64 -> span option
  (** [of_us n] is [n] microseconds, or [None] if the span is out of range. *)

  val of_ms : int64 -> span option
  (** [of_ms n] is like {!of_us} in milliseconds. *)

  val of_s : int64 -> span option
  (** [of_s n] is like {!of_us} in seconds. *)

  val to_ns : span -> int64
  (** [to_ns d] is [d] in nanoseconds. *)

  val to_us : span -> int64
  (** [to_us d] is [d] in whole microseconds, rounded toward negative infinity:
      [to_us (ns (-1))] is [-1L]. *)

  val to_ms : span -> int64
  (** [to_ms d] is like {!to_us} in milliseconds. *)

  val to_s : span -> int64
  (** [to_s d] is like {!to_us} in seconds. *)

  (** {1:predicates Predicates and formatting} *)

  val equal : span -> span -> bool
  (** [equal d0 d1] is [true] iff [d0] and [d1] are the same span. *)

  val compare : span -> span -> int
  (** [compare d0 d1] orders spans from the most negative to the most positive.
      It is compatible with {!equal}. *)

  val pp : Format.formatter -> span -> unit
  (** [pp ppf d] formats [d] as a minus sign when it is negative, then each
      nonzero component in hours, minutes, seconds, milliseconds, microseconds
      and nanoseconds, largest first, suffixed [h], [m], [s], [ms], [us] and
      [ns]: [168h], [1h30m], [1s500ms], [-250us]. The zero span formats as [0s].
      A span never formats in days, which in printed steps are calendar days
      ({!pp_step}). *)
end

(** {1:dates Dates} *)

type date
(** The type for dates: a day of the proleptic Gregorian calendar, counted in
    days since 1970-01-01. Dates range over the days that fit in a signed 32-bit
    integer, about the years -5,877,641 to 5,881,580. *)

(** Dates. *)
module Date : sig
  type t = date
  (** The type for dates. *)

  val of_civil : int * int * int -> date option
  (** [of_civil (y, m, d)] is day [d] of month [m] of year [y] in the proleptic
      Gregorian calendar, or [None] if [m] is not in \[[1];[12]\], [d] is not a
      day of that month, or the date is out of range. Years are astronomical:
      year [0] is 1 BC and year [-1] is 2 BC. *)

  val to_civil : date -> int * int * int
  (** [to_civil t] is [(y, m, d)], the year, month and day of [t].
      [of_civil (to_civil t)] is [Some t]. *)

  val of_days : int -> date option
  (** [of_days n] is the date [n] days after 1970-01-01, or [None] if it is out
      of range. *)

  val to_days : date -> int
  (** [to_days t] is the number of days from 1970-01-01 to [t]. *)

  val equal : date -> date -> bool
  (** [equal t0 t1] is [true] iff [t0] and [t1] are the same day. *)

  val compare : date -> date -> int
  (** [compare t0 t1] orders dates chronologically. It is compatible with
      {!equal}. *)

  val pp : Format.formatter -> date -> unit
  (** [pp ppf t] formats [t] as [YYYY-MM-DD]. A year outside \[[0];[9999]\] is
      signed and has at least four digits: [-0044-03-15], [+12345-01-01]. *)
end

(** {1:steps Calendar steps} *)

(** The type for steps: the units in which temporal expressions floor and offset
    stamps and dates. A step of [n] calendar units has no fixed length, since
    months have 28 to 31 days and, in a zone with daylight saving time, a day
    lasts 23, 24 or 25 hours. *)
type step =
  | Months of int  (** [Months n] is [n] calendar months. *)
  | Weeks of int  (** [Weeks n] is [n] calendar weeks. Weeks begin on Monday. *)
  | Days of int  (** [Days n] is [n] calendar days. *)
  | Exact of span  (** [Exact d] is the span [d]. *)

val pp_step : Format.formatter -> step -> unit
(** [pp_step ppf s] formats [Months n] as [nmo], [Weeks n] as [nw], [Days n] as
    [nd], and [Exact d] as {!Span.pp} formats [d]: [1mo], [2w], [-1d], [15m]. *)
