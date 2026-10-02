(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Instants and the calendar.

    An instant is a point of the POSIX time line: a whole number of nanoseconds
    since the Unix epoch, 1970-01-01 00:00:00 UTC, every day counting 86 400
    seconds (leap seconds are not counted). Data writes instants as int64 counts
    of a {!type-resolution}, as the [time] lift of Hugin takes them. A count is
    a way of writing an instant, not part of it: [v S 1L] and [v Ms 1000L] are
    the same instant.

    The calendar views an instant at a fixed offset from UTC as a civil date and
    time of the proleptic Gregorian calendar, with astronomical year numbering
    (year [0] is 1 BC). There are no time zones and no daylight saving time: an
    offset is a number of seconds, the same all year, and every function taking
    one defaults it to [0]. {{!section-intervals}Intervals} cut the time line at
    the starts of calendar units; time ticks and nice time domains are made of
    their boundaries.

    Days and civil dates convert with Howard Hinnant's [days_from_civil] and
    [civil_from_days]
    ({{:https://howardhinnant.github.io/date_algorithms.html}date algorithms}),
    exact for every instant of this module. *)

(** {1:instants Instants} *)

type t = private {
  sec : int64;  (** Whole seconds since the epoch. *)
  nsec : int;  (** Nanoseconds after [sec], in \[[0];[999_999_999]\]. *)
}
(** The type for instants. The instant [{ sec; nsec }] is [sec] seconds and
    [nsec] nanoseconds after the epoch, so one before the epoch has a negative
    [sec] and a non-negative [nsec]: [v Ms (-1L)] is
    [{ sec = -1L; nsec = 999_000_000 }]. Each instant has one such
    decomposition. An instant is representable iff its [sec] fits in an int64,
    about 2.9 × 10{^ 11} years on each side of the epoch, so every instant an
    int64 count of any {!type-resolution} writes is. Offsets from UTC are whole
    seconds, so [nsec] is the same in local time. *)

(** The type for resolutions, the units of int64 counts of instants. *)
type resolution =
  | S  (** Seconds. *)
  | Ms  (** Milliseconds. *)
  | Us  (** Microseconds. *)
  | Ns  (** Nanoseconds. *)

val v : resolution -> int64 -> t
(** [v r n] is the instant [n] units of [r] after the epoch, or [-n] units
    before it if [n] is negative. *)

val to_int64 : resolution -> t -> int64 option
(** [to_int64 r t] is [Some n] for the greatest [n] such that [v r n] is not
    after [t], and [None] if that [n] does not fit in an int64. For example
    [to_int64 S (v Ms (-1L))] is [Some (-1L)]. *)

val epoch : t
(** [epoch] is 1970-01-01 00:00:00 UTC. *)

(** {1:civil Civil dates and times} *)

type tz_offset_s = int
(** The type for offsets from UTC, in seconds east of UTC: [3600] is UTC+01:00
    and [-18000] is UTC−05:00. Functions taking one raise [Invalid_argument] if
    it is not in \]−86 400;86 400\[. *)

type date = int * int * int
(** The type for civil dates [(y, m, d)]: the year [y], the month [m] in
    \[[1];[12]\] and the day [d], from [1] to the length of month [m] of year
    [y]. *)

type time = int * int * int
(** The type for times of day [(hh, mm, ss)]: the hour [hh] in \[[0];[23]\], and
    the minute [mm] and the second [ss] in \[[0];[59]\]. *)

val of_date_time : ?tz_offset_s:tz_offset_s -> date * time -> t
(** [of_date_time ~tz_offset_s (d, tm)] is the instant whose civil date and time
    at [tz_offset_s] are [d] and [tm], on a whole second.

    Raises [Invalid_argument] if [d] is not a date, [tm] is not a time, or the
    instant is not representable. *)

val of_date : ?tz_offset_s:tz_offset_s -> date -> t
(** [of_date ~tz_offset_s d] is [of_date_time ~tz_offset_s (d, (0, 0, 0))], the
    start of day [d] at [tz_offset_s]. *)

val to_date_time : ?tz_offset_s:tz_offset_s -> t -> date * time
(** [to_date_time ~tz_offset_s t] is the civil date and time of [t] at
    [tz_offset_s], its nanoseconds [t.nsec] dropped.
    [of_date_time ~tz_offset_s (to_date_time ~tz_offset_s t)] is [t] iff
    [t.nsec = 0]. *)

(** {1:intervals Intervals} *)

type interval
(** The type for calendar intervals. An interval is a calendar unit and a stride
    [k >= 1]. At an offset it denotes the set of its {e boundaries}: the
    instants that start a unit whose index is a multiple of [k], units being
    indexed in the offset's local time from these origins:
    - nanoseconds to days: the number of whole units since 1970-01-01 00:00:00
      local time;
    - weeks: the number of weeks since Monday 1969-12-29 local time, the start
      of the first ISO 8601 week of 1970, so weeks start on Monday;
    - months: [12 y + m - 1] for month [m] of year [y];
    - years: the year.

    So [months 3] starts January, April, July and October, [hours 6] starts at
    00:00, 06:00, 12:00 and 18:00 local time, and [days 2] starts every other
    day, evenly across the ends of months. Intervals with the same boundaries at
    every offset are equal: [seconds 60] is [minutes 1] and [hours 24] is
    [days 1], but [days 7] is not [weeks 1].

    The constructors raise [Invalid_argument] if the stride is below [1] or, for
    units up to weeks, the stride spans more than 2{^ 62} nanoseconds, about 146
    years. *)

val nanoseconds : int -> interval
(** [nanoseconds k] is the interval of [k] nanoseconds. *)

val microseconds : int -> interval
(** [microseconds k] is [nanoseconds (1_000 * k)]. *)

val milliseconds : int -> interval
(** [milliseconds k] is [nanoseconds (1_000_000 * k)]. *)

val seconds : int -> interval
(** [seconds k] is the interval of [k] seconds. *)

val minutes : int -> interval
(** [minutes k] is [seconds (60 * k)]. *)

val hours : int -> interval
(** [hours k] is [seconds (3600 * k)]. *)

val days : int -> interval
(** [days k] is [seconds (86_400 * k)]. *)

val weeks : int -> interval
(** [weeks k] is the interval of [k] weeks starting on Monday. *)

val months : int -> interval
(** [months k] is the interval of [k] calendar months. *)

val years : int -> interval
(** [years k] is the interval of [k] calendar years. *)

val floor : ?tz_offset_s:tz_offset_s -> interval -> t -> t
(** [floor ~tz_offset_s i t] is the latest boundary of [i] at [tz_offset_s] that
    is not after [t].

    Raises [Invalid_argument] if that boundary is not representable. *)

val ceil : ?tz_offset_s:tz_offset_s -> interval -> t -> t
(** [ceil ~tz_offset_s i t] is the earliest boundary of [i] at [tz_offset_s]
    that is not before [t].

    Raises [Invalid_argument] if that boundary is not representable. *)

val add : ?tz_offset_s:tz_offset_s -> interval -> int -> t -> t
(** [add ~tz_offset_s i n t] is [t] moved by [n] strides of [i], towards the
    future iff [n] is positive. A stride of a unit up to weeks is a fixed
    duration. A stride of months or years moves the civil date of [t] at
    [tz_offset_s] by that many months, keeps its time of day and nanoseconds,
    and clamps the day to the length of the month reached: one month after
    January 31 is February 28 or 29. The boundaries of [i] at [tz_offset_s] are
    closed under [add ~tz_offset_s i n].

    Raises [Invalid_argument] if the result is not representable. *)

val range : ?tz_offset_s:tz_offset_s -> interval -> t -> t -> t array
(** [range ~tz_offset_s i t t'] is the boundaries of [i] at [tz_offset_s] in
    \[[t];[t']\], from the past to the future. It is empty if [t'] is before
    [t]. It allocates one instant per boundary, so a fine interval over a long
    span makes a large array. *)

val equal_interval : interval -> interval -> bool
(** [equal_interval i i'] is [true] iff [i] and [i'] have the same boundaries at
    every offset. *)

val pp_interval : Format.formatter -> interval -> unit
(** [pp_interval ppf i] formats [i] for debugging, such as [3 months]. *)

(** {1:comparing Comparing and formatting} *)

val equal : t -> t -> bool
(** [equal t t'] is [true] iff [t] and [t'] are the same instant. *)

val compare : t -> t -> int
(** [compare t t'] orders instants from the past to the future. It is compatible
    with {!equal}. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf t] formats [t] at UTC in RFC 3339 form, such as
    [2026-10-01T12:30:00.25Z]: the fraction of a second appears iff it is not
    zero, to the nanosecond with its trailing zeros removed, and a year outside
    \[[0];[9999]\] is written with its sign and at least four digits, as ISO
    8601 extends years. The output is stable. *)
