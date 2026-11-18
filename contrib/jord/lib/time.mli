(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Instants of UTC.

    An {e instant} is a float, the seconds of Coordinated Universal Time (UTC)
    since 1970-01-01T00:00:00Z, each day counted as 86400 of them, as POSIX's
    "seconds since the Epoch" counts them (POSIX.1-2024, Base Definitions,
    §4.19: "each and every day shall be accounted for by exactly 86400
    seconds"). Before 1970 the count is negative, on the proleptic Gregorian
    calendar. Instants batch as float64 tensors, such as a tile's time per
    row, which {!Sun} takes.

    {b Leap seconds.} UTC follows the Earth's rotation by inserting a {e leap
    second}, 23:59:60, at the end of some days (RFC 3339, Appendix D), which
    keeps it within 0.9 s of UT1, the time of that rotation
    (Recommendation ITU-R TF.460-6, §1.2). A leap second has no count of its
    own: 23:59:60.5 and the next day's 00:00:00.5 are one instant. So two
    instants across a leap second differ by one second less than elapsed, and
    an instant needs no table of leap seconds, which the IERS announces only
    months ahead. *)

val of_rfc3339 : string -> (float, string) result
(** [of_rfc3339 s] is the instant [s] states in RFC 3339's [date-time]
    (§5.6), the form of STAC's [datetime], [YYYY-MM-DDTHH:MM:SS], an optional
    fraction of a second, and [Z] or an offset [±HH:MM], which is taken off, so
    ["2024-06-29T12:20:21+02:00"] is ["2024-06-29T10:20:21Z"]. [-00:00],
    RFC 3339's unknown local offset (§4.3), reads as [Z]. [T] and [Z] may be
    lower case, and the separator a space (§5.6, NOTE). A second 60 is a leap
    second, the next UTC day's first second, and must be the last second of a
    UTC day once the offset is taken off (§5.7), on any day. The fraction is
    kept to a float64's spacing ({!section-precision}).

    The result is [Error] for any other text, for a field out of its range,
    the offset's included, as a day 00, 31 June, an hour 24 or a minute 60,
    and for a second 60 before a UTC day's last second. *)

val date : float -> int * int * int
(** [date t] is the year, month and day of UTC's calendar on which the
    instant [t] lies; a leap second's lies on the next day, as above.

    Raises [Invalid_argument] if [t] is not finite or lies 2{^61} days or
    more from 1970, past OCaml's integers. *)

val day_of_year : float -> int
(** [day_of_year t] is the number of the day on which the instant [t] lies
    within its year, from 1 on 1 January to 365, or 366 in a leap year, the
    day of ISO 8601's ordinal date (ISO 8601:2004, 4.1.3.1). A leap second's
    instant lies on the next day, as {!date}'s.

    Raises [Invalid_argument] where {!date} does. *)

val iso_week : float -> int * int
(** [iso_week t] is the week of ISO 8601's week calendar in which the instant
    [t] lies, and the year it belongs to, as [(year, week)]. Weeks run from
    Monday to Sunday, and a year's first week is "that one which includes
    the first Thursday of that year" (ISO 8601:2004, 2.2.10), so a year has
    52 or 53 weeks and its first and last may hold days of the years beside
    it: "Sunday 1 January 1995 is identified by the calendar date
    [1995-01-01] and week date [1994-W52-7]", and Tuesday 31 December 1996
    by [1997-W01-2] (3.2.2, NOTE 2). So near 1 January the year can differ
    from {!date}'s.

    Raises [Invalid_argument] where {!date} does. *)

val of_date : int * int * int -> float
(** [of_date (y, m, d)] is the instant 00:00:00 UTC of day [d] of month [m]
    of year [y], in the proleptic Gregorian calendar, so
    [date (of_date x) = x].

    Raises [Invalid_argument] if [(y, m, d)] is not a day of the calendar or
    [|y| >= 2{^28}], past which a float no longer holds each second of a
    day's instant. *)

(** {1:precision Precision}

    A float64 spaces instants by at most 0.24 µs from 2004 to 2038 and
    0.48 µs to 2106, and more finely before 2004, finer than any acquisition
    states its time. A float32 would space them by minutes. *)
