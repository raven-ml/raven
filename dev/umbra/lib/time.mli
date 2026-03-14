(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Astronomical time with phantom-typed time scales.

    Times are stored internally as Julian Dates (float). Scale conversions are
    type-safe: {!utc_to_tai} accepts a [utc t] and returns a [tai t].

    {[
      let t = Time.of_iso "2024-01-01T00:00:00" in
      let tai = Time.utc_to_tai t in
      let tt = Time.tai_to_tt tai in
      let jd = Time.to_jd tt
    ]} *)

(** {1:types Types} *)

type 'scale t
(** The type for a Julian Date tagged with time scale ['scale]. *)

type utc
(** Coordinated Universal Time. *)

type tai
(** International Atomic Time. *)

type tt
(** Terrestrial Time. *)

type tdb
(** Barycentric Dynamical Time. *)

(** {1:constructors Constructors} *)

val unsafe_of_jd : float -> 'a t
(** [unsafe_of_jd jd] is a time from the Julian Date [jd]. The caller must
    ensure [jd] is in the intended time scale. *)

val unsafe_of_mjd : float -> 'a t
(** [unsafe_of_mjd mjd] is a time from the Modified Julian Date [mjd] (MJD = JD
    \- 2400000.5). The caller must ensure [mjd] is in the intended time scale.
*)

val of_iso : string -> utc t
(** [of_iso s] parses an ISO 8601 date-time string as UTC. Accepted formats:
    ["YYYY-MM-DD"], ["YYYY-MM-DDThh:mm:ss"], and ["YYYY-MM-DDThh:mm:ssZ"].

    {b Warning.} Uses the Gregorian calendar; dates before 1582-10-15 produce
    incorrect Julian Dates. Leap seconds (e.g. [23:59:60]) cannot be represented
    and are parsed as the following second.

    Raises [Invalid_argument] if [s] cannot be parsed. *)

val of_unix : float -> utc t
(** [of_unix u] is the UTC time corresponding to the Unix timestamp [u] (seconds
    since 1970-01-01T00:00:00 UTC). *)

val now : unit -> utc t
(** [now ()] is the current UTC time from the system clock. *)

(** {1:comparison Comparison} *)

val compare : 'a t -> 'a t -> int
(** [compare a b] orders times by their Julian Date values. *)

val equal : 'a t -> 'a t -> bool
(** [equal a b] is [true] iff [a] and [b] have the same Julian Date value. *)

(** {1:eliminators Eliminators} *)

val to_jd : 'a t -> float
(** [to_jd t] is the Julian Date of [t]. *)

val to_mjd : 'a t -> float
(** [to_mjd t] is the Modified Julian Date of [t] (MJD = JD - 2400000.5). *)

val to_iso : utc t -> string
(** [to_iso t] formats [t] as an ISO 8601 string with trailing [Z]. Output is
    ["YYYY-MM-DDThh:mm:ssZ"] when the fractional seconds are below 0.5 ms, or
    ["YYYY-MM-DDThh:mm:ss.sssZ"] otherwise.

    {b Warning.} Leap-second labels like [23:59:60] cannot be produced; times
    within a leap second round to [00:00:00] of the following day. *)

val to_unix : utc t -> float
(** [to_unix t] is the Unix timestamp of [t] (seconds since 1970-01-01T00:00:00
    UTC). *)

(** {1:scales Scale conversions}

    UTC/TAI conversions use the IERS leap-second table (Bulletin C), currently
    covering 1972-01-01 through 2017-01-01 (TAI-UTC = 37 s). Dates before
    1972-01-01 use TAI-UTC = 10 s.

    TT = TAI + 32.184 s (exact by definition).

    TDB-TT uses the first 10 terms of the Fairhead & Bretagnon (1990) series,
    accurate to ~1 us within a few centuries of J2000.0. *)

val utc_to_tai : utc t -> tai t
(** [utc_to_tai t] converts [t] from UTC to TAI. *)

val tai_to_utc : tai t -> utc t
(** [tai_to_utc t] converts [t] from TAI to UTC. *)

val tai_to_tt : tai t -> tt t
(** [tai_to_tt t] converts [t] from TAI to TT. *)

val tt_to_tai : tt t -> tai t
(** [tt_to_tai t] converts [t] from TT to TAI. *)

val tt_to_tdb : tt t -> tdb t
(** [tt_to_tdb t] converts [t] from TT to TDB. *)

val tdb_to_tt : tdb t -> tt t
(** [tdb_to_tt t] converts [t] from TDB to TT. Uses a single Newton iteration;
    accurate to ~1 us. *)

(** {1:duration Duration} *)

val diff : 'a t -> 'a t -> Unit.time Unit.t
(** [diff a b] is the duration [a - b] as a {!Unit.time} quantity. *)

val add : 'a t -> Unit.time Unit.t -> 'a t
(** [add t dt] is [t] offset by the duration [dt]. *)
