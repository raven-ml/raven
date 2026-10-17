(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Decimal steps and their exact multiples, powers, and the steps of time.

    {!Scale} rounds nice domains to these, and {!Ticks} scores them as
    candidates. *)

(* Powers of ten *)

val pow10 : int -> float
(** [pow10 k] is the float nearest [10]{^ [k]}, with infinity above [308]
    and [0.] below [-323]. *)

(* Decimal steps *)

type step = { m : int; k : int }
(** The type for the step [m × 10]{^ [k]}, where [m >= 1]. *)

val value : ?exact:(int -> int -> float -> float) -> step -> int -> float
(** [value ~exact s i] is the float nearest [n × 10]{^ [k]}, where
    [n = i × m], with [0.] for a negative multiple that underflows. Where
    [|n| <= 2]{^ 53} and [|k| <= 22] one multiplication or division rounds
    once. Elsewhere [exact n k approx] moves that approximation to the
    float the exact value rounds to. [exact] defaults to
    {!Decimal.nearest}. *)

val width : step -> float
(** [width s] is [value s 1]. *)

val step : float -> float -> int -> step
(** [step a b count] is the step of mantissa 1, 2, 5 or 10 nearest by ratio
    to the length of \[[a];[b]\] over [count]. [a < b] must hold. *)

val index : step -> float -> float
(** [index s x] is [x / (m × 10]{^ [k]}[)], possibly infinite. *)

val is_fine : step -> float -> float -> bool
(** [is_fine s a b] is [true] iff [s] is below a 32nd of the gap between
    floats at both [a] and [b], so that its multiples are finer than the
    floats around the ends. *)

val collect : ((float -> unit) -> unit) -> float array
(** [collect f] is the floats [f push] pushes, in order, with consecutive
    duplicates once. *)

val multiples :
  ?value:(step -> int -> float) ->
  ?skip:int ->
  ?offset:int ->
  step ->
  float ->
  float ->
  float array
(** [multiples ~value ~skip ~offset s a b] is the distinct floats of the
    multiples of [s] in \[[a];[b]\] whose index is [offset] modulo [skip],
    increasing, or every float of \[[a];[b]\] if [is_fine s a b]. [value]
    computes the multiples and defaults to {!value}. [skip] defaults to [1]
    and [offset], in \[[0];[skip - 1]\], to [0]. *)

val floor_multiple : step -> float -> float
(** [floor_multiple s x] is the greatest multiple of [s] at most [x]. [s]
    must not be fine at [x] ({!is_fine}). *)

val ceil_multiple : step -> float -> float
(** [ceil_multiple s x] is the least multiple of [s] at least [x]. [s] must
    not be fine at [x] ({!is_fine}). *)

(* Powers *)

val is_integer_base : float -> bool
(** [is_integer_base b] is [true] iff [b] is an integer in \[[2];[16]\]. *)

val power : float -> int -> float
(** [power b i] is [b]{^ [i]}, the float nearest it in base 10. *)

val multiple : float -> int -> int -> float
(** [multiple b j i] is [j × b]{^ [i]}, computed as a decimal multiple
    ({!value}) in base 10. *)

val stride : int -> int
(** [stride r] is the [r]th of 1, 2, 5, 10, 20, 50, ..., from [0]. *)

(* Steps of time *)

type calendar
(** The type for how a step of time cuts the time line. *)

type time_step = {
  interval : Time.interval;
  calendar : calendar;
  duration : float;
      (** The duration in nanoseconds, with months of 30 days and years of
          365 days. *)
  longest : float;
      (** The longest span between two boundaries, in nanoseconds. *)
  i : int;  (** The rank, from 1, among the strides of its unit. *)
  n : int;  (** The number of strides of its unit. *)
}
(** The type for steps of time. *)

val time_steps : time_step array
(** [time_steps] is the table of steps of time, finest first. It holds
    decades of nanoseconds up to 500 ms, then seconds, minutes, hours, days,
    weeks, months and decades of years. *)

val ns_diff : Time.t -> Time.t -> float
(** [ns_diff t a] is [t - a] in nanoseconds, rounded once to a float. *)

val nearest_time_step : float -> int
(** [nearest_time_step t] is the index in {!time_steps} of the step whose
    duration is nearest by ratio to [t], the coarser of two equally near.
    [t > 0.] must hold. *)

val minor_step : time_step -> time_step option
(** [minor_step st] is the finest step of {!time_steps} whose boundaries
    include those of [st] and that cuts the span between two of them into
    two to seven parts, if any. *)
