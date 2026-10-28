(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Results as values that tell every two results apart, and print them. *)

(** {1:outcomes Outcomes} *)

(** The type for a call's outcome: its value, or the message of the
    [Invalid_argument] it raised. *)
type 'a outcome = Value of 'a | Raised of string

val outcome : (unit -> 'a) -> 'a outcome
(** [outcome f] is [f ()]'s outcome. Another exception propagates. *)

val outcome_t : 'a Windtrap.Testable.t -> 'a outcome Windtrap.Testable.t
(** [outcome_t t] compares values by [t] and messages exactly. *)

(** {1:bits Bits} *)

val code : float -> int64
(** [code x] is [x]'s bits, [-1L] for every NaN: nx leaves a NaN result's sign
    and payload unspecified. *)

type bits = int array * int64 array
(** The type for a float array's shape and {!code}s. *)

val bits : (float, 's, 'd) Nx.t -> bits
(** [bits x] is [x]'s shape and codes. *)

val bits_t : bits Windtrap.Testable.t
(** [bits_t] compares bits exactly and prints elements as [%h]. *)

type codes = int array * string array
(** The type for an array's shape and elements as text. *)

val codes : ('v, 's, 'd) Nx.t -> codes
(** [codes x] is [x]'s shape and elements: a float's {!code}, a complex number's
    two parts' codes, other values as {!Nx.Dtype.pp_value} prints them. *)

val codes_t : codes Windtrap.Testable.t
(** [codes_t] compares codes exactly. *)

(** {1:ulps Distances} *)

val ulps : (float, 's) Nx.dtype -> float -> float -> float
(** [ulps dt a b] is the number of [dt] floats from [a] to [b], two [dt] values,
    [-0.] and [+0.] one apart: [0.] between two NaNs, and infinite between a NaN
    and a number or between an infinity and any other value, so that an infinity
    is never within a bound of the largest finite float.

    Raises [Invalid_argument] unless [dt] is [Float32] or [Float64]. *)
