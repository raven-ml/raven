(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Values computed on first use, once.

    Any domain may call any function. The first domain to {!get} an
    uncomputed value computes it under the value's lock, and the others
    asking meanwhile wait for it. A computation that raises leaves the value
    uncomputed, for the next {!get} to try again. *)

type 'a t
(** The type for values of type ['a] computed on first use. *)

val make : (unit -> 'a) -> 'a t
(** [make f] is the value [f ()], computed on the first {!get}. *)

val of_value : 'a -> 'a t
(** [of_value v] is [v], already computed. *)

val is_computed : 'a t -> bool
(** [is_computed t] is [true] iff a {!get} of [t] has returned or [t] was
    made by {!of_value}. *)

val get : 'a t -> 'a
(** [get t] is [t]'s value. It blocks while another domain computes it, and
    raises what the computation raises. *)
