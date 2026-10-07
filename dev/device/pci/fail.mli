(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The world's refusals of a request, inside the library (private).

    {!Failed} never leaves the library: each request converts it to [Error] with
    {!result}. *)

exception Failed of string
(** [Failed why] is a request the world refused, [why] naming what failed. *)

val fail : ('a, unit, string, 'b) format4 -> 'a
(** [fail fmt ...] raises {!Failed} with the formatted message. *)

val step : string -> (unit -> 'a) -> 'a
(** [step what f] is [f ()], whose [Unix.Unix_error] raises {!Failed} as
    ["what: cause"], the cause the system's message for the error. *)

val result : (unit -> 'a) -> ('a, string) result
(** [result f] is [Ok (f ())], or [Error why] if [f] raises [Failed why]. *)

val bug : string -> (unit -> 'a) -> 'a
(** [bug what f] is [f ()] for a system call that fails only if the library is
    wrong, such as unmapping its own mapping: its [Unix.Unix_error] raises
    [Failure] as ["what: cause"]. *)
