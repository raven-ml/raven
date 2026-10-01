(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Failures found in data and in the environment.

    Reading a file that is missing or malformed is not a programming error, so
    talon returns it as [Error e], where [e] says what failed and where. *)

type t
(** The type for errors. An error holds a message and, when known, the file it
    was found in and the bytes of that file where it was found. *)

val v : ?file:string -> ?bytes:int * int -> string -> t
(** [v ?file ?bytes msg] is the error [msg], found in [file] at the bytes
    [bytes]. [bytes] is [(first, last)], the zero-based positions of the first
    and the last byte of the range, both included.

    Raises [Invalid_argument] if [bytes] is given and [first < 0] or
    [last < first]. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf e] formats [e] for people, the location first, as in
    [zoneinfo/Europe/Paris: bytes 106-113: transition 1 is not after the
     previous one]. *)

val get_ok : ('a, t) result -> 'a
(** [get_ok r] is [v] if [r] is [Ok v].

    Raises [Failure] with [e] formatted by {!pp} if [r] is [Error e]. *)
