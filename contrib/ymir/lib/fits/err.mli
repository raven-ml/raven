(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Failures in data, and the places that start their messages.

    Decoders raise {!Fail} where a check fails, and each public function
    turns it into its [Error] with {!catch}. A message starts with its
    place, as in ["f.fits: HDU 1 (SCI), card 7 (BUNIT): what"]. *)

exception Fail of string
(** The exception for a failed check, carrying the whole message. *)

val fail : ('a, unit, string, 'b) format4 -> 'a
(** [fail fmt ...] raises {!Fail} with the formatted message. *)

val ok_or_fail : ('a, string) result -> 'a
(** [ok_or_fail r] is [r]'s value. Raises {!Fail} with [r]'s error. *)

val catch : (unit -> 'a) -> ('a, string) result
(** [catch f] is [Ok (f ())], or the [Error] of a {!Fail} or of a
    [Sys_error], which a read of a file that ended or vanished since it was
    opened raises. *)

(** {1:places Places} *)

type place
(** The type for where a failure is: a file's name, then the places inside
    it, in order. *)

val nowhere : place
(** [nowhere] has no name and no part. *)

val file : string -> place
(** [file name] is the place of the file or bytes named [name]. *)

val sub : place -> string -> place
(** [sub p part] is [p] followed by [part], such as ["card 7 (BUNIT)"]. *)

val msg : place -> string -> string
(** [msg p what] is [what] after [p]'s name and parts, if any. *)

val fail_at : place -> ('a, unit, string, 'b) format4 -> 'a
(** [fail_at p fmt ...] raises {!Fail} with the formatted message at [p]. *)

(** {1:sizes Sizes} *)

val mul : int -> int -> int option
(** [mul a b] is [a * b], or [None] if it passes [max_int]. Raises {!Fail}
    if [a] or [b] is negative. *)

val add : int -> int -> int option
(** [add a b] is [a + b] for [b >= 0], or [None] if it passes [max_int]. *)
