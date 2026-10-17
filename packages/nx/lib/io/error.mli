(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Errors of nx.io's NumPy and text readers and writers, and the messages
    that every format's errors share.

    {!Nx_io} raises an error as [Failure] with the text of {!to_string}. *)

type t =
  | Io_error of string
  | Format_error of string
  | Unsupported_dtype
  | Unsupported_shape
  | Missing_entry of string  (** The name of an archive entry. *)
  | Other of string  (** A message shown as it is. *)
(** The type for errors. *)

val to_string : t -> string
(** [to_string e] is [e]'s message, prefixed with its kind. *)

val fail_msg : ('a, unit, string, 'b) format4 -> 'a
(** [fail_msg fmt ...] raises [Failure] with the formatted message. *)

val no_bit : string -> string
(** [no_bit format] is the message that refuses a [bit] tensor in [format],
    which has no 1-bit dtype. It names the two ways to save one. *)
