(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** SHA-256 (FIPS 180-4), fed a message in pieces. *)

type t
(** The type for the state of a digest being computed. *)

val init : unit -> t
(** [init ()] is the state of the empty message. *)

val feed : t -> Bytes.t -> int -> int -> unit
(** [feed t b off len] appends [len] bytes of [b] from [off] to [t]'s
    message. *)

val hex : t -> string
(** [hex t] is the digest of [t]'s message, 64 lowercase hexadecimal digits.
    [t] is spent: it is not fed again. *)
