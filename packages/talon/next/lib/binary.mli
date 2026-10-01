(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Byte strings.

    A binary cell holds any bytes, while a string cell holds valid UTF-8. The
    two are distinct OCaml types so that a string handle never binds binary
    data. Binary columns ({!Type.binary}) read as {!t}. *)

type t = private string
(** The type for byte strings. Coerce with [(b :> string)] to read the bytes.
    Equality and order are those of [string]: the bytes, compared one by one as
    unsigned integers. *)

val of_string : string -> t
(** [of_string s] is the bytes of [s]. It costs O(1). *)

val pp : Format.formatter -> t -> unit
(** [pp ppf b] formats [b] as [0x] followed by two lowercase hexadecimal digits
    per byte, first byte first: [0x48690a]. The empty byte string formats as
    [0x]. *)
