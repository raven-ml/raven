(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** String literals for debugging printers. *)

val pp : Format.formatter -> string -> unit
(** [pp ppf s] formats [s] as an OCaml string literal of its bytes that keeps
    its characters: valid UTF-8 is written as is, except that double quotes,
    backslashes and the control characters U+0000 to U+001F and U+007F to U+009F
    are escaped as [%S] escapes them, as are the bytes of malformed UTF-8. Line
    breaking counts a character as one column. *)
