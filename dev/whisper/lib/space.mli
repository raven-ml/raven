(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** White space as upstream's Python reads it. *)

val is_space : Uchar.t -> bool
(** [is_space u] is [true] iff [u] is white space: Python's [str.isspace], which
    [str.strip], [str.split] and a regular expression's [\s] read. *)

val strip : string -> string
(** [strip s] is the UTF-8 text [s] without its leading and trailing white
    space, as Python's [str.strip ()]. *)
