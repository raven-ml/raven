(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Refusals, named after the function the user called. *)

val reading : by:string -> (('a -> 'b) -> 'b) -> ('a -> 'b) -> 'b
(** [reading ~by read f] is [read f], whose refusal before it calls [f], an
    [Invalid_argument], names [by] in place of the function that raised it. What
    [f] raises passes unchanged. *)
