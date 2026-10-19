(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Errors, as {!Nx_io} states them. *)

val invalid_argf : ('a, unit, string, 'b) format4 -> 'a
val failf : ('a, unit, string, 'b) format4 -> 'a

val pp_shape : int array -> string
(** [pp_shape s] is [s] as ["[2; 3]"]. *)

val file : string -> (unit -> 'a) -> 'a
(** [file path f] is [f ()], whose [Failure], [Sys_error] and [Unix.Unix_error]
    become a [Failure] whose message starts with [path]. [f] raises [Failure]
    without the path. *)

val renamed : by:string -> ('a -> 'b) -> 'a -> 'b
(** [renamed ~by f x] is [f x], whose [Invalid_argument] names [by] in place of
    the function that raised it. *)

val reading : by:string -> (('a -> 'b) -> 'b) -> ('a -> 'b) -> 'b
(** [reading ~by read f] is [read f], whose refusal before it calls [f], an
    [Invalid_argument], names [by] in place of the function that raised it.
    What [f] raises passes unchanged. *)
