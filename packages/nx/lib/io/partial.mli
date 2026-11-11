(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Partial files. {!Nx_io.Partial} documents them. *)

type t

val v : name:string -> length:int -> t
val add : t -> at:int -> string -> t

val name : t -> string
(** [name p] is [p]'s name. *)

val length : t -> int
(** [length p] is the length of the file [p] holds part of. *)

val lacking : t -> pos:int -> int -> (int * int) list
(** [lacking p ~pos n] is the ranges of the bytes [pos] to [pos + n - 1]
    that [p] does not hold, in order, each as long as it can be. *)

val bytes : t -> pos:int -> int -> Storage.bytes
(** [bytes p ~pos n] is the bytes [pos] to [pos + n - 1], all of which [p]
    must hold: a piece of [p]'s own memory where one piece holds them, and a
    copy otherwise. They must not be written. *)

val string : t -> pos:int -> int -> string
(** [string p ~pos n] is {!bytes} as a string. *)

val ranges : (int * int) list -> (int * int) list
(** [ranges rs] is the bytes of the ranges [rs] as ranges sorted by offset,
    disjoint, merged where they touch or overlap, none empty. *)
