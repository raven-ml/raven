(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Bus addresses, ["DDDD:BB:DD.F"]. {!Machine} exports them. *)

val v : domain:int -> bus:int -> device:int -> fn:int -> string
(** [v ~domain ~bus ~device ~fn] is {!Machine.address}. *)

val numbers : string -> (int * int * int * int) option
(** [numbers a] is the domain, bus, device and function numbers of [a], or
    [None] if [a] is no bus address. *)

val compare : string -> string -> int
(** [compare a b] is {!Machine.compare_address}. *)
