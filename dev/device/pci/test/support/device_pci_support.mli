(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What pci's suites and benches share. *)

(** {1:sizes Sizes and addresses} *)

val kib : int
val mib : int
val gib : int

val round_up : int -> int -> int
(** [round_up n a] is the least multiple of [a] at or above [n], for [n >= 0]
    and [a > 0]. *)

val pp_hex : Format.formatter -> int -> unit
(** [pp_hex] prints an integer in hexadecimal, as [0x1f]. *)

val hex : int Windtrap.Testable.t
(** [hex] is integers printed in hexadecimal and ordered as numbers. *)

val on_linux : bool
(** [on_linux] is [true] iff this machine has [/sys/bus/pci]. *)

val now_ns : unit -> int
(** [now_ns ()] is the monotonic clock in nanoseconds. *)

(** {1:memory Process memory and far machines}

    A far machine holds the [size] bytes at its addresses from [base] on, and
    nothing else, reached through a transport whose C structure is at the
    machine's address. Its transport logs every access, fails once the machine
    is broken, and fails an access outside its bytes. Held, its accesses block
    until it is let go, at most 2 s, as a link's round trip does. *)

val memory : int -> int
(** [memory n] is the address of [n] new zeroed bytes of the process, never
    freed. *)

val far : int -> int -> int
(** [far base size] is a new far machine, as the address of its transport. *)

val break : int -> unit
(** [break far] fails [far]'s transport with the reason ["far: the link broke"].
*)

val hold : int -> unit
(** [hold far] makes [far]'s accesses wait until {!let_go}. *)

val waiting : int -> bool
(** [waiting far] is [true] iff an access of [far] waits on its hold. *)

val let_go : int -> unit
(** [let_go far] ends {!hold}. *)

val log : int -> (bool * int * int) list
(** [log far] is [far]'s accesses since the last call, oldest first, as
    [(write, address, bytes)]. *)
