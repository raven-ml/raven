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

val failed : ?substring:string -> exn -> bool
(** [failed ?substring e] is [true] iff [e] is [Device_pci.Failed why] and [why]
    contains [substring], when given: the predicate of [Windtrap.raises_match]
    for the world's failures. *)

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

(** {1:tables Page tables in a fake format} *)

(** Page tables in a fake format, in a fake GPU memory that keeps their entries
    by address.

    Four levels of 512 entries, the root numbered 0: level [l] indexes the bits
    from [shifts.(l)] on. Pages map at levels 1 to 3: 1 GiB, 2 MiB and 4 KiB.
    Entries: bit 0 valid, bit 1 a page, bits 2-3 the target, bit 4 uncached, bit
    5 snooped, bits 6-11 the fragment, bits 12-51 the address. *)
module Tables : sig
  type memory = {
    entries : (int, int64) Hashtbl.t;  (** Entries by physical address. *)
    mutable zeroed : (int * int) list;  (** [zero] calls, newest first. *)
    mutable unflushed : int;  (** Entries written since the last [flush]. *)
    mutable touches : int;  (** Entries read and written, zeroes and flushes. *)
  }
  (** The type for a GPU memory that holds page tables. *)

  val memory : unit -> memory
  (** [memory ()] is a memory that holds no entry. *)

  val format : memory -> Device_pci.Page_table.format
  (** [format m] is the format, its entries in [m]. *)

  val shifts : int array
  (** [shifts.(l)] is the lowest bit of a virtual address level [l] indexes. *)

  val address_mask : int
  (** [address_mask] is an entry's address bits. *)

  val leaf : int
  (** [leaf] is the level of the smallest pages. *)

  val large : int -> bool
  (** [large l] is [true] iff pages map at level [l]. *)

  type entry = {
    va : int;
    level : int;
    pa : int;
    target : Device_pci.Page_table.target;
    uncached : bool;
    snooped : bool;
    fragment : int;
  }
  (** The type for an entry that maps a page. *)

  val pp_target : Format.formatter -> Device_pci.Page_table.target -> unit
  val target : Device_pci.Page_table.target Windtrap.Testable.t
  val pp_entry : Format.formatter -> entry -> unit
  val entry : entry Windtrap.Testable.t

  val walk : memory -> Device_pci.Page_table.t -> entry list * int list
  (** [walk m t] is the pages [t] maps in [m], by virtual address, and the
      tables reached from the root, root first. It touches nothing. *)
end
