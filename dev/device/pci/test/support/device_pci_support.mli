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

val is_pow2 : int -> bool
(** [is_pow2 n] is [true] iff [n] is a positive power of two. *)

val pow2_floor : int -> int
(** [pow2_floor n] is the largest power of two not above [n], for [n >= 1]. *)

val pp_hex : Format.formatter -> int -> unit
(** [pp_hex] prints an integer in hexadecimal, as [0x1f]. *)

val hex : int Windtrap.Testable.t
(** [hex] is integers printed in hexadecimal and ordered as numbers. *)

val free_base : int
(** [free_base] starts a range of this process's addresses far from what the
    runtime maps, which a test may reserve. *)

val largest_gap : int -> int -> (int * int) list -> int
(** [largest_gap lo hi live] is the length of the longest run of addresses from
    [lo] up to [hi] that no [(address, length)] range of [live] covers. *)

val fits : gap:int -> int -> int -> bool
(** [fits ~gap n a] is [2 * (n + a) <= gap], without overflow. *)

val on_linux : bool
(** [on_linux] is [true] iff this machine has [/sys/bus/pci]. *)

val patience : float
(** [patience] is 10 s, how long a test waits for what must happen: the
    [~timeout] of each suite's top groups, and {!poll}'s bound. *)

val poll : (unit -> bool) -> bool
(** [poll f] calls [f] until it is [true] or {!patience} passed, and is [true]
    iff [f] became [true]: the hang guard of a test that waits for another
    domain. *)

val now_ns : unit -> int
(** [now_ns ()] is the monotonic clock in nanoseconds. *)

(** {1:host This machine's GPUs}

    Taking a GPU of this machine is a hardware opt-in. A test that takes a
    function of this machine takes only GPUs, and only while it holds the
    machine's GPU lock, so that it never holds another user's device. *)

val host_gpus : unit -> Device_pci.Machine.id list
(** [host_gpus ()] is this machine's display controllers, class [0x03]. *)

val with_gpu_lock : (unit -> 'a) -> 'a
(** [with_gpu_lock f] is [f ()] while the process holds this machine's GPU lock:
    [flock] on the file the variable [DEVICE_PCI_TEST_GPU_LOCK] names. It skips
    the running test if the variable is unset or another process holds the lock.
*)

(** {1:memory Process memory and far machines}

    A far machine holds the [size] bytes at its addresses from [base] on, and
    nothing else, reached through a transport whose C structure is at the
    machine's address. Its transport logs every access, fails once the machine
    is broken, and fails an access outside its bytes. Held, its accesses block
    until it is let go, at most 10 s, as a link's round trip does. *)

val memory : int -> int
(** [memory n] is the address of [n] new zeroed bytes of the process, never
    freed. *)

val far : int -> int -> int
(** [far base size] is a new far machine, as the address of its transport. *)

val break : int -> unit
(** [break far] fails [far]'s transport with the reason ["far: the link broke"].
*)

val break_at : int -> int -> unit
(** [break_at far k] breaks [far] at its access [k] from now on, counting from
    0: [k] accesses succeed, and from the next on [far] is broken. *)

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
    5 snooped, bits 6-11 the fragment, bits 12-51 the address, bits 52-55 a
    peer's number. *)
module Tables : sig
  type memory = {
    entries : (int, int64) Hashtbl.t;  (** Entries by physical address. *)
    mutable zeroed : (int * int) list;  (** [zero] calls, newest first. *)
    mutable unflushed : int;  (** Entries written since the last [flush]. *)
    mutable touches : int;  (** Entries written, zeroes and flushes. *)
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

(** {1:buffer Page tables in a buffer} *)

(** Page tables of a GPU of 1 GiB, in a buffer, as a bench times them.

    The tables come from a pool: 1 MiB of boot memory, then 2 MiB of tables,
    both in the buffer, which keeps nothing past them. Pages map at 2 MiB and 4
    KiB. An entry holds the address, bit 0 valid and bit 1 a table. *)
module Buffer_tables : sig
  val create : Device_pci.Space.t -> Device_pci.Page_table.t
  (** [create s] is booted page tables whose virtual addresses come from [s]. *)
end

(** {1:hosts Hosts in a fixture tree} *)

(** A host's files as Linux shows them, written under the test's own directory,
    for {!Device_pci.Machine.at}.

    A function's directory holds [vendor], [device] and [class] in hexadecimal,
    [enable], [resource] (one line per BAR: start, end and flags), the first 64
    bytes of [config] with its identity and BAR registers, for each memory BAR
    [N] a [resourceN] of 4 KiB of zeroes, and a [resourceN_wc] if it is
    prefetchable, an empty [remove], and the links [driver] and [iommu_group]
    when it has them. A driver's directory holds empty [bind] and [unbind], the
    bus directory empty [rescan] and [drivers_probe], and an IOMMU group's
    directory its [type] and its functions. The host takes what a change writes
    as plain files and acts on none of it. Names hold [:], so trees are written
    only where the file system allows it: {!make} skips the test on Windows. *)
module Host : sig
  (** The type for a BAR, in BAR order from BAR 0. *)
  type bar =
    | Mem32 of int * int  (** A 32-bit memory BAR: bus address, bytes. *)
    | Mem64 of int * int
        (** A 64-bit memory BAR, prefetchable, which takes two indices. *)
    | Io of int * int  (** An I/O BAR. *)

  type fn = {
    bus : string;
    vendor : int;
    device : int;
    class_ : int;  (** The base class, such as [0x03]. *)
    driver : string option;
    group : string option;  (** Its IOMMU group. *)
    enabled : bool;
    bars : bar list;
  }
  (** The type for a function of a host. *)

  val gpu : ?driver:string -> ?group:string -> ?enabled:bool -> string -> fn
  (** [gpu bus] is an AMD display controller at [bus], enabled and bound to no
      driver, with a 64-bit BAR 0 of 256 MiB at [0x7c_0000_0000], a 64-bit BAR 2
      of 2 MiB at [0xfc00_0000], an I/O BAR 4 of 256 bytes at [0xe000] and a
      32-bit BAR 5 of 1 MiB at [0xfcc0_0000]. *)

  val make :
    ?lockdown:string ->
    ?groups:(string * string) list ->
    ?noiommu:string list ->
    fn list ->
    string
  (** [make fns] is the root of a new host whose functions are [fns]. [groups]
      gives each IOMMU group's type (defaults to [[]], a group whose type is not
      given is ["DMA"]), [noiommu] the groups VFIO's no-IOMMU mode holds, and
      [lockdown] the kernel's lockdown file (defaults to
      ["[none] integrity confidentiality"]). *)
end
