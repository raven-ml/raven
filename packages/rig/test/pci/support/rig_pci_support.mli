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
(** [free_base] starts a range of this process's addresses, at 1 TiB, far from
    what the runtime maps and left to the program under AddressSanitizer and
    ThreadSanitizer, which a test may reserve. *)

val largest_gap : int -> int -> (int * int) list -> int
(** [largest_gap lo hi live] is the length of the longest run of addresses from
    [lo] up to [hi] that no [(address, length)] range of [live] covers. *)

val fits : gap:int -> int -> int -> bool
(** [fits ~gap n a] is [2 * (n + a) <= gap], without overflow. *)

val on_linux : bool
(** [on_linux] is [true] iff this machine has [/sys/bus/pci]. *)

val flocks : bool
(** [flocks] is [true] iff the system locks a function's file with flock and
    maps its BARs, as every system but Windows does. System memory for a
    function needs Linux ({!on_linux}). *)

val ofd_locks : bool
(** [ofd_locks] is [true] iff the system has open file description locks,
    which tell a memory file's process alive, as Linux and macOS do. *)

val patience : float
(** [patience] is 10 s, how long a test waits for what must happen: the
    [~timeout] of each suite's top groups, and {!poll}'s bound. It is
    multiplied by the factor [RIG_TEST_TIME_SCALE] names, as
    [Rig_support.time] multiplies. *)

val poll : (unit -> bool) -> bool
(** [poll f] calls [f] until it is [true] or {!patience} passed, and is [true]
    iff [f] became [true]: the hang guard of a test that waits for another
    domain. *)

val now_ns : unit -> int
(** [now_ns ()] is the monotonic clock in nanoseconds. *)

(** {1:this This machine's GPUs}

    A test that takes a function of this machine takes only GPUs, so that it
    never holds another user's device. *)

val this_gpus : unit -> Rig_pci.Machine.id list
(** [this_gpus ()] is this machine's display controllers, base class [0x03]. *)

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

val unlisted_why : string
(** [unlisted_why] is ["far: the link broke"]. *)

val unlisted : unit -> Rig_pci.Machine.t
(** [unlisted ()] is a machine reached through a broken far machine, whose
    listing of its functions answers [Error unlisted_why], as a transport
    that failed does. Its take and reserve fail the test. *)

(** {1:tables Page tables in a fake format} *)

(** Page tables in a fake format, in a fake GPU memory that keeps their entries
    by address.

    Four levels of 512 entries, the root numbered 0: level [l] indexes the bits
    from [shifts.(l)] on, and entries hold physical addresses of 52 bits. Pages map at levels 1 to 3: 1 GiB, 2 MiB and 4 KiB.
    Entries: bit 0 valid, bit 1 a page, bits 2-3 the target, bit 4 uncached, bit
    5 snooped, bits 6-11 the fragment, bits 12-51 the address, bits 52-55 a
    peer's number. *)
module Tables : sig
  type memory = {
    entries : (int, int64) Hashtbl.t;  (** Entries by physical address. *)
    mutable zeroed : (int * int) list;  (** [zero] calls, newest first. *)
    mutable unflushed : int;  (** Entries written since the last [flush]. *)
    mutable touches : int;  (** Entries written, zeroes and flushes. *)
    mutable confirms : bool;
        (** What [flush] answers: whether the GPU confirmed it. *)
  }
  (** The type for a GPU memory that holds page tables. *)

  val memory : unit -> memory
  (** [memory ()] is a memory that holds no entry, whose flushes confirm. *)

  val format : memory -> Rig_pci.Page_table.format
  (** [format m] is the format, its entries in [m]. *)

  val shifts : int array
  (** [shifts.(l)] is the lowest bit of a virtual address level [l] indexes. *)

  val leaf : int
  (** [leaf] is the level of the smallest pages. *)

  val large : int -> bool
  (** [large l] is [true] iff pages map at level [l]. *)

  type entry = {
    va : int;
    level : int;
    pa : int;
    target : Rig_pci.Page_table.target;
    uncached : bool;
    snooped : bool;
    fragment : int;
  }
  (** The type for an entry that maps a page. *)

  val pp_target : Format.formatter -> Rig_pci.Page_table.target -> unit
  val target : Rig_pci.Page_table.target Windtrap.Testable.t
  val pp_entry : Format.formatter -> entry -> unit
  val entry : entry Windtrap.Testable.t

  val walk : memory -> Rig_pci.Page_table.t -> entry list * int list
  (** [walk m t] is the pages [t] maps in [m], by virtual address, and the
      tables reached from the root, root first. It touches nothing. *)
end

(** {1:buffer Page tables in a buffer} *)

(** Page tables of a GPU of 1 GiB, in a buffer, as a bench times them.

    The tables come from a pool: 1 MiB of boot memory, then 2 MiB of tables,
    both in the buffer, which keeps nothing past them. Pages map at 2 MiB and 4
    KiB. An entry holds the address, bit 0 valid and bit 1 a table. *)
module Buffer_tables : sig
  val create : Rig_pci.Space.t -> Rig_pci.Page_table.t
  (** [create s] is booted page tables whose virtual addresses come from [s]. *)
end

(** {1:trees Machines in a fixture tree} *)

(** A machine's files as Linux shows them, for {!Rig_pci.Machine.at}, written
    under a temporary directory of the process's own, which it removes at its
    exit: no two runs share a tree.

    A function's directory holds [vendor], [device] and [class] in hexadecimal,
    [enable], [resource] (one line per BAR: start, end and flags), the first 64
    bytes of [config] with its identity and BAR registers, for each memory BAR
    [N] a [resourceN] of zeroes as long as the BAR, sparse, and a
    [resourceN_wc] if it is prefetchable, an empty [remove], a
    [driver_override] that names no driver, as ["(null)"], and the links [driver] and [iommu_group] when it has them. A
    driver's directory holds empty [bind] and [unbind], the bus directory empty
    [rescan] and [drivers_probe], and an IOMMU group's directory its [type] and
    its functions. The process's [proc/self/fd] is an empty directory, to which
    a test adds links, and so is [dev/hugepages], which its [proc/self/mounts]
    shows as a hugetlbfs of 2 MiB pages. The machine takes what a
    change writes as plain files and acts on none of it. Names hold [:], so
    trees are written only where the file system allows it: {!make} skips the
    test on Windows. *)
module Tree : sig
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
    class_ : int;  (** Its 24-bit class code, such as [0x030000]. *)
    driver : string option;
    group : string option;  (** Its IOMMU group. *)
    enabled : bool;
    reset : string option;
        (** Its [reset_method], the methods Linux resets it with, in order, or
            [None] for a function without a reset. *)
    host : string;  (** Its host bridge, as [/sys/devices] names it. *)
    bridges : string list;
        (** The bridges above it, by bus address, from its host bridge's root
            bus down. Each is an AMD root port where no function of the tree
            made it first, with no ACS. *)
    bars : bar list;
  }
  (** The type for a function of a machine. *)

  val gpu :
    ?driver:string ->
    ?group:string ->
    ?enabled:bool ->
    ?reset:string option ->
    ?host:string ->
    ?bridges:string list ->
    string ->
    fn
  (** [gpu bus] is an AMD display controller at [bus], enabled and bound to no
      driver, reset through its bus ([reset] defaults to [Some "bus"]), under
      the root port [0000:00:01.0] of the host bridge [pci0000:00] ([host]
      and [bridges]), with a
      64-bit BAR 0 of 256 MiB at [0x7c_0000_0000], a 64-bit BAR 2
      of 2 MiB at [0xfc00_0000], an I/O BAR 4 of 256 bytes at [0xe000] and a
      32-bit BAR 5 of 1 MiB at [0xfcc0_0000]. *)

  val make :
    ?lockdown:string ->
    ?groups:(string * string) list ->
    ?noiommu:string list ->
    ?privileged:bool ->
    fn list ->
    string
  (** [make fns] is the root of a new tree whose functions are [fns]. [groups]
      gives each IOMMU group's type (defaults to [[]], a group whose type is not
      given is ["DMA"]), [noiommu] the groups VFIO's no-IOMMU mode holds, and
      [lockdown] the kernel's lockdown file (defaults to
      ["[none] integrity confidentiality"]). Each function's directory lies
      under its host bridge and bridges in [sys/devices], linked from
      [sys/bus/pci/devices]. Configuration files hold 64 bytes, as a reader
      without [CAP_SYS_ADMIN] sees them, or 4096 with [~privileged:true]
      (defaults to [false]). *)

  val device_number : string -> string
  (** [device_number file] is the number of the character device at [file], as a
      [dev] file under [/sys] spells it: ["MAJOR:MINOR"], such as ["1:3"] for
      Linux's [/dev/null]. Raises [Unix.Unix_error] if [file] cannot be read. *)

  val stored : string -> int
  (** [stored file] is the bytes the file system holds for [file], which its
      holes do not count. Raises [Unix.Unix_error] if [file] cannot be read. *)

  val flocked : string -> bool
  (** [flocked file] is [true] iff an open file holds flock's lock on [file],
      shared or exclusive, so that a new open of it cannot take the exclusive
      one. Raises [Unix.Unix_error] if [file] cannot be opened. *)

  val flocked_fd : Unix.file_descr -> bool
  (** [flocked_fd fd] is {!flocked} of the file [fd] is open on, asked
      through [fd], which must hold no lock of the file: it makes no open, so
      a check repeated after every call costs two system calls. *)

  val hold : string -> Unix.file_descr
  (** [hold file] takes a read lock of a new open file description over
      [file], as a memory file's live process holds one, and is the
      descriptor whose close gives it back. Raises [Unix.Unix_error] if
      [file] cannot be opened or a write lock holds it. *)

  val held : string -> bool
  (** [held file] is [true] iff an open file description holds a lock on
      [file], as its live process's read lock, tested without placing one.
      Raises [Unix.Unix_error] if [file] cannot be opened. *)

  val add : string -> string -> string -> unit
  (** [add root file s] writes [s] to [file], a path from the tree [root],
      making its directories. *)

  val pagemap : string -> page:int -> int -> int list -> unit
  (** [pagemap root ~page a frames] writes the tree's [proc/self/pagemap]
      entries of the pages from [a] on, one present page per frame of [frames],
      each page [page] bytes. The file is sparse. *)

  val link : string -> string -> string -> unit
  (** [link root file target] makes [file], a path from the tree [root], a link
      to [target], making its directories. *)

  val driver : string -> string -> string option
  (** [driver root bus] is the driver the tree [root] binds to the function at
      [bus]. *)

  (** {2:kernels Kernels}

      A tree's kernel acts on what a machine writes to the tree's
      [sys/bus/pci] as Linux does inside the write, and records each bind and
      unbind of a function. *)

  (** The type for a change to a function's driver. *)
  type change = Bind of string | Unbind of string

  type event = {
    fn : string;  (** The function whose driver changed. *)
    change : change;
    others : (string * string option) list;
        (** Each other function of its device, with its driver while the
            driver of [fn] runs: after a bind, before an unbind. *)
  }
  (** The type for a bind or an unbind. *)

  type kernel
  (** The type for a tree's kernel. *)

  val kernel :
    ?refuse:(int -> bool) ->
    ?unloaded:string list ->
    drivers:(string * string) list ->
    string ->
    kernel
  (** [kernel ~drivers root] is the kernel of the tree [root], whose functions
      at [bus] have their own driver [d] for each [(bus, d)] of [drivers],
      each loaded but those [unloaded] names (defaults to [[]]). It acts on
      writes as Linux does:
      - [drivers/D/unbind] given a function bound to [D] unbinds it, and
        refuses another with [ENODEV];
      - [drivers_probe] given an unbound function binds it to the driver its
        [driver_override] names, or to its own with none named, where that
        driver is loaded: the tree has its directory;
      - a function's [remove] unbinds and removes it, and [rescan] brings
        back each removed function with no override and probes it.

      [refuse i] is [true] for each write, counted from [0], that the kernel
      refuses with [EIO], acting on none of it (defaults to none). A write
      the kernel refuses leaves its file as it was. *)

  val load : kernel -> string -> unit
  (** [load k d] loads the driver [d], as a module's load does, and probes
      each unbound function whose [driver_override] names [d], or names none
      while [d] is its own. *)

  val machine : kernel -> Rig_pci.Machine.t
  (** [machine k] is the machine of [k]'s tree, whose writes [k] acts on. *)

  val events : kernel -> event list
  (** [events k] is the binds and unbinds [k] made, oldest first. *)

  val writes : kernel -> int
  (** [writes k] is the number of writes [k] acted on or refused. *)
end
