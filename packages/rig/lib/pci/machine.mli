(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Machines this process reaches, and their PCI functions.

    A machine is a computer whose PCI functions this process reaches. It is
    this one ({!this}), or another one that a transport reaches ({!make}). Every
    operation of this library works on either machine, and a driver never
    learns which.
    This library knows no transport: another library implements one by giving
    the operations of {!ops}.

    A machine lists its functions ({!val-functions}) without changing anything.
    {!Function.take} takes one of them for the process, and
    {!val-reserve} sets aside the process's addresses that the functions'
    system memory goes at. *)

(** {1:addresses Bus addresses}

    A function's bus address names it on its machine, as Linux spells it:
    ["DDDD:BB:DD.F"], its domain, bus, device and function numbers in lowercase
    hexadecimal, the domain in at least four digits. *)

val compare_address : string -> string -> int
(** [compare_address a b] orders bus addresses in bus order: by domain, then
    bus, device and function, each as a number.

    Raises [Invalid_argument] if [a] or [b] is not a bus address. *)

(** {1:machines Machines} *)

type t
(** The type for machines. *)

val this : t
(** [this] is the machine this process runs on. *)

val name : t -> string option
(** [name m] is the name of [m] as its transport gives it, such as
    ["host:7000"], or [None] if [m] is reached without a transport. *)

val failed : t -> string option
(** [failed m] is the reason [m] can no longer be reached, if it cannot. A
    machine that failed stays failed. A machine reached without a transport
    never fails. *)

val page : t -> int
(** [page m] is the page size of [m] in bytes. *)

type id = Ops.id = {
  bus : string;  (** Its bus address. *)
  vendor : int;  (** Its vendor's identifier, such as [0x1002] for AMD. *)
  device : int;  (** Its device identifier, assigned by its vendor. *)
  class_ : int;
      (** Its 24-bit class code: base class, subclass and programming interface,
          such as [0x030000] for a VGA controller. *)
}
(** The type for the identity of a function. *)

val functions : t -> (id list, string) result
(** [functions m] is the PCI functions of [m], in bus order
    ({!compare_address}). Listing them changes nothing on [m]. For a machine
    without a transport it is [Ok []] where the system has no
    [/sys/bus/pci], and a function whose files cannot be read, as while the
    kernel removes it, is left out.

    [Error why] if the functions cannot be listed, such as when the process
    has no file left or the transport has failed. *)

val reserve : t -> base:int -> int -> (unit, string) result
(** [reserve m ~base n] reserves the [n] addresses from [base] on, in the
    address space of the process that holds [m]'s functions, so that only
    {!Function.alloc_dma} maps memory there. A range stays reserved while that
    process runs, and reserving it again does nothing.

    [Error why] if part of the range is in use, or if [m] is reached without a
    transport and its system is not Linux. *)

(** {1:routes Routes}

    A machine's bus carries one function's requests to another function's
    BAR, as from a network adapter to a GPU's memory, where Linux's p2pdma
    allows it ({!Memory.reaches} states the rule). The values below read each
    function's way up to its host bridge from the machine's files. A function
    is {e unread} if its bus is not a bus address, if it names no function of
    the machine, or if its files cannot be read, and every function of a
    machine reached through a transport is. *)

val forwards : t -> provider:string -> client:string -> bool
(** [forwards m ~provider ~client] is [true] iff the bus of [m] carries the
    requests of the function at [client] to a BAR of the function at
    [provider], as {!Memory.reaches} states Linux's p2pdma decides, for a
    process that reads every bridge's ACS controls. It is [false] if either
    function is unread.

    A process without [CAP_SYS_ADMIN] cannot read the ACS controls past the
    first 64 bytes of configuration space, so for it [forwards] answers as
    the root complex's rule alone decides: [false] also where Linux carries
    the requests below a common bridge. *)

(** The type for how far apart two functions of a machine are, nearest
    first. *)
type distance =
  | Switch
      (** Below a bridge that both ways up to the host bridge pass, such as a
          switch's port or a root port. *)
  | Bridge  (** Below one host bridge, with no such bridge. *)
  | System  (** Below two host bridges. *)

val distance : t -> string -> string -> distance option
(** [distance m a b] is how far apart the functions at [a] and [b] of [m]
    are, or [None] if either is unread. *)

(** {1:transports Transports}

    A library that reaches another machine makes it a machine with {!make},
    from the operations below. A transport runs each operation on the other
    machine as the same operation of {!this} runs here. A request it cannot run
    returns [Error why], with [why] starting with the machine's name, and so
    does the reason its transport failed ({!failed}). {!Window} states what
    accesses give once the transport has failed.

    {!Function} refuses misuse before it calls an operation, and counts the
    windows and pins of each function. An operation is called only with
    arguments {!Function} accepts. The size of [alloc_dma] is rounded up to
    {!val-page}, and its [va] lies inside a range {!val-reserve} reserved. A
    pin's range starts on a page. Once {!Function.release} has begun on a
    function, no use of it ({!Function}) starts, and the uses running finish
    before the give-back's own [unmap], [config], [set_config] and [release].
    The release, the last use, a use refused after the release or
    {!Function.finish_releases} makes them, never two at once. [free_dma] and
    [unpin] are still called on a released function, once its give-back has
    turned its bus mastering off, outside any use. An exception, such as an
    asynchronous one, may end [unmap] or [release] part-way, and {!Function}
    then calls it again with the same arguments. Every pin and unpin is passed
    on, so a transport counts them too.

    A use's operations run inside a lock of the calling thread
    ({{!Rig_pci.errors}errors}), as the give-back's do: [alloc_dma], [pin],
    [free_dma] and [unpin] but those above, the [config] of a wait's bound, and
    those of a request of {!Memory} that writes a GPU's tables. Each must call
    nothing of rig's or of a driver's, and must return on the thread that
    called it. *)

(** The type for how a function reaches system memory. *)
type addressing = Ops.addressing =
  | Physical
      (** At physical addresses of its machine, which no IOMMU translates. *)
  | Iommu
      (** Through an IOMMU, at device addresses the process maps for it alone.
      *)

type fn = Ops.fn = {
  addressing : addressing;  (** How it reaches system memory. *)
  inherited : bool;
      (** Whether, when it was taken, the function still reached memory that
          a dead process left, which it may still write. It is read once,
          at the take. *)
  config : int -> int -> int;
      (** [config off n] is the [n]-byte value at byte [off] of its
          configuration space, for [n] of 1, 2 or 4. Only {!Function}'s
          accessors call it. *)
  set_config : int -> int -> int -> unit;
      (** [set_config off n x] writes the low [n] bytes of [x] at byte [off],
          for [n] of 1, 2 or 4. Only {!Function}'s accessors call it. *)
  bar : int -> (int * int) option;  (** {!Function.bar}. *)
  map : combine:bool -> int -> int -> int -> (Window.t, string) result;
      (** [map ~combine i off n] is {!Function.map} of BAR [i]. *)
  unmap : Window.t -> unit;
      (** {!Function.unmap}. It must raise only where an exception cuts it.
          Called again on a window whose unmap an exception ended part-way, it
          must unmap what remains of the window and nothing twice. *)
  interrupt : int -> bool;  (** {!Function.interrupt}. *)
  reset : unit -> (unit, string) result;
      (** {!Function.reset}, without waiting for the function to answer. *)
  forget : unit -> (unit, string) result;
      (** [forget ()] records that the function's device was reset, so that it
          reaches none of the memory that dead processes left. That
          memory goes once no other function reaches it. *)
  alloc_dma :
    contiguous:bool ->
    va:int option ->
    below:int ->
    int ->
    ((Window.t * (int * int) list) option, string) result;
      (** {!Function.alloc_dma}, whose runs end at or below [below] where the
          machine chooses their addresses, as behind an IOMMU
          ({!Function.set_dma_bits}). *)
  free_dma : Window.t -> unit;  (** {!Function.free_dma}. *)
  pin : below:int -> int -> int -> ((int * int) list option, string) result;
      (** {!Function.pin}, as [alloc_dma] places its runs. *)
  unpin : int -> int -> unit;  (** {!Function.unpin}. *)
  release : unit -> bool;
      (** {!Function.release}, which turns the function's bus mastering off:
          [false] if the memory the function reached may still be written once
          given back, as when the system refused the write that turns its bus
          mastering off. It must raise only where an exception cuts it.
          Called again after an exception ended it part-way, it must give
          back what it had not, and nothing twice. *)
}
(** The type for the operations on a function that a transport took. *)

type ops = Ops.ops = {
  transport : Window.transport;
      (** The C accesses to the machine's addresses. The machine's windows use
          them, and their [failed] function is {!failed}'s. *)
  page : int;  (** {!val-page}. *)
  functions : unit -> (id list, string) result;
      (** {!val-functions}, in any order. *)
  take : string -> (fn, string) result;  (** {!Function.take} on the machine. *)
  reserve : base:int -> int -> (unit, string) result;  (** {!val-reserve}. *)
}
(** The type for the operations of a transport. *)

val make : name:string -> ops -> t
(** [make ~name ops] is the machine named [name] that [ops] reaches. *)

(**/**)

(* [address ~domain ~bus ~device ~fn] is the bus address of function [fn] of
   device [device] on bus [bus] of domain [domain], such as ["0000:03:00.0"]. *)
val address : domain:int -> bus:int -> device:int -> fn:int -> string

(* [at root] is this machine as the directory [root] shows it, {!this} being
   [at "/"]. Its functions are read under [root/sys/bus/pci], its IOMMU groups
   under [root/sys/kernel/iommu_groups], its lockdown state at
   [root/sys/kernel/security/lockdown], its VFIO files under [root/dev/vfio],
   and the memory of functions taken physically under [root/dev/hugepages],
   [root/proc/self/mounts] and [root/proc/self/pagemap]. {!Driver.detach} also
   reads its processes under [root/proc] and its DRM clients under
   [root/sys/kernel/debug/dri]. A root that shows no PCI functions makes a
   machine with none.

   Each call makes another machine. Two machines at one root share no holds
   ({!Driver}), and the kernel still refuses a second take of one function.

   [kernel file s] is called at each write of [s] that the machine makes to
   an attribute [file] of a driver, the bus or a function under
   [root/sys/bus/pci], such as [unbind], [drivers_probe], [remove] or
   [driver_override]. A function's [config] is written without it. It is
   called once [file] is open and before [s] is written, to act on it as
   Linux does inside the write, such as unbinding a driver. It may raise
   [Unix.Unix_error] as Linux's refusal of the write, and [s] is then not
   written. The open truncated [file], so a fixture's kernel that refuses
   restores [file] itself. It defaults to doing nothing, as for {!this},
   whose kernel acts itself. *)
val at : ?kernel:(string -> string -> unit) -> string -> t

val window : t -> int -> int -> Window.t
(** [window m a n] is the [n] bytes at address [a] on [m], reached through [m]'s
    transport. It neither maps nor keeps the memory; the caller keeps it valid.
    Raises [Invalid_argument] if [n < 0]. *)

(* [take m bus] is [m]'s take of the function at [bus], or [Error] if [bus]
   is not a bus address: {!Function.take} builds on it. *)
val take : t -> string -> (fn, string) result

(* [take_locked m l] is [m]'s take of the function [l] locks, under [l]:
   {!Function.take_locked} builds on it. Raises [Invalid_argument] if [l] is not
   of [m]'s {!files}. *)
val take_locked : t -> 's Sysfs.lock -> (fn, string) result

(* [reserved m a n] is [true] iff the [n] bytes at [a] lie in one range
   {!val-reserve} reserved on [m]. *)
val reserved : t -> int -> int -> bool

(* [files m] is the files of [m] if the process reaches [m] without a transport:
   {!this}'s, or those of a machine {!at} made. {!Driver} reads them, to lock a
   function across its checks, its take and its changes to kernel drivers, and
   {!Memory} the way up the bus from a GPU. *)
val files : t -> Sysfs.t option

(* [wait m ~us f] is the loop of {!Function.wait} on [m], which builds on it:
   it calls [f] until it is [true], [m] failed, or at least [us] microseconds
   passed, and is [true] iff [f] became [true] while [m] had not failed. *)
val wait : t -> us:int -> (unit -> bool) -> bool
