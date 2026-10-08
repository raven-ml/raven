(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Machines this process reaches, and their PCI functions.

    A machine is this one ({!this}) or another one a {e transport} reaches
    ({!make}). Every function of this library works on either, and a driver
    never learns which. This library knows no transport: another library
    implements one by giving the operations of {!ops}. *)

(** {1:addresses Bus addresses}

    A function's bus address names it on its machine, as Linux spells it:
    ["DDDD:BB:DD.F"], its domain, bus, device and function numbers in lowercase
    hexadecimal, the domain in at least four digits. *)

val address : domain:int -> bus:int -> device:int -> fn:int -> string
(** [address ~domain ~bus ~device ~fn] is the bus address of function [fn] of
    device [device] on bus [bus] of domain [domain], such as ["0000:03:00.0"].
*)

val compare_address : string -> string -> int
(** [compare_address a b] orders bus addresses in bus order: by domain, then
    bus, device and function, each as a number.

    Raises [Invalid_argument] if [a] or [b] is no bus address. *)

(** {1:machines Machines} *)

type t
(** The type for machines. *)

val this : t
(** [this] is the machine this process runs on. *)

val name : t -> string option
(** [name m] is the name of [m] as its transport gives it, such as
    ["host:7000"], or [None] if [m] is {!this}. *)

val failed : t -> string option
(** [failed m] is the reason [m] can no longer be reached, if it cannot. A
    machine that failed stays failed; {!this} never fails. *)

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

val functions : t -> id list
(** [functions m] is the PCI functions of [m], in bus order
    ({!compare_address}). Listing them changes nothing on [m]. It is [[]] where
    [m] is {!this} and the system has no [/sys/bus/pci]. *)

val reserve : t -> base:int -> int -> (unit, string) result
(** [reserve m ~base n] reserves the [n] addresses from [base] on in the address
    space of the process that holds [m]'s functions, so that only
    {!Function.alloc_dma} maps memory there. A range is reserved once and stays
    reserved while that process runs; reserving it again does nothing.

    [Error why] if part of the range is in use, or if [m] is {!this} and not
    Linux. *)

val wait : t -> ms:int -> (unit -> bool) -> bool
(** [wait m ~ms f] calls [f], at least once, until it is [true], [m] failed, or
    at least [ms] milliseconds passed on a monotonic clock. It is [true] iff [f]
    became [true] while [m] had not failed: [m]'s state is read after each call
    of [f], so an answer computed from the all ones of a failed machine is not
    trusted. For its first millisecond it calls [f] back to back, relaxing the
    processor; then it sleeps 0.1 ms between calls, so that a long wait holds no
    core. Drivers wait for their devices in it. *)

(** {1:transports Transports}

    A library that reaches another machine makes it a machine with {!make}, from
    the operations below. A transport runs each operation on the other machine
    as the same operation of {!this} runs here. A request it cannot run returns
    [Error why], [why] starting with the machine's name, as does the reason its
    transport failed ({!failed}); once it failed, its accesses read all ones and
    drop writes, and raise nothing.

    {!Function} refuses misuse before an operation is called, and counts the
    windows and pins of each function. An operation is called only with
    arguments {!Function} accepts, with sizes of system memory rounded up to
    {!page} and addresses of memory inside ranges {!reserve} reserved, and on a
    released function only [free_dma], [unpin] and one [release] are. Every pin
    and unpin is passed on, so a transport counts them too. *)

(** The type for how a function reaches system memory. *)
type addressing = Ops.addressing =
  | Physical
      (** At physical addresses of its machine, which no IOMMU translates. *)
  | Iommu
      (** Through an IOMMU, at device addresses the process maps for it alone.
      *)

type fn = Ops.fn = {
  addressing : addressing;  (** How it reaches system memory. *)
  config8 : int -> int;  (** {!Function.config8}. *)
  config16 : int -> int;  (** {!Function.config16}. *)
  config32 : int -> int;  (** {!Function.config32}. *)
  set_config8 : int -> int -> unit;  (** {!Function.set_config8}. *)
  set_config16 : int -> int -> unit;  (** {!Function.set_config16}. *)
  set_config32 : int -> int -> unit;  (** {!Function.set_config32}. *)
  bar : int -> (int * int) option;  (** {!Function.bar}. *)
  map : combine:bool -> int -> int -> int -> (Window.t, string) result;
      (** [map ~combine i off n] is {!Function.map} of BAR [i]. *)
  unmap : Window.t -> unit;  (** {!Function.unmap}. *)
  interrupt : int -> bool;  (** {!Function.interrupt}. *)
  reset : unit -> (unit, string) result;
      (** {!Function.reset}, without waiting for the function to answer. *)
  alloc_dma :
    contiguous:bool ->
    va:int option ->
    int ->
    ((Window.t * (int * int) list) option, string) result;
      (** {!Function.alloc_dma}. *)
  free_dma : Window.t -> unit;  (** {!Function.free_dma}. *)
  pin : int -> int -> ((int * int) list, string) result;
      (** {!Function.pin}. *)
  unpin : int -> int -> unit;  (** {!Function.unpin}. *)
  release : unit -> unit;
      (** {!Function.release}, which turns the function's bus mastering off. *)
}
(** The type for the operations on a function a transport took. *)

type ops = Ops.ops = {
  transport : Window.transport;
      (** The C accesses to the machine's addresses, which its windows use and
          whose [failed] function is {!failed}'s. *)
  page : int;  (** {!page}. *)
  functions : unit -> id list;  (** {!functions}, in any order. *)
  take : string -> (fn, string) result;  (** {!Function.take} on the machine. *)
  reserve : base:int -> int -> (unit, string) result;  (** {!reserve}. *)
}
(** The type for the operations of a transport. *)

val make : name:string -> ops -> t
(** [make ~name ops] is the machine named [name] that [ops] reaches. *)

(**/**)

(* [take m bus] is [m]'s take: {!Function.take} builds on it. *)
val take : t -> string -> (fn, string) result

(* [reserved m a n] is [true] iff the [n] bytes at [a] lie in one range
   {!reserve} reserved on [m]. *)
val reserved : t -> int -> int -> bool

(* [files m] is the files of [m] if the process reaches [m] without a transport:
   {!this}'s, or those of a machine {!at} made. *)
val files : t -> Sysfs.t option

(* [at root] is this machine as the directory [root] shows it: its functions
   under [root/sys/bus/pci], its VFIO files under [root/dev/vfio]. {!this} is
   [at "/"]. Tests run it on a fixture tree. *)
val at : string -> t
