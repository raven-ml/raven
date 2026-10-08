(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A GPU's registers, accessed by name at the bases its discovery table gives:
   through the register BAR, past the BAR's end through the RSMU index window,
   and, on a virtual function, those the RLC guards through its gateway. GC's
   registers are Device_amd_abi.Register's; the other blocks' are this library's
   tables, for the version of each block.

   The GPU's owner serializes calls. *)

type t

(* Raised when the GPU does not answer in time, or its function or machine
   failed, with the step and the reason. An open or a reset returns it as its
   [Error]; a sleep raises it as [Device_amd.Fault]. *)
exception Stuck of string

(* [make f mmio d ~vf] is the registers of the GPU of [f], whose register BAR is
   mapped as [mmio], described by [d]; [vf] if [f] is a virtual function. *)
val make :
  Device_pci.Function.t -> Device_pci.Window.t -> Discovery.t -> vf:bool -> t

val fn : t -> Device_pci.Function.t
val discovery : t -> Discovery.t
val vf : t -> bool

(* [version r b] is the version of block [b]. Raises [Invalid_argument] if the
   GPU has no block [b], which Discovery.supported refuses first. *)
val version : t -> int -> Discovery.version

(* [has r name] is [true] iff the GPU has the register [name]. *)
val has : t -> string -> bool

(* [address ~inst r name] is the address of instance [inst] (defaults to [0]) of
   the register [name], in 32-bit words. A name the tables lack, or an instance
   the GPU lacks, raises [Invalid_argument]: the library's tables miss a
   register of a GPU it boots. *)
val address : ?inst:int -> t -> string -> int
val read : ?inst:int -> t -> string -> int

(* [write ~inst ~value r name fs] writes [value] (defaults to [0]) with the
   fields [fs] set, as Device_amd_abi.Register.encode does. *)
val write :
  ?inst:int -> ?value:int -> t -> string -> (string * int) list -> unit

(* [update ~inst r name fs] sets the fields [fs] of the register's value, its
   other bits kept. *)
val update : ?inst:int -> t -> string -> (string * int) list -> unit
val field : ?inst:int -> t -> string -> string -> int
val fields : ?inst:int -> t -> string -> (string * int) list

(* [write64 ~inst r base ~lo ~hi x] writes the low and high 32 bits of [x] to
   the registers [base ^ lo] and [base ^ hi]. *)
val write64 : ?inst:int -> t -> string -> lo:string -> hi:string -> int -> unit

(* [get r a] and [set r a x] access the register at address [a], in 32-bit
   words. *)
val get : t -> int -> int
val set : t -> int -> int -> unit

(* [set_pcie ~aid r a x] writes [x] to the PCIe register [a] of the accelerator
   die [aid] (defaults to [0]) through the indirect window. The caller never
   names a die fused off: a write there stalls the fabric. *)
val set_pcie : ?aid:int -> t -> int -> int -> unit

(* [wait ~ms r what f] returns once [f ()] is [true], polled for at most [ms]
   milliseconds (defaults to 10,000). Raises [Stuck] naming [what] otherwise,
   with the function's failure if it failed. *)
val wait : ?ms:int -> t -> string -> (unit -> bool) -> unit

(* [pause r ms] waits [ms] milliseconds, the time a block needs with no state to
   poll. *)
val pause : t -> int -> unit
