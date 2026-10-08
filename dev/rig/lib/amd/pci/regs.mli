(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A GPU's registers, by name.

    A register is a 32-bit word of fields at an offset in one of its block's
    segments, whose base the GPU's discovery table gives per instance. GC's
    registers are {!Rig_amd_abi.Register}'s; the other blocks' are this
    library's tables, for each block at the latest version with tables at or
    before its own of the same major. A {!layout} places them, purely; a {!t}
    reads and writes them: through the register BAR, past the BAR's end through
    the RSMU index window, and, on a virtual function, those the RLC guards
    through its gateway.

    The GPU's owner serializes calls on a {!t}. *)

(** {1:layouts Layouts} *)

type layout
(** The type for the registers of one GPU, placed. *)

val layout : Discovery.t -> (layout, string) result
(** [layout d] places the registers of the GPU [d] describes. [Error msg] if a
    block a boot programs (GC, SDMA, MP0, MP1, MMHUB, OSSSYS, NBIO, HDP) is
    missing or has a version with no table, as
    ["MMHUB 1.7.0 is a version this library does not boot"]. *)

val gpu : layout -> Rig_amd_abi.Gpu.t
(** [gpu l] is the GPU as its formats depend on it: its compiler target (GC
    9.4.3 runs gfx942 code), its GC and SDMA versions, its live dies, and its
    GC's shape. *)

val discovery : layout -> Discovery.t
(** [discovery l] is the table [l] was placed from. *)

val version : layout -> int -> Discovery.version
(** [version l b] is the version of block [b], which {!layout} checked. *)

val has : layout -> string -> bool
(** [has l name] is [true] iff the GPU has the register [name]. *)

val register : layout -> string -> Rig_amd_abi.Register.t
(** [register l name] is the register [name].

    Raises [Invalid_argument] if the GPU has none: the tables lack a register
    the library programs. *)

val address : ?inst:int -> layout -> string -> int
(** [address ~inst l name] is the address of instance [inst] (defaults to [0])
    of the register [name], in 32-bit words from the start of the register
    space.

    Raises [Invalid_argument] as {!register}, or if the block has no instance
    [inst]. *)

val guarded : layout -> (int * int) list
(** [guarded l] is the ranges of addresses, first and last, of GC registers a
    virtual function reaches through the RLC: in each segment of each live GC
    instance, from its base to the last register of the segment this library
    programs. *)

(** {1:access Access} *)

type t
(** The type for a GPU's registers, reached through its register BAR. *)

exception Stuck of string
(** Raised when the GPU does not answer in time, or its function or machine
    failed, with the step and the reason. An open or a reset returns it as its
    [Error]; a sleep raises it as [Rig_amd.Fault]. *)

val make : Rig_pci.Function.t -> Rig_pci.Window.t -> layout -> vf:bool -> t
(** [make f mmio l ~vf] is the registers [l] of the GPU of [f], whose register
    BAR is mapped as [mmio]; [vf] if [f] is a virtual function. *)

val fn : t -> Rig_pci.Function.t
val layout_of : t -> layout
val vf : t -> bool

val read : ?inst:int -> t -> string -> int
(** [read ~inst r name] is the value of the register. *)

val write :
  ?inst:int -> ?value:int -> t -> string -> (string * int) list -> unit
(** [write ~inst ~value r name fs] writes [value] (defaults to [0]) with the
    fields [fs] set, as {!Rig_amd_abi.Register.encode} does. *)

val update : ?inst:int -> t -> string -> (string * int) list -> unit
(** [update ~inst r name fs] sets the fields [fs] of the register's value, its
    other bits kept. *)

val field : ?inst:int -> t -> string -> string -> int
(** [field ~inst r name f] is the field [f] of the register's value. *)

val fields : ?inst:int -> t -> string -> (string * int) list
(** [fields ~inst r name] is every field of the register's value. *)

val write64 : ?inst:int -> t -> string -> lo:string -> hi:string -> int -> unit
(** [write64 ~inst r base ~lo ~hi x] writes the low and high 32 bits of [x] to
    the registers [base ^ lo] and [base ^ hi]. *)

val get : t -> int -> int
(** [get r a] is the register at address [a], in 32-bit words. *)

val set : t -> int -> int -> unit
(** [set r a x] writes [x] to the register at address [a]. *)

val set_pcie : ?aid:int -> t -> int -> int -> unit
(** [set_pcie ~aid r a x] writes [x] to the PCIe register [a] of the accelerator
    die [aid] (defaults to [0]) through the indirect window. The caller never
    names a die fused off: a write there stalls the fabric. *)

val wait : ?ms:int -> t -> string -> (unit -> bool) -> unit
(** [wait ~ms r what f] returns once [f ()] is [true], polled for at most [ms]
    milliseconds (defaults to 10,000). Raises {!Stuck} naming [what] otherwise,
    with the function's or machine's failure if it failed. *)

val pause : t -> int -> unit
(** [pause r us] waits [us] microseconds, the time a block needs with no state
    to poll. *)
