(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Driver-less NVIDIA GPU device core.

    Opens an NVIDIA GPU over PCI without a kernel driver and exposes
    the state everything else builds on: the mapped BARs, registers
    addressed by name with named bitfields, the detected chip
    generation, and the device's memory geometry. Opening recovers a
    device left armed by a previous boot with a full PCI reset.

    This is the passive device state only. Bringing the hardware up
    (page tables, firmware loading, the security-processor boot chain)
    builds on top of it. *)

(** {1:registers Registers} *)

(** Registers bound to a device access path.

    A register pairs a definition from the generated register maps
    (see {!Nv_reg_defs}) with the functions that reach it on a
    concrete device. Its address is the sum of a block base and an
    offset; hardware blocks that repeat at several bases (the falcon
    microcontrollers) share one definition rebased with {!with_base},
    and arrayed registers resolve one element with {!with_idx}.

    Bitfield-only descriptors (the page-table entry encodings) carry
    named fields but no address: field access works, address access
    raises. Fields are inclusive [(lo, hi)] bit ranges; a descriptor
    field reaching past bit 62 does not fit the native [int], so
    {!encode}, {!decode} and {!mask} raise on it and its consumers
    work from the raw ranges in {!fields}. *)
module Nv_reg : sig
  type t
  (** The type for device registers. *)

  val make :
    name:string ->
    entry:Nv_reg_defs.entry ->
    rreg:(int -> int) ->
    wreg:(int -> int -> unit) ->
    t
  (** [make ~name ~entry ~rreg ~wreg] is the register or bitfield
      descriptor [entry] accessed through [rreg] and [wreg], which
      read and write 32-bit values at absolute byte addresses. Raises
      [Invalid_argument] if [entry] is a constant. *)

  val name : t -> string
  (** [name t] is the register's name. *)

  val fields : t -> (string * (int * int)) list
  (** [fields t] is the register's field names with their inclusive
      [(lo, hi)] bit ranges. *)

  val addr : t -> int
  (** [addr t] is the register's absolute byte address. Raises
      [Invalid_argument] on a bitfield descriptor and on an arrayed
      register not yet resolved with {!with_idx}. *)

  val with_base : t -> int -> t
  (** [with_base t base] is [t] with [base] added to its block base.
      Raises [Invalid_argument] on a bitfield descriptor. *)

  val with_idx : t -> int -> t
  (** [with_idx t i] is element [i] of the arrayed register [t], at
      offset [base + stride * i]. Raises [Invalid_argument] if [t] is
      not arrayed. *)

  val read : t -> int
  (** [read t] is the register's current 32-bit value. *)

  val read_bitfields : t -> (string * int) list
  (** [read_bitfields t] is {!read} decoded into the register's named
      fields. *)

  val write : t -> ?value:int -> (string * int) list -> unit
  (** [write t fields] stores the named field assignments ored with
      [value] (defaults to [0]); unnamed bits are written from [value]
      alone. Raises [Invalid_argument] on an unknown field name. *)

  val update : t -> (string * int) list -> unit
  (** [update t fields] is a read-modify-write of [fields]: bits
      outside the named fields keep their current value. Raises
      [Invalid_argument] on an unknown field name. *)

  val encode : t -> (string * int) list -> int
  (** [encode t values] ors each value shifted to its field's low bit.
      Values are not masked to the field width. Raises
      [Invalid_argument] on an unknown field name or a field past the
      native [int]. *)

  val decode : t -> int -> (string * int) list
  (** [decode t v] extracts every field of [t] from the register value
      [v]. Raises [Invalid_argument] if a field reaches past the
      native [int]. *)

  val mask : t -> string list -> int
  (** [mask t names] is the bitmask covering the named fields. Raises
      [Invalid_argument] on an unknown field name or a field past the
      native [int]. *)
end

(** {1:devices Devices} *)

type t
(** The type for driver-less NVIDIA GPU devices. *)

val create : Tolk_hcq.System.Pci_device.t -> t
(** [create pci_dev] opens the GPU behind [pci_dev]: maps its register
    BAR, resolves the base register families, and checks the
    write-protected firmware region. When that region is still armed —
    secure firmware from a previous boot is resident and the device
    cannot boot again — bus mastering is dropped and the device takes
    a full PCI reset followed by a settle delay. Bus mastering is then
    enabled, the chip generation is detected from the boot registers
    (GA1xx, AD1xx and GB2xx parts are supported), the page-table
    format and firmware directory are selected for it, VRAM is sized
    from the firmware scratch register, and the VRAM BAR is mapped.

    Raises [Failure] on an unsupported chip (naming the architecture
    and boot id) or when a BAR cannot be mapped. *)

val make :
  ?pci_dev:Tolk_hcq.System.Pci_device.t ->
  ?read_config:(offset:int -> size:int -> int) ->
  ?write_config_flush:(offset:int -> value:int -> size:int -> unit) ->
  ?reset:(unit -> unit) ->
  ?now_ms:(unit -> int) ->
  ?alloc_sysmem:(contiguous:bool -> int -> Tolk_hcq.Hcq.Mmio.t * int list) ->
  ?bar1_base:int ->
  ?palloc:(int -> int) ->
  rreg:(int -> int) ->
  wreg:(int -> int -> unit) ->
  mmio:Tolk_hcq.Hcq.Mmio.t ->
  vram:Tolk_hcq.Hcq.Mmio.t ->
  devfmt:string ->
  unit ->
  t
(** [make ~rreg ~wreg ~mmio ~vram ~devfmt ()] is a device over
    caller-provided parts, opened exactly like {!create}: every
    register access goes through [rreg] and [wreg] (32-bit values at
    absolute byte addresses), so construction reads the
    write-protected-region, boot and VRAM-size registers through them
    and they must already answer.

    This is the device's injection seam: {!create} is the PCI client
    of the same state, while tests and tooling supply scripted
    register access and anonymous-memory mappings. The PCI actions of
    the recovery path — [read_config] and [write_config_flush] on
    configuration space, the full [reset] — and the boot-memory seams
    — [alloc_sysmem] for pinned system memory, [bar1_base] for the
    VRAM BAR's bus address (defaults to [0]), [palloc] for device
    memory (defaults to raising) — default to the corresponding
    operations of [pci_dev] when given and to inert stand-ins
    otherwise. [now_ms] is the monotonic millisecond clock behind
    {!now_ms} (defaults to the system's); the reset settle delay waits
    on it, so injecting a clock makes the delay scriptable. *)

val pci_dev : t -> Tolk_hcq.System.Pci_device.t option
(** [pci_dev t] is the underlying PCI device; [None] for devices built
    by {!make} without one. *)

val devfmt : t -> string
(** [devfmt t] is the device's PCI bus address, for messages. *)

val mmio : t -> Tolk_hcq.Hcq.Mmio.t
(** [mmio t] is the mapping of the register BAR. *)

val vram : t -> Tolk_hcq.Hcq.Mmio.t
(** [vram t] is the mapping of the VRAM BAR. It covers all of VRAM
    only when {!large_bar} is [true]. *)

val vram_size : t -> int
(** [vram_size t] is the device's memory size in bytes. *)

val large_bar : t -> bool
(** [large_bar t] is [true] iff the VRAM BAR covers all of VRAM. *)

val chip_id : t -> int
(** [chip_id t] is the raw value of the boot-0 identification
    register. *)

val chip_name : t -> string
(** [chip_name t] is the detected chip, e.g. ["GA102"] or ["GB202"]. *)

val fw_name : t -> string
(** [fw_name t] is the firmware directory serving the chip: one
    firmware build covers each architecture, so e.g. every AD1xx part
    loads the ["ad102"] images. *)

val mmu_ver : t -> int
(** [mmu_ver t] is the generation of the chip's page-table entry
    format, [2] or [3]. *)

val fmc_boot : t -> bool
(** [fmc_boot t] is [true] when the chip boots firmware through the
    secure chain-of-trust microcontroller rather than the falcon
    bootloader. *)

val is_booting : t -> bool
(** [is_booting t] is [true] while the device is booting; only
    boot-region memory can be allocated then. *)

val set_is_booting : t -> bool -> unit
(** [set_is_booting t v] records whether the device is booting. *)

val is_err_state : t -> bool
(** [is_err_state t] is [true] once a hardware fault was observed on
    the device (see {!set_err_state}). *)

val set_err_state : t -> bool -> unit
(** [set_err_state t v] records whether the device is in a fault
    state. *)

val now_ms : t -> int
(** [now_ms t] is the device's monotonic clock in milliseconds. Boot
    protocols time their register waits and settle delays against it;
    injecting a clock through {!make} makes those waits scriptable. *)

(** {2:regaccess Register access} *)

val include_regs : t -> family:string -> arch:string -> unit
(** [include_regs t ~family ~arch] adds the register family [family]'s
    table for the architecture [arch] (see {!Nv_reg_defs.families}) to
    the device's register namespace; an entry redefines any earlier
    entry of the same name. The empty [arch] selects a family's
    unversioned table. Opening resolves the boot, framebuffer, island,
    virtual-function and page-table families; blocks brought up later
    resolve theirs on top. Raises [Invalid_argument] when the family
    has no table for [arch]. *)

val reg : t -> string -> Nv_reg.t
(** [reg t name] is the register or bitfield descriptor [name]
    resolved against the included families, bound to the device.
    Raises [Invalid_argument] if no included family defines [name] or
    if [name] is a constant. *)

val const : t -> string -> int
(** [const t name] is the named constant of the included families.
    Raises [Invalid_argument] if no included family defines [name] or
    if [name] is a register. *)

val rreg : t -> int -> int
(** [rreg t addr] is the 32-bit value of the register at byte address
    [addr] in the register BAR. *)

val wreg : t -> int -> int -> unit
(** [wreg t addr v] writes the 32-bit value [v] to the register at
    byte address [addr], like {!rreg}. *)

(** {2:bootmem Boot memory} *)

val alloc_boot_mem :
  t ->
  ?data:bytes ->
  ?contiguous:bool ->
  ?sysmem:bool ->
  int ->
  Tolk_hcq.Hcq.Mmio.t * int option * int list
(** [alloc_boot_mem t size] allocates [size] bytes for structures the
    device must address physically while it boots, before its page
    tables exist, and is the CPU view of the memory, its device-local
    physical address, and the bus address of each of its 4KB pages.

    [sysmem] selects pinned system memory over device memory; when
    unset, device memory is used exactly when the VRAM BAR reaches all
    of VRAM (see {!large_bar}). System memory is allocated at [size]
    ([contiguous] makes it physically contiguous, defaults to [false])
    and has no device-local address; device memory is allocated at
    [size] rounded up to whole 4KB pages and reached through the VRAM
    BAR. [data] initializes the memory. Raises [Failure] when the
    memory cannot be allocated. *)
