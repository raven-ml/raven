(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Driver-less AMD GPU device core.

    Opens an AMD GPU over PCI without a kernel driver and exposes the
    state everything else builds on: the mapped BARs, registers
    addressed by name with named bitfields, the IP discovery table the
    hardware publishes at the end of VRAM, and a device memory manager
    over the GPU's multi-level page tables.

    This is the passive device state only. Bringing the hardware up
    (firmware loading, IP-block initialization, boot and recovery)
    builds on top of it. *)

(** {1:registers Registers} *)

(** Registers bound to a device access path.

    A register pairs its definition (absolute address and named
    bitfields, see {!Amd_tables.Reg}) with the functions that reach it
    on a concrete device. *)
module Am_register : sig
  type t
  (** The type for device registers. *)

  val make :
    reg:Amd_tables.Reg.t -> rreg:(int -> int) -> wreg:(int -> int -> unit) -> t
  (** [make ~reg ~rreg ~wreg] is [reg] accessed through [rreg] and
      [wreg], which read and write 32-bit values at absolute dword
      addresses. *)

  val reg : t -> Amd_tables.Reg.t
  (** [reg t] is the underlying register definition. *)

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
end

(** {1:firmware Firmware} *)

(** Firmware images for device boot.

    Loads the firmware files a device generation needs from the local
    firmware directory, verifies each file against its pinned SHA-256
    digest, and splits the files into the images handed to the
    security processor while the device boots. *)
module Firmware : sig
  type desc = int list * bytes
  (** The type for loadable firmware images: the firmware-type ids the
      image serves (the [gfx_fw_type_*] values of
      {!Amd_tables.Am_defs}) and the image bytes. *)

  type t = {
    sos_fw : (int * bytes) list;
        (** Components of the security-processor operating system
            container, keyed by component type (the [psp_fw_type_*]
            values of {!Amd_tables.Am_defs}). *)
    ucode_start : (string * int) list;
        (** Instruction start address by compute-engine name (["PFP"],
            ["ME"], ["MEC"]), for engines whose image carries one. *)
    smu_psp_desc : desc option;
        (** The power-management firmware image, on generations that
            load it through the security processor. *)
    descs : desc list;  (** The remaining images, in load order. *)
  }
  (** The type for a device's firmware set. *)

  val fetch_fw : ?dir:string -> string -> sha256:string -> bytes
  (** [fetch_fw name ~sha256] is the content of the firmware file
      [name] in the directory [dir] (defaults to [$AMD_FW_PATH], or
      [/lib/firmware/amdgpu] when unset): the plain file when present,
      otherwise the [.zst]-suffixed variant decompressed with the
      [zstd] tool. Raises [Failure] naming the searched paths and the
      pinned upstream source when neither file exists, or naming the
      expected and actual digests when the content's SHA-256 is not
      [sha256]. *)

  val load_fw : ?dir:string -> string -> bytes
  (** [load_fw name] is {!fetch_fw} with the digest pinned for [name]
      in {!Amd_tables.Fw_defs.hashes}. Raises [Failure] when no digest
      is pinned. *)

  val create : ?load:(string -> bytes) -> (int * (int * int * int)) list -> t
  (** [create ip_ver] loads and splits the firmware set for the
      discovered hardware-IP versions [ip_ver] (hardware-IP id to
      version, as in {!type-discovery}): the security-processor OS
      container, then the power-management (skipped on management
      processors that boot their own), SDMA, compute-engine, and
      graphics-core support images the generation needs. [load]
      fetches one firmware file by name and defaults to {!load_fw}.
      Raises [Failure] on a file whose image header version is
      unknown, [Invalid_argument] when [ip_ver] lacks a needed
      hardware IP. *)
end

(** {1:pt Page tables} *)

(** Page tables in device memory.

    Implements {!Tolk.Memory.pt_ops} over 4KB page tables stored in
    VRAM, with the 64-bit entry encoding of the discovered
    graphics-core generation. Entry words are read and written as
    single volatile 64-bit accesses through the VRAM mapping. *)
module Am_page_table : sig
  type t
  (** The type for views of one page table. *)

  val pte_flags :
    gc_ver:int * int * int ->
    lv:int ->
    table:bool ->
    frag:int ->
    uncached:bool ->
    system:bool ->
    snooped:bool ->
    valid:bool ->
    int64
  (** [pte_flags ~gc_ver ~lv ...] is the flag word of a page-table
      entry at level [lv] for the graphics-core generation [gc_ver],
      without the physical-address bits. [table] marks an entry
      pointing at a child page table; otherwise the entry maps a page,
      gains read, write and execute permission bits and, above the leaf
      level, the generation's huge-page marker. [frag] is the TLB
      fragment-size exponent, [uncached] selects the generation's
      uncached memory type, [system] points the entry at host memory
      and [snooped] makes it cache-coherent with the host. Several
      generations use bit 63, so the word must stay an [int64]. *)

  val is_pte_huge_page : gc_ver:int * int * int -> lv:int -> int64 -> bool
  (** [is_pte_huge_page ~gc_ver ~lv pte] is [true] iff the entry word
      [pte] at level [lv] maps a page directly rather than pointing at
      a child page table. *)

  val paddr : t -> int
  (** [paddr pt] is the physical address of the page table [pt]. *)

  val lv : t -> int
  (** [lv pt] is the level of the page table [pt] in the tree. *)

  val ops :
    vram:Tolk_hcq.Hcq.Mmio.t ->
    gc_ver:int * int * int ->
    ?paddr_base:(unit -> int) ->
    unit ->
    t Tolk.Memory.pt_ops
  (** [ops ~vram ~gc_ver ()] are page-table operations over tables
      stored in [vram]. Device-local physical addresses are rebased by
      [paddr_base ()] when written and un-rebased when read back
      (defaults to no rebase, for devices whose local memory starts at
      physical address [0]); the rebased address must fit the
      generation's physical address width or [set_entry] raises
      [Invalid_argument]. *)
end

(** {1:discovery IP discovery} *)

type gc_info =
  | Gc_info_v1 of {
      num_se : int;
      num_wgp0_per_sa : int;
      num_wgp1_per_sa : int;
      num_sa_per_se : int;
      max_scratch_slots_per_cu : int;
      max_waves_per_simd : int;
      lds_size : int;
    }
  | Gc_info_v2 of {
      num_se : int;
      num_cu_per_sh : int;
      num_sh_per_se : int;
      max_scratch_slots_per_cu : int;
      max_waves_per_simd : int;
      lds_size : int;
    }
      (** The type for the graphics-core geometry published in the
          discovery table, by table major version. *)

type discovery = {
  ip_ver : (int * (int * int * int)) list;
      (** Hardware-IP id (e.g. {!Amd_tables.Am_defs.gc_hwip}) to its
          discovered [(major, minor, revision)] version, in increasing
          id order. *)
  regs_offset : (int * (int * int array) list) list;
      (** Hardware-IP id to per-instance register address-space segment
          bases, ids and instance numbers in increasing order. *)
  gc_info : gc_info;  (** Graphics-core geometry. *)
}
(** The type for parsed IP discovery tables. *)

val parse_discovery : bytes -> discovery
(** [parse_discovery blob] parses the IP discovery table [blob], the
    10KB block located 64KB before the end of VRAM: the die headers and
    their IP entries, each carrying one IP instance's version and
    register-aperture base addresses, plus the graphics-core geometry
    table. Raises [Failure] if a signature does not match or the
    geometry table has an unknown major version. *)

(** {1:devices Devices} *)

type t
(** The type for driver-less AMD GPU devices. *)

val create : Tolk_hcq.System.Pci_device.t -> t
(** [create pci_dev] opens the GPU behind [pci_dev]: maps its VRAM,
    doorbell and register BARs, sizes VRAM, reads and parses the IP
    discovery table, resolves register families for the discovered IP
    versions, reads the die's address topology (see {!paddr2mc}), and
    creates the device memory manager (a 32MB boot region, a dedicated
    page-table region when VRAM exceeds the VRAM BAR, and the main
    region behind them; four page-table levels over a 48-bit virtual
    space shared by all devices). The device starts in the booting
    state: only boot-region memory can be allocated until boot
    completes. Raises [Failure] if a BAR cannot be mapped or the
    discovery table is malformed. *)

val make :
  ?pci_dev:Tolk_hcq.System.Pci_device.t ->
  ?now_ms:(unit -> int) ->
  ?is_booting:bool ref ->
  ?on_range_mapped:(unit -> unit) ref ->
  rreg:(int -> int) ->
  wreg:(int -> int -> unit) ->
  vram:Tolk_hcq.Hcq.Mmio.t ->
  doorbell64:Tolk_hcq.Hcq.Mmio.t ->
  mmio:Tolk_hcq.Hcq.Mmio.t ->
  vram_size:int ->
  large_bar:bool ->
  reserved_vram_size:int ->
  discovery:discovery ->
  mm:Am_page_table.t Tolk.Memory.t ->
  devfmt:string ->
  unit ->
  t
(** [make ~rreg ~wreg ... ()] is a device over caller-provided parts:
    every register access goes through [rreg] and [wreg] (32-bit values
    at absolute dword addresses, replacing the register-BAR path of
    {!create} entirely), the BAR mappings, discovery table and memory
    manager are taken as given, and [now_ms] is the monotonic
    millisecond clock behind {!now_ms} (defaults to the system's).
    Construction reads the address-topology registers through [rreg],
    so they must already answer (see {!paddr2mc}).

    This is the device's injection seam: {!create} is the PCI client of
    the same state, while tests and tooling can supply scripted
    register access and anonymous-memory mappings. [pci_dev] is absent
    for such devices and [is_booting] defaults to a fresh reference
    holding [true]; pass the reference the memory manager's booting
    predicate reads to keep the two in step. Likewise
    [on_range_mapped] is the hook cell behind {!set_on_range_mapped}
    (defaults to a fresh cell holding a no-op); pass the reference the
    memory manager's mapping hook dereferences so installed hooks
    reach it. *)

val pci_dev : t -> Tolk_hcq.System.Pci_device.t option
(** [pci_dev t] is the underlying PCI device; [None] for devices built
    by {!make} without one. *)

val devfmt : t -> string
(** [devfmt t] is the device's PCI bus address, for messages. *)

val vram : t -> Tolk_hcq.Hcq.Mmio.t
(** [vram t] is the mapping of the VRAM BAR. It covers all of VRAM only
    when {!large_bar} is [true]. *)

val doorbell64 : t -> Tolk_hcq.Hcq.Mmio.t
(** [doorbell64 t] is the mapping of the doorbell BAR. *)

val mmio : t -> Tolk_hcq.Hcq.Mmio.t
(** [mmio t] is the mapping of the register BAR. *)

val vram_size : t -> int
(** [vram_size t] is the device's memory size in bytes. *)

val large_bar : t -> bool
(** [large_bar t] is [true] iff the VRAM BAR covers all of VRAM. *)

val reserved_vram_size : t -> int
(** [reserved_vram_size t] is the size of the VRAM tail reserved for
    firmware structures; the memory manager stays below it. *)

val discovery : t -> discovery
(** [discovery t] is the device's parsed IP discovery table. *)

val ip_ver : t -> int -> int * int * int
(** [ip_ver t hwip] is the discovered version of the hardware IP
    [hwip]. Raises [Invalid_argument] if discovery did not list it. *)

val gc_info : t -> gc_info
(** [gc_info t] is the device's graphics-core geometry. *)

val is_booting : t -> bool
(** [is_booting t] is [true] while the device is booting; only
    boot-region memory can be allocated then. *)

val is_err_state : t -> bool
(** [is_err_state t] is [true] once a hardware fault was observed on
    the device (see {!set_err_state}); recovery clears it. *)

val set_err_state : t -> bool -> unit
(** [set_err_state t v] records whether the device is in a fault
    state. The interrupt handler raises the flag on faults; recovery
    lowers it. *)

val mm : t -> Am_page_table.t Tolk.Memory.t
(** [mm t] is the device's memory manager. *)

val set_on_range_mapped : t -> (unit -> unit) -> unit
(** [set_on_range_mapped t f] installs [f] as the memory manager's
    after-mapping hook: it runs after every mapping the manager
    creates. The hook starts as a no-op; boot installs the TLB flush
    once the memory hubs answer. *)

val now_ms : t -> int
(** [now_ms t] is the device's monotonic clock in milliseconds. The
    boot protocols time their register waits and settle delays against
    it; injecting a clock through {!make} makes those waits scriptable. *)

(** {2:addr Address topology}

    A die can be one link of a larger memory fabric. The device reads
    its position at creation time: its local physical addresses are
    offset into the fabric's shared physical space, and the memory
    controller additionally rebases them behind the framebuffer window.
    On a single-device topology both conversions collapse to the
    framebuffer base alone. *)

val is_hive : t -> bool
(** [is_hive t] is [true] iff the device is part of a multi-die memory
    fabric. *)

val paddr2mc : t -> int -> int
(** [paddr2mc t paddr] is the device-local physical address [paddr] as
    the memory controller sees it. *)

val paddr2xgmi : t -> int -> int
(** [paddr2xgmi t paddr] is the device-local physical address [paddr]
    in the fabric's shared physical space. *)

val xgmi2paddr : t -> int -> int
(** [xgmi2paddr t addr] is the inverse of {!paddr2xgmi}. *)

(** {2:regaccess Register access} *)

val reg : t -> string -> Am_register.t
(** [reg t name] is the register [name] (e.g. ["regSCRATCH_REG7"])
    resolved against the register families of the device's discovered
    IP versions, bound to the device. Names are matched exactly; when
    several families define the same name, the family resolved last
    wins. Raises [Invalid_argument] if no family defines [name]. *)

val rreg : t -> int -> int
(** [rreg t reg] is the 32-bit value of the register at dword address
    [reg], read through the register BAR, or through the indirect
    index/data window for addresses beyond it. *)

val wreg : t -> int -> int -> unit
(** [wreg t reg v] writes the 32-bit value [v] to the register at dword
    address [reg], like {!rreg}. *)

val wreg_pair : t -> string -> lo:string -> hi:string -> int -> unit
(** [wreg_pair t base ~lo ~hi v] writes the 64-bit value [v] across the
    register pair named [base ^ lo] (low half) and [base ^ hi] (high
    half). *)

val indirect_wreg_pcie : t -> ?aid:int -> int -> int -> unit
(** [indirect_wreg_pcie t reg v] writes [v] to the register at dword
    address [reg] through the PCIe index/data window; [aid] addresses a
    die other than the first (defaults to [0]). *)
