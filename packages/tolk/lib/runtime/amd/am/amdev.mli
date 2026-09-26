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
    reg:Amd_tables.Reg.t ->
    rreg:(direct:bool -> int -> int) ->
    wreg:(direct:bool -> int -> int -> unit) -> t
  (** [make ~reg ~rreg ~wreg] is [reg] accessed through [rreg] and
      [wreg], which read and write 32-bit values at absolute dword
      addresses. [direct] bypasses the virtual-function register gateway. *)

  val reg : t -> Amd_tables.Reg.t
  (** [reg t] is the underlying register definition. *)

  val read : ?direct:bool -> t -> int
  (** [read t] is the register's current 32-bit value. *)

  val read_bitfields : ?direct:bool -> t -> (string * int) list
  (** [read_bitfields t] is {!read} decoded into the register's named
      fields. *)

  val write : t -> ?direct:bool -> ?value:int -> (string * int) list -> unit
  (** [write t fields] stores the named field assignments ored with
      [value] (defaults to [0]); unnamed bits are written from [value]
      alone. Raises [Invalid_argument] on an unknown field name. *)

  val update : t -> ?direct:bool -> (string * int) list -> unit
  (** [update t fields] is a read-modify-write of [fields]: bits
      outside the named fields keep their current value. Raises
      [Invalid_argument] on an unknown field name. *)
end

(** {1:firmware Firmware} *)

(** Firmware images for device boot.

    Loads the firmware files a device generation needs from the local
    firmware directory or pinned upstream source, verifies their SHA-256
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
  (** [fetch_fw name ~sha256] is the verified firmware content. It first
      checks [name] and its [.zst] variant in [dir] (defaults to
      [$AMD_FW_PATH], or [/lib/firmware/amdgpu] when unset). Compressed
      files require the [zstd] tool. If neither local file has the expected
      digest, it uses the disk cache or downloads from the pinned upstream
      source with [curl]. Cached and downloaded bytes must also match
      [sha256]; only verified downloads are cached. Local files are unchanged.
      Raises [Failure] if the download command fails or its digest differs,
      and [Unix.Unix_error] if [curl] cannot be started. *)

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
  harvested : (int * int list) list;
      (** Disabled instances from the optional harvest table, grouped by
          hardware-IP id with sorted, unique instance numbers. *)
  gc_info : gc_info;  (** Graphics-core geometry. *)
}
(** The type for parsed IP discovery tables. *)

val parse_discovery : bytes -> discovery
(** [parse_discovery blob] parses the IP discovery table [blob], the
    10KB block located 64KB before the end of VRAM: the die headers and
    their IP entries, each carrying one IP instance's version and
    register-aperture base addresses, plus graphics-core geometry and optional
    harvest information. Raises [Failure] if a required signature does not
    match, the geometry table has an unknown major version, or a present
    harvest table is truncated. Unknown harvest signatures are ignored. *)

(** {1:devices Devices} *)

val va_base : int
(** [va_base] is the base of the virtual address space shared by every
    device's memory manager. *)

val va_size : int
(** [va_size] is the size of the shared virtual address space in
    bytes. *)

type t
(** The type for driver-less AMD GPU devices. *)

val create : Tolk_hcq.System.Pci_device.t -> t
(** [create pci_dev] opens the GPU behind [pci_dev]: maps its VRAM,
    doorbell and register BARs, sizes VRAM, reads and parses the IP
    discovery table, resolves register families for the discovered IP
    versions, reads the die's address topology (see {!paddr2mc}), and
    creates the device memory manager (a 3 MiB boot region, a dedicated
    page-table region when VRAM exceeds the VRAM BAR, and the main
    region behind them; four page-table levels over a 48-bit virtual
    space shared by all devices). The device starts in the booting
    state: only boot-region memory can be allocated until boot
    completes. Resident boot memory is left intact; {!Am_boot.init}
    clears the root page table after marking the new session active.
    Raises [Failure] if a BAR cannot be mapped or the
    discovery table is malformed. *)

val make :
  ?pci_dev:Tolk_hcq.System.Pci_device.t ->
  ?now_ms:(unit -> int) ->
  ?sleep_ms:(int -> unit) ->
  ?is_booting:bool ref ->
  ?on_range_mapped:(unit -> unit) ref ->
  read_config:(offset:int -> size:int -> int) ->
  rreg:(int -> int) ->
  wreg:(int -> int -> unit) ->
  rreg8:(int -> int) ->
  wreg8:(int -> int -> unit) ->
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
(** [make ~read_config ~rreg ~wreg ... ()] is a device over caller-provided parts:
    every register access goes through [rreg] and [wreg] (32-bit values
    at absolute dword addresses, replacing the register-BAR path of
    {!create} entirely). [rreg8] and [wreg8] access mailbox bytes at byte
    offsets without changing adjacent bytes. The BAR mappings, discovery table and memory
    manager are taken as given, and [now_ms] is the monotonic
    millisecond clock behind {!now_ms} (defaults to the system's).
    [sleep_ms] suspends execution for a settling delay in milliseconds
    (defaults to [Unix.sleepf]); injected clocks should advance during it.
    [read_config] supplies PCI configuration reads at byte offsets,
    including vendor readiness after a reset.
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

val read_config : t -> offset:int -> size:int -> int
(** [read_config t ~offset ~size] reads the little-endian value of [size]
    bytes at byte [offset] in PCI configuration space. Read failures
    propagate from the underlying device or supplied reader. *)

val devfmt : t -> string
(** [devfmt t] is the device's PCI bus address, for messages. *)

val live_instances : t -> int -> int list
(** [live_instances t hwip] lists discovered instances of [hwip] excluding
    entries disabled by the harvest table, in increasing order. *)

val aids : t -> int list
(** [aids t] lists live accelerator I/O dies: die zero and each additional
    die whose four SDMA instances have live mask [0xf], [0x3] or [0xc]. *)

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

val is_vf : t -> bool
(** [is_vf t] is [true] for a PCI virtual function. Construction obtains
    its initialization access lease from the physical function. *)

val release_vf_access : t -> unit
(** [release_vf_access t] returns the held access lease, if any. A timed-out
    handback is ignored and the local lease is cleared before the request. *)

val acquire_fini_access : t -> unit
(** [acquire_fini_access t] requests finalization access for a virtual
    function with no held lease. A timed-out request is ignored. *)

exception Timeout_error of string

val wait_cond :
  t -> ?timeout_ms:int -> value:int -> msg:string -> (unit -> int) -> unit
(** [wait_cond t ~value ~msg read] polls [read] until it equals [value].
    Raises [Timeout_error] after [timeout_ms] milliseconds (default [10000])
    according to the device clock. *)

val is_booting : t -> bool
(** [is_booting t] is [true] while the device is booting; only
    boot-region memory can be allocated then. *)

val set_is_booting : t -> bool -> unit
(** [set_is_booting t v] records whether the device is booting. Boot
    lowers the flag once the blocks whose state lives in boot memory
    are up, unlocking allocation from the main memory region. *)

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
    boot protocols use it to bound register polling. *)

val sleep_ms : t -> int -> unit
(** [sleep_ms t ms] suspends execution for [ms] milliseconds while hardware
    settles. Uses the delay function supplied to {!make}, or [Unix.sleepf]
    for a PCI device. *)

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

val reg : t -> ?inst:int -> string -> Am_register.t
(** [reg t ?inst name] is the register [name] (e.g. ["regSCRATCH_REG7"])
    resolved against the register families of the device's discovered
    IP versions, bound to the device. Names are matched exactly; when
    several families define the same name, the family resolved last
    wins. [inst] selects its discovered IP instance and defaults to [0].
    Raises [Invalid_argument] if the register or instance is absent. *)

val rreg : t -> ?inst:int -> ?direct:bool -> int -> int
(** [rreg t reg] is the 32-bit value of the register at dword address
    [reg], read through the register BAR, or through the indirect
    index/data window for addresses beyond it. Virtual functions route gated
    registers through the RLC gateway of [inst] (default [0]); [direct=true]
    bypasses that gateway. *)

val wreg : t -> ?inst:int -> ?direct:bool -> int -> int -> unit
(** [wreg t reg v] writes the 32-bit value [v] to the register at dword
    address [reg], like {!rreg}. *)

val wreg_pair : t -> ?inst:int -> ?direct:bool -> string -> lo:string -> hi:string -> int -> unit
(** [wreg_pair t ?inst base ~lo ~hi v] writes the 64-bit value [v] across the
    register pair named [base ^ lo] (low half) and [base ^ hi] (high
    half), using discovered IP instance [inst] (default [0]). *)

val indirect_wreg_pcie : t -> ?aid:int -> int -> int -> unit
(** [indirect_wreg_pcie t reg v] writes [v] to the register at dword
    address [reg] through the PCIe index/data window; [aid] addresses a
    die other than the first (defaults to [0]). *)
