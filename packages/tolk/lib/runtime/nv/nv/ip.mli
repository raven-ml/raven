(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Firmware and boot images for the driver-less NVIDIA boot chain.

    Bringing an NVIDIA GPU up without a kernel driver means feeding the
    security-processor boot chain a set of prepared images: the VBIOS
    ucode that reserves the frame-buffer resident tables, the falcon
    bootloaders, the chain-of-trust firmware, and the GSP image with its
    page hierarchy. This module is the parsing and patching layer that
    turns raw firmware files and the on-board VBIOS ROM into those images
    and the offsets that drive them onto the hardware.

    Every function here is a pure transform over byte images (or, for the
    VBIOS window, over an injected register read). Loading the images onto
    the device — allocating boot memory, programming falcon registers, and
    running the boot handshakes — builds on top of the values these
    functions produce.

    Once the GSP firmware runs, the CPU talks to it over shared-memory
    message queues; {!Rpc_queue} is that transport, and
    {!Gsp.run_cpu_seq} interprets the register command sequences the
    firmware sends back over it during boot. *)

exception Timeout_error of string
(** Raised when a bounded hardware wait — a queue waiting to come up, a
    register poll — does not reach its expected value in time. The
    message names the wait and the last observed value. *)

(** {1:firmware Firmware files} *)

module Firmware : sig
  val fetch : ?dir:string -> chip_dir:string -> string -> bytes
  (** [fetch ~chip_dir name] loads and verifies the firmware file [name]
      for chip directory [chip_dir]. The file is read from
      [dir/chip_dir/gsp/name], where [dir] defaults to the [NV_FW_PATH]
      environment variable or [/lib/firmware/nvidia]; a [.zst]-compressed
      copy is accepted and decoded through the system [zstd]. Its SHA-256
      is checked against the pinned digest for [(name, chip_dir)].

      Raises [Failure] when no digest is pinned for the file, when neither
      the plain nor the compressed file is found (naming both searched
      paths, the expected digest, and the pinned upstream URL), or when the
      digest does not match. *)
end

(** {1:rpcq RPC message queues} *)

(** The shared-memory RPC transport to the GSP firmware.

    The CPU and the GSP each own one queue: a page-sized header region
    followed by a ring of fixed-size elements. The owner publishes
    messages by writing elements and advancing the write pointer in its
    own header; the consumer acknowledges them by advancing a read
    pointer that lives in the {e other} queue's header region, so each
    side polls only memory the other side writes.

    A message is one or more records. Each record spans whole ring
    elements and carries a 48-byte element header (a frame checksum, a
    sequence number, and the element count) followed by a 32-byte RPC
    header (function id, byte length including that header, and result
    fields) and the payload, zero-padded to the element boundary. A
    payload that does not fit the 16-element frame limit continues in
    follow-on continuation records.

    Publishing and consuming touch only the queue memory and the
    injected callbacks, so a queue can be driven against anonymous
    memory with no device. *)
module Rpc_queue : sig
  type t
  (** The type for RPC queues. *)

  val make :
    ?completion_q_view:Tolk_hcq.Hcq.Mmio.t ->
    ?now_ms:(unit -> int) ->
    ?devfmt:string ->
    notify:(unit -> unit) ->
    on_run_cpu_seq:(bytes -> unit) ->
    on_error:(unit -> unit) ->
    Tolk_hcq.Hcq.Mmio.t ->
    t
  (** [make view] is the queue whose memory is [view]: the transmit
      header at its start and the element ring at the header's entry
      offset. Construction waits up to ten seconds for the header's
      entry offset to read [0x1000] — the queue owner publishing its
      header — then snapshots the element geometry from it; the wait
      raises {!Timeout_error} ("RPC queue not initialized").

      [completion_q_view] is the memory of the opposite queue; when
      given, the read pointer is taken from it at the offset its header
      names (see {!rx_hdr_off}). A queue made without it cannot read
      responses until {!set_rx_view}.

      [notify] is invoked after each published record to signal the
      consumer (the doorbell write). [on_run_cpu_seq] receives the
      payload of a CPU-sequencer command message (see
      {!Gsp.run_cpu_seq}), and [on_error] is invoked when an error-log
      or MMU-fault event arrives; both fire while responses are being
      read. [now_ms] is the monotonic clock behind the construction
      wait and {!wait_resp} (defaults to the system's); [devfmt] names
      the device in logged messages. *)

  val rx_hdr_off : t -> int
  (** [rx_hdr_off t] is the byte offset, within this queue's memory,
      where the header places the {e consumer's} read pointer — the
      offset at which the opposite queue finds its response pointer. *)

  val set_rx_view : t -> Tolk_hcq.Hcq.Mmio.t -> unit
  (** [set_rx_view t rx] sets the view holding [t]'s read pointer (a
      32-bit element index at its start), for queues made without
      [completion_q_view]. *)

  val send_rpc : t -> int -> bytes -> unit
  (** [send_rpc t fn msg] publishes the RPC call [fn] with payload
      [msg]: the record is framed, checksummed, written into the ring
      at the current write pointer (wrapping across the ring's end),
      the write pointer advances, and [notify] rings the consumer. A
      payload beyond the 16-element frame limit is split into
      continuation records, each published the same way. Every record
      consumes one sequence number. *)

  val read_resp : t -> (int * bytes) Seq.t
  (** [read_resp t] lazily consumes the messages published between
      [t]'s read pointer and the producer's write pointer. Forcing an
      element parses one message, dispatches it ([on_run_cpu_seq] for a
      CPU-sequencer command, a log print and [on_error] for an
      error-log event, [on_error] for a queued MMU fault), advances the
      read pointer past it, and yields its function id and payload; the
      payload is read at the length the RPC header declares, so it
      carries 32 trailing ring bytes. The sequence ends when the queue
      is drained; abandoning it leaves the remaining messages
      unconsumed. Frame checksums are not verified on this path.

      Raises [Failure] when a message carries a nonzero result, naming
      the function id and the result, and [Invalid_argument] if the
      queue has no read pointer yet. *)

  val wait_resp : t -> ?timeout_ms:int -> int -> bytes
  (** [wait_resp t fn] drains {!read_resp} until a message for the
      function id [fn] arrives and is its payload; other messages are
      consumed and dropped. Raises [Failure] when [timeout_ms]
      (defaults to ten seconds) elapses without one, or as
      {!read_resp} does. *)
end

(** {1:falcon Falcon boot images} *)

module Flcn : sig
  type desc = {
    imem_load_size : int;  (** Bytes of instruction memory to load. *)
    imem_phys_base : int;  (** Physical base of the instruction memory. *)
    imem_virt_base : int;  (** Virtual base of the instruction memory. *)
    dmem_phys_base : int;  (** Physical base of the data memory. *)
    dmem_load_size : int;  (** Bytes of data memory to load. *)
    pkc_data_offset : int;  (** Offset of the signature region in the image. *)
    engine_id_mask : int;  (** Engine-id mask the ucode is bound to. *)
    ucode_id : int;  (** Ucode id the boot ROM checks. *)
    stored_size : int;  (** Stored image size, before 256-byte rounding. *)
    interface_offset : int;
        (** Offset of the application-interface header within the image. *)
  }
  (** The FWSEC ucode descriptor, holding the geometry a falcon boot needs
      to load and verify the image. *)

  type ucode = {
    desc : desc;  (** The parsed descriptor. *)
    frts_offset : int;
        (** Byte offset in VRAM reserved for the frame-buffer resident
            tables. *)
    frts_image : bytes;  (** The patched, boot-ready FWSEC image. *)
  }
  (** The prepared FWSEC ucode. *)

  val read_vbios : read32:(int -> int) -> bytes
  (** [read_vbios ~read32] reads the 1 MiB VBIOS ROM out of the register
      window at byte offset [0x300000] into a flat byte image. [read32 a]
      is the unsigned 32-bit value at byte address [a] in the register
      aperture. *)

  val prep_ucode : rom:bytes -> vram_size:int -> ucode
  (** [prep_ucode ~rom ~vram_size] parses the FWSEC ucode out of the VBIOS
      image [rom] and patches it for boot. [vram_size] is the device's
      memory size in bytes, which fixes the reserved-tables offset.

      Raises [Failure] if the VBIOS structures are malformed or the
      production FWSEC descriptor is absent, and [Invalid_argument] on a
      read past the end of [rom]. *)

  type booter = {
    image : bytes;  (** The patched bootloader image. *)
    data_off : int;  (** Byte offset of the data segment in the image. *)
    data_sz : int;  (** Size of the data segment. *)
    code_off : int;  (** Byte offset of the code segment in the image. *)
    code_sz : int;  (** Size of the code segment. *)
  }
  (** A prepared heavy-secured bootloader. *)

  val prep_booter : blob:bytes -> booter
  (** [prep_booter ~blob] parses the heavy-secured bootloader firmware
      [blob] and splices its production signature into the boot image,
      returning the image and its code and data spans.

      Raises [Invalid_argument] on a read past the end of [blob]. *)

  (** {2:falcon_exec Falcon execution} *)

  type t
  (** The type for the falcon execution layer of one device. It drives
      two microcontrollers: the GSP falcon that runs the resident
      firmware and the SEC2 booter that unlocks the write-protected
      region. *)

  val create : Nvdev.t -> t
  (** [create nvdev] is the falcon execution layer of [nvdev]. It holds
      no prepared state until {!init_sw}. *)

  val falcon : t -> int
  (** [falcon t] is the register-block base of the GSP falcon. *)

  val sec2 : t -> int
  (** [sec2 t] is the register-block base of the SEC2 booter. *)

  val wait_for_reset : t -> unit
  (** [wait_for_reset t] blocks until the boot-progress scratch reports
      the device is out of reset and secure boot has completed. Raises
      {!Timeout_error} if it does not within ten seconds. *)

  val init_sw : t -> unit
  (** [init_sw t] resolves the falcon register families on the device,
      reads the VBIOS, and loads the FWSEC ucode and the booter image
      into device memory, remembering where they landed for {!init_hw}.

      Raises [Failure] on a malformed VBIOS, a missing or mismatched
      firmware file, or a boot-memory allocation that yields no
      device-local address (see {!Firmware.fetch} and {!prep_ucode}). *)

  val init_hw : t -> libos_args_sysmem:int -> wpr_meta_sysmem:int -> unit
  (** [init_hw t ~libos_args_sysmem ~wpr_meta_sysmem] brings the falcons
      up: it runs the FWSEC ucode to reserve the resident tables,
      restarts the GSP falcon as RISC-V with [libos_args_sysmem] in its
      mailbox, then runs the booter with [wpr_meta_sysmem] to unlock the
      write-protected region, and confirms the GSP core is active. The
      two addresses are the boot-argument and write-protect-metadata
      regions the GSP client sets up.

      Requires {!init_sw} to have run. Raises [Failure] if the resident
      tables are not initialized, the booter reports a nonzero mailbox,
      or the GSP core does not come up active, and {!Timeout_error} if a
      hardware wait expires. *)

  val reset : t -> ?riscv:bool -> int -> unit
  (** [reset t base] resets the microcontroller at [base]: it pulses the
      engine reset, waits for memory scrubbing to finish, and brings the
      RISC-V core out of reset. [riscv] (defaults to [false]) leaves it
      fetching its own bootloader; otherwise the falcon core is selected
      and the chip id is stamped. Raises {!Timeout_error} on a wait
      expiry. *)

  val disable_ctx_req : t -> int -> unit
  (** [disable_ctx_req t base] configures the microcontroller at [base]
      to allow context-free physical DMA. *)

  val start_cpu : t -> int -> unit
  (** [start_cpu t base] starts the core of the microcontroller at
      [base], through its alias register when the alias path is enabled
      and directly otherwise. *)

  val wait_cpu_halted : t -> int -> unit
  (** [wait_cpu_halted t base] blocks until the core at [base] halts.
      Raises {!Timeout_error} if it does not within ten seconds. *)

  val execute_hs :
    t ->
    int ->
    img_paddr:int ->
    code_off:int ->
    data_off:int ->
    imem_pa:int ->
    imem_va:int ->
    imem_sz:int ->
    dmem_pa:int ->
    dmem_va:int ->
    dmem_sz:int ->
    pkc_off:int ->
    engid:int ->
    ucodeid:int ->
    ?mailbox:int ->
    unit ->
    (int * int) option
  (** [execute_hs t base ~img_paddr ~code_off ~data_off ~imem_pa ~imem_va
      ~imem_sz ~dmem_pa ~dmem_va ~dmem_sz ~pkc_off ~engid ~ucodeid ()]
      runs a heavy-secured image on the microcontroller at [base]: it
      loads the image's instruction memory (from [img_paddr + code_off],
      [imem_sz] bytes to [imem_pa] at virtual base [imem_va]) and data
      memory (from [img_paddr + data_off], [dmem_sz] bytes to [dmem_pa]
      at virtual base [dmem_va]) over physical DMA, programs the boot ROM
      with the signature parameters ([pkc_off], [engid], [ucodeid]), and
      runs the core to completion.

      [mailbox], when given, seeds the mailbox before the run — the
      booter reads its write-protect metadata address there — and the
      result is the mailbox pair read back after the halt; without it the
      result is [None]. Raises {!Timeout_error} on a DMA or halt wait
      expiry. *)
end

(** {1:cot Chain-of-trust boot image} *)

module Flcn_cot : sig
  type fmc = {
    image : bytes;  (** The bootable firmware image. *)
    hash : int array;  (** Image hash, as 32-bit words. *)
    signature : int array;  (** Image signature, as 32-bit words. *)
    public_key : int array;  (** Verification public key, as 32-bit words. *)
  }
  (** A chain-of-trust firmware image and its verification blobs. *)

  val init_fmc_image : blob:bytes -> fmc
  (** [init_fmc_image ~blob] splits the chain-of-trust firmware ELF [blob]
      into its bootable image and the hash, signature and public-key blobs.

      Raises [Failure] if a required section is missing, and
      [Invalid_argument] if [blob] is not a valid ELF object. *)

  (** {2:cot_exec Chain-of-trust execution} *)

  type t
  (** The type for the chain-of-trust boot layer of one device, used on
      chips that boot the GSP through the secure security processor
      rather than the falcon bootloader. *)

  val create : Nvdev.t -> t
  (** [create nvdev] is the chain-of-trust boot layer of [nvdev]. It
      holds no prepared state until {!init_sw}. *)

  val wait_for_reset : t -> unit
  (** [wait_for_reset t] resolves the thermal registers and blocks until
      their scratch reports the device is out of reset. Raises
      {!Timeout_error} if it does not within ten seconds. *)

  val init_sw : t -> unit
  (** [init_sw t] resolves the chain-of-trust register families on the
      device, reserves the GSP boot-argument region, and loads the
      chain-of-trust firmware image into device memory for {!init_hw}.

      Raises [Failure] on a missing or mismatched firmware file (see
      {!Firmware.fetch}). *)

  val init_hw : t -> libos_args_sysmem:int -> wpr_meta_sysmem:int -> unit
  (** [init_hw t ~libos_args_sysmem ~wpr_meta_sysmem] boots the GSP
      through the security processor: it fills the boot-argument region
      with the GSP-RM block (naming [wpr_meta_sysmem]) and the RM block
      (naming [libos_args_sysmem]), builds the chain-of-trust payload
      over the firmware's hash, signature and public key, hands it to the
      security processor, and waits for the RISC-V boot lockdown to clear.
      The two addresses are the boot-argument and write-protect-metadata
      regions the GSP client sets up.

      Requires {!init_sw} to have run. Raises {!Timeout_error} on a
      security-processor or lockdown wait expiry, and [Failure] if the
      payload exceeds the security processor's message window. *)

  val kfsp_send_msg : t -> nvmd:int -> bytes -> unit
  (** [kfsp_send_msg t ~nvmd msg] sends the single-packet message [msg]
      on NVDM channel [nvmd] to the security processor through its
      external memory window, waits for a reply, and drains the reply
      queue. [msg] is framed with the packet and channel headers and
      padded to a word; the framed message must be under 1 KiB. Raises
      [Failure] when it is not, and {!Timeout_error} if no reply arrives
      within ten seconds. *)
end

(** {1:gsp GSP firmware image} *)

module Gsp : sig
  type radix3 = {
    npages : int array;
        (** Page count at each of the four levels, deepest last. *)
    offsets : int array;
        (** Byte offset of each level within the hierarchy region. *)
    image_off : int;  (** Byte offset where the image begins. *)
  }
  (** The layout of a three-level page hierarchy over the GSP image. *)

  val radix3 : image_len:int -> radix3
  (** [radix3 ~image_len] computes the page-hierarchy layout for a GSP
      image of [image_len] bytes: the deepest level covers the image's
      4 KiB pages, and each level above holds one 8-byte pointer per page
      of the level below. The layout is enough to size and fill the
      hierarchy region once its pages have physical addresses. *)

  type split = {
    image : bytes;  (** The GSP firmware image. *)
    signature : bytes;  (** The per-chip signature section. *)
  }
  (** The GSP image split from its firmware ELF. *)

  val split_gsp_image : blob:bytes -> chip_name:string -> split
  (** [split_gsp_image ~blob ~chip_name] extracts the firmware image and
      the signature section for [chip_name] from the GSP firmware ELF
      [blob]. The signature section is named for the chip family (the first
      four characters of [chip_name], lowercased).

      Raises [Failure] if a required section is missing, and
      [Invalid_argument] if [blob] is not a valid ELF object or
      [chip_name] is shorter than four characters. *)

  type bootloader = {
    image : bytes;  (** The RISC-V bootloader image. *)
    monitor_code_offset : int;  (** Offset of the monitor code segment. *)
    monitor_data_offset : int;  (** Offset of the monitor data segment. *)
    manifest_offset : int;  (** Offset of the boot manifest. *)
  }
  (** A parsed RISC-V bootloader container. *)

  val init_boot_binary_image : blob:bytes -> bootloader
  (** [init_boot_binary_image ~blob] parses the RISC-V bootloader firmware
      [blob], returning the bootloader image and the monitor code, data and
      manifest offsets its descriptor names.

      Raises [Invalid_argument] on a read past the end of [blob]. *)

  val run_cpu_seq :
    rreg:(int -> int) ->
    wreg:(int -> int -> unit) ->
    now_ms:(unit -> int) ->
    sleep_us:(int -> unit) ->
    core_reset:(unit -> unit) ->
    core_start:(unit -> unit) ->
    core_wait_halted:(unit -> unit) ->
    core_resume:(unit -> unit) ->
    bytes ->
    unit
  (** [run_cpu_seq buf] interprets the command sequence the GSP firmware
      sends the CPU during boot: a 40-byte header whose second word
      counts the command words that follow, then that many 32-bit
      words. Words past the count are ignored. The commands are:

      - [0x0] [addr value] — write [value] to the register at [addr]
        through [wreg].
      - [0x1] [addr value mask] — read-modify-write: the register keeps
        its bits outside [mask] and takes [value]'s inside it.
      - [0x2] [addr mask value _ _] — poll [rreg addr land mask] until
        it equals [value], up to ten seconds on [now_ms]; the two
        trailing words are unused. Raises {!Timeout_error} naming the
        register and value on expiry.
      - [0x3] [us] — delay [us] microseconds through [sleep_us].
      - [0x4] [addr index] — read the register at [addr] into slot
        [index] of an eight-word scratch save area. Raises
        [Invalid_argument] on an out-of-range slot.
      - [0x5]–[0x8] — hand control to the injected boot actions:
        [core_reset] (reset the firmware processor and detach its
        memory interface), [core_start] (start it), [core_wait_halted]
        (wait for it to halt), and [core_resume] (restart it through
        the secure-boot handshake).

      Raises [Failure] on an unknown command code (naming it) or when a
      command's arguments run past the counted words. *)

  (** {2:gsp_builders Payload builders} *)

  val registry_table : unit -> bytes
  (** [registry_table ()] is the packed registry table the GSP reads at
      boot: a header, one fixed-size entry per key, then the trailing
      NUL-terminated key names each entry points at by offset. *)

  val radix3_fill : image:bytes -> layout:radix3 -> page_addrs:int array -> bytes
  (** [radix3_fill ~image ~layout ~page_addrs] is the byte image of the
      page-hierarchy region for [layout]: [image] copied in at its offset,
      and each directory level filled with the physical addresses (from
      [page_addrs], the region's own page addresses) of the pages of the
      level below. *)

  val wpr_meta :
    vram_size:int ->
    fmc_boot:bool ->
    radix3_addr:int ->
    radix3_size:int ->
    booter_addr:int ->
    booter_size:int ->
    signature_addr:int ->
    code_off:int ->
    data_off:int ->
    manifest_off:int ->
    frts_offset:int ->
    bytes
  (** [wpr_meta ~vram_size ~fmc_boot ...] is the write-protect-region
      metadata that places the GSP image, bootloader, signature and heap
      in VRAM. The chain-of-trust boot ([fmc_boot]) uses a fixed
      reservation layout; the falcon boot packs the regions down from the
      top of [vram_size] and requires the reserved-tables offset it
      computes to equal [frts_offset] (the FWSEC ucode's), raising
      [Failure] ("FRTS mismatch") otherwise. [frts_offset] is ignored on
      the chain-of-trust boot. *)

  val bdf_as_int : string -> int
  (** [bdf_as_int pcibus] packs the bus address [pcibus] into the integer
      the GSP system info carries: the bus, device and function digits of a
      ["0000:03:00.0"]-style address; a non-PCI (USB or remote) transport
      reports zero. *)

  val gsp_system_info :
    gpu_phys:int ->
    gpu_phys_fb:int ->
    gpu_phys_inst:int ->
    fmc_boot:bool ->
    bdf:int ->
    pci_device_id:int ->
    pci_subdevice_id:int ->
    pci_revision_id:int ->
    bytes
  (** [gsp_system_info ~gpu_phys ...] is the system-info structure the GSP
      reads at boot, carrying the device's BAR base addresses, its
      configuration-mirror window (selected by [fmc_boot]), its bus
      address [bdf], and its PCI identity. *)

  val rm_alloc_request :
    client:int ->
    hparent:int ->
    hobject:int ->
    hclass:int ->
    params:bytes ->
    bytes
  (** [rm_alloc_request ~client ~hparent ~hobject ~hclass ~params] is the
      GSP_RM_ALLOC payload: the allocation envelope followed by the
      class's parameter bytes. *)

  val rm_control_request :
    client:int -> hobject:int -> cmd:int -> params:bytes -> bytes
  (** [rm_control_request ~client ~hobject ~cmd ~params] is the
      GSP_RM_CONTROL payload: the control envelope followed by the
      command's parameter bytes. *)

  val rm_control_apply :
    chip_name:string ->
    cmd:int ->
    response:bytes ->
    params:Nv_tables.blob ->
    unit
  (** [rm_control_apply ~chip_name ~cmd ~response ~params] copies the
      driver's in-place parameter update out of a GSP_RM_CONTROL
      [response] (past its envelope echo) back into [params]. On a GB20x
      part, the work-submit-token control additionally has the enable bit
      patched into the returned token. *)

  val set_channel_gpfifo_descs :
    params:Nv_tables.blob ->
    ramfc_paddr:int ->
    method_paddr:int ->
    userd:int option ->
    unit
  (** [set_channel_gpfifo_descs ~params ~ramfc_paddr ~method_paddr ~userd]
      fills the embedded memory descriptors of a channel-allocation blob:
      the RAM-FC and instance memory from the contiguous page at
      [ramfc_paddr], the method buffer at [method_paddr], and — when
      [userd] is the user-D physical base — the error-notifier and user-D
      descriptors a user channel needs. *)

  val reserved_pdes_params :
    page_size:int ->
    virt_addr_lo:int ->
    virt_addr_hi:int ->
    num_levels:int ->
    levels:(int * int * int * int) list ->
    bytes
  (** [reserved_pdes_params ~page_size ~virt_addr_lo ~virt_addr_hi
      ~num_levels ~levels] is the parameters that copy the server-reserved
      page directory entries covering [\[virt_addr_lo, virt_addr_hi\]] into
      a privileged address space. Each of [levels] is a page-table level's
      [(physaddress, size, pageshift, aperture)]. *)

  type promote_entry = {
    buffer_id : int;  (** Which graphics context buffer this promotes. *)
    gpu_virt_addr : int;  (** Virtual address to bind, or [0]. *)
    gpu_phys_addr : int;  (** Physical address to bind, or [0]. *)
    size : int;  (** Buffer size when promoting by physical address. *)
    phys_attr : int;  (** Physical attributes when binding physically. *)
    b_initialize : bool;  (** Initialize the buffer on promotion. *)
    b_nonmapped : bool;  (** Bind physically without a virtual mapping. *)
  }
  (** One graphics-context-buffer promotion entry. *)

  val promote_ctx_params :
    client:int -> obj:int -> entries:promote_entry list -> bytes
  (** [promote_ctx_params ~client ~obj ~entries] is the parameters that
      promote the graphics context buffers [entries] to the channel [obj]
      of [client]. *)

  (** {2:gsp_boot Boot and RPC layer} *)

  type boot = Falcon of Flcn.t | Cot of Flcn_cot.t
  (** How the chip boots the GSP: through the falcon bootloader or, on the
      newest parts, the chain-of-trust security processor. *)

  val falcon_boot : Flcn.t -> boot
  (** [falcon_boot f] is the falcon boot path over [f]. *)

  val cot_boot : Flcn_cot.t -> boot
  (** [cot_boot c] is the chain-of-trust boot path over [c]. *)

  type core_actions = {
    core_reset : unit -> unit;
    core_start : unit -> unit;
    core_wait_halted : unit -> unit;
    core_resume : unit -> unit;
  }
  (** The boot actions the CPU sequencer drives during GSP boot (see
      {!run_cpu_seq}). *)

  val falcon_core_actions : Flcn.t -> libos_args_sysmem:int -> core_actions
  (** [falcon_core_actions f ~libos_args_sysmem] is the CPU-sequencer boot
      actions over the falcon [f]: reset and detach the GSP falcon, start
      and wait on its core, and — for resume — restart it as RISC-V with
      [libos_args_sysmem] in its mailbox, run the SEC2 booter, and wait for
      the secure-boot handoff (raising {!Timeout_error} or [Failure] on a
      handoff failure). *)

  type t
  (** The type for the GSP boot and RPC layer of one device. *)

  val create : Nvdev.t -> boot:boot -> t
  (** [create nvdev ~boot] is the GSP layer of [nvdev] booting through
      [boot]. It holds no prepared state until {!init_sw}. *)

  val libos_args_sysmem : t -> int
  (** [libos_args_sysmem t] is the physical address of the GSP
      boot-argument region, valid after {!init_sw}. *)

  val wpr_meta_sysmem : t -> int
  (** [wpr_meta_sysmem t] is the physical address of the
      write-protect-region metadata, valid after {!init_sw}. *)

  val gpfifo_class : t -> int
  (** [gpfifo_class t] is the channel class selected for the chip, valid
      after {!init_sw}. *)

  val compute_class : t -> int
  (** [compute_class t] is the compute engine class selected for the chip,
      valid after {!init_sw}. *)

  val dma_class : t -> int
  (** [dma_class t] is the copy engine class selected for the chip, valid
      after {!init_sw}. *)

  val init_sw : t -> unit
  (** [init_sw t] allocates the shared command and status queues and the
      GSP boot arguments in system memory, loads the GSP firmware image
      behind its page hierarchy and the RISC-V bootloader, builds the
      write-protect metadata, prefills the command queue with the system
      info and registry table, and selects the engine classes.

      Requires the device's falcon layer to have run its [init_sw] first
      (the write-protect metadata checks the reserved-tables offset
      against it). Raises [Failure] on a missing or mismatched firmware
      file, and {!Timeout_error} if the command queue does not come up. *)

  val init_hw : t -> unit
  (** [init_hw t] brings the status queue up, waits for the GSP to report
      initialization done, programs the BAR1 block, and builds the golden
      image: a privileged client, device, subdevice and address space, the
      server-reserved page directory copy, a channel, and the promoted
      graphics context buffers.

      Requires {!init_sw} and the falcon layer's [init_hw] to have run.
      Raises {!Timeout_error} if the GSP does not report ready, and
      [Failure] on a driver error during the golden-image build. *)

  val fini_hw : t -> unit
  (** [fini_hw t] tells the GSP the driver is unloading, for a fast device
      shutdown. Requires {!init_hw} to have brought the queues up. *)

  val drain_responses : t -> unit
  (** [drain_responses t] consumes any pending status-queue messages,
      dispatching their events; an error-log or fault event latches the
      device's error state. Requires {!init_hw} to have brought the status
      queue up. *)

  val rpc_rm_alloc :
    t ->
    hparent:int ->
    hclass:int ->
    ?params:Nv_tables.blob ->
    client:int ->
    unit ->
    int
  (** [rpc_rm_alloc t ~hparent ~hclass ?params ~client ()] allocates a
      driver object of class [hclass] under [hparent] for [client] and is
      its handle (the client handle for the root class). A channel
      allocation gets its memory descriptors filled, a user address space
      gets its page directory set, and a user compute object promotes the
      graphics context to its channel. *)

  val rpc_rm_control :
    t ->
    hobject:int ->
    cmd:int ->
    ?params:Nv_tables.blob ->
    client:int ->
    unit ->
    unit
  (** [rpc_rm_control t ~hobject ~cmd ?params ~client ()] invokes control
      command [cmd] on [hobject] for [client]. The driver's in-place
      update of [params] is copied back into it, with the GB20x
      work-submit-token patch applied where it fires. *)
end
