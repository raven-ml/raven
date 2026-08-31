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
end
