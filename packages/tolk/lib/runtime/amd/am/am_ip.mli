(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Hardware-IP bring-up protocols for driver-less AMD GPU devices.

    Each module drives one IP block of an {!Amdev.t} over its mailbox
    registers: {!Psp} feeds firmware to the security processor, which
    alone can start the other engines, and {!Smu} speaks the
    power-management message protocol. Both address registers by name
    through the device, poll them against the device clock
    ({!Amdev.now_ms}), and place their in-memory structures in the
    device's boot memory region, so a device built with {!Amdev.make}
    runs the full protocols over scripted register access. *)

exception Timeout_error of string
(** Raised when a polled register fails to reach its expected value
    within a protocol's deadline, measured against the device clock. *)

(** {1:smu Power management}

    The management processor accepts one message at a time through a
    register triple: the response register is cleared, the argument and
    message-id registers are written, and completion is polled as the
    response register turning [1]. Message ids differ per firmware
    interface version; the device's discovered version selects its
    table (see {!Amd_tables.smu}). *)

module Smu : sig
  type t
  (** The type for the power-management block of one device. *)

  val create : Amdev.t -> t
  (** [create adev] prepares the block: selects the message table for
      the device's discovered management-processor version and
      allocates the driver table (the shared-memory buffer table
      transfers fill) from boot memory. Requires [adev] to be booting.
      Raises [Invalid_argument] when no message table covers the
      discovered version. *)

  val driver_table_paddr : t -> int
  (** [driver_table_paddr t] is the device-local physical address of
      the driver table. *)

  val init_hw : t -> unit
  (** [init_hw t] points the firmware at the driver table and enables
      all firmware features. Raises {!Timeout_error} if a message goes
      unanswered for 10 seconds. *)

  val is_smu_alive : t -> bool
  (** [is_smu_alive t] is [true] iff the firmware answers a version
      query within 100ms. *)

  val mode1_reset : t -> unit
  (** [mode1_reset t] asks the firmware to reset the whole device,
      through the message the device's generation expects, and gives
      the hardware 500ms to settle (skipped on multi-die fabrics, which
      reset as a group elsewhere). *)

  val read_table : t -> size:int -> int -> bytes
  (** [read_table t ~size arg] asks the firmware to export the table
      selected by [arg] and is the driver table's first [size] bytes
      after the transfer. *)

  val read_clocks : t -> int list -> (int * int list) list
  (** [read_clocks t clks] is the supported frequency levels of each
      clock domain in [clks] that reports any, queried once and cached
      per [clks]. *)

  val set_clocks : t -> level:int option -> unit
  (** [set_clocks t ~level] pins every adjustable clock domain to the
      frequency at index [level] of its level list, counting from the
      end when negative ([-1] is the highest level). [None] lifts the
      limits instead. Domains whose limit messages go unanswered are
      skipped. *)

  val set_power_limit : t -> float -> unit
  (** [set_power_limit t watts] limits the device's power draw to
      [watts], rounded to whole watts and at least 1. *)

  val aca_read_banks : t -> ue:bool -> int64 list list
  (** [aca_read_banks t ~ue] dumps the hardware-error banks the
      firmware reports: 16 registers per bank, uncorrectable errors
      when [ue], correctable otherwise. The empty list when the
      device's interface version cannot report them. *)
end

(** {1:psp Security processor}

    The security processor gates all engine firmware. Its bring-up has
    two stages, both fed through a 1MB staging buffer in device
    memory: the bootloader mailbox loads the components of the secure
    OS, and once that OS runs, commands submitted over a shared-memory
    ring load every other engine's firmware and set up the trusted
    memory region. Each ring submission is a fixed-size command frame
    whose completion is a fence dword the firmware writes back. *)

module Psp : sig
  type t
  (** The type for the security-processor block of one device. *)

  val create : Amdev.t -> fw:Amdev.Firmware.t -> t
  (** [create adev ~fw] prepares the block for loading the firmware set
      [fw]: picks the mailbox register family of the device's
      security-processor generation and allocates the staging buffer,
      command buffer, fence buffer, submission ring and, on generations
      whose trusted memory region is not reserved by the boot firmware,
      that region too, all from boot memory. Requires [adev] to be
      booting. *)

  val is_sos_alive : t -> bool
  (** [is_sos_alive t] is [true] iff the secure OS reports itself
      running. *)

  val init_hw : t -> unit
  (** [init_hw t] runs the bring-up: loads the secure-OS components
      through the bootloader mailbox unless the OS already runs,
      creates the submission ring, sizes and sets up the trusted memory
      region as the generation requires, and loads every firmware image
      of the set in order (power management first, then the remaining
      engines, ending with the graphics-core loader or its autoload
      trigger). Raises {!Timeout_error} when a mailbox or ring response
      never arrives, [Failure] when the firmware rejects a command. *)

  val spatial_partition_cmd : t -> int -> unit
  (** [spatial_partition_cmd t mode] asks the secure OS to partition
      the device's compute dies into [mode] partitions. *)

  (** {2:layout Boot-memory layout}

      Device-local physical addresses of the block's structures, fixed
      at {!create} time. *)

  val msg1_paddr : t -> int
  (** [msg1_paddr t] is the 1MB firmware staging buffer. *)

  val cmd_paddr : t -> int
  (** [cmd_paddr t] is the command frame buffer. *)

  val fence_paddr : t -> int
  (** [fence_paddr t] is the fence buffer the firmware completes
      submissions into. *)

  val ring_paddr : t -> int
  (** [ring_paddr t] is the submission ring. *)

  val tmr_paddr : t -> int
  (** [tmr_paddr t] is the trusted memory region, or [0] when the boot
      firmware reserves it instead. *)
end
