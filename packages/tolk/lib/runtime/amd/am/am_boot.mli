(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Boot state machine for driver-less AMD GPU devices.

    Bundles an {!Amdev.t} with its firmware set and hardware-IP blocks
    and drives them through boot ({!val-init}), shutdown ({!fini}) and
    fault recovery ({!recover}).

    Two scratch registers carry the boot protocol between sessions, so
    a later session — from this process or any other driver of the same
    protocol — can tell what state the hardware is in:

    - [regSCRATCH_REG7] holds {!version} while the device was set up by
      this protocol and its boot-memory structures are intact.
    - [regSCRATCH_REG6] holds [1] while a session is active, and on
      {!fini} the session's error flag ([1] after a hardware fault, [0]
      on a clean shutdown).

    A device whose [regSCRATCH_REG7] matches boots partially — only the
    compute and DMA engines are re-initialized, everything else keeps
    the previous session's state in boot memory. A mismatch, a session
    that never finalized ([regSCRATCH_REG6] non-zero), a latched
    translation fault, or [AM_RESET=1] in the environment forces the
    full bring-up instead, resetting the device first when a previous
    driver left its firmware running. *)

type t = {
  adev : Amdev.t;  (** The device the blocks drive. *)
  fw : Amdev.Firmware.t;  (** The firmware set fed to the blocks. *)
  soc : Am_ip.Soc.t;
  gmc : Am_ip.Gmc.t;
  ih : Am_ip.Ih.t;
  psp : Am_ip.Psp.t;
  smu : Am_ip.Smu.t;
  gfx : Am_ip.Gfx.t;
  sdma : Am_ip.Sdma.t;
  mutable partial_boot : bool;
      (** [true] when {!val-init} kept the previous session's state and
          only re-initialized the engines. *)
}
(** The type for bootable devices: the device, its firmware set and its
    hardware-IP blocks. *)

val version : int
(** [version] is the boot-protocol stamp held in [regSCRATCH_REG7]
    while a device is set up: [0xA0000008]. The value is a cross-driver
    contract; changing it strands devices booted by other drivers of
    the protocol and vice versa. *)

val create : ?fw:Amdev.Firmware.t -> Amdev.t -> t
(** [create adev] prepares [adev] for boot: loads the firmware set for
    the discovered hardware (see {!Amdev.Firmware.create}; [fw]
    overrides it) and creates every block, in the fixed order that
    gives each its boot-memory addresses — a partial boot reuses the
    previous session's layout, so the order is part of the protocol.
    Also installs the memory manager's after-mapping hook, flushing
    both memory hubs' TLBs once they are up. [adev] must be booting. *)

val init : t -> unit
(** [init t] boots the device: decides between a partial and a full
    boot from the scratch registers (see the module preamble, recorded
    in [t.partial_boot]), on a full boot resets a device another driver
    left running (dropping PCI bus mastering around the reset) and
    brings up the die, memory hubs, interrupt rings, security processor
    and power management, then in either case starts the compute and
    DMA engines, raises the clocks ([AM_POWER_LIMIT] in the environment
    instead caps the power draw in watts and lets the firmware manage
    the clocks), enables clock gating, and stamps the scratch
    registers. Leaves the device out of the booting state ({!Amdev.is_booting}).

    Raises [Failure] when the device is part of a multi-die fabric in a
    malformed state (resetting a fabric one die at a time would wedge
    it), or when a block rejects its bring-up; {!Am_ip.Timeout_error}
    when the hardware stops answering. *)

val fini : t -> unit
(** [fini t] shuts the session down: drains and disables the engine
    queues, drops the clocks, handles pending interrupts, and writes
    the session's error flag to [regSCRATCH_REG6] so the next session
    can trust (or distrust) the state left behind. *)

val recover : ?force:bool -> t -> bool
(** [recover t] restores a faulted device without a reboot: handles
    pending interrupts, resets and restarts the compute processors, and
    clears the device's error state. Runs only when the device is in
    the error state ({!Amdev.is_err_state}) or [force] is set (defaults
    to [false]); [true] iff it ran. The caller re-creates its hardware
    queues afterwards — the processors lost them. *)
