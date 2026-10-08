(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The GPU's power manager (MP1): messages to its firmware through three
    registers, its clocks, the GPU's whole reset (mode 1) and its machine-check
    banks.

    A message is a 32-bit ID and a 32-bit argument; the firmware answers in the
    response register: 1 for done, another value for a refusal. The IDs depend
    on the power manager's version, as amdgpu's message headers state them.

    The GPU's owner serializes calls. *)

val message : Discovery.version -> string -> int option
(** [message mp1 name] is the ID of the message or clock [name], as
    ["PPSMC_MSG_SetSoftMinByFreq"] or ["PPCLK_UCLK"], for a power manager of
    version [mp1], if it has one. Pure. *)

val dpm : Discovery.version -> features:int -> string -> bool
(** [dpm mp1 ~features clock] is [true] iff the DPM of [clock], a [PPCLK_] name,
    is among the enabled [features] the power manager of MP1 [mp1] reports:
    [false] for a clock it has no DPM feature for. Pure. *)

val clock_request : clock:int -> int -> int
(** [clock_request ~clock v] is the argument of a message about clock [clock]
    and value [v], below 2{^ 16}: the clock in bits 16-31, the value below.
    Pure. *)

type t
(** The type for power managers. *)

val make : Regs.t -> Gmc.t -> table:int -> t
(** [make r gmc ~table] is the power manager, whose driver table is at the
    physical address [table] of the GPU's memory.

    Raises [Invalid_argument] if its version has no messages, which
    {!Regs.layout} refuses first. *)

val alive : t -> bool
(** [alive s] is [true] iff the power manager's firmware answers. *)

val start : t -> unit
(** [start s] gives the firmware its driver table and enables its features. *)

val clocks : t -> [ `Lowest | `Highest ] -> unit
(** [clocks s level] holds the memory, fabric and SoC clocks, and the graphics
    clock where the power manager lets the driver set it, at their lowest or
    highest level, each whose DPM the power manager reports enabled: a clock
    whose DPM is off, as before the power manager's features are enabled, keeps
    its boot frequency. A refused request of a clock whose DPM runs raises
    {!Regs.Stuck}. *)

val reset : t -> unit
(** [reset s] resets the GPU whole (mode 1) and waits for its function to answer
    again: 500 ms, then its vendor ID read from configuration space, which fails
    fast where a register read would stall the bus, for at most 2 s. Outside a
    fabric only; a fabric's GPUs reset together. Raises {!Regs.Stuck} if the
    function does not answer. *)

val banks : t -> string
(** [banks s] is the machine-check banks the firmware holds, uncorrectable then
    correctable, for the report of a fatal hardware error. *)
