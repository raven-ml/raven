(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The GPU's power manager (MP1), which this library asks by message to
    set clocks, reset the GPU whole (mode 1) and report machine-check banks.

    A message is a 32-bit ID and a 32-bit argument the driver writes to two
    registers. The firmware writes 1 into a third register when it has run
    the message, and any other value when it refuses it. The IDs depend on
    the power manager's version.

    The values of {!section-messages} are pure, and any domain may call
    them. Calls for one GPU's power manager ({!section-power_manager}) run
    one at a time. *)

(** {1:messages Messages} *)

val message : Rig_amd_abi.Gpu.version -> string -> int option
(** [message mp1 name] is the ID of the message or clock [name], such as
    ["PPSMC_MSG_SetSoftMinByFreq"] or ["PPCLK_UCLK"], for a power manager of
    version [mp1], if it has one. *)

val dpm : Rig_amd_abi.Gpu.version -> features:int -> string -> bool
(** [dpm mp1 ~features clock] is [true] iff the DPM of [clock] is among the
    enabled [features] a power manager of version [mp1] reports. It is
    [false] for a clock it has no DPM feature for.

    Raises [Invalid_argument] if [clock] does not start with [PPCLK_]. *)

val clock_request : clock:int -> int -> int
(** [clock_request ~clock v] is the argument of a message about clock
    [clock] and value [v], [v] below 2{^16}: the clock in bits 16 to 31 and
    [v] below them. *)

(** {1:power_manager The power manager} *)

type t
(** The type for power managers. *)

type levels
(** The type for the frequencies of each clock that power managers of one
    GPU reported, by the version of the firmware that ran: the firmware's
    tables give them. *)

val levels : unit -> levels
(** [levels ()] holds no frequencies. *)

val make : Regs.t -> Gmc.t -> table:int -> levels:levels -> t
(** [make r gmc ~table ~levels] is the power manager, whose driver table is at
    the physical address [table] of the GPU's memory. Before its first
    clock request it asks the firmware's version, then asks the firmware
    for a clock's frequencies only if [levels] lacks them for that version,
    and adds them there, so that the GPU's later power managers made with
    [levels] over the same firmware ask for none. The caller gives [levels]
    to the power managers of one GPU alone, one at a time. *)

val alive : t -> bool
(** [alive s] is [true] iff the power manager's firmware responds to a message
    within 100 ms. *)

val start : t -> unit
(** [start s] gives the firmware the address of its driver table and enables
    all its features.

    Raises {!Regs.Stuck} if the firmware refuses a message or does not
    respond within 10 s. *)

val features : t -> int
(** [features s] is the power manager's enabled features, one bit each, as
    it reports them. *)

val clocks : t -> [ `Lowest | `Highest ] -> unit
(** [clocks s level] holds the memory, fabric, SoC and graphics clocks at
    their lowest or highest frequency. It sets only the clocks whose DPM the
    power manager reports enabled, so a clock whose DPM is off, as before
    {!start}, keeps its boot frequency.

    Raises {!Regs.Stuck} if the power manager refuses a request, or counts
    more than 16 frequencies of a clock. *)

val reset : t -> unit
(** [reset s] resets the GPU whole (mode 1). Outside a fabric it then waits
    500 ms, 1 s from SMU 14 on, and reads the function's vendor ID from its
    configuration space for at most 2 s, until the function responds. On a
    GPU of a fabric it returns immediately. A caller resets only a GPU
    outside a fabric, since a fabric's GPUs reset together.

    Raises {!Regs.Stuck} if the function does not respond. *)

val banks : t -> string
(** [banks s] is the machine-check banks the firmware holds, uncorrectable
    then correctable, for the report of a fatal hardware error. *)
