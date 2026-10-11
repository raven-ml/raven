(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The messages of a GPU's power manager (MP1).

    A message is a 32-bit ID and a 32-bit argument the driver writes to two
    registers. The firmware writes 1 into a third register when it has run
    the message, and any other value when it refuses it. The IDs depend on
    the power manager's version. Every value is pure, and any domain may
    call it. *)

val message : Gpu.version -> string -> int option
(** [message mp1 name] is the ID of the message or clock [name], such as
    ["PPSMC_MSG_SetSoftMinByFreq"] or ["PPCLK_UCLK"], for a power manager of
    version [mp1], if it has one. *)

val dpm : Gpu.version -> features:int -> string -> bool
(** [dpm mp1 ~features clock] is [true] iff the DPM of [clock] is among the
    enabled [features] a power manager of version [mp1] reports. It is
    [false] for a clock it has no DPM feature for.

    Raises [Invalid_argument] if [clock] does not start with [PPCLK_]. *)

val clock_request : clock:int -> int -> int
(** [clock_request ~clock v] is the argument of a message about clock
    [clock] and value [v], [v] below 2{^16}: the clock in bits 16 to 31 and
    [v] below them. *)
