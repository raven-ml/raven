(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type answer = string * int array

external cores : unit -> int = "rig_pool_test_hooked_cores"

(* Entering and closing *)

external stale_add : unit -> answer = "rig_pool_test_stale_add"
external late_entry : int -> answer = "rig_pool_test_late_entry"

(* Waiting and waking *)

external caller_sleeps : held:bool -> answer = "rig_pool_test_caller_sleeps"

type wake_order = Bit_first | Decision_first | Asleep

external wake : wake_order -> answer = "rig_pool_test_wake"
external narrow_burst : unit -> answer = "rig_pool_test_narrow_burst"

(* Fork *)

external fork_parked : unit -> answer = "rig_pool_test_fork_parked"
external fork_running : unit -> answer = "rig_pool_test_fork_running"
