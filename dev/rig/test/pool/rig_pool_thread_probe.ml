(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A burst of jobs *)

external burst : unit -> unit = "rig_pool_test_burst"
external burst_stop : unit -> unit = "rig_pool_test_burst_stop"
external burst_jobs : unit -> int = "rig_pool_test_burst_jobs"

(* Bodies on a worker *)

external worker_mask : unit -> bool * (string * bool) list
  = "rig_pool_test_worker_mask"

let faults =
  [ "SIGSEGV"; "SIGBUS"; "SIGFPE"; "SIGILL"; "SIGTRAP"; "SIGABRT"; "SIGSYS" ]

(* Children made by fork *)

type scenario = Threads | Job | Stack | Faults | Limited

external in_child : scenario -> string * int array = "rig_pool_test_in_child"
external fork : unit -> unit = "rig_pool_test_fork"
external limits_threads : unit -> bool = "rig_pool_test_limits_threads"

(* Threads *)

external running_threads : unit -> int = "rig_pool_test_running_threads"
