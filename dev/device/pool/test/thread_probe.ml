(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A burst of jobs *)

external burst : unit -> unit = "probe_burst"
external burst_stop : unit -> unit = "probe_burst_stop"

(* Bodies on a worker *)

external worker_mask : unit -> bool * (string * bool) list = "probe_worker_mask"

let faults =
  [ "SIGSEGV"; "SIGBUS"; "SIGFPE"; "SIGILL"; "SIGTRAP"; "SIGABRT"; "SIGSYS" ]

(* Children made by fork *)

type scenario = Threads | Job | Stack | Faults | Limited

external in_child : scenario -> string * int array = "probe_in_child"
external fork : unit -> unit = "probe_fork"
external limits_threads : unit -> bool = "probe_limits_threads"

(* Threads *)

external running_threads : unit -> int = "probe_running_threads"
