(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Bodies on a worker *)

external worker_mask_raw : bool -> bool * (string * bool) list
  = "rig_pool_test_worker_mask"

let blocking = function Rig_pool_probe.Compute -> false | Blocking -> true
let worker_mask kind = worker_mask_raw (blocking kind)

let faults =
  [ "SIGSEGV"; "SIGBUS"; "SIGFPE"; "SIGILL"; "SIGTRAP"; "SIGABRT"; "SIGSYS" ]

(* Children made by fork *)

type scenario = Threads | Job | Stack | Faults | Limited

external in_child_raw : bool -> scenario -> string * int array
  = "rig_pool_test_in_child"

let in_child kind s = in_child_raw (blocking kind) s
external fork_beside_blocking : unit -> bool * string
  = "rig_pool_test_fork_beside_blocking"

external limits_threads : unit -> bool = "rig_pool_test_limits_threads"

(* Threads *)

external running_threads : unit -> int = "rig_pool_test_running_threads"
