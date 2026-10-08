(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the NVIDIA path's suites share. *)

val gpu_lock : string
(** [gpu_lock] is the environment variable that names the machine's GPU lock
    file: ["DEVICE_NV_TEST_GPU_LOCK"]. *)

val hold_gpu : unit -> unit
(** [hold_gpu ()] returns once this process holds the GPU lock, which it keeps
    until it exits. It skips the test if the machine has no NVIDIA GPU, if
    {!gpu_lock} names no file, or if another process holds the lock. *)

val files : unit -> int
(** [files ()] is the number of files the process has open. *)

val limit_for : int -> int
(** [limit_for k] is the limit on the process's files ([RLIMIT_NOFILE]) under
    which it may open [k] files more than it has open now. *)

val with_limit : int -> (unit -> 'a) -> 'a
(** [with_limit n f] is [f ()], run with the process's soft limit on its files
    lowered to [n], and set back after it. Only the calling process is limited.
*)
