(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the NVIDIA path's suites share. *)

val hold_gpu : unit -> unit
(** [hold_gpu ()] is {!Rig_gpu_lock.hold} if the machine has an NVIDIA GPU. A
    suite calls it before [Windtrap.run], and a test again before it opens the
    GPU. *)

val files : unit -> int
(** [files ()] is the number of files the process has open. *)

val limit_for : int -> int
(** [limit_for k] is the limit on the process's files ([RLIMIT_NOFILE]) under
    which it may open [k] files more than it has open now. *)

val with_limit : int -> (unit -> 'a) -> 'a
(** [with_limit n f] is [f ()], run with the process's soft limit on its files
    lowered to [n], and set back after it. Only the calling process is limited.
*)

val occupy : int -> int -> bool
(** [occupy at n] maps [n] inaccessible bytes at the address [at] of the
    process, unless something is mapped there: [true] if it did. *)

val vacate : int -> int -> unit
(** [vacate at n] unmaps the [n] bytes at [at]. *)
