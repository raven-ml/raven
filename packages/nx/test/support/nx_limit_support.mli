(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A limit of a test process's memory, so that an allocation past it fails in
    the process rather than take the machine's memory. *)

val limit_memory : int -> int
(** [limit_memory room] limits the process to [room] bytes past what it holds
    now: on Linux its address space past what it maps, on Windows its committed
    memory past what it commits. It never raises a limit set outside. It is [0],
    or the error code of a refusal, [ENOSYS] on macOS, whose limits do not bound
    allocations. *)
