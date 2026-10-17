(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** nx.cpu's kernels, the module behind {!Nx_cpu.backend}.

    Each kernel runs the C engine on the calling domain, which may share the
    work with the engine's thread pool and release the runtime lock
    meanwhile, and returns once the destination is written.

    The elementwise kernels, reductions, scans, scatters, window sums and
    matrix products compute [int4] and [uint4] at the byte width. They widen
    the operands to [int8] or [uint8], run the byte kernel and cast a result
    of the operands' dtype back, which keeps it modulo 16. *)

include Nx_backend.S
