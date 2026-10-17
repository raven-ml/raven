(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** nx.amd's kernels, the implementation of {!Nx_amd.backend}.

    A device's kernels load from {!Archive} on their first use on it, and
    stay loaded while the device does. A GPU that no carried target covers
    raises [Nx_backend.Refused] from every kernel. *)

val backend : Nx_backend.t
