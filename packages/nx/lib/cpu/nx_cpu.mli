(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** nx.cpu: nx's backend over host memory.

    The C engine, run on the calling domain and the engine's thread pool, over
    arrays in memory the host addresses. Named ["nx.cpu"], it computes eager
    operations on every device the host computes on ({!Nx_device.runs_on_host}):
    the host and test devices. Programs use [Nx], whose values on the host are
    these kernels' arrays. *)

include Nx_backend.S
(** @inline *)
