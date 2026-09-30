(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** nx.cpu: nx's kernels over host memory.

    The C engine, run on the calling domain and the engine's thread pool, over
    arrays in memory the host addresses. {!backend} is the backend of
    [Nx.Placement.host]: programs use [Nx], whose values on the host are these
    kernels' arrays. *)

include Nx_backend.S
(** @inline *)

val backend : Nx_backend.t
(** [backend] is nx.cpu's backend, named ["cpu"]. It runs on the devices whose
    memory is the host's ({!Nx_device.shares_host_memory}). *)
