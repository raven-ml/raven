(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** nx.c's kernels over host memory.

    A value of [Nx_c] is an array over a buffer on {!Nx_device.host}. Its
    operations are nx's vocabulary, run by the C engine on the calling domain
    and the engine's thread pool. [Nx_c] is the kernel layer of
    [Nx.Backend.host]: programs use [Nx], whose values on the host are these. *)

type context = unit
(** nx.c needs no context: the host is its only device. *)

include
  Nx_array.Backend_intf.S
    with type ('a, 'b) t = ('a, 'b) Nx_array.t
     and type context := context
