(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The representation behind {!Launch.t}, which {!Qmd} reads, and the
    descriptor layout a GPU's compute class reads. *)

type launch = {
  kernel : Cubin.kernel;
  gpu : Gpu.t;
  shared_bytes : int;  (** The block's shared memory, the driver's included. *)
  shared_config : int;  (** The multiprocessor's shared memory configuration. *)
  max_shared_config : int;
      (** The configuration of the most shared memory a launch can take. *)
}

val layout : Gpu.t -> Defs.qmd option
(** [layout g] is the launch descriptor's layout [g]'s compute class reads:
    version 5 for Blackwell's class, version 3 for Ampere's and Ada's, [None]
    for any other. It allocates nothing. *)
