(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The representation behind {!Launch.t}, which {!Qmd} reads. *)

type launch = {
  kernel : Cubin.kernel;
  gpu : Gpu.t;
  layout : Defs.qmd;  (** The descriptor version the GPU's class reads. *)
  shared_bytes : int;  (** The block's shared memory, the driver's included. *)
  shared_config : int;  (** The multiprocessor's shared memory configuration. *)
  max_shared_config : int;
      (** The configuration of the most shared memory a launch can take. *)
}
