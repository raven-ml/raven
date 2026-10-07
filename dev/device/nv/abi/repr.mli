(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The representations one module builds and another reads, which their
   interfaces export private: Qmd builds structures, and reads the launches
   Launch builds. *)

type 'v hole = { at : int; bits : int; value : 'v Packet.term }
type 'v structure = { bytes : string; holes : 'v hole list }

type launch = {
  kernel : Cubin.kernel;
  gpu : Gpu.t;
  layout : Defs.qmd;  (** The descriptor version the GPU's class reads. *)
  shared_bytes : int;  (** The block's shared memory, the driver's included. *)
  shared_config : int;  (** The multiprocessor's shared memory configuration. *)
}
