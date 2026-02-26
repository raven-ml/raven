(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Batch evaluation.

    Evaluate all code cells in a document sequentially, collecting outputs. *)

val run :
  create_kernel:(on_event:(Kernel.event -> unit) -> Kernel.t) -> Doc.t -> Doc.t
(** [run ~create_kernel doc] creates a kernel, executes all code cells in [doc]
    in order, collects outputs into each cell, shuts down the kernel, and
    returns the updated document. *)
