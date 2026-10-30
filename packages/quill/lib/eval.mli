(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Batch evaluation.

    Evaluate all code cells in a document sequentially, collecting outputs. *)

val run :
  create_kernel:(on_event:(Kernel.event -> unit) -> Kernel.t) -> Doc.t -> Doc.t
(** [run ~create_kernel doc] creates a kernel, runs each code cell of [doc] in
    order with {!Kernel.execute}, adds their outputs with {!Doc.add_output},
    shuts down the kernel, and returns the updated document. A cell the kernel
    refuses gets its [Error] output, as a cell that raised does, and the cells
    after it still run. *)
