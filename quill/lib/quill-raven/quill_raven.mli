(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Raven-flavored kernel for Quill.

    Wraps {!Quill_top} with automatic setup of all raven packages (Nx, Rune,
    Kaun, Hugin, etc.). Package include paths are resolved via findlib and
    toplevel printers are installed on first execution. *)

val create : on_event:(Quill.Kernel.event -> unit) -> Quill.Kernel.t
(** [create ~on_event] creates a kernel with all raven packages available. On
    the first cell execution, the toplevel is initialized, raven package include
    paths are added, and pretty-printers are installed. *)
