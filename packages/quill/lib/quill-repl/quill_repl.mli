(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Interactive OCaml toplevel.

    Provides a REPL using the Mosaic framework in primary screen mode. The
    prompt supports syntax highlighting, phrase-aware submission, tab
    completion, and persistent history. Completed interactions scroll into
    native terminal scrollback. *)

val run :
  create_kernel:(on_event:(Quill.Kernel.event -> unit) -> Quill.Kernel.t) ->
  unit
(** [run ~create_kernel] launches the interactive toplevel. [create_kernel] is
    called once to obtain a kernel; the REPL owns the kernel lifecycle and calls
    [shutdown] on exit. *)

val run_pipe :
  create_kernel:(on_event:(Quill.Kernel.event -> unit) -> Quill.Kernel.t) ->
  unit
(** [run_pipe ~create_kernel] reads OCaml code from stdin, executes it, and
    prints outputs to stdout. For non-tty usage. *)
