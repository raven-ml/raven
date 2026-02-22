(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Terminal notebook interface.

    Provides a full-screen TUI for viewing and executing notebooks using the
    Mosaic TEA framework. The TUI is kernel-agnostic: callers supply a kernel
    factory function. *)

val run :
  create_kernel:(on_event:(Quill.Kernel.event -> unit) -> Quill.Kernel.t) ->
  string ->
  unit
(** [run ~create_kernel path] launches the notebook TUI for the file at [path].
    [create_kernel] is called once to obtain a kernel; the TUI owns the kernel
    lifecycle and calls [shutdown] on exit. The notebook is loaded from [path]
    using {!Quill_markdown.of_string} and saved back on request. *)
