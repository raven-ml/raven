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

(**/**)

(* The footer's key hints, exposed for the tests. *)

type mode =
  | Normal
  | Editing
      (** The type for whether the focused cell is being edited ([Editing]) or
          the notebook is navigated ([Normal]). *)

type footer_action = { key : string; label : string }
(** The type for a key hint: a [key] and what it does, its [label]. *)

val footer_actions : mode -> Quill.Session.t -> footer_action list
(** [footer_actions mode s] is the key hints the footer offers, most important
    first, in [mode] over the notebook [s] while no quit awaits confirmation.
    While a cell of [s] runs, the hint to interrupt it, [Ctrl-C], takes the
    place of the hint to run the focused cell. *)

(**/**)
