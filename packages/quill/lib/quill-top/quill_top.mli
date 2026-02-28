(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** OCaml toplevel kernel for Quill.

    Provides an in-process OCaml toplevel as a {!Quill.Kernel.t}. Stdout and
    stderr are streamed in real time during execution. Rich outputs (images,
    HTML) are emitted via {!Quill.Cell.Display_tag} semantic tags on the
    toplevel formatter. *)

val initialize_if_needed : unit -> unit
(** [initialize_if_needed ()] ensures the OCaml toplevel environment is
    initialized. Safe to call multiple times; only the first call has effect. *)

val add_packages : string list -> unit
(** [add_packages pkgs] resolves each findlib package name and adds its
    directory to the toplevel load path. Unknown packages are silently skipped.
    Must be called after {!initialize_if_needed}. *)

val install_printer : string -> unit
(** [install_printer name] installs a toplevel pretty-printer by evaluating
    [#install_printer name;;]. The printer must be resolvable in the current
    toplevel environment (i.e. its module directory was previously added via
    {!add_packages}). Silently does nothing on failure. *)

val install_printer_fn :
  ty:string -> (Format.formatter -> Obj.t -> unit) -> unit
(** [install_printer_fn ~ty f] registers [f] as a pretty-printer for values of
    type [ty] (e.g. ["Hugin.figure"]). The type is looked up in the toplevel
    environment. Unlike {!install_printer}, the function does not need to be
    resolvable by name -- it is passed directly. Silently does nothing if the
    type cannot be resolved. *)

val create :
  ?setup:(unit -> unit) ->
  on_event:(Quill.Kernel.event -> unit) ->
  unit ->
  Quill.Kernel.t
(** [create ?setup ~on_event ()] creates a new OCaml toplevel kernel. Kernel
    events are delivered by calling [on_event]. [setup] is called once before
    the first cell execution, after toplevel initialization -- use it to call
    {!add_packages} and {!install_printer}. *)
