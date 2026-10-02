(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** OCaml toplevel kernel for Quill.

    Provides an in-process OCaml toplevel as a {!Quill.Kernel.t}. Stdout and
    stderr are streamed in real time during execution. The
    {{!Quill.Cell.display}display tags} a cell prints produce
    {!Quill.Cell.Display} outputs, in order with its stdout and stderr. *)

val initialize_if_needed : unit -> unit
(** [initialize_if_needed ()] ensures the OCaml toplevel environment is
    initialized. Safe to call multiple times; only the first call has effect. *)

val add_packages : string list -> unit
(** [add_packages pkgs] resolves each findlib package name and adds its
    directory to the toplevel load path, marking each as already linked into the
    executable. Unknown packages are silently skipped. *)

val load_package : string -> unit
(** [load_package pkg] resolves the findlib package [pkg] and all its transitive
    dependencies, adds their directories, and dynamically loads their bytecode
    archives. Packages already loaded or marked in-core via {!add_packages} are
    skipped. Raises if the package is not found. *)

val install_printer : string -> (unit, string) result
(** [install_printer name] installs a toplevel pretty-printer by evaluating
    [#install_printer name;;]. The printer must be resolvable in the current
    toplevel environment (i.e. its module directory was previously added via
    {!add_packages}).

    [Error report] if the toplevel does not install it, with the toplevel's
    report: [name] is unbound, or is not a printer. *)

val create :
  ?setup:(unit -> unit) ->
  on_event:(Quill.Kernel.event -> unit) ->
  unit ->
  Quill.Kernel.t
(** [create ?setup ~on_event ()] creates a new OCaml toplevel kernel. Kernel
    events are delivered by calling [on_event]. [setup] is called once before
    the first cell execution, after toplevel initialization -- use it to call
    {!add_packages} and {!install_printer}. *)
