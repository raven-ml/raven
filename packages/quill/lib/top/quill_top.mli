(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** OCaml toplevel kernel for Quill.

    Provides an in-process OCaml toplevel as a {!Quill.Kernel.t}. Stdout and
    stderr are streamed in real time during execution. The
    {{!Quill.Cell.display}display tags} a cell prints produce
    {!Quill.Cell.Display} outputs, in order with its stdout and with its
    stderr. Stdout and stderr are captured apart, so text on one keeps no order
    with text on the other. An uncaught [Failure msg] or [Invalid_argument msg]
    is reported on stdout with [msg] as written, newlines and quotes intact, as
    in [Exception: Failure: msg]. *)

val initialize_if_needed : unit -> unit
(** [initialize_if_needed ()] ensures the OCaml toplevel environment is
    initialized. Safe to call multiple times; only the first call has effect. *)

val add_packages : string list -> unit
(** [add_packages pkgs] resolves each findlib package name and adds its
    directory to the toplevel load path, marking each as already linked into the
    executable. Unknown packages are silently skipped. *)

(** Why {!load_package} did not load a package. *)
type load_error =
  | Not_installed of string
      (** [Not_installed p] says that no installed package is named [p], the
          package asked for or one it requires. *)
  | Failed of { package : string; report : string }
      (** [Failed { package; report }] says that [package], the package asked
          for or one it requires, does not load, and [report] says why. A
          package fails when findlib cannot read its META, or when one of its
          archives is not bytecode, needs a shared library that does not load,
          or refers to a module that no package defines. For an archive,
          [report] is the toplevel's report. *)

val load_package : string -> (unit, load_error) result
(** [load_package pkg] resolves the findlib package [pkg] and all its transitive
    dependencies, adds their directories, and dynamically loads their bytecode
    archives. Packages already loaded or marked in-core via {!add_packages} are
    skipped.

    [Error] if a package is not installed or does not load. The packages loaded
    before it stay loaded. A package is recorded as loaded only once all its
    archives are, so a later [load_package] tries it again. *)

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
(** [create ?setup ~on_event ()] creates a new OCaml toplevel kernel, whose
    {!Quill.Kernel.language} is ["ocaml"]. Kernel events are delivered by
    calling [on_event]. [setup] is called once before the first cell execution,
    after toplevel initialization -- use it to call {!add_packages} and
    {!install_printer}. *)
