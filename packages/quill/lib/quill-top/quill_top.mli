(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** OCaml toplevel kernel for Quill.

    Provides an in-process OCaml toplevel as a {!Quill.Kernel.t}. Captures
    stdout and stderr separately via Unix file descriptor redirection. *)

val initialize_if_needed : unit -> unit
(** [initialize_if_needed ()] ensures the OCaml toplevel environment is
    initialized. Safe to call multiple times; only the first call has effect. *)

val create : on_event:(Quill.Kernel.event -> unit) -> Quill.Kernel.t
(** [create ~on_event] creates a new OCaml toplevel kernel. Kernel events are
    delivered by calling [on_event]. The toplevel is initialized on first
    execution. *)
