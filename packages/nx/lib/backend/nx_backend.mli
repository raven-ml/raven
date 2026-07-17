(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The eager execution engine, as a dune virtual library.

    This module is the seam that selects nx's eager engine at link time. The
    public [Nx] module is a single frontend instantiation whose tensors wrap
    this module's tensor type, and whose operations fall back to this module
    when no effect handler intercepts them. Linking an alternative
    implementation therefore swaps the engine underneath the whole ecosystem —
    the tensor type, the effect vocabulary, and every consumer (rune, kaun, …)
    are unchanged.

    [nx.c] is the default implementation. To provide another engine, implement
    this interface in a library with [(implements nx.backend)] and link it in
    the final executable; one engine is selected per executable. Device-level
    execution (GPU kernels under [Rune.jit]) is a separate mechanism and does
    not go through this seam. *)

include Nx_core.Backend_intf.S

(* TODO: [unit -> context] cannot serve an engine that genuinely needs
   construction parameters (a GPU device index, memory limits). Resolving that
   means growing THIS virtual interface — e.g. a virtual [type config] with
   [val default_config : config] and
   [create_context : ?config:config -> unit -> context] — which changes the
   contract for every engine and is a deliberate, coordinated decision, not
   something a single implementation may do on its own. Unresolved; revisit with
   the device-surface work. *)
val create_context : unit -> context
(** [create_context ()] builds a fresh execution context for this engine.

    This signature is part of the virtual-library interface, so every engine
    must satisfy it as written: construction from [unit], with the engine
    choosing its own defaults, since no engine can unilaterally vary it.
    Configuration knobs (device index, memory limits) do not enter through this
    seam; device-level execution is rune/tolk's domain. *)
