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

(* TODO: [create_context : unit -> context] won't work for backends that need
   parameters (e.g. GPU device index, memory limits). We'll likely need
   [create_context : ?config:config -> unit -> context] or similar, with a
   default config for each backend. *)
val create_context : unit -> context
