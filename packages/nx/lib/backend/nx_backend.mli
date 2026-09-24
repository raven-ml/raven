(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The host's execution engine, as a dune virtual library.

    This module is the seam that selects the engine of host tensors at link
    time. The public [Nx] module is a single frontend instantiation whose host
    tensors wrap this module's tensor type, and whose operations on them fall
    back to this module when no effect handler intercepts them. Linking an
    alternative implementation therefore swaps the host engine underneath the
    whole ecosystem: the tensor type, the effect vocabulary, and every consumer
    (rune, kaun, …) are unchanged.

    [nx.c] is the default implementation. To provide another engine, implement
    this interface in a library with [(implements nx.backend)] and link it in
    the final executable; one engine is selected per executable. Devices are
    engines carried by values ([Nx.Device]); this seam selects the host engine.
*)

include Nx_core.Backend_intf.S

val create_context : unit -> context
(** [create_context ()] builds a fresh execution context for this engine. The
    host engine needs no parameters: a device index is part of a device's name,
    and devices are not selected here. *)
