(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Probabilistic inference.

    Norn turns a log density over a structure of the caller's own type into
    draws and their diagnostics. Samplers move in unconstrained coordinates that
    a bijector ({!Bij}) maps onto a value's support. *)

module Bij = Bij
