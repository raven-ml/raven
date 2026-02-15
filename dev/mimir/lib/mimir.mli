(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Mimir - Text generation with composable logits processors.

    Experimental inference/generation library for the Raven ML ecosystem.
    Provides the autoregressive decode loop, composable logits processors,
    stopping criteria, and generation configuration. *)

include module type of Sampler
