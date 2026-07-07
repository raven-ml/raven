(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_uop

(** Transcendental decompositions matching tinygrad
    [codegen/decomp/transcendental.py].

    This module owns polynomial/bit-manipulation expansions for [Exp2],
    [Log2], [Sin], [Sqrt], and the shared exponent-bias helper used by float
    dtype decomposition. *)

val exponent_bias : Tolk_uop.Dtype.t -> int
(** [exponent_bias dtype] is the IEEE-like exponent bias for [dtype]. *)

val get_transcendental_patterns :
  Decomp_op.supported_ops -> Uop.t -> Uop.t option
(** [get_transcendental_patterns ops u] is the transcendental rewrite for
    [u], if one applies under [ops]. *)
