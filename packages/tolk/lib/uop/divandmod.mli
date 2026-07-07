(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Integer division and modulo simplification.

    Pattern matcher that folds expressions built from {!Ops.Floordiv} and
    {!Ops.Floormod}. Rewrites preserve Python-style floor division and
    modulo semantics and run before these ops lower to truncating division
    and modulo. *)

(** {1:matcher Matcher} *)

val div_and_mod_symbolic : Upat.Pattern_matcher.t
(** [div_and_mod_symbolic] bundles the div/mod simplification rules into a
    single {!Upat.Pattern_matcher}. It raises [Division_by_zero] when the
    denominator is provably zero; unsupported or ambiguous cases are left
    unchanged. *)
