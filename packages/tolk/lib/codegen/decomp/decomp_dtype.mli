(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_uop

(** Dtype decomposition matchers matching tinygrad [codegen/decomp/dtype.py].

    This module owns the long-as-two-int matcher and float storage emulation
    matcher. Graph-level dtype detection and scheduling are still performed by
    codegen lowering and renderer hooks. *)

val pm_long_decomp : Upat.Pattern_matcher.t
(** [pm_long_decomp] rewrites int64 and uint64 values as pairs of 32-bit
    values. *)

type float_decomp_ctx = {
  from_dtype : Tolk_uop.Dtype.scalar;
  (** Source storage dtype to emulate. *)

  to_dtype : Tolk_uop.Dtype.scalar;
  (** Arithmetic dtype used for emulation. *)
}
(** The type for one float decomposition direction. *)

val pm_float_decomp : float_decomp_ctx -> Upat.Pattern_matcher.t
(** [pm_float_decomp ctx] rewrites loads, stores, casts, bitcasts, and float
    operations from [ctx.from_dtype] storage through [ctx.to_dtype]
    arithmetic. *)

val do_dtype_decomps : Renderer.t -> Tolk_uop.Uop.t -> Tolk_uop.Uop.t
(** [do_dtype_decomps renderer sink] detects unsupported long and compact
    float dtypes reachable from [sink], then applies the minimal matching
    decomposition passes required by [renderer]. *)
