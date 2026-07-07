(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_uop

(** Operation decompositions matching tinygrad [codegen/decomp/op.py].

    This module owns the non-dtype operation rewrite families: Threefry,
    floor div/mod lowering, integer division by constants, late algebraic
    rewrites, and backend capability flags. *)

val threefry2x32 : Uop.t -> Uop.t -> Uop.t
(** [threefry2x32 x key] is the software Threefry2x32 expansion of the
    64-bit counter [x] and 64-bit [key]. *)

val magicgu : int -> int -> int * int
(** [magicgu vmax d] is [(m, s)] such that unsigned [x / d] is
    [(x * m) >> s] for all [0 <= x <= vmax].

    Raises [Invalid_argument] if [d <= 0] or no multiplier exists in the
    host integer range. *)

type supported_ops = {
  has_exp2 : bool;
  has_log2 : bool;
  has_sin : bool;
  has_sqrt : bool;
  has_neg : bool;
  has_sub : bool;
  has_max : bool;
  has_shl : bool;
  has_shr : bool;
  has_and : bool;
  has_or : bool;
  has_cmplt : bool;
  has_cmpeq : bool;
  has_fdiv : bool;
  has_threefry : bool;
  has_mulacc : bool;
  is_metal : bool;
  supports_dtype : Dtype.t -> bool;
  disable_fast_idiv : bool;
  force_transcendental : bool;
}
(** The type for backend operation capabilities used by decomp rewrites. *)

val get_simplifying_rewrite_patterns : supported_ops -> Uop.t -> Uop.t option
(** [get_simplifying_rewrite_patterns ops u] is the early rewrite for [u],
    if one applies under [ops]. *)

val get_late_rewrite_patterns : supported_ops -> Uop.t -> Uop.t option
(** [get_late_rewrite_patterns ops u] is the late non-dtype rewrite for [u],
    if one applies under [ops]. *)
