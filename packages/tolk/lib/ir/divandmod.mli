(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Division and modulo folding for index-typed expressions. *)

val div_and_mod_symbolic : Kernel.t -> Kernel.t option
(** [div_and_mod_symbolic node] tries all div/mod folding rules on [node].
    Only fires on index-typed {!Op.Idiv} and {!Op.Mod} nodes. *)

(** {1 Expression analysis helpers} *)

val vmin : Kernel.t -> int64
(** [vmin node] is a conservative lower bound for an index-typed expression. *)

val vmax : Kernel.t -> int64
(** [vmax node] is a conservative upper bound for an index-typed expression. *)

val split_add : Kernel.t -> Kernel.t list
(** [split_add node] flattens an addition tree into its additive terms. *)

val const_factor : Kernel.t -> int64
(** [const_factor node] extracts the constant multiplicative factor.
    Returns [1L] for non-multiply or non-constant-factor terms. *)

val divides : Kernel.t -> int64 -> Kernel.t option
(** [divides node c] returns [Some (node / c)] if [node] is evenly divisible
    by [c], or [None] otherwise. *)
