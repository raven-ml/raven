(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Automatic differentiation over parameter trees.

    {!value_and_grad} differentiates scalar losses with respect to {!Ptree.t}
    leaves. By default, all leaves must share one floating dtype; this enables a
    single forward/backward pass.

    Use {!value_and_grad_aux} to return auxiliary data (for example updated
    layer state) alongside the loss. Use {!value_and_grad_mixed} when mixed
    dtypes are required. *)

(** {1:core Core} *)

val value_and_grad :
  (Ptree.t -> (float, 'l) Rune.t) -> Ptree.t -> (float, 'l) Rune.t * Ptree.t
(** [value_and_grad f params] is [(f params, grads)].

    [params] must contain only floating-point leaves and all leaves must have
    the same dtype/layout witness.

    Raises [Invalid_argument] if a leaf is non-float, or if dtypes/layout differ
    across leaves. Error messages include leaf paths. *)

val value_and_grad_aux :
  (Ptree.t -> (float, 'l) Rune.t * 'aux) ->
  Ptree.t ->
  (float, 'l) Rune.t * Ptree.t * 'aux
(** [value_and_grad_aux f params] differentiates [fst (f params)] and returns
    [(loss, grads, aux)].

    The same dtype constraints and errors as {!value_and_grad} apply. *)

val value_and_grad_mixed :
  (Ptree.t -> (float, 'l) Rune.t) -> Ptree.t -> (float, 'l) Rune.t * Ptree.t
(** [value_and_grad_mixed f params] supports mixed floating dtypes/layouts by
    grouping leaves and running multiple autodiff passes.

    {b Warning.} [f] may be evaluated multiple times (once per dtype/layout
    group).

    Raises [Invalid_argument] if any leaf is non-float. Error messages include
    leaf paths. *)

val grad : (Ptree.t -> (float, 'l) Rune.t) -> Ptree.t -> Ptree.t
(** [grad f params] is [snd (value_and_grad f params)]. *)
