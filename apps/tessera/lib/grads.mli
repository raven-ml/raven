(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A training step's gradient over its groups, a group at a time, for
    [Distil.step] and [Barlow.step]. Only rune differentiates, and this is
    three of its functions composed, so it stays with the app. *)

val sum :
  ('p, 'd) Nx.Ptree.t ->
  ('g, 'd) Nx.Ptree.t ->
  ('g -> 'p -> (float, 's, 'd) Nx.t * ('a, 'b, 'd) Nx.t) ->
  'p ->
  'g ->
  'p * ('a, 'b, 'd) Nx.t
(** [sum p g f params gs] is, over the groups [gs] stacked along axis [0]
    (structure [g]), the sum of the gradients at [params] of each group's
    objective [fst (f group params)], added in group order, and each group's
    [snd (f group params)], stacked along axis [0]. It is a [Rune.scan] of
    [Rune.value_and_grad_aux], so a step holds what one group's backward
    reads, not all of its groups'. *)
