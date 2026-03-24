(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Parameter-tree optimizer.

    Bridges {!Vega} with {!Ptree}: applies per-parameter optimizer steps across
    all leaves of a heterogeneous parameter tree. *)

(** {1:types Types} *)

type state
(** Optimizer state for a parameter tree. Packs per-leaf {!Vega} states. *)

(** {1:core Core} *)

val init : Vega.t -> Ptree.t -> state
(** [init tx params] initializes optimizer state for all leaves of [params]. *)

val update : state -> Ptree.t -> Ptree.t -> Ptree.t * state
(** [update state params grads] returns [(updates, new_state)].

    Applies {!Vega.update} to each matching leaf pair. The returned [updates]
    tree has the same structure as [params] and can be applied via
    {!apply_updates}. *)

val apply_updates : Ptree.t -> Ptree.t -> Ptree.t
(** [apply_updates params updates] adds [updates] to [params] element-wise
    across all leaves. *)

val step : state -> Ptree.t -> Ptree.t -> Ptree.t * state
(** [step state params grads] returns [(new_params, new_state)].

    Convenience for:
    {[
    let updates, state = update state params grads in
    (apply_updates params updates, state)
    ]} *)

(** {1:serialization Serialization} *)

val state_to_trees : state -> int * Ptree.t list
(** [state_to_trees st] is [(count, trees)] where [count] is the optimizer step
    count and [trees] are the internal state as parameter trees.

    Transforms with no state tensors return an empty list. *)

val state_of_trees : Vega.t -> count:int -> Ptree.t list -> state
(** [state_of_trees tx ~count trees] reconstructs optimizer state from a
    transformation, step count, and serialized trees.

    Raises [Invalid_argument] if the number of trees does not match the
    transformation's expectation. *)

(** {1:grad Gradient Utilities} *)

val clip_by_global_norm : float -> Ptree.t -> Ptree.t
(** [clip_by_global_norm max_norm grads] rescales [grads] so their global L2
    norm does not exceed [max_norm]. Returns [grads] unchanged if the norm is
    already within bounds.

    Raises [Invalid_argument] if a leaf tensor is not floating point. *)

val global_norm : Ptree.t -> float
(** [global_norm t] is the L2 norm across all leaf tensors of [t].

    Raises [Invalid_argument] if a leaf tensor is not floating point. *)
