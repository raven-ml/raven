(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The gpt-oss mixture-of-experts block (OpenAI, 2025).

    A routing says, for each token, which experts it goes to and with what
    weight. The model computes one from its router's logits, with {!route} for
    gpt-oss, and {!apply} takes it as it comes: a model that routes differently
    brings another function. An expert is a gated feed-forward: its first
    projection yields interleaved features, even ones the gate and odd ones the
    linear part, combined by {!activation}; a second projection returns to the
    model width. The block's output is the selected experts' outputs weighted by
    the routing.

    Every function acts on each token alone: a token's output depends on no
    other token of the batch. *)

(** {1:params Parameters} *)

(** The type for the weights of one projection of every expert. *)
type 'a weight =
  | Float of 'a  (** [[| experts; inputs; outputs |]]. *)
  | Quant of Nx_quant.t
      (** The checkpoint's packed form, of shape
          [[| experts; outputs; inputs |]]. *)

type 'a t = {
  gate_up : 'a weight;  (** Model width to [2 * intermediate]. *)
  gate_up_bias : 'a;  (** [[| experts; 2 * intermediate |]]. *)
  down : 'a weight;  (** [intermediate] to model width. *)
  down_bias : 'a;  (** [[| experts; model width |]]. *)
}
(** The type for the experts' parameters over payload ['a]. *)

val walk : ('a, 'b) Nx.Ptree.Walk.cursor -> 'a t -> 'b t
(** [walk c p] walks [p]'s parts at [gate_up], [gate_up_bias], [down] and
    [down_bias], in that order. A weight reports its case at its path, ["float"]
    or ["quant"]: a float weight is a position of the parameter, and a packed
    one is walked by {!Nx_quant.walk}, its codes and scales fixed tensors. So
    [Nx.Ptree.cast (module Moe) dt p] converts precision and keeps the packed
    weights. *)

(** {1:forward Forward} *)

val route :
  k:int -> (float, 'b) Nx.t -> (int32, Nx.int32_elt) Nx.t * (float, 'b) Nx.t
(** [route ~k logits] is gpt-oss's routing [(experts, weights)] for router
    logits of shape [[| ...; experts |]], both of shape [[| ...; k |]]: the [k]
    experts with the greatest logits, greatest first, and the softmax of those
    [k] logits. Exactly [k] distinct experts are selected whatever the logits:
    among equal logits the lowest expert comes first, as for {!Nx.top_k}. *)

val activation : limit:float -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [activation ~limit h] splits the last axis of [h] into interleaved [gate]
    (even) and [linear] (odd) features and is
    [gate' * sigmoid (1.702 * gate') * (linear' + 1)], with [gate'] the gate
    clamped above at [limit] and [linear'] the linear part clamped to
    \[[-limit], [limit]\]. The last axis halves. *)

val apply :
  limit:float ->
  (float, 'b) Nx.t t ->
  (int32, Nx.int32_elt) Nx.t * (float, 'b) Nx.t ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t
(** [apply ~limit p (experts, weights) x] is the experts applied to [x], whose
    last axis is the model width, and mixed by the routing [(experts, weights)]:
    for each token of [x], the experts to use and their weights, with shape
    [x]'s leading axes followed by the number of experts per token. The result
    has [x]'s shape.

    Packed weights are multiplied with {!Nx_quant.apply}, which decodes a
    bounded chunk at a time eagerly and chooses its form under [Rune.jit]. Float
    weights are gathered for each token's experts, a copy of every selected
    expert: they suit small checkpoints. *)
