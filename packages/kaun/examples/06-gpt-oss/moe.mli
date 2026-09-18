(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The gpt-oss mixture-of-experts block (OpenAI, 2025).

    A biased linear router scores every expert for a token, the [k] best are
    kept, and their scores alone go through a softmax. An expert is a gated
    feed-forward: its first projection yields interleaved features, even ones
    the gate and odd ones the linear part, combined by {!activation}; a second
    projection returns to the model width. The block's output is the selected
    experts' outputs weighted by the router.

    Every function acts on each token alone: a token's output depends on no
    other token of the batch. *)

(** {1:params Parameters} *)

(** The type for the weights of one projection of every expert. *)
type 'a weight =
  | Float of 'a  (** [[| experts; inputs; outputs |]]. *)
  | Mxfp4 of { blocks : Mxfp4.blocks; scales : Mxfp4.scales }
      (** The checkpoint's packed form, [blocks] of shape
          [[| experts; outputs; inputs / 32; 16 |]]. *)

type 'a t = {
  router : 'a Kaun.Linear.t;  (** Model width to [experts], with a bias. *)
  gate_up : 'a weight;  (** Model width to [2 * intermediate]. *)
  gate_up_bias : 'a;  (** [[| experts; 2 * intermediate |]]. *)
  down : 'a weight;  (** [intermediate] to model width. *)
  down_bias : 'a;  (** [[| experts; model width |]]. *)
}
(** The type for the block's parameters over payload ['a]. *)

(** {1:forward Forward} *)

val route :
  k:int ->
  (float, 'b) Nx.t t ->
  (float, 'b) Nx.t ->
  (int32, Nx.int32_elt) Nx.t * (float, 'b) Nx.t
(** [route ~k p x] is [(experts, weights)] for the tokens [x] of shape
    [[| tokens; width |]], both of shape [[| tokens; k |]]: the [k] experts with
    the greatest router logits, greatest first, and the softmax of those [k]
    logits. Exactly [k] distinct experts are selected whatever the logits: among
    equal logits the lowest expert comes first, as for {!Nx.top_k}. *)

val activation : limit:float -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [activation ~limit h] splits the last axis of [h] into interleaved [gate]
    (even) and [linear] (odd) features and is
    [gate' * sigmoid (1.702 * gate') * (linear' + 1)], with [gate'] the gate
    clamped above at [limit] and [linear'] the linear part clamped to
    \[[-limit], [limit]\]. The last axis halves. *)

(** The type for formulations of the block. Both compute the same function;
    packed weights are dequantised for the rows each form reads. *)
type form =
  | Gather
      (** Gathers the [k] selected experts' weights for each token and applies
          them as a batched product: arithmetic proportional to [k], and
          [tokens * k] expert rows read. The decode form. *)
  | Dense
      (** Applies every expert to every token and keeps the selected ones by
          their weights, zero elsewhere: arithmetic proportional to [experts],
          each weight read once. The prefill form. *)

val apply :
  form ->
  k:int ->
  limit:float ->
  (float, 'b) Nx.t t ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t
(** [apply form ~k ~limit p x] is the block applied to [x], whose last axis is
    the model width; the result has [x]'s shape. *)
