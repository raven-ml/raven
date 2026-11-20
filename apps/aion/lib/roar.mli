(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Draws of a modality's tokens given others, by random-order autoregression
    (ROAR; Mizrahi et al., "4M: Massively Multimodal Masked Modeling", NeurIPS
    2023, Appendix A.3).

    A draw reveals the tokens of a {e target} modality over {e steps}, for each
    object of a batch. Each object's [n] positions are taken in an {e order}
    drawn uniformly at random, its own. Step [s] reveals the next
    [List.nth reveal s] positions of each object's order: it encodes the inputs
    followed by the target, whose tokens are present where revealed
    ({!Transformer.encode}); decodes the logits at those positions
    ({!Transformer.decode}); draws a token at each; and makes the tokens present
    for the steps after it. The target starts with no token revealed.

    What a draw samples depends on [reveal]. One step, [[n]], draws each token
    independently given the inputs: the product of the model's per-token
    conditionals (AION-1, arXiv:2510.17960, Eq. 6). One token per step, [n]
    ones, draws each token given the inputs and the tokens before it in the
    order: the joint distribution the model defines along a uniformly random
    order (Hoogeboom et al., "Autoregressive Diffusion Models", ICLR 2022, §2).
    Several tokens per step lie between the two: the tokens of one step are
    independent given those of the steps before it. A one-token target is one
    step, where all three agree.

    A token is drawn with probability proportional to [exp ((l - m) / T)], as
    {!Nx.Rng.categorical} draws it, [l] its logit, [m] the largest logit of its
    row and [T] the {e temperature}: at [1.] the model's distribution, and at
    [0.] the token of the largest logit, the first on ties. A draw ranges over
    the target's whole vocabulary ({!Modality.vocab}): redshift's holds one
    token more than its codec decodes ({!Scalar.decode}).

    Each object draws from a key of its own: [keys], of shape [[|b; 2|]], holds
    one per object. Object [i]'s order is [Nx.argsort] of {!Nx.Rng.uniform_at}
    at the inputs' placement, from key [i] of [Nx.Rng.fold_in keys 0], of dtype
    float64 and shape [[|n|]]. Its step-[s] tokens are {!Nx.Rng.categorical}
    from key [i] of [Nx.Rng.fold_in keys (s + 1)] of its [[|k; v|]] logits less
    each row's largest, divided by the temperature. So an object's draws are a
    function of its key, its inputs, the transformer's weights and dtype,
    [reveal] and the temperature, and not of the objects drawn beside it, where
    its logits are the same bit for bit in either batch. Logits that differ in
    their last bits, as on another device set or possibly in a batch of another
    size, can move a draw near a boundary between two tokens.

    AION-1's code reveals the [n] tokens in near-equal counts, larger first. For
    a DESI spectrum from an object's Gaia XP coefficients, at eight steps:

    {[
    let desi xp_bp xp_rp =
      let input m tokens : _ Tokens.t =
        let present = Nx.ones Nx.bool (Nx.shape tokens) in
        { modality = Modality.Scalar m; tokens; present }
      in
      Roar.sample
        ~keys:(Nx.Rng.split_batch ~n:(Nx.dim 0 xp_bp) (Nx.Rng.key 0))
        ~reveal:[ 35; 34; 34; 34; 34; 34; 34; 34 ]
        (Transformer.of_archive (Hub.weights Base))
        [ input Xp_bp xp_bp; input Xp_rp xp_rp ]
        (Modality.Spectrum Desi)
    ]} *)

val sample :
  keys:'d Nx.Rng.key ->
  ?temperature:float ->
  reveal:int list ->
  ((float, 'b, 'd) Nx.t, 'd) Transformer.t ->
  'd Tokens.t list ->
  Modality.t ->
  'd Nx.int64_t
(** [sample ~keys ~reveal t inputs target] is a draw of [target]'s tokens,
    [[|b; n|]], for each of the [b] objects of [inputs] from its key of [keys],
    [n] the target's {!Modality.tokens}: the {!tokens} of {!step}[ t] folded
    over [reveal] from {!start}[ ~keys ~temperature inputs target], so step [s]
    reveals [List.nth reveal s] tokens of each object. [temperature] defaults to
    [1.].

    It computes eagerly, so [t]'s set must have eager kernels: on another set,
    such as a GPU's, compile {!start} and {!step} and fold them as [sample]
    does.

    Raises [Invalid_argument] if a count of [reveal] is below [1] or they do
    not sum to [n], and as {!start} and {!step} do. *)

(** {1:steps Steps}

    A draw in progress is a value: {!start} makes it and {!step} takes it one
    step further. Compiled with [Rune.jit], with the transformer an argument of
    [step] and its count reported by [Nx.Ptree.Walk.int], they draw on any
    set:

    {[
    let draw () = Nx.Ptree.v Roar.walk_draw in
    let start =
      Rune.jit
        Nx.Ptree.(Nx.Rng.ptree @-> list (v Tokens.walk) @-> returns (draw ()))
        (fun keys x -> Roar.start ~keys ~temperature x target)
    and step =
      Rune.jit
        Nx.Ptree.(
          instantiate (module Transformer)
          @-> v Walk.int @-> draw () @-> returns (draw ()))
        Roar.step
    in
    Roar.tokens (List.fold_left (fun d k -> step t k d) (start keys x) reveal)
    ]}

    They draw {!sample}'s tokens where the logits agree, to the accuracy
    [Rune.jit] states. [step] keeps a program per count, list of inputs'
    modalities and shapes, placement, and whether the temperature is [0.]:
    every positive temperature shares one program. [start]'s temperature is
    captured, a constant of its program, and [start] refuses as it traces. *)

type 'd draw
(** The type for draws in progress of a target's tokens for a batch of
    objects, on the device set ['d]. *)

val walk_draw : ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> 'd draw -> 'e draw
(** [walk_draw c d] reports [d]'s target, its inputs' modalities and whether
    its temperature is [0.], and walks its arrays as fixed ones. *)

val start :
  keys:'d Nx.Rng.key ->
  ?temperature:float ->
  'd Tokens.t list ->
  Modality.t ->
  'd draw
(** [start ~keys inputs target] is a draw of [target]'s tokens given [inputs]
    with no token revealed, each object's order drawn from its key of [keys], at
    the temperature [temperature] (defaults to [1.]).

    Raises [Invalid_argument] if [temperature] is negative or not finite, if
    [inputs] holds [target] or is empty, or if [keys] is not a batch of [b]
    keys, [b] the objects of [inputs]. *)

val step :
  ((float, 'b, 'd) Nx.t, 'd) Transformer.t -> int -> 'd draw -> 'd draw
(** [step t k d] is [d] after one more step, which reveals the next [k]
    positions of each object's order with [t]. Object [i] draws from key [i] of
    [Nx.Rng.fold_in keys (s + 1)], [keys] those {!start} took and [s] the number
    of steps [d] has taken.

    Raises [Invalid_argument] if [k] is not in [[1, n]], [n] the target's
    {!Modality.tokens}, if a position it would reveal is revealed already, as
    after counts that sum past [n], and as {!Transformer.encode} and
    {!Transformer.decode} do. The positions are checked as {!Nx.check} does:
    at the call on the host, eagerly or compiled, and at a later wait on
    another set. *)

val tokens : 'd draw -> 'd Nx.int64_t
(** [tokens d] is [d]'s tokens of the target, [[|b; n|]]: the tokens drawn at
    the positions revealed, and [0] at the others. *)
