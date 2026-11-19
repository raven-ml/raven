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

    The draws are a function of the key, the transformer's weights and dtype,
    the inputs, [reveal] and the temperature. Each object's order is
    [Nx.argsort ~axis:1] of {!Nx.Rng.uniform_at} at the inputs' placement, from
    [Nx.Rng.fold_in key 0], of dtype float64 and shape [[|b; n|]]. Step [s]'s
    tokens are {!Nx.Rng.categorical} from [Nx.Rng.fold_in key (s + 1)] of the
    step's [[|b; k; v|]] logits less each row's largest, divided by the
    temperature. Logits that differ in their last bits, as on another device
    set, can move a draw near a boundary between two tokens.

    AION-1's code reveals the [n] tokens in near-equal counts, larger first. For
    a DESI spectrum from an object's Gaia XP coefficients, at eight steps:

    {[
    let desi xp_bp xp_rp =
      let input m tokens : _ Tokens.t =
        { modality = Modality.Scalar m; tokens; present = None }
      in
      Roar.sample ~key:(Nx.Rng.key 0)
        ~reveal:[ 35; 34; 34; 34; 34; 34; 34; 34 ]
        (Transformer.of_archive (Hub.weights Base))
        [ input Xp_bp xp_bp; input Xp_rp xp_rp ]
        (Modality.Spectrum Desi)
    ]} *)

val sample :
  key:'d Nx.Rng.key ->
  ?temperature:float ->
  reveal:int list ->
  ('b, 'd) Transformer.t ->
  'd Tokens.t list ->
  Modality.t ->
  'd Nx.int64_t
(** [sample ~key ~reveal t inputs target] is a draw of [target]'s tokens,
    [[|b; n|]], for each of the [b] objects of [inputs], [n] the target's
    {!Modality.tokens}: step [s] reveals [List.nth reveal s] tokens of each
    object. [temperature] defaults to [1.].

    Each step encodes and decodes with [t] as {!Transformer.encode} and
    {!Transformer.decode} do, checks included, and draws eagerly, so [t]'s set
    must have eager kernels: on another set, such as a GPU's, draw with a
    {!sampler}.

    Raises [Invalid_argument] if [temperature] is negative or not finite, if a
    count of [reveal] is below [1] or they do not sum to [n], if [inputs] is
    empty or holds [target], and as {!Transformer.encode} and
    {!Transformer.decode} do. *)

(** {1:samplers Compiled samplers} *)

type ('b, 'd) sampler
(** The type for compiled samplers of one target modality with a transformer
    computing at float dtype ['b] on the device set ['d]. *)

val sampler : ('b, 'd) Transformer.t -> Modality.t -> ('b, 'd) sampler
(** [sampler t target] draws [target]'s tokens as {!sample} with [t] does, each
    step one program [Rune.jit] compiles from [t]'s computation
    ({!Transformer.eager}), with [t]'s weights captured.

    The sampler keeps, for as long as it lives, a program of the orders per
    batch size and placement, and step programs per count of [reveal], list
    of inputs' modalities and shapes, and placement: one for a positive
    temperature, which is the program's argument, and one for the temperature
    [0.]. So a sweep over temperatures adds no program, and the programs go
    with the sampler. *)

val draw :
  ('b, 'd) sampler ->
  key:'d Nx.Rng.key ->
  ?temperature:float ->
  reveal:int list ->
  'd Tokens.t list ->
  'd Nx.int64_t
(** [draw s ~key ~reveal inputs] is {!sample}[ ~key ~reveal t inputs target]
    for [s]'s transformer and target, computed by [s]'s programs: the same
    draws where the logits agree, to the accuracy [Rune.jit] states. It checks
    [inputs] on the host first, as {!Transformer.check} does, so a refusal
    comes before any program runs.

    Raises [Invalid_argument] as {!sample}. *)
