(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Dropout regularization (Srivastava et al., 2014).

    Dropout has no parameters and no state, only a train/eval mode: it is a
    single function taking the same [~training] flag as the other mode-sensitive
    layers (see {!Batch_norm}), so one model forward serves both training and
    evaluation:

    {[
    let forward p ~training x =
      Linear.apply p.l1 x |> Fn.relu
      |> Dropout.apply ~rate:0.1 ~training
      |> Linear.apply p.l2
    ]}

    Training-mode masks draw from the implicit RNG scope of {!Nx.Rng}; wrap the
    training loop in {!Nx.Rng.run} for reproducibility. *)

val apply : rate:float -> training:bool -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [apply ~rate ~training x] is [x] with dropout applied.

    With [training = true], each element of [x] is zeroed independently with
    probability [rate] and the survivors are scaled by [1 / (1 - rate)]
    (inverted dropout), so the result's expectation is [x]. Each call draws a
    fresh mask from the implicit RNG scope; the mask is a constant of
    differentiation, so gradients flow to [x] through the surviving elements
    only.

    With [training = false] (or [rate = 0.]), the result is [x], unchanged.

    {b Note.} Under {!Rune.val-vmap} the implicit RNG draws an identical mask
    for every lane (see the note there); vectorize over a batch axis of [x]
    directly rather than [vmap]ing a dropout forward when the lanes must drop
    independently.

    Raises [Invalid_argument] if [rate] is outside \[[0];[1]). *)
