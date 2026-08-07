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

    Training-mode masks come from an explicit {!Nx.Rng} key when [?key] is given
    — the mask is a pure function of the key and [x]'s shape — and from the
    ambient {!Nx.Rng} scope otherwise; open one with {!Nx.Rng.with_key} for
    reproducibility. Either form compiles under {!Rune.val-jit}, provided the
    key it ultimately draws from is an input of the jitted step rather than one
    the trace closes over. *)

val apply :
  rate:float ->
  training:bool ->
  ?key:Nx.Rng.key ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t
(** [apply ~rate ~training ?key x] is [x] with dropout applied.

    With [training = true], each element of [x] is zeroed independently with
    probability [rate] and the survivors are scaled by [1 / (1 - rate)]
    (inverted dropout), so the result's expectation is [x]. With [?key], the
    mask is [Nx.Rng.bernoulli key ~p:(1. -. rate)] at [x]'s shape: the same key,
    rate and shape give the same mask, so derive a fresh key per call
    ({!Nx.Rng.split}, {!Nx.Rng.fold_in}) for a fresh mask. Without [?key], each
    call draws a fresh mask from the implicit RNG scope. Either way the mask
    selects at [x]'s dtype and is a constant of differentiation, so gradients
    flow to [x] through the surviving elements only.

    With [training = false] (or [rate = 0.]), the result is [x], unchanged.

    {b Under jit.} The mask must come from a key that depends on the jitted
    function's inputs: pass [?key] as a key leaf of the step's input structure,
    or one {!Nx.Rng.fold_in}-derived from such a leaf. Keyless dropout works
    too, if the step body runs inside {!Nx.Rng.with_key} on such a key — what
    raises {!Rune.Jit_error} is a key the trace closes over, whichever form
    drew from it, since the mask would be a constant replayed on every call.

    {b Under vmap.} A key the mapped function closes over is a constant of the
    map, so every lane draws the identical mask. For independent lanes, fold the
    lane index into the key with {!Nx.Rng.fold_in_axis}; a manual
    {!Nx.Rng.split} into per-lane keys stacked along a mapped axis does the same
    thing by hand.

    Raises [Invalid_argument] if [rate] is outside \[[0];[1]). *)
