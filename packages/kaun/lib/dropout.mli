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
    reproducibility. A traced function, as [Rune.jit]'s, refuses a draw without
    a key, so a compiled step passes [~key]. *)

val apply :
  rate:float ->
  training:bool ->
  ?key:'d Nx.Rng.key ->
  (float, 'b, 'd) Nx.t ->
  (float, 'b, 'd) Nx.t
(** [apply ~rate ~training ?key x] is [x] with dropout applied.

    With [training = true], each element of [x] is zeroed independently with
    probability [rate] and the survivors are scaled by [1 / (1 - rate)]
    (inverted dropout), so the result's expectation is [x]. With [?key], the
    mask is [Nx.Rng.bernoulli key] at probability [1. -. rate] and [x]'s shape:
    the same key, rate and shape give the same mask, so derive a fresh key per
    call ({!Nx.Rng.split}, {!Nx.Rng.fold_in}) for a fresh mask. Without [?key],
    each call draws a fresh mask from the implicit RNG scope. Either way the
    mask selects at [x]'s dtype and is a constant of differentiation, so
    gradients flow to [x] through the surviving elements only.

    With [training = false] (or [rate = 0.]), the result is [x], unchanged.

    {b Under jit.} A key leaf of the step's input structure, or one
    {!Nx.Rng.fold_in}-derived from such a leaf, gives a fresh mask per call. A
    key the compiled function closes over draws the same mask on every call, and
    a draw without a key raises [Invalid_argument], as every traced function's
    does.

    {b Under vmap.} A key the mapped function closes over is a constant of the
    map, so every lane draws the identical mask. For independent lanes, fold the
    lane index into the key,
    [Nx.Rng.fold_in_tensor key (Rune.lane_index axis x)] inside
    [Rune.vmap ~axis], or map over a batch of keys from {!Nx.Rng.split_batch}.

    Raises [Invalid_argument] if [rate] is outside \[[0];[1]). *)
