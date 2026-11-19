(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Dropout regularization (Srivastava et al., 2014).

    Dropout has no parameters and no state, only a train/eval mode: {!apply}
    takes the same [~training] flag as the other mode-sensitive layers (see
    {!Batch_norm}), so one model forward serves both training and
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
    a key from a scope around it, so a compiled step passes its dropout key as
    an input, as [~key] or as a scope it opens at that key.

    [~key] may be a batch of keys whose batch shape broadcasts to [x]'s shape,
    aligned on the right as {!Nx.Rng} aligns keys with a sampler's parameters.
    Each key draws the elements of [x] it reaches, as it alone would draw
    them, so keys of batch shape [[| r; t; 1 |]] ({!Nx.Rng.reshape}) give each
    row of an [x] of shape [[| r; t; c |]] a mask that does not depend on the
    other rows or their number. *)

val mask :
  rate:float -> ?key:'d Nx.Rng.key -> (float, 'b, 'd) Nx.t -> 'd Nx.bool_t
(** [mask ~rate ?key x] is [Nx.Rng.bernoulli ?key p], [p] of [x]'s shape
    holding [1 - rate] at float32 whatever [x]'s dtype: [true] where
    {!apply} keeps an element of [x] in training mode. It reads no element of
    [x].

    Raises [Invalid_argument] if [rate] is outside \[[0];[1]), or if [key]'s
    batch shape does not broadcast to [x]'s shape. *)

val apply :
  rate:float ->
  training:bool ->
  ?key:'d Nx.Rng.key ->
  (float, 'b, 'd) Nx.t ->
  (float, 'b, 'd) Nx.t
(** [apply ~rate ~training ?key x] is [x] with dropout applied.

    With [training = true], each element of [x] is zeroed independently with
    probability [rate] and the survivors are scaled by [1 / (1 - rate)]
    (inverted dropout), so the result's expectation is [x]. The result is
    [x * m / (1 - rate)], where [m] is {!mask}[ ~rate ?key x] as [0] and [1].
    The same key, rate and shape give the same mask, so derive a fresh key
    per call ({!Nx.Rng.split}, {!Nx.Rng.fold_in}) for a fresh mask. Without
    [?key], each call draws a fresh mask from the implicit RNG scope. Either way the
    mask selects at [x]'s dtype and is a constant of differentiation, so
    gradients flow to [x] through the surviving elements only.

    The mask lies where [x] does, whole where [x] is a sum
    ({!Nx.Placement.partial}): scaling each term alike, dropout keeps a sum
    one.

    With [training = false] (or [rate = 0.]), the result is [x], unchanged.

    {b Under jit.} A compiled step passes its dropout key as an input, as
    [~key] or as a scope it opens at that key ({!Nx.Rng.with_key}), and the
    key lies on [x]'s set, as [~key] must. A key leaf of the step's input
    structure, or one {!Nx.Rng.fold_in}-derived from such a leaf, gives a
    fresh mask per call. A key the compiled function closes over draws the
    same mask on every call, and a draw without a key outside a scope the
    function opens raises [Invalid_argument], as every traced function's
    does.

    {b Under vmap.} A key the mapped function closes over is a constant of the
    map, so every lane draws the identical mask. For independent lanes, fold the
    lane index into the key,
    [Nx.Rng.fold_in_tensor key (Nx.cast Nx.int64 (Rune.lane_index axis x))]
    inside [Rune.vmap ~axis], or map over a batch of keys from
    {!Nx.Rng.split_batch}.

    Raises [Invalid_argument] if [rate] is outside \[[0];[1]), or if [key]'s
    batch shape does not broadcast to [x]'s shape, in either mode. *)
