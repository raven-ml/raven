(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** How the port's outputs are compared with upstream's.

    Two comparisons, each answering its own question.

    {b One function.} The port and upstream, both at float64 on the same weights
    and inputs, differ only by rounding. Its size is estimated so: a sum of [K]
    products rounds [K] times by at most [u] relative; unrelated roundings add
    as a random walk, about [√K u] relative to the terms; along the model's
    longest chain of sums, about [σ = u Σ √K] relative to the output's largest
    magnitude [y]; and the largest of [n] differences of two such errors is
    typically near [2 √(ln n) σ]. This is an estimate, not a bound: an outlier
    channel ahead of a layer norm amplifies errors past it, and a maximum
    exceeds its typical value at times. What decides the check is the gap around
    it. Float64's rounding is [2{^29}] times smaller than float32's, so two
    implementations of one function agree far below any float32 rounding. A
    different function (another layer norm eps, GELU's tanh form, a stored table
    recomputed) differs by what that change moves, whatever the precision, and a
    change smaller than float32's rounding is one no user of float32 outputs can
    see. The threshold sits in that gap, [2{^29}] wide.

    {b As accurate as upstream.} At float32 the port owes upstream's accuracy.
    Its distance to upstream's float64 output is compared with upstream's own
    float32 distance to it, on the same input:
    [max |ours32 - up64| <= c * max |up32 - up64|], with {!c} = 16. The rule
    exists to catch lost precision: a float32 path whose products, sums or
    accumulators narrow to float16 loses 13 bits of significand, and to bfloat16
    16, so its error grows by [2{^13}] or more. Without lost precision the ratio
    of the two distances still varies, since a maximum taken over a few elements
    of high sensitivity varies from one rounding pattern to another. [c] sits
    between the two: [2{^4}] leaves four bits to that variation and nine below
    the smallest loss the rule must catch. A change in what is computed is the
    float64 comparison's to catch. The reference is upstream's measured
    distance, so the comparison holds the port to upstream on each model and
    input. *)

val u64 : float
(** [u64] is float64's unit roundoff, [2{^-53}]. *)

val encoder : Whisper.Model.dims -> int list
(** [encoder dims] is the depths of the encoder's sums along its longest path:
    the two convolutions, then per block its layer norm, query, scores (the
    head's width), weighted sum (the audio context), output, layer norm and the
    MLP's two projections, then the last layer norm. *)

val decoder : Whisper.Model.dims -> tokens:int -> int list
(** [decoder dims ~tokens] is the depths of the logits' sums at a prefix of
    [tokens] tokens: {!encoder}'s, the key projection of the audio features,
    then per block its self-attention (whose weighted sum is over at most
    [tokens] tokens), its cross-attention (over the audio context) and its MLP,
    then the last layer norm and the output projection. *)

val same : int list -> n:int -> float -> float
(** [same depths ~n y] is the float64 estimate [2 √(ln n) u64 (Σ √K) y] for the
    depths [depths] of an output of [n] elements whose largest magnitude is [y],
    with [ln n] at least [1]: the threshold of the comparison of one function.
*)

val c : float
(** [c] is [16.], the factor of the comparison of accuracy. *)
