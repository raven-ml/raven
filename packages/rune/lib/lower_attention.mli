(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Attention as kernels of its own.

    {!lower} computes {!Nx.Prim.Attention} without storing anything of the
    pair loop's shape: a kernel folds each query row over its keys in tiles,
    with a running maximum, sum and accumulator of weighted values (online
    softmax), and divides at the end. A tile of keys that every pair of a
    workgroup's rows hides is skipped.

    {b Numerics.} Scores, their maxima and sums, the weights and [lse] compute
    in the accumulator, in base 2: each score is multiplied by [log₂ e] and
    exponentiated by [exp2], and [lse] is [(m + log₂ l) ln 2] for the row's
    maximum [m] and sum [l]. A row's weights are each
    [2{^x - m{_t}}] at its tile's running maximum [m{_t}], times one
    [α = 2{^m{_t'-1} - m{_t'}}] for each later tile [t'] where the maximum
    rose. The arguments' roundings sum to one rounding of [x - m], as in a
    two-pass softmax, and each rise adds an [exp2] and a product,
    [(2E + 1) u] for [E] the ulps of {!Nx.exp2}; the sum [l] of [n] terms
    over [n / 32] tiles adds one rescale a tile. Together these stay within
    the bound rune's {!Nx.logsumexp} states ({!Rune.section-numerics}) for
    tiles of at least 9 keys. [o] is [acc / l] rounded once into its dtype,
    each [acc] a sum of products in key order. A tile skipped holds no
    visible pair; a NaN or an infinity at a hidden key within a row's
    length reaches [o] only through a tile that runs. *)

open Tolk

val lower :
  device:Ops.device ->
  renderer:Renderer.t ->
  ('v 's 'd. ('v, 's, 'd) Nx.t -> Ops.t) ->
  'r Nx.Prim.t ->
  Ops.t array
(** [lower ~device ~renderer node op] is the node of [o] and of [lse] of the
    [Attention] [op], computed on [device] by kernels [renderer] renders, each
    operand [x] read from [node x]. Where [op] has no row, no key or no key
    head, [o] is zeros and [lse] is [-∞] and no kernel runs.

    Raises [Invalid_argument] if [op] is not an [Attention]. *)
