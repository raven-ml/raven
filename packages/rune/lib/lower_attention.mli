(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Attention as kernels of its own.

    {!lower} computes {!Nx.Prim.Attention} without storing anything of the
    pair loop's shape: a kernel folds each query row over its keys in tiles,
    with a running maximum, sum and accumulator of weighted values (online
    softmax), and divides at the end. A tile of keys that every pair of a
    workgroup's rows hides is skipped, and one whose every pair the mask
    shows runs without evaluating it.

    {b Numerics.} Scores, their maxima and sums, the weights and [lse] compute
    in the accumulator, in base 2: each score is multiplied by [log₂ e] and
    exponentiated by [exp2], and [lse] is [(m + log₂ l) ln 2] for the row's
    maximum [m] and sum [l]. A row's weights are each [2{^x - m{_t}}] at its
    tile's running maximum [m{_t}], times one [α = 2{^m{_t'-1} - m{_t'}}]
    for each later tile [t'] where the maximum rose. The arguments' roundings
    sum to one rounding of [x - m], as in a two-pass softmax, and each rise
    adds an [exp2] and a product, [(2E + 1) u] for [E] the ulps of
    {!Nx.exp2}; the sum [l] adds one rescale a tile, of 32 or 64 keys.
    Together these stay within the bound rune's {!Nx.logsumexp} states
    ({!Rune.section-numerics}) for tiles of at least 9 keys. On a tensor
    core each weight rounds once to [v]'s dtype before the value product,
    which sums in the core's order; elsewhere each [acc] sums its products
    in key order. Keys split into runs fold the runs' states in order, one
    rescale and one addition more a weight. [o] is [acc / l] rounded once
    into its dtype. A tile skipped holds no visible pair; a NaN or an
    infinity at a hidden key within a row's length reaches [o] only through
    a tile that runs.

    {!linear} computes the derivative's transpose as three kernels: [δ] by
    rows, then [q̄] by query rows over their keys and [k̄] and [v̄] by keys
    over their group's rows, each pair's weight recomputed from [lse] and
    no sum split across workgroups. Each product runs on the tensor core,
    the weights and the scores' cotangents rounded once to the operands'
    dtype; each sum adds in a fixed order of the shapes and the target. *)

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

type operand = Nx_array.Dtype.any * int array * Ops.t
(** The type for an operand of the derivative: its dtype, shape and node. *)

val linear :
  device:Ops.device ->
  renderer:Renderer.t ->
  Attention.spec ->
  transposed:bool ->
  operand array ->
  operand array ->
  Ops.t array option
(** [linear ~device ~renderer spec ~transposed data args] is the nodes of
    {!Attention.linear}'s results, computed on [device] by kernels [renderer]
    renders, where it has them: transposed, dense keys, [o] of the query's
    dtype, no load [spec.tangents] marks, keys and values of [1] to [128]
    dimensions, rows and keys, and a float16 or bfloat16 query on a tensor
    core that chains its accumulator, as {!lower}'s. Elsewhere it is
    [None], and the derivative computes as its default. *)

(**/**)

val derivative_tile : Tc.t -> wide:bool -> int
(** [derivative_tile tc ~wide] is the keys or rows a tile of {!linear}'s
    kernels holds on [tc], a whole number of steps of its [N] and [K]; [wide]
    for keys or values past 64 dimensions. *)

(**/**)
