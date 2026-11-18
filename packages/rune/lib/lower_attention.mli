(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Attention as kernels of its own.

    {!lower} computes {!Nx.Prim.Attention} without storing anything of the
    pair loop's shape: a kernel folds each query row over its keys in tiles,
    with a running maximum, sum and accumulator of weighted values (online
    softmax), and divides at the end. A tile of keys whose every pair of
    a workgroup's rows the lengths and the mask hide is skipped, and one
    whose every pair the mask shows runs without evaluating it. The page
    table skips no tile: a tile where the lengths and the mask show a pair
    runs, reading the pages of its keys whose entries are in range.

    A kernel's code, registers and workgroup memory are bounded by its tiles,
    whatever [dk] and [dv]: it loops over the keys' tiles and the head's
    dimensions. The kernel that folds a row's runs of keys holds one term a
    run: at most the row's tiles of keys, and the least power of two that
    reaches the target's [stream_threads] over a workgroup's threads, its
    [local_workgroups] on the host. On a GPU, on a tensor core where one
    chains its accumulator and otherwise on an FFMA micro-tile, the keys'
    dimensions go in chunks of at most 256 bytes and the values' columns in
    blocks of at most 512 bytes of accumulators, 128 float32 columns, a
    block a workgroup, which computes the tile's scores for itself; the
    weights stay in the registers of the product they chain into where the
    core chains its accumulator, and otherwise go through workgroup memory,
    and the block's sums stay in registers. On the host a tile's scores are
    computed once, then its weights multiply the values a few columns at a
    time.

    {b Numerics.} Scores, their maxima and sums, the weights and [lse] compute
    in the accumulator, in base 2: each score is multiplied by [log₂ e] and
    exponentiated by [exp2], and [lse] is [(m + log₂ l) ln 2] for the row's
    maximum [m] and sum [l]. A row's weights are each [2{^x - m{_t}}] at its
    tile's running maximum [m{_t}], times one [α = 2{^m{_t'-1} - m{_t'}}]
    for each later tile [t'] where the maximum rose. The arguments' roundings
    sum to one rounding of [x - m], as in a two-pass softmax, and each rise
    adds an [exp2] and a product, [(2E + 1) u] for [E] the ulps of
    {!Nx.exp2}; the sum [l] adds one rescale a tile, of 16, 32 or 64 keys.
    Together these stay within the bound rune's {!Nx.logsumexp} states
    ({!Rune.section-numerics}) for tiles of at least 9 keys. On a tensor
    core each weight rounds once to [v]'s dtype before the value product,
    which sums in the core's order; elsewhere, on an FFMA micro-tile and on
    the host, each [acc] sums its products in key order, each weight in the
    accumulator. Keys split into runs fold the runs' states in order, one
    rescale and one addition more a weight. [o] is [acc / l] rounded once
    into its dtype. A tile skipped holds no visible pair; a NaN or an
    infinity at a hidden key within a row's length reaches [o] only through
    a tile that runs.

    {!linear} computes the derivative's transpose as three kernels: [δ] by
    rows, then [q̄] by query rows over their keys and [k̄] and [v̄] by keys
    over their group's rows, each pair's weight recomputed from [lse] and
    no sum split across workgroups. Past 64 dimensions of keys or values,
    [k̄] and [v̄] take a kernel each, four kernels in all, which compute
    eight products a tile of pairs: [S], [dP] and [q̄] for [q̄], [S], [dP] and
    [k̄] for [k̄], and [S] and [v̄] for [v̄]. A call over values of no column,
    whose output is its [lse], takes no [v̄] kernel and no [dP]. Its [lse],
    [q̄] and [k̄] are the bits of the same call over values with [ō = 0]
    where both calls' tiles hold as many keys, in as many runs, and their
    derivatives' tiles as many keys and rows. A tile's keys share workgroup
    memory with their values' block, so values of more columns may take
    fewer keys a tile, and then the bits may differ; both calls hold the
    bound rune's Numerics section states. Each product runs on the tensor
    core, the weights and the scores' cotangents rounded once to the
    operands' dtype; each sum adds in a fixed order of the shapes and the
    target. To 160 dimensions each operand's dimensions are one block; past
    160 of keys or values the derivative's default computes it, as products
    over the stored weights: on an idle RTX 5000 Ada the kernels take 0.19
    to 0.94 of the default's time at 129 and 160 dimensions, and their
    footprint is 1.4 to 17 times smaller, but at 256 one block or blocks of
    128 take 1.5 to 1.6 times the default's time unmasked. *)

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
    dtype, no load [spec.tangents] marks, keys and values of 1 to 128
    dimensions, rows and keys, and a float16 or bfloat16 query on a tensor
    core that chains its accumulator, as {!lower}'s. Elsewhere it is
    [None], and the derivative computes as its default. *)

(**/**)

val forward_tiles :
  Renderer.t ->
  Nx_array.Dtype.any ->
  q:int array ->
  k:int array ->
  dv:int ->
  (int * int) option
(** [forward_tiles r dt ~q ~k ~dv] is [(keys, runs)] of {!lower}'s kernel on
    [r]'s tensor core for dense queries and keys of [dt] shaped [q] and [k],
    [[lead…; heads; rows; dims]], values of [dv] columns and a float32
    accumulator: the keys a tile holds and the runs each row's keys split
    into. It is [None] where the call takes no core, whose tiles hold 32 keys
    and whose runs do not depend on [dv]. *)

val derivative_tile : Tc.t -> wide:bool -> int
(** [derivative_tile tc ~wide] is the keys or rows a tile of {!linear}'s
    kernels holds on [tc], a whole number of steps of its [N] and [K]; [wide]
    for keys or values past 64 dimensions. *)

val lower_transposed :
  device:Ops.device ->
  renderer:Renderer.t ->
  ('v 's 'd. ('v, 's, 'd) Nx.t -> Ops.t) ->
  'r Nx.Prim.t ->
  Ops.t array option
(** [lower_transposed ~device ~renderer node op] is {!linear}'s nodes of the
    query's, keys' and values' cotangents of the [Attention] [op] at its own
    [o] and [lse] ({!lower}), each operand [x] read from [node x], for
    cotangents of ones of [o] and [lse], where [renderer]'s target takes
    the derivative's kernels: the kernels a compiled gradient runs, made
    without a device, for the suites to render on the host. [None] for
    paged keys, and where {!linear} is.

    Raises [Invalid_argument] if [op] is not an [Attention]. *)

(**/**)
