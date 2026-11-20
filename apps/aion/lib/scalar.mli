(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** AION-1's scalar codecs: tokens for numbers.

    A scalar codec turns a number, such as a galaxy's flux in one band or one
    coefficient of a Gaia spectrum, into a {e token}, an integer, and a token
    back into a number. It holds a {e transform}, which maps a value into the
    space its codebooks were built in, and one {e codebook} per {e channel}: [n]
    float32 {e edges} [e(0) … e(n-1)], quantiles of the training sample in
    increasing order (the redshift codec's are an even grid).

    Encoding transforms a value [v] to [y], in [v]'s dtype, and searches [y]
    among [e(0) … e(n-2)]: the token is the number of these edges below [y]. The
    comparisons are exact: a float64 [y] is compared with the edges as float64,
    any other in float32. Token [k] thus covers [y] in (e(k-1), e(k)\], a [y]
    equal to an edge takes the lower bin, a [y] above [e(n-2)] takes [n-1], and
    NaN takes [n-1]. Decoding reads [e(k)], the upper end of bin [k], and
    inverts the transform in float32, so a value decodes to the top of its bin.

    Some codebooks end in NaN edges, where the training sample held NaN. The
    search is the bisection AION's tokens were made with: start with [lo = 0]
    and [hi = n - 1]; while [lo < hi], with [mid = lo + (hi - lo) / 2], continue
    with [hi = mid] if [e(mid) >= y], and with [lo = mid + 1] otherwise; the
    token is [lo]. On edges in order it counts as above. It goes right at a NaN
    edge, so where NaN edges end a codebook, a search that meets one ends at
    [n - 1], even for a [y] below some finite edges.

    A codec of [c] channels tokenizes values whose last axis has [c] entries,
    the [i]th by codebook [i]. A codec of one channel takes values of any shape.
    Tokens have their values' shape.

    A codec is a structure ({!walk}) whose codebooks are fixed arrays on a
    device set, so [Nx.Ptree.place (module Scalar)] moves it and
    [Nx.Ptree.cast] keeps its codebooks. It computes with values of its set.
    Compiled by [Rune.jit] with the codec as an argument, its tokens and values
    are the eager ones bit for bit on a faithful device: the computation is
    comparisons, gathers and the transform's functions, which [Rune.jit]
    computes exactly there. *)

(** {1:codecs Codecs} *)

(** The type for transforms. *)
type transform =
  | Identity  (** The value itself. *)
  | Log of float
      (** [Log m] is [log (max v m)], with [m] rounded to [v]'s dtype and NaN
          kept, decoded by [exp]. *)
  | Asinh_asinh  (** [asinh (asinh v)], decoded by [sinh (sinh y)]. *)

type ('a, 'd) t
(** The type for scalar codecs on the device set ['d]. A codec holds no
    parameter, so ['a] is free: it is there because [Nx.Ptree.place] and
    [Nx.Ptree.cast] take a structure of a payload and a set. *)

val walk : ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> ('a, 'd) t -> ('b, 'e) t
(** [walk c t] reports [t]'s transform as a case, with the floor of a [Log]
    by its bits, and walks its codebooks as a fixed array at ["edges"]. *)

val v : transform -> 'd Nx.float32_t -> ('a, 'd) t
(** [v tr edges] is the codec of [tr] whose codebooks are the rows of [edges],
    of shape [[|c; n|]].

    Raises [Invalid_argument] if [edges] is not 2-d, or [c] or [n] is [0]. *)

val transform : ('a, 'd) t -> transform
(** [transform t] is [t]'s transform. *)

val edges : ('a, 'd) t -> 'd Nx.float32_t
(** [edges t] is [t]'s codebooks, of shape [[|c; n|]]. *)

(** {1:aion AION-1's codecs} *)

val transform_of : Modality.scalar -> transform
(** [transform_of m] is the transform of [m]'s codec: [Log] at [e{^-3}] for the
    Legacy Survey fluxes and radius, the Gaia fluxes and the parallax,
    [Asinh_asinh] for the XP coefficients, and [Identity] for the others. *)

val of_archive : Modality.scalar -> 'd Nx_io.Archive.t -> ('a, 'd) t
(** [of_archive m a] is [m]'s codec, of transform {!transform_of}[ m], with the
    codebooks of [a], named as in its file in {!Hub.codec}: one for each of
    {!Modality.tokens}[ (Scalar m)], stored as [_quantizer._index_to_val] for a
    codec of one channel, [_quantizer.buckets] for redshift's, and
    [_quantizer.quantizers.i._index_to_val] for channel [i] of the others.
    Each codebook's [_reservoir], the training sample it was computed from,
    is not read.

    Raises [Failure] if [a] does not hold that many float32 codebooks of one
    length, or holds an entry it does not read. *)

val init : key:'d Nx.Rng.key -> Modality.scalar -> ('a, 'd) t
(** [init ~key m] is a codec of [m]'s transform and shapes, at [key]'s set:
    {!Modality.tokens}[ (Scalar m)] codebooks of 1024 edges, as AION-1's, each
    of sorted standard normal draws from [key]. *)

(** {1:tokens Tokens} *)

val encode : ('a, 'd) t -> (float, 's, 'd) Nx.t -> 'd Nx.int64_t
(** [encode t v] is the token of each value of [v].

    Raises [Invalid_argument] if [t] has more than one channel and [v]'s last
    axis does not have one entry per channel. *)

val decode : ('a, 'd) t -> 'd Nx.int64_t -> 'd Nx.float32_t
(** [decode t k] is the value of each token of [k].

    Raises [Invalid_argument] if a token is outside [\[0, n)], or if [t] has
    more than one channel and [k]'s last axis does not have one entry per
    channel. The token [n] is the class past the last edge, which the
    transformer predicts for redshift ({!Modality.vocab} is [1025] there) and
    whose value is unknown: its refusal names it. *)
