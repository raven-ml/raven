(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Finite scalar quantization: tokens for short vectors.

    Finite scalar quantization (Mentzer et al., "Finite Scalar Quantization:
    VQ-VAE Made Simple", 2023) turns a vector of [d] numbers into a {e code}, an
    integer, by rounding each number onto a few values. A quantizer has [d]
    {e levels} [L(0) … L(d-1)], each at least [2]: number [i] takes one of
    [L(i)] values, and a code is one of the [L(0) ⋯ L(d-1)] tuples of them.

    {e Bounding} maps number [i], [z], into an interval of [L(i)] integers:
    [y = tanh (z + s) h - o], with [h = (L(i) - 1) (1 + ε) / 2] and
    [ε = 10{^-3}], an offset [o] of [0] for an odd [L(i)] and [1/2] for an even
    one, and [s = atanh (o / h)], which makes [y] [0] at [z = 0]. Its {e digit}
    is [q + ⌊L(i)/2⌋], [q] the integer nearest [y], halves to even, so that
    digits lie in [\[0, L(i))]. A vector's code is [Σ digit(i) B(i)], where
    [B(0) = 1] and [B(i+1) = B(i) L(i)]: the first number is the least
    significant. Decoding inverts the code into digits and maps digit [k] of
    number [i] to [(k - ⌊L(i)/2⌋) / ⌊L(i)/2⌋], at most [1] in magnitude.

    A quantizer reads its vectors along the last axis of its input. AION-1's
    images use the levels [[7; 5; 5; 5; 5]], 4375 codes, and its segmentation
    maps [[8; 5; 5; 5]], 1000 codes.

    A quantizer holds no array. Compiled by [Rune.jit] with the quantizer as
    an argument, described by [Nx.Ptree.v walk], its codes and vectors are the
    eager ones bit for bit on a faithful device, a program for each list of
    levels. *)

type t
(** The type for quantizers. *)

val walk : ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> t -> t
(** [walk c q] reports [q]'s levels, so that [Rune.jit] keys a program on
    them. *)

val v : int list -> t
(** [v levels] is the quantizer of [levels].

    Raises [Invalid_argument] if [levels] is empty, a level is below [2], or the
    number of codes passes [max_int]. *)

val levels : t -> int list
(** [levels q] is [q]'s levels. *)

val size : t -> int
(** [size q] is [q]'s number of codes, the product of its levels. *)

val encode : t -> (float, 's, 'd) Nx.t -> 'd Nx.int64_t
(** [encode q z] is the code of each vector along [z]'s last axis, of [z]'s
    shape without that axis, from the digits of its bounded numbers, computed
    in [z]'s dtype. The bounding's constants are float32's, the trained
    model's, for any dtype but float64: [h] is [(L(i) - 1)] times [1 + ε]
    rounded to float32, that product rounded, then halved, and [s] is [atanh]
    of [o / h] rounded, rounded. At float64 each is the exact value rounded.

    Raises [Invalid_argument] if [z]'s last axis does not have one entry per
    level, or if [z] holds NaN, which has no digit; on a set other than the
    host it raises at the next wait ({!Nx.check}). *)

val bound : t -> (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t
(** [bound q z] is each number of [z] bounded, as {!encode} bounds it before
    rounding.

    Raises [Invalid_argument] as {!encode} does for [z]'s shape. *)

val decode : (float, 's) Nx.dtype -> t -> 'd Nx.int64_t -> (float, 's, 'd) Nx.t
(** [decode dt q k] is the vector of each code of [k], along a last axis added
    to [k]'s shape, each number computed in [dt]: [(k - ⌊L/2⌋) / ⌊L/2⌋] rounded
    once.

    Raises [Invalid_argument] if a code is outside [\[0, size q)], as {!encode}
    raises for NaN. *)
