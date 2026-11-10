(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Pixels packed for one call of a model.

    A model sees each sensor's observations of a pixel as a set of {e tokens},
    each an observation with a {e count}: the number of copies of it the model
    computes on. An attention or pooling key of count [m] whose score gains
    [log m] weighs exactly as [m] copies of it do, since a set's transformer and
    its pooling read no position, so the copies are never computed.

    A batch lays tokens out in {e rows} of [width] tokens. A row holds the
    tokens of several pixels, each in a {e slot}: a run of consecutive tokens.
    Each token carries its slot's bounds [start] and [stop] in the row. Its
    queries see the keys [start] to [stop − 1], and the pooling of a slot reads
    the same keys. Tokens past a row's last slot are {e padding}, of empty
    range, so they see no key and no query sees them. A batch has a fixed
    {!shape}, so a compiled model compiles once per shape.

    Tokens are raw: a model standardises them with its own statistics. *)

(** {1:counts Counts} *)

type counts = { s2 : Nx.host Nx.int32_t; s1 : Nx.host Nx.int32_t }
(** The type for a count per observation: [s2.{j}] is the count of the
    observation at index [j] of [Nx_ragged.values] of a {!Pixels.t}'s [s2], and
    [s1] likewise. An observation of count [0] is left out. *)

val v2_counts : Nx.host Pixels.t -> counts
(** [v2_counts px] is the counts with which TESSERA v2 computes on [px]. A
    pixel's [n] observations of a sensor fill [b], the least multiple of [8] at
    or above [n], at most [256], and each counts the times the fill takes it:
    - for [n ≤ b], each observation once, then [r = b − n] more. For [r ≤ n],
      the observations split in order into [r] runs, the first [n mod r] of
      [⌊n / r⌋ + 1] and the others of [⌊n / r⌋], and the observation at offset
      [⌊s / 2⌋] of each run of [s] counts once more. For [r > n], observation
      [i mod n] counts once more for each [i] from [0] to [r − 1];
    - for [n > 256], numpy's [linspace]: observation [⌊j · ((n − 1) / 255)⌋] for
      each [j] from [0] to [254], the quotient and the product in float64, and
      observation [n − 1], each once.

    Cost: one pass over [px]'s offsets. *)

val expand : Nx.host Pixels.t -> counts -> Nx.host Pixels.t * counts
(** [expand px c] is [px] with each observation repeated as many times as it
    counts, each copy counting [1]: the duplicate-filled pixels a model trains
    on, as copies dropped out independently. A row lays out its copies in
    passes, pass [k] holding in order the observations of count at least [k], so
    [expand px (v2_counts px)] fills each row in the order TESSERA v2's
    reference does: every observation once, then the fill.

    Raises [Invalid_argument] if [c] does not have a count for each of [px]'s
    observations. *)

(** {1:batches Batches} *)

type shape = {
  pixels : int;  (** Pixels per batch, at most. *)
  s2_rows : int;  (** Rows of Sentinel-2 tokens. *)
  s1_rows : int;  (** Rows of Sentinel-1 tokens. *)
  slots : int;  (** Pixels per row, at most. *)
  width : int;  (** Tokens per row. *)
}
(** The type for the static shape of a batch. *)

val typical : pixels:int -> slots:int -> shape
(** [typical ~pixels ~slots] is the shape of [pixels] pixels, at most [slots] a
    row, whose rows of [256] tokens hold [40] Sentinel-2 and [64] Sentinel-1
    tokens a pixel, the typical counts of a tile. A batch of [pixels] whose
    pixels hold more stops before its pixels are full. *)

type 'd rows = {
  tokens : 'd Nx.float32_t;
      (** [[rows; width; bands + 2]]: a token's {!Pixels.s2_bands} or
          {!Pixels.s1_bands} bands, its day of year, and its {e source}, which
          is [0] for a Sentinel-2 observation or an ascending Sentinel-1 one,
          [1] for a descending one, and [-1] for a {e stand-in}. A stand-in is
          the token of a sensor that observed nothing, and standardises to
          zeros. *)
  log_count : 'd Nx.float32_t;
      (** [[rows; width]]: [log count], [0.] at padding. *)
  start : 'd Nx.int32_t;  (** [[rows; width]]: the token's slot's first key. *)
  stop : 'd Nx.int32_t;  (** [[rows; width]]: the slot's last key, plus 1. *)
  slot_start : 'd Nx.int32_t;  (** [[rows; slots]]: each slot's first key. *)
  slot_stop : 'd Nx.int32_t;  (** [[rows; slots]]: its last key, plus 1. *)
  place : 'd Nx.int64_t;
      (** [[rows; width]]: the place in {!field-pixel} of the token's pixel, [0]
          at padding. *)
  at : 'd Nx.int64_t;
      (** [[pixels]]: each pixel's slot, [row · slots + slot]. *)
}
(** The type for one sensor's rows. An empty slot, and padding, has bounds [0]
    and [0]. *)

type 'd t = {
  s2 : 'd rows;
  s1 : 'd rows;
  pixel : 'd Nx.int64_t;
      (** [[pixels]]: the index in the packed {!Pixels.t} of each pixel place,
          [-1] at a place left empty. *)
}
(** The type for batches. *)

val pack : shape -> Nx.host Pixels.t -> counts -> from:int -> Nx.host t * int
(** [pack shape px counts ~from] packs [px]'s pixels from index [from] in order,
    until the batch's pixels are full or a pixel's tokens fit in no row, and is
    the batch with the index of the first pixel it did not consider. Each
    sensor's tokens of a pixel are its observations of nonzero count, in [px]'s
    order; a sensor with none, beside one with some, has one stand-in of count
    [1], day [0]. A pixel with no token in either sensor is skipped, and no
    place holds it. A pixel takes the first row of each sensor with room for its
    tokens and a free slot.

    Raises [Invalid_argument] if a field of [shape] is not positive, if [px]'s
    sensors have different numbers of rows, if [counts] does not have a count
    for each of [px]'s observations, or if a pixel has more than [width] tokens
    of a sensor. *)

val walk : ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> 'd t -> 'e t
(** [walk c b] walks [b]'s arrays, so [Nx.Ptree.v walk] describes a batch. *)

val place : 'e Nx.Placement.t -> 'd t -> 'e t
(** [place at b] is [b] with each array at [at], as {!Nx.place}. *)

val stack : 'd t list -> 'd t
(** [stack bs] is the batches [bs], of one shape, each array stacked along a new
    axis [0]: row [i] of each array is [bs]'s [i]-th batch's, as a [Rune.scan]
    over the batches passes them.

    Raises [Invalid_argument] if [bs] is empty or two batches differ in shape.
*)
