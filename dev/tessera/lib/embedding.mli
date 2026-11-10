(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Embeddings of pixels, and the files GeoTessera reads.

    An embedding array is float32 [[pixels; dims]], a row per pixel of a
    {!Pixels.t}. A pixel with no observation in either sensor has no embedding,
    and its row is NaN. A student's Matryoshka prefix of [k] values
    ({!Model.prefixes}) is the slice [Nx.slice [A; R (0, k)]], as it is, and
    every function here takes it as it takes a whole embedding.

    The int8 form of an embedding is GeoTessera's. Per pixel, it is a scale
    [s = max (max |e|, 1e-12) / 127] and codes [round (e / s)] clipped to
    [[−127, 127]], each step in float32, halves rounded to even. A pixel with no
    embedding has scale NaN and codes [0], which GeoTessera reads as no data. *)

val encode :
  shape:Batch.shape ->
  'd Nx.Placement.t ->
  ('d Batch.t -> 'd Nx.float32_t) ->
  Nx.host Pixels.t ->
  Nx.host Nx.float32_t
(** [encode ~shape at f px] is the embedding of [px]'s pixels by [f], a model's
    {!Model.apply} or its compiled form. It packs [px] into batches of [shape]
    with {!Batch.v2_counts}, places each at [at] and applies [f]. The host packs
    the next batch while [at]'s devices compute one, and waits for each result
    in turn.

    Raises what {!Batch.pack} raises. *)

val quantize : 'd Nx.float32_t -> 'd Nx.int8_t * 'd Nx.float32_t
(** [quantize e] is the int8 form of [e], [[…; dims]]: the codes, of [e]'s
    shape, and the scales, of [e]'s shape without its last axis. *)

val save : string -> grid:int * int -> Nx.host Nx.float32_t -> unit
(** [save name ~grid:(h, w) e] writes [e] to [name.npy], float32 [[h; w; dims]].

    Raises [Invalid_argument] if [e] does not have [h · w] rows. *)

val save_int8 : string -> grid:int * int -> Nx.host Nx.float32_t -> unit
(** [save_int8 name ~grid:(h, w) e] writes the int8 form of [e]: the codes to
    [name.npy], int8 [[h; w; dims]], and the scales to [name_scales.npy],
    float32 [[h; w]].

    Raises [Invalid_argument] if [e] does not have [h · w] rows. *)
