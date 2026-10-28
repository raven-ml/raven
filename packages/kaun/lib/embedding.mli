(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Embedding layers.

    An embedding is a lookup table mapping integer token ids to learned dense
    vectors: a record with one [vocab × dim] payload hole. Construct one with
    {!init} or {!make} and turn id tensors into vector tensors with {!apply}.
    Like the other layers, it composes into models through record nesting, and
    {!walk} makes it a structure. *)

(** {1:types Types} *)

type ('a, _) t = { table : 'a }
(** The type for embedding parameters over payload ['a]. At tensor payloads,
    [table] has shape [[| vocab; dim |]]: one row per token id. *)

(** {1:constructors Constructors} *)

val make :
  ?init:('b, 'd) Init.t ->
  key:'d Nx.Rng.key ->
  vocab:int ->
  dim:int ->
  (float, 'b) Nx.dtype ->
  ((float, 'b, 'd) Nx.t, 'd) t
(** [make ~key ~vocab ~dim dtype] is a fresh table of [vocab] rows of [dim]
    features, drawn from [key] and lying where it lies. [init] initializes the
    table and is applied with [~fan_in:dim] and [~fan_out:dim], so
    variance-scaling initializers target a variance proportional to [1 / dim].
    It defaults to [Init.normal ~stddev:1.0].

    Raises [Invalid_argument] if [vocab] or [dim] is not positive. *)

val init : key:'d Nx.Rng.key -> vocab:int -> dim:int -> ('d Nx.float32_t, 'd) t
(** [init ~key ~vocab ~dim] is [make ~key ~vocab ~dim Nx.float32]: rows drawn
    from [N(0, 1)]. *)

(** {1:applying Applying} *)

val apply :
  ((float, 'b, 'd) Nx.t, 'd) t -> 'd Nx.int64_t -> (float, 'b, 'd) Nx.t
(** [apply p ids] gathers the table row of each id: the result has [ids]'s shape
    with a trailing axis of size [dim] appended, and its [(i, ..., :)] slice is
    row [ids.(i, ...)] of [p.table]. A scalar id yields a single row of shape
    [[| dim |]].

    The gather is differentiable through Rune: the table's gradient accumulates
    each row's cotangent as many times as its id occurs.

    An id outside \[[0], [vocab]), negative included, yields a row of zeros,
    eagerly and under [Rune.jit] alike, and passes no gradient to the table. *)

(** {1:structure Structure} *)

val walk : ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> ('a, 'd) t -> ('b, 'e) t
(** [walk c p] walks [p]'s parameters, [table], at that path: the layer's
    {!Nx.Ptree.S} instance. [Nx.Ptree.instantiate (module Embedding)] is the
    layer at one dtype, and [Nx.Ptree.cast (module Embedding) dtype p] converts
    its precision. *)
