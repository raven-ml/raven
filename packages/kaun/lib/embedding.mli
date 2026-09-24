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

type 'a t = { table : 'a }
(** The type for embedding parameters over payload ['a]. At tensor payloads,
    [table] has shape [[| vocab; dim |]]: one row per token id. *)

(** {1:constructors Constructors} *)

val make :
  ?init:'b Init.t ->
  vocab:int ->
  dim:int ->
  (float, 'b) Nx.dtype ->
  (float, 'b) Nx.t t
(** [make ~vocab ~dim dtype] is a fresh table of [vocab] rows of [dim] features.
    [init] initializes the table and is applied with [~fan_in:dim] and
    [~fan_out:dim], so variance-scaling initializers target a variance
    proportional to [1 / dim]. It defaults to [Init.normal ~stddev:1.0].

    Random initializers draw from the implicit RNG scope (see {!Nx.Rng}).

    Raises [Invalid_argument] if [vocab] or [dim] is not positive. *)

val init : vocab:int -> dim:int -> Nx.float32_t t
(** [init ~vocab ~dim] is [make ~vocab ~dim Nx.float32]: rows drawn from
    [N(0, 1)]. *)

(** {1:applying Applying} *)

val apply : (float, 'b) Nx.t t -> (int32, Nx.int32_elt) Nx.t -> (float, 'b) Nx.t
(** [apply p ids] gathers the table row of each id: the result has [ids]'s shape
    with a trailing axis of size [dim] appended, and its [(i, ..., :)] slice is
    row [ids.(i, ...)] of [p.table]. A scalar id yields a single row of shape
    [[| dim |]].

    The gather is differentiable through Rune: the table's gradient accumulates
    each row's cotangent as many times as its id occurs.

    Raises [Invalid_argument] if an id is negative or not below [vocab]. *)

(** {1:structure Structure} *)

val walk : ('a, 'b) Nx.Ptree.Walk.cursor -> 'a t -> 'b t
(** [walk c p] walks [p]'s parameters, [table], at that path: the layer's
    {!Nx.Ptree.S} instance. [Nx.Ptree.instantiate (module Embedding)] is the
    layer at one dtype, and [Nx.Ptree.cast (module Embedding) dtype p] converts
    its precision. *)
