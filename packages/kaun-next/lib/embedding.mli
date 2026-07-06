(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Embedding layers.

    An embedding is a lookup table mapping integer token ids to learned dense
    vectors: a plain record holding one [vocab × dim] parameter. Construct one
    with {!init} or {!make} and turn id tensors into vector tensors with
    {!apply}. Like the other layers, it composes into models through record
    nesting; {!map}, {!map2}, {!iter} and {!names} supply the {!Nx.Ptree.S} and
    checkpoint plumbing. *)

(** {1:types Types} *)

type 'b params = { table : (float, 'b) Nx.t }
(** The type for embedding parameters with float dtype layout ['b]. [table] has
    shape [[| vocab; dim |]]: one row per token id. *)

type t = Nx.float32_elt params
(** The type for single-precision embeddings, the common case. *)

(** {1:constructors Constructors} *)

val make :
  ?init:'b Init.t -> vocab:int -> dim:int -> (float, 'b) Nx.dtype -> 'b params
(** [make ~vocab ~dim dtype] is a fresh table of [vocab] rows of [dim] features.
    [init] initializes the table and is applied with [~fan_in:dim] and
    [~fan_out:dim], so variance-scaling initializers target a variance
    proportional to [1 / dim]. It defaults to [Init.normal ~stddev:1.0].

    Random initializers draw from the implicit RNG scope (see {!Nx.Rng}).

    Raises [Invalid_argument] if [vocab] or [dim] is not positive. *)

val init : vocab:int -> dim:int -> t
(** [init ~vocab ~dim] is [make ~vocab ~dim Nx.float32]: rows drawn from
    [N(0, 1)]. *)

(** {1:applying Applying} *)

val apply : 'b params -> (int32, Nx.int32_elt) Nx.t -> (float, 'b) Nx.t
(** [apply p ids] gathers the table row of each id: the result has [ids]'s shape
    with a trailing axis of size [dim] appended, and its [(i, ..., :)] slice is
    row [ids.(i, ...)] of [p.table]. A scalar id yields a single row of shape
    [[| dim |]].

    The gather is differentiable through Rune-next: the table's gradient
    accumulates each row's cotangent as many times as its id occurs.

    Raises [Failure] if an id is negative or not below [vocab]. *)

(** {1:traversals Traversals}

    Plain traversals over the single parameter leaf. They satisfy the
    {!Nx.Ptree.S} contract at any fixed ['b]. *)

val map : ('a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t) -> 'b params -> 'b params
(** [map f p] is [p] with [f] applied to the table. *)

val map2 :
  ('a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t -> ('a, 'c) Nx.t) ->
  'b params ->
  'b params ->
  'b params
(** [map2 f p q] combines the tables of [p] and [q] with [f]. *)

val iter : ('a 'c. ('a, 'c) Nx.t -> unit) -> 'b params -> unit
(** [iter f p] applies [f] to the table. *)

val names : 'b params -> string list
(** [names p] is [["table"]], the checkpoint name of the single parameter leaf.
    See {!Checkpoint.Named}. *)
