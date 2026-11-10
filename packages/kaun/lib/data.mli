(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Minibatch sequences over in-memory tensors, in order.

    A dataset is a tensor, or a pair of tensors, whose axis 0 indexes
    examples. {!batches} and {!batches2} cut it into a standard [Seq.t] of
    minibatches, consecutive slices of axis 0 that are views of the dataset.
    The training loop is ordinary [Seq] iteration, and stdlib combinators
    ([Seq.map], [Seq.take], [Seq.fold_left], ...) compose any further
    transformation. This module defines no iterator type of its own, and a
    traversal draws nothing.

    An epoch in a shuffled order is the dataset taken in an order drawn from a
    key, then cut. This draws the order of each epoch from its own key, so the
    run is reproducible and epoch [e] can be drawn again on its own:
    {[
    let epoch key e (x, y) =
      let key = Nx.Rng.fold_in key e in
      let order = Nx.Rng.permutation ~key (Nx.dim 0 x) in
      let take t = Nx.take ~axis:0 order t in
      Data.batches2 ~batch_size:32 (take x, take y)
    ]}
    For a dataset too large to hold, {!Nx.Rng.permute} gives the example at
    each position of an epoch's order without the whole order. *)

val batches :
  ?drop_last:bool ->
  batch_size:int ->
  ('a, 'b, 'd) Nx.t ->
  ('a, 'b, 'd) Nx.t Seq.t
(** [batches ~batch_size x] is the sequence of minibatches of [x]: consecutive
    slices of [batch_size] examples along axis 0, in order, each a view of [x].
    Each batch has shape [batch_size] followed by [x]'s remaining axes. A
    dataset with no examples is the empty sequence.

    [drop_last] drops the final batch when fewer than [batch_size] examples
    remain; otherwise that final batch is smaller. Defaults to [false].

    Raises [Invalid_argument] if [batch_size <= 0] or [x] is a scalar. *)

val batches2 :
  ?drop_last:bool ->
  batch_size:int ->
  ('a, 'b, 'd) Nx.t * ('c, 'e, 'd) Nx.t ->
  (('a, 'b, 'd) Nx.t * ('c, 'e, 'd) Nx.t) Seq.t
(** [batches2 ~batch_size (x, y)] is {!batches} of a dataset of paired
    examples, inputs [x] and targets [y] with equal axis-0 extents: each batch
    pairs [x]'s slice with [y]'s.

    Raises [Invalid_argument] if [batch_size <= 0], if [x] or [y] is a scalar,
    or if [x] and [y] disagree on the number of examples. *)
