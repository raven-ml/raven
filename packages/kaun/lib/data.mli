(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Minibatch sequences over in-memory tensors.

    A dataset is a tensor — or a pair of tensors — whose axis 0 indexes
    examples. {!batches} and {!batches2} cut it into a standard [Seq.t] of
    minibatch tensors; the training loop is ordinary [Seq] iteration, and stdlib
    combinators ([Seq.map], [Seq.take], [Seq.fold_left], ...) compose any
    further transformation. This module defines no iterator type of its own.

    Traversing a sequence built with [~shuffle:true] draws a fresh permutation
    from the ambient RNG scope (see {!Nx.Rng}): iterating the same sequence once
    per epoch reshuffles every epoch, and running the loop under {!Nx.Rng.run}
    makes the whole schedule of permutations — hence the whole run —
    reproducible.

    {[
    Nx.Rng.run ~seed:42 @@ fun () ->
    let data = Data.batches2 ~shuffle:true ~batch_size:32 (x, y) in
    let state = ref (params, Vega.adam_init (module Model) params) in
    for _epoch = 1 to 10 do
      data |> Seq.iter (fun batch -> state := fst (step !state batch))
    done
    ]} *)

val batches :
  ?shuffle:bool ->
  ?drop_last:bool ->
  batch_size:int ->
  ('a, 'b) Nx.t ->
  ('a, 'b) Nx.t Seq.t
(** [batches ~batch_size x] is the sequence of minibatches of [x]: consecutive
    slices of [batch_size] examples along axis 0, in order. Each batch has shape
    [batch_size] followed by [x]'s remaining axes. Without shuffling, batches
    are views of [x], not copies. A dataset with no examples is the empty
    sequence.

    - [shuffle] visits the examples in a random order instead. The permutation
      is drawn from the ambient RNG scope at each traversal of the sequence, so
      every traversal reshuffles (see the module preamble); batches are then
      copies. Defaults to [false].
    - [drop_last] drops the final batch when fewer than [batch_size] examples
      remain; otherwise that final batch is smaller. Defaults to [false].

    Raises [Invalid_argument] if [batch_size <= 0] or [x] is a scalar. *)

val batches2 :
  ?shuffle:bool ->
  ?drop_last:bool ->
  batch_size:int ->
  ('a, 'b) Nx.t * ('c, 'd) Nx.t ->
  (('a, 'b) Nx.t * ('c, 'd) Nx.t) Seq.t
(** [batches2 ~batch_size (x, y)] is like {!batches} for a dataset of paired
    examples — inputs [x] and targets [y] with equal axis-0 sizes. Batches pair
    [x]'s slice with [y]'s, and shuffling permutes both with the same
    permutation, keeping every example aligned with its target.

    Raises [Invalid_argument] if [batch_size <= 0], if [x] or [y] is a scalar,
    or if [x] and [y] disagree on the number of examples. *)
