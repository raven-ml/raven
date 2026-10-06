(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Preparing a tensor graph for ranges.

    A function's tensor graph states what to compute: values, movements of
    values, copies between devices and stores into storage. Before ranges index
    it ({!Indexing.run_rangeify}), it is brought to a normal form: each output
    is stored into its storage, sharded values compute on shards
    ({!Multi.multi_pm}), inline calls are inlined, copies and materialisations
    store into new storage, and the operations that the later passes cannot
    express are rewritten into ones they can. *)

val pm_mops : (unit, Ops.t) Ops.Pattern_matcher.t
(** [pm_mops] moves movements towards the storage they view:

    - an {!Op.Index} of a movement indexes the movement's source with the
      indices the movement maps to it ({!Indexing.apply_movement_op}). An index
      with fewer indices than a reshape has axes indexes its source when the
      reshape leaves the trailing axes alone;
    - a movement or an index ordered after effects ({!Op.After}) is the movement
      or index of its source ordered after them;
    - an {!Op.End} of a movement is the end of the movement's source. *)

val prepare_rangeify : Ops.t -> Ops.t
(** [prepare_rangeify sink] is the tensor graph of the function [sink] in the
    form ranges index. In turn:

    {ol
     {- each output that stores a value into storage stores it in place: a value
        computed into new storage ({!Op.Alloc}) of the output's size is computed
        into the output's storage directly, and every other use of it reads the
        output. So is a value materialised by an {!Op.Stage}, or storage, when
        the output is the store itself rather than storage ordered after it;
     }
     {- sharded values compute on shards ({!Multi.multi_pm}); }
     {- movements move towards storage ({!pm_mops}), except that a gather, an
        index by a value with axes ({!Indexing.is_gather}), is left whole; each
        call to a function without its own compilation ({!Ops.is_inline_call})
        is replaced by its body, its parameters bound to its arguments viewed as
        flat storage and its local storage renamed; a value ordered after a
        call's stores into one of its outputs is that output's value
        ({!Ops.resolve_returned_after}); and the movements of a copy from a disk
        move to the copy's result;
     }
     {- from the leaves up, with {!Shape.mop_cleanup} and {!pm_mops}, gathers
        left whole:
        - an {!Op.Allreduce} is a call of its own
          ({!Allreduce.create_allreduce_function});
        - a large reduction over few outputs is split in two, when the setting
          {!Setting.split_reduceop} is on: the reduced axis of the input is
          split by the largest divisor from [256] down to [8] that keeps the
          first reduction's output within [2]{^ n} elements, [n] the setting
          {!Setting.reduceop_split_size}, and the first reduction is
          materialised. It applies when the input's shape is known and the input
          has at least {!Setting.reduceop_split_threshold} times as many
          elements as the output, and only to an axis the input is not broadcast
          along;
        - {!Op.Detach} and {!Op.Contiguous_backward} are their source;
        - the sink's sources lose their movements and sharding;
        - a copy to the device the value is on is the value; a store of a copy
          into storage on the copy's device stores the value, the store being
          the copy; any other copy stores the value, padded to its greatest
          shape, into new storage on the copy's device;
        - a materialisation ({!Op.Stage}) of storage or of a copy is its source;
          any other stores its source into new storage on its device;
        - a store of a reshape into a reshape of the same shape stores the
          sources; a store across devices first materialises its value on its
          own device; a store whose value reads its destination through a
          permutation, a flip, a gather's gathered source or, when the
          destination is itself shrunk, a shrink materialises the value first;
          the second of two equal stores into the same storage is dropped, and
          so is a store of a storage's contents into itself; a store into a
          bitcast of storage stores the value bitcast to the storage's type;
        - a bitcast between types of different sizes is shifts and masks on
          unsigned integers, except on a disk;
        - a reduction of an empty axis to a non-empty result is the operation's
          identity ({!Ops.identity_element}), and any other value with an empty
          axis is [0];
        - the effects that a sink or an {!Op.After} lists lose their movements,
          and {!Op.Noop}s among them are dropped.
     }
    }

    Raises [Invalid_argument] if an inline call's argument does not fit its
    parameter: a storage parameter's size, a scalar's shape, or the data type.
*)

val contiguous_view : Ops.t -> (Ops.t * int) option
(** [contiguous_view u] is [Some (b, offset)] if the elements of [u], in
    row-major order, are the elements of [b] from [offset] on, in order, where
    [u] is [b] viewed through movements and bitcasts that neither reorder nor
    skip elements. [b] is storage, storage ordered after effects, or a bitcast
    whose view does not start and end on whole elements of its source, which [u]
    views through. [offset] counts elements of [b]'s type. It is [None]
    otherwise: when [u] is no view of storage, as a constant or a computed value
    is not, when it is empty, and when its size is symbolic. *)
