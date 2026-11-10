(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Preparing a tensor graph for ranges.

    A function's tensor graph states what to compute: values, movements of
    values, copies between devices and stores into storage. Before ranges index
    it ({!Indexing.run_rangeify}), it is brought to a normal form: each output
    is stored into its storage, inline calls are inlined, copies and
    materialisations store into new storage, and the operations that the later
    passes cannot express are rewritten into ones they can. *)

val pm_mops : (unit, Ops.t) Ops.Pattern_matcher.t
(** [pm_mops] moves movements towards the storage they view:

    - an {!Op.Index} of a movement indexes the movement's source with the
      indices the movement maps to it ({!Indexing.apply_movement_op}). An index
      with fewer indices than a reshape has axes indexes its source when the
      reshape leaves the trailing axes alone;
    - a store through a selection of an index and of [0], as a padded view
      makes, stores through the index made valid ({!Shape.valid}) only where
      the selection's conditions hold: a selection of such a selection and of
      [`Invalid] gates on both conditions;
    - a movement or an index of storage ordered after effects ({!Op.After})
      is the movement or index of its source ordered after them, and an index
      of a value ordered after effects is the index of the value ordered after
      them;
    - an {!Op.End} of a movement is the end of the movement's source. *)

val expand_bitcast : Ops.t -> Ops.t option
(** [expand_bitcast bc] is the {!Op.Bitcast} [bc] of a value to a type of
    another size as shifts and casts of unsigned integers, lane by lane along
    its last axis: to a wider type, each group of narrow lanes is shifted into
    place and summed; to a narrower one, each wide lane is shifted and cast
    into each of its narrow lanes, the least significant first. [None] where
    the sizes are equal or the value is on a disk. *)

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
     {- movements move towards storage ({!pm_mops}), except that a gather, an
        index by a value with axes ({!Indexing.is_gather}), is left whole; each
        call to a function without its own compilation ({!Ops.is_inline_call})
        is replaced by its body, its parameters bound to its arguments viewed as
        flat storage and its local storage renamed; and the movements and
        bitcasts of a copy from a disk move to the copy's result;
     }
     {- the graph's effects are sequenced ({!sequence});
     }
     {- from the leaves up, with {!Shape.mop_cleanup} and {!pm_mops}, gathers
        left whole:
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
        - a call compiled on its own, a kernel given as code or a precompiled
          call, takes as an argument that is a run of storage's elements, or
          of a state of it, from the first, of the storage's type, the
          storage or the state ({!contiguous_view}), which the call writes
          through; storage or a state of it that the call only reads, outside
          loops, that starts elsewhere than its parameter assumes
          ({!Shape.storage_phase}), and that the call does not write through
          another argument ({!Shape.reads_storage_written}), is stored into new
          storage, where new storage starts as the parameter assumes;
        - the sink's sources lose their movements;
        - a copy to the device the value is on is the value; a store of a copy
          into storage on the copy's device, or into a run of its elements
          that shrinks and reshapes view ({!contiguous_view}), stores the
          value, the store being the copy; any other copy stores the value, padded to its greatest
          shape, into new storage on the copy's device;
        - a materialisation ({!Op.Stage}) of storage or of a copy is its source;
          any other in global memory stores its source into new storage on
          its device, each slice at the coordinates of the stage's ranges;
        - a store of a reshape into a reshape of the same shape stores the
          sources; a store across devices first materialises its value on its
          own device; a store of a storage's contents into itself is
          dropped, while a store of the value the store before it stored
          stays, a state of its own; a store into a bitcast of storage stores
          the value bitcast to the storage's type; any other store through
          movements and bitcasts whose elements are a run of whole elements
          of storage or of a state of it ({!contiguous_view}) stores the
          value's bits into that run;
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
    Raises [Invalid_argument] if a store's destination, inline calls inlined,
    is a broadcast: an {!Op.Expand} that stretches an axis of one element to
    more, a symbolic size included, or a movement, index, bitcast,
    {!Op.After}, {!Op.Detach} or {!Op.Contiguous_backward} of one. Several of
    its elements are one cell, which the store would write once per element,
    in no order; a view of the broadcast whose elements share no cell, such
    as one row of it, is refused too. An expand that only adds axes of one,
    or empties an axis, stretches none. *)

val contiguous_view : Ops.t -> (Ops.t * Ops.sint) option
(** [contiguous_view u] is [Some (b, offset)] if the elements of [u], in
    row-major order, are the elements of [b] from [offset] on, in order, where
    [u] is [b] viewed through movements and bitcasts that neither reorder nor
    skip elements. [b] is the node the view reads, simplified: storage, storage
    ordered after effects, a constant, a computed value, or a bitcast whose view
    does not start and end on whole elements of its source at a constant
    offset. [offset] counts elements of [b]'s type: a constant, or an
    expression of constants and bound variables, through a bitcast only where
    the view starts and ends on whole elements of its source at every value
    of the variables. A scalar parameter, in a call's body, counts as a bound
    variable at its least value. It is [None]
    otherwise: when the movements reorder, skip, repeat or pad elements, when
    the offset reads a range or an unbound variable, when [u] is empty, and when
    its size is symbolic.

    A run is found when [u] is shrinks and permutes of a reshape of [b]. Other
    runs are found where the simplified index of [u]'s elements is the range
    plus a constant, and some are not: 27 elements viewed as [[3; 3; 3]], shrunk
    to the first two along axis 0, reshaped to [[18]] and shrunk to elements 6
    to 9, whose index keeps divisions the simplifier does not recombine; flips
    that cancel across a reshape; and a read within one copy of an expanded
    value. A caller then treats [u] as a value of its own. *)

val sequence : Ops.t -> Ops.t
(** [sequence sink] is [sink] with each storage's states one chain, which
    holds what {!Ops.after} says the graph means. Each effect writes the
    states of its destinations, or, where its body shows none, of the
    {!Op.After}s that list it; it runs after the state of each {!Op.After}
    listing it whose storage it writes, and the effects listed before its first
    place there. A write is over the state {!Ops.after} says. Each state of the chain lists the one
    effect, or the stores of one disjoint write ({!Ops.after_disjoint}), that
    makes it, over the state before; every read names the state of the chain
    it reads, a join or a value ordered after effects the value under it; a
    destination, and an argument a call writes, names the state its write is
    over; a store through a bitcast of one element size stores the value's
    bits into the state. [sink] lists each chain's last state, the effects
    that write no storage, and its items that are no effect and no state.
    Effects whose order no state decides keep the order a walk from [sink]
    reaches them: two writes over one state are refused, so that order writes
    nothing another order would not.

    Raises [Invalid_argument] on two writes over one state, neither reading
    what the other left nor ordered after it, naming both, and if an effect is
    listed over a state that needs it. *)
