(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** From tensor graphs to kernel graphs.

    A kernel graph is what runs: storage, and calls to kernels that read and
    write it. Each kernel is a {!Op.Sink} carrying {!Ops.type-kernel_info},
    whose loops are ranges ({!Op.Range}) and whose storage is its parameters.
    The graph holds no movements: a value is its storage. *)

val get_kernel_graph : Ops.t -> Ops.t
(** [get_kernel_graph sink] is the kernel graph of the prepared tensor graph
    [sink] ({!Prepare.prepare_rangeify}). In turn:

    {ol
     {- ranges index the graph ({!Indexing.run_rangeify});
     }
     {- the graph is simplified ({!Shape.val-symbolic},
        {!Simplify.pm_reduce_simplify}), and storage that need not exist is
        removed:
        - a stage ({!Op.Stage}) that a later pass may inline drops the axes its
          value does not vary along, and a stage of a constant is the constant;
        - an index of a stage by the stage's own ranges is the stage's value;
        - an index of a stage that a later pass may inline back is the staged
          value, recomputed at the index, unless it reads more than three
          storages, a reduction in it reads one, or its copies beyond the first
          exceed 360 nodes, so that copies of copies cannot grow the kernels
          faster than the program, nor recomputation the work of a reduction
          past what storing the value costs. Its copies beyond the first are the
          nodes of its value inside its ranges, which each index copies, times
          the copies its reads make less one. A read makes one copy at each
          distinct index, and in a reduction one per trip of the reduction's
          ranges its index runs in, but at most its trips per element of the
          stage: a running sum's windows of [w] compute each element [w] times,
          and a sum computes each element once. A read whose index runs in no
          reduction range makes one copy, as a broadcast through an expand, a
          reshape or padding does. Trips in padding count, since inlined there
          the value's loads are skipped and its arithmetic is not. A sum over
          one range that selects one trip, where an index equals the range
          ({!Simplify.load_collapse}), is computed at that index alone, once for
          each point of the ranges the index runs in, so its range counts as
          the ranges its selecting conditions read besides it, as a read at
          that index would: a factor over another range inside the sum
          broadcasts the selected value along it; reads of stages a later
          pass may inline whose values read no storage, as a range's
          positions do, are seen as those values there, and count such a
          sum's own trips, so that they are inlined as that test assumes. An
          axis the value does not vary along counts neither elements nor trips,
          since inlining at its index leaves the value as it is, and a symbolic
          size counts at its largest. The reads are those of the graph this step
          starts from: a read the step itself makes, such as inlining a reader
          of the stage at several indices, is not counted. They are counted at
          each range of the stage, taking the least count, and stages over one
          device range share it: two such stages read once each, with no range
          of their own, count as read twice;
        - an index of storage whose every write is invalid is invalid;
        - a store of a value into itself does nothing;
     }
     {- a value of no range that [k] kernels compute, each kernel being a
        stage or a store or end whose ranges are all closed, is staged once,
        in one element, and read by each, when [(k - 1) n > 360], [n] the
        computed nodes of its value (elementwise operations and reductions on
        a device) down to storage. Values are weighed from the kernels down,
        each where some consumer computes it in fewer kernels; a weak value
        is never staged, and below it, as below a load or a value with
        ranges, the value beneath is weighed in its place. A
        value whose nodes, counted down to the values staged below it, are
        within the bound is computed in each kernel again, and those below it
        count its kernels as theirs. A value of [n] nodes that 3 kernels
        compute is staged past 180 nodes. With the bound on stages, a program
        of [N] nodes makes kernels of at most [N] nodes plus 360 for each
        stage inlined and for each value of no range left in several kernels;
        values with ranges that several kernels share through a range are not
        bounded;
     }
     {- when a kernel may access at most [n] buffers (the setting
        {!Setting.max_kernel_buffers}, if not [0]), an operation that reads [n]
        buffers or more stores its elementwise sources first; and a store
        whose value reads its destination's storage at another index than the
        one it writes, other than through a stage or an {!Op.After}, stores
        its value first, since the store would overwrite elements the value
        reads;
     }
     {- each stage becomes a store into new storage ({!Op.Alloc}) of its
        committed type ({!Ops.commit_dtype}), closed by an {!Op.End} over its
        ranges and read after it; a stage of storage that effects write ends
        those writes instead. Invalid writes are dropped;
     }
     {- each store or end whose ranges are all closed becomes a call to a
        kernel of its own: each state of storage it reads or writes becomes a
        parameter that names no device, the kernel running on its call's
        arguments' devices, numbered from [0] in the order the kernel reaches
        them, its
        ranges are renumbered from [0], and the call's arguments are those
        states, a state read through a join ({!Ops.is_join}) being the state
        under it, after what the join waits for. An end of a call over {!Ops.Axis_type.Loop}
        ranges is a loop that runs the call once per trip, and no kernel;
     }
     {- the calls' arguments and the storage lose their indices and their views,
        but for the views a loop's call reads that move with the loop's
        ranges.
     }
    }

    With the setting {!Setting.spec} at [1] or more, the result is checked
    against {!Spec.kernel_graph}.

    Raises [Invalid_argument] if a stage has no elements, which a prepared
    graph does not have, if a kernel's value is ordered after effects with no
    storage to hold it, as a constant after a store, or if the check
    fails. *)
