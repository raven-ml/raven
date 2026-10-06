(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** From tensor graphs to kernel graphs.

    A kernel graph is what runs: storage, and calls to kernels that read and
    write it. Each kernel is a {!Op.Sink} carrying {!Ops.kernel_info}, whose
    loops are ranges ({!Op.Range}) and whose storage is its parameters. The
    graph holds no movements: a value is its storage. *)

val get_kernel_graph : Ops.t -> Ops.t
(** [get_kernel_graph sink] is the kernel graph of the prepared tensor graph
    [sink] ({!Prepare.prepare_rangeify}). In turn:

    {ol
     {- ranges index the graph ({!Indexing.run_rangeify}), which prints them
        when the setting {!Setting.debug_rangeify} is on;
     }
     {- the graph is simplified ({!Shape.symbolic},
        {!Simplify.pm_reduce_simplify}), and storage that need not exist is
        removed:
        - a stage ({!Op.Stage}) that a later pass may inline drops the axes its
          value does not vary along, and a stage of a constant is the constant;
        - an index of a stage by the stage's own ranges is the stage's value;
        - an index of a stage that a later pass may inline back is the staged
          value, recomputed at the index, unless it reads more than three
          storages or a reduction in it reads one;
        - an index of storage whose every write is invalid is invalid;
        - a store of a value into itself does nothing;
     }
     {- when a kernel may access at most [n] buffers (the setting
        {!Setting.max_kernel_buffers}, if not [0]), an operation that reads [n]
        buffers or more stores its elementwise sources first;
     }
     {- each stage becomes a store into new storage ({!Op.Alloc}) of its
        committed type ({!Ops.commit_dtype}), closed by an {!Op.End} over its
        ranges and read after it; a stage of storage that effects write ends
        those writes instead. Invalid writes are dropped;
     }
     {- each store or end whose ranges are all closed, but for device ranges,
        becomes a call to a kernel of its own: its storage becomes parameters,
        numbered from [0] in the order the kernel reaches them, its ranges are
        renumbered from [0], and the call's arguments are that storage, after
        the kernels that write it. An end of a call over {!Ops.Axis_type.Loop}
        ranges is a loop that runs the call once per trip, and no kernel;
     }
     {- the calls' arguments and the storage lose their indices and their views,
        but for the views a loop's call reads that move with the loop's ranges,
        and the loops' ranges lose the tags of kernel ranges.
     }
    }

    With the setting {!Setting.spec} at [1] or more, the result is checked
    against {!Spec.kernel_graph}.

    Raises [Invalid_argument] if a kernel reads one storage in two different
    states, which is a cycle, if a stage has no elements, which a prepared graph
    does not have, or if the check fails. *)
