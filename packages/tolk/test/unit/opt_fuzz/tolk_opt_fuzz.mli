(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Differential correctness of the kernel optimisation actions.

    Beam search ranks candidates by measured runtime and never inspects their
    results, so an action that changes what a kernel computes is invisible to
    it. {!check} runs a workload's kernels under every applicable sequence of
    actions and compares against the unoptimised kernel.

    Which actions survive depends on the renderer — a target without local
    memory drops every LOCAL, GROUP and tensor-core action — so the device is
    a parameter. *)

type workload
(** A named tensor graph exercising one kernel shape. *)

val workloads : workload list
(** Element-wise, reduce, matmul, matmul-with-reduced-output, one recurrent
    step, and a two-store kernel whose column axis is an output axis for one
    store and a reduce axis for the other. *)

val name : workload -> string
(** [name w] identifies [w] in test output. *)

type result = {
  sequences : int;  (** Action sequences that compiled, ran, and were compared. *)
  miscompiled : (string * string) list;
      (** Sequences that ran but returned different values, as
          [(sequence, detail)]. Each is a defect: beam search compares only
          runtimes, so it can select one of these silently. *)
  rejected : (string * string) list;
      (** Sequences that applied but then failed to compile or run. Beam
          search discards these, so they are not correctness defects — but a
          legal optimisation that cannot be compiled is a capability gap
          worth knowing about. *)
}

val check : Tolk.Device.t -> workload -> result
(** [check dev w] compiles and runs each of [w]'s kernels on [dev] under the
    unoptimised schedule and under every applicable action sequence, on
    byte-identical inputs, comparing every buffer afterwards so that an
    optimisation writing outside its output is caught too.

    Sequence length defaults to 2 and is set by [OPT_FUZZ_DEPTH]; cost grows
    exponentially in it. Float comparison is relative to [1e-4], which admits
    the reassociation that GROUP and UNROLL introduce into reductions. *)
