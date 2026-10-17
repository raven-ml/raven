(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** AMD GPUs, as their formats depend on them.

    A GPU is built of blocks, each with a version of its own. Its GC runs
    the compute queues and sets the layout of their registers and PM4
    packets. Its SDMA engines run the copy queues. Its instruction set,
    which code objects target, has a version too ({!processor}). A large GPU
    has several dies, each a GC of its own, and every die has the same
    number of shader engines and compute units. A driver reads these facts
    when it opens the GPU, from the GPU's discovery table or from the kernel
    driver's topology. *)

type version = int * int * int
(** The type for versions, as [(major, minor, stepping)], such as
    [(9, 4, 3)] for the GC of an MI300. *)

type t = {
  target : version;
      (** The instruction set its compiler targets, such as [(12, 0, 1)] for
          gfx1201. *)
  gc : version;
      (** Its GC, the block that runs compute queues, whose registers and
          PM4 packets follow this version. *)
  sdma : version;  (** Its SDMA engines, which run copy queues. *)
  xccs : int;  (** Its dies, each a GC of its own; [1] on most GPUs. *)
  shader_engines : int;  (** The shader engines of one die. *)
  compute_units : int;
      (** The compute units of one die that run work. Those disabled at
          manufacture or by the firmware are not counted. *)
  array_units : int;
      (** The compute units each shader array is built with, those that run
          no work included: two to a work-group processor from GC 10. *)
  scratch_slots : int;
      (** The waves a compute unit runs concurrently with scratch
          memory. *)
}
(** The type for GPUs. Every count is positive, which this library's
    functions assume and do not check. A device checks it when it is
    made. *)

val processor : t -> string
(** [processor g] is the processor [g] runs code objects of, as LLVM names
    it: ["gfx"] followed by [g.target]'s major in decimal and its minor and
    stepping in hexadecimal, such as ["gfx1201"] or ["gfx90a"]. *)
