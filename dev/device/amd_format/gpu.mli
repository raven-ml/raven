(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** AMD GPUs, as their formats depend on them.

    A GPU's packets, registers and scratch memory follow the versions of its
    blocks and the number of its parts. A driver reads them from the GPU at
    open, from its discovery table or the kernel driver's topology. *)

type version = int * int * int
(** The type for versions, as [(major, minor, stepping)]: [(9, 4, 3)] for the GC
    of an MI300. *)

type t = {
  target : version;
      (** The instruction set its compiler targets: [(12, 0, 1)] for gfx1201. *)
  gc : version;
      (** Its GC, the block that runs compute queues, whose registers and PM4
          packets it takes. *)
  sdma : version;  (** Its SDMA engines, which run copy queues. *)
  xccs : int;  (** Its dies, each a GC of its own; [1] on most GPUs. *)
  shader_engines : int;  (** The shader engines of one die. *)
  compute_units : int;  (** The compute units of one die. *)
  scratch_slots : int;
      (** The waves a compute unit runs at once with scratch memory. *)
}
(** The type for GPUs. Every count is positive. *)

val processor : t -> string
(** [processor g] is the processor [g] runs code objects of, as LLVM names it:
    ["gfx"] followed by [g.target]'s major in decimal and its minor and stepping
    in hexadecimal, such as ["gfx1201"] or ["gfx90a"]. *)
