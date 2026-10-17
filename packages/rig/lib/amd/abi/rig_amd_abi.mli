(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The bytes an AMD GPU shares with its driver and with compiled code, as
    values.

    This library describes what an AMD GPU reads and writes: the code
    objects it runs, and the thread traces and counters a run records.
    Nothing here acts on a GPU. Each value describes bytes, and its caller
    writes them where the GPU reads them, or reads them where the GPU wrote
    them. A format whose bytes depend on the GPU takes a {!Gpu.t}, the
    versions of the GPU's blocks and the number of its parts.

    A {e code object} ({!Code_object.t}) is a program the GPU runs. A device
    loads its image into the GPU's memory and launches its kernels from
    there.

    A device's driver declares what compiled code needs of the device as a
    {!Capability.t} ({!declare}): its GPU, its clock, what its compute queue
    reads and its trace buffers. Compiled code finds it under {!key}.

    Compiled code checks that a device [d] runs a code object [co]:
    {[
    let runs d co =
      match Rig.capability d key with
      | Some c -> Code_object.runs_on co c.gpu
      | None -> false
    ]}

    The modules below are in reading order.

    {1:conventions Conventions}

    Misuse raises [Invalid_argument], such as an integer outside the range
    its function states, or a register the GPU's GC does not have. A code
    object that is not one for AMD GPUs is [Error msg], [msg] saying why.

    The library has no state, and any domain may use any of its values.

    {1:references References}

    {ul
    {- LLVM's
       {{:https://llvm.org/docs/AMDGPUUsage.html}User Guide for AMDGPU
       Backend}: code objects, their relocation records, their metadata and
       kernel descriptors.}} *)

(** {1:gpus GPUs} *)

module Gpu = Gpu
(** AMD GPUs, as their formats depend on them. *)

(** {1:programs Programs} *)

module Code_object = Code_object
(** Code objects, the programs AMD GPUs run. *)

(** {1:records Records of runs} *)

module Thread_trace = Thread_trace
(** Thread traces, which a GPU's shader engines record while they run
    waves. *)

module Counter = Counter
(** Performance counters, which count events of a kernel's run. *)

(** {1:devices Devices} *)

module Capability = Capability
(** What compiled code needs from an AMD device. *)

val key : Capability.t Type.Id.t
(** [key] is the key of an AMD device's record. Compiled code finds the record
    of a device [d] with [Rig.capability d key], which is [None] if [d] is
    not an AMD device. *)

val declare : Capability.t -> Rig_edge.capability
(** [declare c] is [c] under {!key}, as the facts of an AMD device carry it
    ({!Rig_edge.type-facts}). *)

(**/**)

(* The packets a GPU's queues read, and the registers and scratch memory
   they set up. rig's AMD driver encodes its queues' work with them; tests
   check them against AMD's headers without a GPU.

   A packet ([Packet.t]) is a list of a queue's 32-bit words, some fixed by
   the layout and the others values of the caller's type ['v], which the
   caller encodes or templates. This library cannot check such an operand:
   each function states the alignment and range it must have, and the
   caller must keep them. The HSA Foundation's HSA Platform System
   Architecture Specification defines the Architected Queuing Language
   (AQL) packets. *)

module Packet = Packet
(** Packets, the 32-bit words a GPU's queue reads. *)

module Register = Register
(** The registers of a GPU's GC. *)

module Scratch = Scratch
(** Scratch memory, which holds the private memory of kernels'
    work-items. *)

module Pm4 = Pm4
(** The PM4 packets a compute queue reads. *)

module Aql = Aql
(** The AQL packets of a queue that dispatches over every die. *)

module Sdma = Sdma
(** The SDMA packets a copy queue reads. *)
