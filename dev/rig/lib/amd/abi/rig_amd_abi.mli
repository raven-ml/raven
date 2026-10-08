(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** AMD GPUs' formats, as values.

    What compiled code and the driver of an AMD GPU share: the packets its
    queues read, the registers and scratch memory those packets set up, the code
    objects they run, the thread traces they record, and the record that
    describes the device. Nothing here acts on a GPU. Each value describes
    bytes, and its caller writes them.

    Three ideas carry the library:
    - A {e packet} ({!Packet.t}) is a queue's 32-bit words around values of the
      caller's type ['v]: each word is a constant, or a {e term} that computes
      on a value. The caller interprets it: {!Rig_packet.encode} with integers,
      {!Rig_packet.template} with some values left for later, a compiler as its
      own nodes. {!Pm4}, {!Aql}, {!Sdma} and {!Thread_trace} give each command's
      words as a packet.
    - A {e GPU} ({!Gpu.t}) is what the formats depend on: the versions of its
      blocks and the number of its parts. An encoder whose words depend on it
      takes it, and so do {!Register}, which gives its GC's registers, and
      {!Scratch}, which sizes its kernels' private memory.
    - A {e code object} ({!Code_object.t}) is a program the GPU runs: an image a
      loader writes once into its destination, and the kernels it holds, each as
      its kernel descriptor describes it.

    The driver declares what compiled code needs from a device, its GPU among
    it, as a {!Capability.t} under {!Capability.key}.

    {v
    code object bytes --Code_object.of_string--> Code_object.t --kernel--> kernel
                                                       |                     |
                                               size, elf, patches  Pm4.dispatch, Aql.dispatch
                                                       |                     |
                                           a loader writes the image         v
    Gpu.t --> Pm4.*, Aql.*, Sdma.*, Thread_trace.* -------------------> 'v Packet.t
          --> Register.*, Scratch.*: values the packets carry                |
                                            encode, template, a compiler's own nodes
    v}

    The modules below are in reading order.

    {1:conventions Conventions}

    An operand of the caller's type ['v] cannot be checked here: its alignment
    and range, which each function states, are the caller's to keep.

    Misuse raises [Invalid_argument]: an integer outside the range its function
    states, a register or packet the GPU's GC does not have. A code object that
    is not one for AMD GPUs is [Error msg], [msg] saying why.

    The library has no state: any domain may use any of its values.

    {1:references References}

    - The Linux kernel's amdgpu driver (ROCK-Kernel-Driver): the PM4 packet
      headers [soc15d.h], [nvd.h] and [kfd_pm4_headers_ai.h], the SDMA packet
      headers, and the GC register headers [gc_*_offset.h] and [gc_*_sh_mask.h].
    - GPUOpen's PAL: the PM4 packet layouts of the compute engine's firmware,
      and the register spaces PM4 packets set ([gfx9_plus_merged_enum.h]).
    - ROCm's runtime (ROCR-Runtime): [hsa.h] (AQL packets), [registers.h] and
      [amd_aql_queue.cpp] (the scratch buffer descriptor and the vendor packet
      of PM4 commands).
    - LLVM's
      {{:https://llvm.org/docs/AMDGPUUsage.html}User Guide for AMDGPU Backend}:
      code objects, their relocation records and kernel descriptors.
    - Mesa's [src/amd/common/ac_sqtt.c]: the program that records a thread trace
      on a compute queue. *)

(** {1:words Queue words} *)

module Packet = Packet
(** Packets: the 32-bit words a GPU's queue reads, around the caller's values.
*)

(** {1:gpus GPUs} *)

module Gpu = Gpu
(** AMD GPUs, as their formats depend on them. *)

module Register = Register
(** The registers of a GPU's GC. *)

(** {1:programs Programs} *)

module Code_object = Code_object
(** Code objects: the programs AMD GPUs run. *)

module Scratch = Scratch
(** Scratch memory: the private memory of kernels' work-items. *)

(** {1:commands Commands} *)

module Pm4 = Pm4
(** PM4 packets: the commands of a compute queue. *)

module Aql = Aql
(** AQL packets: the commands of a queue that dispatches over every die. *)

module Sdma = Sdma
(** SDMA packets: the commands of a copy queue. *)

module Thread_trace = Thread_trace
(** Thread traces: what a GPU's shader engines record while they run waves. *)

module Counter = Counter
(** Performance counters: what a kernel's run counts, and where its values lie.
*)

(** {1:devices Devices} *)

module Capability = Capability
(** What compiled code needs from an AMD device. *)
