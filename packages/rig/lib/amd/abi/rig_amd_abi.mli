(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The bytes an AMD GPU shares with its driver and with compiled code, as
    values.

    This library describes what an AMD GPU reads and writes: the packets its
    queues read, the registers and scratch memory those packets set up, the
    code objects they run, and the counters and thread traces a run records.
    Nothing here acts on a GPU. Each value describes bytes, and its caller
    writes them where the GPU reads them.

    A {e packet} ({!Packet.t}) is a list of a queue's 32-bit words, some
    fixed by the layout and the others values of the caller's type ['v],
    which the caller encodes, templates or compiles ({!Packet}). An
    encoder whose words depend on the GPU takes a {!Gpu.t}, the
    versions of the GPU's blocks and the number of its parts.

    A {e code object} ({!Code_object.t}) is a program the GPU runs. A loader
    writes its image into the GPU's memory, and a dispatch ({!Pm4.dispatch},
    {!Aql.dispatch}) launches one of its kernels from there.

    A device's driver declares what compiled code needs to encode work for
    the device as a {!Capability.t} ({!Capability.declare}): its GPU, its
    clock, the packets its compute queue reads and its trace buffers.
    Compiled code finds it under {!Capability.key}.

    {v
    code object bytes
      --Code_object.of_string--> Code_object.t --kernel--> kernel
                                      |                      |
                            size, image, patches  Pm4.dispatch, Aql.dispatch
                                      |                      |
                            a loader writes them             v
    Gpu.t --> Pm4, Aql, Sdma, Thread_trace ------------> 'v Packet.t
          --> Register, Scratch: values the packets carry      |
                                   encode, template, a compiler's own nodes
    v}

    A driver that knows its addresses encodes a launch of the kernel
    ["add"] of a code object [co] loaded at [base]:
    {[
    let k = Option.get (Code_object.kernel co "add") in
    let p =
      Aql.dispatch k ~descriptor:(base + k.descriptor) ~args
        ~threads:(64, 1, 1) ~grid:(1024, 1, 1)
    in
    Packet.encode Int64.of_int p
    ]}
    The result is one AQL packet, 64 bytes, which the driver places in its
    queue's ring as {!Aql} says.

    The modules below are in reading order.

    {1:conventions Conventions}

    This library cannot check an operand of the caller's type ['v]. Each
    function states the alignment and range such an operand must have, and
    the caller must keep them.

    Misuse raises [Invalid_argument], such as an integer outside the range
    its function states, or a register or packet the GPU's GC does not have.
    A code object that is not one for AMD GPUs is [Error msg], [msg] saying
    why.

    The library has no state, and any domain may use any of its values.

    {1:references References}

    {ul
    {- HSA Foundation. {e HSA Platform System Architecture Specification}:
       the Architected Queuing Language (AQL) packets.}
    {- LLVM's
       {{:https://llvm.org/docs/AMDGPUUsage.html}User Guide for AMDGPU
       Backend}: code objects, their relocation records, their metadata and
       kernel descriptors.}} *)

(** {1:words Queue words} *)

module Packet = Packet
(** Packets, the 32-bit words a GPU's queue reads. *)

(** {1:gpus GPUs} *)

module Gpu = Gpu
(** AMD GPUs, as their formats depend on them. *)

module Register = Register
(** The registers of a GPU's GC. *)

(** {1:programs Programs} *)

module Code_object = Code_object
(** Code objects, the programs AMD GPUs run. *)

module Scratch = Scratch
(** Scratch memory, which holds the private memory of kernels'
    work-items. *)

(** {1:commands Commands} *)

module Pm4 = Pm4
(** The PM4 packets a compute queue reads. *)

module Aql = Aql
(** The AQL packets of a queue that dispatches over every die. *)

module Sdma = Sdma
(** The SDMA packets a copy queue reads. *)

module Thread_trace = Thread_trace
(** Thread traces, which a GPU's shader engines record while they run
    waves. *)

module Counter = Counter
(** Performance counters, which count events of a kernel's run. *)

(** {1:devices Devices} *)

module Capability = Capability
(** What compiled code needs from an AMD device. *)

(** {1:boot The boot}

    What a driver that boots the GPU itself reads from the GPU and its
    firmware, and writes for the GPU's blocks. *)

module Discovery = Discovery
(** The discovery table: the GPU's blocks, their versions and where their
    registers lie. *)

module Firmware = Firmware
(** The firmware a GPU boots with, cut into the pieces its security
    processor loads. *)

module Psp = Psp
(** The security processor's commands. *)

module Smu = Smu
(** The power manager's messages. *)

module Gmc = Gmc
(** The memory controller's window and page-table entries. *)

module Gfx = Gfx
(** The compute queues' descriptors and the shader array index. *)

module Ih = Ih
(** The interrupt rings' entries. *)
