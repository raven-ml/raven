(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** NVIDIA GPUs' formats, as values.

    What compiled code and the driver of an NVIDIA GPU share: the words a
    channel runs, the entries of its ring, the descriptors its launches read,
    the cubins they run, and the record that describes the GPU. Nothing here
    acts on a GPU. Each value describes bytes, and its caller writes them.

    Three ideas carry the library:
    - A {e packet} ({!Packet.t}) is a channel's 32-bit words around values of
      the caller's type ['v]: each word is a constant, or a {e term} that
      computes on a value. The caller interprets it: {!Packet.encode} with
      integers, {!Packet.template} with some values left for later, a compiler
      as its own nodes. {!Method} and {!Gpfifo} give each operation's words as a
      packet.
    - A {e structure} ({!Structure.t}) is bytes in memory around the caller's
      values: fields the layout knows, and {e holes} that terms fill. A launch
      descriptor ({!Qmd}) is built as one.
    - A {e GPU} ({!Gpu.t}) is what the formats depend on: its classes, its
      geometry and the windows its driver chose, declared by the driver under
      {!Gpu.key}. A cubin's kernel ({!Cubin}) set up for launch on a GPU is a
      {!Launch.t}, from which its descriptor is built; {!Local_memory} sizes the
      local memory its threads take.

    {v
    cubin bytes --Cubin.of_string--> Cubin.t --kernel--> Cubin.kernel
                                       |                      |
                              size, elf, patches       Launch.make Gpu.t
                                       |                      |
                          a loader writes the image      Launch.t
                                                              |
                               Qmd.make, set_*, release, chain, structure
                                                              |
    Method.*, Gpfifo.entry --> 'v Packet.t              'v Structure.t
                                      \                      /
                         encode, template, a compiler's own nodes
    v}

    The modules below are in reading order.

    {1:conventions Conventions}

    Addresses are the GPU's virtual addresses. An operand of the caller's type
    ['v] cannot be checked here: its alignment and range, which each function
    states, are the caller's to keep.

    Misuse raises [Invalid_argument]: an integer outside the range its function
    states, a class or version no launch descriptor holds. A cubin that is not
    one, or a kernel that takes more than a launch can, is [Error msg], [msg]
    saying why.

    The library has no state: any domain may use any of its values.

    {1:references References}

    - NVIDIA's class headers in
      {{:https://github.com/NVIDIA/open-gpu-kernel-modules}open-gpu-kernel-modules}
      570.144: [clc56f.h] (a channel's host methods, ring entries and pushbuffer
      headers), [clc7c0.h] (compute) and [clc7b5.h] (copy).
    - NVIDIA's launch descriptor headers in
      {{:https://github.com/NVIDIA/open-gpu-doc}open-gpu-doc}: [clc7c0qmd.h]
      (version 3) and [clcec0qmd.h] (version 5). *)

(** {1:words Channel words} *)

module Packet = Packet
(** Packets: the 32-bit words a channel runs, around the caller's values. *)

module Method = Method
(** Channel methods. *)

module Gpfifo = Gpfifo
(** Ring entries. *)

(** {1:gpus GPUs} *)

module Gpu = Gpu
(** NVIDIA GPUs, as their formats depend on them. *)

module Local_memory = Local_memory
(** Kernels' local memory. *)

(** {1:launches Launches} *)

module Cubin = Cubin
(** Cubins: the ELF objects NVIDIA's compilers make for a GPU architecture. *)

module Launch = Launch
(** Kernels set up for launch on a GPU. *)

module Structure = Structure
(** Structures in memory around values. *)

module Qmd = Qmd
(** Launch descriptors. *)
