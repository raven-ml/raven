(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** NVIDIA GPUs' formats, as values.

    What compiled code and the driver of an NVIDIA GPU share: the words a
    channel runs, the entries of its ring, the descriptors its launches read,
    the cubins they run, and the record that describes the GPU. Nothing here
    acts on a GPU. Each value describes bytes, and its caller writes them.

    The library describes bytes with two kinds of value:
    - A {e template} ({!Template.t}) is bytes around the caller's values: each
      value is an integer known now, written in place, or a {e slot}, a value
      the caller numbers and fills later through a {e patch}. {!Method} and
      {!Gpfifo} give each operation's words as a template, and {!Qmd} a launch's
      descriptor and parameters.
    - A {e GPU} ({!Gpu.t}) is what the formats depend on: its classes, its
      geometry and the windows its driver chose, declared by the driver
      ({!declare}). A cubin's kernel ({!Cubin}) set up for launch on a GPU
      is a {!Launch.t}, from which its descriptor is built; {!Local_memory}
      sizes the local memory its threads take.

    {v
    cubin bytes --Cubin.of_string--> Cubin.t --kernel--> Cubin.kernel
                                       |                      |
                              size, elf, patches       Launch.make Gpu.t
                                       |                      |
                          a loader writes the image      Launch.t
                                                              |
                                                 Qmd.make, Qmd.parameters
                                                              |
    Method.*, Gpfifo.entry --------> Template.t <-------------+
                                         |
                       bytes as they are, or filled at each use
    v}

    {1:conventions Conventions}

    Addresses are the GPU's virtual addresses. A slot cannot be checked here:
    the caller keeps the alignment and range each function states for the
    value that fills it.

    Misuse raises [Invalid_argument]: an integer outside the range its function
    states, a class or version no launch descriptor holds. A cubin that is not
    one, or a kernel that takes more than a launch can, is [Error msg], [msg]
    saying why.

    The library has no state: any domain may use any of its values.

    {1:references References}

    {!Qmd} and {!Launch} name the documents of NVIDIA's that they follow. *)

(** {1:words Channel words} *)

module Template = Template
(** Bytes around the caller's values. *)

module Method = Method
(** Channel methods. *)

module Gpfifo = Gpfifo
(** Ring entries. *)

(** {1:gpus GPUs} *)

module Gpu = Gpu
(** NVIDIA GPUs, as their formats depend on them. *)

val key : Gpu.t Type.Id.t
(** [key] is the key of an NVIDIA GPU's record. Compiled code finds the record
    of a device [d] with [Rig.capability d key], which is [None] if [d] is
    not an NVIDIA GPU. *)

val declare : Gpu.t -> Rig_edge.capability
(** [declare c] is [c] under {!key}, as the facts of an NVIDIA GPU carry it
    ({!Rig_edge.type-facts}). *)

module Local_memory = Local_memory
(** Kernels' local memory. *)

(** {1:launches Launches} *)

module Cubin = Cubin
(** Cubins: the ELF objects NVIDIA's compilers make for a GPU architecture. *)

module Launch = Launch
(** Kernels set up for launch on a GPU. *)

module Qmd = Qmd
(** Launch descriptors. *)
