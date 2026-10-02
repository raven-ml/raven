(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** NVIDIA GPUs as nx devices, through nx's own runtime.

    A device of this library is an NVIDIA GPU's memory, which {!Nx_nv_device}
    opens, through one of two interfaces: NVIDIA's kernel driver ({!get}, named
    [NV:i]), or nx's own driver over PCI ({!get_pci}, named [NV-PCI:i]), which
    detaches the GPU's kernel driver and keeps the GPU for the process. Only a
    call of {!get_pci} or {!device_pci} takes a GPU over PCI. The first open of
    either fixes the interface for the process: the other then fails with that
    reason.

    A device computes eagerly with no backend: [Rune.jit] compiles for it, and
    an eager operation on its values raises [Invalid_argument] naming the
    remedies. Constants, views, reads and [Nx.place] work on it.

    A program with both the CUDA driver and NVIDIA's kernel driver can open one
    GPU as [Nx_cuda.device i] and as [device i]: they are two memories, which
    copy between them. *)

val get : int -> (Nx.Device.t, string) result
(** [get i] is NVIDIA GPU [i] of the kernel driver, opened now, or [Error msg]
    with the reason, such as that [/dev/nvidiactl] is absent. It never touches
    PCI. Every [get i] that succeeds gives an equal device.

    Raises [Invalid_argument] if [i < 0]. *)

val device : int -> Nx.Device.t
(** [device i] is {!get}[ i].

    Raises [Failure] with {!get}'s reason if it does not open, and as {!get}
    does. *)

val get_pci : int -> (Nx.Device.t, string) result
(** [get_pci i] is NVIDIA GPU [i], taken from its kernel driver and opened over
    PCI by nx's own driver, or [Error msg] with the reason, such as a missing
    privilege or firmware image. See {!Nx_nv_device} for the privileges, the
    GPUs supported and the firmware it boots them with.

    Raises [Invalid_argument] if [i < 0]. *)

val device_pci : int -> Nx.Device.t
(** [device_pci i] is {!get_pci}[ i].

    Raises [Failure] with {!get_pci}'s reason if it does not open, and as
    {!get_pci} does. *)
