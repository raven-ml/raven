(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** NVIDIA GPUs as nx devices, through nx's own runtime.

    A device of this library is an NVIDIA GPU's memory, which {!Nx_nv_device}
    opens, through one of two interfaces: NVIDIA's kernel driver ({!get}, named
    [NV:i]), or nx's own driver over PCI ({!get_pci}, named [NV-PCI:i]), which
    keeps the GPU for the process. Opening changes nothing on the machine: a GPU
    opened over PCI must first be detached from its kernel driver ({!detach}),
    reset ({!reset}), and have its firmware at hand ({!fetch_firmware}), each a
    change to the machine that persists after the process. The first open of
    either fixes the interface for the process: the other then fails with that
    reason. Both number the machine's NVIDIA GPUs in bus order, so [NV:i] and
    [NV-PCI:i] are the same GPU.

    A device computes eagerly with no backend: [Rune.jit] compiles for it, and
    an eager operation on its values raises [Invalid_argument] naming the
    remedies. Constants, views, reads and [Nx.place] work on it.

    A program with both the CUDA driver and NVIDIA's kernel driver can open one
    GPU as [Nx_cuda.device i] and as [device i]: they are two memories, which
    copy between them. *)

val get : int -> (Nx.Device.t, string) result
(** [get i] is NVIDIA GPU [i] of the kernel driver, opened now, or [Error msg]
    with the reason, such as that [/dev/nvidiactl] is absent. It never touches
    PCI. Every [get i] that succeeds gives an equal device until the device is
    lost ({!Nx_device.Lost}); a [get i] then opens the GPU anew, an unequal
    device.

    Raises [Invalid_argument] if [i < 0]. *)

val device : int -> Nx.Device.t
(** [device i] is {!get}[ i].

    Raises [Failure] with {!get}'s reason if it does not open, and as {!get}
    does. *)

val get_pci : int -> (Nx.Device.t, string) result
(** [get_pci i] is NVIDIA GPU [i] opened over PCI by nx's own driver, or
    [Error msg] with the reason, such as a missing privilege, or a GPU not
    detached, not reset or without its firmware, naming {!detach}, {!reset} or
    {!fetch_firmware}, or lost ({!Nx_device.Lost}), which only a {!reset} in
    another process recovers. See {!Nx_nv_device} for the privileges, the GPUs
    supported and the firmware it boots them with.

    Raises [Invalid_argument] if [i < 0]. *)

val device_pci : int -> Nx.Device.t
(** [device_pci i] is {!get_pci}[ i].

    Raises [Failure] with {!get_pci}'s reason if it does not open, and as
    {!get_pci} does. *)

(** {1:machine Changes to the machine}

    Each is {!Nx_nv_device}'s, which documents the privileges it needs. *)

val detach : int -> (unit, string) result
(** [detach i] detaches GPU [i] from its kernel driver, which loses it, for
    {!get_pci}. It persists until {!attach} or a reboot. *)

val attach : int -> (unit, string) result
(** [attach i] gives GPU [i] back to its kernel driver, for {!get}. *)

val reset : int -> (unit, string) result
(** [reset i] resets detached GPU [i], clearing what an earlier boot left on it,
    for {!get_pci}. *)

val fetch_firmware : int -> (unit, string) result
(** [fetch_firmware i] downloads the firmware images GPU [i] boots with under
    {!get_pci} into the user's cache. *)
