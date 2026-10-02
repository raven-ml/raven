(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** AMD GPUs as nx devices, through nx's own runtime.

    A device of this library is an AMD GPU's memory, which {!Nx_amd_device}
    opens, through one of two interfaces: the compute interface of the [amdgpu]
    kernel driver ({!get}, named [AMD:i]), or nx's own driver over PCI
    ({!get_pci}, named [AMD-PCI:i]), which detaches the GPU's kernel driver and
    keeps the GPU for the process. Only a call of {!get_pci} or {!device_pci}
    takes a GPU over PCI. The first open of either fixes the interface for the
    process: the other then fails with that reason.

    A device computes eagerly with no backend: [Rune.jit] compiles for it, and
    an eager operation on its values raises [Invalid_argument] naming the
    remedies. Constants, views, reads and [Nx.place] work on it. *)

val get : int -> (Nx.Device.t, string) result
(** [get i] is AMD GPU [i] of the kernel driver, opened now, or [Error msg] with
    the reason, such as that [/dev/kfd] is absent. It never touches PCI. Every
    [get i] that succeeds gives an equal device.

    Raises [Invalid_argument] if [i < 0]. *)

val device : int -> Nx.Device.t
(** [device i] is {!get}[ i].

    Raises [Failure] with {!get}'s reason if it does not open, and as {!get}
    does. *)

val get_pci : int -> (Nx.Device.t, string) result
(** [get_pci i] is AMD GPU [i], taken from its kernel driver and opened over PCI
    by nx's own driver, or [Error msg] with the reason, such as a missing
    privilege or firmware image. See {!Nx_amd_device} for the privileges, the
    GPUs supported and the firmware it boots them with.

    Raises [Invalid_argument] if [i < 0]. *)

val device_pci : int -> Nx.Device.t
(** [device_pci i] is {!get_pci}[ i].

    Raises [Failure] with {!get_pci}'s reason if it does not open, and as
    {!get_pci} does. *)
