(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** AMD GPUs as nx devices, through nx's own runtime.

    A device of this library is an AMD GPU's memory, which {!Nx_amd_device}
    opens, through one of two interfaces: the compute interface of the [amdgpu]
    kernel driver ({!get}, named [AMD:i]), or nx's own driver over PCI
    ({!get_pci}, named [AMD-PCI:i]), which keeps the GPU for the process.
    Opening changes nothing on the machine: a GPU opened over PCI must first be
    detached from its kernel driver ({!detach}), reset if another booted it
    ({!reset}), and have its firmware at hand ({!fetch_firmware}), each a change
    to the machine that persists after the process. The first open of either
    fixes the interface for the process: the other then fails with that reason.
    Both number the machine's AMD GPUs in bus order, so [AMD:i] and [AMD-PCI:i]
    are the same GPU.

    A device computes eagerly through nx.amd's backend, AMD memory's own
    ([Nx_backend.S.owns]), from code objects the library carries, compiled for
    [gfx12-generic]. GPUs of the gfx12 generation (gfx1200, gfx1201) copy values
    ({!Nx.copy}, {!Nx.contiguous}) and cast them ({!Nx.cast}), and compute the
    elementwise functions of one, two and three operands, the comparisons and
    {!Nx.where}, and reduce them ({!Nx.sum}, {!Nx.prod}, {!Nx.max}, {!Nx.min},
    {!Nx.argmax}, {!Nx.argmin}), of every dtype but the complex ones, [int4],
    [uint4] and [bit]: bit for bit as the host does, but the transcendental
    functions, which keep the accuracy {!Nx} states, the sign and payload of a
    NaN an arithmetic operation makes, the NaN that {!Nx.max} or {!Nx.min} over
    several axes takes, the first in C order, and the rounding of float sums and
    products, whose terms associate differently. Any other eager operation,
    dtype or GPU raises [Invalid_argument] naming the remedies: [Rune.jit]
    compiles for every AMD device, and [Nx.place] moves values to the host.
    Constants, views and reads work on every AMD device. *)

val get : int -> (Nx.Device.t, string) result
(** [get i] is AMD GPU [i] of the kernel driver, opened now, or [Error msg] with
    the reason, such as that [/dev/kfd] is absent. It never touches PCI. Every
    [get i] that succeeds gives an equal device until the device is lost
    ({!Nx_device.Lost}); a [get i] then opens the GPU anew, an unequal device.

    Raises [Invalid_argument] if [i < 0]. *)

val device : int -> Nx.Device.t
(** [device i] is {!get}[ i].

    Raises [Failure] with {!get}'s reason if it does not open, and as {!get}
    does. *)

val get_pci : int -> (Nx.Device.t, string) result
(** [get_pci i] is AMD GPU [i] opened over PCI by nx's own driver, or
    [Error msg] with the reason, such as a missing privilege, or a GPU not
    detached, not reset or without its firmware, naming {!detach}, {!reset} or
    {!fetch_firmware}, or lost ({!Nx_device.Lost}), which only a {!reset} in
    another process recovers. See {!Nx_amd_device} for the privileges, the GPUs
    supported and the firmware it boots them with.

    Raises [Invalid_argument] if [i < 0]. *)

val device_pci : int -> Nx.Device.t
(** [device_pci i] is {!get_pci}[ i].

    Raises [Failure] with {!get_pci}'s reason if it does not open, and as
    {!get_pci} does. *)

(** {1:machine Changes to the machine}

    Each is {!Nx_amd_device}'s, which documents the privileges it needs. *)

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
