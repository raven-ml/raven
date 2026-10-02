(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** RDMA network adapters.

    Opens the Broadcom NetXtreme RoCE adapters (BCM57608, [14e4:1760]) of a
    machine as {!Nx_device.t}s named ["RDMA"], ["RDMA:1"], ..., and
    ["RDMA@HOST:PORT"], ["RDMA:1@HOST:PORT"], ... for another machine's
    ([nx.remote.device]). The runtime drives an adapter itself, as
    [nx.amd.device] and [nx.nv.device] drive GPUs under their PCI interface: it
    takes the adapter's PCI function, which {!detach} detached from its kernel
    driver, sets up its firmware's queues and the RoCE engine's context memory,
    and gives it an address on the fabric, [10.x.y.z] from the last three bytes
    of its MAC address. It needs Linux and the privileges of
    {!Nx_device_support.Pci} and {!Nx_device_support.Sysmem} on the adapter's
    machine.

    {b Copies between machines.} Once an adapter is open on each of two
    machines, {!Nx_device.Buffer.copy} between the memory of GPUs of those
    machines, opened over PCI, moves the bytes from GPU to GPU over RoCE: a
    reliable connected queue pair joins each pair of GPUs, through the adapter
    of each machine closest to its GPU on the PCI bus; the adapter reads and
    writes GPU memory through the GPU's memory BAR, which must be large. Each
    chunk of at most 1 GiB is a send of the source's adapter and a receive of
    the destination's, which the host posts and waits for. The adapters register
    the GPU memory they move at its first copy, and deregister it when the GPU
    frees it. Other copies between machines go through the hosts.

    {b Buffers} of an adapter are locked system memory of its machine. It loads
    no programs and has no copy queue.

    {b Failures.} A copy whose completion does not arrive within the adapters'
    {!Nx_device.timeout}, or completes in error, loses both adapters of the
    queue pair ({!Nx_device.Lost}), whose state is then unknown, and the
    destination stays in their reach. At exit the process unregisters from each
    healthy adapter's firmware, and turns every adapter's bus mastering off, so
    that none reaches memory the process releases. The fabric is trusted: a
    packet with the right queue pair and sequence number writes the memory it
    names. *)

val count : ?host:Nx_device.t -> unit -> int
(** [count ()] is the number of adapters of the machine of [host] (defaults to
    {!Nx_device.host}), whatever driver they have. [0] where the machine has no
    PCI functions, such as off Linux.

    Raises [Invalid_argument] if [host] is no host, and {!Nx_device.Lost} with
    [host] if [host]'s machine cannot be reached. *)

val get : ?host:Nx_device.t -> int -> (Nx_device.t, string) result
(** [get i] is the adapter [i] of the machine of [host] (defaults to
    {!Nx_device.host}), opened by the first call that succeeds; every later call
    returns the same value.

    [Error msg] says why the adapter cannot be opened, for example that
    [i >= count ()], that the adapter is not detached, naming {!detach}, that a
    privilege is missing, or that its firmware refused a request, after the
    adapter's name, such as ["RDMA:2: no adapter 2; there are 2"]. A failed open
    gives the function back, its bus mastering off.

    Raises [Invalid_argument] if [i < 0] or if [host] is no host, and
    {!Nx_device.Lost} with [host] if [host]'s machine cannot be reached. *)

val v : ?host:Nx_device.t -> int -> Nx_device.t
(** [v i] is like {!get} but raises [Failure] with [get]'s message when the
    adapter cannot be opened. *)

(** {1:machine Changes to the machine}

    Each acts on adapter [i] of this machine, refuses one the process has open,
    and persists after the process. [Error msg] starts with the adapter's name.
*)

val detach : int -> (unit, string) result
(** [detach i] detaches adapter [i] from its kernel driver, which loses it, its
    network interfaces among them, so that {!get} can take it: see
    {!Nx_device_support.Pci.detach}, whose privileges it needs. It persists
    until {!attach} or a reboot. *)

val attach : int -> (unit, string) result
(** [attach i] gives adapter [i] back to its kernel driver: see
    {!Nx_device_support.Pci.attach}, whose privileges it needs. *)
