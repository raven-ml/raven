(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** AMD GPUs of this machine, opened through Linux's [amdgpu] driver.

    The kernel driver owns the GPU and shares it with other programs. This
    library opens a GPU through its compute interface, [/dev/kfd], and gives the
    GPU's memory, queues and interrupts to {!Device_amd}, which drives it
    ({!Device_amd.make}):
    {[
    let g = Result.get_ok (Device_amd_amdgpu.open_ 0) in
    Device_amd.arch g (* "gfx1201" *)
    ]}

    {b Numbering.} GPU [i] is the [i]th of the machine's AMD GPUs in bus order
    ({!Device_amd.is_gpu}), whichever driver holds each: a GPU taken from the
    kernel driver keeps its index, and the others theirs. The GPUs are named
    ["AMD"], ["AMD:1"], ["AMD:2"], ....

    {b Privileges.} Opening needs read and write access to [/dev/kfd] and to the
    GPU's render node, [/dev/dri/renderD*], which members of the [render] group
    have, and no other privilege. Elsewhere than Linux, {!count} is [0].

    {b The process's GPU.} The kernel driver gives a process one address space
    per GPU, which the first open takes and the process keeps until it exits.
    Every device of a GPU works in it: an open after a device of the GPU was
    lost makes new queues there, and a fault of one device's work is reported to
    every device of the GPU.

    {b Faults.} The kernel driver reports a fault of the GPU's work, a page
    fault or a hardware exception such as a reset, with its address or cause;
    {!Device_amd.sleep} raises it. It also bounds long work by its own rules:
    nothing in this library decides that work hangs.

    {b Tracing.} Thread traces hold the GPU's stable power state, its clocks and
    shader engines steady, from the first trace to the end of the process,
    unless another process holds it.

    {b References.}
    - The Linux kernel's [include/uapi/linux/kfd_ioctl.h] and
      [include/uapi/drm/amdgpu_drm.h]: the compute interface and the render
      node's queries.
    - The Linux kernel's [drivers/gpu/drm/amd/amdkfd]: [kfd_events.c] (signal
      events and their slots), [kfd_queue.c] (the sizes of a queue's context
      save area) and the topology it lists under
      [/sys/devices/virtual/kfd/kfd/topology]. *)

val count : unit -> int
(** [count ()] is the number of the machine's AMD GPUs, whichever driver holds
    them: the indices [0] to [count () - 1]. It is [0] elsewhere than Linux. *)

val device_name : int -> string
(** [device_name i] is the name of GPU [i]: ["AMD"] for [0], ["AMD:i"]
    otherwise.

    Raises [Invalid_argument] if [i < 0]. *)

val open_ : int -> (Device_amd.t, string) result
(** [open_ i] opens GPU [i] through the [amdgpu] driver, with queues of its own.
    The result is [Error msg] if [i >= count ()], saying how many GPUs there
    are, if the [amdgpu] driver does not hold the GPU, if a file cannot be
    opened, naming it and the reason, or with {!Device_amd.make}'s message.

    Raises [Invalid_argument] if [i < 0]. *)

(**/**)

(* [gpus_at root] is the bus addresses of the AMD GPUs of the machine whose
   files are under the directory [root], in bus order: the functions of
   [root/sys/bus/pci] that [Device_amd.is_gpu] names. [/] for this machine, a
   fixture's directory in tests. *)
val gpus_at : string -> string list

(* [gpu_at root bus] is the GPU at bus address [bus] as the [amdgpu] driver
   describes it under the directory [root]: its topology node and its blocks'
   versions. [Error msg] if the driver does not hold it. *)
val gpu_at : string -> string -> (Device_amd_abi.Gpu.t, string) result
