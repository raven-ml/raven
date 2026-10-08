(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** AMD GPUs of this machine, opened through Linux's [amdgpu] driver.

    The kernel driver owns the GPU and shares it with other programs. This
    library opens a GPU through its compute interface, [/dev/kfd], and gives the
    GPU's memory, queues and interrupts to {!Rig_amd}, which drives it
    ({!Rig_amd.make}):
    {[
    let g = Result.get_ok (Rig_amd_amdgpu.open_ 0) in
    Rig_amd.arch g (* "gfx1201" *)
    ]}

    {b Numbering.} GPU [i] is the [i]th of the machine's AMD GPUs in bus order
    ({!Rig_amd.is_gpu}), whichever driver holds each: a GPU taken from the
    kernel driver keeps its index, and the others theirs. The GPUs are named
    ["AMD"], ["AMD:1"], ["AMD:2"], ....

    {b Privileges.} Opening needs read and write access to [/dev/kfd] and to the
    GPU's render node, [/dev/dri/renderD*], which members of the [render] group
    have, and no other privilege.

    {b The process's GPU.} The kernel driver gives a process one address space
    per GPU, which the first open takes and the process keeps until it exits.
    Every device of a GPU works in it, and a fault of one device's work is
    reported to every device of the GPU. An open after a device of the GPU was
    lost makes new queues there, unless the kernel driver reported a fault of
    the GPU's work in the process, which {!open_} asks the GPU's open devices:
    the kernel driver then schedules none of the process's queues on the GPU
    again, so {!open_} answers [Error], every later {!Rig_amd.sleep} of a device
    of the GPU raises the fault, and another process opens the GPU. A forked
    child opens a GPU anew, in an address space of its own: the parent's devices
    stay the parent's.

    {b Faults.} The kernel driver reports a fault of the GPU's work, a page
    fault or a hardware exception such as a reset, with its address or cause;
    {!Rig_amd.sleep} raises it. It also bounds long work by its own rules:
    nothing in this library decides that work hangs.

    {b Tracing.} The first thread trace takes the GPU's stable power state,
    which holds its clocks and shader engines steady, until the process exits.
    The trace fails while another process holds that state. On a GFX 9 GPU,
    traces take no power state.

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

val open_ : int -> (Rig_amd.t, string) result
(** [open_ i] opens GPU [i] through the [amdgpu] driver, with queues of its own.
    The result is [Error msg] if [i >= count ()], saying how many GPUs there
    are, if the [amdgpu] driver does not hold the GPU, if a file cannot be
    opened, naming it and the reason, if the kernel driver refuses the GPU's
    address space or events, naming the step and the reason, if the GPU faulted
    in this process, with the fault, or with {!Rig_amd.make}'s message.

    Raises [Invalid_argument] if [i < 0]. *)

(**/**)

(* [gpus_at root] is the bus addresses of the AMD GPUs of the machine whose
   files are under the directory [root], in bus order: the functions of
   [root/sys/bus/pci] that [Rig_amd.is_gpu] names. [/] for this machine, a
   fixture's directory in tests. *)
val gpus_at : string -> string list

(* [gpu_at root bus] is the GPU at bus address [bus] as the [amdgpu] driver
   describes it under the directory [root]: its topology node and its blocks'
   versions. [Error msg] if the driver does not hold it. *)
val gpu_at : string -> string -> (Rig_amd_abi.Gpu.t, string) result

(* The type for a machine's AMD GPUs as the [amdgpu] driver describes them, read
   once, but for a GPU the driver did not hold at the last look. *)
type machine

(* [machine_at root] is the AMD GPUs of the machine whose files are under the
   directory [root], as {!gpus_at} lists them, each with the node the driver has
   for it. *)
val machine_at : string -> machine

(* [gpus_of m] is the GPUs of [m] in bus order as {!gpu_at} describes each. A
   GPU [Error] at the last look is looked at again; one found is kept. *)
val gpus_of : machine -> (string * (Rig_amd_abi.Gpu.t, string) result) list

(* [save_area_at root bus] is the bytes of a compute queue's context save area
   for the GPU at [bus] as the [amdgpu] driver describes it under the directory
   [root]: each die's area and its debugger area, as the kernel driver requires.
   [Error msg] as [gpu_at]. *)
val save_area_at : string -> string -> (int, string) result

(* [wgps_of gpu ~arrays ~per_array cus] is the work-group processors of each
   shader array of [gpu], engines numbered across dies ({!Rig_amd.path}'s
   [wgps]), from the render node's compute-unit bitmap [cus] (16 ints, engine by
   engine as [cu_bitmap] lays them out), which describes the first die: later
   dies have each of an array's [per_array] compute units set. [arrays] is the
   shader arrays of an engine. *)
val wgps_of :
  Rig_amd_abi.Gpu.t ->
  arrays:int ->
  per_array:int ->
  int array ->
  int array array
