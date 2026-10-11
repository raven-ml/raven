(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The path that opens AMD GPUs through Linux's [amdgpu] kernel driver.

    The kernel driver owns the GPU and shares it with other programs. This
    library opens a GPU through the driver's compute interface, [/dev/kfd],
    and the GPU's render node, and gives {!Rig_amd} what the device needs
    ({!Rig_amd.type-path}): memory the kernel driver allocates and maps,
    queues it schedules, and events it signals on the GPU's interrupts. The
    device then runs as every {!Rig_amd} device does. The kernel driver
    keeps the GPU's firmware, power and resets, and reports faults.

    A program opens each of the machine's AMD GPUs through its opener:
    {[
    let devices =
      List.init (Rig_amd_amdgpu.count ()) (fun i ->
          Rig.open_ (Rig_amd_amdgpu.gpu i))
    ]}

    {b Numbering.} GPU [i] is the [i]th of the machine's AMD GPUs in bus
    order ({!Rig_amd.is_gpu}), whichever driver holds each, so a GPU taken
    from the kernel driver keeps its index, and the others theirs. The GPUs
    are named ["AMD"], ["AMD:1"], ["AMD:2"], ….

    {b The machine's files.} {!buses}, {!count} and an opener's [make] read
    the machine's files under the directory [root] (defaults to ["/"]): its
    PCI functions under [root/sys/bus/pci], the kernel driver's description
    of its GPUs under [root/sys], and the kernel driver's files under
    [root/dev], such as a container's view of its host. Every [root] a
    process uses must show the same kernel driver.

    {b Privileges.} Opening needs read and write access to [/dev/kfd] and to
    the GPU's render node, [/dev/dri/renderD*], which members of the
    [render] group have, and no other privilege. *)

val buses : ?root:string -> unit -> string list
(** [buses ~root ()] is the bus addresses of the machine's AMD GPUs
    ({!Rig_amd.is_gpu}) under [root] (defaults to ["/"]), in bus order: by
    domain, bus, device and function, each as a number. GPU [i] is the
    [i]th, whichever driver holds it. It is [[]] where [root/sys/bus/pci]
    does not exist, such as off Linux, and where the machine's PCI functions
    cannot be read, such as when the process has no file left; {!gpu}'s
    [make] then says why. Each call reads the files again, and changes
    nothing on the machine. *)

val count : ?root:string -> unit -> int
(** [count ~root ()] is [List.length (buses ~root ())], so GPUs are numbered
    [0] to [count ~root () - 1]. *)

val gpu : ?root:string -> int -> Rig_amd.t Rig_edge.opener
(** [gpu ~root i] is the opener of GPU [i] under [root] (defaults to ["/"]),
    named ["AMD"] for [0] and ["AMD:i"] otherwise. Each [make ()] opens a
    new device of the GPU with queues of its own, also while other devices
    of the GPU are open. The process reads the list of GPUs under a [root]
    at the first [make] there that can read it, and keeps it, so after a GPU
    is plugged in or removed it may differ from {!buses}.

    [make ()] returns [Error msg]:
    {ul
    {- if the machine's PCI functions cannot be read, saying why;}
    {- if [i] is past the GPUs listed, saying how many there are;}
    {- if the [amdgpu] driver does not hold the GPU;}
    {- if a file cannot be opened, naming it and the reason;}
    {- if the kernel driver refuses a request of the open, naming the step
       and the reason;}
    {- if the GPU's work faulted in this process, with the fault
       ({!section-process});}
    {- with {!Rig_amd.make}'s message, or the report of a fault while the
       device opens.}}

    Raises [Invalid_argument] if [i < 0]. *)

(** {1:process The process's GPU}

    The process opens [/dev/kfd] once, under the [root] of the first open
    that succeeds in making it, and every later open uses that file. The
    kernel driver gives a process one address space per GPU, which the
    first open of the GPU takes and the process keeps until it exits. Every
    device of the GPU works in it, and a fault of one device's work is
    reported to every device of the GPU.

    An open after a device of the GPU was lost makes new queues in that
    address space. Once the kernel driver has reported a fault of the GPU's
    work, though, it schedules none of the process's queues on that GPU
    again. Every later [make] of the GPU then returns [Error], every
    {!Rig_amd.val-sleep} of its devices raises the fault, and only another
    process can use the GPU. A forked child opens a GPU anew, in an address
    space of its own, and the parent keeps its devices.

    {b Memory.} The kernel driver maps a page of host memory once per GPU,
    for every device of the GPU the process opens. A region whose pages all
    lie within a region {!Rig_amd.val-map_host} gave a device of the same
    GPU shares that mapping, which lasts until the last region over it is
    freed, also once those devices have stopped. {!Rig_amd.val-map_host}
    refuses pages of which some, but not all, lie within such a region. A
    device reaches the memory of another GPU's device where the kernel
    driver's topology lists a link between the two GPUs.

    {b Faults.} The kernel driver reports a fault of the GPU's work, a page
    fault or a hardware exception such as a reset, with its address or
    cause, and {!Rig_amd.val-sleep} raises it. The kernel driver bounds no
    work on a KFD queue, so the device's [hang_ms] is 30 s
    ({!Rig_edge.type-facts}), and rig loses the device once its work has
    made no progress for that long.

    {b Tracing.} The first thread trace takes the GPU's stable power state,
    which holds its clocks and shader engines steady, until the process
    exits. The trace fails while another process holds that state. On a
    GFX 9 GPU, traces take no power state. *)

(**/**)

(* [gpu_at root bus] is the GPU at bus address [bus] as the [amdgpu] driver
   describes it under the directory [root]: its topology node and its blocks'
   versions. [Error msg] if the driver does not hold it. *)
val gpu_at : string -> string -> (Rig_amd_abi.Gpu.t, string) result

(* The type for a machine's AMD GPUs as the [amdgpu] driver describes them, read
   once, but for a GPU the driver did not hold at the last look. *)
type machine

(* [machine_at root] is the AMD GPUs of the machine whose files are under the
   directory [root], as {!buses} lists them, each with the node the driver has
   for it. [Error] with the system's error if its PCI functions cannot be
   read. *)
val machine_at : string -> (machine, string) result

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
