(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The path that drives AMD GPUs over PCI, with no kernel driver.

    The process takes the GPU's PCI function from Linux, boots the GPU from
    firmware files, and gives {!Rig_amd} the GPU's memory, queues and
    interrupt ring ({!Rig_amd.type-path}). The device then runs as every
    {!Rig_amd} device does, and work reaches the GPU with no system call.
    This library does what a kernel driver would do. It boots the GPU,
    bounds how long work runs, reports faults from the interrupt ring, and
    resets a GPU whose state it cannot trust. {!Rig_pci} holds the mechanism
    this path shares with other vendors' drivers, such as taking a function
    and holding a GPU.

    A GPU passes through these states:
    {v
                detach                Rig.open_
    amdgpu ---------------> detached ------------> held
       ^                     |  ^  ^                 |
       +-------attach--------+  |  +-----close-------+
                                |                    | fault, hang,
                                |                    v killed process
                                +------reset------- lost
    v}
    {ul
    {- While [amdgpu] is bound, the kernel driver holds the GPU. {!detach}
       takes it from the kernel driver, and {!attach} gives it back.}
    {- A detached GPU ({!Rig_pci.Gpus}) is bound to no kernel driver, or to
       [vfio-pci], [amdgpu] has let go of it, and no process holds it.}
    {- A held GPU belongs to this process, from [Rig.open_] until its device
       stops ({!Rig_pci.Gpus.type-hold}). Closing the device stops the
       GPU's work and leaves it detached, marked clean, so that the next
       open boots it in part ({!section-opening}).}
    {- A GPU is lost when a fault or a hang ended its device, a boot failed
       after its first write, or the process that held it was killed. Its
       state is then unknown. {!reset} makes it detached, and its next open
       resets it first ({!section-opening}).}}

    A program opens the first GPU with the firmware that the [rig] command
    downloaded ({!section-needs}):
    {[
    let d = Rig.open_ (Rig_amd_pci.gpu 0)
    ]}

    {b Hardware status.} Virtual functions and XGMI fabrics are written,
    never run on hardware.

    {!Rig_pci.section-needs} states what any open of a GPU with no kernel
    driver needs, and {!Rig_pci.Function} how a process takes a GPU's
    function. *)

(** {1:needs What an open needs}

    {b GPUs.} The library boots GPUs whose blocks have versions it holds
    register tables and firmware for: GC 9.4.3 and 9.5.0 (the Instinct
    MI300 and MI350 series, physical and virtual functions), 11.0.0 and
    11.0.2, and 12.0.0 and 12.0.1. It reads the GPU's discovery table
    without changing the GPU's state, and refuses a GPU with a block of
    another version before the boot writes. {!val-report} tells, from a GPU's
    discovery table, whether the library boots it.

    {b Firmware.} The images are files of linux-firmware's [amdgpu/]
    directory at commit [0a6871b1], each pinned by its BLAKE2b-256 digest.
    They are named for the versions of the GPU's blocks: its security
    processor's [psp_X_sos.bin], its power manager's [smu_X.bin], its copy
    engines' [sdma_X.bin], and its GC's [gc_X_mec.bin] and [gc_X_rlc.bin],
    with [gc_X_imu.bin] from GC 11 and [gc_X_pfp.bin] and [gc_X_me.bin] from
    GC 12. The R9700, for one, boots with [psp_14_0_3_sos.bin],
    [smu_14_0_3.bin], [sdma_7_0_1.bin] and
    [gc_12_0_1_{pfp,me,mec,imu,rlc}.bin]. The library reads uncompressed
    files only. The [rig] command downloads the pinned images into rig's
    firmware directory ({!Rig_pci.Firmware.directory}):
    {v rig firmware amd v}

    {b Host memory.} Behind an IOMMU, the device's host memory counts
    against the process's locked-memory limit ([RLIMIT_MEMLOCK]). Taken
    physically, a GPU reaches only the host memory this library allocates
    for it, in 2 MiB huge pages from [/dev/hugepages]. The device takes one
    page for each 2 MiB block of its host memory, staging included, and the
    system must have them free ([vm.nr_hugepages]). The process also needs
    [CAP_SYS_ADMIN], to read physical addresses in [/proc/self/pagemap].
    Such a device does not declare [maps_host] ({!Rig_edge.type-facts}). *)

(** {1:gpus GPUs}

    GPU [i] of a machine is the [i]th of its AMD GPUs in bus order
    ({!Rig_amd.is_gpu}), whichever driver holds each, so it is the same GPU
    whichever path opens it. The GPUs are named ["AMD-PCI"], ["AMD-PCI:1"],
    ["AMD-PCI:2"], …. {!buses}, {!gpu} and {!reset} also reach another
    machine ({!Rig_pci.Machine.t}). *)

val buses : ?machine:Rig_pci.Machine.t -> unit -> string list
(** [buses ~machine ()] is the bus addresses of the AMD GPUs of [machine]
    (defaults to {!Rig_pci.Machine.this}) ({!Rig_amd.is_gpu}), in bus order
    ({!Rig_pci.Machine.val-functions}): GPU [i] is the [i]th, whichever driver
    holds it. It is [[]] for a machine that lists no PCI functions, such as
    this one off Linux. Listing them changes nothing on the machine. *)

val count : ?machine:Rig_pci.Machine.t -> unit -> int
(** [count ~machine ()] is [List.length (buses ~machine ())], so GPUs are
    numbered [0] to [count ~machine () - 1]. *)

(** {1:opening Opening}

    {b Times.} A device measures the GPU's clock against the host's at open
    only, with stamps of its compute queue ({!Rig_amd.Round_trip}), so the
    times of its profiles drift from the host's clock at the clock's rate
    error since open.

    {b Sessions.} An open writes a {e mark}, a value in one of the GC's
    scratch registers that says this library booted the GPU and whether its
    session ended cleanly. The process stops every GPU it holds at exit. A
    GPU whose work stopped with no fault is marked clean, and the next open,
    by this process or another, boots only its compute and copy blocks, from
    the state the last session left. A child of [fork] stops nothing at its
    exit.

    The open resets the GPU as {!reset} does, then boots it in full, when
    this library cannot vouch for the GPU's state, whoever left it. Such a
    GPU may run firmware without the mark, as its kernel driver leaves it,
    or have been left by a killed process, or by a process that lost it to
    a fault or a hang, this process or another. The open returns [Error] if
    firmware this library did not start still runs after the reset. On
    GC 9.5.0 this reset is written and has never run on hardware.

    A boot turns the GPU's bus mastering on only once both of its memory
    hubs translate addresses, so that no address a boot step gets wrong
    reaches host memory.

    {b Fabrics.} This library cannot tell which fabric (XGMI) a GPU belongs
    to. It takes a machine's AMD GPUs for one fabric when their number is a
    fabric's size, and they then reach each other's memory over its links.
    Otherwise they reach each other through their memory BARs. Two halves of
    two fabrics are taken for one. A fabric's GPUs reset together, as their
    kernel driver resets them when it takes them back, so {!reset} only
    stops such a GPU, and an open refuses one left running.

    {b Faults and hangs.} The GPU reports faults on its interrupt ring: page
    faults with their address, shader errors, and fatal hardware errors with
    its machine-check banks. {!Rig_amd.val-sleep} raises them. No kernel
    bounds the GPU's work, so the device declares [hang_ms]
    ({!Rig_edge.type-facts}) of {!Rig_pci.Gpus.hang_ms}. A device stopped
    with the fault it was lost for, a hang included, leaves its GPU lost,
    since its stop does not wait for the queues to leave.

    {b Domains.} Any domain may call any value. Opens, resets and changes to
    the machine of AMD GPUs run one at a time, so a boot delays the
    others. *)

val gpu :
  ?machine:Rig_pci.Machine.t ->
  ?firmware:string list ->
  int ->
  Rig_amd.t Rig_edge.opener
(** [gpu ~machine ~firmware i] is the opener of GPU [i] of [machine]
    (defaults to {!Rig_pci.Machine.this}), named ["AMD-PCI"] for [0] and
    ["AMD-PCI:i"] otherwise. Its [make ()] takes the GPU's function, boots
    the GPU from the images in the directories [firmware] (defaults to
    [[]]), then {!Rig_pci.Firmware.directory}, then ["/lib/firmware"], the
    first directory holding an image winning, and opens its device. Run as
    root, an open searches root's home.

    [make ()] returns [Error msg], [msg] starting with the GPU's name.
    Before the boot writes to the GPU, it refuses:
    {ul
    {- a GPU [i] the machine does not have, saying how many it has;}
    {- a GPU this process holds, or one [amdgpu] has not let go of;}
    {- a function the process cannot take, with
       {!Rig_pci.Function.take}'s reason;}
    {- a GPU whose {!val-report} is [Error] or lacks an image, with the report's
       reason;}
    {- a GPU in a fabric left running, or one whose memory its memory
       controller cannot place whole;}
    {- a machine whose windows ({!Rig_pci.Window}) are not mapped into the
       process.}}
    A reset the open needs that fails returns its reason. After the boot's
    first write, [make] returns the step at which a block stopped
    responding, or {!Rig_amd.make}'s refusal. In both cases the GPU is
    lost.

    Raises [Invalid_argument] if [i < 0]. *)

(** {1:reports Boot reports}

    A report says what an open loads to boot a GPU, and whether it refuses
    the GPU for its blocks or its firmware, from the GPU's discovery table.
    A report needs no GPU and writes nothing. *)

type block = {
  name : string;  (** As the kernel names it, such as ["GC"]. *)
  version : int * int * int;
      (** (major, minor, revision), as the table and the firmware files' names
          give them, such as [(13, 0, 3)]. *)
  instances : int list;
      (** The instances the boot drives, in order; harvested ones are absent. *)
}
(** The type for a block a boot programs. *)

type image = {
  file : string;
      (** The file's path under a firmware directory, such as
          ["amdgpu/psp_14_0_3_sos.bin"]. *)
  found : string option;
      (** The path of the file with its pinned digest, in the first directory
          that holds one, if any does. *)
}
(** The type for a firmware file a boot loads. *)

type report = {
  blocks : block list;
      (** The blocks a boot programs: GC, SDMA0, MP0, MP1, MMHUB, OSSSYS, NBIF
          and HDP, in that order. *)
  images : image list;  (** In the order the boot loads them. *)
}
(** The type for boot reports. *)

val report : firmware:string list -> string -> (report, string) result
(** [report ~firmware table] is what an open loads to boot a GPU whose IP
    discovery table is [table]: its blocks, their versions and the instances
    harvesting left, and its firmware files, each looked up in the
    directories [firmware] as an open looks one up in its own. An open
    computes the same report before the boot writes, and again after any
    reset.

    [table] is the 10 KiB the GPU's firmware leaves 64 KiB below the end of
    its memory. While [amdgpu] is bound to the GPU, root reads it in the
    debugfs file [/sys/kernel/debug/dri/<bus address>/amdgpu_discovery].

    The result is [Error why] if [table] is not a discovery table, if a
    block a boot programs is missing or has a version this library does not
    boot or no pinned image for, naming the block and its version, or if a
    file found with its pinned digest is not laid out as its format says. *)

(** {1:changes Changes to the machine}

    {!detach}, {!attach} and {!reset} change the machine as
    {!Rig_pci.Gpus} describes, and their changes persist after the process.
    Each refuses a GPU that a process holds, this one included, over PCI or
    through a file of its kernel driver, such as the GPU's render node.
    {!detach} and {!attach} act on this machine alone, and need
    [CAP_SYS_ADMIN] and write access to the files under [/sys/bus/pci] they
    write, which root has. An [Error] or a [Part_way] for a file names it. *)

val detach : int -> (Rig_pci.Gpus.progress, string) result
(** [detach i] takes GPU [i] of this machine from [amdgpu] as
    {!Rig_pci.Gpus.detach} does, and makes its memory BAR as large as the
    BAR and its bridge allow. It waits up to 30 s for KFD to drop the files
    of the GPU's DRM nodes that it holds for processes that have exited,
    reading them in debugfs, which must be mounted at [/sys/kernel/debug].
    Once [detach] returns [Ok Detached], [amdgpu]'s last write to the GPU is
    done.

    The result is as {!Rig_pci.Gpus.detach} returns it. While KFD's
    topology still lists the GPU or its [ip_discovery] directory stays after
    the unbind, [amdgpu] has not let go of the GPU. [detach] then returns
    [Ok (Unreleased msg)], and a later [detach] waits up to 30 s for both
    to go and finishes, or returns [Error msg] if they stay.

    Raises [Invalid_argument] if [i < 0]. *)

val attach : int -> (unit, string) result
(** [attach i] gives GPU [i] of this machine back to [amdgpu] as
    {!Rig_pci.Gpus.attach} does, resetting it first as {!reset} does if it
    is detached. The result is [Error msg] as {!Rig_pci.Gpus.attach}
    returns it.

    After a detach, [amdgpu] serves the GPU's render node again, but KFD
    refuses every process until the [amdgpu] module is reloaded, so
    [Rig_amd_amdgpu] opens the GPU only after that.

    Raises [Invalid_argument] if [i < 0]. *)

val reset : ?machine:Rig_pci.Machine.t -> int -> (unit, string) result
(** [reset ~machine i] stops whatever runs on GPU [i] of [machine] (defaults
    to {!Rig_pci.Machine.this}) and, if its security processor's OS runs,
    resets the GPU whole (mode 1), which clears what any driver left. A GPU
    no OS runs on keeps its state, but for its interrupt rings, which
    [reset] turns off. A GPU of a fabric is only stopped ({!section-opening}),
    and a virtual function gives back its access for its physical function
    to reset. Afterwards the GPU opens with a full boot.

    The result is [Error msg] if [i >= count ~machine ()], if a process
    holds the GPU, if its function cannot be taken, or if after the reset
    the GPU does not respond, its configuration differs, its security
    processor or interrupt rings still run, or its bootloader is not ready
    with no OS running. Only a power cycle recovers a GPU that does not
    respond or whose bootloader is not ready.

    Raises [Invalid_argument] if [i < 0]. *)

(** {1:references References}

    {ul
    {- PCI-SIG. {e PCI Express Base Specification}, 7.5.3.7 "Link Control
       Register": ASPM.}} *)

(**/**)

(* What a driver that boots the GPU itself reads from the GPU and its
   firmware, and writes for the GPU's blocks. Tests check them without a
   GPU. *)

module Discovery = Discovery
(** The discovery table: the GPU's blocks, their versions and where their
    registers lie. *)

module Firmware = Firmware
(** The firmware a GPU boots with, cut into the pieces its security
    processor loads. *)

module Psp = Psp
(** The security processor's commands. *)

module Smu = Smu
(** The power manager's messages. *)

module Gmc = Gmc
(** The memory controller's window and page-table entries. *)

module Gfx = Gfx
(** The compute queues' descriptors and the shader array index. *)

module Ih = Ih
(** The interrupt rings' entries. *)
