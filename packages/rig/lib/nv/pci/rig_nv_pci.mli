(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** NVIDIA GPUs driven over PCI, with no kernel driver.

    This library is the {!Rig_nv} path for a GPU that no kernel driver
    holds. The process takes the GPU's PCI function, boots the GPU's system
    processor (GSP) with NVIDIA's signed firmware, and then acts as the
    kernel's half of the resource manager that the GSP runs. It gives the
    GSP the memory of the objects the GSP makes and runs the register
    sequences the GSP asks of it; the two talk through message queues in
    system memory. Once the GSP runs, a {!Rig_nv} device runs submissions
    on the GPU as it does through NVIDIA's kernel driver.

    This library is written, never run on hardware.

    A machine is prepared once, outside the program, and a program then
    opens the GPU:
    {[
    Rig.open_ (Rig_nv_pci.gpu 0)
    ]}

    {v
                    detach                 Rig.open_
    kernel driver ---------> detached ------------------> held
          ^                   ^                            |
          +----- attach ------+---------- stop ------------+
    v}

    {!detach} takes the GPU from its kernel driver, and {!attach} gives it
    back. An open holds the GPU until its device is stopped, and no other
    open, in this process or another, can take it meanwhile. A GPU the
    process held keeps its GSP running after that, and its next open or
    {!attach} resets it first, as {!reset} does. A GPU whose reset has
    failed is lost, and its next open resets it again.

    {!Rig_pci.section-needs} states what any open of a GPU with no kernel
    driver needs. *)

(** {1:requirements Requirements}

    Opening requires Linux and a PCI function the process can take. How the
    process takes it depends on whether an IOMMU translates the GPU's
    accesses to system memory:
    - With an IOMMU, bind the function to [vfio-pci] and grant access to its
      group's [/dev/vfio/N]. {!Rig_pci.Function.take} states the group's
      rules.
    - Without one, the process takes the GPU {e physically}. Detach its
      kernel driver ({!detach}) and grant write access to its files under
      [/sys/bus/pci]. The kernel must allow PCI mappings, which kernel
      lockdown refuses, and the process must read [/proc/self/pagemap].

    The chips whose boot this library lays out open: GA102, GA103, GA104,
    GA106 and GA107 (Ampere), AD102, AD103, AD104, AD106 and AD107 (Ada),
    and GB202, GB203, GB205, GB206 and GB207 (Blackwell), as the GPU's
    [NV_PMC_BOOT_42] register names them. An open needs the firmware files
    {!section-firmware} lists.

    [Mapped] memory ({!Rig_nv.section-regions}) is the GPU's, through its
    memory BAR (BAR 1), the range of the GPU's memory that the host
    addresses over PCI. [Pinned] memory is the machine's system memory
    ({!Rig_pci.Function.alloc_dma}). The GPUs' addresses and the process's
    addresses of system memory coincide, from 64 GiB to {!Rig_nv.windows},
    which the process reserves at its first open; every GPU it opens takes
    its addresses from that range.
    - Behind an IOMMU, system memory counts against the process's limit on
      locked memory ([RLIMIT_MEMLOCK]), and {!Rig_nv.val-alloc} returns
      [None] once it is reached. Raise the limit to allocate more.
    - Taken physically, system memory lies in huge pages of 2 MiB, and
      {!Rig_nv.val-alloc} returns [None] once none is free. Reserve more
      huge pages ([vm.nr_hugepages]) or free memory. Such a GPU's device has
      [maps_host] [false], since the process's pages would go back to the
      system at its death with the GPU still writing them.

    A boot also takes system memory for the GSP, most of it for the
    firmware's image, and it counts against the same limit. The GSP's logs
    and bootloader, and on Blackwell the FMC, each lie in one run of bus
    addresses.

    The GSP's objects live as long as the GSP: its RM frees none
    ([Rig_nv.rm]'s [free] is [None]). A device of this path keeps the
    channels a growth replaces until it stops ({!Rig_nv.section-submitting}):
    at most 26 sets, each with its rings, the instance blocks and method
    buffers of its two channels, and the compute engine's context buffers,
    in the GPU's memory. *)

(** {1:gpus GPUs}

    GPU [i] here is GPU [i] ({!Rig_nv.is_gpu}) of a machine. {!buses},
    {!count} and {!reset} also reach another machine through a
    {!Rig_pci.Machine.t} transport, which an open refuses. Listing GPUs
    changes nothing on the machine. *)

val buses : ?machine:Rig_pci.Machine.t -> unit -> string list
(** [buses ~machine ()] is the bus addresses of the NVIDIA GPUs of
    [machine] (defaults to {!Rig_pci.Machine.this}), in bus order
    ({!Rig_pci.Machine.val-functions}): GPU [i] is the [i]th. It is [[]] where
    the machine lists no PCI functions, as off Linux. *)

val count : ?machine:Rig_pci.Machine.t -> unit -> int
(** [count ~machine ()] is the number of {!buses}. *)

(** {1:firmware Firmware}

    A boot loads three files of linux-firmware's [nvidia/] directory at
    commit [0a6871b1], each pinned by its BLAKE2b-256 digest: the GSP's
    firmware [nvidia/ga102/gsp/gsp-570.144.bin], the same for every family,
    then its bootloader and the image that starts it:
    - Ampere: [ga102/gsp/bootloader-570.144.bin] and
      [ga102/gsp/booter_load-570.144.bin];
    - Ada: the same two under [ad102];
    - Blackwell: [gb202/gsp/bootloader-570.144.bin] and
      [gb202/gsp/fmc-570.144.bin].

    They are read from directories the caller names, then from rig's
    firmware directory ({!Rig_pci.Firmware.directory}), then from
    [/lib/firmware] ({!gpu}), and this library downloads nothing. The [rig]
    command downloads them into rig's directory with [curl]:
    {v rig firmware nv v}
    Keep a directory of the caller's apart from [/lib/firmware]
    ({!Rig_pci.section-needs}). On Ampere and Ada the boot also runs the
    FWSEC ucode of the GPU's VBIOS, which it reads from the GPU. *)

(** {1:opening Opening}

    The GSP reports a channel it stopped and a fault its MMU queued, and the
    device raises them from {!Rig_nv.sleep}. An exception in a sleep, such
    as [Sys.Break], drops no report: the next sleep raises it. The GSP may
    also ask the process to run a sequence of register operations, which
    runs once, front to back: one an exception cuts short is a fault, which
    loses the GPU. Only the process uses a GPU it has booted, so the GPU's
    work waits for nobody else's, and rig loses the device once its timeline
    makes no progress for {!Rig_pci.Gpus.hang_ms} (the device's
    [hang_ms]).

    Two GPUs map each other's memory only if both are taken physically, on
    one machine, each BAR reaches all of its GPU's memory, and the bus carries
    requests between them ({!Rig_pci.Memory.reaches}).

    A device reads the GPU's timer registers through BAR 0
    ({!Rig_nv.Registers}) at open only, so the times of its profiles drift
    from the host's clock at the timer's rate error since open.

    Any domain may call any function. Opens and changes of NVIDIA GPUs run
    one at a time, so a boot delays the others. {!gpu}'s [make], {!detach},
    {!attach} and {!reset}, called from a callback inside a lock ([Rig]'s
    domains section), raise [Invalid_argument], having changed nothing
    ({!Rig_edge.section-drivers}). *)

val gpu :
  ?machine:Rig_pci.Machine.t ->
  ?firmware:string list ->
  int ->
  Rig_nv.t Rig_edge.opener
(** [gpu ~machine ~firmware i] is the opener of GPU [i] of [machine]
    (defaults to {!Rig_pci.Machine.this}), named ["NV-PCI"] for [0] and
    ["NV-PCI:i"] otherwise. Its [make] boots the GPU that is [i]th when it
    runs, reading each image from the first of the directories [firmware]
    (defaults to [[]]), then {!Rig_pci.Firmware.directory}, then
    ["/lib/firmware"], that holds it with its pinned digest
    ({!Rig_pci.Firmware.find}). Run as root, an open searches root's home.

    Each [Error] names the GPU, then the reason. [make] gives back what it
    took and leaves the GPU as the open found it if:
    - [i >= count ~machine ()], saying how many GPUs there are;
    - a kernel driver holds GPU [i], or the process holds it already;
    - its function cannot be taken, with {!Rig_pci.Function.take}'s reason;
    - [machine]'s windows are not mapped into the process, as through a
      transport;
    - its chip is outside the chips {!section-requirements} lists, naming
      it;
    - on Ampere and Ada, its VBIOS lacks a FWSEC this library runs, naming
      what it lacks;
    - an image is missing, naming the directories and the files found with
      another digest, or one found is not laid out as its format says;
    - its GSP still runs after its reset;
    - the machine refuses the addresses or the GSP's memory.

    A GPU reset first stays reset. Once the GSP's memory is taken, a
    failure loses the GPU, which its next open resets, as does a failed
    reset.

    Raises [Invalid_argument] if [i < 0]. *)

(** {1:changes Changes to the machine}

    These changes persist after the process, and each refuses a GPU this
    process holds. {!detach} and {!attach} act on this machine: they write
    its [/sys/bus/pci] files, and need [CAP_SYS_ADMIN] and write access to
    those files, which root has. Detaching or attaching another machine's
    GPUs needs a program there. *)

val detach : int -> (Rig_pci.Gpus.progress, string) result
(** [detach i] detaches GPU [i] of this machine from its kernel driver, so
    that its opener can take it, unless it is detached already
    ({!Rig_pci.Gpus.detach}). The kernel driver's users, a display among
    them, lose the GPU until {!attach} or a reboot. It refuses a GPU while a
    process holds its [/dev/nvidiaN] or nvidia-drm's DRM nodes open or
    mapped, and with nvidia-drm loaded it needs debugfs mounted at
    [/sys/kernel/debug].

    It returns [Error why] only before its first write to the machine, and
    after that write [Ok Detached] or [Ok (Part_way why)]
    ({!Rig_pci.Gpus.type-progress}). It is [Error why], changing nothing, if
    there is no GPU [i], if this process holds it over PCI, if a process
    holds a file of its devices, naming it, or as {!Rig_pci.Gpus.detach}.

    Raises [Invalid_argument] if [i < 0]. *)

val attach : int -> (unit, string) result
(** [attach i] gives GPU [i] of this machine back to its kernel driver,
    resetting it first as {!reset} does if no kernel driver has it
    ({!Rig_pci.Gpus.attach}).

    [Error why] if there is no GPU [i], if this process holds it, or as
    {!Rig_pci.Gpus.attach}.

    Raises [Invalid_argument] if [i < 0]. *)

val reset : ?machine:Rig_pci.Machine.t -> int -> (unit, string) result
(** [reset ~machine i] resets GPU [i] of [machine] (defaults to
    {!Rig_pci.Machine.this}), which ends what an earlier boot left running:
    it takes the GPU's function, turns its bus mastering off and resets the
    function ({!Rig_pci.Function.reset}).

    [Error why] if there is no GPU [i], if this process holds it, if its
    function cannot be taken, or if Linux has no reset for it or it does not
    respond after one.

    Raises [Invalid_argument] if [i < 0]. *)

(**/**)

(* Boot reports, for the suites that check, without a GPU, what an open
   loads to boot a GPU and whether it refuses the GPU for its chip, its VBIOS
   or its firmware, from what the open reads before the boot writes. Making
   one needs no GPU and writes nothing. If [report] returns [Error why], an
   open of that GPU returns [Error] with [why] after the GPU's name; if an
   image is not [found], the open returns [Error] naming it. Otherwise none
   of the open's refusals is about the chip, the VBIOS or the firmware. *)

type image = {
  file : string;
      (** The file's path under a firmware directory, such as
          ["nvidia/ad102/gsp/booter_load-570.144.bin"]. *)
  found : string option;
      (** The path of the file with its pinned digest, in the first directory
          that holds one, if any does. *)
}
(** The type for a firmware file a boot loads. *)

type report = {
  chip : string;  (** The chip's name, such as ["AD102"]. *)
  images : image list;
      (** The GSP's firmware, its bootloader, then the booter or the FMC. *)
}
(** The type for boot reports. *)

val report :
  firmware:string list -> chip:int -> vbios:string -> (report, string) result
(** [report ~firmware ~chip ~vbios] is what an open loads to boot a GPU
    whose [NV_PMC_BOOT_42] register holds [chip] and whose VBIOS is [vbios]:
    its chip and its firmware files, each looked up in the directories
    [firmware] as {!gpu} looks one up in its own. It reads files and nothing
    else.
    [vbios] is the first MiB of the GPU's ROM read through [NV_PROM_DATA]
    (BAR 0 from [0x300000]), as an open reads it. On Blackwell, [report]
    ignores [vbios].

    [Error why] if [chip] is none of the chips this library boots, naming
    it; on Ampere and Ada, if [vbios] lacks a FWSEC this library runs,
    naming what it lacks; or if a file found with its pinned digest is not
    laid out as its format says. *)

(* The GSP's firmware, the same for every family: its path under a firmware
   directory and its pinned digest, for the suites that look for it as an
   open does. *)
val gsp_firmware : file:string * digest:string

(* The GSP's formats and the falcons' register accesses, for the suites that
   check them. *)
module Rpc = Rpc
module Fb_layout = Fb_layout
module Page_entry = Page_entry

module Falcon : sig
  type cond = Falcon.cond = Is of int | Not of int | Differs of int

  type op = Falcon.op =
    | Write of int * int
    | Modify of int * int * int
    | Copy of int * int
    | Poll of string * int * int * cond
    | Delay of int
    | When of int * int * cond * op list
    | Expect of string * int * int * cond

  val gsp : int
  val sec2 : int
  val reset : int -> [ `Falcon | `Riscv ] -> op list
  val start : int -> op list
  val wait_halt : int -> op list
end
