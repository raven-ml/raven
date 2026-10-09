(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** NVIDIA GPUs driven over PCI, through the resource manager in their GSP.

    Opens an NVIDIA GPU that no kernel driver holds as a {!Rig_nv.t}. {!count}
    and {!reset} also reach the GPUs of another machine through its transport
    ({!Rig_pci.Machine.t}); {!open_} opens those of a machine the process
    reaches without one. The process takes the GPU's PCI function, boots the
    GPU's system processor (GSP) with NVIDIA's signed firmware, and then calls
    the resource manager (RM) the GSP runs, through two message queues in the
    machine's memory. The process owns the GPU while it drives it: it writes its
    page tables, places its memory and answers the GSP's requests, which a
    kernel driver does otherwise.

    {b Numbering.} GPU [i] of a machine is the [i]th of its NVIDIA GPUs
    ({!Rig_nv.is_gpu}) in bus order, whichever kernel driver holds each: GPU [i]
    here is GPU [i] of {!Rig_nv_nvidia}.

    {b GPUs.} The chips whose boot this library lays out open: GA102, GA103,
    GA104, GA106 and GA107 (Ampere), AD102, AD103, AD104, AD106 and AD107 (Ada),
    GB202, GB203, GB205, GB206 and GB207 (Blackwell), as the GPU's
    [NV_PMC_BOOT_42] register names them, with NVIDIA's GSP firmware 570.144.
    Another GPU answers [Error] naming its chip before anything is written to
    it.

    {b Firmware.} A boot loads the GSP's firmware, its bootloader and the image
    that starts it: files of linux-firmware's [nvidia/] directory at commit
    [0a6871b1], each pinned by its BLAKE2b-256 digest. Every family takes
    [nvidia/ga102/gsp/gsp-570.144.bin]; Ampere
    [ga102/gsp/bootloader-570.144.bin] and [ga102/gsp/booter_load-570.144.bin],
    Ada the same two under [ad102], and Blackwell
    [gb202/gsp/bootloader-570.144.bin] and [gb202/gsp/fmc-570.144.bin]. They are
    read from directories the caller names; this library downloads nothing. The
    VBIOS's FWSEC ucode, which Ampere and Ada also run, is read from the GPU
    itself.

    {b The machine.} {!open_} changes nothing on the machine but the GPU it
    boots. A GPU becomes openable by {!detach}, which unbinds its kernel driver
    until {!attach} gives it back, whatever process makes either.

    {b Faults and hangs.} The GSP reports the faults of the GPU's work: a
    channel it stopped, or a fault its MMU queued. The device raises them from
    its waits ({!Rig_nv.sleep}). No other program shares a GPU the process
    boots, so its work never waits for theirs: work whose timeline makes no
    progress for {!Rig_pci.Gpus.hang_ms} is a hang, which the device raises as a
    fault. A device lost so, or stopped, leaves the GSP running, and the GPU's
    next open resets it, as it does a GPU that no longer answered at the stop.

    {b Memory.} The memory a device gives the host is the GPU's own, through its
    memory BAR, while the BAR reaches it ([Mapped] of {!Rig_nv.alloc}), and the
    machine's system memory otherwise. GPU addresses and the process's addresses
    of system memory coincide, from [64 GiB] to [1 TiB], which the process
    reserves at its first open; every memory of the GPUs it opens takes its GPU
    addresses from that range. Behind an IOMMU, system memory counts against the
    process's limit on locked memory ([RLIMIT_MEMLOCK]): once it is reached,
    {!Rig_nv.alloc} answers [None], as for memory the GPU lacks, and raising the
    limit is the cure. Taken physically, it lies in huge pages of 2 MiB from the
    hugetlbfs at [/dev/hugepages], one for each 2 MiB block of addresses that
    holds some ({!Rig_pci.Function.alloc_dma}): once no huge page is free,
    {!Rig_nv.alloc} answers [None], and reserving more ([vm.nr_hugepages]) or
    freeing memory is the cure. Such a GPU maps no host memory: its device's
    [maps_host] fact is [false], as the process's pages would go back to the
    system at its death with the GPU still writing them.

    {b Requirements.} Linux. Taking a GPU's function needs either an IOMMU and
    the GPU bound to [vfio-pci] with its group's file granted to the user, or
    write access to the function's files ({!Rig_pci.Function.take}). A boot
    takes about 64 MiB of system memory for the GSP, its image chief among them:
    behind an IOMMU it counts against [RLIMIT_MEMLOCK]; taken physically, it
    takes free huge pages of 2 MiB for as much ([vm.nr_hugepages]). Its logs and
    bootloader, and on Blackwell the FMC, each lie in one run of bus addresses.
    Two GPUs map each other's memory only if both are taken physically, on one
    machine, and each BAR reaches all of its GPU's memory.

    {b Domains.} Any domain may call any function. Opens and changes of NVIDIA
    GPUs are serialized: a boot of a few seconds delays the others.

    {b References.}
    - NVIDIA's
      {{:https://github.com/NVIDIA/open-gpu-kernel-modules}
       open-gpu-kernel-modules}, release 570.144, the firmware's:
      [src/nvidia/src/kernel/gpu/gsp/] ([kernel_gsp.c], [message_queue_cpu.c],
      [kernel_gsp_fwsec.c], [kernel_gsp_booter.c], and per architecture
      [arch/turing/kernel_gsp_tu102.c], [arch/ampere/kernel_gsp_ga102.c],
      [arch/hopper/kernel_gsp_gh100.c], [arch/blackwell/kernel_gsp_gb100.c]),
      [src/nvidia/inc/kernel/vgpu/rpc_global_enums.h],
      [src/nvidia/arch/nvalloc/common/inc/gsp/gsp_fw_wpr_meta.h] and
      [rmgspseq.h], [src/common/shared/msgq/inc/msgq/msgq_priv.h],
      [src/common/uproc/os/common/include/libos_init_args.h], the registers of
      [src/common/inc/swref/published/] for [tu102], [ga102], [gh100] and
      [gb202], and
      [kernel-open/nvidia-uvm/hwref/{turing/tu102,hopper/gh100}/dev_mmu.h]
      (page-table entries).
    - Linux's nouveau driver: [include/nvfw/fw.h] and [hs.h] (the firmware
      containers), and [nvkm/subdev/mmu/vmmtu102.c] (the TLB invalidation and
      its wait).
    - The {{:https://gitlab.com/kernel-firmware/linux-firmware}linux-firmware}
      repository's [nvidia/] directory and its [LICENCE.nvidia]. *)

(** {1:gpus GPUs} *)

val buses : ?machine:Rig_pci.Machine.t -> unit -> string list
(** [buses ~machine ()] is the bus addresses of the NVIDIA GPUs of [machine]
    (defaults to {!Rig_pci.Machine.this}) ({!Rig_nv.is_gpu}), in bus order: GPU
    [i] is the [i]th, whichever kernel driver holds it. It is [[]] where the
    machine has no PCI functions, as off Linux. Listing them changes nothing on
    the machine. *)

val count : ?machine:Rig_pci.Machine.t -> unit -> int
(** [count ~machine ()] is the number of {!buses}. *)

val device_name : int -> string
(** [device_name i] is the name of GPU [i]: ["NV-PCI"] for [0], ["NV-PCI:i"]
    otherwise. Every [Error] about GPU [i] starts with it.

    Raises [Invalid_argument] if [i < 0]. *)

(** {1:reports Boot reports}

    A report says what {!open_} loads to boot a GPU, and whether it refuses the
    GPU for its chip, its VBIOS or its firmware, from what an open reads before
    it writes to it. It needs no GPU and writes nothing. *)

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
(** [report ~firmware ~chip ~vbios] is what {!open_} loads to boot a GPU whose
    [NV_PMC_BOOT_42] register holds [chip] and whose VBIOS is [vbios]: its chip
    and its firmware files, each looked up in the directories [firmware] as
    {!open_} looks it up. It reads files and nothing else.

    [vbios] is the first MiB of the GPU's ROM as its registers show it, from
    [NV_PROM_DATA] (BAR 0 from [0x300000]), which {!open_} reads. Linux's [rom]
    file of the GPU's function reads the ROM through its expansion ROM BAR
    instead, which may show less of it. Blackwell's boot reads no VBIOS, and
    [vbios] is then unread.

    {!open_} of that GPU reads what [report] reads before it writes to the GPU,
    but for the reset it gives a GPU whose GSP runs, that this process lost or
    that a dead process left. If [report] answers [Error why], {!open_} answers
    [Error] with [why] after the GPU's name; if an image is not [found], [Error]
    naming it. Otherwise no refusal of {!open_} is about the chip, the VBIOS or
    the firmware.

    [Error why] if [chip] is no chip this library boots, naming it; on Ampere
    and Ada, if [vbios] holds no FWSEC this library runs, naming what it lacks;
    or if a file found with its pinned digest is not laid out as its format
    says. *)

(** {1:opening Opening} *)

val open_ :
  ?machine:Rig_pci.Machine.t ->
  firmware:string list ->
  int ->
  (Rig_nv.t, string) result
(** [open_ ~machine ~firmware i] boots GPU [i] of [machine] (defaults to
    {!Rig_pci.Machine.this}) and opens it. It reads the GPU's firmware from the
    first of the directories [firmware] that holds each image with its pinned
    digest ({!Rig_pci.Firmware.find}), in order.

    The process holds the GPU until [Rig.close] ends the device or it is lost:
    no other open of it succeeds, in this process or another, and {!detach},
    {!attach} and {!reset} refuse it.

    A GPU this process lost, or whose GSP runs, as its kernel driver, a stopped
    device or a process that died leaves it, is reset as {!reset} does before
    anything is written to it: the open's take proves that no process holds it.

    The result is [Error why], having given back what it took, if
    [i >= count ~machine ()], saying how many GPUs there are, if a kernel driver
    holds GPU [i] (naming {!detach}), if the process holds it, if its function
    cannot be taken ({!Rig_pci.Function.take}'s reason), if [machine]'s windows
    are not mapped into the process, as through a transport, before anything is
    written, if its reset fails, the GPU then lost, if its GSP still runs after
    that reset, if {!report} of it answers [Error] or finds an image missing
    (naming the directories, and the files found with another digest), before
    anything but that reset is written to it, if the machine refuses the memory
    or the addresses the GPU needs, or if a step of the boot fails, naming it. A
    boot that failed after it started the GPU leaves it lost, reset by its next
    open.

    Raises [Invalid_argument] if [i < 0]. *)

(** {1:changes Changes to the machine}

    These change the machine and persist after the process. Each refuses a GPU
    this process holds. {!detach} and {!attach} write [/sys/bus/pci] of this
    machine, and need [CAP_SYS_ADMIN] and write access to the files they write,
    which root has: another machine's GPUs are changed by a program on that
    machine. *)

val detach : int -> (unit, string) result
(** [detach i] detaches GPU [i] of this machine from its kernel driver, so that
    {!open_} can take it, unless it is detached already. It unbinds the driver
    (other than [vfio-pci]), removes the GPU's other functions, such as its
    audio function, enables the GPU and makes its memory BAR (BAR 1) the largest
    size the BAR and its bridge take ({!Rig_pci.Gpus.detach}). The kernel
    driver's users, a display among them, lose the GPU until {!attach} or a
    reboot.

    It refuses a GPU a process holds a file of open or mapped ([/dev/nvidiaN], N
    its minor in [/proc/driver/nvidia/gpus/<bus>/information], and nvidia-drm's
    DRM nodes): the driver's unbind would wait for them. With nvidia-drm loaded
    it reads the DRM nodes' files from debugfs, which must be mounted at
    [/sys/kernel/debug].

    The result is [Error why] if there is no GPU [i], if this process holds it
    over PCI, if a process holds a file of its devices, naming it, or as
    {!Rig_pci.Gpus.detach}.

    Raises [Invalid_argument] if [i < 0]. *)

val attach : int -> (unit, string) result
(** [attach i] gives GPU [i] of this machine back to its kernel driver,
    resetting it first as {!reset} does if no kernel driver has it
    ({!Rig_pci.Gpus.attach}).

    The result is [Error why] if there is no GPU [i], if this process holds it,
    or as {!Rig_pci.Gpus.attach}.

    Raises [Invalid_argument] if [i < 0]. *)

val reset : ?machine:Rig_pci.Machine.t -> int -> (unit, string) result
(** [reset ~machine i] resets GPU [i] of [machine] (defaults to
    {!Rig_pci.Machine.this}): it takes its function, turns its bus mastering off
    and resets the function ({!Rig_pci.Function.reset}), which ends what an
    earlier boot left running.

    The result is [Error why] if there is no GPU [i], if this process holds it,
    if its function cannot be taken, or if Linux has no reset for it or it does
    not answer after one.

    Raises [Invalid_argument] if [i < 0]. *)
