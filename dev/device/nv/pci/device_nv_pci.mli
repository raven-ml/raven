(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** NVIDIA GPUs driven over PCI, through the resource manager in their GSP.

    Opens an NVIDIA GPU that no kernel driver holds as a {!Device_nv.t}.
    {!count} and {!reset} also reach the GPUs of another machine through its
    transport ({!Device_pci.Machine.t}); {!open_} opens those of a machine the
    process reaches without one. The process takes the GPU's PCI function, boots
    the GPU's system processor (GSP) with NVIDIA's signed firmware, and then
    calls the resource manager (RM) the GSP runs, through two message queues in
    the machine's memory. The process owns the GPU while it drives it: it writes
    its page tables, places its memory and answers the GSP's requests, which a
    kernel driver does otherwise.

    {b Numbering.} GPU [i] of a machine is the [i]th of its NVIDIA GPUs
    ({!Device_nv.is_gpu}) in bus order, whichever kernel driver holds each: GPU
    [i] here is GPU [i] of {!Device_nv_nvidia}.

    {b GPUs.} The chips whose boot this library lays out open: GA102, GA103,
    GA104, GA106 and GA107 (Ampere), AD102, AD103, AD104, AD106 and AD107 (Ada),
    GB202, GB203, GB205, GB206 and GB207 (Blackwell), as the GPU's
    [NV_PMC_BOOT_42] register names them, with NVIDIA's GSP firmware 570.144.
    Another GPU answers [Error] naming its chip before anything is written to
    it.

    {b Firmware.} A boot loads the GSP's firmware, its bootloader and the image
    that starts it (the booter on Ampere and Ada, the FMC on Blackwell): the
    files of linux-firmware's [nvidia/] directory that {!pinned} lists, each
    pinned by its BLAKE2b-256 digest. They are read from directories the caller
    names; this library downloads nothing. The VBIOS's FWSEC ucode, which Ampere
    and Ada also run, is read from the GPU itself.

    {b The machine.} {!open_} changes nothing on the machine but the GPU it
    boots. A GPU becomes openable by {!detach}, which unbinds its kernel driver,
    and a GPU booted before, by its kernel driver or another process, by
    {!reset}. Both persist after the process; {!attach} gives a GPU back to its
    kernel driver.

    {b Faults and hangs.} The GSP reports the faults of the GPU's work: a
    channel it stopped, or a fault its MMU queued. The device raises them from
    its waits ({!Device_nv.sleep}). No other program shares a GPU the process
    boots, so its work never waits for theirs: work whose timeline makes no
    progress for 30 seconds is a hang, which the device raises as a fault. A
    device lost so, or stopped, is the GPU's last: the GPU then opens again only
    after {!reset}.

    {b Memory.} The memory a device gives the host is the GPU's own, through its
    memory BAR, while the BAR reaches it ([`Mapped] of {!Device_nv.alloc}), and
    the machine's system memory otherwise. GPU addresses and the process's
    addresses of system memory coincide, from [272 GiB] to [384 GiB], which the
    process reserves at its first open. System memory counts against the
    process's limit on locked memory ([RLIMIT_MEMLOCK]): once it is reached,
    {!Device_nv.alloc} answers [None], as for memory the GPU lacks, and raising
    the limit is the cure.

    {b Requirements.} Linux. Taking a GPU's function needs either an IOMMU and
    the GPU bound to [vfio-pci] with its group's file granted to the user, or
    write access to the function's files ({!Device_pci.Function.take}). Two GPUs
    map each other's memory only if both are taken physically, on one machine,
    and each BAR reaches all of its GPU's memory.

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

val count : ?machine:Device_pci.Machine.t -> unit -> int
(** [count ~machine ()] is the number of NVIDIA GPUs of [machine] (defaults to
    {!Device_pci.Machine.this}): [0] where the machine has no PCI functions, as
    off Linux. Counting changes nothing on the machine. *)

val device_name : int -> string
(** [device_name i] is the name of GPU [i]: ["NV-PCI"] for [0], ["NV-PCI:i"]
    otherwise. Every [Error] about GPU [i] starts with it.

    Raises [Invalid_argument] if [i < 0]. *)

val pinned : (string * string) list
(** [pinned] is the firmware images this library boots GPUs with: each file's
    path under a firmware directory, such as
    ["nvidia/ad102/gsp/booter_load-570.144.bin"], with its BLAKE2b-256 digest
    ({!Device_pci.Firmware.digest}). A tool that fills a firmware directory
    fetches these. *)

(** {1:opening Opening} *)

val open_ :
  ?machine:Device_pci.Machine.t ->
  firmware:string list ->
  int ->
  (Device_nv.t, string) result
(** [open_ ~machine ~firmware i] boots GPU [i] of [machine] (defaults to
    {!Device_pci.Machine.this}) and opens it. A device on a machine reached
    through a transport would need submissions through it, which {!Device_nv}
    does not make. It reads the GPU's firmware from the first of the directories
    [firmware] that holds each image with its pinned digest
    ({!Device_pci.Firmware.find}), in order.

    The process holds the GPU until the device is stopped ({!Device_nv.stop}):
    no other open of it succeeds, in this process or another, and {!detach},
    {!attach} and {!reset} refuse it.

    The result is [Error why], having given back what it took, if [machine] is
    reached through a transport, before anything is taken, if
    [i >= count ~machine ()], saying how many GPUs there are, if a kernel driver
    holds GPU [i] (naming {!detach}), if the process holds it, if its function
    cannot be taken ({!Device_pci.Function.take}'s reason), if it is no chip
    this library boots, if it was booted before or lost (naming {!reset}), if an
    image is missing (naming the directories, and the files found with another
    digest), if the machine refuses the memory or the addresses the GPU needs,
    or if a step of the boot fails, naming it. A boot that failed after it
    started the GPU leaves it to {!reset}.

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
    size the BAR and its bridge take ({!Device_pci.Gpus.detach}). The kernel
    driver's users, a display among them, lose the GPU until {!attach} or a
    reboot.

    The result is [Error why] if there is no GPU [i], if this process holds it
    over PCI, or holds a file of its kernel driver for it open ([/dev/nvidiaN],
    N its minor in [/proc/driver/nvidia/gpus/<bus>/information]), whose
    unbinding would wait for this process, or as {!Device_pci.Gpus.detach}.

    Raises [Invalid_argument] if [i < 0]. *)

val attach : int -> (unit, string) result
(** [attach i] gives GPU [i] of this machine back to its kernel driver
    ({!Device_pci.Gpus.attach}).

    The result is [Error why] if there is no GPU [i], if this process holds it,
    or as {!Device_pci.Gpus.attach}.

    Raises [Invalid_argument] if [i < 0]. *)

val reset : ?machine:Device_pci.Machine.t -> int -> (unit, string) result
(** [reset ~machine i] resets GPU [i] of [machine] (defaults to
    {!Device_pci.Machine.this}): it takes its function, turns its bus mastering
    off and resets the function ({!Device_pci.Function.reset}), which ends what
    an earlier boot left running. A GPU booted before, or lost by this process,
    opens again after it.

    The result is [Error why] if there is no GPU [i], if this process holds it,
    if its function cannot be taken, or if Linux has no reset for it or it does
    not answer after one.

    Raises [Invalid_argument] if [i < 0]. *)

(**/**)

(* The parts of a boot that read files and lay out bytes, for tests on fixtures
   of the pinned firmware and on the layouts NVIDIA's sources state. *)
module Chip = Chip
module Held = Held
module Images = Images
module Layout = Layout
module Mmu = Mmu
module Msgq = Msgq
module Vbios = Vbios
