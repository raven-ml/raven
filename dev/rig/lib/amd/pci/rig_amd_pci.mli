(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** AMD GPUs booted over PCI, with no kernel driver.

    This library does the kernel driver's work from the process. It takes the
    GPU's PCI function ({!Rig_pci.Function}), loads the GPU's firmware, brings
    up its blocks, writes its page tables and reads its interrupts, then gives
    the GPU's memory, queues and interrupts to {!Rig_amd}, which drives it
    ({!Rig_amd.make}):
    {[
    let g = Result.get_ok (Rig_amd_pci.open_ ~firmware:[ "/lib/firmware" ] 0) in
    Rig_amd.arch g (* "gfx1201" *)
    ]}

    GPUs are counted and reset on this machine or on another one a transport
    reaches ({!Rig_pci.Machine}); only this machine's open.

    {b Numbering.} GPU [i] of a machine is the [i]th of its AMD GPUs in bus
    order ({!Rig_amd.is_gpu}), whichever driver holds each, so it is the same
    GPU whichever path opens it. The GPUs are named ["AMD-PCI"], ["AMD-PCI:1"],
    ["AMD-PCI:2"], ....

    {b GPUs.} The library boots GPUs whose blocks have versions it holds
    register tables and firmware for: GC 9.4.3 and 9.5.0 (the Instinct MI300 and
    MI350 series, physical and virtual functions), 11.0.0 and 11.0.2, and 12.0.0
    and 12.0.1. It reads the GPU's discovery table without changing the GPU's
    state, and refuses a GPU with a block of another version before it writes to
    it.

    {b Taking a GPU.} A GPU opens once its function can be taken: bound to
    [vfio-pci] or to no driver ({!detach}), with the privileges
    {!Rig_pci.Function.take} lists. The GPU then belongs to the process: its
    display, if any, and the kernel driver's users lose it.

    {b Host memory.} Taken without an IOMMU, a GPU reaches only the host memory
    this library allocates for it: huge pages of 2 MiB from the hugetlbfs at
    [/dev/hugepages], one for each 2 MiB block of addresses that holds the
    device's host memory, the copies the library stages through it included,
    which the system must have free ([vm.nr_hugepages]). They outlive a process
    killed while its GPU runs, until the GPU's next successful {!reset}. The
    kernel may still move them to allocate a contiguous area or to take memory
    offline, which only an IOMMU prevents ({!Rig_pci.Function.alloc_dma}). Such
    a GPU maps no other host memory: {!Rig_amd.maps_host} is [false] for its
    device, as the process's pages would go back to the system at its death with
    the GPU still writing them.

    {b Firmware.} An open reads the GPU's firmware images from the directories
    its caller names, files of linux-firmware's [amdgpu/] directory at commit
    [0a6871b1], each pinned by its BLAKE2b-256 digest; a file with another
    digest is skipped, and nothing is downloaded. They are named for the
    versions of the GPU's blocks: its security processor's [psp_X_sos.bin], its
    power manager's [smu_X.bin] and its copy engines' [sdma_X.bin], and its GC's
    [gc_X_mec.bin] and [gc_X_rlc.bin], with [gc_X_imu.bin] from GC 11 and
    [gc_X_pfp.bin] and [gc_X_me.bin] from GC 12. The R9700, for one, boots with
    [psp_14_0_3_sos.bin], [smu_14_0_3.bin], [sdma_7_0_1.bin] and
    [gc_12_0_1_{pfp,me,mec,imu,rlc}.bin]. A compressed file, such as
    [psp_13_0_0_sos.bin.zst], is another file, which the library does not read.

    {b Sessions.} An open leaves a mark on the GPU, and the process stops every
    GPU it holds at exit, leaving it marked clean: the next open, by this
    process or another, boots only the GPU's compute and copy blocks, from the
    state the last one left, which takes milliseconds where a full boot takes a
    second. A child of [fork] stops nothing at its exit. A GPU a process that
    died left running is reset by the open, as {!reset} does. One booted by its
    kernel driver opens only after a {!reset}. So does a GPU this process lost,
    to a fault or a hang, since its state is then unknown.

    {b Faults and hangs.} The GPU reports faults on its interrupt ring: page
    faults with their address, shader errors, and fatal hardware errors with its
    machine-check banks. {!Rig_amd.sleep} raises them. No kernel bounds the
    GPU's work, so this library states a bound: work that leaves the device's
    timeline word below its last value for 30 seconds is a hang, which
    {!Rig_amd.sleep} raises too.

    {b Domains.} Every value may be called from any domain. Opens, resets and
    changes to the machine of AMD GPUs run one at a time, so a boot delays the
    others.

    {b References.}
    - The Linux kernel's amdgpu driver ([drivers/gpu/drm/amd]): the discovery
      table ([discovery.h], [amdgpu_discovery.c]), firmware headers
      ([amdgpu_ucode.h]), the security processor ([psp_gfx_if.h], [psp_v13_0.c],
      [psp_v14_0.c]), the power manager ([smu_v13_0.c], [smu_v14_0.c]), page
      tables and hubs ([amdgpu_vm.h], [gmc_v9_0.c], [gmc_v11_0.c],
      [gmc_v12_0.c], [mmhub_v*.c], [gfxhub_v*.c]), compute queues
      ([gfx_v9_4_3.c], [gfx_v11_0.c], [gfx_v12_0.c], [v11_structs.h],
      [v12_structs.h]), copy engines ([sdma_v4_4_2.c], [sdma_v6_0.c],
      [sdma_v7_0.c]), interrupts ([ih_v6_0.c], [ih_v7_0.c], [amdgpu_irq.h]), and
      virtual functions ([mxgpu_nv.c], [amdgpu_virt.c]).
    - PCI-SIG. {e PCI Express Base Specification}, 7.5.3.7 "Link Control
      Register": ASPM. *)

val count : ?machine:Rig_pci.Machine.t -> unit -> int
(** [count ~machine ()] is the number of AMD GPUs of [machine] (defaults to
    {!Rig_pci.Machine.this}), whichever driver holds them: the indices [0] to
    [count ~machine () - 1]. It is [0] for a machine with no PCI functions, such
    as this one elsewhere than Linux. *)

val device_name : int -> string
(** [device_name i] is the name of GPU [i]: ["AMD-PCI"] for [0], ["AMD-PCI:i"]
    otherwise.

    Raises [Invalid_argument] if [i < 0]. *)

val open_ :
  ?machine:Rig_pci.Machine.t ->
  firmware:string list ->
  int ->
  (Rig_amd.t, string) result
(** [open_ ~machine ~firmware i] boots GPU [i] of [machine] (defaults to
    {!Rig_pci.Machine.this}) and opens it, each firmware image read from the
    first of the directories [firmware] that holds it with its pinned digest.

    The process holds the GPU until the device is stopped ({!Rig_amd.stop}); a
    device stopped after a fault or a hang leaves the GPU lost, to be {!reset}
    before it opens again.

    The result is [Error msg], the GPU left as it was, if [machine] is reached
    through a transport, if [i >= count ~machine ()], saying how many GPUs there
    are, if the process holds the GPU, if this process lost it and did not reset
    it since, if amdgpu, unbound from it, has not let go of it yet (KFD's
    topology still lists it, or its [ip_discovery] directory stays), naming the
    holder's reason, if its function cannot be taken, with
    {!Rig_pci.Function.take}'s reason, if one of its blocks has a version this
    library does not boot, naming the block and the version, if an image is
    missing, with {!Rig_pci.Firmware.find}'s reason, or if firmware this library
    did not start runs on it, which a {!reset} stops, or if the memory
    controller does not place all of the GPU's memory. It is also [Error msg] if
    a block does not answer during the boot, naming the step, or with
    {!Rig_amd.make}'s message; the GPU is then stopped, and opens again only
    after a reset. An exception raised after the boot's first write stops the
    GPU the same way and passes through.

    Raises [Invalid_argument] if [i < 0]. *)

(** {1:changes Changes to the machine}

    These change this machine and persist after the process. Each refuses a GPU
    this process holds: over PCI, or through a file of its kernel driver it has
    open, such as the GPU's render node. {!detach} and {!attach} need
    [CAP_SYS_ADMIN] and write access to the files under [/sys/bus/pci] they
    write, which root has; an [Error] for a file names it. An open of the GPU
    through its kernel driver that completes while a change runs is not held
    back by it. *)

val detach : int -> (unit, string) result
(** [detach i] takes GPU [i] of this machine from its kernel driver, so that
    {!open_} can take its function, unless no driver but [vfio-pci] holds it. It
    removes the GPU's other functions, such as its audio, and makes its memory
    BAR as large as the BAR and its bridge allow, so that the process reaches
    all of the GPU's memory. Its display and the kernel driver's users lose the
    GPU until {!attach} or a reboot.

    amdgpu lets go of the GPU, writing to it, once the last file of its DRM
    nodes goes, which its unbind does not wait for; KFD holds one for a process
    that computed on the GPU until some time after the process exits. So
    [detach] waits up to 30 s for those to go, and amdgpu then lets go inside
    the unbind: once [detach] is [Ok ()], amdgpu's last write to the GPU is
    done. It reads them from debugfs, which must be mounted at
    [/sys/kernel/debug]. After the unbind, KFD's topology no longer listing the
    GPU and its [ip_discovery] directory gone confirm that amdgpu let go: the
    release removes the first at its start and the second at its end.

    The result is [Error msg], changing nothing, if [i] is no GPU, if a process
    holds a file of its DRM nodes open or mapped, this one included, naming it,
    if KFD still holds one after 30 s, if an IOMMU translates its addresses and
    it is not bound to [vfio-pci], or if debugfs or [/proc] cannot be read. It
    is [Error msg] with the GPU detached and its memory BAR as it was if KFD's
    topology still lists the GPU or its [ip_discovery] directory stays after the
    unbind: a process opened a DRM node meanwhile, or memory of the GPU exported
    to another device or process (a dma-buf) holds it, and amdgpu lets go when
    that holder does. [detach] called again waits up to 30 s for both to go, the
    end of amdgpu's release, and finishes. A release that never comes, as after
    a failed resume of the GPU, leaves the directory for good: [detach] then
    answers [Error msg] past 30 s, and only a reboot clears it. It is
    [Error msg] if a file cannot be written, or if the GPU is still not
    detached, saying why.

    Raises [Invalid_argument] if [i < 0]. *)

val attach : int -> (unit, string) result
(** [attach i] gives GPU [i] of this machine back to its kernel driver,
    resetting it first as {!reset} does if no kernel driver has it.

    The result is [Error msg] if [i] is no GPU, if this process holds it, if its
    reset fails, the GPU then left detached, if a file cannot be written, or if
    no driver takes it.

    Raises [Invalid_argument] if [i < 0]. *)

val reset : ?machine:Rig_pci.Machine.t -> int -> (unit, string) result
(** [reset ~machine i] stops whatever runs on GPU [i] of [machine] (defaults to
    {!Rig_pci.Machine.this}) and, if its security processor's OS runs, resets it
    with the GPU's whole reset (mode 1), which clears what any driver left. Its
    bus mastering is turned off, and its engines are stopped and its clocks
    lowered first. A GPU no OS runs on is left as it is, but for its interrupt
    rings, which it turns off. A GPU of several joined by a fabric (XGMI) is
    only stopped, since such GPUs reset together, as their kernel driver does
    when it takes them back; it then opens only after that. A virtual function
    gives back its access instead, and its physical function resets it.
    Afterwards the GPU opens with a full boot.

    The result is [Error msg] if [i] is no GPU, if a process holds it, if its
    function cannot be taken, if it does not answer after the reset, in which
    case only a power cycle recovers it, if its configuration differs after it,
    if its security processor or interrupt rings still run after it, or if its
    security processor's bootloader is not ready with no OS running, which only
    a power cycle recovers.

    Raises [Invalid_argument] if [i < 0]. *)

(**/**)

(* The parts of a boot: for tests on fixture tables, synthetic images and the
   layouts amdgpu's headers state, and for bringing a GPU up one block at a
   time. *)
module Discovery = Discovery
module Regs = Regs
module Images = Images
module Gmc = Gmc
module Soc = Soc
module Sdma = Sdma
module Smu = Smu
module Psp = Psp
module Gfx = Gfx
module Boot = Boot
module Ih = Ih

(* [gpus_at root] is the bus addresses of the AMD GPUs of the machine whose
   files are under the directory [root], in bus order, as {!count} numbers
   them. *)
val gpus_at : string -> string list
