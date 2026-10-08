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

    {b Firmware.} An open reads the GPU's firmware images from the directories
    its caller names. Each image has the BLAKE2b-256 digest this library pins
    ({!pinned}); a file with another digest is skipped, and nothing is
    downloaded. A directory of linux-firmware at the pinned commit serves every
    GPU the library boots; a compressed file, such as [psp_13_0_0_sos.bin.zst],
    is another file, which the library does not read.

    {b Sessions.} An open leaves a mark on the GPU, and the process stops every
    GPU it holds at exit, leaving it marked clean: the next open, by this
    process or another, boots only the GPU's compute and copy blocks, from the
    state the last one left, which takes milliseconds where a full boot takes a
    second. A child of [fork] stops nothing at its exit. A GPU that runs
    firmware without this mark, booted by its kernel driver or left by a process
    that died, opens only after a {!reset}. So does a GPU this process lost, to
    a fault or a hang, since its state is then unknown.

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
    it since, if its function cannot be taken, with {!Rig_pci.Function.take}'s
    reason, if one of its blocks has a version this library does not boot,
    naming the block and the version, if an image is missing, with
    {!Rig_pci.Firmware.find}'s reason, or if firmware this library did not start
    runs on it, which a {!reset} stops. It is also [Error msg] if a block does
    not answer during the boot, naming the step, or with {!Rig_amd.make}'s
    message; the GPU then opens again only after a reset.

    Raises [Invalid_argument] if [i < 0]. *)

(** {1:firmware Firmware} *)

val pinned : (string * string) list
(** [pinned] is every firmware image an open may read: its path under a firmware
    directory, such as ["amdgpu/psp_13_0_0_sos.bin"], and its lowercase
    hexadecimal BLAKE2b-256 digest ({!Rig_pci.Firmware.digest}). *)

val origin : string
(** [origin] is the URL of linux-firmware's tree at the pinned commit:
    [origin ^ path] downloads the image at [path] of {!pinned}. *)

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

    The result is [Error msg] if [i] is no GPU, if this process holds it, if a
    file cannot be written, or if the GPU is still not detached, saying why.

    Raises [Invalid_argument] if [i < 0]. *)

val attach : int -> (unit, string) result
(** [attach i] gives GPU [i] of this machine back to its kernel driver.

    The result is [Error msg] if [i] is no GPU, if this process holds it, if a
    file cannot be written, if it is bound to [vfio-pci] by its
    [driver_override], or if no driver takes it.

    Raises [Invalid_argument] if [i < 0]. *)

val reset : ?machine:Rig_pci.Machine.t -> int -> (unit, string) result
(** [reset ~machine i] stops whatever runs on GPU [i] of [machine] (defaults to
    {!Rig_pci.Machine.this}) and resets it with the GPU's whole reset (mode 1),
    which clears what any driver left. A GPU no firmware runs on is left as it
    is. A virtual function gives back its access instead, and its physical
    function resets it. Afterwards the GPU opens with a full boot.

    The result is [Error msg] if [i] is no GPU, if this process holds it, if its
    function cannot be taken, if it is one of several GPUs joined by a fabric
    (XGMI), which reset together outside the process, or if it does not answer
    after the reset, in which case only a power cycle recovers it.

    Raises [Invalid_argument] if [i < 0]. *)

(**/**)

(* The parts of a boot that read bytes and lay them out, for tests on fixture
   tables, synthetic images and the layouts amdgpu's headers state. *)
module Discovery = Discovery
module Regs = Regs
module Images = Images
module Gmc = Gmc
module Smu = Smu
module Psp = Psp
module Gfx = Gfx
module Ih = Ih

(* [gpus_at root] is the bus addresses of the AMD GPUs of the machine whose
   files are under the directory [root], in bus order, as {!count} numbers
   them. *)
val gpus_at : string -> string list
