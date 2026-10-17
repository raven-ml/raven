(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** GPUs driven through their PCI function.

    A driver that drives a GPU without a kernel driver does that driver's work
    from the process. It takes the GPU's PCI function, reads and writes its
    registers and memory, gives it system memory, writes its page tables and
    loads its firmware. This library is what such a driver needs below it. It
    knows no vendor: each driver gives its vendor's GPUs, page-table format and
    reset. It is the expert layer under [Rig_amd_pci] and [Rig_nv_pci], public
    for the driver of another GPU; a program opens a GPU through those.

    The library has six types:
    - A {e machine} ({!Machine.t}) is a computer whose PCI functions the process
      reaches, either this one or another one that a {e transport} reaches. A
      transport is a library that runs this library's operations on another
      computer. The same driver code drives a GPU on either.
    - A {e function} ({!Function.t}) is a PCI function of a machine that the
      process has taken: its configuration space, BARs, interrupts and reset,
      and the system memory it reaches by DMA.
    - A {e window} ({!Window.t}) is a range of a machine's addresses that the
      process reads and writes, such as a function's BAR or system memory
      allocated for it. A driver's C code accesses windows through
      [rig_pci.h], so a submission is the same C on every machine.
    - A {e space} ({!Space.t}) is the range of virtual addresses that the GPUs
      of one vendor share. An address in it means the same memory on each of
      them and, for system memory, in the process.
    - A GPU's {e page tables} ({!Page_table.t}) map the space onto the GPU's
      physical memory.
    - A GPU's {e memory} ({!Memory.t}) is what the GPU addresses through its
      page tables, placed by kind.

    {v
    Machine.t --take--> Function.t --map, alloc_dma--> Window.t <-- rig_pci.h
                             |
    Space.t --> Page_table.t --> Memory.t --alloc--> Memory.region
    v}

    A driver's open of a GPU usually goes like this:
    + {!Gpus.open_} holds GPU [i] of a machine, takes its function and calls
      the driver's [f], which starts the GPU.
    + [f] reserves its vendor's space on the machine
      ({!Machine.val-reserve}), maps the function's BARs ({!Function.map}) and
      boots the GPU through them, with the images {!Firmware.find} verified.
      It makes the GPU's page tables over the space ({!Page_table.create}) and
      the GPU's {!Memory} over them, from which it allocates its rings and its
      users' buffers.
    + Before the GPU can be left running, [f] gives the hold the GPU's stop
      ({!Gpus.set_stop}).
    + The driver submits work through windows.
    + {!Gpus.stop} ends the GPU's work and releases its function. The
      process's exit does this for each GPU it still holds.

    The modules below are in reading order, from the hardware up; each builds
    on the ones before it.

    {1:needs What an open needs}

    A {e PCI function} is one device on a machine's PCI bus, named by its bus
    address, such as [0000:03:00.0]. A GPU is one function, beside others of
    its card such as its audio function. GPU [i] of a machine is the [i]th GPU
    of its vendor in bus order, whichever driver holds it ({!Gpus}).

    An open of a GPU with no kernel driver needs:
    - The GPU detached from its kernel driver, once, outside the program
      ({!Gpus.detach}). {!Gpus.section-changes} states the privileges this
      needs and what persists.
    - A function the process can take, behind an IOMMU or physically, as the
      machine's state decides ({!Function}).
    - For a function taken physically, the free huge pages its system memory
      lies in ({{!Function.dma}system memory}).
    - The firmware images the driver boots the GPU with, verified by digest
      ({!Firmware}), in rig's firmware directory ({!Firmware.directory}) or
      in directories the driver's caller names. An open downloads nothing,
      and the [rig firmware] command fetches the pinned images into rig's
      directory. Keep a directory of the caller's apart from [/lib/firmware]:
      the kernel's drivers load their images from there, and an image
      replaced there changes what they load.

    {!Gpus} defines the states a GPU then passes through: detached, held and
    lost. Each driver states what its next open does in each.

    {1:safety What a function can reach}

    A function reaches system memory by DMA once its bus mastering is on.
    Behind an IOMMU it reaches only the memory the process mapped for it. Taken
    physically, nothing confines it. It can read and write any physical memory
    of its machine, the kernel's included. It keeps writing after its process
    dies by a signal or a crash, so the library gives it memory that outlives
    the process.
    {!Function} states both ways, the privileges each needs, and what stays
    after the process. {!Gpus} states the changes to the machine that
    persist.

    {!Function} also states when a function's access ends, after its
    release, at the process's exit and after its death, and
    {{!Function.dma}system memory} which pages the kernel can still move under
    a function taken physically. An IOMMU is the only protection from a wrong
    address in a function's work and from a page the kernel moves, so a
    machine whose GPU is taken physically trusts that GPU's work with all of
    its memory.

    {1:errors Errors}

    - A {e request} is a call that the system or the machine may refuse:
      taking a function, mapping a BAR, allocating or pinning system memory,
      resetting a function, reserving addresses, opening or changing GPUs,
      finding firmware, mapping memory for a GPU. A request returns
      [Error why].
    - An allocator that runs out of GPU memory or virtual addresses for what was
      asked returns [None].
    - An {e access} never fails. A function that has left the bus reads as all
      ones and drops writes, and nothing tells the process. Through a machine
      whose transport has failed, accesses do the same. Failure is state, which
      {!Machine.failed} reports for a machine and {!Function.failed} for a
      function. {!Function.wait} returns [Error] once its machine has failed,
      and notices a function that has left the bus only at its bound.
    - {e Misuse} raises [Invalid_argument], such as an index below zero, bytes
      outside a window, or memory given back twice.
    - The library's locks count among the locks a thread holds ([rig_hold] in
      [rig_edge.h]), so a call of rig's or of a driver's from a callback
      inside one raises [Invalid_argument] as it starts ([Rig]'s domains
      section). A call of this library's from such a callback raises
      [Invalid_argument] where it would take a lock its thread holds,
      without taking it. What it did before stays done, as after any
      exception there, and a {!Gpus.stop} that raised so gives its GPU back
      at the next stop.

    A message names what failed and, where something grants what is missing, the
    command, privilege or setting that grants it.

    A driver checks for failure in four places, each just before it acts on
    bytes it read:
    + Its wait for the device. {!Function.wait} returns in its [Error] a
      failure of the machine when it happens, and a failure of the function
      at the wait's bound.
    + Its submission from C. It calls [rig_pci_failed] once, after its last
      access ([rig_pci.h]).
    + A timeline word read through a window. It reads the word, then asks
      {!Function.failed}. Once the function has failed, the word keeps its
      last value.
    + A read whose bytes leave the driver, such as a copy of results. It asks
      {!Function.failed} after the read. On a mapped window this is needed only
      when the bytes hold a word of all ones.

    {1:platforms Platforms}

    The library builds everywhere. This machine's PCI functions, and the
    addresses {!Machine.val-reserve} reserves for them, need Linux. Elsewhere
    this machine has none, and another machine's functions are reached through
    its transport.

    {1:references References}

    - PCI-SIG. {e PCI Express Base Specification}, chapter 7, "Software
      Initialization and Configuration": configuration space, BARs and their
      resizing.
    - The Linux kernel's {{:https://docs.kernel.org/driver-api/vfio.html}VFIO}
      documentation: functions behind an IOMMU.
    - The Linux kernel's
      {{:https://www.kernel.org/doc/Documentation/ABI/testing/sysfs-bus-pci}
       sysfs-bus-pci}
      ABI: functions taken physically, detached and attached.
    - The Linux kernel's
      {{:https://docs.kernel.org/admin-guide/mm/pagemap.html}pagemap}
      documentation: the physical addresses of system memory. *)

(** {1:hardware Reaching the hardware} *)

module Machine = Machine
(** Machines, their PCI functions, and transports to other machines. *)

module Function = Function
(** PCI functions the process has taken. *)

module Window = Window
(** Ranges of a machine's addresses that the process reads and writes. *)

(** {1:memory A GPU's memory} *)

module Space = Space
(** Virtual addresses the GPUs of one vendor share. *)

module Page_table = Page_table
(** A GPU's physical memory and the page tables that map it. *)

module Memory = Memory
(** The memory a GPU addresses, placed by kind. *)

(** {1:gpus GPUs} *)

module Firmware = Firmware
(** Firmware images, verified by digest. *)

module Gpus = Gpus
(** A vendor's GPUs, and the process's hold on them. *)
