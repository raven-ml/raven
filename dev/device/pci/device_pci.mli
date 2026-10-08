(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** GPUs driven through their PCI function.

    A driver that drives a GPU without a kernel driver does that driver's work
    from the process: it takes the GPU's PCI function, reads and writes its
    registers and memory, gives it system memory, writes its page tables and
    loads its firmware. This library is what such a driver needs below it. It
    knows no vendor: each driver gives its vendor's GPUs, page-table format and
    reset.

    Five types carry the library:
    - A {e machine} ({!Machine.t}) is a computer whose PCI functions the process
      reaches: this one, or another one a {e transport} reaches. The same code
      drives a GPU on either.
    - A {e function} ({!Function.t}) is a PCI function of a machine that the
      process has taken: its configuration space, BARs, interrupts and reset,
      and the system memory it reaches by DMA.
    - A {e window} ({!Window.t}) is a range of a machine's addresses that the
      process reads and writes: a function's BAR, or system memory allocated for
      it. A driver's C code accesses windows through [device_pci.h], so a
      submission is one section of C on every machine.
    - A {e space} ({!Space.t}) is the virtual addresses the GPUs of one vendor
      share, so that an address means the same memory on each of them and, for
      system memory, in the process.
    - A GPU's {e memory} ({!Memory.t}) is what the GPU addresses, placed by
      kind. Its page tables ({!Page_table.t}) map the space onto it.

    {v
    Machine.t --take--> Function.t --map, alloc_dma--> Window.t <-- device_pci.h
                             |
    Space.t --> Page_table.t --> Memory.t --alloc--> Memory.region
    v}

    A driver starts at {!Gpus}, which numbers a vendor's GPUs on a machine,
    holds those the process drives, and opens one with its function taken.
    {!Firmware} finds the images a GPU boots with. The modules below are in
    reading order; each builds on the ones before it.

    {1:errors Errors}

    - A {e request} that the world may refuse returns [Error why]: taking a
      function, mapping a BAR, allocating or pinning system memory, resetting a
      function, reserving addresses, opening or changing GPUs, finding firmware,
      mapping memory for a GPU. The caller decides there.
    - An allocator that runs out of GPU memory or virtual addresses for what was
      asked answers [None].
    - An {e access} never fails. A function that left the bus answers reads with
      all ones and drops writes, and nothing tells the process; through a
      machine whose transport failed, accesses do the same. Failure is state:
      {!Machine.failed} for a machine, {!Function.failed} for a function, which
      also reads the function's vendor ID. {!Machine.wait} answers [false] once
      its machine failed.
    - {e Misuse} raises [Invalid_argument]: an index below zero, bytes outside a
      window, memory given back twice.

    A message names what failed and, where something grants what is missing, the
    command, privilege or setting that does.

    A driver owes four checks, each where it acts on what it read:
    + Its wait for the device: when the wait ends [false], {!Function.failed}
      says whether the function or its machine failed, and why.
    + Its submission from C: one call of [device_pci_failed] after its last
      access ([device_pci.h]).
    + A progress word read through a window: read it, then ask
      {!Function.failed}; once failed, the word keeps its last value.
    + A read whose bytes leave the driver, such as a copy of results: ask
      {!Function.failed} after it, which on a mapped window is needed only when
      the bytes hold a word of all ones.

    {1:platforms Platforms}

    The library builds everywhere. This machine's PCI functions, and the
    addresses {!Machine.reserve} reserves for them, need Linux; elsewhere this
    machine has none, and another machine's are reached through its transport.

    {1:references References}

    - PCI-SIG. {e PCI Express Base Specification}, chapter 7, "Software
      Initialization and Configuration": configuration space, BARs and their
      resizing.
    - The Linux kernel's {{:https://docs.kernel.org/driver-api/vfio.html}VFIO}
      documentation, and [include/uapi/linux/vfio.h]: functions behind an IOMMU.
    - The Linux kernel's
      {{:https://www.kernel.org/doc/Documentation/ABI/testing/sysfs-bus-pci}sysfs-bus-pci}
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
