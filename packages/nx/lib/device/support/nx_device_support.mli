(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What GPU runtimes that drive their GPU without its kernel driver share.

    Such a runtime does a kernel driver's work: it takes the GPU's PCI function
    ({!Pci}), reads and writes its registers and memory ({!Mmio}), allocates and
    pins the system memory the GPU reaches ({!Sysmem}), manages the GPU's
    physical memory, address space and page tables ({!Tlsf}, {!Page_table},
    {!Pci_memory}), and loads the firmware it was validated with ({!Firmware}).
    Programs and firmware are laid out with [nx.device.elf].

    {b Admission.} A module belongs here iff the runtimes of several vendors
    share it and it knows nothing about any of them. A vendor's formats and
    rules, such as its page-table entries, come in as values; a module that
    names a vendor's registers, packets or firmware layouts stays in that
    vendor's library.

    {b Platforms and privileges.} The library builds everywhere; {!Pci} and
    {!Sysmem} need Linux and raise [Failure] elsewhere. Nothing escalates
    privileges: a missing privilege or kernel setting raises [Failure] naming
    what grants it. *)

module Pci = Pci
module Mmio = Mmio
module Sysmem = Sysmem
module Tlsf = Tlsf
module Page_table = Page_table
module Pci_memory = Pci_memory
module Firmware = Firmware
