(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** GPUs driven through their PCI function.

    A driver that drives a GPU without a kernel driver does a kernel driver's
    work. It finds the GPU among the PCI functions of a {e machine}
    ({!Machine}), takes its function ({!Function}), reads and writes its
    registers and memory through {e windows} ({!Window}), and gives it system
    memory it reaches by DMA. It manages the GPU's virtual addresses ({!Space}),
    physical memory and page tables ({!Page_table}), and places what it
    allocates ({!Memory}). It loads the firmware the GPU was validated with
    ({!Firmware}). {!Gpus} numbers a vendor's GPUs in bus order and keeps the
    process's hold on them, and changes the machine for them.

    A machine is this one or another one that a transport reaches; the same code
    drives a GPU on either. A driver's C code accesses windows through
    [device_pci.h], so a submission stays one section of C on every machine.

    {1:errors Errors}

    - {e Requests} return [Error why] when the world refuses them: taking a
      function ({!Function.take}), opening and changing GPUs ({!Gpus}), finding
      and fetching firmware ({!Firmware}), mapping memory for a GPU
      ({!Memory.map_host}, {!Memory.map_peer}). The caller decides there.
    - {e Accesses and waits} on what the process holds raise [Failure] when the
      world fails under them: a device that stops answering, a machine whose
      transport failed, a limit of the machine reached while allocating. A
      driver turns these into its own errors once, where it can act: its open
      into an [Error], its wait into a fault. A mapped access cannot fail, so
      nothing is wrapped per access.
    - {e Misuse} raises [Invalid_argument]: an index below zero, bytes outside a
      window, memory given back twice.

    A message names what failed and, where something grants it, what to run or
    which privilege or setting is missing.

    {1:platforms Platforms}

    The library builds everywhere. This machine's PCI functions need Linux;
    elsewhere this machine has none, and another machine's are reached through
    its transport. *)

module Window = Window
module Machine = Machine
module Function = Function
module Space = Space
module Page_table = Page_table
module Memory = Memory
module Firmware = Firmware
module Gpus = Gpus
