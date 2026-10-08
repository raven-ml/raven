(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** An AMD GPU this process booted over PCI: its BARs, its blocks brought up,
    its page tables, its memory, and the session mark it leaves for the next
    open.

    {v
    take the function
      -> read: BARs, discovery, blocks supported, mark, firmware   (state unchanged)
      -> ASPM off, VF access, page tables, boot pool
      -> full boot:    MM hub, IH, PSP and its firmware, SMU
         partial boot: the TMR the last boot left, MEC reset
      -> GC hub and MEC, SDMA, clocks, gating, mark
    v}

    A full boot needs a GPU that runs no firmware; a partial one a GPU that
    carries this library's clean mark. A boot marks the GPU dirty before it
    touches a block, so one that fails midway is booted fully next time.

    One mutex per GPU serializes the register sequences and page-table edits of
    every domain; [sleep] waits for interrupts outside it. *)

type t

(* [space] is the GPU addresses every GPU this library boots shares, in the
   process and on this machine, which reserves them before the first take: 2^44
   bytes from 0x2000_0000_0000. *)
val space : Device_pci.Space.t

(* [start f find] boots the GPU of [f], whose function the caller took and whose
   machine has [space] reserved, its firmware read with [find]: a partial boot
   if the GPU carries the clean mark, else a full one. [Error msg] if the GPU
   has a block of a version this library does not boot (Discovery.supported), if
   it runs firmware without the mark, saying that a reset stops it, if it is in
   a fabric left running, if firmware is missing, or if a block does not answer,
   naming the step. Nothing is written to the GPU before its firmware is read
   and its state checked. *)
val start :
  Device_pci.Function.t ->
  (string -> digest:string -> (string, string) result) ->
  (t, string) result

(* [reset f] resets the GPU of [f] as Device_amd_pci.reset states. *)
val reset : Device_pci.Function.t -> (unit, string) result
val gpu : t -> Device_amd_abi.Gpu.t
val gc : t -> Discovery.gc
val mec : t -> int
val wgps : t -> int array array

(* [memory g] is the GPU's memory, placed by kind. *)
val memory : t -> Device_pci.Memory.t

(* [hive g] is Gmc.hive. *)
val hive : t -> bool

(* [queue g kind ~ring ~bytes ~read ~write] is Device_amd.path's [queue]: the
   end-of-pipe buffer of a compute queue is the GPU's own memory, freed by
   [stop]. *)
val queue :
  t ->
  [ `Pm4 | `Aql | `Sdma ] ->
  ring:int ->
  bytes:int ->
  read:int ->
  write:int ->
  (int, string) result

(* [hdp g] is the host address of the HDP flush register, if the register BAR is
   mapped into the process. *)
val hdp : t -> int option

(* [sleep g ~ms] is Device_amd.path's [sleep]: it waits at most [ms]
   milliseconds for an interrupt, or for the interrupt ring to move where the
   function has no interrupts, then reads the ring. It raises Device_amd.Fault
   with every report of a fault, and with the function's or machine's failure.
   Once raised, every later sleep raises the same report. *)
val sleep : t -> ms:int -> unit

(* [stop g] stops the GPU's queues and leaves it for the next open, clocks
   lowered: - [`Clean] if every queue left and no fault was reported: the mark
   is clean, and the next open boots partially; - [`Lost] if a fault was
   reported or a queue did not leave, whose GPU then loses its bus mastering and
   reaches no memory outside its own: the mark is dirty, and only a reset
   recovers it; - [`Unknown] if its machine failed, or a queue of a GPU in a
   fabric did not leave, whose writes to its peers bus mastering does not stop.
   [`Clean] and [`Lost] answer Device_amd's [`Stopped]. *)
val stop : t -> [ `Clean | `Lost | `Unknown ]
