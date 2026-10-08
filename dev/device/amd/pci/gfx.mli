(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The GPU's GC: its compute micro-engines (MEC) and their hardware queue
   descriptors (MQD), the doorbells that wake them, the KIQ through which a
   virtual function asks for what the host keeps, clock gating, and which
   work-group processors run work.

   The GPU's owner serializes calls. *)

type t

(* [make r gmc vram images ~mqds] is the GC, whose queue descriptors are at the
   physical addresses [mqds] of the GPU's memory, one page per die each: the
   compute queue's, the KIQ's on a virtual function. *)
val make :
  Regs.t -> Gmc.t -> Device_pci.Window.t -> Images.t -> mqds:int array -> t

(* [start g tables ~partial] waits for the RLC's firmware, programs the GC's hub
   over [tables] and starts the micro-engines: a full boot configures them and
   their doorbell ranges, and a virtual function its KIQ; a partial boot
   dequeues the queues the last boot left and resets the engines. *)
val start : t -> Device_pci.Page_table.t -> partial:bool -> unit

(* [queue g kind ~ring ~bytes ~read ~write ~eop] programs compute queue 0 of
   [kind]: on every die for AQL, on the first for PM4. Its ring is the [bytes]
   bytes at the GPU address [ring], its read position written to [read], its
   write position polled at [write], its end-of-pipe buffer at [eop]. It is the
   queue's doorbell index. *)
val queue :
  t ->
  [ `Pm4 | `Aql ] ->
  ring:int ->
  bytes:int ->
  read:int ->
  write:int ->
  eop:int ->
  int

(* [dequeue g ~wait] removes every queue it programmed. With [wait], it waits
   for each to leave, and is [false] if one did not: a wave the reset could not
   stop holds it. *)
val dequeue : t -> wait:bool -> bool

(* [halt g] halts the micro-engines. *)
val halt : t -> unit

(* [gate g] enables the GC's clock gating, under the RLC's safe mode. *)
val gate : t -> unit

(* [wgps g] is the work-group processors that run work, as Device_amd.path's
   [wgps]: those the GPU's configuration does not mark inactive, read per shader
   engine and array. *)
val wgps : t -> int array array

(* [invalidate g] asks the KIQ of each die to invalidate both hubs' TLBs. On a
   virtual function only. *)
val invalidate : t -> unit
