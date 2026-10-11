(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The registers of the GPU's SoC that set up its doorbells, the dummy read
    of its interrupt handler and its HDP's memory power gating.

    On NBIO 7.9 (GC 9.4.3 and 9.5.0) this module programs each live
    accelerator die through the indirect window ({!Regs.set_pcie}). Calls
    for one GPU run one at a time. *)

val nbio79 : Regs.t -> bool
(** [nbio79 r] is [true] iff the GPU's NBIO is 7.9, which routes doorbells
    per accelerator die. *)

val start : Regs.t -> unit
(** [start r] enables the doorbell aperture. Before that, on NBIO 7.9, it
    fences the doorbells of the dies fused off and grants the physical
    function doorbell access, and on other NBIOs it clears the soft reset of
    the third function's strap. On a physical function it then remaps the
    HDP flush registers to {!hdp_flush}. *)

val hdp_flush : Regs.t -> int
(** [hdp_flush r] is the byte offset in the register BAR of the register
    whose store flushes the HDP, the first word of the register hole, once
    {!start} has remapped it there. *)

val interrupts : Regs.t -> dummy:int -> unit
(** [interrupts r ~dummy] makes the page of system memory at bus address
    [dummy] the one the interrupt handler reads before each interrupt it
    signals without MSI. *)

val route :
  ?aid:int ->
  ?offset:int ->
  ?size:int ->
  Regs.t ->
  port:int ->
  awid:int ->
  awaddr:int ->
  unit
(** [route ~aid ~offset ~size r ~port ~awid ~awaddr] routes doorbell port
    [port] of die [aid] (defaults to [0]) to the engine of write ID [awid] at
    address bits [awaddr], for [size] doorbells from [offset]. [size]
    defaults to [0], which the hardware reads as all of the port's
    doorbells, and [offset] to [0]. *)

val gate : Regs.t -> unit
(** [gate r] enables the HDP's memory power gating from HDP 5.2.1 on, and
    does nothing on earlier HDPs. *)
