(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The GPU's SoC, as the kernel's soc15.c, soc21.c and soc24.c start it: its
    doorbell aperture, the routes of its doorbells to their engines, and its
    HDP's clock gating. NBIO 7.9 (GC 9.4.3 and 9.5.0) programs each live
    accelerator die through the indirect window.

    The GPU's owner serializes calls. *)

val nbio79 : Regs.t -> bool
(** [nbio79 r] is [true] iff the GPU's NBIO is 7.9, which routes doorbells per
    accelerator die. *)

val start : Regs.t -> unit
(** [start r] enables the doorbell aperture: on NBIO 7.9, the doorbell fence of
    the dies fused off and the doorbell access of the physical function; on
    others, the soft reset of the third function's strap cleared. *)

val route :
  ?aid:int ->
  ?offset:int ->
  ?size:int ->
  Regs.t ->
  port:int ->
  awid:int ->
  awaddr:int ->
  unit
(** [route ~aid ~offset ~size r ~port ~awid ~awaddr] routes doorbell port [port]
    of die [aid] (defaults to [0]) to the engine of write ID [awid] at address
    bits [awaddr], for [size] doorbells (defaults to [0]: all) from [offset]
    (defaults to [0]). *)

val gate : Regs.t -> unit
(** [gate r] enables the HDP's memory power gating, from HDP 5.2.1. *)
