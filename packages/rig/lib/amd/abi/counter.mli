(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Performance counters, which count events of a kernel's run.

    A counter is an event of one block of a GPU's GC, such as ["SQ_WAVES"]
    or ["GRBM_GUI_ACTIVE"], which one of the block's counter registers
    selects. A run counts it in each instance of its block, and an SQ
    counter in each shader engine and, after GFX9, in each shader array and
    work-group processor of the engine. The run writes each value as a
    64-bit word into its samples. For the counters a profile asks for,
    {!val-layout} gives the register each takes and where its values lie,
    which the program that counts writes and the reader reads.

    The counters are rocprofiler's for gfx942, gfx950, and GFX11 and GFX12
    GPUs. A GPU of another processor counts none. *)

type t = {
  name : string;  (** Its name, such as ["SQ_WAVES"]. *)
  block : string;  (** Its block: ["GRBM"], ["GL2C"], ["TCC"] or ["SQ"]. *)
  event : int;  (** The event its block's counter register selects. *)
  register : int;
      (** Which of its block's counter registers counts it. The counters of
          a block take its registers in the profile's order, from [0]. *)
  instances : int;  (** The instances of its block in one die. *)
  engines : int;  (** The shader engines it is counted in. *)
  arrays : int;  (** The shader arrays of an engine it is counted in. *)
  wgps : int;  (** The work-group processors of an array it is counted in. *)
  offset : int;  (** The byte offset of its values in a run's samples. *)
}
(** The type for a counter as a run counts it. Its values are
    [xccs * instances * engines * arrays * wgps] 64-bit words, one per die,
    instance, engine, array and work-group processor, the last varying
    fastest. *)

type layout = {
  counters : t list;  (** The counters, in the profile's order. *)
  bytes : int;  (** The bytes of a run's samples. *)
}
(** The type for a run's samples: each counter's values, one after
    another. *)

val names : Gpu.t -> string list
(** [names g] is the counters [g] counts, in increasing order. *)

val layout : Gpu.t -> string list -> (layout, string) result
(** [layout g names] is the samples of a run of [g] counting [names], in
    order. The result is [Error msg]:
    {ul
    {- if [g] counts no counter of [names], naming it, as
       ["gfx1201 counts no SQ_FOO"];}
    {- if [names] lists a counter twice, naming it, as
       ["SQ_WAVES is listed twice"];}
    {- if [names] has more counters of a block than the block has counter
       registers, naming the block, as
       ["gfx1201 counts at most 4 GL2C counters at once"].}} *)
