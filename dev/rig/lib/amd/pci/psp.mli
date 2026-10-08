(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The GPU's security processor (MP0).

    Its bootloader starts the processor's own OS from the SOS components; the OS
    then loads every other block's firmware, which the process hands it through
    a ring in the GPU's memory, and keeps the trusted memory region (TMR) the
    firmware runs from. A command is a 1024-byte buffer the ring's 64-byte
    frames point to; the processor answers in the same buffer, and writes each
    frame's fence value once it has run it ([psp_gfx_if.h]).

    The GPU's owner serializes calls. *)

(** {1:commands Commands}

    Each is a command buffer, its addresses as the GPU's memory controller gives
    them. Pure. *)

val load_ip_fw : at:int -> bytes:int -> fw_type:int -> string
(** [load_ip_fw ~at ~bytes ~fw_type] has the processor load the [bytes] bytes at
    [at] as firmware of type [fw_type]. *)

val load_toc : at:int -> bytes:int -> string
(** [load_toc ~at ~bytes] has the processor read the table of contents of the
    [bytes] bytes at [at], and answer the TMR's size. *)

val setup_tmr : at:int -> fabric:int -> bytes:int -> string
(** [setup_tmr ~at ~fabric ~bytes] gives the processor the TMR of [bytes] bytes
    at [at], at [fabric] in the fabric's addresses; [bytes] [0] has it use the
    TMR it set up at boot. The processor is told both addresses. *)

val autoload_rlc : string
(** [autoload_rlc] tells the processor every firmware is loaded, so that the RLC
    loads the GC's engines. *)

val partition : mode:int -> string
(** [partition ~mode] sets the compute partition mode of a GPU of several dies.
*)

val frame : command:int -> fence:int -> value:int -> string
(** [frame ~command ~fence ~value] is the ring's frame for the command buffer at
    [command]: the processor writes [value] to [fence] once it has run it. *)

val status : string -> int
(** [status b] is the status the processor answered in command buffer [b], [0]
    for success. *)

val tmr_bytes : string -> int
(** [tmr_bytes b] is the TMR size the processor answered a {!load_toc} with. *)

(** {1:processor The processor} *)

type t
(** The type for security processors. *)

type memory = {
  message : int;  (** 1 MiB, aligned on 1 MiB: what a command loads. *)
  command : int;  (** The command buffer. *)
  fence : int;  (** The fence the ring writes. *)
  ring : int;  (** The ring of frames, 64 KiB. *)
}
(** The type for the GPU memory the processor works in, laid out by [Boot] in
    the boot pool, so that a partial boot finds it where the last one left it.
    Addresses are physical. *)

val make :
  Regs.t -> Gmc.t -> Rig_pci.Window.t -> Rig_pci.Page_table.t -> memory -> t
(** [make r gmc vram tables m] is the processor, whose memory is [m]. *)

val alive : t -> bool
(** [alive p] is [true] iff the processor's OS runs. *)

val start : t -> Images.t -> partial:bool -> unit
(** [start p images ~partial] starts the processor and loads [images]: a full
    boot loads the SOS components, makes the ring, sets up the TMR and loads
    every piece; a partial boot finds the TMR the last boot left, whose size it
    reads from a scratch register. In both the TMR is the first runtime
    allocation of the GPU's memory, so it keeps its address. Raises
    {!Regs.Stuck} if the processor does not answer, or answers a command with an
    error status, naming the command and the status. *)

val tmr : t -> int
(** [tmr p] is the TMR's size in bytes. *)

val set_partition : t -> int -> unit
(** [set_partition p mode] runs {!partition}. *)
