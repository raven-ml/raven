(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The commands of a GPU's security processor (MP0), which loads every
    other block's firmware.

    Its bootloader starts the processor's own OS from the SOS components
    ({!steps}). The OS then loads the firmware it is handed through a ring
    in the GPU's memory, and keeps the trusted memory region (TMR) the
    firmware runs from. A command is a 1024-byte buffer that a 64-byte frame
    of the ring points to. Once the processor has run a command, it writes
    its response into the same buffer and the frame's fence value. Every
    value is pure, and any domain may call it. *)

(** {1:commands Commands}

    Each value is a command buffer, its addresses as the GPU's memory
    controller gives them. *)

val load_ip_fw : at:int -> bytes:int -> fw_type:int -> string
(** [load_ip_fw ~at ~bytes ~fw_type] has the processor load the [bytes]
    bytes at [at] as firmware of type [fw_type]. *)

val load_toc : at:int -> bytes:int -> string
(** [load_toc ~at ~bytes] has the processor read the table of contents of
    the [bytes] bytes at [at] and return the TMR's size. *)

val setup_tmr : at:int -> fabric:int -> bytes:int -> string
(** [setup_tmr ~at ~fabric ~bytes] gives the processor the TMR of [bytes]
    bytes at [at], which is [fabric] in the fabric's addresses. With [bytes]
    of [0] it uses the TMR it set up at boot. *)

val autoload_rlc : string
(** [autoload_rlc] tells the processor every firmware is loaded, so that the
    RLC loads the GC's engines. *)

val partition : mode:int -> string
(** [partition ~mode] sets the compute partition mode of a GPU of several
    dies. *)

val frame : command:int -> fence:int -> value:int -> string
(** [frame ~command ~fence ~value] is the ring's frame for the command buffer
    at [command]. The processor writes [value] to [fence] once it has run
    the command. *)

val status : string -> int
(** [status b] is the status the processor returned in command buffer [b],
    [0] for success. *)

val tmr_bytes : string -> int
(** [tmr_bytes b] is the TMR size the processor returned for a
    {!load_toc}. *)

(** {1:bootloader The bootloader} *)

val steps : (int * int) list
(** [steps] is the bootloader's steps, in the order it takes them: the type
    of the SOS component each loads ([PSP_FW_TYPE_*]) and the command that
    loads it ([PSP_BL__LOAD_*]). The last step loads the SOS itself. *)
