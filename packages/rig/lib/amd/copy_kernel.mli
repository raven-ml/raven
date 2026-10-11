(** The driver's private byte-copy code objects, compiled from
    [kernel/copy.cl]. Their kernel [copy_bytes] uses no private or group
    memory, takes 24 bytes of arguments and runs groups of 256 work-items. *)

val find : string -> string option
(** [find processor] is the object compatible with [processor], as LLVM names
    it, or [None] when copies use the copy engine. *)
