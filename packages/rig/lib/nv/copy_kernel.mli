(** The driver's private byte-copy cubin, compiled from [kernel/copy.cu]. Its
    kernel [copy_bytes] uses no stack or shared memory. *)

val find : string -> string option
(** [find arch] is the object compatible with [arch], or [None] when copies use
    the copy engine. *)
