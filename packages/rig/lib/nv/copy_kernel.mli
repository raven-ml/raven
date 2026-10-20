(** The driver's private byte-copy cubin, compiled from [kernel/copy.cu]. Its
    kernels [copy_bytes] and [fill_bytes] use no stack or shared memory. *)

val find : string -> string option
(** [find arch] is the object compatible with [arch], or [None] when copies and
    fills use the copy engine. *)
