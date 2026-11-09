(** The driver's private counter-filter code objects, compiled from
    [kernel/filter.cl]. Their kernel [filter] stores at its one argument's
    address the counter filter of its wave's VMID
    ({!Rig_amd_abi.Counter.filter}). It uses no private or group memory,
    takes 8 bytes of arguments and runs waves of 32 lanes. *)

val find : string -> string option
(** [find processor] is the object compatible with [processor], as LLVM names
    it, or [None] when no device of it measures counters. *)
