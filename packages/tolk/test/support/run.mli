(** Running kernels on the host, or on a device, through the engine.

    Kernels are compiled for the host's target
    ({!Tolk_engine.target}[ Rig.host]), or the device's, and run on buffers
    made from arrays of values, whose contents after the run are read back. *)

open Tolk

val program : Renderer.t -> Ops.t list -> Ops.t
(** [program r uops] is the linearized kernel [uops] rendered with [r] and
    compiled with [r]'s compiler ({!Codegen.compile}).

    Raises {!Renderer.Compiler.Compile_error} if the compiler rejects the
    source. *)

val on_host :
  ?vars:(string * int) list ->
  Ops.t ->
  (int * Dtype.value array) list ->
  (int * Dtype.value array) list
(** [on_host ~vars prg buffers] runs the compiled program [prg] once on the host
    ({!Tolk_engine.link_call}) and is the contents of each of its buffer
    parameters after the run, by slot, in the order [prg] declares them. The
    buffer of slot [s] starts as the array [buffers] gives [s], or as zeros if
    it gives none, and lies at its parameter's phase ({!Ops.type-param_arg}).
    A variable of [prg] is bound by [vars] to the value of its name.

    Raises [Invalid_argument] if an array's length is not its buffer's size, and
    as {!Tolk_engine.link_call} and {!Tolk_engine.run} do. *)

val on :
  ?vars:(string * int) list ->
  Rig.t ->
  Ops.t ->
  (int * Dtype.value array) list ->
  (int * Dtype.value array) list
(** [on ~vars d prg buffers] is {!on_host} on the device [d]: [prg], compiled
    for [d]'s target ({!Tolk_engine.target}), runs once on buffers of [d]. *)

(** {1:buffers Buffers of values} *)

val buffer : Rig.t -> Dtype.t -> Dtype.value array -> Rig.Buffer.t
(** [buffer d dt values] is a new buffer of [d] holding [values], elements of
    type [dt]; one byte if [values] is empty. *)

val values : Dtype.t -> Rig.Buffer.t -> Dtype.value array
(** [values dt b] is the elements of type [dt] that [b] holds, as many as fit,
    once the work that writes them is done ({!Rig.Buffer.blit_to_bytes}). *)

(** {1:devices Test devices} *)

val devices : unit -> (string * Rig.t) list
(** [devices ()] maps ["CPU"] to {!Rig.host}, and ["CPU:1"], ["CPU:2"] and
    ["CPU:3"] to software devices ({!Rig.software}), which run the host's
    code. The software devices are opened by the first call. *)
