(** Running kernels on the host through the engine.

    Kernels are compiled for the host's target
    ({!Tolk_engine.target}[ Nx_device.host]) and run on buffers made from arrays
    of values, whose contents after the run are read back. *)

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
    ({!Tolk_engine.Program.run}) and is the contents of each of its buffer
    parameters after the run, by slot, in the order [prg] declares them. The
    buffer of slot [s] starts as the array [buffers] gives [s], or as zeros if
    it gives none.

    Raises [Invalid_argument] if an array's length is not its buffer's size, and
    as {!Tolk_engine.Program.load} and {!Tolk_engine.Program.run} do. *)

(** {1:buffers Buffers of values} *)

val buffer : Nx_device.t -> Dtype.t -> Dtype.value array -> Nx_device.Buffer.t
(** [buffer d dt values] is a new buffer of [d] holding [values], elements of
    type [dt]; one byte if [values] is empty. *)

val values : Dtype.t -> Nx_device.Buffer.t -> Dtype.value array
(** [values dt b] is the elements of type [dt] that [b] holds, as many as fit,
    once the work that writes them is done ({!Nx_device.Buffer.copy}). *)

(** {1:devices Test devices} *)

val devices : unit -> (string * Nx_device.t) list
(** [devices ()] maps ["CPU"] to {!Nx_device.host}, and ["CPU:1"], ["CPU:2"] and
    ["CPU:3"] to test devices of the host's memory
    ({!Nx_device.Driver.host_memory}) that address it as it is
    ({!Nx_device.Driver.mapping}[ Identity]), and ["CPU:4"] to one that maps
    whole pages of it ([Pages]), as a GPU does, and so borrows no host buffer of
    less than 64 KiB. Their calls run one by one here, and from command queues
    through {!Null_device}. The test devices are opened by the first call. *)
