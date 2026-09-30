(** Running kernels on the host through the engine.

    Kernels are compiled for the host's target
    ({!Tolk_next_engine.target}[ Nx_device.host]) and run on buffers made from
    arrays of values, whose contents after the run are read back. *)

open Tolk_next

val program : Renderer.t -> Ops.t list -> Ops.t
(** [program r uops] is the linearized kernel [uops] rendered with [r] and
    compiled with [r]'s compiler ({!Codegen.to_program}).

    Raises {!Renderer.Compiler.Compile_error} if the compiler rejects the
    source. *)

val on_host :
  ?vars:(string * int) list ->
  Ops.t ->
  (int * Dtype.value array) list ->
  (int * Dtype.value array) list
(** [on_host ~vars prg buffers] runs the compiled program [prg] once on the host
    ({!Tolk_next_engine.Program.run}) and is the contents of each of its buffer
    parameters after the run, by slot, in the order [prg] declares them. The
    buffer of slot [s] starts as the array [buffers] gives [s], or as zeros if
    it gives none.

    Raises [Invalid_argument] if an array's length is not its buffer's size, and
    as {!Tolk_next_engine.Program.load} and {!Tolk_next_engine.Program.run} do.
*)
