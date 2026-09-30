(** Running kernels on the host.

    A kernel rendered for the host is a C function of its parameters, in order:
    a pointer for each buffer and a value for each variable. The host runs a
    program as [void f(void **buffers, const int64_t *values)], so the source is
    compiled with an entry of that form that passes each buffer and each
    variable on to the kernel. *)

open Tolk_next

val target : Helpers.Target.t
(** [target] is the host's processor as the Clang renderer's target: device
    ["CPU"], renderer ["CLANG"] and architecture ["x86_64,native"] or
    ["arm64,native"]. *)

type t
(** The type for kernels loaded on the host. *)

val load : Renderer.t -> Ops.t list -> t
(** [load r uops] is the linearized kernel [uops] rendered with [r], compiled
    with [r]'s compiler and loaded on the host.

    Raises {!Renderer.Compiler.Compile_error} if the compiler rejects the
    source, and [Invalid_argument] if the host cannot load the program. *)

val run :
  ?vars:(string * int) list ->
  t ->
  (int * Dtype.value array) list ->
  (int * Dtype.value array) list
(** [run ~vars k buffers] runs [k] once and is the contents of each of its
    buffer parameters after the run, by slot, in the order [k] declares them.
    The buffer of slot [s] starts as the array [buffers] gives [s], or as zeros
    if it gives none, and each variable is the value [vars] binds to its name,
    or else its bound value ([vars] defaults to [[]]).

    Raises [Invalid_argument] if an array's length is not its buffer's size, or
    if a variable is bound neither by [vars] nor by its parameter. *)
