(** Kernel optimisations, the settings they run under, the renderers they
    target, and what an optimised kernel must still write.

    The readers take golden cells as tinygrad prints them. *)

open Tolk

(** {1:opts Optimisations} *)

val opt : Opt.t Windtrap.testable
(** [opt] compares optimisations structurally, orders them by {!Opt.compare} and
    prints them as tinygrad does ({!Opt.pp}). *)

val opt_of_cell : string -> Opt.t
(** [opt_of_cell s] is the optimisation tinygrad prints as [s]:
    [Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST))].

    Raises [Failure] naming [s] if it is not one. *)

val opts_of_cell : string -> Opt.t list
(** [opts_of_cell s] is the optimisations of the tuple tinygrad prints as [s]:
    [()], [(o,)] or [(o0, o1)], each [o] as {!opt_of_cell} reads it.

    Raises [Failure] naming [s] if it is not one. *)

(** {1:settings Settings} *)

val settings_of_cell : string -> Setting.binding list
(** [settings_of_cell s] binds the settings of [s], space-separated [NAME=value]
    pairs such as [TC=2 ALLOW_TF32=1], which name the environment variables of
    {!Setting.use_tc}, {!Setting.tc_opt}, {!Setting.tc_select},
    {!Setting.tc_min_globals}, {!Setting.allow_tf32}, {!Setting.noopt},
    {!Setting.emulated_dtypes}, {!Setting.disable_fast_idiv} and
    {!Setting.transcendental}. The empty cell binds nothing.

    Raises [Failure] naming a pair that is none of these. *)

(** {1:renderers Renderers} *)

val renderer_of_row : (string -> string) -> Renderer.t
(** [renderer_of_row cell] is the renderer of a row with the columns [device],
    [arch], [has_local], [has_shared] and [shared_max], and the tensor cores of
    its device and architecture: {!Tc.metal} on [METAL], [Tc.cuda arch] on
    [CUDA] and [NV], and [Tc.amd arch] on [AMD], none elsewhere.

    Raises [Failure] if their number is not the row's [tensor_cores]. *)

(** {1:writes Writes} *)

val inputs : Ops.t -> (int * Dtype.value array) list
(** [inputs k] is the storage of [k]'s parameters, by slot: small integers,
    exact in every data type, so that a sum of products of them is computed
    exactly whatever its order. *)

val variables : Ops.t -> (string * int) list
(** [variables k] binds each of [k]'s variables to its greatest value. *)

val writes : Ops.t -> (int * int * Dtype.value) list
(** [writes k] is what the kernel [k] writes ({!Interpreter.writes}) from its
    {!inputs}, its {!variables} bound. *)

val write : (int * int * Dtype.value) Windtrap.testable
(** [write] compares writes ({!Interpreter.writes}) by slot, index and value,
    floats up to a relative [1e-5], which reassociating a floating point sum
    stays within. *)
