(** Shared plumbing for parity case [main.ml] scripts.

    Each case's [main.ml] builds its own [Uop.t] kernel AST and calls
    {!dump} (selecting which stages to diff) or {!dump_stage7} (stage 7
    only). *)

val global_fptr : Tolk_uop.Dtype.t
(** [float32 *] with address space [Global] and unknown size (as a
    {!Tolk_uop.Dtype.Ptr} wrapped in the unified [Dtype.t]). *)

val idx : int -> Tolk_uop.Uop.t
(** [idx n] is a constant of type [Dtype.Val.weakint] with value [n]. *)

val all_backends : (string * Tolk.Renderer.t) list
(** [(short_name, renderer)] for ["cpu"] / ["cuda"] / ["metal"] / ["opencl"]. *)

val gpu_backends : (string * Tolk.Renderer.t) list
(** {!all_backends} minus ["cpu"]. *)

(** Pipeline-stage snapshot for parity diffing. *)
type stage =
  | Stage5  (** Kernel AST after [full_rewrite_to_sink]. Columnar uop dump. *)
  | Stage7  (** Rendered backend source. *)

val mk_shape : int list -> Tolk_uop.Uop.t
(** [mk_shape dims] encodes a concrete shape as either a single index const
    (rank 1) or a [STACK] of index consts. *)

val mk_param :
  idx:int -> ?dtype:Tolk_uop.Dtype.t -> ?device:string -> int list ->
  Tolk_uop.Uop.t
(** [mk_param ~idx ?dtype ?device shape] builds a tensor-level [Param] node
    with dtype [?dtype] (default [float32]), device [?device] (default
    ["CPU"]), and the given concrete shape. *)

val mk_param_multi :
  idx:int -> ?dtype:Tolk_uop.Dtype.t -> devices:string list ->
  ?axis:int -> int list -> Tolk_uop.Uop.t
(** [mk_param_multi ~idx ~devices ?axis shape] builds a multi-device
    tensor-level [Param]. [shape] is the per-shard shape; when [axis] is
    given, the stored shape has that axis multiplied by the device count,
    mirroring the reference param constructor. *)

val wrap_sink : Tolk_uop.Uop.t list -> Tolk_uop.Uop.t
(** [wrap_sink srcs] wraps each [src] in [Contiguous] and groups them under
    a tensor-level [Sink]. *)

val stage5_tensor :
  ?optimize:bool -> Tolk.Renderer.t -> Tolk_uop.Uop.t -> string
(** [stage5_tensor ren tensor_sink] runs [tensor_sink] through
    {!Tolk.Rangeify.get_kernel_graph}, then for every [Call] node wrapping an
    inline kernel AST, runs [full_rewrite_to_sink] and dumps the resulting
    uop list. Multiple kernels are separated by [=== kernel N ===] headers. *)

val stage7_tensor :
  ?optimize:bool -> Tolk.Renderer.t -> Tolk_uop.Uop.t -> string
(** [stage7_tensor ren tensor_sink] runs [tensor_sink] through
    {!Tolk.Rangeify.get_kernel_graph}, then for every [Call] node wrapping an
    inline kernel AST, runs the full codegen + linearize + render pipeline and
    concatenates the rendered sources with ["\n---\n"] between kernels. *)

val dump_tensor :
  ?backends:(string * Tolk.Renderer.t) list ->
  ?optimize:bool ->
  stages:stage list ->
  out_dir:string ->
  Tolk_uop.Uop.t ->
  unit
(** [dump_tensor ~stages ~out_dir tensor_sink] is the tensor-graph counterpart
    of {!dump}: writes [<out_dir>/<stage>_<backend>.actual] for every
    (stage, backend) pair, running the rangeify pass first. *)

val stage5 :
  ?optimize:bool -> Tolk.Renderer.t -> Tolk_uop.Uop.t -> string

val stage7 :
  ?optimize:bool -> Tolk.Renderer.t -> Tolk_uop.Uop.t -> string

val dump :
  ?backends:(string * Tolk.Renderer.t) list ->
  ?optimize:bool ->
  stages:stage list ->
  out_dir:string ->
  Tolk_uop.Uop.t ->
  unit
(** [dump ~stages ~out_dir sink] writes
    [<out_dir>/<stage>_<backend>.actual] for every (stage, backend) pair. *)

val dump_stage7 :
  ?backends:(string * Tolk.Renderer.t) list ->
  ?optimize:bool ->
  out_dir:string ->
  Tolk_uop.Uop.t ->
  unit
(** Stage-7-only convenience wrapper around {!dump}. *)

val stage7_program :
  ?name:string -> Tolk.Renderer.t -> Tolk.Program_spec.program -> string
(** [stage7_program ren program] renders [program] directly with [ren]. No
    codegen rewrite or linearize pass is run — the program is expected to be
    a pre-linearized flat SSA form. Defaults to kernel name ["test"]. *)

val dump_stage7_program :
  ?backends:(string * Tolk.Renderer.t) list ->
  ?name:string ->
  out_dir:string ->
  Tolk.Program_spec.program ->
  unit
(** [dump_stage7_program ~out_dir program] writes
    [<out_dir>/stage7_<backend>.actual] for every backend in [backends]
    (defaults to {!all_backends}). *)
