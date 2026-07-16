(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Unified hash-consed DAG IR.

    A {!t} is a node with an operation tag ({!Ops.t}), a data type
    ({!Dtype.t}), an ordered tuple of child nodes ({!src}), a structured
    per-op payload ({!arg}), and an optional diagnostic {!node_tag}.
    Nodes are interned in a global hash-cons table, so structurally equal
    nodes are physically identical: {!equal} reduces to physical equality
    on the interned {!tag}.

    The same type flows through every stage of the pipeline — tensor graph,
    kernel AST, and linearized program. Stage membership is enforced by
    the {!Spec} validators at pass boundaries rather than by the type
    system. Per-op {!src} layouts are fixed and documented at each smart
    constructor; pattern matches should prefer the
    {{!section-viewacc}view accessors}, which encode these layouts once.

    {1:design Design}

    Users construct nodes through labelled-argument smart constructors
    and inspect them either by matching on {!op} plus {!src}/{!arg}/{!dtype},
    or by using the view accessors which return structured records for
    ops with non-trivial src/arg contracts. There is no public raw [mk]
    escape hatch; every node must be built through a dedicated smart
    constructor so that per-op invariants and src layouts stay
    centralised.

    {b Example.}

    {[
      open Tolk_uop

      let sum =
        Uop.alu_binary ~op:Ops.Add ~lhs:a
          ~rhs:(Uop.alu_binary ~op:Ops.Mul ~lhs:b ~rhs:c)

      (* Inspection via view accessors. *)
      match Uop.as_range u with
      | Some { size; axis; kind; _ } -> ...
      | None -> ...
    ]}

    {1:stages Stages}

    Smart constructor docstrings are tagged with the stage at which the
    node is legal:

    {ul
    {- {b Shared.} Valid at every stage.}
    {- {b Tensor.} Valid before scheduling.}
    {- {b Kernel.} Valid in the kernel-AST stage.}
    {- {b Program.} Valid in the linearized-program stage.}}

    Stage membership is not enforced by the type system; the {!Spec}
    validators check it at pass boundaries. *)

(** {1:main Main type} *)

type t
(** A hash-consed IR node. Structurally identical nodes are physically
    identical ([==]), and {!equal} on them reduces to physical equality
    on {!tag}. *)

(** {1:aux Auxiliary types} *)

(** Device placement.

    [Single] names a concrete device; [Multi] names a group of devices
    that share a shard. [Index] selects a device by position while rewriting
    one shard of a multi-device value. *)
type device =
  | Single of string
  | Multi of string list
  | Index of int

(** Schedule options attached to kernel metadata.

    Each variant except {!Nolocals} carries an [axis] — the schedule
    axis it applies to — and most carry an [amount] interpreted by the
    scheduler (tile width, unroll factor, lane count, padding target).

    {b Note.} The variant declaration order is load-bearing: total
    ordering over {!t} uses {!Stdlib.compare}, which reads the
    constructor ordinal first. *)
module Opt : sig
  type t =
    | Tc of { axis : int; tc_select : int; tc_opt : int; use_tc : int }
        (** Tensor-core configuration on [axis]. *)
    | Upcast of { axis : int; amount : int }
        (** Vectorize [axis] by [amount] lanes. *)
    | Unroll of { axis : int; amount : int }
        (** Unroll [axis] by [amount] iterations. *)
    | Local of { axis : int; amount : int }
        (** Split [axis] into workgroup-shared tiles of [amount]. *)
    | Thread of { axis : int; amount : int }
        (** Split [axis] into per-thread tiles of [amount]. *)
    | Group of { axis : int; amount : int }
        (** Split [axis] into workgroups of [amount]. *)
    | Grouptop of { axis : int; amount : int }
        (** Like {!Group} but takes the top portion of [axis]. *)
    | Nolocals  (** Disable local memory for this kernel. *)
    | Padto of { axis : int; amount : int }
        (** Pad [axis] to a multiple of [amount]. *)
    | Swap of { axis : int; with_axis : int }
        (** Swap [axis] and [with_axis] in the schedule. *)
  (** The type for schedule options. *)

  val to_string : t -> string
  (** [to_string opt] is a compact textual form of [opt]
      (e.g. ["UPCAST:0:4"]). *)

  val pp : Format.formatter -> t -> unit
  (** [pp] formats options with {!to_string}. *)

  val axis : t -> int option
  (** [axis opt] is the axis carried by [opt], or [None] for
      {!Nolocals}. *)

  val amount : t -> int option
  (** [amount opt] is the amount carried by [opt], or [None] for
      {!Tc}, {!Swap}, and {!Nolocals}. *)

  val with_amount : t -> int -> t
  (** [with_amount opt n] is [opt] with its amount replaced by [n].
      Returns [opt] unchanged for {!Tc}, {!Swap}, and {!Nolocals}. *)
end

type stage_opts = {
  device : device option;
      (** Target device, or [None] for default placement. When a
          positional selector is needed, use {!device.Index}. *)
  addrspace : Dtype.addr_space;  (** Memory address space. *)
  removable : bool;
      (** [true] if the buffer can be elided by later passes. *)
}
(** Options attached to an {!Ops.Stage} node. *)

type metadata = {
  name : string;  (** Operation name. *)
  caller : string;  (** Call-site identifier. *)
  backward : bool;  (** [true] if emitted during backward pass. *)
}
(** Side metadata attached to tensor-stage call sites or individual uops.
    Per-uop side metadata does not participate in hash-consing. *)

type param_arg = {
  slot : int;  (** Parameter slot. [-1] denotes a symbolic variable. *)
  dtype : Dtype.t;
      (** Scalar element dtype of the parameter or buffer. Equal to the node's
          dtype; carried in the arg so index-lowering rewrites can re-infer the
          node dtype without desynchronising from the parameter. *)
  vmin_vmax : (int * int) option;
      (** Symbolic lower and upper bounds, when known. *)
  name : string option;  (** Symbolic or debug name, when known. *)
  addrspace : Dtype.addr_space;
      (** Memory address space. Defaults to {!Dtype.Global}.
          {!Dtype.Alu} denotes ALU symbolic parameters. *)
  axis : int option;  (** Sharding axis, when applicable. *)
  device : device option;  (** Concrete or multi-device placement. *)
}
(** Payload for {!Ops.Param} and {!Ops.Buffer}. *)

type realization_state =
  | Never_realized
      (** Tinygrad's [realized] and [is_realized] would be false without
          consulting runtime allocation state. This includes non-buffer bases
          and LOCAL/REG scratch buffers. *)
  | Runtime_dependent of t list
      (** Tinygrad's [realized] and [is_realized] depend on allocation state
          for these concrete {!Ops.Buffer} identities. *)
(** Static part of tinygrad's runtime-backed realization properties.

    The actual [Buffer]/[MultiBuffer] object belongs to the execution runtime,
    not to [tolk.uop]. This type records only the graph facts [Uop] can answer
    faithfully. *)

type reduce_arg = {
  op : Ops.t;  (** Reduction operation. *)
  num_axes : int;
      (** Number of leading axes reduced. The reduced axes are permuted to
          the front of the source, so this counts them rather than naming
          them. *)
}
(** Payload for {!Ops.Reduce}. [num_axes = 0] denotes the lowered kernel form
    whose reduced ranges are carried in {!src}. *)

type const_value =
  | Const_scalar of Dtype.storage_scalar
      (** Scalar payload coerced by the requested dtype. *)
  | Const_invalid  (** Tinygrad [Invalid] sentinel. *)
  | Const_tuple of const_value list
      (** Tuple payload represented as a {!Ops.Stack} of scalar constants. *)
(** Payload accepted by {!const_of_dtype}. *)

(** Static or symbolic count used in kernel cost estimates. *)
type estimate =
  | Int of int  (** Concrete count. *)
  | Sym of t
      (** Symbolic expression over runtime variables (typically a
          {!variable}-rooted arithmetic uop). *)

type estimates = {
  ops : estimate;  (** Arithmetic operation count. *)
  lds : estimate;  (** Local data share access count. *)
  mem : estimate;  (** Global memory access count. *)
}
(** Kernel cost estimates. *)

type kernel_info = {
  name : string;  (** Kernel name, used for debugging and codegen. *)
  axis_types : Axis_type.t list;  (** Kind assignment per schedule axis. *)
  dont_use_locals : bool;
      (** [true] if local memory was disabled (e.g. via
          {!Opt.Nolocals}). *)
  applied_opts : Opt.t list;  (** Schedule options already applied. *)
  opts_to_apply : Opt.t list option;
      (** Remaining options to apply, or [None] for auto-tuning. *)
  estimates : estimates option;  (** Cost estimates, if computed. *)
  beam : int;  (** Beam-search score. [0] means no beam score. *)
}
(** Non-semantic kernel annotations attached to the kernel-stage
    root {!Ops.Sink}. *)

type grad_fxn = grad_output:t -> call:t -> t option list
(** Custom gradient callback. Given the upstream gradient
    [~grad_output] and the original [~call] node, returns a gradient
    (or [None] for non-differentiable positions) per call argument, in
    positional order. *)

type call_info = {
  grad_fxn : grad_fxn option;  (** Custom gradient callback, if any. *)
  name : string option;  (** Optional callable name for debugging. *)
  precompile : bool;
      (** [true] to precompile the forward callee. *)
  precompile_backward : bool;
      (** [true] to precompile the backward callee. *)
  aux : string option;  (** Auxiliary call payload for cache/runtime users. *)
}
(** Annotations attached to a {!Ops.Call} or {!Ops.Function} node. *)

type launch_dim =
  | Launch_int of int  (** Concrete integer launch dimension. *)
  | Launch_float of float  (** Concrete float launch dimension. *)
  | Launch_sym of t  (** Symbolic launch dimension. *)
(** Program launch dimension. *)

type launch_value =
  | Launch_value_int of int  (** Resolved integer launch dimension. *)
  | Launch_value_float of float  (** Resolved floating launch dimension. *)
(** Concrete launch dimension returned by {!program_launch_dims}. *)

type program_info = {
  name : string;  (** Program name before sanitization. *)
  global_size : launch_dim list;  (** Global launch dimensions. *)
  local_size : int list option;  (** Local launch dimensions, if fixed. *)
  vars : t list;  (** Runtime symbolic parameters. *)
  globals : int list;  (** Global buffer slots. *)
  outs : int list;  (** Output buffer slots. *)
  ins : int list;  (** Input buffer slots. *)
  aux : string list;  (** Auxiliary program payload. *)
}
(** Tinygrad-shaped program metadata attached to {!Ops.Program}. *)

val sanitize_function_name : string -> string
(** [sanitize_function_name name] rewrites [name] to a valid C identifier:
    characters outside [[A-Za-z0-9_]] are replaced by the uppercase
    hexadecimal of their code point. *)

val kernel_function_name : kernel_info -> string
(** [kernel_function_name info] is [info.name] sanitized for backend
    function emission. *)

val program_function_name : program_info -> string
(** [program_function_name info] is [info.name] sanitized for backend
    function emission. *)

val program_info_from_sink : ?aux:string list -> t -> program_info
(** [program_info_from_sink ?aux sink] derives tinygrad-style program metadata
    from [sink]. It scans [sink]'s topological order for ALU {!Ops.Param}
    runtime variables, non-ALU {!Ops.Param} global buffer slots, load/store
    buffer slots, and {!Ops.Special} launch dimensions.

    [aux] defaults to [[]]. The resulting [vars], [globals], [outs], and [ins]
    are deduplicated and sorted like tinygrad's [ProgramInfo.from_sink].

    Raises [Invalid_argument] if a launch axis is outside the three tinygrad
    launch dimensions, or if a local launch dimension is not a concrete
    integer. *)

val program_runtimevars : program_info -> (string * int) list
(** [program_runtimevars info] maps runtime-owned variable names to their
    positions in [info.vars]. Currently this matches tinygrad's [core_id]
    rule. *)

val program_launch_dims :
  program_info -> var_vals:(string * int) list ->
  launch_value list * int list option
(** [program_launch_dims info ~var_vals] resolves [info.global_size] and
    [info.local_size] using [var_vals]. Symbolic global dimensions are
    evaluated as integer UOp expressions over named runtime variables.

    Raises [Not_found] if a symbolic dimension references a missing variable
    or an expression outside the UOp-local evaluator. *)

val program_vals : program_info -> var_vals:(string * int) list -> int option list
(** [program_vals info ~var_vals] is the runtime argument tuple for
    [info.vars]. Variables listed by {!program_runtimevars} return [None];
    other variables return [Some value] from [var_vals].

    Raises [Not_found] if a non-runtime variable has no supplied value. *)

type wmma_info = {
  name : string;  (** Tensor-core primitive name. *)
  dims : int * int * int;  (** Matrix dimensions [(M, N, K)]. *)
  dtype_in : Dtype.t;  (** Input operand scalar type. *)
  dtype_out : Dtype.t;  (** Output operand scalar type. *)
  device : string;  (** Target device name. *)
  threads : int;  (** Warp thread count. *)
  upcast_axes :
    (int * int) list * (int * int) list * (int * int) list;
      (** [(axis, amount)] pairs for the [A]/[B]/[C] operands. *)
  reduce_axes : int list;  (** Reduction axis ids. *)
}
(** Configuration of a kernel-stage tensor-core matrix-multiply
    accumulate ({!Ops.Wmma}). *)

(** {1:arg Arg}

    Per-op structured payload. One flat sum with generic shapes (ints,
    strings, consts) reused across ops and named record variants for
    rich data. The pairing between variant and op is documented at the
    smart constructors; construction is always through those
    constructors.

    Prefer the {{!section-viewacc}view accessors} over raw pattern
    matching when the full per-op contract (both [src] and [arg]) is
    of interest. *)

module Arg : sig
  type t =
    | Empty
    | Int of int
    | Ints of int list
    | Bools of bool list
    | String of string
    | Value of Const.t
    | Op of Ops.t
    | Range_info of { axis : int; sub : int list; kind : Axis_type.t }
    | Param_arg of param_arg
    | Reduce_arg of reduce_arg
        (** For [Reduce]: reduction op plus leading-axis count. *)
    | Device of device
        (** For [Copy]: device placement. *)
    | Op_device of Ops.t * device
        (** For [Allreduce]: reduction op and device group. *)
    | Stage_info of stage_opts
    | Opts of Opt.t list
        (** For [Contiguous]: schedule options attached to the boundary. *)
    | Kernel_info of kernel_info
    | Call_info of call_info
    | Program_info of program_info
    | Wmma_info of wmma_info

  val equal : t -> t -> bool
  (** [equal a b] is structural equality on payloads. {!Call_info}
      values are compared with physical equality on their [grad_fxn]
      closure, and structural equality on the rest. *)

  val compare : t -> t -> int
  (** [compare] is {!Stdlib.compare} on payloads. Used only for total
      ordering; for semantic comparison use {!equal}. *)

  (** {2:payload_acc Payload accessors}

      One accessor per generic payload variant, returning [None] when
      the variant does not match. Use [Option.is_some] for a boolean
      check. The richer named-record payloads are destructured through
      the {{!section-viewacc}view accessors} on nodes. *)

  val as_int : t -> int option
  val as_ints : t -> int list option
  val as_bools : t -> bool list option
  val as_string : t -> string option
  val as_value : t -> Const.t option
  val as_op : t -> Ops.t option
  val as_param_arg : t -> param_arg option
  val as_reduce_arg : t -> reduce_arg option
  val as_device : t -> device option
  val as_opts : t -> Opt.t list option
  val as_stage_info : t -> stage_opts option
  val as_program_info : t -> program_info option
end

type arg = Arg.t

(** {1:accessors Accessors} *)

val op : t -> Ops.t
(** [op u] is [u]'s operation tag. *)

val dtype : t -> Dtype.t
(** [dtype u] is [u]'s dtype. *)

val src : t -> t array
(** [src u] is [u]'s ordered children. Child positions are part of each
    op's contract, documented at each smart constructor. The returned
    array is shared — do not mutate it. Prefer the
    {{!section-viewacc}view accessors} over indexing into this array. *)

val arg : t -> arg
(** [arg u] is [u]'s structured payload. *)

val tag : t -> int
(** [tag u] is [u]'s hash-cons identity. Two nodes are physically
    identical iff their [tag]s are equal. Tags are stable within a
    single run of the program but not across runs — use
    {!compare_structure} when a deterministic order is required. *)

val node_tag : t -> string option
(** [node_tag u] is the optional string tag attached to [u],
    orthogonal to {!tag}. It participates in hash-consing, so nodes
    differing only in [node_tag] are distinct. It does not participate
    in {!semantic_key}. Use {!metadata} for diagnostic side data. *)

val metadata : t -> metadata list
(** [metadata u] is the side metadata attached to [u]. It does not
    participate in hash-consing or {!semantic_key}. *)

val with_metadata : metadata list -> t -> t
(** [with_metadata md u] attaches side metadata [md] to [u] and returns
    [u]. This mutates the module-local side table. *)

exception Bottom_up_gate
(** Raised by a pre-matcher to keep the current node and skip its
    children and post-matcher. Caught only by {!graph_rewrite}'s
    pre-matcher path. *)

val children : t -> t list
(** [children u] is [Array.to_list (src u)]. *)

val child_ops : t -> Ops.t list
(** [child_ops u] is the deduplicated set of operation tags appearing among
    [u]'s direct children. Memoised on the node across calls: it backs the
    pattern matcher's early-reject, which consults it on every candidate. *)

(** {1:predicates Predicates} *)

val equal : t -> t -> bool
(** [equal a b] is [tag a = tag b], i.e. physical equality on
    interned nodes. *)

val compare : t -> t -> int
(** [compare a b] orders by hash-cons tag. Total and consistent with
    {!equal} but not stable across runs; see {!compare_structure} for
    a stable alternative. *)

(** {1:viewacc View accessors}

    Structured views over per-op [src]/[arg] contracts. Each [as_<op>]
    returns [Some r] when [u]'s op matches and its payload is well
    formed, and [None] otherwise. These encode the positional [src]
    conventions once, so rewrite rules and passes do not repeat them. *)

type index_view = { ptr : t; idxs : t list }
(** View of an {!Ops.Index} node: pointer and logical index axes. *)

type load_view = { src : t; alt : t option; gate : t option }
(** View of an {!Ops.Load} node: pointer source plus optional alternate
    value and gate. *)

type store_view = { dst : t; value : t; gate : t option }
(** View of an {!Ops.Store} node: destination pointer, value to store, and
    optional gate. *)

type range_view = {
  size : t;
  parents : t list;
  axis : int;
  sub : int list;
  kind : Axis_type.t;
}
(** View of an {!Ops.Range} node: loop bound, outer ordering parents,
    schedule axis, sub-axis ids, and axis kind. *)

type end_view = { value : t; ranges : t list }
(** View of an {!Ops.End} node: value produced by the loop body and
    the ranges closed around it. *)

type if_view = { cond : t; idx_for_dedup : t }
(** View of an {!Ops.If} node: condition and the index used to
    deduplicate guards over the same region. *)

type reduce_view = { src : t; ranges : t list; op : Ops.t; num_axes : int }
(** View of an {!Ops.Reduce} node: body, lowered loop ranges reduced
    over, reduction op, and leading-axis count. Tensor-stage reductions have
    [ranges = []] and [num_axes > 0]; lowered kernel reductions have
    [num_axes = 0] and carry source ranges after [src]. *)

type allreduce_view = { src : t; device : device; op : Ops.t }
(** View of an {!Ops.Allreduce} node: body, device group, and
    reduction op. *)

type stage_view = { src : t; ranges : t list; opts : stage_opts }
(** View of an {!Ops.Stage} node: materialised body, indexing
    ranges, and placement options. *)

type slice_view = { src : t; offset : t; size : int }
(** View of an {!Ops.Slice} node: underlying buffer, symbolic offset,
    and element count. *)

type wait_view = { src : t }
(** View of an {!Ops.Wait} node: the boolean condition being waited on. *)

type param_view = { param : param_arg; shape : t }
(** View of an {!Ops.Param} node: tinygrad-style payload plus shape child.
    Unknown shape is represented by an empty void {!Ops.Noop}
    sentinel. Device placement is carried by [param.device]. *)

type buffer_view = { buffer : param_arg; shape : t }
(** View of an {!Ops.Buffer} node: tinygrad-style payload plus shape child.
    Unknown shape is represented by an empty void {!Ops.Noop} sentinel.
    Device placement is carried by [buffer.device]. *)

type wmma_view = { a : t; b : t; c : t; info : wmma_info }
(** View of an {!Ops.Wmma} node: operands and hardware configuration. *)

type call_view = { body : t; args : t list; info : call_info }
(** View of an {!Ops.Call} or {!Ops.Function} node: callee body,
    positional arguments, and call annotations. *)

type special_view = { name : string; size : t }
(** View of an {!Ops.Special} node: raw hardware index name and its upper
    bound. *)

type bind_view = { var : t; value : t }
(** View of an {!Ops.Bind} node: symbolic parameter and concrete value. *)

type marg =
  | Marg_shape of t list
      (** Target shape for {!Ops.Reshape} and {!Ops.Expand}. *)
  | Marg_bounds of (t * t) list
      (** Per-axis [(offset, size)] bounds for {!Ops.Pad} and
          {!Ops.Shrink}. *)
  | Marg_permute of int list
      (** Axis order for {!Ops.Permute}. *)
  | Marg_flip of bool list
      (** Per-axis reversal flags for {!Ops.Flip}. *)
(** Tinygrad-style movement argument decoded from a movement op. *)

val as_index : t -> index_view option
(** [as_index u] matches {!Ops.Index}. *)

val as_load : t -> load_view option
(** [as_load u] matches {!Ops.Load}. *)

val as_store : t -> store_view option
(** [as_store u] matches {!Ops.Store}. *)

val as_range : t -> range_view option
(** [as_range u] matches {!Ops.Range}. *)

val as_end : t -> end_view option
(** [as_end u] matches {!Ops.End}. *)

val as_if : t -> if_view option
(** [as_if u] matches {!Ops.If}. *)

val as_reduce : t -> reduce_view option
(** [as_reduce u] matches {!Ops.Reduce}. *)

val as_allreduce : t -> allreduce_view option
(** [as_allreduce u] matches {!Ops.Allreduce}. *)

val as_stage : t -> stage_view option
(** [as_stage u] matches {!Ops.Stage}. *)

val as_slice : t -> slice_view option
(** [as_slice u] matches {!Ops.Slice}. *)

val as_wait : t -> wait_view option
(** [as_wait u] matches {!Ops.Wait}. *)

val as_param : t -> param_view option
(** [as_param u] matches {!Ops.Param} nodes carrying {!Param_arg}. *)

val as_buffer : t -> buffer_view option
(** [as_buffer u] matches {!Ops.Buffer} nodes carrying {!Param_arg}. *)

val as_wmma : t -> wmma_view option
(** [as_wmma u] matches {!Ops.Wmma}. *)

val as_call : t -> call_view option
(** [as_call u] matches both {!Ops.Call} and {!Ops.Function}. *)

val as_special : t -> special_view option
(** [as_special u] matches {!Ops.Special}. *)

val as_bind : t -> bind_view option
(** [as_bind u] matches {!Ops.Bind}. *)

val as_contiguous_opts : t -> Opt.t list option
(** [as_contiguous_opts u] is [Some opts] when [u] is an {!Ops.Contiguous}
    carrying schedule options, and [None] otherwise. *)

val as_kernel_info : t -> kernel_info option
(** [as_kernel_info u] is [Some ki] when [u] is an {!Ops.Sink} carrying
    kernel metadata, and [None] otherwise. *)

val as_call_info : t -> call_info option
(** [as_call_info u] is [Some info] when [u] is an {!Ops.Call} or
    {!Ops.Function}, and [None] otherwise. *)

val as_program_info : t -> program_info option
(** [as_program_info u] is [Some info] when [u] is an {!Ops.Program}
    carrying {!Program_info}, and [None] otherwise. *)

(** {1:ctors Smart constructors}

    Labelled-argument constructors that intern the result in the global
    hash-cons table. Each constructor documents the child layout in
    {!src} and, where relevant, its stage and dtype inheritance rules.
    Constructors that depend on runtime invariants raise
    {!Invalid_argument} on violation. *)

(** {2:ctors_struct Structural} *)

val sink : ?kernel_info:kernel_info -> t list -> t
(** [sink ?kernel_info srcs] is the graph or kernel root gathering
    [srcs]. [kernel_info] attaches non-semantic kernel annotations at
    kernel stage. Void dtype. Shared. *)

val group : t list -> t
(** [group srcs] groups effect-like children without introducing a
    value. Returns the single element unchanged when [srcs] is a
    singleton. Void dtype. Shared. *)

val after : src:t -> deps:t list -> t
(** [after ~src ~deps] sequences [src] after [deps] as an ordering
    dependency. Returns [src] unchanged when [deps] is empty. Dtype is
    inherited from [src]. Shared. *)

val noop : ?src:t -> dtype:Dtype.t -> unit -> t
(** [noop ?src ~dtype ()] is a pass-through scheduling marker with
    [dtype]. Optional single [src]. Tensor. *)

val shape_to_shape_arg : t option -> t
(** [shape_to_shape_arg shape] is [shape] when supplied and an empty void
    {!Ops.Noop} sentinel otherwise. *)

val linear : t list -> t
(** [linear srcs] is the linearized schedule of [srcs]. Void dtype.
    Program. *)

(** {2:ctors_buffers Parameters and buffers} *)

val param :
  slot:int -> dtype:Dtype.t -> ?shape:t -> ?device:device ->
  ?vmin_vmax:int * int -> ?name:string ->
  ?addrspace:Dtype.addr_space -> ?axis:int -> unit -> t
(** [param ~slot ~dtype ?shape ?device ?vmin_vmax ?name ?addrspace ?axis ()]
    is a {!Ops.Param} carrying {!param_arg} and exactly one shape child.
    [shape] defaults to {!shape_to_shape_arg} [None]. Shared. *)

val variable :
  name:string -> min_val:int -> max_val:int -> ?dtype:Dtype.t -> unit -> t
(** [variable ~name ~min_val ~max_val ?dtype ()] is a symbolic
    {!Ops.Param} in {!Dtype.Alu} address space. [dtype] defaults to
    {!Dtype.index}. Shared. *)

val buffer :
  slot:int -> dtype:Dtype.t -> ?shape:t -> ?name:string ->
  ?addrspace:Dtype.addr_space -> ?axis:int -> ?device:device -> unit -> t
(** [buffer ~slot ~dtype ?shape ?name ?addrspace ?axis ?device ()] is an
    {!Ops.Buffer} carrying {!param_arg} and exactly one shape child. [shape]
    defaults to {!shape_to_shape_arg} [None]. Tensor. *)

val fresh_buffer_slot : unit -> int
(** [fresh_buffer_slot ()] draws the next process-unique buffer slot. Two
    buffers with the same slot, dtype, shape, and device hash-cons to the same
    node, so every distinct allocation must draw a fresh slot. *)

val reserve_buffer_slots : int -> unit
(** [reserve_buffer_slots n] raises the {!fresh_buffer_slot} counter so that
    subsequent draws are at least [n]. Call it before allocating alongside a
    graph whose buffers were numbered by hand. *)

val stage : src:t -> ranges:t list -> opts:stage_opts -> t
(** [stage ~src ~ranges ~opts] materialises [src] into a staged value
    indexed by loop [ranges]. Dtype is inherited from [src]. Kernel. *)

val slice : src:t -> offset:t -> size:int -> dtype:Dtype.t -> t
(** [slice ~src ~offset ~size ~dtype] is a {!Ops.Slice} view of [src]
    at symbolic [offset], with [size] elements. Tensor. *)

(** {2:ctors_scalars Variables, binds, constants} *)

val bind : var:t -> value:t -> t
(** [bind ~var ~value] binds symbolic parameter [var] to concrete
    [value]. Dtype is inherited from [var]; [src] is [(var, value)].

    Raises [Invalid_argument] if a constant [value] is outside [var]'s
    known bounds. Tensor. *)

val const : ?srcs:t list -> Const.t -> t
(** [const ?srcs v] is a compile-time constant [v]. [srcs] carries
    scheduling dependencies and is empty at kernel stage. Dtype is
    that of [v]. Shared. *)

val const_of_dtype : ?shape:t -> Dtype.t -> const_value -> t
(** [const_of_dtype ?shape dtype value] is a constant node for [value] at
    [dtype]. Scalar values produce a {!Ops.Const}. Tuple values produce a
    {!Ops.Stack} of scalar constants with lane dtype [dtype]; the tuple length
    is the lane count.

    If [shape] is supplied and is not scalar, the result is reshaped from
    singleton dimensions and expanded to [shape]. *)

val invalid : ?dtype:Dtype.t -> unit -> t
(** [invalid ?dtype ()] is the [Invalid] sentinel expressed as a
    [Const]. [dtype] defaults to {!Dtype.index}. Shared. *)

val const_int : int -> t
(** [const_int n] is a {!Dtype.index} integer constant, used for shape
    dimensions and loop indices. Shared. *)

val const_float : float -> t
(** [const_float x] is a {!Dtype.weakfloat} float constant. Shared. *)

val const_bool : bool -> t
(** [const_bool b] is a {!Dtype.bool} boolean constant. Shared. *)

val zero_like : t -> t
(** [zero_like u] is a [Const] zero with [u]'s value dtype.

    @raise Invalid_argument if [u] has a pointer dtype. *)

val const_like : t -> int -> t
(** [const_like u n] is an integer [Const] with value [n] and [u]'s
    value dtype.

    @raise Invalid_argument if [u] has a pointer dtype, or if [u]'s
    dtype is not a scalar integer. *)

(** {2:ctors_mem Indexing and memory}

    {!Ops.Index}: [src = ptr :: idxs]. The tail carries one logical
    index expression per indexed axis, mirroring tinygrad's variadic
    [INDEX].

    {!Ops.Load}: [src = \[| idx |\]] or [\[| idx; alt; gate |\]].

    {!Ops.Store}: [src = \[| dst; value |\]] or [\[| dst; value; gate |\]]. *)

val index : ptr:t -> idxs:t list -> unit -> t
(** [index ~ptr ~idxs ()] indexes [ptr] by [idxs], selecting an element. The
    result dtype is [ptr]'s scalar dtype (a buffer or vector already carries
    its element type). A constant index into a {!Ops.Stack} selects the lane
    node directly. Kernel. *)

val load : src:t -> ?dtype:Dtype.t -> ?alt:t -> ?gate:t -> unit -> t
(** [load ~src ?dtype ?alt ?gate ()] loads the element addressed by [src]. The
    result dtype is [dtype] when specified, and otherwise [src]'s dtype, which
    the indexed source already carries. [alt] is the value substituted when
    [gate] is false. [alt] and [gate] must be supplied together. Kernel. *)

val store : dst:t -> value:t -> ?gate:t -> unit -> t
(** [store ~dst ~value ?gate ()] stores [value] through pointer [dst],
    optionally guarded by [gate]. Void dtype. Shared. *)

(** {2:ctors_alu Arithmetic} *)

val alu_unary : op:Ops.t -> src:t -> t
(** [alu_unary ~op ~src] is a unary ALU node. Dtype is inherited from
    [src]. Shared.

    @raise Invalid_argument if [op] is not in {!Ops.Group.unary}. *)

val alu_binary : op:Ops.t -> lhs:t -> rhs:t -> t
(** [alu_binary ~op ~lhs ~rhs] is a binary ALU node. Dtype is inherited
    from [lhs], except for comparisons which produce a bool of matching
    vector width. Shared.

    @raise Invalid_argument if [op] is not in {!Ops.Group.binary}, or
    if [op] is a comparison on a pointer dtype. *)

val alu_ternary : op:Ops.t -> a:t -> b:t -> c:t -> t
(** [alu_ternary ~op ~a ~b ~c] is a ternary ALU node. Dtype is
    inherited from [b] for {!Ops.Where} and from [a] otherwise.
    Shared.

    @raise Invalid_argument if [op] is not in {!Ops.Group.ternary}. *)

val valid : src:t -> cond:t -> t
(** [valid ~src ~cond] is [where cond src invalid]: it masks [src] to the
    {!Const.invalid} sentinel of [src]'s dtype wherever [cond] is false.
    Used to gate index expressions. Dtype is inherited from [src]. *)

val cast : src:t -> dtype:Dtype.t -> t
(** [cast ~src ~dtype] converts [src] to [dtype] with the usual
    numeric-conversion semantics. When [dtype] is scalar and [src] is
    vector-valued, [dtype]'s vector count is adjusted to match [src].
    Returns [src] unchanged when the adjusted dtype is already [src]'s
    dtype. Shared. *)

val bitcast : src:t -> dtype:Dtype.t -> t
(** [bitcast ~src ~dtype] reinterprets [src]'s bits as [dtype] without
    conversion. Returns [src] unchanged when [dtype] is already [src]'s
    dtype. Shared. *)

(** {2:ctors_vec Vector manipulation} *)

val stack : ?dtype:Dtype.t -> t list -> t
(** [stack ?dtype srcs] packs scalar [srcs] into a {!Ops.Stack} value.
    Non-empty stacks carry the scalar lane dtype; the lane count is represented
    by the number of sources and the resulting shape, as in tinygrad.
    Empty [srcs] produces a void empty stack. Shared. *)

val getaddr : src:t -> t
(** [getaddr ~src] is a {!Ops.Getaddr} node extracting the address of
    [src]. Its dtype is {!Dtype.uint64}. *)

val broadcast : t -> int -> t
(** [broadcast u n] repeats [u] into an [n]-wide {!Ops.Stack}. Returns
    [u] unchanged when [n <= 1]. Shared. *)

(** {2:ctors_control Control flow}

    {!Ops.Range}: [src = \[| size; parent0; parent1; ... |\]].

    {!Ops.End}: [src = \[| value; range0; range1; ... |\]].

    {!Ops.If}: [src = \[| cond; idx_for_dedup |\]].

    {!Ops.Endif}: [src = \[| if_ |\]].

    {!Ops.Wait}: [src = \[| cond |\]]. *)

val range :
  size:t -> axis:int -> kind:Axis_type.t -> ?sub:int list ->
  ?dtype:Dtype.t -> ?parents:t list -> unit -> t
(** [range ~size ~axis ~kind ?sub ?dtype ?parents ()] is a loop variable
    over \[[0];[size-1]\] bound to schedule [axis] with semantic [kind]
    (see {!Axis_type}). [sub] defaults to [[]]. [dtype] defaults to
    {!Dtype.index}. [parents] defaults to [[]] and lists the outer
    {!range} nodes this loop must be emitted under, used as
    control-flow ordering dependencies. Shared. *)

val end_ : value:t -> ranges:t list -> t
(** [end_ ~value ~ranges] closes loop [ranges] around [value]. Void dtype;
    the value flows through [value] ([src.(0)]), whose shape it keeps.
    Returns [value] unchanged when [ranges] is empty. Kernel. *)

val if_ : cond:t -> idx_for_dedup:t -> t
(** [if_ ~cond ~idx_for_dedup] is a predicated control-flow gate.
    [idx_for_dedup] is used to deduplicate {!Ops.If} nodes that guard
    the same region. Void dtype. Kernel. *)

val endif : if_:t -> t
(** [endif ~if_] closes an {!if_} region. Void dtype. Kernel. *)

val barrier : ?srcs:t list -> unit -> t
(** [barrier ?srcs ()] is a workgroup barrier. [srcs] defaults to [[]]
    and carries ordering dependencies. Void dtype. Kernel. *)

val wait : src:t -> t
(** [wait ~src] waits on the boolean condition [src], with
    [src = \[| src |\]]. Void dtype. Kernel. *)

val special : name:string -> size:t -> ?dtype:Dtype.t -> unit -> t
(** [special ~name ~size ?dtype ()] is a backend-provided hardware index
    named [name] and bounded by [size], ranging over \[[0];[size-1]\].
    [size] is cast to [dtype]. [dtype] defaults to {!Dtype.index}.
    Kernel. *)

(** {2:ctors_reduce Reduction}

    Tensor-stage {!Ops.Reduce}: [src = \[| body |\]] where [body] has the
    reduced axes permuted to the front, [arg = Reduce_arg { op; num_axes }].

    Lowered/kernel {!Ops.Reduce}: [src = \[| body; range0; range1; ... |\]],
    [arg = Reduce_arg { op; num_axes = 0 }].

    {!Ops.Allreduce}: [src = \[| body |\]], [arg = Op_device (r, device)]. *)

val reduce :
  src:t -> ranges:t list -> op:Ops.t -> dtype:Dtype.t -> t
(** [reduce ~src ~ranges ~op ~dtype] reduces [src] using [op] over the
    loop [ranges], producing a value of [dtype]. The payload has
    [num_axes = 0]. Kernel. *)

val reduce_axis : src:t -> op:Ops.t -> axes:int list -> t
(** [reduce_axis ~src ~op ~axes] reduces tensor [src] over tensor [axes]
    using [op]. The genuinely reduced axes (those not of size one) are
    permuted to the front and reduced as a leading block; size-one axes are
    dropped by reshape. Dtype is inherited from [src]. Returns [src]
    unchanged when [axes] is empty. Tensor. *)

val allreduce : src:t -> device:device -> op:Ops.t -> t
(** [allreduce ~src ~device ~op] reduces [src] using [op] across
    [device]. Dtype is inherited from [src]. Tensor. *)

(** {2:ctors_multi Sharding} *)

val multi : src:t -> axis:int -> t
(** [multi ~src ~axis] distributes [src] along [axis] for multi-device
    execution. Dtype is inherited from [src]. Tensor. *)

val mstack : t list -> t
(** [mstack srcs] stacks per-device shards into a multi-device tensor.
    Dtype is inherited from the first shard. Tensor.

    @raise Invalid_argument if [srcs] is empty. *)

val mselect : src:t -> index:int -> t
(** [mselect ~src ~index] selects shard [index] of a multi-device
    [src]. Dtype is inherited from [src]. Tensor. *)

val copy : src:t -> device:device -> unit -> t
(** [copy ~src ~device ()] copies [src] to [device]. Dtype is
    inherited from [src]. Tensor. *)

(** {2:ctors_movement Movement}

    Tensor-stage only. {!Ops.Reshape}: [src = \[| input; shape |\]];
    {!Ops.Expand}: [src = \[| input; dims |\]]. {!Ops.Pad} and {!Ops.Shrink}:
    [src = \[| input; offset; size |\]], where [offset] is the per-axis
    start offset and [size] is the resulting per-axis output size. *)

val reshape : src:t -> shape:t -> t
(** [reshape ~src ~shape] rearranges the elements of [src] into
    [shape] without changing the total count. Dtype is inherited from
    [src]. Tensor. *)

val expand : src:t -> dims:t -> t
(** [expand ~src ~dims] prepends [dims] as new leading axes of [src]: the
    result shape is [dims] followed by [src]'s shape. This is the primitive
    add-leading-dims op; whole-shape broadcasting is composed at the tensor
    level from reshape, expand, and permute. Returns [src] unchanged when
    [dims] is the empty shape. Dtype is inherited from [src]. Tensor. *)

val pad : src:t -> offset:t -> size:t -> t
(** [pad ~src ~offset ~size] pads [src] with zeros to the per-axis output
    size [size], offsetting the input by [offset]. Dtype is inherited from
    [src]. Tensor. *)

val shrink : src:t -> offset:t -> size:t -> t
(** [shrink ~src ~offset ~size] slices [src] at per-axis offsets [offset]
    with per-axis sizes [size]. Dtype is inherited from [src]. Tensor. *)

val permute : src:t -> order:int list -> t
(** [permute ~src ~order] permutes the axes of [src] according to
    [order]. Dtype is inherited from [src]. Tensor. *)

val flip : src:t -> dims:bool list -> t
(** [flip ~src ~dims] reverses [src] along each axis flagged in
    [dims]. Dtype is inherited from [src]. Tensor. *)

(** {2:ctors_sched Scheduling} *)

val detach : src:t -> t
(** [detach ~src] detaches [src] from the gradient tape. Dtype is
    inherited from [src]. Tensor. *)

val contiguous : src:t -> ?ranges:t list -> ?force:bool -> unit -> t
(** [contiguous ~src ?ranges ?force ()] forces [src] into contiguous layout.
    [ranges] defaults to [[]]. Schedule options live on the enclosing
    {!sink}'s {!kernel_info}, not here. Dtype is inherited from [src].
    Returns [src] unchanged for duplicate {!Ops.Contiguous} sources and
    for empty-range buffer-identity sources ({!Ops.Buffer}, {!Ops.Param},
    {!Ops.Slice}), unless [force] is [true].
    Tensor. *)

val contiguous_backward : src:t -> t
(** [contiguous_backward ~src] is a backward-pass contiguous marker.
    Dtype is inherited from [src]. Tensor. *)

val call : body:t -> args:t list -> info:call_info -> t
(** [call ~body ~args ~info] is tinygrad's call constructor. Opaque
    bodies ({!Ops.Sink}, {!Ops.Program}, {!Ops.Linear}, {!Ops.Copy},
    {!Ops.Slice}, and {!Ops.Custom_function}) produce {!Ops.Call}.
    Value-producing bodies produce {!Ops.Function}; non-tuple bodies are
    wrapped in {!Ops.Tuple}. Dtype is void and [src] is [(body, arg0,
    arg1, ...)].

    Raises [Invalid_argument] if [body] has in-scope ranges. Tensor. *)

val tuple : t list -> t
(** [tuple srcs] is an {!Ops.Tuple} with void dtype and
    [src = srcs]. Used as the body of a value-producing {!call}. Tensor. *)

val gettuple : src:t -> index:int -> t
(** [gettuple ~src ~index] projects element [index] out of a
    {!Ops.Tuple} or {!Ops.Function} body. Dtype is inherited from the
    [index]-th element. Tensor.

    @raise Invalid_argument if [src] is not a {!Ops.Tuple} or
    {!Ops.Function}, or if [index] is out of range. *)

val program :
  sink:t -> ?linear:t -> ?source:t -> ?binary:t -> info:program_info ->
  unit -> t
(** [program ~sink ?linear ?source ?binary ~info ()] is an
    {!Ops.Program} node with void dtype and [src = (sink, linear?,
    source?, binary?)]. [info] is carried as {!Arg.Program_info}.
    Program. *)

val set : target:t -> value:t -> ?extras:t list -> unit -> t
(** [set ~target ~value ?extras ()] is
    [after ~src:target ~deps:(store target value :: extras)]: a
    {!store} followed by an {!after} that sequences the store and any
    [extras] before [target]. Tensor. *)

(** {2:ctors_tc Tensor-core} *)

val wmma :
  a:t -> b:t -> c:t -> info:wmma_info -> dtype:Dtype.t -> t
(** [wmma ~a ~b ~c ~info ~dtype] is a concrete tensor-core
    matrix-multiply-accumulate. See {!wmma_info} for the per-device
    configuration. Kernel. *)

(** {2:ctors_custom Backend escape hatches} *)

val custom : fmt:string -> args:t list -> t
(** [custom ~fmt ~args] is a backend-specific effect or statement.
    [fmt] is the rendered source template; [args] are substituted into
    it. Void dtype. Kernel. *)

val custom_inline :
  fmt:string -> args:t list -> dtype:Dtype.t -> t
(** [custom_inline ~fmt ~args ~dtype] is like {!custom} but produces a
    value of [dtype] rather than an effect. Kernel. *)

val source : string -> t
(** [source s] carries rendered source text [s] as its arg. Void dtype,
    no [src]. Used as a child of {!program}. Program. *)

val binary : string -> t
(** [binary bytes] carries compiled machine-code [bytes] as its arg.
    {!Dtype.uint8} dtype with one shape dimension per byte, no [src].
    Used as a child of {!program}. Program. *)

val rewrite_error : src:t array -> msg:string -> t
(** [rewrite_error ~src ~msg] records a rewrite failure. [src] is
    typically copied from the node that failed; [msg] is the error
    message. Void dtype. Shared. *)

val ins : mnemonic:string -> operands:t list -> ?dtype:Dtype.t -> unit -> t
(** [ins ~mnemonic ~operands ?dtype ()] is a backend machine
    instruction. [mnemonic] is the assembly opcode; [operands] become
    [src]. [dtype] defaults to void. Program. *)

val custom_function : name:string -> srcs:t list -> t
(** [custom_function ~name ~srcs] is an {!Ops.Custom_function} named
    [name] with [src = srcs]. Void dtype. Tensor. *)

(** {2:ctors_replace Replace}

    There is no public [mk] escape hatch; every op must be built through
    a dedicated smart constructor above. This keeps per-op invariants
    and src layout centralised. *)

val replace :
  t -> ?op:Ops.t -> ?src:t array -> ?arg:arg -> ?dtype:Dtype.t ->
  ?node_tag:string option -> unit -> t
(** [replace u ?op ?src ?arg ?dtype ?node_tag ()] rebuilds [u] with
    the supplied fields overridden and the rest inherited from [u].
    Pass [~node_tag:None] to clear the diagnostic tag; omit it to
    preserve it. The result is hash-consed, so it is physically equal
    to [u] when every override matches the existing field.

    Bypasses the per-op validation performed by the dedicated smart
    constructors; callers are responsible for preserving the op's
    src/arg contract. *)

val with_tag : string -> t -> t
(** [with_tag s u] is [u] with its {!node_tag} replaced by [Some s].
    The result is hash-consed. *)

(** {1:traversal Traversal} *)

val toposort : ?gate:(t -> bool) -> ?enter_calls:bool -> t -> t list
(** [toposort ?gate ?enter_calls root] is the transitive dependencies
    of [root] in topological order, leaves first, [root] last. Each
    node appears at most once.

    [gate] defaults to [fun _ -> true]; children of nodes for which it
    returns [false] are not entered, though the node itself is still
    emitted.

    [enter_calls] defaults to [true]; when [false], {!Ops.Call} and
    {!Ops.Function} bodies (i.e. [src.(0)]) are not entered, but their
    argument children are. *)

val topovisit : (t -> 'a) -> (int, 'a) Hashtbl.t -> t -> 'a
(** [topovisit visitor cache root] folds over the DAG rooted at [root] in
    dependency order, leaves first, applying [visitor] to each node exactly
    once and memoizing the result in [cache] (keyed by {!tag}). A subtree
    whose root is already in [cache] is not re-descended, so successive calls
    sharing one [cache] short-circuit shared work across roots. Returns the
    result computed for [root]. *)

val backward_slice : t -> t list
(** [backward_slice root] is [toposort root] without [root] itself. *)

val find_nodes : (t -> bool) -> t -> t list
(** [find_nodes p root] is the nodes of the DAG rooted at [root] that
    satisfy [p], in topological order. *)

val in_backward_slice : t -> t -> bool
(** [in_backward_slice needle haystack] is [true] iff [needle] occurs
    strictly before [haystack] in [toposort haystack]. *)

val runtime_realization_state : t -> realization_state
(** [runtime_realization_state u] is the static graph portion of tinygrad's
    runtime-backed [realized] and [is_realized] properties.

    [Runtime_dependent buffers] means [u]'s tinygrad [base] can be realized
    iff the runtime has allocated every buffer in [buffers]. [Never_realized]
    means the answer is statically false, either because [u]'s base is not a
    realizable buffer node or because it contains LOCAL/REG scratch storage.

    This function does not allocate, look up, or cache runtime buffers. *)

val ranges : t -> t list
(** [ranges u] is the set of {!Ops.Range} nodes that [u] is nested
    within. A [Range] is included in its own [ranges]. Ops that close
    a range (e.g. {!Ops.Reduce}, {!Ops.Stage}, {!Ops.End},
    {!Ops.Wmma}, {!Ops.Call}, {!Ops.Function}, {!Ops.Copy},
    {!Ops.Slice}) drop ended ranges from the
    propagated set. *)

val ranges_subset : t -> t -> bool
(** [ranges_subset sub sup] is [true] iff every {!Ops.Range} in
    [ranges sub] also appears in [ranges sup]. *)

val device_of : t -> device option
(** [device_of u] resolves the device [u] is placed on by walking the
    DAG: a {!Ops.Stage} reports its buffer's device,
    {!Ops.After} inherits from [src.(0)],
    {!Ops.Mselect} indexes into the [Multi] device of its source,
    {!Ops.Mstack} stacks per-shard [Single] devices into [Multi],
    {!Ops.Param} and {!Ops.Buffer} read [Param_arg.device], and
    {!Ops.Copy} and {!Ops.Allreduce} read their payload device.
    Other ops report the device of their first child that has one, or
    [None]. *)

val addrspace : t -> Dtype.addr_space option
(** [addrspace u] is [u]'s tinygrad-style address-space property.
    {!Ops.Param} and {!Ops.Buffer} read their {!param_arg}. {!Ops.Special},
    {!Ops.Range}, and {!Ops.Load} are in {!Dtype.Alu}. Index-like,
    movement, reduction, and shard-selection nodes inherit from their first
    source. Elementwise and stack-like nodes report an address space only when
    all address-spaced sources agree. *)

val base : t -> t
(** [base u] walks through movement ops and {!Ops.Detach} to the
    underlying node. Other ops, including {!Ops.Multi}, {!Ops.Stage},
    {!Ops.Slice}, {!Ops.Bind}, {!Ops.Param}, and {!Ops.Buffer}, are their
    own base. *)

val buf_uop : t -> t
(** [buf_uop u] is the buffer-identity node reached by following tinygrad's
    buffer property rules. {!Ops.Param} and {!Ops.Buffer} return themselves;
    {!Ops.Slice} resolves through its source; {!Ops.Stage} and {!Ops.Mstack}
    stop the walk. *)

val has_buffer_identity : t -> bool
(** [has_buffer_identity u] is [true] iff [u] is a concrete graph buffer
    identity under tinygrad's shortcut rules: {!Ops.Param}, {!Ops.Buffer},
    {!Ops.Slice}, or those identities through {!Ops.Reshape}, {!Ops.Multi},
    or direct {!Ops.Gettuple} from a {!Ops.Tuple}. *)

val as_shape : t -> t list
(** [as_shape u] decodes [u] as a shape argument. Scalar constants and
    symbolic expressions become one dimension; {!Ops.Stack} becomes its
    source list. *)

val marg : t -> marg
(** [marg u] is the movement argument carried by [u]. It decodes
    {!Ops.Reshape}, {!Ops.Expand}, {!Ops.Pad}, {!Ops.Shrink},
    {!Ops.Permute}, and {!Ops.Flip}.

    Raises [Invalid_argument] if [u] is not a movement op or if the op's
    payload does not match its movement layout. *)

val shape : t -> t list
(** [shape u] is [u]'s symbolic shape.

    For {!Ops.Gettuple} through a {!Ops.Function}, shape expressions from
    the function body have internal {!Ops.Param} nodes substituted by the
    corresponding function call arguments by parameter slot.

    Raises [Invalid_argument] if [u] has no tensor shape. *)

val shape_opt : t -> t list option
(** [shape_opt u] is [Some (shape u)] when [u] has a shape and [None] when it
    does not, never raising. Memoised like {!shape}. *)

val max_shape : t -> int list
(** [max_shape u] is {!shape} with every symbolic dimension replaced by
    its conservative upper bound. *)

val shard_shape : t -> t list
(** [shard_shape u] is [shape u], except that a multi-device tensor with a
    known sharding axis has that axis divided by the number of devices. *)

val max_shard_shape : t -> int list
(** [max_shard_shape u] is {!shard_shape} with every symbolic dimension
    replaced by its conservative upper bound. *)

val axis : t -> int option
(** [axis u] is [u]'s sharding axis. {!Ops.Param} reads [param_arg.axis],
    {!Ops.Multi} reads its integer arg, {!Ops.Copy} clears the axis, direct
    tuple projections read the projected element, ALU ops use the last
    non-[None] source axis, and movement/reduction ops remap or clear the
    axis using tinygrad's shape rules. *)

val bounds : t -> (t * t) list
(** [bounds u] is the per-device shard interval on [u]'s sharding axis.

    Raises [Invalid_argument] if [u] has no sharding axis or no multi-device
    placement. *)

val contiguous_view_offset : t -> int option
(** [contiguous_view_offset u] is the element offset when [u] is a
    statically contiguous view of a parameter, buffer, or slice. Returns
    [None] when the layout is non-contiguous or the offset cannot be proven as
    an exact integer from UOp-local shape information. *)

(** {1:rewrite Rewriting} *)

val graph_rewrite :
  ?loc:string * int * int * int -> ?name:string -> ?enter_calls:bool ->
  ?bottom_up:bool ->
  ?bpm:(t -> t option) -> ?walk:bool ->
  ?on_rebuild:(old_n:t -> new_n:t -> unit) ->
  (t -> t option) -> t -> t
(** [graph_rewrite ?loc ?name ?enter_calls ?bottom_up ?walk ?bpm
    ?on_rebuild f root] rewrites the DAG rooted at [root] by applying [f]
    to every node and replacing any node for which [f] returns [Some u']
    with [u']. Returns the rewritten root. Results are memoised on
    physical identity so each input node is visited once per
    [graph_rewrite] call.

    {ul
    {- [loc] is optional source-position context for cycle diagnostics.}
    {- [name] defaults to [""]. It is included in cycle diagnostics.}
    {- [enter_calls] defaults to [false]; when [false], {!Ops.Call} and
       {!Ops.Function} bodies ([src.(0)]) are not rewritten.}
    {- [bottom_up] defaults to [false]. When [false], [f] is applied to
       each rewritten node after its children have been rewritten and
       [bpm] is an optional pre-matcher. When [true], [f] is used as a
       fixed-point pre-matcher and no post-matcher is used.}
    {- [walk] defaults to [false]. When [true], replacement subtrees are
       not recursively traversed by the same rewrite pass and
       pre/post-matchers run at most once per visited node.}
    {- [bpm] defaults to [None]. When [bottom_up] is [false], it is a
       fixed-point pre-matcher applied before descending into children.}
    {- [on_rebuild] defaults to the no-op; it is called with the
       original and rewritten node every time a node's identity
       changes, useful for change tracking.}}

    Raises [Invalid_argument] if fixed-point rewriting cycles back to a node
    already being rewritten. *)

val remove_all_tags : t -> t
(** [remove_all_tags root] clears every {!node_tag} reachable from [root],
    including call/function bodies. The resulting graph keeps the same
    operations, dtypes, payloads, side {!metadata}, and children modulo
    tag-stripped rebuilding. *)

val substitute : ?walk:bool -> (t * t) list -> t -> t
(** [substitute ?walk mappings root] rewrites [root] replacing every
    occurrence of the first component of each pair with the second.
    Bottom-up; performed via {!graph_rewrite} with the default
    no-enter-calls traversal. Lookup in [mappings] uses physical
    equality. [walk] defaults to [false]; when [true] replacement
    values are final: they are not traversed by the same pass, so a
    value may contain its own key without cycling. *)

val first_match : (t -> t option) list -> t -> t option
(** [first_match rules u] returns the first [Some _] result when
    applying each rule in order, or [None] if every rule fails. *)

(** {1:serial Serialization} *)

val export : t -> string
(** [export u] is a serialized form of the graph rooted at [u], suitable
    for {!import} in another process. The traversal covers the [src] edges
    and the uops embedded in node arguments: {!kernel_info} estimates
    ({!Sym}), {!program_info} variables, and symbolic launch dimensions
    ({!Launch_sym}). Node tags ({!node_tag}) are preserved; {!metadata}
    side data is not.

    Raises [Invalid_argument] if any node carries a gradient function
    ([grad_fxn] in its {!call_info}): gradient functions are closures and
    cannot be serialized. Compiled programs never carry them. *)

val import : string -> t
(** [import s] rebuilds a graph previously produced by {!export} in this
    process's hash-cons universe. Every node is re-interned bottom-up:
    a node structurally equal to a live node {b is} that node ([==]),
    genuinely new nodes are assigned fresh {!tag}s, and uops embedded in
    node arguments are remapped consistently with the [src] edges, so
    sharing between arguments and sources survives the round-trip.

    {b Warning.} {!Ops.Buffer} nodes hash-cons on their slot, so importing
    a graph whose internal buffer slots were minted by another process can
    make an imported buffer collide with a distinct local buffer that
    reuses the same slot, silently aliasing their storage. Renumber
    imported internal (negative) buffer slots before use.

    Raises [Failure] on malformed or version-incompatible input. Inputs
    are trusted: [import] rejects truncated data and unknown format
    versions, but it does not defend against adversarially crafted
    payloads — feed it only locally produced data, such as this machine's
    compile cache. *)

(** {1:tables Hash tables} *)

module Tbl : Hashtbl.S with type key = t
(** Hashtable keyed by {!equal}. Since {!equal} reduces to physical
    equality on hash-consed nodes, this is functionally equivalent to
    {!Ref_tbl} and is kept as the nominal "structural" entry point. *)

module Ref_tbl : Hashtbl.S with type key = t
(** Hashtable keyed by physical equality ([==]), hashing on {!tag}.
    Use this for caches that want to avoid re-hashing on each lookup. *)

(** {1:analysis Analysis} *)

val vmin : t -> int
(** [vmin u] is a conservative lower bound for [u]'s integer value.
    Analyses {!Ops.Const}, bounded symbolic {!Ops.Param},
    {!Ops.Range}, {!Ops.Special}, {!Ops.Bind}, {!Ops.Stack},
    {!Ops.Cast}, {!Ops.Where}, {!Ops.Neg}
    and the integer binary ALU ops (including {!Ops.Add}, {!Ops.Sub},
    {!Ops.Mul}, {!Ops.Max}, {!Ops.Cmod}, {!Ops.Cdiv},
    {!Ops.Floordiv}, {!Ops.Floormod}, {!Ops.Shl}, {!Ops.Shr},
    {!Ops.And}, {!Ops.Or}, {!Ops.Xor}, {!Ops.Cmplt}, {!Ops.Cmpne}).
    Empty intervals are represented by [vmin u > vmax u], as for
    [RANGE(0)]. Bounds that exceed OCaml's native [int] range are
    saturated to [min_int] or [max_int]. For unanalysed nodes the
    bound is the minimum value representable by [u]'s dtype
    ({!Dtype.min} for integer dtypes; [min_int] for non-integer
    dtypes). Memoised. *)

val vmax : t -> int
(** [vmax u] is a conservative upper bound, symmetric to {!vmin}.
    Fallback is {!Dtype.max} for integer dtypes, [max_int]
    otherwise. *)

val const_int_value : t -> int option
(** [const_int_value u] is [Some n] when [u] is a scalar integer
    {!Ops.Const} of value [n] that fits in OCaml's native [int], and
    [None] otherwise. *)

val const_factor : t -> int
(** [const_factor u] is a best-effort integer divisor of [u]: the
    integer value of [u] when [u] is an integer constant, the GCD of
    the lanes for {!Ops.Stack}, the summands for {!Ops.Add}, the known
    integer factor for {!Ops.Mul}, and [1] otherwise. Empty stacks have
    factor [0], matching tinygrad's use of [math.gcd ()]. *)

val divides : t -> int -> t option
(** [divides u n] is [Some q] when [u] is syntactically a multiple of
    [n], with [q] the quotient expressed as a uop. Returns [None] when
    divisibility cannot be proved. Handles {!Ops.Const}, {!Ops.Stack}
    (when every lane divides), {!Ops.Add} (when every term divides), and
    {!Ops.Mul} (when either factor divides). *)

val pop_const : t -> t * int
(** [pop_const u] splits [u = rest + c] into [(rest, c)] when [u] is
    [rest + const_int c], and returns [(u, 0)] otherwise. *)

val split_uop : t -> Ops.t -> t list
(** [split_uop u op] flattens a binary tree of [op] nodes into its
    leaves. E.g. [split_uop (a + b + c) Ops.Add] is [[a; b; c]].
    Returns [[u]] when [u]'s op is not [op]. *)

val usum : t list -> t
(** [usum xs] left-folds [xs] with {!Ops.Add}.
    @raise Invalid_argument if [xs] is empty. *)

val uprod : t list -> t
(** [uprod xs] left-folds [xs] with {!Ops.Mul}.
    @raise Invalid_argument if [xs] is empty. *)

val divide_exact : t -> t -> t option
(** [divide_exact u d] returns [Some q] such that [q * d] is provably
    equal to [u], or [None] otherwise. Handles [u == d], delegates to
    {!divides} for constant [d], and for symbolic [d] supports sums whose
    every term divides by [d] and products whose multiplicative factors
    contain the factors in [d]. *)

val gcd : t list -> t
(** [gcd xs] is a syntactic uop-level greatest common divisor. It combines
    the integer GCD of the individual {!const_factor}s with symbolic
    multiplicative factors common to every input.

    @raise Invalid_argument if [xs] is empty. *)

val simplify : t -> t
(** [simplify u] applies the installed symbolic-simplifier rules to
    fixed point. Before {!Symbolic} installs its rules, returns [u]
    unchanged. Dereferences {!simplify_ref}. *)

val simplify_ref : (t -> t) ref
(** Mutable hook backing {!simplify}. {!Symbolic} assigns it on
    module initialisation to break the dependency cycle between [Uop]
    and the symbolic rewriter. *)

val exec_alu : ?truncate_output:bool -> Ops.t -> Dtype.t -> Const.t list -> Const.t option
(** [exec_alu ?truncate_output op target args] folds ALU op [op] applied to
    constant [args], producing a constant of [target] dtype, or [None] when the
    op or operand shapes are not foldable. Any binary op with an {!Const.invalid}
    operand folds to {!Const.invalid} regardless of dtype.

    [truncate_output] defaults to [true]: the folded value is narrowed to
    [target]'s value domain (a no-op for {!Dtype.weakint}, {!Dtype.index}, and
    {!Dtype.weakfloat}, which have no finite width). Symbolic fold sites pass
    [false] to keep full host precision, deferring narrowing to emission.

    Bool comparisons follow IEEE for floats (nan differs from nan, [0.0] equals
    [-0.0]); integer division and modulo use C-truncating ({!Ops.Cdiv},
    {!Ops.Cmod}) or flooring ({!Ops.Floordiv}, {!Ops.Floormod}) semantics. *)

(** {1:sint Symbolic integers}

    A symbolic integer is a plain node: a concrete value is a
    {!Ops.Const} and a symbolic value is any other integer-valued
    expression, typically rooted in a {!variable} or a {!bind}. Tensor
    shapes are lists of such dimensions. *)

val resolve : ?default:bool -> t -> bool
(** [resolve ?default u] decides the boolean expression [u]. [u] is
    simplified first; when its value bounds agree the concrete truth
    value is returned, and otherwise [default] (defaulting to [true]).

    @raise Invalid_argument if [u] is not a boolean expression. *)

val smax : t list -> t
(** [smax xs] is the simplified maximum of [xs], staying symbolic when
    the maximum cannot be decided.

    @raise Invalid_argument if [xs] is empty. *)

val smin : t list -> t
(** [smin xs] is the simplified minimum of [xs], symmetric to {!smax}.

    @raise Invalid_argument if [xs] is empty. *)

val sprod : t list -> t
(** [sprod dims] is the simplified product of [dims]. The empty product
    is the constant one. *)

val broadcast_shape : t list list -> t list
(** [broadcast_shape shapes] is the common shape all of [shapes]
    broadcast to: shapes are aligned from the last axis, size-one axes
    stretch, and a zero along an axis makes the result zero there. A
    symbolic dimension broadcasts when every shape carries the same
    expression there or a constant one.

    @raise Invalid_argument if the shapes are incompatible. *)

val unbind : t -> t * int
(** [unbind u] splits a {!bind} node into its symbolic variable and
    bound integer value.

    @raise Invalid_argument if [u] is not a {!bind} of a variable to an
    integer constant. *)

(** {1:compare Comparison} *)

val compare_structure : t -> t -> int
(** [compare_structure a b] orders [a] and [b] by recursive structural
    comparison of op, dtype, arg, and children. Unlike {!compare}, the
    result is independent of hash-cons tags and therefore stable
    across runs, at the cost of worst-case traversal of the DAG. *)

val semantic_key : t -> string
(** [semantic_key u] is a digest of [u]'s op, dtype, payload, and child
    keys. It excludes hash-cons identity, {!node_tag}, and side
    {!metadata}. *)

(** {1:operators Operators}

    Infix sugar for common ALU expressions. Open locally to avoid
    shadowing the stdlib arithmetic operators.

    {[
      let open Uop.O in
      let e = (x + y) * int_ 2
    ]} *)

module O : sig
  val ( + ) : t -> t -> t
  (** [a + b] is {!alu_binary} with {!Ops.Add}. *)

  val ( * ) : t -> t -> t
  (** [a * b] is {!alu_binary} with {!Ops.Mul}. *)

  val ( - ) : t -> t -> t
  (** [a - b] is {!alu_binary} with {!Ops.Sub}. *)

  val ( / ) : t -> t -> t
  (** [a / b] is {!alu_binary} with {!Ops.Fdiv}, matching tinygrad's
      true-division shorthand. *)

  val ( // ) : t -> t -> t
  (** [a // b] is {!alu_binary} with {!Ops.Floordiv}, matching tinygrad's
      floor-division shorthand. *)

  val ( mod ) : t -> t -> t
  (** [a mod b] is {!alu_binary} with {!Ops.Floormod}. *)

  val ( < ) : t -> t -> t
  (** [a < b] is {!alu_binary} with {!Ops.Cmplt}. *)

  val cdiv : t -> t -> t
  (** [cdiv a b] is {!alu_binary} with {!Ops.Cdiv}, the truncating
      C-style division op. *)

  val cmod : t -> t -> t
  (** [cmod a b] is {!alu_binary} with {!Ops.Cmod}, the truncating
      C-style remainder op. *)

  val floordiv : t -> t -> t
  (** [floordiv a b] is {!alu_binary} with {!Ops.Floordiv}. *)

  val floormod : t -> t -> t
  (** [floormod a b] is {!alu_binary} with {!Ops.Floormod}. *)

  val ne : t -> t -> t
  (** [ne a b] is {!alu_binary} with {!Ops.Cmpne}. *)

  val where : t -> t -> t -> t
  (** [where c t e] is {!alu_ternary} with {!Ops.Where}. *)

  val neg : t -> t
  (** [neg a] is {!alu_unary} with {!Ops.Neg}. *)

  val not_ : t -> t
  (** [not_ a] is [ne a (const_bool true)], matching tinygrad's
      logical-not UOp form after boolean casting. *)

  val cast : Dtype.t -> t -> t
  (** [cast dt a] is {!Uop.cast} with [~src:a ~dtype:dt]; argument
      order is swapped for point-free use. *)

  val int_ : int -> t
  (** [int_ n] is {!Uop.const_int}. *)

  val float_ : float -> t
  (** [float_ x] is {!Uop.const_float}. *)

  val bool_ : bool -> t
  (** [bool_ b] is {!Uop.const_bool}. *)
end

(** {1:fmt Formatting} *)

val pp : Format.formatter -> t -> unit
(** [pp ppf u] formats [u] as a nested prefix expression
    ["OP:dtype(child0, child1, ...)"]. Traverses the DAG as a tree and
    may emit shared subterms multiple times; suitable for small terms and
    debugging. For stable graph listings, use {!Render.uops_to_string}. *)
