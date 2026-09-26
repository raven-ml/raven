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

(** Schedule options attached to kernel metadata. Axes are absolute indices
    in the scheduler's current, ordered range list. Tensor-core [axis] selects
    an eligible matrix-axis triple instead. Constructor order determines the
    total ordering used by beam search. *)
module Opt : sig
  type t =
    | Tc of { axis : int; tc_select : int; tc_opt : int; use_tc : int }
        (** Tensor-core configuration. *)
    | Split of { axis : int; amount : int; kind : Axis_type.t; top : bool }
        (** Split [axis] into [amount] lanes of [kind]: Upcast, Unroll or Local.
            [amount = 0] takes the whole axis; otherwise it must exceed one.
            [top] takes the outer portion instead of the inner portion. *)
    | Padto of { axis : int; amount : int }
        (** Pad [axis] to a multiple of [amount], which must exceed one. *)
    | Swap of { axis : int; with_axis : int }
        (** Swap two global axes in the schedule. *)
  (** The type for schedule options. *)

  val to_string : t -> string
  (** [to_string opt] is a compact textual form of [opt]. *)

  val pp : Format.formatter -> t -> unit
  (** [pp] formats options with {!to_string}. *)

  val axis : t -> int
  (** [axis opt] is the axis carried by [opt]. *)

  val amount : t -> int option
  (** [amount opt] is the split or padding amount, or [None] for {!Tc} and {!Swap}. *)

  val with_amount : t -> int -> t
  (** [with_amount opt n] replaces the split or padding amount by [n].
      Returns [opt] unchanged for {!Tc} and {!Swap}. *)
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
  size : int option;
      (** Maximum flat storage size, or [None] for a scalar parameter. *)
  image : (int * int) option;
      (** Image height and width; its storage shape is [(height, width, 4)]. *)
  vmin_vmax : (Bound.t * Bound.t) option;
      (** Exact inclusive numeric bounds, when known. Floating endpoints must
          not be NaN. *)
  multiple_of : int option;
      (** A known divisor of every value this parameter can take, when the
          producer declares one. Lets index folding discharge a remainder
          statically and treat the quotient as irreducible. [None] means no
          such promise, not [1]. *)
  name : string option;  (** Symbolic or debug name, when known. *)
  addrspace : Dtype.addr_space;
      (** Memory address space. Defaults to {!Dtype.Global}.
          {!Dtype.Alu} denotes ALU symbolic parameters. *)
  device : device option;  (** Concrete or multi-device placement. *)
  volatile : bool;
      (** Preserve individual memory accesses and emit volatile parameters.
          Defaults to [false]; does not provide atomicity or synchronization. *)
  bind_on_realize : bool;
      (** Bind this ALLOC to persistent tensor storage during bufferization. *)
  allocation : (string * string) option;
      (** Backend name and serialized allocation descriptor, interpreted at link time. *)
  buffer : Storage.t list option;
      (** Storage owned by a global BUFFER, one buffer per device. Parameters
          and kernel-local buffers do not own runtime storage. *)
}
(** Payload for {!Ops.Param}, {!Ops.Buffer}, and {!Ops.Alloc}. *)

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

type queue_info = {
  fallback : t list; (** Original calls, parameterized by the submission arguments. *)
  devices : string list; (** Devices submitted by the host program. *)
  host : string; (** Device executing the host program. *)
  table : int; (** Call argument containing runtime addresses, or [-1]. *)
  inputs : (int * string) list; (** Source argument and target address space per table row. *)
  outputs : int list; (** Written arguments of the original calls. *)
  timings : (string * string * int * int * int) list; (** Device, queue, argument slot and start/end uint64 word offsets. *)
  independent_accesses : (int * int) list; (** Argument pairs that must not overlap at replay. *)
  host_deps : (string * string) list; (** Memory owner and submitting device outside the batch timelines. *)
  accesses : int list list; (** Argument slots touched by each original dispatch, in order. *)
}
(** Metadata for compiled hardware-queue submission. *)

(** The collective a precompiled call implements over its (dst, src)
    arguments, with what a backend needs to replace its body. *)
type collective =
  | Allreduce of Ops.t
      (** Every device's [dst] is the reduction with the op of every
          device's [src]. *)
  | Allgather of int list
      (** Every target's [dst] is the whole value whose shards the source
          devices' [src] hold, split along these axes in device order. *)
  | Reducescatter of Ops.t * int
      (** Device k's [dst] is block k, along the axis, of the reduction with
          the op of every device's [src]. *)

val collective_name : collective -> string
(** [collective_name c] is [c]'s name: ["allreduce"], ["allgather"] or
    ["reducescatter"]. *)

(** What names a call: a free label, or the collective a precompiled call
    implements. *)
type call_name =
  | Label of string  (** A callable name for debugging. *)
  | Collective of collective  (** The collective the call implements. *)

type call_info = {
  grad_fxn : grad_fxn option;  (** Custom gradient callback, if any. *)
  name : call_name option;  (** What names the call, if anything. *)
  precompile : bool;
      (** [true] to precompile the forward callee. *)
  precompile_backward : bool;
      (** [true] to precompile the backward callee. *)
  aux : queue_info option;  (** Compiled queue submission metadata. *)
  dtype : Dtype.t;  (** Scalar return dtype, or {!Dtype.void} for effects. *)
}
(** Result type and scheduling attributes of a {!Ops.Call} node. *)

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
  target : Target.t;  (** Resolved compilation target. *)
  global_size : launch_dim list;  (** Global launch dimensions. *)
  local_size : launch_dim list;  (** Declared local launch dimensions. *)
  vars : t list;  (** Runtime symbolic parameters. *)
  globals : int list;  (** Global buffer slots. *)
  outs : int list;  (** Output buffer slots. *)
  ins : int list;  (** Input buffer slots. *)
}
(** Tinygrad-shaped program metadata attached to {!Ops.Program}. *)

val sanitize_function_name : string -> string
(** [sanitize_function_name name] rewrites [name] to a valid C identifier:
    characters outside [[A-Za-z0-9_]] are replaced by the uppercase
    hexadecimal of their code point. *)

val kernel_function_name : kernel_info -> string
(** [kernel_function_name info] is [info.name] sanitized for backend
    function emission. *)

val program_var_name : t -> string option
(** [program_var_name u] is the name carried by a parameter or buffer, if any. *)

val program_function_name : t -> string
(** [program_function_name program] is the name of its kernel, sanitized for
    backend function emission. An unnamed kernel uses ["test"], as in rendering.
    Raises [Invalid_argument] unless [program] is a PROGRAM with a SINK body. *)

val program_info_from_sink : ?target:Target.t -> t -> program_info
(** [program_info_from_sink ?target sink] derives tinygrad-style program metadata
    from [sink]. [target] defaults to an unspecified target. It scans [sink]'s topological order for ALU {!Ops.Param}
    runtime variables, non-ALU {!Ops.Param} global buffer slots, load/store
    buffer slots, and {!Ops.Special} launch dimensions.

    The resulting [vars], [globals], [outs], and [ins] are deduplicated
    and sorted. If no reads or writes can be inferred, every global buffer
    is conservatively treated as both an input and an output.

    Raises [Invalid_argument] if a launch axis is outside the three tinygrad
    launch dimensions. *)

val program_launch_dims :
  program_info -> var_vals:(string * int) list ->
  launch_value list * launch_value list
(** [program_launch_dims info ~var_vals] resolves [info.global_size] and
    [info.local_size] using [var_vals]. Symbolic dimensions are
    evaluated as integer UOp expressions over named runtime variables.

    Raises [Invalid_argument], naming the variable, if a symbolic
    dimension references a missing variable. Raises [Not_found] for an
    expression outside the UOp-local evaluator. *)

val program_vals : program_info -> var_vals:(string * int) list -> int list
(** [program_vals info ~var_vals] is the runtime argument tuple for
    [info.vars], in their declared order, resolved from [var_vals].

    Raises [Invalid_argument], naming the variable, if a
    variable has no supplied value. *)

type wmma_info = {
  dims : int * int * int;  (** Matrix dimensions [(M, N, K)]. *)
  dtype_in : Dtype.t;
      (** Input operand scalar type. Carried rather than read off [src.(0)],
          which bitcast rewrites are free to retype. *)
  threads : int;  (** Warp thread count. *)
  tc_upcast_axes :
    ((int list * int) list * (int list * int) list * (int list * int) list) option;
      (** [(axis_id, amount)] pairs for the [A]/[B]/[C] operands, pending
          expansion. [None] once the operands have been contracted, which is
          what marks the node as already expanded.

          The output type is not carried here: it is the node's own dtype. *)
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
    | Dtype of Dtype.t  (** Destination dtype for casts and bitcasts. *)
    | Typed of string * Dtype.t  (** Custom instruction text and result dtype. *)
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
        (** Placement and lifetime of a kernel stage. Bare tensor stages
            carry {!Empty}. *)
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
(** View of an {!Ops.Call} node: callee body,
    positional arguments, and call annotations. *)

type special_view = { name : string; size : t }
(** View of an {!Ops.Special} node: raw hardware index name and its upper
    bound. *)

type bind_view = { var : t; value : t }
(** View of a variable binding effect: scalar variable and stored constant. *)

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

val axis_id : t -> int list
(** [axis_id r] is the full identity of range [r]: its root axis followed by
    every split component. The range kind is not part of its identity.

    Raises [Invalid_argument] if [r] is not a range. *)

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

val as_param : t -> param_view option
(** [as_param u] matches {!Ops.Param} nodes carrying {!Param_arg}. *)

val as_buffer : t -> buffer_view option
(** [as_buffer u] matches {!Ops.Buffer} nodes carrying {!Param_arg}. *)

val as_wmma : t -> wmma_view option
(** [as_wmma u] matches {!Ops.Wmma}. *)

val as_call : t -> call_view option
(** [as_call u] matches {!Ops.Call}. *)

val as_special : t -> special_view option
(** [as_special u] matches {!Ops.Special}. *)

val as_bind : t -> bind_view option
(** [as_bind u] matches [AFTER(var, STORE(var, CONST))]. *)

val is_variable : t -> bool
(** [is_variable u] is true for a ranged scalar BUFFER in ALU address space. *)

val is_bound_var : t -> bool
(** [is_bound_var u] is true when {!as_bind} recognizes a binding effect. *)

val as_kernel_info : t -> kernel_info option
(** [as_kernel_info u] is [Some ki] when [u] is an {!Ops.Sink} carrying
    kernel metadata, and [None] otherwise. *)

val as_call_info : t -> call_info option
(** [as_call_info u] is [Some info] when [u] is an {!Ops.Call},
    and [None] otherwise. *)

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

val without_after : t -> t
(** [without_after u] removes outer {!Ops.After} dependency wrappers. *)

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
  slot:int -> dtype:Dtype.t -> ?shape:t -> ?image:int * int -> ?device:device ->
  ?vmin_vmax:Bound.t * Bound.t -> ?multiple_of:int -> ?name:string ->
  ?addrspace:Dtype.addr_space -> ?axis:int -> ?volatile:bool -> unit -> t
(** [param ~slot ~dtype ?shape ?image ?device ?vmin_vmax ?multiple_of ?name
    ?addrspace ?axis ?volatile ()]
    is a flat {!Ops.Param} with no shape child, viewed at [shape]. Multidimensional
    shapes add a reshape; symbolic extents shrink the maximum-sized storage.
    With [axis], [shape] is the full logical shape; storage holds a single
    shard and an UNSHARD view carries the device range.
    Omitting [shape] creates a scalar. [image] creates an image parameter with
    explicit height and width. Shared. *)

val variable :
  name:string -> min_val:int -> max_val:int -> ?dtype:Dtype.t ->
  ?multiple_of:int -> ?param:bool -> unit -> t
(** [variable ~name ~min_val ~max_val ?dtype ?multiple_of ?param ()] is a scalar
    {!Ops.Buffer} in {!Dtype.Alu} address space. [param = true] creates the
    kernel-side {!Ops.Param} form. [dtype] defaults to
    {!Dtype.weakint}. [multiple_of] declares a known divisor of every value
    the variable can take and defaults to [1], which every integer divides;
    see {!param_arg}. Shared. *)

val buffer :
  slot:int -> dtype:Dtype.t -> ?shape:t -> ?name:string ->
  ?addrspace:Dtype.addr_space -> ?axis:int -> ?device:device ->
  ?volatile:bool -> unit -> t
(** [buffer ~slot ~dtype ?shape ?name ?addrspace ?axis ?device ?volatile ()] is a
    flat {!Ops.Buffer} viewed at [shape]. Its maximum storage size lives in
    {!param_arg}; placed global buffers own their storage directly. With [axis],
    [shape] is the full logical shape and each device owns a single shard. Tensor. *)

val alloc :
  slot:int -> dtype:Dtype.t -> ?shape:t -> ?device:device ->
  ?bind_on_realize:bool -> unit -> t
(** [alloc ~slot ~dtype ?shape ?device ?bind_on_realize ()] declares unbound
    global storage. Scheduling gives each invocation a fresh owner unless
    [bind_on_realize] (default [false]) requests persistent tensor storage.

    @raise Invalid_argument if [dtype] is weak. *)

val from_buffer : Storage.t -> t
(** [from_buffer b] is a flat BUFFER retaining [b], including its external
    backing and view owner. Repeated calls with [b] return the same node. *)

val fresh_buffer_slot : unit -> int
(** [fresh_buffer_slot ()] draws the next process-unique buffer slot for
    graph construction. Bound storage also participates in node identity. *)

val reserve_buffer_slots : int -> unit
(** [reserve_buffer_slots n] raises the {!fresh_buffer_slot} counter so that
    subsequent draws are at least [n]. Call it before allocating alongside a
    graph whose buffers were numbered by hand. *)

val stage : src:t -> ranges:t list -> opts:stage_opts -> t
(** [stage ~src ~ranges ~opts] materialises [src] into a staged value
    indexed by loop [ranges]. Dtype is inherited from [src]. Kernel. *)

(** {2:ctors_scalars Variables, binds, constants} *)

val bind : var:t -> value:t -> t
(** [bind ~var ~value] stores a constant into a scalar variable and returns
    its AFTER effect. The variable supplies the dtype; the stored constant
    remains weak.

    @raise Invalid_argument if [var] is not a variable, [value] is not a
    constant, or its value violates the variable's bounds or divisor. *)

val const : Const.t -> t
(** [const v] is [v] as a weak CONST, with a typed CAST for concrete numeric
    dtypes. Boolean and Invalid constants remain bare. Shared. *)

val as_const : t -> Const.t option
(** [as_const u] reads a bare CONST or a CAST of a CONST at its stated dtype. *)

val ccast : src:t -> dtype:Dtype.t -> t
(** [ccast ~src ~dtype] converts a bare constant's payload before stating its
    dtype. Other inputs receive a regular cast. *)

val cconst : Const.t -> Dtype.t -> t
(** [cconst value dtype] forces a CAST around [value], including boolean
    literals, for the final program representation. *)

val const_of_dtype : ?shape:t -> Dtype.t -> const_value -> t
(** [const_of_dtype ?shape dtype value] is a constant node for [value] at
    [dtype]. Scalar values use {!const}. Tuple values produce a
    {!Ops.Stack} of scalar constants with lane dtype [dtype]; the tuple length
    is the lane count.

    If [shape] is supplied and is not scalar, the result is reshaped from
    singleton dimensions and expanded to [shape]. *)

val invalid : unit -> t
(** [invalid ()] is the [Invalid] sentinel expressed as a {!Dtype.bool}
    [Const]. It is the bottom of the promotion lattice and satisfies whatever
    dtype its consumer demands. Shared. *)

val const_int : int -> t
(** [const_int n] is a {!Dtype.weakint} integer constant, used for shape
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

val promo_dtype : t list -> Dtype.t
(** [promo_dtype srcs] is the dtype an elementwise operation over [srcs]
    produces: the dtype they share if they all carry the same one, and
    {!Dtype.least_upper_dtype} of them otherwise.

    The shared-dtype case does not consult the lattice, so operands whose
    dtype has no upper bound with anything — pointers, {!Dtype.Void} — are
    carried through rather than rejected.

    @raise Invalid_argument if [srcs] is empty, or if the operands have no
    common upper bound. *)

val alu_unary : op:Ops.t -> src:t -> t
(** [alu_unary ~op ~src] is a unary ALU node. Dtype is inherited from
    [src], except that the transcendental ops ({!Ops.Sin}, {!Ops.Log2},
    {!Ops.Exp2}, {!Ops.Sqrt}, {!Ops.Reciprocal}) promote an integer argument
    to floating point via {!Dtype.least_upper_float}. Shared.

    @raise Invalid_argument if [op] is not in {!Ops.Group.unary}. *)

val alu_binary : op:Ops.t -> lhs:t -> rhs:t -> t
(** [alu_binary ~op ~lhs ~rhs] is a binary ALU node. Dtype is
    {!promo_dtype} of the two operands, so a mixed-dtype operation widens
    to their least upper bound rather than favouring either side. Two
    exceptions: comparisons produce a bool of matching vector width, and a
    shift ({!Ops.Shl}, {!Ops.Shr}) keeps [lhs]'s dtype, since the shift
    amount's width does not affect the result's. Shared.

    @raise Invalid_argument if [op] is not in {!Ops.Group.binary}, if [op]
    is a comparison on a pointer dtype, if [op] is a shift and either
    operand is not an integer, or if the operands have no common upper
    bound. *)

val alu_ternary : op:Ops.t -> a:t -> b:t -> c:t -> t
(** [alu_ternary ~op ~a ~b ~c] is a ternary ALU node. Dtype is
    {!promo_dtype} of the value operands — [b] and [c] for {!Ops.Where},
    whose condition [a] does not take part in the promotion, and all three
    otherwise. Shared.

    @raise Invalid_argument if [op] is not in {!Ops.Group.ternary}, if [op]
    is {!Ops.Where} and [a] is not bool, or if the promoted operands have
    no common upper bound. *)

val valid : src:t -> cond:t -> t
(** [valid ~src ~cond] is [where cond src invalid]: it masks [src] to the
    {!Const.invalid} sentinel wherever [cond] is false. Used to gate index
    expressions. Dtype is inherited from [src]. *)

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

val getaddr : ?device:string -> src:t -> unit -> t
(** [getaddr ?device ~src ()] extracts the address of [src] in [device],
    defaulting to its owning device. Its dtype is {!Dtype.uint64}. *)

val broadcast : t -> int -> t
(** [broadcast u n] repeats [u] into an [n]-wide {!Ops.Stack}. Returns
    [u] unchanged when [n <= 1]. Shared. *)

val is_invalid_const : t -> bool
(** [is_invalid_const u] is [true] iff [u] is an {!Ops.Const} carrying the
    {!Const.invalid} sentinel. *)

val get_idx : t -> t
(** [get_idx u] recovers the index expression from a possibly-gated index. A
    gate is a [where cond idx invalid]; [get_idx] returns [idx] there, recurses
    lane-wise through an {!Ops.Stack}, and is the identity otherwise.

    @raise Invalid_argument if [u] is not an integer index expression. *)

val get_valid : t -> t
(** [get_valid u] recovers the boolean guard of a possibly-gated index: [cond]
    for a [where cond idx invalid], recursed lane-wise through an
    {!Ops.Stack}; a bare {!Const.invalid} yields false and any other
    expression yields true.

    @raise Invalid_argument if [u] is not an integer index expression. *)

(** {2:ctors_control Control flow}

    {!Ops.Range}: [src = \[| size; parent0; parent1; ... |\]].

    {!Ops.End}: [src = \[| value; range0; range1; ... |\]].

    {!Ops.If}: [src = \[| cond; idx_for_dedup |\]].

    {!Ops.Endif}: [src = \[| if_ |\]]. *)

val range :
  size:t -> axis:int -> kind:Axis_type.t -> ?sub:int list ->
  ?dtype:Dtype.t -> ?parents:t list -> unit -> t
(** [range ~size ~axis ~kind ?sub ?dtype ?parents ()] is a loop variable
    over \[[0];[size-1]\] bound to schedule [axis] with semantic [kind]
    (see {!Axis_type}). [sub] defaults to [[]]. [dtype] defaults to
    {!Dtype.weakint}. [parents] defaults to [[]] and lists the outer
    {!range} nodes this loop must be emitted under, used as
    control-flow ordering dependencies. Shared. *)

val loop : axis:int -> t
(** [loop ~axis] is an unbounded, void loop header with identifier [axis]. *)

val backedge : body:t -> loop:t -> cond:t -> t
(** [backedge ~body ~loop ~cond] executes [body] and repeats the unbounded
    [loop] while the scalar boolean [cond] is true. It produces no value
    and closes only [loop]; enclosing ranges used by [cond] remain live. *)

val end_ : value:t -> ranges:t list -> t
(** [end_ ~value ~ranges] closes bounded loop [ranges] around the void
    effect [value], preserving its shape. [value] must have void dtype and
    each range must have integer dtype. Use {!backedge} for unbounded loops.
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

val special : name:string -> size:t -> ?dtype:Dtype.t -> unit -> t
(** [special ~name ~size ?dtype ()] is a backend-provided hardware index
    named [name] and bounded by [size], ranging over \[[0];[size-1]\].
    [size] is cast to [dtype]. [dtype] defaults to {!Dtype.weakint}.
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

val unshard : ?ranges:t list -> src:t -> axes:int list -> unit -> t
(** [unshard ~src ~axes ?ranges ()] assembles the logical shape of [src]'s
    shards. Axes are sorted together with their ranges. Each range's maximum
    plus one determines its shard count. Without [ranges], a single axis uses
    a DEVICE range over [src]'s devices.
    @raise Invalid_argument if axes repeat or their count differs from ranges. *)

val sharding : t -> (int * t) list
(** [sharding u] is the sorted list of axes and their owning ranges for an
    UNSHARD, or the empty list for another operation. *)

val mstack : t list -> t
(** [mstack srcs] stacks per-device shards into a multi-device tensor.
    Dtype is inherited from the first shard. Tensor.

    @raise Invalid_argument if [srcs] is empty. *)

val mselect : src:t -> index:int -> t
(** [mselect ~src ~index] selects shard [index] of a multi-device
    [src]. Dtype is inherited from [src]. Tensor. *)

val copy : src:t -> device:device -> unit -> t
(** [copy ~src ~device ()] copies [src] to [device]. Dtype is
    inherited from [src]. Tensor.

    @raise Invalid_argument if [src] has a weak dtype or [device] contains
    a disk destination. Use an explicit store to write disk storage. *)

(** {2:ctors_movement Movement}

    Tensor-stage only. {!Ops.Reshape}: [src = \[| input; shape |\]];
    {!Ops.Expand}: [src = \[| input; dims |\]]. {!Ops.Pad} and {!Ops.Shrink}:
    [src = \[| input; offset; size |\]], where [offset] is the per-axis
    start offset and [size] is the resulting per-axis output size. *)

val reshape : src:t -> shape:t -> t
(** [reshape ~src ~shape] rearranges the elements of [src] into
    [shape] without changing the total count. Dtype is inherited from
    [src]. It is [src] itself when [shape] is [src]'s shape. Tensor. *)

val expand : src:t -> dims:t -> t
(** [expand ~src ~dims] prepends [dims] as new leading axes of [src]: the
    result shape is [dims] followed by [src]'s shape. This is the primitive
    add-leading-dims op; whole-shape broadcasting is composed from it by
    {!broadcast_to}. Returns [src] unchanged when [dims] is the empty shape.
    Dtype is inherited from [src]. Tensor. *)

val broadcast_to : src:t -> shape:t -> t
(** [broadcast_to ~src ~shape] broadcasts [src] to [shape] by the array
    broadcasting rules: [src]'s shape is right-aligned under [shape] (missing
    leading axes are treated as size one), and each existing axis must either
    match its target or be size one (in which case it stretches). Built from
    the primitives: the size-one axes that must grow are squeezed out with
    {!reshape}, prepended at their target sizes with {!expand}, and permuted
    back into [shape]'s order. Returns [src] unchanged when its shape already
    equals [shape]. Dtype is inherited from [src].

    @raise Invalid_argument if [src] has more axes than [shape], or if an axis
    neither matches nor is size one. Tensor. *)

val pad : src:t -> offset:t -> size:t -> t
(** [pad ~src ~offset ~size] pads [src] with zeros to the per-axis output
    size [size], offsetting the input by [offset]. Dtype is inherited from
    [src]. Tensor. *)

val shrink : src:t -> offset:t -> size:t -> t
(** [shrink ~src ~offset ~size] slices [src] at per-axis offsets [offset]
    with per-axis sizes [size]. Dtype is inherited from [src]. Tensor.

    Offsets and sizes may be symbolic. Shape inference rejects a bound only
    when it is provably out of range; a symbolically-undecidable offset (whose
    out-of-range accesses are masked by a surrounding gate) is accepted. *)

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

val contiguous : src:t -> ?force:bool -> unit -> t
(** [contiguous ~src ?force ()] forces [src] into contiguous layout.
    Schedule options live on the enclosing
    {!sink}'s {!kernel_info}, not here. Dtype is inherited from [src].
    Returns [src] unchanged for duplicate bare {!Ops.Stage} sources and
    for buffer-identity sources ({!Ops.Buffer}, {!Ops.Alloc},
    {!Ops.Param}), unless [force] is [true].
    Tensor. *)

val contiguous_backward : src:t -> t
(** [contiguous_backward ~src] is a backward-pass contiguous marker.
    Dtype is inherited from [src]. Tensor. *)

val call : body:t -> args:t list -> info:call_info -> t
(** [call ~body ~args ~info] invokes an opaque effect body with explicit
    arguments. Its return dtype is [info.dtype]. Use {!call_with_outputs} for
    bodies that produce tensor values.

    @raise Invalid_argument if [body] is not an opaque effect body or has
    in-scope ranges other than device ranges. *)

val param_like : t -> slot:int -> t
(** [param_like u ~slot] is a call formal with [u]'s shape and placement.
    Variables become scalar ALU formals with positional names. *)

val store_call : dst:t -> src:t -> t
(** [store_call ~dst ~src] is an executable bulk transfer into [dst]. Its
    body stores between two formal buffers, bound to [dst] and [src]. *)

val call_with_outputs :
  ?output_pos:int list -> values:t list -> args:t list -> info:call_info ->
  unit -> t list
(** [call_with_outputs ?output_pos ~values ~args ~info ()] calls a body that
    stores [values] into explicit output arguments. Returns fresh allocation
    views sequenced after the call. Output positions default to the slots
    following the inputs; explicit positions must be distinct and ascending.
    Symbolic result shapes substitute input arguments by their final slots.

    @raise Invalid_argument if output positions are invalid or outputs have
    weak dtypes. *)

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

val placeholder :
  shape:int list -> dtype:Dtype.t -> slot:int -> ?addrspace:Dtype.addr_space ->
  ?device:device -> ?volatile:bool -> ?allocation:(string * string) -> unit -> t
(** [placeholder ~shape ~dtype ~slot ?addrspace ?device ?volatile ?allocation ()] is storage for
    [shape] elements of [dtype] that a kernel body addresses before any buffer
    is bound to it. The storage is flat, holding the product of [shape], and a
    {!reshape} restores a [shape] of rank above one. A weak [dtype] commits to
    its default width. [addrspace] defaults to {!Dtype.Global}, which gives a
    {!Ops.Param}; {!Dtype.Local} and {!Dtype.Reg} give an {!Ops.Buffer}.
    [volatile] defaults to [false] and applies to global parameters.
    [allocation] carries a backend allocation descriptor for the link phase.

    @raise Invalid_argument
      if [addrspace] is {!Dtype.Alu}, or if [device] is given for a local or
      register placeholder. *)

val placeholder_like : t -> slot:int -> ?addrspace:Dtype.addr_space -> unit -> t
(** [placeholder_like u ~slot ?addrspace ()] is a {!placeholder} with [u]'s
    dtype and per-device shape.

    @raise Invalid_argument if [u]'s shape is symbolic. *)

val custom_kernel : ?grad_fxn:grad_fxn -> fxn:(t list -> t) -> t list -> t list
(** [custom_kernel ?grad_fxn ~fxn srcs] runs a kernel written in uops over
    [srcs]. [fxn] receives one {!placeholder_like} per source, slot [i] for the
    [i]-th source, and returns the kernel's {!sink}. The result lists every
    source {!after} the {!call} of that kernel, in order: read a source the
    kernel writes through its entry here. Sources are realized before the
    kernel runs. Tensor.

    @raise Invalid_argument if a source has a symbolic shape. *)

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
    [name] with [src = srcs]. With a single function-pointer source, it is
    the body of an indirect host {!call}. The body has void dtype; the
    call’s result dtype is declared in {!call_info}. *)

(** {2:ctors_replace Replace}

    There is no public [mk] escape hatch; every op must be built through
    a dedicated smart constructor above. This keeps per-op invariants
    and src layout centralised. *)

val replace :
  t -> ?op:Ops.t -> ?src:t array -> ?arg:arg ->
  ?node_tag:string option -> unit -> t
(** [replace u ?op ?src ?arg ?node_tag ()] rebuilds [u] with
    the supplied fields overridden and the rest inherited from [u].
    Pass [~node_tag:None] to clear the diagnostic tag; omit it to
    preserve it. The result is hash-consed, so it is physically equal
    to [u] when every override matches the existing field. Result dtypes are
    derived from the new sources and argument. To change a cast, storage or
    custom instruction's dtype, replace its typed payload explicitly.

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

    [enter_calls] defaults to [true]; when [false], {!Ops.Call} bodies (i.e. [src.(0)]) are not entered, but their
    argument children are. *)

val topovisit : (t -> 'a) -> (int, 'a) Hashtbl.t -> t -> 'a
(** [topovisit visitor cache root] folds over the DAG rooted at [root] in
    dependency order, leaves first, applying [visitor] to each node exactly
    once and memoizing the result in [cache] (keyed by {!tag}). A subtree
    whose root is already in [cache] is not re-descended, so successive calls
    sharing one [cache] short-circuit shared work across roots. Returns the
    result computed for [root]. *)

val backward_slice : t -> t list
(** [backward_slice root] is [toposort root] without [root] itself.
    The ordered slice and its membership index are memoized per domain;
    cached properties do not keep an otherwise unreachable graph alive. *)

val find_nodes : (t -> bool) -> t -> t list
(** [find_nodes p root] is the nodes of the DAG rooted at [root] that
    satisfy [p], in topological order. *)

val in_backward_slice : t -> t -> bool
(** [in_backward_slice needle haystack] is [true] iff [needle] occurs
    strictly before [haystack] in [toposort haystack]. *)

val bool_slice_mem : t -> t -> bool
(** [bool_slice_mem root u] is [true] iff [u] is bool-typed and reachable
    from [root], counting [root] itself.

    This is a reachability test restricted to the one dtype a condition can
    have, which is what makes it cheap enough to run per node: the bool
    subgraph is a small fraction of a kernel. Memoized on [root]. *)

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
    a range (e.g. {!Ops.Reduce}, {!Ops.Stage}, {!Ops.End}, {!Ops.Backedge},
    {!Ops.Wmma}, {!Ops.Call}, {!Ops.Copy}) drop ended ranges from the
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

val on_disk : t -> bool
(** [on_disk u] is [true] when [u] resides on one disk device. *)

val is_virtual : t -> bool
(** [is_virtual u] is [true] iff [u] cannot back a buffer as it stands:
    either it has no {!device_of} — nowhere to store it — or its dtype is
    weak ({!Dtype.is_weak}) — no committed width to store it at. Realizing
    such a node requires placing it on a device and committing its width
    first. *)

val addrspace : t -> Dtype.addr_space option
(** [addrspace u] is [u]'s tinygrad-style address-space property.
    {!Ops.Param} and {!Ops.Buffer} read their {!param_arg}. {!Ops.Special},
    {!Ops.Range}, and {!Ops.Load} are in {!Dtype.Alu}. Index-like,
    movement, reduction, and shard-selection nodes inherit from their first
    source. Elementwise and stack-like nodes report an address space only when
    all address-spaced sources agree. *)

val base : t -> t
(** [base u] walks through movement ops and {!Ops.Detach} to the
    underlying node. Other ops, including {!Ops.Unshard}, {!Ops.Stage},
    {!Ops.Param}, and {!Ops.Buffer}, are their
    own base. *)

val storage_base : t -> t
(** [storage_base u] is the node targeted by [u]'s storage views. It walks
    through movement ops, {!Ops.Detach}, {!Ops.Bitcast}, {!Ops.After}, and
    {!Ops.Unshard}, stopping at storage identities or pending computations. *)

val buf_uop : t -> t
(** [buf_uop u] is the buffer-identity node reached by following tinygrad's
    buffer property rules. {!Ops.Param} and {!Ops.Buffer} return themselves;
    {!Ops.Stage} and {!Ops.Mstack}
    stop the walk. *)

val has_buffer_identity : ?after_ok:bool -> t -> bool
(** [has_buffer_identity ?after_ok u] is [true] iff [u] is a concrete graph
    buffer identity: {!Ops.Param}, {!Ops.Buffer}, {!Ops.Alloc}, or those
    identities through {!Ops.Reshape}, {!Ops.Unshard}, or {!Ops.Mselect}. With [after_ok] (default
    [false]) an {!Ops.After} over such an identity also qualifies. *)

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

    Raises [Invalid_argument] if [u] has no tensor shape. *)

val shape_opt : t -> t list option
(** [shape_opt u] is [Some (shape u)] when [u] has a shape and [None] when it
    does not, never raising. Memoised like {!shape}. *)

val max_shape : t -> int list
(** [max_shape u] is {!shape} with every symbolic dimension replaced by
    its conservative upper bound. *)

val max_numel : t -> int
(** [max_numel u] is the number of scalar lanes [u] holds: the product of
    {!max_shape}.

    Multiplication is exact before converting the result to a host integer,
    so a zero dimension yields zero even when another dimension is larger
    than the host integer range.

    Raises [Invalid_argument] if [u] has no tensor shape or the product does
    not fit in a host integer. *)

val shard_shape : t -> t list
(** [shard_shape u] is the local source shape of an UNSHARD. For another
    multi-device tensor with a known single sharding axis, it divides that
    logical axis by the device count. Otherwise it is [shape u]. *)

val max_shard_shape : t -> int list
(** [max_shard_shape u] is {!shard_shape} with every symbolic dimension
    replaced by its conservative upper bound. *)

val max_shard_numel : t -> int
(** [max_shard_numel u] is the product of the upper bounds of {!shard_shape}.
    Multiplication and host-range checks have the same contract as
    {!max_numel}. *)

val axis : t -> int option
(** [axis u] is [u]'s sharding axis. {!Ops.Param} has no axis;
    a single-axis {!Ops.Unshard} reads its axis tuple, {!Ops.Copy} clears the axis,
    ALU ops use the last
    non-[None] source axis, and movement/reduction ops remap or clear the
    axis using tinygrad's shape rules. Raises [Invalid_argument] for an
    UNSHARD with multiple axes; use {!sharding} to inspect those axes. *)

val bounds : t -> (t * t) list
(** [bounds u] is the per-device shard interval on [u]'s sharding axis.

    Raises [Invalid_argument] if [u] has no sharding axis or no multi-device
    placement. *)

val contiguous_view : t -> (t * int) option
(** [contiguous_view u] is the underlying storage node and byte offset when
    [u] is a contiguous view. The offset must be statically known; leading
    dimensions may have bounded symbolic lengths. Returns [None] if the
    layout or offset cannot be established from the graph. *)

val storage_window : t -> (t * int) option
(** [storage_window u] is the storage [u] reads or writes and [u]'s byte
    offset in it, when [u] is that storage or a contiguous window of it.
    Storage is a {!Ops.Buffer}, {!Ops.Alloc}, {!Ops.Param}, {!Ops.Mselect}
    or {!Ops.Mstack}, reached through movement ops, bitcasts and
    {!Ops.After}, or an empty-arg {!Ops.Stage}, which becomes a buffer of its
    own. It is {!contiguous_view} except that a stage counts as storage
    rather than being looked through. *)

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
    {- [enter_calls] defaults to [false]; when [false], {!Ops.Call}
       program bodies ([src.(0)]) are not rewritten. Native callees represented
       by {!Ops.Custom_function} contain caller-scope pointer expressions and
       are always rewritten.}
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

val substitute : ?walk:bool -> ?enter_calls:bool -> (t * t) list -> t -> t
(** [substitute ?walk ?enter_calls mappings root] rewrites [root] replacing every
    occurrence of the first component of each pair with the second.
    Bottom-up; performed via {!graph_rewrite}. [enter_calls] defaults to
    [false], keeping callee bodies opaque. Lookup in [mappings] uses physical
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
    side data is not. Bound storage is serialized as bytes and ownership
    relationships; allocator closures and native pointers are excluded.

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

    Bound buffers acquire independent storage while their bytes and shared
    base/view relationships are preserved. Unallocated buffers stay lazy.
    Kernel-local buffers and unbound placeholders remain structural nodes.

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

module Weak_tbl : Ephemeron.S with type key = t
(** Ephemeron table keyed by physical equality, hashing on {!tag}.
    Entries retain their values only while their keys are otherwise reachable,
    including when a value refers back to its key. Use for long-lived caches
    and node-owned resources. *)

(** {1:analysis Analysis} *)

val vmin : t -> Bound.t
(** [vmin u] is a conservative numeric lower bound for [u]. Integer bounds
    retain arbitrary precision; float bounds preserve infinities and exclude
    NaN. Unknown values use the dtype's limits, as do integer values whose
    arithmetic may wrap around their dtype's width. Empty intervals have a
    lower bound greater than their upper bound, as for [RANGE(0)]. Memoised. *)

val vmax : t -> Bound.t
(** [vmax u] is the upper bound symmetric to {!vmin}. *)

val commit_dtype : ?default_int:Dtype.t -> t -> Dtype.t
(** [commit_dtype u] is the concrete dtype for storing [u]. Concrete dtypes
    are unchanged and [weakfloat] becomes {!Dtype.default_float}. For
    [weakint], it selects the first of [default_int] (default:
    {!Dtype.default_int}), [int32], [int64], and [uint64] containing both
    bounds. [default_int] must be a concrete integer dtype.

    An unresolved interval that fits no candidate uses [int64]. Raises
    [Invalid_argument] if an exact integer fits neither [int64] nor [uint64]. *)

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
(** [usum xs] left-folds [xs] with {!Ops.Add}, or with {!Ops.Or} when the
    first element is boolean.
    @raise Invalid_argument if [xs] is empty. *)

val uprod : t list -> t
(** [uprod xs] left-folds [xs] with {!Ops.Mul}, or with {!Ops.And} when the
    first element is boolean.
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


val symbolic_vars : t -> (t * string * Bound.t * Bound.t) list
(** [symbolic_vars u] is the named, bounded variables [u] reaches, each as
    [(node, name, vmin, vmax)]. *)

val sym_infer : t -> (string * int) list -> int
(** [sym_infer u var_vals] is the integer [u] evaluates to once its variables
    take their values in [var_vals].

    Raises [Invalid_argument] if [u] does not reduce to a constant. *)

val exec_alu : Ops.t -> Dtype.t -> Const.t list -> Const.t option
(** [exec_alu op target args] folds ALU op [op] applied to
    constant [args], producing a constant of [target] dtype, or [None] when the
    op or operand shapes are not foldable. Any binary op with an {!Const.invalid}
    operand folds to {!Const.invalid} regardless of dtype.

    The folded value is [target]'s: a fixed-width integer wraps, as every
    {!Const.integer} does, and a weak one stays exact.

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
    comparison of op, then arg, then dtype, then children — the
    comparison order is load-bearing for schedulers that break ties
    structurally. Unlike {!compare}, the result is independent of
    hash-cons tags and therefore stable across runs, at the cost of
    worst-case traversal of the DAG. *)

val semantic_key : t -> string
(** [semantic_key u] is a digest of [u]'s op, dtype, payload, and child
    keys. It excludes hash-cons identity, {!node_tag}, and side
    {!metadata}. *)

val program_signature : program_info -> t list -> Tiny_elf.argument list
(** [program_signature info linear] extracts the compiled argument signature
    from [linear], using [info]'s buffer slots and scalar binding order.
    Symbolic dimensions use their maximum bounds.

    Raises [Invalid_argument] if [info.globals] and the linear buffer formals
    disagree, or a scalar formal is not a parameter. *)

val to_elf : t -> Tiny_elf.t
(** [to_elf program] is [program]'s binary, entry point, target and signature.
    Raises [Invalid_argument] unless [program] is a compiled {!Ops.Program}
    with a linear body and binary. *)

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

(** Promoting operators.

    The operators of tinygrad's UOp promote their operands before building
    the node, as its tensors do; {!O} builds the node as given. A rule body
    ported from tinygrad uses these where tinygrad's source uses an operator
    or [maximum]/[minimum], so a weak non-literal meeting a committed operand
    is cast rather than left as a mixed node that symbolic rules do not fold.
    Integer literals written in the reference's source are weak: pass
    {!const_int}. *)
module Promoting : sig
  val broadcasted : t -> t -> t * t
  (** [broadcasted a b] brings [a] and [b] to their {!promo_dtype} [out].
      An {!Const.invalid} operand is returned as is, a weak constant (under
      movement ops) is rebuilt at [Dtype.weak_dtype out], and any other
      operand is {!cast} to [out]. *)

  val ( + ) : t -> t -> t
  (** [a + b] is {!Ops.Add} of the {!broadcasted} operands. *)

  val ( - ) : t -> t -> t
  (** [a - b] is {!Ops.Add} of the {!broadcasted} [a] and [neg b]. *)

  val ( * ) : t -> t -> t
  (** [a * b] is {!Ops.Mul} of the {!broadcasted} operands. *)

  val ( // ) : t -> t -> t
  (** [a // b] is {!Ops.Floordiv} of the {!broadcasted} operands. tinygrad's
      float path (a floored product with the reciprocal) is not ported.

      Raises [Invalid_argument] unless both promote to an integer dtype. *)

  val ( < ) : t -> t -> t
  (** [a < b] is {!Ops.Cmplt} of the {!broadcasted} operands. *)

  val ne : t -> t -> t
  (** [ne a b] is {!Ops.Cmpne} of the {!broadcasted} operands. *)

  val xor : t -> t -> t
  (** [xor a b] is {!Ops.Xor} of the {!broadcasted} operands. *)

  val pow : t -> t -> t
  (** [pow a b] is {!Ops.Pow} of the {!broadcasted} operands. *)

  val maximum : t -> t -> t
  (** [maximum a b] is {!Ops.Max} of the {!broadcasted} operands. *)

  val minimum : t -> t -> t
  (** [minimum a b] is the smaller of the {!broadcasted} operands, built
      from {!Ops.Max}: [neg (max (neg a) (neg b))] on floats, and
      [(max (a ^ k) (b ^ k)) ^ k] on integers and booleans, where [k] is the
      sum of the dtype's bounds ([-1] for signed integers and
      {!Dtype.weakint}, all ones for unsigned ones, [true] for booleans). *)

  val neg : t -> t
  (** [neg a] is [ne a true] on booleans and [a * const_int (-1)]
      otherwise. *)
end

(** {1:fmt Formatting} *)

val pp : Format.formatter -> t -> unit
(** [pp ppf u] formats [u] as a nested prefix expression
    ["OP:dtype(child0, child1, ...)"]. Traverses the DAG as a tree and
    may emit shared subterms multiple times; suitable for small terms and
    debugging. For stable graph listings, use {!Render.uops_to_string}. *)
