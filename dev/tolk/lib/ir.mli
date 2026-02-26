(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Three-level intermediate representation for tolk.

    - {!Tensor}: scheduling-level tensor graph (shape ops, multi-device,
      autograd)
    - {!Kernel}: codegen-oriented DAG (ranges, reductions, memory indexing)
    - {!Program}: flat SSA for code emission (loops, if/endif, no graph)

    All three levels use a flat instruction-array representation where
    [type ref = int] indexes into the array. Instructions may only reference
    earlier indices (SSA dominance). Each module provides [dtype_of], [refs_of],
    [map_refs], [validate], and pretty-printing.

    Data types for elements, vectors, and pointer address spaces are defined in
    {!Dtype}. *)

(** {1:shared Shared Types} *)

(** GPU thread dimension. The int specifies the axis (0, 1, or 2 by convention;
    not enforced by the type). *)
type special_dim =
  | Group_id of int
      (** Group/block ID (blockIdx in CUDA, get_group_id in OpenCL) *)
  | Local_id of int
      (** Local/thread ID within workgroup (threadIdx in CUDA, get_local_id in
          OpenCL) *)
  | Global_idx of int
      (** Global thread ID (blockIdx * blockDim + threadIdx, or get_global_id)
      *)

val special_axis : special_dim -> int
(** [special_axis dim] is the axis index of [dim]. *)

(** Range axis kind. Determines how a [Range] dimension maps to hardware
    parallelism or loop structure. Variant declaration order encodes scheduling
    priority (used by structural comparison). Used by [Range] instructions in
    {!Kernel}, {!Tensor}, and {!Program}. *)
type axis_kind =
  | Global  (** Grid-level parallelism (maps to block/group dimensions). *)
  | Thread  (** Thread-level parallelism within a block/workgroup. *)
  | Local  (** Shared/local memory dimension for workgroup data sharing. *)
  | Warp  (** Warp/sub-group level parallelism (e.g., 32 threads on NVIDIA). *)
  | Loop  (** Sequential iteration (not parallelized). *)
  | Group_reduce
      (** Grouped reduction within a workgroup. Same scheduling priority as
          [Local]. *)
  | Reduce  (** Reduction dimension (excluded from local indexing). *)
  | Upcast
      (** Register tiling: multiple elements per thread, no sync needed. *)
  | Unroll  (** Fully unrolled loop (body duplicated, no loop overhead). *)
  | Outer  (** Outermost reduction scope spanning the entire computation. *)
  | Placeholder
      (** Transient placeholder used during reshape range rewriting. Not
          expected in finalized IR. *)

(** Codegen-oriented DAG produced by rangeify.

    This IR sits between {!Tensor} (scheduling) and {!Program} (rendering). It
    retains DAG structure with high-level operations like {!instr.Reduce},
    {!instr.Unroll}, {!instr.Contract}, and {!instr.Bufferize} that are lowered
    during linearization.

    Unlike {!Program}, this IR is not flattened into SSA. Instructions reference
    operands by index into the instruction array. Use {!intern} to deduplicate
    structurally equal instructions.

    {2 Invariants}

    Well-formed programs must satisfy:
    - All references point to earlier instructions (SSA dominance).
    - [Param] has [Global] addrspace, [Define_local] has [Local], [Define_reg]
      has [Reg].
    - [Define_var] is scalar int/index with valid bounds ([lo <= hi]).
    - [Bufferize] dtype addrspace must match its [opts.addrspace].
    - [Index] base must be a pointer definition ([Param], [Define_local],
      [Define_reg], [Bufferize], or [Ptrcat]); indices must be index scalars;
      optional gate must be bool scalar.
    - [Load] src must reference a pointer; [alt] requires a gated [Index].
    - [Store] dst must reference a pointer.
    - ALU operand dtypes must match the result dtype; comparisons return bool.
    - [Idiv]/[Mod] must have int/index dtype.
    - Shifts must have int/index dtype; rhs must match lhs dtype or be [uint32].
    - [Vectorize] source count must equal [dtype.count]; all sources scalar.
    - [Gep] result is scalar of source vector type; index in bounds.
    - [End] closes a computation scope with a value and range references.
      Distinct from {!Program.instr.End_range}, which explicitly closes a loop.
*)
module Kernel : sig
  (** {1:kernel-types Types} *)

  (** Constant literal values. Same variants as {!Tensor.const}. [Invalid]
      represents an uninitialized or undefined value (requires [Index] dtype).
  *)
  type const = Bool of bool | Int of int | Float of float | Invalid

  (** Reduction operation kind. Same variants as {!Tensor.reduce_op}. *)
  type reduce_op = Add | Mul | Max

  (** Device specification for bufferization. *)
  type bufferize_device =
    | Device_single of string  (** Target a single named device. *)
    | Device_multi of string list  (** Sharded across multiple named devices. *)
    | Device_index of int
        (** Index into a multi-device buffer's device list. *)

  (** Estimated cost metric (exact integer or symbolic expression). *)
  type estimate = Int of int | Symbolic of string

  type estimates = {
    ops : estimate;  (** Operation count (arithmetic instructions). *)
    lds : estimate;  (** Load count (memory read instructions). *)
    mem : estimate;  (** Memory traffic (bytes transferred). *)
  }
  (** Cost estimates for a kernel. *)

  type bufferize_opts = {
    device : bufferize_device option;
    addrspace : Dtype.addr_space;
    removable : bool;
        (** Whether this buffer can be optimized away if unused. [false] for
            buffers that must be materialized (e.g., before a [Copy]). *)
  }
  (** Options controlling buffer materialization. *)

  type kernel_info = {
    name : string;  (** Kernel name for debugging and profiling. *)
    axis_kinds : axis_kind list;
        (** Axis kinds for each range dimension, in dimension order. *)
    dont_use_locals : bool;
        (** When [true], inhibit shared/local memory usage for this kernel. *)
    applied_opts : string list;
        (** Optimization passes already applied to this kernel. *)
    opts_to_apply : string list option;
        (** Optimization passes still to apply. [None] means use defaults. *)
    estimates : estimates option;
        (** Cost estimates. [None] when cost information is not yet available.
        *)
    metadata_tags : string list;
        (** Deduplicated scheduler metadata names collected during rangeify
            splitting. *)
  }
  (** Kernel metadata attached to {!instr.Sink} to describe the compiled kernel.
  *)

  type ref = int
  (** Reference to another instruction by index in the instruction array. *)

  (** {1:kernel-instrs Instructions} *)

  (** Kernel IR instruction. *)
  type instr =
    | Sink of { srcs : ref list; kernel_info : kernel_info option }
        (** Program root. Collects side-effecting instructions. *)
    | Group of { srcs : ref list }
        (** Group related instructions for scheduling. *)
    | After of { src : ref; deps : ref list }
        (** Execution dependency: [src] must execute after [deps]. Unlike
            {!Tensor.instr.After}, does not carry dtype. *)
    | Param of { idx : int; dtype : Dtype.ptr }
        (** Kernel parameter at position [idx]. *)
    | Define_local of { size : int; dtype : Dtype.ptr }
        (** Allocate shared memory for [size] elements. *)
    | Define_reg of { size : int; dtype : Dtype.ptr }
        (** Allocate register storage for [size] elements. *)
    | Define_var of { name : string; lo : int; hi : int; dtype : Dtype.t }
        (** Scalar kernel parameter with runtime bounds [lo..hi]. *)
    | Bufferize of {
        src : ref;
        idx : ref;
        ranges : ref list;
        dtype : Dtype.ptr;
        opts : bufferize_opts;
      }
        (** Materialize [src] into a buffer at linear index [idx]. [ranges]
            constrain the materialization scope. Lowered during linearization.
        *)
    | Const of { value : const; dtype : Dtype.t }  (** Constant literal. *)
    | Index of {
        ptr : ref;
        idxs : ref list;
        gate : ref option;
        dtype : Dtype.ptr;
      }
        (** Pointer arithmetic: [ptr + sum(idxs)], with optional [gate]. Uses
            {!Dtype.ptr} (unlike {!Tensor.instr.Index} which uses {!Dtype.t}).
        *)
    | Ptrcat of { srcs : ref list; dtype : Dtype.ptr }
        (** Concatenate pointer vectors. Total pointer vector width of sources
            must equal [dtype.v]. *)
    | Load of { src : ref; alt : ref option; dtype : Dtype.t }
        (** Load value from memory. If [alt] is provided, the source index must
            be gated and [alt] is used when the gate is false. *)
    | Store of { dst : ref; value : ref; ranges : ref list }
        (** Store [value] to memory at [dst]. [ranges] constrain the store
            scope. Unlike {!Tensor.instr.Store} and {!Program.instr.Store},
            carries explicit range constraints. *)
    | Neg of { src : ref; dtype : Dtype.t }  (** Negation. *)
    | Exp2 of { src : ref; dtype : Dtype.t }  (** Base-2 exponential. *)
    | Log2 of { src : ref; dtype : Dtype.t }  (** Base-2 logarithm. *)
    | Sin of { src : ref; dtype : Dtype.t }  (** Sine. *)
    | Sqrt of { src : ref; dtype : Dtype.t }  (** Square root. *)
    | Recip of { src : ref; dtype : Dtype.t }  (** Reciprocal (1/x). *)
    | Trunc of { src : ref; dtype : Dtype.t }  (** Truncate toward zero. *)
    | Add of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Addition. *)
    | Sub of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Subtraction. *)
    | Mul of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Multiplication. *)
    | Fdiv of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Float division. *)
    | Idiv of { lhs : ref; rhs : ref; dtype : Dtype.t }
        (** Integer division. *)
    | Mod of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Modulo. *)
    | Max of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Maximum. *)
    | Pow of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Power. *)
    | Shl of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Left shift. *)
    | Shr of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Right shift. *)
    | And of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Bitwise AND. *)
    | Or of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Bitwise OR. *)
    | Xor of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Bitwise XOR. *)
    | Threefry of { lhs : ref; rhs : ref; dtype : Dtype.t }
        (** Threefry2x32 mixing function (random number generation). *)
    | Cmplt of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Less than. *)
    | Cmpeq of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Equal. *)
    | Cmpne of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Not equal. *)
    | Where of { cond : ref; a : ref; b : ref; dtype : Dtype.t }
        (** Conditional select: if [cond] then [a] else [b]. *)
    | Mulacc of { a : ref; b : ref; c : ref; dtype : Dtype.t }
        (** Fused multiply-accumulate: [a * b + c]. *)
    | Cast of { src : ref; dtype : Dtype.t }
        (** Type conversion with value preservation. *)
    | Bitcast of { src : ref; dtype : Dtype.t }
        (** Reinterpret bits as different type. *)
    | Vectorize of { srcs : ref list; dtype : Dtype.t }
        (** Pack scalars into a vector. *)
    | Cat of { srcs : ref list; dtype : Dtype.t }
        (** Concatenate vectors. Total element count of sources must equal
            [dtype.count]. *)
    | Gep of { src : ref; idx : int; dtype : Dtype.t }
        (** Extract element [idx] from a vector. *)
    | Range of { size : ref; dtype : Dtype.t; axis : int; kind : axis_kind }
        (** Begin loop from 0 to [size - 1]. Produces the loop variable. [axis]
            identifies the loop dimension. [kind] encodes the axis semantics. *)
    | End of { value : ref; ranges : ref list }
        (** Close a computation scope, producing [value]. [ranges] are
            references to associated [Range] or index-like instructions.
            Distinct from {!Program.instr.End_range}, which explicitly closes a
            loop. *)
    | Barrier  (** Workgroup synchronization barrier. *)
    | Special of { dim : special_dim; size : ref; dtype : Dtype.t }
        (** GPU thread ID for [dim], bounded by [size]. *)
    | Reduce of {
        op : reduce_op;
        src : ref;
        ranges : ref list;
        dtype : Dtype.t;
      }
        (** Reduction over [ranges] using [op]. Lowered during linearization. *)
    | Unroll of { src : ref; axes : (int * int) list; dtype : Dtype.t }
        (** Unroll vector [src] along [axes]. Each [(axis, size)] pair specifies
            a dimension to unroll. Lowered during linearization. *)
    | Contract of { src : ref; axes : (int * int) list; dtype : Dtype.t }
        (** Contract (fold) dimensions of [src] along [axes]. Inverse of
            {!instr.Unroll}. Lowered during linearization. *)
    | Wmma of {
        name : string;
        a : ref;
        b : ref;
        c : ref;
        ranges : ref list;
        dtype : Dtype.t;
        dims : int * int * int;
        dtype_in : Dtype.scalar;
        dtype_out : Dtype.scalar;
        device : string;
        threads : int;
        upcast_axes : (int * int) list * (int * int) list * (int * int) list;
        reduce_axes : int list;
      }
        (** Tensor core matrix multiply-accumulate. [dims] is [(N, M, K)] matrix
            dimensions. [dtype_in] is the input element type. [dtype_out] is the
            accumulator type. [name] encodes the operation for preamble
            generation. [threads] is the number of cooperating threads per WMMA
            operation. [ranges] constrain the reduction scope. [upcast_axes] are
            [(axis, size)] pairs for register tiling of [a], [b], and [c]
            respectively. [reduce_axes] are reduction axis indices. *)
    | Custom of { fmt : string; args : ref list }
        (** Inject custom code as a statement. [fmt] is a backend format string
            with positional placeholders substituted by the rendered forms of
            [args] during code emission. *)
    | Custom_inline of { fmt : string; args : ref list; dtype : Dtype.t }
        (** Inject custom code as an expression. Same [fmt] convention as
            {!instr.Custom}. *)

  (** {1:kernel-program Program} *)

  type t = instr array
  (** A program is an array of instructions. Instruction indices serve as
      references. *)

  (** {1:kernel-inspection Inspection} *)

  val dtype_of : instr -> Dtype.t option
  (** [dtype_of instr] is the result dtype, or [None] for instructions that
      produce no value ([Sink], [Group], [After], [Store], [End], [Barrier],
      [Custom]).

      For pointer-producing instructions ([Param], [Define_local], [Define_reg],
      [Bufferize], [Index], [Ptrcat]), returns the {e base type}, not a pointer
      type. *)

  val refs_of : instr -> ref list
  (** [refs_of instr] is all operand references in definition order. The list is
      complete (every ref in the instruction is included) and may contain
      duplicates. *)

  val map_refs : (ref -> ref) -> instr -> instr
  (** [map_refs f instr] applies [f] to every operand reference in [instr].
      Non-reference fields (dtype, constants, options) are preserved.
      [map_refs Fun.id instr] is structurally equal to [instr]. *)

  val is_unary : instr -> bool
  (** [is_unary instr] is [true] for [Neg], [Exp2], [Log2], [Sin], [Sqrt],
      [Recip], [Trunc]. *)

  val is_binary : instr -> bool
  (** [is_binary instr] is [true] for arithmetic and comparison operations with
      two operands. *)

  val is_ternary : instr -> bool
  (** [is_ternary instr] is [true] for [Where], [Mulacc]. *)

  val is_alu : instr -> bool
  (** [is_alu instr] is [true] iff [instr] is a unary, binary, or ternary
      arithmetic operation. *)

  (** {1:kernel-transform Transformation} *)

  val intern : t -> t
  (** [intern t] deduplicates structurally equal instructions, producing a
      semantically equivalent program. References are remapped to point to the
      canonical (first-seen) copy of each instruction. Idempotent. *)

  (** {1:kernel-validation Validation} *)

  val validate : t -> unit
  (** [validate t] checks that [t] satisfies the Kernel IR invariants.

      Raises [Failure] if any invariant is violated, with a message describing
      the violation and instruction index. *)

  (** {1:kernel-fmt Pretty Printing} *)

  val pp_instr : Format.formatter -> instr -> unit
  (** [pp_instr fmt instr] formats a single instruction on [fmt], one line. *)

  val pp : Format.formatter -> t -> unit
  (** [pp fmt t] formats all instructions with their indices on [fmt], one per
      line. *)
end

(** High-level tensor graph for scheduling and fusion.

    Top-level IR representing the full computation graph including buffers,
    movement ops, multi-device operations, and autograd metadata. Lowered to
    {!Kernel} during scheduling, then to {!Program} for code emission.

    This IR includes operations not present in {!Kernel} or {!Program}: buffer
    management ([Buffer], [Buffer_view], [Unique], [Lunique], [Device]),
    movement ops ([Reshape], [Expand], [Pad], [Shrink], [Permute], [Flip]),
    multi-device ops ([Multi], [Mstack], [Mselect], [Allreduce]), and autograd
    support ([Detach], [Contiguous_backward], [Assign]).

    Metadata (names, gradient functions, custom kernels) is stored in registries
    and referenced by opaque IDs to keep instruction records small and avoid
    duplication.

    {2 Invariants}

    Well-formed programs must satisfy:
    - All references point to earlier instructions (SSA dominance).
    - [Buffer.unique] must reference [Unique] or [Lunique]; [Buffer.device] must
      reference [Device]; [Buffer.size] must be non-negative.
    - [Buffer_view.src] must reference [Index]; size and offset non-negative.
    - [Define_var] is scalar int/index with valid bounds ([lo <= hi]).
    - [Bind.var] must reference [Define_var]; optional [value] must match dtype.
    - [Param.device] must reference [Device]; [Param.shape] must be index vector
      (empty vectors with [count = 0] are allowed).
    - [Copy] and [Allreduce] device must reference [Device].
    - Movement ops ([Reshape], [Expand]) take index-vector shapes.
    - [Pad]/[Shrink] before and after must be index vectors of equal width.
    - [Range] must be scalar int/index; [size] must match dtype.
    - [Index] must have at least one index (index scalars); optional gate must
      be bool scalar. Unlike {!Kernel}, [Index.ptr] has no structural constraint
      (only [idxs] and [gate] are validated).
    - ALU operand dtypes must match the result dtype; comparisons return bool.
    - [Idiv]/[Mod] must have int/index dtype.
    - Shifts must have int/index dtype; rhs must match lhs dtype or be [uint32].
    - [Vectorize] source count must equal [dtype.count]; all sources scalar.
    - [Flip] uses a [bool list] (one per dimension; [true] = flip).
      [Flip { dims = [false; true; false] }] flips axis 1. *)
module Tensor : sig
  (** {1:tensor-registries Registries}

      Metadata types stored by reference. Use [register] to intern a value and
      [get] to retrieve it by ID. *)

  (** Tensor operation metadata for debugging and profiling. *)
  module Metadata : sig
    type t = { name : string; caller : string; backward : bool }
    type id

    val register : t -> id
    (** [register t] interns metadata and returns its unique ID. *)

    val get : id -> t
    (** [get id] retrieves previously registered metadata.

        Raises [Not_found] if [id] was not returned by {!register}. *)
  end

  (** Interned {!Kernel.kernel_info} values. *)
  module Kernel_info : sig
    type t = Kernel.kernel_info
    type id

    val register : t -> id
    (** [register t] interns kernel info and returns its unique ID. *)

    val get : id -> t
    (** [get id] retrieves previously registered kernel info.

        Raises [Not_found] if [id] was not returned by {!register}. *)
  end

  (** Gradient function reference. Stored as a name string for serialization. *)
  module Grad_fxn : sig
    type t = { name : string }
    type id

    val register : t -> id
    (** [register t] interns a gradient function and returns its unique ID. *)

    val get : id -> t
    (** [get id] retrieves a previously registered gradient function.

        Raises [Not_found] if [id] was not returned by {!register}. *)
  end

  (** Custom kernel reference.

      [ast] optionally stores a pre-lowered Kernel IR body for custom kernels
      that can be resolved during earliest rewrites. *)
  module Custom_kernel : sig
    type t = {
      name : string;  (** Kernel name for identification and codegen. *)
      grad : Grad_fxn.id option;
          (** Gradient function. [None] for non-differentiable kernels. *)
      ast : Kernel.t option;
          (** Pre-lowered Kernel IR body, when the kernel can be resolved during
              earliest rewrites. [None] for opaque kernels. *)
      metadata : Metadata.id list;  (** Associated operation metadata entries. *)
    }

    type id

    val register : t -> id
    (** [register t] interns a custom kernel and returns its unique ID. *)

    val get : id -> t
    (** [get id] retrieves a previously registered custom kernel.

        Raises [Not_found] if [id] was not returned by {!register}. *)
  end

  (** {1:tensor-types Types} *)

  (** Constant literal values. Same variants as {!Kernel.const}. [Invalid]
      represents an uninitialized or undefined value (requires [Index] dtype).
  *)
  type const = Bool of bool | Int of int | Float of float | Invalid

  (** Reduction operation kind. Same variants as {!Kernel.reduce_op}. *)
  type reduce_op = Add | Mul | Max

  (** Device specification. [Single] for one device, [Multi] for sharded tensors
      across devices. *)
  type device = Single of string | Multi of string list

  type metadata = Metadata.id
  (** Opaque ID referencing {!Metadata.t}. *)

  type kernel_info = Kernel_info.id
  (** Opaque ID referencing {!Kernel_info.t}. *)

  type grad_fxn = Grad_fxn.id
  (** Opaque ID referencing {!Grad_fxn.t}. *)

  type custom_kernel = Custom_kernel.id
  (** Opaque ID referencing {!Custom_kernel.t}. *)

  type kernel = {
    ast : Kernel.t;  (** Lowered Kernel IR body. *)
    metadata : metadata list;  (** Associated operation metadata. *)
    grad : grad_fxn option;
        (** Gradient function for autograd. [None] for non-differentiable
            kernels. *)
  }
  (** A scheduled kernel with its lowered AST and associated metadata. *)

  type ref = int
  (** Reference to another instruction by index in the instruction array. *)

  (** {1:tensor-instrs Instructions} *)

  (** Tensor IR instruction. *)
  type instr =
    | Sink of { srcs : ref list; kernel_info : kernel_info option }
        (** Program root. Collects side-effecting instructions. *)
    | Group of { srcs : ref list }
        (** Group related instructions for scheduling. *)
    | After of { src : ref; deps : ref list; dtype : Dtype.t }
        (** Execution dependency: [src] must execute after [deps]. Carries
            [src]'s dtype. *)
    | Unique of { id : int }
        (** Unique buffer identity marker. Referenced by [Buffer.unique]. *)
    | Lunique of { id : int }
        (** Lazy unique buffer identity marker. Like [Unique] but for buffers
            that may not yet be materialized. *)
    | Device of { device : device }
        (** Device specification node. Referenced by [Buffer], [Copy],
            [Allreduce], and [Param]. *)
    | Buffer of { unique : ref; device : ref; size : int; dtype : Dtype.t }
        (** Allocated buffer. [unique] must reference [Unique]/[Lunique];
            [device] must reference [Device]; [size] is element count
            (non-negative). *)
    | Buffer_view of { src : ref; size : int; offset : int; dtype : Dtype.t }
        (** View into an existing buffer at byte [offset] for [size] elements.
            [src] must reference [Index]. *)
    | Const of { value : const; dtype : Dtype.t; srcs : ref list }
        (** Constant literal. [srcs] are data-dependency references that
            establish shape context for broadcasting; empty for scalar constants
            with no shape dependencies. *)
    | Vconst of { values : const list; dtype : Dtype.t; srcs : ref list }
        (** Vector constant. Length of [values] must equal [dtype.count]. [srcs]
            serve the same role as in {!instr.Const}. *)
    | Define_var of { name : string; lo : int; hi : int; dtype : Dtype.t }
        (** Symbolic variable with runtime bounds [lo..hi]. *)
    | Bind of { var : ref; value : ref option; dtype : Dtype.t }
        (** Bind a concrete [value] to a symbolic [var]. [var] must reference
            [Define_var]. *)
    | Param of {
        slot : int;
        dtype : Dtype.t;
        shape : ref option;
        device : ref option;
      }
        (** Function parameter at [slot]. Optional [shape] (index vector) and
            [device] ([Device] node). *)
    | Call of {
        fn : ref;
        args : ref list;
        grad : grad_fxn option;
        dtype : Dtype.t;
      }  (** Function call. *)
    | Custom_kernel of { srcs : ref list; kernel : custom_kernel }
        (** User-defined kernel. Identified by name. *)
    | Kernel of { srcs : ref list; kernel : kernel }
        (** Scheduled kernel. [srcs] are the buffer and dependency inputs. *)
    | Assign of {
        target : ref;
        value : ref;
        extras : ref list;
        dtype : Dtype.t;
      }
        (** In-place assignment: write [value] into [target]. [extras] are
            additional [Assign] dependencies for ordering. *)
    | Detach of { src : ref; dtype : Dtype.t }
        (** Detach [src] from the autograd graph. *)
    | Contiguous of { src : ref; ranges : ref list; dtype : Dtype.t }
        (** Force [src] to contiguous memory layout. [ranges] are index scalar
            constraints. *)
    | Contiguous_backward of { src : ref; dtype : Dtype.t }
        (** Backward pass marker for contiguous. *)
    | Copy of { src : ref; device : ref; dtype : Dtype.t }
        (** Copy [src] to [device]. *)
    | Allreduce of { src : ref; device : ref; op : reduce_op; dtype : Dtype.t }
        (** All-reduce [src] across devices using [op]. *)
    | Multi of { src : ref; axis : int; dtype : Dtype.t }
        (** Mark [src] as sharded along [axis] for multi-device. *)
    | Mstack of { srcs : ref list; dtype : Dtype.t }
        (** Stack single-device tensors into a multi-device tensor. *)
    | Mselect of { src : ref; index : int; dtype : Dtype.t }
        (** Select shard [index] from a multi-device tensor. *)
    | Encdec of { srcs : ref list; shape : int list; dtype : Dtype.t }
        (** Encode/decode operation with target [shape]. *)
    | Reduce_axis of {
        src : ref;
        op : reduce_op;
        axes : int list;
        dtype : Dtype.t;
      }
        (** Reduce [src] along [axes] using [op]. High-level form before range
            lowering. *)
    | Reduce of {
        src : ref;
        ranges : ref list;
        op : reduce_op;
        dtype : Dtype.t;
      }
        (** Reduce [src] over [ranges] using [op]. Lowered form with explicit
            range references. *)
    | Reshape of { src : ref; shape : ref; dtype : Dtype.t }
        (** Reshape [src] to [shape] (index vector). *)
    | Expand of { src : ref; shape : ref; dtype : Dtype.t }
        (** Broadcast [src] to [shape] (index vector). *)
    | Pad of { src : ref; before : ref; after : ref; dtype : Dtype.t }
        (** Pad [src] with zeros. [before] and [after] are index vectors of
            equal width specifying padding per dimension. *)
    | Shrink of { src : ref; before : ref; after : ref; dtype : Dtype.t }
        (** Shrink (crop) [src]. [before] and [after] are index vectors of equal
            width specifying bounds per dimension. *)
    | Permute of { src : ref; order : int list; dtype : Dtype.t }
        (** Transpose [src] according to [order]. *)
    | Flip of { src : ref; dims : bool list; dtype : Dtype.t }
        (** Reverse [src] along selected dimensions. [true] at position [i]
            means flip axis [i]. *)
    | Range of { size : ref; dtype : Dtype.t; axis : int; kind : axis_kind }
        (** Begin loop from 0 to [size - 1]. Produces the loop variable. *)
    | End of { value : ref; ranges : ref list }
        (** Close a computation scope, producing [value]. *)
    | Index of {
        ptr : ref;
        idxs : ref list;
        gate : ref option;
        dtype : Dtype.t;
      }  (** Pointer arithmetic: [ptr + sum(idxs)], with optional [gate]. *)
    | Store of { dst : ref; value : ref }
        (** Store [value] to memory at [dst]. *)
    | Vectorize of { srcs : ref list; dtype : Dtype.t }
        (** Pack scalars into a vector. *)
    | Cast of { src : ref; dtype : Dtype.t }
        (** Type conversion with value preservation. *)
    | Bitcast of { src : ref; dtype : Dtype.t }
        (** Reinterpret bits as different type. *)
    | Neg of { src : ref; dtype : Dtype.t }  (** Negation. *)
    | Exp2 of { src : ref; dtype : Dtype.t }  (** Base-2 exponential. *)
    | Log2 of { src : ref; dtype : Dtype.t }  (** Base-2 logarithm. *)
    | Sin of { src : ref; dtype : Dtype.t }  (** Sine. *)
    | Sqrt of { src : ref; dtype : Dtype.t }  (** Square root. *)
    | Recip of { src : ref; dtype : Dtype.t }  (** Reciprocal (1/x). *)
    | Trunc of { src : ref; dtype : Dtype.t }  (** Truncate toward zero. *)
    | Add of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Addition. *)
    | Sub of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Subtraction. *)
    | Mul of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Multiplication. *)
    | Fdiv of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Float division. *)
    | Idiv of { lhs : ref; rhs : ref; dtype : Dtype.t }
        (** Integer division. *)
    | Mod of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Modulo. *)
    | Max of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Maximum. *)
    | Pow of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Power. *)
    | Shl of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Left shift. *)
    | Shr of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Right shift. *)
    | And of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Bitwise AND. *)
    | Or of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Bitwise OR. *)
    | Xor of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Bitwise XOR. *)
    | Threefry of { lhs : ref; rhs : ref; dtype : Dtype.t }
        (** Threefry2x32 mixing function (random number generation). *)
    | Cmplt of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Less than. *)
    | Cmpeq of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Equal. *)
    | Cmpne of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Not equal. *)
    | Where of { cond : ref; a : ref; b : ref; dtype : Dtype.t }
        (** Conditional select: if [cond] then [a] else [b]. *)
    | Mulacc of { a : ref; b : ref; c : ref; dtype : Dtype.t }
        (** Fused multiply-accumulate: [a * b + c]. *)

  (** {1:tensor-program Program} *)

  type t = instr array
  (** A program is an array of instructions. Instruction indices serve as
      references. *)

  (** {1:tensor-inspection Inspection} *)

  val dtype_of : instr -> Dtype.t option
  (** [dtype_of instr] is the result dtype, or [None] for instructions that
      produce no value ([Sink], [Group], [Unique], [Lunique], [Device],
      [Custom_kernel], [Kernel], [End], [Store]). *)

  val refs_of : instr -> ref list
  (** [refs_of instr] is all operand references in definition order. The list is
      complete (every ref in the instruction is included) and may contain
      duplicates. *)

  val map_refs : (ref -> ref) -> instr -> instr
  (** [map_refs f instr] applies [f] to every operand reference in [instr].
      Non-reference fields (dtype, constants, options) are preserved.
      [map_refs Fun.id instr] is structurally equal to [instr]. *)

  val is_unary : instr -> bool
  (** [is_unary instr] is [true] for [Neg], [Exp2], [Log2], [Sin], [Sqrt],
      [Recip], [Trunc]. *)

  val is_binary : instr -> bool
  (** [is_binary instr] is [true] for arithmetic and comparison operations with
      two operands. *)

  val is_ternary : instr -> bool
  (** [is_ternary instr] is [true] for [Where], [Mulacc]. *)

  val is_alu : instr -> bool
  (** [is_alu instr] is [true] iff [instr] is a unary, binary, or ternary
      arithmetic operation. *)

  (** {1:tensor-transform Transformation} *)

  val intern : t -> t
  (** [intern t] deduplicates structurally equal instructions, producing a
      semantically equivalent program. References are remapped to point to the
      canonical (first-seen) copy of each instruction. Idempotent. *)

  (** {1:tensor-validation Validation} *)

  val validate : t -> unit
  (** [validate t] checks that [t] satisfies the Tensor IR invariants.

      Raises [Failure] if any invariant is violated, with a message describing
      the violation and instruction index. *)

  (** {1:tensor-fmt Pretty Printing} *)

  val pp_instr : Format.formatter -> instr -> unit
  (** [pp_instr fmt instr] formats a single instruction on [fmt], one line. *)

  val pp : Format.formatter -> t -> unit
  (** [pp fmt t] formats all instructions with their indices on [fmt], one per
      line. *)
end

module Program : sig
  (** Low-level IR for code generation.

      Instructions map directly to rendered code (C, CUDA, Metal, etc.). The IR
      uses a flattened SSA representation where each instruction references its
      operands by index into the instruction array.

      This IR is post-devectorizer and render-ready. Vector masks, per-lane
      vector constants, multi-element GEPs, and Index-typed values must be
      lowered before constructing a {!Program.t}. The renderer assumes only
      backend-legal operations and scalar control flow.

      {2 Invariants}

      Well-formed programs must satisfy:
      - [Range]/[End_range] and [If]/[Endif] pairs are properly nested and
        explicitly paired.
      - Instruction references point to earlier instructions (SSA dominance).
      - Each [Special] dimension (e.g., [Group 0]) appears at most once. The
        flat array representation does not deduplicate automatically, and the
        renderer emits variable declarations, so duplicates cause redeclaration
        errors in generated code.
      - [Where] conditions must be scalar. C-style ternary [cond ? a : b]
        requires scalar conditions; vector masks need [select(b, a, cond)] which
        we don't emit. The devectorizer ensures WHERE conditions are scalar
        before codegen. *)

  (** {1:program-types Types} *)

  (** Constant literal values. Unlike {!Kernel.const} and {!Tensor.const},
      [Invalid] is not valid in rendered programs. *)
  type const = Bool of bool | Int of int | Float of float

  type ref = int
  (** Reference to another instruction by index in the instruction array. *)

  (** {1:program-instrs Instructions} *)

  (** Program IR instruction. *)
  type instr =
    | Param of { idx : int; dtype : Dtype.ptr }
        (** Kernel parameter at position [idx]. *)
    | Define_local of { size : int; dtype : Dtype.ptr }
        (** Allocate shared memory for [size] elements. *)
    | Define_reg of { size : int; dtype : Dtype.ptr }
        (** Allocate register storage for [size] elements. *)
    | Define_var of { name : string; lo : int; hi : int; dtype : Dtype.t }
        (** Scalar kernel parameter with runtime bounds [lo..hi]. *)
    | Const of { value : const; dtype : Dtype.t }  (** Constant literal. *)
    | Index of {
        ptr : ref;
        idxs : ref list;
        gate : ref option;
        dtype : Dtype.ptr;
      }  (** Pointer arithmetic: [ptr + sum(idxs)], with optional [gate]. *)
    | Load of { src : ref; alt : ref option; dtype : Dtype.t }
        (** Load value from memory. If [alt] is provided, the source index must
            be gated and [alt] is used when the gate is false. *)
    | Store of { dst : ref; value : ref }
        (** Store [value] to memory at [dst]. *)
    | Neg of { src : ref; dtype : Dtype.t }  (** Negation. *)
    | Exp2 of { src : ref; dtype : Dtype.t }  (** Base-2 exponential. *)
    | Log2 of { src : ref; dtype : Dtype.t }  (** Base-2 logarithm. *)
    | Sin of { src : ref; dtype : Dtype.t }  (** Sine. *)
    | Sqrt of { src : ref; dtype : Dtype.t }  (** Square root. *)
    | Recip of { src : ref; dtype : Dtype.t }  (** Reciprocal (1/x). *)
    | Trunc of { src : ref; dtype : Dtype.t }  (** Truncate toward zero. *)
    | Add of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Addition. *)
    | Sub of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Subtraction. *)
    | Mul of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Multiplication. *)
    | Fdiv of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Float division. *)
    | Idiv of { lhs : ref; rhs : ref; dtype : Dtype.t }
        (** Integer division. *)
    | Mod of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Modulo. *)
    | Max of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Maximum. *)
    | Pow of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Power. *)
    | Shl of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Left shift. *)
    | Shr of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Right shift. *)
    | And of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Bitwise AND. *)
    | Or of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Bitwise OR. *)
    | Xor of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Bitwise XOR. *)
    | Threefry of { lhs : ref; rhs : ref; dtype : Dtype.t }
        (** Threefry2x32 mixing function (random number generation). *)
    | Cmplt of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Less than. *)
    | Cmpeq of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Equal. *)
    | Cmpne of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Not equal. *)
    | Where of { cond : ref; a : ref; b : ref; dtype : Dtype.t }
        (** Conditional select: if [cond] then [a] else [b]. *)
    | Mulacc of { a : ref; b : ref; c : ref; dtype : Dtype.t }
        (** Fused multiply-accumulate: [a * b + c]. *)
    | Cast of { src : ref; dtype : Dtype.t }
        (** Type conversion with value preservation. *)
    | Bitcast of { src : ref; dtype : Dtype.t }
        (** Reinterpret bits as different type. *)
    | Vectorize of { srcs : ref list; dtype : Dtype.t }
        (** Pack scalars into a vector. *)
    | Gep of { src : ref; idx : int; dtype : Dtype.t }
        (** Extract element [idx] from a vector. *)
    | Range of { size : ref; dtype : Dtype.t; axis : int; kind : axis_kind }
        (** Begin loop from 0 to [size - 1]. Produces the loop variable. [axis]
            identifies the loop dimension (0=x, 1=y, etc.) for variable naming
            and axis-specific optimizations. [kind] encodes the axis semantics.
        *)
    | End_range of { range : ref }  (** End loop started by [Range]. *)
    | If of { cond : ref; idx_for_dedup : ref }
        (** Begin conditional block. [idx_for_dedup] must be an [Index] (or
            casted index), used for structural ordering guarantees. *)
    | Endif of { if_ : ref }  (** End conditional block. *)
    | Barrier  (** Workgroup synchronization barrier. *)
    | Special of { dim : special_dim; size : ref; dtype : Dtype.t }
        (** GPU thread ID for [dim], bounded by [size]. *)
    | Wmma of {
        name : string;
        a : ref;
        b : ref;
        c : ref;
        dtype : Dtype.t;
        dims : int * int * int;
        dtype_in : Dtype.scalar;
        dtype_out : Dtype.scalar;
        device : string;
        threads : int;
        upcast_axes : (int * int) list * (int * int) list * (int * int) list;
        reduce_axes : int list;
      }
        (** Tensor core matrix multiply-accumulate. [dims] is [(N, M, K)] matrix
            dimensions. [dtype_in] is the input element type. [dtype_out] is the
            accumulator type. [name] encodes the operation for preamble
            generation. [threads] is the number of cooperating threads per WMMA
            operation. [upcast_axes] are [(axis, size)] pairs for register
            tiling of [a], [b], and [c] respectively. [reduce_axes] are
            reduction axis indices. *)
    | Custom of { fmt : string; args : ref list }
        (** Inject custom code as a statement. [fmt] is a backend format string
            with positional placeholders substituted by the rendered forms of
            [args] during code emission. *)
    | Custom_inline of { fmt : string; args : ref list; dtype : Dtype.t }
        (** Inject custom code as an expression. Same [fmt] convention as
            {!instr.Custom}. *)

  (** {1:program-program Program} *)

  type t = instr array
  (** A program is an array of instructions. Instruction indices correspond to
      SSA variable names. *)

  (** {1:program-inspection Inspection} *)

  val dtype_of : instr -> Dtype.t option
  (** [dtype_of instr] is the result dtype, or [None] for instructions that
      produce no value ([Store], [End_range], [If], [Endif], [Barrier],
      [Custom]).

      For pointer-producing instructions ([Param], [Define_local], [Define_reg],
      [Index]), returns the {e base type}, not a pointer type. Most consumers
      want "what data flows here" for type checking ALU ops; pointer-ness can be
      determined by pattern matching on the instruction. *)

  val refs_of : instr -> ref list
  (** [refs_of instr] is all operand references in definition order. The list is
      complete (every ref in the instruction is included) and may contain
      duplicates. *)

  val map_refs : (ref -> ref) -> instr -> instr
  (** [map_refs f instr] applies [f] to every operand reference in [instr].
      Non-reference fields (dtype, constants, options) are preserved.
      [map_refs Fun.id instr] is structurally equal to [instr]. *)

  val map_alu : map_ref:(ref -> ref) -> dtype:Dtype.t -> instr -> instr
  (** [map_alu ~map_ref ~dtype instr] applies [map_ref] to operand references
      and overrides the result dtype.

      Raises [Failure] if [instr] is not an ALU instruction. *)

  val is_unary : instr -> bool
  (** [is_unary instr] is [true] for [Neg], [Exp2], [Log2], [Sin], [Sqrt],
      [Recip], [Trunc]. *)

  val is_binary : instr -> bool
  (** [is_binary instr] is [true] for arithmetic and comparison operations with
      two operands. *)

  val is_ternary : instr -> bool
  (** [is_ternary instr] is [true] for [Where], [Mulacc]. *)

  val is_alu : instr -> bool
  (** [is_alu instr] is [true] iff [instr] is a unary, binary, or ternary
      arithmetic operation. *)

  (** {1:program-validation Validation} *)

  val validate : t -> unit
  (** [validate t] checks that [t] satisfies the Program IR invariants.

      Raises [Failure] if any invariant is violated, with a message describing
      the violation and instruction index. Call this before rendering to catch
      malformed IR early.

      Checks performed (post-devectorizer, render-ready subset):
      - All references point to earlier instructions (SSA dominance)
      - No [Index] dtype in linearized program (should be lowered)
      - [Param] has [Global] addrspace, [Define_local] has [Local], [Define_reg]
        has [Reg]
      - [Define_var] is scalar and has valid bounds
      - [Range]/[End_range] and [If]/[Endif] pairs are properly nested and
        explicitly paired
      - [Range] dtype is integer, scalar, and matches the [size] dtype
      - [If] condition must be bool, [idx_for_dedup] must reference an [Index]
      - [Index] base is a pointer definition, each index is scalar int, optional
        gate is scalar bool
      - [Load] src must reference an [Index] (or casted [Index]); [alt] requires
        a gated [Index]
      - [Store] dst must reference an [Index] (or casted [Index])
      - [Special] dtype is [int32] scalar and matches the [size] dtype
      - Each [Special] dimension appears at most once
      - [Where] condition is scalar bool, branches match result dtype
      - Comparisons return bool, operands have matching dtypes
      - [Idiv]/[Mod] must have int dtype
      - Binary/unary ALU operands match result dtype; shift rhs matches lhs or
        is [uint32]
      - [Vectorize] source count equals [dtype.count] and is > 1; all sources
        are scalar
      - [Gep] result is scalar of source vector type, index in bounds
      - [Wmma] dims are positive, result dtype matches [dtype_out] *)

  (** {1:program-rebuilding Rebuilding} *)

  val rebuild :
    (emit:(instr -> int) -> map_ref:(int -> int) -> instr -> int option) ->
    t ->
    t
  (** [rebuild f program] constructs a new program by iterating forward through
      [program]. For each instruction, calls [f ~emit ~map_ref instr]:

      - [map_ref r] translates an original ref [r] to its new index in the
        output.
      - [emit instr] appends [instr] to the output and returns its new index.
      - Return [Some idx] to map the current instruction to [idx].
      - Return [None] to copy the instruction with refs remapped via [map_ref].
  *)

  (** {1:program-fmt Pretty Printing} *)

  val pp_instr : Format.formatter -> instr -> unit
  (** [pp_instr fmt instr] formats a single instruction on [fmt], one line. *)

  val pp : Format.formatter -> t -> unit
  (** [pp fmt t] formats all instructions with their indices on [fmt], one per
      line. *)
end
