(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Three-level intermediate representation for tensor computation.

    Tolk defines three IRs at decreasing abstraction levels:

    - {!Tensor}: scheduling-level tensor graph (shape ops, multi-device,
      autograd)
    - {!Kernel}: codegen-oriented DAG (ranges, reductions, memory indexing)
    - {!Program}: flat SSA for code emission (loops, if/endif, no graph)

    All three share a flat instruction-array representation where
    [type ref = int] indexes into the array. Instructions may only reference
    earlier indices (SSA dominance). Each module provides
    {!val:Kernel.dtype_of}, {!val:Kernel.refs_of}, {!val:Kernel.map_refs},
    {!val:Kernel.validate}, and pretty-printing ({!val:Kernel.pp}).

    Data types for elements, vectors, and pointer address spaces are defined in
    {!Dtype}. *)

(** {1:shared Shared Definitions} *)

(** GPU thread dimension. The [int] payload specifies the axis (0, 1, or 2 by
    convention; not enforced by the type). *)
type special_dim =
  | Group_id of int
      (** Group/block ID ([blockIdx] in CUDA, [get_group_id] in OpenCL). *)
  | Local_id of int
      (** Local/thread ID within workgroup ([threadIdx] in CUDA, [get_local_id]
          in OpenCL). *)
  | Global_idx of int
      (** Global thread ID ([blockIdx * blockDim + threadIdx] in CUDA,
          [get_global_id] in OpenCL). *)

val special_axis : special_dim -> int
(** [special_axis dim] is the axis index of [dim]. *)

(** Range axis kind.

    Determines how a [Range] dimension maps to hardware parallelism or loop
    structure. Variant declaration order encodes scheduling priority (used by
    structural comparison). Used by [Range] instructions in {!Kernel},
    {!Tensor}, and {!Program}. *)
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

(** {1:kernel Kernel IR}

    Codegen-oriented DAG produced by rangeify.

    This IR sits between {!Tensor} (scheduling) and {!Program} (rendering). It
    retains DAG structure with high-level operations like [Reduce], [Unroll],
    [Contract], and [Bufferize] that are lowered during linearization.

    Unlike {!Program}, this IR is not flattened into SSA. Instructions reference
    operands by index into the instruction array. Use {!Kernel.intern} to
    deduplicate structurally equal instructions.

    {2:kernel-invariants Invariants}

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
    - [Cat] source scalar types must match; total element count must equal
      [dtype.count].
    - [Gep] result is scalar of source vector type; index in bounds.
    - [Ptrcat] sources must all be pointers with matching addrspace, base dtype,
      and image metadata; total pointer vector width must equal [dtype.v].
    - [End] closes a computation scope with a value and range references.
      Distinct from {!Program.instr.End_range}, which explicitly closes a loop.
*)
module Kernel : sig
  (** {1:kernel-types Types} *)

  (** Constant literal values.

      [Invalid] represents an uninitialized or undefined value (requires [Index]
      dtype). Same variants as {!Tensor.const}. *)
  type const = Bool of bool | Int of int | Float of float | Invalid

  (** Reduction operation kind. Same variants as {!Tensor.reduce_op}. *)
  type reduce_op = Add | Mul | Max

  (** Device specification for bufferization. *)
  type bufferize_device =
    | Device_single of string  (** Target a single named device. *)
    | Device_multi of string list  (** Sharded across multiple named devices. *)
    | Device_index of int
        (** Index into a multi-device buffer's device list. *)

  (** Estimated cost metric: exact integer or symbolic expression string. *)
  type estimate = Int of int | Symbolic of string

  type estimates = {
    ops : estimate;  (** Arithmetic instruction count. *)
    lds : estimate;  (** Memory read instruction count. *)
    mem : estimate;  (** Bytes transferred. *)
  }
  (** Cost estimates for a kernel.

      All fields use {!estimate} to support both concrete and symbolic cost
      values. *)

  type bufferize_opts = {
    device : bufferize_device option;
        (** Target device. [None] inherits from context. *)
    addrspace : Dtype.addr_space;
        (** Address space for the materialized buffer. Must match the
            [dtype.addrspace] of the enclosing [Bufferize] instruction. *)
    removable : bool;
        (** When [true], this buffer can be optimized away if unused. [false]
            for buffers that must be materialized (e.g., before a [Copy]). *)
  }
  (** Options controlling buffer materialization during bufferization. *)

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
        (** Cost estimates. [None] when not yet computed. *)
    metadata_tags : string list;
        (** Deduplicated scheduler metadata names collected during rangeify
            splitting. *)
  }
  (** Kernel metadata attached to {!instr.Sink}. *)

  type ref = int
  (** The type for instruction references. An index into the instruction array.
  *)

  (** {1:kernel-instrs Instructions} *)

  (** Kernel IR instruction.

      Organized into graph management, memory definitions, constants, memory
      operations, ALU (unary, binary, ternary), type conversion, vector ops,
      control flow, GPU special, reduction, tensor core, and custom code
      injection. *)
  type instr =
    (* {2 Graph management} *)
    | Sink of { srcs : ref list; kernel_info : kernel_info option }
        (** Program root. Collects side-effecting instructions. [kernel_info] is
            [None] before kernel metadata is attached. *)
    | Group of { srcs : ref list }
        (** Group related instructions for scheduling. *)
    | After of { src : ref; deps : ref list }
        (** Execution dependency: [src] must execute after [deps]. Unlike
            {!Tensor.instr.After}, does not carry dtype (the value flows through
            [src]). *)
    (* {2 Memory definitions} *)
    | Param of { idx : int; dtype : Dtype.ptr }
        (** Kernel parameter at position [idx]. Address space must be [Global].
        *)
    | Define_local of { size : int; dtype : Dtype.ptr }
        (** Allocate shared memory for [size] elements. Address space must be
            [Local]. *)
    | Define_reg of { size : int; dtype : Dtype.ptr }
        (** Allocate register storage for [size] elements. Address space must be
            [Reg]. *)
    | Define_var of { name : string; lo : int; hi : int; dtype : Dtype.t }
        (** Scalar kernel parameter with runtime bounds \[[lo];[hi]\]. Must be
            scalar int/index with [lo <= hi]. *)
    | Bufferize of {
        src : ref;
        idx : ref;
        ranges : ref list;
        dtype : Dtype.ptr;
        opts : bufferize_opts;
      }
        (** Materialize [src] into a buffer at linear index [idx]. [ranges]
            constrain the materialization scope. [idx] and each element of
            [ranges] must be index-like (index or int32 scalar). Lowered during
            linearization. *)
    (* {2 Constants} *)
    | Const of { value : const; dtype : Dtype.t }
        (** Constant literal. The [value] variant must match the [dtype] scalar
            kind: [Bool] requires bool dtype, [Int] requires int/index, [Float]
            requires float, [Invalid] requires index. *)
    (* {2 Memory operations} *)
    | Index of {
        ptr : ref;
        idxs : ref list;
        gate : ref option;
        dtype : Dtype.ptr;
      }
        (** Pointer arithmetic: [ptr + sum(idxs)], with optional boolean [gate].
            [ptr] must be a pointer definition ([Param], [Define_local],
            [Define_reg], [Bufferize], or [Ptrcat]). [idxs] must be non-empty
            index scalars. [gate], when present, must be a bool scalar. [dtype]
            must match the base pointer type. Uses {!Dtype.ptr} (unlike
            {!Tensor.instr.Index} which uses {!Dtype.t}). *)
    | Ptrcat of { srcs : ref list; dtype : Dtype.ptr }
        (** Concatenate pointer vectors. Sources must be non-empty, all pointers
            with matching addrspace, base dtype, and image metadata. Total
            pointer vector width of sources must equal [dtype.v]. *)
    | Load of { src : ref; alt : ref option; dtype : Dtype.t }
        (** Load value from memory. [src] must reference a pointer. [dtype] must
            match the pointer's base type. If [alt] is provided, the source
            [Index] must be gated and [alt] supplies the fallback value when the
            gate is false. *)
    | Store of { dst : ref; value : ref; ranges : ref list }
        (** Store [value] to memory at [dst]. [dst] must reference a pointer.
            [value] dtype must match the pointer's base type. [ranges] constrain
            the store scope. Unlike {!Tensor.instr.Store} and
            {!Program.instr.Store}, carries explicit range constraints. *)
    (* {2 Unary ALU} *)
    | Neg of { src : ref; dtype : Dtype.t }  (** Negation. *)
    | Exp2 of { src : ref; dtype : Dtype.t }  (** Base-2 exponential. *)
    | Log2 of { src : ref; dtype : Dtype.t }  (** Base-2 logarithm. *)
    | Sin of { src : ref; dtype : Dtype.t }  (** Sine. *)
    | Sqrt of { src : ref; dtype : Dtype.t }  (** Square root. *)
    | Recip of { src : ref; dtype : Dtype.t }  (** Reciprocal ([1/x]). *)
    | Trunc of { src : ref; dtype : Dtype.t }  (** Truncate toward zero. *)
    (* {2 Binary ALU} *)
    | Add of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Addition. *)
    | Sub of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Subtraction. *)
    | Mul of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Multiplication. *)
    | Fdiv of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Float division. *)
    | Idiv of { lhs : ref; rhs : ref; dtype : Dtype.t }
        (** Integer division. Requires int/index dtype. *)
    | Mod of { lhs : ref; rhs : ref; dtype : Dtype.t }
        (** Modulo. Requires int/index dtype. *)
    | Max of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Maximum. *)
    | Pow of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Power. *)
    | Shl of { lhs : ref; rhs : ref; dtype : Dtype.t }
        (** Left shift. Requires int/index dtype. [rhs] must match [lhs] dtype
            or be [uint32]. *)
    | Shr of { lhs : ref; rhs : ref; dtype : Dtype.t }
        (** Right shift. Requires int/index dtype. [rhs] must match [lhs] dtype
            or be [uint32]. *)
    | And of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Bitwise AND. *)
    | Or of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Bitwise OR. *)
    | Xor of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Bitwise XOR. *)
    | Threefry of { lhs : ref; rhs : ref; dtype : Dtype.t }
        (** Threefry2x32 mixing function for random number generation. *)
    | Cmplt of { lhs : ref; rhs : ref; dtype : Dtype.t }
        (** Less than. Result dtype must be bool. *)
    | Cmpeq of { lhs : ref; rhs : ref; dtype : Dtype.t }
        (** Equal. Result dtype must be bool. *)
    | Cmpne of { lhs : ref; rhs : ref; dtype : Dtype.t }
        (** Not equal. Result dtype must be bool. *)
    (* {2 Ternary ALU} *)
    | Where of { cond : ref; a : ref; b : ref; dtype : Dtype.t }
        (** Conditional select: if [cond] then [a] else [b]. [cond] must be bool
            scalar. *)
    | Mulacc of { a : ref; b : ref; c : ref; dtype : Dtype.t }
        (** Fused multiply-accumulate: [a * b + c]. All operands must have
            matching dtypes. *)
    (* {2 Type conversion} *)
    | Cast of { src : ref; dtype : Dtype.t }
        (** Type conversion with value preservation. *)
    | Bitcast of { src : ref; dtype : Dtype.t }
        (** Reinterpret bits as a different type (no value conversion). *)
    (* {2 Vector operations} *)
    | Vectorize of { srcs : ref list; dtype : Dtype.t }
        (** Pack scalars into a vector. [srcs] must be non-empty, all scalar,
            matching [dtype.scalar]. Length of [srcs] must equal [dtype.count].
        *)
    | Cat of { srcs : ref list; dtype : Dtype.t }
        (** Concatenate vectors. All sources must have matching scalar type.
            Total element count of sources must equal [dtype.count]. *)
    | Gep of { src : ref; idx : int; dtype : Dtype.t }
        (** Extract element [idx] from a vector (Get Element Pointer). [src]
            must be a vector; [idx] must be in bounds. Result is scalar of
            [src]'s scalar type. *)
    (* {2 Control flow} *)
    | Range of { size : ref; dtype : Dtype.t; axis : int; kind : axis_kind }
        (** Begin iteration from [0] to [size - 1]. Produces the loop variable.
            [axis] identifies the loop dimension. [kind] encodes the axis
            semantics. [dtype] must be scalar int/index and must match the dtype
            of [size]. *)
    | End of { value : ref; ranges : ref list }
        (** Close a computation scope, producing [value]. [ranges] are
            references to associated [Range] or index-like instructions.
            Distinct from {!Program.instr.End_range}, which explicitly closes a
            single loop. *)
    | Barrier  (** Workgroup synchronization barrier. *)
    (* {2 GPU special} *)
    | Special of { dim : special_dim; size : ref; dtype : Dtype.t }
        (** GPU thread ID for [dim], bounded by [size]. [dtype] must be index or
            int32 scalar and must match [size]'s dtype. *)
    (* {2 Reduction} *)
    | Reduce of {
        op : reduce_op;
        src : ref;
        ranges : ref list;
        dtype : Dtype.t;
      }
        (** Reduction over [ranges] using [op]. [src] dtype must match [dtype].
            Lowered during linearization. *)
    | Unroll of { src : ref; axes : (int * int) list; dtype : Dtype.t }
        (** Unroll vector [src] along [axes]. Each [(axis, size)] pair specifies
            a dimension to unroll. Source vector count must equal
            [product(sizes) * dtype.count]. Lowered during linearization. *)
    | Contract of { src : ref; axes : (int * int) list; dtype : Dtype.t }
        (** Contract (fold) dimensions of [src] along [axes]. Inverse of
            {!instr.Unroll}. [dtype.count] must equal [product(sizes)]. Lowered
            during linearization. *)
    (* {2 Tensor core} *)
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
        (** Tensor core matrix multiply-accumulate.
            - [dims] is [(N, M, K)] matrix dimensions.
            - [dtype_in] is the input element type.
            - [dtype_out] is the accumulator element type.
            - [name] encodes the operation for preamble generation.
            - [threads] is the number of cooperating threads per WMMA operation.
            - [ranges] constrain the reduction scope.
            - [upcast_axes] are [(axis, size)] pairs for register tiling of [a],
              [b], and [c] respectively.
            - [reduce_axes] are reduction axis indices. *)
    (* {2 Custom code injection} *)
    | Custom of { fmt : string; args : ref list }
        (** Inject custom code as a statement. [fmt] is a backend format string
            with positional placeholders substituted by the rendered forms of
            [args] during code emission. Produces no value. *)
    | Custom_inline of { fmt : string; args : ref list; dtype : Dtype.t }
        (** Inject custom code as an expression. Same [fmt] convention as
            {!instr.Custom}. Produces a value of type [dtype]. *)

  (** {1:kernel-representation Representation} *)

  type t = instr array
  (** The type for kernel programs. An array of instructions where indices serve
      as references. *)

  (** {1:kernel-inspection Inspection} *)

  val dtype_of : instr -> Dtype.t option
  (** [dtype_of instr] is the result dtype of [instr], or [None] for
      instructions that produce no value ([Sink], [Group], [After], [Store],
      [End], [Barrier], [Custom]).

      For pointer-producing instructions ([Param], [Define_local], [Define_reg],
      [Bufferize], [Index], [Ptrcat]), returns the {e base type}, not a pointer
      type. *)

  val refs_of : instr -> ref list
  (** [refs_of instr] is all operand references of [instr] in definition order.
      The list is complete and may contain duplicates. *)

  val map_refs : (ref -> ref) -> instr -> instr
  (** [map_refs f instr] is [instr] with [f] applied to every operand reference.
      Non-reference fields (dtype, constants, options) are preserved. *)

  (** {1:kernel-predicates Predicates} *)

  val is_unary : instr -> bool
  (** [is_unary instr] is [true] iff [instr] is [Neg], [Exp2], [Log2], [Sin],
      [Sqrt], [Recip], or [Trunc]. *)

  val is_binary : instr -> bool
  (** [is_binary instr] is [true] iff [instr] is a two-operand arithmetic or
      comparison operation ([Add], [Sub], [Mul], [Fdiv], [Idiv], [Mod], [Max],
      [Pow], [Shl], [Shr], [And], [Or], [Xor], [Threefry], [Cmplt], [Cmpeq],
      [Cmpne]). *)

  val is_ternary : instr -> bool
  (** [is_ternary instr] is [true] iff [instr] is [Where] or [Mulacc]. *)

  val is_alu : instr -> bool
  (** [is_alu instr] is [true] iff [instr] is a unary, binary, or ternary ALU
      operation. Equivalent to
      [is_unary instr || is_binary instr || is_ternary instr]. *)

  (** {1:kernel-transform Transformation} *)

  val intern : t -> t
  (** [intern t] deduplicates structurally equal instructions, producing a
      semantically equivalent program with no duplicate instructions. References
      are remapped to point to the canonical (first-seen) copy. Uses structural
      equality and hashing. Idempotent. *)

  (** {1:kernel-validation Validation} *)

  val validate : t -> unit
  (** [validate t] checks that [t] satisfies the Kernel IR invariants described
      in {!section:kernel-invariants}.

      Raises [Failure] with a message identifying the instruction index and the
      specific violation. *)

  (** {1:kernel-fmt Formatting} *)

  val pp_instr : Format.formatter -> instr -> unit
  (** [pp_instr] formats a single instruction for debugging, one line. *)

  val pp : Format.formatter -> t -> unit
  (** [pp] formats all instructions with their indices, one per line. Output
      format: [{i}: {instr}] where [{i}] is the zero-based index. *)
end

(** {1:tensor Tensor IR}

    High-level tensor graph for scheduling and fusion.

    Top-level IR representing the full computation graph including buffers,
    movement ops, multi-device operations, and autograd metadata. Lowered to
    {!Kernel} during scheduling, then to {!Program} for code emission.

    This IR includes operations not present in {!Kernel} or {!Program}:
    - Buffer management: [Buffer], [Buffer_view], [Unique], [Lunique], [Device].
    - Movement ops: [Reshape], [Expand], [Pad], [Shrink], [Permute], [Flip].
    - Multi-device ops: [Multi], [Mstack], [Mselect], [Allreduce].
    - Autograd support: [Detach], [Contiguous_backward], [Assign].

    Metadata (names, gradient functions, custom kernels) is stored in registries
    and referenced by opaque IDs to keep instruction records small and avoid
    duplication.

    {2:tensor-invariants Invariants}

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
    - [Reshape] shape must not contain negative dimensions.
    - [Permute.order] must be a valid permutation of [0..n-1].
    - [Mstack] must have at least one source.
    - [Reduce_axis] must have at least one axis; axes must be unique.
    - [Range] must be scalar int/index; [size] must match dtype.
    - [Index] must have at least one index (index scalars); optional gate must
      be bool scalar. Unlike {!Kernel}, [Index.ptr] has no structural constraint
      (only [idxs] and [gate] are validated).
    - ALU operand dtypes must match the result dtype; comparisons return bool.
    - [Idiv]/[Mod] must have int/index dtype.
    - Shifts must have int/index dtype; rhs must match lhs dtype or be [uint32].
    - [Cast] must preserve vector width.
    - [Vectorize] source count must equal [dtype.count]; all sources scalar.
    - [Flip] uses a [bool list] (one per dimension; [true] = flip).
      [Flip { dims = [false; true; false] }] flips axis 1. *)
module Tensor : sig
  (** {1:tensor-registries Registries}

      Metadata types stored by reference. Use [register] to intern a value and
      [get] to retrieve it by ID. All registries are global and mutable; IDs are
      valid for the lifetime of the process. *)

  (** Tensor operation metadata for debugging and profiling. *)
  module Metadata : sig
    type t = {
      name : string;  (** Operation name. *)
      caller : string;  (** Source location or caller identifier. *)
      backward : bool;
          (** [true] when this operation is part of the backward pass. *)
    }

    type id
    (** The type for metadata identifiers. *)

    val register : t -> id
    (** [register t] interns metadata and returns its unique ID. *)

    val get : id -> t
    (** [get id] is the metadata previously registered with [id].

        Raises [Not_found] if [id] was not returned by {!register}. *)
  end

  (** Interned {!Kernel.kernel_info} values. *)
  module Kernel_info : sig
    type t = Kernel.kernel_info
    (** The type for kernel info values. *)

    type id
    (** The type for kernel info identifiers. *)

    val register : t -> id
    (** [register t] interns kernel info and returns its unique ID. *)

    val get : id -> t
    (** [get id] is the kernel info previously registered with [id].

        Raises [Not_found] if [id] was not returned by {!register}. *)
  end

  (** Gradient function reference. Stored as a name string for serialization. *)
  module Grad_fxn : sig
    type t = { name : string }
    (** The type for gradient function references. *)

    type id
    (** The type for gradient function identifiers. *)

    val register : t -> id
    (** [register t] interns a gradient function and returns its unique ID. *)

    val get : id -> t
    (** [get id] is the gradient function previously registered with [id].

        Raises [Not_found] if [id] was not returned by {!register}. *)
  end

  (** Custom kernel reference.

      [ast] optionally stores a pre-lowered {!Kernel.t} body for custom kernels
      that can be resolved during earliest rewrites. *)
  module Custom_kernel : sig
    type t = {
      name : string;  (** Kernel name for identification and codegen. *)
      grad : Grad_fxn.id option;
          (** Gradient function. [None] for non-differentiable kernels. *)
      ast : Kernel.t option;
          (** Pre-lowered Kernel IR body. [None] for opaque kernels. *)
      metadata : Metadata.id list;  (** Associated operation metadata entries. *)
    }

    type id
    (** The type for custom kernel identifiers. *)

    val register : t -> id
    (** [register t] interns a custom kernel and returns its unique ID. *)

    val get : id -> t
    (** [get id] is the custom kernel previously registered with [id].

        Raises [Not_found] if [id] was not returned by {!register}. *)
  end

  (** {1:tensor-types Types} *)

  (** Constant literal values.

      [Invalid] represents an uninitialized or undefined value (requires [Index]
      dtype). Same variants as {!Kernel.const}. *)
  type const = Bool of bool | Int of int | Float of float | Invalid

  (** Reduction operation kind. Same variants as {!Kernel.reduce_op}. *)
  type reduce_op = Add | Mul | Max

  (** Device specification. *)
  type device =
    | Single of string  (** Single named device. *)
    | Multi of string list  (** Sharded across multiple devices. *)

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
  (** The type for instruction references. An index into the instruction array.
  *)

  (** {1:tensor-instrs Instructions} *)

  (** Tensor IR instruction.

      Organized into graph management, identity and device, buffers, constants
      and variables, parameters and calls, kernels, assignment and autograd,
      data movement, multi-device, shape operations, control flow, memory,
      vectors, type conversion, and ALU. *)
  type instr =
    (* {2 Graph management} *)
    | Sink of { srcs : ref list; kernel_info : kernel_info option }
        (** Program root. Collects side-effecting instructions. *)
    | Group of { srcs : ref list }
        (** Group related instructions for scheduling. *)
    | After of { src : ref; deps : ref list; dtype : Dtype.t }
        (** Execution dependency: [src] must execute after [deps]. Carries
            [src]'s dtype (unlike {!Kernel.instr.After}). *)
    (* {2 Identity and device} *)
    | Unique of { id : int }
        (** Unique buffer identity marker. Referenced by [Buffer.unique]. *)
    | Lunique of { id : int }
        (** Lazy unique buffer identity marker. Like [Unique] but for buffers
            that may not yet be materialized. *)
    | Device of { device : device }
        (** Device specification node. Referenced by [Buffer], [Copy],
            [Allreduce], and [Param]. *)
    (* {2 Buffers} *)
    | Buffer of { unique : ref; device : ref; size : int; dtype : Dtype.t }
        (** Allocated buffer. [unique] must reference [Unique]/[Lunique];
            [device] must reference [Device]; [size] is element count
            (non-negative). *)
    | Buffer_view of { src : ref; size : int; offset : int; dtype : Dtype.t }
        (** View into an existing buffer at byte [offset] for [size] elements.
            [src] must reference [Index]. Both [size] and [offset] must be
            non-negative. *)
    (* {2 Constants and variables} *)
    | Const of { value : const; dtype : Dtype.t; srcs : ref list }
        (** Constant literal. [srcs] are data-dependency references that
            establish shape context for broadcasting; empty for scalar constants
            with no shape dependencies. The [value] variant must match the
            [dtype] scalar kind. *)
    | Vconst of { values : const list; dtype : Dtype.t; srcs : ref list }
        (** Vector constant. Length of [values] must equal [dtype.count]. Each
            element variant must match the [dtype] scalar kind. [srcs] serve the
            same role as in {!instr.Const}. *)
    | Define_var of { name : string; lo : int; hi : int; dtype : Dtype.t }
        (** Symbolic variable with runtime bounds \[[lo];[hi]\]. Must be scalar
            int/index with [lo <= hi]. *)
    | Bind of { var : ref; value : ref option; dtype : Dtype.t }
        (** Bind a concrete [value] to a symbolic [var]. [var] must reference
            [Define_var]. [dtype] must match the variable's dtype. *)
    (* {2 Parameters and calls} *)
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
      }  (** Function call. [fn] dtype must match [dtype]. *)
    (* {2 Kernels} *)
    | Custom_kernel of { srcs : ref list; kernel : custom_kernel }
        (** User-defined kernel. Identified by name via {!Custom_kernel}. *)
    | Kernel of { srcs : ref list; kernel : kernel }
        (** Scheduled kernel. [srcs] are the buffer and dependency inputs. *)
    (* {2 Assignment and autograd} *)
    | Assign of {
        target : ref;
        value : ref;
        extras : ref list;
        dtype : Dtype.t;
      }
        (** In-place assignment: write [value] into [target]. [extras] are
            additional [Assign] dependencies for ordering. Both [target] and
            [value] must match [dtype]. *)
    | Detach of { src : ref; dtype : Dtype.t }
        (** Detach [src] from the autograd graph. *)
    | Contiguous of { src : ref; ranges : ref list; dtype : Dtype.t }
        (** Force [src] to contiguous memory layout. [ranges] are index scalar
            constraints. *)
    | Contiguous_backward of { src : ref; dtype : Dtype.t }
        (** Backward pass marker for contiguous. *)
    (* {2 Data movement} *)
    | Copy of { src : ref; device : ref; dtype : Dtype.t }
        (** Copy [src] to [device]. [device] must reference [Device]. *)
    | Allreduce of { src : ref; device : ref; op : reduce_op; dtype : Dtype.t }
        (** All-reduce [src] across devices using [op]. [device] must reference
            [Device]. *)
    (* {2 Multi-device} *)
    | Multi of { src : ref; axis : int; dtype : Dtype.t }
        (** Mark [src] as sharded along [axis] for multi-device. *)
    | Mstack of { srcs : ref list; dtype : Dtype.t }
        (** Stack single-device tensors into a multi-device tensor. [srcs] must
            be non-empty; all sources must match [dtype]. *)
    | Mselect of { src : ref; index : int; dtype : Dtype.t }
        (** Select shard [index] from a multi-device tensor. *)
    | Encdec of { srcs : ref list; shape : int list; dtype : Dtype.t }
        (** Encode/decode operation with target [shape]. *)
    (* {2 Reduction} *)
    | Reduce_axis of {
        src : ref;
        op : reduce_op;
        axes : int list;
        dtype : Dtype.t;
      }
        (** Reduce [src] along [axes] using [op]. High-level form before range
            lowering. [axes] must be non-empty and contain no duplicates. *)
    | Reduce of {
        src : ref;
        ranges : ref list;
        op : reduce_op;
        dtype : Dtype.t;
      }
        (** Reduce [src] over [ranges] using [op]. Lowered form with explicit
            range references. *)
    (* {2 Shape operations} *)
    | Reshape of { src : ref; shape : ref; dtype : Dtype.t }
        (** Reshape [src] to [shape] (index vector). Shape dimensions must be
            non-negative. *)
    | Expand of { src : ref; shape : ref; dtype : Dtype.t }
        (** Broadcast [src] to [shape] (index vector). *)
    | Pad of { src : ref; before : ref; after : ref; dtype : Dtype.t }
        (** Pad [src] with zeros. [before] and [after] are index vectors of
            equal width specifying padding per dimension. *)
    | Shrink of { src : ref; before : ref; after : ref; dtype : Dtype.t }
        (** Shrink (crop) [src]. [before] and [after] are index vectors of equal
            width specifying bounds per dimension. *)
    | Permute of { src : ref; order : int list; dtype : Dtype.t }
        (** Transpose [src] according to [order]. [order] must be a valid
            permutation of [0..n-1]. *)
    | Flip of { src : ref; dims : bool list; dtype : Dtype.t }
        (** Reverse [src] along selected dimensions. [true] at position [i]
            means flip axis [i]. *)
    (* {2 Control flow} *)
    | Range of { size : ref; dtype : Dtype.t; axis : int; kind : axis_kind }
        (** Begin iteration from [0] to [size - 1]. Produces the loop variable.
            [dtype] must be scalar int/index and must match [size]'s dtype. *)
    | End of { value : ref; ranges : ref list }
        (** Close a computation scope, producing [value]. *)
    (* {2 Memory operations} *)
    | Index of {
        ptr : ref;
        idxs : ref list;
        gate : ref option;
        dtype : Dtype.t;
      }
        (** Pointer arithmetic: [ptr + sum(idxs)], with optional boolean [gate].
            [idxs] must be non-empty index scalars. [gate], when present, must
            be bool scalar. Uses {!Dtype.t} (unlike {!Kernel.instr.Index} which
            uses {!Dtype.ptr}). *)
    | Store of { dst : ref; value : ref }
        (** Store [value] to memory at [dst]. *)
    (* {2 Vector operations} *)
    | Vectorize of { srcs : ref list; dtype : Dtype.t }
        (** Pack scalars into a vector. [srcs] must be non-empty, all scalar,
            matching [dtype.scalar]. Length of [srcs] must equal [dtype.count].
        *)
    (* {2 Type conversion} *)
    | Cast of { src : ref; dtype : Dtype.t }
        (** Type conversion with value preservation. Must preserve vector width.
        *)
    | Bitcast of { src : ref; dtype : Dtype.t }
        (** Reinterpret bits as a different type (no value conversion). *)
    (* {2 Unary ALU} *)
    | Neg of { src : ref; dtype : Dtype.t }  (** Negation. *)
    | Exp2 of { src : ref; dtype : Dtype.t }  (** Base-2 exponential. *)
    | Log2 of { src : ref; dtype : Dtype.t }  (** Base-2 logarithm. *)
    | Sin of { src : ref; dtype : Dtype.t }  (** Sine. *)
    | Sqrt of { src : ref; dtype : Dtype.t }  (** Square root. *)
    | Recip of { src : ref; dtype : Dtype.t }  (** Reciprocal ([1/x]). *)
    | Trunc of { src : ref; dtype : Dtype.t }  (** Truncate toward zero. *)
    (* {2 Binary ALU} *)
    | Add of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Addition. *)
    | Sub of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Subtraction. *)
    | Mul of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Multiplication. *)
    | Fdiv of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Float division. *)
    | Idiv of { lhs : ref; rhs : ref; dtype : Dtype.t }
        (** Integer division. Requires int/index dtype. *)
    | Mod of { lhs : ref; rhs : ref; dtype : Dtype.t }
        (** Modulo. Requires int/index dtype. *)
    | Max of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Maximum. *)
    | Pow of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Power. *)
    | Shl of { lhs : ref; rhs : ref; dtype : Dtype.t }
        (** Left shift. Requires int/index dtype. [rhs] must match [lhs] dtype
            or be [uint32]. *)
    | Shr of { lhs : ref; rhs : ref; dtype : Dtype.t }
        (** Right shift. Requires int/index dtype. [rhs] must match [lhs] dtype
            or be [uint32]. *)
    | And of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Bitwise AND. *)
    | Or of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Bitwise OR. *)
    | Xor of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Bitwise XOR. *)
    | Threefry of { lhs : ref; rhs : ref; dtype : Dtype.t }
        (** Threefry2x32 mixing function for random number generation. *)
    | Cmplt of { lhs : ref; rhs : ref; dtype : Dtype.t }
        (** Less than. Result dtype must be bool. *)
    | Cmpeq of { lhs : ref; rhs : ref; dtype : Dtype.t }
        (** Equal. Result dtype must be bool. *)
    | Cmpne of { lhs : ref; rhs : ref; dtype : Dtype.t }
        (** Not equal. Result dtype must be bool. *)
    (* {2 Ternary ALU} *)
    | Where of { cond : ref; a : ref; b : ref; dtype : Dtype.t }
        (** Conditional select: if [cond] then [a] else [b]. [cond] must be bool
            scalar. *)
    | Mulacc of { a : ref; b : ref; c : ref; dtype : Dtype.t }
        (** Fused multiply-accumulate: [a * b + c]. All operands must have
            matching dtypes. *)

  (** {1:tensor-representation Representation} *)

  type t = instr array
  (** The type for tensor programs. An array of instructions where indices serve
      as references. *)

  (** {1:tensor-inspection Inspection} *)

  val dtype_of : instr -> Dtype.t option
  (** [dtype_of instr] is the result dtype of [instr], or [None] for
      instructions that produce no value ([Sink], [Group], [Unique], [Lunique],
      [Device], [Custom_kernel], [Kernel], [End], [Store]). *)

  val refs_of : instr -> ref list
  (** [refs_of instr] is all operand references of [instr] in definition order.
      The list is complete and may contain duplicates. *)

  val map_refs : (ref -> ref) -> instr -> instr
  (** [map_refs f instr] is [instr] with [f] applied to every operand reference.
      Non-reference fields (dtype, constants, options) are preserved. *)

  (** {1:tensor-predicates Predicates} *)

  val is_unary : instr -> bool
  (** [is_unary instr] is [true] iff [instr] is [Neg], [Exp2], [Log2], [Sin],
      [Sqrt], [Recip], or [Trunc]. *)

  val is_binary : instr -> bool
  (** [is_binary instr] is [true] iff [instr] is a two-operand arithmetic or
      comparison operation ([Add], [Sub], [Mul], [Fdiv], [Idiv], [Mod], [Max],
      [Pow], [Shl], [Shr], [And], [Or], [Xor], [Threefry], [Cmplt], [Cmpeq],
      [Cmpne]). *)

  val is_ternary : instr -> bool
  (** [is_ternary instr] is [true] iff [instr] is [Where] or [Mulacc]. *)

  val is_alu : instr -> bool
  (** [is_alu instr] is [true] iff [instr] is a unary, binary, or ternary ALU
      operation. Equivalent to
      [is_unary instr || is_binary instr || is_ternary instr]. *)

  (** {1:tensor-transform Transformation} *)

  val intern : t -> t
  (** [intern t] deduplicates structurally equal instructions, producing a
      semantically equivalent program with no duplicate instructions. References
      are remapped to point to the canonical (first-seen) copy. Uses structural
      equality and hashing. Idempotent. *)

  (** {1:tensor-validation Validation} *)

  val validate : t -> unit
  (** [validate t] checks that [t] satisfies the Tensor IR invariants described
      in {!section:tensor-invariants}.

      Raises [Failure] with a message identifying the instruction index and the
      specific violation. *)

  (** {1:tensor-fmt Formatting} *)

  val pp_instr : Format.formatter -> instr -> unit
  (** [pp_instr] formats a single instruction for debugging, one line. *)

  val pp : Format.formatter -> t -> unit
  (** [pp] formats all instructions with their indices, one per line. Output
      format: [{i}: {instr}] where [{i}] is the zero-based index. *)
end

(** {1:program Program IR}

    Low-level IR for code generation.

    Instructions map directly to rendered code (C, CUDA, Metal, etc.). The IR
    uses a flattened SSA representation where each instruction references its
    operands by index into the instruction array.

    This IR is post-devectorizer and render-ready. Vector masks, per-lane vector
    constants, multi-element GEPs, and Index-typed values must be lowered before
    constructing a {!Program.t}. The renderer assumes only backend-legal
    operations and scalar control flow.

    {2:program-invariants Invariants}

    Well-formed programs must satisfy:
    - All references point to earlier instructions (SSA dominance).
    - No [Index] dtype values remain (lowered before linearization).
    - [Param] has [Global] addrspace, [Define_local] has [Local], [Define_reg]
      has [Reg].
    - [Define_var] is scalar with valid bounds ([lo <= hi]).
    - [Range]/[End_range] and [If]/[Endif] pairs are properly nested and
      explicitly paired.
    - [Range] dtype is integer, scalar, and matches [size]'s dtype.
    - [If] condition must be scalar bool; [idx_for_dedup] must reference an
      [Index] (or casted [Index]).
    - [Index] base must be a pointer definition ([Param], [Define_local],
      [Define_reg]); each index operand must be scalar int; optional gate must
      be scalar bool.
    - [Load] src must reference an [Index] (or casted [Index]); [alt] requires a
      gated [Index].
    - [Store] dst must reference an [Index] (or casted [Index]).
    - Each [Special] dimension (e.g., [Group_id 0]) appears at most once. The
      flat array representation does not deduplicate automatically, and the
      renderer emits variable declarations, so duplicates cause redeclaration
      errors in generated code.
    - [Special] dtype must be [int32] scalar and match [size]'s dtype.
    - [Where] condition must be scalar bool; branches must match result dtype.
      C-style ternary requires scalar conditions; the devectorizer ensures this
      before codegen.
    - Comparisons return bool with matching operand dtypes.
    - [Idiv]/[Mod] must have int dtype.
    - Binary/unary ALU operands match result dtype; shift rhs matches lhs dtype
      or is [uint32].
    - [Vectorize] source count equals [dtype.count] and is > 1; all sources are
      scalar with matching scalar type.
    - [Gep] source must be a vector; index in bounds; result is scalar of source
      type.
    - [Wmma] dims are positive; result dtype matches [dtype_out]. *)
module Program : sig
  (** {1:program-types Types} *)

  (** Constant literal values.

      Unlike {!Kernel.const} and {!Tensor.const}, [Invalid] is not valid in
      rendered programs (Index dtype is lowered away). *)
  type const = Bool of bool | Int of int | Float of float

  type ref = int
  (** The type for instruction references. An index into the instruction array.
  *)

  (** {1:program-instrs Instructions} *)

  (** Program IR instruction.

      Organized into memory definitions, constants, memory operations, ALU
      (unary, binary, ternary), type conversion, vector operations, control
      flow, synchronization, GPU special, tensor core, and custom code
      injection. *)
  type instr =
    (* {2 Memory definitions} *)
    | Param of { idx : int; dtype : Dtype.ptr }
        (** Kernel parameter at position [idx]. Address space must be [Global].
        *)
    | Define_local of { size : int; dtype : Dtype.ptr }
        (** Allocate shared memory for [size] elements. Address space must be
            [Local]. *)
    | Define_reg of { size : int; dtype : Dtype.ptr }
        (** Allocate register storage for [size] elements. Address space must be
            [Reg]. *)
    | Define_var of { name : string; lo : int; hi : int; dtype : Dtype.t }
        (** Scalar kernel parameter with runtime bounds \[[lo];[hi]\]. Must be
            scalar with [lo <= hi]. *)
    (* {2 Constants} *)
    | Const of { value : const; dtype : Dtype.t }  (** Constant literal. *)
    (* {2 Memory operations} *)
    | Index of {
        ptr : ref;
        idxs : ref list;
        gate : ref option;
        dtype : Dtype.ptr;
      }
        (** Pointer arithmetic: [ptr + sum(idxs)], with optional boolean [gate].
            [ptr] must be a pointer definition ([Param], [Define_local],
            [Define_reg]). [idxs] must be non-empty scalar ints. [gate], when
            present, must be scalar bool. *)
    | Load of { src : ref; alt : ref option; dtype : Dtype.t }
        (** Load value from memory. [src] must reference an [Index] (or casted
            [Index]). If [alt] is provided, the source [Index] must be gated and
            [alt] supplies the fallback value when the gate is false; [alt]
            dtype must match [dtype]. *)
    | Store of { dst : ref; value : ref }
        (** Store [value] to memory at [dst]. [dst] must reference an [Index]
            (or casted [Index]). *)
    (* {2 Unary ALU} *)
    | Neg of { src : ref; dtype : Dtype.t }  (** Negation. *)
    | Exp2 of { src : ref; dtype : Dtype.t }  (** Base-2 exponential. *)
    | Log2 of { src : ref; dtype : Dtype.t }  (** Base-2 logarithm. *)
    | Sin of { src : ref; dtype : Dtype.t }  (** Sine. *)
    | Sqrt of { src : ref; dtype : Dtype.t }  (** Square root. *)
    | Recip of { src : ref; dtype : Dtype.t }  (** Reciprocal ([1/x]). *)
    | Trunc of { src : ref; dtype : Dtype.t }  (** Truncate toward zero. *)
    (* {2 Binary ALU} *)
    | Add of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Addition. *)
    | Sub of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Subtraction. *)
    | Mul of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Multiplication. *)
    | Fdiv of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Float division. *)
    | Idiv of { lhs : ref; rhs : ref; dtype : Dtype.t }
        (** Integer division. Requires int dtype. *)
    | Mod of { lhs : ref; rhs : ref; dtype : Dtype.t }
        (** Modulo. Requires int dtype. *)
    | Max of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Maximum. *)
    | Pow of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Power. *)
    | Shl of { lhs : ref; rhs : ref; dtype : Dtype.t }
        (** Left shift. Requires int dtype. [rhs] must match [lhs] dtype or be
            [uint32]. *)
    | Shr of { lhs : ref; rhs : ref; dtype : Dtype.t }
        (** Right shift. Requires int dtype. [rhs] must match [lhs] dtype or be
            [uint32]. *)
    | And of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Bitwise AND. *)
    | Or of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Bitwise OR. *)
    | Xor of { lhs : ref; rhs : ref; dtype : Dtype.t }  (** Bitwise XOR. *)
    | Threefry of { lhs : ref; rhs : ref; dtype : Dtype.t }
        (** Threefry2x32 mixing function for random number generation. *)
    | Cmplt of { lhs : ref; rhs : ref; dtype : Dtype.t }
        (** Less than. Result dtype must be bool. *)
    | Cmpeq of { lhs : ref; rhs : ref; dtype : Dtype.t }
        (** Equal. Result dtype must be bool. *)
    | Cmpne of { lhs : ref; rhs : ref; dtype : Dtype.t }
        (** Not equal. Result dtype must be bool. *)
    (* {2 Ternary ALU} *)
    | Where of { cond : ref; a : ref; b : ref; dtype : Dtype.t }
        (** Conditional select: if [cond] then [a] else [b]. [cond] must be
            scalar bool. Branches must match [dtype]. *)
    | Mulacc of { a : ref; b : ref; c : ref; dtype : Dtype.t }
        (** Fused multiply-accumulate: [a * b + c]. All operands must have
            matching dtypes. *)
    (* {2 Type conversion} *)
    | Cast of { src : ref; dtype : Dtype.t }
        (** Type conversion with value preservation. *)
    | Bitcast of { src : ref; dtype : Dtype.t }
        (** Reinterpret bits as a different type (no value conversion). *)
    (* {2 Vector operations} *)
    | Vectorize of { srcs : ref list; dtype : Dtype.t }
        (** Pack scalars into a vector. [srcs] must have more than one element,
            all scalar, matching [dtype.scalar]. Length of [srcs] must equal
            [dtype.count]. *)
    | Gep of { src : ref; idx : int; dtype : Dtype.t }
        (** Extract element [idx] from a vector (Get Element Pointer). [src]
            must be a vector with count > 1; [idx] must be in bounds. Result is
            scalar of [src]'s scalar type. *)
    (* {2 Control flow} *)
    | Range of { size : ref; dtype : Dtype.t; axis : int; kind : axis_kind }
        (** Begin loop from [0] to [size - 1]. Produces the loop variable.
            [axis] identifies the loop dimension (0=x, 1=y, etc.) for variable
            naming and axis-specific optimizations. [kind] encodes the axis
            semantics. [dtype] must be int, scalar, and match [size]'s dtype. *)
    | End_range of { range : ref }
        (** End loop started by [range]. [range] must reference a [Range]
            instruction. Must be properly nested with respect to other
            [Range]/[End_range] and [If]/[Endif] pairs. *)
    | If of { cond : ref; idx_for_dedup : ref }
        (** Begin conditional block. [cond] must be scalar bool. [idx_for_dedup]
            must reference an [Index] (or casted [Index]); it provides a
            structural key for deduplication during optimization. *)
    | Endif of { if_ : ref }
        (** End conditional block started by [if_]. [if_] must reference an [If]
            instruction. Must be properly nested with respect to other
            [Range]/[End_range] and [If]/[Endif] pairs. *)
    (* {2 Synchronization} *)
    | Barrier  (** Workgroup synchronization barrier. *)
    (* {2 GPU special} *)
    | Special of { dim : special_dim; size : ref; dtype : Dtype.t }
        (** GPU thread ID for [dim], bounded by [size]. [dtype] must be [int32]
            scalar and match [size]'s dtype. Each [dim] must appear at most once
            in a program. *)
    (* {2 Tensor core} *)
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
        (** Tensor core matrix multiply-accumulate.
            - [dims] is [(N, M, K)] matrix dimensions. All must be positive.
            - [dtype_in] is the input element type.
            - [dtype_out] is the accumulator element type. Must match
              [dtype.scalar].
            - [name] encodes the operation for preamble generation.
            - [threads] is the number of cooperating threads per WMMA operation.
            - [upcast_axes] are [(axis, size)] pairs for register tiling of [a],
              [b], and [c] respectively.
            - [reduce_axes] are reduction axis indices. *)
    (* {2 Custom code injection} *)
    | Custom of { fmt : string; args : ref list }
        (** Inject custom code as a statement. [fmt] is a backend format string
            with positional placeholders substituted by the rendered forms of
            [args] during code emission. Produces no value. *)
    | Custom_inline of { fmt : string; args : ref list; dtype : Dtype.t }
        (** Inject custom code as an expression. Same [fmt] convention as
            {!instr.Custom}. Produces a value of type [dtype]. *)

  (** {1:program-representation Representation} *)

  type t = instr array
  (** The type for programs. An array of instructions where indices correspond
      to SSA variable names. *)

  (** {1:program-inspection Inspection} *)

  val dtype_of : instr -> Dtype.t option
  (** [dtype_of instr] is the result dtype of [instr], or [None] for
      instructions that produce no value ([Store], [End_range], [If], [Endif],
      [Barrier], [Custom]).

      For pointer-producing instructions ([Param], [Define_local], [Define_reg],
      [Index]), returns the {e base type}, not a pointer type. Most consumers
      want "what data flows here" for type checking ALU ops; pointer-ness can be
      determined by pattern matching on the instruction. *)

  val refs_of : instr -> ref list
  (** [refs_of instr] is all operand references of [instr] in definition order.
      The list is complete and may contain duplicates. *)

  val map_refs : (ref -> ref) -> instr -> instr
  (** [map_refs f instr] is [instr] with [f] applied to every operand reference.
      Non-reference fields (dtype, constants, options) are preserved. *)

  val map_alu : map_ref:(ref -> ref) -> dtype:Dtype.t -> instr -> instr
  (** [map_alu ~map_ref ~dtype instr] is [instr] with [map_ref] applied to
      operand references and [dtype] replacing the result dtype. Only valid for
      ALU instructions (unary, binary, ternary).

      Raises [Failure] if [instr] is not an ALU instruction. *)

  (** {1:program-predicates Predicates} *)

  val is_unary : instr -> bool
  (** [is_unary instr] is [true] iff [instr] is [Neg], [Exp2], [Log2], [Sin],
      [Sqrt], [Recip], or [Trunc]. *)

  val is_binary : instr -> bool
  (** [is_binary instr] is [true] iff [instr] is a two-operand arithmetic or
      comparison operation ([Add], [Sub], [Mul], [Fdiv], [Idiv], [Mod], [Max],
      [Pow], [Shl], [Shr], [And], [Or], [Xor], [Threefry], [Cmplt], [Cmpeq],
      [Cmpne]). *)

  val is_ternary : instr -> bool
  (** [is_ternary instr] is [true] iff [instr] is [Where] or [Mulacc]. *)

  val is_alu : instr -> bool
  (** [is_alu instr] is [true] iff [instr] is a unary, binary, or ternary ALU
      operation. Equivalent to
      [is_unary instr || is_binary instr || is_ternary instr]. *)

  (** {1:program-validation Validation} *)

  val validate : t -> unit
  (** [validate t] checks that [t] satisfies the Program IR invariants described
      in {!section:program-invariants}.

      Raises [Failure] with a message identifying the instruction index and the
      specific violation. Call this before rendering to catch malformed IR
      early. *)

  (** {1:program-rebuilding Rebuilding} *)

  val rebuild :
    (emit:(instr -> int) -> map_ref:(int -> int) -> instr -> int option) ->
    t ->
    t
  (** [rebuild f program] constructs a new program by iterating forward through
      [program]. For each instruction, calls [f ~emit ~map_ref instr]:

      - [map_ref r] translates an original ref [r] to its new index in the
        output program.
      - [emit instr] appends [instr] to the output and returns its new index.
      - Return [Some idx] to map the current instruction to [idx] (the caller is
        responsible for having emitted it).
      - Return [None] to copy the instruction with refs automatically remapped
        via [map_ref]. *)

  (** {1:program-fmt Formatting} *)

  val pp_instr : Format.formatter -> instr -> unit
  (** [pp_instr] formats a single instruction for debugging, one line. *)

  val pp : Format.formatter -> t -> unit
  (** [pp] formats all instructions with their indices, one per line. Output
      format: [{i}: {instr}] where [{i}] is the zero-based index. *)
end
