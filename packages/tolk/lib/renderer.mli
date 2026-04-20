(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** GPU kernel renderer.

    A renderer converts {!Ir.Program.t} programs to backend-specific source
    code and owns its {!Compiler.t}. The abstract type {!type-t} encapsulates
    target capabilities (memory hierarchy, grid limits, supported operations),
    a rendering function, and an optional compiler.

    Backends construct renderers via {!make}, supplying only the fields that
    differ from the defaults. The compiler is typically attached by the
    device backend via {!with_compiler} or the [?compiler] parameter of
    {!make}.

    See {!Cstyle} for C-family language backends (CUDA, Metal, OpenCL, HIP,
    Clang). *)

(** {1:types Types} *)

(** ALU operations that a backend can provide custom rendering for.

    The decomposition pass uses {!type-supported_ops} to decide which composite
    operations to lower; {!val-code_for_op} lists the operations a renderer
    handles natively.

    Operations without a corresponding flag in {!type-supported_ops} ([Add],
    [Sub], [Mul], [Mod], [Idiv], [Cmpne], [Xor], [Where], [Trunc]) are always
    required and never decomposed. *)
type code_op =
  | Sqrt  (** Square root. *)
  | Recip  (** Reciprocal ([1/x]). *)
  | Neg  (** Arithmetic negation. *)
  | Exp2  (** Base-2 exponential. *)
  | Log2  (** Base-2 logarithm. *)
  | Sin  (** Sine. *)
  | Trunc  (** Truncation to integer. *)
  | And  (** Bitwise AND. *)
  | Xor  (** Bitwise XOR. *)
  | Or  (** Bitwise OR. *)
  | Add  (** Addition. *)
  | Sub  (** Subtraction. *)
  | Mul  (** Multiplication. *)
  | Mod  (** Modulo. *)
  | Idiv  (** Integer division. *)
  | Cmpne  (** Not-equal comparison. *)
  | Shr  (** Bitwise right shift. *)
  | Shl  (** Bitwise left shift. *)
  | Cmplt  (** Less-than comparison. *)
  | Where  (** Ternary select ([cond ? a : b]). *)
  | Cmpeq  (** Equality comparison. *)
  | Fdiv  (** Floating-point division. *)
  | Max  (** Maximum. *)
  | Mulacc  (** Fused multiply-accumulate. *)
  | Threefry  (** Threefry 2x32 PRNG mixing function. *)

(** {2:supported_ops Supported operations} *)

val all_supported_ops : Tolk_ir.Decompositions.supported_ops
(** [all_supported_ops] marks every decomposable operation as natively
    supported. *)

val supported_ops_of_code_for_op : code_op list -> Tolk_ir.Decompositions.supported_ops
(** [supported_ops_of_code_for_op ops] derives capability flags from a list of
    natively rendered operations. An operation absent from [ops] is marked
    unsupported. *)

(** {1:renderer Renderer} *)

type t
(** The type for renderers. *)

(** {1:properties Properties} *)

val name : t -> string
(** [name r] is the renderer name (e.g., ["metal"], ["cuda"]). *)

val device : t -> string
(** [device r] is the target device identifier (e.g., ["NV"], ["HIP"], ["CPU"]).
    Passed as context to codegen rewrite passes for device-specific
    transformations. *)

val compiler : t -> Compiler.t option
(** [compiler r] is [r]'s compiler, or [None] if the renderer has no
    associated compiler (e.g., interpreter backends). *)

val has_local : t -> bool
(** [has_local r] is [true] iff [r] supports local thread IDs. *)

val has_threads : t -> bool
(** [has_threads r] is [true] iff [r] supports host-side threading instead of
    GPU grid dimensions. *)

val has_shared : t -> bool
(** [has_shared r] is [true] iff [r] supports shared memory. *)

val global_max : t -> int list option
(** [global_max r] is the maximum global grid dimensions [[x; y; z]], or [None]
    when unconstrained. The list has exactly three elements when present. *)

val global_prod_max : t -> int list option
(** [global_prod_max r] is the per-axis product limit for global dimensions, or
    [None] when unconstrained. When present, each global dimension is capped at
    [min(global_max.(i), global_prod_max.(i) / local_hw.(i))]. *)

val local_max : t -> int list option
(** [local_max r] is the maximum local workgroup dimensions [[x; y; z]], or
    [None] when unconstrained. The list has exactly three elements when present.
*)

val shared_max : t -> int
(** [shared_max r] is the maximum shared memory size in bytes.

    - [0] when shared memory is unsupported ({!has_shared} is [false]).
    - For GPU backends, a conservative default (e.g., 32 KB for OpenCL, 48 KB
      for CUDA). Actual limits may vary by device. *)

val tensor_cores : t -> Tc.t list
(** [tensor_cores r] is the list of {!type-tensor_core} configurations supported
    by [r]. Empty when the backend has no hardware matrix-multiply support. *)

(** {1:capabilities Capabilities} *)

val code_for_op : t -> code_op list
(** [code_for_op r] is the list of ALU operations that [r] provides custom
    rendering for.

    See also {!val-supported_ops}. *)

val supported_ops : t -> Tolk_ir.Decompositions.supported_ops
(** [supported_ops r] is the backend capability flags for the decomposition
    pass, derived from {!val-code_for_op} unless explicitly overridden via
    {!make}. *)

val supports_dtype : t -> Tolk_ir.Dtype.t -> bool
(** [supports_dtype r dt] is [true] iff the backend natively supports [dt].
    When [false], the decomposition pass emulates [dt] using supported types. *)

val emulated_float_dtypes : t -> (Tolk_ir.Dtype.scalar * Tolk_ir.Dtype.scalar) list
(** [emulated_float_dtypes r] is the list of [(from, to)] dtype pairs for
    float emulation. Each [from] float is promoted to [to] (typically f32).
    Empty for backends that natively support all float types. *)

val pre_matcher : t -> (Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t option) option
(** [pre_matcher r] is an optional device-specific rewrite rule applied
    before decompositions. *)

val extra_matcher : t -> (Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t option) option
(** [extra_matcher r] is an optional device-specific rewrite rule composed
    into the final rewrite fixpoint. *)

(** {1:load_store Load/store policy} *)

val supports_float4 : t -> bool
(** [supports_float4 r] is [true] iff [r] supports vectorized (float4/float2)
    load and store operations.  The devectorizer uses this to decide whether
    wide accesses can be folded.  Defaults to [true]. *)

(** {1:rendering Rendering} *)

val render : t -> ?name:string -> Tolk_ir.Program.t -> string
(** [render r ~name program] converts [program] to backend-specific source code.

    [name] defaults to ["kernel"]. *)

(** {1:construction Construction} *)

val make :
  ?tensor_cores:Tc.t list ->
  ?supports_float4:bool ->
  ?has_threads:bool ->
  ?global_max:int list ->
  ?global_prod_max:int list ->
  ?local_max:int list ->
  ?code_for_op:code_op list ->
  ?supported_ops:Tolk_ir.Decompositions.supported_ops ->
  ?compiler:Compiler.t ->
  ?pre_matcher:(Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t option) ->
  ?extra_matcher:(Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t option) ->
  name:string ->
  device:string ->
  has_local:bool ->
  has_shared:bool ->
  shared_max:int ->
  render:(?name:string -> Tolk_ir.Program.t -> string) ->
  unit ->
  t
(** [make ~name ~device ~has_local ~has_shared ~shared_max ~render ()] is a
    renderer with the given capabilities.

    Optional parameters and their defaults:
    - [tensor_cores]: [[]] (none).
    - [supports_float4]: [true].
    - [has_threads]: [false].
    - [global_max]: [Some [0x8FFFFFFF; 0x8FFFFFFF; 0x8FFFFFFF]].
    - [global_prod_max]: [None].
    - [local_max]: [Some [0x8FFFFFFF; 0x8FFFFFFF; 0x8FFFFFFF]].
    - [code_for_op]: [[]] (no custom ops).
    - [supported_ops]: derived from [code_for_op] via
      {!supported_ops_of_code_for_op}. When [code_for_op] is [[]], defaults to
      {!all_supported_ops}.
    - [compiler]: [None]. *)

val with_compiler : Compiler.t -> t -> t
(** [with_compiler c r] is [r] with compiler set to [Some c]. *)
