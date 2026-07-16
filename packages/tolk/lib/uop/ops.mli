(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** UOp operation tags.

    Constructor declaration order is part of the semantics: {!compare},
    toposort stability, and commutative-operand canonicalisation use the
    constructor ordinal. Do not add, remove, or reorder constructors without
    updating the parity tests. *)

(** {1:types Types} *)

type t =
  (** {2 Defines and special indices} *)

  | Bind  (** Pairs a symbolic {!Param} with a concrete value. *)
  | Special  (** Hardware index, such as group, local, or global ids. *)
  | Buffer  (** Buffer allocation or buffer identity. *)

  (** {2 Non-op uops} *)

  | Noop  (** Pass-through scheduling marker. *)
  | Rewrite_error  (** Rewrite failure marker. *)
  | Param  (** Function, symbolic, or buffer parameter. *)
  | Function  (** Gradient-able function body. *)
  | Call  (** Opaque kernel invocation. *)
  | Program  (** Program root. *)
  | Linear  (** Linearized uop sequence. *)
  | Source  (** Human-readable rendered source. *)
  | Binary  (** Compiled machine-code bytes. *)
  | Sink  (** Graph or kernel root gathering children. *)
  | After  (** Passes [src.(0)] through after [src.(1..)]. *)
  | Group  (** Merges dependencies without producing a value. *)
  | Stack  (** Constructs value vectors and shape tuples. *)
  | Tuple  (** Multi-result function body. *)
  | Gettuple  (** Projects one element from a {!Tuple} or {!Function}. *)
  | Getaddr  (** HCQ address extraction op. *)

  (** {2 Load and store} *)

  | Index  (** Pointer arithmetic over a base pointer and offsets. *)
  | Shrink  (** Trims edges per axis; declared in tinygrad's load/store block. *)
  | Load  (** Loads from an indexed pointer. *)
  | Store  (** Stores through an indexed pointer. *)

  (** {2 Math} *)

  | Wmma  (** Tensor-core matrix multiply-accumulate. *)
  | Cast  (** Value-preserving type conversion. *)
  | Bitcast  (** Bit-preserving reinterpretation. *)
  | Exp2  (** Base-2 exponential. *)
  | Log2  (** Base-2 logarithm. *)
  | Sin  (** Sine. *)
  | Sqrt  (** Square root. *)
  | Reciprocal  (** Reciprocal, [1 / x]. *)
  | Neg  (** Arithmetic negation. *)
  | Trunc  (** Truncation toward zero. *)
  | Add  (** Addition. *)
  | Mul  (** Multiplication. *)
  | Shl  (** Bitwise left shift. *)
  | Shr  (** Bitwise right shift. *)
  | Cdiv  (** C-style truncating integer division. *)
  | Max  (** Pointwise maximum. *)
  | Cmod  (** C-style integer remainder. *)
  | Cmplt  (** Less-than comparison. *)
  | Cmpne  (** Inequality comparison. *)
  | Cmpeq  (** Equality comparison. *)
  | Xor  (** Bitwise XOR. *)
  | Or  (** Bitwise OR. *)
  | And  (** Bitwise AND. *)
  | Threefry  (** Threefry counter-based RNG round. *)
  | Sub  (** Subtraction. *)
  | Fdiv  (** Floating-point division. *)
  | Pow  (** Exponentiation. *)
  | Floordiv  (** Floor integer division. *)
  | Floormod  (** Floor integer modulo. *)
  | Where  (** Ternary select. *)
  | Mulacc  (** Fused multiply-accumulate. *)

  (** {2 Control flow, constants, backend escapes} *)

  | Barrier  (** Workgroup barrier. *)
  | Range  (** Loop variable. *)
  | If  (** Predicated control-flow gate. *)
  | End  (** Closes one or more loops around a value. *)
  | Endif  (** Closes an {!If} region. *)
  | Wait  (** Wait/synchronisation point. *)
  | Const  (** Compile-time constant. *)
  | Custom  (** Backend-specific statement. *)
  | Customi  (** Backend-specific inline expression. *)
  | Ins  (** Backend machine instruction. *)

  (** {2 Tensor graph and expansion ops} *)

  | Contiguous  (** Forces contiguous layout. *)
  | Contiguous_backward  (** Backward-pass contiguous marker. *)
  | Detach  (** Detaches from gradient tracking. *)
  | Stage  (** Staged buffer before final buffer materialisation. *)
  | Copy  (** Cross-device copy. *)
  | Slice  (** Slice of a buffer identity. *)
  | Mselect  (** Selects one shard of a sharded value. *)
  | Mstack  (** Stacks per-device shards. *)
  | Custom_function  (** User-defined tensor-stage function. *)
  | Reshape  (** Rearranges elements into a new shape. *)
  | Permute  (** Permutes axes. *)
  | Expand  (** Broadcasts to a target shape. *)
  | Pad  (** Pads per axis. *)
  | Flip  (** Reverses selected axes. *)
  | Multi  (** Distributes a value across devices. *)
  | Reduce  (** Reduction by [Add], [Mul], or [Max]. *)
  | Allreduce  (** Cross-device reduction. *)

  (** {2 Pattern compiler IR} *)

  | Pyliteral  (** Carries a literal payload for custom pattern predicates. *)
(** One flat tag per uop kind. *)

val equal : t -> t -> bool
(** [equal a b] is [true] iff [a] and [b] are the same constructor. *)

val compare : t -> t -> int
(** [compare a b] orders by tinygrad declaration order. *)

val name : t -> string
(** [name op] is the uppercase tinygrad operator name. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf op] prints [op] using {!name}. *)

(** {1:groups Groups}

    {!module-Group} exposes operation families used by validation and rewrites.
    Lists use stable declaration-oriented order; predicates and lists agree. *)

module Group : sig
  val mem : t -> t list -> bool
  (** [mem op group] is [true] iff [op] occurs in [group]. *)

  val union : t list -> t list -> t list
  (** [union a b] appends members of [b] not already in [a]. *)

  val without : t list -> t list -> t list
  (** [without group excluded] is [group] without members of [excluded]. *)

  val unary : t list
  (** Unary ALU ops. *)

  val binary : t list
  (** Binary ALU ops. *)

  val ternary : t list
  (** Ternary ALU ops. *)

  val alu : t list
  (** [unary @ binary @ ternary]. *)

  val broadcastable : t list
  (** Broadcastable ops: binary and ternary ops. *)

  val elementwise : t list
  (** Elementwise ops: {!alu}, {!Cast}, and {!Bitcast}. *)

  val defines : t list
  (** Defines: {!Buffer} and {!Param}. *)

  val irreducible : t list
  (** Irreducible leaves: {!Special}, {!Param}, {!Range}, {!Const}, and
      {!Getaddr}. *)

  val movement : t list
  (** Movement ops: {!Shrink}, {!Reshape}, {!Permute}, {!Expand}, {!Pad},
      and {!Flip}. *)

  val commutative : t list
  (** Binary ops whose operands may be swapped. *)

  val associative : t list
  (** Binary ops satisfying associativity. *)

  val idempotent : t list
  (** Binary ops satisfying [f x x = x]. *)

  val reduce : t list
  (** Ops valid as {!Reduce} and {!Allreduce} arguments: {!Add}, {!Mul},
      and {!Max}. *)

  val comparison : t list
  (** Binary ops producing booleans: {!Cmplt}, {!Cmpne}, and {!Cmpeq}. *)

  val all : t list
  (** Every constructor of {!t}, in tinygrad declaration order. *)

  val is_unary : t -> bool
  (** [is_unary op] is [true] iff [op] is in {!unary}. *)

  val is_binary : t -> bool
  (** [is_binary op] is [true] iff [op] is in {!binary}. *)

  val is_ternary : t -> bool
  (** [is_ternary op] is [true] iff [op] is in {!ternary}. *)

  val is_alu : t -> bool
  (** [is_alu op] is [true] iff [op] is in {!alu}. *)

  val is_broadcastable : t -> bool
  (** [is_broadcastable op] is [true] iff [op] is in {!broadcastable}. *)

  val is_elementwise : t -> bool
  (** [is_elementwise op] is [true] iff [op] is in {!elementwise}. *)

  val is_define : t -> bool
  (** [is_define op] is [true] iff [op] is in {!defines}. *)

  val is_irreducible : t -> bool
  (** [is_irreducible op] is [true] iff [op] is in {!irreducible}. *)

  val is_movement : t -> bool
  (** [is_movement op] is [true] iff [op] is in {!movement}. *)

  val is_commutative : t -> bool
  (** [is_commutative op] is [true] iff [op] is in {!commutative}. *)

  val is_associative : t -> bool
  (** [is_associative op] is [true] iff [op] is in {!associative}. *)

  val is_idempotent : t -> bool
  (** [is_idempotent op] is [true] iff [op] is in {!idempotent}. *)

  val is_reduce : t -> bool
  (** [is_reduce op] is [true] iff [op] is in {!reduce}. *)

  val is_comparison : t -> bool
  (** [is_comparison op] is [true] iff [op] is in {!comparison}. *)
end
