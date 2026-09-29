(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Operations.

    An operation names what a node of the intermediate representation does. One
    vocabulary serves every stage: tensor graphs, kernels, programs and command
    sequences.

    Operations are totally ordered by their declaration order in {!t}. The order
    is part of the contract: graph orderings and canonical operand orders are
    derived from it.

    The arithmetic operations name their sources [x], [y] and [z], in order. *)

(** {1:ops Operations} *)

type t =
  (* Definitions *)
  | Special
      (** A hardware index, such as a workgroup or thread id. Its source bounds
          it and its argument names it. *)
  | Buffer
      (** Storage bound to a runtime buffer, or storage private to a program:
          registers and workgroup memory. *)
  | Alloc
      (** Storage declared without a runtime buffer. The call that owns it binds
          one. *)
  (* Structure *)
  | Noop  (** Computes nothing. *)
  | Param  (** A parameter of a function: a buffer or a scalar variable. *)
  | Call
      (** An invocation of the body in its first source on the remaining
          sources. A call that produces a value writes it into {!Alloc}
          arguments. *)
  | Program
      (** A program in compilation. Its sources accumulate, in order: the
          kernel, its {!Linear} order, its {!Source} and its {!Binary}. *)
  | Linear  (** Operations in execution order. *)
  | Source  (** The rendered source text of a program. *)
  | Binary
      (** A constant string of bytes: the machine code of a program, or bytes
          placed in a command packet or a buffer. *)
  | Sink  (** A root gathering what a graph must compute. *)
  | After
      (** Its first source, ordered after the remaining sources: every consumer
          runs after them. *)
  | Group  (** Merges effects without producing a value. *)
  | Stack  (** A vector of its sources. *)
  | Getaddr
      (** The device address of a buffer or of a command sequence, for command
          queues. *)
  (* Memory *)
  | Index  (** A pointer into its first source, offset by the others. *)
  | Shrink
      (** Restricts each axis of its first source to the range between its
          second and third sources. In a program, a window of a buffer. *)
  | Load  (** Reads through a pointer. *)
  | Store  (** Writes its second source through its first. *)
  (* Arithmetic *)
  | Wmma  (** A matrix multiply-accumulate on matrix units. *)
  | Cast
      (** Converts the value of its source to another type. {!Bitcast} keeps the
          bits instead. *)
  | Bitcast  (** Reinterprets the bits as another type of the same width. *)
  | Exp2  (** [2]{^ [x]}. *)
  | Log2  (** The base-2 logarithm. *)
  | Sin  (** The sine. *)
  | Sqrt  (** The square root. *)
  | Reciprocal  (** [1 / x]. *)
  | Neg  (** [-x]. *)
  | Trunc  (** Rounds toward zero. *)
  | Add  (** [x + y]. *)
  | Mul  (** [x * y]. *)
  | Shl  (** Shifts [x] left by [y] bits. *)
  | Shr  (** Shifts [x] right by [y] bits. *)
  | Cdiv  (** Integer division rounding toward zero. *)
  | Max  (** The greater of [x] and [y]. *)
  | Cmod  (** The remainder of {!Cdiv}: it has the sign of [x]. *)
  | Cmplt  (** [x < y]. *)
  | Cmpne  (** [x <> y]. *)
  | Cmpeq  (** [x = y]. *)
  | Xor  (** Bitwise exclusive or. *)
  | Or  (** Bitwise or. *)
  | And  (** Bitwise and. *)
  | Threefry
      (** The Threefry-2x32 bijection of the counter [x] under the key [y]. *)
  | Sub  (** [x - y]. *)
  | Fdiv  (** Floating-point division. *)
  | Pow  (** [x]{^ [y]}. *)
  | Floordiv  (** Integer division rounding toward negative infinity. *)
  | Floormod  (** The remainder of {!Floordiv}: it has the sign of [y]. *)
  | Where  (** [y] where [x] holds, [z] otherwise. *)
  | Mulacc  (** [x * y + z]. *)
  (* Control flow, constants and target code *)
  | Barrier  (** Synchronises the threads of a workgroup. *)
  | Range
      (** A loop variable counting from 0 to its source, exclusive, or an
          unbounded loop when its type is void. *)
  | If  (** Runs the operations it guards only when its condition holds. *)
  | End  (** Closes the loops of its range sources around an effect. *)
  | Endif  (** Closes an {!If}. *)
  | Backedge
      (** Runs its body, then repeats its unbounded {!Range} while its condition
          holds. *)
  | Const  (** A constant. *)
  | Custom  (** A statement written in the target's source language. *)
  | Customi
      (** An expression written in the target's source language, rendered
          inline. *)
  | Ins
      (** An instruction named by its argument: a machine instruction, or a
          command of a device queue such as a wait, a barrier or a timestamp. *)
  (* Tensor graph *)
  | Contiguous_backward  (** Its source, whose gradient is made contiguous. *)
  | Detach  (** Its source, through which no gradient flows. *)
  | Stage  (** Realizes its source into a new buffer. *)
  | Copy  (** Copies its source to another device. *)
  | Mselect  (** One device's shard of a value held on several devices. *)
  | Mstack  (** Gathers per-device values into one multi-device value. *)
  | Custom_function
      (** A function named by its argument and implemented outside the graph,
          such as a device submission or an external callee. *)
  | Reshape  (** Gives its source a new shape with the same element count. *)
  | Permute  (** Permutes the axes of its source. *)
  | Expand  (** Broadcasts axes of size one of its source. *)
  | Pad  (** Pads each axis of its source. *)
  | Flip  (** Reverses axes of its source. *)
  | Unshard
      (** Reassembles a value sharded across devices: its source is one shard
          and its result is the whole. *)
  | Reduce  (** Reduces axes of its source with {!Add}, {!Mul} or {!Max}. *)
  | Allreduce  (** Reduces a value across devices. *)

val equal : t -> t -> bool
(** [equal o o'] is [true] iff [o] and [o'] are the same operation. *)

val compare : t -> t -> int
(** [compare o o'] orders [o] and [o'] by their declaration order in {!t}. *)

val to_int : t -> int
(** [to_int o] is the position of [o] in the declaration order of {!t}, counting
    from [0]. *)

val name : t -> string
(** [name o] is the name of [o] in upper case, words separated by underscores:
    ["ADD"], ["CONTIGUOUS_BACKWARD"]. *)

val pp : Format.formatter -> t -> unit
(** [pp] formats an operation as ["Ops."] followed by its {!name}: [Ops.ADD]. *)

(** {1:sets Sets} *)

(** Sets of operations.

    The named sets are the families that rewrites and checks match on, such as
    {!Set.alu} or {!Set.movement}; the set operations combine them into new
    sets. *)
module Set : sig
  type op := t

  type t
  (** The type for sets of operations. *)

  val of_list : op list -> t
  (** [of_list ops] is the set of the operations in [ops]. *)

  val mem : op -> t -> bool
  (** [mem o s] is [true] iff [o] is in [s]. *)

  val union : t -> t -> t
  (** [union s s'] is the set of the operations in [s] or in [s']. *)

  val diff : t -> t -> t
  (** [diff s s'] is the set of the operations in [s] and not in [s']. *)

  val to_list : t -> op list
  (** [to_list s] is the operations of [s] in declaration order. *)

  val equal : t -> t -> bool
  (** [equal s s'] is [true] iff [s] and [s'] have the same operations. *)

  val pp : Format.formatter -> t -> unit
  (** [pp] formats a set as its operations in declaration order, between braces
      and separated by commas: [{Ops.ADD, Ops.MUL}]. The empty set is [{}]. *)

  (** {1:named Named sets} *)

  val unary : t
  (** [unary] is the operations of one operand: {!Exp2}, {!Log2}, {!Sin},
      {!Sqrt}, {!Reciprocal}, {!Neg} and {!Trunc}. *)

  val binary : t
  (** [binary] is the operations of two operands: {!Add}, {!Mul}, {!Cdiv},
      {!Max}, {!Cmod}, {!Cmplt}, {!Cmpne}, {!Cmpeq}, {!Xor}, {!Shl}, {!Shr},
      {!Or}, {!And}, {!Threefry}, {!Sub}, {!Fdiv}, {!Pow}, {!Floordiv} and
      {!Floormod}. *)

  val ternary : t
  (** [ternary] is the operations of three operands: {!Where} and {!Mulacc}. *)

  val alu : t
  (** [alu] is the arithmetic and logic operations: {!unary}, {!binary} and
      {!ternary}. *)

  val broadcastable : t
  (** [broadcastable] is {!binary} and {!ternary}: the operations whose operands
      are brought to a common shape and type. *)

  val elementwise : t
  (** [elementwise] is {!alu}, {!Cast} and {!Bitcast}: the operations applied
      element by element. *)

  val defines : t
  (** [defines] is the operations that declare storage or variables: {!Param},
      {!Buffer} and {!Alloc}. *)

  val irreducible : t
  (** [irreducible] is the operations that index arithmetic treats as atoms:
      {!Const}, {!Special}, {!Range}, {!Param} and {!Getaddr}. *)

  val movement : t
  (** [movement] is the operations that move elements without computing:
      {!Reshape}, {!Expand}, {!Permute}, {!Pad}, {!Shrink} and {!Flip}. *)

  val commutative : t
  (** [commutative] is the binary operations [f] with [f x y = f y x]: {!Add},
      {!Mul}, {!Max}, {!Cmpne}, {!Cmpeq}, {!Xor}, {!And} and {!Or}. *)

  val associative : t
  (** [associative] is the binary operations that rewrites may regroup, each
      satisfying [f (f x y) z = f x (f y z)]: {!Add}, {!Mul}, {!And}, {!Or} and
      {!Max}. *)

  val idempotent : t
  (** [idempotent] is the binary operations [f] with [f x x = x]: {!Or}, {!And}
      and {!Max}. *)

  val reduce : t
  (** [reduce] is the operations a {!Reduce} or an {!Allreduce} can reduce with:
      {!Add}, {!Mul} and {!Max}. *)

  val comparison : t
  (** [comparison] is the operations whose result is a boolean whatever the type
      of their operands: {!Cmplt}, {!Cmpne} and {!Cmpeq}. *)

  val all : t
  (** [all] is every operation. *)
end
