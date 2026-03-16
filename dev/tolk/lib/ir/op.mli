(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Arithmetic and logical operations.

    Operations are represented as polymorphic variants grouped by arity. Each
    group has {!equal_reduce}, {!compare_reduce}, {!pp_reduce} (and likewise for
    {!unary}, {!binary}, {!ternary}). *)

(** {1:types Types} *)

type reduce = [ `Add | `Mul | `Max ]
(** The type for reduction operations. *)

type unary = [ `Neg | `Exp2 | `Log2 | `Sin | `Sqrt | `Recip | `Trunc ]
(** The type for unary operations. *)

type binary =
  [ `Add
  | `Sub
  | `Mul
  | `Fdiv  (** Floating-point division. *)
  | `Idiv  (** Integer division. *)
  | `Mod
  | `Max
  | `Pow
  | `Shl  (** Left shift. *)
  | `Shr  (** Right shift. *)
  | `And  (** Bitwise and. *)
  | `Or  (** Bitwise or. *)
  | `Xor  (** Bitwise xor. *)
  | `Threefry  (** Threefry PRNG mixing. *)
  | `Cmplt  (** Less-than comparison (result is bool). *)
  | `Cmpeq  (** Equality comparison (result is bool). *)
  | `Cmpne  (** Not-equal comparison (result is bool). *) ]
(** The type for binary operations.

    Comparison operators ([`Cmplt], [`Cmpeq], [`Cmpne]) produce a boolean dtype
    regardless of their operand dtype. All other operators preserve the operand
    dtype. *)

type ternary = [ `Where | `Mulacc ]
(** The type for ternary operations.

    [`Where] selects between two values based on a boolean condition. [`Mulacc]
    is fused multiply-accumulate. *)

(** {1:reduce Reduce operations} *)

val equal_reduce : reduce -> reduce -> bool
(** [equal_reduce a b] is [true] iff [a] and [b] are the same. *)

val compare_reduce : reduce -> reduce -> int
(** [compare_reduce a b] totally orders reduce operations. *)

val pp_reduce : Format.formatter -> reduce -> unit
(** [pp_reduce] formats a reduce operation as a lowercase string. *)

(** {1:unary Unary operations} *)

val equal_unary : unary -> unary -> bool
(** [equal_unary a b] is [true] iff [a] and [b] are the same. *)

val compare_unary : unary -> unary -> int
(** [compare_unary a b] totally orders unary operations. *)

val pp_unary : Format.formatter -> unary -> unit
(** [pp_unary] formats a unary operation as a lowercase string. *)

(** {1:binary Binary operations} *)

val equal_binary : binary -> binary -> bool
(** [equal_binary a b] is [true] iff [a] and [b] are the same. *)

val compare_binary : binary -> binary -> int
(** [compare_binary a b] totally orders binary operations. *)

val pp_binary : Format.formatter -> binary -> unit
(** [pp_binary] formats a binary operation as a lowercase string. *)

(** {1:ternary Ternary operations} *)

val equal_ternary : ternary -> ternary -> bool
(** [equal_ternary a b] is [true] iff [a] and [b] are the same. *)

val compare_ternary : ternary -> ternary -> int
(** [compare_ternary a b] totally orders ternary operations. *)

val pp_ternary : Format.formatter -> ternary -> unit
(** [pp_ternary] formats a ternary operation as a lowercase string. *)
