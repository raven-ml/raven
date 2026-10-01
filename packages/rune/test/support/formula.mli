(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Generated differentiable programs over one float64 tensor of any shape.

    A program grows from its argument by operations of every differentiable
    family: elementwise, reductions, movements, matrix products, selections and
    gathers. Each step keeps the result's shape known, so shapes with a zero or
    a one-element axis, and scalar arguments, are drawn among the others. Every
    operation is total on every input: a logarithm of [1 + x²], a division by
    [1 + b²], an exponential of a bounded value, a gather whose out-of-range
    indices read zero. *)

type unary =
  | Sin
  | Tanh
  | Exp_tanh  (** [exp (tanh x)]. *)
  | Neg
  | Log1p_sq  (** [log (1 + x²)]. *)
  | Abs  (** A kink at [0]. *)
  | Relu  (** A kink at [0]. *)

type binary =
  | Add
  | Sub
  | Mul
  | Div_safe  (** [a / (1 + b²)]. *)
  | Maximum  (** A kink where the operands tie. *)

type term =
  | X  (** The argument. *)
  | Const of int array * float array  (** A captured tensor, its shape. *)
  | Un of unary * term
  | Bin of binary * term * term  (** The operands broadcast. *)
  | Where of int array * bool array * term * term
      (** A selection by a captured mask, its shape. *)
  | Sum of int * bool * term  (** Along one axis, keeping it or not. *)
  | Max of int * term  (** Along one non-empty axis; a kink at ties. *)
  | Permute of int list * term
  | Reshape of int array * term  (** Of a contiguous copy. *)
  | Flip of int * term
  | Pad of (int * int) array * term  (** With zeros. *)
  | Shrink of (int * int) array * term
  | Expand of int * term  (** Broadcast along a new leading axis. *)
  | Concat of int * term * term
  | Matmul of term * term
  | Take of int * int array * term
      (** Along an axis, some indices out of range. *)

type t = { term : term; input : int array; output : int array }
(** A program, the shape of its argument and of its result. *)

val eval : t -> Nx.float64_t -> Nx.float64_t
(** [eval p x] is [p] at [x], computed with nx's operations, so every
    transformation around the call sees them. *)

val objective : t -> Nx.float64_t -> Nx.float64_t
(** [objective p x] is the sum of [eval p x], a scalar. *)

val families : t -> string list
(** [families p] names the operation families [p] uses: ["elementwise"],
    ["reduction"], ["movement"], ["matmul"], ["where"] and ["take"]. *)

val all_families : string list
(** [all_families] is every name {!families} gives. *)

val pp : Format.formatter -> t -> unit
(** [pp] formats a program as a formula in [x], with its argument's shape. *)

val gen : t Windtrap.Gen.t
(** [gen] draws programs of up to five operations, kinks included. *)

val smooth : t Windtrap.Gen.t
(** [smooth] draws programs as {!gen} does, without [Abs], [Relu], [Maximum] or
    [Max]: they are differentiable everywhere, as a finite difference needs. *)

val point : int array -> Nx.float64_t Windtrap.Gen.t
(** [point shape] draws arguments, and directions, of [shape] with elements in
    [[-2, 2]]. *)

val points : int -> int array -> Nx.float64_t Windtrap.Gen.t
(** [points n shape] draws a batch of [n] arguments, of shape [n :: shape]. *)
