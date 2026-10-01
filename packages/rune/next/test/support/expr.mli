(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Generated programs over one float64 tensor of shape [[2; 3]].

    A program is a small expression whose operations are total on every input (a
    logarithm of [1 + x²], a division by [1 + b²], an exponential of a bounded
    value), so a law over programs needs no discarded case. It prints as a
    formula, so a counterexample reads as one. *)

type unary =
  | Sin
  | Tanh
  | Exp_tanh  (** [exp (tanh x)]. *)
  | Neg
  | Log1p_sq  (** [log (1 + x²)]. *)

type binary = Add | Sub | Mul | Div_safe  (** [a / (1 + b²)]. *) | Maximum

type t =
  | X  (** The argument. *)
  | Const of float array  (** A captured tensor of the argument's shape. *)
  | Un of unary * t
  | Bin of binary * t * t
  | Row_sum of t  (** Each row's sum, broadcast back along the row. *)
  | Transposed of t  (** Transposed and transposed back. *)
  | Matmul_const of t  (** Times a captured [[3; 3]] matrix. *)

val shape : int array
(** [shape] is [[|2; 3|]], the argument's and every program's result's. *)

val eval : t -> Nx.float64_t -> Nx.float64_t
(** [eval p x] is [p] at [x], computed with nx's operations, so every
    transformation around the call sees them. *)

val objective : t -> Nx.float64_t -> Nx.float64_t
(** [objective p x] is the sum of [eval p x], a scalar. *)

val pp : Format.formatter -> t -> unit
(** [pp] formats a program as a formula in [x]. *)

val gen : t Windtrap.Gen.t
(** [gen] draws programs of depth up to 4 that read the argument. *)

val smooth : t Windtrap.Gen.t
(** [smooth] draws programs as {!gen} does, without [Maximum]: they are
    differentiable everywhere, as a finite difference needs. *)

val point : Nx.float64_t Windtrap.Gen.t
(** [point] draws arguments, and directions, of {!shape} with elements in
    [[-2, 2]]. *)

val points : int -> Nx.float64_t Windtrap.Gen.t
(** [points n] draws a batch of [n] arguments, of shape [n :: shape]. *)
