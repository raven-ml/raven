(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type unary =
  | Neg
  | Recip
  | Abs
  | Sqrt
  | Sign
  | Exp
  | Log
  | Sin
  | Cos
  | Tan
  | Asin
  | Acos
  | Atan
  | Sinh
  | Cosh
  | Tanh
  | Trunc
  | Ceil
  | Floor
  | Round
  | Erf

type binary =
  | Add
  | Sub
  | Mul
  | Fdiv
  | Idiv
  | Mod
  | Pow
  | Atan2
  | Maximum
  | Minimum
  | And
  | Or
  | Xor

type compare = Equal | Not_equal | Less | Less_equal
type reduce = Sum | Prod | Max | Min
type arg_reduce = Argmax | Argmin
type conversion = Cast | Bitcast
