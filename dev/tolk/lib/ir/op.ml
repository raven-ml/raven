(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

type reduce = [ `Add | `Mul | `Max ]
type unary = [ `Neg | `Exp2 | `Log2 | `Sin | `Sqrt | `Recip | `Trunc ]

type binary =
  [ `Add
  | `Sub
  | `Mul
  | `Fdiv
  | `Idiv
  | `Mod
  | `Max
  | `Pow
  | `Shl
  | `Shr
  | `And
  | `Or
  | `Xor
  | `Threefry
  | `Cmplt
  | `Cmpeq
  | `Cmpne ]

type ternary = [ `Where | `Mulacc ]

let equal_reduce = ( = )
let compare_reduce = Stdlib.compare
let equal_unary = ( = )
let compare_unary = Stdlib.compare
let equal_binary = ( = )
let compare_binary = Stdlib.compare
let equal_ternary = ( = )
let compare_ternary = Stdlib.compare

let string_of_reduce = function
  | `Add -> "add" | `Mul -> "mul" | `Max -> "max"

let string_of_unary = function
  | `Neg -> "neg" | `Exp2 -> "exp2" | `Log2 -> "log2" | `Sin -> "sin"
  | `Sqrt -> "sqrt" | `Recip -> "recip" | `Trunc -> "trunc"

let string_of_binary = function
  | `Add -> "add" | `Sub -> "sub" | `Mul -> "mul" | `Fdiv -> "fdiv"
  | `Idiv -> "idiv" | `Mod -> "mod" | `Max -> "max" | `Pow -> "pow"
  | `Shl -> "shl" | `Shr -> "shr" | `And -> "and" | `Or -> "or"
  | `Xor -> "xor" | `Threefry -> "threefry" | `Cmplt -> "cmplt"
  | `Cmpeq -> "cmpeq" | `Cmpne -> "cmpne"

let string_of_ternary = function
  | `Where -> "where" | `Mulacc -> "mulacc"

let pp_reduce fmt op = Format.pp_print_string fmt (string_of_reduce op)
let pp_unary fmt op = Format.pp_print_string fmt (string_of_unary op)
let pp_binary fmt op = Format.pp_print_string fmt (string_of_binary op)
let pp_ternary fmt op = Format.pp_print_string fmt (string_of_ternary op)
