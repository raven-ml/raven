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

let pp_reduce fmt = function
  | `Add -> Format.pp_print_string fmt "add"
  | `Mul -> Format.pp_print_string fmt "mul"
  | `Max -> Format.pp_print_string fmt "max"

let pp_unary fmt = function
  | `Neg -> Format.pp_print_string fmt "neg"
  | `Exp2 -> Format.pp_print_string fmt "exp2"
  | `Log2 -> Format.pp_print_string fmt "log2"
  | `Sin -> Format.pp_print_string fmt "sin"
  | `Sqrt -> Format.pp_print_string fmt "sqrt"
  | `Recip -> Format.pp_print_string fmt "recip"
  | `Trunc -> Format.pp_print_string fmt "trunc"

let pp_binary fmt = function
  | `Add -> Format.pp_print_string fmt "add"
  | `Sub -> Format.pp_print_string fmt "sub"
  | `Mul -> Format.pp_print_string fmt "mul"
  | `Fdiv -> Format.pp_print_string fmt "fdiv"
  | `Idiv -> Format.pp_print_string fmt "idiv"
  | `Mod -> Format.pp_print_string fmt "mod"
  | `Max -> Format.pp_print_string fmt "max"
  | `Pow -> Format.pp_print_string fmt "pow"
  | `Shl -> Format.pp_print_string fmt "shl"
  | `Shr -> Format.pp_print_string fmt "shr"
  | `And -> Format.pp_print_string fmt "and"
  | `Or -> Format.pp_print_string fmt "or"
  | `Xor -> Format.pp_print_string fmt "xor"
  | `Threefry -> Format.pp_print_string fmt "threefry"
  | `Cmplt -> Format.pp_print_string fmt "cmplt"
  | `Cmpeq -> Format.pp_print_string fmt "cmpeq"
  | `Cmpne -> Format.pp_print_string fmt "cmpne"

let pp_ternary fmt = function
  | `Where -> Format.pp_print_string fmt "where"
  | `Mulacc -> Format.pp_print_string fmt "mulacc"
