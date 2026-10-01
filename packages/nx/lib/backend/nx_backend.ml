(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type unary = Nx_backend_intf.unary =
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

type binary = Nx_backend_intf.binary =
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

type compare = Nx_backend_intf.compare = Equal | Not_equal | Less | Less_equal
type reduce = Nx_backend_intf.reduce = Sum | Prod | Max | Min
type arg_reduce = Nx_backend_intf.arg_reduce = Argmax | Argmin
type index_array = Nx_backend_intf.index_array

module type S = Nx_backend_intf.S

exception Refused of string

exception
  Linalg_error of {
    op : string;
    kind : [ `Not_positive_definite | `Singular | `No_convergence ];
  }

let () =
  Printexc.register_printer (function
    | Refused reason -> Some (Printf.sprintf "Nx_backend.Refused(%S)" reason)
    | Linalg_error { op; kind } ->
        let detail =
          match kind with
          | `Not_positive_definite -> "matrix is not positive-definite"
          | `Singular -> "matrix is singular"
          | `No_convergence -> "algorithm failed to converge"
        in
        Some (Printf.sprintf "Nx.Linalg_error(%s): %s" op detail)
    | _ -> None)

type t = { kernels : (module S) }

let make kernels = { kernels }
let kernels b = b.kernels
let name { kernels = (module B) } = B.name
let runs_on { kernels = (module B) } d = B.runs_on d
let equal = ( == )
