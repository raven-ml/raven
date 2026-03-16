(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

type view = Bool of bool | Int of int64 | Float of float
type t = { dtype : Dtype.t; view : view }

let view t = t.view
let dtype t = t.dtype

let check_scalar kind (dtype : Dtype.t) =
  if dtype.count <> 1 then
    invalid_arg
      (Printf.sprintf "Const.%s expects a scalar dtype, got %s" kind
         (Dtype.to_string dtype))

let bool value = { dtype = Dtype.bool; view = Bool value }

let int64 dtype value =
  check_scalar "int64" dtype;
  if not (Dtype.is_int dtype || Dtype.is_bool dtype) then
    invalid_arg
      (Printf.sprintf "Const.int64 expects an integer dtype, got %s"
         (Dtype.to_string dtype));
  { dtype; view = Int value }

let int dtype value = int64 dtype (Int64.of_int value)

let float dtype value =
  check_scalar "float" dtype;
  if not (Dtype.is_float dtype) then
    invalid_arg
      (Printf.sprintf "Const.float expects a floating-point dtype, got %s"
         (Dtype.to_string dtype));
  { dtype; view = Float value }

let equal a b = Dtype.equal a.dtype b.dtype && a.view = b.view

let compare a b =
  match Dtype.compare a.dtype b.dtype with
  | 0 -> Stdlib.compare a.view b.view
  | c -> c

let to_string t =
  match t.view with
  | Bool value -> Printf.sprintf "%b:%s" value (Dtype.to_string t.dtype)
  | Int value -> Printf.sprintf "%Ld:%s" value (Dtype.to_string t.dtype)
  | Float value -> Printf.sprintf "%g:%s" value (Dtype.to_string t.dtype)

let pp fmt t = Format.pp_print_string fmt (to_string t)
