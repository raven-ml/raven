(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

let err_not_int dt =
  strf "Const.int64 expects an integer dtype, got %s" (Dtype.to_string dt)

let err_not_float dt =
  strf "Const.float expects a floating-point dtype, got %s" (Dtype.to_string dt)

type view = Bool of bool | Int of int64 | Float of float | Invalid
type t = { dtype : Dtype.t; view : view }

let view t = t.view
let dtype t = t.dtype
let bool value = { dtype = Dtype.bool; view = Bool value }

let invalid ?(dtype = Dtype.weakint) () = { dtype; view = Invalid }

let int64 (dtype : Dtype.t) value =
  if not (Dtype.is_int dtype) then invalid_arg (err_not_int dtype);
  { dtype; view = Int value }

let int dtype value = int64 dtype (Int64.of_int value)

let float (dtype : Dtype.t) value =
  if not (Dtype.is_float dtype) then invalid_arg (err_not_float dtype);
  let value = if Float.is_nan value then Float.nan else value in
  { dtype; view = Float value }

let storage_bool = function
  | `Bool b -> b
  | `Int n -> n <> 0L
  | `Float f -> f <> 0.0

let storage_float = function
  | `Bool b -> if b then 1.0 else 0.0
  | `Int n -> Int64.to_float n
  | `Float f -> f

let storage_int64 = function
  | `Bool b -> if b then 1L else 0L
  | `Int n -> n
  | `Float f -> Int64.of_float f

let of_scalar dtype value =
  if Dtype.is_float dtype then
    let value = storage_float value in
    let value = if Float.is_nan value then Float.nan else value in
    { dtype; view = Float value }
  else if Dtype.is_bool dtype then
    { dtype; view = Bool (storage_bool value) }
  else { dtype; view = Int (storage_int64 value) }

let of_view dtype = function
  | Invalid -> invalid ~dtype ()
  | Bool b -> of_scalar dtype (`Bool b)
  | Int n -> of_scalar dtype (`Int n)
  | Float f -> of_scalar dtype (`Float f)

let equal_view a b =
  match a, b with
  | Bool x, Bool y -> Bool.equal x y
  | Int x, Int y -> Int64.equal x y
  | Float x, Float y -> Int64.equal (Int64.bits_of_float x) (Int64.bits_of_float y)
  | Invalid, Invalid -> true
  | _ -> false

let equal a b = Dtype.equal a.dtype b.dtype && equal_view a.view b.view

let compare_view a b =
  match a, b with
  | Bool x, Bool y -> Bool.compare x y
  | Int x, Int y -> Int64.compare x y
  | Float x, Float y -> Int64.compare (Int64.bits_of_float x) (Int64.bits_of_float y)
  | Invalid, Invalid -> 0
  | Bool _, _ -> -1 | _, Bool _ -> 1
  | Int _, _ -> -1 | _, Int _ -> 1
  | Float _, _ -> -1 | _, Float _ -> 1

let compare a b =
  let c = Dtype.compare a.dtype b.dtype in
  if c <> 0 then c else compare_view a.view b.view

let to_string t =
  let s = Dtype.to_string t.dtype in
  match t.view with
  | Bool v -> strf "%b:%s" v s
  | Int v -> strf "%Ld:%s" v s
  | Float v -> strf "%g:%s" v s
  | Invalid -> strf "Invalid:%s" s

let pp fmt t = Format.pp_print_string fmt (to_string t)

let zero dtype =
  if Dtype.is_float dtype then float dtype 0.0
  else if Dtype.is_bool dtype then bool false
  else int dtype 0

let one dtype =
  if Dtype.is_float dtype then float dtype 1.0
  else if Dtype.is_bool dtype then bool true
  else int dtype 1

let min_value dtype =
  match Dtype.min dtype with
  | `Float f -> float dtype f
  | `SInt i | `UInt i -> int64 dtype i
  | `Bool _ -> bool false

let max_value dtype =
  match Dtype.max dtype with
  | `Float f -> float dtype f
  | `SInt i | `UInt i -> int64 dtype i
  | `Bool _ -> bool true
