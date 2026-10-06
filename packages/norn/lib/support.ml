(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t =
  | Real
  | Greater of float
  | Interval of float * float
  | Simplex of int
  | Ordered
  | Correlation_cholesky of int
  | Sum_to_zero
  | Integers_from of int
  | Integer_interval of int * int
  | Boolean

let equal s s' =
  match (s, s') with
  | Real, Real | Ordered, Ordered | Sum_to_zero, Sum_to_zero -> true
  | Boolean, Boolean -> true
  | Greater a, Greater a' -> Float.equal a a'
  | Interval (a, b), Interval (a', b') -> Float.equal a a' && Float.equal b b'
  | Simplex n, Simplex n' | Correlation_cholesky n, Correlation_cholesky n' ->
      Int.equal n n'
  | Integers_from n, Integers_from n' -> Int.equal n n'
  | Integer_interval (a, b), Integer_interval (a', b') ->
      Int.equal a a' && Int.equal b b'
  | ( ( Real | Greater _ | Interval _ | Simplex _ | Ordered
      | Correlation_cholesky _ | Sum_to_zero | Integers_from _
      | Integer_interval _ | Boolean ),
      _ ) ->
      false

let pp_bound ppf x =
  if Float.is_integer x && Float.abs x < 1e15 then Format.fprintf ppf "%.0f" x
  else Format.fprintf ppf "%g" x

let pp ppf = function
  | Real -> Format.pp_print_string ppf "(-inf, inf)"
  | Greater a -> Format.fprintf ppf "(%a, inf)" pp_bound a
  | Interval (a, b) -> Format.fprintf ppf "(%a, %a)" pp_bound a pp_bound b
  | Simplex n -> Format.fprintf ppf "simplex of %d" n
  | Ordered -> Format.pp_print_string ppf "ordered vectors"
  | Correlation_cholesky n ->
      Format.fprintf ppf "Cholesky factors of %d x %d correlations" n n
  | Sum_to_zero -> Format.pp_print_string ppf "vectors summing to zero"
  | Integers_from a -> Format.fprintf ppf "{%d, %d, %d, ...}" a (a + 1) (a + 2)
  | Integer_interval (a, b) when b - a <= 2 ->
      let xs = List.init (b - a + 1) (fun i -> string_of_int (a + i)) in
      Format.fprintf ppf "{%s}" (String.concat ", " xs)
  | Integer_interval (a, b) ->
      Format.fprintf ppf "{%d, %d, ..., %d}" a (a + 1) b
  | Boolean -> Format.pp_print_string ppf "{false, true}"
