(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Shared test utilities for Rune test suite *)

open Windtrap

(* Check Rune tensors for approximate equality *)
let check_rune ?eps msg expected actual =
  let testable =
    match eps with
    | None ->
        Testable.make ~pp:Rune.pp
          ~equal:(fun a b ->
            if Rune.shape a <> Rune.shape b then false
            else
              let eq_tensor = Rune.array_equal a b in
              (* array_equal returns a scalar boolean tensor *)
              let result = Rune.item [] eq_tensor in
              result)
          ()
    | Some eps ->
        Testable.make ~pp:Rune.pp
          ~equal:(fun a b ->
            let diff = Rune.sub a b in
            let abs_diff = Rune.abs diff in
            let max_diff = Rune.max abs_diff in
            let max_diff_val = Rune.item [] max_diff in
            Float.compare max_diff_val eps < 0)
          ()
  in
  equal ~msg testable expected actual

(* Check scalar values *)
let check_scalar ?eps msg expected actual =
  let eps = Option.value ~default:1e-6 eps in
  equal ~msg (float eps) expected actual

(* Extract scalar from Rune tensor *)
let scalar_value t = Rune.item [] t

(* Check shape of Rune tensor *)
let check_shape msg expected_shape tensor =
  equal ~msg (array int) expected_shape (Rune.shape tensor)

(* Common failure checks *)
let check_invalid_arg msg pattern f = raises ~msg (Invalid_argument pattern) f
let check_failure msg pattern f = raises ~msg (Failure pattern) f
