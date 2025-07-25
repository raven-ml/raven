(* Shared test utilities for Rune test suite *)

open Alcotest

(* Check Rune tensors for approximate equality *)
let check_rune ?eps msg expected actual =
  let testable =
    match eps with
    | None ->
        let equal_int a b =
          match (a, b) with 1, 1 -> true | 0, 0 -> true | _ -> false
        in
        Alcotest.testable Rune.pp (fun a b ->
            if Rune.shape a <> Rune.shape b then false
            else
              let eq_tensor = Rune.array_equal a b in
              (* array_equal returns a scalar uint8 tensor with 1 for true, 0
                 for false *)
              let result = Rune.unsafe_get [] eq_tensor in
              equal_int result 1)
    | Some eps ->
        Alcotest.testable Rune.pp (fun a b ->
            let diff = Rune.sub a b in
            let abs_diff = Rune.abs diff in
            let max_diff = Rune.max abs_diff in
            let max_diff_val = Rune.unsafe_get [] max_diff in
            Float.compare max_diff_val eps < 0)
  in
  check testable msg expected actual

(* Check scalar values *)
let check_scalar ?eps msg expected actual =
  let eps = Option.value ~default:1e-6 eps in
  check (float eps) msg expected actual

(* Extract scalar from Rune tensor *)
let scalar_value t = Rune.unsafe_get [] t

(* Check shape of Rune tensor *)
let check_shape msg expected_shape tensor =
  check (array int) msg expected_shape (Rune.shape tensor)

(* Common failure checks *)
let check_invalid_arg msg pattern f =
  check_raises msg (Invalid_argument pattern) f

let check_failure msg pattern f = check_raises msg (Failure pattern) f
