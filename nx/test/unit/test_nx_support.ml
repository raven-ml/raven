(* Shared test utilities for Nx test suite *)

open Alcotest

module Make (Backend : Nx_core.Backend_intf.S) = struct
  module Nx = Nx_core.Make_frontend (Backend)

  let check_invalid_arg msg pattern f =
    check_raises msg (Invalid_argument pattern) (fun () -> ignore (f ()))

  let check_failure msg pattern f = check_raises msg (Failure pattern) f

  let testable_of_dtype (type a b) ?(eps = 1e-6)
      (dtype : (a, b) Nx_core.Dtype.t) : a testable =
    match dtype with
    | Nx_core.Dtype.Float16 -> Alcotest.float eps
    | Nx_core.Dtype.Float32 -> Alcotest.float eps
    | Nx_core.Dtype.Float64 -> Alcotest.float eps
    | Nx_core.Dtype.Int8 -> Alcotest.int
    | Nx_core.Dtype.Int16 -> Alcotest.int
    | Nx_core.Dtype.Int32 -> Alcotest.int32
    | Nx_core.Dtype.Int64 -> Alcotest.int64
    | Nx_core.Dtype.UInt8 -> Alcotest.int
    | Nx_core.Dtype.UInt16 -> Alcotest.int
    | Nx_core.Dtype.Int -> Alcotest.int
    | Nx_core.Dtype.NativeInt ->
        Alcotest.testable
          (fun ppf v -> Format.fprintf ppf "%nd" v)
          Nativeint.equal
    | Nx_core.Dtype.Complex32 ->
        Alcotest.testable
          (fun ppf v -> Format.fprintf ppf "(%f, %f)" v.Complex.re v.Complex.im)
          (fun a b -> a.re = b.re && a.im = b.im)
    | Nx_core.Dtype.Complex64 ->
        Alcotest.testable
          (fun ppf v -> Format.fprintf ppf "(%f, %f)" v.Complex.re v.Complex.im)
          (fun a b -> a.re = b.re && a.im = b.im)

  (* Check function to test a tensor against an array *)
  let check_data (type a b) ?eps msg (expected : a array) (actual : (a, b) Nx.t)
      =
    let dt_testable = testable_of_dtype ?eps (Nx.dtype actual) in
    let actual = Nx.unsafe_to_array actual in
    check (array dt_testable) msg expected actual

  let check_shape msg expected_shape tensor =
    check (array int) msg expected_shape (Nx.shape tensor)

  let check_t ?eps msg shape data actual =
    check_shape msg shape actual;
    check_data ?eps msg data actual

  (* Approximate equality for floating-point comparisons *)
  let approx_equal (type b) ?(epsilon = 1e-6) (a : (float, b) Nx.t)
      (b : (float, b) Nx.t) =
    if Nx.shape a <> Nx.shape b then false
    else
      let diff = Nx.sub a b in
      let abs_diff = Nx.abs diff in
      let max_diff = Nx.unsafe_get [] (Nx.max abs_diff) in
      max_diff < epsilon

  (* Common check functions *)
  let check_nx (type a b) ?epsilon msg (expected : (a, b) Nx.t)
      (actual : (a, b) Nx.t) =
    if Nx.shape expected <> Nx.shape actual then
      Alcotest.failf "%s: shapes differ - expected %s, got %s" msg
        (String.concat "x"
           (List.map string_of_int (Array.to_list (Nx.shape expected))))
        (String.concat "x"
           (List.map string_of_int (Array.to_list (Nx.shape actual))))
    else
      let test_float expected actual =
        let approx_equal = approx_equal ?epsilon in
        if not (approx_equal expected actual) then
          Alcotest.failf "%s: tensors not equal\nExpected:\n%s\nActual:\n%s" msg
            (Nx.to_string expected) (Nx.to_string actual)
      in
      match Nx.dtype expected with
      | Float16 -> test_float expected actual
      | Float32 -> test_float expected actual
      | Float64 -> test_float expected actual
      | _ ->
          let equal = Nx.array_equal expected actual in
          if not (equal |> Nx.unsafe_get [] = 0) then
            Alcotest.failf "%s: tensors not equal\nExpected:\n%s\nActual:\n%s"
              msg (Nx.to_string expected) (Nx.to_string actual)

  let check_nx_scalar ctx dtype msg expected actual =
    let expected_t = Nx.scalar ctx dtype expected in
    let actual_t = Nx.scalar ctx dtype actual in
    check_nx msg expected_t actual_t
end
