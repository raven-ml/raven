(* Shared test utilities for Nx test suite *)

open Alcotest

let check_invalid_arg msg pattern f =
  check_raises msg (Invalid_argument pattern) (fun () -> ignore (f ()))

let check_failure msg pattern f = check_raises msg (Failure pattern) f

let testable_of_dtype (type a b) ?(eps = 1e-6) (dtype : (a, b) Nx.dtype)
    : a testable =
  match dtype with
  | Nx.Float16 -> Alcotest.float eps
  | Nx.Float32 -> Alcotest.float eps
  | Nx.Float64 -> Alcotest.float eps
  | Nx.BFloat16 -> Alcotest.float eps
  | Nx.Float8_e4m3 -> Alcotest.float eps
  | Nx.Float8_e5m2 -> Alcotest.float eps
  | Nx.Int8 -> Alcotest.int
  | Nx.Int16 -> Alcotest.int
  | Nx.Int32 -> Alcotest.int32
  | Nx.Int64 -> Alcotest.int64
  | Nx.UInt8 -> Alcotest.int
  | Nx.UInt16 -> Alcotest.int
  | Nx.Int -> Alcotest.int
  | Nx.NativeInt ->
      Alcotest.testable
        (fun ppf v -> Format.fprintf ppf "%nd" v)
        Nativeint.equal
  | Nx.Int4 -> Alcotest.int
  | Nx.UInt4 -> Alcotest.int
  | Nx.QInt8 -> Alcotest.int
  | Nx.QUInt8 -> Alcotest.int
  | Nx.Bool -> Alcotest.bool
  | Nx.Complex32 ->
      Alcotest.testable
        (fun ppf v -> Format.fprintf ppf "(%f, %f)" v.Complex.re v.Complex.im)
        (fun a b ->
          Float.abs (a.re -. b.re) < eps && Float.abs (a.im -. b.im) < eps)
  | Nx.Complex64 ->
      Alcotest.testable
        (fun ppf v -> Format.fprintf ppf "(%f, %f)" v.Complex.re v.Complex.im)
        (fun a b ->
          Float.abs (a.re -. b.re) < eps && Float.abs (a.im -. b.im) < eps)
  | Nx.Complex16 ->
      Alcotest.testable
        (fun ppf v -> Format.fprintf ppf "(%f, %f)" v.Complex.re v.Complex.im)
        (fun a b ->
          Float.abs (a.re -. b.re) < eps && Float.abs (a.im -. b.im) < eps)

(* Check function to test a tensor against an array *)
let check_data (type a b) ?eps msg (expected : a array) (actual : (a, b) Nx.t) =
  let dt_testable = testable_of_dtype ?eps (Nx.dtype actual) in
  let actual = Nx.to_array actual in
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
    let max_diff = Nx.item [] (Nx.max abs_diff) in
    max_diff < epsilon

(* Approximate equality for complex numbers *)
let approx_equal_complex (type b) ?(epsilon = 1e-6) (a : (Complex.t, b) Nx.t)
    (b : (Complex.t, b) Nx.t) =
  if Nx.shape a <> Nx.shape b then false
  else
    let a_arr = Nx.to_array a in
    let b_arr = Nx.to_array b in
    Array.for_all2
      (fun x y ->
        Float.abs (x.Complex.re -. y.Complex.re) < epsilon
        && Float.abs (x.Complex.im -. y.Complex.im) < epsilon)
      a_arr b_arr

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
    let test_complex expected actual =
      let approx_equal_complex = approx_equal_complex ?epsilon in
      if not (approx_equal_complex expected actual) then
        Alcotest.failf "%s: tensors not equal\nExpected:\n%s\nActual:\n%s" msg
          (Nx.to_string expected) (Nx.to_string actual)
    in
    match Nx.dtype expected with
    | Float16 -> test_float expected actual
    | Float32 -> test_float expected actual
    | Float64 -> test_float expected actual
    | Complex16 -> test_complex expected actual
    | Complex32 -> test_complex expected actual
    | Complex64 -> test_complex expected actual
    | _ ->
        let equal = Nx.array_equal expected actual in
        if not (equal |> Nx.item [] = 0) then
          Alcotest.failf "%s: tensors not equal\nExpected:\n%s\nActual:\n%s" msg
            (Nx.to_string expected) (Nx.to_string actual)

let check_nx_scalar dtype msg expected actual =
  let expected_t = Nx.scalar dtype expected in
  let actual_t = Nx.scalar dtype actual in
  check_nx msg expected_t actual_t
