(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Import

let failed = ref 0
let passed = ref 0

let check name ~expected ~actual cond =
  if cond then incr passed
  else (
    incr failed;
    Printf.printf "FAIL: %s\n  expected: %s\n  actual:   %s\n%!" name expected actual)

let check_float name ~eps exp act =
  check name
    ~expected:(string_of_float exp)
    ~actual:(string_of_float act)
    (Float.abs (exp -. act) <= eps)

let check_float_array name ~eps exp act =
  check (name ^ ":len")
    ~expected:(string_of_int (Stdlib.Array.length exp))
    ~actual:(string_of_int (Stdlib.Array.length act))
    (Stdlib.Array.length exp = Stdlib.Array.length act);
  for i = 0 to Stdlib.Array.length exp - 1 do
    check_float (Printf.sprintf "%s[%d]" name i) ~eps exp.(i) act.(i)
  done

let check_int_array name exp act =
  check (name ^ ":len")
    ~expected:(string_of_int (Stdlib.Array.length exp))
    ~actual:(string_of_int (Stdlib.Array.length act))
    (Stdlib.Array.length exp = Stdlib.Array.length act);
  for i = 0 to Stdlib.Array.length exp - 1 do
    check (Printf.sprintf "%s[%d]" name i)
      ~expected:(string_of_int exp.(i))
      ~actual:(string_of_int act.(i))
      (exp.(i) = act.(i))
  done

let check_int32_array name exp act =
  check (name ^ ":len")
    ~expected:(string_of_int (Stdlib.Array.length exp))
    ~actual:(string_of_int (Stdlib.Array.length act))
    (Stdlib.Array.length exp = Stdlib.Array.length act);
  for i = 0 to Stdlib.Array.length exp - 1 do
    check (Printf.sprintf "%s[%d]" name i)
      ~expected:(Int32.to_string exp.(i))
      ~actual:(Int32.to_string act.(i))
      (Int32.equal exp.(i) act.(i))
  done

let check_int64_array name exp act =
  check (name ^ ":len")
    ~expected:(string_of_int (Stdlib.Array.length exp))
    ~actual:(string_of_int (Stdlib.Array.length act))
    (Stdlib.Array.length exp = Stdlib.Array.length act);
  for i = 0 to Stdlib.Array.length exp - 1 do
    check (Printf.sprintf "%s[%d]" name i)
      ~expected:(Int64.to_string exp.(i))
      ~actual:(Int64.to_string act.(i))
      (Int64.equal exp.(i) act.(i))
  done

let check_bool_array name exp act =
  check (name ^ ":len")
    ~expected:(string_of_int (Stdlib.Array.length exp))
    ~actual:(string_of_int (Stdlib.Array.length act))
    (Stdlib.Array.length exp = Stdlib.Array.length act);
  for i = 0 to Stdlib.Array.length exp - 1 do
    check (Printf.sprintf "%s[%d]" name i)
      ~expected:(string_of_bool exp.(i))
      ~actual:(string_of_bool act.(i))
      (exp.(i) = act.(i))
  done
  
let ub_float64_of_array a =
  Array.ba_to_unboxed_float_array
    (Array1.of_array Bigarray.float64 Bigarray.c_layout a)

let ub_float32_of_array a =
  Array.ba_to_unboxed_float32_array
    (Array1.of_array Bigarray.float32 Bigarray.c_layout a)

let ub_int8_of_array a =
  Array.ba_to_unboxed_int8_array
    (Array1.of_array Bigarray.int8_signed Bigarray.c_layout a)

let ub_int16_of_array a =
  Array.ba_to_unboxed_int16_array
    (Array1.of_array Bigarray.int16_signed Bigarray.c_layout a)

let ub_int32_of_array a =
  Array.ba_to_unboxed_int32_array
    (Array1.of_array Bigarray.int32 Bigarray.c_layout a)

let ub_int64_of_array a =
  Array.ba_to_unboxed_int64_array
    (Array1.of_array Bigarray.int64 Bigarray.c_layout a)

let array_of_ub_float64 a n =
  let ba = Array.unboxed_float64_to_ba a n in
  Stdlib.Array.init n (fun i -> Bigarray.Array1.unsafe_get ba i)

let array_of_ub_float32 a n =
  let ba = Array.unboxed_float32_to_ba a n in
  Stdlib.Array.init n (fun i -> Bigarray.Array1.unsafe_get ba i)

let array_of_ub_int8 a n =
  let ba = Array.unboxed_int8_to_ba a n in
  Stdlib.Array.init n (fun i -> Bigarray.Array1.unsafe_get ba i)

let array_of_ub_int16 a n =
  let ba = Array.unboxed_int16_to_ba a n in
  Stdlib.Array.init n (fun i -> Bigarray.Array1.unsafe_get ba i)

let array_of_ub_int32 a n =
  let ba = Array.unboxed_int32_to_ba a n in
  Stdlib.Array.init n (fun i -> Bigarray.Array1.unsafe_get ba i)

let array_of_ub_int64 a n =
  let ba = Array.unboxed_int64_to_ba a n in
  Stdlib.Array.init n (fun i -> Bigarray.Array1.unsafe_get ba i)

let test_cast_float64_group () =
  let src = ub_float64_of_array [| -2.5; 0.0; 3.9; 1.0 |] in
  let n = 4 in
  let shape = [| n |] in
  let src_view = View.create shape in
  let dst_view = View.create shape in

  let dst_f32 = Array.make_float32 n in
  Op_cast.cast_float64_float32 src dst_f32 src_view dst_view;
  check_float_array "cast_f64_f32" ~eps:1e-6 [| -2.5; 0.0; 3.9; 1.0 |]
    (array_of_ub_float32 dst_f32 n);

  let dst_i8 = Array.make_int8 n in
  Op_cast.cast_float64_int8 src dst_i8 src_view dst_view;
  check_int_array "cast_f64_i8" [| -2; 0; 3; 1 |] (array_of_ub_int8 dst_i8 n);

  let dst_i16 = Array.make_int16 n in
  Op_cast.cast_float64_int16 src dst_i16 src_view dst_view;
  check_int_array "cast_f64_i16" [| -2; 0; 3; 1 |]
    (array_of_ub_int16 dst_i16 n);

  let dst_i32 = Array.make_int32 n in
  Op_cast.cast_float64_int32 src dst_i32 src_view dst_view;
  check_int32_array "cast_f64_i32" [| -2l; 0l; 3l; 1l |]
    (array_of_ub_int32 dst_i32 n);

  let dst_i64 = Array.make_int64 n in
  Op_cast.cast_float64_int64 src dst_i64 src_view dst_view;
  check_int64_array "cast_f64_i64" [| -2L; 0L; 3L; 1L |]
    (array_of_ub_int64 dst_i64 n);

  let dst_bool = Stdlib.Array.make n false in
  Op_cast.cast_float64_bool src dst_bool src_view dst_view;
  check_bool_array "cast_f64_bool" [| true; false; true; true |] dst_bool

let test_cast_float32_group () =
  let src = ub_float32_of_array [| -2.5; 0.0; 3.9; 1.0 |] in
  let n = 4 in
  let shape = [| n |] in
  let src_view = View.create shape in
  let dst_view = View.create shape in

  let dst_f64 = Array.make_float64 n in
  Op_cast.cast_float32_float64 src dst_f64 src_view dst_view;
  check_float_array "cast_f32_f64" ~eps:1e-6 [| -2.5; 0.0; 3.9; 1.0 |]
    (array_of_ub_float64 dst_f64 n);

  let dst_i8 = Array.make_int8 n in
  Op_cast.cast_float32_int8 src dst_i8 src_view dst_view;
  check_int_array "cast_f32_i8" [| -2; 0; 3; 1 |] (array_of_ub_int8 dst_i8 n);

  let dst_i16 = Array.make_int16 n in
  Op_cast.cast_float32_int16 src dst_i16 src_view dst_view;
  check_int_array "cast_f32_i16" [| -2; 0; 3; 1 |]
    (array_of_ub_int16 dst_i16 n);

  let dst_i32 = Array.make_int32 n in
  Op_cast.cast_float32_int32 src dst_i32 src_view dst_view;
  check_int32_array "cast_f32_i32" [| -2l; 0l; 3l; 1l |]
    (array_of_ub_int32 dst_i32 n);

  let dst_i64 = Array.make_int64 n in
  Op_cast.cast_float32_int64 src dst_i64 src_view dst_view;
  check_int64_array "cast_f32_i64" [| -2L; 0L; 3L; 1L |]
    (array_of_ub_int64 dst_i64 n);

  let dst_bool = Stdlib.Array.make n false in
  Op_cast.cast_float32_bool src dst_bool src_view dst_view;
  check_bool_array "cast_f32_bool" [| true; false; true; true |] dst_bool

let test_cast_int8_group () =
  let src = ub_int8_of_array [| -2; 0; 3; 1 |] in
  let n = 4 in
  let shape = [| n |] in
  let src_view = View.create shape in
  let dst_view = View.create shape in

  let dst_f64 = Array.make_float64 n in
  Op_cast.cast_int8_float64 src dst_f64 src_view dst_view;
  check_float_array "cast_i8_f64" ~eps:0.0 [| -2.0; 0.0; 3.0; 1.0 |]
    (array_of_ub_float64 dst_f64 n);

  let dst_f32 = Array.make_float32 n in
  Op_cast.cast_int8_float32 src dst_f32 src_view dst_view;
  check_float_array "cast_i8_f32" ~eps:0.0 [| -2.0; 0.0; 3.0; 1.0 |]
    (array_of_ub_float32 dst_f32 n);

  let dst_i16 = Array.make_int16 n in
  Op_cast.cast_int8_int16 src dst_i16 src_view dst_view;
  check_int_array "cast_i8_i16" [| -2; 0; 3; 1 |]
    (array_of_ub_int16 dst_i16 n);

  let dst_i32 = Array.make_int32 n in
  Op_cast.cast_int8_int32 src dst_i32 src_view dst_view;
  check_int32_array "cast_i8_i32" [| -2l; 0l; 3l; 1l |]
    (array_of_ub_int32 dst_i32 n);

  let dst_i64 = Array.make_int64 n in
  Op_cast.cast_int8_int64 src dst_i64 src_view dst_view;
  check_int64_array "cast_i8_i64" [| -2L; 0L; 3L; 1L |]
    (array_of_ub_int64 dst_i64 n);

  let dst_bool = Stdlib.Array.make n false in
  Op_cast.cast_int8_bool src dst_bool src_view dst_view;
  check_bool_array "cast_i8_bool" [| true; false; true; true |] dst_bool

let test_cast_int16_group () =
  let src = ub_int16_of_array [| -2; 0; 3; 1 |] in
  let n = 4 in
  let shape = [| n |] in
  let src_view = View.create shape in
  let dst_view = View.create shape in

  let dst_f64 = Array.make_float64 n in
  Op_cast.cast_int16_float64 src dst_f64 src_view dst_view;
  check_float_array "cast_i16_f64" ~eps:0.0 [| -2.0; 0.0; 3.0; 1.0 |]
    (array_of_ub_float64 dst_f64 n);

  let dst_f32 = Array.make_float32 n in
  Op_cast.cast_int16_float32 src dst_f32 src_view dst_view;
  check_float_array "cast_i16_f32" ~eps:0.0 [| -2.0; 0.0; 3.0; 1.0 |]
    (array_of_ub_float32 dst_f32 n);

  let dst_i8 = Array.make_int8 n in
  Op_cast.cast_int16_int8 src dst_i8 src_view dst_view;
  check_int_array "cast_i16_i8" [| -2; 0; 3; 1 |] (array_of_ub_int8 dst_i8 n);

  let dst_i32 = Array.make_int32 n in
  Op_cast.cast_int16_int32 src dst_i32 src_view dst_view;
  check_int32_array "cast_i16_i32" [| -2l; 0l; 3l; 1l |]
    (array_of_ub_int32 dst_i32 n);

  let dst_i64 = Array.make_int64 n in
  Op_cast.cast_int16_int64 src dst_i64 src_view dst_view;
  check_int64_array "cast_i16_i64" [| -2L; 0L; 3L; 1L |]
    (array_of_ub_int64 dst_i64 n);

  let dst_bool = Stdlib.Array.make n false in
  Op_cast.cast_int16_bool src dst_bool src_view dst_view;
  check_bool_array "cast_i16_bool" [| true; false; true; true |] dst_bool

let test_cast_int32_group () =
  let src = ub_int32_of_array [| -2l; 0l; 3l; 1l |] in
  let n = 4 in
  let shape = [| n |] in
  let src_view = View.create shape in
  let dst_view = View.create shape in

  let dst_f64 = Array.make_float64 n in
  Op_cast.cast_int32_float64 src dst_f64 src_view dst_view;
  check_float_array "cast_i32_f64" ~eps:0.0 [| -2.0; 0.0; 3.0; 1.0 |]
    (array_of_ub_float64 dst_f64 n);

  let dst_f32 = Array.make_float32 n in
  Op_cast.cast_int32_float32 src dst_f32 src_view dst_view;
  check_float_array "cast_i32_f32" ~eps:0.0 [| -2.0; 0.0; 3.0; 1.0 |]
    (array_of_ub_float32 dst_f32 n);

  let dst_i8 = Array.make_int8 n in
  Op_cast.cast_int32_int8 src dst_i8 src_view dst_view;
  check_int_array "cast_i32_i8" [| -2; 0; 3; 1 |] (array_of_ub_int8 dst_i8 n);

  let dst_i16 = Array.make_int16 n in
  Op_cast.cast_int32_int16 src dst_i16 src_view dst_view;
  check_int_array "cast_i32_i16" [| -2; 0; 3; 1 |]
    (array_of_ub_int16 dst_i16 n);

  let dst_i64 = Array.make_int64 n in
  Op_cast.cast_int32_int64 src dst_i64 src_view dst_view;
  check_int64_array "cast_i32_i64" [| -2L; 0L; 3L; 1L |]
    (array_of_ub_int64 dst_i64 n);

  let dst_bool = Stdlib.Array.make n false in
  Op_cast.cast_int32_bool src dst_bool src_view dst_view;
  check_bool_array "cast_i32_bool" [| true; false; true; true |] dst_bool

let test_cast_int64_group () =
  let src = ub_int64_of_array [| -2L; 0L; 3L; 1L |] in
  let n = 4 in
  let shape = [| n |] in
  let src_view = View.create shape in
  let dst_view = View.create shape in

  let dst_f64 = Array.make_float64 n in
  Op_cast.cast_int64_float64 src dst_f64 src_view dst_view;
  check_float_array "cast_i64_f64" ~eps:0.0 [| -2.0; 0.0; 3.0; 1.0 |]
    (array_of_ub_float64 dst_f64 n);

  let dst_f32 = Array.make_float32 n in
  Op_cast.cast_int64_float32 src dst_f32 src_view dst_view;
  check_float_array "cast_i64_f32" ~eps:0.0 [| -2.0; 0.0; 3.0; 1.0 |]
    (array_of_ub_float32 dst_f32 n);

  let dst_i8 = Array.make_int8 n in
  Op_cast.cast_int64_int8 src dst_i8 src_view dst_view;
  check_int_array "cast_i64_i8" [| -2; 0; 3; 1 |] (array_of_ub_int8 dst_i8 n);

  let dst_i16 = Array.make_int16 n in
  Op_cast.cast_int64_int16 src dst_i16 src_view dst_view;
  check_int_array "cast_i64_i16" [| -2; 0; 3; 1 |]
    (array_of_ub_int16 dst_i16 n);

  let dst_i32 = Array.make_int32 n in
  Op_cast.cast_int64_int32 src dst_i32 src_view dst_view;
  check_int32_array "cast_i64_i32" [| -2l; 0l; 3l; 1l |]
    (array_of_ub_int32 dst_i32 n);

  let dst_bool = Stdlib.Array.make n false in
  Op_cast.cast_int64_bool src dst_bool src_view dst_view;
  check_bool_array "cast_i64_bool" [| true; false; true; true |] dst_bool

let test_cast_bool_group () =
  let src = [| true; false; true; false |] in
  let n = 4 in
  let shape = [| n |] in
  let src_view = View.create shape in
  let dst_view = View.create shape in

  let dst_f64 = Array.make_float64 n in
  Op_cast.cast_bool_float64 src dst_f64 src_view dst_view;
  check_float_array "cast_bool_f64" ~eps:0.0 [| 1.0; 0.0; 1.0; 0.0 |]
    (array_of_ub_float64 dst_f64 n);

  let dst_f32 = Array.make_float32 n in
  Op_cast.cast_bool_float32 src dst_f32 src_view dst_view;
  check_float_array "cast_bool_f32" ~eps:0.0 [| 1.0; 0.0; 1.0; 0.0 |]
    (array_of_ub_float32 dst_f32 n);

  let dst_i8 = Array.make_int8 n in
  Op_cast.cast_bool_int8 src dst_i8 src_view dst_view;
  check_int_array "cast_bool_i8" [| 1; 0; 1; 0 |] (array_of_ub_int8 dst_i8 n);

  let dst_i16 = Array.make_int16 n in
  Op_cast.cast_bool_int16 src dst_i16 src_view dst_view;
  check_int_array "cast_bool_i16" [| 1; 0; 1; 0 |]
    (array_of_ub_int16 dst_i16 n);

  let dst_i32 = Array.make_int32 n in
  Op_cast.cast_bool_int32 src dst_i32 src_view dst_view;
  check_int32_array "cast_bool_i32" [| 1l; 0l; 1l; 0l |]
    (array_of_ub_int32 dst_i32 n);

  let dst_i64 = Array.make_int64 n in
  Op_cast.cast_bool_int64 src dst_i64 src_view dst_view;
  check_int64_array "cast_bool_i64" [| 1L; 0L; 1L; 0L |]
    (array_of_ub_int64 dst_i64 n)

let test_cast_float64_float32_strided () =
  let src = ub_float64_of_array [| 1.0; 2.0; 3.0; 4.0 |] in
  let dst = Array.make_float32 4 in
  let src_view = View.create ~strides:[| 2; 1 |] [| 2; 2 |] in
  let dst_view = View.create ~strides:[| 1; 2 |] [| 2; 2 |] in
  Op_cast.cast_float64_float32 src dst src_view dst_view;
  check_float_array "cast_f64_f32_strided" ~eps:1e-6 [| 1.0; 3.0; 2.0; 4.0 |]
    (array_of_ub_float32 dst 4)

let () =
  print_endline "Running Op_cast tests...";
  test_cast_float64_group ();
  test_cast_float32_group ();
  test_cast_int8_group ();
  test_cast_int16_group ();
  test_cast_int32_group ();
  test_cast_int64_group ();
  test_cast_bool_group ();
  test_cast_float64_float32_strided ();
  Printf.printf "\nResults: %d passed, %d failed\n" !passed !failed;
  if !failed > 0 then exit 1
