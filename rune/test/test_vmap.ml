open Test_rune_support
module T = Rune

let ctx = T.c
let eps = 1e-6

(* Test basic vmap functionality *)
let test_vmap_simple () =
  let x = T.create ctx T.float32 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let f t = T.mul_s t 2. in
  let vmapped_f = T.vmap f in
  let result = vmapped_f x in
  let expected =
    T.create ctx T.float32 [| 3; 2 |] [| 2.; 4.; 6.; 8.; 10.; 12. |]
  in
  check_rune ~eps "vmap simple" expected result

(* Test vmap with matrix multiplication *)
let test_vmap_matmul () =
  let batch_x =
    T.create ctx T.float32 [| 2; 3; 3 |]
      [|
        1.;
        2.;
        3.;
        4.;
        5.;
        6.;
        7.;
        8.;
        9.;
        10.;
        11.;
        12.;
        13.;
        14.;
        15.;
        16.;
        17.;
        18.;
      |]
  in
  let w = T.create ctx T.float32 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let batched_matmul = T.vmap (fun x -> T.matmul x w) in
  let result = batched_matmul batch_x in

  (* Expected: batch of 2 matrix multiplications *)
  let expected_shape = [| 2; 3; 2 |] in
  check_shape "vmap matmul shape" expected_shape result;

  (* Check first batch result *)
  let first_batch = T.get [ 0 ] result in
  let expected_first = T.matmul (T.get [ 0 ] batch_x) w in
  check_rune ~eps "vmap matmul first batch" expected_first first_batch

(* Test vmap with different axis *)
let test_vmap_axis () =
  let x = T.create ctx T.float32 [| 2; 3; 4 |] (Array.init 24 float_of_int) in
  let f = T.vmap ~in_axes:(T.Single (T.Map 1)) (fun t -> T.sum t) in
  let result = f x in
  let expected_shape = [| 3; 2 |] in
  check_shape "vmap axis shape" expected_shape result

(* Test vmap with no output axis *)
let test_vmap_no_out_axis () =
  let x = T.create ctx T.float32 [| 5; 3 |] (Array.init 15 float_of_int) in
  let f = T.vmap ~out_axes:(T.OutSingle None) (fun t -> T.sum t) in
  let result = f x in
  check_scalar ~eps "vmap no out axis" 105. (scalar_value result)

(* Test vmap with broadcasting *)
let test_vmap_broadcast () =
  let x = T.create ctx T.float32 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let y = T.create ctx T.float32 [| 2 |] [| 10.; 20. |] in
  let f = T.vmap (fun t -> T.add t y) in
  let result = f x in
  let expected =
    T.create ctx T.float32 [| 3; 2 |] [| 11.; 22.; 13.; 24.; 15.; 26. |]
  in
  check_rune ~eps "vmap broadcast" expected result

(* Test nested vmap *)
let test_nested_vmap () =
  let x = T.create ctx T.float32 [| 2; 3; 4 |] (Array.init 24 float_of_int) in
  let inner_vmap = T.vmap (fun t -> T.mul_s t 2.) in
  let outer_vmap = T.vmap inner_vmap in
  let result = outer_vmap x in
  let expected_shape = [| 2; 3; 4 |] in
  check_shape "nested vmap shape" expected_shape result;

  (* Check that all values are doubled *)
  let first_val = T.unsafe_get [ 0; 0; 0 ] result in
  check_scalar ~eps "nested vmap first value" 0. first_val

(* Test vmap with reduction *)
let test_vmap_reduction () =
  let x = T.create ctx T.float32 [| 4; 3; 2 |] (Array.init 24 float_of_int) in
  let f = T.vmap (fun t -> T.sum t ~axes:[| 1 |]) in
  let result = f x in
  let expected_shape = [| 4; 3 |] in
  check_shape "vmap reduction shape" expected_shape result

(* Test vmap with where operation *)
let test_vmap_where () =
  let cond = T.create ctx T.uint8 [| 3; 2 |] [| 1; 0; 1; 1; 0; 1 |] in
  let x = T.create ctx T.float32 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let y =
    T.create ctx T.float32 [| 3; 2 |] [| 10.; 20.; 30.; 40.; 50.; 60. |]
  in
  let f = T.vmap (fun c -> T.where c x y) in
  let result = f cond in
  let expected =
    T.create ctx T.float32 [| 3; 2 |] [| 1.; 20.; 3.; 4.; 50.; 6. |]
  in
  check_rune ~eps "vmap where" expected result

(* Test vmap with transpose *)
let test_vmap_transpose () =
  let x = T.create ctx T.float32 [| 2; 3; 4 |] (Array.init 24 float_of_int) in
  let f = T.vmap (fun t -> T.transpose t) in
  let result = f x in
  let expected_shape = [| 2; 4; 3 |] in
  check_shape "vmap transpose shape" expected_shape result

(* Test vmap with elementwise operations *)
let test_vmap_elementwise () =
  let x =
    T.create ctx T.float32 [| 3; 4 |]
      (Array.init 12 (fun i -> float_of_int (i + 1)))
  in
  let y =
    T.create ctx T.float32 [| 3; 4 |]
      (Array.init 12 (fun i -> float_of_int (i + 1)))
  in

  (* Test various elementwise operations *)
  let test_op name op expected_fn =
    let f = T.vmap (fun a -> op a y) in
    let result = f x in
    let expected =
      T.create ctx T.float32 [| 3; 4 |]
        (Array.init 12 (fun i -> expected_fn (float_of_int (i + 1))))
    in
    check_rune ~eps (Printf.sprintf "vmap %s" name) expected result
  in

  test_op "add" T.add (fun x -> x +. x);
  test_op "mul" T.mul (fun x -> x *. x);
  test_op "sub" T.sub (fun _ -> 0.);
  test_op "div" T.div (fun _ -> 1.)

let () =
  let open Alcotest in
  run "Vmap tests"
    [
      ( "basic",
        [
          test_case "simple" `Quick test_vmap_simple;
          test_case "matmul" `Quick test_vmap_matmul;
          test_case "axis" `Quick test_vmap_axis;
          test_case "no_out_axis" `Quick test_vmap_no_out_axis;
          test_case "broadcast" `Quick test_vmap_broadcast;
          test_case "nested" `Quick test_nested_vmap;
          test_case "reduction" `Quick test_vmap_reduction;
          test_case "where" `Quick test_vmap_where;
          test_case "transpose" `Quick test_vmap_transpose;
          test_case "elementwise" `Quick test_vmap_elementwise;
        ] );
    ]
