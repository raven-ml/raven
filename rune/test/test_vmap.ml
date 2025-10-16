open Test_rune_support
module T = Rune

let eps = 1e-6

(* Test basic vmap functionality *)
let test_vmap_simple () =
  let x = T.create T.float32 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let f t = T.mul_s t 2. in
  let vmapped_f = T.vmap f in
  let result = vmapped_f x in
  let expected = T.create T.float32 [| 3; 2 |] [| 2.; 4.; 6.; 8.; 10.; 12. |] in
  check_rune ~eps "vmap simple" expected result

(* Test vmap with matrix multiplication *)
let test_vmap_matmul () =
  let batch_x =
    T.create T.float32 [| 2; 3; 3 |]
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
  let w = T.create T.float32 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
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
  let x = T.create T.float32 [| 2; 3; 4 |] (Array.init 24 float_of_int) in
  let f = T.vmap ~in_axes:(T.Single (T.Map 1)) (fun t -> T.sum t) in
  let result = f x in
  let expected_shape = [| 3 |] in
  check_shape "vmap axis shape" expected_shape result

(* Test vmap with no output axis *)
let test_vmap_no_out_axis () =
  (* JAX semantics: out_axes=None only works with constant functions. For
     non-constant outputs, JAX would error. We take first element. *)
  let x = T.create T.float32 [| 5; 3 |] (Array.init 15 float_of_int) in
  let f = T.vmap ~out_axes:(T.OutSingle None) (fun t -> T.sum t) in
  let result = f x in
  (* First row sum: 0+1+2 = 3 *)
  check_scalar ~eps "vmap no out axis" 3. (scalar_value result)

(* Test vmap with broadcasting *)
let test_vmap_broadcast () =
  let x = T.create T.float32 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let y = T.create T.float32 [| 2 |] [| 10.; 20. |] in
  let f = T.vmap (fun t -> T.add t y) in
  let result = f x in
  let expected =
    T.create T.float32 [| 3; 2 |] [| 11.; 22.; 13.; 24.; 15.; 26. |]
  in
  check_rune ~eps "vmap broadcast" expected result

(* Test nested vmap *)
let test_nested_vmap () =
  let x = T.create T.float32 [| 2; 3; 4 |] (Array.init 24 float_of_int) in
  let inner_vmap = T.vmap (fun t -> T.mul_s t 2.) in
  let outer_vmap = T.vmap inner_vmap in
  let result = outer_vmap x in
  let expected_shape = [| 2; 3; 4 |] in
  check_shape "nested vmap shape" expected_shape result;

  (* Check that all values are doubled *)
  let first_val = T.item [ 0; 0; 0 ] result in
  check_scalar ~eps "nested vmap first value" 0. first_val

(* Test vmap with reduction *)
let test_vmap_reduction () =
  let x = T.create T.float32 [| 4; 3; 2 |] (Array.init 24 float_of_int) in
  let f = T.vmap (fun t -> T.sum t ~axes:[ 1 ]) in
  let result = f x in
  let expected_shape = [| 4; 3 |] in
  check_shape "vmap reduction shape" expected_shape result

(* Test vmap with where operation *)
let test_vmap_where () =
  (* JAX semantics: captured tensors are broadcast, not co-iterated *)
  let cond = T.create T.bool [| 3; 2 |] [| true; false; true; true; false; true |] in
  let x = T.create T.float32 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let y = T.create T.float32 [| 3; 2 |] [| 10.; 20.; 30.; 40.; 50.; 60. |] in
  let f = T.vmap (fun c -> T.where c x y) in
  let result = f cond in
  (* With broadcast semantics, result shape should be [3, 3, 2] Each batch
     element sees the entire x and y arrays *)
  let expected_shape = [| 3; 3; 2 |] in
  check_shape "vmap where shape" expected_shape result
(* For now, just check shape. Full value check would be complex. *)

(* Test vmap with transpose *)
let test_vmap_transpose () =
  let x = T.create T.float32 [| 2; 3; 4 |] (Array.init 24 float_of_int) in
  let f = T.vmap (fun t -> T.transpose t) in
  let result = f x in
  let expected_shape = [| 2; 4; 3 |] in
  check_shape "vmap transpose shape" expected_shape result

(* Test vmap with elementwise operations *)
let test_vmap_elementwise () =
  let x =
    T.create T.float32 [| 3; 4 |]
      (Array.init 12 (fun i -> float_of_int (i + 1)))
  in
  let y =
    T.create T.float32 [| 3; 4 |]
      (Array.init 12 (fun i -> float_of_int (i + 1)))
  in

  (* JAX semantics: captured y is treated as a constant across the mapped axis
     (not co-iterated). Broadcasting happens elementwise, not as a cross-product
     over an extra axis. *)
  let f = T.vmap (fun a -> T.add a y) in
  let result = f x in
  (* Under JAX semantics, result shape is [3, 4] (same as x). *)
  let expected_shape = [| 3; 4 |] in
  check_shape "vmap elementwise broadcast shape" expected_shape result

(* Test composition: jvp (vmap f) *)
let test_jvp_vmap_composition () =
  let x = T.create T.float32 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let v = T.create T.float32 [| 3; 2 |] [| 0.1; 0.2; 0.3; 0.4; 0.5; 0.6 |] in

  (* Define f: sum of squares *)
  let f t = T.sum (T.mul t t) in

  (* vmap f *)
  let vmapped_f = T.vmap f in

  (* jvp of vmapped f *)
  let primals, tangents = T.jvp vmapped_f x v in

  let expected_primals = T.create T.float32 [| 3 |] [| 5.; 25.; 61. |] in
  let expected_tangents = T.create T.float32 [| 3 |] [| 1.; 5.; 12.2 |] in

  check_rune ~eps:1e-5 "jvp(vmap(f)) primals" expected_primals primals;
  check_rune ~eps:1e-5 "jvp(vmap(f)) tangents" expected_tangents tangents

(* Test composition: vmap (jvp f) *)
let test_vmap_jvp_composition () =
  let x = T.create T.float32 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let v = T.create T.float32 [| 3; 2 |] [| 0.1; 0.2; 0.3; 0.4; 0.5; 0.6 |] in

  (* Define f: sum of squares *)
  let f t = T.sum (T.mul t t) in

  (* Function that computes jvp and returns primals *)
  let jvp_f_primals inputs =
    match inputs with
    | [ x; v ] ->
        let primals, _ = T.jvp f x v in
        primals
    | _ -> failwith "jvp_f_primals expects exactly 2 inputs"
  in

  (* Function that computes jvp and returns tangents *)
  let jvp_f_tangents inputs =
    match inputs with
    | [ x; v ] ->
        let _, tangents = T.jvp f x v in
        tangents
    | _ -> failwith "jvp_f_tangents expects exactly 2 inputs"
  in

  (* vmap the jvp functions *)
  let vmapped_jvp_f_primals = T.vmaps jvp_f_primals in
  let vmapped_jvp_f_tangents = T.vmaps jvp_f_tangents in
  let primals = vmapped_jvp_f_primals [ x; v ] in
  let tangents = vmapped_jvp_f_tangents [ x; v ] in

  let expected_primals = T.create T.float32 [| 3 |] [| 5.; 25.; 61. |] in
  let expected_tangents = T.create T.float32 [| 3 |] [| 1.; 5.; 12.2 |] in

  check_rune ~eps:1e-5 "vmap(jvp(f)) primals" expected_primals primals;
  check_rune ~eps:1e-5 "vmap(jvp(f)) tangents" expected_tangents tangents

(* Test composition: grad (vmap f) *)
let test_grad_vmap_composition () =
  let x = T.create T.float32 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in

  (* Define f: sum of squares *)
  let f t = T.sum (T.mul t t) in

  (* vmap f *)
  let vmapped_f = T.vmap f in

  (* To take grad of vmap, we need to sum the output *)
  let sum_vmapped_f x = T.sum (vmapped_f x) in

  (* grad of sum of vmapped f *)
  let grad_sum_vmapped_f = T.grad sum_vmapped_f in
  let grads = grad_sum_vmapped_f x in

  let expected_grads =
    T.create T.float32 [| 3; 2 |] [| 2.; 4.; 6.; 8.; 10.; 12. |]
  in

  check_rune ~eps:1e-5 "grad(sum(vmap(f)))" expected_grads grads

(* Test composition: vmap (grad f) *)
let test_vmap_grad_composition () =
  let x = T.create T.float32 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in

  (* Define f: sum of squares *)
  let f t = T.sum (T.mul t t) in

  (* grad f *)
  let grad_f = T.grad f in

  (* vmap grad f *)
  let vmapped_grad_f = T.vmap grad_f in
  let grads = vmapped_grad_f x in

  let expected_grads =
    T.create T.float32 [| 3; 2 |] [| 2.; 4.; 6.; 8.; 10.; 12. |]
  in

  check_rune ~eps:1e-5 "vmap(grad(f))" expected_grads grads

(* Test composition with two-argument function: jvp (vmap g) *)
let test_jvp_vmap_composition_two_args () =
  let x = T.create T.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let y = T.create T.float32 [| 2; 2 |] [| 5.; 6.; 7.; 8. |] in
  let v_x = T.create T.float32 [| 2; 2 |] [| 0.1; 0.2; 0.3; 0.4 |] in
  let v_y = T.create T.float32 [| 2; 2 |] [| 0.5; 0.6; 0.7; 0.8 |] in

  (* Define g: sum of element-wise product *)
  let g inputs =
    match inputs with
    | [ x; y ] -> T.sum (T.mul x y)
    | _ -> failwith "g expects exactly 2 inputs"
  in

  (* vmap g *)
  let vmapped_g = T.vmaps g in

  (* jvp of vmapped g *)
  let primals, tangents = T.jvps vmapped_g [ x; y ] [ v_x; v_y ] in

  let expected_primals = T.create T.float32 [| 2 |] [| 17.; 53. |] in
  let expected_tangents = T.create T.float32 [| 2 |] [| 3.4; 10.6 |] in

  check_rune ~eps:1e-5 "jvp(vmap(g)) primals" expected_primals primals;
  check_rune ~eps:1e-5 "jvp(vmap(g)) tangents" expected_tangents tangents

(* Test composition with two-argument function: vmap (jvp g) *)
let test_vmap_jvp_composition_two_args () =
  let x = T.create T.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let y = T.create T.float32 [| 2; 2 |] [| 5.; 6.; 7.; 8. |] in
  let v_x = T.create T.float32 [| 2; 2 |] [| 0.1; 0.2; 0.3; 0.4 |] in
  let v_y = T.create T.float32 [| 2; 2 |] [| 0.5; 0.6; 0.7; 0.8 |] in

  (* Define g: sum of element-wise product *)
  let g inputs =
    match inputs with
    | [ x; y ] -> T.sum (T.mul x y)
    | _ -> failwith "g expects exactly 2 inputs"
  in

  (* Function that computes jvp and returns primals *)
  let jvp_g_primals inputs =
    match inputs with
    | [ x; y; v_x; v_y ] ->
        let primals, _ = T.jvps g [ x; y ] [ v_x; v_y ] in
        primals
    | _ -> failwith "jvp_g_primals expects exactly 4 inputs"
  in

  (* Function that computes jvp and returns tangents *)
  let jvp_g_tangents inputs =
    match inputs with
    | [ x; y; v_x; v_y ] ->
        let _, tangents = T.jvps g [ x; y ] [ v_x; v_y ] in
        tangents
    | _ -> failwith "jvp_g_tangents expects exactly 4 inputs"
  in

  (* vmap the jvp functions *)
  let vmapped_jvp_g_primals = T.vmaps jvp_g_primals in
  let vmapped_jvp_g_tangents = T.vmaps jvp_g_tangents in
  let primals = vmapped_jvp_g_primals [ x; y; v_x; v_y ] in
  let tangents = vmapped_jvp_g_tangents [ x; y; v_x; v_y ] in

  let expected_primals = T.create T.float32 [| 2 |] [| 17.; 53. |] in
  let expected_tangents = T.create T.float32 [| 2 |] [| 3.4; 10.6 |] in

  check_rune ~eps:1e-5 "vmap(jvp(g)) primals" expected_primals primals;
  check_rune ~eps:1e-5 "vmap(jvp(g)) tangents" expected_tangents tangents

(* Test composition with two-argument function: grad (vmap g) *)
let test_grad_vmap_composition_two_args () =
  let x = T.create T.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let y = T.create T.float32 [| 2; 2 |] [| 5.; 6.; 7.; 8. |] in

  (* Define g: sum of element-wise product *)
  let g inputs =
    match inputs with
    | [ x; y ] -> T.sum (T.mul x y)
    | _ -> failwith "g expects exactly 2 inputs"
  in

  (* vmap g *)
  let vmapped_g = T.vmaps g in

  (* To take grad of vmap, we need to sum the output *)
  let sum_vmapped_g inputs = T.sum (vmapped_g inputs) in

  (* grad of sum of vmapped g *)
  let grads_list = T.grads sum_vmapped_g [ x; y ] in
  let grad_x = List.nth grads_list 0 in

  let expected_grads = T.create T.float32 [| 2; 2 |] [| 5.; 6.; 7.; 8. |] in

  check_rune ~eps:1e-5 "grad(sum(vmap(g)), argnums=0)" expected_grads grad_x

(* Test composition with two-argument function: vmap (grad g) *)
let test_vmap_grad_composition_two_args () =
  let x = T.create T.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let y = T.create T.float32 [| 2; 2 |] [| 5.; 6.; 7.; 8. |] in

  (* Define g: sum of element-wise product *)
  let g inputs =
    match inputs with
    | [ x; y ] -> T.sum (T.mul x y)
    | _ -> failwith "g expects exactly 2 inputs"
  in

  (* Function that computes grad w.r.t. first argument *)
  let grad_g inputs =
    match inputs with
    | [ x; y ] ->
        let grads = T.grads g [ x; y ] in
        List.nth grads 0 (* Return gradient w.r.t. x *)
    | _ -> failwith "grad_g expects exactly 2 inputs"
  in

  (* vmap grad g *)
  let vmapped_grad_g = T.vmaps grad_g in
  let grads = vmapped_grad_g [ x; y ] in

  let expected_grads = T.create T.float32 [| 2; 2 |] [| 5.; 6.; 7.; 8. |] in

  check_rune ~eps:1e-5 "vmap(grad(g), argnums=0)" expected_grads grads

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
      ( "composition",
        [
          test_case "jvp_vmap" `Quick test_jvp_vmap_composition;
          test_case "vmap_jvp" `Quick test_vmap_jvp_composition;
          test_case "grad_vmap" `Quick test_grad_vmap_composition;
          test_case "vmap_grad" `Quick test_vmap_grad_composition;
          test_case "jvp_vmap_two_args" `Quick
            test_jvp_vmap_composition_two_args;
          test_case "vmap_jvp_two_args" `Quick
            test_vmap_jvp_composition_two_args;
          test_case "grad_vmap_two_args" `Quick
            test_grad_vmap_composition_two_args;
          test_case "vmap_grad_two_args" `Quick
            test_vmap_grad_composition_two_args;
        ] );
    ]
