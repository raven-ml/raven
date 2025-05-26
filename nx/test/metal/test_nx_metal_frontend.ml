open Alcotest

(* Create the Metal frontend *)
module Nx = Nx_core.Make_frontend (Nx_metal)

(* Create a context *)
let ctx = Nx_metal.create_context ()

(* Helper to extract scalar value from tensor *)
let get_scalar t = Nx.unsafe_get [] t

(* Helper to create tensors *)
let tensor_from_list dtype lst =
  let n = List.length lst in
  let t = Nx.zeros ctx dtype [| n |] in
  List.iteri
    (fun i v ->
      let v_tensor = Nx.full ctx dtype [||] v in
      Nx.set [ i ] t v_tensor)
    lst;
  t

(* Helper to compare tensors *)
let check_tensor ?(eps = 1e-5) msg expected actual =
  let expected_shape = Nx.shape expected in
  let actual_shape = Nx.shape actual in
  check (array int) (msg ^ " shape") expected_shape actual_shape;

  let n = Array.fold_left ( * ) 1 expected_shape in
  for i = 0 to n - 1 do
    let idx =
      let rec compute_idx i dims =
        match dims with
        | [] -> []
        | d :: rest ->
            let q = i / d in
            let r = i mod d in
            r :: compute_idx q rest
      in
      Array.of_list
        (List.rev (compute_idx i (List.rev (Array.to_list expected_shape))))
    in

    let expected_tensor = Nx.get (Array.to_list idx) expected in
    let actual_tensor = Nx.get (Array.to_list idx) actual in
    let expected_val = get_scalar expected_tensor in
    let actual_val = get_scalar actual_tensor in

    if Nx_core.Dtype.is_float (Nx.dtype expected) then
      let diff = abs_float (expected_val -. actual_val) in
      if diff > eps then
        failf "%s[%s]: expected %f, got %f (diff: %f)" msg
          (String.concat "," (Array.to_list (Array.map string_of_int idx)))
          expected_val actual_val diff
      else if expected_val <> actual_val then
        failf "%s[%s]: expected %s, got %s" msg
          (String.concat "," (Array.to_list (Array.map string_of_int idx)))
          (string_of_float expected_val)
          (string_of_float actual_val)
  done

(* Test basic tensor creation *)
let test_creation () =
  (* Zeros *)
  let z = Nx.zeros ctx Nx_core.Dtype.Float32 [| 3; 4 |] in
  check (array int) "zeros shape" [| 3; 4 |] (Nx.shape z);
  check (float 0.001) "zeros[0,0]" 0.0 (get_scalar (Nx.get [ 0; 0 ] z));

  (* Ones *)
  let o = Nx.ones ctx Nx_core.Dtype.Float32 [| 2; 3 |] in
  check (array int) "ones shape" [| 2; 3 |] (Nx.shape o);
  check (float 0.001) "ones[1,2]" 1.0 (get_scalar (Nx.get [ 1; 2 ] o));

  (* Full *)
  let f = Nx.full ctx Nx_core.Dtype.Float32 [| 2; 2 |] 42.0 in
  check (float 0.001) "full[0,1]" 42.0 (get_scalar (Nx.get [ 0; 1 ] f));

  (* Eye *)
  let e = Nx.eye ctx Nx_core.Dtype.Float32 3 in
  check (float 0.001) "eye[0,0]" 1.0 (get_scalar (Nx.get [ 0; 0 ] e));
  check (float 0.001) "eye[0,1]" 0.0 (get_scalar (Nx.get [ 0; 1 ] e));
  check (float 0.001) "eye[1,1]" 1.0 (get_scalar (Nx.get [ 1; 1 ] e))

(* Test arithmetic operations *)
let test_arithmetic () =
  let a = tensor_from_list Nx_core.Dtype.Float32 [ 1.0; 2.0; 3.0; 4.0 ] in
  let b = tensor_from_list Nx_core.Dtype.Float32 [ 5.0; 6.0; 7.0; 8.0 ] in

  (* Add *)
  let sum = Nx.add a b in
  check_tensor "add"
    (tensor_from_list Nx_core.Dtype.Float32 [ 6.0; 8.0; 10.0; 12.0 ])
    sum;

  (* Subtract *)
  let diff = Nx.sub a b in
  check_tensor "sub"
    (tensor_from_list Nx_core.Dtype.Float32 [ -4.0; -4.0; -4.0; -4.0 ])
    diff;

  (* Multiply *)
  let prod = Nx.mul a b in
  check_tensor "mul"
    (tensor_from_list Nx_core.Dtype.Float32 [ 5.0; 12.0; 21.0; 32.0 ])
    prod;

  (* Divide *)
  let quot = Nx.div a b in
  let expected = [ 1.0 /. 5.0; 2.0 /. 6.0; 3.0 /. 7.0; 4.0 /. 8.0 ] in
  check_tensor "div" (tensor_from_list Nx_core.Dtype.Float32 expected) quot

(* Test broadcasting *)
let test_broadcasting () =
  let a =
    Nx.reshape [| 3; 1 |]
      (tensor_from_list Nx_core.Dtype.Float32 [ 1.0; 2.0; 3.0 ])
  in
  let b =
    Nx.reshape [| 1; 2 |]
      (tensor_from_list Nx_core.Dtype.Float32 [ 10.0; 20.0 ])
  in

  (* Add with broadcasting *)
  let sum = Nx.add a b in
  check (array int) "broadcast shape" [| 3; 2 |] (Nx.shape sum);
  check (float 0.001) "broadcast[0,0]" 11.0 (get_scalar (Nx.get [ 0; 0 ] sum));
  check (float 0.001) "broadcast[0,1]" 21.0 (get_scalar (Nx.get [ 0; 1 ] sum));
  check (float 0.001) "broadcast[1,0]" 12.0 (get_scalar (Nx.get [ 1; 0 ] sum));
  check (float 0.001) "broadcast[2,1]" 23.0 (get_scalar (Nx.get [ 2; 1 ] sum))

(* Test reduction operations *)
let test_reductions () =
  let a =
    Nx.reshape [| 2; 3 |]
      (tensor_from_list Nx_core.Dtype.Float32 [ 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 ])
  in

  (* Sum all *)
  let sum_all = Nx.sum a in
  check (float 0.001) "sum all" 21.0 (get_scalar (Nx.get [] sum_all));

  (* Sum along axis 0 *)
  let sum0 = Nx.sum ~axes:[| 0 |] a in
  check_tensor "sum axis 0"
    (tensor_from_list Nx_core.Dtype.Float32 [ 5.0; 7.0; 9.0 ])
    sum0;

  (* Sum along axis 1 *)
  let sum1 = Nx.sum ~axes:[| 1 |] a in
  check_tensor "sum axis 1"
    (tensor_from_list Nx_core.Dtype.Float32 [ 6.0; 15.0 ])
    sum1;

  (* Mean *)
  let mean_all = Nx.mean a in
  check (float 0.001) "mean all" 3.5 (get_scalar (Nx.get [] mean_all));

  (* Max *)
  let max_all = Nx.max a in
  check (float 0.001) "max all" 6.0 (get_scalar (Nx.get [] max_all))

(* Test mathematical functions *)
let test_math_functions () =
  let a = tensor_from_list Nx_core.Dtype.Float32 [ 1.0; 4.0; 9.0; 16.0 ] in

  (* Sqrt *)
  let sqrt_a = Nx.sqrt a in
  check_tensor "sqrt"
    (tensor_from_list Nx_core.Dtype.Float32 [ 1.0; 2.0; 3.0; 4.0 ])
    sqrt_a;

  (* Square *)
  let b = tensor_from_list Nx_core.Dtype.Float32 [ 1.0; 2.0; 3.0; 4.0 ] in
  let square_b = Nx.square b in
  check_tensor "square"
    (tensor_from_list Nx_core.Dtype.Float32 [ 1.0; 4.0; 9.0; 16.0 ])
    square_b;

  (* Exp and log *)
  let c = tensor_from_list Nx_core.Dtype.Float32 [ 0.0; 1.0; 2.0 ] in
  let exp_c = Nx.exp c in
  let log_exp_c = Nx.log exp_c in
  check_tensor ~eps:1e-4 "log(exp(x))" c log_exp_c

(* Test comparison operations *)
let test_comparisons () =
  let a = tensor_from_list Nx_core.Dtype.Float32 [ 1.0; 2.0; 3.0; 4.0 ] in
  let b = tensor_from_list Nx_core.Dtype.Float32 [ 3.0; 2.0; 1.0; 4.0 ] in

  (* Less than *)
  let lt = Nx.less a b in
  check int "lt[0]" 1 (get_scalar (Nx.get [ 0 ] lt));
  (* 1 < 3 *)
  check int "lt[1]" 0 (get_scalar (Nx.get [ 1 ] lt));
  (* 2 < 2 *)
  check int "lt[2]" 0 (get_scalar (Nx.get [ 2 ] lt));
  (* 3 < 1 *)
  check int "lt[3]" 0 (get_scalar (Nx.get [ 3 ] lt));

  (* 4 < 4 *)

  (* Equal *)
  let eq = Nx.equal a b in
  check int "eq[0]" 0 (get_scalar (Nx.get [ 0 ] eq));
  check int "eq[1]" 1 (get_scalar (Nx.get [ 1 ] eq));
  check int "eq[3]" 1 (get_scalar (Nx.get [ 3 ] eq))

(* Test where operation *)
let test_where () =
  let cond = tensor_from_list Nx_core.Dtype.UInt8 [ 1; 0; 1; 0 ] in
  let a = tensor_from_list Nx_core.Dtype.Float32 [ 10.0; 20.0; 30.0; 40.0 ] in
  let b =
    tensor_from_list Nx_core.Dtype.Float32 [ 100.0; 200.0; 300.0; 400.0 ]
  in

  let result = Nx.where cond a b in
  check_tensor "where"
    (tensor_from_list Nx_core.Dtype.Float32 [ 10.0; 200.0; 30.0; 400.0 ])
    result

(* Test reshape and transpose *)
let test_shape_manipulation () =
  let a =
    tensor_from_list Nx_core.Dtype.Float32 [ 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 ]
  in

  (* Reshape *)
  let a_2x3 = Nx.reshape [| 2; 3 |] a in
  check (array int) "reshape shape" [| 2; 3 |] (Nx.shape a_2x3);
  check (float 0.001) "reshape[0,0]" 1.0 (get_scalar (Nx.get [ 0; 0 ] a_2x3));
  check (float 0.001) "reshape[1,2]" 6.0 (get_scalar (Nx.get [ 1; 2 ] a_2x3));

  (* Transpose *)
  let a_t = Nx.transpose a_2x3 in
  check (array int) "transpose shape" [| 3; 2 |] (Nx.shape a_t);
  check (float 0.001) "transpose[0,0]" 1.0 (get_scalar (Nx.get [ 0; 0 ] a_t));
  check (float 0.001) "transpose[0,1]" 4.0 (get_scalar (Nx.get [ 0; 1 ] a_t));
  check (float 0.001) "transpose[2,0]" 3.0 (get_scalar (Nx.get [ 2; 0 ] a_t))

(* Test concatenation *)
let test_concatenation () =
  let a =
    Nx.reshape [| 2; 2 |]
      (tensor_from_list Nx_core.Dtype.Float32 [ 1.0; 2.0; 3.0; 4.0 ])
  in
  let b =
    Nx.reshape [| 2; 2 |]
      (tensor_from_list Nx_core.Dtype.Float32 [ 5.0; 6.0; 7.0; 8.0 ])
  in

  (* Concatenate along axis 0 *)
  let cat0 = Nx.concatenate [ a; b ] ~axis:0 in
  check (array int) "cat0 shape" [| 4; 2 |] (Nx.shape cat0);
  check (float 0.001) "cat0[0,0]" 1.0 (get_scalar (Nx.get [ 0; 0 ] cat0));
  check (float 0.001) "cat0[2,0]" 5.0 (get_scalar (Nx.get [ 2; 0 ] cat0));

  (* Concatenate along axis 1 *)
  let cat1 = Nx.concatenate [ a; b ] ~axis:1 in
  check (array int) "cat1 shape" [| 2; 4 |] (Nx.shape cat1);
  check (float 0.001) "cat1[0,0]" 1.0 (get_scalar (Nx.get [ 0; 0 ] cat1));
  check (float 0.001) "cat1[0,2]" 5.0 (get_scalar (Nx.get [ 0; 2 ] cat1))

(* Test suite *)
let () =
  run "Nx_metal_frontend"
    [
      ("creation", [ test_case "basic" `Quick test_creation ]);
      ( "arithmetic",
        [
          test_case "basic" `Quick test_arithmetic;
          test_case "broadcasting" `Quick test_broadcasting;
        ] );
      ("reductions", [ test_case "basic" `Quick test_reductions ]);
      ("math", [ test_case "functions" `Quick test_math_functions ]);
      ("comparison", [ test_case "basic" `Quick test_comparisons ]);
      ("selection", [ test_case "where" `Quick test_where ]);
      ( "shape",
        [
          test_case "manipulation" `Quick test_shape_manipulation;
          test_case "concatenation" `Quick test_concatenation;
        ] );
    ]
