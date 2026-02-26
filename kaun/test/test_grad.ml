(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module Grad = Kaun.Grad
module Ptree = Kaun.Ptree

let string_contains s sub =
  let slen = String.length s in
  let sub_len = String.length sub in
  let rec loop i =
    if i + sub_len > slen then false
    else if String.sub s i sub_len = sub then true
    else loop (i + 1)
  in
  if sub_len = 0 then true else loop 0

let raises_invalid_arg_contains needle f =
  raises_match
    (fun exn ->
      match exn with
      | Invalid_argument msg -> string_contains msg needle
      | _ -> false)
    f

(* f(x) = 0.5 * sum(x^2), gradient = x *)
let test_scalar_quadratic () =
  let x = Rune.create Rune.float32 [| 2 |] [| 3.0; -4.0 |] in
  let params = Ptree.tensor x in
  let loss, grads =
    Grad.value_and_grad
      (fun p ->
        let (Ptree.P t) = Ptree.as_tensor_exn p in
        let t = Ptree.Tensor.to_typed_exn Rune.float32 (Ptree.P t) in
        Rune.mul (Rune.scalar Rune.float32 0.5) (Rune.sum (Rune.mul t t)))
      params
  in
  equal ~msg:"loss value" (float 1e-6) 12.5 (Rune.item [] loss);
  let g = Ptree.Tensor.to_typed_exn Rune.float32 (Ptree.as_tensor_exn grads) in
  equal ~msg:"grad[0]" (float 1e-5) 3.0 (Rune.item [ 0 ] g);
  equal ~msg:"grad[1]" (float 1e-5) (-4.0) (Rune.item [ 1 ] g)

(* f(w, b) = sum(w * x + b), dw = x, db = ones *)
let test_multi_leaf_dict () =
  let w = Rune.create Rune.float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let b = Rune.create Rune.float32 [| 3 |] [| 0.1; 0.2; 0.3 |] in
  let x = Rune.create Rune.float32 [| 3 |] [| 4.0; 5.0; 6.0 |] in
  let params = Ptree.dict [ ("w", Ptree.tensor w); ("b", Ptree.tensor b) ] in
  let loss, grads =
    Grad.value_and_grad
      (fun p ->
        let fields = Ptree.Dict.fields_exn p in
        let w = Ptree.Dict.get_tensor_exn fields ~name:"w" Rune.float32 in
        let b = Ptree.Dict.get_tensor_exn fields ~name:"b" Rune.float32 in
        Rune.sum (Rune.add (Rune.mul w x) b))
      params
  in
  equal ~msg:"loss value" (float 1e-4) 32.6 (Rune.item [] loss);
  let grad_fields = Ptree.Dict.fields_exn grads in
  let gw = Ptree.Dict.get_tensor_exn grad_fields ~name:"w" Rune.float32 in
  let gb = Ptree.Dict.get_tensor_exn grad_fields ~name:"b" Rune.float32 in
  equal ~msg:"dw[0]" (float 1e-5) 4.0 (Rune.item [ 0 ] gw);
  equal ~msg:"dw[1]" (float 1e-5) 5.0 (Rune.item [ 1 ] gw);
  equal ~msg:"dw[2]" (float 1e-5) 6.0 (Rune.item [ 2 ] gw);
  equal ~msg:"db[0]" (float 1e-5) 1.0 (Rune.item [ 0 ] gb);
  equal ~msg:"db[1]" (float 1e-5) 1.0 (Rune.item [ 1 ] gb);
  equal ~msg:"db[2]" (float 1e-5) 1.0 (Rune.item [ 2 ] gb)

let test_nested_tree () =
  let a = Rune.create Rune.float32 [| 2 |] [| 1.0; 2.0 |] in
  let b = Rune.create Rune.float32 [| 2 |] [| 3.0; 4.0 |] in
  let params =
    Ptree.dict
      [
        ("layer1", Ptree.dict [ ("w", Ptree.tensor a) ]);
        ("layer2", Ptree.dict [ ("w", Ptree.tensor b) ]);
      ]
  in
  let _loss, grads =
    Grad.value_and_grad
      (fun p ->
        let f1 = Ptree.Dict.fields_exn p in
        let l1 = Ptree.Dict.fields_exn (Ptree.Dict.find_exn "layer1" f1) in
        let l2 = Ptree.Dict.fields_exn (Ptree.Dict.find_exn "layer2" f1) in
        let w1 = Ptree.Dict.get_tensor_exn l1 ~name:"w" Rune.float32 in
        let w2 = Ptree.Dict.get_tensor_exn l2 ~name:"w" Rune.float32 in
        Rune.add (Rune.sum (Rune.mul w1 w1)) (Rune.sum (Rune.mul w2 w2)))
      params
  in
  let gf = Ptree.Dict.fields_exn grads in
  let gl1 = Ptree.Dict.fields_exn (Ptree.Dict.find_exn "layer1" gf) in
  let gl2 = Ptree.Dict.fields_exn (Ptree.Dict.find_exn "layer2" gf) in
  let ga = Ptree.Dict.get_tensor_exn gl1 ~name:"w" Rune.float32 in
  let gb = Ptree.Dict.get_tensor_exn gl2 ~name:"w" Rune.float32 in
  equal ~msg:"ga[0]" (float 1e-5) 2.0 (Rune.item [ 0 ] ga);
  equal ~msg:"ga[1]" (float 1e-5) 4.0 (Rune.item [ 1 ] ga);
  equal ~msg:"gb[0]" (float 1e-5) 6.0 (Rune.item [ 0 ] gb);
  equal ~msg:"gb[1]" (float 1e-5) 8.0 (Rune.item [ 1 ] gb)

let test_value_and_grad_aux () =
  let x = Rune.create Rune.float32 [| 2 |] [| 3.0; 4.0 |] in
  let params = Ptree.tensor x in
  let loss, grads, aux =
    Grad.value_and_grad_aux
      (fun p ->
        let (Ptree.P t) = Ptree.as_tensor_exn p in
        let t = Ptree.Tensor.to_typed_exn Rune.float32 (Ptree.P t) in
        (Rune.sum (Rune.mul t t), Rune.item [ 0 ] t))
      params
  in
  equal ~msg:"loss value" (float 1e-6) 25.0 (Rune.item [] loss);
  equal ~msg:"aux value" (float 1e-6) 3.0 aux;
  let g = Ptree.Tensor.to_typed_exn Rune.float32 (Ptree.as_tensor_exn grads) in
  equal ~msg:"grad[0]" (float 1e-5) 6.0 (Rune.item [ 0 ] g);
  equal ~msg:"grad[1]" (float 1e-5) 8.0 (Rune.item [ 1 ] g)

let test_grad_convenience () =
  let x = Rune.create Rune.float32 [| 2 |] [| 5.0; -3.0 |] in
  let params = Ptree.tensor x in
  let f p =
    let (Ptree.P t) = Ptree.as_tensor_exn p in
    let t = Ptree.Tensor.to_typed_exn Rune.float32 (Ptree.P t) in
    Rune.sum t
  in
  let grads = Grad.grad f params in
  let g = Ptree.Tensor.to_typed_exn Rune.float32 (Ptree.as_tensor_exn grads) in
  equal ~msg:"grad[0]" (float 1e-5) 1.0 (Rune.item [ 0 ] g);
  equal ~msg:"grad[1]" (float 1e-5) 1.0 (Rune.item [ 1 ] g)

let test_empty_tree () =
  let params = Ptree.list [] in
  let loss, grads =
    Grad.value_and_grad (fun _p -> Rune.scalar Rune.float32 42.0) params
  in
  equal ~msg:"empty tree loss" (float 1e-6) 42.0 (Rune.item [] loss);
  match grads with
  | Ptree.List [] -> ()
  | _ -> fail "expected empty list gradient"

let test_non_float_leaf_error () =
  let params = Ptree.tensor (Rune.zeros Rune.int32 [| 3 |]) in
  raises_invalid_arg_contains "<root> expected float dtype" (fun () ->
      ignore
        (Grad.value_and_grad (fun _p -> Rune.scalar Rune.float32 0.0) params))

let test_mixed_dtype_error () =
  let params =
    Ptree.dict
      [
        ("a", Ptree.tensor (Rune.ones Rune.float16 [| 2 |]));
        ("b", Ptree.tensor (Rune.ones Rune.float32 [| 2 |]));
      ]
  in
  raises_invalid_arg_contains "has dtype/layout" (fun () ->
      ignore
        (Grad.value_and_grad
           (fun p ->
             let fields = Ptree.Dict.fields_exn p in
             let a = Ptree.Dict.get_tensor_exn fields ~name:"a" Rune.float16 in
             let b = Ptree.Dict.get_tensor_exn fields ~name:"b" Rune.float32 in
             Rune.add (Rune.sum (Rune.cast Rune.float32 a)) (Rune.sum b))
           params))

let test_value_and_grad_mixed () =
  let params =
    Ptree.dict
      [
        ("a", Ptree.tensor (Rune.ones Rune.float16 [| 2 |]));
        ("b", Ptree.tensor (Rune.ones Rune.float32 [| 2 |]));
      ]
  in
  let loss, grads =
    Grad.value_and_grad_mixed
      (fun p ->
        let fields = Ptree.Dict.fields_exn p in
        let a = Ptree.Dict.get_tensor_exn fields ~name:"a" Rune.float16 in
        let b = Ptree.Dict.get_tensor_exn fields ~name:"b" Rune.float32 in
        Rune.add (Rune.sum (Rune.cast Rune.float32 a)) (Rune.sum b))
      params
  in
  equal ~msg:"loss value" (float 1e-5) 4.0 (Rune.item [] loss);
  let grad_fields = Ptree.Dict.fields_exn grads in
  let ga = Ptree.Dict.get_tensor_exn grad_fields ~name:"a" Rune.float16 in
  let gb = Ptree.Dict.get_tensor_exn grad_fields ~name:"b" Rune.float32 in
  equal ~msg:"ga[0]" (float 1e-5) 1.0
    (Rune.item [ 0 ] (Rune.cast Rune.float32 ga));
  equal ~msg:"gb[0]" (float 1e-5) 1.0 (Rune.item [ 0 ] gb)

let test_structure_preserved () =
  let params =
    Ptree.list
      [
        Ptree.tensor (Rune.ones Rune.float32 [| 2 |]);
        Ptree.tensor (Rune.ones Rune.float32 [| 3 |]);
      ]
  in
  let grads =
    Grad.grad
      (fun p ->
        let items = Ptree.List.items_exn p in
        let t0 =
          Ptree.Tensor.to_typed_exn Rune.float32
            (Ptree.as_tensor_exn (List.nth items 0))
        in
        let t1 =
          Ptree.Tensor.to_typed_exn Rune.float32
            (Ptree.as_tensor_exn (List.nth items 1))
        in
        Rune.add (Rune.sum t0) (Rune.sum t1))
      params
  in
  let items = Ptree.List.items_exn grads in
  equal ~msg:"list length" int 2 (List.length items);
  let g0 =
    Ptree.Tensor.to_typed_exn Rune.float32
      (Ptree.as_tensor_exn (List.nth items 0))
  in
  let g1 =
    Ptree.Tensor.to_typed_exn Rune.float32
      (Ptree.as_tensor_exn (List.nth items 1))
  in
  equal ~msg:"g0 shape" (list int) [ 2 ] (Array.to_list (Rune.shape g0));
  equal ~msg:"g1 shape" (list int) [ 3 ] (Array.to_list (Rune.shape g1))

let () =
  run "Kaun.Grad"
    [
      group "value_and_grad"
        [
          test "scalar quadratic" test_scalar_quadratic;
          test "multi-leaf dict" test_multi_leaf_dict;
          test "nested tree" test_nested_tree;
          test "structure preserved" test_structure_preserved;
          test "value_and_grad_aux" test_value_and_grad_aux;
          test "value_and_grad_mixed" test_value_and_grad_mixed;
        ];
      group "grad" [ test "grad convenience" test_grad_convenience ];
      group "edge cases"
        [
          test "empty tree" test_empty_tree;
          test "non-float leaf error" test_non_float_leaf_error;
          test "mixed dtype error" test_mixed_dtype_error;
        ];
    ]
