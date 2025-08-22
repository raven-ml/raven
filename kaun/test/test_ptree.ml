open Alcotest
open Kaun.Ptree

let ctx = Rune.c

let test_map () =
  let tree =
    Kaun.Ptree.Record
      [
        ("layer1", Tensor (Rune.ones ctx Rune.float32 [| 2; 3 |]));
        ( "layer2",
          List
            [
              Tensor (Rune.ones ctx Rune.float32 [| 4 |]);
              Tensor (Rune.ones ctx Rune.float32 [| 5 |]);
            ] );
      ]
  in
  let doubled =
    Kaun.Ptree.map (fun t -> Rune.mul t (Rune.scalar ctx Rune.float32 2.0)) tree
  in

  (* Check that all tensors are doubled *)
  Kaun.Ptree.iter
    (fun t ->
      let first_val = Rune.unsafe_get [ 0 ] (Rune.reshape [| -1 |] t) in
      check (float 0.01) "tensor doubled" 2.0 first_val)
    doubled

let test_map2 () =
  let tree1 = Kaun.Ptree.Tensor (Rune.ones ctx Rune.float32 [| 3 |]) in
  let tree2 = Kaun.Ptree.Tensor (Rune.full ctx Rune.float32 [| 3 |] 2.0) in
  let sum = Kaun.Ptree.add tree1 tree2 in

  match sum with
  | Tensor t ->
      let first_val = Rune.unsafe_get [ 0 ] t in
      check (float 0.01) "1 + 2 = 3" 3.0 first_val
  | _ -> fail "Expected Tensor"

let test_count () =
  let tree =
    Kaun.Ptree.Record
      [
        ("a", Tensor (Rune.zeros ctx Rune.float32 [| 2; 3 |]));
        ("b", Tensor (Rune.zeros ctx Rune.float32 [| 4; 5 |]));
        ( "c",
          List
            [
              Tensor (Rune.zeros ctx Rune.float32 [| 6 |]);
              Tensor (Rune.zeros ctx Rune.float32 [| 7 |]);
            ] );
      ]
  in

  check int "count tensors" 4 (Kaun.Ptree.count_tensors tree);
  check int "count parameters"
    (6 + 20 + 6 + 7)
    (Kaun.Ptree.count_parameters tree)

let test_flat_list () =
  let tree =
    Kaun.Ptree.List
      [
        Tensor (Rune.full ctx Rune.float32 [| 2 |] 1.0);
        Tensor (Rune.full ctx Rune.float32 [| 3 |] 2.0);
      ]
  in

  let flat = Kaun.Ptree.to_flat_list tree in
  check int "flat list length" 2 (List.length flat);

  (* Modify the flat list *)
  let modified =
    List.map (fun t -> Rune.add t (Rune.scalar ctx Rune.float32 10.0)) flat
  in

  (* Reconstruct the tree *)
  let reconstructed = Kaun.Ptree.from_flat_list tree modified in

  match reconstructed with
  | List [ Tensor t1; Tensor t2 ] ->
      check (float 0.01) "first tensor modified" 11.0 (Rune.unsafe_get [ 0 ] t1);
      check (float 0.01) "second tensor modified" 12.0
        (Rune.unsafe_get [ 0 ] t2)
  | _ -> fail "Unexpected structure"

let test_equal_structure () =
  let tree1 =
    Kaun.Ptree.Record
      [ ("a", Tensor (Rune.zeros ctx Rune.float32 [| 2 |])); ("b", List []) ]
  in
  let tree2 =
    Kaun.Ptree.Record
      [ ("a", Tensor (Rune.ones ctx Rune.float32 [| 3 |])); ("b", List []) ]
  in
  let tree3 =
    Kaun.Ptree.Record [ ("a", Tensor (Rune.zeros ctx Rune.float32 [| 2 |])) ]
  in

  check bool "same structure" true (Kaun.Ptree.equal_structure tree1 tree2);
  check bool "different structure" false
    (Kaun.Ptree.equal_structure tree1 tree3)

let test_arithmetic () =
  let a = Kaun.Ptree.Tensor (Rune.full ctx Rune.float32 [| 2 |] 10.0) in
  let b = Kaun.Ptree.Tensor (Rune.full ctx Rune.float32 [| 2 |] 3.0) in

  let test_op name op expected =
    match op a b with
    | Tensor t -> check (float 0.01) name expected (Rune.unsafe_get [ 0 ] t)
    | _ -> fail "Expected Tensor"
  in

  test_op "add" Kaun.Ptree.add 13.0;
  test_op "sub" Kaun.Ptree.sub 7.0;
  test_op "mul" Kaun.Ptree.mul 30.0;
  test_op "div" Kaun.Ptree.div (10.0 /. 3.0);

  match Kaun.Ptree.scale 2.0 a with
  | Tensor t -> check (float 0.01) "scale" 20.0 (Rune.unsafe_get [ 0 ] t)
  | _ -> fail "Expected Tensor"

let () =
  run "Kaun.Ptree"
    [
      ( "operations",
        [
          test_case "map" `Quick test_map;
          test_case "map2" `Quick test_map2;
          test_case "count" `Quick test_count;
          test_case "flat_list" `Quick test_flat_list;
          test_case "equal_structure" `Quick test_equal_structure;
          test_case "arithmetic" `Quick test_arithmetic;
        ] );
    ]
