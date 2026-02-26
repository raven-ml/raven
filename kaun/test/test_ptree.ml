(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
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

let raises_invalid_arg_any f =
  raises_match
    (fun exn -> match exn with Invalid_argument _ -> true | _ -> false)
    f

let raises_invalid_arg_contains needle f =
  raises_match
    (fun exn ->
      match exn with
      | Invalid_argument msg -> string_contains msg needle
      | _ -> false)
    f

let f32_leaf v = Ptree.tensor (Rune.full Rune.float32 [| 1 |] v)

let f32_value_of_tensor p =
  let t = Ptree.Tensor.to_typed_exn Rune.float32 p in
  Rune.item [ 0 ] t

let f32_value_of_tree t =
  let p = Ptree.as_tensor_exn t in
  f32_value_of_tensor p

let collect_f32_values t =
  let values = ref [] in
  Ptree.iter
    (fun p ->
      let v =
        Ptree.with_tensor p
          {
            run =
              (fun (type a) (type layout) (x : (a, layout) Rune.t) ->
                let y = Rune.cast Rune.float32 x in
                Rune.item [ 0 ] (Rune.reshape [| -1 |] y));
          }
      in
      values := v :: !values)
    t;
  List.rev !values

let test_dict_key_validation () =
  raises_invalid_arg "duplicate key \"w\"" (fun () ->
      Ptree.dict [ ("w", f32_leaf 1.0); ("w", f32_leaf 2.0) ]);
  raises_invalid_arg "empty key" (fun () -> Ptree.dict [ ("", f32_leaf 1.0) ]);
  raises_invalid_arg_contains "reserved character '.'" (fun () ->
      Ptree.dict [ ("a.b", f32_leaf 1.0) ]);
  raises_invalid_arg_contains "reserved character '['" (fun () ->
      Ptree.dict [ ("a[0]", f32_leaf 1.0) ]);
  ignore
    (Ptree.dict
       [
         ("weight", f32_leaf 1.0);
         ("bias", f32_leaf 2.0);
         ("layer_1", f32_leaf 3.0);
       ])

let test_tensor_module () =
  let p = Ptree.P (Rune.zeros Rune.float32 [| 2; 3 |]) in
  let dtype_matches =
    match Ptree.Tensor.dtype p with
    | Nx_core.Dtype.Pack dt -> Nx_core.Dtype.equal dt Rune.float32
  in
  equal ~msg:"dtype" bool true dtype_matches;
  equal ~msg:"shape" (list int) [ 2; 3 ] (Array.to_list (Ptree.Tensor.shape p));
  equal ~msg:"numel" int 6 (Ptree.Tensor.numel p);
  equal ~msg:"to_typed hit" bool true
    (Option.is_some (Ptree.Tensor.to_typed Rune.float32 p));
  equal ~msg:"to_typed miss" bool true
    (Option.is_none (Ptree.Tensor.to_typed Rune.float64 p));
  raises_invalid_arg_contains "dtype mismatch" (fun () ->
      Ptree.Tensor.to_typed_exn Rune.float64 p)

let test_leaf_access () =
  let p = Ptree.P (Rune.full Rune.float32 [| 1 |] 7.0) in
  let v =
    Ptree.with_tensor p
      {
        run =
          (fun (type a) (type layout) (t : (a, layout) Rune.t) ->
            let t = Rune.cast Rune.float32 t in
            Rune.item [ 0 ] t);
      }
  in
  equal ~msg:"with_tensor" (float 1e-6) 7.0 v;
  equal ~msg:"as_tensor_exn" (float 1e-6) 7.0
    (f32_value_of_tree (Ptree.tensor (Rune.full Rune.float32 [| 1 |] 7.0)));
  raises_invalid_arg_contains "ctx" (fun () ->
      Ptree.as_tensor_exn ~ctx:"ctx" (Ptree.list []))

let test_dict_access () =
  let fields =
    Ptree.Dict.fields_exn
      (Ptree.dict [ ("w", f32_leaf 3.0); ("b", f32_leaf 4.0) ])
  in
  equal ~msg:"find hit" bool true (Option.is_some (Ptree.Dict.find "w" fields));
  equal ~msg:"find miss" bool true (Option.is_none (Ptree.Dict.find "x" fields));
  equal ~msg:"find_exn" (float 1e-6) 4.0
    (f32_value_of_tree (Ptree.Dict.find_exn "b" fields));
  raises_invalid_arg_contains "ctx" (fun () ->
      Ptree.Dict.find_exn ~ctx:"ctx" "x" fields);
  equal ~msg:"get_tensor_exn" (float 1e-6) 3.0
    (Rune.item [ 0 ] (Ptree.Dict.get_tensor_exn fields ~name:"w" Rune.float32));
  raises_invalid_arg_any (fun () ->
      Ptree.Dict.get_tensor_exn fields ~name:"x" Rune.float32);
  raises_invalid_arg_any (fun () ->
      Ptree.Dict.get_tensor_exn fields ~name:"w" Rune.float64);
  raises_invalid_arg_any (fun () ->
      Ptree.Dict.get_tensor_exn
        (Ptree.Dict.fields_exn (Ptree.dict [ ("node", Ptree.list []) ]))
        ~name:"node" Rune.float32);
  raises_invalid_arg_contains "ctx" (fun () ->
      Ptree.Dict.fields_exn ~ctx:"ctx"
        (Ptree.tensor (Rune.zeros Rune.float32 [| 1 |])))

let test_list_access () =
  let items =
    Ptree.List.items_exn
      (Ptree.list [ f32_leaf 1.0; f32_leaf 2.0; Ptree.list [ f32_leaf 3.0 ] ])
  in
  equal ~msg:"items_exn length" int 3 (List.length items);
  raises_invalid_arg_contains "ctx" (fun () ->
      Ptree.List.items_exn ~ctx:"ctx" (f32_leaf 1.0))

let test_map () =
  let tree =
    Ptree.dict
      [ ("a", f32_leaf 1.0); ("b", Ptree.list [ f32_leaf 2.0; f32_leaf 3.0 ]) ]
  in
  let mapped =
    Ptree.map
      {
        run =
          (fun (type a) (type layout) (t : (a, layout) Rune.t) ->
            let dt = Rune.dtype t in
            let ten = Nx_core.Dtype.of_float dt 10.0 in
            Rune.add t (Rune.scalar dt ten));
      }
      tree
  in
  equal ~msg:"map values"
    (list (float 1e-6))
    [ 11.0; 12.0; 13.0 ]
    (collect_f32_values mapped)

let test_map2_success_and_order () =
  let lhs = Ptree.dict [ ("z", f32_leaf 1.0); ("a", f32_leaf 2.0) ] in
  let rhs = Ptree.dict [ ("a", f32_leaf 20.0); ("z", f32_leaf 10.0) ] in
  let out = Ptree.map2 { run = Rune.add } lhs rhs in
  let fields = Ptree.Dict.fields_exn out in
  equal ~msg:"preserve lhs key order" (list string) [ "z"; "a" ]
    (List.map fst fields);
  equal ~msg:"z value" (float 1e-6) 11.0
    (Rune.item [ 0 ] (Ptree.Dict.get_tensor_exn fields ~name:"z" Rune.float32));
  equal ~msg:"a value" (float 1e-6) 22.0
    (Rune.item [ 0 ] (Ptree.Dict.get_tensor_exn fields ~name:"a" Rune.float32))

let test_map2_errors () =
  raises_invalid_arg_contains "structure mismatch" (fun () ->
      Ptree.map2 { run = Rune.add } (f32_leaf 1.0)
        (Ptree.dict [ ("x", f32_leaf 1.0) ]));
  raises_invalid_arg_contains "list length mismatch" (fun () ->
      Ptree.map2 { run = Rune.add }
        (Ptree.list [ f32_leaf 1.0 ])
        (Ptree.list [ f32_leaf 1.0; f32_leaf 2.0 ]));
  raises_invalid_arg_contains "dict size mismatch" (fun () ->
      Ptree.map2 { run = Rune.add }
        (Ptree.dict [ ("a", f32_leaf 1.0) ])
        (Ptree.dict [ ("a", f32_leaf 1.0); ("b", f32_leaf 2.0) ]));
  raises_invalid_arg_contains "not found in second dict" (fun () ->
      Ptree.map2 { run = Rune.add }
        (Ptree.dict [ ("a", f32_leaf 1.0) ])
        (Ptree.dict [ ("b", f32_leaf 1.0) ]));
  raises_invalid_arg_contains "dtype mismatch" (fun () ->
      Ptree.map2 { run = Rune.add }
        (Ptree.tensor (Rune.ones Rune.float32 [| 1 |]))
        (Ptree.tensor (Rune.ones Rune.int32 [| 1 |])))

let test_iter_and_fold_order () =
  let tree =
    Ptree.dict
      [
        ("a", f32_leaf 1.0);
        ("b", Ptree.list [ f32_leaf 2.0; Ptree.dict [ ("c", f32_leaf 3.0) ] ]);
        ("d", f32_leaf 4.0);
      ]
  in
  let iter_values = collect_f32_values tree in
  equal ~msg:"iter order" (list (float 1e-6)) [ 1.0; 2.0; 3.0; 4.0 ] iter_values;
  let fold_values =
    Ptree.fold
      (fun acc p ->
        let v = f32_value_of_tensor p in
        v :: acc)
      [] tree
    |> List.rev
  in
  equal ~msg:"fold order" (list (float 1e-6)) [ 1.0; 2.0; 3.0; 4.0 ] fold_values

let test_flatten_and_rebuild () =
  let tree =
    Ptree.dict
      [ ("a", f32_leaf 1.0); ("b", Ptree.list [ f32_leaf 2.0; f32_leaf 3.0 ]) ]
  in
  let leaves, rebuild = Ptree.flatten tree in
  equal ~msg:"flatten length" int 3 (List.length leaves);
  equal ~msg:"flatten order"
    (list (float 1e-6))
    [ 1.0; 2.0; 3.0 ]
    (List.map f32_value_of_tensor leaves);
  let leaves_plus_10 =
    List.map
      (fun (Ptree.P t) ->
        Ptree.P
          (Rune.add t
             (Rune.scalar (Rune.dtype t)
                (Nx_core.Dtype.of_float (Rune.dtype t) 10.0))))
      leaves
  in
  let rebuilt = rebuild leaves_plus_10 in
  equal ~msg:"rebuild values"
    (list (float 1e-6))
    [ 11.0; 12.0; 13.0 ]
    (collect_f32_values rebuilt);
  let first_leaf =
    match leaves with
    | first :: _ -> first
    | [] -> fail "flatten returned no leaves"
  in
  raises_invalid_arg_contains "not enough tensors" (fun () ->
      rebuild [ first_leaf ]);
  raises_invalid_arg_contains "too many tensors" (fun () ->
      rebuild (leaves @ [ first_leaf ]))

let test_flatten_with_paths () =
  let root = f32_leaf 3.0 in
  equal ~msg:"tensor root path" (list string) [ "" ]
    (List.map fst (Ptree.flatten_with_paths root));
  let tree =
    Ptree.dict
      [
        ("w", f32_leaf 1.0);
        ( "layers",
          Ptree.list [ f32_leaf 2.0; Ptree.dict [ ("b", f32_leaf 3.0) ] ] );
      ]
  in
  equal ~msg:"nested paths" (list string)
    [ "w"; "layers.0"; "layers.1.b" ]
    (List.map fst (Ptree.flatten_with_paths tree))

let test_zeros_like_and_count_parameters () =
  let tree =
    Ptree.dict
      [
        ("w", Ptree.tensor (Rune.ones Rune.float32 [| 2; 3 |]));
        ("b", Ptree.tensor (Rune.full Rune.float32 [| 4 |] 5.0));
      ]
  in
  equal ~msg:"count parameters" int 10 (Ptree.count_parameters tree);
  let zeros = Ptree.zeros_like tree in
  equal ~msg:"count preserved" int 10 (Ptree.count_parameters zeros);
  Ptree.iter
    (fun p ->
      Ptree.with_tensor p
        {
          run =
            (fun (type a) (type layout) (x : (a, layout) Rune.t) ->
              let y = Rune.cast Rune.float32 x in
              let flat = Rune.reshape [| -1 |] y in
              let n = Rune.numel flat in
              for i = 0 to n - 1 do
                equal ~msg:"zeros_like values" (float 1e-6) 0.0
                  (Rune.item [ i ] flat)
              done);
        })
    zeros

let test_pp () =
  let tree =
    Ptree.dict
      [
        ("w", Ptree.tensor (Rune.ones Rune.float32 [| 2 |]));
        ("b", Ptree.list [ f32_leaf 1.0 ]);
      ]
  in
  let s = Format.asprintf "%a" Ptree.pp tree in
  equal ~msg:"pp has Dict" bool true
    (String.length s >= String.length "Dict" && String.sub s 0 4 = "Dict");
  equal ~msg:"pp has key" bool true (string_contains s "w");
  equal ~msg:"pp non-empty" bool true (String.length s > 0)

let () =
  run "Kaun.Ptree"
    [
      group "constructors"
        [
          test "dict key validation" test_dict_key_validation;
          test "leaf access" test_leaf_access;
        ];
      group "tensor" [ test "Tensor module" test_tensor_module ];
      group "containers"
        [
          test "dict access" test_dict_access;
          test "list access" test_list_access;
        ];
      group "functional"
        [
          test "map" test_map;
          test "map2 success and order" test_map2_success_and_order;
          test "map2 errors" test_map2_errors;
          test "iter/fold traversal order" test_iter_and_fold_order;
        ];
      group "flatten"
        [
          test "flatten/rebuild" test_flatten_and_rebuild;
          test "flatten_with_paths" test_flatten_with_paths;
        ];
      group "utilities"
        [
          test "zeros_like/count_parameters"
            test_zeros_like_and_count_parameters;
          test "pp" test_pp;
        ];
    ]
