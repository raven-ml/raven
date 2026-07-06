(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The Ptree instance: traversal semantics, structural errors, and
   differentiation through a dynamically-typed tree. *)

open Windtrap
open Rune_test_support.Support
module P = Rune.Ptree

let tree () =
  P.dict
    [
      ("w", P.tensor (vec32 [| 1.0; -2.0; 3.0 |]));
      ("b", P.tensor (vec32 [| 0.5 |]));
    ]

let test_map_preserves_structure () =
  match P.map (fun t -> Nx.mul t t) (tree ()) with
  | P.Dict [ ("w", P.Tensor (P.P w)); ("b", P.Tensor (P.P b)) ] ->
      check_arr ~msg:"w" [| 1.0; 4.0; 9.0 |] (as_f32 w);
      check_arr ~msg:"b" [| 0.25 |] (as_f32 b)
  | _ -> fail "map changed the tree structure"

let test_map2_combines_leafwise () =
  match P.map2 (fun a b -> Nx.add a b) (tree ()) (tree ()) with
  | P.Dict [ ("w", P.Tensor (P.P w)); _ ] ->
      check_arr ~msg:"w" [| 2.0; -4.0; 6.0 |] (as_f32 w)
  | _ -> fail "map2 changed the tree structure"

let test_iter_visits_every_leaf () =
  let count = ref 0 in
  P.iter (fun _ -> incr count) (P.list [ tree (); tree () ]);
  equal ~msg:"leaves" int 4 !count

let test_map2_structure_mismatch () =
  let a = P.list [ P.tensor (vec32 [| 1.0 |]) ] in
  let b = P.dict [ ("x", P.tensor (vec32 [| 1.0 |])) ] in
  raises_invalid_arg (fun () -> ignore (P.map2 (fun x _ -> x) a b))

let test_map2_dict_key_mismatch () =
  let a = P.dict [ ("x", P.tensor (vec32 [| 1.0 |])) ] in
  let b = P.dict [ ("y", P.tensor (vec32 [| 1.0 |])) ] in
  raises_invalid_arg (fun () -> ignore (P.map2 (fun x _ -> x) a b))

let test_map2_dtype_mismatch () =
  let a = P.tensor (vec32 [| 1.0 |]) in
  let b = P.tensor (vec64 [| 1.0 |]) in
  raises_invalid_arg (fun () -> ignore (P.map2 (fun x _ -> x) a b))

let test_grad_over_ptree () =
  let f tree =
    match tree with
    | P.Dict [ ("w", P.Tensor (P.P w)); ("b", P.Tensor (P.P b)) ] ->
        let w = as_f32 w and b = as_f32 b in
        Nx.add (Nx.sum (Nx.mul w w)) (Nx.mul_s (Nx.sum b) 3.0)
    | _ -> assert false
  in
  let g = Rune.grad (module P) f (tree ()) in
  match g with
  | P.Dict [ ("w", P.Tensor (P.P gw)); ("b", P.Tensor (P.P gb)) ] ->
      check_arr ~msg:"dw" [| 2.0; -4.0; 6.0 |] (as_f32 gw);
      check_arr ~msg:"db" [| 3.0 |] (as_f32 gb)
  | _ -> fail "gradient tree does not match parameter tree structure"

let tests =
  [
    group "traversals"
      [
        test "map preserves structure" test_map_preserves_structure;
        test "map2 combines leafwise" test_map2_combines_leafwise;
        test "iter visits every leaf" test_iter_visits_every_leaf;
      ];
    group "structural errors"
      [
        test "map2 rejects structure mismatch" test_map2_structure_mismatch;
        test "map2 rejects dict key mismatch" test_map2_dict_key_mismatch;
        test "map2 rejects dtype mismatch" test_map2_dtype_mismatch;
      ];
    group "differentiation"
      [ test "grad works over a ptree" test_grad_over_ptree ];
  ]

let () = run "rune ptree" tests
