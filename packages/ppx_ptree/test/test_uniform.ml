(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Uniform mode and mirror mode of [@@deriving ptree], through the ppx. *)

open Windtrap

(* ——— uniform declarations ——— *)

module Params = struct
  type 'a layer = { lw : 'a; lb : 'a option }

  and 'a t = {
    layers : 'a layer list;
    head : 'a layer;
    pair : 'a * 'a;
    tag : string;
  }
  [@@deriving ptree]
end

module Params_as_uniform : Nx.Ptree.Uniform with type 'a t = 'a Params.t =
  Params

let params =
  {
    Params.layers =
      [ { Params.lw = 1; lb = Some 2 }; { Params.lw = 3; lb = None } ];
    head = { Params.lw = 4; lb = Some 5 };
    pair = (6, 7);
    tag = "model";
  }

let leaf_paths =
  [
    "layers.0.lw";
    "layers.0.lb";
    "layers.1.lw";
    "head.lw";
    "head.lb";
    "pair.0";
    "pair.1";
  ]

let test_map () =
  let strings = Params.map string_of_int params in
  equal ~msg:"payloads change type" string "1" (List.hd strings.layers).lw;
  equal ~msg:"option payloads map" (option string) (Some "2")
    (List.hd strings.layers).lb;
  equal ~msg:"absent payloads stay absent" (option string) None
    (List.nth strings.layers 1).lb;
  equal ~msg:"tuple payloads map" (pair string string) ("6", "7") strings.pair;
  equal ~msg:"static fields are preserved" string "model" strings.tag

let test_map2_heterogeneous () =
  let strings = Params.map string_of_int params in
  let zipped = Params.map2 (fun n s -> (n, s)) params strings in
  equal ~msg:"map2 zips trees of different payload types" (pair int string)
    (4, "4") zipped.head.lw;
  equal ~msg:"map2 keeps the left static fields" string "model" zipped.tag

let test_map2_mismatch () =
  let shorter = { params with Params.layers = [] } in
  raises_invalid_arg "Test_uniform.Params.map2: list length mismatch at layers"
    (fun () -> Stdlib.ignore (Params.map2 (fun a _ -> a) params shorter));
  let absent = { params with Params.head = { Params.lw = 4; lb = None } } in
  raises_invalid_arg
    "Test_uniform.Params.map2_layer: option constructor mismatch at lb"
    (fun () -> Stdlib.ignore (Params.map2 (fun a _ -> a) params absent))

let test_fold2_mismatch () =
  let shorter = { params with Params.layers = [] } in
  raises_invalid_arg "Test_uniform.Params.fold2: list length mismatch at layers"
    (fun () ->
      Stdlib.ignore (Params.fold2 (fun _ acc _ _ -> acc) 0 params shorter))

let test_iter () =
  let visited = ref [] in
  Params.iter (fun n -> visited := n :: !visited) params;
  equal ~msg:"iter visits every payload in declaration order" (list int)
    [ 1; 2; 3; 4; 5; 6; 7 ] (List.rev !visited)

let test_fold_paths () =
  let folded =
    Params.fold (fun path acc n -> (path, n) :: acc) [] params |> List.rev
  in
  equal ~msg:"fold threads dot-joined paths in traversal order"
    (list (pair string int))
    (List.combine leaf_paths [ 1; 2; 3; 4; 5; 6; 7 ])
    folded

let test_fold2 () =
  let doubled = Params.map (fun n -> n * 10) params in
  let folded =
    Params.fold2 (fun path acc a b -> (path, a, b) :: acc) [] params doubled
    |> List.rev
  in
  equal ~msg:"fold2 pairs corresponding payloads with paths"
    (list (triple string int int))
    (List.map
       (fun (p, n) -> (p, n, n * 10))
       (List.combine leaf_paths [ 1; 2; 3; 4; 5; 6; 7 ]))
    folded

let test_names () =
  let names = Params.names params in
  equal ~msg:"names mirrors the value's structure" (option string)
    (Some "head.lb") names.head.lb;
  equal ~msg:"names copies static fields" string "model" names.tag;
  let collected =
    Params.fold (fun _ acc path -> path :: acc) [] names |> List.rev
  in
  equal ~msg:"names agrees with fold's paths" (list string) leaf_paths collected

(* ——— delegates whose payload sits at their own root ——— *)

module Nested = struct
  type 'a leaf = 'a

  and 'a t = { root : 'a leaf; items : 'a leaf list; grid : 'a leaf array }
  [@@deriving ptree]
end

let nested = { Nested.root = 1; items = [ 2; 3 ]; grid = [| 4; 5 |] }
let nested_paths = [ "root"; "items.0"; "items.1"; "grid.0"; "grid.1" ]

let test_delegate_root_paths () =
  let folded =
    Nested.fold (fun path acc n -> (path, n) :: acc) [] nested |> List.rev
  in
  equal ~msg:"a delegate's root payload contributes no segment"
    (list (pair string int))
    (List.combine nested_paths [ 1; 2; 3; 4; 5 ])
    folded;
  let zipped =
    Nested.fold2 (fun path acc _ _ -> path :: acc) [] nested nested |> List.rev
  in
  equal ~msg:"fold2 agrees with fold" (list string) nested_paths zipped;
  let names = Nested.names nested in
  equal ~msg:"names agrees with fold" (list string) nested_paths
    (Nested.fold (fun _ acc path -> path :: acc) [] names |> List.rev)

(* ——— nested containers ——— *)

module Deep = struct
  type 'a t = { rows : 'a list array; cells : ('a * 'a option) list }
  [@@deriving ptree]
end

let deep =
  { Deep.rows = [| [ 1; 2 ]; [ 3 ] |]; cells = [ (4, Some 5); (6, None) ] }

let deep_paths =
  [ "rows.0.0"; "rows.0.1"; "rows.1.0"; "cells.0.0"; "cells.0.1"; "cells.1.0" ]

let test_nested_container_paths () =
  let folded =
    Deep.fold (fun path acc n -> (path, n) :: acc) [] deep |> List.rev
  in
  equal ~msg:"indices nest through arrays and lists"
    (list (pair string int))
    (List.combine deep_paths [ 1; 2; 3; 4; 5; 6 ])
    folded;
  let zipped =
    Deep.fold2 (fun path acc _ _ -> path :: acc) [] deep deep |> List.rev
  in
  equal ~msg:"fold2 agrees with fold" (list string) deep_paths zipped;
  equal ~msg:"names agrees with fold" (list string) deep_paths
    (Deep.fold (fun _ acc path -> path :: acc) [] (Deep.names deep) |> List.rev)

let test_nested_container_mismatch () =
  let shorter = { deep with Deep.rows = [| [ 1 ]; [ 3 ] |] } in
  raises_invalid_arg "Test_uniform.Deep.fold2: list length mismatch at rows.0"
    (fun () -> Stdlib.ignore (Deep.fold2 (fun _ acc _ _ -> acc) 0 deep shorter))

(* ——— Ptree.S bridge and differentiation ——— *)

module Lin = struct
  type 'a t = { w : 'a; b : 'a } [@@deriving ptree]
end

module P = Nx.Ptree.Make (Lin)

let vec32 values = Nx.create Nx.float32 [| Array.length values |] values
let raw tensor = Nx.to_array (Nx.reshape [| -1 |] (Nx.contiguous tensor))

let check_arr ~msg expected tensor =
  let actual = raw tensor in
  equal ~msg int (Array.length expected) (Array.length actual);
  Array.iteri
    (fun i x ->
      equal ~msg:(Printf.sprintf "%s[%d]" msg i) (float 1e-5) x actual.(i))
    expected

let test_grad_through_bridge () =
  let p : P.t =
    {
      Lin.w = Nx.Ptree.P (vec32 [| 1.0; -2.0; 3.0 |]);
      b = Nx.Ptree.P (vec32 [| 0.5 |]);
    }
  in
  let loss (t : P.t) =
    let w = Nx.Ptree.unpack Nx.float32 t.Lin.w in
    let b = Nx.Ptree.unpack Nx.float32 t.Lin.b in
    Nx.add (Nx.sum (Nx.mul w w)) (Nx.mul_s (Nx.sum b) 3.0)
  in
  let g = Rune.grad (module P) loss p in
  check_arr ~msg:"dw" [| 2.0; -4.0; 6.0 |] (Nx.Ptree.unpack Nx.float32 g.Lin.w);
  check_arr ~msg:"db" [| 3.0 |] (Nx.Ptree.unpack Nx.float32 g.Lin.b)

(* ——— mirror mode ——— *)

module Dense = struct
  type t = {
    w : Nx.float32_t;
    b : Nx.float32_t option;
    dim : int; [@ptree.ignore]
  }
  [@@deriving ptree ~mirror]
end

let dense =
  { Dense.w = vec32 [| 1.0; 2.0 |]; b = Some (vec32 [| 3.0 |]); dim = 2 }

let test_mirror_paths () =
  let paths =
    Dense.Uniform.fold
      (fun path acc _ -> path :: acc)
      [] (Dense.to_uniform dense)
    |> List.rev
  in
  equal ~msg:"the mirror folds over packed leaves with paths" (list string)
    [ "w"; "b" ] paths

let test_mirror_round_trip () =
  let restored = Dense.of_uniform (Dense.to_uniform dense) in
  check_arr ~msg:"w survives the round trip" [| 1.0; 2.0 |] restored.Dense.w;
  check_arr ~msg:"b survives the round trip" [| 3.0 |]
    (Option.get restored.Dense.b);
  equal ~msg:"static fields survive the round trip" int 2 restored.Dense.dim

let test_mirror_metadata_zip () =
  let scales = { Dense.Uniform.w = 2.0; b = Some 0.5; dim = 2 } in
  let scaled =
    Dense.Uniform.map2
      (fun (Nx.Ptree.P x) scale -> float_of_int (Nx.numel x) *. scale)
      (Dense.to_uniform dense) scales
  in
  equal ~msg:"map2 zips the mirror against metadata" (float 1e-9) 4.0
    scaled.Dense.Uniform.w;
  equal ~msg:"optional metadata zips leafwise"
    (option (float 1e-9))
    (Some 0.5) scaled.Dense.Uniform.b

let test_mirror_dtype_mismatch () =
  let forged =
    let u = Dense.to_uniform dense in
    { u with Dense.Uniform.w = Nx.Ptree.P (Nx.zeros Nx.float64 [| 2 |]) }
  in
  raises_invalid_arg "Ptree.unpack: dtype mismatch at w" (fun () ->
      Stdlib.ignore (Dense.of_uniform forged))

(* ——— mirror over containers ——— *)

module Stack = struct
  type t = { layers : Nx.float32_t list; grid : Nx.float32_t array }
  [@@deriving ptree ~mirror]
end

let stack =
  {
    Stack.layers = [ vec32 [| 1.0 |]; vec32 [| 2.0 |] ];
    grid = [| vec32 [| 3.0 |] |];
  }

let test_mirror_container_paths () =
  let paths =
    Stack.Uniform.fold
      (fun path acc _ -> path :: acc)
      [] (Stack.to_uniform stack)
    |> List.rev
  in
  equal ~msg:"container elements carry their index" (list string)
    [ "layers.0"; "layers.1"; "grid.0" ]
    paths

let test_mirror_container_dtype_mismatch () =
  let forged =
    let u = Stack.to_uniform stack in
    {
      u with
      Stack.Uniform.layers =
        [
          List.hd u.Stack.Uniform.layers;
          Nx.Ptree.P (Nx.zeros Nx.float64 [| 1 |]);
        ];
    }
  in
  raises_invalid_arg "Ptree.unpack: dtype mismatch at layers.1" (fun () ->
      Stdlib.ignore (Stack.of_uniform forged))

let tests =
  [
    group "uniform traversals"
      [
        test "map changes the payload type" test_map;
        test "map2 zips heterogeneous trees" test_map2_heterogeneous;
        test "map2 rejects structural mismatches" test_map2_mismatch;
        test "iter visits payloads in order" test_iter;
      ];
    group "uniform paths"
      [
        test "fold threads leaf paths" test_fold_paths;
        test "fold2 pairs payloads with paths" test_fold2;
        test "fold2 rejects structural mismatches" test_fold2_mismatch;
        test "names is the tree of paths" test_names;
        test "delegates keep the parent path intact" test_delegate_root_paths;
        test "nested containers nest their indices" test_nested_container_paths;
        test "a nested mismatch names its position"
          test_nested_container_mismatch;
      ];
    group "Ptree.S bridge"
      [ test "grad through Nx.Ptree.Make" test_grad_through_bridge ];
    group "mirror"
      [
        test "fold over the packed mirror" test_mirror_paths;
        test "to_uniform/of_uniform round trip" test_mirror_round_trip;
        test "map2 against a metadata tree" test_mirror_metadata_zip;
        test "of_uniform reports dtype mismatches with paths"
          test_mirror_dtype_mismatch;
        test "container elements fold with their index"
          test_mirror_container_paths;
        test "of_uniform locates a mismatch inside a container"
          test_mirror_container_dtype_mismatch;
      ];
  ]

let () = run "ppx_ptree uniform" tests
