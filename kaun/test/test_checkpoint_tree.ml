(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Alcotest
module CT = Kaun.Checkpoint.Snapshot
module Json = Yojson.Basic

let mk_tensor = Rune.create Rune.float32 [| 2; 2 |] [| 0.; 1.; 2.; 3. |]

let test_constructors () =
  let tree =
    CT.record
      [
        ("params", CT.tensor mk_tensor);
        ("step", CT.int 3);
        ("flags", CT.list [ CT.bool true; CT.bool false ]);
        ( "nested",
          CT.record [ ("lr", CT.float 1e-3); ("note", CT.string "checkpoint") ]
        );
      ]
  in
  check bool "is record" true (Option.is_some (CT.get_record tree));
  check bool "has tensor" true (CT.is_tensor (CT.tensor mk_tensor));
  check bool "has scalar" true (CT.is_scalar (CT.int 1));
  let tensors = CT.flatten_tensors tree in
  let tensor_paths = List.map fst tensors in
  check (list string) "tensor paths" [ "params" ] tensor_paths;
  let scalars = CT.flatten_scalars tree in
  let scalar_paths = List.map fst scalars |> List.sort String.compare in
  check (list string) "scalar paths"
    [ "flags[0]"; "flags[1]"; "nested.lr"; "nested.note"; "step" ]
    scalar_paths

let test_json_roundtrip () =
  let scalar =
    CT.json (`Assoc [ ("dataset", `String "CartPole"); ("epochs", `Int 10) ])
  in
  match CT.get_scalar scalar with
  | None -> fail "expected scalar"
  | Some s -> (
      (* Roundtrip through the typed JSON representation *)
      let recovered = s |> CT.scalar_to_yojson |> CT.scalar_of_yojson in
      match (s, recovered) with
      | CT.Json orig, CT.Json payload ->
          check string "json preserved" (Json.to_string orig)
            (Json.to_string payload)
      | _ -> fail "expected json scalar")

let suite =
  [
    test_case "constructors and flatten" `Quick test_constructors;
    test_case "json roundtrip" `Quick test_json_roundtrip;
  ]

let () = run "Snapshot" [ ("Structure", suite) ]
