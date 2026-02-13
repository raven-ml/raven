(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
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
  equal ~msg:"is record" bool true (Option.is_some (CT.get_record tree));
  equal ~msg:"has tensor" bool true (CT.is_tensor (CT.tensor mk_tensor));
  equal ~msg:"has scalar" bool true (CT.is_scalar (CT.int 1));
  let tensors = CT.flatten_tensors tree in
  let tensor_paths = List.map fst tensors in
  equal ~msg:"tensor paths" (list string) [ "params" ] tensor_paths;
  let scalars = CT.flatten_scalars tree in
  let scalar_paths = List.map fst scalars |> List.sort String.compare in
  equal ~msg:"scalar paths" (list string)
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
          equal ~msg:"json preserved" string (Json.to_string orig)
            (Json.to_string payload)
      | _ -> fail "expected json scalar")

let tests =
  [
    test "constructors and flatten" test_constructors;
    test "json roundtrip" test_json_roundtrip;
  ]

let () = run "Snapshot" [ group "Structure" tests ]
