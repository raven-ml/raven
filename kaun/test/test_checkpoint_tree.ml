open Alcotest
module CT = Kaun.Checkpoint.Snapshot
module Json = Yojson.Basic

let mk_tensor = Rune.create Rune.float32 [| 2; 2 |] [| 0.; 1.; 2.; 3. |]

let test_constructors () =
  let tree =
    CT.record_of
      [
        ("params", CT.tensor mk_tensor);
        ("step", CT.scalar_int 3);
        ("flags", CT.list_of [ CT.scalar_bool true; CT.scalar_bool false ]);
        ( "nested",
          CT.record_of
            [
              ("lr", CT.scalar_float 1e-3);
              ("note", CT.scalar_string "checkpoint");
            ] );
      ]
  in
  check bool "is record" true (Option.is_some (CT.to_record tree));
  check bool "has tensor" true (CT.is_tensor (CT.tensor mk_tensor));
  check bool "has scalar" true (CT.is_scalar (CT.scalar_int 1));
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
    CT.scalar_json
      (`Assoc [ ("dataset", `String "CartPole"); ("epochs", `Int 10) ])
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
