(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Quill

let accessor_tests =
  [
    test "empty doc" (fun () ->
        let d = Doc.empty () in
        equal int 0 (Doc.length d);
        equal int 0 (List.length (Doc.cells d)));
    test "of_cells" (fun () ->
        let c1 = Cell.text "a" in
        let c2 = Cell.code "b" in
        let d = Doc.of_cells [ c1; c2 ] in
        equal int 2 (Doc.length d));
    test "nth" (fun () ->
        let c1 = Cell.text "first" in
        let c2 = Cell.text "second" in
        let d = Doc.of_cells [ c1; c2 ] in
        (match Doc.nth 0 d with
        | Some c -> equal string "first" (Cell.source c)
        | None -> fail "expected Some for nth 0");
        (match Doc.nth 1 d with
        | Some c -> equal string "second" (Cell.source c)
        | None -> fail "expected Some for nth 1");
        is_none (Doc.nth 2 d));
    test "find" (fun () ->
        let c1 = Cell.text "hello" in
        let id = Cell.id c1 in
        let d = Doc.of_cells [ c1 ] in
        match Doc.find id d with
        | Some c -> equal string "hello" (Cell.source c)
        | None -> fail "expected Some");
    test "find_index" (fun () ->
        let c1 = Cell.text "a" in
        let c2 = Cell.text "b" in
        let d = Doc.of_cells [ c1; c2 ] in
        some int 1 (Doc.find_index (Cell.id c2) d));
  ]

let modification_tests =
  [
    test "insert at beginning" (fun () ->
        let c1 = Cell.text "existing" in
        let c2 = Cell.text "new" in
        let d = Doc.of_cells [ c1 ] |> Doc.insert ~pos:0 c2 in
        equal int 2 (Doc.length d);
        match Doc.nth 0 d with
        | Some c -> equal string "new" (Cell.source c)
        | None -> fail "expected Some");
    test "insert at end" (fun () ->
        let c1 = Cell.text "first" in
        let c2 = Cell.text "last" in
        let d = Doc.of_cells [ c1 ] |> Doc.insert ~pos:1 c2 in
        match Doc.nth 1 d with
        | Some c -> equal string "last" (Cell.source c)
        | None -> fail "expected Some");
    test "remove" (fun () ->
        let c1 = Cell.text "keep" in
        let c2 = Cell.text "remove" in
        let d = Doc.of_cells [ c1; c2 ] |> Doc.remove (Cell.id c2) in
        equal int 1 (Doc.length d));
    test "replace" (fun () ->
        let c1 = Cell.text "old" in
        let c2 = Cell.text "new" in
        let d = Doc.of_cells [ c1 ] |> Doc.replace (Cell.id c1) c2 in
        match Doc.nth 0 d with
        | Some c -> equal string "new" (Cell.source c)
        | None -> fail "expected Some");
    test "move" (fun () ->
        let c1 = Cell.text "a" in
        let c2 = Cell.text "b" in
        let c3 = Cell.text "c" in
        let d = Doc.of_cells [ c1; c2; c3 ] |> Doc.move (Cell.id c3) ~pos:0 in
        match Doc.nth 0 d with
        | Some c -> equal string "c" (Cell.source c)
        | None -> fail "expected Some");
    test "update" (fun () ->
        let c = Cell.text "old" in
        let d =
          Doc.of_cells [ c ] |> Doc.update (Cell.id c) (Cell.set_source "new")
        in
        match Doc.nth 0 d with
        | Some c -> equal string "new" (Cell.source c)
        | None -> fail "expected Some");
    test "clear_all_outputs" (fun () ->
        let c = Cell.code "x" |> Cell.set_outputs [ Cell.Stdout "out" ] in
        let d = Doc.of_cells [ c ] |> Doc.clear_all_outputs in
        match Doc.nth 0 d with
        | Some (Cell.Code { outputs; _ }) -> equal int 0 (List.length outputs)
        | _ -> fail "expected Code cell");
  ]

let () =
  run "Doc"
    [
      group "Accessors" accessor_tests; group "Modifications" modification_tests;
    ]
