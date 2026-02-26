(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Quill

let constructor_tests =
  [
    test "code cell defaults" (fun () ->
        let c = Cell.code "let x = 1" in
        equal string "let x = 1" (Cell.source c);
        match c with
        | Cell.Code { language; outputs; _ } ->
            equal string "ocaml" language;
            equal int 0 (List.length outputs)
        | _ -> fail "expected Code cell");
    test "code cell with language" (fun () ->
        let c = Cell.code ~language:"python" "print(1)" in
        match c with
        | Cell.Code { language; _ } -> equal string "python" language
        | _ -> fail "expected Code cell");
    test "text cell" (fun () ->
        let c = Cell.text "# Hello" in
        equal string "# Hello" (Cell.source c);
        match c with Cell.Text _ -> () | _ -> fail "expected Text cell");
    test "unique ids" (fun () ->
        let a = Cell.code "a" in
        let b = Cell.code "b" in
        is_true ~msg:"distinct ids" (not (String.equal (Cell.id a) (Cell.id b))));
  ]

let transformation_tests =
  [
    test "set_source on code" (fun () ->
        let c = Cell.code "old" |> Cell.set_source "new" in
        equal string "new" (Cell.source c));
    test "set_source on text" (fun () ->
        let c = Cell.text "old" |> Cell.set_source "new" in
        equal string "new" (Cell.source c));
    test "set_outputs" (fun () ->
        let c = Cell.code "x" |> Cell.set_outputs [ Cell.Stdout "hello" ] in
        match c with
        | Cell.Code { outputs; _ } -> equal int 1 (List.length outputs)
        | _ -> fail "expected Code cell");
    test "set_outputs on text is noop" (fun () ->
        let c = Cell.text "x" |> Cell.set_outputs [ Cell.Stdout "hello" ] in
        match c with Cell.Text _ -> () | _ -> fail "expected Text cell");
    test "append_output" (fun () ->
        let c =
          Cell.code "x"
          |> Cell.append_output (Cell.Stdout "a")
          |> Cell.append_output (Cell.Stdout "b")
        in
        match c with
        | Cell.Code { outputs; _ } -> equal int 2 (List.length outputs)
        | _ -> fail "expected Code cell");
    test "clear_outputs" (fun () ->
        let c =
          Cell.code "x"
          |> Cell.set_outputs [ Cell.Stdout "hello" ]
          |> Cell.clear_outputs
        in
        match c with
        | Cell.Code { outputs; _ } -> equal int 0 (List.length outputs)
        | _ -> fail "expected Code cell");
  ]

let () =
  run "Cell"
    [
      group "Constructors" constructor_tests;
      group "Transformations" transformation_tests;
    ]
