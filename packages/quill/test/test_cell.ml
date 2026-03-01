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

let attrs_tests =
  [
    test "default attrs" (fun () ->
        let c = Cell.code "x" in
        let a = Cell.attrs c in
        is_false ~msg:"not collapsed" a.collapsed;
        is_false ~msg:"not hide_source" a.hide_source);
    test "default attrs on text" (fun () ->
        let c = Cell.text "x" in
        let a = Cell.attrs c in
        is_false ~msg:"not collapsed" a.collapsed;
        is_false ~msg:"not hide_source" a.hide_source);
    test "code with attrs" (fun () ->
        let a = { Cell.collapsed = true; hide_source = false } in
        let c = Cell.code ~attrs:a "x" in
        let a' = Cell.attrs c in
        is_true ~msg:"collapsed" a'.collapsed;
        is_false ~msg:"not hide_source" a'.hide_source);
    test "set_attrs on code" (fun () ->
        let c = Cell.code "x" in
        let c = Cell.set_attrs { collapsed = false; hide_source = true } c in
        let a = Cell.attrs c in
        is_false ~msg:"not collapsed" a.collapsed;
        is_true ~msg:"hide_source" a.hide_source);
    test "set_attrs on text" (fun () ->
        let c = Cell.text "x" in
        let c = Cell.set_attrs { collapsed = true; hide_source = false } c in
        is_true ~msg:"collapsed" (Cell.attrs c).collapsed);
    test "set_source preserves attrs" (fun () ->
        let a = { Cell.collapsed = true; hide_source = true } in
        let c = Cell.code ~attrs:a "old" |> Cell.set_source "new" in
        let a' = Cell.attrs c in
        is_true ~msg:"collapsed preserved" a'.collapsed;
        is_true ~msg:"hide_source preserved" a'.hide_source);
  ]

let () =
  run "Cell"
    [
      group "Constructors" constructor_tests;
      group "Transformations" transformation_tests;
      group "Attributes" attrs_tests;
    ]
