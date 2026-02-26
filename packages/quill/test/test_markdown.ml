(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Quill

let parsing_tests =
  [
    test "empty document" (fun () ->
        let doc = Quill_markdown.of_string "" in
        equal int 0 (Doc.length doc));
    test "text only" (fun () ->
        let doc = Quill_markdown.of_string "# Hello\n\nSome text." in
        equal int 1 (Doc.length doc);
        match Doc.nth 0 doc with
        | Some (Cell.Text _) -> ()
        | _ -> fail "expected Text cell");
    test "code only" (fun () ->
        let doc = Quill_markdown.of_string "```ocaml\nlet x = 1\n```\n" in
        equal int 1 (Doc.length doc);
        match Doc.nth 0 doc with
        | Some (Cell.Code { language; source; _ }) ->
            equal string "ocaml" language;
            equal string "let x = 1" source
        | _ -> fail "expected Code cell");
    test "mixed content" (fun () ->
        let md =
          "# Title\n\n\
           ```ocaml\n\
           let x = 1\n\
           ```\n\n\
           Some text.\n\n\
           ```ocaml\n\
           let y = 2\n\
           ```\n"
        in
        let doc = Quill_markdown.of_string md in
        equal int 4 (Doc.length doc);
        (match Doc.nth 0 doc with
        | Some (Cell.Text _) -> ()
        | _ -> fail "expected Text cell at 0");
        (match Doc.nth 1 doc with
        | Some (Cell.Code { source; _ }) -> equal string "let x = 1" source
        | _ -> fail "expected Code cell at 1");
        (match Doc.nth 2 doc with
        | Some (Cell.Text _) -> ()
        | _ -> fail "expected Text cell at 2");
        match Doc.nth 3 doc with
        | Some (Cell.Code { source; _ }) -> equal string "let y = 2" source
        | _ -> fail "expected Code cell at 3");
    test "code without language" (fun () ->
        let doc = Quill_markdown.of_string "```\nsome code\n```\n" in
        equal int 1 (Doc.length doc);
        match Doc.nth 0 doc with
        | Some (Cell.Code { language; _ }) -> equal string "" language
        | _ -> fail "expected Code cell");
    test "parse output markers" (fun () ->
        let md =
          "```ocaml\n\
           let x = 1\n\
           ```\n\
           <!-- quill:output -->\n\
           val x : int = 1\n\
           <!-- /quill:output -->\n"
        in
        let doc = Quill_markdown.of_string md in
        equal int 1 (Doc.length doc);
        match Doc.nth 0 doc with
        | Some (Cell.Code { outputs; _ }) -> (
            equal int 1 (List.length outputs);
            match List.hd outputs with
            | Cell.Stdout s -> equal string "val x : int = 1" s
            | _ -> fail "expected Stdout output")
        | _ -> fail "expected Code cell with outputs");
    test "parse strips output markers as text" (fun () ->
        let md =
          "# Title\n\n\
           ```ocaml\n\
           let x = 1\n\
           ```\n\
           <!-- quill:output -->\n\
           val x : int = 1\n\
           <!-- /quill:output -->\n\n\
           Some text.\n"
        in
        let doc = Quill_markdown.of_string md in
        equal int 3 (Doc.length doc);
        (match Doc.nth 0 doc with
        | Some (Cell.Text _) -> ()
        | _ -> fail "expected Text cell at 0");
        (match Doc.nth 1 doc with
        | Some (Cell.Code { outputs; _ }) -> equal int 1 (List.length outputs)
        | _ -> fail "expected Code cell at 1");
        match Doc.nth 2 doc with
        | Some (Cell.Text { source; _ }) ->
            is_true ~msg:"text is 'Some text.'"
              (String.trim source = "Some text.")
        | _ -> fail "expected Text cell at 2");
    test "roundtrip with outputs" (fun () ->
        let c =
          Cell.code "let x = 1"
          |> Cell.set_outputs [ Cell.Stdout "val x : int = 1\n" ]
        in
        let doc = Doc.of_cells [ c ] in
        let md = Quill_markdown.to_string_with_outputs doc in
        let doc2 = Quill_markdown.of_string md in
        let md2 = Quill_markdown.to_string_with_outputs doc2 in
        equal string md md2);
    test "fmt strips outputs" (fun () ->
        let md =
          "```ocaml\n\
           let x = 1\n\
           ```\n\
           <!-- quill:output -->\n\
           val x : int = 1\n\
           <!-- /quill:output -->\n"
        in
        let doc = Quill_markdown.of_string md in
        let doc = Doc.clear_all_outputs doc in
        let result = Quill_markdown.to_string doc in
        let has_marker =
          String.split_on_char '\n' result
          |> List.exists (fun l -> String.trim l = "<!-- quill:output -->")
        in
        is_false ~msg:"no output marker after fmt" has_marker);
  ]

let rendering_tests =
  [
    test "render text cell" (fun () ->
        let doc = Doc.of_cells [ Cell.text "# Hello" ] in
        let md = Quill_markdown.to_string doc in
        let lines = String.split_on_char '\n' md in
        is_true ~msg:"contains heading"
          (List.exists (fun l -> String.trim l = "# Hello") lines));
    test "render code cell" (fun () ->
        let doc = Doc.of_cells [ Cell.code ~language:"ocaml" "let x = 1" ] in
        let md = Quill_markdown.to_string doc in
        let lines = String.split_on_char '\n' md in
        is_true ~msg:"has fence"
          (List.exists (fun l -> String.trim l = "```ocaml") lines);
        is_true ~msg:"has source"
          (List.exists (fun l -> String.trim l = "let x = 1") lines));
    test "render with outputs" (fun () ->
        let c =
          Cell.code "let x = 1"
          |> Cell.set_outputs [ Cell.Stdout "val x : int = 1\n" ]
        in
        let doc = Doc.of_cells [ c ] in
        let md = Quill_markdown.to_string_with_outputs doc in
        let has_marker =
          String.split_on_char '\n' md
          |> List.exists (fun l -> String.trim l = "<!-- quill:output -->")
        in
        is_true ~msg:"has output marker" has_marker);
    test "render without outputs omits markers" (fun () ->
        let c =
          Cell.code "let x = 1"
          |> Cell.set_outputs [ Cell.Stdout "val x : int = 1\n" ]
        in
        let doc = Doc.of_cells [ c ] in
        let md = Quill_markdown.to_string doc in
        let has_marker =
          String.split_on_char '\n' md
          |> List.exists (fun l -> String.trim l = "<!-- quill:output -->")
        in
        is_false ~msg:"no output marker" has_marker);
  ]

let id_persistence_tests =
  [
    test "cell IDs survive roundtrip" (fun () ->
        let c1 = Cell.text ~id:"t_1" "# Hello" in
        let c2 = Cell.code ~id:"c_2" "let x = 1" in
        let doc = Doc.of_cells [ c1; c2 ] in
        let md = Quill_markdown.to_string doc in
        let doc2 = Quill_markdown.of_string md in
        (match Doc.nth 0 doc2 with
        | Some (Cell.Text { id; _ }) -> equal string "t_1" id
        | _ -> fail "expected Text cell");
        match Doc.nth 1 doc2 with
        | Some (Cell.Code { id; _ }) -> equal string "c_2" id
        | _ -> fail "expected Code cell");
    test "fresh IDs for unmarked cells" (fun () ->
        let md = "# Hello\n\n```ocaml\nlet x = 1\n```\n" in
        let doc = Quill_markdown.of_string md in
        (match Doc.nth 0 doc with
        | Some c -> is_true ~msg:"text cell has id" (Cell.id c <> "")
        | None -> fail "expected cell");
        match Doc.nth 1 doc with
        | Some c -> is_true ~msg:"code cell has id" (Cell.id c <> "")
        | None -> fail "expected cell");
    test "IDs preserved with outputs" (fun () ->
        let c =
          Cell.code ~id:"c_99" "let x = 1"
          |> Cell.set_outputs [ Cell.Stdout "val x : int = 1\n" ]
        in
        let doc = Doc.of_cells [ c ] in
        let md = Quill_markdown.to_string_with_outputs doc in
        let doc2 = Quill_markdown.of_string md in
        match Doc.nth 0 doc2 with
        | Some (Cell.Code { id; outputs; _ }) ->
            equal string "c_99" id;
            equal int 1 (List.length outputs)
        | _ -> fail "expected Code cell");
  ]

let structured_output_tests =
  [
    test "roundtrip stderr" (fun () ->
        let c =
          Cell.code "let x = 1"
          |> Cell.set_outputs [ Cell.Stderr "Warning 26: unused variable x" ]
        in
        let doc = Doc.of_cells [ c ] in
        let md = Quill_markdown.to_string_with_outputs doc in
        let doc2 = Quill_markdown.of_string md in
        match Doc.nth 0 doc2 with
        | Some (Cell.Code { outputs; _ }) -> (
            equal int 1 (List.length outputs);
            match List.hd outputs with
            | Cell.Stderr s -> equal string "Warning 26: unused variable x" s
            | _ -> fail "expected Stderr output")
        | _ -> fail "expected Code cell");
    test "roundtrip error" (fun () ->
        let c =
          Cell.code "let x = " |> Cell.set_outputs [ Cell.Error "Syntax error" ]
        in
        let doc = Doc.of_cells [ c ] in
        let md = Quill_markdown.to_string_with_outputs doc in
        let doc2 = Quill_markdown.of_string md in
        match Doc.nth 0 doc2 with
        | Some (Cell.Code { outputs; _ }) -> (
            equal int 1 (List.length outputs);
            match List.hd outputs with
            | Cell.Error s -> equal string "Syntax error" s
            | _ -> fail "expected Error output")
        | _ -> fail "expected Code cell");
    test "roundtrip display" (fun () ->
        let c =
          Cell.code "plot ()"
          |> Cell.set_outputs
               [ Cell.Display { mime = "image/png"; data = "iVBORw0KGgo=" } ]
        in
        let doc = Doc.of_cells [ c ] in
        let md = Quill_markdown.to_string_with_outputs doc in
        let doc2 = Quill_markdown.of_string md in
        match Doc.nth 0 doc2 with
        | Some (Cell.Code { outputs; _ }) -> (
            equal int 1 (List.length outputs);
            match List.hd outputs with
            | Cell.Display { mime; data } ->
                equal string "image/png" mime;
                equal string "iVBORw0KGgo=" data
            | _ -> fail "expected Display output")
        | _ -> fail "expected Code cell");
    test "roundtrip mixed outputs" (fun () ->
        let c =
          Cell.code "let x = 1"
          |> Cell.set_outputs
               [
                 Cell.Stdout "val x : int = 1";
                 Cell.Stderr "Warning 26: unused";
                 Cell.Display { mime = "text/html"; data = "<b>hello</b>" };
               ]
        in
        let doc = Doc.of_cells [ c ] in
        let md = Quill_markdown.to_string_with_outputs doc in
        let doc2 = Quill_markdown.of_string md in
        match Doc.nth 0 doc2 with
        | Some (Cell.Code { outputs; _ }) -> (
            equal int 3 (List.length outputs);
            match outputs with
            | [ Cell.Stdout s; Cell.Stderr e; Cell.Display { mime; data } ] ->
                equal string "val x : int = 1" s;
                equal string "Warning 26: unused" e;
                equal string "text/html" mime;
                equal string "<b>hello</b>" data
            | _ -> fail "expected Stdout, Stderr, Display")
        | _ -> fail "expected Code cell");
    test "backward compat: untagged output parsed as stdout" (fun () ->
        let md =
          "```ocaml\n\
           let x = 1\n\
           ```\n\
           <!-- quill:output -->\n\
           val x : int = 1\n\
           <!-- /quill:output -->\n"
        in
        let doc = Quill_markdown.of_string md in
        match Doc.nth 0 doc with
        | Some (Cell.Code { outputs; _ }) -> (
            equal int 1 (List.length outputs);
            match List.hd outputs with
            | Cell.Stdout s -> equal string "val x : int = 1" s
            | _ -> fail "expected Stdout")
        | _ -> fail "expected Code cell");
  ]

let () =
  run "Markdown"
    [
      group "Parsing" parsing_tests;
      group "Rendering" rendering_tests;
      group "ID persistence" id_persistence_tests;
      group "Structured outputs" structured_output_tests;
    ]
