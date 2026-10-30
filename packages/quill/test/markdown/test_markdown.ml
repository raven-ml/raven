(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Quill

let pp_output ppf = function
  | Cell.Stdout s -> Format.fprintf ppf "Stdout %S" s
  | Cell.Stderr s -> Format.fprintf ppf "Stderr %S" s
  | Cell.Error s -> Format.fprintf ppf "Error %S" s
  | Cell.Display { mime; id; data } ->
      Format.fprintf ppf "Display {mime = %S; id = %s; data = %S}" mime
        (match id with
        | None -> "None"
        | Some id -> Printf.sprintf "Some %S" id)
        data

let output = Testable.make ~pp:pp_output ~equal:( = )

(* A cell as the parser promises it: a text cell's source, or the language and
   source of a code cell. *)
type view = Txt of string | Code of string * string

let pp_view ppf = function
  | Txt s -> Format.fprintf ppf "Txt %S" s
  | Code (lang, s) -> Format.fprintf ppf "Code (%S, %S)" lang s

let cell = Testable.make ~pp:pp_view ~equal:( = )

let view = function
  | Cell.Text { source; _ } -> Txt source
  | Cell.Code { language; source; _ } -> Code (language, source)

let cells doc = List.map view (Doc.cells doc)

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

(* A top-level block of a notebook and the cell it is part of. *)
let fragments =
  let txt md = (md, Txt md) in
  [
    txt "Some prose.";
    txt "<!-- quill:cell id=\"t_stale\" -->\nText after a stale id.";
    ("```ocaml\nlet x = 1\n```", Code ("ocaml", "let x = 1"));
    ("```bash\nls\n```", Code ("bash", "ls"));
    ("```\nplain\n```", Code ("", "plain"));
    ("~~~ocaml\nlet z = 3\n~~~", Code ("ocaml", "let z = 3"));
    ( "````ocaml\nlet s = {|\n```\n|}\n````",
      Code ("ocaml", "let s = {|\n```\n|}") );
  ]

let fragment =
  Gen.of_list ~pp:(fun ppf (md, _) -> Format.fprintf ppf "%S" md) fragments

(* Adjacent text blocks form one text cell, byte for byte. *)
let rec expected = function
  | [] -> []
  | Txt a :: Txt b :: rest -> expected (Txt (a ^ "\n\n" ^ b) :: rest)
  | v :: rest -> v :: expected rest

let render md = Quill_markdown.to_string (Quill_markdown.of_string md)

let round_trip_tests =
  [
    prop "saving keeps every cell"
      (Gen.list ~size:(Gen.int_range 0 8) fragment)
      (fun frags ->
        let md = String.concat "\n\n" (List.map fst frags) ^ "\n" in
        let doc = Quill_markdown.of_string md in
        let views = expected (List.map snd frags) in
        equal (list cell) views (cells doc);
        equal (list cell) views
          (cells (Quill_markdown.of_string (Quill_markdown.to_string doc)));
        Law.idempotent text render md);
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
    test "a fence outlasts the backticks in its source" (fun () ->
        let source = "let s = {|\n```\n  ``````\n|}" in
        let md =
          Quill_markdown.to_string (Doc.of_cells [ Cell.code ~id:"c_1" source ])
        in
        equal text
          ("<!-- quill:cell id=\"c_1\" -->\n```````ocaml\n" ^ source
         ^ "\n```````\n")
          md;
        equal (list cell)
          [ Code ("ocaml", source) ]
          (cells (Quill_markdown.of_string md)));
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
    test "code cell IDs survive roundtrip" (fun () ->
        let c1 = Cell.text ~id:"t_1" "# Hello" in
        let c2 = Cell.code ~id:"c_2" "let x = 1" in
        let doc = Doc.of_cells [ c1; c2 ] in
        let md = Quill_markdown.to_string doc in
        let doc2 = Quill_markdown.of_string md in
        (match Doc.nth 0 doc2 with
        | Some (Cell.Text _) -> ()
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
               [
                 Cell.Display
                   { mime = "image/png"; id = None; data = "iVBORw0KGgo=" };
               ]
        in
        let doc = Doc.of_cells [ c ] in
        let md = Quill_markdown.to_string_with_outputs doc in
        let doc2 = Quill_markdown.of_string md in
        match Doc.nth 0 doc2 with
        | Some (Cell.Code { outputs; _ }) -> (
            equal int 1 (List.length outputs);
            match List.hd outputs with
            | Cell.Display { mime; data; _ } ->
                equal string "image/png" mime;
                equal string "iVBORw0KGgo=" data
            | _ -> fail "expected Display output")
        | _ -> fail "expected Code cell");
    test "saving drops display ids" (fun () ->
        let displays id =
          [
            Cell.Display { mime = "image/png"; id; data = "iVBORw0KGgo=" };
            Cell.Display { mime = "text/html"; id; data = "<b>hello</b>" };
          ]
        in
        let save outputs =
          Cell.code ~id:"c" "plot ()" |> Cell.set_outputs outputs |> fun c ->
          Quill_markdown.to_string_with_outputs (Doc.of_cells [ c ])
        in
        let md = save (displays (Some "fig")) in
        equal text (save (displays None)) md;
        match Doc.nth 0 (Quill_markdown.of_string md) with
        | Some (Cell.Code { outputs; _ }) ->
            equal (list output) (displays None) outputs
        | _ -> fail "expected Code cell");
    test "roundtrip mixed outputs" (fun () ->
        let c =
          Cell.code "let x = 1"
          |> Cell.set_outputs
               [
                 Cell.Stdout "val x : int = 1";
                 Cell.Stderr "Warning 26: unused";
                 Cell.Display
                   { mime = "text/html"; id = None; data = "<b>hello</b>" };
               ]
        in
        let doc = Doc.of_cells [ c ] in
        let md = Quill_markdown.to_string_with_outputs doc in
        let doc2 = Quill_markdown.of_string md in
        match Doc.nth 0 doc2 with
        | Some (Cell.Code { outputs; _ }) -> (
            equal int 3 (List.length outputs);
            match outputs with
            | [ Cell.Stdout s; Cell.Stderr e; Cell.Display { mime; data; _ } ]
              ->
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

let attrs_tests =
  [
    test "parse collapsed attr" (fun () ->
        let md =
          "<!-- quill:cell id=\"c_1\" collapsed -->\n```ocaml\nlet x = 1\n```\n"
        in
        let doc = Quill_markdown.of_string md in
        match Doc.nth 0 doc with
        | Some (Cell.Code { attrs; _ }) ->
            is_true ~msg:"collapsed" attrs.collapsed;
            is_false ~msg:"not hide_source" attrs.hide_source
        | _ -> fail "expected Code cell");
    test "parse hide-source attr" (fun () ->
        let md =
          "<!-- quill:cell id=\"c_1\" hide-source -->\n\
           ```ocaml\n\
           let x = 1\n\
           ```\n"
        in
        let doc = Quill_markdown.of_string md in
        match Doc.nth 0 doc with
        | Some (Cell.Code { attrs; _ }) ->
            is_false ~msg:"not collapsed" attrs.collapsed;
            is_true ~msg:"hide_source" attrs.hide_source
        | _ -> fail "expected Code cell");
    test "parse multiple attrs" (fun () ->
        let md =
          "<!-- quill:cell id=\"c_1\" collapsed hide-source -->\n\
           ```ocaml\n\
           let x = 1\n\
           ```\n"
        in
        let doc = Quill_markdown.of_string md in
        match Doc.nth 0 doc with
        | Some (Cell.Code { attrs; _ }) ->
            is_true ~msg:"collapsed" attrs.collapsed;
            is_true ~msg:"hide_source" attrs.hide_source
        | _ -> fail "expected Code cell");
    test "unknown attrs are ignored" (fun () ->
        let md =
          "<!-- quill:cell id=\"c_1\" collapsed future-flag -->\n\
           ```ocaml\n\
           let x = 1\n\
           ```\n"
        in
        let doc = Quill_markdown.of_string md in
        match Doc.nth 0 doc with
        | Some (Cell.Code { attrs; id; _ }) ->
            equal string "c_1" id;
            is_true ~msg:"collapsed" attrs.collapsed
        | _ -> fail "expected Code cell");
    test "no attrs is backward compatible" (fun () ->
        let md = "<!-- quill:cell id=\"c_1\" -->\n```ocaml\nlet x = 1\n```\n" in
        let doc = Quill_markdown.of_string md in
        match Doc.nth 0 doc with
        | Some (Cell.Code { attrs; id; _ }) ->
            equal string "c_1" id;
            is_false ~msg:"not collapsed" attrs.collapsed;
            is_false ~msg:"not hide_source" attrs.hide_source
        | _ -> fail "expected Code cell");
    test "an id comment before text is text" (fun () ->
        let comment = "<!-- quill:cell id=\"t_1\" collapsed -->" in
        let md = comment ^ "\n# Hidden section\n" in
        let doc = Quill_markdown.of_string md in
        equal (list cell) [ Txt (comment ^ "\n# Hidden section") ] (cells doc);
        equal text md (Quill_markdown.to_string doc));
    test "roundtrip collapsed" (fun () ->
        let a = { Cell.collapsed = true; hide_source = false } in
        let c = Cell.code ~id:"c_1" ~attrs:a "let x = 1" in
        let doc = Doc.of_cells [ c ] in
        let md = Quill_markdown.to_string doc in
        let doc2 = Quill_markdown.of_string md in
        match Doc.nth 0 doc2 with
        | Some (Cell.Code { id; attrs; _ }) ->
            equal string "c_1" id;
            is_true ~msg:"collapsed survives" attrs.collapsed;
            is_false ~msg:"hide_source unchanged" attrs.hide_source
        | _ -> fail "expected Code cell");
    test "roundtrip hide-source" (fun () ->
        let a = { Cell.collapsed = false; hide_source = true } in
        let c = Cell.code ~id:"c_2" ~attrs:a "let x = 1" in
        let doc = Doc.of_cells [ c ] in
        let md = Quill_markdown.to_string doc in
        let doc2 = Quill_markdown.of_string md in
        match Doc.nth 0 doc2 with
        | Some (Cell.Code { id; attrs; _ }) ->
            equal string "c_2" id;
            is_false ~msg:"not collapsed" attrs.collapsed;
            is_true ~msg:"hide_source survives" attrs.hide_source
        | _ -> fail "expected Code cell");
    test "roundtrip both attrs" (fun () ->
        let a = { Cell.collapsed = true; hide_source = true } in
        let c = Cell.code ~id:"c_3" ~attrs:a "let x = 1" in
        let doc = Doc.of_cells [ c ] in
        let md = Quill_markdown.to_string doc in
        let doc2 = Quill_markdown.of_string md in
        match Doc.nth 0 doc2 with
        | Some (Cell.Code { attrs; _ }) ->
            is_true ~msg:"collapsed survives" attrs.collapsed;
            is_true ~msg:"hide_source survives" attrs.hide_source
        | _ -> fail "expected Code cell");
    test "default attrs produce no tokens" (fun () ->
        let c = Cell.code ~id:"c_4" "let x = 1" in
        let doc = Doc.of_cells [ c ] in
        let md = Quill_markdown.to_string doc in
        is_true ~msg:"no collapsed token"
          (not
             (String.split_on_char ' ' md
             |> List.exists (fun w -> w = "collapsed"))));
  ]

let () =
  exit
    (run "Markdown"
       [
         group "Parsing" parsing_tests;
         group "Round trip" round_trip_tests;
         group "Rendering" rendering_tests;
         group "ID persistence" id_persistence_tests;
         group "Structured outputs" structured_output_tests;
         group "Attributes" attrs_tests;
       ])
