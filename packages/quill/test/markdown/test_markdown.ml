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

(* Files *)

(* The base64 of [s], as an image output's [data] holds it. *)
let base64 s =
  let alphabet =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
  in
  let buf = Buffer.create (String.length s * 2) in
  let byte i = if i < String.length s then Char.code s.[i] else 0 in
  let rec loop i =
    if i < String.length s then (
      let n = (byte i lsl 16) lor (byte (i + 1) lsl 8) lor byte (i + 2) in
      for k = 0 to 3 do
        if i + k - 1 < String.length s then
          Buffer.add_char buf alphabet.[(n lsr (18 - (6 * k))) land 63]
        else Buffer.add_char buf '='
      done;
      loop (i + 3))
  in
  loop 0;
  Buffer.contents buf

let all_bytes = String.init 256 Char.chr
let image mime raw = Cell.Display { mime; id = None; data = base64 raw }

let pp_cell ppf = function
  | Cell.Code { id; source; outputs; _ } ->
      Format.fprintf ppf
        "@[<hv 2>Code {id = %S; source = %S;@ outputs = [%a]}@]" id source
        (Format.pp_print_list
           ~pp_sep:(fun ppf () -> Format.fprintf ppf ";@ ")
           pp_output)
        outputs
  | Cell.Text { source; _ } -> Format.fprintf ppf "Text %S" source

let doc =
  Testable.make
    ~pp:(fun ppf d -> Format.pp_print_list pp_cell ppf (Doc.cells d))
    ~equal:(fun a b -> Doc.cells a = Doc.cells b)

let figures =
  Testable.make
    ~pp:(fun ppf f ->
      Format.pp_print_string ppf
        (match f with Quill_markdown.Inline -> "Inline" | Files -> "Files"))
    ~equal:( = )

let raws d =
  List.concat_map
    (function
      | Cell.Code { outputs; _ } ->
          List.filter_map
            (function
              | Cell.Display { mime; data; _ }
                when String.starts_with ~prefix:"image/" mime ->
                  Some data
              | _ -> None)
            outputs
      | Cell.Text _ -> [])
    (Doc.cells d)

let gen_image_bytes =
  let open Gen in
  one_of
    [
      constant ~pp:Format.pp_print_string "";
      string_of ~size:(constant 1) char;
      constant
        ~pp:(fun ppf _ -> Format.pp_print_string ppf "all_bytes")
        all_bytes;
      string;
    ]

let gen_output =
  let open Gen in
  one_of
    [
      constant ~pp:pp_output (Cell.Stdout "val x : int = 1");
      constant ~pp:pp_output
        (Cell.Display { mime = "text/html"; id = None; data = "<b>hi</b>" });
      (let+ mime =
         of_list ~pp:Format.pp_print_string [ "image/png"; "image/svg+xml" ]
       and+ raw = gen_image_bytes in
       image mime raw);
    ]

let gen_doc =
  let open Gen in
  let+ outputs = list (list gen_output) in
  Doc.of_cells
    (List.mapi
       (fun i os ->
         Cell.code ~id:(Printf.sprintf "c_%d" i) "plot ()"
         |> Cell.set_outputs os)
       outputs)

let cover_images d =
  let images os =
    List.filter
      (function
        | Cell.Display { mime; _ } -> String.starts_with ~prefix:"image/" mime
        | _ -> false)
      os
  in
  let cells =
    List.filter_map
      (function
        | Cell.Code { outputs; _ } -> Some outputs | Cell.Text _ -> None)
      (Doc.cells d)
  in
  let raws = raws d in
  cover "an empty image" (List.mem "" raws);
  (* The base64 of one byte is two digits and two pads. *)
  cover "a one-byte image"
    (List.exists
       (fun r -> String.length r = 4 && String.ends_with ~suffix:"==" r)
       raws);
  cover "an image of all 256 bytes" (List.mem (base64 all_bytes) raws);
  cover "several images in one cell"
    (List.exists (fun os -> List.length (images os) >= 2) cells);
  cover "an image beside another output"
    (List.exists
       (fun os -> images os <> [] && List.length (images os) < List.length os)
       cells);
  cover "no image" (raws = [])

let files_in dir =
  if Sys.file_exists dir then Array.length (Sys.readdir dir) else 0

let loaded = result (pair doc figures) string
let saved = result unit string

(* [save_result f path d] is [Quill_markdown.save f path d] without its
   stamp. *)
let save_result f path d = Result.map ignore (Quill_markdown.save f path d)
let save f path d = equal saved (Ok ()) (save_result f path d)
let load = Quill_markdown.load

(* A notebook of one cell, [c_x], that stores its images as files and shows the
   image [src]. *)
let notebook src =
  "<!-- quill:figures files -->\n\n\
   <!-- quill:cell id=\"c_x\" -->\n\
   ```ocaml\n\
   plot ()\n\
   ```\n\
   <!-- quill:output -->\n\
   <!-- out:display image/svg+xml -->\n\
   <img src=\"" ^ src ^ "\">\n<!-- /quill:output -->\n"

let one_figure =
  Doc.of_cells
    [
      Cell.code ~id:"c_x" "plot ()"
      |> Cell.set_outputs [ image "image/svg+xml" "<svg/>" ];
    ]

let write path s =
  let oc = open_out_bin path in
  output_string oc s;
  close_out oc

let mkdir_p path =
  let rec loop p =
    if not (Sys.file_exists p) then (
      loop (Filename.dirname p);
      Sys.mkdir p 0o755)
  in
  loop path

let file_tests =
  [
    prop "load reads what save Inline wrote" gen_doc (fun d ->
        cover_images d;
        let path = Filename.concat (temp_dir ()) "nb.md" in
        save Inline path d;
        equal loaded (Ok (d, Inline)) (load path));
    prop "load reads what save Files wrote, over an earlier save"
      (Gen.pair gen_doc gen_doc) (fun (before, d) ->
        cover_images d;
        let dir = temp_dir () in
        let path = Filename.concat dir "nb.md" in
        save Files path before;
        save Files path d;
        equal loaded (Ok (d, Files)) (load path);
        equal int
          ~msg:"figures/nb/ holds the images of the notebook and nothing else"
          (List.length (raws d))
          (files_in (Filename.concat dir "figures/nb")));
    prop "notebooks in one directory keep each other's figures"
      (Gen.triple gen_doc gen_doc gen_doc) (fun (a, b, a') ->
        let dir = temp_dir () in
        let a_md = Filename.concat dir "a.md"
        and b_md = Filename.concat dir "b.md" in
        (* An image a text cell shows, named as a figure of cell c_0 was in
           earlier versions. *)
        mkdir_p (Filename.concat dir "figures");
        let user_image = Filename.concat dir "figures/c_0.png" in
        write user_image "user image";
        save Files a_md a;
        save Files b_md b;
        save Files a_md a';
        equal loaded (Ok (b, Files)) (load b_md);
        equal loaded (Ok (a', Files)) (load a_md);
        equal string "user image"
          (In_channel.with_open_bin user_image In_channel.input_all));
    test "save Files stores an image at figures/<stem>/<cell id>.<ext>"
      (fun () ->
        let dir = temp_dir () in
        let path = Filename.concat dir "nb.md" in
        let d =
          Doc.of_cells
            [
              Cell.code ~id:"c_x" "plot ()"
              |> Cell.set_outputs
                   [ image "image/svg+xml" "<svg/>"; image "image/png" "png" ];
            ]
        in
        save Files path d;
        let md = In_channel.with_open_bin path In_channel.input_all in
        starts_with ~prefix:"<!-- quill:figures files -->\n" md;
        contains ~sub:"<img src=\"figures/nb/c_x.svg\">" md;
        contains ~sub:"<img src=\"figures/nb/c_x-2.png\">" md;
        equal string "<svg/>"
          (In_channel.with_open_bin
             (Filename.concat dir "figures/nb/c_x.svg")
             In_channel.input_all));
    test "save Files through a figures directory that is a link" (fun () ->
        if Sys.win32 then skip ~reason:"symbolic links need privileges" ();
        let dir = temp_dir () in
        let shared = temp_dir () in
        Unix.symlink shared (Filename.concat dir "figures");
        let path = Filename.concat dir "nb.md" in
        save Files path one_figure;
        equal loaded (Ok (one_figure, Files)) (load path);
        equal int ~msg:"the figure is in the linked directory" 1
          (files_in (Filename.concat shared "nb")));
    cases "load refuses a reference save never writes" ~id:fst
      [
        ("figures/nb/c_x.svg", "no such file");
        ("figures/nb/sub", "not a regular file");
        ("../c_x.svg", "not in figures/nb/");
        ("/etc/passwd", "not in figures/nb/");
        ("figures/other/c_x.svg", "not in figures/nb/");
        ("figures/nb/../other/c_x.svg", "not in figures/nb/");
        ("figures/nb/sub/c_x.svg", "not in figures/nb/");
        ("figures/nb/..", "not in figures/nb/");
        ("figures/nb/", "not in figures/nb/");
        ("https://example.com/c_x.svg", "not in figures/nb/");
      ]
      (fun (src, reason) ->
        let dir = temp_dir () in
        List.iter
          (fun d -> mkdir_p (Filename.concat dir d))
          [ "figures/nb/sub"; "figures/other" ];
        List.iter
          (fun f -> write (Filename.concat dir f) "<svg/>")
          [ "c_x.svg"; "figures/other/c_x.svg"; "figures/nb/sub/c_x.svg" ];
        let path = Filename.concat dir "nb.md" in
        write path (notebook src);
        equal loaded
          (Error (Printf.sprintf "%s: cell c_x: figure %s: %s" path src reason))
          (load path));
    test "load refuses a figure that is a FIFO, without reading it" (fun () ->
        if Sys.win32 then skip ~reason:"no FIFOs" ();
        let dir = temp_dir () in
        mkdir_p (Filename.concat dir "figures/nb");
        Unix.mkfifo (Filename.concat dir "figures/nb/c_x.svg") 0o600;
        let path = Filename.concat dir "nb.md" in
        write path (notebook "figures/nb/c_x.svg");
        equal loaded
          (Error
             (path ^ ": cell c_x: figure figures/nb/c_x.svg: not a regular file"))
          (load path));
    test "load reports a notebook it cannot read" (fun () ->
        let path = Filename.concat (temp_dir ()) "nb.md" in
        contains ~sub:path (require_error (load path)));
    test "load ~outputs:false reads no figure" (fun () ->
        let path = Filename.concat (temp_dir ()) "nb.md" in
        write path (notebook "figures/nb/c_x.svg");
        equal loaded
          (Ok (Doc.of_cells [ Cell.code ~id:"c_x" "plot ()" ], Files))
          (load ~outputs:false path));
    cases "of_string refuses a notebook that stores figures as files" ~id:fst
      [
        ( "a figure reference",
          ( "<!-- quill:cell id=\"c_x\" -->\n\
             ```ocaml\n\
             plot ()\n\
             ```\n\
             <!-- quill:output -->\n\
             <!-- out:display image/svg+xml -->\n\
             <img src=\"figures/nb/c_x.svg\">\n\
             <!-- /quill:output -->\n",
            "Quill_markdown.of_string: cell c_x: figure figures/nb/c_x.svg is \
             a file, which Quill_markdown.load reads" ) );
        ( "the figures line",
          ( "<!-- quill:figures files -->\n\n# Title\n",
            "Quill_markdown.of_string: the notebook stores its figures as \
             files, which Quill_markdown.load reads" ) );
      ]
      (fun (_, (md, msg)) ->
        raises (Invalid_argument msg) (fun () -> Quill_markdown.of_string md));
    test "save writes through a notebook that is a link" (fun () ->
        if Sys.win32 then skip ~reason:"symbolic links need privileges" ();
        let dir = temp_dir () in
        mkdir_p (Filename.concat dir "real");
        mkdir_p (Filename.concat dir "book");
        let target = Filename.concat dir "real/nb.md" in
        let link = Filename.concat dir "book/nb.md" in
        write target "";
        Unix.symlink "../real/nb.md" link;
        let d =
          Doc.of_cells
            [
              Cell.code ~id:"c_x" "1"
              |> Cell.set_outputs [ Cell.Stdout "- : int = 1" ];
            ]
        in
        save Inline link d;
        equal string ~msg:"the link stays" "../real/nb.md" (Unix.readlink link);
        equal loaded (Ok (d, Inline)) (load target));
    test "save refuses a read-only notebook" (fun () ->
        if Sys.win32 then skip ~reason:"no POSIX permissions" ();
        let path = Filename.concat (temp_dir ()) "nb.md" in
        write path "# Read-only\n";
        Unix.chmod path 0o444;
        equal saved
          (Error (path ^ " is read-only"))
          (save_result Inline path one_figure);
        equal string "# Read-only\n"
          (In_channel.with_open_bin path In_channel.input_all));
    test "a notebook without the files line reads its figures as Inline"
      (fun () ->
        let dir = temp_dir () in
        let path = Filename.concat dir "nb.md" in
        save Files path one_figure;
        let md = In_channel.with_open_bin path In_channel.input_all in
        let line = "<!-- quill:figures files -->\n\n" in
        starts_with ~prefix:line md;
        let n = String.length line in
        write path (String.sub md n (String.length md - n));
        equal loaded (Ok (one_figure, Inline)) (load path));
    test "save Files refuses a figures path that is no directory" (fun () ->
        let dir = temp_dir () in
        let figures = Filename.concat dir "figures" in
        write figures "not a directory";
        let path = Filename.concat dir "nb.md" in
        equal saved
          (Error (path ^ ": " ^ figures ^ " is not a directory"))
          (save_result Files path one_figure);
        equal bool ~msg:"nothing is written" false (Sys.file_exists path));
    cases "save Files refuses images it cannot name" ~id:fst
      [
        ( "a cell id holding /",
          ( [
              Cell.code ~id:"a/b" "plot ()"
              |> Cell.set_outputs [ image "image/png" "" ];
            ],
            "cell a/b: figure figures/nb/a/b.png: the cell id makes no file \
             name" ) );
        ( "two images of one name",
          ( [
              Cell.code ~id:"a" "plot ()"
              |> Cell.set_outputs [ image "image/png" ""; image "image/png" "" ];
              Cell.code ~id:"a-2" "plot ()"
              |> Cell.set_outputs [ image "image/png" "" ];
            ],
            "cell a-2: figure figures/nb/a-2.png: also stored by cell a" ) );
      ]
      (fun (_, (cells, msg)) ->
        let dir = temp_dir () in
        let path = Filename.concat dir "nb.md" in
        equal saved
          (Error (path ^ ": " ^ msg))
          (save_result Files path (Doc.of_cells cells));
        equal bool ~msg:"nothing is written" false (Sys.file_exists path));
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
         group "Files" file_tests;
       ])
