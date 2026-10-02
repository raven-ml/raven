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
          |> Cell.append_output (Cell.Stderr "b")
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

(* Display protocol *)

let pp_output ppf = function
  | Cell.Stdout s -> Format.fprintf ppf "Stdout %S" s
  | Cell.Stderr s -> Format.fprintf ppf "Stderr %S" s
  | Cell.Error s -> Format.fprintf ppf "Error %S" s
  | Cell.Display { mime; data } ->
      Format.fprintf ppf "Display {mime = %S; data = %S}" mime data

let output = Testable.make ~pp:pp_output ~equal:( = )
let display mime data = Some (Cell.Display { mime; data })
let tag mime content = "quill.display\n" ^ mime ^ "\n\n" ^ content

(* [base64_decode s] is the bytes that the RFC 4648 base64 text [s] encodes. *)
let base64_decode s =
  let value c =
    match c with
    | 'A' .. 'Z' -> Char.code c - Char.code 'A'
    | 'a' .. 'z' -> Char.code c - Char.code 'a' + 26
    | '0' .. '9' -> Char.code c - Char.code '0' + 52
    | '+' -> 62
    | '/' -> 63
    | _ -> failf "%C is not a base64 digit" c
  in
  let b = Buffer.create (String.length s) in
  let acc = ref 0 and bits = ref 0 in
  String.iter
    (fun c ->
      if c <> '=' then begin
        acc := (!acc lsl 6) lor value c;
        bits := !bits + 6;
        if !bits >= 8 then begin
          bits := !bits - 8;
          Buffer.add_char b (Char.chr ((!acc lsr !bits) land 0xff))
        end
      end)
    s;
  Buffer.contents b

let image_data content =
  match Cell.output_of_tag (tag "image/png" content) with
  | Some (Cell.Display { data; _ }) -> data
  | o -> failf "%a" (Format.pp_print_option pp_output) o

let display_tests =
  [
    cases ~name:(Printf.sprintf "%S")
      "a string without the display line is no display"
      [
        ""; "quill.display"; "quill.displayx\nimage/png\n\n"; " quill.display\n";
      ] (fun s -> equal (option output) None (Cell.output_of_tag s));
    test "a text display holds its content" (fun () ->
        equal (option output)
          (display "text/html" "<b>x</b>")
          (Cell.output_of_tag (tag "text/html" "<b>x</b>")));
    test "an image display holds its content in base64" (fun () ->
        equal (option output)
          (display "image/png" "iVBORw0KGgo=")
          (Cell.output_of_tag (tag "image/png" "\x89PNG\r\n\x1a\n")));
    test "the display id is not part of the content" (fun () ->
        equal (option output)
          (display "image/svg+xml" "PHN2Zy8+")
          (Cell.output_of_tag "quill.display\nimage/svg+xml\nfig-1\n<svg/>"));
    test "an empty content is a display" (fun () ->
        equal (option output) (display "text/plain" "")
          (Cell.output_of_tag (tag "text/plain" "")));
    cases
      ~name:(fun (c, _) -> Printf.sprintf "%S" c)
      "base64 follows RFC 4648"
      [
        ("", "");
        ("f", "Zg==");
        ("fo", "Zm8=");
        ("foo", "Zm9v");
        ("foob", "Zm9vYg==");
        ("fooba", "Zm9vYmE=");
        ("foobar", "Zm9vYmFy");
      ]
      (fun (content, data) -> equal string data (image_data content));
    prop "a non-image display keeps any bytes" Gen.string (fun content ->
        equal (option output)
          (display "text/plain" content)
          (Cell.output_of_tag (tag "text/plain" content)));
    prop "an image display keeps any bytes" Gen.string (fun content ->
        Law.round_trip string string image_data base64_decode content);
    cases
      ~name:(fun (s, _) -> Printf.sprintf "%S" s)
      "a malformed display tag is an error"
      [
        ("quill.display\n", "no line ends the MIME type");
        ("quill.display\nimage/png", "no line ends the MIME type");
        ("quill.display\nimage/png\n", "no line ends the display id");
        ("quill.display\nimage/png\nfig-1", "no line ends the display id");
        ("quill.display\n\n\n<svg/>", "the MIME type is empty");
      ]
      (fun (s, why) ->
        equal (option output)
          (Some (Cell.Error ("malformed display tag: " ^ why)))
          (Cell.output_of_tag s));
  ]

let () =
  exit
    (run "Cell"
       [
         group "Constructors" constructor_tests;
         group "Transformations" transformation_tests;
         group "Attributes" attrs_tests;
         group "Display protocol" display_tests;
       ])
