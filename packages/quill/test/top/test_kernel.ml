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

(* A printer for [fig] values whose tag is the string [fig] holds. *)
let printer =
  {|type fig = Fig of string
let pp_fig ppf (Fig tag) =
  Format.pp_open_stag ppf (Format.String_tag tag);
  Format.pp_print_string ppf "a figure";
  Format.pp_close_stag ppf ();;
#install_printer pp_fig|}

(* [outputs cells] is the outputs a notebook shows for the last of [cells], each
   run in turn by one kernel. *)
let outputs cells =
  let last = ref (Cell.code "") in
  let on_event = function
    | Kernel.Output { output; _ } -> last := Cell.append_output output !last
    | _ -> ()
  in
  let kernel = Quill_top.create ~on_event () in
  List.iter
    (fun code ->
      last := Cell.code "";
      kernel.execute ~cell_id:"c" ~code)
    cells;
  kernel.shutdown ();
  match !last with Cell.Code { outputs; _ } -> outputs | Cell.Text _ -> []

let svg data = Cell.Display { mime = "image/svg+xml"; id = None; data }
let svg_tag content = "quill.display\nimage/svg+xml\n\n" ^ content
let figure tag = outputs [ printer; Printf.sprintf "Fig %S" tag ]

let printed_figures () =
  let cell =
    Printf.sprintf
      {|let () =
  Format.printf "a@.";
  Format.printf "%%a@." pp_fig (Fig %S);
  print_string "b\n";
  Format.printf "%%a@." pp_fig (Fig %S);
  prerr_string "e\n";
  Format.eprintf "%%a@." pp_fig (Fig %S)|}
      (svg_tag "<a/>") (svg_tag "<b/>") (svg_tag "<c/>")
  in
  equal (list output)
    [
      Cell.Stdout "a\n";
      svg "PGEvPg==";
      Cell.Stdout "a figure\nb\n";
      svg "PGIvPg==";
      Cell.Stdout "a figure\n";
      Cell.Stderr "e\n";
      svg "PGMvPg==";
      Cell.Stderr "a figure\n";
    ]
    (outputs [ printer; cell ])

(* [printed fmt] is the outputs of a cell printing a figure with the format
   [fmt] on [Format.std_formatter]. *)
let printed fmt =
  let cell =
    Printf.sprintf "let () = Format.printf %S pp_fig (Fig %S)" fmt
      (svg_tag "<a/>")
  in
  outputs [ printer; cell ]

(* The standard formatters' tag settings are the test's before and after a cell
   that raises with a tag open on each. *)
let restored () =
  let marks = ref [] in
  let stags =
    {
      Format.mark_open_stag = (fun _ -> "");
      mark_close_stag = (fun _ -> "");
      print_open_stag =
        (function Format.String_tag s -> marks := s :: !marks | _ -> ());
      print_close_stag = ignore;
    }
  in
  let formatters = [ Format.std_formatter; Format.err_formatter ] in
  let saved =
    List.map
      (fun ppf ->
        ( Format.pp_get_mark_tags ppf (),
          Format.pp_get_print_tags ppf (),
          Format.pp_get_formatter_stag_functions ppf () ))
      formatters
  in
  let restore () =
    List.iter2
      (fun ppf (mark, print, stags) ->
        Format.pp_set_mark_tags ppf mark;
        Format.pp_set_print_tags ppf print;
        Format.pp_set_formatter_stag_functions ppf stags)
      formatters saved
  in
  Fun.protect ~finally:restore @@ fun () ->
  List.iter
    (fun ppf ->
      Format.pp_set_mark_tags ppf false;
      Format.pp_set_print_tags ppf true;
      Format.pp_set_formatter_stag_functions ppf stags)
    formatters;
  ignore
    (outputs
       [
         {|let () =
  Format.pp_open_stag Format.std_formatter (Format.String_tag "t");
  Format.pp_open_stag Format.err_formatter (Format.String_tag "t");
  failwith "x"|};
       ]);
  equal
    (list (pair bool bool))
    [ (false, true); (false, true) ]
    (List.map
       (fun ppf ->
         (Format.pp_get_mark_tags ppf (), Format.pp_get_print_tags ppf ()))
       formatters);
  List.iter
    (fun ppf ->
      Format.pp_open_stag ppf (Format.String_tag "after");
      Format.pp_close_stag ppf ();
      Format.pp_print_flush ppf ())
    formatters;
  equal (list string) [ "after"; "after" ] !marks

(* A cell prints a figure after unflushed text, leaves the line unflushed and
   raises: its text and display come out in order, then the toplevel's report of
   the exception, which it prints on standard output. *)
let raised_mid_print () =
  let cell =
    Printf.sprintf
      {|let () =
  Format.printf "Epoch 1: %%a" pp_fig (Fig %S);
  failwith "boom"|}
      (svg_tag "<a/>")
  in
  match outputs [ printer; cell ] with
  | [ text; display; Cell.Stdout rest ] ->
      equal (list output)
        [ Cell.Stdout "Epoch 1: "; svg "PGEvPg==" ]
        [ text; display ];
      starts_with ~affix:"a figure" rest;
      contains ~sub:"Failure \"boom\"" rest
  | outs -> failf "%a" (Format.pp_print_list pp_output) outs

let kernel_tests =
  [
    test "display tags printed on the standard formatters display in order"
      printed_figures;
    test "a display follows the text before it on its line" (fun () ->
        equal (list output)
          [ Cell.Stdout "Epoch 1: "; svg "PGEvPg=="; Cell.Stdout "a figure\n" ]
          (printed "Epoch 1: %a@."));
    test "a cell that raises mid-print shows its display and its error"
      raised_mid_print;
    test "a display follows the text before it in its box" (fun () ->
        equal (list output)
          [ Cell.Stdout "a\nb "; svg "PGEvPg=="; Cell.Stdout "a figure\n" ]
          (printed "@[<v>a@,b %a@]@."));
    test "a cell that raises mid-print leaves the standard formatters as found"
      restored;
    test "a value's display tag is a display output" (fun () ->
        equal (list output)
          [
            Cell.Display
              { mime = "image/svg+xml"; id = None; data = "PHN2Zy8+" };
            Cell.Stdout "- : fig = a figure\n";
          ]
          (figure "quill.display\nimage/svg+xml\n\n<svg/>"));
    test "a display id reaches the display output" (fun () ->
        equal (list output)
          [
            Cell.Display
              { mime = "image/svg+xml"; id = Some "loss"; data = "PHN2Zy8+" };
            Cell.Stdout "- : fig = a figure\n";
          ]
          (figure "quill.display\nimage/svg+xml\nloss\n<svg/>"));
    test "a malformed display tag is an error output" (fun () ->
        equal (list output)
          [
            Cell.Error "malformed display tag: the MIME type is empty";
            Cell.Stdout "- : fig = a figure\n";
          ]
          (figure "quill.display\n\n\n<svg/>"));
    test "another string tag prints its text alone" (fun () ->
        equal (list output)
          [ Cell.Stdout "- : fig = a figure\n" ]
          (figure "bold"));
  ]

(* Printers *)

(* [installed name cell] is the outcome of installing the printer [name] after
   running [cell] in a kernel, and the outputs of the cell [T] then. *)
let installed name cell =
  let outputs = ref [] in
  let on_event = function
    | Kernel.Output { output; _ } -> outputs := output :: !outputs
    | _ -> ()
  in
  let kernel = Quill_top.create ~on_event () in
  kernel.execute ~cell_id:"c" ~code:cell;
  let r = Quill_top.install_printer name in
  outputs := [];
  kernel.execute ~cell_id:"c" ~code:"T";
  kernel.shutdown ();
  (r, List.rev !outputs)

let tee =
  {|type t = T
let pp_t ppf T = Format.pp_print_string ppf "tee"
let n = 3|}

let printer_tests =
  [
    test "a printer installs and prints values of its type" (fun () ->
        let r, outputs = installed "pp_t" tee in
        is_ok ~pp:Format.pp_print_string r;
        equal (list output) [ Cell.Stdout "- : t = tee\n" ] outputs);
    test "an unbound printer is reported" (fun () ->
        let r, _ = installed "no_such_printer" tee in
        expect (require_error r)
        @@ __POS_OF__ {| Unbound value no_such_printer. |});
    test "a value that is not a printer is reported" (fun () ->
        let r, _ = installed "n" tee in
        expect (require_error r)
        @@ __POS_OF__ {| n has the wrong type for a printing function. |});
  ]

let () =
  exit
    (run "Kernel"
       [ group "Display tags" kernel_tests; group "Printers" printer_tests ])
