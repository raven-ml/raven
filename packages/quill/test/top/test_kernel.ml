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

let cell code = Cell.code ~id:"c" ~language:"ocaml" code

(* [outputs cells] is the outputs a notebook shows for the last of [cells], each
   run in turn by one kernel. *)
let outputs cells =
  let last = ref (Cell.code ~language:"ocaml" "") in
  let on_event = function
    | Kernel.Output { output; _ } -> last := Cell.append_output output !last
    | _ -> ()
  in
  let kernel = Quill_top.create ~on_event () in
  List.iter
    (fun code ->
      last := Cell.code ~language:"ocaml" "";
      Kernel.execute kernel (cell code))
    cells;
  Kernel.shutdown kernel;
  match !last with Cell.Code { outputs; _ } -> outputs | Cell.Text _ -> []

let svg data = Cell.Display { mime = "image/svg+xml"; id = None; data }
let svg_tag content = "quill.display\nimage/svg+xml\n\n" ^ content
let figure tag = outputs [ printer; Printf.sprintf "Fig %S" tag ]

(* [stream kept outs] is [outs] without the other standard stream's text, with
   adjacent text joined. The two streams reach the kernel apart, so each keeps
   its order with the displays and none with the other. *)
let stream kept outs =
  let drop = function
    | Cell.Stdout _ -> kept = `Stderr
    | Cell.Stderr _ -> kept = `Stdout
    | Cell.Error _ | Cell.Display _ -> false
  in
  let join o acc =
    match (o, acc) with
    | _ when drop o -> acc
    | Cell.Stdout a, Cell.Stdout b :: rest -> Cell.Stdout (a ^ b) :: rest
    | Cell.Stderr a, Cell.Stderr b :: rest -> Cell.Stderr (a ^ b) :: rest
    | o, acc -> o :: acc
  in
  List.fold_right join outs []

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
  let outs = outputs [ printer; cell ] in
  equal (list output)
    [
      Cell.Stdout "a\n";
      svg "PGEvPg==";
      Cell.Stdout "a figure\nb\n";
      svg "PGIvPg==";
      Cell.Stdout "a figure\n";
      svg "PGMvPg==";
    ]
    (stream `Stdout outs);
  equal (list output)
    [
      svg "PGEvPg==";
      svg "PGIvPg==";
      Cell.Stderr "e\n";
      svg "PGMvPg==";
      Cell.Stderr "a figure\n";
    ]
    (stream `Stderr outs)

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
      starts_with ~prefix:"a figure" rest;
      contains ~sub:"Exception: Failure: boom" rest
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

(* [installed name code] is the outcome of installing the printer [name] after
   running [code] in a kernel, and the outputs of the cell [T] then. *)
let installed name code =
  let outputs = ref [] in
  let on_event = function
    | Kernel.Output { output; _ } -> outputs := output :: !outputs
    | _ -> ()
  in
  let kernel = Quill_top.create ~on_event () in
  Kernel.execute kernel (cell code);
  let r = Quill_top.install_printer name in
  outputs := [];
  Kernel.execute kernel (cell "T");
  Kernel.shutdown kernel;
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

(* Packages *)

let write path contents =
  Out_channel.with_open_bin path (fun oc -> output_string oc contents)

(* [findlib_root broken] is a new directory, first on the findlib search path,
   with the package [broken], whose archive is not bytecode, and
   [needs_<broken>], which requires it. Findlib keeps a package's directory once
   it has looked it up, so each test names its own packages. *)
let findlib_root broken =
  let root = temp_dir ~prefix:"findlib" () in
  let package name meta =
    let dir = Filename.concat root name in
    Sys.mkdir dir 0o755;
    write (Filename.concat dir "META") meta
  in
  package broken (Printf.sprintf "archive(byte) = %S" (broken ^ ".cma"));
  write
    (Filename.concat (Filename.concat root broken) (broken ^ ".cma"))
    "this file is not a bytecode archive\n";
  package ("needs_" ^ broken) (Printf.sprintf "requires = %S" broken);
  let sep = if Sys.win32 then ";" else ":" in
  let path =
    match Sys.getenv_opt "OCAMLPATH" with
    | None | Some "" -> root
    | Some path -> root ^ sep ^ path
  in
  setenv "OCAMLPATH" (Some path);
  Quill_top.initialize_if_needed ();
  root

(* The toplevel's report on the archive of [broken] in [root]. *)
let not_bytecode root broken =
  let archive =
    Filename.concat (Filename.concat root broken) (broken ^ ".cma")
  in
  Printf.sprintf "File %s is not a bytecode object file." archive

let no_such_package = function
  | Fl_package_base.No_such_package ("absent", _) -> true
  | _ -> false

let package_tests =
  [
    test "an absent package raises No_such_package" (fun () ->
        ignore (findlib_root "present");
        raises_match no_such_package (fun () ->
            Quill_top.load_package "absent"));
    test "an archive that does not load is reported at each load" (fun () ->
        let root = findlib_root "reloaded" in
        let first = Quill_top.load_package "reloaded" in
        let second = Quill_top.load_package "reloaded" in
        let report = Error (not_bytecode root "reloaded") in
        equal (result unit string) report first;
        equal (result unit string) report second);
    test "a package whose dependency does not load is reported" (fun () ->
        let root = findlib_root "required" in
        equal (result unit string)
          (Error (not_bytecode root "required"))
          (Quill_top.load_package "needs_required"));
  ]

(* Exceptions *)

let message = "2 problems:\n  no column \"carier\"\n  no column \"x\""

let exception_tests =
  [
    cases "a raised message shows as written" ~id:fst
      [ ("Failure", "failwith"); ("Invalid_argument", "invalid_arg") ]
      (fun (name, raise) ->
        (* A recorded backtrace would follow the report. *)
        let recording = Printexc.backtrace_status () in
        Printexc.record_backtrace false;
        Fun.protect ~finally:(fun () -> Printexc.record_backtrace recording)
        @@ fun () ->
        equal (list output)
          [ Cell.Stdout (Printf.sprintf "Exception: %s: %s\n" name message) ]
          (outputs [ Printf.sprintf "%s %S" raise message ]));
  ]

(* Languages *)

let pp_event ppf = function
  | Kernel.Output { cell_id; output } ->
      Format.fprintf ppf "Output (%S, %a)" cell_id pp_output output
  | Kernel.Finished { cell_id; success } ->
      Format.fprintf ppf "Finished (%S, %b)" cell_id success
  | Kernel.Status_changed _ -> Format.fprintf ppf "Status_changed"

let event = Testable.make ~pp:pp_event ~equal:( = )

let language_tests =
  [
    test "the toplevel runs OCaml" (fun () ->
        let kernel = Quill_top.create ~on_event:ignore () in
        equal string "ocaml" (Kernel.language kernel));
    test "a python cell is refused and leaves the toplevel as it was" (fun () ->
        let events = ref [] in
        let on_event = function
          | Kernel.Status_changed _ -> ()
          | e -> events := e :: !events
        in
        let kernel = Quill_top.create ~on_event () in
        Kernel.execute kernel
          (Cell.code ~id:"a" ~language:"ocaml" "let kept = 41");
        events := [];
        Kernel.execute kernel (Cell.code ~id:"p" ~language:"python" "kept = 0");
        Kernel.execute kernel
          (Cell.code ~id:"b" ~language:"ocaml" "let () = print_int (kept + 1)");
        Kernel.shutdown kernel;
        equal (list event)
          [
            Output
              {
                cell_id = "p";
                output =
                  Error
                    "no kernel runs python code: this notebook's kernel runs \
                     ocaml";
              };
            Finished { cell_id = "p"; success = false };
            Output { cell_id = "b"; output = Stdout "42" };
            Finished { cell_id = "b"; success = true };
          ]
          (List.rev !events));
    test "a python cell has no completions" (fun () ->
        let kernel = Quill_top.create ~on_event:ignore () in
        equal (list string) []
          (List.map
             (fun (i : Kernel.completion_item) -> i.label)
             (Kernel.complete kernel
                (Cell.code ~language:"python" "List.ma")
                ~pos:7));
        Kernel.shutdown kernel);
  ]

let () =
  exit
    (run "Kernel"
       [
         group "Languages" language_tests;
         group "Display tags" kernel_tests;
         group "Printers" printer_tests;
         group "Packages" package_tests;
         group "Exceptions" exception_tests;
       ])
