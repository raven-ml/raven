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

(* [outputs cells] is the outputs of the last of [cells], each run in turn by
   one kernel. *)
let outputs cells =
  let last = ref [] in
  let on_event = function
    | Kernel.Output { output; _ } -> last := output :: !last
    | _ -> ()
  in
  let kernel = Quill_top.create ~on_event () in
  List.iter
    (fun code ->
      last := [];
      kernel.execute ~cell_id:"c" ~code)
    cells;
  kernel.shutdown ();
  List.rev !last

let figure tag = outputs [ printer; Printf.sprintf "Fig %S" tag ]

let kernel_tests =
  [
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

let () = exit (run "Kernel" [ group "Display tags" kernel_tests ])
