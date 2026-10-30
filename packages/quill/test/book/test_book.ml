(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Quill

let create_kernel ~on_event:_ = fail "a build that skips evaluation ran a cell"
let read path = In_channel.with_open_bin path In_channel.input_all
let built = result unit string

(* [build path] is the page of the notebook [path], built without running it. *)
let build path =
  let output = temp_dir () in
  equal built (Ok ())
    (Quill_book.Build.build_file ~create_kernel ~skip_eval:true ~output path);
  read
    (Filename.concat output
       (Filename.remove_extension (Filename.basename path) ^ ".html"))

let save path doc =
  equal built (Ok ()) (Result.map ignore (Quill_markdown.save Files path doc))

let figure data = Cell.Display { mime = "image/svg+xml"; id = None; data }

let skip_eval_tests =
  [
    test "embeds a figure stored as a file" (fun () ->
        let path = Filename.concat (temp_dir ()) "nb.md" in
        (* The base64 of "<svg/>". *)
        let data = "PHN2Zy8+" in
        let doc =
          Doc.of_cells
            [
              Cell.code ~id:"c_x" "plot ()" |> Cell.set_outputs [ figure data ];
            ]
        in
        save path doc;
        contains
          ~sub:("<img src=\"data:image/svg+xml;base64," ^ data ^ "\">")
          (build path));
    test "reports a missing figure with its notebook and cell" (fun () ->
        let dir = temp_dir () in
        let path = Filename.concat dir "nb.md" in
        let doc =
          Doc.of_cells
            [
              Cell.code ~id:"c_x" "plot ()"
              |> Cell.set_outputs [ figure "PHN2Zy8+" ];
            ]
        in
        save path doc;
        Sys.remove (Filename.concat dir "figures/nb/c_x.svg");
        equal built
          (Error (path ^ ": cell c_x: figure figures/nb/c_x.svg: no such file"))
          (Quill_book.Build.build_file ~create_kernel ~skip_eval:true
             ~output:(temp_dir ()) path));
  ]

let () = exit (run "Book" [ group "Skip eval" skip_eval_tests ])
