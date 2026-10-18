(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

let write path s = Out_channel.with_open_bin path (fun oc -> output_string oc s)
let read path = In_channel.with_open_bin path In_channel.input_all

let contains ~sub s =
  let n = String.length sub in
  let rec go i =
    i + n <= String.length s && (String.sub s i n = sub || go (i + 1))
  in
  go 0

let temp_dir () =
  let d = Filename.temp_file "quill_book" "" in
  Sys.remove d;
  Sys.mkdir d 0o755;
  d

(* A notebook run with [quill run --figures-dir figures] stores its figure
   beside it and refers to it from the output. *)
let notebook =
  {|# Figures

<!-- quill:cell id="c_fig" -->
```ocaml
let fig = ()
```
<!-- quill:output -->
<!-- out:display image/svg+xml -->
<img src="figures/c_fig.svg">
<!-- /quill:output -->
|}

let no_kernel ~on_event:_ = failwith "skip_eval runs no kernel"

let tests =
  [
    test "a figure stored on disk is embedded" (fun () ->
        let dir = temp_dir () in
        Sys.mkdir (Filename.concat dir "figures") 0o755;
        write (Filename.concat dir "figures/c_fig.svg") "<svg/>";
        let nb = Filename.concat dir "notebook.md" in
        write nb notebook;
        Quill_book.Build.build_file ~create_kernel:no_kernel ~skip_eval:true
          ~output:dir nb;
        let html = read (Filename.concat dir "notebook.html") in
        (* base64 of "<svg/>" *)
        is_true (contains ~sub:"data:image/svg+xml;base64,PHN2Zy8+" html));
  ]

let () = exit (run "Book" [ group "Figures" tests ])
