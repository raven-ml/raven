(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Generates assets.ml from a frontend directory.

   Usage: ocaml gen_assets.ml <frontend_dir> <output.ml>

   Reads index.html, dist/ (JS/CSS), and fonts/ (woff2), writes an OCaml module
   with: - [index_html] : the HTML page - [lookup] : maps asset paths to
   contents *)

let read_file path =
  let ic = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in ic)
    (fun () ->
      let len = in_channel_length ic in
      really_input_string ic len)

let walk_dir root =
  let rec aux prefix dir acc =
    let entries = Sys.readdir dir in
    Array.sort String.compare entries;
    Array.fold_left
      (fun acc name ->
        let path = Filename.concat dir name in
        let rel = if prefix = "" then name else prefix ^ "/" ^ name in
        if Sys.is_directory path then aux rel path acc else (rel, path) :: acc)
      acc entries
  in
  List.rev (aux "" root [])

let () =
  let frontend_dir = Sys.argv.(1) in
  let output_path = Sys.argv.(2) in
  let index_path = Filename.concat frontend_dir "index.html" in
  let dist_dir = Filename.concat frontend_dir "dist" in
  let oc = open_out output_path in
  Fun.protect
    ~finally:(fun () -> close_out oc)
    (fun () ->
      Printf.fprintf oc "let index_html = %S\n\n" (read_file index_path);
      let dist_files = walk_dir dist_dir in
      let fonts_dir = Filename.concat frontend_dir "fonts" in
      let font_files = walk_dir fonts_dir in
      let font_files =
        List.map (fun (rel, path) -> ("fonts/" ^ rel, path)) font_files
      in
      Printf.fprintf oc "let lookup = function\n";
      List.iter
        (fun (rel, path) ->
          Printf.fprintf oc "  | %S -> Some %S\n" rel (read_file path))
        (dist_files @ font_files);
      Printf.fprintf oc "  | _ -> None\n")
