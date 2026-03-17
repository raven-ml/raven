(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* ───── File utilities ───── *)

let read_file path =
  let ic = open_in path in
  Fun.protect
    ~finally:(fun () -> close_in ic)
    (fun () -> really_input_string ic (in_channel_length ic))

let write_file path content =
  let oc = open_out path in
  Fun.protect
    ~finally:(fun () -> close_out oc)
    (fun () -> output_string oc content)

let rec mkdir_p dir =
  if Sys.file_exists dir then ()
  else (
    mkdir_p (Filename.dirname dir);
    try Unix.mkdir dir 0o755 with Unix.Unix_error (Unix.EEXIST, _, _) -> ())

let copy_file ~src ~dst =
  let ic = open_in_bin src in
  Fun.protect
    ~finally:(fun () -> close_in ic)
    (fun () ->
      let oc = open_out_bin dst in
      Fun.protect
        ~finally:(fun () -> close_out oc)
        (fun () ->
          let buf = Bytes.create 8192 in
          let rec loop () =
            let n = input ic buf 0 8192 in
            if n > 0 then (
              output oc buf 0 n;
              loop ())
          in
          loop ()))

let rec copy_dir_contents ~src_dir ~dst_dir =
  if Sys.file_exists src_dir && Sys.is_directory src_dir then (
    mkdir_p dst_dir;
    let entries = Sys.readdir src_dir in
    Array.iter
      (fun name ->
        let src = Filename.concat src_dir name in
        let dst = Filename.concat dst_dir name in
        if not (Sys.is_directory src) then copy_file ~src ~dst
        else copy_dir_contents ~src_dir:src ~dst_dir:dst)
      entries)

(* ───── Path computation ───── *)

let notebook_dir (project : Quill_project.t) (nb : Quill_project.notebook) =
  let dir = Filename.dirname nb.path in
  if dir = "." then project.root else Filename.concat project.root dir

let prelude_path (project : Quill_project.t) (nb : Quill_project.notebook) =
  let dir = notebook_dir project nb in
  let path = Filename.concat dir "prelude.ml" in
  if Sys.file_exists path then Some path else None

let relative_root_path (nb : Quill_project.notebook) =
  let dir = Filename.dirname nb.path in
  if dir = "." then "./"
  else
    let parts = String.split_on_char '/' dir in
    let depth =
      List.length (List.filter (fun s -> s <> "" && s <> ".") parts)
    in
    if depth = 0 then "./"
    else String.concat "" (List.init depth (fun _ -> "../"))

(* ───── Build ───── *)

let build_notebook ~create_kernel ~skip_eval ~output_dir ~live_reload_script
    (project : Quill_project.t) (nb : Quill_project.notebook) =
  let nb_path = Filename.concat project.root nb.path in
  let nb_dir = notebook_dir project nb in
  let md = read_file nb_path in
  let doc = Quill_markdown.of_string md in
  let doc =
    if skip_eval then doc
    else
      let create_kernel ~on_event =
        let k = create_kernel ~on_event in
        (match prelude_path project nb with
        | Some p ->
            let code = read_file p in
            k.Quill.Kernel.execute ~cell_id:"__prelude__" ~code
        | None -> ());
        k
      in
      let prev_cwd = Sys.getcwd () in
      Sys.chdir nb_dir;
      Fun.protect
        ~finally:(fun () -> Sys.chdir prev_cwd)
        (fun () ->
          let doc = Quill.Doc.clear_all_outputs doc in
          Quill.Eval.run ~create_kernel doc)
  in
  let content = Render.chapter_html doc in
  let root_path = relative_root_path nb in
  let toc = Render.toc_html project ~current:nb ~root_path in
  let prev =
    match Quill_project.prev_notebook project nb with
    | Some p -> Some (root_path ^ Render.notebook_output_path p, p.title)
    | None -> None
  in
  let next =
    match Quill_project.next_notebook project nb with
    | Some n -> Some (root_path ^ Render.notebook_output_path n, n.title)
    | None -> None
  in
  let edit_url =
    match project.config.edit_url with
    | Some base -> Some (base ^ nb.path)
    | None -> None
  in
  let html =
    Render.page_html ~book_title:project.title ~chapter_title:nb.title
      ~toc_html:toc ~prev ~next ~root_path ~content ~edit_url
      ~live_reload_script
  in
  let output_path =
    Filename.concat output_dir (Render.notebook_output_path nb)
  in
  mkdir_p (Filename.dirname output_path);
  write_file output_path html;
  let asset_dirs = [ "figures"; "images"; "assets" ] in
  List.iter
    (fun name ->
      let src = Filename.concat nb_dir name in
      let dst =
        Filename.concat output_dir
          (Filename.concat (Filename.dirname nb.path) name)
      in
      copy_dir_contents ~src_dir:src ~dst_dir:dst)
    asset_dirs;
  Printf.printf "  %s\n%!" nb.title;
  content

(* ───── Search index ───── *)

let json_escape_string s =
  let buf = Buffer.create (String.length s + 16) in
  Buffer.add_char buf '"';
  String.iter
    (function
      | '"' -> Buffer.add_string buf {|\"|}
      | '\\' -> Buffer.add_string buf {|\\|}
      | '\n' -> Buffer.add_string buf {|\n|}
      | '\r' -> Buffer.add_string buf {|\r|}
      | '\t' -> Buffer.add_string buf {|\t|}
      | c -> Buffer.add_char buf c)
    s;
  Buffer.add_char buf '"';
  Buffer.contents buf

let search_entry ~title ~url ~body =
  Printf.sprintf {|{"title":%s,"url":%s,"body":%s}|} (json_escape_string title)
    (json_escape_string url) (json_escape_string body)

let build_search_index ~output_dir ~toc
    (notebooks : (Quill_project.notebook * string) list) =
  let entries =
    List.map
      (fun (nb, content_html) ->
        let number_prefix =
          match Quill_project.number_string (Quill_project.number toc nb) with
          | "" -> ""
          | s -> s ^ ". "
        in
        let title = number_prefix ^ nb.title in
        let url = Render.notebook_output_path nb in
        let body = Render.strip_html_tags content_html in
        search_entry ~title ~url ~body)
      notebooks
  in
  let json = "[" ^ String.concat "," entries ^ "]" in
  write_file (Filename.concat output_dir "searchindex.json") json

let build_print_page ~output_dir ~toc (project : Quill_project.t)
    (notebooks : (Quill_project.notebook * string) list) =
  let chapter_pairs =
    List.map
      (fun (nb, content_html) ->
        let number_prefix =
          match Quill_project.number_string (Quill_project.number toc nb) with
          | "" -> ""
          | s -> s ^ ". "
        in
        (number_prefix ^ nb.title, content_html))
      notebooks
  in
  let html =
    Render.print_page_html ~book_title:project.title ~chapters:chapter_pairs
  in
  write_file (Filename.concat output_dir "print.html") html

let build_index ~output_dir (project : Quill_project.t) ~live_reload_script =
  match Quill_project.notebooks project with
  | [] -> ()
  | first :: _ ->
      let url = Render.notebook_output_path first in
      let html =
        Printf.sprintf
          {|<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="0; url=%s">
<title>%s</title>
</head>
<body><p>Redirecting to <a href="%s">%s</a>...</p>%s</body>
</html>|}
          (Render.escape_html url)
          (Render.escape_html project.title)
          (Render.escape_html url)
          (Render.escape_html first.title)
          live_reload_script
      in
      write_file (Filename.concat output_dir "index.html") html

let build ~create_kernel ?(skip_eval = false) ?output ?(live_reload_script = "")
    (project : Quill_project.t) =
  let output_dir =
    match output with
    | Some dir -> dir
    | None -> Filename.concat project.root "build"
  in
  mkdir_p output_dir;
  write_file (Filename.concat output_dir "style.css") Theme.style_css;
  let nbs = Quill_project.notebooks project in
  Printf.printf "Building %s (%d notebooks)\n%!" project.title (List.length nbs);
  let notebook_contents =
    List.map
      (fun nb ->
        let content =
          build_notebook ~create_kernel ~skip_eval ~output_dir
            ~live_reload_script project nb
        in
        (nb, content))
      nbs
  in
  build_search_index ~output_dir ~toc:project.toc notebook_contents;
  build_print_page ~output_dir ~toc:project.toc project notebook_contents;
  build_index ~output_dir project ~live_reload_script;
  Printf.printf "Done → %s\n%!" output_dir

let build_file ~create_kernel ?(skip_eval = false) ?output
    ?(live_reload_script = "") path =
  let abs_path =
    if Filename.is_relative path then Filename.concat (Sys.getcwd ()) path
    else path
  in
  let nb_dir = Filename.dirname abs_path in
  let basename = Filename.basename abs_path in
  let title = Quill_project.title_of_filename basename in
  let md = read_file abs_path in
  let doc = Quill_markdown.of_string md in
  let doc =
    if skip_eval then doc
    else
      let create_kernel ~on_event =
        let k = create_kernel ~on_event in
        let prelude = Filename.concat nb_dir "prelude.ml" in
        (if Sys.file_exists prelude then
           let code = read_file prelude in
           k.Quill.Kernel.execute ~cell_id:"__prelude__" ~code);
        k
      in
      let prev_cwd = Sys.getcwd () in
      Sys.chdir nb_dir;
      Fun.protect
        ~finally:(fun () -> Sys.chdir prev_cwd)
        (fun () ->
          let doc = Quill.Doc.clear_all_outputs doc in
          Quill.Eval.run ~create_kernel doc)
  in
  let content = Render.chapter_html doc in
  let html =
    Render.standalone_page_html ~title ~content ~live_reload_script
  in
  let output_dir =
    match output with Some dir -> dir | None -> nb_dir
  in
  mkdir_p output_dir;
  let html_name = Filename.remove_extension basename ^ ".html" in
  let output_path = Filename.concat output_dir html_name in
  write_file output_path html;
  if output_dir <> nb_dir then begin
    let asset_dirs = [ "figures"; "images"; "assets" ] in
    List.iter
      (fun name ->
        let src = Filename.concat nb_dir name in
        let dst = Filename.concat output_dir name in
        copy_dir_contents ~src_dir:src ~dst_dir:dst)
      asset_dirs
  end;
  Printf.printf "Built %s → %s\n%!" title output_path
