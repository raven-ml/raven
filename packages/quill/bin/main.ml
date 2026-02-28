(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* ───── Helpers ───── *)

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

(* ───── Eval ───── *)

let eval_run inplace path =
  let md = read_file path in
  let doc = Quill_markdown.of_string md in
  let doc = Quill.Eval.run ~create_kernel:Quill_raven.create doc in
  let result = Quill_markdown.to_string_with_outputs doc in
  if inplace then (
    write_file path result;
    Printf.printf "Updated %s\n%!" path)
  else print_string result

let get_mtime path =
  try Some (Unix.stat path).Unix.st_mtime with Unix.Unix_error _ -> None

let rec watch_loop path last_mtime eval_fn =
  Unix.sleepf 1.0;
  match get_mtime path with
  | None ->
      Printf.eprintf "File %s no longer exists\n%!" path;
      exit 1
  | Some mtime when mtime > last_mtime ->
      let tm = Unix.localtime (Unix.gettimeofday ()) in
      Printf.printf "\n[%02d:%02d:%02d] File changed, re-evaluating...\n%!"
        tm.Unix.tm_hour tm.Unix.tm_min tm.Unix.tm_sec;
      eval_fn ();
      let new_mtime = Option.value ~default:mtime (get_mtime path) in
      watch_loop path new_mtime eval_fn
  | Some _ -> watch_loop path last_mtime eval_fn

let eval_cmd inplace watch path =
  if not (Sys.file_exists path) then (
    Printf.eprintf "Error: %s not found\n%!" path;
    exit 1);
  eval_run inplace path;
  if watch then (
    match get_mtime path with
    | None ->
        Printf.eprintf "Error: Cannot watch %s\n%!" path;
        exit 1
    | Some mtime ->
        Printf.printf "\nWatching %s for changes... (Ctrl+C to stop)\n%!" path;
        watch_loop path mtime (fun () -> eval_run inplace path))
  else ()

(* ───── Fmt ───── *)

let fmt_cmd inplace path =
  if not (Sys.file_exists path) then (
    Printf.eprintf "Error: %s not found\n%!" path;
    exit 1);
  let md = read_file path in
  let doc = Quill_markdown.of_string md in
  let doc = Quill.Doc.clear_all_outputs doc in
  let result = Quill_markdown.to_string doc in
  if inplace then (
    write_file path result;
    Printf.printf "Stripped outputs from %s\n%!" path)
  else print_string result

(* ───── Cmdliner ───── *)

open Cmdliner

let path_arg =
  Arg.(
    required
    & pos 0 (some string) None
    & info [] ~docv:"FILE" ~doc:"Path to a markdown notebook file.")

let inplace_flag =
  Arg.(
    value & flag
    & info [ "inplace"; "i" ] ~doc:"Write changes back into the file.")

let watch_flag =
  Arg.(
    value & flag
    & info [ "watch"; "w" ]
        ~doc:"Watch for file changes and re-evaluate automatically.")

let default_term =
  Term.(
    const (fun path -> Quill_tui.run ~create_kernel:Quill_raven.create path)
    $ path_arg)

let eval_term =
  let doc = "Evaluate code blocks in a notebook." in
  Cmd.v (Cmd.info "eval" ~doc)
    Term.(const eval_cmd $ inplace_flag $ watch_flag $ path_arg)

let fmt_term =
  let doc = "Strip outputs from a notebook." in
  Cmd.v (Cmd.info "fmt" ~doc) Term.(const fmt_cmd $ inplace_flag $ path_arg)

let port_flag =
  Arg.(
    value & opt int 8888
    & info [ "port"; "p" ] ~docv:"PORT" ~doc:"Port to listen on (default 8888).")

let serve_term =
  let doc = "Start the web notebook server." in
  Cmd.v (Cmd.info "serve" ~doc)
    Term.(
      const (fun port path -> Quill_httpd.serve ~port path)
      $ port_flag $ path_arg)

let quill_cmd =
  let doc = "Interactive notebooks for OCaml." in
  let info = Cmd.info "quill" ~version:"1.0.0" ~doc in
  Cmd.group ~default:default_term info [ eval_term; fmt_term; serve_term ]

let () = exit (Cmd.eval quill_cmd)
