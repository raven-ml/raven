(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* ───── Template ───── *)

let default_template =
  {|# Welcome to Quill

Interactive OCaml notebooks — run each cell with **Enter** to see results.
All [Raven](https://github.com/raven-ml) packages are pre-loaded.

## Arrays with Nx

Nx provides n-dimensional arrays, like NumPy for OCaml.

```ocaml
let x = Nx.linspace Nx.float32 0. 5. 6
let y = Nx.sin x
```

## Plotting with Hugin

Hugin renders plots directly in the notebook.

```ocaml
let _fig =
  Hugin.plot ~title:"A sine wave" ~xlabel:"x" ~ylabel:"y"
    (Nx.linspace Nx.float32 0. 6.28 200)
    (Nx.sin (Nx.linspace Nx.float32 0. 6.28 200))
```

## Automatic Differentiation with Rune

Rune computes gradients automatically — define any function and differentiate it.

```ocaml
let f x = Nx.pow_s x 3.                      (* f(x) = x³ *)

let x = Nx.scalar Nx.float32 2.0
let value = f x                                (* f(2) = 8 *)
let gradient = Rune.grad f x                   (* f'(2) = 3·2² = 12 *)
```

## Putting It Together

Plot a function alongside its derivative.

```ocaml
let fig =
  let xs = Nx.linspace Nx.float32 (-2.) 3. 200 in
  let ys = Nx.pow_s xs 3. in
  let gs = Nx.mul_s (Nx.pow_s xs 2.) 3. in
  let fig = Hugin.figure ~width:600 ~height:300 () in
  let ax = Hugin.subplot fig in
  let ax = Hugin.Plotting.plot ~x:xs ~y:ys ~label:"f(x) = x³" ax in
  let ax = Hugin.Plotting.plot ~x:xs ~y:gs ~label:"f'(x) = 3x²" ax in
  let ax = Hugin.Axes.set_xlabel "x" ax in
  let _ = Hugin.Axes.set_ylabel "y" ax in
  fig
```
|}

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

let ensure_file path =
  if not (Sys.file_exists path) then (
    write_file path default_template;
    Printf.printf "Created %s\n%!" path)

let open_browser url =
  let cmd = if Sys.file_exists "/usr/bin/open" then "open" else "xdg-open" in
  ignore (Sys.command (cmd ^ " " ^ Filename.quote url))

(* ───── Run ───── *)

let run_once inplace path =
  let md = read_file path in
  let doc = Quill_markdown.of_string md in
  let doc = Quill.Eval.run ~create_kernel:Quill_raven.create doc in
  let result = Quill_markdown.to_string_with_outputs doc in
  if inplace then (
    write_file path result;
    Printf.printf "Updated %s\n%!" path)
  else print_string result

let run_cmd inplace path =
  if not (Sys.file_exists path) then (
    Printf.eprintf "Error: %s not found\n%!" path;
    exit 1);
  run_once inplace path

(* ───── Watch ───── *)

let get_mtime path =
  try Some (Unix.stat path).Unix.st_mtime with Unix.Unix_error _ -> None

let rec watch_loop path last_mtime =
  Unix.sleepf 1.0;
  match get_mtime path with
  | None ->
      Printf.eprintf "File %s no longer exists\n%!" path;
      exit 1
  | Some mtime when mtime > last_mtime ->
      let tm = Unix.localtime (Unix.gettimeofday ()) in
      Printf.printf "\n[%02d:%02d:%02d] File changed, re-evaluating...\n%!"
        tm.Unix.tm_hour tm.Unix.tm_min tm.Unix.tm_sec;
      run_once true path;
      let new_mtime = Option.value ~default:mtime (get_mtime path) in
      watch_loop path new_mtime
  | Some _ -> watch_loop path last_mtime

let watch_cmd path =
  if not (Sys.file_exists path) then (
    Printf.eprintf "Error: %s not found\n%!" path;
    exit 1);
  run_once true path;
  match get_mtime path with
  | None ->
      Printf.eprintf "Error: Cannot watch %s\n%!" path;
      exit 1
  | Some mtime ->
      Printf.printf "\nWatching %s for changes... (Ctrl-C to stop)\n%!" path;
      watch_loop path mtime

(* ───── Clean ───── *)

let clean_cmd inplace path =
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

(* ───── New ───── *)

let new_cmd path =
  if Sys.file_exists path then (
    Printf.eprintf "Error: %s already exists\n%!" path;
    exit 1);
  write_file path default_template;
  Printf.printf "Created %s\n%!" path

(* ───── Cmdliner ───── *)

open Cmdliner

let required_path_arg =
  Arg.(
    required
    & pos 0 (some string) None
    & info [] ~docv:"FILE" ~doc:"Path to a markdown notebook file.")

let optional_path_arg default =
  Arg.(value & pos 0 string default & info [] ~docv:"FILE")

let inplace_flag =
  Arg.(
    value & flag
    & info [ "inplace"; "i" ] ~doc:"Write changes back into the file.")

let port_flag =
  Arg.(
    value & opt int 8888
    & info [ "port"; "p" ] ~docv:"PORT" ~doc:"Port to listen on (default 8888).")

(* Default: TUI *)
let default_term =
  Term.(
    const (fun path ->
        ensure_file path;
        Quill_tui.run ~create_kernel:Quill_raven.create path)
    $ optional_path_arg "notebook.md")

(* serve: web UI *)
let serve_term =
  let doc = "Start the web notebook server." in
  Cmd.v (Cmd.info "serve" ~doc)
    Term.(
      const (fun port path ->
          ensure_file path;
          let url = Printf.sprintf "http://127.0.0.1:%d" port in
          Quill_httpd.serve ~port ~on_ready:(fun () -> open_browser url) path)
      $ port_flag
      $ optional_path_arg "notebook.md")

(* run: batch execution *)
let run_term =
  let doc = "Execute all code blocks in a notebook." in
  Cmd.v (Cmd.info "run" ~doc)
    Term.(const run_cmd $ inplace_flag $ required_path_arg)

(* watch: live editing *)
let watch_term =
  let doc = "Watch a notebook and re-execute on every save." in
  Cmd.v (Cmd.info "watch" ~doc) Term.(const watch_cmd $ required_path_arg)

(* clean: strip outputs *)
let clean_term =
  let doc = "Strip outputs from a notebook." in
  Cmd.v (Cmd.info "clean" ~doc)
    Term.(const clean_cmd $ inplace_flag $ required_path_arg)

(* new: create notebook *)
let new_term =
  let doc = "Create a new notebook from a starter template." in
  Cmd.v (Cmd.info "new" ~doc)
    Term.(const new_cmd $ optional_path_arg "notebook.md")

let quill_cmd =
  let doc = "Interactive notebooks for OCaml." in
  let info = Cmd.info "quill" ~version:"1.0.0" ~doc in
  Cmd.group ~default:default_term info
    [ serve_term; run_term; watch_term; clean_term; new_term ]

let () = exit (Cmd.eval quill_cmd)
