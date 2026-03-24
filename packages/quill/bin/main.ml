(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let raven_packages =
  [
    "nx.c";
    "nx.io";
    "rune";
    "vega";
    "norn";
    "kaun";
    "kaun.datasets";
    "hugin";
    "sowilo";
    "talon";
    "talon.csv";
    "brot";
    "fehu";
  ]

let raven_printers = [ "Nx.pp_data"; "Hugin.pp"; "Talon.pp_display" ]

let load_optional pkg =
  match Quill_top.load_package pkg with
  | () -> true
  | exception Fl_package_base.No_such_package _ -> false
  | exception exn ->
      Printf.eprintf "[quill] failed to load %s: %s\n%!" pkg
        (Printexc.to_string exn);
      false

let setup () =
  (* Mark packages already linked into the quill executable so that load_package
     does not try to load their .cma archives again. *)
  Quill_top.add_packages
    [
      "compiler-libs";
      "compiler-libs.common";
      "compiler-libs.bytecomp";
      "compiler-libs.toplevel";
      "findlib";
      "findlib.internal";
      "unix";
      "threads";
      "threads.posix";
    ];
  (* Load raven packages individually. We skip the .top packages (nx.top,
     hugin.top) — they only install printers during module init, which fails
     inside dir_load. We install printers ourselves below. *)
  List.iter (fun pkg -> ignore (load_optional pkg)) raven_packages;
  List.iter Quill_top.install_printer raven_printers

let create_kernel ~on_event = Quill_top.create ~setup ~on_event ()

(* ───── Template ───── *)

let default_template =
  {|# Welcome to Quill

Interactive OCaml notebooks — run each cell with **Enter** to see results.
[Raven](https://github.com/raven-ml) packages are loaded automatically when installed.

## Arrays with Nx

Nx provides n-dimensional arrays, like NumPy for OCaml.

```ocaml
let x = Nx.linspace Nx.float32 0. 5. 6
let y = Nx.sin x
```

## Plotting with Hugin

Hugin renders plots directly in the notebook.

```ocaml
let x = Nx.linspace Nx.float32 0. 6.28 200
let y = Nx.sin x

let _fig =
  Hugin.line ~x ~y ()
  |> Hugin.title "A sine wave"
  |> Hugin.xlabel "x" |> Hugin.ylabel "y"
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
let f x = Nx.pow_s x 3.

let xs = Nx.linspace Nx.float32 (-2.) 3. 200
let ys = f xs
let gs = Rune.grad f xs

let _fig =
  Hugin.layers [
    Hugin.line ~x:xs ~y:ys ~label:"f(x) = x³" ();
    Hugin.line ~x:xs ~y:gs ~label:"f'(x) = 3x²" ();
  ]
  |> Hugin.xlabel "x" |> Hugin.ylabel "y"
  |> Hugin.legend
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

let resolve_path path =
  if Filename.is_relative path then Filename.concat (Sys.getcwd ()) path
  else path

let ensure_file path =
  if not (Sys.file_exists path) then (
    write_file path default_template;
    Printf.printf "Created %s\n%!" path)

let open_browser url =
  let cmd = if Sys.file_exists "/usr/bin/open" then "open" else "xdg-open" in
  ignore (Sys.command (cmd ^ " " ^ Filename.quote url))

(* ───── Project loading ───── *)

let is_dir path = Sys.file_exists path && Sys.is_directory path

let discover_notebooks dir =
  let entries = Sys.readdir dir in
  let mds =
    Array.to_list entries
    |> List.filter (fun name -> Filename.check_suffix name ".md")
    |> List.sort String.compare
  in
  List.map
    (fun name ->
      let title = Quill_project.title_of_filename name in
      Quill_project.Notebook ({ title; path = name }, []))
    mds

let load_project dir =
  let conf_path = Filename.concat dir "quill.conf" in
  if Sys.file_exists conf_path then (
    let source = read_file conf_path in
    match Quill_project.parse_config source with
    | Ok (config, toc) ->
        let title =
          match config.title with Some t -> t | None -> Filename.basename dir
        in
        { Quill_project.title; root = dir; toc; config }
    | Error msg ->
        Printf.eprintf "Error: %s\n%!" msg;
        exit 1)
  else
    let toc = discover_notebooks dir in
    let title = Filename.basename dir in
    {
      Quill_project.title;
      root = dir;
      toc;
      config = Quill_project.default_config;
    }

let scratch_dir =
  match Sys.getenv_opt "XDG_DATA_HOME" with
  | Some dir -> Filename.concat dir "quill"
  | None ->
      Filename.concat
        (Filename.concat (Sys.getenv "HOME") ".local/share")
        "quill"

let scratch_path = Filename.concat scratch_dir "scratch.md"

let ensure_scratch_dir () =
  if not (Sys.file_exists scratch_dir) then
    ignore
      (Sys.command (Printf.sprintf "mkdir -p %s" (Filename.quote scratch_dir)))

(* ───── Default: TUI notebook ───── *)

let default_cmd path =
  let path =
    match path with
    | Some p -> p
    | None ->
        ensure_scratch_dir ();
        scratch_path
  in
  let abs_path = resolve_path path in
  Sys.chdir (Filename.dirname abs_path);
  Quill_tui.run ~create_kernel abs_path

(* ───── Serve: web notebook ───── *)

let serve_notebook port path =
  let abs_path = resolve_path path in
  ensure_file abs_path;
  Sys.chdir (Filename.dirname abs_path);
  let url = Printf.sprintf "http://127.0.0.1:%d" port in
  Quill_server.serve ~create_kernel ~port
    ~on_ready:(fun () -> open_browser url)
    abs_path

let serve_project port project =
  let prelude nb_path =
    let nb_dir =
      Filename.concat project.Quill_project.root (Filename.dirname nb_path)
    in
    let path = Filename.concat nb_dir "prelude.ml" in
    if Sys.file_exists path then Some (read_file path) else None
  in
  let url = Printf.sprintf "http://127.0.0.1:%d" port in
  Quill_server.serve_dir ~create_kernel ~port ~prelude ~toc:project.toc
    ~on_ready:(fun () -> open_browser url)
    project.root

let serve_cmd port path =
  if is_dir path then serve_project port (load_project path)
  else serve_notebook port path

(* ───── Run: batch execution ───── *)

let run_file ?prelude ?figures_dir inplace path =
  let abs_path = resolve_path path in
  let abs_prelude = Option.map resolve_path prelude in
  let nb_dir = Filename.dirname abs_path in
  let figures_dir =
    Option.map
      (fun d -> if Filename.is_relative d then Filename.concat nb_dir d else d)
      figures_dir
  in
  let md = read_file abs_path in
  let doc = Quill_markdown.of_string md in
  let create_kernel ~on_event =
    let k = create_kernel ~on_event in
    (match abs_prelude with
    | Some p ->
        let code = read_file p in
        k.Quill.Kernel.execute ~cell_id:"__prelude__" ~code
    | None -> ());
    k
  in
  let doc = Quill.Doc.clear_all_outputs doc in
  let prev_cwd = Sys.getcwd () in
  Sys.chdir nb_dir;
  let doc =
    Fun.protect
      ~finally:(fun () -> Sys.chdir prev_cwd)
      (fun () -> Quill.Eval.run ~create_kernel doc)
  in
  let result = Quill_markdown.to_string_with_outputs ?figures_dir doc in
  if inplace then (
    write_file abs_path result;
    Printf.printf "Updated %s\n%!" abs_path)
  else print_string result

let get_mtime path =
  try Some (Unix.stat path).Unix.st_mtime with Unix.Unix_error _ -> None

let rec watch_loop ?prelude ?figures_dir path last_mtime =
  Unix.sleepf 1.0;
  match get_mtime path with
  | None ->
      Printf.eprintf "File %s no longer exists\n%!" path;
      exit 1
  | Some mtime when mtime > last_mtime ->
      let tm = Unix.localtime (Unix.gettimeofday ()) in
      Printf.printf "\n[%02d:%02d:%02d] File changed, re-evaluating...\n%!"
        tm.Unix.tm_hour tm.Unix.tm_min tm.Unix.tm_sec;
      run_file ?prelude ?figures_dir true path;
      let new_mtime = Option.value ~default:mtime (get_mtime path) in
      watch_loop ?prelude ?figures_dir path new_mtime
  | Some _ -> watch_loop ?prelude ?figures_dir path last_mtime

let run_project ?prelude ?figures_dir project =
  List.iter
    (fun (nb : Quill_project.notebook) ->
      let path = Filename.concat project.Quill_project.root nb.path in
      if Sys.file_exists path then (
        Printf.printf "  Running %s...\n%!" nb.title;
        run_file ?prelude ?figures_dir true path))
    (Quill_project.notebooks project)

let run_cmd watch inplace prelude figures_dir path =
  if not (Sys.file_exists path) then (
    Printf.eprintf "Error: %s not found\n%!" path;
    exit 1);
  if is_dir path then run_project ?prelude ?figures_dir (load_project path)
  else if watch then begin
    run_file ?prelude ?figures_dir true path;
    match get_mtime path with
    | None ->
        Printf.eprintf "Error: Cannot watch %s\n%!" path;
        exit 1
    | Some mtime ->
        Printf.printf "\nWatching %s for changes... (Ctrl-C to stop)\n%!" path;
        watch_loop ?prelude ?figures_dir path mtime
  end
  else run_file ?prelude ?figures_dir inplace path

(* ───── Build: static HTML ───── *)

let build_cmd skip_eval output path =
  if not (Sys.file_exists path) then (
    Printf.eprintf "Error: %s not found\n%!" path;
    exit 1);
  if is_dir path then
    let project = load_project path in
    Quill_book.Build.build ~create_kernel ~skip_eval ?output project
  else Quill_book.Build.build_file ~create_kernel ~skip_eval ?output path

(* ───── Clean: strip outputs ───── *)

let rec rm_rf path =
  if Sys.file_exists path then
    if Sys.is_directory path then (
      Array.iter
        (fun name -> rm_rf (Filename.concat path name))
        (Sys.readdir path);
      Unix.rmdir path)
    else Sys.remove path

let clean_figures_dir dir =
  let figures = Filename.concat dir "figures" in
  if Sys.file_exists figures && Sys.is_directory figures then rm_rf figures

let clean_cmd path =
  if not (Sys.file_exists path) then (
    Printf.eprintf "Error: %s not found\n%!" path;
    exit 1);
  if is_dir path then begin
    let project = load_project path in
    List.iter
      (fun (nb : Quill_project.notebook) ->
        let path = Filename.concat project.root nb.path in
        if Sys.file_exists path then (
          let md = read_file path in
          let doc = Quill_markdown.of_string md in
          let doc = Quill.Doc.clear_all_outputs doc in
          let result = Quill_markdown.to_string doc in
          write_file path result;
          let nb_dir = Filename.dirname path in
          clean_figures_dir nb_dir;
          Printf.printf "  Cleaned %s\n%!" nb.title))
      (Quill_project.notebooks project);
    Printf.printf "Done.\n%!"
  end
  else begin
    let md = read_file path in
    let doc = Quill_markdown.of_string md in
    let doc = Quill.Doc.clear_all_outputs doc in
    let result = Quill_markdown.to_string doc in
    write_file path result;
    let nb_dir = Filename.dirname path in
    clean_figures_dir nb_dir;
    Printf.printf "Stripped outputs from %s\n%!" path
  end

(* ───── Cmdliner ───── *)

open Cmdliner

let optional_path_arg =
  Arg.(
    value
    & pos 0 (some string) None
    & info [] ~docv:"FILE"
        ~doc:"Path to a notebook file. If omitted, opens a scratch notebook.")

let serve_path_arg =
  Arg.(
    value & pos 0 string "notebook.md"
    & info [] ~docv:"PATH"
        ~doc:
          "Path to a notebook file or project directory (contains quill.conf).")

let required_path_arg =
  Arg.(
    required
    & pos 0 (some string) None
    & info [] ~docv:"PATH"
        ~doc:
          "Path to a notebook file or project directory (contains quill.conf).")

let port_flag =
  Arg.(
    value & opt int 8888
    & info [ "port"; "p" ] ~docv:"PORT" ~doc:"Port to listen on (default 8888).")

let watch_flag =
  Arg.(
    value & flag & info [ "watch"; "w" ] ~doc:"Re-execute on every file save.")

let inplace_flag =
  Arg.(
    value & flag
    & info [ "inplace"; "i" ] ~doc:"Write changes back into the file.")

let prelude_flag =
  Arg.(
    value
    & opt (some string) None
    & info [ "prelude" ] ~docv:"FILE"
        ~doc:"Execute OCaml code from $(docv) before the notebook cells.")

let figures_dir_flag =
  Arg.(
    value
    & opt (some string) None
    & info [ "figures-dir" ] ~docv:"DIR"
        ~doc:
          "Write images to $(docv) and reference by path instead of inlining.")

let skip_eval_flag =
  Arg.(
    value & flag
    & info [ "skip-eval" ]
        ~doc:"Render HTML from existing outputs without re-executing code.")

let output_flag =
  Arg.(
    value
    & opt (some string) None
    & info [ "output"; "o" ] ~docv:"DIR"
        ~doc:"Output directory (default: build/ inside the project directory).")

(* Default: TUI notebook *)
let default_term = Term.(const default_cmd $ optional_path_arg)

(* serve: web notebook *)
let serve_term =
  let doc = "Open a notebook or project in the browser." in
  Cmd.v (Cmd.info "serve" ~doc)
    Term.(const serve_cmd $ port_flag $ serve_path_arg)

(* run: batch execution *)
let run_term =
  let doc = "Execute all code blocks in a notebook or project." in
  Cmd.v (Cmd.info "run" ~doc)
    Term.(
      const run_cmd $ watch_flag $ inplace_flag $ prelude_flag
      $ figures_dir_flag $ required_path_arg)

(* build: static HTML *)
let build_term =
  let doc = "Build a notebook or project as static HTML." in
  Cmd.v (Cmd.info "build" ~doc)
    Term.(const build_cmd $ skip_eval_flag $ output_flag $ required_path_arg)

(* clean: strip outputs *)
let clean_term =
  let doc = "Strip outputs from a notebook or all project notebooks." in
  Cmd.v (Cmd.info "clean" ~doc) Term.(const clean_cmd $ required_path_arg)

let subcommands = [ serve_term; run_term; build_term; clean_term ]
let known_commands = List.map Cmd.name subcommands

let quill_cmd =
  let doc = "Interactive OCaml notebooks." in
  let info = Cmd.info "quill" ~version:"1.0.0" ~doc in
  Cmd.group ~default:default_term info subcommands

let () =
  (* cmdliner's Cmd.group matches the first positional arg against subcommand
     names before falling through to the default term. Pre-parse argv to insert
     "--" when the first arg is not a known subcommand, so that [quill file.md]
     works without requiring [quill -- file.md]. *)
  let argv =
    let a = Sys.argv in
    if
      Array.length a >= 2
      && String.length a.(1) > 0
      && a.(1).[0] <> '-'
      && not (List.mem a.(1) known_commands)
    then Array.concat [ [| a.(0); "--" |]; Array.sub a 1 (Array.length a - 1) ]
    else a
  in
  exit (Cmd.eval ~argv quill_cmd)
