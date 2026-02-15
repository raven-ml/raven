(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let open_temp_file prefix suffix =
  let filename = Filename.temp_file prefix suffix in
  let fd = Unix.openfile filename Unix.[ O_WRONLY; O_CREAT; O_TRUNC ] 0o600 in
  (fd, filename)

let read_all_file filename =
  try
    let ic = open_in filename in
    let len = in_channel_length ic in
    let buf = Buffer.create len in
    Buffer.add_channel buf ic len;
    close_in ic;
    Buffer.contents buf
  with _ -> ""

let capture_separated f =
  let stdout_backup = Unix.dup ~cloexec:true Unix.stdout in
  let stderr_backup = Unix.dup ~cloexec:true Unix.stderr in
  let fd_out, fname_out = open_temp_file "quill-out-" ".tmp" in
  let fd_err, fname_err = open_temp_file "quill-err-" ".tmp" in

  let ppf_out =
    Format.formatter_of_out_channel (Unix.out_channel_of_descr fd_out)
  in
  let ppf_err =
    Format.formatter_of_out_channel (Unix.out_channel_of_descr fd_err)
  in

  let result = ref None in
  Fun.protect
    (fun () ->
      flush stdout;
      flush stderr;
      Unix.dup2 ~cloexec:false fd_out Unix.stdout;
      Unix.dup2 ~cloexec:false fd_err Unix.stderr;
      result := Some (f ppf_out ppf_err))
    ~finally:(fun () ->
      Format.pp_print_flush ppf_out ();
      Format.pp_print_flush ppf_err ();
      flush stdout;
      flush stderr;

      Unix.close fd_out;
      Unix.close fd_err;

      Unix.dup2 ~cloexec:false stdout_backup Unix.stdout;
      Unix.dup2 ~cloexec:false stderr_backup Unix.stderr;
      Unix.close stdout_backup;
      Unix.close stderr_backup);

  let captured_output = read_all_file fname_out in
  let captured_error = read_all_file fname_err in

  (try Sys.remove fname_out with _ -> ());
  (try Sys.remove fname_err with _ -> ());

  match !result with
  | None -> failwith "Capture logic failed unexpectedly"
  | Some success_status ->
      if success_status then
        {
          Quill_top.output = captured_output;
          error = (if captured_error = "" then None else Some captured_error);
          status = `Success;
        }
      else
        (* On error, combine stdout and stderr to get full error context *)
        let combined_error =
          let parts =
            List.filter (fun s -> s <> "") [ captured_output; captured_error ]
          in
          String.concat "\n" parts
        in
        { Quill_top.output = ""; error = Some combined_error; status = `Error }

let initialized = ref false
let initialization_mutex = Mutex.create ()

let format_error_to_string prefix formatter_fn =
  let buf = Buffer.create 512 in
  let fmt = Format.formatter_of_buffer buf in
  Format.fprintf fmt "%s@.@." prefix;
  formatter_fn fmt;
  Format.pp_print_flush fmt ();
  Buffer.contents buf

let execute_directive directive =
  try
    let lexbuf = Lexing.from_string directive in
    let phrases = !Toploop.parse_use_file lexbuf in
    List.iter
      (fun phrase ->
        let result = Toploop.execute_phrase true Format.err_formatter phrase in
        if not result then
          Printf.eprintf "Failed to execute directive: %s\n%!" directive)
      phrases
  with ex ->
    Printf.eprintf "Exception executing directive '%s': %s\n%!" directive
      (Printexc.to_string ex);
    raise ex

let load_plugins () =
  try
    let plugins_locations = Quill_sites.Sites.toplevel_libs in

    let lookup_file filename =
      List.find_map
        (fun dir ->
          let filename' = Filename.concat dir filename in
          if Sys.file_exists filename' then Some filename' else None)
        plugins_locations
    in

    let cmas =
      (* Standard library modules that don't need to be looked up *)
      let stdlib_cmas = [ "unix.cma" ] in

      (* Our project libraries *)
      let project_cmas =
        [
          "nx_core.cma";
          "nx.cma";
          "nx_top.cma";
          "nx_c.cma";
          "bigarray_compat.cma";
          "integers.cma";
          "ctypes.cma";
          "ctypes_foreign.cma";
          "objc_c.cma";
          "objc.cma";
          "zip.cma";
          "npy.cma";
          "stb_image.cma";
          "stb_image_write.cma";
          "nx_io.cma";
          "csv.cma";
          "nx_datasets.cma";
          "re.cma";
          "uucp.cma";
          "saga.cma";
          "cairo.cma";
          "usdl.cma";
          "logs.cma";
          "hugin.cma";
          "hugin_top.cma";
          "rune.cma";
          "rune_top.cma";
          "sowilo.cma";
          "kaun.cma";
          "kaun_datasets.cma";
        ]
      in

      let all_cmas =
        stdlib_cmas
        @ List.filter_map
            (fun name ->
              match lookup_file name with
              | Some path -> Some path
              | None ->
                  Printf.eprintf "Warning: %s not found\n%!" name;
                  None)
            project_cmas
      in

      Printf.eprintf "Found %d CMA files to load\n%!" (List.length all_cmas);
      all_cmas
    in

    if plugins_locations = [] then (
      let error_msg =
        "No site directories found for 'quill.toplevel_libs'. Check \
         installation."
      in
      Printf.eprintf "ERROR: %s\n%!" error_msg;
      failwith error_msg)
    else (
      Printf.eprintf "Plugin locations: %s\n%!"
        (String.concat ", " plugins_locations);
      (* Add directories to search path *)
      List.iter Topdirs.dir_directory plugins_locations;

      (* Also add subdirectories for libraries with dots in their public_name *)
      List.iter
        (fun dir ->
          let kaun_dir = Filename.concat (Filename.dirname dir) "kaun" in
          if Sys.file_exists kaun_dir then
            let datasets_dir = Filename.concat kaun_dir "datasets" in
            if Sys.file_exists datasets_dir then
              Topdirs.dir_directory datasets_dir)
        plugins_locations;

      (* Load CMA files *)
      List.iter
        (fun cma -> execute_directive (Printf.sprintf "#load %S;;" cma))
        cmas;

      (* Set up a simple Logs reporter without ANSI codes *)
      (* Only set up logs if it's available *)
      let () =
        try
          execute_directive
            {|
(* Set up a simple Logs reporter without ANSI codes *)
let setup_logs () =
  let report src level ~over k msgf =
    let k _ = over (); k () in
    msgf @@ fun ?header:_ ?tags:_ fmt ->
    Format.kfprintf k Format.std_formatter ("[%s] %s: " ^^ fmt ^^ "@.")
      (Logs.Src.name src)
      (Logs.level_to_string (Some level))
  in
  { Logs.report }
;;
Logs.set_reporter (setup_logs ());;
|}
        with _ -> Printf.eprintf "Warning: Could not set up Logs reporter\n%!"
      in
      ())
  with
  | Env.Error e ->
      (* Create a more descriptive error message *)
      failwith
        (Format.asprintf "Plugin loading failed with environment error: %a"
           (fun fmt err ->
             (* OCaml 5.4+ removed Env.report_error, use generic formatting *)
             Format.fprintf fmt "%s" (Printexc.to_string (Env.Error err)))
           e)
  | Typecore.Error (loc, env, err) ->
      (* Create a more descriptive error message *)
      failwith
        (format_error_to_string "Plugin loading failed with type error:"
           (fun fmt ->
             let report = Typecore.report_error ~loc env err in
             Location.print_report fmt report))
  | ex ->
      Printf.eprintf "Error during plugin loading: %s\n%!"
        (Printexc.to_string ex);
      raise ex

let initialize_if_needed () =
  Mutex.lock initialization_mutex;
  Fun.protect
    ~finally:(fun () -> Mutex.unlock initialization_mutex)
    (fun () ->
      if not !initialized then (
        (* Perform initialization steps *)
        Quill_top.initialize_toplevel ();
        (* Load plugins - this will raise an exception on failure *)
        load_plugins ();
        (* Only set initialized to true if we get here without exceptions *)
        initialized := true))

let eval ?(print_all = true) code : Quill_top.execution_result =
  try
    initialize_if_needed ();
    capture_separated (fun ppf_out ppf_err ->
        Quill_top.execute print_all ppf_out ppf_err code)
  with
  | Failure msg when String.starts_with ~prefix:"Plugin loading failed" msg ->
      (* Plugin loading error with formatted message *)
      { Quill_top.output = ""; error = Some msg; status = `Error }
  | Env.Error e ->
      (* Format the environment error properly *)
      let error_msg =
        Format.asprintf "Toplevel initialization failed: Environment error: %s"
          (Printexc.to_string (Env.Error e))
      in
      { Quill_top.output = ""; error = Some error_msg; status = `Error }
  | Typecore.Error (loc, env, err) ->
      (* Format the type error properly *)
      let error_msg =
        format_error_to_string "Toplevel initialization failed: Type error"
          (fun fmt ->
            let report = Typecore.report_error ~loc env err in
            Location.print_report fmt report)
      in
      { Quill_top.output = ""; error = Some error_msg; status = `Error }
  | ex ->
      (* Generic error with backtrace *)
      let error_msg =
        Printf.sprintf "Toplevel initialization failed: %s\nBacktrace:\n%s"
          (Printexc.to_string ex)
          (Printexc.get_backtrace ())
      in
      { Quill_top.output = ""; error = Some error_msg; status = `Error }
