let toplevel_envs : (string, Env.t) Hashtbl.t = Hashtbl.create 10
let toplevel_mutex = Mutex.create ()
let global_directives_run = ref false

module My_sites = struct
  let plugins_locations : string list = Quill_sites.Sites.toplevel_libs

  let lookup_dirs dirs =
    List.filter Sys.file_exists dirs
    |> List.map (fun dir -> Array.to_list (Sys.readdir dir))
    |> List.concat

  let find_available_plugins () = lookup_dirs plugins_locations

  let lookup_file filename =
    List.find_map
      (fun dir ->
        let filename' = Filename.concat dir filename in
        if Sys.file_exists filename' then Some filename' else None)
      plugins_locations

  let cmas =
    let all_cmas =
      [
        ("nx_core.cma", lookup_file "nx_core.cma");
        ("nx_native.cma", lookup_file "nx_native.cma");
        ("nx.cma", lookup_file "nx.cma");
        ("nx_c.cma", lookup_file "nx_c.cma");
        ("bigarray.cma", lookup_file "bigarray.cma");
        ("bigarray_compat.cma", lookup_file "bigarray_compat.cma");
        ("integers.cma", lookup_file "integers.cma");
        ("ctypes.cma", lookup_file "ctypes.cma");
        ("ctypes_foreign.cma", lookup_file "ctypes_foreign.cma");
        ("objc_c.cma", lookup_file "objc_c.cma");
        ("objc.cma", lookup_file "objc.cma");
        ("metal.cma", lookup_file "metal.cma");
        ("nx_metal.cma", lookup_file "nx_metal.cma");
        ("zip.cma", lookup_file "zip.cma");
        ("npy.cma", lookup_file "npy.cma");
        ("stb_image.cma", lookup_file "stb_image.cma");
        ("stb_image_write.cma", lookup_file "stb_image_write.cma");
        ("nx_io.cma", lookup_file "nx_io.cma");
        ("unix.cma", lookup_file "unix.cma");
        ("curl.cma", lookup_file "curl.cma");
        ("csv.cma", lookup_file "csv.cma");
        ("nx_datasets.cma", lookup_file "nx_datasets.cma");
        ("re.cma", lookup_file "re.cma");
        ("uutf.cma", lookup_file "uutf.cma");
        ("uucp.cma", lookup_file "uucp.cma");
        ("nx_text.cma", lookup_file "nx_text.cma");
        ("cairo.cma", lookup_file "cairo.cma");
        ("usdl.cma", lookup_file "usdl.cma");
        ("base64.cma", lookup_file "base64.cma");
        ("hugin.cma", lookup_file "hugin.cma");
        ("rune_jit.cma", lookup_file "rune_jit.cma");
        ("rune_jit_metal.cma", lookup_file "rune_jit_metal.cma");
        ("rune_metal.cma", lookup_file "rune_metal.cma");
        ("rune.cma", lookup_file "rune.cma");
        ("sowilo.cma", lookup_file "sowilo.cma");
        ("kaun.cma", lookup_file "kaun.cma");
        ("kaun_datasets.cma", lookup_file "kaun_datasets.cma");
      ]
    in
    List.filter_map snd all_cmas
end

let execute_directive directive =
  try
    let lexbuf = Lexing.from_string directive in
    let phrases = !Toploop.parse_use_file lexbuf in
    List.iter
      (fun phrase ->
        let result = Toploop.execute_phrase true Format.err_formatter phrase in
        if not result then
          Printf.eprintf "[DEBUG] Failed to execute directive: %s\n%!" directive)
      phrases
  with
  | Symtable.Error error as ex ->
      Printf.eprintf "[DEBUG] Symtable.Error executing directive '%s'\n%!"
        directive;
      Printf.eprintf "[DEBUG] Error details: %s\n%!"
        (match error with
        | Symtable.Undefined_global g ->
            Printf.sprintf "Undefined_global: %s" (Symtable.Global.name g)
        | Symtable.Unavailable_primitive s ->
            Printf.sprintf "Unavailable_primitive: %s" s
        | Symtable.Wrong_vm s -> Printf.sprintf "Wrong_vm: %s" s
        | Symtable.Uninitialized_global _ -> "Uninitialized_global");
      Printf.eprintf "[DEBUG] Backtrace:\n%s\n%!" (Printexc.get_backtrace ());
      raise ex
  | ex ->
      Printf.eprintf "[DEBUG] Exception executing directive '%s': %s\n%!"
        directive (Printexc.to_string ex);
      Printf.eprintf "[DEBUG] Backtrace:\n%s\n%!" (Printexc.get_backtrace ());
      raise ex

let initialize_toplevel_unsafe () : bool =
  if !global_directives_run then true
  else
    try
      Printf.eprintf "[DEBUG] Starting toplevel initialization...\n%!";
      let site_dirs = Quill_sites.Sites.toplevel_libs in
      Printf.eprintf "[DEBUG] Site directories: %s\n%!"
        (String.concat ", " site_dirs);

      if site_dirs = [] then (
        Printf.eprintf
          "ERROR: No site directories found for 'quill.toplevel_libs'. Check \
           installation.\n\
           %!";
        false)
      else (
        Printf.eprintf "[DEBUG] Adding site directories to search path...\n%!";
        List.iter
          (fun dir ->
            Printf.eprintf "[DEBUG] Adding directory: %s\n%!" dir;
            Topdirs.dir_directory dir)
          site_dirs;

        Printf.eprintf "[DEBUG] Available plugins: %s\n%!"
          (String.concat ", " (My_sites.find_available_plugins ()));

        Printf.eprintf "[DEBUG] Loading CMA files...\n%!";
        Printf.eprintf "[DEBUG] Module loading state check\n%!";

        List.iter
          (fun cma ->
            Printf.eprintf "[DEBUG] About to load %s\n%!" cma;
            (* Special handling for ctypes to get more info *)
            execute_directive (Printf.sprintf "#load %S;;" cma))
          My_sites.cmas;

        Printf.eprintf "[DEBUG] Setting up Nx printer...\n%!";
        execute_directive
          {|
let pp_nx fmt arr =
  Format.fprintf fmt "```ocaml@\n";
  Format.pp_open_vbox fmt 0;
  Nx.pp_data fmt arr;
  Format.pp_close_box fmt ();
  Format.fprintf fmt "@\n```";;
|};
        execute_directive "#install_printer pp_nx;;";

        Printf.eprintf "[DEBUG] Setting up Rune printer...\n%!";
        execute_directive
          {|
let pp_rune fmt arr =
  Format.fprintf fmt "```ocaml@\n";
  Format.pp_open_vbox fmt 0;
  Rune.pp_data fmt arr;
  Format.pp_close_box fmt ();
  Format.fprintf fmt "@\n```";;
|};
        execute_directive "#install_printer pp_nx;;";

        Printf.eprintf "[DEBUG] Setting up Hugin printer...\n%!";
        execute_directive
          {|
let pp_hugin_figure fmt figure =
  let image_data = Hugin.render figure in
  let base64_data = Base64.encode_string image_data in
  Format.fprintf fmt "![figure](data:image/png;base64,%s)" base64_data;;
|};
        execute_directive "#install_printer pp_hugin_figure;;";

        Printf.eprintf "[DEBUG] Toplevel initialization complete!\n%!";
        global_directives_run := true;
        true)
    with ex ->
      Printf.eprintf "Error during toplevel initialization: %s\n%!"
        (Printexc.to_string ex);
      Printf.eprintf "Backtrace:\n%s\n%!" (Printexc.get_backtrace ());
      false

let get_or_create_env_unsafe id =
  match Hashtbl.find_opt toplevel_envs id with
  | Some env -> env
  | None -> (
      Printf.eprintf "[DEBUG] Creating new toplevel environment for '%s'\n%!" id;
      Printexc.record_backtrace true;

      let current_env = !Toploop.toplevel_env in
      let current_input_name = !Toploop.input_name in
      let current_interactive = !Sys.interactive in

      try
        Printf.eprintf "[DEBUG] Initializing toplevel environment...\n%!";
        Toploop.initialize_toplevel_env ();
        Toploop.input_name := Printf.sprintf "//toplevel-init-%s//" id;
        Sys.interactive := true;

        let init_ok = initialize_toplevel_unsafe () in
        if not init_ok then
          Printf.eprintf
            "Warning: Toplevel initialization failed for new env '%s'.\n%!" id;

        let new_env_after_init = !Toploop.toplevel_env in

        Hashtbl.add toplevel_envs id new_env_after_init;

        Toploop.toplevel_env := current_env;
        Toploop.input_name := current_input_name;
        Sys.interactive := current_interactive;

        new_env_after_init
      with ex ->
        Printf.eprintf "[DEBUG] Exception in get_or_create_env_unsafe: %s\n%!"
          (Printexc.to_string ex);
        Printf.eprintf "Backtrace:\n%s\n%!" (Printexc.get_backtrace ());
        (* Restore state and re-raise *)
        Toploop.toplevel_env := current_env;
        Toploop.input_name := current_input_name;
        Sys.interactive := current_interactive;
        raise ex)

let eval ~id code : Quill_api.code_execution_result =
  Mutex.lock toplevel_mutex;
  try
    let target_env = get_or_create_env_unsafe id in

    let saved_env = !Toploop.toplevel_env in
    let saved_input_name = !Toploop.input_name in
    let saved_interactive = !Sys.interactive in

    Toploop.toplevel_env := target_env;
    Toploop.input_name := Printf.sprintf "//toplevel-%s//" id;
    Sys.interactive := true;

    let result : Quill_top.execution_result =
      try Quill_top_unix.eval code
      with exn ->
        let err_msg = Printexc.to_string exn in
        let backtrace = Printexc.get_backtrace () in
        let detailed_error =
          Printf.sprintf "Internal error during eval execution: %s\n%s" err_msg
            backtrace
        in
        { output = ""; error = Some detailed_error; status = `Error }
    in

    Hashtbl.replace toplevel_envs id !Toploop.toplevel_env;

    Toploop.toplevel_env := saved_env;
    Toploop.input_name := saved_input_name;
    Sys.interactive := saved_interactive;

    Mutex.unlock toplevel_mutex;
    { output = result.output; error = result.error; status = result.status }
  with ex ->
    Printf.eprintf "!!! Uncaught Exception in eval for ID %s: %s\n%s\n%!" id
      (Printexc.to_string ex)
      (Printexc.get_backtrace ());
    if Mutex.try_lock toplevel_mutex then () else Mutex.unlock toplevel_mutex;
    raise ex
