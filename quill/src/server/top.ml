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
    [
      lookup_file "nx_core.cma";
      lookup_file "nx_native.cma";
      lookup_file "nx.cma";
      lookup_file "nx_metal.cma";
      lookup_file "bigarray.cma";
      lookup_file "zip.cma";
      lookup_file "npy.cma";
      lookup_file "stb_image.cma";
      lookup_file "stb_image_write.cma";
      lookup_file "nx_io.cma";
      lookup_file "unix.cma";
      lookup_file "curl.cma";
      lookup_file "csv.cma";
      lookup_file "nx_datasets.cma";
      lookup_file "cairo.cma";
      lookup_file "usdl.cma";
      lookup_file "sowilo.cma";
      lookup_file "hugin.cma";
      lookup_file "base64.cma";
      lookup_file "rune.cma";
    ]
    |> List.filter_map Fun.id
end

let execute_directive directive =
  let lexbuf = Lexing.from_string directive in
  let phrases = !Toploop.parse_use_file lexbuf in
  List.iter
    (fun phrase ->
      ignore (Toploop.execute_phrase true Format.err_formatter phrase))
    phrases

let initialize_toplevel_unsafe () : bool =
  if !global_directives_run then true
  else
    try
      let site_dirs = Quill_sites.Sites.toplevel_libs in

      if site_dirs = [] then (
        Printf.eprintf
          "ERROR: No site directories found for 'quill.toplevel_libs'. Check \
           installation.\n\
           %!";
        false)
      else (
        List.iter (fun dir -> Topdirs.dir_directory dir) site_dirs;

        List.iter
          (fun cma -> execute_directive (Printf.sprintf "#load %S;;" cma))
          My_sites.cmas;

        execute_directive
          {|
let pp_nx fmt arr =
  Format.fprintf fmt "```ocaml@\n";
  Format.pp_open_vbox fmt 0;
  Nx.pp fmt arr;
  Format.pp_close_box fmt ();
  Format.fprintf fmt "@\n```";;
|};
        execute_directive "#install_printer pp_nx;;";

        execute_directive
          {|
let pp_hugin_figure fmt figure =
  let image_data = Hugin.render figure in
  let base64_data = Base64.encode_string image_data in
  Format.fprintf fmt "![figure](data:image/png;base64,%s)" base64_data;;
|};
        execute_directive "#install_printer pp_hugin_figure;;";

        global_directives_run := true;
        true)
    with ex ->
      Printf.eprintf "Error during toplevel initialization: %s\n%!"
        (Printexc.to_string ex);
      false

let get_or_create_env_unsafe id =
  match Hashtbl.find_opt toplevel_envs id with
  | Some env -> env
  | None ->
      let current_env = !Toploop.toplevel_env in
      let current_input_name = !Toploop.input_name in
      let current_interactive = !Sys.interactive in

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
