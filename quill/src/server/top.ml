let toplevel_envs : (string, Env.t) Hashtbl.t = Hashtbl.create 10
let toplevel_mutex = Mutex.create ()

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
        (* Initialize using quill_top which handles everything *)
        Quill_top.initialize_toplevel ();
        Toploop.input_name := Printf.sprintf "//toplevel-init-%s//" id;

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
    (* Format the output before sending to client *)
    let formatted_output =
      if result.status = `Success then Quill_top.format_output result.output
      else result.output
    in
    (* Format errors as code blocks to preserve compiler error context *)
    let formatted_error =
      match result.error with
      | Some err -> Some ("```\n" ^ err ^ "\n```")
      | None -> None
    in
    {
      output = formatted_output;
      error = formatted_error;
      status = result.status;
    }
  with ex ->
    Printf.eprintf "!!! Uncaught Exception in eval for ID %s: %s\n%s\n%!" id
      (Printexc.to_string ex)
      (Printexc.get_backtrace ());
    if Mutex.try_lock toplevel_mutex then () else Mutex.unlock toplevel_mutex;
    raise ex
