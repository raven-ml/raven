type execution_result = {
  output : string;
  error : string option;
  status : [ `Success | `Error ];
}

(* Borrowed from Mdx *)
let redirect f =
  let stdout_backup = Unix.dup ~cloexec:true Unix.stdout in
  let stderr_backup = Unix.dup ~cloexec:true Unix.stderr in
  let filename = Filename.temp_file "quill-" ".stdout" in
  let fd_out =
    Unix.openfile filename Unix.[ O_WRONLY; O_CREAT; O_TRUNC; O_CLOEXEC ] 0o600
  in
  Unix.dup2 ~cloexec:false fd_out Unix.stdout;
  Unix.dup2 ~cloexec:false fd_out Unix.stderr;
  let ic = open_in filename in
  let read_up_to = ref 0 in
  let capture buf =
    flush stdout;
    flush stderr;
    let pos = Unix.lseek fd_out 0 Unix.SEEK_CUR in
    let len = pos - !read_up_to in
    read_up_to := pos;
    Buffer.add_channel buf ic len
  in
  Fun.protect
    (fun () -> f ~capture)
    ~finally:(fun () ->
      close_in_noerr ic;
      Unix.close fd_out;
      Unix.dup2 ~cloexec:false stdout_backup Unix.stdout;
      Unix.dup2 ~cloexec:false stderr_backup Unix.stderr;
      Unix.close stdout_backup;
      Unix.close stderr_backup;
      Sys.remove filename)

let toplevels : (string, Env.t) Hashtbl.t =
  Hashtbl.create 10 (* Map document_id to toplevel *)

let load_dynamic_library filename =
  try
    let ppf = Format.std_formatter in
    Topdirs.dir_load ppf filename;

    (* Register the module *)
    let module_name =
      Filename.basename filename |> Filename.chop_extension
      |> String.capitalize_ascii
    in
    let mod_id = Ident.create_persistent module_name in
    let cmi_file = Filename.chop_extension filename ^ ".cmi" in
    let unit_info = Unit_info.Artifact.from_filename cmi_file in
    let signature = Env.read_signature unit_info in
    let mod_type = Types.Mty_signature signature in
    let updated_env =
      Env.add_module mod_id Mp_present mod_type !Toploop.toplevel_env
    in
    Toploop.toplevel_env := updated_env;

    true
  with exn ->
    Printf.sprintf "Failed to load %s: %s\n" filename (Printexc.to_string exn)
    |> print_endline;
    false

let initialize_toplevel ?(libraries = []) id =
  if not (Hashtbl.mem toplevels id) then (
    let env = Compmisc.initial_env () in
    Toploop.toplevel_env := env;

    List.iter (fun lib_path -> ignore (load_dynamic_library lib_path)) libraries;

    Hashtbl.add toplevels id !Toploop.toplevel_env)

(* Parse a string into a toplevel phrase *)
let parse_phrase str =
  let lexbuf = Lexing.from_string (str ^ ";;") in
  (* Set pos_fname to an empty string to suppress the file reference *)
  lexbuf.Lexing.lex_curr_p <-
    { lexbuf.Lexing.lex_curr_p with pos_fname = "//toplevel//" };
  lexbuf.Lexing.lex_start_p <-
    { lexbuf.Lexing.lex_start_p with pos_fname = "//toplevel//" };
  try Ok (!Toploop.parse_toplevel_phrase lexbuf)
  with exn ->
    let error_msg =
      match Location.error_of_exn exn with
      | Some (`Ok error) -> Format.asprintf "%a" Location.print_report error
      | _ -> Printexc.to_string exn
    in
    Error error_msg

let is_unit_type ty =
  match ty with
  | Outcometree.Otyp_constr (Oide_ident { printed_name = "unit" }, []) -> true
  | _ -> false

let eval ~id code =
  let env = Hashtbl.find toplevels id in
  Toploop.toplevel_env := env;

  match parse_phrase code with
  | Error error -> { output = ""; error = Some error; status = `Error }
  | Ok phrase ->
      (* Buffers for side-effecting output and phrase output *)
      let side_effect_buffer = Buffer.create 1024 in
      let phrase_buffer = Buffer.create 1024 in
      let phrase_ppf = Format.formatter_of_buffer phrase_buffer in

      (* Store the phrase output for later processing *)
      let phrase_output = ref None in
      let custom_out_phrase _ppf phr =
        phrase_output := Some phr (* Store the out_phrase instead of printing *)
      in
      let old_out_phrase = !Oprint.out_phrase in
      Oprint.out_phrase := custom_out_phrase;

      let result =
        redirect (fun ~capture ->
            try
              (* Redirect warnings to stderr, which goes to
                 side_effect_buffer *)
              let old_warnings = !Location.formatter_for_warnings in
              Location.formatter_for_warnings := Format.err_formatter;

              (* Execute phrase, storing output in phrase_output *)
              let success = Toploop.execute_phrase true phrase_ppf phrase in

              (* Restore warnings *)
              Location.formatter_for_warnings := old_warnings;

              (* Capture side-effecting output *)
              capture side_effect_buffer;
              let side_effect_output = Buffer.contents side_effect_buffer in

              (* Process phrase output *)
              let phrase_output_str =
                match !phrase_output with
                | Some (Outcometree.Ophr_eval (value, ty)) ->
                    if is_unit_type ty && side_effect_output <> "" then ""
                      (* Suppress () if there are side effects *)
                    else
                      (* Print the value *)
                      let buf = Buffer.create 1024 in
                      let ppf = Format.formatter_of_buffer buf in
                      let _ = !Oprint.out_value ppf value in
                      Format.pp_print_flush ppf ();
                      Buffer.contents buf
                | Some (Outcometree.Ophr_signature _) ->
                    "" (* Suppress output for definitions *)
                | Some (Outcometree.Ophr_exception (exn, _)) ->
                    (* Print exception message *)
                    let buf = Buffer.create 1024 in
                    let ppf = Format.formatter_of_buffer buf in
                    Location.report_exception ppf exn;
                    Format.pp_print_flush ppf ();
                    Buffer.contents buf
                | None -> "" (* No phrase output *)
              in
              (* Combine outputs *)
              let output = side_effect_output ^ phrase_output_str in
              {
                output;
                error = None;
                status = (if success then `Success else `Error);
              }
            with exn ->
              (* Handle uncaught exceptions *)
              capture side_effect_buffer;
              (* Capture any output before the exception *)
              let side_effect_output = Buffer.contents side_effect_buffer in
              let error_msg =
                match Location.error_of_exn exn with
                | Some (`Ok error) ->
                    Format.asprintf "%a" Location.print_report error
                | _ -> Printexc.to_string exn
              in
              {
                output = side_effect_output;
                error = Some error_msg;
                status = `Error;
              })
      in
      (* Restore original out_phrase *)
      Oprint.out_phrase := old_out_phrase;
      Hashtbl.replace toplevels id !Toploop.toplevel_env;
      result
