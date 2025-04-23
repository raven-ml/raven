type execution_result = {
  output : string;
  error : string option;
  status : [ `Success | `Error ];
}

let redirect f =
  let stdout_backup = Unix.dup ~cloexec:true Unix.stdout in
  let stderr_backup = Unix.dup ~cloexec:true Unix.stderr in
  let filename = Filename.temp_file "quill-" ".stdout" in
  let fd_out =
    Unix.openfile filename Unix.[ O_WRONLY; O_CREAT; O_TRUNC; O_CLOEXEC ] 0o600
  in
  flush stdout;
  flush stderr;
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
      flush stdout;
      flush stderr;
      Unix.close fd_out;
      Unix.dup2 ~cloexec:false stdout_backup Unix.stdout;
      Unix.dup2 ~cloexec:false stderr_backup Unix.stderr;
      Unix.close stdout_backup;
      Unix.close stderr_backup;
      Sys.remove filename)

let toplevels : (string, Env.t) Hashtbl.t =
  Hashtbl.create 10 (* Map document_id to toplevel *)

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

let initialize_toplevel id =
  if not (Hashtbl.mem toplevels id) then (
    let env = Compmisc.initial_env () in
    Toploop.toplevel_env := env;
    (* Initialize findlib *)
    (match parse_phrase "#use \"topfind\";;" with
    | Ok phrase ->
        let _ = Toploop.execute_phrase false Format.err_formatter phrase in
        ()
    | Error err ->
        prerr_endline ("Warning: Failed to initialize findlib: " ^ err));
    (* Load libraries and install printers *)
    let phrases =
      [
        "#require \"ndarray\";;";
        "#install_printer Ndarray.print;;";
        "#require \"ndarray-io\";;";
        "#require \"ndarray-cv\";;";
        "#require \"ndarray-datasets\";;";
        "#require \"hugin\";;";
        "#require \"rune\";;";
        "#install_printer Rune.print;;";
      ]
    in
    List.iter
      (fun code ->
        match parse_phrase code with
        | Ok phrase ->
            let _ = Toploop.execute_phrase false Format.err_formatter phrase in
            ()
        | Error err ->
            prerr_endline ("Warning: Failed to execute phrase: " ^ err))
      phrases;
    Hashtbl.add toplevels id !Toploop.toplevel_env)

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

      let result =
        redirect (fun ~capture ->
            try
              (* Redirect warnings to stderr, which goes to
                 side_effect_buffer *)
              let old_warnings = !Location.formatter_for_warnings in
              Location.formatter_for_warnings := Format.err_formatter;

              (* Execute phrase and print to phrase_ppf using default
                 out_phrase *)
              let success = Toploop.execute_phrase true phrase_ppf phrase in

              (* Restore warnings *)
              Location.formatter_for_warnings := old_warnings;

              (* Capture side-effecting output *)
              capture side_effect_buffer;
              let side_effect_output = Buffer.contents side_effect_buffer in

              (* Capture phrase output *)
              Format.pp_print_flush phrase_ppf ();
              let phrase_output_str = Buffer.contents phrase_buffer in

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
      Hashtbl.replace toplevels id !Toploop.toplevel_env;
      result
