type execution_result = {
  output : string; (* Captured stdout + value results *)
  error : string option; (* Captured stderr (errors and warnings) *)
  status : [ `Success | `Error ];
}

let refill_lexbuf s p ppf buffer len =
  if !p = String.length s then 0
  else
    let len', nl =
      try (String.index_from s !p '\n' - !p + 1, false)
      with _ -> (String.length s - !p, true)
    in
    let len'' = min len len' in
    StringLabels.blit ~src:s ~src_pos:!p ~dst:buffer ~dst_pos:0 ~len:len'';
    (match ppf with
    | Some ppf ->
        Format.fprintf ppf "%s"
          (BytesLabels.sub_string buffer ~pos:0 ~len:len'');
        if nl then Format.pp_print_newline ppf ();
        Format.pp_print_flush ppf ()
    | None -> ());
    p := !p + len'';
    len''

let initialize_toplevel () =
  Sys.interactive := false;
  Toploop.initialize_toplevel_env ();
  Toploop.input_name := "//toplevel//";
  Sys.interactive := true

let ensure_terminator code =
  let trimmed_code = String.trim code in
  if trimmed_code = "" || String.ends_with ~suffix:";;" trimmed_code then code
  else code ^ ";;"

let execute printval pp_out pp_err s =
  let s = ensure_terminator s in

  let lb = Lexing.from_function (refill_lexbuf s (ref 0) None) in
  let overall_success = ref true in

  let old_warnings_formatter = !Location.formatter_for_warnings in
  Location.formatter_for_warnings := pp_err;

  Fun.protect
    (fun () ->
      try
        while true do
          try
            let phr = !Toploop.parse_toplevel_phrase lb in
            let exec_success = Toploop.execute_phrase printval pp_out phr in
            overall_success := !overall_success && exec_success
          with
          | End_of_file -> raise End_of_file
          | Sys.Break ->
              overall_success := false;
              Format.fprintf pp_err "Interrupted.@.";
              raise End_of_file
          | x ->
              overall_success := false;
              Errors.report_error pp_err x
        done
      with End_of_file -> ())
    ~finally:(fun () ->
      Location.formatter_for_warnings := old_warnings_formatter;
      Format.pp_print_flush pp_out ();
      Format.pp_print_flush pp_err ());

  !overall_success
