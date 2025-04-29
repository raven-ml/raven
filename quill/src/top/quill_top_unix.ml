let initialized = ref false

let initialize_if_needed () =
  if not !initialized then (
    Quill_top.initialize_toplevel ();
    initialized := true)

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
      {
        Quill_top.output = captured_output;
        error = (if captured_error = "" then None else Some captured_error);
        status = (if success_status then `Success else `Error);
      }

let eval code : Quill_top.execution_result =
  initialize_if_needed ();
  capture_separated (fun ppf_out ppf_err ->
      Quill_top.execute true ppf_out ppf_err code)
