(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* ───── Initialization ───── *)

let initialized = ref false
let init_mutex = Mutex.create ()

let initialize_if_needed () =
  Mutex.lock init_mutex;
  Fun.protect
    ~finally:(fun () -> Mutex.unlock init_mutex)
    (fun () ->
      if not !initialized then (
        Sys.interactive := false;
        Topeval.init ();
        Toploop.initialize_toplevel_env ();
        Toploop.input_name := "//toplevel//";
        Sys.interactive := true;
        initialized := true))

(* ───── Output capture ───── *)

let drain_into fd buf =
  let tmp = Bytes.create 4096 in
  let rec loop () =
    match Unix.read fd tmp 0 4096 with
    | 0 -> ()
    | n ->
        Buffer.add_subbytes buf tmp 0 n;
        loop ()
  in
  loop ();
  Unix.close fd

let capture f =
  let buf_out = Buffer.create 256 in
  let buf_err = Buffer.create 256 in
  let ppf_out = Format.formatter_of_buffer buf_out in
  let ppf_err = Format.formatter_of_buffer buf_err in
  (* Pipes for raw stdout/stderr from user code (e.g. print_string) *)
  let rd_out, wr_out = Unix.pipe ~cloexec:true () in
  let rd_err, wr_err = Unix.pipe ~cloexec:true () in
  let stdout_backup = Unix.dup ~cloexec:true Unix.stdout in
  let stderr_backup = Unix.dup ~cloexec:true Unix.stderr in
  (* Drain pipes in background threads to avoid deadlock when user code writes
     more than the OS pipe buffer (~64KB). *)
  let buf_raw_out = Buffer.create 256 in
  let buf_raw_err = Buffer.create 256 in
  let t_out = Thread.create (fun () -> drain_into rd_out buf_raw_out) () in
  let t_err = Thread.create (fun () -> drain_into rd_err buf_raw_err) () in
  let result = ref None in
  Fun.protect
    (fun () ->
      flush stdout;
      flush stderr;
      Unix.dup2 ~cloexec:false wr_out Unix.stdout;
      Unix.dup2 ~cloexec:false wr_err Unix.stderr;
      result := Some (f ppf_out ppf_err))
    ~finally:(fun () ->
      Format.pp_print_flush ppf_out ();
      Format.pp_print_flush ppf_err ();
      flush stdout;
      flush stderr;
      Unix.dup2 ~cloexec:false stdout_backup Unix.stdout;
      Unix.dup2 ~cloexec:false stderr_backup Unix.stderr;
      Unix.close stdout_backup;
      Unix.close stderr_backup;
      (* Close write ends so drain threads see EOF *)
      Unix.close wr_out;
      Unix.close wr_err);
  Thread.join t_out;
  Thread.join t_err;
  let stdout_text =
    let toplevel = Buffer.contents buf_out in
    let raw = Buffer.contents buf_raw_out in
    if raw = "" then toplevel else if toplevel = "" then raw else toplevel ^ raw
  in
  let stderr_text =
    let warnings = Buffer.contents buf_err in
    let raw = Buffer.contents buf_raw_err in
    if raw = "" then warnings else if warnings = "" then raw else warnings ^ raw
  in
  match !result with
  | None -> failwith "capture: unreachable"
  | Some ok -> (ok, stdout_text, stderr_text)

(* ───── Execution ───── *)

let ensure_terminator code =
  let trimmed = String.trim code in
  if trimmed = "" || String.ends_with ~suffix:";;" trimmed then code
  else code ^ ";;"

let execute_code ppf_out ppf_err code =
  let code = ensure_terminator code in
  let lb = Lexing.from_string code in
  lb.lex_curr_p <-
    { pos_fname = "//toplevel//"; pos_lnum = 1; pos_bol = 0; pos_cnum = 0 };
  let old_warnings_fmt = !Location.formatter_for_warnings in
  Location.formatter_for_warnings := ppf_err;
  let orig_input_lexbuf = !Location.input_lexbuf in
  Location.input_lexbuf := Some lb;
  let phrases = ref [] in
  (try
     while true do
       let phr = !Toploop.parse_toplevel_phrase lb in
       phrases := phr :: !phrases
     done
   with End_of_file -> ());
  let phrases = List.rev !phrases in
  let num_phrases = List.length phrases in
  let success = ref true in
  Fun.protect
    (fun () ->
      List.iteri
        (fun i phr ->
          try
            let is_last = i = num_phrases - 1 in
            let ok = Toploop.execute_phrase is_last ppf_out phr in
            success := !success && ok
          with
          | Sys.Break ->
              success := false;
              Format.fprintf ppf_err "Interrupted.@."
          | x ->
              success := false;
              Errors.report_error ppf_err x)
        phrases)
    ~finally:(fun () ->
      Location.formatter_for_warnings := old_warnings_fmt;
      Location.input_lexbuf := orig_input_lexbuf;
      Format.pp_print_flush ppf_out ();
      Format.pp_print_flush ppf_err ());
  !success

(* ───── Completion ───── *)

let clamp lo hi x = if x < lo then lo else if x > hi then hi else x

let starts_with ~prefix s =
  let lp = String.length prefix and ls = String.length s in
  lp <= ls && String.sub s 0 lp = prefix

let is_ident_char = function
  | 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '_' | '\'' -> true
  | _ -> false

let is_path_char c = is_ident_char c || Char.equal c '.'

let parse_completion_context code pos =
  let len = String.length code in
  let pos = clamp 0 len pos in
  let i = ref (pos - 1) in
  while !i >= 0 && is_path_char code.[!i] do
    decr i
  done;
  let start = !i + 1 in
  let token = if pos > start then String.sub code start (pos - start) else "" in
  let token =
    if String.starts_with ~prefix:"." token then
      String.sub token 1 (String.length token - 1)
    else token
  in
  if token = "" then (None, "")
  else
    let trailing_dot = String.ends_with ~suffix:"." token in
    let parts = String.split_on_char '.' token |> List.filter (( <> ) "") in
    if trailing_dot then (Longident.unflatten parts, "")
    else
      match List.rev parts with
      | [] -> (None, "")
      | prefix :: rev_qual ->
          let qualifier = Longident.unflatten (List.rev rev_qual) in
          (qualifier, prefix)

let format_type env ty =
  Printtyp.wrap_printing_env ~error:false env (fun () ->
      Format.asprintf "%a" Printtyp.type_scheme ty)

let collect_env_items env qualifier =
  let open Quill.Kernel in
  let add label kind detail acc =
    if String.length label = 0 then acc else { label; kind; detail } :: acc
  in
  let items =
    Env.fold_values
      (fun name _path (vd : Types.value_description) acc ->
        add name Value (format_type env vd.val_type) acc)
      qualifier env []
  in
  let items =
    Env.fold_types
      (fun name _path (td : Types.type_declaration) acc ->
        let detail =
          match td.type_manifest with
          | Some ty -> "= " ^ format_type env ty
          | None -> (
              match td.type_kind with
              | Type_abstract _ -> "abstract"
              | Type_record _ -> "record"
              | Type_variant _ -> "variant"
              | Type_open -> "open")
        in
        add name Type detail acc)
      qualifier env items
  in
  let items =
    Env.fold_modules
      (fun name _path (_md : Types.module_declaration) acc ->
        add name Module "module" acc)
      qualifier env items
  in
  let items =
    Env.fold_modtypes
      (fun name _path (_mtd : Types.modtype_declaration) acc ->
        add name Module_type "module type" acc)
      qualifier env items
  in
  let items =
    Env.fold_constructors
      (fun (c : Data_types.constructor_description) acc ->
        let detail = format_type env c.cstr_res in
        add c.cstr_name Constructor detail acc)
      qualifier env items
  in
  Env.fold_labels
    (fun (l : Data_types.label_description) acc ->
      let detail = format_type env l.lbl_arg in
      add l.lbl_name Label detail acc)
    qualifier env items

let complete_names ~code ~pos =
  let qualifier, prefix = parse_completion_context code pos in
  let env = !Toploop.toplevel_env in
  collect_env_items env qualifier
  |> List.filter (fun (item : Quill.Kernel.completion_item) ->
      String.length prefix = 0 || starts_with ~prefix item.label)
  |> List.sort_uniq (fun (a : Quill.Kernel.completion_item) b ->
      String.compare a.label b.label)

(* ───── Kernel interface ───── *)

let status_ref = ref Quill.Kernel.Idle

let create ~on_event =
  let execute ~cell_id ~code =
    initialize_if_needed ();
    status_ref := Quill.Kernel.Busy;
    on_event (Quill.Kernel.Status_changed Busy);
    let ok, stdout_text, stderr_text =
      capture (fun ppf_out ppf_err -> execute_code ppf_out ppf_err code)
    in
    (* Emit outputs *)
    if stdout_text <> "" then
      on_event
        (Quill.Kernel.Output { cell_id; output = Quill.Cell.Stdout stdout_text });
    if stderr_text <> "" then
      on_event
        (Quill.Kernel.Output { cell_id; output = Quill.Cell.Stderr stderr_text });
    (* Signal completion *)
    on_event (Quill.Kernel.Finished { cell_id; success = ok });
    status_ref := Quill.Kernel.Idle;
    on_event (Quill.Kernel.Status_changed Idle)
  in
  let interrupt () =
    (* Send SIGINT to the current thread - this will cause Sys.Break *)
    try Unix.kill (Unix.getpid ()) Sys.sigint with _ -> ()
  in
  let complete ~code ~pos =
    initialize_if_needed ();
    try complete_names ~code ~pos with _ -> []
  in
  let status () = !status_ref in
  let shutdown () =
    status_ref := Quill.Kernel.Shutting_down;
    on_event (Quill.Kernel.Status_changed Shutting_down)
  in
  {
    Quill.Kernel.execute;
    interrupt;
    complete;
    type_at = None;
    diagnostics = None;
    status;
    shutdown;
  }
