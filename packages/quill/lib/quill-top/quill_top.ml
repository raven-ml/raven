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

(* ───── Toplevel primitives ───── *)

let add_packages pkgs =
  (try Findlib.init () with _ -> ());
  List.iter
    (fun pkg ->
      match Findlib.package_directory pkg with
      | dir -> Topdirs.dir_directory dir
      | exception Findlib.No_such_package _ -> ())
    pkgs

let install_printer name =
  try
    let phrase =
      Printf.sprintf "#install_printer %s;;" name
      |> Lexing.from_string
      |> !Toploop.parse_toplevel_phrase
    in
    ignore (Toploop.execute_phrase false Format.err_formatter phrase)
  with _ -> ()

let install_printer_fn ~ty f =
  try
    let parts = String.split_on_char '.' ty in
    match Longident.unflatten parts with
    | None -> ()
    | Some lid ->
        let path, _decl = Env.find_type_by_name lid !Toploop.toplevel_env in
        let ty_expr = Ctype.newconstr path [] in
        let printer_path = Path.Pident (Ident.create_local ty) in
        Toploop.install_printer printer_path ty_expr f
  with _ -> ()

(* ───── Output capture ───── *)

(** [read_available fd] reads whatever bytes are currently available on [fd]
    without blocking indefinitely (the caller uses [Unix.select] first). Returns
    [None] on EOF. *)
let read_available fd =
  let tmp = Bytes.create 4096 in
  match Unix.read fd tmp 0 4096 with
  | 0 -> None
  | n -> Some (Bytes.sub_string tmp 0 n)
  | exception Unix.Unix_error (Unix.EAGAIN, _, _) -> Some ""

(** [drain_remaining fd] reads all remaining bytes after the write end is
    closed. *)
let drain_remaining fd =
  let buf = Buffer.create 256 in
  let tmp = Bytes.create 4096 in
  let rec loop () =
    match Unix.read fd tmp 0 4096 with
    | 0 -> ()
    | n ->
        Buffer.add_subbytes buf tmp 0 n;
        loop ()
  in
  loop ();
  Unix.close fd;
  Buffer.contents buf

let capture ~on_stdout ~on_stderr ~on_display f =
  let buf_out = Buffer.create 256 in
  let buf_err = Buffer.create 256 in
  let ppf_out = Format.formatter_of_buffer buf_out in
  let ppf_err = Format.formatter_of_buffer buf_err in
  (* Intercept Display_tag semantic tags on the toplevel formatter *)
  Format.pp_set_print_tags ppf_out true;
  Format.pp_set_formatter_stag_functions ppf_out
    {
      mark_open_stag = (fun _ -> "");
      mark_close_stag = (fun _ -> "");
      print_open_stag =
        (fun stag ->
          match stag with
          | Quill.Cell.Display_tag { mime; data } ->
              on_display (Quill.Cell.Display { mime; data })
          | _ -> ());
      print_close_stag = (fun _ -> ());
    };
  (* Pipes for raw stdout/stderr from user code (e.g. print_string) *)
  let rd_out, wr_out = Unix.pipe ~cloexec:true () in
  let rd_err, wr_err = Unix.pipe ~cloexec:true () in
  let stdout_backup = Unix.dup ~cloexec:true Unix.stdout in
  let stderr_backup = Unix.dup ~cloexec:true Unix.stderr in
  (* Poll pipes in a background thread, streaming output as it arrives. Uses
     Unix.select with a 50ms timeout so training progress prints (Printf.printf
     "\rstep %d loss: %.4f%!" ...) appear in real time. *)
  let stop = Atomic.make false in
  let poll_thread =
    Thread.create
      (fun () ->
        while not (Atomic.get stop) do
          let ready, _, _ =
            try Unix.select [ rd_out; rd_err ] [] [] 0.05
            with Unix.Unix_error (Unix.EINTR, _, _) -> ([], [], [])
          in
          List.iter
            (fun fd ->
              match read_available fd with
              | Some s when s <> "" ->
                  if fd == rd_out then on_stdout s else on_stderr s
              | _ -> ())
            ready
        done)
      ()
  in
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
      (* Close write ends so poll thread and drain see EOF *)
      Unix.close wr_out;
      Unix.close wr_err);
  (* Stop the poll thread and drain any remaining bytes *)
  Atomic.set stop true;
  Thread.join poll_thread;
  let rest_out = drain_remaining rd_out in
  let rest_err = drain_remaining rd_err in
  if rest_out <> "" then on_stdout rest_out;
  if rest_err <> "" then on_stderr rest_err;
  (* Format buffer output (toplevel results like "val x = ...") *)
  let toplevel_out = Buffer.contents buf_out in
  let toplevel_err = Buffer.contents buf_err in
  match !result with
  | None -> failwith "capture: unreachable"
  | Some ok -> (ok, toplevel_out, toplevel_err)

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

(* ───── Parse and typecheck ───── *)

let parse_phrases code =
  let code = ensure_terminator code in
  let lb = Lexing.from_string code in
  lb.lex_curr_p <-
    { pos_fname = "//toplevel//"; pos_lnum = 1; pos_bol = 0; pos_cnum = 0 };
  let phrases = ref [] in
  (try
     while true do
       let phr = !Toploop.parse_toplevel_phrase lb in
       phrases := phr :: !phrases
     done
   with End_of_file -> ());
  List.rev !phrases

let typecheck_structure env structure =
  let tstr, _sig, _names, _shape, _env =
    Typemod.type_toplevel_phrase env structure
  in
  tstr

(* ───── Type at position ───── *)

let loc_contains (loc : Location.t) pos =
  (not loc.loc_ghost)
  && loc.loc_start.pos_cnum <= pos
  && pos <= loc.loc_end.pos_cnum

let loc_span (loc : Location.t) = loc.loc_end.pos_cnum - loc.loc_start.pos_cnum

let find_type_at_pos env (tstr : Typedtree.structure) pos =
  let best = ref None in
  let update loc ty =
    if loc_contains loc pos then
      match !best with
      | Some (_, prev_loc, _) when loc_span loc >= loc_span prev_loc -> ()
      | _ ->
          let typ = format_type env ty in
          best := Some (typ, loc, None)
  in
  let iter =
    {
      Tast_iterator.default_iterator with
      expr =
        (fun self (e : Typedtree.expression) ->
          update e.exp_loc e.exp_type;
          Tast_iterator.default_iterator.expr self e);
      pat =
        (fun (type k) self (p : k Typedtree.general_pattern) ->
          update p.pat_loc p.pat_type;
          Tast_iterator.default_iterator.pat self p);
    }
  in
  iter.structure iter tstr;
  match !best with
  | None -> None
  | Some (typ, loc, doc) ->
      Some
        Quill.Kernel.
          {
            typ;
            doc;
            from_pos = loc.loc_start.pos_cnum;
            to_pos = loc.loc_end.pos_cnum;
          }

let type_at_pos ~code ~pos =
  let env = !Toploop.toplevel_env in
  let phrases = parse_phrases code in
  let rec try_phrases = function
    | [] -> None
    | Parsetree.Ptop_def structure :: rest -> (
        match typecheck_structure env structure with
        | tstr -> (
            match find_type_at_pos env tstr pos with
            | Some _ as result -> result
            | None -> try_phrases rest)
        | exception _ -> try_phrases rest)
    | _ :: rest -> try_phrases rest
  in
  try_phrases phrases

(* ───── Diagnostics ───── *)

let loc_to_positions (loc : Location.t) =
  (loc.loc_start.pos_cnum, loc.loc_end.pos_cnum)

let error_loc_of_exn exn =
  match exn with
  | Location.Error report -> report.main.loc
  | _ -> Location.in_file "//toplevel//"

let format_exn exn =
  match Location.error_of_exn exn with
  | Some (`Ok report) -> Format.asprintf "%a" Location.print_report report
  | _ -> Printexc.to_string exn

let compute_diagnostics ~code =
  let env = !Toploop.toplevel_env in
  let diags = ref [] in
  let len = String.length code in
  let add_diag severity loc message =
    let from_pos, to_pos = loc_to_positions loc in
    (* Clamp to valid range; skip diagnostics with no usable location *)
    let from_pos = clamp 0 len from_pos in
    let to_pos = clamp 0 len to_pos in
    let to_pos =
      if to_pos <= from_pos then min (from_pos + 1) len else to_pos
    in
    if from_pos < len then
      diags := Quill.Kernel.{ from_pos; to_pos; severity; message } :: !diags
  in
  (match parse_phrases code with
  | phrases ->
      List.iter
        (function
          | Parsetree.Ptop_def structure -> (
              try ignore (Typemod.type_toplevel_phrase env structure)
              with exn ->
                add_diag Error (error_loc_of_exn exn) (format_exn exn))
          | _ -> ())
        phrases
  | exception exn -> add_diag Error (error_loc_of_exn exn) (format_exn exn));
  List.rev !diags

(* ───── Kernel interface ───── *)

let status_ref = ref Quill.Kernel.Idle

let create ?setup ~on_event () =
  let setup_done = ref false in
  let ensure_setup () =
    if not !setup_done then (
      initialize_if_needed ();
      (match setup with Some f -> f () | None -> ());
      setup_done := true)
  in
  let execute ~cell_id ~code =
    ensure_setup ();
    status_ref := Quill.Kernel.Busy;
    on_event (Quill.Kernel.Status_changed Busy);
    let emit output = on_event (Quill.Kernel.Output { cell_id; output }) in
    let ok, toplevel_out, toplevel_err =
      capture
        ~on_stdout:(fun s -> emit (Quill.Cell.Stdout s))
        ~on_stderr:(fun s -> emit (Quill.Cell.Stderr s))
        ~on_display:emit
        (fun ppf_out ppf_err -> execute_code ppf_out ppf_err code)
    in
    (* Emit toplevel formatter output (val bindings, type info) *)
    if toplevel_out <> "" then emit (Quill.Cell.Stdout toplevel_out);
    if toplevel_err <> "" then emit (Quill.Cell.Stderr toplevel_err);
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
    ensure_setup ();
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
    type_at =
      Some
        (fun ~code ~pos ->
          ensure_setup ();
          try type_at_pos ~code ~pos with _ -> None);
    diagnostics =
      Some
        (fun ~code ->
          ensure_setup ();
          try compute_diagnostics ~code with _ -> []);
    status;
    shutdown;
  }
