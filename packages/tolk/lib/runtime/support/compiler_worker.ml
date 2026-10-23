(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external memfd : string -> string -> int = "caml_tolk_worker_memfd"

type t = {
  name : string;
  program : string;
  lock : Mutex.t;
  mutable path : string option;
}

let v ~name program = { name; program; lock = Mutex.create (); path = None }

(* The program *)

let rec mkdir_p dir =
  if not (Sys.file_exists dir) then begin
    mkdir_p (Filename.dirname dir);
    try Sys.mkdir dir 0o755 with Sys_error _ when Sys.file_exists dir -> ()
  end

(* The program is read shared for deletion: on Windows, renaming a file holds
   it open for deletion, and a reader that does not share it fails while
   another process renames the program into place. *)
let file_digest path =
  let fd = Unix.openfile path [ O_RDONLY; O_CLOEXEC; O_SHARE_DELETE ] 0 in
  let ic = Unix.in_channel_of_descr fd in
  Fun.protect ~finally:(fun () -> close_in ic) (fun () ->
      Digest.channel ic (-1))

(* The file in [Helpers.cache_dir] named by the program's digest, written aside
   and renamed into place unless a whole one is there. *)
let program_file w =
  let digest = Digest.to_hex (Digest.string w.program) in
  let prefix = w.name ^ "-worker-" in
  let path =
    Filename.concat Helpers.cache_dir
      (prefix ^ digest ^ if Sys.win32 then ".exe" else "")
  in
  let whole () =
    Sys.file_exists path && Digest.to_hex (file_digest path) = digest
  in
  try
    if not (whole ()) then begin
      mkdir_p Helpers.cache_dir;
      let tmp, oc =
        Filename.open_temp_file ~mode:[ Open_binary ] ~perms:0o755
          ~temp_dir:Helpers.cache_dir prefix ".tmp"
      in
      Fun.protect
        ~finally:(fun () -> close_out_noerr oc)
        (fun () -> output_string oc w.program);
      try Sys.rename tmp path with Sys_error _ when whole () -> Sys.remove tmp
    end;
    if not (whole ()) then failwith (path ^ ": differs from the program");
    path
  with
  | Sys_error e -> failwith (Printf.sprintf "%s worker %s: %s" w.name path e)
  | Unix.Unix_error (e, _, _) ->
      failwith
        (Printf.sprintf "%s worker %s: %s" w.name path (Unix.error_message e))

(* Written once, by whichever domain first runs [w]. A failure is raised to
   that run and tried again by the next. *)
let program_path w =
  Mutex.protect w.lock @@ fun () ->
  match w.path with
  | Some path -> path
  | None ->
      let path =
        if Host_config.system = "linux" then
          let fd = memfd ("tolk-" ^ w.name ^ "-worker") w.program in
          "/proc/self/fd/" ^ string_of_int fd
        else program_file w
      in
      w.path <- Some path;
      path

(* Running *)

(* [reply s] is the reply [s] holds if it is complete. *)
let reply s =
  match
    Scanf.sscanf_opt s "%s %u\n%n" (fun kind n start -> (kind, n, start))
  with
  | Some (kind, n, start) when start + n = String.length s -> (
      let data = String.sub s start n in
      match kind with
      | "ok" -> Some (Ok data)
      | "error" -> Some (Error data)
      | _ -> None)
  | _ -> None

let signal_name s =
  let names =
    Sys.
      [
        (sigsegv, "SIGSEGV");
        (sigbus, "SIGBUS");
        (sigabrt, "SIGABRT");
        (sigill, "SIGILL");
        (sigfpe, "SIGFPE");
        (sigkill, "SIGKILL");
        (sigterm, "SIGTERM");
        (sigint, "SIGINT");
      ]
  in
  match List.assoc_opt s names with
  | Some name -> name
  | None -> "signal " ^ string_of_int s

(* The request, the reply and the diagnostics are files, so that neither
   process ever waits for the other to read. *)
let run w library request =
  let program = program_path w in
  let temp_file suffix contents =
    let path = Filename.temp_file ("tolk-" ^ w.name) suffix in
    Out_channel.with_open_bin path (fun oc -> output_string oc contents);
    path
  in
  let request_file = temp_file ".request" request
  and reply_file = temp_file ".reply" ""
  and diagnostics_file = temp_file ".log" "" in
  Fun.protect ~finally:(fun () ->
      List.iter Sys.remove [ request_file; reply_file; diagnostics_file ])
  @@ fun () ->
  let args =
    [ library; string_of_int (Unix.getpid ()); request_file; reply_file ]
  in
  let status =
    match
      Helpers.run_program ~stdin:Filename.null ~stdout:diagnostics_file
        ~stderr:diagnostics_file program args
    with
    | Ok status -> status
    | Error why -> failwith (w.name ^ " worker " ^ why)
  in
  let read path = In_channel.with_open_bin path In_channel.input_all in
  let diagnostics = read diagnostics_file in
  match reply (read reply_file) with
  | Some r ->
      prerr_string diagnostics;
      r
  | None ->
      let why =
        match status with
        | Some (WSIGNALED s) -> "was killed by " ^ signal_name s
        | Some (WEXITED n) -> "exited with " ^ string_of_int n
        | Some (WSTOPPED s) -> "was stopped by " ^ signal_name s
        | None -> "ended"
      in
      Error
        (Printf.sprintf "%s worker %s without a reply\n%s" w.name why
           diagnostics)
