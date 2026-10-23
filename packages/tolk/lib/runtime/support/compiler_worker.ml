(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external memfd : string -> string -> int = "caml_tolk_worker_memfd"
external send : Unix.file_descr -> string -> bool = "caml_tolk_worker_send"

external above_stdio : Unix.file_descr -> Unix.file_descr
  = "caml_tolk_worker_above_stdio"

external no_sigpipe : Unix.file_descr -> unit = "caml_tolk_worker_no_sigpipe"

external spawn :
  string ->
  string ->
  Unix.file_descr ->
  Unix.file_descr ->
  Unix.file_descr ->
  int = "caml_tolk_worker_spawn"

(* A serving worker: the process that started it, its process, the library
   it loaded, the descriptors of its replies and its requests (one socket but
   on Windows), and of its diagnostics, a file that no name reaches. *)
type serving = {
  owner : int;
  pid : int;
  library : string;
  replies : Unix.file_descr;
  requests : Unix.file_descr;
  log : Unix.file_descr;
}

(* [lock] guards [path] and [idle], the serving workers no call holds. *)
type t = {
  name : string;
  program : string;
  lock : Mutex.t;
  mutable path : string option;
  mutable idle : serving list;
}

let v ~name program =
  { name; program; lock = Mutex.create (); path = None; idle = [] }

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
let ending = function
  | Some (Unix.WSIGNALED s) -> "was killed by " ^ signal_name s
  | Some (WEXITED n) -> "exited with " ^ string_of_int n
  | Some (WSTOPPED s) -> "was stopped by " ^ signal_name s
  | None -> "ended"

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
      Error
        (Printf.sprintf "%s worker %s without a reply\n%s" w.name
           (ending status) diagnostics)

(* Serving *)

(* [f ()], or else [fds] closed and its exception raised. *)
let guarded fds f =
  match f () with
  | v -> v
  | exception e ->
      List.iter Unix.close fds;
      raise e

(* A file for a worker's diagnostics, created for reading and appending, its
   name removed at once: it lasts as long as a descriptor of it. Its name is
   drawn from a state of its own, so that the user's draws stay theirs. *)
let log_file w =
  let dir = Filename.get_temp_dir_name () in
  let names = Random.State.make_self_init () in
  let rec create tries =
    let path =
      Filename.concat dir
        (Printf.sprintf "tolk-%s-%d-%06x.log" w.name (Unix.getpid ())
           (Random.State.bits names land 0xffffff))
    in
    let flags = Unix.[ O_RDWR; O_APPEND; O_CREAT; O_EXCL; O_CLOEXEC ] in
    match Unix.openfile path (O_SHARE_DELETE :: flags) 0o600 with
    | fd -> (fd, path)
    | exception Unix.Unix_error (EEXIST, _, _) when tries > 0 ->
        create (tries - 1)
  in
  let fd, path = create 100 in
  guarded [ fd ] (fun () -> Sys.remove path);
  guarded [ fd ] (fun () -> above_stdio fd)

let close_pair (a, b) =
  Unix.close a;
  if b != a then Unix.close b

(* This process's ends, then the worker's, of the channel of requests and
   replies: a socket, whose sends to a closed peer fail rather than raise
   SIGPIPE, or on Windows, which has no SIGPIPE, two pipes. Each end is
   close-on-exec, so that no other child holds the worker's end, which would
   keep its replies open past its death, and above standard input, output and
   error, which it would replace. *)
let channel () =
  if Sys.win32 then
    let replies, worker_out = Unix.pipe ~cloexec:true () in
    let worker_in, requests =
      guarded [ replies; worker_out ] (Unix.pipe ~cloexec:true)
    in
    ((replies, requests), (worker_in, worker_out))
  else
    let mine, theirs = Unix.socketpair ~cloexec:true PF_UNIX SOCK_STREAM 0 in
    (* A lift that fails leaves its descriptor open. *)
    let mine = guarded [ mine; theirs ] (fun () -> above_stdio mine) in
    let theirs = guarded [ mine; theirs ] (fun () -> above_stdio theirs) in
    guarded [ mine; theirs ] (fun () -> no_sigpipe mine);
    ((mine, mine), (theirs, theirs))

(* Starts [program] on [library], its standard input, output and error the
   descriptors [input], [output] and [errors]. On Windows the child inherits
   these alone ([caml_tolk_worker_spawn]): Unix.create_process would let it
   inherit any handle another start makes inheritable meanwhile, such as
   another worker's pipe. *)
let spawn_worker program library input output errors =
  if Sys.win32 then
    let cmdline =
      String.concat " " (List.map Filename.quote [ program; library ])
    in
    spawn program cmdline input output errors
  else Unix.create_process program [| program; library |] input output errors

let start w library =
  let program = program_path w in
  let fail why =
    failwith (Printf.sprintf "%s worker %s: %s" w.name program why)
  in
  let reason = function
    | Sys_error e | Failure e -> e
    | Unix.Unix_error (e, _, _) -> Unix.error_message e
    | e -> raise e
  in
  let log = try log_file w with e -> fail (reason e) in
  let (replies, requests), theirs =
    try guarded [ log ] channel with e -> fail (reason e)
  in
  match spawn_worker program library (fst theirs) (snd theirs) log with
  | pid ->
      close_pair theirs;
      { owner = Unix.getpid (); pid; library; replies; requests; log }
  | exception e ->
      close_pair theirs;
      close_pair (replies, requests);
      Unix.close log;
      fail (reason e)

(* An idle worker of [library]. Workers started by another process, the one
   this process was forked from, are dropped, their descriptors closed: their
   parent serves its own compiles with them, and only it can wait for them. *)
let take w library =
  Mutex.protect w.lock @@ fun () ->
  let mine, inherited =
    List.partition (fun p -> p.owner = Unix.getpid ()) w.idle
  in
  List.iter
    (fun p ->
      close_pair (p.replies, p.requests);
      Unix.close p.log)
    inherited;
  match List.partition (fun p -> String.equal p.library library) mine with
  | [], _ ->
      w.idle <- mine;
      None
  | p :: same, others ->
      w.idle <- same @ others;
      Some p

let give w p = Mutex.protect w.lock @@ fun () -> w.idle <- p :: w.idle

(* Sends [message] whole: false if the worker has closed its end. *)
let write p message =
  if not Sys.win32 then send p.requests message
  else
    match
      Unix.write_substring p.requests message 0 (String.length message)
    with
    | _ -> true
    | exception Unix.Unix_error (EPIPE, _, _) -> false

(* What a worker answers: a reply; nothing, its end closed before the request
   was sent or before the reply was whole; or something that is no reply. *)
type outcome =
  | Reply of (string, string) result
  | Unsent
  | Ended
  | Broken

(* Reads the worker's reply: [Ended] if its end closes first, [Broken] if what
   it sends is no reply. *)
let read_reply p =
  let chunk = Bytes.create 65536 in
  let rec read buf at n =
    match Unix.read p.replies buf at n with
    | k -> k
    | exception Unix.Unix_error (EINTR, _, _) -> read buf at n
    | exception Unix.Unix_error ((ECONNRESET | EPIPE), _, _) -> 0
  in
  (* "ok N" or "error N", at most 26 bytes, then a line break. *)
  let rec header seen =
    match String.index_opt seen '\n' with
    | Some i ->
        let rest = String.sub seen (i + 1) (String.length seen - i - 1) in
        Ok (String.sub seen 0 i, rest)
    | None when String.length seen > 26 -> Error Broken
    | None -> (
        match read chunk 0 (Bytes.length chunk) with
        | 0 -> Error Ended
        | k -> header (seen ^ Bytes.sub_string chunk 0 k))
  in
  let rec fill data at =
    at = Bytes.length data
    ||
    match read data at (Bytes.length data - at) with
    | 0 -> false
    | k -> fill data (at + k)
  in
  match header "" with
  | Error outcome -> outcome
  | Ok (line, rest) -> (
      match Scanf.sscanf_opt line "%s %u%!" (fun kind n -> (kind, n)) with
      | Some ((("ok" | "error") as kind), n) when String.length rest <= n ->
          let data = Bytes.create n in
          Bytes.blit_string rest 0 data 0 (String.length rest);
          if not (fill data (String.length rest)) then Ended
          else
            let data = Bytes.unsafe_to_string data in
            Reply (if kind = "ok" then Ok data else Error data)
      | _ -> Broken)

let log_size p = (Unix.fstat p.log).st_size

(* What the worker printed past the first [printed] bytes. *)
let log_since p printed =
  let n = log_size p - printed in
  if n <= 0 then ""
  else begin
    ignore (Unix.lseek p.log printed SEEK_SET);
    let data = Bytes.create n in
    let rec fill at =
      match Unix.read p.log data at (n - at) with
      | 0 -> at
      | k -> if at + k = n then n else fill (at + k)
    in
    Bytes.sub_string data 0 (fill 0)
  end

(* Ends [p], killing it if [kill], and says how it ended. *)
let stop p ~kill =
  if kill then (
    try Unix.kill p.pid Sys.sigkill
    with Unix.Unix_error (ESRCH, _, _) -> ());
  let rec wait () =
    match Unix.waitpid [] p.pid with
    | _, status -> Some status
    | exception Unix.Unix_error (EINTR, _, _) -> wait ()
    | exception Unix.Unix_error (ECHILD, _, _) -> None
  in
  let status = wait () in
  close_pair (p.replies, p.requests);
  Unix.close p.log;
  ending status

(* A worker taken idle whose end is closed when the request is sent died
   meanwhile: the request never reached it, and the next worker gets it. A
   worker started for the request fails it. An exception in the exchange
   stops the worker, so that none is left half through a request. *)
let call w library request =
  let message = string_of_int (String.length request) ^ "\n" ^ request in
  let rec exchange () =
    let p, fresh =
      match take w library with
      | Some p -> (p, false)
      | None -> (start w library, true)
    in
    let failed why diagnostics =
      Error (Printf.sprintf "%s worker %s\n%s" w.name why diagnostics)
    in
    match
      let printed = log_size p in
      let outcome = if write p message then read_reply p else Unsent in
      (outcome, log_since p printed)
    with
    | exception e ->
        let bt = Printexc.get_raw_backtrace () in
        ignore (stop p ~kill:true);
        Printexc.raise_with_backtrace e bt
    | Reply r, diagnostics ->
        give w p;
        prerr_string diagnostics;
        r
    | Unsent, _ when not fresh ->
        ignore (stop p ~kill:false);
        exchange ()
    | (Unsent | Ended), diagnostics ->
        failed (stop p ~kill:false ^ " without a reply") diagnostics
    | Broken, diagnostics ->
        failed ("sent no reply and " ^ stop p ~kill:true) diagnostics
  in
  exchange ()
