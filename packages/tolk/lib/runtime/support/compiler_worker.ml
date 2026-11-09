(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external memfd : string -> string -> int = "caml_tolk_worker_memfd"
external send : Unix.file_descr -> string -> bool = "caml_tolk_worker_send"

external above_stdio : Unix.file_descr -> Unix.file_descr
  = "caml_tolk_worker_above_stdio"

external no_sigpipe : Unix.file_descr -> unit = "caml_tolk_worker_no_sigpipe"

external quarantined : Unix.file_descr -> bool
  = "caml_tolk_worker_quarantined"
[@@noalloc]

external spawn :
  string ->
  string ->
  Unix.file_descr ->
  Unix.file_descr ->
  Unix.file_descr ->
  int = "caml_tolk_worker_spawn"

(* A serving worker: the process that started it, its process, the library
   it loaded, the descriptors of its replies and its requests (one socket but
   on Windows), and of its diagnostics, a file that no name reaches, and the
   directory of its temporary files. *)
type serving = {
  owner : int;
  pid : int;
  library : string;
  replies : Unix.file_descr;
  requests : Unix.file_descr;
  log : Unix.file_descr;
  temp : string;
}

(* [lock] guards [path], [refused], the directories whose program the system
   refused to run, each with why, and [idle], the serving workers no call
   holds. *)
type t = {
  name : string;
  program : string;
  lock : Mutex.t;
  mutable path : string option;
  mutable refused : (string * string) list;
  mutable idle : serving list;
}

let rec remove_tree path =
  match (Unix.lstat path).st_kind with
  | S_DIR ->
      let names = Sys.readdir path in
      Array.iter (fun n -> remove_tree (Filename.concat path n)) names;
      Unix.rmdir path
  | _ -> Unix.unlink path

(* Removes the file or directory tree [path], following no symbolic link: the
   temporary directory of a worker that crashed, which could not. What cannot
   be removed stays: it is in the system's temporary directory, and the
   compile it served has its outcome already. *)
let remove path =
  try remove_tree path with Unix.Unix_error _ | Sys_error _ -> ()

let v ~name program =
  {
    name;
    program;
    lock = Mutex.create ();
    path = None;
    refused = [];
    idle = [];
  }

(* The program *)

let rec mkdir_p dir =
  if not (Sys.file_exists dir) then begin
    mkdir_p (Filename.dirname dir);
    try Sys.mkdir dir 0o755 with Sys_error _ when Sys.file_exists dir -> ()
  end

(* The directories the program is written to, in order: the cache, then the
   temporary directory, for a cache that cannot be written, that other users
   can write, or whose files cannot run, as on a volume mounted noexec. *)
let program_dirs () = [ Helpers.cache_dir; Filename.get_temp_dir_name () ]
let digest w = Digest.to_hex (Digest.string w.program)
let prefix w = w.name ^ "-worker-"

(* The file in [dir] named by the program's digest. *)
let program_name w dir =
  Filename.concat dir (prefix w ^ digest w ^ if Sys.win32 then ".exe" else "")

(* Owners and modes: Windows has neither here. *)

let own (st : Unix.stats) = Sys.win32 || st.st_uid = Unix.geteuid ()

let others_write (st : Unix.stats) =
  (not Sys.win32) && st.st_perm land 0o022 <> 0

(* Whether no other user can replace what [dir] holds: a directory of this
   user's or the system's that no other can write, or whose sticky bit keeps
   each file to its owner, as /tmp's. *)
let safe_dir dir =
  match Unix.stat dir with
  | exception Unix.Unix_error _ -> false
  | st ->
      st.st_kind = S_DIR
      && (own st || st.st_uid = 0)
      && ((not (others_write st)) || st.st_perm land 0o1000 <> 0)

(* Whether [path] is the program and may run: a regular file reached through
   no link, this user's, that no other can write, without macOS's quarantine,
   whose bytes are the program's. It is read shared for deletion: on Windows,
   renaming a file holds it open for deletion, and a reader that does not
   share it fails while another process renames the program into place. *)
let trusted w path =
  match Unix.lstat path with
  | exception Unix.Unix_error _ -> false
  | entry when entry.st_kind <> S_REG || (not (own entry)) || others_write entry
    ->
      false
  | entry -> (
      match Unix.openfile path [ O_RDONLY; O_CLOEXEC; O_SHARE_DELETE ] 0 with
      | exception Unix.Unix_error _ -> false
      | fd ->
          let ic = Unix.in_channel_of_descr fd in
          Fun.protect ~finally:(fun () -> close_in_noerr ic) @@ fun () ->
          (* A link put in its place since lstat opens another file. *)
          let st = Unix.fstat fd in
          (Sys.win32 || (st.st_dev = entry.st_dev && st.st_ino = entry.st_ino))
          && (not (quarantined fd))
          && Digest.to_hex (Digest.channel ic (-1)) = digest w)

(* The program in [dir]: [program_name w dir] if it is [trusted], or else
   written aside under a name no other process can take and renamed there.
   Where that name holds another user's file, which a sticky directory keeps,
   the program runs from the file written aside, removed when this process
   exits. *)
let program_file w dir =
  let path = program_name w dir in
  let fail e = failwith (Printf.sprintf "%s worker %s: %s" w.name path e) in
  try
    mkdir_p dir;
    if not (safe_dir dir) then fail "other users can write its directory";
    if trusted w path then path
    else
      let tmp, oc =
        Filename.open_temp_file ~mode:[ Open_binary ] ~perms:0o755 ~temp_dir:dir
          (prefix w) ".tmp"
      in
      Fun.protect
        ~finally:(fun () -> close_out_noerr oc)
        (fun () -> output_string oc w.program);
      (match Sys.rename tmp path with
      | () -> ()
      | exception Sys_error _ when trusted w path -> Sys.remove tmp
      | exception Sys_error _ -> ());
      if trusted w path then path
      else if trusted w tmp then begin
        at_exit (fun () -> try Sys.remove tmp with Sys_error _ -> ());
        tmp
      end
      else fail "cannot be made the program: quarantined, or another's"
  with
  | Sys_error e -> fail e
  | Unix.Unix_error (e, _, _) -> fail (Unix.error_message e)

(* The program written to the first of [dirs] that takes it. A failure names
   each directory's. *)
let rec first_file w errors = function
  | [] -> failwith (String.concat "; " (List.rev errors))
  | dir :: dirs -> (
      try program_file w dir with Failure e -> first_file w (e :: errors) dirs)

(* Written by whichever domain first runs [w], to the first directory whose
   program the system has not refused to run, and written again when it is no
   longer [trusted]: removed, as a cache cleaner does, or replaced. A failure
   is raised to that run and tried again by the next. *)
let program_path w =
  Mutex.protect w.lock @@ fun () ->
  match w.path with
  | Some path when Host_config.system = "linux" || trusted w path -> path
  | Some _ | None ->
      let path =
        if Host_config.system = "linux" then
          let fd = memfd ("tolk-" ^ w.name ^ "-worker") w.program in
          "/proc/self/fd/" ^ string_of_int fd
        else
          let usable dir = not (List.mem_assoc dir w.refused) in
          first_file w [] (List.filter usable (program_dirs ()))
      in
      w.path <- Some path;
      path

(* [Ok ()] if the program at [path], which the system refused to run for
   [why], has a directory left to be written to: then the directory of [path]
   is refused, and the next [program_path] writes the program to a later one.
   Otherwise [Error] names each refusal. *)
let refuse w path why =
  Mutex.protect w.lock @@ fun () ->
  let in_dir dir =
    Filename.dirname (program_name w dir) = Filename.dirname path
  in
  let refusal = path ^ ": " ^ why in
  match List.rev (program_dirs ()) with
  | last :: earlier when not (in_dir last) -> (
      match List.find_opt in_dir earlier with
      | Some dir ->
          if not (List.mem_assoc dir w.refused) then
            w.refused <- (dir, refusal) :: w.refused;
          if w.path = Some path then w.path <- None;
          Ok ()
      | None -> Error refusal)
  | _ -> Error (String.concat "; " (List.rev_map snd w.refused @ [ refusal ]))

(* Running *)

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

(* How a worker ended, as its status says. *)
let ending = function
  | Some (Unix.WSIGNALED s) -> "was killed by " ^ signal_name s
  | Some (WEXITED n) -> "exited with " ^ string_of_int n
  | Some (WSTOPPED s) -> "was stopped by " ^ signal_name s
  | None -> "ended"

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

(* Starts [program] on [library], [temp] and this process's pid, its standard
   input, output and error the descriptors [input], [output] and [errors]. On
   Windows the child inherits these alone ([caml_tolk_worker_spawn]):
   Unix.create_process would let it inherit any handle another start makes
   inheritable meanwhile, such as another worker's pipe. *)
let spawn_worker program library temp input output errors =
  let args = [ program; library; temp; string_of_int (Unix.getpid ()) ] in
  if Sys.win32 then
    let cmdline = String.concat " " (List.map Filename.quote args) in
    spawn program cmdline input output errors
  else Unix.create_process program (Array.of_list args) input output errors

let rec start w library =
  let program = program_path w in
  let fail why =
    failwith (Printf.sprintf "%s worker %s: %s" w.name program why)
  in
  let reason = function
    | Sys_error e | Failure e -> e
    | Unix.Unix_error (e, _, _) -> Unix.error_message e
    | e -> raise e
  in
  let temp =
    try Filename.temp_dir ("tolk-" ^ w.name ^ "-") "" with e -> fail (reason e)
  in
  let fail_removing e =
    remove temp;
    fail (reason e)
  in
  let log = try log_file w with e -> fail_removing e in
  let (replies, requests), theirs =
    try guarded [ log ] channel with e -> fail_removing e
  in
  let close_all () =
    close_pair theirs;
    close_pair (replies, requests);
    Unix.close log
  in
  match spawn_worker program library temp (fst theirs) (snd theirs) log with
  | pid ->
      close_pair theirs;
      { owner = Unix.getpid (); pid; library; replies; requests; log; temp }
  | exception Unix.Unix_error (((EACCES | EPERM) as e), _, _) -> (
      close_all ();
      remove temp;
      match refuse w program (Unix.error_message e) with
      | Ok () -> start w library
      | Error refusals -> failwith (w.name ^ " worker " ^ refusals))
  | exception e ->
      close_all ();
      fail_removing e

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
  remove p.temp;
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
