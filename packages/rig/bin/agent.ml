(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

let say l = Proc.write Unix.stdout (Line.to_string l)

(* The job's key: the first line of [fd], read a byte at a time so that nothing
   after it is taken, once [Rig_remote.key] accepts it. A line that grows from a
   key into none is longer than any key: it is refused there, read no
   further. *)
let read_key fd =
  let b = Buffer.create 64 and c = Bytes.create 1 in
  let key () = Rig_remote.key (Buffer.contents b) in
  let rec go was_key =
    match Unix.read fd c 0 1 with
    | 0 when Buffer.length b = 0 -> Error "no key on standard input"
    | n when n = 0 || Bytes.get c 0 = '\n' ->
        Result.map (fun _ -> Buffer.contents b) (key ())
    | _ -> (
        Buffer.add_char b (Bytes.get c 0);
        match key () with
        | Error _ as e when was_key -> e
        | k -> go (Result.is_ok k))
    | exception Unix.Unix_error (Unix.EINTR, _, _) -> go was_key
  in
  go false

let fail why =
  say (Line.Failed why);
  exit 123

(* The half *)

let half ~firmware address =
  say (Line.Agent Version.session);
  let key =
    match read_key Unix.stdin with Ok k -> k | Error why -> fail why
  in
  Proc.signals [];
  let key_r, key_w = Proc.pipe () and out_r, out_w = Proc.pipe () in
  let env = Array.append (Unix.environment ()) [| "RIG_REMOTE_REPORT=1" |] in
  let exe = Sys.executable_name in
  let options = List.concat_map (fun d -> [ "--firmware"; d ]) firmware in
  let pid =
    Proc.spawn ~env exe
      (Array.of_list ((exe :: "agent" :: options) @ [ address ]))
      ~stdin:key_r ~stdout:out_w ~stderr:Unix.stderr
  in
  Unix.close key_r;
  Unix.close out_w;
  (* [key_w] stays open until the half exits, however it exits: the agent ends
     when its input ends. *)
  Proc.write key_w (key ^ "\n");
  let lines = Line.reader out_r and input = Line.reader Unix.stdin in
  let reported = ref None and killed = ref false and status = ref None in
  let relay l =
    match Line.of_string l with
    | Some ((Line.Closed | Line.Failed _) as l) when !reported = None ->
        reported := Some l;
        say l
    | Some ((Line.Waiting | Line.Listening _) as l) -> say l
    | _ -> ()
  in
  let rec loop () =
    Proc.wait [ lines; input ];
    if !status = None then status := Proc.reap pid;
    ignore (Line.read input);
    if Line.ended input && !status = None && not !killed then begin
      killed := true;
      Proc.kill pid
    end;
    List.iter relay (Line.read lines);
    if !status = None || not (Line.ended lines) then loop ()
  in
  loop ();
  (* The agent died unreported, unless this half killed it. *)
  match (!status, !reported) with
  | Some (Unix.WSIGNALED s), None when !killed && s = Sys.sigkill -> exit 123
  | Some st, None ->
      say (Line.Died (Proc.cause st));
      exit 123
  | _, r -> exit (if r = Some Line.Closed then 0 else 123)

(* The agent *)

(* Holds the machine's lock for the life of the process: one agent of this user
   runs on a machine at a time. The lock's file holds the holder's process
   id. *)
let lock () =
  let path =
    Filename.concat
      (Filename.get_temp_dir_name ())
      (strf "rig-agent-%d.lock" (Unix.getuid ()))
  in
  let error e = Error (strf "%s: %s" path (Unix.error_message e)) in
  (* The holder's process id is the lock's promise: a write that fails, on a
     full disk say, fails the lock. *)
  let rec write fd s at =
    if at < String.length s then
      match Unix.single_write_substring fd s at (String.length s - at) with
      | n -> write fd s (at + n)
      | exception Unix.Unix_error (Unix.EINTR, _, _) -> write fd s at
  in
  let hold fd =
    (try Unix.lockf fd Unix.F_TLOCK 0
     with Unix.Unix_error ((Unix.EAGAIN | Unix.EACCES), _, _) ->
       say Line.Waiting;
       let rec wait () =
         try Unix.lockf fd Unix.F_LOCK 0
         with Unix.Unix_error (Unix.EINTR, _, _) -> wait ()
       in
       wait ());
    Unix.ftruncate fd 0;
    write fd (strf "%d\n" (Unix.getpid ())) 0
  in
  let take () =
    match
      Unix.openfile path [ Unix.O_RDWR; Unix.O_CREAT; Unix.O_CLOEXEC ] 0o600
    with
    | exception Unix.Unix_error (e, _, _) -> error e
    | fd -> (
        match hold fd with
        | () -> Ok ()
        | exception Unix.Unix_error (e, _, _) ->
            Unix.close fd;
            error e)
  in
  match Unix.lstat path with
  | { st_kind = Unix.S_REG; st_uid; _ } when st_uid <> Unix.getuid () ->
      Error (strf "%s belongs to another user" path)
  | { st_kind = Unix.S_REG; _ } -> take ()
  | _ -> Error (strf "%s is no regular file" path)
  | exception Unix.Unix_error (Unix.ENOENT, _, _) -> take ()
  | exception Unix.Unix_error (e, _, _) -> error e

(* Ends the agent once its input ends: its half holds the input open until it
   exits, however it exits. *)
let orphaned () =
  let b = Bytes.create 1 in
  let rec watch () =
    match Unix.read Unix.stdin b 0 1 with
    | 0 -> Unix._exit 123
    | _ -> watch ()
    | exception Unix.Unix_error (Unix.EINTR, _, _) -> watch ()
  in
  ignore (Thread.create watch ())

(* The kinds the agent serves, as rig agent's help names them: one per path of
   Rig_gpus.paths, by its name. A kind is every GPU of its path, listed and
   opened at the controller's first call for it, which lists no other path:
   no device where the path sees none, and the path's refusal where it cannot
   list them. The driver-less paths search [firmware] first. *)
let kinds firmware =
  let rec open_all acc = function
    | [] -> Ok (List.rev acc)
    | (g : Rig_gpus.gpu) :: gs -> (
        match g.open_ () with
        | Ok d -> open_all (d :: acc) gs
        | Error _ as e -> e)
  in
  let kind (p : Rig_gpus.path) =
    let all () =
      match p.gpus () with
      | Ok gpus -> open_all [] gpus
      | Error why -> Error (p.kind ^ ": " ^ why)
    in
    (p.kind, all)
  in
  List.map kind (Rig_gpus.paths ~firmware ())

let agent ~firmware host port =
  let key =
    match Result.bind (read_key Unix.stdin) Rig_remote.key with
    | Ok k -> k
    | Error why -> fail why
  in
  orphaned ();
  (match lock () with Ok () -> () | Error why -> fail why);
  let a =
    match Rig_remote.listen ~key host port with
    | Ok a -> a
    | Error why -> fail why
  in
  say (Line.Listening (Address.with_port host (Rig_remote.port a)));
  match Rig_remote.serve a (kinds firmware) with
  | Ok () -> exit 0
  | Error _ -> exit 123
