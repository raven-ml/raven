(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

(* Each line in one write, so that it never splits around the program's output,
   and whatever stderr has become. *)
let sayf fmt =
  Printf.ksprintf (fun s -> Proc.write Unix.stderr ("rig: " ^ s ^ "\n")) fmt

(* The failures in a row with one cause after which rig run gives up. *)
let restarts = 3

(* rig.remote's silence bound: a job fails when a connection carries no byte for
   this long (Rig_remote, Failure). *)
let silence_s = 10.

(* How long the processes of a failed job have to end by themselves. It must
   exceed [silence_s]: then each process learns of the failure through the job
   and reports it before rig run ends it, and the cause does not depend on which
   came first. *)
let exit_s = silence_s +. 5.

(* Between tries to reach a machine that does not answer, and ssh's connect
   timeout. *)
let retry_s = 5.
let key_bytes = 32
let after s = Unix.gettimeofday () +. s

(* Machines *)

type machine = {
  name : string;  (** As written in [--on]. *)
  host : string;  (** For ssh: the name, its brackets taken off. *)
  address : string;  (** Where its agent listens, resolved here. *)
}

(* The host name ssh reaches [host] at: [ssh -G host]'s. *)
let hostname host =
  match Unix.open_process_args_in "ssh" [| "ssh"; "-G"; host |] with
  | exception Unix.Unix_error (e, _, _) ->
      Error (strf "ssh: %s" (Unix.error_message e))
  | ic -> (
      let rec find found =
        match In_channel.input_line ic with
        | None -> found
        | Some l when String.starts_with ~prefix:"hostname " l ->
            find (Some (String.sub l 9 (String.length l - 9)))
        | Some _ -> find found
      in
      let found = find None in
      match (Unix.close_process_in ic, found) with
      | Unix.WEXITED 0, Some h -> Ok h
      | Unix.WEXITED 0, None -> Error "ssh -G printed no host name"
      | st, _ -> Error (strf "ssh -G %s" (Proc.cause st)))

let addresses host =
  match Unix.getaddrinfo host "" [ Unix.AI_SOCKTYPE Unix.SOCK_STREAM ] with
  | l -> List.map (fun a -> a.Unix.ai_addr) l
  | exception Not_found -> []

(* Whether [a] is an address of this machine: whether a socket binds to it. A
   system without the address's family has none. *)
let binds a =
  match Unix.socket (Unix.domain_of_sockaddr a) Unix.SOCK_STREAM 0 with
  | exception Unix.Unix_error _ -> false
  | fd -> (
      Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
      match Unix.bind fd a with
      | () -> true
      | exception Unix.Unix_error _ -> false)

let machines ~misuse names =
  let found name =
    let host = Result.get_ok (Address.machine name) in
    match hostname host with
    | Ok h -> (name, host, h)
    | Error why ->
        sayf "%s: %s" name why;
        exit 123
  in
  let named = List.map found names in
  let first, _, h = List.hd named in
  if not (List.exists binds (addresses h)) then
    misuse
      (strf "--on: '%s' is not this machine; name this machine first" first);
  let machine (name, host, h) =
    match addresses h with
    | Unix.ADDR_INET (a, _) :: _ ->
        { name; host; address = Unix.string_of_inet_addr a }
    | _ ->
        sayf "%s: %s does not resolve here" name h;
        exit 123
  in
  List.map machine (List.tl named)

let key () =
  let b =
    In_channel.with_open_bin "/dev/urandom" (fun ic ->
        really_input_string ic key_bytes)
  in
  String.concat ""
    (List.init key_bytes (fun i -> strf "%02x" (Char.code b.[i])))

let silent m = strf "%s does not answer; waiting for it" m.name
let was c = if String.starts_with ~prefix:"killed" c then "was " ^ c else c

(* Sessions: [ssh M exec rig agent ADDRESS:0], a machine's half. The remote
   shell execs the half, whatever shell it is, so the session's process is the
   half and ends with it. *)

type session = {
  m : machine;
  ssh : int;
  mutable input : Unix.file_descr option;  (** Its half ends when closed. *)
  out : Line.reader;
  err : Line.reader;
  quiet : bool;  (** A try at a machine waited for: its errors are dropped. *)
  mutable greeted : bool;
  mutable port : int option;  (** Once its agent listens. *)
  mutable final : Line.t option;  (** [Closed], [Failed] or [Died]. *)
  mutable broken : string option;  (** It broke the session's protocol. *)
  mutable status : Unix.process_status option;
  mutable lost : bool;  (** ssh ended, or was killed, with no final line. *)
}

let session ~quiet key m =
  let in_r, in_w = Proc.pipe () in
  let out_r, out_w = Proc.pipe () in
  let err_r, err_w = Proc.pipe () in
  let agent = strf "'%s'" (Address.with_port m.address 0) in
  let timeout = strf "ConnectTimeout=%.0f" retry_s in
  let args =
    [|
      "ssh";
      "-T";
      "-o";
      "BatchMode=yes";
      "-o";
      timeout;
      m.host;
      "exec";
      "rig";
      "agent";
      agent;
    |]
  in
  let ssh =
    Fun.protect
      ~finally:(fun () -> List.iter Unix.close [ in_r; out_w; err_w ])
      (fun () -> Proc.spawn "ssh" args ~stdin:in_r ~stdout:out_w ~stderr:err_w)
  in
  Proc.write in_w (key ^ "\n");
  {
    m;
    ssh;
    input = Some in_w;
    out = Line.reader out_r;
    err = Line.reader err_r;
    quiet;
    greeted = false;
    port = None;
    final = None;
    broken = None;
    status = None;
    lost = false;
  }

let on_line s l =
  let broke why = s.broken <- Some why in
  match Line.of_string l with
  | _ when s.final <> None || s.broken <> None -> ()
  | Some (Line.Agent v) when not s.greeted ->
      if v = Line.version then s.greeted <- true
      else broke (strf "runs rig %s; this is rig %s" v Line.version)
  | _ when not s.greeted -> ()
  | Some Line.Waiting ->
      sayf "%s runs another agent of this user; waiting for it to end" s.m.name
  | Some (Line.Listening a) -> (
      match Address.host_port a with
      | Ok (_, p) -> s.port <- Some p
      | Error _ -> broke (strf "wrote %S" l))
  | Some ((Line.Closed | Line.Failed _ | Line.Died _) as f) -> s.final <- Some f
  | Some (Line.Agent _ | Line.Started) | None -> broke (strf "wrote %S" l)

(* Reaps, then reads: once ssh has exited, every byte it wrote is in its
   pipes. *)
let poll_session s =
  let exit = if s.status = None then Proc.reap s.ssh else None in
  List.iter (on_line s) (Line.read s.out);
  List.iter
    (fun l ->
      if not s.quiet then Proc.write Unix.stderr (strf "%s: %s\n" s.m.name l))
    (Line.read s.err);
  Option.iter
    (fun st ->
      s.status <- Some st;
      s.lost <- s.lost || (s.input <> None && s.final = None && s.broken = None);
      Option.iter Unix.close s.input;
      s.input <- None)
    exit

(* The program *)

type program = {
  prog : string;
  pid : int;
  report : Line.reader;
  mutable started : bool;
  mutable end_ : Line.t option;  (** [Closed] or [Failed]. *)
  mutable status : Unix.process_status option;
  mutable killed : bool;
}

let environment ss key report =
  let agent s =
    strf "%s=%s" s.m.name (Address.with_port s.m.address (Option.get s.port))
  in
  let own v = not (String.starts_with ~prefix:"RIG_REMOTE_" v) in
  Array.append
    (Array.of_list (List.filter own (Array.to_list (Unix.environment ()))))
    [|
      "RIG_REMOTE_AGENTS=" ^ String.concat "," (List.map agent ss);
      "RIG_REMOTE_KEY=" ^ key;
      strf "RIG_REMOTE_REPORT=%d" (Proc.fd_number report);
    |]

let program ss key prog args =
  let r, w = Proc.pipe () in
  let env = environment ss key w in
  let spawn () =
    Proc.spawn ~env prog
      (Array.of_list (prog :: args))
      ~stdin:Unix.stdin ~stdout:Unix.stdout ~stderr:Unix.stderr
  in
  match Proc.inherited w spawn with
  | pid ->
      Unix.close w;
      Ok
        {
          prog;
          pid;
          report = Line.reader r;
          started = false;
          end_ = None;
          status = None;
          killed = false;
        }
  | exception Unix.Unix_error (e, _, _) ->
      Unix.close r;
      Unix.close w;
      Error (strf "%s: %s" prog (Unix.error_message e))

let on_report p l =
  match Line.of_string l with
  | Some Line.Started -> p.started <- true
  | Some ((Line.Closed | Line.Failed _) as e) when p.end_ = None ->
      p.end_ <- Some e
  | _ -> ()

let poll_program p =
  if p.status = None then p.status <- Proc.reap p.pid;
  List.iter (on_report p) (Line.read p.report)

(* An attempt: its sessions and, once they listen, its program. *)

type attempt = { sessions : session list; mutable program : program option }

let exited a =
  Option.fold ~none:true ~some:(fun p -> p.status <> None) a.program

let ended a =
  exited a && List.for_all (fun (s : session) -> s.status <> None) a.sessions

(* Takes in what came, after waiting for anything, until [until] at most. An
   interrupt ends the job and rig run with it. It is taken after the polls:
   [waitpid] runs pending handlers, so a death the interrupt caused is seen with
   it. *)
let rec step ?until a =
  let readers =
    Option.fold ~none:[] ~some:(fun p -> [ p.report ]) a.program
    @ List.concat_map (fun s -> [ s.out; s.err ]) a.sessions
  in
  Proc.wait ?until readers;
  Option.iter poll_program a.program;
  List.iter poll_session a.sessions;
  Option.iter (interrupted a) (Proc.interrupted ())

(* Steps until [f a], or until [until]. *)
and wait ?until a f =
  let late () =
    Option.fold ~none:false ~some:(fun t -> Unix.gettimeofday () >= t) until
  in
  if not (f a || late ()) then begin
    step ?until a;
    wait ?until a f
  end

(* Ends every process of the attempt: the program killed, each session's input
   closed, so that its half ends its agent, then ssh killed if it does not end
   within [exit_s]. *)
and finish a =
  Option.iter
    (fun p ->
      if p.status = None then begin
        p.killed <- true;
        Proc.kill p.pid
      end)
    a.program;
  List.iter
    (fun s ->
      Option.iter Unix.close s.input;
      s.input <- None)
    a.sessions;
  wait ~until:(after exit_s) a ended;
  List.iter
    (fun (s : session) ->
      if s.status = None then begin
        s.lost <- true;
        Proc.kill s.ssh
      end)
    a.sessions;
  wait a ended

(* The program gets the signal, and [exit_s] to end, with its agents, before it
   is killed: a scheduler that preempts a job leaves it time to save its
   work. *)
and interrupted a s =
  sayf "interrupted; ending the job";
  Option.iter
    (fun p ->
      if p.status = None then begin
        Proc.signal p.pid s;
        wait ~until:(after exit_s) a (fun _ -> p.status <> None)
      end)
    a.program;
  finish a;
  Proc.die_by s

let quit a code =
  finish a;
  exit code

(* Ranking *)

(* How a process of a failed attempt ended. A death after a report, or one rig
   run caused, is none. *)
type end_ = Death of string | Report of string | Silent of machine

let ends a =
  let program =
    match a.program with
    | Some { end_ = Some (Line.Failed why); _ } -> [ Report why ]
    | Some
        {
          end_ = None;
          status = Some st;
          started = true;
          killed = false;
          prog;
          _;
        } ->
        [ Death (strf "%s %s" prog (was (Proc.cause st))) ]
    | _ -> []
  in
  (* An agent that fails before it listens (its key, its lock) names no machine:
     the reason gets its machine's name. *)
  let session s =
    let named why = strf "%s: %s" s.m.name why in
    match (s.final, s.broken) with
    | Some (Line.Died c), _ ->
        [ Death (strf "the agent on %s %s" s.m.name (was c)) ]
    | Some (Line.Failed why), _ when s.port <> None -> [ Report why ]
    | Some (Line.Failed why), _ | _, Some why -> [ Report (named why) ]
    | _ when s.lost -> [ Silent s.m ]
    | _ -> []
  in
  program @ List.concat_map session a.sessions

let started a = Option.fold ~none:false ~some:(fun p -> p.started) a.program

(* A failed attempt: its cause, the machines that do not answer, and whether the
   program had started the job. *)
type failure = { cause : string; down : machine list; started : bool }

(* The cause is the first death no process reported, then the first report, then
   the first machine that does not answer. *)
let rank a =
  let ends = ends a in
  let death = function Death c -> Some c | _ -> None
  and report = function Report c -> Some c | _ -> None
  and silence = function Silent m -> Some (silent m) | _ -> None in
  let cause =
    List.find_map (fun f -> List.find_map f ends) [ death; report; silence ]
    |> Option.value ~default:"the job ended"
  in
  let down = List.filter_map (function Silent m -> Some m | _ -> None) ends in
  { cause; down; started = started a }

(* Lets the processes of a failed attempt end by themselves, ends the rest, and
   ranks how they ended. Before the program started the job, no job tells the
   agents of the failure: only the program is waited for. *)
let collect a =
  wait ~until:(after exit_s) a (if started a then ended else exited);
  finish a;
  rank a

(* A start *)

(* Why a session ended before its agent listened, if it did. *)
let refused s =
  match (s.broken, s.final, s.status) with
  | Some why, _, _ | _, Some (Line.Failed why), _ -> Some why
  | _, Some (Line.Died c), _ -> Some (strf "its agent %s" (was c))
  | _, Some Line.Closed, _ -> Some "its agent ended"
  | _, _, Some st ->
      Some (strf "ssh %s before its agent listened" (Proc.cause st))
  | _ -> None

(* Starts a session per machine and waits until each listens; with the machines
   waited for. The first start ends at its first failure. A restart waits for
   every machine: when one does not answer, it ends the start and tries again
   after [retry_s], the errors of that machine's tries dropped. *)
let rec start ~first ~waited key machines =
  let quiet m = List.memq m waited in
  let ss = List.map (fun m -> session ~quiet:(quiet m) key m) machines in
  let a = { sessions = ss; program = None } in
  let listening s = s.port <> None in
  let settled s = listening s || refused s <> None in
  wait a (fun _ ->
      List.for_all settled ss
      || (first && List.exists (fun s -> refused s <> None) ss));
  match List.filter (fun s -> not (listening s)) ss with
  | [] -> Ok (a, waited)
  | down when first ->
      let s, why =
        List.find_map (fun s -> Option.map (fun w -> (s, w)) (refused s)) down
        |> Option.get
      in
      sayf "%s: %s" s.m.name why;
      quit a 123
  (* No program runs: nothing tells the listening agents the start failed. *)
  | down when List.exists (fun s -> not s.lost) down ->
      finish a;
      Error (rank a, waited)
  | down ->
      let lost = List.map (fun s -> s.m) down in
      List.iter (fun m -> if not (quiet m) then sayf "%s" (silent m)) lost;
      finish a;
      wait ~until:(after retry_s) a (fun _ -> false);
      let waited =
        lost @ List.filter (fun m -> not (List.memq m lost)) waited
      in
      start ~first ~waited key machines

(* Attempts *)

(* What ends a running attempt: the program's close, its exit before it started
   the job, or any failure. *)
let event a p =
  (* An agent closes its job, and may end, before the program reports its close:
     a session that ended closed is no failure. *)
  let fails s =
    match (s.broken, s.final) with
    | Some _, _ | _, Some (Line.Failed _ | Line.Died _) -> true
    | _, Some Line.Closed -> false
    | _, _ -> s.status <> None
  in
  match (p.end_, p.status) with
  | Some Line.Closed, _ -> Some `Closed
  | None, Some st when not p.started -> Some (`Early st)
  | Some _, _ | _, Some _ -> Some `Failed
  | None, None when List.exists fails a.sessions -> Some `Failed
  | None, None -> None

let early a p st =
  let how =
    match st with
    | Unix.WEXITED n -> strf "exited with status %d" n
    | st -> was (Proc.cause st)
  in
  sayf "%s %s before starting the job" p.prog how;
  quit a (Proc.status st)

(* The program closed the job: it ends when it ends, and its agents after it. *)
let orderly a p =
  wait a (fun _ -> p.status <> None);
  wait ~until:(after exit_s) a ended;
  finish a;
  exit (Proc.status (Option.get p.status))

let restarting ~count = function
  | [] -> sayf "restarting the job (%d of %d)" count restarts
  | ms ->
      let names = String.concat ", " (List.map (fun m -> m.name) ms) in
      let verb = if List.length ms = 1 then "answers" else "answer" in
      sayf "%s %s; restarting the job (%d of %d)" names verb count restarts

(* Runs one attempt to its end, and is its failure. A restart says so once its
   start ended, failed or not, so that every attempt is counted aloud. *)
let attempt ~first ~count ~waited machines prog args =
  let key = key () in
  let started = start ~first ~waited key machines in
  (match started with
  | (Ok (_, waited) | Error (_, waited)) when not first ->
      restarting ~count waited
  | _ -> ());
  match started with
  | Error (f, _) -> f
  | Ok (a, _) -> (
      match program a.sessions key prog args with
      | Error why when first ->
          sayf "%s" why;
          quit a 123
      | Error why ->
          finish a;
          { cause = why; down = []; started = false }
      | Ok p -> (
          a.program <- Some p;
          wait a (fun a -> event a p <> None);
          match event a p with
          | Some `Closed -> orderly a p
          | Some (`Early st) -> early a p st
          | Some `Failed | None -> (
              let f = collect a in
              match p with
              | {
               started = false;
               end_ = None;
               killed = false;
               status = Some st;
               _;
              } ->
                  early a p st
              | { started = false; _ } when first ->
                  sayf "the job did not start: %s" f.cause;
                  exit 123
              | _ -> f)))

(* Failures in a row count toward [restarts]: those before the program started
   the job whatever their causes, those after it with one cause. *)
let run ~misuse names prog args =
  (* Before ssh -G runs: an ignored SIGCHLD would have it reaped by the
     kernel. *)
  Proc.signals [ Sys.sigint; Sys.sigterm; Sys.sighup ];
  let machines = machines ~misuse names in
  let rec loop ~count ~last ~waited =
    let f = attempt ~first:(count = 0) ~count ~waited machines prog args in
    sayf "job failed: %s" f.cause;
    List.iter (fun m -> if silent m <> f.cause then sayf "%s" (silent m)) f.down;
    let cause = if f.started then Some f.cause else None in
    let n = if cause = last then count + 1 else 1 in
    if n > restarts then begin
      if f.started then
        sayf "the job failed %d times in a row with this cause; giving up" n
      else sayf "the job failed %d times in a row before starting; giving up" n;
      exit 123
    end;
    loop ~count:n ~last:cause ~waited:f.down
  in
  loop ~count:0 ~last:None ~waited:[]
