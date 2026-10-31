(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Wire = Rig_remote_proxy.Wire
module Link = Rig_remote_proxy.Link

let strf = Printf.sprintf

(* The largest listen backlog: Windows' SOMAXCONN, which asks for the system's
   largest, and POSIX systems cut a larger one to their own bound (somaxconn on
   Linux and macOS). *)
let backlog = 0x7fff_ffff

(* How long a connection and the job's other agents' connections take at
   most. *)
let join_s = 10.

(* Signals

   The library's calls that build its state run with the calling thread's
   signals blocked, and unblock them only across their waits ([rig_mask],
   [rig_unmask], [rig_unblock] and [rig_block] in [rig_edge.h]); its
   finalisers and exit handler take none, even in their waits
   ([rig_shield]). A thread it starts takes the mask its creator has inside
   [masked], all blocked, for its life. *)
external mask : unit -> unit = "caml_rig_edge_mask" [@@noalloc]
external unmask : unit -> unit = "caml_rig_edge_unmask"
external shield : unit -> unit = "caml_rig_edge_shield" [@@noalloc]
external unshield : unit -> unit = "caml_rig_edge_unshield"
external unblock : unit -> unit = "caml_rig_edge_unblock" [@@noalloc]
external block : unit -> unit = "caml_rig_edge_block"
external reraise : exn -> 'a = "%reraise"

let masked f =
  mask ();
  match f () with
  | r ->
      unmask ();
      r
  | exception e ->
      unmask ();
      reraise e

let shielded f =
  shield ();
  match f () with
  | r ->
      unshield ();
      r
  | exception e ->
      unshield ();
      reraise e

let unblocked f x =
  unblock ();
  match f x with
  | r ->
      block ();
      r
  | exception e ->
      block ();
      reraise e

let spawn f x = masked (fun () -> Thread.create f x)

(* Reports to a launcher *)

external report_open : int -> Unix.file_descr option
  = "caml_rig_remote_report_open"

external process_id : unit -> int = "caml_rig_remote_process_id" [@@noalloc]

let report_var = "RIG_REMOTE_REPORT"

type report = {
  fd : Unix.file_descr;
  pid : int;  (** The process that reports. *)
  lock : Mutex.t;  (** Held across a line's write. *)
  holder : int Atomic.t;  (** The thread that holds [lock], or [nobody]. *)
  mutable started : bool;
  mutable ended : bool;
}

let nobody = -1

let take name =
  let v = Sys.getenv_opt name in
  if Option.is_some v then Unix.unsetenv name;
  v

let report fd =
  let n =
    if fd <> "" && String.for_all (fun c -> '0' <= c && c <= '9') fd then
      int_of_string_opt fd
    else None
  in
  match Option.bind n report_open with
  | Some fd ->
      Ok
        {
          fd;
          pid = process_id ();
          lock = Mutex.create ();
          holder = Atomic.make nobody;
          started = false;
          ended = false;
        }
  | None -> Error (strf "%s=%S is no open file descriptor" report_var fd)

(* [report_line fd pid line] writes [line] on [fd] if this process is [pid],
   ignoring every error, and runs no OCaml code: no signal handler, finaliser
   or Gc.Memprof callback. *)
external report_line : Unix.file_descr -> int -> string -> unit
  = "caml_rig_remote_report_line"

(* [s] as a line, a reason's newlines as spaces. *)
let line s = String.map (function '\n' -> ' ' | c -> c) s ^ "\n"

(* [locked r f] is [f ()] under [r]'s lock, so that a call from another thread
   returns once the line [f] writes is written. Each [f] claims its line and
   writes it in one step, which allocates nothing and raises nothing. A signal
   handler, finaliser or Gc.Memprof callback that finds this thread the holder
   runs [f ()] without the lock, which it would otherwise wait for for good:
   [f] writes the line unless the call it interrupts wrote it. A signal
   handler that runs inside [Mutex.lock] runs before the take, with the lock
   not this thread's. *)
let locked r f =
  let me = Thread.id (Thread.self ()) in
  if Atomic.get r.holder = me then f ()
  else begin
    Mutex.lock r.lock;
    Atomic.set r.holder me;
    f ();
    Atomic.set r.holder nobody;
    Mutex.unlock r.lock
  end

let started = function
  | None -> ()
  | Some r ->
      let l = line "started" in
      locked r @@ fun () ->
      if not (r.started || r.ended) then begin
        r.started <- true;
        report_line r.fd r.pid l
      end

let ended report e =
  match report with
  | None -> ()
  | Some r ->
      let l =
        line (match e with `Closed -> "closed" | `Failed why -> "failed " ^ why)
      in
      locked r @@ fun () ->
      if not r.ended then begin
        r.ended <- true;
        report_line r.fd r.pid l
      end

(* The job's fate *)

(* Fails the agent's process with [j]'s root cause once [j] fails, after it
   reports the failure on [report]; and fails [j] once a device of the process
   is lost other than by a close. Every device an agent opens is its job's. *)
let watch report j =
  let rec loop () =
    match Link.wait j ~ms:1000 with
    | Link.Failed why ->
        ended report (`Failed why);
        Rig.fail why
    | Link.Closed -> ()
    | Link.Open ->
        Option.iter (Link.fail j) (Rig.failure ());
        loop ()
  in
  ignore (spawn loop ())

(* Addresses *)

let resolve host port =
  match
    Unix.getaddrinfo host (string_of_int port)
      [ Unix.AI_SOCKTYPE Unix.SOCK_STREAM ]
  with
  | [] -> Error (strf "%s does not resolve" host)
  | a :: _ -> Ok a.Unix.ai_addr
  | exception Not_found -> Error (strf "%s does not resolve" host)

(* A TCP connection to [host] and [port], within [s] seconds. A signal that
   interrupts the wait leaves it to go on. *)
let dial_tcp ~s host port =
  match resolve host port with
  | Error _ as e -> e
  | Ok addr -> (
      let until = Unix.gettimeofday () +. s in
      let rec finish fd =
        let left = Float.max 0. (until -. Unix.gettimeofday ()) in
        match Wire.ready [ fd ] `Write left with
        | exception Unix.Unix_error (Unix.EINTR, _, _) -> finish fd
        | None -> Error (strf "no answer within %.0f s" s)
        | Some _ -> (
            match Unix.getsockopt_error fd with
            | None -> Ok ()
            | Some e -> Error (Unix.error_message e))
      in
      let connect fd =
        Unix.set_nonblock fd;
        let r =
          match Unix.connect fd addr with
          | () -> Ok ()
          | exception
              Unix.Unix_error
                ((Unix.EINPROGRESS | Unix.EWOULDBLOCK | Unix.EINTR), _, _) ->
              finish fd
        in
        if Result.is_ok r then Unix.clear_nonblock fd;
        r
      in
      let domain = Unix.domain_of_sockaddr addr in
      match Unix.socket ~cloexec:true domain Unix.SOCK_STREAM 0 with
      | exception Unix.Unix_error (e, _, _) -> Error (Unix.error_message e)
      | fd -> (
          match connect fd with
          | Ok () -> Ok fd
          | Error _ as e ->
              Unix.close fd;
              e
          | exception Unix.Unix_error (e, _, _) ->
              Unix.close fd;
              Error (Unix.error_message e)))

(* Changes

   A count of an agent's changes, read and counted under its lock, whose wait
   for the next one returns to OCaml at least every 200 ms, where a pending
   signal's handler runs: OCaml's Condition has no timed wait. *)
type changes

external changes : unit -> changes = "caml_rig_remote_changes"
external change : changes -> unit = "caml_rig_remote_change" [@@noalloc]
external count : changes -> int = "caml_rig_remote_changes_count" [@@noalloc]

(* [await c n] waits until [c] counts more than [n] changes, at most 200 ms. *)
external await : changes -> int -> unit = "caml_rig_remote_changes_await"

(* Agents *)

(* The objects the controller names on this machine. *)
type obj =
  | Buffer of Rig.Buffer.t
  | Rail of Rig_remote_abi.end_
  | Program of Rig_program.loaded

(* Where an agent is in its life once a serve claimed it: each change is one
   step under its lock, so two controllers find each other. *)
type phase =
  | Claimed  (** A serve waits for a controller. *)
  | Admitting of Link.job
      (** A controller was admitted, its handshake not yet done: the job. *)
  | Serving of { job : Link.job; self : int; controller : Link.t }
      (** A job, this agent's process in it and the controller's link. *)

module Ids = Map.Make (Int)

type t = {
  key : string;
  socket : Unix.file_descr;
  port : int;
  claimed : bool Atomic.t;
      (** Once a serve claimed it. The claim takes no lock, and the sections
          under [lock] run only once the agent is claimed, so a callback
          inside one that calls [serve] raises [Invalid_argument]. *)
  lock : Mutex.t;
  changes : changes;
  mutable phase : phase;  (** Read once [claimed]. *)
  mutable accepted : Unix.file_descr Ids.t;
      (** Connections of the job's agents before this one, admitted, waiting for
          the join that names their machines. Each change is one store, so an
          exception, an asynchronous one too, keeps or drops it whole. *)
  mutable peers : Link.t Ids.t;  (** The job's other agents' links, as above. *)
  mutable stopped : bool;  (** Once it listens no more. *)
  mutable broken : string option;
      (** Once its socket failed for good, so that it accepts no more: why. *)
}

let close_quietly fd = try Unix.close fd with Unix.Unix_error _ -> ()

let listen ~key host port =
  masked @@ fun () ->
  match resolve host port with
  | Error _ as e -> e
  | Ok addr -> (
      let refused e =
        Error (strf "listening at %s:%d: %s" host port (Unix.error_message e))
      in
      let domain = Unix.domain_of_sockaddr addr in
      match Unix.socket ~cloexec:true domain Unix.SOCK_STREAM 0 with
      | exception Unix.Unix_error (e, _, _) -> refused e
      | fd -> (
          (* From the socket on, an exception closes it: a [Unix_error] of
             its setup answers [Error], and another, such as Out_of_memory or
             a Gc.Memprof callback's, is raised again. The handler allocates
             nothing before the close. *)
          match
            (* On POSIX, a port whose closed connections linger binds again;
               on Windows the option would let two sockets bind one port. *)
            if not Sys.win32 then Unix.setsockopt fd Unix.SO_REUSEADDR true;
            Unix.bind fd addr;
            Unix.listen fd backlog;
            (* A connection reset between the wait and accept leaves nothing
               to accept: a blocking accept would then wait for the next one,
               and [stop] for it (Stevens, UNIX Network Programming, 16.6). *)
            Unix.set_nonblock fd;
            let port =
              match Unix.getsockname fd with
              | Unix.ADDR_INET (_, p) -> p
              | Unix.ADDR_UNIX _ -> port
            in
            Ok
              {
                key;
                socket = fd;
                port;
                claimed = Atomic.make false;
                lock = Mutex.create ();
                changes = changes ();
                phase = Claimed;
                accepted = Ids.empty;
                peers = Ids.empty;
                stopped = false;
                broken = None;
              }
          with
          | r -> r
          | exception Unix.Unix_error (e, _, _) ->
              close_quietly fd;
              refused e
          | exception e ->
              close_quietly fd;
              reraise e))

let port a = a.port

(* A handshake's state, made before its thread: the job its controller's
   admission made, and whether a step took its connection. *)
type shake = { mutable mine : Link.job option; mutable handed : bool }

(* Admits one controller, then the agents of its job before this one. A
   controller's admission makes the job, which [st] keeps for its handshake: a
   second controller finds it. An exception between the job's making and the
   stores that record it closes it, which would otherwise refuse every later
   job of the process; the stores come after the last allocation. *)
let admit a st p =
  Mutex.protect a.lock @@ fun () ->
  match (p, a.phase) with
  | Wire.Controller, Claimed when Option.is_some a.broken ->
      Error (Option.get a.broken)
  | Wire.Controller, Claimed when a.stopped -> Error "the agent stopped"
  | Wire.Controller, Claimed -> (
      match Link.job () with
      | exception Invalid_argument why -> Error why
      | j -> (
          match (Admitting j, Some j) with
          | admitting, mine ->
              a.phase <- admitting;
              st.mine <- mine;
              Ok ()
          | exception e ->
              Link.close j;
              reraise e))
  | Wire.Controller, (Admitting _ | Serving _) ->
      Error "the agent serves another job"
  | Wire.Agent j, Serving { self; _ } when j < self -> Ok ()
  | Wire.Agent _, _ -> Error "the agent does not expect this agent"

(* Runs one connection's handshake, and makes its link if it is admitted. A
   controller admitted whose handshake then fails gives its job back, and so
   does one admitted once [a] stopped, which no serve would serve. *)
let shake a st fd =
  let give fd =
    st.handed <- true;
    close_quietly fd
  in
  let resolve r () =
    match (r, st.mine) with
    | Ok (Wire.Controller, Wire.Agent self), Some job when not a.stopped ->
        st.handed <- true;
        let controller =
          Link.make job fd ~name:"controller" ~peer:Wire.Controller
        in
        a.phase <- Serving { job; self; controller };
        change a.changes
    | Ok (Wire.Agent i, _), None ->
        if a.stopped || Ids.mem i a.accepted || Ids.mem i a.peers then give fd
        else begin
          a.accepted <- Ids.add i fd a.accepted;
          st.handed <- true
        end
    | _, Some job ->
        Link.close job;
        a.phase <- Claimed;
        change a.changes;
        give fd
    | _, None -> give fd
  in
  let r = Wire.accept fd ~key:a.key ~admit:(admit a st) in
  Mutex.protect a.lock (resolve r)

(* The handshake's thread. [fd] is the handshake's until [st.handed] is set:
   the step that gives it to a link, to [accepted] or to a close sets
   [st.handed] first, so an exception that lands between leaves it open,
   never closed twice. An exception, such as Out_of_memory or a Gc.Memprof
   callback's, that ends the handshake gives the job back if [a] still admits
   it, and closes [fd] if no step took it. Its handler comes before the
   thread's first allocation, and allocates nothing before the close. *)
let handshake a st fd =
  match shake a st fd with
  | () -> ()
  | exception e ->
      Mutex.lock a.lock;
      let admitted =
        match (st.mine, a.phase) with
        | Some j, Admitting j' when j == j' ->
            a.phase <- Claimed;
            change a.changes;
            true
        | _ -> false
      in
      Mutex.unlock a.lock;
      (match st.mine with Some j when admitted -> Link.close j | _ -> ());
      if not st.handed then close_quietly fd;
      reraise e

(* Accepting *)

(* The pauses after an error of accept that passes, as Go's net/http server
   makes them: the first, doubled after each error in a row up to the last.
   Once the system has descriptors again, a connection waits a pause at
   most. *)
let first_pause = 0.005
let last_pause = 1.

external eproto : unit -> int = "caml_rig_remote_eproto"
external enonet : unit -> int = "caml_rig_remote_enonet"

(* The system's EPROTO and ENONET, -1 where it has none. *)
let eproto = eproto ()
let enonet = enonet ()

(* Whether an error [e] of accept passes. The system may lack descriptors,
   buffers or memory for now; or the connection accept took failed, which
   Linux's accept(2) reports as the network's errors and asks a server to
   retry (ENETDOWN, EPROTO, ENOPROTOOPT, EHOSTDOWN, ENONET, EHOSTUNREACH,
   EOPNOTSUPP, ENETUNREACH), and as EPERM when a firewall forbids it. Another
   error, of the socket itself such as EBADF or EINVAL, or one no system
   documents for accept, recurs: retrying it would leave the agent deaf
   without saying so. *)
let passes (e : Unix.error) =
  match e with
  | EMFILE | ENFILE | ENOBUFS | ENOMEM | ECONNABORTED | ECONNRESET | EPERM
  | ENETDOWN | ENETUNREACH | EHOSTDOWN | EHOSTUNREACH | ENOPROTOOPT
  | EOPNOTSUPP | ETIMEDOUT ->
      true
  | EUNKNOWNERR n -> n = eproto || n = enonet
  | _ -> false

(* Ends [a]'s accepting after an error that recurs, [broken] its reason:
   fails the job it serves, and wakes a serve that waits for a controller.
   The failure, which may wait for an abort to leave, runs outside [a]'s
   lock. It allocates nothing, so that the acceptor's handler can end
   accepting with a reason made before. *)
let break a broken =
  Mutex.lock a.lock;
  a.broken <- broken;
  change a.changes;
  let phase = a.phase in
  Mutex.unlock a.lock;
  match (phase, broken) with
  | Serving { job; _ }, Some why -> Link.fail job why
  | _ -> ()

(* Why [a] breaks when an exception ends its acceptor, such as Out_of_memory
   or a Gc.Memprof callback's. *)
let acceptor_raised = Some "the agent's acceptor raised an exception"

(* Starts the handshake of the accepted [fd] on a thread of its own, or
   refuses it if the system gives it no thread. [fd] blocks first: macOS and
   Windows give it the listener's non-blocking mode. [fd] is the handshake's
   once its thread is made; an exception before closes it. *)
let start a fd =
  match Unix.clear_nonblock fd with
  | exception Unix.Unix_error _ -> close_quietly fd
  | exception e ->
      close_quietly fd;
      reraise e
  | () -> (
      match spawn (handshake a { mine = None; handed = false }) fd with
      | _ -> ()
      | exception Sys_error why ->
          Wire.refuse fd ("the agent cannot start a handshake: " ^ why)
      | exception Out_of_memory ->
          Wire.refuse fd
            "the agent cannot start a handshake: no memory for a thread"
      | exception e ->
          close_quietly fd;
          reraise e)

(* Accepts connections until [wake] is readable, each handshake on a thread of
   its own. A job assumes a trusted network, so only the job's processes
   connect: the listener takes every connection, and refuses one before its
   proof only when the system gives it no thread. After an error that passes
   it pauses, then accepts again; after another it breaks [a]. An exception
   that ends it, which its thread blocks every signal against, breaks [a]
   too: an acceptor that ended silently would leave the agent deaf, and a
   serve waiting for a controller that never comes. *)
let accept a wake =
  let err what e =
    Some (strf "the agent's %s failed: %s" what (Unix.error_message e))
  in
  let rec loop pause =
    match Wire.ready [ wake; a.socket ] `Read (-1.) with
    | exception Unix.Unix_error (Unix.EINTR, _, _) -> loop pause
    | exception Unix.Unix_error (e, _, _) -> break a (err "wait" e)
    | Some fd when fd == wake -> ()
    | Some _ | None -> accept_one pause
  and accept_one pause =
    match Unix.accept ~cloexec:true a.socket with
    | exception
        Unix.Unix_error ((Unix.EINTR | Unix.EAGAIN | Unix.EWOULDBLOCK), _, _) ->
        loop pause
    | exception Unix.Unix_error (e, _, _) when passes e -> rest pause
    | exception Unix.Unix_error (e, _, _) -> break a (err "accept" e)
    | fd, _ ->
        start a fd;
        loop first_pause
  (* Waits [pause], or until [wake] is readable. *)
  and rest pause =
    let next = Float.min last_pause (2. *. pause) in
    match Wire.ready [ wake ] `Read pause with
    | exception Unix.Unix_error (Unix.EINTR, _, _) -> loop next
    | exception Unix.Unix_error (e, _, _) -> break a (err "wait" e)
    | None -> loop next
    | Some _ -> ()
  in
  loop first_pause

(* The acceptor's thread: its handler comes before its first allocation. *)
let accept_loop a wake =
  match accept a wake with
  | () -> ()
  | exception e ->
      break a acceptor_raised;
      reraise e

(* What a serve took, so that one handler gives it back: made before it
   takes anything, each part recorded by stores with nothing between that
   allocates or polls, as it is taken. *)
type held = {
  mutable report : report option;
  mutable wake : Unix.file_descr;
  mutable waker : Unix.file_descr;
  mutable paired : bool;  (** Once [wake] and [waker] are the pair's. *)
  mutable acceptor : Thread.t;
  mutable accepting : bool;  (** Once [acceptor] is the acceptor's. *)
  mutable released : bool;
}

let held () =
  {
    report = None;
    wake = Unix.stdin;
    waker = Unix.stdin;
    paired = false;
    acceptor = Thread.self ();
    accepting = false;
    released = false;
  }

(* Ends [a]'s listening, once: wakes the acceptor through [waker], waits for
   it, and closes the socket, the pair and the connections no join named. A
   handshake that ends once [a] stopped closes its own connection. It takes
   no signal, even in its wait for the acceptor, which ends as soon as the
   acceptor reads [waker]. Under the shield nothing allocates or polls:
   [Thread.join] blocks in a section that runs only signal handlers, all
   blocked here. Should an exception come, the shield ends before it is
   raised again. The connections no join named close after the shield, in
   a loop whose polls may cut it, which leaves only those not yet closed
   open. *)
let release a h =
  if not h.released then begin
    h.released <- true;
    shield ();
    match
      Mutex.lock a.lock;
      a.stopped <- true;
      let accepted = a.accepted in
      a.accepted <- Ids.empty;
      Mutex.unlock a.lock;
      if h.accepting then begin
        (try ignore (Unix.write_substring h.waker "x" 0 1)
         with Unix.Unix_error _ -> ());
        Thread.join h.acceptor
      end;
      close_quietly a.socket;
      if h.paired then begin
        close_quietly h.wake;
        close_quietly h.waker
      end;
      accepted
    with
    | accepted ->
        unshield ();
        Ids.iter (fun _ fd -> close_quietly fd) accepted
    | exception e ->
        unshield ();
        reraise e
  end

(* Serving *)

type state = {
  agent : t;
  self : int;  (** This agent's process in the job. *)
  job : Link.job;
  link : Link.t;  (** The controller's. *)
  kinds : (string * (unit -> (Rig.t list, string) result)) list;
  devices : (int, Rig.t) Hashtbl.t;  (** By id, the host [0]. *)
  opened : (string, Wire.account list) Hashtbl.t;
  objects : (int, obj) Hashtbl.t;
}

(* A refusal of the controller's request, or a failure of the job. *)
exception Refused of string

let no_memory d n = strf "%s: no memory for %d bytes" (Rig.name d) n

let device s id =
  match Hashtbl.find_opt s.devices id with
  | Some d -> d
  | None -> raise (Refused (strf "no device %d" id))

let buffer s id =
  match Hashtbl.find_opt s.objects id with
  | Some (Buffer b) -> b
  | _ -> raise (Refused (strf "no memory %d" id))

(* Refuses a new object [id] while the job holds one of that id. *)
let fresh s id =
  if Hashtbl.mem s.objects id then
    raise (Refused (strf "id %d names an object already" id))

let account s id d : Wire.account =
  let reaches, reached_by =
    Hashtbl.fold
      (fun id' d' (reaches, reached_by) ->
        if id' >= id then (reaches, reached_by)
        else
          ( (if Rig.reaches d d' then id' :: reaches else reaches),
            if Rig.reaches d' d then id' :: reached_by else reached_by ))
      s.devices ([], [])
  in
  {
    id;
    name = Rig.name d;
    arch = Rig.arch d;
    budget = Rig.budget d;
    alignment = Rig.alignment d;
    reaches = List.sort compare reaches;
    reached_by = List.sort compare reached_by;
  }

(* A join's dial of an agent, on a thread of its own: its outcome, or the
   exception it raised, once [state] has left [`Running] for [`Ended]. A join
   that ends first marks it [`Abandoned], and the dial then closes its own
   connection. *)
type dialing = {
  mutable dialled : (int * string * Unix.file_descr, string) result;
  mutable raised : exn;  (** [Not_raised] unless the dial raised. *)
  state : [ `Running | `Ended | `Abandoned ] Atomic.t;
}

exception Not_raised

let close_dialled d =
  match d.dialled with Ok (_, _, fd) -> close_quietly fd | Error _ -> ()

(* Connects to the job's agents after this one, each on a thread of its own,
   then makes links of the connections of those before it, all within
   [join_s]: a join takes as long as its slowest connection, whatever the
   job's size. Each link is named after its agent's machine, as the
   controller names it in [agents]. *)
let join s agents =
  let a = s.agent in
  if s.self > List.length agents then
    raise (Refused (strf "the job has no agent %d" s.self));
  let until = Unix.gettimeofday () +. join_s in
  let late name = strf "%s: no answer within %.0f s" name join_s in
  let dial j ({ name; host; port } : Wire.agent) =
    let left = until -. Unix.gettimeofday () in
    if left <= 0. then Error (late name)
    else
      match dial_tcp ~s:left host port with
      | Error why -> Error (strf "%s: %s" name why)
      | Ok fd -> (
          match
            Wire.dial fd ~key:a.key ~self:(Wire.Agent s.self)
              ~peer:(Wire.Agent j)
          with
          | Error why ->
              Unix.close fd;
              Error (strf "%s: %s" name why)
          | Ok () when Unix.gettimeofday () > until ->
              Unix.close fd;
              Error (late name)
          | Ok () -> Ok (j, name, fd))
  in
  (* Each dial runs on a thread of its own, its exception raised again on this
     thread. The join waits until every dial has ended, at most 200 ms at a
     time, then closes each connection no link took. An exception that ends
     the wait, a Ctrl-C's in it, abandons the dials still running: each
     closes its connection as it ends, within its bounds. A dial stores its
     outcome and ends with nothing between that allocates. *)
  let ended = changes () in
  let started = ref [] and linked = ref [] in
  let start (j, m) =
    let d =
      {
        dialled = Error (late m.Wire.name);
        raised = Not_raised;
        state = Atomic.make `Running;
      }
    in
    let run () =
      (match d.dialled <- dial j m with
      | () -> ()
      | exception e -> d.raised <- e);
      if not (Atomic.compare_and_set d.state `Running `Ended) then
        close_dialled d;
      change ended
    in
    (* [d] is recorded before its thread starts, so the join closes its
       connection whatever exception comes after. A dial whose thread does
       not start is marked ended. *)
    started := d :: !started;
    match spawn run () with
    | _ -> ()
    | exception e ->
        Atomic.set d.state `Ended;
        reraise e
  in
  let rec wait () =
    let n = count ended in
    if List.exists (fun d -> Atomic.get d.state = `Running) !started then begin
      unblocked (await ended) n;
      wait ()
    end
  in
  let close_unlinked () =
    List.iter
      (fun d ->
        if not (Atomic.compare_and_set d.state `Running `Abandoned) then
          match d.dialled with
          | Ok (_, _, fd) when not (List.memq fd !linked) -> close_quietly fd
          | Ok _ | Error _ -> ())
      !started
  in
  let result d =
    match d.raised with Not_raised -> d.dialled | e -> raise e
  in
  (* The note that a link took [fd] comes before the take: an exception
     between them leaves [fd] open, where one after the take would have
     [close_unlinked] close the link's descriptor, which the system may since
     have given to another file. *)
  let link (j, name, fd) =
    linked := fd :: !linked;
    let l = Link.make s.job fd ~name ~peer:(Wire.Agent j) in
    Mutex.protect a.lock (fun () -> a.peers <- Ids.add j l a.peers)
  in
  let refusal = function Error why -> Some why | Ok _ -> None in
  ( Fun.protect ~finally:close_unlinked @@ fun () ->
    List.filteri (fun i _ -> i + 1 > s.self) agents
    |> List.iteri (fun k m -> start (s.self + 1 + k, m));
    wait ();
    let dialled = List.rev_map result !started in
    match List.find_map refusal dialled with
    | Some why -> raise (Refused why)
    | None -> List.iter (function Ok d -> link d | Error _ -> ()) dialled );
  (* The agents before this one dial it; their handshakes run on the acceptor's
     threads. *)
  let before = List.filteri (fun i _ -> i + 1 < s.self) agents in
  let accepted () =
    Mutex.protect a.lock (fun () -> Ids.cardinal a.accepted)
  in
  while accepted () < List.length before && Unix.gettimeofday () < until do
    unblocked Thread.delay 0.01
  done;
  let missing =
    Mutex.protect a.lock @@ fun () ->
    List.concat
      (List.mapi
         (fun i ({ name; _ } : Wire.agent) ->
           let j = i + 1 in
           match Ids.find_opt j a.accepted with
           | None -> [ name ]
           | Some fd ->
               a.accepted <- Ids.remove j a.accepted;
               let l = Link.make s.job fd ~name ~peer:(Wire.Agent j) in
               a.peers <- Ids.add j l a.peers;
               [])
         before)
  in
  if missing <> [] then
    raise
      (Refused
         (strf "%s did not connect within %.0f s"
            (String.concat ", " missing)
            join_s));
  account s 0 Rig.host

let open_kind s kind =
  match Hashtbl.find_opt s.opened kind with
  | Some accounts -> accounts
  | None -> (
      match List.assoc_opt kind s.kinds with
      | None ->
          raise (Refused (strf "the agent serves no devices of kind %S" kind))
      | Some opener -> (
          match opener () with
          | Error why -> raise (Refused why)
          | Ok ds ->
              let first = Hashtbl.length s.devices in
              List.iteri (fun i d -> Hashtbl.replace s.devices (first + i) d) ds;
              let accounts =
                List.mapi (fun i d -> account s (first + i) d) ds
              in
              Hashtbl.replace s.opened kind accounts;
              accounts))

let peer_link s = function
  | Wire.Controller -> s.link
  | Wire.Agent j -> (
      match
        Mutex.protect s.agent.lock (fun () -> Ids.find_opt j s.agent.peers)
      with
      | Some l -> l
      | None -> raise (Refused (strf "no link to agent %d" j)))

let answer : type a. state -> a Wire.request -> a =
 fun s -> function
  | Wire.Join { agents } -> join s agents
  | Wire.Open kind -> open_kind s kind
  | Wire.Alloc { id; device = d; memory = m; bytes } -> (
      fresh s id;
      match Rig.Buffer.create ~memory:m (device s d) bytes with
      | b ->
          Hashtbl.replace s.objects id (Buffer b);
          true
      | exception Rig.Out_of_memory _ -> false)
  | Wire.Map { id; device = d; region } -> (
      fresh s id;
      match Rig.Buffer.borrow (device s d) (buffer s region) with
      | Some b ->
          Hashtbl.replace s.objects id (Buffer b);
          true
      | None -> false)
  | Wire.Load { id; binary } -> (
      fresh s id;
      let rails id =
        match Hashtbl.find_opt s.objects id with
        | Some (Rail e) -> Some e
        | _ -> None
      in
      match
        Rig_program.load_received ~rails binary
          ~device:(Hashtbl.find_opt s.devices)
      with
      | Ok p ->
          Hashtbl.replace s.objects id (Program p);
          Rig_program.threads p
      | Error why -> raise (Refused why)
      | exception Rig.Out_of_memory (d, n) -> raise (Refused (no_memory d n)))
  | Wire.Entry { image; name } -> (
      (* A program has one function, its run, which a run's words name by the
         program's id. *)
      match Hashtbl.find_opt s.objects image with
      | Some (Program _) when name = "run" -> Some image
      | _ -> None)
  | Wire.Rail { id; peer; send; receive } ->
      fresh s id;
      let l = peer_link s peer in
      let e =
        match Link.rail l ~id ~send ~receive with
        | e -> e
        | exception Out_of_memory ->
            raise (Refused (strf "rail %d: no memory for its areas" id))
      in
      (* Drop removes the name; a loaded program still owns the end it uses. *)
      Gc.finalise (fun _ -> shielded (fun () -> Link.release_rail l id)) e;
      Hashtbl.replace s.objects id (Rail e)

(* [f] of [b]'s bytes as the host reads them in place: [b]'s own memory on
   this machine's host, a copy of them elsewhere. *)
let with_host_bytes b f =
  let b =
    if Rig.Buffer.device b == Rig.host then b
    else begin
      let h = Rig.Buffer.create Rig.host (Rig.Buffer.length b) in
      Rig.Buffer.copy ~src:b ~dst:h;
      h
    end
  in
  Rig.Buffer.read b Bigarray.char f

let program s id =
  match Hashtbl.find_opt s.objects id with
  | Some (Program p) -> Some p
  | _ -> None

let region s id =
  match Hashtbl.find_opt s.objects id with
  | Some (Buffer b) -> Some b
  | _ -> None

(* Runs a program's run, whose words name it and its frame, and waits for the
   points its work ends at. *)
let run s words =
  match
    Rig_program.run_received words ~program:(program s) ~region:(region s)
  with
  | Error why -> raise (Refused why)
  | Ok points -> Array.iter Rig.Point.wait points

(* Copies [rows] rows of [width] bytes, row [r] from [r * src_pitch] bytes
   into [src] to [r * dst_pitch] bytes into [dst]: one copy where the rows
   lie end to end on both sides, and one per row otherwise. *)
let copy_rows ~src ~dst ~width ~rows ~src_pitch ~dst_pitch =
  let row b at = Rig.Buffer.view b ~first:at ~length:width in
  if width > 0 && rows > 0 then
    if rows = 1 || (src_pitch = width && dst_pitch = width) then
      Rig.Buffer.copy
        ~src:(Rig.Buffer.view src ~first:0 ~length:(width * rows))
        ~dst:(Rig.Buffer.view dst ~first:0 ~length:(width * rows))
    else
      for r = 0 to rows - 1 do
        Rig.Buffer.copy ~src:(row src (r * src_pitch))
          ~dst:(row dst (r * dst_pitch))
      done

(* Runs a hand-over's parts in order, sending the bytes of each copy into the
   controller's memory as it comes, then the word of its value. [local] holds
   one area per copy from Local, an empty one included, its rows one after
   another. A copy or a fill of no bytes into memory skips it: an empty remote
   buffer may have no driver object. *)
let hand_over s (h : Wire.handover) local =
  let next = ref 0 in
  let upload () =
    let a = local.(!next) in
    incr next;
    a
  in
  (* The bytes of a region a box of [rows] rows of [width] bytes at [pitch]
     spans. *)
  let region id offset ~width ~rows ~pitch =
    let length = if rows = 0 then 0 else ((rows - 1) * pitch) + width in
    Rig.Buffer.view (buffer s id) ~first:offset ~length
  in
  Array.iter
    (function
      | Wire.Words w -> run s w
      | Wire.Copy { src = Wire.Local; dst = Wire.Local; _ } ->
          raise (Refused "a copy from Local to Local")
      | Wire.Copy
          {
            src = Wire.Region { id; offset };
            dst = Wire.Local;
            width;
            rows;
            src_pitch;
            _;
          } ->
          let src = region id offset ~width ~rows ~pitch:src_pitch in
          let packed =
            if rows <= 1 || src_pitch = width then src
            else begin
              let h = Rig.Buffer.create Rig.host (width * rows) in
              copy_rows ~src ~dst:h ~width ~rows ~src_pitch ~dst_pitch:width;
              h
            end
          in
          with_host_bytes packed
            (Link.bytes s.link ~device:h.device ~value:h.value)
      | Wire.Copy
          {
            src = Wire.Local;
            dst = Wire.Region { id; offset };
            width;
            rows;
            dst_pitch;
            _;
          } ->
          let a = upload () in
          if width * rows > 0 then
            copy_rows ~src:(Rig.Buffer.of_bigarray a)
              ~dst:(region id offset ~width ~rows ~pitch:dst_pitch)
              ~width ~rows ~src_pitch:width ~dst_pitch
      | Wire.Copy
          {
            src = Wire.Region { id = i; offset = o };
            dst = Wire.Region { id = j; offset = p };
            width;
            rows;
            src_pitch;
            dst_pitch;
          } ->
          if width * rows > 0 then
            copy_rows
              ~src:(region i o ~width ~rows ~pitch:src_pitch)
              ~dst:(region j p ~width ~rows ~pitch:dst_pitch)
              ~width ~rows ~src_pitch ~dst_pitch
      | Wire.Fill { dst = Wire.Region { id; offset }; bytes; byte } ->
          if bytes > 0 then
            Rig.Buffer.fill (Rig.Buffer.view (buffer s id) ~first:offset
              ~length:bytes) byte
      | Wire.Fill { dst = Wire.Local; _ } -> raise (Refused "a fill of Local"))
    h.parts;
  Link.word s.link ~device:h.device h.value

(* Applies the controller's commands in order until it closes the job, then
   releases the job's memory and rails and closes its devices; or until the job
   fails. *)
let rec apply s =
  match Link.next s.link with
  | Error why -> Error why
  | Ok Wire.Close ->
      Hashtbl.clear s.objects;
      Hashtbl.iter (fun id d -> if id <> 0 then Rig.close d) s.devices;
      Ok ()
  | Ok (Wire.Request r) ->
      (* The agent's rig raises [Invalid_argument] on what the frame asks of it,
         such as a rail of no transfer: a refusal too. *)
      (match answer s r with
      | v -> Link.answer s.link r (Ok v)
      | exception (Refused why | Invalid_argument why) ->
          Link.answer s.link r (Error why));
      apply s
  | Ok (Wire.Drop id) ->
      (* A drop takes no answer: one of no object fails the job. *)
      if Hashtbl.mem s.objects id then Hashtbl.remove s.objects id
      else Link.fail s.job (strf "%s: a malformed frame" (Link.name s.link));
      apply s
  | Ok (Wire.Handover (h, local)) -> (
      match hand_over s h local with
      | () -> apply s
      | exception (Refused why | Invalid_argument why) ->
          Link.fail s.job why;
          apply s
      | exception Rig.Out_of_memory (d, n) ->
          Link.fail s.job (no_memory d n);
          apply s)

(* The job [a] serves, once a controller's handshake made it. A job that a
   broken accept met fails with its reason; a serve that has no job yet ends
   with it. The wait for a change starts from the count read with the phase,
   so a change after the read ends it. *)
let job a =
  let look () =
    match (a.phase, a.broken) with
    | Serving { job; self; controller }, broken ->
        `Serving ((job, self, controller), broken)
    | Claimed, Some why -> `Broken why
    | (Claimed | Admitting _), _ -> `Waiting (count a.changes)
  in
  let rec serving () =
    match Mutex.protect a.lock look with
    | `Waiting n ->
        unblocked (await a.changes) n;
        serving ()
    | `Broken why -> Error why
    | `Serving (((job, _, _) as serving), broken) ->
        Option.iter (Link.fail job) broken;
        Ok serving
  in
  serving ()

(* [unmasked f x] is [f x] with the calling thread's signals as [serve]'s
   caller had them, for the program's code that serving runs. It ends
   [serve]'s pair across [f] and takes it again after, where [unblocked]
   would open a window that the first wait inside [f] closes. *)
let unmasked f x =
  unmask ();
  match f x with
  | r ->
      mask ();
      r
  | exception e ->
      mask ();
      reraise e

(* Why a job fails when an exception ends [serve] other than its commands'. *)
let serve_raised = "the agent's serve raised an exception"

(* Why a job fails when a [Sys.Break], a Ctrl-C's, ends [serve] wherever it
   comes: its text, as for an exception of its commands. *)
let interrupted = Printexc.to_string Sys.Break

(* Serves [job], reporting its end on [h]'s report, and stops listening.
   From the job's watcher on, an exception, a thread the system does not give
   or one [apply] does not handle (a device's loss or a bug of the agent or an
   opener), leaves the job's objects in a state the controller cannot know:
   it fails the job with its text, which ends the controller's wait. A
   [Sys.Break] ends [serve] as it does in serve's own steps. *)
let serve_job a h kinds job self link =
  let r =
    match
      watch h.report job;
      let devices = Hashtbl.create 8 in
      Hashtbl.replace devices 0 Rig.host;
      let s =
        {
          agent = a;
          self;
          job;
          link;
          kinds;
          devices;
          opened = Hashtbl.create 4;
          objects = Hashtbl.create 64;
        }
      in
      unmasked apply s
    with
    | r -> r
    | exception (Sys.Break as e) -> reraise e
    | exception e ->
        let why =
          match e with
          | Rig.Lost (d, why) -> strf "%s lost: %s" (Rig.name d) why
          | e -> Printexc.to_string e
        in
        Link.fail job why;
        Error why
  in
  (* The agent listens no more before its close tells the controller the job
     ended. *)
  release a h;
  let r =
    match r with
    | Error _ as e -> e
    | Ok () -> (
        Link.close job;
        match Link.failed job with Some why -> Error why | None -> Ok ())
  in
  (match r with
  | Ok () -> ended h.report `Closed
  | Error why ->
      ended h.report (`Failed why);
      Rig.fail why;
      (* Each peer reads this process's abort before serve's caller exits. *)
      Link.close job);
  r

(* Ends the job [a] serves, if any, after an exception [e] that ends [serve]
   elsewhere than in its commands, or a [Sys.Break] anywhere: fails it,
   closes it, and reports its end, failing the process if the job failed. A
   controller still in its handshake finds [a] stopped and gives its job
   back. It allocates nothing before the close. *)
let abandon a report e =
  let why = match e with Sys.Break -> interrupted | _ -> serve_raised in
  Mutex.lock a.lock;
  let phase = a.phase in
  Mutex.unlock a.lock;
  match phase with
  | Serving { job; _ } -> (
      Link.fail job why;
      Link.close job;
      match Link.failed job with
      | None -> ended report `Closed
      | Some why ->
          ended report (`Failed why);
          Rig.fail why)
  | Claimed | Admitting _ -> ended report (`Failed why)

(* Serves one job. The pair and the acceptor are recorded in [h] as they are
   taken. *)
let run a h kinds =
  let wake, waker = Link.socket_pair () in
  h.wake <- wake;
  h.waker <- waker;
  h.paired <- true;
  let acceptor = spawn (accept_loop a) wake in
  h.acceptor <- acceptor;
  h.accepting <- true;
  match job a with
  | Ok (job, self, link) -> serve_job a h kinds job self link
  | Error why ->
      release a h;
      ended h.report (`Failed why);
      Error why

(* [serve] blocks signals but around the program's code it runs ([unmasked]),
   so that a Ctrl-C lands in its waits. From its claim the socket is
   [serve]'s, and an exception that ends it elsewhere than in its commands,
   or a Ctrl-C's [Sys.Break] wherever it comes, stops listening and ends the
   job [a] took, which no serve would serve. *)
let serve a kinds =
  let names = List.map fst kinds in
  if List.length (List.sort_uniq String.compare names) <> List.length names then
    invalid_arg "Rig_remote.serve: kinds names a kind twice";
  masked @@ fun () ->
  let h = held () in
  if Atomic.exchange a.claimed true then
    invalid_arg "Rig_remote.serve: the agent served already";
  match
    match take report_var with
    | None -> run a h kinds
    | Some fd -> (
        match report fd with
        | Ok r ->
            h.report <- Some r;
            run a h kinds
        | Error why ->
            release a h;
            Error why)
  with
  | r -> r
  | exception e ->
      release a h;
      abandon a h.report e;
      reraise e
