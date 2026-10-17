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

(* Reports to a launcher *)

external report_open : int -> Unix.file_descr option
  = "caml_rig_remote_report_open"

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
          pid = Unix.getpid ();
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

(* Fails the process with [j]'s root cause once [j] fails, after it reports the
   failure on [report]; and fails [j] once a device of the process is lost other
   than by a close. *)
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
  ignore (Thread.create loop ())

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

type t = {
  key : string;
  socket : Unix.file_descr;
  port : int;
  claimed : bool Atomic.t;
      (** Once a serve claimed it. The claim takes no lock, and the sections
          under [lock] run only once the agent is claimed, so a callback
          inside one that calls [serve] raises [Invalid_argument]. *)
  lock : Mutex.t;
  changed : Condition.t;
  mutable phase : phase;  (** Read once [claimed]. *)
  accepted : (int, Unix.file_descr) Hashtbl.t;
      (** Connections of the job's agents before this one, admitted, waiting for
          the join that names their machines. *)
  peers : (int, Link.t) Hashtbl.t;  (** The job's other agents' links. *)
  mutable stopped : bool;  (** Once it listens no more. *)
  mutable broken : string option;
      (** Once its socket failed for good, so that it accepts no more: why. *)
}

let listen ~key host port =
  match resolve host port with
  | Error _ as e -> e
  | Ok addr -> (
      let domain = Unix.domain_of_sockaddr addr in
      let fd = Unix.socket ~cloexec:true domain Unix.SOCK_STREAM 0 in
      match
        (* On POSIX, a port whose closed connections linger binds again; on
           Windows the option would let two sockets bind one port. *)
        if not Sys.win32 then Unix.setsockopt fd Unix.SO_REUSEADDR true;
        Unix.bind fd addr;
        Unix.listen fd backlog;
        (* A connection reset between the wait and accept leaves nothing to
           accept: a blocking accept would then wait for the next one, and
           [stop] for it (Stevens, UNIX Network Programming, 16.6). *)
        Unix.set_nonblock fd
      with
      | exception Unix.Unix_error (e, _, _) ->
          Unix.close fd;
          Error (strf "listening at %s:%d: %s" host port (Unix.error_message e))
      | () ->
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
              changed = Condition.create ();
              phase = Claimed;
              accepted = Hashtbl.create 4;
              peers = Hashtbl.create 4;
              stopped = false;
              broken = None;
            })

let port a = a.port
let close_quietly fd = try Unix.close fd with Unix.Unix_error _ -> ()

(* Admits one controller, then the agents of its job before this one. A
   controller's admission makes the job, which [mine] keeps for its handshake: a
   second controller finds it. *)
let admit a mine p =
  Mutex.protect a.lock @@ fun () ->
  match (p, a.phase) with
  | Wire.Controller, Claimed when Option.is_some a.broken ->
      Error (Option.get a.broken)
  | Wire.Controller, Claimed -> (
      match Link.job () with
      | j ->
          a.phase <- Admitting j;
          mine := Some j;
          Ok ()
      | exception Invalid_argument why -> Error why)
  | Wire.Controller, (Admitting _ | Serving _) ->
      Error "the agent serves another job"
  | Wire.Agent j, Serving { self; _ } when j < self -> Ok ()
  | Wire.Agent _, _ -> Error "the agent does not expect this agent"

(* Runs one connection's handshake, and makes its link if it is admitted. A
   controller admitted whose handshake then fails gives its job back. *)
let handshake a fd =
  let mine = ref None in
  let r = Wire.accept fd ~key:a.key ~admit:(admit a mine) in
  Mutex.protect a.lock @@ fun () ->
  match (r, !mine) with
  | Ok (Wire.Controller, Wire.Agent self), Some job ->
      let controller =
        Link.make job fd ~name:"controller" ~peer:Wire.Controller
      in
      a.phase <- Serving { job; self; controller };
      Condition.broadcast a.changed
  | Ok (Wire.Agent i, _), None ->
      if a.stopped || Hashtbl.mem a.accepted i || Hashtbl.mem a.peers i then
        close_quietly fd
      else Hashtbl.replace a.accepted i fd
  | _, Some job ->
      Link.close job;
      a.phase <- Claimed;
      Condition.broadcast a.changed;
      close_quietly fd
  | _, None -> close_quietly fd

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

(* Ends [a]'s accepting after an error that recurs, [why]: fails the job it
   serves, and wakes a serve that waits for a controller. The failure, which
   waits for its aborts to leave, runs outside [a]'s lock. *)
let break a why =
  let serving =
    Mutex.protect a.lock @@ fun () ->
    a.broken <- Some why;
    Condition.broadcast a.changed;
    match a.phase with
    | Serving { job; _ } -> Some job
    | Claimed | Admitting _ -> None
  in
  Option.iter (fun job -> Link.fail job why) serving

(* Starts the handshake of the accepted [fd] on a thread of its own, or
   refuses it if the system gives it no thread. [fd] blocks first: macOS and
   Windows give it the listener's non-blocking mode. *)
let start a fd =
  match Unix.clear_nonblock fd with
  | exception Unix.Unix_error _ -> close_quietly fd
  | () -> (
      match Thread.create (handshake a) fd with
      | _ -> ()
      | exception Sys_error why ->
          Wire.refuse fd ("the agent cannot start a handshake: " ^ why)
      | exception Out_of_memory ->
          Wire.refuse fd
            "the agent cannot start a handshake: no memory for a thread")

(* Accepts connections until [wake] is readable, each handshake on a thread of
   its own. A job assumes a trusted network, so only the job's processes
   connect: the listener takes every connection, and refuses one before its
   proof only when the system gives it no thread. After an error that passes
   it pauses, then accepts again; after another it breaks [a]. *)
let accept_loop a wake =
  let err what e =
    strf "the agent's %s failed: %s" what (Unix.error_message e)
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

(* Ends [a]'s listening: wakes the acceptor through [waker], waits for it, and
   closes the socket and the connections no join named. *)
let stop a waker acceptor =
  Mutex.protect a.lock (fun () -> a.stopped <- true);
  ignore (Unix.write_substring waker "x" 0 1);
  Thread.join acceptor;
  Unix.close a.socket;
  Mutex.protect a.lock @@ fun () ->
  Hashtbl.iter (fun _ fd -> Unix.close fd) a.accepted;
  Hashtbl.reset a.accepted

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
    most_parts = (if Rig.computes d then Rig.most_parts d else max_int);
    alignment = Rig.alignment d;
    reaches = List.sort compare reaches;
    reached_by = List.sort compare reached_by;
  }

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
     thread. Every thread ends before the join reads a result or leaves, an
     exception's way included, and a connection no link took is closed: a
     failed join leaves no dial running and no connection open. *)
  let started = ref [] and linked = ref [] in
  let start (j, m) =
    let r = ref (Ok (Error (late m.Wire.name))) in
    let run () = r := match dial j m with v -> Ok v | exception e -> Error e in
    started := (r, Thread.create run ()) :: !started
  in
  let close_unlinked () =
    List.iter
      (fun (r, t) ->
        Thread.join t;
        match !r with
        | Ok (Ok (_, _, fd)) when not (List.memq fd !linked) -> (
            try Unix.close fd with Unix.Unix_error _ -> ())
        | Ok _ | Error _ -> ())
      !started
  in
  let result (r, _) = match !r with Ok v -> v | Error e -> raise e in
  let link (j, name, fd) =
    let l = Link.make s.job fd ~name ~peer:(Wire.Agent j) in
    linked := fd :: !linked;
    Mutex.protect a.lock (fun () -> Hashtbl.replace a.peers j l)
  in
  let refusal = function Error why -> Some why | Ok _ -> None in
  ( Fun.protect ~finally:close_unlinked @@ fun () ->
    List.filteri (fun i _ -> i + 1 > s.self) agents
    |> List.iteri (fun k m -> start (s.self + 1 + k, m));
    List.iter (fun (_, t) -> Thread.join t) !started;
    let dialled = List.rev_map result !started in
    match List.find_map refusal dialled with
    | Some why -> raise (Refused why)
    | None -> List.iter (function Ok d -> link d | Error _ -> ()) dialled );
  (* The agents before this one dial it; their handshakes run on the acceptor's
     threads. *)
  let before = List.filteri (fun i _ -> i + 1 < s.self) agents in
  let accepted () =
    Mutex.protect a.lock (fun () -> Hashtbl.length a.accepted)
  in
  while accepted () < List.length before && Unix.gettimeofday () < until do
    Thread.delay 0.01
  done;
  let missing =
    Mutex.protect a.lock @@ fun () ->
    List.concat
      (List.mapi
         (fun i ({ name; _ } : Wire.agent) ->
           let j = i + 1 in
           match Hashtbl.find_opt a.accepted j with
           | None -> [ name ]
           | Some fd ->
               Hashtbl.remove a.accepted j;
               Hashtbl.replace a.peers j
                 (Link.make s.job fd ~name ~peer:(Wire.Agent j));
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
        Mutex.protect s.agent.lock (fun () -> Hashtbl.find_opt s.agent.peers j)
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
      Gc.finalise (fun _ -> Link.release_rail l id) e;
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

(* Runs a hand-over's parts in order, sending the bytes of each copy into the
   controller's memory as it comes, then the word of its value. [local] holds
   one area per copy from Local, an empty one included. A copy of no bytes into
   memory skips it: an empty remote buffer may have no driver object. *)
let hand_over s (h : Wire.handover) local =
  let next = ref 0 in
  let upload () =
    let a = local.(!next) in
    incr next;
    a
  in
  let side bytes = function
    | Wire.Region { id; offset } ->
        Rig.Buffer.view (buffer s id) ~first:offset ~length:bytes
    | Wire.Local -> Rig.Buffer.of_bigarray (upload ())
  in
  Array.iter
    (function
      | Wire.Words w -> run s w
      | Wire.Copy { src; dst = Wire.Local; bytes } ->
          with_host_bytes (side bytes src)
            (Link.bytes s.link ~device:h.device ~value:h.value)
      | Wire.Copy { src = Wire.Local; dst; bytes } ->
          let a = upload () in
          if bytes > 0 then
            Rig.Buffer.copy ~src:(Rig.Buffer.of_bigarray a)
              ~dst:(side bytes dst)
      | Wire.Copy { src; dst; bytes } ->
          if bytes > 0 then
            Rig.Buffer.copy ~src:(side bytes src) ~dst:(side bytes dst))
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
   with it. *)
let job a =
  let r =
    Mutex.protect a.lock @@ fun () ->
    let rec serving () =
      match (a.phase, a.broken) with
      | Serving { job; self; controller }, broken ->
          Ok ((job, self, controller), broken)
      | Claimed, Some why -> Error why
      | (Claimed | Admitting _), _ ->
          Condition.wait a.changed a.lock;
          serving ()
    in
    serving ()
  in
  match r with
  | Error _ as e -> e
  | Ok (((job, _, _) as serving), broken) ->
      Option.iter (Link.fail job) broken;
      Ok serving

(* Serves [job], reporting its end on [report]; [close] ends [a]'s listening. *)
let serve_job a kinds report ~close job self link =
  watch report job;
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
  (* An exception [apply] does not handle, a device's loss or a bug of the agent
     or an opener, leaves the job's objects in a state the controller cannot
     know: it fails the job with its text, which ends the controller's wait. *)
  let r =
    match apply s with
    | r -> r
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
  close ();
  let r =
    match r with
    | Error _ as e -> e
    | Ok () -> (
        Link.close job;
        match Link.failed job with Some why -> Error why | None -> Ok ())
  in
  (match r with
  | Ok () -> ended report `Closed
  | Error why ->
      ended report (`Failed why);
      Rig.fail why);
  r

(* Serves one job, reporting its end on [report]. *)
let run a kinds report =
  let wake, waker = Link.socket_pair () in
  let acceptor = Thread.create (accept_loop a) wake in
  let close () =
    stop a waker acceptor;
    Unix.close wake;
    Unix.close waker
  in
  match job a with
  | Ok (job, self, link) -> serve_job a kinds report ~close job self link
  | Error why ->
      close ();
      ended report (`Failed why);
      Error why

let serve a kinds =
  let names = List.map fst kinds in
  if List.length (List.sort_uniq String.compare names) <> List.length names then
    invalid_arg "Rig_remote.serve: kinds names a kind twice";
  if Atomic.exchange a.claimed true then
    invalid_arg "Rig_remote.serve: the agent served already";
  match take report_var with
  | None -> run a kinds None
  | Some fd -> (
      match report fd with
      | Ok r -> run a kinds (Some r)
      | Error why ->
          (* From its claim, the socket is [serve]'s: [run]'s stop closes it,
             and this exit before [run] closes it here. *)
          close_quietly a.socket;
          Error why)
