(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Wire = Rig_remote_proxy.Wire
module Link = Rig_remote_proxy.Link

let strf = Printf.sprintf

(* Errors *)

let err_key fn n =
  strf "Rig_remote.%s: a key of %d bytes is outside %d to %d" fn n Wire.min_key
    Wire.max_key

(* The connections waiting for their handshakes beyond which the listener
   refuses more. *)
let max_pending = 64

(* How long a connection and the job's other agents' connections take at
   most. *)
let join_s = 10.

let check_key fn key =
  let n = String.length key in
  if n < Wire.min_key || n > Wire.max_key then invalid_arg (err_key fn n)

(* Reports to a launcher *)

external unsetenv : string -> unit = "caml_rig_remote_unsetenv"
external report_open : int -> bool = "caml_rig_remote_report_open"
external report_write : int -> string -> unit = "caml_rig_remote_report_write"

let report_var = "RIG_REMOTE_REPORT"

type report = {
  fd : int;
  pid : int;  (** The process that reports. *)
  lock : Mutex.t;
  mutable started : bool;
  mutable ended : bool;
}

let take name =
  let v = Sys.getenv_opt name in
  if Option.is_some v then unsetenv name;
  v

let report fd =
  let n =
    if fd <> "" && String.for_all (fun c -> '0' <= c && c <= '9') fd then
      int_of_string_opt fd
    else None
  in
  match n with
  | Some n when report_open n ->
      Ok
        {
          fd = n;
          pid = Unix.getpid ();
          lock = Mutex.create ();
          started = false;
          ended = false;
        }
  | _ -> Error (strf "%s=%S is no open file descriptor" report_var fd)

(* Writes [line] in one write, a reason's newlines as spaces, unless this
   process is a child of the one that reports. *)
let write r line =
  if Unix.getpid () = r.pid then
    report_write r.fd (String.map (function '\n' -> ' ' | c -> c) line ^ "\n")

let started = function
  | None -> ()
  | Some r ->
      Mutex.protect r.lock @@ fun () ->
      if not (r.started || r.ended) then begin
        r.started <- true;
        write r "started"
      end

let ended report e =
  match report with
  | None -> ()
  | Some r ->
      Mutex.protect r.lock @@ fun () ->
      if not r.ended then begin
        r.ended <- true;
        write r
          (match e with `Closed -> "closed" | `Failed why -> "failed " ^ why)
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

(* A TCP connection to [host] and [port], within [s] seconds. *)
let dial_tcp ~s host port =
  match resolve host port with
  | Error _ as e -> e
  | Ok addr -> (
      let fd = Unix.socket (Unix.domain_of_sockaddr addr) Unix.SOCK_STREAM 0 in
      let fail e =
        Unix.close fd;
        Error (Unix.error_message e)
      in
      Unix.set_nonblock fd;
      match Unix.connect fd addr with
      | () ->
          Unix.clear_nonblock fd;
          Ok fd
      | exception Unix.Unix_error ((Unix.EINPROGRESS | Unix.EWOULDBLOCK), _, _)
        -> (
          match Unix.select [] [ fd ] [] s with
          | [], [], [] ->
              Unix.close fd;
              Error (strf "no answer within %.0f s" s)
          | _ -> (
              match Unix.getsockopt_error fd with
              | None ->
                  Unix.clear_nonblock fd;
                  Ok fd
              | Some e -> fail e))
      | exception Unix.Unix_error (e, _, _) -> fail e)

(* Agents *)

(* The objects the controller names on this machine. *)
type obj = Buffer of Rig.Buffer.t | Rail of Link.t

type t = {
  key : string;
  socket : Unix.file_descr;
  port : int;
  lock : Mutex.t;
  changed : Condition.t;
  mutable served : bool;
  mutable pending : int;  (** Handshakes in flight. *)
  mutable job : Link.job option;
  mutable self : int;  (** This agent's process, once a controller came. *)
  mutable controller : Link.t option;
  accepted : (int, Unix.file_descr) Hashtbl.t;
      (** Connections of the job's agents before this one, admitted, waiting for
          the join that names their machines. *)
  peers : (int, Link.t) Hashtbl.t;  (** The job's other agents' links. *)
  mutable stopped : bool;  (** Once it listens no more. *)
}

let listen ~key host port =
  check_key "listen" key;
  match resolve host port with
  | Error _ as e -> e
  | Ok addr -> (
      let fd = Unix.socket (Unix.domain_of_sockaddr addr) Unix.SOCK_STREAM 0 in
      match
        Unix.setsockopt fd Unix.SO_REUSEADDR true;
        Unix.bind fd addr;
        Unix.listen fd max_pending
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
              lock = Mutex.create ();
              changed = Condition.create ();
              served = false;
              pending = 0;
              job = None;
              self = 0;
              controller = None;
              accepted = Hashtbl.create 4;
              peers = Hashtbl.create 4;
              stopped = false;
            })

let port a = a.port

(* Admits one controller, then the agents of its job before this one. *)
let admit a p =
  Mutex.protect a.lock @@ fun () ->
  match (p, a.job) with
  | Wire.Controller, None -> Ok ()
  | Wire.Controller, Some _ -> Error "the agent serves another job"
  | Wire.Agent j, Some _ when a.self > 0 && j < a.self -> Ok ()
  | Wire.Agent _, _ -> Error "the agent does not expect this agent"

(* Runs one connection's handshake, and makes its link if it is admitted. *)
let handshake a fd =
  let finish () =
    Mutex.protect a.lock (fun () ->
        a.pending <- a.pending - 1;
        Condition.broadcast a.changed)
  in
  Fun.protect ~finally:finish @@ fun () ->
  match Wire.accept fd ~key:a.key ~admit:(admit a) with
  | Error _ -> ( try Unix.close fd with Unix.Unix_error _ -> ())
  | Ok (Wire.Controller, Wire.Agent self) ->
      Mutex.protect a.lock @@ fun () ->
      let j = Link.job () in
      a.job <- Some j;
      a.self <- self;
      a.controller <-
        Some (Link.make j fd ~name:"controller" ~peer:Wire.Controller)
  | Ok (Wire.Agent i, _) ->
      Mutex.protect a.lock @@ fun () ->
      if a.stopped || Hashtbl.mem a.accepted i || Hashtbl.mem a.peers i then
        Unix.close fd
      else Hashtbl.replace a.accepted i fd
  | Ok (Wire.Controller, Wire.Controller) -> Unix.close fd

(* Accepts connections until [wake] is readable, each handshake on a thread of
   its own. *)
let accept_loop a wake =
  let rec loop () =
    match Unix.select [ a.socket; wake ] [] [] (-1.) with
    | exception Unix.Unix_error (Unix.EINTR, _, _) -> loop ()
    | ready, _, _ when List.mem wake ready -> ()
    | _ -> accept_one ()
  and accept_one () =
    match Unix.accept ~cloexec:true a.socket with
    | exception Unix.Unix_error ((Unix.EINTR | Unix.ECONNABORTED), _, _) ->
        loop ()
    | exception Unix.Unix_error _ -> ()
    | fd, _ ->
        let refuse =
          Mutex.protect a.lock @@ fun () ->
          if a.pending >= max_pending then true
          else begin
            a.pending <- a.pending + 1;
            false
          end
        in
        if refuse then Wire.refuse fd "the agent has too many connections"
        else ignore (Thread.create (handshake a) fd);
        loop ()
  in
  loop ()

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
  job : Link.job;
  link : Link.t;  (** The controller's. *)
  kinds : (string * (unit -> (Rig.t list, string) result)) list;
  devices : (int, Rig.t) Hashtbl.t;  (** By id, the host [0]. *)
  opened : (string, Wire.account list) Hashtbl.t;
  objects : (int, obj) Hashtbl.t;
}

(* A refusal of the controller's request, or a failure of the job. *)
exception Refused of string

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
  let reaches =
    Hashtbl.fold
      (fun id' d' acc ->
        if id' <> id && Rig.reaches d d' then id' :: acc else acc)
      s.devices []
  in
  {
    id;
    name = Rig.name d;
    arch = Rig.arch d;
    budget = Rig.budget d;
    reaches = List.sort compare reaches;
  }

(* Connects to the job's agents after this one, then makes links of the
   connections of those before it, all within [join_s]. Each link is named after
   its agent's machine, as the controller names it in [agents]. *)
let join s agents =
  let a = s.agent in
  if a.self > List.length agents then
    raise (Refused (strf "the job has no agent %d" a.self));
  let until = Unix.gettimeofday () +. join_s in
  let late name = Refused (strf "%s: no answer within %.0f s" name join_s) in
  let dial j ({ name; host; port } : Wire.agent) =
    let left = until -. Unix.gettimeofday () in
    if left <= 0. then raise (late name);
    match dial_tcp ~s:left host port with
    | Error why -> raise (Refused (strf "%s: %s" name why))
    | Ok fd -> (
        match
          Wire.dial fd ~key:a.key ~self:(Wire.Agent a.self) ~peer:(Wire.Agent j)
        with
        | Error why ->
            Unix.close fd;
            raise (Refused (strf "%s: %s" name why))
        | Ok () when Unix.gettimeofday () > until ->
            Unix.close fd;
            raise (late name)
        | Ok () ->
            let l = Link.make s.job fd ~name ~peer:(Wire.Agent j) in
            Mutex.protect a.lock (fun () -> Hashtbl.replace a.peers j l))
  in
  List.iteri (fun i m -> if i + 1 > a.self then dial (i + 1) m) agents;
  (* The agents before this one dial it; their handshakes run on the acceptor's
     threads. *)
  let before = List.filteri (fun i _ -> i + 1 < a.self) agents in
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

let memory = function
  | `Device -> Rig.Buffer.Device
  | `Pinned -> Rig.Buffer.Pinned
  | `Mapped -> Rig.Buffer.Mapped

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
      match Rig.Buffer.create ~memory:(memory m) (device s d) bytes with
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
  | Wire.Load _ -> raise (Refused "the agent loads no code")
  | Wire.Entry _ -> None
  | Wire.Rail { id; peer; send; receive } ->
      fresh s id;
      let l = peer_link s peer in
      ignore (Link.rail l ~id ~send ~receive);
      Hashtbl.replace s.objects id (Rail l)

let drop s id =
  match Hashtbl.find_opt s.objects id with
  | Some (Rail l) ->
      Link.release_rail l id;
      Hashtbl.remove s.objects id
  | Some (Buffer _) -> Hashtbl.remove s.objects id
  | None -> ()

(* [b]'s bytes as the host reads them in place: [b]'s own memory on this
   machine's host, a copy of them elsewhere. *)
let host_bytes b =
  let b =
    if Rig.Buffer.device b == Rig.host then b
    else begin
      let h = Rig.Buffer.create Rig.host (Rig.Buffer.length b) in
      Rig.Buffer.copy ~src:b ~dst:h;
      h
    end
  in
  Rig.Buffer.wait b Rig.Buffer.Read;
  Rig.Buffer.bigarray Bigarray.char b

(* Runs a hand-over's parts in order, sending the bytes of each copy into the
   controller's memory as it comes, then the word of its value. *)
let hand_over s (h : Wire.handover) local =
  let next = ref 0 in
  let side bytes = function
    | Wire.Region { id; offset } ->
        Rig.Buffer.view (buffer s id) ~first:offset ~length:bytes
    | Wire.Local ->
        let a = local.(!next) in
        incr next;
        Rig.Buffer.of_bigarray a
  in
  Array.iter
    (function
      | Wire.Words _ -> raise (Refused "the agent runs no code")
      | Wire.Copy { src; dst = Wire.Local; bytes } ->
          Link.bytes s.link ~device:h.device ~value:h.value
            (host_bytes (side bytes src))
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
      Hashtbl.iter (fun id _ -> drop s id) (Hashtbl.copy s.objects);
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
      if Hashtbl.mem s.objects id then drop s id
      else Link.fail s.job (strf "%s: a malformed frame" (Link.name s.link));
      apply s
  | Ok (Wire.Handover (h, local)) -> (
      match hand_over s h local with
      | () -> apply s
      | exception (Refused why | Invalid_argument why) ->
          Link.fail s.job why;
          apply s)

(* Serves one job, reporting its end on [report]. *)
let run a kinds report =
  let wake, waker =
    Unix.socketpair ~cloexec:true Unix.PF_UNIX Unix.SOCK_STREAM 0
  in
  let acceptor = Thread.create (accept_loop a) wake in
  let link =
    Mutex.protect a.lock @@ fun () ->
    while a.controller = None do
      Condition.wait a.changed a.lock
    done;
    Option.get a.controller
  in
  let job = Option.get a.job in
  watch report job;
  let devices = Hashtbl.create 8 in
  Hashtbl.replace devices 0 Rig.host;
  let s =
    {
      agent = a;
      job;
      link;
      kinds;
      devices;
      opened = Hashtbl.create 4;
      objects = Hashtbl.create 64;
    }
  in
  let r =
    match apply s with
    | r -> r
    | exception Rig.Lost (d, why) ->
        let why = strf "%s lost: %s" (Rig.name d) why in
        Link.fail job why;
        Error why
  in
  (* The agent listens no more before its close tells the controller the job
     ended. *)
  stop a waker acceptor;
  Unix.close wake;
  Unix.close waker;
  let r =
    match r with
    | Error _ as e -> e
    | Ok () -> (
        Link.close job;
        match Link.failure job with Some why -> Error why | None -> Ok ())
  in
  (match r with
  | Ok () -> ended report `Closed
  | Error why ->
      ended report (`Failed why);
      Rig.fail why);
  r

let serve a kinds =
  let names = List.map fst kinds in
  if List.length (List.sort_uniq String.compare names) <> List.length names then
    invalid_arg "Rig_remote.serve: kinds names a kind twice";
  if Mutex.protect a.lock (fun () -> a.served) then
    invalid_arg "Rig_remote.serve: the agent served already";
  Mutex.protect a.lock (fun () -> a.served <- true);
  match take report_var with
  | None -> run a kinds None
  | Some fd -> (
      match report fd with
      | Ok r -> run a kinds (Some r)
      | Error why -> Error why)
