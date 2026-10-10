(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Wire = Rig_remote_proxy.Wire
module Link = Rig_remote_proxy.Link
module Proxy = Rig_remote_proxy

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

(* Machines *)

(* A machine of a job: its agent's link and the devices opened there. *)
type machine = {
  name : string;
  agent : int;  (** Its agent's process. *)
  link : Link.t;
  mutable host : Rig.t option;
  kinds : (string, Rig.t list) Hashtbl.t;  (** Guarded by [lock]. *)
  lock : Mutex.t;
}

type t = {
  job : Link.job;
  machines : machine list;
  report : Agent.report option;  (** Its launcher's, if {!launched} made it. *)
}

(* Each machine's name names it for the life of the process: the [n]th
   connection named [base], such as ["HOST:PORT"], is ["base#n"] from the second
   on. *)
let names : (string, int) Hashtbl.t = Hashtbl.create 4
let names_lock = Mutex.create ()

let machine_name base =
  Mutex.protect names_lock @@ fun () ->
  let n = 1 + Option.value ~default:0 (Hashtbl.find_opt names base) in
  Hashtbl.replace names base n;
  if n = 1 then base else strf "%s#%d" base n

(* The machines of the open job, by name, which a host's record names. *)
let machines : (string, machine) Hashtbl.t = Hashtbl.create 4
let current : t option ref = ref None
let lock = Mutex.create ()

let machine_of_name name =
  Mutex.protect lock (fun () -> Hashtbl.find_opt machines name)

(* Requests *)

(* The agent's answer, its refusal named after the machine; a failed job's root
   cause as it is. *)
let request m q =
  match Link.request m.link q with
  | Ok v -> Ok v
  | Error (`Refused why) -> Error (strf "%s: %s" m.name why)
  | Error (`Failed why) -> Error why

(* Rails *)

(* [f] made to run at its first call alone: a rail's release runs once. *)
let once f =
  let ran = Atomic.make false in
  fun () -> if not (Atomic.exchange ran true) then f ()

(* The rail function of [m]'s host. *)
let rail m (peer : Rig_remote_abi.host option) ~send ~receive :
    (Rig_remote_abi.rail, string) result =
  (match peer with
  | Some h when h.machine = m.name ->
      invalid_arg "Rig_remote_abi.host.rail: the peer is this host"
  | _ -> ());
  (match Rig_remote_abi.check_transfers ~send ~receive with
  | Ok () -> ()
  | Error why -> invalid_argf "Rig_remote_abi.host.rail: %s" why);
  let id = Link.fresh () in
  match peer with
  | None -> (
      let local = Link.rail m.link ~id ~send:receive ~receive:send in
      let release =
        once (fun () ->
            Link.release_rail m.link id;
            Link.drop m.link id)
      in
      match
        request m (Wire.Rail { id; peer = Wire.Controller; send; receive })
      with
      | Ok () -> Ok { id; local = Some local; release }
      | Error why ->
          Link.release_rail m.link id;
          Error why)
  | Some h -> (
      match machine_of_name h.machine with
      | None -> Error (strf "%s is of another job" h.machine)
      | Some m' -> (
          let release =
            once (fun () ->
                Link.drop m.link id;
                Link.drop m'.link id)
          in
          let rail' =
            Wire.Rail { id; peer = Wire.Agent m'.agent; send; receive }
          in
          let rail'' =
            Wire.Rail
              { id; peer = Wire.Agent m.agent; send = receive; receive = send }
          in
          match request m rail' with
          | Error why -> Error why
          | Ok () -> (
              match request m' rail'' with
              | Ok () -> Ok { id; local = None; release }
              | Error why ->
                  Link.drop m.link id;
                  Error why)))

(* Opening *)

let open_host m (a : Wire.account) =
  let record = Rig_remote_abi.Host { machine = m.name; rail = rail m } in
  Rig.open_host
    (module Proxy)
    ~machine:m.name ~name:a.name
    (fun () -> Ok (Proxy.make m.link a record))

let machine_of_host h =
  match Rig.capability h Rig_remote_abi.key with
  | Some (Rig_remote_abi.Host { machine; _ }) -> (
      match machine_of_name machine with
      | Some m when Option.fold ~none:false ~some:(Rig.equal h) m.host -> Some m
      | _ -> None)
  | _ -> None

(* The job's state comes first, under the machine's lock: a kind opened before
   the job ended answers as a new one does. *)
let devices h kind =
  match machine_of_host h with
  | None -> invalid_arg "Rig_remote.devices: the device is no host of a job"
  | Some m -> (
      Mutex.protect m.lock @@ fun () ->
      match
        (Link.wait (Link.job_of m.link) ~ms:0, Hashtbl.find_opt m.kinds kind)
      with
      | Link.Closed, _ -> Error (strf "%s: the job is closed" m.name)
      | Link.Failed why, _ -> Error why
      | Link.Open, Some ds -> Ok ds
      | Link.Open, None -> (
          match request m (Wire.Open kind) with
          | Error why -> Error why
          | Ok accounts ->
              let open_one (a : Wire.account) =
                let record = Rig_remote_abi.Device { id = a.id } in
                Rig.open_
                  (module Proxy)
                  ~machine:h ~name:a.name
                  (fun () -> Ok (Proxy.make m.link a record))
              in
              let rec all acc = function
                | [] -> Ok (List.rev acc)
                | a :: rest -> (
                    match open_one a with
                    | Ok d -> all (d :: acc) rest
                    | Error why -> Error why)
              in
              let r = all [] accounts in
              Result.iter (Hashtbl.replace m.kinds kind) r;
              r))

(* Jobs *)

let hosts j = List.filter_map (fun m -> m.host) j.machines
let failure j = Link.failure j.job

let close j =
  (match Link.wait j.job ~ms:0 with
  | Link.Closed | Link.Failed _ -> ()
  | Link.Open ->
      List.iter Rig.close (hosts j);
      Link.close j.job);
  Agent.ended j.report
    (match Link.failure j.job with Some why -> `Failed why | None -> `Closed)

let () =
  at_exit (fun () ->
      Option.iter close (Mutex.protect lock (fun () -> !current)))

(* Fails [j] with [why] and is [why] after the machine's name. *)
let abandon job name why =
  let why = strf "%s: %s" name why in
  Link.fail job why;
  Error why

(* Connects to each agent and runs its handshake, in order, naming each machine
   after its [base]. *)
let rec dial ~key job acc i = function
  | [] -> Ok (List.rev acc)
  | (base, host, port) :: rest -> (
      let name = machine_name base in
      match Agent.dial_tcp ~s:Agent.join_s host port with
      | Error why -> abandon job name why
      | Ok fd -> (
          match
            Wire.dial fd ~key ~self:Wire.Controller ~peer:(Wire.Agent i)
          with
          | Error why ->
              Unix.close fd;
              abandon job name why
          | Ok () ->
              let link = Link.make job fd ~name ~peer:(Wire.Agent i) in
              let m =
                {
                  name;
                  agent = i;
                  link;
                  host = None;
                  kinds = Hashtbl.create 4;
                  lock = Mutex.create ();
                }
              in
              dial ~key job (m :: acc) (i + 1) rest))

(* How long the controller waits for an agent's answer to its join: the agent's
   own bound on joining, and a second for the answer to come. *)
let join_answer_s = Agent.join_s +. 1.

(* Joins every agent at once, each named as [ms] names it: each waits for the
   ones before it to connect. The first refusal fails the job, which ends the
   other requests, and so does an agent that does not answer within
   [join_answer_s]. *)
let join job agents ms =
  let agents =
    List.map2
      (fun m (_, host, port) -> { Wire.name = m.name; host; port })
      ms agents
  in
  let answers = Array.make (List.length ms) None in
  let lock = Mutex.create () in
  let ask i m =
    match request m (Wire.Join { agents }) with
    | Ok a -> Mutex.protect lock (fun () -> answers.(i) <- Some a)
    | Error why -> Link.fail job why
  in
  let threads = List.mapi (fun i m -> Thread.create (ask i) m) ms in
  let until = Unix.gettimeofday () +. join_answer_s in
  let rec wait () =
    let unanswered =
      Mutex.protect lock (fun () ->
          List.filteri (fun i _ -> Option.is_none answers.(i)) ms)
    in
    match unanswered with
    | [] -> ()
    | m :: _ when Unix.gettimeofday () >= until ->
        Link.fail job
          (strf "%s: no answer to its join within %.0f s" m.name join_answer_s)
    | _ -> if Link.wait job ~ms:10 = Link.Open then wait ()
  in
  wait ();
  List.iter Thread.join threads;
  match Link.failure job with
  | Some why -> Error why
  | None -> Ok (List.filter_map Fun.id (Array.to_list answers))

(* Opens each machine's host. On a failure it closes those it opened and fails
   the job. *)
let open_hosts job ms accounts =
  let rec go = function
    | [] -> Ok ()
    | (m, a) :: rest -> (
        match open_host m a with
        | Ok h ->
            m.host <- Some h;
            go rest
        | Error why -> Error (strf "%s: %s" m.name why))
  in
  match go (List.combine ms accounts) with
  | Ok () -> Ok ()
  | Error why ->
      List.iter (fun m -> Option.iter Rig.close m.host) ms;
      Link.fail job why;
      Error why

let check_agents agents =
  if agents = [] then invalid_arg "Rig_remote.connect: no agent";
  if List.length (List.sort_uniq compare agents) <> List.length agents then
    invalid_arg "Rig_remote.connect: an address is listed twice"

(* Starts a job with the agents at [agents]' hosts and ports, each machine named
   after its base, reporting on [report]. [fn] names the caller. *)
let start ~fn ~key ~report agents =
  Mutex.protect lock (fun () ->
      match !current with
      | Some j when Link.wait j.job ~ms:0 = Link.Open ->
          invalid_argf "Rig_remote.%s: a job of the process is open" fn
      | _ -> ());
  match Rig.failure () with
  | Some why -> Error why
  | None -> (
      let job = Link.job () in
      match dial ~key job [] 1 agents with
      | Error _ as e -> e
      | Ok ms -> (
          match Result.bind (join job agents ms) (open_hosts job ms) with
          | Error _ as e -> e
          | Ok () ->
              let j = { job; machines = ms; report } in
              Mutex.protect lock (fun () ->
                  Hashtbl.reset machines;
                  List.iter (fun m -> Hashtbl.replace machines m.name m) ms;
                  current := Some j);
              Agent.watch report job;
              Ok j))

let connect ~key agents =
  check_agents agents;
  let named = List.map (fun (h, p) -> (strf "%s:%d" h p, h, p)) agents in
  start ~fn:"connect" ~key ~report:None named

(* Launched jobs *)

let agents_var = "RIG_REMOTE_AGENTS"
let key_var = "RIG_REMOTE_KEY"
let digits s = s <> "" && String.for_all (fun c -> '0' <= c && c <= '9') s

let is_hex = function
  | '0' .. '9' | 'a' .. 'f' | 'A' .. 'F' -> true
  | _ -> false

(* One machine of [agents_var]: ["NAME=ADDRESS:PORT"], an IPv6 address in
   brackets, as ["[fd00::2]=[fd00::2]:41234"]. *)
let launched_agent s =
  let unbracket a =
    let n = String.length a in
    if n >= 2 && a.[0] = '[' && a.[n - 1] = ']' then String.sub a 1 (n - 2)
    else a
  in
  let ( let* ) = Option.bind in
  let* i = String.index_opt s '=' in
  let name = String.sub s 0 i in
  let address = String.sub s (i + 1) (String.length s - i - 1) in
  let* k = String.rindex_opt address ':' in
  let host = unbracket (String.sub address 0 k) in
  let port = String.sub address (k + 1) (String.length address - k - 1) in
  let* p = if digits port then int_of_string_opt port else None in
  if name = "" || host = "" || p < 1 || p > 65535 then None
  else Some (name, host, p)

let launched_agents = function
  | None -> Error (strf "%s is not set" agents_var)
  | Some v -> (
      let rec all acc = function
        | [] -> Ok (List.rev acc)
        | s :: rest -> (
            match launched_agent s with
            | Some a -> all (a :: acc) rest
            | None ->
                Error (strf "%s: %S is not NAME=ADDRESS:PORT" agents_var s))
      in
      match all [] (String.split_on_char ',' v) with
      | Error _ as e -> e
      | Ok agents ->
          let unique l =
            List.length (List.sort_uniq compare l) = List.length l
          in
          if not (unique (List.map (fun (n, _, _) -> n) agents)) then
            Error (strf "%s names a machine twice" agents_var)
          else if not (unique (List.map (fun (_, h, p) -> (h, p)) agents)) then
            Error (strf "%s lists an address twice" agents_var)
          else Ok agents)

let launched_key = function
  | None -> Error (strf "%s is not set" key_var)
  | Some k when String.length k = 64 && String.for_all is_hex k -> Ok k
  | Some _ -> Error (strf "%s is not 64 hexadecimal characters" key_var)

let called = Atomic.make false

let launched () =
  if Atomic.exchange called true then
    invalid_arg "Rig_remote.launched: the process called it before";
  match Agent.take Agent.report_var with
  | None -> None
  | Some fd -> (
      let agents = Agent.take agents_var in
      let key = Agent.take key_var in
      match Agent.report fd with
      | Error _ as e -> Some e
      | Ok r ->
          let report = Some r in
          let job =
            match (launched_key key, launched_agents agents) with
            | (Error _ as e), _ | _, (Error _ as e) -> e
            | Ok key, Ok agents -> start ~fn:"launched" ~key ~report agents
          in
          (match job with
          | Ok _ -> Agent.started report
          | Error why -> Agent.ended report (`Failed why));
          Some job)

(* Agents *)

type agent = Agent.t

let listen = Agent.listen
let port = Agent.port
let serve = Agent.serve

(* Keys *)

type key = string

let key s =
  let n = String.length s in
  if n >= Wire.min_key && n <= Wire.max_key then Ok s
  else
    Error
      (strf "the key has %d bytes, outside %d to %d" n Wire.min_key Wire.max_key)

let read_key file =
  let err why = Error (strf "%s: %s" file why) in
  (* Without [O_NONBLOCK] an open of a FIFO would wait for a writer before the
     kind is known; a regular file's reads ignore it. *)
  let flags = Unix.[ O_RDONLY; O_NONBLOCK; O_NOCTTY; O_CLOEXEC ] in
  match Unix.openfile file flags 0 with
  | exception Unix.Unix_error (e, _, _) -> err (Unix.error_message e)
  | fd -> (
      Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
      let st = Unix.fstat fd in
      if st.st_kind <> Unix.S_REG then err "is no regular file"
      else if (not Sys.win32) && st.st_uid <> Unix.getuid () then
        err "belongs to another user"
      else if (not Sys.win32) && st.st_perm land 0o077 <> 0 then
        err "may be read or written by other users; chmod 600 it"
      else if st.st_size < Wire.min_key || st.st_size > Wire.max_key then
        err
          (strf "holds %d bytes, outside %d to %d" st.st_size Wire.min_key
             Wire.max_key)
      else
        let b = Bytes.create st.st_size in
        let rec go off =
          if off < st.st_size then
            match Unix.read fd b off (st.st_size - off) with
            | 0 -> err "ended early"
            | k -> go (off + k)
          else Ok (Bytes.unsafe_to_string b)
        in
        match go 0 with
        | r -> r
        | exception Unix.Unix_error (e, _, _) -> err (Unix.error_message e))
