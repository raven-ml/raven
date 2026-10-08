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

type t = { job : Link.job; machines : machine list }

(* Each machine's name names it for the life of the process: the [n]th
   connection to an address is ["HOST:PORT#n"] from the second on. *)
let names : (string, int) Hashtbl.t = Hashtbl.create 4
let names_lock = Mutex.create ()

let machine_name host port =
  let address = strf "%s:%d" host port in
  Mutex.protect names_lock @@ fun () ->
  let n = 1 + Option.value ~default:0 (Hashtbl.find_opt names address) in
  Hashtbl.replace names address n;
  if n = 1 then address else strf "%s#%d" address n

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

let check_transfers send receive =
  if Array.length send = 0 && Array.length receive = 0 then
    invalid_arg "Rig_remote_abi.host.rail: the rail carries no transfer";
  let check (t : Rig_remote_abi.transfer) =
    if t.length <= 0 || t.src < 0 || t.dst < 0 then
      invalid_argf
        "Rig_remote_abi.host.rail: a transfer of %d bytes from %d to %d is \
         invalid"
        t.length t.src t.dst
  in
  Array.iter check send;
  Array.iter check receive

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
  check_transfers send receive;
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

let devices h kind =
  match machine_of_host h with
  | None -> invalid_arg "Rig_remote.devices: the device is no host of a job"
  | Some m when Link.wait (Link.job_of m.link) ~ms:0 = Link.Closed ->
      Error (strf "%s: the job is closed" m.name)
  | Some m -> (
      Mutex.protect m.lock @@ fun () ->
      match Hashtbl.find_opt m.kinds kind with
      | Some ds -> Ok ds
      | None -> (
          match request m (Wire.Open kind) with
          | Error why -> Error why
          | Ok accounts ->
              let open_one (a : Wire.account) =
                let record = Rig_remote_abi.Device { id = a.id } in
                Rig.open_
                  (module Proxy)
                  ~machine:m.name ~name:a.name
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
  match Link.wait j.job ~ms:0 with
  | Link.Closed | Link.Failed _ -> ()
  | Link.Open ->
      List.iter Rig.close (hosts j);
      Link.close j.job

let () =
  at_exit (fun () ->
      Option.iter close (Mutex.protect lock (fun () -> !current)))

(* Fails [j] with [why] and is [why] after the machine's name. *)
let abandon job name why =
  let why = strf "%s: %s" name why in
  Link.fail job why;
  Error why

(* Connects to each agent and runs its handshake, in order. *)
let rec dial ~key job acc i = function
  | [] -> Ok (List.rev acc)
  | (host, port) :: rest -> (
      let name = machine_name host port in
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

(* Joins every agent at once: each waits for the ones before it to connect. *)
let join agents ms =
  let answers = Array.make (List.length ms) (Error "") in
  let threads =
    List.mapi
      (fun i m ->
        Thread.create
          (fun () -> answers.(i) <- request m (Wire.Join { agents }))
          ())
      ms
  in
  List.iter Thread.join threads;
  Array.to_list answers

let check_agents agents =
  if agents = [] then invalid_arg "Rig_remote.connect: no agent";
  if List.length (List.sort_uniq compare agents) <> List.length agents then
    invalid_arg "Rig_remote.connect: an address is listed twice"

let connect ~key agents =
  Agent.check_key "connect" key;
  check_agents agents;
  Mutex.protect lock (fun () ->
      match !current with
      | Some j when Link.wait j.job ~ms:0 = Link.Open ->
          invalid_arg "Rig_remote.connect: a job of the process is open"
      | _ -> ());
  match Rig.failure () with
  | Some why -> Error why
  | None -> (
      let job = Link.job () in
      match dial ~key job [] 1 agents with
      | Error _ as e -> e
      | Ok ms -> (
          let rec hosts_of = function
            | [] -> Ok ()
            | (m, Ok a) :: rest -> (
                match open_host m a with
                | Ok h ->
                    m.host <- Some h;
                    hosts_of rest
                | Error why -> abandon job m.name why)
            | (_, Error why) :: _ ->
                Link.fail job why;
                Error why
          in
          match hosts_of (List.combine ms (join agents ms)) with
          | Error _ as e -> e
          | Ok () ->
              let j = { job; machines = ms } in
              Mutex.protect lock (fun () ->
                  Hashtbl.reset machines;
                  List.iter (fun m -> Hashtbl.replace machines m.name m) ms;
                  current := Some j);
              Agent.watch job;
              Ok j))

(* Agents *)

type agent = Agent.t

let listen = Agent.listen
let port = Agent.port
let serve = Agent.serve

(* Keys *)

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
