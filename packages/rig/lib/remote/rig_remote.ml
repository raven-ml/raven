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

(* The state below changes by atomic swaps of immutable values and takes no
   lock, so a signal handler, finaliser or Gc.Memprof callback inside a call
   of this module waits for no lock of this module's. *)
module Names = Map.Make (String)
module Ids = Map.Make (Int)

(* A machine of a job: its agent's link and the devices opened there. *)
type machine = {
  name : string;
  agent : int;  (** Its agent's process. *)
  link : Link.t;
  mutable host : Rig.t option;
  kinds : Rig.t list Names.t Atomic.t;  (** The devices of each kind opened. *)
  proxies : Rig.t Ids.t Atomic.t;  (** The proxy of each account opened. *)
}

type t = {
  job : Link.job;
  machines : machine list;
  report : Agent.report option;  (** Its launcher's, if {!launched} made it. *)
}

(* Each machine's name names it for the life of the process: the [n]th
   connection named [base], such as ["HOST:PORT"], is ["base#n"] from the second
   on. *)
let names : int Names.t Atomic.t = Atomic.make Names.empty

let rec machine_name base =
  let seen = Atomic.get names in
  let n = 1 + Option.value ~default:0 (Names.find_opt base seen) in
  if not (Atomic.compare_and_set names seen (Names.add base n seen)) then
    machine_name base
  else if n = 1 then base
  else strf "%s#%d" base n

(* The process's last job to start, whose machines a host's record names. *)
let current : t option Atomic.t = Atomic.make None

let machine_of_name name =
  match Atomic.get current with
  | None -> None
  | Some j -> List.find_opt (fun m -> String.equal m.name name) j.machines

(* Requests *)

(* The agent's answer, its refusal named after the machine; a failed job's root
   cause as it is. *)
let request m q =
  match Link.request m.link q with
  | Ok v -> Ok v
  | Error (`Refused why) -> Error (strf "%s: %s" m.name why)
  | Error (`Failed why) -> Error why

(* Rails *)

(* The rail function of [m]'s host. A function of a capability, it blocks
   signals as rig's do, so a Ctrl-C lands once both machines hold the rail,
   but on Windows, which blocks none. *)
let rail m (peer : Rig_remote_abi.host option) ~send ~receive :
    (Rig_remote_abi.rail, string) result =
  Agent.masked @@ fun () ->
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
      let collect _ =
        Link.release_rail m.link id;
        Link.drop m.link id
      in
      let local = Link.rail m.link ~id ~send:receive ~receive:send in
      let rail = Rig_remote_abi.rail ~id ~local:(Some local) in
      match
        request m (Wire.Rail { id; peer = Wire.Controller; send; receive })
      with
      | Ok () ->
          (* The rail ends once its end is unreachable: the rail holds the end,
             and work that uses it holds either. *)
          Gc.finalise (fun l -> Agent.shielded (fun () -> collect l)) local;
          Ok rail
      | Error why ->
          Link.release_rail m.link id;
          Error why
      | exception exn ->
          Link.release_rail m.link id;
          raise exn)
  | Some h -> (
      match machine_of_name h.machine with
      | None -> Error (strf "%s is of another job" h.machine)
      | Some m' -> (
          let rail' =
            Wire.Rail { id; peer = Wire.Agent m'.agent; send; receive }
          in
          let rail'' =
            Wire.Rail
              { id; peer = Wire.Agent m.agent; send = receive; receive = send }
          in
          let collect _ =
            Link.drop m.link id;
            Link.drop m'.link id
          in
          let rail = Rig_remote_abi.rail ~id ~local:None in
          match request m rail' with
          | Error why -> Error why
          | Ok () -> (
              match request m' rail'' with
              | Ok () ->
                  Gc.finalise
                    (fun r -> Agent.shielded (fun () -> collect r))
                    rail;
                  Ok rail
              | Error why ->
                  Link.drop m.link id;
                  Error why
              | exception exn ->
                  (* [m'] may hold its half, which a drop of none would fail
                     the job for: only [m]'s goes. *)
                  Link.drop m.link id;
                  raise exn)))

(* Opening *)

let open_host m (a : Wire.account) =
  let record =
    Rig_remote_abi.Host
      {
        machine = m.name;
        rail = rail m;
        threads = Proxy.threads;
      }
  in
  Rig.open_host ~machine:m.name (Proxy.device m.link a record)

let machine_of_host h =
  match Rig.capability h Rig_remote_abi.key with
  | Some (Rig_remote_abi.Host { machine; _ }) -> (
      match machine_of_name machine with
      | Some m when Option.fold ~none:false ~some:(Rig.equal h) m.host -> Some m
      | _ -> None)
  | _ -> None

(* Keeps [ds] as [m]'s devices of [kind], unless a call kept some first:
   those. *)
let rec keep m kind ds =
  let kept = Atomic.get m.kinds in
  match Names.find_opt kind kept with
  | Some ds -> ds
  | None ->
      if Atomic.compare_and_set m.kinds kept (Names.add kind ds kept) then ds
      else keep m kind ds

(* Keeps [d] as [m]'s proxy of account [id]: no other is made once it is. *)
let rec keep_proxy m id d =
  let kept = Atomic.get m.proxies in
  if not (Atomic.compare_and_set m.proxies kept (Ids.add id d kept)) then
    keep_proxy m id d

(* [m]'s proxy of [a], opened unless a call opened one first, which may be
   closed since. Once a proxy of an account handed work over, the link takes
   no other of that account. The lookup runs in the opener's [make], which
   rig runs while no other open of the proxy's name does, so it is one step
   with the open: a call that a callback interrupts before the open finds
   the proxy the callback opened and closed. A proxy that rig gave and that
   is not kept is open, and rig gives it again. *)
let open_proxy m h (a : Wire.account) =
  let o = Proxy.device m.link a (Rig_remote_abi.Device { id = a.id }) in
  let found = ref None in
  let make ~word =
    match Ids.find_opt a.id (Atomic.get m.proxies) with
    | Some d ->
        found := Some d;
        Error "opened"
    | None -> o.make ~word
  in
  match (Rig.open_ ~machine:h { o with make }, !found) with
  | Ok d, _ ->
      keep_proxy m a.id d;
      Ok d
  | Error _, Some d -> Ok d
  | (Error _ as e), None -> e

(* The job's state comes first: a kind opened before the job ended answers as
   a new one does. Two calls that find no devices of [kind], from two domains
   or from a callback inside the other, both ask the agent, which answers an
   open of a kind with the accounts it gave first, so both give the same
   proxies ({!open_proxy}). *)
let devices h kind =
  Agent.masked @@ fun () ->
  match machine_of_host h with
  | None -> invalid_arg "Rig_remote.devices: the device is no host of a job"
  | Some m -> (
      match
        ( Link.wait (Link.job_of m.link) ~ms:0,
          Names.find_opt kind (Atomic.get m.kinds) )
      with
      | Link.Closed, _ -> Error (strf "%s: the job is closed" m.name)
      | Link.Failed why, _ -> Error why
      | Link.Open, Some ds -> Ok ds
      | Link.Open, None -> (
          match request m (Wire.Open kind) with
          | Error why -> Error why
          | Ok accounts ->
              let rec all acc = function
                | [] -> Ok (List.rev acc)
                | a :: rest -> (
                    match open_proxy m h a with
                    | Ok d -> all (d :: acc) rest
                    | Error why -> Error why)
              in
              Result.map (keep m kind) (all [] accounts)))

(* Jobs *)

let hosts j = List.filter_map (fun m -> m.host) j.machines
let failed j = Link.failed j.job

(* A failed job's hosts are closed too: a lost device's stop runs only at its
   close, and until then its memory and slot stay held. Their stops wait for
   nothing once the job failed. *)
let close j =
  Agent.masked @@ fun () ->
  (match Link.wait j.job ~ms:0 with
  | Link.Closed -> ()
  | Link.Failed _ -> List.iter Rig.close (hosts j)
  | Link.Open ->
      List.iter Rig.close (hosts j);
      Link.close j.job);
  Agent.ended j.report
    (match Link.failed j.job with Some why -> `Failed why | None -> `Closed)

(* The process closes its last job at exit. A close that rig refuses, from a
   callback inside a call of rig's on the exiting thread, fails the job with
   the refusal instead: its exception would end the process with a fatal
   error, and leave the job's report without its end. The process exits once
   the links of a failed job have ended, so each agent reads its abort. *)
let () =
  at_exit @@ fun () ->
  Agent.shielded @@ fun () ->
  match Atomic.get current with
  | None -> ()
  | Some j ->
      (match close j with
      | () -> ()
      | exception Invalid_argument why ->
          Link.fail j.job why;
          let why = Option.value ~default:why (Link.failed j.job) in
          Agent.ended j.report (`Failed why));
      Link.close j.job

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
                  kinds = Atomic.make Names.empty;
                  proxies = Atomic.make Ids.empty;
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
  let answers = Array.init (List.length ms) (fun _ -> Atomic.make None) in
  let ask i m =
    match request m (Wire.Join { agents }) with
    | Ok a -> Atomic.set answers.(i) (Some a)
    | Error why -> Link.fail job why
  in
  let threads = List.mapi (fun i m -> Agent.spawn (ask i) m) ms in
  let until = Unix.gettimeofday () +. join_answer_s in
  let rec wait () =
    let unanswered =
      List.filteri (fun i _ -> Option.is_none (Atomic.get answers.(i))) ms
    in
    match unanswered with
    | [] -> ()
    | m :: _ when Unix.gettimeofday () >= until ->
        Link.fail job
          (strf "%s: no answer to its join within %.0f s" m.name join_answer_s)
    | _ -> if Link.wait job ~ms:10 = Link.Open then wait ()
  in
  wait ();
  List.iter (Agent.unblocked Thread.join) threads;
  match Link.failed job with
  | Some why -> Error why
  | None -> Ok (List.filter_map Atomic.get (Array.to_list answers))

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

(* The cause of a loss on the controller that fails [j], as {!Rig.Lost}
   prints it: the first loss of one of [j]'s hosts or proxies other than by a
   close, whose reason is ["closed"] (rig.mli's Errors and failure). A loss
   of another device of the controller reaches nothing of [j], and [Rig.fail]
   loses [j]'s devices too. *)
let loss j =
  let lost d =
    match Rig.lost d with
    | None | Some "closed" -> None
    | Some why -> Some (strf "%s lost: %s" (Rig.name d) why)
  in
  let opened m =
    Option.to_list m.host @ List.map snd (Ids.bindings (Atomic.get m.proxies))
  in
  List.find_map lost (List.concat_map opened j.machines)

(* Fails [j] within a second of a loss on the controller that fails it, and
   reports [j]'s failure on its report. The controller's process goes on: the
   job's hosts and proxies fault with the root cause at their next use, and
   its own devices are no job's. *)
let watch j =
  let rec loop () =
    match Link.wait j.job ~ms:1000 with
    | Link.Failed why -> Agent.ended j.report (`Failed why)
    | Link.Closed -> ()
    | Link.Open ->
        Option.iter (Link.fail j.job) (loss j);
        loop ()
  in
  ignore (Agent.spawn loop ())

external reraise : exn -> 'a = "%reraise"

(* Closes [m]'s host, if it has one, unless rig refuses, from a callback
   inside a lock. *)
let close_host m =
  match m.host with
  | Some h -> ( try Rig.close h with Invalid_argument _ -> ())
  | None -> ()

(* The reason a start fails its job with when it raises. *)
let start_raised = "the controller raised while it started the job"

let check_agents agents =
  if agents = [] then invalid_arg "Rig_remote.connect: no agent";
  if List.length (List.sort_uniq compare agents) <> List.length agents then
    invalid_arg "Rig_remote.connect: an address is listed twice"

(* Starts a job with the agents at [agents]' hosts and ports, each machine named
   after its base, reporting on [report]. [fn] names the caller. A process that
   has failed gets its failure from the first host's open. A loss that fails
   the open job fails it here, before its watcher sees it. *)
let start ~fn ~key ~report agents =
  (match Atomic.get current with
  | Some j when Link.wait j.job ~ms:0 = Link.Open -> (
      Option.iter (Link.fail j.job) (loss j);
      match Link.wait j.job ~ms:0 with
      | Link.Open ->
          invalid_argf "Rig_remote.%s: a job of the process is open" fn
      | Link.Failed _ | Link.Closed -> ())
  | _ -> ());
  let job = Link.job () in
  let dialled = ref [] in
  let started () =
    match dial ~key job [] 1 agents with
    | Error _ as e -> e
    | Ok ms ->
        dialled := ms;
        Result.map
          (fun () -> ms)
          (Result.bind (join job agents ms) (open_hosts job ms))
  in
  (* From the job's making on, an exception fails and closes the job, which
     would otherwise refuse every later start: a refusal of rig's, such as
     [Rig.open_host]'s inside a lock, a thread the system does not give, or
     an asynchronous exception. It then closes the hosts the start opened,
     whose memory and slots a failed job's hosts keep until their close; the
     stops wait for nothing once the job failed. The handler allocates
     nothing before the job's close. [current] may keep the job, which then
     reads as failed. *)
  match
    match started () with
    | Error _ as e -> e
    | Ok ms ->
        let j = { job; machines = ms; report } in
        Atomic.set current (Some j);
        watch j;
        Ok j
  with
  | Error _ as e ->
      (* The agents reached read the abort before the caller may exit. *)
      Link.close job;
      e
  | Ok _ as r -> r
  | exception e ->
      Link.fail job start_raised;
      Link.close job;
      List.iter close_host !dialled;
      reraise e

let connect ~key agents =
  Agent.masked @@ fun () ->
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
   brackets, as ["[fd00::2]=[fd00::2]:41234"]. An IPv6 address without them is
   refused: in ["fd00::2:4123"], ["4123"] may be its last group or a port. *)
let launched_agent s =
  let malformed = Error (strf "%s: %S is not NAME=ADDRESS:PORT" agents_var s) in
  let ( let* ) o f = match o with Some x -> f x | None -> malformed in
  let* i = String.index_opt s '=' in
  let name = String.sub s 0 i in
  let address = String.sub s (i + 1) (String.length s - i - 1) in
  let* k = String.rindex_opt address ':' in
  let host = String.sub address 0 k in
  let port = String.sub address (k + 1) (String.length address - k - 1) in
  let n = String.length host in
  let bracketed = n >= 2 && host.[0] = '[' && host.[n - 1] = ']' in
  let host = if bracketed then String.sub host 1 (n - 2) else host in
  if (not bracketed) && String.contains host ':' then
    Error
      (strf "%s: %S: write an IPv6 address in brackets, as NAME=[ADDRESS]:PORT"
         agents_var s)
  else
    let* p = if digits port then int_of_string_opt port else None in
    if name = "" || host = "" || p < 1 || p > 65535 then malformed
    else Ok (name, host, p)

let launched_agents = function
  | None -> Error (strf "%s is not set" agents_var)
  | Some v -> (
      let rec all acc = function
        | [] -> Ok (List.rev acc)
        | s :: rest -> (
            match launched_agent s with
            | Ok a -> all (a :: acc) rest
            | Error _ as e -> e)
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
  Agent.masked @@ fun () ->
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
