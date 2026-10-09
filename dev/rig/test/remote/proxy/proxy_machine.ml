(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Another machine for the proxy suites: a real link on loopback whose far end
   is an agent, a loop on a thread of this process that keeps the machine's
   memory as bytes, runs each hand-over's parts in order and reports its word,
   as an agent does. *)

open Windtrap
module Wire = Rig_remote_proxy.Wire
module Link = Rig_remote_proxy.Link
module Proxy = Rig_remote_proxy
module B = Rig.Buffer
module Sub = Rig.Submission

(* The most anything the test awaits may take. *)
let patience = 5.

(* Sockets and threads *)

let connected () =
  let l = Unix.socket ~cloexec:true Unix.PF_INET Unix.SOCK_STREAM 0 in
  Fun.protect
    ~finally:(fun () -> Unix.close l)
    (fun () ->
      Unix.bind l (Unix.ADDR_INET (Unix.inet_addr_loopback, 0));
      Unix.listen l 1;
      let port =
        match Unix.getsockname l with
        | Unix.ADDR_INET (_, p) -> p
        | Unix.ADDR_UNIX _ -> assert false
      in
      let d = Unix.socket ~cloexec:true Unix.PF_INET Unix.SOCK_STREAM 0 in
      Unix.connect d (Unix.ADDR_INET (Unix.inet_addr_loopback, port));
      let a, _ = Unix.accept ~cloexec:true l in
      (d, a))

(* Runs [f] on a thread: [join] is its result, or the exception it raised. *)
let spawn f =
  let r = ref None in
  let t =
    Thread.create
      (fun () ->
        r := Some (match f () with v -> Ok v | exception e -> Error e))
      ()
  in
  ( (fun () -> Option.is_some !r),
    fun () ->
      Thread.join t;
      match Option.get !r with Ok v -> v | Error e -> raise e )

let until ~what cond =
  let t0 = Unix.gettimeofday () in
  while not (cond ()) do
    if Unix.gettimeofday () -. t0 > patience then
      failf "%s: not within %.0f s" what patience;
    Thread.delay 0.001
  done

(* The agent *)

(* A region of the machine: its bytes once something wrote them. Copies between
   regions nothing wrote move nothing, so large regions cost no memory. *)
type region = { size : int; mutable bytes : Bytes.t option }

let bytes_of r =
  match r.bytes with
  | Some b -> b
  | None ->
      let b = Bytes.make r.size '\000' in
      r.bytes <- Some b;
      b

(* What the agent received, oldest first. *)
type event =
  | Alloc of { id : int; device : int; bytes : int }
  | Map of { id : int; device : int; region : int }
  | Drop of int
  | Handover of {
      device : int;
      value : int;
      waits : (int * int) list;
      local : int; (* copies whose bytes came from the controller *)
      parts : Wire.part list;
    }

type agent = {
  job : Link.job;
  link : Link.t; (* the agent's end *)
  memory : (int, region) Hashtbl.t;
  done_ : (int, int) Hashtbl.t; (* each device's last value reported *)
  mutable events : event list;
  mutable violations : string list;
  mutable room : int; (* bytes it still allocates *)
  lock : Mutex.t;
  mutable paused : bool;
  resumed : Condition.t;
}

let record ag e = Mutex.protect ag.lock (fun () -> ag.events <- e :: ag.events)
let events ag = Mutex.protect ag.lock (fun () -> List.rev ag.events)

let violate ag fmt =
  Printf.ksprintf
    (fun s ->
      Mutex.protect ag.lock (fun () -> ag.violations <- s :: ag.violations))
    fmt

let pause ag = Mutex.protect ag.lock (fun () -> ag.paused <- true)

let resume ag =
  Mutex.protect ag.lock (fun () ->
      ag.paused <- false;
      Condition.broadcast ag.resumed)

let gate ag =
  Mutex.protect ag.lock (fun () ->
      while ag.paused do
        Condition.wait ag.resumed ag.lock
      done)

(* The binary the host loads lists its functions, separated by commas; one that
   starts with "bad" is refused. *)
let functions = Hashtbl.create 8

let answer : type a. agent -> a Wire.request -> (a, string) result =
 fun ag -> function
  | Wire.Alloc { id; device; bytes; _ } ->
      if bytes > ag.room then Ok false
      else begin
        ag.room <- ag.room - bytes;
        Hashtbl.replace ag.memory id { size = bytes; bytes = None };
        record ag (Alloc { id; device; bytes });
        Ok true
      end
  | Wire.Map { id; device; region } -> (
      match Hashtbl.find_opt ag.memory region with
      | None -> Ok false
      | Some r ->
          Hashtbl.replace ag.memory id r;
          record ag (Map { id; device; region });
          Ok true)
  | Wire.Load { id; binary } ->
      if String.starts_with ~prefix:"bad" binary then
        Error "the agent refuses it"
      else begin
        Hashtbl.replace functions id (String.split_on_char ',' binary);
        Ok ()
      end
  | Wire.Entry { image; name } -> (
      match Hashtbl.find_opt functions image with
      | None -> Ok None
      | Some fs ->
          let rec index i = function
            | [] -> None
            | f :: _ when f = name -> Some i
            | _ :: fs -> index (i + 1) fs
          in
          Ok (index 0 fs))
  | Wire.Open _ | Wire.Join _ | Wire.Rail _ -> Error "not served here"

let region ag id =
  match Hashtbl.find_opt ag.memory id with
  | Some r -> r
  | None -> failwith (Printf.sprintf "no region %d" id)

let run_handover ag (h : Wire.handover) (local : Rig_remote_abi.area array) =
  gate ag;
  Array.iter
    (fun (d, v) ->
      let reached = Option.value ~default:0 (Hashtbl.find_opt ag.done_ d) in
      if reached < v then
        violate ag "device %d's value %d waits on device %d's %d, reached %d"
          h.device h.value d v reached)
    h.waits;
  let k = ref 0 in
  Array.iter
    (function
      | Wire.Words _ -> ()
      | Wire.Copy { src = Wire.Local; dst = Wire.Region { id; offset }; bytes }
        ->
          let a = local.(!k) in
          incr k;
          if Bigarray.Array1.dim a <> bytes then
            violate ag "a local copy of %d bytes came with %d" bytes
              (Bigarray.Array1.dim a);
          let m = bytes_of (region ag id) in
          for i = 0 to bytes - 1 do
            Bytes.set m (offset + i) a.{i}
          done
      | Wire.Copy { src = Wire.Region { id; offset }; dst = Wire.Local; bytes }
        ->
          let m = bytes_of (region ag id) in
          let a =
            Bigarray.Array1.create Bigarray.char Bigarray.c_layout bytes
          in
          for i = 0 to bytes - 1 do
            a.{i} <- Bytes.get m (offset + i)
          done;
          Link.bytes ag.link ~device:h.device ~value:h.value a
      | Wire.Copy
          {
            src = Wire.Region { id = s; offset = so };
            dst = Wire.Region { id = d; offset = doff };
            bytes;
          } -> (
          let rs = region ag s and rd = region ag d in
          match (rs.bytes, rd.bytes) with
          | None, None -> ()
          | _ -> Bytes.blit (bytes_of rs) so (bytes_of rd) doff bytes)
      | Wire.Copy { src = Wire.Local; dst = Wire.Local; _ } ->
          violate ag "a copy from Local to Local")
    h.parts;
  record ag
    (Handover
       {
         device = h.device;
         value = h.value;
         waits = Array.to_list h.waits;
         local = !k;
         parts = Array.to_list h.parts;
       });
  Hashtbl.replace ag.done_ h.device h.value;
  Link.word ag.link ~device:h.device h.value

let rec serve ag =
  match Link.next ag.link with
  | Error _ | Ok Wire.Close -> ()
  | Ok (Wire.Request r) ->
      Link.answer ag.link r (answer ag r);
      serve ag
  | Ok (Wire.Drop id) ->
      record ag (Drop id);
      serve ag
  | Ok (Wire.Handover (h, local)) ->
      run_handover ag h local;
      serve ag

(* A machine *)

type machine = {
  ag : agent;
  far : Link.t; (* the controller's end *)
  host : Rig.t;
  raw : Proxy.t; (* the host's proxy, which [host] drives *)
  devices : Rig.t array; (* MEM:1, MEM:2, … *)
}

let count = Atomic.make 0

(* A machine's name, which names one machine for the life of the process. *)
let fresh_machine () =
  Printf.sprintf "proxy-test:%d" (Atomic.fetch_and_add count 1)

let account id ~reaches : Wire.account =
  {
    id;
    name = (if id = 0 then "CPU" else Printf.sprintf "MEM:%d" id);
    arch = (if id = 0 then "arm64" else Printf.sprintf "mem%d" id);
    budget = (1 lsl 30) + id;
    reaches;
  }

let no_rails _ ~send:_ ~receive:_ = Error "no rails"
let ok_or_fail = function Ok d -> d | Error e -> failf "open: %s" e

(* A job, a link to a fresh machine whose agent serves on a thread, its host
   opened through rig, and a device for each list of [reaches] (the ids the
   device reaches). Ends every device and the job, and joins the agent. *)
let with_machine ?(reaches = []) ?(room = max_int) f =
  let job = Link.job () in
  let d, a = connected () in
  let far = Link.make job d ~name:"far" ~peer:(Wire.Agent 1) in
  let link = Link.make job a ~name:"controller" ~peer:Wire.Controller in
  let ag =
    {
      job;
      link;
      memory = Hashtbl.create 16;
      done_ = Hashtbl.create 4;
      events = [];
      violations = [];
      room;
      lock = Mutex.create ();
      paused = false;
      resumed = Condition.create ();
    }
  in
  let _, served =
    spawn (fun () ->
        try serve ag
        with e ->
          Link.fail job ("the agent raised " ^ Printexc.to_string e);
          raise e)
  in
  let machine = fresh_machine () in
  let host_account =
    account 0 ~reaches:(List.init (List.length reaches) succ)
  in
  let finish host =
    resume ag;
    Option.iter Rig.close host;
    (match Link.wait job ~ms:0 with
    | Link.Open -> Link.fail job "the test ended"
    | _ -> ());
    served ()
  in
  let raw =
    Proxy.make far host_account
      (Rig_remote_abi.Host { machine; rail = no_rails })
  in
  match
    Rig.open_host (module Proxy) ~machine ~name:"CPU" (fun () -> Ok raw)
  with
  | Error e ->
      finish None;
      failf "open: %s" e
  | Ok host ->
      Fun.protect
        ~finally:(fun () -> finish (Some host))
        (fun () ->
          let accounts =
            Array.of_list
              (host_account
              :: List.mapi (fun i r -> account (i + 1) ~reaches:r) reaches)
          in
          let devices =
            Array.init (List.length reaches) (fun i ->
                let a = accounts.(i + 1) in
                ok_or_fail
                  (Rig.open_
                     (module Proxy)
                     ~machine ~name:a.name
                     (fun () ->
                       Ok
                         (Proxy.make far a
                            (Rig_remote_abi.Device { id = a.id })))))
          in
          let m = { ag; far; host; raw; devices } in
          f m;
          match Mutex.protect ag.lock (fun () -> ag.violations) with
          | [] -> ()
          | vs -> failf "the agent saw: %s" (String.concat "; " (List.rev vs)))

(* Buffers *)

let host_buffer s = B.of_string s

let read_host b =
  let s = Bytes.create (B.length b) in
  B.blit_to_bytes b 0 s 0 (B.length b);
  Bytes.to_string s

(* [b]'s bytes, through a copy to this process. *)
let read b =
  let h = B.create Rig.host (B.length b) in
  B.copy ~src:b ~dst:h;
  read_host h

let far_of_string d s =
  let b = B.create d (String.length s) in
  B.copy ~src:(host_buffer s) ~dst:b;
  b

let random_string n = String.init n (fun _ -> Char.chr (Random.int 256))

let copy_submission d ~src ~dst =
  Sub.make ~reads:0 ~writes:0 d
    [| { Sub.queue = "COPY:0"; after = [||]; work = Sub.Copy { src; dst } } |]

let submit ?(waits = [||]) s =
  Rig.submit s ~run:(Sub.Run.make ()) ~reads:[||] ~writes:[||] ~waits

let lost_w =
  Testable.structural ~pp:(fun ppf -> function
    | None -> Format.pp_print_string ppf "not lost"
    | Some why -> Format.fprintf ppf "lost: %S" why)

(* A job with one link [far], whose peer end nothing serves. *)
let with_job_link f =
  let job = Link.job () in
  Fun.protect
    ~finally:(fun () ->
      match Link.wait job ~ms:0 with
      | Link.Open -> Link.fail job "the test ended"
      | _ -> ())
    (fun () ->
      let d, a = connected () in
      let far = Link.make job d ~name:"far" ~peer:(Wire.Agent 1) in
      ignore (Link.make job a ~name:"controller" ~peer:Wire.Controller);
      f far)
