(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Wire = Wire
module Link = Link

let strf = Printf.sprintf

exception Fault of string

(* A region is memory of the agent, named by its id, or a proxy's word: its
   agent's id and the shadow's host address. *)
type region = Memory of { id : int } | Word of { id : int; at : int }
type image = { id : int; link : Link.t }

(* The C state's address comes first: the C entries take it as [self]. *)
type t = {
  c : nativeint;
  link : Link.t;
  account : Wire.account;
  capability : Rig_remote_abi.t;
  word : region;
}

external make_c : Link.t -> int -> nativeint -> nativeint
  = "caml_rig_remote_proxy_make"
external word_at : nativeint -> int = "caml_rig_remote_proxy_word"
external signaled_c : nativeint -> int = "caml_rig_remote_proxy_signaled"

external sleep_c : nativeint -> int -> int -> bool
  = "caml_rig_remote_proxy_sleep"

external stop_c : nativeint -> unit = "caml_rig_remote_proxy_stop"

let make l (a : Wire.account) c ~word =
  (match c with
  | Rig_remote_abi.Host _ when a.id <> 0 ->
      invalid_arg "Rig_remote_proxy.device: a host's record for a device"
  | Rig_remote_abi.Device { id } when id <> a.id ->
      invalid_arg "Rig_remote_proxy.device: a record of another device"
  | _ -> ());
  let c' = make_c l a.id word in
  {
    c = c';
    link = l;
    account = a;
    capability = c;
    word = Word { id = a.id; at = word_at c' };
  }

(* The job's failure raises; the agent's refusal is the request's [Error]. *)
let request l q =
  match Link.request l q with
  | Ok v -> Ok v
  | Error (`Failed root) -> raise (Fault root)
  | Error (`Refused why) -> Error why

(* Facts *)

let key : t Type.Id.t = Type.Id.make ()

(* The machine's host runs words too: the runs of its loaded code. *)
let facts d =
  let runs = Rig_edge.(if d.account.id = 0 then [ Words; Copy ] else [ Copy ]) in
  {
    Rig_edge.arch = d.account.arch;
    budget = d.account.budget;
    queues = [ { name = "COMPUTE:0"; runs }; { name = "COPY:0"; runs } ];
    copy_engine = Some 1;
    completion = Host;
    waits = { stores = false; hosts = true; most = max_int };
    may_block = true;
    hang_ms = None;
    maps_host = false;
    host_addresses = false;
    alignment = d.account.alignment;
    capability = Rig_remote_abi.declare d.capability;
    word = d.word;
    edge = d.c;
  }

let capability d = d.capability

(* Memory *)

let alloc d memory bytes =
  let id = Link.fresh () in
  match
    request d.link (Wire.Alloc { id; device = d.account.id; memory; bytes })
  with
  | Ok true -> Some (Memory { id })
  | Ok false | Error _ -> None

(* A word region that rig frees is another device's mapping of a word, which
   [map_peer] answers with the word itself: nothing to give back. *)
let free d = function Memory { id } -> Link.drop d.link id | Word _ -> ()

let locate = function
  | Memory { id } ->
      { Rig_edge.address = None; host = None; handle = Nativeint.of_int id }
  | Word { id; at } -> { address = Some id; host = Some at; handle = 0n }

let peer d d' =
  d.link == d'.link
  &&
  if d.account.id > d'.account.id then List.mem d'.account.id d.account.reaches
  else List.mem d.account.id d'.account.reached_by

let map_peer d d' r =
  if d.link != d'.link then None
  else
    match r with
    | Word _ -> Some r
    | Memory { id = region } -> (
        let id = Link.fresh () in
        match
          request d.link (Wire.Map { id; device = d.account.id; region })
        with
        | Ok true -> Some (Memory { id })
        | Ok false | Error _ -> None)

let map_host _ _ _ = None

(* Code *)

(* What the agents stated of each program they loaded, by id, until its
   unload: the most threads a group of each of its launches has. It changes
   by atomic swaps and takes no lock, so a signal handler, finaliser or
   Gc.Memprof callback inside a load or an unload may load or unload too. *)
module Ids = Map.Make (Int)

let launches = Atomic.make Ids.empty

let rec change f =
  let m = Atomic.get launches in
  if not (Atomic.compare_and_set launches m (f m)) then change f

let threads id =
  Option.value ~default:[||] (Ids.find_opt id (Atomic.get launches))

let image d binary =
  if d.account.id <> 0 then Error (strf "%s loads no code" d.account.name)
  else
    let id = Link.fresh () in
    match request d.link (Wire.Load { id; binary }) with
    | Ok limits ->
        change (Ids.add id limits);
        Ok (Rig_edge.Loaded { id; link = d.link })
    | Error why -> Error why

let entry (i : image) name =
  match request i.link (Wire.Entry { image = i.id; name }) with
  | Ok (Some code) ->
      Rig_edge.Entry { code; launch = 0n; params = None; threads = None }
  | Ok None | Error _ -> Absent

let unload d (i : image) =
  change (Ids.remove i.id);
  Link.drop d.link i.id

(* Timeline *)

let signaled d = signaled_c d.c

let fault d =
  Fault
    (Option.value ~default:"the job failed" (Link.failed (Link.job_of d.link)))

let sleep d ~seen ~still_ms = if sleep_c d.c seen still_ms then raise (fault d)
let stop d ~fault:_ = stop_c d.c

(* Its room never answers RIG_GROW: the device's own driver grows its
   queues on its machine. *)
let grow _ = Rig_edge.Grown

module Driver = struct
  type nonrec t = t
  type nonrec region = region
  type nonrec image = image

  exception Fault = Fault

  let key = key
  let facts = facts
  let alloc = alloc
  let free = free
  let locate = locate
  let peer = peer
  let map_peer = map_peer
  let map_host = map_host
  let image = image
  let entry = entry
  let unload = unload
  let signaled = signaled
  let sleep = sleep
  let stop = stop
  let grow = grow
end

let device l (a : Wire.account) c : t Rig_edge.opener =
  {
    name = a.name;
    driver = (module Driver);
    make = (fun ~word -> Ok (make l a c ~word));
  }
