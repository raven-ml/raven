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

external make_c : Link.t -> int -> nativeint = "caml_rig_remote_proxy_make"
external word_at : nativeint -> int = "caml_rig_remote_proxy_word"
external signaled_c : nativeint -> int = "caml_rig_remote_proxy_signaled"

external sleep_c : nativeint -> int -> int -> bool
  = "caml_rig_remote_proxy_sleep"

external stop_c : nativeint -> unit = "caml_rig_remote_proxy_stop"

let make l (a : Wire.account) c =
  (match c with
  | Rig_remote_abi.Host _ when a.id <> 0 ->
      invalid_arg "Rig_remote_proxy.make: a host's record for a device"
  | Rig_remote_abi.Device { id } when id <> a.id ->
      invalid_arg "Rig_remote_proxy.make: a record of another device"
  | _ -> ());
  let c' = make_c l a.id in
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
    completion = Host;
    waits = { stores = false; hosts = true; objects = false; most = max_int };
    may_block = true;
    hang_ms = None;
    maps_host = false;
    host_addresses = false;
    capability = Capability (Rig_remote_abi.key, d.capability);
    word = d.word;
    edge = d.c;
  }

let capability d = d.capability

(* Memory *)

let alloc d (memory : Rig_edge.memory) bytes =
  let memory =
    match memory with Device -> `Device | Pinned -> `Pinned | Mapped -> `Mapped
  in
  let id = Link.fresh () in
  match
    request d.link (Wire.Alloc { id; device = d.account.id; memory; bytes })
  with
  | Ok true -> Some (Memory { id })
  | Ok false | Error _ -> None

let free d = function Memory { id } -> Link.drop d.link id | Word _ -> ()
let locate = function
  | Memory { id } ->
      { Rig_edge.address = None; host = None; handle = Nativeint.of_int id }
  | Word { id; at } -> { address = Some id; host = Some at; handle = 0n }

let peer d d' = d.link == d'.link && List.mem d'.account.id d.account.reaches

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

let image d binary =
  if d.account.id <> 0 then Error (strf "%s loads no code" d.account.name)
  else
    let id = Link.fresh () in
    match request d.link (Wire.Load { id; binary }) with
    | Ok () -> Ok (Rig_edge.Loaded { id; link = d.link })
    | Error why -> Error why

let entry (i : image) name =
  match request i.link (Wire.Entry { image = i.id; name }) with
  | Ok e -> Option.map (fun code -> { Rig_edge.code; launch = 0n }) e
  | Error _ -> None

let unload d (i : image) = Link.drop d.link i.id

(* Timeline *)

let signaled d = signaled_c d.c

let fault d =
  Fault
    (Option.value ~default:"the job failed" (Link.failure (Link.job_of d.link)))

let sleep d ~seen ~still_ms = if sleep_c d.c seen still_ms then raise (fault d)
let stop d ~fault:_ = stop_c d.c
