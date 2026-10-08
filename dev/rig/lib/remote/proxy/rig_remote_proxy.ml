(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Wire = Wire
module Link = Link

let strf = Printf.sprintf

type capability = Rig_remote_abi.t

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
  capability : capability;
  word : region;
}

external make_c : Link.t -> int -> nativeint = "caml_rig_remote_proxy_make"
external word_at : nativeint -> int = "caml_rig_remote_proxy_word"
external signaled_c : nativeint -> int = "caml_rig_remote_proxy_signaled"

external sleep_c : nativeint -> int -> int -> bool
  = "caml_rig_remote_proxy_sleep"

external stop_c : nativeint -> unit = "caml_rig_remote_proxy_stop"
external room : unit -> nativeint = "caml_rig_remote_proxy_room_entry"
external submit : unit -> nativeint = "caml_rig_remote_proxy_submit_entry"

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
let arch d = d.account.arch
let budget d = d.account.budget
let queues _ = [ "COMPUTE:0"; "COPY:0" ]
let completion _ = `Host
let waits_on _ = function `Host -> true | `Store | `Object -> false
let max_waits _ = max_int
let blocks _ = `May_block
let maps_host _ = false
let capability d = d.capability
let capability_key = Rig_remote_abi.key

(* Memory *)

let alloc d memory bytes =
  let id = Link.fresh () in
  match
    request d.link (Wire.Alloc { id; device = d.account.id; memory; bytes })
  with
  | Ok true -> Some (Memory { id })
  | Ok false | Error _ -> None

let free d = function Memory { id } -> Link.drop d.link id | Word _ -> ()
let address = function Memory _ -> None | Word { id; _ } -> Some id
let handle = function Memory { id } -> Nativeint.of_int id | Word _ -> 0n
let host = function Memory _ -> None | Word { at; _ } -> Some at
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
    | Ok () -> Ok (`Loaded { id; link = d.link })
    | Error why -> Error why

let entry (i : image) name =
  match request i.link (Wire.Entry { image = i.id; name }) with
  | Ok e -> e
  | Error _ -> None

let unload d (i : image) = Link.drop d.link i.id

(* Timeline *)

let word d = d.word
let signaled d = signaled_c d.c

let fault d =
  Fault
    (Option.value ~default:"the job failed" (Link.failure (Link.job_of d.link)))

let sleep d ~seen ~still_ms = if sleep_c d.c seen still_ms then raise (fault d)
let room_entry = room ()
let submit_entry = submit ()
let self d = d.c
let stop d = stop_c d.c
