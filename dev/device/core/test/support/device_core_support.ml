(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external polled_new : int -> bool -> nativeint = "device_core_test_polled_new"
external polled_fail : nativeint -> unit = "device_core_test_polled_fail"
external polled_run : nativeint -> int = "device_core_test_polled_run"
external polled_queued : nativeint -> int = "device_core_test_polled_queued"
external polled_submits : nativeint -> int = "device_core_test_polled_submits"
external polled_room : unit -> nativeint = "device_core_test_polled_room"
external polled_submit : unit -> nativeint = "device_core_test_polled_submit"
external polled_word : nativeint -> int = "device_core_test_polled_word"

external polled_set_word : nativeint -> int -> unit
  = "device_core_test_polled_set_word"

external host_alloc : int -> int = "device_core_test_alloc"
external host_free : int -> unit = "device_core_test_free"
external bump : unit -> nativeint = "device_core_test_bump"
external load : int -> int = "device_core_test_load"
external store : int -> int -> unit = "device_core_test_store"

let bump = bump ()

module Driver = struct
  type t = {
    c : nativeint;
    may_block : bool;
    waits_host : bool;
    answer : [ `Stopped | `Unknown ];
    lock : Mutex.t;
    mutable calls : string list;
    mutable fault_next : string option;
  }

  type region = { at : int; owned : bool }
  type image = region
  type part = unit
  type capability = unit

  exception Fault of string

  let note d call = Mutex.protect d.lock (fun () -> d.calls <- call :: d.calls)
  let log d = Mutex.protect d.lock (fun () -> List.rev d.calls)
  let key : t Type.Id.t = Type.Id.make ()
  let arch _ = "polled"
  let budget _ = 1 lsl 30
  let queues _ = [ "COMPUTE:0"; "COPY:0" ]

  let alloc d _ n =
    note d "alloc";
    let at = host_alloc n in
    if at = 0 then None else Some { at; owned = true }

  let free d r =
    note d "free";
    if r.owned then host_free r.at

  let address r = Some r.at
  let handle r = Nativeint.of_int r.at
  let host r = Some (Nativeint.of_int r.at)
  let peer _ _ = true

  let map_peer d _ r =
    note d "map_peer";
    Some { r with owned = false }

  let map_host d p _ =
    note d "map_host";
    Some { at = Nativeint.to_int p; owned = false }

  let unmap d _ = note d "unmap"

  let image d b =
    note d "image";
    match String.split_on_char ':' b with
    | [ "code"; n ] ->
        let n = int_of_string n in
        let r = { at = host_alloc n; owned = true } in
        Ok (r, Some (r, String.make n 'c'))
    | [ "nomem"; n ] -> Error (`No_memory (int_of_string n))
    | _ -> Error (`Refused "not a polled binary")

  let entry r f = if f = "main" then Some r.at else None

  let unload d r =
    note d "unload";
    host_free r.at

  let word d = { at = Nativeint.to_int d.c; owned = false }
  let signaled d = polled_word d.c

  let sleep d ~seen:_ ~still_ms:_ =
    note d "sleep";
    match
      Mutex.protect d.lock (fun () ->
          let f = d.fault_next in
          d.fault_next <- None;
          f)
    with
    | Some why -> raise (Fault why)
    | None -> ignore (polled_run d.c)

  let completion _ = `Host
  let waits_on d c = d.waits_host && c = `Host
  let blocks d = if d.may_block then `May_block else `Returns

  let part _ ~queue:_ ?after:_ = function
    | `Words _ -> invalid_arg "Polled.part: no words"
    | _ -> ()

  let room _ _ = `Fits
  let submit _ ~v:_ ~waits:_ ~handles:_ _ = `Failed "Polled runs from C only"
  let room_entry = polled_room ()
  let submit_entry = polled_submit ()
  let self d = d.c
  let capability _ = ()
  let capability_key : capability Type.Id.t = Type.Id.make ()

  let stop d =
    note d "stop";
    d.answer
end

module Polled = struct
  include Driver

  let make ?(capacity = 1024) ?(may_block = false) ?(waits_host = false)
      ?(answer = `Stopped) () =
    {
      c = polled_new capacity may_block;
      may_block;
      waits_host;
      answer;
      lock = Mutex.create ();
      calls = [];
      fault_next = None;
    }

  let open_ ?capacity ?may_block ?waits_host ?answer name =
    let p = make ?capacity ?may_block ?waits_host ?answer () in
    match Device_core.open_ (module Driver) ~name (fun () -> Ok p) with
    | Ok d -> (d, p)
    | Error e -> failwith e

  let run d = polled_run d.c
  let queued d = polled_queued d.c
  let submits d = polled_submits d.c
  let fail d = polled_fail d.c
  let fault d why = Mutex.protect d.lock (fun () -> d.fault_next <- Some why)
  let set_word d v = polled_set_word d.c v
end
