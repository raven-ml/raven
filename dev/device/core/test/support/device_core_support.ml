(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external polled_new : int -> bool -> nativeint = "device_core_test_polled_new"
external polled_fail : nativeint -> unit = "device_core_test_polled_fail"
external polled_run : nativeint -> int = "device_core_test_polled_run"
external polled_queued : nativeint -> int = "device_core_test_polled_queued"
external polled_submits : nativeint -> int = "device_core_test_polled_submits"
external polled_blocked : nativeint -> int = "device_core_test_polled_blocked"
external polled_room : unit -> nativeint = "device_core_test_polled_room"
external polled_submit : unit -> nativeint = "device_core_test_polled_submit"
external polled_word : nativeint -> int = "device_core_test_polled_word"

external polled_set_word : nativeint -> int -> unit
  = "device_core_test_polled_set_word"

external host_alloc : int -> int = "device_core_test_alloc"
external host_free : int -> unit = "device_core_test_free"
external bump : unit -> nativeint = "device_core_test_bump"
external poke : unit -> nativeint = "device_core_test_poke"
external interrupt : unit -> unit = "device_core_test_interrupt"
external load : int -> int = "device_core_test_load"
external store : int -> int -> unit = "device_core_test_store"

let bump = bump ()
let poke = poke ()

module Driver = struct
  type kind = [ `Device | `Pinned | `Mapped ]

  type t = {
    c : nativeint;
    copies : bool;
    host_visible : bool;
    transport : bool;
    peers : bool;
    budget : int;
    limits : kind -> int;
    may_block : bool;
    waits_host : bool;
    answer : [ `Stopped | `Unknown ];
    lock : Mutex.t;
    opened : Condition.t;
    mutable calls : string list;
    mutable frees : (int * int) list;
    mutable allocs : (kind * int * bool) list;
    mutable maps : int list;
    mutable held : (kind * int) list;
    mutable fault : string option;
    mutable interrupt_next : bool;
    mutable stalls : int;
    mutable gated : bool;
    mutable sleepers : int;
  }

  (* A region the driver allocated has a kind; a mapping has none. *)
  type region = { at : int; kind : kind option; bytes : int; visible : bool }
  type image = region
  type part = unit
  type capability = unit

  exception Fault of string

  let note d call = Mutex.protect d.lock (fun () -> d.calls <- call :: d.calls)
  let log d = Mutex.protect d.lock (fun () -> List.rev d.calls)
  let key : t Type.Id.t = Type.Id.make ()
  let arch _ = "polled"
  let budget d = d.budget
  let queues d = if d.copies then [ "COMPUTE:0"; "COPY:0" ] else [ "COMPUTE:0" ]

  (* A counted call of a faulted device raises its fault. *)
  let counted d call =
    note d call;
    match Mutex.protect d.lock (fun () -> d.fault) with
    | Some why -> raise (Fault why)
    | None -> ()

  let holding d kind =
    List.fold_left (fun n (k, b) -> if k = kind then n + b else n) 0 d.held

  let alloc d kind n =
    counted d "alloc";
    let fits =
      Mutex.protect d.lock (fun () ->
          let fits = holding d kind + n <= d.limits kind in
          d.allocs <- (kind, n, fits) :: d.allocs;
          if fits then d.held <- (kind, n) :: d.held;
          fits)
    in
    if not fits then None
    else
      let at = host_alloc n in
      let visible = d.host_visible || kind <> `Device in
      Some { at; kind = Some kind; bytes = n; visible }

  let rec remove x = function
    | [] -> []
    | y :: l -> if y = x then l else y :: remove x l

  let free d r =
    note d "free";
    let w = polled_word d.c in
    Mutex.protect d.lock (fun () ->
        d.frees <- (r.at, w) :: d.frees;
        match r.kind with
        | Some k -> d.held <- remove (k, r.bytes) d.held
        | None -> ());
    if r.kind <> None then host_free r.at

  let address r = Some r.at
  let handle r = Nativeint.of_int r.at
  let host r = if r.visible then Some (Nativeint.of_int r.at) else None
  let peer d _ = d.peers

  let map_peer d _ r =
    counted d "map_peer";
    if d.peers then Some { r with kind = None } else None

  let map_host d p n =
    counted d "map_host";
    Mutex.protect d.lock (fun () -> d.maps <- n :: d.maps);
    Some { at = Nativeint.to_int p; kind = None; bytes = n; visible = true }

  let unmap d _ = note d "unmap"

  let image d b =
    counted d "image";
    match String.split_on_char ':' b with
    | [ "code"; n ] ->
        let n = int_of_string n in
        Ok (`Place (n, fun r -> (r, String.make n 'c')))
    | _ -> Error "not a polled binary"

  let entry r f = if f = "main" then Some r.at else None
  let unload d _ = note d "unload"

  let word d =
    {
      at = Nativeint.to_int d.c;
      kind = None;
      bytes = 8;
      visible = not d.transport;
    }

  let signaled d = polled_word d.c

  (* What a sleep does once its gate opens. *)
  let next d =
    if d.gated then begin
      d.sleepers <- d.sleepers + 1;
      while d.gated do
        Condition.wait d.opened d.lock
      done;
      d.sleepers <- d.sleepers - 1
    end;
    match d.fault with
    | Some why -> `Fault why
    | None when d.interrupt_next ->
        d.interrupt_next <- false;
        `Interrupt
    | None when d.stalls > 0 ->
        d.stalls <- d.stalls - 1;
        `Stall
    | None -> `Run

  let sleep d ~seen:_ ~still_ms =
    note d "sleep";
    match Mutex.protect d.lock (fun () -> next d) with
    | `Fault why -> raise (Fault why)
    | `Interrupt -> interrupt ()
    | `Stall -> Thread.delay (float still_ms /. 1000.)
    | `Run -> ignore (polled_run d.c)

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

  let make ?(capacity = 1024) ?(copies = true) ?(host_visible = true)
      ?(transport = false) ?(peers = true) ?(budget = 1 lsl 30)
      ?(memory = max_int) ?(window = max_int) ?(may_block = false)
      ?(waits_host = false) ?(answer = `Stopped) () =
    let limits = function
      | `Device -> memory
      | `Mapped -> window
      | `Pinned -> max_int
    in
    {
      c = polled_new capacity may_block;
      copies;
      host_visible;
      transport;
      peers;
      budget;
      limits;
      may_block;
      waits_host;
      answer;
      lock = Mutex.create ();
      opened = Condition.create ();
      calls = [];
      frees = [];
      allocs = [];
      maps = [];
      held = [];
      fault = None;
      interrupt_next = false;
      stalls = 0;
      gated = false;
      sleepers = 0;
    }

  let open_ ?capacity ?copies ?host_visible ?transport ?peers ?budget ?memory
      ?window ?may_block ?waits_host ?answer name =
    let p =
      make ?capacity ?copies ?host_visible ?transport ?peers ?budget ?memory
        ?window ?may_block ?waits_host ?answer ()
    in
    match Device_core.open_ (module Driver) ~name (fun () -> Ok p) with
    | Ok d -> (d, p)
    | Error e -> failwith e

  let run d = polled_run d.c
  let queued d = polled_queued d.c
  let submits d = polled_submits d.c
  let fail d = polled_fail d.c
  let fault d why = Mutex.protect d.lock (fun () -> d.fault <- Some why)
  let set_word d v = polled_set_word d.c v
  let blocked d = polled_blocked d.c
  let frees d = Mutex.protect d.lock (fun () -> List.rev d.frees)
  let allocs d = Mutex.protect d.lock (fun () -> List.rev d.allocs)
  let host_maps d = Mutex.protect d.lock (fun () -> List.rev d.maps)
  let allocated d kind = Mutex.protect d.lock (fun () -> holding d kind)
  let interrupt d = Mutex.protect d.lock (fun () -> d.interrupt_next <- true)
  let stall d n = Mutex.protect d.lock (fun () -> d.stalls <- n)
  let gate d = Mutex.protect d.lock (fun () -> d.gated <- true)
  let sleepers d = Mutex.protect d.lock (fun () -> d.sleepers)

  let open_gate d =
    Mutex.protect d.lock (fun () ->
        d.gated <- false;
        Condition.broadcast d.opened)
end

(* Waiting for a signal *)

let watchdog_s = 10.

let await what f =
  let until = Unix.gettimeofday () +. watchdog_s in
  while not (f ()) do
    if Unix.gettimeofday () > until then
      failwith (Printf.sprintf "await: no %s after %.0f s" what watchdog_s);
    Thread.yield ()
  done

(* Readers *)

module Reader = struct
  external host : Device_core.Buffer.t -> int = "device_core_test_reader_host"
  external bytes : Device_core.Buffer.t -> int = "device_core_test_reader_bytes"

  external why : Device_core.Buffer.t -> string option
    = "device_core_test_reader_why"
end
