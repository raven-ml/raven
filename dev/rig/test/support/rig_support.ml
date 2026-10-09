(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external polled_new : int -> bool -> int -> nativeint = "rig_test_polled_new"
external polled_fail : nativeint -> unit = "rig_test_polled_fail"
external polled_fail_commit : nativeint -> unit = "rig_test_polled_fail_commit"
external polled_run : nativeint -> int = "rig_test_polled_run"
external polled_drive : nativeint -> int = "rig_test_polled_drive"
external polled_start : nativeint -> unit = "rig_test_polled_start"
external polled_queued : nativeint -> int = "rig_test_polled_queued"
external polled_submits : nativeint -> int = "rig_test_polled_submits"
external polled_blocked : nativeint -> int = "rig_test_polled_blocked"
external polled_step : nativeint -> int = "rig_test_polled_step" [@@noalloc]
external polled_steps : nativeint -> int = "rig_test_polled_steps"

external polled_fail_at : nativeint -> int -> int -> string -> unit
  = "rig_test_polled_fail_at"

external polled_last_waits : nativeint -> int array
  = "rig_test_polled_last_waits"

external polled_last_handles : nativeint -> int array
  = "rig_test_polled_last_handles"

external polled_copy_sides : nativeint -> int array
  = "rig_test_polled_copy_sides"

external rig_local : unit -> int * int * int = "rig_test_rig_local"
external rig_word : unit -> int = "rig_test_rig_word"
external rig_object : unit -> int = "rig_test_rig_object"
external polled_word_at : nativeint -> int = "rig_test_polled_word_at"
external polled_word : nativeint -> int = "rig_test_polled_word"
external polled_stop : nativeint -> unit = "rig_test_polled_stop"
external polled_set_word : nativeint -> int -> unit = "rig_test_polled_set_word"
external host_alloc : int -> int = "rig_test_alloc"
external host_free : int -> unit = "rig_test_free"
external bump : unit -> nativeint = "rig_test_bump"
external poke : unit -> nativeint = "rig_test_poke"
external carry : unit -> nativeint = "rig_test_carry"
external countdown : unit -> nativeint = "rig_test_countdown"
external interrupt : unit -> unit = "rig_test_interrupt"

external shares : ('a, 'b, 'c) Bigarray.Array1.t -> int = "rig_test_shares"
[@@noalloc]

external host_held : unit -> int = "rig_test_host_held"
external heap_bytes : unit -> int = "rig_test_heap_bytes"
external load : int -> int = "rig_test_load"
external store : int -> int -> unit = "rig_test_store"
external move : dst:int -> src:int -> int -> unit = "rig_test_move"

let bump = bump ()
let rig_word = rig_word ()
let rig_object = rig_object ()
let poke = poke ()
let carry = carry ()
let countdown = countdown ()

module Driver = struct
  type kind = Rig_edge.memory

  type t = {
    c : nativeint;
    word_at : int;  (** The word's host address. *)
    copies : bool;
    host_visible : bool;
    transport : bool;
    peers : bool;
    maps_host : bool;
    budget : int;
    limits : kind -> int;
    may_block : bool;
    objects : bool;
    waits : [ `Store | `Object | `Host ] list;
    max_waits : int;
    hang_ms : int option;
    answer : [ `Stopped | `Unknown ];
    lock : Mutex.t;
    opened : Condition.t;
    mutable calls : string list;
    mutable frees : (int * int) list;
    mutable allocs : (kind * int * bool) list;
    mutable maps : int list;
    mutable held : (kind * int) list;
    mutable mapped : int list;
    mutable failure : string;
    mutable fault : string option;
    mutable word_fault : string option;
    mutable interrupt_next : bool;
    mutable stalls : int;
    mutable gated : bool;
    mutable sleepers : int;
    mutable stop_fault : string option;  (** What its stop was given. *)
  }

  (* A region the driver allocated has a kind; a mapping has none. *)
  type region = { at : int; kind : kind option; bytes : int; visible : bool }
  type image = { code : region; owner : t }

  exception Fault of string

  let note d call = Mutex.protect d.lock (fun () -> d.calls <- call :: d.calls)
  let log d = Mutex.protect d.lock (fun () -> List.rev d.calls)
  let key : t Type.Id.t = Type.Id.make ()

  (* Counts a fallible call: the call [fail_at] fails raises, and the one it
     refuses answers [true]. [d]'s lock is not held. *)
  let countdown d =
    match polled_step d.c with
    | 2 -> raise (Fault (Mutex.protect d.lock (fun () -> d.failure)))
    | s -> s = 1

  (* A counted call or a fact: a faulted device's raises its fault. *)
  let step d =
    match Mutex.protect d.lock (fun () -> d.fault) with
    | Some why -> raise (Fault why)
    | None -> countdown d

  let capability_key : unit Type.Id.t = Type.Id.make ()

  (* Its queues run fills and, with copies, copies. The facts are one call,
     which goes on whether or not it refuses. *)
  let facts d =
    ignore (step d : bool);
    let runs = Rig_edge.(if d.copies then [ Fill; Copy ] else [ Fill ]) in
    let compute = { Rig_edge.name = "COMPUTE:0"; runs } in
    let has c = List.mem c d.waits in
    {
      Rig_edge.arch = "polled";
      budget = d.budget;
      queues =
        (if d.copies then [ compute; { name = "COPY:0"; runs } ]
         else [ compute ]);
      completion =
        (if d.objects then Object (Nativeint.of_int d.word_at) else Host);
      waits =
        {
          stores = has `Store;
          hosts = has `Host;
          objects = has `Object;
          most = d.max_waits;
        };
      may_block = d.may_block;
      hang_ms = d.hang_ms;
      maps_host = d.maps_host;
      capability = Capability (capability_key, ());
      word =
        { at = d.word_at; kind = None; bytes = 8; visible = not d.transport };
      edge = d.c;
    }

  let counted d call =
    note d call;
    step d

  let holding d kind =
    List.fold_left (fun n (k, b) -> if k = kind then n + b else n) 0 d.held

  let alloc d kind n =
    let refused = counted d "alloc" in
    let fits =
      Mutex.protect d.lock (fun () ->
          let fits = (not refused) && holding d kind + n <= d.limits kind in
          d.allocs <- (kind, n, fits) :: d.allocs;
          if fits then d.held <- (kind, n) :: d.held;
          fits)
    in
    if not fits then None
    else
      let at = host_alloc n in
      let visible = d.host_visible || kind <> Device in
      Some { at; kind = Some kind; bytes = n; visible }

  let rec remove x = function
    | [] -> []
    | y :: l -> if y = x then l else y :: remove x l

  (* A mapping's free is logged as ["unmap"]. *)
  let free_region d r =
    note d (if r.kind = None then "unmap" else "free");
    let w = polled_word d.c in
    Mutex.protect d.lock (fun () ->
        d.frees <- (r.at, w) :: d.frees;
        match r.kind with
        | Some k -> d.held <- remove (k, r.bytes) d.held
        | None -> d.mapped <- remove r.bytes d.mapped);
    if r.kind <> None then host_free r.at

  (* The word is the device's own record: its free is logged as ["word"]
     alone. *)
  let free d r =
    if r.kind = None && r.at = d.word_at then note d "word"
    else free_region d r

  let locate r =
    {
      Rig_edge.address = Some r.at;
      host = (if r.visible then Some r.at else None);
      handle = Nativeint.of_int r.at;
    }

  let peer d _ = d.peers

  let mapping d m =
    Mutex.protect d.lock (fun () -> d.mapped <- m.bytes :: d.mapped);
    Some m

  let map_peer d _ r =
    if counted d "map_peer" || not d.peers then None
    else mapping d { r with kind = None }

  let map_host d p n =
    if counted d "map_host" then None
    else begin
      Mutex.protect d.lock (fun () -> d.maps <- n :: d.maps);
      mapping d { at = p; kind = None; bytes = n; visible = true }
    end

  let image d b =
    if counted d "image" then Error "refused"
    else
      match String.split_on_char ':' b with
      | [ "code"; n ] ->
          let n = int_of_string n in
          Ok
            (Rig_edge.Place
               (n, fun r -> ({ code = r; owner = d }, String.make n 'c')))
      | _ -> Error "not a polled binary"

  let entry i f =
    if counted i.owner "entry" || f <> "main" then None else Some i.code.at

  let unload d _ = ignore (counted d "unload" : bool)

  (* Reads the fault under the lock without a closure, so that a read of the
     word allocates nothing, as a driver's must not. *)
  let signaled d =
    Mutex.lock d.lock;
    let fault = d.word_fault in
    Mutex.unlock d.lock;
    match fault with
    | Some why -> raise (Fault why)
    | None ->
        ignore (countdown d : bool);
        polled_word d.c

  (* What a sleep does once its gate opens. *)
  (* Blocks at [d]'s gate while it is shut. [d]'s lock is held. *)
  let at_gate d =
    if d.gated then begin
      d.sleepers <- d.sleepers + 1;
      while d.gated do
        Condition.wait d.opened d.lock
      done;
      d.sleepers <- d.sleepers - 1
    end

  let next d =
    at_gate d;
    match d.fault with
    | Some why -> `Fault why
    | None when polled_step d.c = 2 -> `Fault d.failure
    | None when d.interrupt_next ->
        d.interrupt_next <- false;
        `Interrupt
    | None when d.stalls > 0 ->
        d.stalls <- d.stalls - 1;
        `Stall
    | None -> `Run

  let sleep d ~seen ~still_ms =
    note d "sleep";
    match Mutex.protect d.lock (fun () -> next d) with
    | `Fault why -> raise (Fault why)
    | `Interrupt -> interrupt ()
    | `Stall -> Thread.delay (float still_ms /. 1000.)
    | `Run ->
        if polled_drive d.c < 0 && polled_word d.c = seen then
          failwith "Polled: nothing committed"

  let stop d ~fault =
    note d "stop";
    Mutex.protect d.lock (fun () -> d.stop_fault <- fault);
    Mutex.protect d.lock (fun () -> at_gate d);
    if d.answer = `Stopped then polled_stop d.c
end

module Polled = struct
  include Driver

  let make ?(capacity = 1024) ?(copies = true) ?(host_visible = true)
      ?(transport = false) ?(peers = true) ?(maps_host = true)
      ?(budget = 1 lsl 30) ?(memory = max_int) ?(window = max_int)
      ?(may_block = false) ?(completion = `Host) ?(waits_on = [])
      ?(max_waits = max_int) ?(answer = `Stopped) ?(runs = `When_slept)
      ?(lag = 1) ?hang_ms () =
    if lag < 1 then invalid_arg "Polled.make: lag is below 1";
    let limits = function
      | Rig_edge.Device -> memory
      | Mapped -> window
      | Pinned -> max_int
    in
    let c = polled_new capacity may_block lag in
    let d =
      {
        c;
        word_at = polled_word_at c;
        copies;
        host_visible;
        transport;
        peers;
        maps_host;
        budget;
        limits;
        may_block;
        objects = completion = `Object;
        waits = waits_on;
        max_waits;
        hang_ms;
        answer;
        lock = Mutex.create ();
        opened = Condition.create ();
        calls = [];
        frees = [];
        allocs = [];
        maps = [];
        held = [];
        mapped = [];
        failure = "";
        fault = None;
        word_fault = None;
        interrupt_next = false;
        stalls = 0;
        gated = false;
        sleepers = 0;
        stop_fault = None;
      }
    in
    if runs = `Itself then polled_start d.c;
    d

  let open_ ?capacity ?copies ?host_visible ?transport ?peers ?maps_host ?budget
      ?memory ?window ?may_block ?completion ?waits_on ?max_waits ?answer ?runs
      ?lag ?hang_ms name =
    let p =
      make ?capacity ?copies ?host_visible ?transport ?peers ?maps_host ?budget
        ?memory ?window ?may_block ?completion ?waits_on ?max_waits ?answer
        ?runs ?lag ?hang_ms ()
    in
    match Rig.open_ (module Driver) ~name (fun () -> Ok p) with
    | Ok d -> (d, p)
    | Error e -> failwith e

  let run d = polled_run d.c
  let queued d = polled_queued d.c
  let submits d = polled_submits d.c
  let fail d = polled_fail d.c
  let fail_commit d = polled_fail_commit d.c
  let fault d why = Mutex.protect d.lock (fun () -> d.fault <- Some why)

  let fail_at d n how =
    match how with
    | `Fault why ->
        Mutex.protect d.lock (fun () -> d.failure <- why);
        polled_fail_at d.c n (-1) why
    | `Refuse k ->
        if k < 0 then invalid_arg "Polled.fail_at: negative count";
        polled_fail_at d.c n k ""

  let steps d = polled_steps d.c

  let outstanding d =
    Mutex.protect d.lock (fun () -> List.map snd d.held @ d.mapped)

  let fault_word d why =
    Mutex.protect d.lock (fun () -> d.word_fault <- Some why)

  let set_word d v = polled_set_word d.c v
  let word_at d = d.word_at
  let stop_fault d = Mutex.protect d.lock (fun () -> d.stop_fault)
  let blocked d = polled_blocked d.c

  let last_waits d =
    let a = polled_last_waits d.c in
    List.init
      (Array.length a / 3)
      (fun i -> (a.(3 * i), a.((3 * i) + 1), a.((3 * i) + 2)))

  let last_handles d = Array.to_list (polled_last_handles d.c)

  let copy_sides d =
    let none, src, dst = rig_local () in
    let side k =
      if k = none then `None
      else if k = src then `Src
      else if k = dst then `Dst
      else invalid_arg "Polled.copy_sides: an unknown copy_local"
    in
    List.map side (Array.to_list (polled_copy_sides d.c))

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

(* Census *)

let heap_bytes () = match heap_bytes () with -1 -> None | n -> Some n

let descriptors () =
  if Sys.win32 then None else Some (Array.length (Sys.readdir "/dev/fd"))

(* Waiting for a signal *)

let watchdog_s = 10.

let await what f =
  let until = Unix.gettimeofday () +. watchdog_s in
  while not (f ()) do
    if Unix.gettimeofday () > until then
      failwith (Printf.sprintf "await: no %s after %.0f s" what watchdog_s);
    Thread.yield ()
  done

(* Io devices and machines *)

module Empty = struct
  type t = unit
  type region = unit

  exception Fault of string

  let region_key : region Type.Id.t = Type.Id.make ()
  let budget () = max_int
  let alloc () _ = Some ()
  let free () () = ()
  let read () () ~at:_ ~dst:_ ~len:_ = ()
  let write () () ~at:_ ~src:_ ~len:_ = ()
  let pages () () = None
  let prefetch () () ~at:_ ~len:_ = ()
  let stop () = ()
end

let io name =
  match Rig.open_io (module Empty) ~name (fun () -> Ok ()) with
  | Ok d -> d
  | Error e -> failwith e

let machine m =
  let make () =
    Ok (Polled.make ~host_visible:false ~peers:false ~maps_host:false ())
  in
  match Rig.open_host (module Polled) ~machine:m ~name:"CPU" make with
  | Ok d -> d
  | Error e -> failwith e

(* Readers *)

module Reader = struct
  external host : Rig.Buffer.t -> int = "rig_test_reader_host"
  external bytes : Rig.Buffer.t -> int = "rig_test_reader_bytes"
  external why : Rig.Buffer.t -> string option = "rig_test_reader_why"

  type answer = Claimed | Wait | Dead | Exclusive | Read_only

  let pp_answer ppf a =
    Format.pp_print_string ppf
      (match a with
      | Claimed -> "Claimed"
      | Wait -> "Wait"
      | Dead -> "Dead"
      | Exclusive -> "Exclusive"
      | Read_only -> "Read_only")

  external claim : Rig.Buffer.t -> Rig.Buffer.access -> answer
    = "rig_test_reader_claim"
  [@@noalloc]

  external wait : Rig.Buffer.t -> Rig.Buffer.access -> unit
    = "rig_test_reader_wait"

  external release : Rig.Buffer.t -> unit = "rig_test_reader_release"
  [@@noalloc]
end
