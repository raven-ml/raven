(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type polled

external polled_new : int -> bool -> int -> polled = "rig_test_polled_new"
external polled_edge : polled -> nativeint = "rig_test_polled_edge"
external polled_grows : polled -> int -> unit = "rig_test_polled_grows"
external polled_grow : polled -> bool -> bool = "rig_test_polled_grow"
external polled_fail : polled -> unit = "rig_test_polled_fail"
external polled_fail_commit : polled -> unit = "rig_test_polled_fail_commit"
external polled_fail_settle : polled -> unit = "rig_test_polled_fail_settle"
external polled_run : polled -> int = "rig_test_polled_run"
external polled_drive : polled -> int = "rig_test_polled_drive"
external polled_start : polled -> unit = "rig_test_polled_start"
external polled_queued : polled -> int = "rig_test_polled_queued"
external polled_submits : polled -> int = "rig_test_polled_submits"
external polled_stopping : polled -> unit = "rig_test_polled_stopping"
[@@noalloc]
external polled_stops : polled -> int = "rig_test_polled_stops" [@@noalloc]
external polled_blocked : polled -> int = "rig_test_polled_blocked"
external polled_step : polled -> int = "rig_test_polled_step" [@@noalloc]
external polled_steps : polled -> int = "rig_test_polled_steps"

external polled_fail_at : polled -> int -> int -> string -> unit
  = "rig_test_polled_fail_at"

external polled_last_times : polled -> int = "rig_test_polled_last_times"
external polled_last_waits : polled -> int array
  = "rig_test_polled_last_waits"

external polled_layout : polled -> int array = "rig_test_polled_layout"

external polled_copy_facts : polled -> (int * bool) array
  = "rig_test_polled_copy_facts"

external polled_kernel : string -> nativeint = "rig_test_polled_kernel"

external polled_launches : polled -> string array
  = "rig_test_polled_launches"

external rig_local : unit -> int * int * int = "rig_test_rig_local"
external polled_word_at : polled -> int = "rig_test_polled_word_at"
external polled_word : polled -> int = "rig_test_polled_word"
external polled_stop : polled -> unit = "rig_test_polled_stop"
external polled_set_word : polled -> int -> unit = "rig_test_polled_set_word"
external polled_free_word : polled -> int = "rig_test_polled_free_word"


external polled_mapped : polled -> int -> int -> unit
  = "rig_test_polled_mapped"

external polled_unmapped : polled -> int -> int -> unit
  = "rig_test_polled_unmapped"

external polled_gate_room : polled -> bool -> unit = "rig_test_polled_gate_room"
external polled_room_waiters : polled -> int = "rig_test_polled_room_waiters"
external polled_gate_settle : polled -> bool -> unit
  = "rig_test_polled_gate_settle"

external polled_settle_waiters : polled -> int
  = "rig_test_polled_settle_waiters"
external polled_broke : int -> unit = "rig_test_polled_broke"
external polled_broken : unit -> int array = "rig_test_polled_broken"
external polled_bind : polled -> nativeint -> int -> bool -> unit
  = "rig_test_polled_bind"
external host_alloc : int -> int = "rig_test_alloc"
external host_free : int -> unit = "rig_test_free"
external noop : unit -> nativeint = "rig_test_noop"
external bump : unit -> nativeint = "rig_test_bump"
external waiting : unit -> nativeint = "rig_test_waiting"
external poke : unit -> nativeint = "rig_test_poke"
external far : unit -> nativeint = "rig_test_far"
external carry : unit -> nativeint = "rig_test_carry"
external countdown : unit -> nativeint = "rig_test_countdown"
external ready : unit -> nativeint = "rig_test_ready"
external interrupt : unit -> unit = "rig_test_interrupt" [@@noalloc]
external interrupt_after_ms : int -> unit = "rig_test_interrupt_after"
external interrupt_join : unit -> unit = "rig_test_interrupt_join"
external park : unit -> unit = "rig_test_park"
external parked : unit -> bool = "rig_test_parked"
external unpark : unit -> unit = "rig_test_unpark"
external sees_waits : unit -> bool = "rig_test_sees_waits"
external interrupter_start : unit -> unit = "rig_test_interrupter_start"
external interrupter_stop : unit -> bool = "rig_test_interrupter_stop"
external thread_self : unit -> int = "rig_test_thread_self"
external waits_for_lock : int -> bool = "rig_test_waits_for_lock"
external waits_in_condition : int -> bool = "rig_test_waits_in_condition"
external set_address_space : int -> int = "rig_test_set_address_space"

external shares : ('a, 'b, 'c) Bigarray.Array1.t -> int = "rig_test_shares"
[@@noalloc]

external host_held : unit -> int = "rig_test_host_held"
external heap_bytes : unit -> int = "rig_test_heap_bytes"
external gcc_asan : unit -> bool = "rig_test_gcc_asan" [@@noalloc]
external load : int -> int = "rig_test_load"
external store : int -> int -> unit = "rig_test_store"

external load64 :
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t ->
  int ->
  int64 = "rig_test_load64"
external move : dst:int -> src:int -> int -> unit = "rig_test_move"

let noop = noop ()
let bump = bump ()
let waiting = waiting ()
let poke = poke ()
let far = far ()
let carry = carry ()
let countdown = countdown ()
let ready = ready ()

(* The locks the calling thread holds, and a count of one more and one fewer
   ([rig_holds], [rig_hold] and [rig_unhold] in [rig_edge.h]). *)
external holds : unit -> int = "caml_rig_edge_holds_byte"
external hold : unit -> unit = "caml_rig_edge_hold" [@@noalloc]
external unhold : unit -> unit = "caml_rig_edge_unhold" [@@noalloc]

module Driver = struct
  type kind = Rig_edge.memory

  (* Locks

     A device's lock counts among the locks its thread holds, as a driver's
     must (Rig_edge's drivers section), so rig refuses a callback's call
     inside it. No section takes a second lock. *)

  external reraise : exn -> 'a = "%reraise"

  (* [take m] takes [m], counted from before the take, so that a signal handler
     that runs inside a contended take sees it held. An exception that ends the
     take gives the count back first, having taken nothing: one a handler
     raised, or the error-checking [Mutex]'s [Sys_error] for a take by the
     thread that holds [m], which only a call that skipped its check makes and
     which raises [Invalid_argument]. *)
  let take m =
    hold ();
    match Mutex.lock m with
    | () -> ()
    | exception e -> (
        unhold ();
        match e with
        | Sys_error why when String.starts_with ~prefix:"Mutex.lock:" why ->
            invalid_arg
              "Polled: a lock taken again by the thread that holds it, from a \
               signal handler, finaliser or Gc.Memprof callback"
        | e -> reraise e)

  (* Gives [m] back, then its count, with nothing between that allocates or
     polls. *)
  let give m =
    Mutex.unlock m;
    unhold ()

  let protect m f =
    take m;
    match f () with
    | x ->
        give m;
        x
    | exception e ->
        give m;
        reraise e

  let rec push a x =
    let l = Atomic.get a in
    if not (Atomic.compare_and_set a l (x :: l)) then push a x

  type t = {
    c : polled;
    mutable word_at : int;  (** The word's host address. *)
    most_parts : int;  (** The parts its facts state an idle queue runs. *)
    copies : bool;
    host_visible : bool;
    addresses : bool;
    transport : bool;
    peers : bool;
    maps_host : bool;
    alignment : int;
    budget : int;
    limits : kind -> int;
    may_block : bool;
    stores : bool;
    waits : [ `Store | `Host ] list;
    max_waits : int;
    hang_ms : int option;
    answer : [ `Stopped | `Unknown ];
    capability : Rig_edge.capability;
    lock : Mutex.t;
    opened : Condition.t;
    mutable calls : string list;
    mutable sleeps : int;
    frees : (int * int) list Atomic.t;
    mutable allocs : (kind * int * bool) list;
    mutable maps : int list;
    mutable held : (kind * int) list;
    mutable mapped : int list;
    failure : string Atomic.t;
    mutable fault : string option;
    word_fault : string option Atomic.t;
    word_break : bool Atomic.t;
    mutable interrupt_next : bool;
    mutable stalls : int;
    mutable gated : bool;
    mutable sleepers : int;
    mutable frees_gated : bool;
    mutable freers : int;
    mutable locates_gated : bool;
    mutable locaters : int;
    mutable stop_fault : string option;  (** What its stop was given. *)
    before : (string * (unit -> unit)) list Atomic.t;
        (** Functions to run at the next call of each name. *)
  }

  (* A region the driver allocated has a kind; a mapping has none, and names
     the device that made it; the word has neither. *)
  type region = {
    at : int;
    kind : kind option;
    bytes : int;
    visible : bool;
    addressed : bool;
    mapper : t option;
  }

  (* An image's code is in a region of the device, or, loaded by the driver,
     nowhere: its functions are the host's. *)
  type image = { code : region option; owner : t }

  exception Fault of string

  let note d call = protect d.lock (fun () -> d.calls <- call :: d.calls)
  let log d = protect d.lock (fun () -> List.rev d.calls)
  let sleeps d = protect d.lock (fun () -> d.sleeps)
  let key : t Type.Id.t = Type.Id.make ()

  (* Counts a fallible call: the call [fail_at] fails raises, and the one it
     refuses answers [true]. [d]'s lock is not held. *)
  let countdown d =
    match polled_step d.c with
    | 2 -> raise (Fault (Atomic.get d.failure))
    | s -> s = 1

  (* A counted call or a fact: a faulted device's raises its fault. *)
  let step d =
    match protect d.lock (fun () -> d.fault) with
    | Some why -> raise (Fault why)
    | None -> countdown d

  let capability_key : unit Type.Id.t = Type.Id.make ()
  let entry_scratch = 1 lsl 16

  (* Its queues run fills and, with copies, copies; its compute queue also runs
     launches and words. The facts are one call, which goes on whether or not
     it refuses. *)
  let facts d =
    ignore (step d : bool);
    let runs = Rig_edge.(if d.copies then [ Fill; Copy ] else [ Fill ]) in
    let compute =
      { Rig_edge.name = "COMPUTE:0"; runs = Launch :: Words :: runs }
    in
    let has c = List.mem c d.waits in
    {
      Rig_edge.arch = "polled";
      budget = d.budget;
      queues =
        (if d.copies then [ compute; { name = "COPY:0"; runs } ]
         else [ compute ]);
      copy_engine = (if d.copies then Some 1 else None);
      completion = (if d.stores then Store else Host);
      waits = { stores = has `Store; hosts = has `Host; most = d.max_waits };
      most_parts = d.most_parts;
      may_block = d.may_block;
      hang_ms = d.hang_ms;
      maps_host = d.maps_host;
      host_addresses = d.host_visible && not d.transport;
      alignment = d.alignment;
      capability = d.capability;
      word =
        {
          at = d.word_at;
          kind = None;
          bytes = 8;
          visible = not d.transport;
          addressed = true;
          mapper = None;
        };
      edge = polled_edge d.c;
    }

  (* Runs the functions {!before} set for [call], taking no lock. *)
  let rec run_before d call =
    let all = Atomic.get d.before in
    match List.partition (fun (c, _) -> c = call) all with
    | [], _ -> ()
    | mine, rest ->
        if Atomic.compare_and_set d.before all rest then
          List.iter (fun (_, f) -> f ()) mine
        else run_before d call

  let counted d call =
    note d call;
    run_before d call;
    step d

  let holding d kind =
    List.fold_left (fun n (k, b) -> if k = kind then n + b else n) 0 d.held

  let alloc d kind n =
    let refused = counted d "alloc" in
    let fits =
      protect d.lock (fun () ->
          let fits = (not refused) && holding d kind + n <= d.limits kind in
          d.allocs <- (kind, n, fits) :: d.allocs;
          if fits then d.held <- (kind, n) :: d.held;
          fits)
    in
    if not fits then None
    else
      let at = host_alloc n in
      let visible = d.host_visible || kind <> Device in
      let addressed = d.addresses || kind <> Device in
      Some
        { at; kind = Some kind; bytes = n; visible; addressed; mapper = None }

  let rec remove x = function
    | [] -> []
    | y :: l -> if y = x then l else y :: remove x l

  (* What rig must not do, in [rig_support_stubs.c]'s order of [BROKE_*]. *)
  let broken =
    [|
      "Polled: a freed word was read";
      "Polled: a word was freed twice";
      "Polled: a wait read a mapping rig gave back";
      "Polled: rig freed a word in its array";
      "Polled: a time pair went to a value before the one it timed settled";
    |]

  let read_freed = 0

  let broke what =
    polled_broke what;
    failwith broken.(what)

  let violations () =
    Array.to_list (Array.map (fun w -> broken.(w)) (polled_broken ()))

  let is_word r = r.kind = None && r.mapper = None

  (* rig frees only a word outside its array, which is the driver's own,
     once, after which a read of it breaks rig's contract. *)
  let free_word d r =
    note d "free";
    let w = polled_word d.c in
    push d.frees (r.at, w);
    (match polled_free_word d.c with -1 -> () | what -> broke what);
    run_before d "free"

  (* A free waits at the free gate while it is shut. A mapping's is logged as
     ["unmap"]. *)
  let free_region d r =
    protect d.lock (fun () ->
        if d.frees_gated then begin
          d.freers <- d.freers + 1;
          while d.frees_gated do
            Condition.wait d.opened d.lock
          done;
          d.freers <- d.freers - 1
        end);
    let call = if r.kind = None then "unmap" else "free" in
    note d call;
    run_before d call;
    let w = polled_word d.c in
    push d.frees (r.at, w);
    protect d.lock (fun () ->
        match r.kind with
        | Some k -> d.held <- remove (k, r.bytes) d.held
        | None -> d.mapped <- remove r.bytes d.mapped);
    if r.kind <> None then host_free r.at
    else polled_unmapped d.c r.at r.bytes

  (* A faulted device's free gives the region back, then raises. *)
  let free d r =
    if is_word r then free_word d r else free_region d r;
    match protect d.lock (fun () -> d.fault) with
    | Some why -> raise (Fault why)
    | None -> ()

  (* A locate of a mapping waits at its mapper's locate gate while it is
     shut. *)
  let locate r =
    Option.iter
      (fun d ->
        protect d.lock (fun () ->
            if d.locates_gated then begin
              d.locaters <- d.locaters + 1;
              while d.locates_gated do
                Condition.wait d.opened d.lock
              done;
              d.locaters <- d.locaters - 1
            end))
      r.mapper;
    {
      Rig_edge.address = (if r.addressed then Some r.at else None);
      host = (if r.visible then Some r.at else None);
      handle = Nativeint.of_int r.at;
    }

  let peer d _ = d.peers

  let mapping d m =
    protect d.lock (fun () -> d.mapped <- m.bytes :: d.mapped);
    polled_mapped d.c m.at m.bytes;
    Some m

  let map_peer d _ r =
    if counted d "map_peer" || not d.peers then None
    else mapping d { r with kind = None; mapper = Some d }

  let map_host d p n =
    if counted d "map_host" then None
    else begin
      protect d.lock (fun () -> d.maps <- n :: d.maps);
      mapping d
        {
          at = p;
          kind = None;
          bytes = n;
          visible = true;
          addressed = true;
          mapper = Some d;
        }
    end

  let image d b =
    if counted d "image" then Error "refused"
    else
      match String.split_on_char ':' b with
      | [ "code"; n ] ->
          let n = int_of_string n in
          Ok
            (Rig_edge.Place
               ( n,
                 fun r ->
                   run_before d "place";
                   ({ code = Some r; owner = d }, String.make n 'c') ))
      | [ "functions" ] -> Ok (Rig_edge.Loaded { code = None; owner = d })
      | _ -> Error "not a polled binary"

  (* Its functions are Polled's host functions of that name. *)
  let entry i f =
    let launch = polled_kernel f in
    if counted i.owner "entry" then Rig_edge.No_memory entry_scratch
    else if launch = 0n then Absent
    else
      let code = match i.code with Some r -> r.at | None -> 0 in
      let params =
        match f with "fill" -> Some 16 | "copy" -> Some 24 | _ -> None
      in
      Entry { code; launch; params; threads = Some 1024 }

  let unload d _ = ignore (counted d "unload" : bool)

  let grow d =
    let refused = counted d "grow" in
    if polled_grow d.c refused then Rig_edge.Grown else Short 4096

  let rec freed_at a = function
    | [] -> false
    | (x, _) :: l -> x = a || freed_at a l

  (* Reads the fault and the frees without a lock, as a driver's [signaled]
     must, since rig calls it under a lock of its own, and with no closure, so
     that a read of the word allocates nothing. *)
  let signaled d =
    let fault = Atomic.get d.word_fault in
    let break = Atomic.exchange d.word_break false in
    let freed = freed_at d.word_at (Atomic.get d.frees) in
    let hooked = Atomic.get d.before != [] in
    if hooked then run_before d "signaled";
    if freed then broke read_freed;
    if break then raise Sys.Break;
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
    | None when polled_step d.c = 2 -> `Fault (Atomic.get d.failure)
    | None when d.interrupt_next ->
        d.interrupt_next <- false;
        `Interrupt
    | None when d.stalls > 0 ->
        d.stalls <- d.stalls - 1;
        `Stall
    | None -> `Run

  (* Consecutive sleeps are one entry of the log: two domains' waits sleep
     without bound, and a device the program keeps would hold every one.
     With no function set by {!before}, a sleep counts and steps under one
     lock, as bench rows time rig's waits through it. *)
  let sleep d ~seen ~still_ms =
    let step =
      protect d.lock (fun () ->
          d.sleeps <- d.sleeps + 1;
          (match d.calls with
          | "sleep" :: _ -> ()
          | calls -> d.calls <- "sleep" :: calls);
          if Atomic.get d.before == [] then next d else `Hooked)
    in
    let step =
      match step with
      | `Hooked ->
          run_before d "sleep";
          protect d.lock (fun () -> next d)
      | (`Fault _ | `Interrupt | `Stall | `Run) as step -> step
    in
    match step with
    | `Fault why -> raise (Fault why)
    | `Interrupt -> interrupt ()
    | `Stall -> Thread.delay (float still_ms /. 1000.)
    | `Run ->
        if polled_drive d.c < 0 && polled_word d.c = seen then
          failwith "Polled: nothing committed"

  let stop d ~fault =
    polled_stopping d.c;
    note d "stop";
    run_before d "stop";
    protect d.lock (fun () -> d.stop_fault <- fault);
    protect d.lock (fun () -> at_gate d);
    if d.answer = `Stopped then polled_stop d.c
end

module Polled = struct
  include Driver

  let make ?(capacity = 1024) ?grows ?most_parts ?(copies = true)
      ?(host_visible = true) ?(addresses = true) ?(transport = false)
      ?(peers = true) ?(maps_host = true) ?(alignment = 4096)
      ?(budget = 1 lsl 30) ?(memory = max_int) ?(window = max_int)
      ?(may_block = false) ?(completion = `Host) ?(waits_on = [])
      ?(max_waits = max_int) ?(answer = `Stopped) ?(runs = `When_slept)
      ?(lag = 1) ?hang_ms
      ?(capability = Rig_edge.Capability (capability_key, ())) () =
    if lag < 1 then invalid_arg "Polled.make: lag is below 1";
    let most_parts = Option.value most_parts ~default:capacity in
    if most_parts > capacity then
      invalid_arg "Polled.make: most_parts is above capacity";
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
        most_parts;
        copies;
        host_visible;
        addresses;
        transport;
        peers;
        maps_host;
        alignment;
        budget;
        limits;
        may_block;
        stores = completion = `Store;
        waits = waits_on;
        max_waits;
        hang_ms;
        answer;
        capability;
        lock = Mutex.create ();
        opened = Condition.create ();
        calls = [];
        sleeps = 0;
        frees = Atomic.make [];
        allocs = [];
        maps = [];
        held = [];
        mapped = [];
        failure = Atomic.make "";
        fault = None;
        word_fault = Atomic.make None;
        word_break = Atomic.make false;
        interrupt_next = false;
        stalls = 0;
        gated = false;
        sleepers = 0;
        frees_gated = false;
        freers = 0;
        locates_gated = false;
        locaters = 0;
        stop_fault = None;
        before = Atomic.make [];
      }
    in
    if runs = `Itself then polled_start d.c;
    Option.iter (polled_grows d.c) grows;
    d

  let bind ?(start = 0) d ~word =
    polled_bind d.c word start d.transport;
    d.word_at <- polled_word_at d.c;
    d

  let opener ?start name d =
    {
      Rig_edge.name;
      driver = (module Driver : Rig_edge.Driver with type t = t);
      make = (fun ~word -> Ok (bind ?start d ~word));
    }

  let open_ ?capacity ?grows ?most_parts ?copies ?host_visible ?addresses
      ?transport ?peers ?maps_host ?alignment ?budget ?memory ?window ?may_block
      ?completion ?waits_on ?max_waits ?answer ?runs ?lag ?hang_ms name =
    let p =
      make ?capacity ?grows ?most_parts ?copies ?host_visible ?addresses
        ?transport ?peers ?maps_host ?alignment ?budget ?memory ?window
        ?may_block ?completion ?waits_on ?max_waits ?answer ?runs ?lag ?hang_ms
        ()
    in
    match Rig.open_ (opener name p) with
    | Ok d -> (d, p)
    | Error e -> failwith e

  let run d = polled_run d.c
  let queued d = polled_queued d.c
  let submits d = polled_submits d.c
  let fail d = polled_fail d.c
  let fail_commit d = polled_fail_commit d.c
  let fail_settle d = polled_fail_settle d.c
  let fault d why = protect d.lock (fun () -> d.fault <- Some why)

  let before d call f = push d.before (call, f)

  let fail_at d n how =
    match how with
    | `Fault why ->
        Atomic.set d.failure why;
        polled_fail_at d.c n (-1) why
    | `Refuse k ->
        if k < 0 then invalid_arg "Polled.fail_at: negative count";
        polled_fail_at d.c n k ""

  let steps d = polled_steps d.c

  let outstanding d =
    protect d.lock (fun () -> List.map snd d.held @ d.mapped)

  let fault_word d why = Atomic.set d.word_fault (Some why)
  let break_word d = Atomic.set d.word_break true
  let set_word d v = polled_set_word d.c v
  let word_at d = d.word_at
  let stop_fault d = protect d.lock (fun () -> d.stop_fault)
  let stops d = polled_stops d.c
  let blocked d = polled_blocked d.c

  let last_times d = polled_last_times d.c

  let last_waits d =
    let a = polled_last_waits d.c in
    List.init (Array.length a / 2) (fun i -> (a.(2 * i), a.((2 * i) + 1)))


  let layout d =
    let a = polled_layout d.c in
    let rec parts i =
      if i >= Array.length a then []
      else
        let n = a.(i + 1) in
        (a.(i), Array.to_list (Array.sub a (i + 2) n)) :: parts (i + 2 + n)
    in
    parts 0

  let copy_sides d =
    let none, src, dst = rig_local () in
    let side k =
      if k = none then `None
      else if k = src then `Src
      else if k = dst then `Dst
      else invalid_arg "Polled.copy_sides: an unknown copy_local"
    in
    List.map (fun (s, _) -> side s) (Array.to_list (polled_copy_facts d.c))

  let copy_device_memory d =
    List.map snd (Array.to_list (polled_copy_facts d.c))

  type launch = {
    groups : int * int * int;
    threads : int * int * int;
    shared : int;
    params : string;
  }

  (* Each block: [rig_edge.h]'s struct rig_block, 32 bytes of header, then the
     parameters. *)
  let launch b =
    let u32 at = Int32.to_int (String.get_int32_le b at) land 0xffff_ffff in
    let axes k = (u32 k, u32 (k + 4), u32 (k + 8)) in
    {
      groups = axes 0;
      threads = axes 12;
      shared = u32 24;
      params = String.sub b 32 (String.length b - 32);
    }

  let launches d = Array.to_list (Array.map launch (polled_launches d.c))
  let frees d = List.rev (Atomic.get d.frees)

  let word_frees d =
    List.length (List.filter (fun (a, _) -> a = d.word_at) (frees d))
  let allocs d = protect d.lock (fun () -> List.rev d.allocs)
  let host_maps d = protect d.lock (fun () -> List.rev d.maps)
  let allocated d kind = protect d.lock (fun () -> holding d kind)
  let interrupt d = protect d.lock (fun () -> d.interrupt_next <- true)
  let stall d n = protect d.lock (fun () -> d.stalls <- n)
  let gate d = protect d.lock (fun () -> d.gated <- true)
  let sleepers d = protect d.lock (fun () -> d.sleepers)

  let open_gate d =
    protect d.lock (fun () ->
        d.gated <- false;
        Condition.broadcast d.opened)

  let gate_frees d = protect d.lock (fun () -> d.frees_gated <- true)
  let freers d = protect d.lock (fun () -> d.freers)

  let open_frees d =
    protect d.lock (fun () ->
        d.frees_gated <- false;
        Condition.broadcast d.opened)

  let gate_locates d =
    protect d.lock (fun () -> d.locates_gated <- true)

  let gate_room d =
    if not d.may_block then
      invalid_arg "Polled.gate_room: a device that does not block";
    polled_gate_room d.c true

  let open_room d = polled_gate_room d.c false
  let in_room d = polled_room_waiters d.c

  let gate_settle d =
    if not d.may_block then
      invalid_arg "Polled.gate_settle: a device that does not block";
    polled_gate_settle d.c true

  let open_settle d = polled_gate_settle d.c false
  let settlers d = polled_settle_waiters d.c

  let locaters d = protect d.lock (fun () -> d.locaters)

  let open_locates d =
    protect d.lock (fun () ->
        d.locates_gated <- false;
        Condition.broadcast d.opened)
end

(* Census *)

let heap_bytes () = match heap_bytes () with -1 -> None | n -> Some n

let descriptors () =
  if Sys.win32 then None else Some (Array.length (Sys.readdir "/dev/fd"))

(* Forks *)

(* A process whose threads it cannot count may run several. *)
let threaded () =
  match Sys.readdir "/proc/self/task" with
  | tasks -> Array.length tasks > 1
  | exception Sys_error _ -> true

let needs_fork () =
  if Sys.win32 then Windtrap.skip ~reason:"Windows has no fork" ();
  if gcc_asan () && threaded () then
    Windtrap.skip
      ~reason:"GCC's AddressSanitizer holds no allocator lock across fork" ()

external recv_timeout : Unix.file_descr -> float -> unit
  = "rig_test_recv_timeout"

external descriptor_limit : unit -> int = "rig_test_descriptor_limit"

external set_descriptor_limit : int -> int = "rig_test_set_descriptor_limit"

external resident : unit -> int = "rig_test_resident"

(* Waiting for a signal *)

let watchdog_s = 10.

let sees_waits = sees_waits ()

type thread = int

let self = thread_self
let waits_for_lock = waits_for_lock
let waits_in_condition = waits_in_condition
let interrupt_after s = interrupt_after_ms (int_of_float (s *. 1000.))

let interrupt_in_wait f =
  interrupter_start ();
  let raised = ref false in
  let r = Fun.protect ~finally:(fun () -> raised := interrupter_stop ()) f in
  (r, !raised)

type landing = Raise | Signal | Call of (unit -> unit)

(* Only the calling thread's allocations count, from [f]'s start: callbacks
   the profile's start deferred run in a [Gc.minor] before. Once [f] ends, a
   [Gc.minor] lands a signal still pending, before the profile stops, so
   nothing of the walk's own meets the landing. *)
let interrupted ?(from = fun _ -> true) ?(landing = Raise) n f =
  let me = Thread.id (Thread.self ()) in
  let left = ref n and armed = ref false in
  let alloc a =
    if !armed && Thread.id (Thread.self ()) = me && from a then begin
      decr left;
      match landing with
      | Raise -> if !left <= 0 then raise Exit
      | Signal -> if !left = 0 then interrupt ()
      | Call g -> if !left = 0 then g ()
    end;
    None
  in
  let tracker =
    { Gc.Memprof.null_tracker with alloc_minor = alloc; alloc_major = alloc }
  in
  let breaks = match landing with Signal -> true | Raise | Call _ -> false in
  let previous =
    if not breaks then Sys.Signal_default
    else Sys.signal Sys.sigint (Sys.Signal_handle (fun _ -> raise Sys.Break))
  in
  let ended profile =
    armed := false;
    (try Gc.minor () with Sys.Break -> ());
    Gc.Memprof.stop ();
    Gc.Memprof.discard profile;
    if breaks then Sys.set_signal Sys.sigint previous
  in
  let profile = Gc.Memprof.start ~sampling_rate:1. tracker in
  Gc.minor ();
  armed := true;
  match f () with
  | () ->
      ended profile;
      !left <= 0
  | exception (Exit | Sys.Break) when !left <= 0 ->
      ended profile;
      true
  | exception e ->
      ended profile;
      raise e

(* Calls inside a lock *)

let inside_lock f =
  hold ();
  Fun.protect ~finally:unhold f

type seen = { mutable inside : int; mutable between : int }

let seen () = { inside = 0; between = 0 }

(* Whether the thread held a lock as the callback ran, what the call
   answered, and the change in the locks it holds across the call. *)
type landed = (bool * string * int) option ref

let landing seen prefix call =
  let landed = ref None in
  let call () =
    let before = holds () in
    if before > 0 then seen.inside <- seen.inside + 1
    else seen.between <- seen.between + 1;
    let got =
      match call () with
      | () -> "returned"
      | exception Invalid_argument why when String.starts_with ~prefix why ->
          "refused"
      | exception e -> Printexc.to_string e
    in
    landed := Some (before > 0, got, holds () - before)
  in
  (Call call, landed)

let answered what (landed : landed) =
  match !landed with
  | None -> None
  | Some (held, got, drift) ->
      let where = if held then "inside a lock" else "between locks" in
      let expected = if held then "refused" else "returned" in
      Windtrap.equal
        ~msg:(Printf.sprintf "%s, a call %s" what where)
        Windtrap.string expected got;
      Windtrap.equal
        ~msg:(Printf.sprintf "%s, the locks held across the call" what)
        Windtrap.int 0 drift;
      Some held

let covered seen =
  Windtrap.greater ~msg:"landings inside a lock" Windtrap.int ~than:0
    seen.inside;
  Windtrap.greater ~msg:"landings between locks" Windtrap.int ~than:0
    seen.between

let time_scale =
  match Sys.getenv_opt "RIG_TEST_TIME_SCALE" with
  | None -> 1.
  | Some s -> (
      match float_of_string_opt s with
      | Some k when k >= 1. -> k
      | _ -> invalid_arg ("RIG_TEST_TIME_SCALE=" ^ s ^ ": not a factor >= 1"))

let time s = s *. time_scale

let await what f =
  let until = Unix.gettimeofday () +. watchdog_s in
  while not (f ()) do
    if Unix.gettimeofday () > until then
      failwith (Printf.sprintf "await: no %s after %.0f s" what watchdog_s);
    Thread.yield ()
  done

(* Each count of the heap allocates its own result, the same bytes each time:
   the second count measures the first, and the third is less by it. *)
let allocated_bytes f =
  let b0 = Gc.allocated_bytes () in
  let b1 = Gc.allocated_bytes () in
  f ();
  let b2 = Gc.allocated_bytes () in
  int_of_float (b2 -. b1 -. (b1 -. b0))

let donated b =
  Rig.Claim.with_ ~read:[] ~donate:[ b ] (fun c ->
      if Rig.Claim.exclusive c b then
        Some (Rig.Claim.consume c ~why:"donated" b)
      else None)

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
  match Rig.open_io { name; io = (module Empty); make = (fun () -> Ok ()) } with
  | Ok d -> d
  | Error e -> failwith e

let machine m =
  let make ~word =
    Ok
      (Polled.bind ~word
         (Polled.make ~host_visible:false ~peers:false ~maps_host:false ()))
  in
  match
    Rig.open_host ~machine:m { name = "CPU"; driver = (module Polled); make }
  with
  | Ok d -> d
  | Error e -> failwith e

(* Readers *)

module Reader = struct
  external host : Rig.Buffer.t -> int = "rig_test_reader_host"
  external bytes : Rig.Buffer.t -> int = "rig_test_reader_bytes"
  external why : Rig.Buffer.t -> string option = "rig_test_reader_why"

  type answer = Claimed | Wait | Lost | Dead | Exclusive | Read_only

  let pp_answer ppf a =
    Format.pp_print_string ppf
      (match a with
      | Claimed -> "Claimed"
      | Wait -> "Wait"
      | Lost -> "Lost"
      | Dead -> "Dead"
      | Exclusive -> "Exclusive"
      | Read_only -> "Read_only")

  external claim : Rig.Buffer.t -> Rig.access -> answer
    = "rig_test_reader_claim"
  [@@noalloc]

  external wait : Rig.Buffer.t -> Rig.access -> unit
    = "rig_test_reader_wait"

  external release : Rig.Buffer.t -> unit = "rig_test_reader_release"
  [@@noalloc]

  type outcome = All of answer array | Refused of int * answer

  external claim_all : Rig.Buffer.t array -> Rig.access array -> outcome
    = "rig_test_reader_claim_all"

  external release_all : Rig.Buffer.t array -> unit
    = "rig_test_reader_release_all"
  [@@noalloc]

  external span : Rig.Buffer.t -> int * int = "rig_test_reader_span"

  let read b =
    match claim b Rig.Read with
    | Claimed | Wait -> ()
    | Lost -> (
        match wait b Rig.Read with
        | () -> ()
        | exception e ->
            release b;
            raise e)
    | (Dead | Exclusive | Read_only) as a ->
        invalid_arg (Format.asprintf "a read claim answered %a" pp_answer a)
end

(* Telling memory apart *)

let at b =
  match Reader.host b with
  | 0 -> Nativeint.to_int (Rig.Buffer.handle b) + Rig.Buffer.offset b
  | h -> h

let touching uses =
  let refs = Array.mapi (fun i r -> r i ~at:(8 * i)) uses in
  Rig.Submission.fill ~refs noop ~params:(8 * Array.length uses)
