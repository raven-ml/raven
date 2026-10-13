(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Def

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

exception Lost of device * string
exception Out_of_memory of device * int

(* The C record *)

external c_new : int -> string -> bool -> nativeint -> nativeint -> c
  = "caml_rig_device_new"

external c_io_new : int -> string -> c = "caml_rig_io_new"
external c_host_new : string -> c = "caml_rig_host_new"
external c_host_word : unit -> int = "caml_rig_host_word"
external c_publish : c -> unit = "caml_rig_publish"
external c_committed : c -> int = "caml_rig_committed" [@@noalloc]
external c_commit : c -> int = "caml_rig_commit"
external c_settle : c -> int -> int = "caml_rig_settle"
external c_done : int -> bool = "caml_rig_done" [@@noalloc]

(* A device's line, read by its index for the life of the process. *)
external c_word : int -> int = "caml_rig_word" [@@noalloc]
external c_set_seen : int -> int -> unit = "caml_rig_set_seen" [@@noalloc]
external c_submitted : int -> int = "caml_rig_submitted" [@@noalloc]
external c_state : int -> int = "caml_rig_state" [@@noalloc]
external c_why : int -> string = "caml_rig_why"

(* The first point of the stamps [st] that a lost device did not reach, among
   their uses' if [every] and every device's last write's otherwise, [0] for
   none. *)
external c_stamps_lost : int -> bool -> int = "caml_rig_stamps_lost"
[@@noalloc]
external c_lose : c -> string -> bool -> bool = "caml_rig_lose"
external c_faulted : c -> bool = "caml_rig_faulted" [@@noalloc]
external c_stopped : c -> unit = "caml_rig_stopped"
external c_await_stop : c -> int -> bool = "caml_rig_await_stop"
external c_owed : unit -> int = "caml_rig_owed" [@@noalloc]
external c_claim : unit -> int = "caml_rig_claim"
external c_fail : string -> bool = "caml_rig_fail"
external c_failed : unit -> string option = "caml_rig_failed"
external c_first : unit -> int = "caml_rig_first" [@@noalloc]
external generation : unit -> int = "caml_rig_generation" [@@noalloc]
external c_enter : c -> int = "caml_rig_enter"
external c_exit : c -> unit = "caml_rig_exit"
external c_spin : c -> int -> int -> int = "caml_rig_spin"
external c_producers : c -> int array = "caml_rig_producers"
external c_waits_in_queue : c -> bool = "caml_rig_waits_in_queue"
external release_list : unit -> int = "caml_rig_release_list"
external c_release : c -> int = "caml_rig_device_release" [@@noalloc]
external host_arch : unit -> string = "caml_rig_arch"

(* [c_next i] is the least index from [i] on of a published device, 0 for
   none. *)
external c_next : int -> int = "caml_rig_next" [@@noalloc]

(* The C line's states: [rig_stubs.h]'s enum. *)
let live_state = 0
let ended_state = 4
let orphaned_state = 5

(* How long a wait sees no progress before it blocks in the driver, and the
   longest a blocking call of a wait lasts: each returns to OCaml within it,
   where a pending [Sys.Break] raises. *)
let still_ms = 200

(* How long a wait reads a word the device writes before it releases the
   domain lock to spin: past a submission's round trip on the GPUs measured
   (3.5 us on kimchi's RTX 5000), so that a wait for work about to complete
   costs no release; the domain's other threads wait at most this long. It
   polls, so other domains' collections never wait on it. *)
let held_ns = 4_000

(* Tables by device index *)

(* Values by device index, from 0 to [Point.max_index], read without a lock:
   chunks of [chunk] entries, each replaced whole when one of its entries is
   set. A set copies one chunk, however many devices the process made, and a
   chunk left with only [empty] entries goes, so a table holds chunks only
   around the indices it has values for. *)
module Table = struct
  let bits = 8
  let chunk = 1 lsl bits

  type 'a t = { empty : 'a; chunks : 'a array Atomic.t array }

  let make empty =
    let n = (Point.max_index lsr bits) + 1 in
    { empty; chunks = Array.init n (fun _ -> Atomic.make [||]) }

  let get t i =
    let c = Atomic.get t.chunks.(i lsr bits) in
    if Array.length c = 0 then t.empty else c.(i land (chunk - 1))

  let rec set t i v =
    let a = t.chunks.(i lsr bits) in
    let c = Atomic.get a in
    let c' =
      if Array.length c = 0 then Array.make chunk t.empty else Array.copy c
    in
    c'.(i land (chunk - 1)) <- v;
    let c' = if Array.for_all (fun x -> x == t.empty) c' then [||] else c' in
    if not (Atomic.compare_and_set a c c') then set t i v
end

module Cache = Tbl.Make (Int)
module Strings = Tbl.Make (String)

let no_waits = { Rig_edge.stores = false; hosts = false; objects = false; most = 0 }

let make_device ~index ~name ~machine ~kind ~c ~arch ~queues ~copy_engine
    ~completion ~waits ~hang_ms ~maps_host ~most_parts ~host_addresses ~word
    ~word_region ~key ~memory_device ~fault ~capability ~budget =
  let live =
    {
      c;
      queues;
      copy_engine;
      completion;
      waits;
      hang_ms;
      progress = { seen = 0; idle = true; since = 0 };
      word;
      word_region;
      fault;
      release = c_release c;
      lock = Lock.create ();
      used = 0;
      cached = 0;
      cache = Cache.create 16;
      retiring = [];
      pending = ([], []);
      pairs = [||];
      word_maps = [];
      mapped = Cache.create 4;
      afters = [];
    }
  in
  {
    index;
    name;
    machine;
    kind;
    arch;
    maps_host;
    most_parts;
    host_addresses;
    key;
    memory_device;
    capability;
    budget;
    closing = 0;
    live;
  }

(* The host runs launches of linked code ([rig_host_code.c]) on its one queue,
   and waits on the host for other devices' points. *)
let host =
  let queues = [| { Rig_edge.name = "COMPUTE:0"; runs = [ Launch ] } |] in
  let d =
    make_device ~index:0 ~name:"CPU" ~machine:Here ~kind:Host
      ~c:(c_host_new "CPU") ~arch:(host_arch ()) ~queues ~copy_engine:None
      ~completion:Host_writes ~waits:no_waits ~hang_ms:None ~maps_host:false
      ~most_parts:max_int ~host_addresses:false ~word:(c_host_word ())
      ~word_region:None ~key:(-1)
      ~memory_device:false
      ~fault:(fun _ -> None)
      ~capability:None ~budget:max_int
  in
  d

(* The table of devices *)

(* Every device by index, for the life of the process; an index whose open
   failed holds the host. *)
let devices = Table.make host

let of_index i = Table.get devices i
let register d = Table.set devices d.index d

let iter f =
  let rec from i =
    match c_next i with
    | 0 -> ()
    | i ->
        f (of_index i);
        from (i + 1)
  in
  from 1

let () =
  Printexc.register_printer (function
    | Lost (d, why) -> Some (strf "%s lost: %s" d.name why)
    | Out_of_memory (d, n) -> Some (strf "%s cannot allocate %d bytes" d.name n)
    | _ -> None)

(* Facts *)

let is_host d = match d.kind with Host -> true | _ -> false
let is_io d = match d.kind with Io _ -> true | _ -> false

(* Whether the device of index [i] is lost, read from its line alone. The host
   is never lost: a use of its memory reads no C line. *)
let[@inline] lost_at i = i <> 0 && c_state i <> live_state
let[@inline] is_lost d = lost_at d.index
let orphaned_at i = c_state i = orphaned_state
let orphaned d = orphaned_at d.index

let why d = c_why d.index
let lost d = if is_lost d then Some (why d) else None
let raise_lost d = raise (Lost (d, why d))

(* [d]'s loss as {!Lost} prints it. *)
let printed d = strf "%s lost: %s" d.name (why d)

let submitted d = c_submitted d.index
let committed d = c_committed d.live.c

(* Locks *)

let protect d f x = Lock.protect d.live.lock f x

(* The host of each other machine named so far: a machine has one for its
   life. *)
let machines : device Strings.t = Strings.create 4

(* The machines whose host an open is making, by the host's full name. *)
let hosting : string Strings.t = Strings.create 4

let table_lock = Lock.create ()

let host_of (d : device) =
  match d.machine with Here -> host | Hosts _ -> d | On h -> h
let same_machine d d' = host_of d == host_of d'

let machine_of = function
  | None -> Here
  | Some d ->
      let h = host_of d in
      if h == host then Here else On h

(* The name of machine [m], [None] for this one. *)
let rec machine_name = function
  | Here -> None
  | Hosts m -> Some m
  | On h -> machine_name h.machine

(* Values per device *)

(* A key's values by device index, set under [locals_lock], so a get that
   finds its value reads it without the lock. Once a device's stop returned,
   its values leave every key. A store keeps nothing on a lost device, so a
   store that raced the stop keeps nothing after the drop, which comes after
   the loss. Indices are never reused. *)
type 'a key = { init : device -> 'a; values : 'a option Table.t }
type any_key = Key : 'a key -> any_key

let keys : any_key list ref = ref []
let locals_lock = Lock.create ()

let add_key k = keys := k :: !keys

let new_key init =
  let k = { init; values = Table.make None } in
  Lock.protect locals_lock add_key (Key k);
  k

let store_locked (k, d, v) =
  match Table.get k.values d.index with
  | Some v' -> v'
  | None ->
      if not (is_lost d) then Table.set k.values d.index (Some v);
      v

(* Keeps [v] as [k]'s value on [d] unless a value came first or [d] is lost:
   the value kept, or [v]. *)
let store k d v = Lock.protect locals_lock store_locked (k, d, v)

let local k d =
  if is_lost d then raise_lost d;
  match Table.get k.values d.index with
  | Some v -> v
  | None -> store k d (k.init d)

let clear d (Key k) =
  if Option.is_some (Table.get k.values d.index) then
    Table.set k.values d.index None

let drop_locked d = List.iter (clear d) !keys

let drop_values d = Lock.protect locals_lock drop_locked d

(* Loss and stops *)

let answered : (device -> unit) ref = ref ignore

(* Reads a lost device's word behind a transport through its driver, which may
   be called after [stop], so rig's copy of it moves on. *)
let refresh d =
  match d.kind with
  | Driver { m; h; _ } when d.live.word = 0 -> (
      let module D = (val m) in
      match D.signaled h with
      | w -> c_set_seen d.index w
      | exception D.Fault _ -> ())
  | _ -> ()

let stop_returned d = c_state d.index >= ended_state

(* Whether the stop of [d] returned and its word shows its last submitted
   value: no work of it runs. An orphaned device's work is its parent's. *)
let stopped d =
  c_state d.index = ended_state
  && begin
    refresh d;
    c_word d.index >= c_submitted d.index
  end

let stop_handle d =
  match d.kind with
  | Driver { m; h; _ } -> (
      let module D = (val m) in
      let fault = if c_faulted d.live.c then Some (why d) else None in
      try D.stop h ~fault with D.Fault _ -> ())
  | Io { m; h } -> (
      let module I = (val m) in
      try I.stop h with I.Fault _ -> ())
  | Host -> ()

(* Removes a device's names from the table of names ({!table}, below), which
   so holds the devices whose stop has not returned. *)
let forget_names : (device -> unit) ref = ref ignore

(* Moves the state on after the driver's stop. Its values and its names go
   first, so a close, which waits for the state, returns after them, and the
   state moves also when they raise. *)
let returned d =
  match
    drop_values d;
    !forget_names d
  with
  | () -> c_stopped d.live.c
  | exception e ->
      c_stopped d.live.c;
      Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ())

(* The state moves on whatever the driver's stop did: the device must count as
   stopped or not. *)
let stop d =
  (match stop_handle d with
  | () -> returned d
  | exception e ->
      returned d;
      Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ()));
  !answered d

let rec run_owed () =
  if c_owed () > 0 then
    match c_claim () with
    | 0 -> ()
    | i ->
        stop (of_index i);
        run_owed ()

let lose d why =
  ignore (c_lose d.live.c why true);
  run_owed ();
  raise_lost d

let leave d =
  c_exit d.live.c;
  run_owed ()

(* Ends the counted call on [d] that raised [e]: a fault of [d] loses it. The
   call ends before anything allocates, since a second exception would keep
   it counted for good. *)
let raised d e =
  c_exit d.live.c;
  let bt = Printexc.get_raw_backtrace () in
  run_owed ();
  match d.live.fault e with
  | Some why -> lose d why
  | None -> Printexc.raise_with_backtrace e bt

let refused d =
  run_owed ();
  raise_lost d

let counted d f =
  match c_enter d.live.c with
  | 0 -> (
      match f () with
      | r ->
          leave d;
          r
      | exception e -> raised d e)
  | _ -> refused d

(* On a stopped device a failure of the call, its fault or its memory's, gives
   back nothing more. On a live one a fault loses the device, which its caller
   never sees: a give is no use of the device. *)
let give d f =
  let l = d.live in
  if orphaned d then true
  else if stopped d then begin
    (match f () with
    | () -> ()
    | exception Sys_error _ -> ()
    | exception e when Option.is_some (l.fault e) -> ());
    true
  end
  else if c_enter l.c <> 0 then begin
    run_owed ();
    false
  end
  else
    match f () with
    | () ->
        leave d;
        true
    | exception e -> (
        c_exit l.c;
        let bt = Printexc.get_raw_backtrace () in
        run_owed ();
        match l.fault e with
        | Some why ->
            ignore (c_lose l.c why true : bool);
            run_owed ();
            true
        | None -> Printexc.raise_with_backtrace e bt)

let fail why = if c_fail why then run_owed ()

let failure () =
  match c_first () with
  | 0 -> None
  | -1 -> c_failed ()
  | i -> Some (printed (of_index i))

(* The timeline *)

(* A word with no host address is read through a counted call, written out
   so that a read allocates nothing. *)
let word d =
  let l = d.live in
  if l.word <> 0 || is_lost d then c_word d.index
  else
    match d.kind with
    | Driver { m; h; _ } ->
        let module D = (val m) in
        if c_enter l.c <> 0 then refused d;
        let w =
          match D.signaled h with
          | w ->
              leave d;
              w
          | exception e -> raised d e
        in
        c_set_seen d.index w;
        w
    | _ -> c_word d.index

let now_ms () = Prof.now () / 1_000_000

(* How long a sleep may watch the word [w] under [d]'s hang bound [hang], or a
   loss of [d] once the bound passed. The clock restarts when the word moved or
   [d] was idle at this look or the last, and runs on while the same committed
   value stays above the word. A device whose queue waits for a producer's
   unreached value counts as idle: its own work may not have started. The word
   is read again here: [w] was read before a commit that may have waited for
   the turn as long as the still interval. *)
let bounded d w ~still_ms hang =
  let w = Int.max w (c_word d.index) in
  let l = d.live in
  let now = now_ms () and p = l.progress in
  let idle = committed d <= w || c_waits_in_queue l.c in
  if idle || p.idle || p.seen <> w then begin
    l.progress <- { seen = w; idle; since = now };
    if idle then still_ms else Int.min still_ms hang
  end
  else
    let left = p.since + hang - now in
    if left <= 0 then lose d (strf "no progress for %d ms" hang);
    Int.min still_ms left

(* A counted call, written out so that a wait's sleep without a hang bound
   allocates nothing. *)
let sleep d ~seen ~still_ms =
  match d.kind with
  | Driver { m; h; _ } -> (
      let module D = (val m) in
      let l = d.live in
      let still_ms =
        match l.hang_ms with
        | None -> still_ms
        | Some hang -> bounded d seen ~still_ms hang
      in
      if c_enter l.c <> 0 then refused d;
      match D.sleep h ~seen ~still_ms with
      | () -> leave d
      | exception e -> raised d e)
  | _ -> ()

module Answer = struct
  let ok = 0
  let busy = 1
  let no_room = 2
  let never = 3
  let producer_lost = 6
  let failed = 7
  let need_record = 8
end

(* Commits [d]'s submitted work if [d]'s turn frees within the still interval:
   whether it did, or found [d] lost. A commit that fails has lost [d], whose
   stop is then owed. *)
let commit_once d =
  let r = c_commit d.live.c in
  if r = Answer.failed then run_owed ();
  r <> Answer.busy

let rec commit d v = if committed d < v && not (commit_once d) then commit d v

let rec settle d v =
  let r = c_settle d.live.c v in
  if r = Answer.failed then run_owed ();
  if r = Answer.busy then settle d v else r = Answer.ok

let signaled = word

let take_afters (l, w) =
  let due, later = List.partition (fun (v, _) -> v <= w) l.afters in
  l.afters <- later;
  due

(* Runs the functions registered on [d]'s values up to [w], re-raising the first
   exception one raised once all ran. *)
let run_afters d w =
  let l = d.live in
  let due = Lock.protect l.lock take_afters (l, w) in
  let first = ref None in
  List.iter
    (fun (_, f) -> try f () with e -> if !first = None then first := Some e)
    (List.rev due);
  Option.iter raise !first

let add_after (l, a) = l.afters <- a :: l.afters

let after d v f =
  let l = d.live in
  Lock.protect l.lock add_after (l, (v, f))

(* A device whose unreached work waits in its queue on another's surfaces that
   producer's unseen fault: its sleep raises it. *)
let look_at_producers d =
  Array.iter
    (fun i ->
      let p = of_index i in
      if not (is_lost p) then
        try sleep p ~seen:(word p) ~still_ms:0 with Lost _ -> ())
    (c_producers d.live.c)

(* Reads [d]'s word until it reaches [v] or the clock [until]: the word. *)
let rec spin_held d v until =
  let w = c_word d.index in
  if w >= v || Prof.now () >= until then w
  else begin
    Domain.cpu_relax ();
    spin_held d v until
  end

(* Waits for [d]'s value [v], the word having read [seen] since [since]: the
   word once it reads [v] or [d] is lost. Until [v] is committed, each round
   commits [d]'s work unless another holds [d]'s turn throughout the still
   interval, such as a submit blocked in its driver, whose earlier work runs
   without another call, so the turn frees and a later round commits. Top
   level, so a wait that finds its value reached allocates nothing. *)
let rec wait_from d v seen since =
  let w = word d in
  if w >= v || is_lost d then w
  else begin
    if committed d < v then ignore (commit_once d : bool);
    let now = now_ms () in
    let since = if w <> seen then now else since in
    let still = now - since in
    let l = d.live in
    (match l.completion with
    | Host_writes -> sleep d ~seen:w ~still_ms
    | (Store | Object _) when l.word = 0 -> sleep d ~seen:w ~still_ms
    | (Store | Object _) when still >= still_ms ->
        look_at_producers d;
        sleep d ~seen:w ~still_ms
    | Store | Object _ ->
        if spin_held d v (Prof.now () + held_ns) < v then
          ignore (c_spin l.c v (still_ms - still)));
    wait_from d v w since
  end

(* How a wait blocks depends on who writes the word. A word the host writes
   ([Host_writes]), or one with no host address, changes only on an event the
   driver sees, so the wait sleeps in the driver from its first read. A word a
   queue or an object writes is read with the domain lock held for [held_ns],
   then spun on with the lock released, and once it stood still for
   [still_ms] the wait sleeps in the driver between reads. *)
let wait_value d v =
  let w = word d in
  let w = if w >= v || is_lost d then w else wait_from d v w (now_ms ()) in
  if is_lost d then raise_lost d;
  if d.live.afters != [] then run_afters d w

let wait d v =
  let s = submitted d in
  if v > s then invalid_argf "Rig: %d is beyond %s's last value %d" v d.name s;
  wait_value d v

(* Waits for [d]'s value [v] from the word [w], until [d] is lost: a loss its
   sleep raises is judged by the caller. *)
let wait_until_lost d v w =
  match wait_from d v w (now_ms ()) with
  | w -> w
  | exception Lost (d', _) when d' == d -> 0

let check p =
  let i = Point.index p in
  if lost_at i && not (c_done p) then raise_lost (of_index i)

(* A lost device's point is done if its word reached it before the loss: its
   stop raises the word whatever ran. The word is read before the loss is, so a
   word the stop raised shows the loss. Before it blocks, it raises the loss of
   any point of [m]'s stamps a lost device did not reach, so that it waits out
   no work ahead of a loss. The stamps are read from [m] there: [m] holds them
   until then. *)
let wait_point m every p =
  let i = Point.index p in
  if lost_at i then check p
  else
    let d = of_index i and v = Point.value p in
    let w = word d in
    let w =
      if w >= v || is_lost d then w
      else
        let q = c_stamps_lost m.entry.stamps every in
        if q <> 0 then raise_lost (of_index (Point.index q));
        wait_until_lost d v w
    in
    if not (is_lost d) then
      begin if d.live.afters != [] then run_afters d w
      end
    else if not (c_done p) then raise_lost d

(* The word is read before the loss is, as [wait_point] reads them. *)
let is_done p =
  let i = Point.index p in
  if lost_at i then c_done p
  else
    let w = word (of_index i) in
    if lost_at i then c_done p else w >= Point.value p

let settled p =
  let i = Point.index p in
  if lost_at i then orphaned_at i || c_word i >= Point.value p
  else
    match word (of_index i) >= Point.value p with
    | r -> r
    | exception Lost _ -> false

(* Closing *)

(* Runs the stop if it is owed and claimable, and waits for it, returning to
   OCaml every still interval. *)
let rec await_stop d =
  run_owed ();
  if not (c_await_stop d.live.c still_ms) then await_stop d

let close_one d =
  (match d.kind with
  | Driver _ when not (is_lost d) -> (
      try wait d (submitted d) with Lost (d', _) when d' == d -> ())
  | _ -> ());
  ignore (c_lose d.live.c "closed" false);
  await_stop d

(* Opening *)


(* An open of a name: its slot is [k], the machine's name and the device's.
   [opening] says whether the slot reads [Opening] for it, [index] is the
   index it took, and [ended] says that an exception ended it. *)
type request = {
  fn : string;
  machine : machine;
  k : string option * string;
  full : string;
  key : int;
  host : bool;
  mutable opening : bool;
  mutable index : int;
  mutable ended : bool; [@atomic]
}

type slot = Opening of request | Open of device

module Names = Tbl.Make (struct
  type t = string option * string

  let equal = ( = )
  let hash = Hashtbl.hash
end)

let table : slot Names.t = Names.create 16
let next_index = Atomic.make 1

(* Under the table's lock: removes [d]'s names, so an open of them makes a new
   device. *)
let forget_locked d =
  let mine k slot ks =
    match slot with Open d' when d' == d -> k :: ks | _ -> ks
  in
  List.iter (Names.remove table) (Names.fold mine table [])

let () = forget_names := Lock.protect table_lock forget_locked

(* The device [r] published, if any. *)
let made r =
  let i = r.index in
  if i <> 0 && c_next i = i then Ok (of_index i) else Error ""

(* Under the table's lock: ends [r]'s open with [result]. The name's slot
   changes last, so an open that an exception stopped half way still reads
   [Opening], and ending it again ends it as once. *)
let end_open (r, result) =
  if r.host then Option.iter (Strings.remove hosting) (fst r.k);
  (match result with
  | Ok d ->
      if r.host then
        Option.iter (fun m -> Strings.replace machines m d) (fst r.k);
      Names.replace table r.k (Open d)
  | Error _ -> Names.remove table r.k);
  Lock.broadcast table_lock

(* Under the table's lock: ends the opens an exception ended, as they stand:
   a device one published stays open under its name, and otherwise its name
   is free. *)
let end_interrupted () =
  let ended _ slot rs =
    match slot with Opening r when r.ended -> r :: rs | _ -> rs
  in
  List.iter (fun r -> end_open (r, made r)) (Names.fold ended table [])

(* A machine has one host, open or opening. *)
let check_host r =
  match r.machine with
  | Hosts m -> (
      (match Strings.find_opt machines m with
      | Some h when h.name <> r.full && not (is_lost h) ->
          invalid_argf "Rig.%s: machine %s's host is %s" r.fn m h.name
      | _ -> ());
      match Strings.find_opt hosting m with
      | Some n when n <> r.full ->
          invalid_argf "Rig.%s: machine %s's host is opening as %s" r.fn m n
      | _ -> ())
  | Here | On _ -> ()

(* A machine whose host is lost or closing opens nothing more, its host
   included: it has one for its life. *)
let host_down machine =
  let down h =
    if is_lost h then Some (printed h)
    else if h.closing > 0 then Some (strf "%s lost: closed" h.name)
    else None
  in
  match machine with
  | Here -> None
  | On h -> down h
  | Hosts m -> Option.bind (Strings.find_opt machines m) down

(* A failed process opens nothing, not calling [make]. *)
let rec find r =
  match (c_failed (), host_down r.machine, Names.find_opt table r.k) with
  | Some why, _, _ | None, Some why, _ -> `Error why
  | None, None, Some (Opening o) when o.ended ->
      end_open (o, made o);
      find r
  | None, None, Some (Opening _) ->
      Lock.wait table_lock;
      find r
  | None, None, Some (Open d) when not (is_lost d) ->
      if d.key <> r.key then
        invalid_argf "Rig.%s: %s is open as another driver's device" r.fn
          r.full;
      `Open d
  | None, None, Some (Open d) ->
      if stop_returned d then `Make
      else `Error (strf "%s is lost and its stop has not answered" r.full)
  | None, None, None -> `Make

(* Under the table's lock: the device open under [r]'s name, or the slot
   claimed for [r]. *)
let claim r =
  end_interrupted ();
  check_host r;
  match find r with
  | `Make ->
      r.opening <- true;
      Names.replace table r.k (Opening r);
      if r.host then
        Option.iter (fun m -> Strings.replace hosting m r.full) (fst r.k);
      `Make
  | found -> found

(* Opens the device [name] on [machine] with [make], which builds it from its
   index and full name, once no open of that name runs. From the moment the
   slot may read [Opening], an exception, such as one a signal handler raises
   at an allocation, ends the open as it stands: a device it published stays
   open under the name, and otherwise the name is free. [make] ends what it
   made and did not publish. *)
let open_named ~fn ~machine ~name ~key make =
  let m = machine_name machine in
  let full = match m with None -> name | Some m -> name ^ "@" ^ m in
  let host = match machine with Hosts _ -> true | Here | On _ -> false in
  let r =
    {
      fn;
      machine;
      k = (m, name);
      full;
      key;
      host;
      opening = false;
      index = 0;
      ended = false;
    }
  in
  let finish result =
    Lock.protect table_lock end_open (r, result);
    result
  in
  match
    match Lock.protect table_lock claim r with
    | `Open d -> Ok d
    | `Error e -> Error e
    | `Make ->
        r.index <- Atomic.fetch_and_add next_index 1;
        if r.index > Point.max_index then
          finish
            (Error
               (strf "%s: the process opened %d devices, the most it can" full
                  Point.max_index))
        else finish (make ~index:r.index ~name:full)
  with
  | result -> result
  | exception e when r.opening ->
      (* Marks [r] ended for the next open or close to end, with no call to an
         OCaml function, where a second exception may arise. The table's lock
         is not taken: a waiter reads [ended] when it wakes, at least every
         200 ms. *)
      r.ended <- true;
      Lock.broadcast table_lock;
      Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ())

let completion_of = function
  | Rig_edge.Store -> Store
  | Object o -> Object (Nativeint.to_int o)
  | Host -> Host_writes

(* Publishes the C record of [d], which is in the tables: spreads, failures and
   the fork handler walk it for good. A device lost at birth by a failure of the
   process is stopped here. *)
let publish d =
  c_publish d.live.c;
  run_owed ()

(* The device of the driver's handle [h], in the tables and not published.
   Every fact is read before the C record is published: a fault reading one
   leaves nothing made. *)
let driver_device (type a) (module D : Rig_edge.Driver with type t = a)
    (h : a) ~index ~name ~machine ~memory_device =
  let m : (a, D.region, D.image) dm = (module D) in
  let rid : D.region Type.Id.t = Type.Id.make () in
  let f = D.facts h in
  Option.iter
    (fun n ->
      if n < 1 then
        invalid_argf "Rig.open_: %s's hang bound is %d ms, expected at least 1"
          name n)
    f.hang_ms;
  if f.most_parts < 1 then
    invalid_argf
      "Rig.open_: %s's submissions hold at most %d parts, expected at least 1"
      name f.most_parts;
  let queues = Array.of_list f.queues in
  Option.iter
    (fun i ->
      if i < 0 || i >= Array.length queues
         || not (List.mem Rig_edge.Copy queues.(i).runs)
      then
        invalid_argf
          "Rig.open_: %s's copy engine is queue %d, which runs no copies" name
          i)
    f.copy_engine;
  let word = Option.value ~default:0 (D.locate f.word).host in
  let c = c_new index name f.may_block f.edge (Nativeint.of_int word) in
  let d =
    make_device ~index ~name ~machine
      ~kind:(Driver { m; h; rid })
      ~c ~arch:f.arch ~queues ~copy_engine:f.copy_engine
      ~completion:(completion_of f.completion) ~waits:f.waits
      ~hang_ms:f.hang_ms ~maps_host:f.maps_host ~most_parts:f.most_parts
      ~host_addresses:f.host_addresses ~word
      ~word_region:(Some (Region { m; h; r = f.word; rid }))
      ~key:(Type.Id.uid D.key) ~memory_device
      ~fault:(function D.Fault why -> Some why | _ -> None)
      ~capability:(Some f.capability) ~budget:f.budget
  in
  register d;
  d

let open_driver (type a) ?(memory_device = false)
    (module D : Rig_edge.Driver with type t = a) ~machine ~name make =
  let fn = if memory_device then "memory_device" else "open_" in
  open_named ~fn ~machine ~name ~key:(Type.Id.uid D.key)
  @@ fun ~index ~name:full ->
  match make () with
  | Error e -> Error e
  | Ok h -> (
      (* Until the device is published, a failure stops the handle: a fault is
         a loss, and any other exception, such as facts that break the
         contract, is raised again. *)
      match
        driver_device (module D) h ~index ~name:full ~machine ~memory_device
      with
      | d ->
          publish d;
          Ok d
      | exception D.Fault why ->
          (try D.stop h ~fault:(Some why) with D.Fault _ -> ());
          Error (strf "%s: %s" full why)
      | exception e ->
          (try D.stop h ~fault:None with D.Fault _ -> ());
          Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ()))

let open_io (type a) (module I : Rig_edge.Io with type t = a) ~machine ~name
    make =
  let key = Type.Id.uid I.region_key in
  open_named ~fn:"open_io" ~machine ~name ~key
  @@ fun ~index ~name:full ->
  match make () with
  | Error e -> Error e
  | Ok h -> (
      (* As for a driver's device. *)
      match
        let budget = I.budget h in
        let c = c_io_new index full in
        let d =
          make_device ~index ~name:full ~machine
            ~kind:(Io { m = (module I); h })
            ~c ~arch:"" ~queues:[||] ~copy_engine:None ~completion:Host_writes
            ~waits:no_waits ~hang_ms:None ~maps_host:false ~most_parts:0
            ~host_addresses:false
            ~word:0 ~word_region:None ~key ~memory_device:false
            ~fault:(function I.Fault why -> Some why | _ -> None)
            ~capability:None ~budget
        in
        register d;
        d
      with
      | d ->
          publish d;
          Ok d
      | exception I.Fault why ->
          (try I.stop h with I.Fault _ -> ());
          Error (strf "%s: %s" full why)
      | exception e ->
          (try I.stop h with I.Fault _ -> ());
          Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ()))

(* Closing *)

(* The open devices of machine [m] but [h]. Holds the table's lock. *)
let devices_of m h =
  Names.fold
    (fun (m', _) slot acc ->
      match slot with Open d when m' = Some m && d != h -> d :: acc | _ -> acc)
    table []

let opening_on m =
  Names.fold
    (fun (m', _) slot acc ->
      let opening = match slot with Opening _ -> true | Open _ -> false in
      acc || (m' = Some m && opening))
    table false

(* Under the table's lock: marks the host [d] of machine [m] closing, waits
   for the opens in flight on [m], ending those an exception ended, and is
   [m]'s other devices. An exception before it returns, such as [Sys.Break]
   in a wait, takes the close back by its handler's first step, an atomic
   one: a call there could take a second exception first. Another close of
   [d] under way keeps its own. *)
let start_close ((d : device), m) =
  match
    ignore (Atomic.Loc.fetch_and_add [%atomic.loc d.closing] 1);
    while
      end_interrupted ();
      opening_on m
    do
      Lock.wait table_lock
    done;
    devices_of m d
  with
  | devices -> devices
  | exception e ->
      ignore (Atomic.Loc.fetch_and_add [%atomic.loc d.closing] (-1));
      Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ())

(* A machine's devices end before its host, so [host_of] of an open device is
   never closed: the machine stops taking opens, the opens in flight finish,
   then its devices close. *)
let close d =
  if is_host d then invalid_arg "Rig.close: the host is never closed";
  match d.machine with
  | Here | On _ -> close_one d
  | Hosts m -> (
      let devices = Lock.protect table_lock start_close (d, m) in
      match
        List.iter close_one devices;
        close_one d
      with
      | () -> ignore (Atomic.Loc.fetch_and_add [%atomic.loc d.closing] (-1))
      | exception e ->
          ignore (Atomic.Loc.fetch_and_add [%atomic.loc d.closing] (-1));
          Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ()))

(* Reach *)

let peer d d' =
  match (d.kind, d'.kind) with
  | Driver { m; h; _ }, Driver { m = m'; h = h'; _ } -> (
      let module D = (val m) in
      let module D' = (val m') in
      match Type.Id.provably_equal D.key D'.key with
      | Some Type.Equal -> D.peer h h'
      | None -> false)
  | _ -> false

let reaches d d' =
  d == d'
  || same_machine d d'
     &&
     match (d.kind, d'.kind) with
     | Io _, _ | _, Io _ -> false
     | Host, Host -> false
     | Host, Driver _ -> d'.host_addresses
     | Driver _, Host -> d.maps_host
     | Driver _, Driver _ -> d'.memory_device || peer d d'
