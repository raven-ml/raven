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
external c_done : int -> bool = "caml_rig_done" [@@noalloc]

(* A device's line, read by its index for the life of the process. *)
external c_word : int -> int = "caml_rig_word" [@@noalloc]
external c_set_seen : int -> int -> unit = "caml_rig_set_seen" [@@noalloc]
external c_submitted : int -> int = "caml_rig_submitted" [@@noalloc]
external c_state : int -> int = "caml_rig_state" [@@noalloc]
external c_why : int -> string = "caml_rig_why"
external c_upgrade : int -> bool = "caml_rig_upgrade" [@@noalloc]

(* The first point of the stamps [st] that a lost device did not reach, among
   their last write's and, if [every], their uses', [0] for none. *)
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
external c_word_retire : c -> bool = "caml_rig_word_retire"
external minors : unit -> int = "caml_rig_minors" [@@noalloc]
external c_enter : c -> int = "caml_rig_enter"
external c_exit : c -> unit = "caml_rig_exit"
external c_spin : c -> int -> int -> int = "caml_rig_spin"
external c_producers : c -> int array = "caml_rig_producers"
external c_waits_in_queue : c -> bool = "caml_rig_waits_in_queue"
external release_list : unit -> int = "caml_rig_release_list"
external c_release : c -> int = "caml_rig_device_release" [@@noalloc]
external host_arch : unit -> string = "caml_rig_arch"
external c_forgotten : unit -> int = "caml_rig_forgotten"
external c_forget : int -> unit = "caml_rig_forget" [@@noalloc]

(* [c_next_live i] is the least index from [i] on whose device's resources
   are not freed, 0 for none; [c_unlive i] says index [i]'s are. *)
external c_next_live : int -> int = "caml_rig_next_live" [@@noalloc]
external c_unlive : int -> unit = "caml_rig_unlive" [@@noalloc]

(* The C line's states: [rig_stubs.h]'s enum. *)
let live_state = 0
let ended_state = 4
let stopped_state = 5
let orphaned_state = 6

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

module Cache = Hashtbl.Make (Int)

let no_waits = { Rig_edge.stores = false; hosts = false; objects = false; most = 0 }

let make_device ~index ~name ~machine ~kind ~c ~arch ~queues ~copy_engine
    ~completion ~waits ~hang_ms ~maps_host ~host_addresses ~word ~word_region
    ~key ~memory_device ~fault ~capability ~budget =
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
      drains = 0;
      entries = 0;
      used = 0;
      cached = 0;
      cache = Cache.create 16;
      retiring = [];
      pending = ([], []);
      pairs = [||];
      pair_maps = [];
      mapped = Cache.create 4;
      word_end = Read;
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
    host_addresses;
    key;
    memory_device;
    capability;
    budget;
    live = Some live;
  }

(* The host runs launches of linked code ([rig_host_code.c]) on its one queue,
   and waits on the host for other devices' points. *)
let host =
  let queues = [| { Rig_edge.name = "COMPUTE:0"; runs = [ Launch ] } |] in
  let d =
    make_device ~index:0 ~name:"CPU" ~machine:Here ~kind:Host
      ~c:(c_host_new "CPU") ~arch:(host_arch ()) ~queues ~copy_engine:None
      ~completion:Host_writes ~waits:no_waits ~hang_ms:None ~maps_host:false
      ~host_addresses:false ~word:(c_host_word ()) ~word_region:None ~key:(-1)
      ~memory_device:false
      ~fault:(fun _ -> None)
      ~capability:None ~budget:max_int
  in
  d

(* The device every point of a device whose record was collected names
   ({!of_index}): lost with the reason ["ended"], its word past every value.
   Its index is past every point's. *)
let forgotten =
  {
    index = c_forgotten ();
    name = "ended";
    machine = Here;
    kind = Ended;
    arch = "";
    maps_host = false;
    host_addresses = false;
    key = -1;
    memory_device = false;
    capability = None;
    budget = 0;
    live = None;
  }

(* The table of devices *)

(* The devices whose resources are not freed, by index; an index whose open
   failed, or whose device's resources are freed, holds the host. *)
let devices = Table.make host

module Int_map = Map.Make (Int)

(* The devices whose resources are freed, held weakly by index: a point finds
   its device while a value of the program refers to it, and {!forgotten} once
   the collector took it, which also takes it out of the map. *)
let freed_devices : device Weak.t Int_map.t Atomic.t = Atomic.make Int_map.empty

let rec update_freed f =
  let m = Atomic.get freed_devices in
  if not (Atomic.compare_and_set freed_devices m (f m)) then update_freed f

let of_index i =
  let d = Table.get devices i in
  if d != host || i = 0 then d
  else
    match Int_map.find_opt i (Atomic.get freed_devices) with
    | None -> forgotten
    | Some w -> ( match Weak.get w 0 with Some d -> d | None -> forgotten)

(* Once the collector takes [d], its weak entry and its line's reason go: its
   points ask {!forgotten} from then on. The finaliser holds [d]'s index
   alone. *)
let register d =
  Table.set devices d.index d;
  let i = d.index in
  Gc.finalise_last
    (fun () ->
      update_freed (Int_map.remove i);
      c_forget i)
    d

let iter f =
  let rec from i =
    match c_next_live i with
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

(* The host is never lost: a use of its memory reads no C line. *)
let[@inline] is_lost d = d != host && c_state d.index <> live_state
let orphaned d = c_state d.index = orphaned_state
let lost d = if is_lost d then Some (c_why d.index) else None
let raise_lost d = raise (Lost (d, c_why d.index))
let submitted d = c_submitted d.index

(* A device whose resources are freed is lost: a use of them raises its
   loss. *)
let[@inline] live d = match d.live with Some l -> l | None -> raise_lost d
let freed d = Option.is_none d.live
let committed d = c_committed (live d).c

(* Locks *)

let protect d f = Lock.protect (live d).lock f
let hold d = Lock.hold (live d).lock
let release d = Lock.release (live d).lock
let busy d = Lock.busy (live d).lock

(* The host of each other machine named so far: a machine has one for its
   life. *)
let machines : (string, device) Hashtbl.t = Hashtbl.create 4

(* The machines whose host an open is making, by the host's full name. *)
let hosting : (string, string) Hashtbl.t = Hashtbl.create 4

(* The hosts a close is ending, by index. *)
let closing : (int, unit) Hashtbl.t = Hashtbl.create 4
let table_lock = Lock.create ()

let host_of d = match d.machine with Here -> host | Hosts _ -> d | On h -> h
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
   finds its value reads it without the lock. A device whose stop
   returned is in [dropped] until its resources are freed, and its values
   leave every key: a store that raced its stop checks [dropped], or finds
   the resources freed, and keeps nothing. Indices are never reused. *)
type 'a key = { init : device -> 'a; values : 'a option Table.t }
type any_key = Key : 'a key -> any_key

let keys : any_key list ref = ref []
let dropped : (int, unit) Hashtbl.t = Hashtbl.create 16
let locals_lock = Lock.create ()

let new_key init =
  let k = { init; values = Table.make None } in
  Lock.protect locals_lock (fun () -> keys := Key k :: !keys);
  k

(* Keeps [v] as [k]'s value on [d] unless a value came first or [d]
   dropped its values: the value kept, or [v]. *)
let store k d v =
  Lock.protect locals_lock @@ fun () ->
  match Table.get k.values d.index with
  | Some v' -> v'
  | None ->
      if not (Hashtbl.mem dropped d.index || freed d) then
        Table.set k.values d.index (Some v);
      v

let local k d =
  if is_lost d then raise_lost d;
  match Table.get k.values d.index with
  | Some v -> v
  | None -> store k d (k.init d)

let drop_values d =
  let clear (Key k) =
    if Option.is_some (Table.get k.values d.index) then
      Table.set k.values d.index None
  in
  Lock.protect locals_lock @@ fun () ->
  Hashtbl.replace dropped d.index ();
  List.iter clear !keys

(* Loss and stops *)

let answered : (device -> unit) ref = ref ignore
let ended_devices : device list Atomic.t = Atomic.make []

let rec add_ended d =
  let l = Atomic.get ended_devices in
  if not (Atomic.compare_and_set ended_devices l (d :: l)) then add_ended d

let ended () = Atomic.get ended_devices

let rec remove_ended d =
  let l = Atomic.get ended_devices in
  let l' = List.filter (fun d' -> d' != d) l in
  if not (Atomic.compare_and_set ended_devices l l') then remove_ended d

(* Reads a lost device's word behind a transport through its driver, which may
   be called after [stop], so rig's copy of it moves on. *)
let refresh d =
  match d.kind with
  | Driver { m; h; _ } when (live d).word = 0 -> (
      let module D = (val m) in
      match D.signaled h with
      | w -> c_set_seen d.index w
      | exception D.Fault _ -> ())
  | _ -> ()

let upgrade d =
  refresh d;
  c_upgrade d.index

let stop_returned d = c_state d.index >= ended_state

let stopped d =
  let s = c_state d.index in
  s = stopped_state || (s = ended_state && upgrade d)

(* The state moves on whatever the driver's stop did: the device must count as
   stopped or not. Its values drop first, so a close, which waits for the
   state, returns after the drop. *)
let stop d =
  let returned () =
    drop_values d;
    c_stopped (live d).c;
    (match d.kind with Driver _ -> ignore (upgrade d) | _ -> ());
    add_ended d
  in
  Fun.protect ~finally:returned (fun () ->
      match d.kind with
      | Driver { m; h; _ } -> (
          let module D = (val m) in
          let fault =
            if c_faulted (live d).c then Some (c_why d.index) else None
          in
          try D.stop h ~fault with D.Fault _ -> ())
      | Io { m; h } -> (
          let module I = (val m) in
          try I.stop h with I.Fault _ -> ())
      | Host | Ended -> ());
  !answered d

let rec run_owed () =
  if c_owed () > 0 then
    match c_claim () with
    | 0 -> ()
    | i ->
        stop (of_index i);
        run_owed ()

let lose d why =
  ignore (c_lose (live d).c why true);
  run_owed ();
  raise_lost d

let leave d =
  c_exit (live d).c;
  run_owed ()

(* Ends the counted call on [d] that raised [e]: a fault of [d] loses it. *)
let raised d e bt =
  leave d;
  match (live d).fault e with
  | Some why -> lose d why
  | None -> Printexc.raise_with_backtrace e bt

let refused d =
  run_owed ();
  raise_lost d

let counted d f =
  match c_enter (live d).c with
  | 0 -> (
      match f () with
      | r ->
          leave d;
          r
      | exception e -> raised d e (Printexc.get_raw_backtrace ()))
  | _ -> refused d

(* On a stopped device a failure of the call, its fault or its memory's, gives
   back nothing more. On a live one a fault loses the device, which its caller
   never sees: a give is no use of the device. *)
let give d f =
  let l = live d in
  let s = c_state d.index in
  if s = orphaned_state then true
  else if s = stopped_state || (s = ended_state && upgrade d) then begin
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
        let bt = Printexc.get_raw_backtrace () in
        leave d;
        match l.fault e with
        | Some why ->
            ignore (c_lose l.c why true : bool);
            run_owed ();
            true
        | None -> Printexc.raise_with_backtrace e bt)

let move_word d = c_word_retire (live d).c
let fail why = if c_fail why then run_owed ()

(* The device whose loss was the process's first failure, once its resources
   are freed: {!failure} names it for the life of the process. *)
let first_lost : device option Atomic.t = Atomic.make None

let failure () =
  match c_first () with
  | 0 -> None
  | -1 -> c_failed ()
  | i ->
      let d = of_index i in
      Some (strf "%s lost: %s" d.name (c_why d.index))

(* The timeline *)

let word d =
  match d.live with
  | Some l when l.word = 0 && not (is_lost d) -> (
      match d.kind with
      | Driver { m; h; _ } ->
          let module D = (val m) in
          let w = counted d (fun () -> D.signaled h) in
          c_set_seen d.index w;
          w
      | _ -> c_word d.index)
  | _ -> c_word d.index

let now_ms () = Prof.now () / 1_000_000

(* How long a sleep may watch the word [w] under [d]'s hang bound [hang], or a
   loss of [d] once the bound passed. The clock restarts when the word moved or
   [d] was idle at this look or the last, and runs on while the same committed
   value stays above the word. A device whose queue waits for a producer's
   unreached value counts as idle: its own work may not have started. *)
let bounded d w ~still_ms hang =
  let l = live d in
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
      let l = live d in
      let still_ms =
        match l.hang_ms with
        | None -> still_ms
        | Some hang -> bounded d seen ~still_ms hang
      in
      if c_enter l.c <> 0 then refused d;
      match D.sleep h ~seen ~still_ms with
      | () -> leave d
      | exception e -> raised d e (Printexc.get_raw_backtrace ()))
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
  let r = c_commit (live d).c in
  if r = Answer.failed then run_owed ();
  r <> Answer.busy

let rec commit d v = if committed d < v && not (commit_once d) then commit d v

let signaled = word

(* Runs the functions registered on [d]'s values up to [w], re-raising the first
   exception one raised once all ran. *)
let run_afters d w =
  let l = live d in
  let due =
    Lock.protect l.lock (fun () ->
        let due, later = List.partition (fun (v, _) -> v <= w) l.afters in
        l.afters <- later;
        due)
  in
  let first = ref None in
  List.iter
    (fun (_, f) -> try f () with e -> if !first = None then first := Some e)
    (List.rev due);
  Option.iter raise !first

let after d v f =
  let l = live d in
  Lock.protect l.lock (fun () -> l.afters <- (v, f) :: l.afters)

(* A device whose unreached work waits in its queue on another's surfaces that
   producer's unseen fault: its sleep raises it. *)
let look_at_producers d =
  Array.iter
    (fun i ->
      let p = of_index i in
      if not (is_lost p) then
        try sleep p ~seen:(word p) ~still_ms:0 with Lost _ -> ())
    (c_producers (live d).c)

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
    let l = live d in
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
  if (live d).afters != [] then run_afters d w

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

(* A lost device's point is done if its word reached it before the loss: its
   stop raises the word whatever ran. The word is read before the loss is, so a
   word the stop raised shows the loss. Before it blocks, it raises the loss of
   any point of [m]'s stamps a lost device did not reach, so that it waits out
   no work ahead of a loss. The stamps are read from [m] there: [m] holds them
   until then. *)
let wait_point m every p =
  let d = of_index (Point.index p) and v = Point.value p in
  let w = word d in
  let w =
    if w >= v || is_lost d then w
    else
      let q = c_stamps_lost m.root.entry.stamps every in
      if q <> 0 then raise_lost (of_index (Point.index q));
      wait_until_lost d v w
  in
  if not (is_lost d) then
    begin if (live d).afters != [] then run_afters d w
    end
  else if not (c_done p) then raise_lost d

(* The word is read before the loss is, as [wait_point] reads them. *)
let is_done p =
  let d = of_index (Point.index p) in
  if is_lost d then c_done p
  else
    let w = word d in
    if is_lost d then c_done p else w >= Point.value p

let settled p =
  let d = of_index (Point.index p) in
  orphaned d
  || match word d >= Point.value p with r -> r | exception Lost _ -> false

let check p =
  let d = of_index (Point.index p) in
  if is_lost d && not (c_done p) then raise_lost d

(* Closing *)

(* Runs the stop if it is owed and claimable, and waits for it, returning to
   OCaml every still interval. *)
let rec await_stop d =
  run_owed ();
  match d.live with
  | Some l when not (c_await_stop l.c still_ms) -> await_stop d
  | _ -> ()

(* A device whose resources are freed stopped long ago. *)
let close_one d =
  match d.live with
  | None -> ()
  | Some l ->
      (match d.kind with
      | Driver _ when not (is_lost d) -> (
          try wait d (submitted d) with Lost (d', _) when d' == d -> ())
      | _ -> ());
      ignore (c_lose l.c "closed" false);
      await_stop d

(* Opening *)

type slot = Opening | Open of device

let table : (string option * string, slot) Hashtbl.t = Hashtbl.create 16
let next_index = Atomic.make 1

(* Opens the device [name] on [machine] with [make], which builds it from its
   index and full name, once no open of that name runs. *)
let open_named ~fn ~machine ~name ~key make =
  let m = machine_name machine in
  let k = (m, name) in
  let full = match m with None -> name | Some m -> name ^ "@" ^ m in
  (* A machine has one host, open or opening. *)
  let check_host () =
    match machine with
    | Hosts m -> (
        (match Hashtbl.find_opt machines m with
        | Some h when h.name <> full && not (is_lost h) ->
            invalid_argf "Rig.%s: machine %s's host is %s" fn m h.name
        | _ -> ());
        match Hashtbl.find_opt hosting m with
        | Some n when n <> full ->
            invalid_argf "Rig.%s: machine %s's host is opening as %s" fn m n
        | _ -> ())
    | Here | On _ -> ()
  in
  (* A machine whose host is lost or closing opens nothing more, its host
     included: it has one for its life. *)
  let host_down () =
    let down h =
      if is_lost h then Some (strf "%s lost: %s" h.name (c_why h.index))
      else if Hashtbl.mem closing h.index then Some (strf "%s lost: closed" h.name)
      else None
    in
    match machine with
    | Here -> None
    | On h -> down h
    | Hosts m -> Option.bind (Hashtbl.find_opt machines m) down
  in
  let host = match machine with Hosts _ -> true | Here | On _ -> false in
  (* A failed process opens nothing, not calling [make]. *)
  let rec find () =
    match (c_failed (), host_down (), Hashtbl.find_opt table k) with
    | Some why, _, _ | None, Some why, _ -> `Error why
    | None, None, Some Opening ->
        Lock.wait table_lock;
        find ()
    | None, None, Some (Open d) when not (is_lost d) ->
        if d.key <> key then
          invalid_argf "Rig.%s: %s is open as another driver's device" fn full;
        `Open d
    | None, None, Some (Open d) ->
        if stop_returned d then `Make
        else `Error (strf "%s is lost and its stop has not answered" full)
    | None, None, None -> `Make
  in
  let finish r =
    Lock.protect table_lock (fun () ->
        if host then Option.iter (Hashtbl.remove hosting) m;
        (match r with
        | Ok d ->
            Hashtbl.replace table k (Open d);
            if host then Option.iter (fun m -> Hashtbl.replace machines m d) m
        | Error _ -> Hashtbl.remove table k);
        Lock.broadcast table_lock);
    r
  in
  (* From the moment the slot may read [Opening], an exception, such as one a
     signal handler raises at an allocation, ends the open as it stands: a
     device it published stays open under the name, and otherwise the name is
     free. [make] ends what it made and did not publish. *)
  let opening = ref false and index = ref 0 in
  let made () =
    let i = !index in
    if i <> 0 && c_next_live i = i then Ok (of_index i) else Error ""
  in
  match
    let found =
      Lock.protect table_lock (fun () ->
          check_host ();
          match find () with
          | `Make ->
              opening := true;
              Hashtbl.replace table k Opening;
              if host then
                Option.iter (fun m -> Hashtbl.replace hosting m full) m;
              `Make
          | r -> r)
    in
    match found with
    | `Open d -> Ok d
    | `Error e -> Error e
    | `Make ->
        index := Atomic.fetch_and_add next_index 1;
        if !index > Point.max_index then
          finish
            (Error
               (strf "%s: the process opened %d devices, the most it can" full
                  Point.max_index))
        else finish (make ~index:!index ~name:full)
  with
  | r -> r
  | exception e when !opening ->
      let bt = Printexc.get_raw_backtrace () in
      ignore (finish (made ()));
      Printexc.raise_with_backtrace e bt

let completion_of = function
  | Rig_edge.Store -> Store
  | Object o -> Object (Nativeint.to_int o)
  | Host -> Host_writes

(* Publishes the C record of [d], which is in the tables: spreads, failures and
   the fork handler walk it for good. A device lost at birth by a failure of the
   process is stopped here. *)
let publish d =
  c_publish (live d).c;
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
      ~hang_ms:f.hang_ms ~maps_host:f.maps_host
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
          let bt = Printexc.get_raw_backtrace () in
          (try D.stop h ~fault:None with D.Fault _ -> ());
          Printexc.raise_with_backtrace e bt)

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
            ~waits:no_waits ~hang_ms:None ~maps_host:false ~host_addresses:false
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
          let bt = Printexc.get_raw_backtrace () in
          (try I.stop h with I.Fault _ -> ());
          Printexc.raise_with_backtrace e bt)

(* Closing *)

(* The open devices of machine [m] but [h]. Holds the table's lock. *)
let devices_of m h =
  Hashtbl.fold
    (fun (m', _) slot acc ->
      match slot with Open d when m' = Some m && d != h -> d :: acc | _ -> acc)
    table []

let opening_on m =
  Hashtbl.fold
    (fun (m', _) slot acc -> acc || (m' = Some m && slot = Opening))
    table false

(* A machine's devices end before its host, so [host_of] of an open device is
   never closed: the machine stops taking opens, the opens in flight finish,
   then its devices close. *)
let close d =
  if is_host d then invalid_arg "Rig.close: the host is never closed";
  let machine =
    Lock.protect table_lock @@ fun () ->
    match d.machine with
    | Hosts m ->
        Hashtbl.replace closing d.index ();
        while opening_on m do
          Lock.wait table_lock
        done;
        Some (devices_of m d)
    | Here | On _ -> None
  in
  match machine with
  | None -> close_one d
  | Some devices ->
      Fun.protect
        ~finally:(fun () ->
          Lock.protect table_lock (fun () -> Hashtbl.remove closing d.index))
        (fun () ->
          List.iter close_one devices;
          close_one d)

(* Freeing *)

(* Its resources go first, so a store of a value that raced the drop finds
   them gone once [dropped] forgets [d]. The table holds [d] weakly from then
   on, set there before its strong entry goes, so a point finds it
   throughout. *)
let free_live d =
  d.live <- None;
  c_unlive d.index;
  remove_ended d;
  if c_first () = d.index then Atomic.set first_lost (Some d);
  let w = Weak.create 1 in
  Weak.set w 0 (Some d);
  update_freed (Int_map.add d.index w);
  Table.set devices d.index host;
  Lock.protect locals_lock (fun () -> Hashtbl.remove dropped d.index);
  let other _ = function Open d' when d' == d -> None | slot -> Some slot in
  Lock.protect table_lock (fun () -> Hashtbl.filter_map_inplace other table)

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
     | Io _, _ | _, Io _ | Ended, _ | _, Ended -> false
     | Host, Host -> false
     | Host, Driver _ -> d'.host_addresses
     | Driver _, Host -> d.maps_host
     | Driver _, Driver _ -> d'.memory_device || peer d d'
