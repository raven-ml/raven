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

external c_new : int -> string -> bool -> nativeint -> nativeint -> int
  = "caml_rig_device_new"

external c_io_new : int -> string -> int = "caml_rig_io_new"
external c_host_new : string -> int = "caml_rig_host_new"
external c_publish : int -> unit = "caml_rig_publish"
external c_word : int -> int = "caml_rig_word" [@@noalloc]
external c_set_seen : int -> int -> unit = "caml_rig_set_seen" [@@noalloc]
external c_submitted : int -> int = "caml_rig_submitted" [@@noalloc]
external c_committed : int -> int = "caml_rig_committed" [@@noalloc]
external c_commit : int -> bool -> int = "caml_rig_commit"
external c_state : int -> int = "caml_rig_state" [@@noalloc]
external c_done : int -> bool = "caml_rig_done" [@@noalloc]
external c_why : int -> string = "caml_rig_why"
external c_lose : int -> string -> bool -> bool = "caml_rig_lose"
external c_faulted : int -> bool = "caml_rig_faulted" [@@noalloc]
external c_stopped : int -> unit = "caml_rig_stopped"
external c_upgrade : int -> bool = "caml_rig_upgrade" [@@noalloc]
external c_await_stop : int -> int -> bool = "caml_rig_await_stop"
external c_owed : unit -> int = "caml_rig_owed" [@@noalloc]
external c_claim : unit -> int = "caml_rig_claim"
external c_fail : string -> bool = "caml_rig_fail"
external c_failed : unit -> string option = "caml_rig_failed"
external c_first : unit -> int = "caml_rig_first" [@@noalloc]
external generation : unit -> int = "caml_rig_generation" [@@noalloc]
external c_word_retire : int -> bool = "caml_rig_word_retire"
external minors : unit -> int = "caml_rig_minors" [@@noalloc]
external c_enter : int -> int = "caml_rig_enter"
external c_exit : int -> unit = "caml_rig_exit"
external c_spin : int -> int -> int -> int = "caml_rig_spin"
external c_producers : int -> int array = "caml_rig_producers"
external c_waits_in_queue : int -> bool = "caml_rig_waits_in_queue"
external release_list : unit -> int = "caml_rig_release_list"
external host_arch : unit -> string = "caml_rig_arch"

(* The C record's states: [rig_stubs.h]'s enum. *)
let live = 0
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

(* The table of devices *)

(* The devices by index; an index whose open failed holds the host. *)
let devices : device array Atomic.t = Atomic.make [||]
let of_index i = (Atomic.get devices).(i)

let iter f =
  Array.iteri (fun i d -> if i > 0 && i = d.index then f d) (Atomic.get devices)

let register d =
  let rec go () =
    let a = Atomic.get devices in
    let n = Int.max (Array.length a) (d.index + 1) in
    let gap i =
      if i < Array.length a then a.(i) else if i > 0 then a.(0) else d
    in
    let a' = Array.init n gap in
    a'.(d.index) <- d;
    if not (Atomic.compare_and_set devices a a') then go ()
  in
  go ()

let same_machine d d' = Option.equal String.equal d.machine d'.machine

module Cache = Hashtbl.Make (Int)

(* The queue a device's copies go to: the first named "COPY:i". *)
let copy_queue queues =
  Array.find_map
    (fun (q : Rig_edge.queue) ->
      if String.starts_with ~prefix:"COPY:" q.name then Some q.name else None)
    queues

let no_waits = { Rig_edge.stores = false; hosts = false; objects = false; most = 0 }

let make_device ~index ~name ~machine ~kind ~c ~arch ~queues ~completion ~waits
    ~hang_ms ~maps_host ~word ~word_region ~key ~memory_device ~fault
    ~capability ~budget =
  {
    index;
    name;
    machine;
    kind;
    c;
    arch;
    queues;
    copy_queue = copy_queue queues;
    completion;
    waits;
    hang_ms;
    progress = { seen = 0; idle = true; since = 0 };
    maps_host;
    word;
    word_region;
    key;
    memory_device;
    fault;
    capability;
    release = release_list ();
    lock = Lock.create ();
    budget;
    used = 0;
    cached = 0;
    cache = Cache.create 16;
    retiring = [];
    pending = [];
    pairs = [||];
    pair_maps = [];
    mapped = Cache.create 4;
    word_end = Read;
    afters = [];
  }

let host =
  let d =
    make_device ~index:0 ~name:"CPU" ~machine:None ~kind:Host
      ~c:(c_host_new "CPU") ~arch:(host_arch ()) ~queues:[||]
      ~completion:Host_writes ~waits:no_waits ~hang_ms:None ~maps_host:false
      ~word:0 ~word_region:None ~key:(-1) ~memory_device:false
      ~fault:(fun _ -> None)
      ~capability:None ~budget:max_int
  in
  register d;
  d

let () =
  Printexc.register_printer (function
    | Lost (d, why) -> Some (strf "%s lost: %s" d.name why)
    | Out_of_memory (d, n) -> Some (strf "%s cannot allocate %d bytes" d.name n)
    | _ -> None)

(* Locks *)

let protect d f = Lock.protect d.lock f
let hold d = Lock.hold d.lock
let release d = Lock.release d.lock
let busy d = Lock.busy d.lock

(* Facts *)

let is_host d = match d.kind with Host -> true | _ -> false
let is_io d = match d.kind with Io _ -> true | _ -> false

(* The host is never lost: a use of its memory reads no C record. *)
let[@inline] is_lost d = d != host && c_state d.c <> live
let orphaned d = c_state d.c = orphaned_state
let lost d = if is_lost d then Some (c_why d.c) else None
let raise_lost d = raise (Lost (d, c_why d.c))
let submitted d = c_submitted d.c
let committed d = c_committed d.c
let machines : (string, device) Hashtbl.t = Hashtbl.create 4

(* The machines whose host an open is making, by the host's full name. *)
let hosting : (string, string) Hashtbl.t = Hashtbl.create 4

(* The machines whose host a close is ending. *)
let closing : (string, unit) Hashtbl.t = Hashtbl.create 4
let table_lock = Lock.create ()

(* A device of another machine opens only once that machine's host is open
   ([open_named]), so every device has one. *)
let host_of d =
  match d.machine with
  | None -> host
  | Some m -> Lock.protect table_lock (fun () -> Hashtbl.find machines m)

(* Loss and stops *)

let answered : (device -> unit) ref = ref ignore
let ended_devices : device list Atomic.t = Atomic.make []

let rec add_ended d =
  let l = Atomic.get ended_devices in
  if not (Atomic.compare_and_set ended_devices l (d :: l)) then add_ended d

let ended () = Atomic.get ended_devices

(* Reads a lost device's word behind a transport through its driver, which may
   be called after [stop], so rig's copy of it moves on. *)
let refresh d =
  match d.kind with
  | Driver { m; h; _ } when d.word = 0 -> (
      let module D = (val m) in
      match D.signaled h with
      | w -> c_set_seen d.c w
      | exception D.Fault _ -> ())
  | _ -> ()

let upgrade d =
  refresh d;
  c_upgrade d.c

let stop_returned d = c_state d.c >= ended_state

let stopped d =
  let s = c_state d.c in
  s = stopped_state || (s = ended_state && upgrade d)

(* The state moves on whatever the driver's stop did: the device must count as
   stopped or not. *)
let stop d =
  let returned () =
    c_stopped d.c;
    (match d.kind with Driver _ -> ignore (upgrade d) | _ -> ());
    add_ended d
  in
  Fun.protect ~finally:returned (fun () ->
      match d.kind with
      | Driver { m; h; _ } -> (
          let module D = (val m) in
          let fault = if c_faulted d.c then Some (c_why d.c) else None in
          try D.stop h ~fault with D.Fault _ -> ())
      | Io { m; h } -> (
          let module I = (val m) in
          try I.stop h with I.Fault _ -> ())
      | Host -> ());
  !answered d

let rec run_owed () =
  if c_owed () > 0 then
    match c_claim () with
    | 0 -> ()
    | i ->
        stop (of_index i);
        run_owed ()

let lose d why =
  ignore (c_lose d.c why true);
  run_owed ();
  raise_lost d

let leave d =
  c_exit d.c;
  run_owed ()

(* Ends the counted call on [d] that raised [e]: a fault of [d] loses it. *)
let raised d e bt =
  leave d;
  match d.fault e with
  | Some why -> lose d why
  | None -> Printexc.raise_with_backtrace e bt

let refused d =
  run_owed ();
  raise_lost d

let counted d f =
  match c_enter d.c with
  | 0 -> (
      match f () with
      | r ->
          leave d;
          r
      | exception e -> raised d e (Printexc.get_raw_backtrace ()))
  | _ -> refused d

(* On a stopped device a failure of the call, its fault or its memory's, gives
   back nothing more. *)
let give d f =
  let s = c_state d.c in
  if s = orphaned_state then ()
  else if s = stopped_state || (s = ended_state && upgrade d) then
    match f () with
    | () -> ()
    | exception Sys_error _ -> ()
    | exception e when Option.is_some (d.fault e) -> ()
  else counted d f

let move_word d = c_word_retire d.c
let fail why = if c_fail why then run_owed ()

let failure () =
  match c_first () with
  | 0 -> None
  | -1 -> c_failed ()
  | i ->
      let d = of_index i in
      Some (strf "%s lost: %s" d.name (c_why d.c))

(* The timeline *)

let word d =
  if d.word <> 0 || is_lost d then c_word d.c
  else
    match d.kind with
    | Driver { m; h; _ } ->
        let module D = (val m) in
        let w = counted d (fun () -> D.signaled h) in
        c_set_seen d.c w;
        w
    | _ -> c_word d.c

let now_ms () = Prof.now () / 1_000_000

(* How long a sleep may watch the word [w] under [d]'s hang bound [hang], or a
   loss of [d] once the bound passed. The clock restarts when the word moved or
   [d] was idle at this look or the last, and runs on while the same committed
   value stays above the word. A device whose queue waits for a producer's
   unreached value counts as idle: its own work may not have started. *)
let bounded d w ~still_ms hang =
  let now = now_ms () and p = d.progress in
  let idle = committed d <= w || c_waits_in_queue d.c in
  if idle || p.idle || p.seen <> w then begin
    d.progress <- { seen = w; idle; since = now };
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
      let still_ms =
        match d.hang_ms with
        | None -> still_ms
        | Some hang -> bounded d seen ~still_ms hang
      in
      if c_enter d.c <> 0 then refused d;
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
  let r = c_commit d.c true in
  if r = Answer.failed then run_owed ();
  r <> Answer.busy

let rec commit d v = if committed d < v && not (commit_once d) then commit d v

let signaled d =
  if (not (is_lost d)) && committed d < submitted d then
    if c_commit d.c false = Answer.failed then run_owed ();
  word d

(* Runs the functions registered on [d]'s values up to [w], re-raising the first
   exception one raised once all ran. *)
let run_afters d w =
  let due =
    protect d (fun () ->
        let due, later = List.partition (fun (v, _) -> v <= w) d.afters in
        d.afters <- later;
        due)
  in
  let first = ref None in
  List.iter
    (fun (_, f) -> try f () with e -> if !first = None then first := Some e)
    (List.rev due);
  Option.iter raise !first

let after d v f = protect d (fun () -> d.afters <- (v, f) :: d.afters)

(* A device whose unreached work waits in its queue on another's surfaces that
   producer's unseen fault: its sleep raises it. *)
let look_at_producers d =
  Array.iter
    (fun i ->
      let p = of_index i in
      if not (is_lost p) then
        try sleep p ~seen:(word p) ~still_ms:0 with Lost _ -> ())
    (c_producers d.c)

(* Reads [d]'s word until it reaches [v] or the clock [until]: the word. *)
let rec spin_held d v until =
  let w = c_word d.c in
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
    (match d.completion with
    | Host_writes -> sleep d ~seen:w ~still_ms
    | (Store | Object _) when d.word = 0 -> sleep d ~seen:w ~still_ms
    | (Store | Object _) when still >= still_ms ->
        look_at_producers d;
        sleep d ~seen:w ~still_ms
    | Store | Object _ ->
        if spin_held d v (Prof.now () + held_ns) < v then
          ignore (c_spin d.c v (still_ms - still)));
    wait_from d v w since
  end

let wait d v =
  let s = submitted d in
  if v > s then
    invalid_argf "Rig.wait: %d is beyond %s's last value %d" v d.name s;
  let w = word d in
  let w = if w >= v || is_lost d then w else wait_from d v w (now_ms ()) in
  if is_lost d then raise_lost d;
  if d.afters != [] then run_afters d w

(* Waits for [d]'s value [v] from the word [w], until [d] is lost: a loss its
   sleep raises is judged by the caller. *)
let wait_until_lost d v w =
  match wait_from d v w (now_ms ()) with
  | w -> w
  | exception Lost (d', _) when d' == d -> 0

(* A lost device's point is done if its word reached it before the loss: its
   stop raises the word whatever ran. The word is read before the loss is, so a
   word the stop raised shows the loss. *)
let wait_point p =
  let d = of_index (Point.index p) and v = Point.value p in
  let w = word d in
  let w = if w >= v || is_lost d then w else wait_until_lost d v w in
  if not (is_lost d) then
    begin if d.afters != [] then run_afters d w
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
  if not (c_await_stop d.c still_ms) then await_stop d

let close_one d =
  (match d.kind with
  | Driver _ when not (is_lost d) -> (
      try wait d (submitted d) with Lost (d', _) when d' == d -> ())
  | _ -> ());
  ignore (c_lose d.c "closed" false);
  await_stop d

(* Opening *)

type slot = Opening | Open of device

let table : (string option * string, slot) Hashtbl.t = Hashtbl.create 16
let next_index = Atomic.make 1

(* Opens the device [name] on [machine] with [make], which builds it from its
   index and full name, once no open of that name runs. *)
let open_named ~fn ~machine ~name ~key ~host make =
  let k = (machine, name) in
  let full = match machine with None -> name | Some m -> name ^ "@" ^ m in
  (* A machine has one host, open or opening, and its devices open after it. *)
  let check_machine () =
    match machine with
    | None when host ->
        invalid_argf "Rig.%s: %s is no other machine's host" fn full
    | None -> ()
    | Some m when host -> (
        (match Hashtbl.find_opt machines m with
        | Some h when h.name <> full && not (is_lost h) ->
            invalid_argf "Rig.%s: machine %s's host is %s" fn m h.name
        | _ -> ());
        match Hashtbl.find_opt hosting m with
        | Some n when n <> full ->
            invalid_argf "Rig.%s: machine %s's host is opening as %s" fn m n
        | _ -> ())
    | Some m ->
        if not (Hashtbl.mem machines m) then
          invalid_argf "Rig.%s: no host of machine %s was opened" fn m
  in
  (* A device of a machine whose host is lost or closing opens no more. *)
  let host_down () =
    match machine with
    | Some m when not host -> (
        match Hashtbl.find_opt machines m with
        | Some h when is_lost h -> Some (strf "%s lost: %s" h.name (c_why h.c))
        | Some h when Hashtbl.mem closing m ->
            Some (strf "%s lost: closed" h.name)
        | _ -> None)
    | _ -> None
  in
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
  let found =
    Lock.protect table_lock (fun () ->
        check_machine ();
        match find () with
        | `Make ->
            Hashtbl.replace table k Opening;
            if host then
              Option.iter (fun m -> Hashtbl.replace hosting m full) machine;
            `Make
        | r -> r)
  in
  match found with
  | `Open d -> Ok d
  | `Error e -> Error e
  | `Make -> (
      let finish r =
        Lock.protect table_lock (fun () ->
            if host then Option.iter (Hashtbl.remove hosting) machine;
            (match r with
            | Ok d ->
                Hashtbl.replace table k (Open d);
                if host then
                  Option.iter (fun m -> Hashtbl.replace machines m d) machine
            | Error _ -> Hashtbl.remove table k);
            Lock.broadcast table_lock);
        r
      in
      let index = Atomic.fetch_and_add next_index 1 in
      if index > Point.max_index then
        finish
          (Error
             (strf "%s: the process opened %d devices, the most it can" full
                Point.max_index))
      else
        match make ~index ~name:full with
        | r -> finish r
        | exception e ->
            let bt = Printexc.get_raw_backtrace () in
            ignore (finish (Error ""));
            Printexc.raise_with_backtrace e bt)

let completion_of = function
  | Rig_edge.Store -> Store
  | Object o -> Object (Nativeint.to_int o)
  | Host -> Host_writes

(* Puts [d] in the tables, then publishes its C record, which spreads, failures
   and the fork handler walk for good: a device lost at birth by a failure of
   the process is stopped here. *)
let publish d =
  register d;
  c_publish d.c;
  run_owed ()

(* The device of the driver's handle [h]. Every fact is read before the C record
   is published: a fault reading one leaves nothing made. *)
let driver_device (type a) (module D : Rig_edge.Driver with type t = a)
    (h : a) ~index ~name ~machine ~memory_device =
  let m : (a, D.region, D.image) dm = (module D) in
  let rid : D.region Type.Id.t = Type.Id.make () in
  let f = D.facts h in
  List.iter
    (fun (q : Rig_edge.queue) ->
      if String.starts_with ~prefix:"COPY:" q.name
         && not (List.mem Rig_edge.Copy q.runs)
      then invalid_argf "Rig.open_: %s's queue %S runs no copies" name q.name)
    f.queues;
  Option.iter
    (fun n ->
      if n < 1 then
        invalid_argf "Rig.open_: %s's hang bound is %d ms, expected at least 1"
          name n)
    f.hang_ms;
  let word = Option.value ~default:0 (D.locate f.word).host in
  let c = c_new index name f.may_block f.edge (Nativeint.of_int word) in
  let d =
    make_device ~index ~name ~machine
      ~kind:(Driver { m; h; rid })
      ~c ~arch:f.arch ~queues:(Array.of_list f.queues)
      ~completion:(completion_of f.completion) ~waits:f.waits
      ~hang_ms:f.hang_ms ~maps_host:f.maps_host ~word
      ~word_region:(Some (Region { m; h; r = f.word; rid }))
      ~key:(Type.Id.uid D.key) ~memory_device
      ~fault:(function D.Fault why -> Some why | _ -> None)
      ~capability:(Some f.capability) ~budget:f.budget
  in
  publish d;
  d

let open_driver (type a) ?(memory_device = false)
    (module D : Rig_edge.Driver with type t = a) ?machine ?(host = false) ~name
    make =
  let fn = if memory_device then "memory_device" else "open_" in
  open_named ~fn ~machine ~name ~key:(Type.Id.uid D.key) ~host
  @@ fun ~index ~name:full ->
  match make () with
  | Error e -> Error e
  | Ok h -> (
      match
        driver_device (module D) h ~index ~name:full ~machine ~memory_device
      with
      | d -> Ok d
      | exception D.Fault why ->
          (* A fault at open is a loss: the handle is stopped. *)
          (try D.stop h ~fault:(Some why) with D.Fault _ -> ());
          Error (strf "%s: %s" full why)
      | exception (Invalid_argument _ as e) ->
          (* Facts that break the contract: the handle is stopped. *)
          (try D.stop h ~fault:None with D.Fault _ -> ());
          raise e)

let open_io (type a) (module I : Rig_edge.Io with type t = a) ?machine ~name
    make =
  let key = Type.Id.uid I.region_key in
  open_named ~fn:"open_io" ~machine ~name ~key ~host:false
  @@ fun ~index ~name:full ->
  match make () with
  | Error e -> Error e
  | Ok h -> (
      match I.budget h with
      | exception I.Fault why ->
          (try I.stop h with I.Fault _ -> ());
          Error (strf "%s: %s" full why)
      | budget ->
          let c = c_io_new index full in
          let d =
            make_device ~index ~name:full ~machine
              ~kind:(Io { m = (module I); h })
              ~c ~arch:"" ~queues:[||] ~completion:Host_writes ~waits:no_waits
              ~hang_ms:None ~maps_host:false ~word:0
              ~word_region:None ~key ~memory_device:false
              ~fault:(function I.Fault why -> Some why | _ -> None)
              ~capability:None ~budget
          in
          publish d;
          Ok d)

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

(* Whether [d] is another machine's host. Holds the table's lock. *)
let is_machine_host d =
  match d.machine with
  | None -> false
  | Some m -> (
      match Hashtbl.find_opt machines m with Some h -> h == d | None -> false)

(* A machine's devices end before its host, so [host_of] of an open device is
   never closed: the machine stops taking opens, the opens in flight finish,
   then its devices close. *)
let close d =
  if is_host d then invalid_arg "Rig.close: the host is never closed";
  let machine =
    Lock.protect table_lock @@ fun () ->
    match d.machine with
    | Some m when is_machine_host d ->
        Hashtbl.replace closing m ();
        while opening_on m do
          Lock.wait table_lock
        done;
        Some (m, devices_of m d)
    | _ -> None
  in
  match machine with
  | None -> close_one d
  | Some (m, devices) ->
      Fun.protect
        ~finally:(fun () ->
          Lock.protect table_lock (fun () -> Hashtbl.remove closing m))
        (fun () ->
          List.iter close_one devices;
          close_one d)

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
     | Host, Driver _ -> d'.memory_device || d'.copy_queue = None
     | Driver _, Host -> d.maps_host
     | Driver _, Driver _ -> d'.memory_device || peer d d'
