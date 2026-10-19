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

external c_new : int -> string -> bool -> nativeint -> int -> c
  = "caml_rig_device_new"

external c_io_new : int -> string -> int -> c = "caml_rig_io_new"
external c_host_new : string -> c = "caml_rig_host_new"
external c_slot_word : int -> nativeint = "caml_rig_slot_word"
(* Publishes a device's record and marks the slot its open took published,
   negating it, in one call: the open's end gives back only an unpublished
   slot. *)
external c_publish : c -> int Atomic.t -> unit = "caml_rig_publish"
external c_committed : c -> int = "caml_rig_committed" [@@noalloc]
external c_commit : c -> int = "caml_rig_commit"
external c_settle : c -> int -> int = "caml_rig_settle"
external c_done : int -> bool = "caml_rig_done" [@@noalloc]
external c_lost : int -> bool = "caml_rig_lost" [@@noalloc]

(* The device whose lost range of its slot holds a point, if one does. *)
external c_range_device : int -> device option = "caml_rig_range_device"

(* The timeline of the device of a slot and an [opened] id, read from the
   slot's line while the line names it: the word and the last value given
   are -1 and the state Ended once a later device's record wrote the line,
   and [c_final] of its record answers the values. *)
external c_word : int -> int -> int = "caml_rig_word" [@@noalloc]
external c_submitted : int -> int -> int = "caml_rig_submitted" [@@noalloc]
external c_state : int -> int -> int = "caml_rig_state" [@@noalloc]
external c_final : c -> int = "caml_rig_final" [@@noalloc]
external c_why : c -> string = "caml_rig_why"
external c_base : c -> int = "caml_rig_base" [@@noalloc]

(* A slot's word, raised to the value read from the word of its device's
   own. *)
external c_raise_word : int -> int -> unit = "caml_rig_raise_word" [@@noalloc]

(* The first point of the stamps [st] that a lost device did not reach, among
   their uses' if [every] and every device's last write's otherwise, [0] for
   none. *)
external c_stamps_lost : int -> bool -> int = "caml_rig_stamps_lost"
[@@noalloc]
external c_lose : c -> string -> bool -> bool = "caml_rig_lose"

(* [caml_rig_raised] reads a scope as [Faults] 0, [Any] 1, or [Any_but]'s
   block. *)
type scope = Faults | Any | Any_but of Obj.Extension_constructor.t

external c_raised : c -> exn -> Obj.Extension_constructor.t -> scope -> bool
  = "caml_rig_raised"

external c_faulted : c -> bool = "caml_rig_faulted" [@@noalloc]
external c_stopped : c -> unit = "caml_rig_stopped"
external c_await_stop : c -> int -> bool = "caml_rig_await_stop"
external c_free : c -> device -> bool = "caml_rig_free"
external c_next_value : c -> int = "caml_rig_next_value"
external c_claim_own : c -> bool = "caml_rig_claim_own"
external c_unclaim : c -> unit = "caml_rig_unclaim"
external c_fail : string -> bool = "caml_rig_fail"
external c_failed : unit -> string option = "caml_rig_failed"
external c_first : unit -> int = "caml_rig_first" [@@noalloc]
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
let owed_state = 2
let ended_state = 4
let orphaned_state = 5

(* The host's clock, [rig_now_ns] of [rig_edge.h]. *)
external now_ns : unit -> (int[@untagged])
  = "caml_rig_edge_now_byte" "caml_rig_edge_now"
[@@noalloc]

external still : unit -> int = "caml_rig_still_ms" [@@noalloc]

(* How long a wait sees no progress before it blocks in the driver, and the
   longest a blocking call of a wait lasts: each returns to OCaml within it,
   where a pending [Sys.Break] raises. [rig_device.c]'s STILL_MS. *)
let still_ms = still ()

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

let no_waits = { Rig_edge.stores = false; hosts = false; most = 0 }

let make_device ~index ~name ~machine ~kind ~c ~arch ~queues ~copy_engine
    ~completion ~waits ~hang_ms ~maps_host ~alignment ~host_addresses ~word
    ~word_region ~key ~memory_device ~fault ~capability ~budget ~opened =
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
      word_unmaps = 1;
      fault;
      release = c_release c;
      lock = Lock.create ();
      used = 0;
      cached = 0;
      cache = Cache.create 16;
      retiring = [||];
      nretiring = 0;
      judged = 0;
      freeing = false;
      pending = ([], []);
      pairs = [||];
      words = 0;
      word_maps = [];
      mapped = Cache.create 4;
      slots = [||];
      claims = 0;
    }
  in
  {
    index;
    name;
    machine;
    kind;
    arch;
    maps_host;
    alignment;
    host_addresses;
    key;
    memory_device;
    capability;
    budget;
    closing = 0;
    locals = [];
    live;
    base = c_base c;
    opened;
  }

(* The host's fault, which nothing raises. *)
exception No_fault

(* The alignment of the C library's [malloc] on every 64-bit host, which a host
   buffer starts on at least ([rig_heap.c]). *)
let malloc_alignment = 16

(* The host runs launches of linked code ([rig_host_code.c]) on its one queue,
   and waits on the host for other devices' points. *)
let host =
  let queues = [| { Rig_edge.name = "COMPUTE:0"; runs = [ Launch ] } |] in
  let d =
    make_device ~index:0 ~name:"CPU" ~machine:Here ~kind:Host
      ~c:(c_host_new "CPU") ~arch:(host_arch ()) ~queues ~copy_engine:None
      ~completion:Rig_edge.Host ~waits:no_waits ~hang_ms:None ~maps_host:false
      ~alignment:malloc_alignment ~host_addresses:false
      ~word:(Nativeint.to_int (c_slot_word 0))
      ~word_region:None ~key:(-1)
      ~memory_device:false
      ~fault:[%extension_constructor No_fault]
      ~capability:None ~budget:max_int ~opened:0
  in
  d

(* The table of devices *)

(* Every device by its slot: the last device that took the slot, which keeps
   it until a later one does; the host for a slot no device took. *)
let devices = Table.make host

let of_index i = Table.get devices i
let register d = Table.set devices d.index d

(* Whether [d] is the device of the point [p]: [p] lies on [d]'s slot, after
   the value [d]'s timeline started after. *)
let[@inline] owns d p = d.index = Point.index p && Point.value p > d.base

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

(* Whether [d] is lost. The host is never lost: a use of its memory reads no C
   record. *)
let[@inline] state (d : device) = c_state d.index d.opened
let[@inline] is_lost d = d.index <> 0 && state d <> live_state
let orphaned d = state d = orphaned_state

let why d = c_why d.live.c
let lost d = if is_lost d then Some (why d) else None
let raise_lost d = raise (Lost (d, why d))

(* [d]'s loss as {!Lost} prints it. *)
let printed d = strf "%s lost: %s" d.name (why d)

let submitted (d : device) =
  let v = c_submitted d.index d.opened in
  if v >= 0 then v else c_final d.live.c

(* [d]'s slot word, or its record's final value once its slot passed on. *)
let slot_word (d : device) =
  let w = c_word d.index d.opened in
  if w >= 0 then w else c_final d.live.c
let committed d = c_committed d.live.c

(* One past [d]'s last value, read under its C mutex: a value that every
   submit whose waits were checked before it took, or will take ([+ 1] covers
   the submit between its check and its value). *)
let next_value d = c_next_value d.live.c

(* Locks *)

let protect d f x = Lock.protect d.live.lock f x

(* The locks the calling thread holds, [rig_holds] of [rig_edge.h]. *)
external holds : unit -> (int[@untagged])
  = "caml_rig_edge_holds_byte" "caml_rig_edge_holds"
[@@noalloc]

let refuse_held fn =
  invalid_arg
    ("Rig." ^ fn
   ^ ": called from a signal handler, finaliser or Gc.Memprof callback \
      inside a lock of the calling thread's")

let[@inline] check_held fn = if holds () <> 0 then refuse_held fn

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

(* A key's value on a device lies in the device's [locals], under the key's
   id, so a later device of its slot starts with none and an init that runs
   past its device's stop finds no other device's value. [locals] is replaced
   whole under [locals_lock], so a get that finds its value reads it without
   the lock. A store on a lost device keeps nothing and raises, and the stop
   empties [locals] after the loss, so no value outlives the stop. *)
type 'a key = { init : device -> 'a; id : 'a Type.Id.t }

let locals_lock = Lock.create ()
let new_key init = { init; id = Type.Id.make () }

let rec find : type a. a key -> local list -> a option =
 fun k -> function
  | [] -> None
  | Local (id, v) :: l -> (
      match Type.Id.provably_equal k.id id with
      | Some Equal -> Some v
      | None -> find k l)

(* [k]'s value on [d], storing [v] unless a value came first: [None] if [d]
   is lost. *)
let store_locked (k, d, v) =
  match find k d.locals with
  | Some _ as kept -> kept
  | None when is_lost d -> None
  | None ->
      d.locals <- Local (k.id, v) :: d.locals;
      Some v

let store k d v =
  match Lock.protect locals_lock store_locked (k, d, v) with
  | Some v -> v
  | None -> raise_lost d

(* [k]'s value among [l], made if absent. It finds a value without
   allocating. *)
let rec value_in : type a. a key -> device -> local list -> a =
 fun k d -> function
  | [] ->
      check_held "Local.get";
      store k d (k.init d)
  | Local (id, v) :: l -> (
      match Type.Id.provably_equal k.id id with
      | Some Equal -> v
      | None -> value_in k d l)

let local k d =
  if is_lost d then raise_lost d;
  value_in k d d.locals

let drop_locked d = d.locals <- []
let drop_values d = Lock.protect locals_lock drop_locked d

(* The table of names *)

(* An open of a name: its slot in the table of names is [k], the machine's
   name and the device's. [thread] is the number of the thread that runs it.
   [opening] says whether the slot reads [Opening] for it, [index] holds the
   device slot it took, [0] for none and negated once its device is
   published, which the C stubs that take, give back and publish store,
   [ended] says that an exception ended it, and [id] is unique for the
   process, [-1] until the open took its slot. *)
type request = {
  fn : string;
  machine : machine;
  k : string option * string;
  full : string;
  key : int;
  host : bool;
  thread : int;
  mutable opening : bool;
  index : int Atomic.t;
  mutable ended : bool; [@atomic]
  mutable id : int;
}

(* Takes a device slot and stores it in [a] ([0] if every slot is held), and
   gives [a]'s back, zeroing it. Each runs no OCaml code, so no asynchronous
   exception comes between the move of the slot and its record. *)
external c_take : int Atomic.t -> int = "caml_rig_take"
external c_give : int Atomic.t -> unit = "caml_rig_give"

(* The calling thread's number, which no other thread of the process has. *)
external c_thread : unit -> int = "caml_rig_thread" [@@noalloc]

(* Drains every device ({!Memory.drain_all}), which may free slots. *)
let drain_all : (unit -> unit) ref = ref ignore

type slot = Opening of request | Open of device

module Names = Tbl.Make (struct
  type t = string option * string

  let equal = ( = )
  let hash = Hashtbl.hash
end)

let table : slot Names.t = Names.create 16
let opens = Atomic.make 1

(* Under the table's lock: removes [d]'s names, so an open of them makes a new
   device. A stop removes them, so the table holds the devices whose stop has
   not returned. *)
let forget_locked d =
  let mine k slot ks =
    match slot with Open d' when d' == d -> k :: ks | _ -> ks
  in
  List.iter (Names.remove table) (Names.fold mine table [])

(* Loss and stops *)

let answered : (device -> unit) ref = ref ignore

(* Reads a lost device's own word through its driver, which may be called
   after [stop], so that its slot's word moves on: only while its record owns
   its line, under its lock, which the drop of its own word unmap passes
   ({!Memory.let_go_word}), so that no read meets a freed word. *)
let refresh_owned d =
  match d.kind with
  | Driver { m; h; _ } when c_word d.index d.opened >= 0 -> (
      let module D = (val m) in
      match D.signaled h with
      | w -> c_raise_word d.index w
      | exception D.Fault _ -> ())
  | _ -> ()

let refresh d = if d.live.word = 0 then protect d refresh_owned d

let stop_returned (d : device) = state d >= ended_state

(* Whether the stop of [d] returned and its word shows its last submitted
   value: no work of it runs. An orphaned device's work is its parent's. *)
let stopped (d : device) =
  state d = ended_state
  && begin
    refresh d;
    slot_word d >= submitted d
  end

(* Frees [d]'s slot once [d] has stopped and nothing of it remains to give
   back: no token its drains have not routed, nothing in its cache or among
   its retiring memory, and no release waiting for a value.
   [caml_rig_free] checks the rest.
   Its record then answers as it stood, and a later device takes the slot,
   continuing its timeline. Whether this call freed it. *)
let free_slot d =
  let l = d.live in
  d.index > 0 && stopped d
  && l.cached = 0
  && l.nretiring = 0
  && fst l.pending == []
  && c_free l.c d

(* Calls the driver's stop, allocating nothing before it does. *)
let stop_handle d fault =
  match d.kind with
  | Driver { m; h; _ } -> (
      let module D = (val m) in
      try D.stop h ~fault with D.Fault _ -> ())
  | Io { m; h } -> (
      let module I = (val m) in
      try I.stop h with I.Fault _ -> ())
  | Host -> ()

(* Moves the state on after the driver's stop. Its values and its names go
   first, so a close, which waits for the state, returns after them, and the
   state moves also when they raise. *)
let returned d =
  match
    drop_values d;
    Lock.protect table_lock forget_locked d
  with
  | () -> c_stopped d.live.c
  | exception e ->
      c_stopped d.live.c;
      Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ())

(* Runs the stop of [d], which the caller claimed just before. It ends in no
   tail call, so the compiler puts no poll at its start, and nothing polls
   between the claim and its handlers. An exception before the driver's stop
   is called owes the stop again; from the driver's stop on, the state moves
   on whatever else it did: the device must count as stopped or not. A stop
   that raised may have left the device's work running, which asks nothing
   more of rig: its memory returns once it is Stopped, as its word shows its
   last value, which a driver writes only once no work of the device runs. *)
let run_stop d =
  (match if c_faulted d.live.c then Some (why d) else None with
  | exception e ->
      c_unclaim d.live.c;
      Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ())
  | fault -> (
      match stop_handle d fault with
      | () -> returned d
      | exception e ->
          returned d;
          Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ())));
  !answered d;
  ()

(* Runs [d]'s stop if it is owed and no counted call runs on [d]: whether
   this call ran it. *)
let own_stop d =
  state d = owed_state
  && c_claim_own d.live.c
  && begin
    run_stop d;
    true
  end

let run_own d = if state d = owed_state then ignore (own_stop d : bool)

(* A wait's hang loss runs no stop: a wait does no device's work. *)
let lose d why =
  ignore (c_lose d.live.c why true);
  raise_lost d

let leave d =
  c_exit d.live.c;
  run_own d

(* Ends the counted call on [d] that raised [e]. Its first step is one C call
   that loses [d] for a fault or an exception in [scope] and ends the count,
   and nothing allocates or polls before it: a second exception landing in
   this handler finds both done. It returns for a fault, and raises another
   exception again. A wait's call runs no stop ([wait_raised]). *)
let raised d scope e =
  let fault = c_raised d.live.c e d.live.fault scope in
  let bt = Printexc.get_raw_backtrace () in
  run_own d;
  if not fault then Printexc.raise_with_backtrace e bt

(* A wait's fault loses [d], and another exception passes through. *)
let wait_raised d e =
  let fault = c_raised d.live.c e d.live.fault Faults in
  let bt = Printexc.get_raw_backtrace () in
  if fault then raise_lost d else Printexc.raise_with_backtrace e bt

let refused d =
  run_own d;
  raise_lost d

let counted d scope f =
  match c_enter d.live.c with
  | 0 -> (
      match f () with
      | r ->
          leave d;
          r
      | exception e ->
          raised d scope e;
          raise_lost d)
  | _ -> refused d

type given = Given | Later

(* On a stopped device a failure of the call, its fault or its memory's, gives
   back nothing more. On a live one a fault loses the device, which its caller
   never sees: a give is no use of the device. An object that did not go back
   is still in the caller's hand, so [d]'s owed stop is the caller's to run
   once the object is back in its place: the stop's drain would otherwise
   give back what waits behind it first, such as an image's code memory
   behind its unload. *)
let give d scope f =
  if orphaned d then Given
  else if stopped d then
    match f () with
    | () -> Given
    | exception Sys_error _ -> Given
    | exception e when Obj.Extension_constructor.of_val e == d.live.fault ->
        Given
  else if c_enter d.live.c <> 0 then Later
  else
    match f () with
    | () ->
        leave d;
        Given
    | exception e ->
        raised d scope e;
        Given

(* A [fail] from a callback inside a lock fails the process and leaves the
   owed stops, which take locks, to later calls. *)
let fail why = if c_fail why && holds () = 0 then iter run_own

let failure () =
  match c_first () with
  | 0 -> None
  | -1 -> c_failed ()
  | i -> Some (printed (of_index i))

(* The timeline *)

(* A word of the device's own is read through a counted call, written out so
   that a read allocates nothing. *)
let word d =
  let l = d.live in
  if l.word <> 0 || is_lost d then slot_word d
  else
    match d.kind with
    | Driver { m; h; _ } ->
        let module D = (val m) in
        if c_enter l.c <> 0 then raise_lost d;
        let w =
          match D.signaled h with
          | w ->
              c_exit l.c;
              w
          | exception e -> wait_raised d e
        in
        c_raise_word d.index w;
        w
    | _ -> slot_word d

let now_ms () = now_ns () / 1_000_000

(* How long a sleep may watch the word [w] under [d]'s hang bound [hang], or a
   loss of [d] once the bound passed. The clock restarts when the word moved or
   [d] was idle at this look or the last, and runs on while the same committed
   value stays above the word. A device whose queue waits for a producer's
   unreached value counts as idle: its own work may not have started. The word
   is read again here, through the driver where it has no host address: [w]
   was read before a commit that may have waited for the turn as long as the
   still interval. *)
let bounded (d : device) w ~still_ms hang =
  let w = Int.max w (word d) in
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
      if c_enter l.c <> 0 then raise_lost d;
      match D.sleep h ~seen ~still_ms with
      | () -> c_exit l.c
      | exception e -> wait_raised d e)
  | _ -> ()

module Answer = struct
  let ok = 0
  let busy = 1
  let no_room = 2
  let never = 3
  let producer_lost = 6
  let failed = 7
  let no_memory = 8
  let grow = 9
end

(* Commits [d]'s submitted work if [d]'s turn frees within the still interval:
   whether it did, or found [d] lost. A commit that fails has lost [d], whose
   stop is then owed: a wait's commit leaves it so. *)
let commit_once d = c_commit d.live.c <> Answer.busy

let rec commit d v =
  if committed d < v then
    let r = c_commit d.live.c in
    if r = Answer.failed then run_own d
    else if r = Answer.busy then commit d v

let rec settle d v =
  let r = c_settle d.live.c v in
  if r = Answer.failed then run_own d;
  if r = Answer.busy then settle d v else r = Answer.ok

let signaled = word

(* The word of a live device, or [-1] once a read of a word of its driver's
   own found it lost, which its line then answers for. A word in rig's array
   is read with no handler, as no read of it raises. *)
let live_word d =
  if d.live.word <> 0 then slot_word d
  else try word d with Lost (d', _) when d' == d -> -1

(* A device whose unreached work waits in its queue on another's surfaces that
   producer's unseen fault: its sleep raises it. A producer is the device its
   slot holds, as no entry of an earlier device's points is unreached. *)
let look_at_producers d =
  Array.iter
    (fun i ->
      let p = of_index i in
      if not (is_lost p) then
        try sleep p ~seen:(word p) ~still_ms:0 with Lost _ -> ())
    (c_producers d.live.c)

(* Reads [d]'s word until it reaches [v] or the clock [until]: the word. *)
let rec spin_held (d : device) v until =
  let w = slot_word d in
  if w >= v || now_ns () >= until then w
  else begin
    Domain.cpu_relax ();
    spin_held d v until
  end

(* Waits for [d]'s value [v], the word having read [seen] since [since],
   until the word reads [v] or [d] is lost. Until [v] is committed, each round
   commits [d]'s work unless another holds [d]'s turn throughout the still
   interval, such as a submit blocked in its driver, whose earlier work runs
   without another call, so the turn frees and a later round commits. Top
   level, so a wait that finds its value reached allocates nothing. *)
let rec wait_from d v seen since =
  let w = word d in
  if w < v && not (is_lost d) then begin
    if committed d < v then ignore (commit_once d : bool);
    let now = now_ms () in
    let since = if w <> seen then now else since in
    let still = now - since in
    let l = d.live in
    (match l.completion with
    | Rig_edge.Host -> sleep d ~seen:w ~still_ms
    | Store when l.word = 0 -> sleep d ~seen:w ~still_ms
    | Store when still >= still_ms ->
        look_at_producers d;
        sleep d ~seen:w ~still_ms
    | Store ->
        if spin_held d v (now_ns () + held_ns) < v then
          ignore (c_spin l.c v (still_ms - still)));
    wait_from d v w since
  end

(* How a wait blocks depends on who writes the word. A word the host writes
   ([Host]), or one with no host address, changes only on an event the driver
   sees, so the wait sleeps in the driver from its first read. A word a queue
   writes is read with the domain lock held for [held_ns], then spun on with
   the lock released, and once it stood still for [still_ms] the wait sleeps
   in the driver between reads. *)
let wait_value d v =
  let w = word d in
  if w < v && not (is_lost d) then wait_from d v w (now_ms ());
  if is_lost d then raise_lost d

let wait d v =
  let s = submitted d in
  if v > s then invalid_argf "Rig: %d is beyond %s's last value %d" v d.name s;
  wait_value d v

(* Waits for [d]'s value [v] from the word [w], until [d] is lost: a loss its
   sleep raises is judged by the caller. *)
let wait_until_lost d v w =
  try wait_from d v w (now_ms ()) with Lost (d', _) when d' == d -> ()

(* Points

   A point's slot answers it from its line ([rig_point_done],
   [rig_point_lost]) whatever device the slot holds now. Only the device that
   owns a point ([owns]) is asked for it. *)

(* Raises [Lost] for the point [p] that a lost device did not reach: the
   device of [p]'s slot, or the earlier one whose lost range holds [p]. A lost
   point of an earlier device lies in a range: the free leaves one for every
   value it dropped. *)
let raise_point_lost p =
  let d = of_index (Point.index p) in
  if owns d p then raise_lost d
  else
    match c_range_device p with
    | Some d -> raise_lost d
    | None -> invalid_arg "Rig: an earlier device's lost point lies in no range"

let check p = if c_lost p then raise_point_lost p

(* A lost device's point is done if its word reached it before the loss: its
   stop raises the word whatever ran. The word is read before the loss is, so a
   word the stop raised shows the loss. Before it blocks, it raises the loss of
   any point of the stamps [st] a lost device did not reach, so that it waits
   out no work ahead of a loss. *)
let wait_point st every p =
  let d = of_index (Point.index p) in
  if not (owns d p) then check p
  else
    let v = Point.value p in
    let w = live_word d in
    if w < v && not (is_lost d) then begin
      let q = c_stamps_lost st every in
      if q <> 0 then raise_point_lost q;
      wait_until_lost d v w
    end;
    if is_lost d then check p

(* The word is read before the loss is, as [wait_point] reads them. *)
let is_done p =
  let d = of_index (Point.index p) in
  if not (owns d p) then c_done p
  else
    let w = live_word d in
    if is_lost d then c_done p else w >= Point.value p

(* [wait_point]'s reads and commit without its wait: a done point answers
   before any commit, and a point not done raises [Lost] once its device is
   lost. *)
let reached_point p =
  if is_done p then true
  else
    let d = of_index (Point.index p) in
    if owns d p && committed d < Point.value p && not (is_lost d) then
      ignore (commit_once d : bool);
    if is_done p then true
    else (
      check p;
      false)

let settled p =
  let d = of_index (Point.index p) in
  if not (owns d p) then true
  else if is_lost d then orphaned d || slot_word d >= Point.value p
  else match word d >= Point.value p with r -> r | exception Lost _ -> false

(* Closing *)

(* Runs the stop if it is owed and claimable, and waits for it, returning to
   OCaml every still interval. *)
let rec await_stop d =
  ignore (own_stop d : bool);
  if not (c_await_stop d.live.c still_ms) then await_stop d

let close_one d =
  (match d.kind with
  | Driver _ when not (is_lost d) -> (
      try wait d (submitted d) with Lost (d', _) when d' == d -> ())
  | _ -> ());
  ignore (c_lose d.live.c "closed" false);
  await_stop d

(* Opening *)

(* The device [r] published, if any: [r]'s slot holds a device of [r]'s open,
   which may have stopped and given the slot to a later one. *)
let made r =
  let i = abs (Atomic.get r.index) in
  if i <> 0 && c_next i = i && (of_index i).opened = r.id then Ok (of_index i)
  else Error ""

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
  | Error _ ->
      c_give r.index;
      Names.remove table r.k);
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

(* A failed process opens nothing, not calling [make]. A lost device whose
   stop has not returned is [`Stop], for the open to run the stop. An open
   of a name another thread is opening is [`Wait]; one of a name that [r]'s
   own thread is opening, from its opener or a callback inside that open,
   would wait for good: it raises. *)
let rec find r =
  match (c_failed (), host_down r.machine, Names.find_opt table r.k) with
  | Some why, _, _ | None, Some why, _ -> `Error why
  | None, None, Some (Opening o) when o.ended ->
      end_open (o, made o);
      find r
  | None, None, Some (Opening o) when o.thread = r.thread ->
      invalid_argf "Rig.%s: %s is opening on the calling thread" r.fn r.full
  | None, None, Some (Opening _) -> `Wait
  | None, None, Some (Open d) when not (is_lost d) ->
      if d.key <> r.key then
        invalid_argf "Rig.%s: %s is open as another driver's device" r.fn
          r.full;
      `Open d
  | None, None, Some (Open d) -> if stop_returned d then `Make else `Stop d
  | None, None, None -> `Make

(* Under the table's lock: the device open under [r]'s name, or the slot
   claimed for [r], or [None] while another thread opens the name. *)
let claim r =
  end_interrupted ();
  check_host r;
  match find r with
  | `Wait -> None
  | `Make ->
      r.opening <- true;
      Names.replace table r.k (Opening r);
      if r.host then
        Option.iter (fun m -> Strings.replace hosting m r.full) (fst r.k);
      Some `Make
  | (`Open _ | `Stop _ | `Error _) as found -> Some found

(* [claim] of [r], after running the owed stop of a lost device that holds
   its name. A stop takes the table's lock as it returns, so it runs holding
   none. *)
let rec claimed r =
  match Lock.await table_lock claim r with
  | `Stop d when own_stop d -> claimed r
  | `Stop _ -> `Error (strf "%s is lost and its stop has not answered" r.full)
  | (`Open _ | `Error _ | `Make) as found -> found

(* Opens the device [name] on [machine] with [make], which builds it from its
   index and full name, once no open of that name runs. From the moment the
   slot may read [Opening], an exception, such as one a signal handler raises
   at an allocation, ends the open as it stands: a device it published stays
   open under the name, and otherwise the name is free. [make] ends what it
   made and did not publish. *)
let open_named ~fn ~machine ~name ~key make =
  check_held fn;
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
      thread = c_thread ();
      opening = false;
      index = Atomic.make 0;
      ended = false;
      id = -1;
    }
  in
  let finish result =
    Lock.protect table_lock end_open (r, result);
    result
  in
  match
    match claimed r with
    | `Open d -> Ok d
    | `Error e -> Error e
    | `Make ->
        let index =
          match c_take r.index with
          | 0 ->
              (* Every slot is held: a collection may return the last
                 memory of lost devices, whose drains then run their stops
                 and free their slots. *)
              Gc.full_major ();
              !drain_all ();
              c_take r.index
          | i -> i
        in
        if index = 0 then
          finish
            (Error
               (strf "%s: %d devices hold every device slot" full
                  Point.max_index))
        else begin
          (* Taken after the slot, so that a slot's later device has a larger
             id: one that took an id before waiting for the slot would not. *)
          r.id <- Atomic.fetch_and_add opens 1;
          finish (make ~index ~slot:r.index ~opened:r.id ~name:full)
        end
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

(* Publishes the C record of [d], which is in the tables: spreads, failures and
   the fork handler walk it for good. A device lost at birth by a failure of the
   process is stopped here. *)
let publish d slot =
  c_publish d.live.c slot;
  run_own d

(* The device of the driver's handle [h], in the tables and not published.
   Every fact is read before the C record is published: a fault reading one
   leaves nothing made. *)
let driver_device (type a) (module D : Rig_edge.Driver with type t = a)
    (h : a) ~index ~opened ~name ~machine ~memory_device =
  let m : (a, D.region, D.image) dm = (module D) in
  let rid : D.region Type.Id.t = Type.Id.make () in
  let f = D.facts h in
  Option.iter
    (fun n ->
      if n < 1 then
        invalid_argf "Rig.open_: %s's hang bound is %d ms, expected at least 1"
          name n)
    f.hang_ms;
  if f.alignment < 1 || f.alignment land (f.alignment - 1) <> 0 then
    invalid_argf
      "Rig.open_: %s's regions start on multiples of %d, expected a power of \
       two"
      name f.alignment;
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
  (* A word other than the one [make] was given is the driver's own, which
     rig reads through [signaled]. *)
  let word =
    let slot = Nativeint.to_int (c_slot_word index) in
    if (D.locate f.word).host = Some slot then slot else 0
  in
  let c = c_new index name f.may_block f.edge opened in
  let d =
    make_device ~index ~name ~machine
      ~kind:(Driver { m; h; rid })
      ~c ~arch:f.arch ~queues ~copy_engine:f.copy_engine
      ~completion:f.completion
      ~waits:f.waits ~hang_ms:f.hang_ms ~maps_host:f.maps_host
      ~alignment:f.alignment ~host_addresses:f.host_addresses ~word
      ~word_region:(Some (Region { m; h; r = f.word; rid }))
      ~key:(Type.Id.uid D.key) ~memory_device
      ~fault:[%extension_constructor D.Fault]
      ~capability:(Some f.capability) ~budget:f.budget ~opened
  in
  register d;
  d

let open_driver (type a) ?(memory_device = false)
    (module D : Rig_edge.Driver with type t = a) ~machine ~name make =
  let fn = if memory_device then "memory_device" else "open_" in
  open_named ~fn ~machine ~name ~key:(Type.Id.uid D.key)
  @@ fun ~index ~slot ~opened ~name:full ->
  match make ~word:(c_slot_word index) with
  | Error e -> Error e
  | Ok h -> (
      (* Until the device is published, a failure stops the handle: a fault is
         a loss, and any other exception, such as facts that break the
         contract, is raised again. *)
      match
        driver_device (module D) h ~index ~opened ~name:full ~machine
          ~memory_device
      with
      | d ->
          publish d slot;
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
  @@ fun ~index ~slot ~opened ~name:full ->
  match make () with
  | Error e -> Error e
  | Ok h -> (
      (* As for a driver's device. *)
      match
        let budget = I.budget h in
        let c = c_io_new index full opened in
        let d =
          make_device ~index ~name:full ~machine
            ~kind:(Io { m = (module I); h })
            ~c ~arch:"" ~queues:[||] ~copy_engine:None ~completion:Rig_edge.Host
            ~waits:no_waits ~hang_ms:None ~maps_host:false ~alignment:1
            ~host_addresses:false ~word:0 ~word_region:None ~key
            ~memory_device:false ~fault:[%extension_constructor I.Fault]
            ~capability:None ~budget ~opened
        in
        register d;
        d
      with
      | d ->
          publish d slot;
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
  Names.fold
    (fun (m', _) slot acc ->
      match slot with Open d when m' = Some m && d != h -> d :: acc | _ -> acc)
    table []

(* Whether an open on machine [m] is in flight. Raises [Invalid_argument]
   for one the calling thread runs: a close of [m]'s host from its opener
   or from a callback inside it would wait for good. *)
let opening_on m =
  let self = c_thread () in
  Names.fold
    (fun (m', _) slot acc ->
      match slot with
      | Opening r when m' = Some m ->
          if r.thread = self then
            invalid_argf "Rig.close: %s is opening on the calling thread"
              r.full;
          true
      | Opening _ | Open _ -> acc)
    table false

(* Under the table's lock: [m]'s devices other than its host [d] once no
   open is in flight on [m], ending those an exception ended, or [None]. *)
let opens_ended ((d : device), m) =
  end_interrupted ();
  if opening_on m then None else Some (devices_of m d)

(* Marks the host [d] of machine [m] closing, waits for the opens in flight
   on [m], and is [m]'s other devices. An exception before it returns, such
   as [Sys.Break] in a wait, takes the close back by its handler's first
   step, an atomic one: a call there could take a second exception first.
   Another close of [d] under way keeps its own. *)
let start_close (d : device) m =
  match
    ignore (Atomic.Loc.fetch_and_add [%atomic.loc d.closing] 1);
    Lock.await table_lock opens_ended (d, m)
  with
  | devices -> devices
  | exception e ->
      ignore (Atomic.Loc.fetch_and_add [%atomic.loc d.closing] (-1));
      Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ())

(* A machine's devices end before its host, so [host_of] of an open device is
   never closed: the machine stops taking opens, the opens in flight finish,
   then its devices close. *)
let close d =
  check_held "close";
  if is_host d then invalid_arg "Rig.close: the host is never closed";
  match d.machine with
  | Here | On _ -> close_one d
  | Hosts m -> (
      let devices = start_close d m in
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
     | Driver _, Driver _ -> (d'.memory_device && d.maps_host) || peer d d'
