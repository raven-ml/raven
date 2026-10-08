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

external c_new :
  int ->
  string ->
  bool ->
  nativeint ->
  nativeint ->
  nativeint ->
  nativeint ->
  int = "caml_rig_device_new_byte" "caml_rig_device_new"

external c_io_new : int -> string -> int = "caml_rig_io_new"
external c_word : int -> int = "caml_rig_word" [@@noalloc]
external c_set_seen : int -> int -> unit = "caml_rig_set_seen" [@@noalloc]
external c_submitted : int -> int = "caml_rig_submitted" [@@noalloc]
external c_is_lost : int -> bool = "caml_rig_is_lost" [@@noalloc]
external c_why : int -> string = "caml_rig_why"
external c_answer : int -> int = "caml_rig_answer" [@@noalloc]
external c_inherited : int -> bool = "caml_rig_inherited" [@@noalloc]
external c_lose : int -> string -> int array = "caml_rig_lose"
external c_set_answer : int -> int -> unit = "caml_rig_set_answer" [@@noalloc]
external c_upgrade : int -> bool = "caml_rig_upgrade" [@@noalloc]
external c_enter : int -> int = "caml_rig_enter"
external c_exit : int -> bool = "caml_rig_exit"
external c_spin : int -> int -> int -> int = "caml_rig_spin"
external c_producers : int -> int array = "caml_rig_producers"
external release_list : unit -> int = "caml_rig_release_list"
external host_arch : unit -> string = "caml_rig_arch"

(* A stop's answers, as the C record holds them. *)
let answer_stopping = 1
let answer_stopped = 2
let answer_unknown = 3

(* How long a wait sees no progress before it blocks in the driver, and the
   longest a blocking call of a wait lasts: each returns to OCaml within it,
   where a pending [Sys.Break] raises. *)
let still_ms = 200

(* The table of devices *)

let devices : device array Atomic.t = Atomic.make [||]
let of_index i = (Atomic.get devices).(i)
let all () = Atomic.get devices

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

let make_device ~index ~name ~machine ~kind ~c ~arch ~queues ~completion ~waits
    ~word ~word_region ~key ~memory_device ~fault ~capability ~budget =
  let waits_store, waits_object, waits_host = waits in
  {
    index;
    name;
    machine;
    kind;
    c;
    arch;
    queues;
    completion;
    waits_store;
    waits_object;
    waits_host;
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
    cache = Hashtbl.create 16;
    retiring = [];
    pending = [];
    pairs = [||];
    pair_maps = [];
    afters = [];
  }

let host =
  let d =
    make_device ~index:0 ~name:"CPU" ~machine:None ~kind:Host ~c:0
      ~arch:(host_arch ()) ~queues:[||] ~completion:Host_writes
      ~waits:(false, false, false) ~word:0 ~word_region:None ~key:(-1)
      ~memory_device:false
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
let busy d = Lock.busy d.lock

(* Facts *)

let is_host d = match d.kind with Host -> true | _ -> false
let is_io d = match d.kind with Io _ -> true | _ -> false
let inherited d = d.c <> 0 && c_inherited d.c
let is_lost d = d.c <> 0 && c_is_lost d.c
let lost d = if is_lost d then Some (c_why d.c) else None
let raise_lost d = raise (Lost (d, c_why d.c))
let submitted d = if d.c = 0 then 0 else c_submitted d.c
let answer d = if d.c = 0 then 0 else c_answer d.c
let copies d = Array.exists (String.starts_with ~prefix:"COPY:") d.queues
let machines : (string, device) Hashtbl.t = Hashtbl.create 4
let table_lock = Lock.create ()

let host_of d =
  match d.machine with
  | None -> host
  | Some m -> (
      match Lock.protect table_lock (fun () -> Hashtbl.find_opt machines m) with
      | Some h -> h
      | None -> d)

(* Loss and stops *)

(* Records what the driver's stop of the claimed [d] answers, then frees what
   [d]'s answer makes due ([answered], set by the memory module). *)
let answered : (device -> unit) ref = ref ignore

(* Reads a lost device's word behind a transport through its driver, which may
   be called after [stop], so the core's copy of it moves on. *)
let refresh d =
  match d.kind with
  | Driver { m; h; _ } when d.word = 0 -> (
      let module D = (val m) in
      match D.signaled h with w -> c_set_seen d.c w | exception _ -> ())
  | _ -> ()

let upgrade d =
  d.c <> 0
  && begin
    refresh d;
    c_upgrade d.c
  end

let stop d =
  (match d.kind with
  | Driver { m; h; _ } -> (
      let module D = (val m) in
      try D.stop h with _ -> ())
  | Io { m; h } -> (
      let module I = (val m) in
      try I.stop h with _ -> ())
  | Host -> ());
  c_set_answer d.c answer_unknown;
  (match d.kind with
  | Driver _ -> ignore (upgrade d)
  | _ -> c_set_answer d.c answer_stopped);
  !answered d

let stop_claimed indices =
  for i = 1 to Array.length indices - 1 do
    stop (of_index indices.(i))
  done

let lose d why =
  stop_claimed (c_lose d.c why);
  raise_lost d

let leave d = if c_exit d.c then stop d

(* [f ()] as a counted call on [d]: a fault it raises loses [d]. *)
let counted d f =
  if d.c = 0 then f ()
  else
    match c_enter d.c with
    | 0 -> (
        match f () with
        | r ->
            leave d;
            r
        | exception e -> (
            leave d;
            match d.fault e with Some why -> lose d why | None -> raise e))
    | 3 ->
        stop d;
        raise_lost d
    | _ -> raise_lost d

(* The timeline *)

(* The last value [d]'s word showed: read at its host address, or through the
   driver behind a transport, whose last answer a lost device keeps. *)
let word d =
  if d.c = 0 then 0
  else if d.word <> 0 || c_is_lost d.c then c_word d.c
  else
    match d.kind with
    | Driver { m; h; _ } ->
        let module D = (val m) in
        let w = counted d (fun () -> D.signaled h) in
        c_set_seen d.c w;
        w
    | _ -> c_word d.c

let signaled d = if d.c = 0 then 0 else word d

let sleep d ~seen ~still_ms =
  match d.kind with
  | Driver { m; h; _ } ->
      let module D = (val m) in
      counted d (fun () -> D.sleep h ~seen ~still_ms)
  | _ -> ()

let now_ms () = Prof.now () / 1_000_000

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

let reached d w =
  if is_lost d then raise_lost d;
  if d.afters <> [] then run_afters d w

(* Waits for [d]'s value [v], the word having read [seen] since [since]. Top
   level, so a wait that finds its value reached allocates nothing. *)
let rec wait_from d v seen since =
  let w = word d in
  if w >= v then reached d w
  else begin
    if is_lost d then raise_lost d;
    let now = now_ms () in
    let since = if w <> seen then now else since in
    let still = now - since in
    (match d.completion with
    | Host_writes -> sleep d ~seen:w ~still_ms
    | (Store | Object _) when d.word = 0 -> sleep d ~seen:w ~still_ms
    | (Store | Object _) when still >= still_ms ->
        look_at_producers d;
        sleep d ~seen:w ~still_ms
    | Store | Object _ -> ignore (c_spin d.c v (still_ms - still)));
    wait_from d v w since
  end

let wait d v =
  let s = submitted d in
  if v > s then
    invalid_argf "Rig.wait: %d is beyond %s's last value %d" v d.name s;
  let w = word d in
  if w >= v then reached d w else wait_from d v w (now_ms ())

(* Whether [p] is reached: its device's word reads it. A forked child reads no
   word of a device it inherited. *)
let point_reached p =
  let d = of_index (Point.index p) in
  if d.c = 0 then true
  else if c_inherited d.c then false
  else word d >= Point.value p

(* Opening *)

type slot = Opening | Open of device

let table : (string option * string, slot) Hashtbl.t = Hashtbl.create 16
let next_index = Atomic.make 1

(* Opens the device [name] on [machine] with [make], which builds it from its
   index and full name, once no open of that name runs. *)
let open_named ~machine ~name ~key ~host make =
  let k = (machine, name) in
  let full = match machine with None -> name | Some m -> name ^ "@" ^ m in
  let rec find () =
    match Hashtbl.find_opt table k with
    | Some Opening ->
        Lock.wait table_lock;
        find ()
    | Some (Open d) when not (is_lost d) ->
        if d.key <> key then
          invalid_argf "Rig.open_: %s is open as another driver's device" full;
        `Open d
    | Some (Open d) ->
        let a = c_answer d.c in
        if a = 0 || a = answer_stopping then
          `Error (strf "%s is lost and its stop has not answered" full)
        else `Make
    | None -> `Make
  in
  let found =
    Lock.protect table_lock (fun () ->
        match find () with
        | `Make ->
            Hashtbl.replace table k Opening;
            `Make
        | r -> r)
  in
  match found with
  | `Open d -> Ok d
  | `Error e -> Error e
  | `Make -> (
      let finish r =
        Lock.protect table_lock (fun () ->
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
            ignore (finish (Error ""));
            raise e)

let completion_of = function
  | `Store -> Store
  | `Object o -> Object (Nativeint.to_int o)
  | `Host -> Host_writes

let driver_device (type a) (module D : Sigs.Driver with type t = a) (h : a)
    ~index ~name ~machine ~memory_device =
  let m : (a, D.region, D.image) dm = (module D) in
  let rid : D.region Type.Id.t = Type.Id.make () in
  let word_region = D.word h in
  let word =
    match D.host word_region with Some p -> Nativeint.of_int p | None -> 0n
  in
  let c =
    c_new index name
      (D.blocks h = `May_block)
      (D.self h) D.room_entry D.submit_entry word
  in
  let d =
    make_device ~index ~name ~machine
      ~kind:(Driver { m; h; rid })
      ~c ~arch:(D.arch h)
      ~queues:(Array.of_list (D.queues h))
      ~completion:(completion_of (D.completion h))
      ~waits:(D.waits_on h `Store, D.waits_on h `Object, D.waits_on h `Host)
      ~word:(Nativeint.to_int word)
      ~word_region:(Some (Region { m; h; r = word_region; rid }))
      ~key:(Type.Id.uid D.key) ~memory_device
      ~fault:(function D.Fault why -> Some why | _ -> None)
      ~capability:(Some (Capability (D.capability_key, D.capability h)))
      ~budget:(D.budget h)
  in
  register d;
  d

let open_driver (type a) ?(memory_device = false)
    (module D : Sigs.Driver with type t = a) ?machine ~name make =
  open_named ~machine ~name ~key:(Type.Id.uid D.key) ~host:false
  @@ fun ~index ~name:full ->
  match make () with
  | Error e -> Error e
  | Ok h -> (
      match
        driver_device (module D) h ~index ~name:full ~machine ~memory_device
      with
      | d -> Ok d
      | exception D.Fault why -> Error (strf "%s: %s" full why))

let open_io (type a) (module I : Sigs.Io with type t = a) ?machine
    ?(host = false) ~name make =
  let key = Type.Id.uid I.region_key in
  open_named ~machine ~name ~key ~host @@ fun ~index ~name:full ->
  match make () with
  | Error e -> Error e
  | Ok h ->
      let c = c_io_new index full in
      let d =
        make_device ~index ~name:full ~machine
          ~kind:(Io { m = (module I); h })
          ~c ~arch:"" ~queues:[||] ~completion:Host_writes
          ~waits:(false, false, false) ~word:0 ~word_region:None ~key
          ~memory_device:false
          ~fault:(function I.Fault why -> Some why | _ -> None)
          ~capability:None ~budget:(I.budget h)
      in
      register d;
      Ok d

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
  || d.machine = d'.machine
     &&
     match (d.kind, d'.kind) with
     | Io _, _ | _, Io _ -> false
     | Host, Host -> false
     | Host, Driver _ -> d'.memory_device || not (copies d')
     | Driver _, Host -> true
     | Driver _, Driver _ -> d'.memory_device || peer d d'
