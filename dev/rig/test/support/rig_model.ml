(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* rig's public API against a reference written from rig.mli, over every kind of
   device the machine has: the host, memory devices, Polled devices in several
   configurations, the disk and the GPUs present.

   The reference holds what rig.mli names: devices and whether they are lost,
   memories with their bytes, stamps, claims and holds, and the buffers over
   them. Every call takes effect in the reference at once, in program order. The
   system observes bytes only through rig's own order (Buffer.wait,
   Buffer.copy), so a correct rig shows the reference's bytes whenever its
   devices ran the work. Polled devices run only when the program runs them or a
   wait sleeps, which keeps that window open.

   Buffers, submissions and holds live in cells: a creation fills a cell, a drop
   empties it, and a call may use its operand for the last time, emptying its
   cell before the call, so the operand is unreachable during it. A collection
   is a command, so a counterexample shows where it fell. Outcomes rig.mli
   leaves open are judged: whether a faulted device's loss shows yet, a borrow a
   driver may refuse, a budget's refusal while memory may be cached.

   The caller's own exclusion is a lock per memory, taken by the calls that
   access bytes on the host: rig orders device work, and leaves host accesses to
   its caller. *)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled
module S = Rig_support

let strf = Printf.sprintf

(* Devices *)

type config = {
  label : string;
  capacity : int;
  copies : bool;
  visible : bool;
  transport : bool;
  peers : bool;
  objects : bool;
  waits : bool;
  unknown : bool;
  maps_host : bool;
}

let polled =
  {
    label = "polled";
    capacity = 1024;
    copies = true;
    visible = true;
    transport = false;
    peers = true;
    objects = false;
    waits = false;
    unknown = false;
    maps_host = true;
  }

(* Polled's configurations and the weight each is drawn with: memory the host
   does not address is what a device's queue copies, so it weighs most. A device
   that runs no copy has memory the host addresses: [copies = false] keeps
   [visible]. *)
let configs =
  [
    (1, polled);
    (5, { polled with label = "polled-hidden"; visible = false });
    ( 1,
      {
        polled with
        label = "polled-hidden-tight";
        visible = false;
        capacity = 2;
      } );
    (1, { polled with label = "polled-tight"; capacity = 1 });
    (1, { polled with label = "polled-transport"; transport = true });
    (1, { polled with label = "polled-objects"; objects = true; waits = true });
    (1, { polled with label = "polled-waits"; waits = true });
    (1, { polled with label = "polled-copyless"; copies = false });
    (1, { polled with label = "polled-unknown"; unknown = true });
    (1, { polled with label = "polled-alone"; peers = false });
    (1, { polled with label = "polled-mapless"; maps_host = false });
  ]

type gpu = Metal | Cuda | Nv | Amd
type kind = Host | Memory | Disk | Polled of config | Gpu of gpu

let kind_name = function
  | Host -> "host"
  | Memory -> "memory"
  | Disk -> "disk"
  | Polled c -> c.label
  | Gpu Metal -> "metal"
  | Gpu Cuda -> "cuda"
  | Gpu Nv -> "nv"
  | Gpu Amd -> "amd"

let gpus =
  List.concat
    [
      (if Sys.file_exists "/System/Library/Frameworks/Metal.framework" then
         [ Metal ]
       else []);
      (if Sys.file_exists "/dev/nvidiactl" then [ Cuda; Nv ] else []);
      (if Rig_amd_amdgpu.count () > 0 then [ Amd ] else []);
    ]

let hold_gpu () =
  Rig_metal_support.hold_gpu ();
  Rig_cuda_support.hold_gpu ();
  Rig_amd_support.hold_gpu ()

let opened_gpu g =
  let get = function Ok d -> d | Error e -> failwith e in
  match g with
  | Metal ->
      get
        (Rig.open_
           (module Rig_metal)
           ~name:(Rig_metal.device_name 0)
           (fun () -> Rig_metal.open_ 0))
  | Cuda ->
      get
        (Rig.open_
           (module Rig_cuda)
           ~name:(Rig_cuda.device_name 0)
           (fun () -> Rig_cuda.open_ 0))
  | Nv ->
      get
        (Rig.open_
           (module Rig_nv)
           ~name:(Rig_nv_nvidia.device_name 0)
           (fun () -> Rig_nv_nvidia.open_ 0))
  | Amd ->
      get
        (Rig.open_
           (module Rig_amd)
           ~name:(Rig_amd_amdgpu.device_name 0)
           (fun () -> Rig_amd_amdgpu.open_ 0))

(* A GPU opens once per process, on first use, and again once a program lost it:
   its queue may wait on a lost Polled device's work. *)
let gpu_device =
  let lock = Mutex.create () and opened = ref [] in
  fun g ->
    Mutex.protect lock @@ fun () ->
    match List.assoc_opt g !opened with
    | Some d when Rig.lost d = None -> d
    | _ ->
        let d = opened_gpu g in
        opened := (g, d) :: List.remove_assoc g !opened;
        d

(* Files *)

(* The model's files, under this test's directory in _build, cleared at the
   start of a run. *)
let files =
  let dir = Filename.concat (Sys.getcwd ()) "model-files" in
  if Sys.file_exists dir then
    Array.iter
      (fun f -> try Sys.remove (Filename.concat dir f) with Sys_error _ -> ())
      (Sys.readdir dir)
  else Sys.mkdir dir 0o755;
  dir

let file_count = Atomic.make 0

(* A forked child names its files apart from its parent's. *)
let file_path () =
  Filename.concat files
    (strf "f%d-%d" (Unix.getpid ()) (Atomic.fetch_and_add file_count 1))

(* Bytes *)

(* A pattern of period 251, a prime, so that no power-of-two offset repeats
   it. *)
let pattern seed n =
  let period = 251 in
  let b = Bytes.create n in
  let k = min n period in
  for i = 0 to k - 1 do
    Bytes.unsafe_set b i (Char.unsafe_chr (((seed * 131) + (i * 7)) land 255))
  done;
  let filled = ref k in
  while !filled < n do
    let m = min !filled (n - !filled) in
    Bytes.blit b 0 b !filled m;
    filled := !filled + m
  done;
  b

(* The reference *)

(* A device: lost once the system showed its loss, [Faulting] while it may show
   it (a fault or a failure armed, or work that waited on such a device). *)
type state = Fine | Faulting | Lost

(* Where a buffer's bytes are for a copy: memory the host addresses, an io
   device's, or neither. *)
type locality = Local | Io | Remote

type rdev = {
  id : int;
  kind : kind;
  mutable state : state;
  mutable budget : int;
  mutable seen : int;  (** The last value the system showed assigned. *)
  mutable made : rmem list;  (** Memories the device allocated. *)
}

(* A memory's stamps as the reference knows them: the device of its last write
   and those of its uses, where the reference knows them, and the devices that
   may have made them, where a copy's route leaves it open. *)
and rstamps = {
  mutable writer : rdev option;
  mutable writers : rdev list;
  mutable users : rdev list;
  mutable maybe : rdev list;
}

and mkind = Heap | Array | Owned of B.memory | File of bool

and rmem = {
  owner : rdev;
  mkind : mkind;
  data : Bytes.t;
  defined : Bytes.t;  (** ['\001'] where [data] is known. *)
  mutable stamps : rstamps;
  mutable hold : rhold option;
  mutable readers : int;  (** Read claims. *)
  mutable generation : int;
  mutable exported : bool;
      (** {!Rig.Buffer.bigarray} exported it: it is outside the claims. *)
  mutable home : locality;  (** Its owner's buffers' locality. *)
}

and rhold = { hstamps : rstamps }

type rbuf = {
  mem : rmem;
  first : int;
  length : int;
  on : rdev;
  gen : int;
  local : locality;
}

type rworld = {
  two : bool;  (** Calls may run on two domains at once. *)
  forks : bool;  (** Calls fork children, whose own demands are the run's. *)
  mutable devs : rdev list;
  mutable next_dev : int;
  mutable subs : rsub list;
  mutable cells : rcell list;  (** In the order they were made. *)
}

and rcell = { w : rworld; mutable rb : rbuf option }
and rsub = { subw : rworld; mutable sub : sub option }

and sub = {
  dev : rdev;
  copy : (rbuf * rbuf) option;  (** A copy part, or a fill. *)
  arg : rmem option;
  held : rhold option;
}

type rhcell = { hw : rworld; mutable h : rhold option; mutable dropped : bool }

let empty_stamps () = { writer = None; writers = []; users = []; maybe = [] }

let new_dev w kind =
  let r =
    {
      id = w.next_dev;
      kind;
      state = Fine;
      budget = 1 lsl 30;
      seen = 0;
      made = [];
    }
  in
  w.next_dev <- w.next_dev + 1;
  w.devs <- w.devs @ [ r ];
  r

let new_world two forks =
  let w = { two; forks; devs = []; next_dev = 0; subs = []; cells = [] } in
  ignore (new_dev w Host);
  ignore (new_dev w Disk);
  w

let host w = List.nth w.devs 0
let disk w = List.nth w.devs 1

let new_mem owner mkind data defined =
  let m =
    {
      owner;
      mkind;
      data;
      defined;
      stamps = empty_stamps ();
      hold = None;
      readers = 0;
      generation = 0;
      exported = false;
      home = Local;
    }
  in
  owner.made <- m :: owner.made;
  m

let whole m local on =
  m.home <- local;
  { mem = m; first = 0; length = Bytes.length m.data; on; gen = 0; local }

let dead b = b.gen < b.mem.generation
let spans b = b.first = 0 && b.length = Bytes.length b.mem.data

(* Memory outside the claims, or [Read] memory, is never exclusive. *)
let never_exclusive m =
  m.exported || match m.mkind with Array | File _ -> true | _ -> false

let read_only m = m.mkind = File false

let overlaps a b =
  a.mem == b.mem && a.length > 0 && b.length > 0
  && a.first < b.first + b.length
  && b.first < a.first + a.length

let is_polled r = match r.kind with Polled _ -> true | _ -> false

let runs_on_host r =
  match r.kind with Host | Memory | Polled _ -> true | _ -> false

(* Verdicts *)

(* What a call may answer besides its result: [invalid], [Invalid_argument]
   required; [must], devices of which a loss is reported if one is lost; [may],
   those whose loss the call may report; [oom], an [Out_of_memory]. *)
type verdict = {
  mutable invalid : bool;
  mutable must : rdev list;
  mutable may : rdev list;
  mutable oom : [ `No | `May | `Must ];
}

let verdict () = { invalid = false; must = []; may = []; oom = `No }

(* The call uses [devs]; with [maybe], it may. *)
let uses ?(maybe = false) v devs =
  List.iter
    (fun r ->
      match r.state with
      | Lost when not maybe -> v.must <- r :: v.must
      | Lost | Faulting -> v.may <- r :: v.may
      | Fine -> ())
    devs

(* The call waits for [st]'s last write, or with [all], for every point. *)
let waits ?(all = false) v (st : rstamps) =
  if all then begin
    uses v (Option.to_list st.writer @ st.users);
    uses ~maybe:true v (st.writers @ st.maybe)
  end
  else begin
    uses v (Option.to_list st.writer);
    uses ~maybe:true v st.writers
  end

(* The call reads memory [m]: it waits for its last write, or, for memory in a
   hold, for every point of the hold. *)
let reads v m = waits ~all:(m.hold <> None) v m.stamps
let invalid_if v c = if c then v.invalid <- true

(* A device whose queue waits on others' work, which a GPU's and Polled's do,
   may be lost with a device whose point its work follows. *)
let follows dev v =
  let waits_in_queue =
    match dev.kind with
    | Polled _ | Gpu _ -> true
    | Host | Memory | Disk -> false
  in
  if
    dev.state = Fine && waits_in_queue
    && List.exists (fun r -> r.state <> Fine && r != dev) (v.must @ v.may)
  then dev.state <- Faulting

exception Empty

(* A device's loss, by its index in the program, [-1] for a device of no
   program, and its message. *)
exception Lost_at of int * string
exception Oom_at of int
exception Skipped

let lose w k = List.iter (fun r -> if r.id = k then r.state <- Lost) w.devs

(* A demand that a case reached [label], on one domain: on two, whether a case
   reaches it is the scheduler's. *)
let seen w label c = if not (w.two || w.forks) then cover label c

(* Judges the system's [outcome] by [v], applying [ok] to a result. A loss the
   call may report runs [lost]: the call's work may have run in part. *)
let judge ?(lost = ignore) w v outcome ok =
  let losable k =
    List.exists (fun r -> r.id = k && r.state <> Fine) (v.must @ v.may)
  in
  seen w "a call reports a loss it must"
    (v.must <> [] && Result.is_error outcome);
  match outcome with
  | Error (Invalid_argument _) when v.invalid -> ()
  | Error (Lost_at (k, _)) when losable k ->
      seen w "a call reports a device's loss" true;
      lose w k;
      lost ()
  | Error (Oom_at _) when v.oom <> `No ->
      seen w "an allocation the budget refuses" true
  | Error e -> raise e
  | Ok x -> (
      if v.invalid then fail "returned where Invalid_argument is required";
      if v.oom = `Must then fail "returned where Out_of_memory is required";
      match List.find_opt (fun r -> r.state = Lost) v.must with
      | Some r -> failf "returned though %s is lost" (kind_name r.kind)
      | None -> ok x)

(* Stamps *)

let add x l = if List.memq x l then l else x :: l
let use st d = st.users <- add d st.users
let maybe_use st d = st.maybe <- add d st.maybe

let write st d =
  st.writer <- Some d;
  st.writers <- [];
  use st d

let maybe_write st d =
  st.writers <- add d st.writers;
  maybe_use st d

let device_work r = match r.kind with Host | Disk -> false | _ -> true

(* The stamps a copy leaves, by its route (rig.mli, Buffer.copy): the host
   copies between memory it addresses and through io devices; otherwise the
   destination's device when only the source is the host's, the source's
   otherwise, directly or through staging memory, whose legs run on each side's
   device. *)
let copy_stamps src dst =
  let sd = src.on and dd = dst.on in
  let st = src.mem.stamps and dt = dst.mem.stamps in
  let on d f = if device_work d then f d in
  match (src.local, dst.local) with
  | (Local | Io), (Local | Io) -> ()
  | Local, Remote ->
      on dd (maybe_use st);
      on dd (write dt)
  | Io, Remote -> on dd (write dt)
  | Remote, (Local | Io) ->
      on sd (use st);
      on sd (maybe_write dt)
  | Remote, Remote ->
      on sd (use st);
      on dd (maybe_use st);
      on sd (maybe_write dt);
      on dd (maybe_write dt)

let blit src dst =
  let sd, sk = (src.mem.data, src.mem.defined)
  and dd, dk = (dst.mem.data, dst.mem.defined) in
  Bytes.blit sd src.first dd dst.first src.length;
  Bytes.blit sk src.first dk dst.first src.length

(* Bytes a call may have written in part. *)
let unknown b =
  let k = b.mem.defined in
  Bytes.fill k b.first b.length '\000'

let set_pattern b seed =
  let d = b.mem.data and k = b.mem.defined in
  Bytes.blit (pattern seed b.length) 0 d b.first b.length;
  Bytes.fill k b.first b.length '\001'

let check_bytes b s =
  let d = b.mem.data and k = b.mem.defined in
  if String.length s <> b.length then
    failf "read %d bytes of a %d-byte buffer" (String.length s) b.length;
  if not (String.equal s (Bytes.sub_string d b.first b.length)) then
    for i = 0 to b.length - 1 do
      let j = b.first + i in
      if Bytes.get k j = '\001' && Bytes.get d j <> s.[i] then
        failf "byte %d of %d is %#x, the reference's %#x" i b.length
          (Char.code s.[i])
          (Char.code (Bytes.get d j))
    done

(* Budgets *)

(* Whether [m] counts in its owner's budget for sure. *)
let counts m =
  match (m.owner.kind, m.mkind) with
  | Polled { copies = false; _ }, Owned _ -> true
  | Polled _, Owned B.Device -> true
  | _ -> false

(* Whether an allocation of [n] bytes on [d] may be refused: past what [d]'s
   memories may hold, or on two domains, where the other domain's allocations
   and budgets change what [d] holds at the same time. *)
let may_refuse w d made n = made + n > d.budget || w.two

(* The bytes of [d]'s budget that live buffers hold for sure: those of cells and
   of submissions' parts. *)
let live_bytes w d =
  let live =
    List.filter_map (fun c -> c.rb) w.cells
    @ List.concat_map
        (fun s ->
          match s.sub with
          | Some { copy = Some (a, b); _ } -> [ a; b ]
          | _ -> [])
        w.subs
  in
  List.fold_left
    (fun n m ->
      if counts m && List.exists (fun b -> b.mem == m) live then
        n + Bytes.length m.data
      else n)
    0 d.made

(* The bytes of [d]'s budget its memories may hold: [Mapped] memory counts where
   the budget held it. *)
let made_bytes d =
  List.fold_left
    (fun n m ->
      if counts m || m.mkind = Owned B.Mapped then n + Bytes.length m.data
      else n)
    0 d.made

(* The system *)

type sdev = { d : Rig.t; skind : kind; p : P.t option; mutable spoiled : bool }

(* Polled and memory devices go back to a pool when their program ends, unless a
   fault or a failure was armed on them. *)
let pool_lock = Mutex.create ()
let pool : sdev list ref = ref []
let opened = Atomic.make 0

let fresh kind =
  let name () =
    strf "model:%s-%d" (kind_name kind) (Atomic.fetch_and_add opened 1)
  in
  let device d = { d; skind = kind; p = None; spoiled = false } in
  match kind with
  | Host -> device Rig.host
  | Disk -> device Rig_disk.device
  | Gpu g -> device (gpu_device g)
  | Memory -> (
      match Rig.memory_device (name ()) with
      | Ok d -> device d
      | Error e -> failwith e)
  | Polled c ->
      let d, p =
        P.open_ ~capacity:c.capacity ~copies:c.copies ~host_visible:c.visible
          ~transport:c.transport ~peers:c.peers ~maps_host:c.maps_host
          ~completion:(if c.objects then `Object else `Host)
          ~waits_on:(if c.waits then [ `Host; `Object ] else [])
          ~answer:(if c.unknown then `Unknown else `Stopped)
          (name ())
      in
      { d; skind = kind; p = Some p; spoiled = false }

let take_device kind =
  let pooled =
    Mutex.protect pool_lock (fun () ->
        match List.partition (fun s -> s.skind = kind) !pool with
        | s :: rest, others ->
            pool := rest @ others;
            Some s
        | [], _ -> None)
  in
  match pooled with Some s -> s | None -> fresh kind

let give_device s =
  match s.skind with
  | Host | Disk | Gpu _ -> ()
  | Memory | Polled _ ->
      if (not s.spoiled) && Rig.lost s.d = None then
        begin match Rig.wait s.d (Rig.submitted s.d) with
        | () ->
            if s.p <> None then Rig.set_budget s.d (1 lsl 30);
            Mutex.protect pool_lock (fun () -> pool := s :: !pool)
        | exception Rig.Lost _ -> ()
        end

type sworld = {
  mutable sdevs : (Rig.t * int) list;
  mutable next : int;
  mems : int Atomic.t;
  mutable scells : scell list;  (** In the order they were made. *)
  mutable gpus : (kind * sdev) list;  (** One device per GPU in a program. *)
}

(* A memory as the caller sees it: the lock it takes around host accesses. *)
and smem = {
  lock : Mutex.t;
  sid : int;
  driver : bool;  (** A driver's device allocated it. *)
}

and sbuf = { b : B.t; m : smem }

(* A cell's lock: a call holds the locks of the cells it names, so that the
   program's own variables change as some order of the calls says. *)
and scell = { cell : sbuf option Atomic.t; sw : sworld; cl : smem }

type ssub = { s : Sub.t; sd : Rig.t; sarg : B.t option; slock : Mutex.t }
type sscell = { ss : ssub option Atomic.t; ssw : sworld; scl : smem }

type shcell = {
  hsw : sworld;
  hcl : smem;
  sh : Rig.Hold.t option Atomic.t;
  used : bool Atomic.t;  (** A cell holds one hold in its life. *)
  runs : int Atomic.t;  (** Runs of the hold's release. *)
}

let index w d =
  match List.find_opt (fun (d', _) -> Rig.equal d d') w.sdevs with
  | Some (_, i) -> i
  | None -> -1

(* Runs [f], answering rig's device exceptions by the devices' indices in the
   program, which the reference names them by. *)
let guard w f =
  try f () with
  | Rig.Lost (d, why) ->
      raise (Lost_at (index w d, strf "%s lost: %s" (Rig.name d) why))
  | Rig.Out_of_memory (d, _) -> raise (Oom_at (index w d))

let new_smem ?(driver = false) w =
  { lock = Mutex.create (); sid = Atomic.fetch_and_add w.mems 1; driver }

let get c = match Atomic.get c.cell with Some sb -> sb | None -> raise Empty

(* A cell's buffer, its cell emptied first when [last]. *)
let take ~last c =
  match if last then Atomic.exchange c.cell None else Atomic.get c.cell with
  | Some sb -> sb
  | None -> raise Empty

(* Runs [f] holding the locks of [ms], taken in one order. *)
let locked ms f =
  let ms = List.sort_uniq (fun a b -> Int.compare a.sid b.sid) ms in
  List.iter (fun m -> Mutex.lock m.lock) ms;
  Fun.protect ~finally:(fun () -> List.iter (fun m -> Mutex.unlock m.lock) ms) f

let on_host b = Rig.equal (B.device b) Rig.host

let host_bytes b =
  let ba = B.bigarray Bigarray.char b in
  String.init (Bigarray.Array1.dim ba) (Bigarray.Array1.unsafe_get ba)

let host_pattern seed n =
  let h = B.create Rig.host n in
  let ba = B.bigarray Bigarray.char h in
  Bytes.iteri (Bigarray.Array1.unsafe_set ba) (pattern seed n);
  h

(* Bytes print by their length and digest past a line. *)
let octets =
  let pp ppf s =
    let n = String.length s in
    if n <= 32 then Format.fprintf ppf "%S" s
    else
      Format.fprintf ppf "%d bytes, md5 %s" n (Digest.to_hex (Digest.string s))
  in
  Testable.make ~pp ~equal:String.equal

(* The first bytes of two buffers, as many in each; of one buffer, its two
   halves. *)
let prefixes a b =
  if a == b then
    let n = a.length / 2 in
    ({ a with length = n }, { a with first = a.first + n; length = n })
  else
    let n = min a.length b.length in
    ({ a with length = n }, { b with length = n })

let prefixes_sys a b =
  if a == b then
    let n = B.length a / 2 in
    (B.view a ~first:0 ~length:n, B.view a ~first:n ~length:n)
  else
    let n = min (B.length a) (B.length b) in
    (B.view a ~first:0 ~length:n, B.view b ~first:0 ~length:n)

(* Generators *)

(* Sizes at the edges rig.mli names: none, a page, a host buffer that starts on
   a page from 64 KiB. A copy larger than a staging slot is test_buffer.ml's. *)
let size =
  Gen.of_list ~pp:Format.pp_print_int
    [ 0; 1; 8; 24; 4096; 65535; 65536; 69632; 1 lsl 20 ]

let seed = Gen.int_range 0 255

(* A call uses its operand for the last time one time in four: a cell it empties
   starves the calls after it. *)
let last =
  Gen.frequency
    [
      (3, Gen.of_list ~pp:Format.pp_print_bool [ false ]);
      (1, Gen.of_list ~pp:Format.pp_print_bool [ true ]);
    ]

let eighth = Gen.int_range 0 8

let pp_memory ppf = function
  | B.Device -> Format.pp_print_string ppf "Device"
  | B.Pinned -> Format.pp_print_string ppf "Pinned"
  | B.Mapped -> Format.pp_print_string ppf "Mapped"

let memory_kind =
  Gen.of_list ~pp:pp_memory [ B.Device; B.Device; B.Device; B.Pinned; B.Mapped ]

let pp_access ppf = function
  | B.Read -> Format.pp_print_string ppf "Read"
  | B.Read_write -> Format.pp_print_string ppf "Read_write"

let access = Gen.of_list ~pp:pp_access [ B.Read; B.Read_write ]

let kinds =
  let pp ppf k = Format.pp_print_string ppf (kind_name k) in
  Gen.frequency
    ([
       (1, Gen.of_list ~pp [ Host ]);
       (2, Gen.of_list ~pp [ Memory; Disk ]);
       ( 6,
         Gen.frequency
           (List.map (fun (k, c) -> (k, Gen.of_list ~pp [ Polled c ])) configs)
       );
     ]
    @
    if gpus = [] then []
    else [ (3, Gen.of_list ~pp (List.map (fun g -> Gpu g) gpus)) ])

let budgets =
  Gen.of_list ~pp:Format.pp_print_int [ 0; 4096; 65536; 1 lsl 20; 1 lsl 30 ]

(* Values *)

let world = abstract "w"

type rdevv = { dw : rworld; r : rdev }

let device =
  abstract
    ~pp:(fun ppf v -> Format.fprintf ppf "%s#%d" (kind_name v.r.kind) v.r.id)
    ~release:(fun (s, _) -> give_device s)
    "d"

let cell = abstract ~release:(fun c -> Atomic.set c.cell None) "b"
let subs = abstract ~release:(fun c -> Atomic.set c.ss None) "s"

(* A hold's release runs at most once, and only once nothing reaches the
   hold. *)
let check_hold r s =
  let runs = Atomic.get s.runs in
  seen r.hw "a hold's release ran" (runs = 1);
  at_most ~msg:"runs of the release" int ~than:1 runs;
  if runs = 1 then begin
    let reachable =
      (not r.dropped)
      || List.exists
           (fun c ->
             match c.sub with
             | Some { held = Some h; _ } -> Option.equal ( == ) (Some h) r.h
             | _ -> false)
           r.hw.subs
    in
    if reachable then fail "a hold's release ran while the hold was reachable"
  end

let holds =
  abstract ~invariant:check_hold ~release:(fun c -> Atomic.set c.sh None) "h"

(* Commands *)

(* The world and its devices *)

let start_ref two forks () = new_world two forks

let start_sys () =
  let w =
    { sdevs = []; next = 2; mems = Atomic.make 0; scells = []; gpus = [] }
  in
  w.sdevs <- [ (Rig.host, 0); (Rig_disk.device, 1) ];
  w

let open_ref kind w =
  let r =
    match kind with
    | Host -> host w
    | Disk -> disk w
    | Gpu _ -> (
        match List.find_opt (fun r -> r.kind = kind) w.devs with
        | Some r -> r
        | None -> new_dev w kind)
    | Memory | Polled _ -> new_dev w kind
  in
  { dw = w; r }

let open_sys kind w =
  let s =
    match (kind, List.assoc_opt kind w.gpus) with
    | Gpu _, Some s -> s
    | Gpu _, None ->
        let s = take_device kind in
        w.gpus <- (kind, s) :: w.gpus;
        s
    | _ -> take_device kind
  in
  if index w s.d < 0 then begin
    w.sdevs <- w.sdevs @ [ (s.d, w.next) ];
    w.next <- w.next + 1
  end;
  (s, w)

(* A cell starts with a host buffer of [n] bytes of the pattern [seed]. *)
let new_cell_ref n seed w =
  let m = new_mem (host w) Heap (pattern seed n) (Bytes.make n '\001') in
  (* Its pattern was written through Buffer.bigarray. *)
  m.exported <- true;
  let c = { w; rb = Some (whole m Local (host w)) } in
  w.cells <- w.cells @ [ c ];
  c

let new_cell_sys n seed w =
  let c =
    {
      cell = Atomic.make (Some { b = host_pattern seed n; m = new_smem w });
      sw = w;
      cl = new_smem w;
    }
  in
  w.scells <- w.scells @ [ c ];
  c

let new_sub_ref w =
  let c = { subw = w; sub = None } in
  w.subs <- c :: w.subs;
  c

let new_sub_sys w = { ss = Atomic.make None; ssw = w; scl = new_smem w }
let new_hold_ref w = { hw = w; h = None; dropped = false }

let new_hold_sys w =
  {
    hsw = w;
    hcl = new_smem w;
    sh = Atomic.make None;
    used = Atomic.make false;
    runs = Atomic.make 0;
  }

(* Creation *)

let create_ref (v : rdevv) memory n c outcome =
  let d = v.r and w = c.w in
  let vd = verdict () in
  uses vd [ d ];
  invalid_if vd (d.kind = Disk && n > 0);
  let budgeted =
    match (d.kind, memory) with
    | Polled { copies = false; _ }, _ | Polled _, B.Device -> n > 0
    | _ -> false
  in
  (* On two domains the other's budgets and allocations change what [d] may hold
     meanwhile. *)
  if w.two && is_polled d && n > 0 then vd.oom <- `May
  else if budgeted then
    if n > d.budget then vd.oom <- `Must
    else if may_refuse w d (made_bytes d) n then vd.oom <- `May;
  judge w vd outcome (fun () ->
      let local =
        match d.kind with
        | Polled cfg when memory = B.Device && not cfg.visible -> Remote
        | Gpu (Cuda | Nv | Amd) when memory = B.Device -> Remote
        | Disk -> Io
        | _ -> Local
      in
      let mkind = if d.kind = Host then Heap else Owned memory in
      if budgeted && (not w.two) && live_bytes w d + n > d.budget then
        failf "%s holds %d bytes and %d more in a budget of %d"
          (kind_name d.kind) (live_bytes w d) n d.budget;
      let m = new_mem d mkind (Bytes.make n '\000') (Bytes.make n '\000') in
      (* Memory from the device's cache keeps the device's own points: its own
         order covers their reuse. *)
      if device_work d then maybe_write m.stamps d;
      c.rb <- Some (whole m local d))

let create_sys ((s : sdev), w) memory n c =
  guard w @@ fun () ->
  let b = B.create ~memory s.d n in
  let driver = match s.skind with Host | Disk -> false | _ -> true in
  Atomic.set c.cell (Some { b; m = new_smem ~driver w })

let array_ref (v : rdevv) n seed c outcome =
  let w = v.dw in
  judge w (verdict ()) outcome (fun () ->
      let m = new_mem (host w) Array (pattern seed n) (Bytes.make n '\001') in
      c.rb <- Some (whole m Local (host w)))

let array_sys (_, w) n seed c =
  let ba = Bigarray.Array1.create Bigarray.char Bigarray.c_layout n in
  Bytes.iteri (Bigarray.Array1.unsafe_set ba) (pattern seed n);
  Atomic.set c.cell (Some { b = B.of_bigarray ba; m = new_smem w })

(* A file the disk creates, of zeros, or one written here and opened for
   reading. *)
let file_ref (v : rdevv) n seed writable c outcome =
  let w = v.dw in
  judge w (verdict ()) outcome (fun () ->
      let data = if writable then Bytes.make n '\000' else pattern seed n in
      let m = new_mem (disk w) (File writable) data (Bytes.make n '\001') in
      c.rb <- Some (whole m Io (disk w)))

let file_sys (_, w) n seed writable c =
  let path = file_path () in
  let b =
    if writable then
      match Rig_disk.create_file path n with Ok b -> b | Error e -> failwith e
    else begin
      Out_channel.with_open_bin path (fun oc ->
          Out_channel.output_bytes oc (pattern seed n));
      match Rig_disk.of_file path with Ok b -> b | Error e -> failwith e
    end
  in
  Atomic.set c.cell (Some { b; m = new_smem w })

(* Borrows *)

type borrowed = [ `Some | `None | `Either ]

(* Whether [d] maps [b]'s memory (rig.mli, Buffer.borrow). *)
let maps d b : borrowed =
  let m = b.mem in
  let heap_page = m.mkind = Heap && Bytes.length m.data >= 65536 in
  if b.on == d || m.owner == d then `Some
  else
    match (d.kind, m.mkind) with
    | Disk, _ -> `None
    | _ when Bytes.length m.data = 0 -> `Either
    | (Host | Memory), (Heap | Array) -> `Some
    | Polled { maps_host = false; _ }, (Heap | Array | File _) -> `None
    (* A GPU maps a file's pages where its kernel driver lets it: NVIDIA's maps
       no file held for writing (rig.mli, Io.pages). *)
    | Gpu _, File _ -> `Either
    | (Host | Memory | Polled _), File _ -> `Some
    (* A host buffer of fewer than 64 KiB may start on a page. *)
    | Polled _, Heap -> if heap_page then `Some else `Either
    | Polled cfg, Owned _ when is_polled m.owner ->
        if cfg.peers then `Some else `Either
    | _ -> `Either

(* [b] borrowed on [d]: the same memory, and where its bytes are for [d]. *)
let borrowed d b =
  if b.on == d then b
  else if b.mem.owner == d then { b with on = d; local = b.mem.home }
  else
    let local =
      match (d.kind, b.mem.owner.kind, b.mem.mkind) with
      | (Host | Memory), _, _ -> Local
      | _, _, File _ -> Local
      | _ when b.local = Local -> Local
      | Polled _, Polled cfg, _ when cfg.visible -> Local
      | _ -> Remote
    in
    { b with on = d; local }

let borrow_ref (v : rdevv) src dst outcome =
  let d = v.r and w = dst.w in
  match src.rb with
  | None -> (
      match outcome with Error Empty -> () | _ -> fail "an empty cell")
  | Some b ->
      let vd = verdict () in
      invalid_if vd (dead b);
      uses vd [ d; b.on ];
      uses ~maybe:true vd [ b.mem.owner ];
      waits ~all:true vd b.mem.stamps;
      let predicted = maps d b in
      judge w vd outcome (fun got ->
          match (predicted, got) with
          | `Some, false -> fail "a borrow rig.mli promises answered None"
          | `None, true -> fail "a borrow rig.mli refuses answered Some"
          | _, false -> dst.rb <- None
          | _, true ->
              seen w "a driver's device borrows host memory"
                ((is_polled d || match d.kind with Gpu _ -> true | _ -> false)
                && b.mem.mkind = Heap);
              seen w "a device borrows a file's pages"
                (match b.mem.mkind with File _ -> d.kind <> Disk | _ -> false);
              seen w "a device borrows another device's memory"
                (is_polled d && is_polled b.mem.owner && b.mem.owner != d);
              dst.rb <- Some (borrowed d b))

let borrow_sys ((s : sdev), w) src dst =
  guard w @@ fun () ->
  let sb = get src in
  match B.borrow s.d sb.b with
  | Some b ->
      Atomic.set dst.cell (Some { b; m = sb.m });
      true
  | None ->
      Atomic.set dst.cell None;
      false

let view_bounds length a l =
  let first = length * a / 8 in
  (first, (length - first) * l / 8)

let view_ref a l src dst outcome =
  match src.rb with
  | None -> (
      match outcome with Error Empty -> () | _ -> fail "an empty cell")
  | Some b ->
      let vd = verdict () in
      invalid_if vd (dead b);
      judge dst.w vd outcome (fun () ->
          let first, length = view_bounds b.length a l in
          dst.rb <- Some { b with first = b.first + first; length })

let view_sys a l src dst =
  let sb = get src in
  let first, length = view_bounds (B.length sb.b) a l in
  Atomic.set dst.cell (Some { sb with b = B.view sb.b ~first ~length })

(* Host access *)

let with_buf c outcome k =
  match c.rb with
  | None -> (
      match outcome with Error Empty -> () | _ -> fail "an empty cell")
  | Some b -> k b

let drop_if last c = if last then c.rb <- None

(* A fill writes through the buffer's bytes on the host, or copies from a host
   buffer of the pattern. *)
let fill_ref seed last c outcome =
  with_buf c outcome @@ fun b ->
  if read_only b.mem then
    match outcome with
    | Error Skipped -> drop_if last c
    | _ -> fail "a fill of memory that admits only reads ran"
  else
    let vd = verdict () in
    invalid_if vd (dead b);
    if b.on.kind = Host then waits ~all:true vd b.mem.stamps
    else begin
      uses vd [ b.on ];
      if b.length > 0 then waits ~all:true vd b.mem.stamps
    end;
    let lost () =
      unknown b;
      if device_work b.on then maybe_write b.mem.stamps b.on
    in
    judge ~lost c.w vd outcome (fun () ->
        seen c.w "a fill whose buffer is unreachable during it" last;
        if b.on.kind = Host then b.mem.exported <- true;
        if b.on.kind <> Host && b.local = Remote && b.length > 0 then begin
          follows b.on vd;
          if device_work b.on then write b.mem.stamps b.on
        end;
        set_pattern b seed);
    drop_if last c

(* Memory that admits only reads is never written: the host would write it only
   by breaking that, through Buffer.bigarray. *)
let fill_sys seed last c =
  let sb = take ~last c in
  if B.access sb.b = B.Read then raise Skipped;
  guard c.sw @@ fun () ->
  locked [ sb.m ] @@ fun () ->
  let b = sb.b in
  if on_host b then begin
    B.wait b B.Read_write;
    let ba = B.bigarray Bigarray.char b in
    Bytes.iteri (Bigarray.Array1.unsafe_set ba) (pattern seed (B.length b))
  end
  else B.copy ~src:(host_pattern seed (B.length b)) ~dst:b

let read_ref last c outcome =
  with_buf c outcome @@ fun b ->
  let vd = verdict () in
  invalid_if vd (dead b);
  if b.on.kind = Host then waits ~all:true vd b.mem.stamps
  else begin
    uses vd [ b.on ];
    if b.length > 0 then reads vd b.mem
  end;
  let lost () = if device_work b.on then maybe_use b.mem.stamps b.on in
  judge ~lost c.w vd outcome (fun s ->
      if b.on.kind = Host then b.mem.exported <- true;
      if
        b.on.kind <> Host && b.local = Remote && device_work b.on
        && b.length > 0
      then begin
        follows b.on vd;
        use b.mem.stamps b.on
      end;
      check_bytes b s);
  drop_if last c

let read_sys last c =
  let sb = take ~last c in
  guard c.sw @@ fun () ->
  locked [ sb.m ] @@ fun () ->
  let b = sb.b in
  if on_host b then begin
    B.wait b B.Read;
    host_bytes b
  end
  else begin
    let h = B.create Rig.host (B.length b) in
    B.copy ~src:b ~dst:h;
    host_bytes h
  end

(* A copy of [a] into [b], [src]'s buffer and [dst]'s, which [invalid] says rig
   refuses. *)
let copy_judge ~invalid last src a b outcome =
  let vd = verdict () in
  invalid_if vd invalid;
  invalid_if vd (read_only b.mem);
  uses vd [ a.on; b.on ];
  if a.length > 0 then begin
    reads vd a.mem;
    waits ~all:true vd b.mem.stamps
  end;
  let w = src.w in
  let lost () =
    unknown b;
    List.iter
      (fun d ->
        if device_work d then begin
          maybe_use a.mem.stamps d;
          maybe_write b.mem.stamps d
        end)
      [ a.on; b.on ]
  in
  judge ~lost w vd outcome (fun () ->
      if a.length > 0 then begin
        seen w "a copy the host makes"
          (a.local <> Remote && b.local <> Remote
          && not (a.local = Io && b.local = Io));
        seen w "a copy a device runs" (a.local = Remote || b.local = Remote);
        seen w "a copy between two devices"
          (device_work a.on && device_work b.on && a.on != b.on);
        seen w "a copy whose source is unreachable during it" last;
        follows a.on vd;
        follows b.on vd;
        copy_stamps a b;
        blit a b
      end);
  drop_if last src

let copy_ref last src dst outcome =
  with_buf src outcome @@ fun a ->
  with_buf dst outcome @@ fun b ->
  copy_judge
    ~invalid:(dead a || dead b || a.length <> b.length || overlaps a b)
    last src a b outcome

(* A copy between the first bytes of two buffers, as many in each: views, made
   only of live buffers. *)
let copy_prefix_ref last src dst outcome =
  with_buf src outcome @@ fun a ->
  with_buf dst outcome @@ fun b ->
  if dead a || dead b then
    match outcome with
    | Error (Invalid_argument _) -> drop_if last src
    | _ -> fail "a view of a dead buffer"
  else
    let a, b = prefixes a b in
    copy_judge ~invalid:(overlaps a b) last src a b outcome

let copy_prefix_sys last src dst =
  let b = get dst in
  let a = take ~last src in
  guard src.sw @@ fun () ->
  locked [ a.m; b.m ] @@ fun () ->
  let a, b = prefixes_sys a.b b.b in
  B.copy ~src:a ~dst:b

let copy_sys last src dst =
  let b = get dst in
  let a = take ~last src in
  guard src.sw @@ fun () ->
  locked [ a.m; b.m ] @@ fun () -> B.copy ~src:a.b ~dst:b.b

(* A transfer: [n] bytes of a pattern written into new memory of one device,
   copied into new memory of another and read back, whatever the two devices:
   the copy routes between every pair of the machine's kinds. *)
let transfer_ref (v : rdevv) (v' : rdevv) n seed outcome =
  let vd = verdict () in
  uses vd [ v.r; v'.r ];
  let asked = if v.r == v'.r then 2 * n else n in
  List.iter
    (fun d ->
      match d.kind with
      | Polled _ when n > 0 ->
          if may_refuse v.dw d (made_bytes d) asked then vd.oom <- `May
      | _ -> ())
    [ v.r; v'.r ];
  judge v.dw vd outcome (fun s ->
      seen v.dw "a copy between two devices"
        (device_work v.r && device_work v'.r && v.r != v'.r && n > 0);
      if not (String.equal s (Bytes.unsafe_to_string (pattern seed n))) then
        failf "a transfer from %s to %s read other bytes" (kind_name v.r.kind)
          (kind_name v'.r.kind))

let transfer_sys ((s : sdev), w) ((s' : sdev), _) n seed =
  guard w @@ fun () ->
  let make (d : sdev) =
    if d.skind = Disk then
      match Rig_disk.create_file (file_path ()) n with
      | Ok b -> b
      | Error e -> failwith e
    else B.create d.d n
  in
  let a = make s and b = make s' in
  B.copy ~src:(host_pattern seed n) ~dst:a;
  B.copy ~src:a ~dst:b;
  let h = B.create Rig.host n in
  B.copy ~src:b ~dst:h;
  host_bytes h

(* Submissions *)

(* A copy part, or a fill that copies its read slot into its write slot where
   both are memory the host runs work on. *)
let make_ref ?hold (v : rdevv) pair sc outcome =
  let d = v.r and w = sc.subw in
  let held = Option.bind hold (fun (h : rhcell) -> h.h) in
  let other_hold m =
    match m.hold with
    | Some h -> not (Option.equal ( == ) (Some h) held)
    | None -> false
  in
  let vd = verdict () in
  let skipped =
    match (pair, d.kind) with
    | Some (a, b), _ -> overlaps a b && not (dead a || dead b)
    | None, Gpu _ -> true
    | None, _ -> false
  in
  invalid_if vd (d.kind = Host || d.kind = Disk);
  (match pair with
  | Some (a, b) ->
      let copies =
        match d.kind with
        | Polled cfg -> cfg.copies
        | Gpu Metal -> false
        | _ -> true
      in
      invalid_if vd (not copies);
      invalid_if vd (dead a || dead b || a.length <> b.length);
      invalid_if vd (a.on != d || b.on != d || read_only b.mem);
      invalid_if vd (other_hold a.mem || other_hold b.mem)
  | None -> ());
  uses vd [ d ];
  if skipped then
    match outcome with
    | Error Skipped -> ()
    | _ -> fail "a submission the model does not make was made"
  else
    judge w vd outcome (fun () ->
        let arg =
          match pair with
          | Some _ -> None
          | None ->
              Some
                (new_mem (host w) Heap (Bytes.make 24 '\000')
                   (Bytes.make 24 '\000'))
        in
        sc.sub <- Some { dev = d; copy = pair; arg; held })

let make_sys ?hold ((s : sdev), w) pair sc =
  guard w @@ fun () ->
  let hold =
    Option.map
      (fun h -> match Atomic.get h.sh with Some h -> h | None -> raise Empty)
      hold
  in
  let part, arg =
    match pair with
    | Some (src, dst) ->
        let src, dst = prefixes_sys (get src).b (get dst).b in
        if B.overlaps src dst then raise Skipped;
        ( { Sub.queue = "COPY:0"; after = [||]; work = Sub.Copy { src; dst } },
          None )
    | None ->
        (match s.skind with Gpu _ -> raise Skipped | _ -> ());
        let arg = B.create Rig.host 24 in
        ( {
            Sub.queue = "COMPUTE:0";
            after = [||];
            work =
              Sub.Fill
                { fill = S.carry; arg; ring_units = 0; segment_bytes = 0 };
          },
          Some arg )
  in
  let sub = Sub.make ?hold ~reads:1 ~writes:1 s.d [| part |] in
  Atomic.set sc.ss
    (Some { s = sub; sd = s.d; sarg = arg; slock = Mutex.create () })

(* A copy between two buffers of [n] bytes that the submission alone holds. *)
let make_own_ref (v : rdevv) n sc outcome =
  let d = v.r and w = sc.subw in
  let vd = verdict () in
  let copies =
    match d.kind with
    | Polled cfg -> cfg.copies
    | Gpu Metal | Host | Disk -> false
    | _ -> true
  in
  invalid_if vd (d.kind = Disk && n > 0);
  invalid_if vd (not copies);
  uses vd [ d ];
  (match d.kind with
  | Polled _ when n > 0 ->
      if n > d.budget && not w.two then vd.oom <- `Must
      else if may_refuse w d (made_bytes d) (2 * n) then vd.oom <- `May
  | _ -> ());
  judge w vd outcome (fun () ->
      let local =
        match d.kind with
        | Polled cfg when not cfg.visible -> Remote
        | Gpu (Cuda | Nv | Amd) -> Remote
        | _ -> Local
      in
      let part () =
        let m =
          new_mem d (Owned B.Device) (Bytes.make n '\000') (Bytes.make n '\000')
        in
        maybe_write m.stamps d;
        whole m local d
      in
      let a = part () in
      let b = part () in
      sc.sub <- Some { dev = d; copy = Some (a, b); arg = None; held = None })

let make_own_sys ((s : sdev), w) n sc =
  guard w @@ fun () ->
  let src = B.create s.d n and dst = B.create s.d n in
  let part =
    { Sub.queue = "COPY:0"; after = [||]; work = Sub.Copy { src; dst } }
  in
  let sub = Sub.make ~reads:1 ~writes:1 s.d [| part |] in
  Atomic.set sc.ss
    (Some { s = sub; sd = s.d; sarg = None; slock = Mutex.create () })

let submit_ref last rc wc sc outcome =
  match sc.sub with
  | None -> (
      match outcome with Error Empty -> () | _ -> fail "an empty cell")
  | Some sub -> (
      with_buf rc outcome @@ fun r ->
      with_buf wc outcome @@ fun wb ->
      let d = sub.dev in
      let vd = verdict () in
      let parts = match sub.copy with Some (a, b) -> [ a; b ] | None -> [] in
      invalid_if vd (List.exists dead (r :: wb :: parts));
      invalid_if vd (r.mem.hold <> None || wb.mem.hold <> None);
      (* A slot not on the device is borrowed on it first. *)
      let slot b = if b.on == d then `Some else maps d b in
      invalid_if vd (slot r = `None || slot wb = `None || read_only wb.mem);
      uses ~maybe:true vd [ r.on; wb.on ];
      invalid_if vd
        (List.exists
           (fun b ->
             match b.mem.hold with
             | Some h -> not (Option.equal ( == ) (Some h) sub.held)
             | None -> false)
           parts);
      uses vd [ d ];
      reads vd r.mem;
      waits ~all:true vd wb.mem.stamps;
      (match sub.copy with
      | Some (a, b) ->
          reads vd a.mem;
          waits ~all:true vd b.mem.stamps
      | None -> ());
      let w = sc.subw in
      let lost () =
        List.iter
          (fun b ->
            maybe_write b.mem.stamps d;
            unknown b)
          (wb :: parts);
        maybe_use r.mem.stamps d
      in
      match outcome with
      | Error (Invalid_argument _) when slot r = `Either || slot wb = `Either ->
          drop_if last rc;
          drop_if last wc
      | _ ->
          let r = borrowed d r and wb = borrowed d wb in
          judge ~lost w vd outcome (fun value ->
              seen w "a submit whose slots are unreachable during it" last;
              if value <= d.seen then
                failf "value %d after the device's %d" value d.seen;
              d.seen <- value;
              follows d vd;
              use r.mem.stamps d;
              write wb.mem.stamps d;
              Option.iter (fun h -> use h.hstamps d) sub.held;
              Option.iter (fun m -> use m.stamps d) sub.arg;
              match sub.copy with
              | Some (a, b) ->
                  use a.mem.stamps d;
                  write b.mem.stamps d;
                  blit a b
              | None ->
                  seen w "a fill copies its read slot into its write slot"
                    (runs_on_host r.on && runs_on_host wb.on && r.length > 0
                   && wb.length > 0);
                  if runs_on_host r.on && runs_on_host wb.on then
                    blit
                      { r with length = min r.length wb.length }
                      { wb with length = min r.length wb.length });
          drop_if last rc;
          drop_if last wc)

let submit_sys last rc wc sc =
  let sub = match Atomic.get sc.ss with Some s -> s | None -> raise Empty in
  let r = get rc and wb = get wc in
  if last then begin
    Atomic.set rc.cell None;
    Atomic.set wc.cell None
  end;
  let w = rc.sw in
  guard w @@ fun () ->
  Mutex.protect sub.slock @@ fun () ->
  (* The run's buffers are on the device: a borrow where they are not. *)
  let onto b =
    if Rig.equal (B.device b) sub.sd then b
    else Option.value (B.borrow sub.sd b) ~default:b
  in
  let r = { r with b = onto r.b } and wb = { wb with b = onto wb.b } in
  (match sub.sarg with
  | Some arg ->
      (* Polled runs its work on host memory, at the addresses its buffers
         have. *)
      let host_run b =
        let d = B.device b in
        Rig.runs_on_host d || Rig.arch d = "polled"
      in
      let n = min (B.length r.b) (B.length wb.b) in
      B.wait arg B.Read_write;
      let at = B.address arg in
      if host_run r.b && host_run wb.b && n > 0 then begin
        S.store at (B.address wb.b);
        S.store (at + 8) (B.address r.b);
        S.store (at + 16) n
      end
      else begin
        S.store at 0;
        S.store (at + 8) 0;
        S.store (at + 16) 0
      end
  | None -> ());
  Rig.Point.value
    (Rig.submit sub.s ~reads:[| r.b |] ~writes:[| wb.b |] ~waits:[||])

(* Holds *)

let hold_ref a b hc outcome =
  match hc.h with
  | Some _ -> (
      match outcome with
      | Error Skipped -> ()
      | _ -> fail "a second hold in a cell")
  | None when hc.dropped -> (
      match outcome with
      | Error Skipped -> ()
      | _ -> fail "a hold in a dropped cell")
  | None ->
      with_buf a outcome @@ fun x ->
      with_buf b outcome @@ fun y ->
      let vd = verdict () in
      invalid_if vd (dead x || dead y);
      invalid_if vd (x.mem.hold <> None || y.mem.hold <> None);
      judge hc.hw vd outcome (fun () ->
          let mems = if x.mem == y.mem then [ x.mem ] else [ x.mem; y.mem ] in
          let st = empty_stamps () in
          List.iter
            (fun m ->
              let s = m.stamps in
              st.writer <-
                (match s.writer with Some _ as wr -> wr | None -> st.writer);
              List.iter (fun d -> st.writers <- add d st.writers) s.writers;
              List.iter
                (fun d -> st.users <- add d st.users)
                (Option.to_list s.writer @ s.users);
              List.iter (fun d -> st.maybe <- add d st.maybe) s.maybe)
            mems;
          let h = { hstamps = st } in
          List.iter
            (fun m ->
              m.stamps <- st;
              m.hold <- Some h)
            mems;
          hc.h <- Some h)

let hold_sys a b hc =
  if Atomic.get hc.used then raise Skipped;
  let x = (get a).b and y = (get b).b in
  let runs = hc.runs in
  guard a.sw @@ fun () ->
  let h = Rig.Hold.make ~release:(fun () -> Atomic.incr runs) [ x; y ] in
  Atomic.set hc.used true;
  Atomic.set hc.sh (Some h)

(* Claims *)

let claim_ref c outcome =
  with_buf c outcome @@ fun b ->
  let vd = verdict () in
  invalid_if vd (dead b);
  waits ~all:true vd b.mem.stamps;
  match outcome with
  | Error (Invalid_argument _) when c.w.two && not (dead b) -> ()
  | _ -> judge c.w vd outcome (fun () -> b.mem.readers <- b.mem.readers + 1)

let claim_sys c = guard c.sw @@ fun () -> Rig.Claim.read (get c).b

let release_ref c outcome =
  with_buf c outcome @@ fun b ->
  let vd = verdict () in
  invalid_if vd (b.mem.readers = 0);
  judge c.w vd outcome (fun () -> b.mem.readers <- b.mem.readers - 1)

let release_sys c = guard c.sw @@ fun () -> Rig.Claim.release (get c).b
let exclusive b = spans b && b.mem.readers = 0 && not (never_exclusive b.mem)

let with_ref r d outcome =
  with_buf r outcome @@ fun a ->
  with_buf d outcome @@ fun b ->
  let vd = verdict () in
  invalid_if vd (dead a || dead b || overlaps a b);
  waits ~all:true vd a.mem.stamps;
  waits ~all:true vd b.mem.stamps;
  match outcome with
  | Error (Invalid_argument _)
    when r.w.two && not (dead a || dead b || overlaps a b) ->
      ()
  | _ ->
      judge r.w vd outcome (fun got ->
          seen r.w "a donation held exclusive" got;
          let expected = a.mem != b.mem && exclusive b in
          if got && not expected then
            fail "a donation held exclusive against its claims";
          if expected && (not got) && not r.w.two then
            fail "a donation refused with no other claim")

let with_sys r d =
  guard r.sw @@ fun () ->
  let a = (get r).b and b = (get d).b in
  Rig.Claim.with_ ~read:[ a ] ~donate:[ [ b ] ] (fun c ->
      Rig.Claim.exclusive c b)

let consume_ref src dst outcome =
  with_buf src outcome @@ fun b ->
  let vd = verdict () in
  invalid_if vd (dead b);
  waits ~all:true vd b.mem.stamps;
  match outcome with
  | Error (Invalid_argument _) when src.w.two && not (dead b) -> ()
  | _ ->
      judge src.w vd outcome (fun got ->
          seen src.w "a memory consumed" got;
          let expected = exclusive b in
          if got && not expected then
            fail "a memory consumed while not exclusive";
          if expected && (not got) && not src.w.two then
            fail "a memory with no claim was not consumed";
          if got then begin
            b.mem.generation <- b.mem.generation + 1;
            dst.rb <- Some { b with gen = b.mem.generation }
          end)

let consume_sys src dst =
  let sb = get src in
  guard src.sw @@ fun () ->
  let b = sb.b in
  match
    Rig.Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
        if Rig.Claim.exclusive c b then
          Some (Rig.Claim.consume c ~why:"consumed" b)
        else None)
  with
  | Some b' ->
      Atomic.set dst.cell (Some { sb with b = b' });
      true
  | None -> false

(* Waits *)

let wait_ref access c outcome =
  with_buf c outcome @@ fun b ->
  let vd = verdict () in
  invalid_if vd (dead b);
  waits ~all:(access = B.Read_write || b.mem.hold <> None) vd b.mem.stamps;
  judge c.w vd outcome Fun.id

let wait_sys access c = guard c.sw @@ fun () -> B.wait (get c).b access

let wait_device_ref (v : rdevv) outcome =
  let vd = verdict () in
  uses vd [ v.r ];
  judge v.dw vd outcome Fun.id

let wait_device_sys ((s : sdev), w) =
  guard w @@ fun () -> Rig.wait s.d (Rig.submitted s.d)

(* A lost device's facts answer: its last value assigned, its word's, and
   whether it is lost. *)
let facts_ref (v : rdevv) outcome =
  let r = v.r in
  judge v.dw (verdict ()) outcome (fun (submitted, signaled, lost) ->
      if submitted < r.seen then failf "submitted %d after %d" submitted r.seen;
      if signaled > submitted then
        failf "signaled %d past submitted %d" signaled submitted;
      r.seen <- submitted;
      match (lost, r.state) with
      | true, Fine -> fail "a device lost with no fault"
      | true, _ -> r.state <- Lost
      | false, Lost -> fail "a lost device answers it is not"
      | false, _ -> ())

(* The word first: it never passes the value assigned before. *)
let facts_sys ((s : sdev), _) =
  let signaled = Rig.signaled s.d in
  let submitted = Rig.submitted s.d in
  (submitted, signaled, Rig.lost s.d <> None)

(* Polled's seams *)

let polled_cmd f ((s : sdev), _) =
  match s.p with
  | Some p ->
      s.spoiled <- true;
      f p
  | None -> ()

let arm_ref (v : rdevv) =
  if is_polled v.r && v.r.state = Fine then v.r.state <- Faulting

let run_sys ((s : sdev), _) =
  match s.p with Some p -> ignore (P.run p) | None -> ()

let budget_ref n (v : rdevv) outcome =
  judge v.dw (verdict ()) outcome (fun () ->
      if is_polled v.r then v.r.budget <- n)

let budget_sys n ((s : sdev), _) =
  match s.p with Some _ -> Rig.set_budget s.d n | None -> ()

let free_cache_ref (v : rdevv) outcome =
  let vd = verdict () in
  uses ~maybe:true vd [ v.r ];
  judge v.dw vd outcome Fun.id

let free_cache_sys ((s : sdev), w) = guard w @@ fun () -> Rig.free_cache s.d

let barrier_ref last c outcome =
  with_buf c outcome @@ fun b ->
  let vd = verdict () in
  invalid_if vd (dead b || b.mem.owner.kind <> Disk);
  reads vd b.mem;
  judge c.w vd outcome Fun.id;
  drop_if last c

let barrier_sys last c =
  let sb = take ~last c in
  guard c.sw @@ fun () -> Rig_disk.barrier sb.b

(* The command list *)

let unit_ref _ = ()

(* Forks *)

(* What a forked child sees of a cell: nothing, a loss, an exception, or its
   bytes read on the host. *)
type seen = Empty_cell | Lost_cell | Raised of string | Read of string

let pp_seen ppf = function
  | Empty_cell -> Format.pp_print_string ppf "empty"
  | Lost_cell -> Format.pp_print_string ppf "lost"
  | Raised e -> Format.fprintf ppf "raised %s" e
  | Read s -> Format.fprintf ppf "%d bytes" (String.length s)

type forked = {
  cells_seen : seen list;
  devices_lost : bool list;  (** Each device's loss, in the world's order. *)
  child_file : bool;  (** A long file read in the child read its bytes. *)
  parent_file : bool;  (** The same in the parent, at the same time. *)
}

let forked =
  let pp ppf f =
    Format.fprintf ppf "@[{ cells = [%a]; lost = [%a]; files %b %b }@]"
      (Format.pp_print_list
         ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ")
         pp_seen)
      f.cells_seen
      (Format.pp_print_list
         ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ")
         Format.pp_print_bool)
      f.devices_lost f.child_file f.parent_file
  in
  Testable.make ~pp ~equal:( = )

(* More than one of the disk's long requests on Linux, 2 MiB. *)
let long_file = (4 lsl 20) + 4096

(* Writes and reads back a file longer than one of the disk's long requests. *)
let file_reads seed =
  let n = long_file in
  let f =
    match Rig_disk.create_file (file_path ()) n with
    | Ok f -> f
    | Error e -> failwith e
  in
  B.copy ~src:(host_pattern seed n) ~dst:f;
  let h = B.create Rig.host n in
  B.copy ~src:f ~dst:h;
  String.equal (host_bytes h) (Bytes.unsafe_to_string (pattern seed n))

let fork_ref w outcome =
  match outcome with
  | Error Skipped when Sys.win32 -> ()
  | Error e -> raise e
  | Ok f ->
      cover "a child sees a device lost" (List.exists Fun.id f.devices_lost);
      cover "a child reads a buffer's bytes"
        (List.exists (function Read s -> s <> "" | _ -> false) f.cells_seen);
      if not f.child_file then
        fail "a long file read in the child read other bytes";
      if not f.parent_file then
        fail "a long file read in the parent beside the child read other bytes";
      List.iter2
        (fun r lost ->
          let driver = device_work r && r.kind <> Disk in
          if lost <> driver then
            failf "in the child, %s is %slost" (kind_name r.kind)
              (if lost then "" else "not "))
        w.devs f.devices_lost;
      List.iter2
        (fun c seen ->
          match (c.rb, seen) with
          | None, Empty_cell -> ()
          | None, _ -> fail "an empty cell read in the child"
          | Some b, _ when dead b -> (
              match seen with
              | Raised "Invalid_argument" -> ()
              | _ -> fail "a dead buffer read in the child")
          | Some b, Lost_cell when device_work b.on || device_work b.mem.owner
            ->
              ()
          | Some b, _ when device_work b.on ->
              failf "a buffer of %s read in the child" (kind_name b.on.kind)
          | Some b, seen -> (
              let st = b.mem.stamps in
              let drivers =
                List.exists device_work
                  (Option.to_list st.writer @ st.writers @ st.users @ st.maybe)
              in
              match seen with
              | (Lost_cell | Read _) when drivers -> ()
              | Read s -> check_bytes b s
              | Raised e -> failf "a read in the child raised %s" e
              | Lost_cell ->
                  fail "a read in the child lost a device its memory never met"
              | Empty_cell -> fail "a full cell empty in the child"))
        w.cells f.cells_seen

(* The child reads every cell on the host, checks its devices' losses and reads
   a long file while the parent reads one too, then leaves without this
   process's exit. *)
let fork_sys w =
  if Sys.win32 then raise Skipped;
  let r, wr = Unix.pipe () in
  match Unix.fork () with
  | 0 ->
      Unix.close r;
      let see c =
        match Atomic.get c.cell with
        | None -> Empty_cell
        (* A forked child that reads a host borrow of its lost device's memory
           dies where the memory is not the child's, as Metal's is not on macOS:
           test_fork.ml holds it as an expected failure. *)
        | Some sb when sb.m.driver && on_host sb.b -> Lost_cell
        | Some sb -> (
            let b = sb.b in
            match
              guard w (fun () ->
                  if on_host b then begin
                    B.wait b B.Read;
                    host_bytes b
                  end
                  else begin
                    let h = B.create Rig.host (B.length b) in
                    B.copy ~src:b ~dst:h;
                    host_bytes h
                  end)
            with
            | s -> Read s
            | exception Lost_at _ -> Lost_cell
            | exception e -> Raised (Printexc.exn_slot_name e))
      in
      let f =
        try
          Ok
            {
              cells_seen = List.map see w.scells;
              devices_lost = List.map (fun (d, _) -> Rig.lost d <> None) w.sdevs;
              child_file = file_reads 1;
              parent_file = true;
            }
        with e -> Error (Printexc.to_string e)
      in
      let oc = Unix.out_channel_of_descr wr in
      Marshal.to_channel oc f [];
      close_out oc;
      Unix._exit 0
  | pid -> (
      Unix.close wr;
      let parent_file = file_reads 2 in
      let ic = Unix.in_channel_of_descr r in
      let f : (forked, string) result =
        match Marshal.from_channel ic with
        | f -> f
        | exception End_of_file -> (
            match Unix.waitpid [] pid with
            | _, Unix.WSIGNALED n ->
                Error (strf "the child died of signal %d" n)
            | _, Unix.WEXITED n -> Error (strf "the child exited %d" n)
            | _, Unix.WSTOPPED n -> Error (strf "the child stopped on %d" n))
      in
      close_in ic;
      (try ignore (Unix.waitpid [] pid) with Unix.Unix_error _ -> ());
      match f with
      | Ok f -> { f with parent_file }
      | Error e -> failwith ("the forked child raised " ^ e))

(* Values of two worlds of one program never meet: a call naming both does
   nothing. *)
let one_world ws outcome k =
  match ws with
  | w :: rest when List.exists (fun w' -> w' != w) rest -> (
      match outcome with
      | Error Skipped -> ()
      | _ -> fail "a call on two worlds' values ran")
  | _ -> k ()

let one_world_sys ws =
  match ws with
  | w :: rest when List.exists (fun w' -> w' != w) rest -> raise Skipped
  | _ -> ()

(* Runs [f] holding the locks of the cells [ls] names, after checking their
   worlds. *)
let cells ?(ws = []) ls f =
  one_world_sys ws;
  locked ls f

let commands ~two ~fork =
  let dw (v : rdevv) = v.dw in
  let one c f = cells [ c.cl ] f in
  let start =
    command "start" (Gen.unit @-> makes world) (start_ref two fork) start_sys
  in
  let open_ =
    command "open" (kinds @-> world ^-> makes device) open_ref open_sys
  in
  let new_cell =
    command "cell"
      (size @-> seed @-> world ^-> makes cell)
      new_cell_ref new_cell_sys
  in
  let new_sub =
    command "submission" (world ^-> makes subs) new_sub_ref new_sub_sys
  in
  let new_hold =
    command "hold cell" (world ^-> makes holds) new_hold_ref new_hold_sys
  in
  let create =
    command "create"
      (device ^-> memory_kind @-> size @-> cell ^-> judges unit)
      (fun v m n c o ->
        one_world [ dw v; c.w ] o (fun () -> create_ref v m n c o))
      (fun v m n c ->
        cells ~ws:[ snd v; c.sw ] [ c.cl ] (fun () -> create_sys v m n c))
  in
  let create_host =
    command "create on host"
      (size @-> cell ^-> judges unit)
      (fun n c o -> create_ref { dw = c.w; r = host c.w } B.Device n c o)
      (fun n c ->
        one c (fun () ->
            create_sys
              ({ d = Rig.host; skind = Host; p = None; spoiled = false }, c.sw)
              B.Device n c))
  in
  let of_bigarray =
    command "of_bigarray"
      (device ^-> size @-> seed @-> cell ^-> judges unit)
      (fun v n k c o ->
        one_world [ dw v; c.w ] o (fun () -> array_ref v n k c o))
      (fun v n k c ->
        cells ~ws:[ snd v; c.sw ] [ c.cl ] (fun () -> array_sys v n k c))
  in
  let file =
    command "file"
      (device ^-> size @-> seed @-> Gen.bool @-> cell ^-> judges unit)
      (fun v n k wr c o ->
        one_world [ dw v; c.w ] o (fun () -> file_ref v n k wr c o))
      (fun v n k wr c ->
        cells ~ws:[ snd v; c.sw ] [ c.cl ] (fun () -> file_sys v n k wr c))
  in
  let borrow =
    command "borrow"
      (device ^-> cell ^-> cell ^-> judges bool)
      (fun v a b o ->
        one_world [ dw v; a.w; b.w ] o (fun () -> borrow_ref v a b o))
      (fun v a b ->
        cells
          ~ws:[ snd v; a.sw; b.sw ]
          [ a.cl; b.cl ]
          (fun () -> borrow_sys v a b))
  in
  let view =
    command "view"
      (eighth @-> eighth @-> cell ^-> cell ^-> judges unit)
      (fun x y a b o -> one_world [ a.w; b.w ] o (fun () -> view_ref x y a b o))
      (fun x y a b ->
        cells ~ws:[ a.sw; b.sw ] [ a.cl; b.cl ] (fun () -> view_sys x y a b))
  in
  let fill =
    command "fill"
      (seed @-> last @-> cell ^-> judges unit)
      fill_ref
      (fun k l c -> one c (fun () -> fill_sys k l c))
  in
  let read =
    command "read"
      (last @-> cell ^-> judges octets)
      read_ref
      (fun l c -> one c (fun () -> read_sys l c))
  in
  let copy =
    command "copy"
      (last @-> cell ^-> cell ^-> judges unit)
      (fun l a b o -> one_world [ a.w; b.w ] o (fun () -> copy_ref l a b o))
      (fun l a b ->
        cells ~ws:[ a.sw; b.sw ] [ a.cl; b.cl ] (fun () -> copy_sys l a b))
  in
  let copy_prefix =
    command "copy prefix"
      (last @-> cell ^-> cell ^-> judges unit)
      (fun l a b o ->
        one_world [ a.w; b.w ] o (fun () -> copy_prefix_ref l a b o))
      (fun l a b ->
        cells ~ws:[ a.sw; b.sw ] [ a.cl; b.cl ] (fun () ->
            copy_prefix_sys l a b))
  in
  let transfer =
    command "transfer"
      (device ^-> device ^-> size @-> seed @-> judges octets)
      (fun v v' n k o ->
        one_world [ dw v; dw v' ] o (fun () -> transfer_ref v v' n k o))
      (fun v v' n k ->
        one_world_sys [ snd v; snd v' ];
        transfer_sys v v' n k)
  in
  let make_copy =
    command "make copy"
      (device ^-> cell ^-> cell ^-> subs ^-> judges unit)
      (fun v a b sc o ->
        one_world [ dw v; a.w; b.w; sc.subw ] o @@ fun () ->
        with_buf a o @@ fun x ->
        with_buf b o @@ fun y -> make_ref v (Some (prefixes x y)) sc o)
      (fun v a b sc ->
        cells ~ws:[ snd v; a.sw; b.sw; sc.ssw ] [ a.cl; b.cl; sc.scl ]
        @@ fun () -> make_sys v (Some (a, b)) sc)
  in
  let make_own =
    command "make own copy"
      (device ^-> size @-> subs ^-> judges unit)
      (fun v n sc o ->
        one_world [ dw v; sc.subw ] o (fun () -> make_own_ref v n sc o))
      (fun v n sc ->
        cells ~ws:[ snd v; sc.ssw ] [ sc.scl ] (fun () -> make_own_sys v n sc))
  in
  let make_fill =
    command "make fill"
      (device ^-> subs ^-> judges unit)
      (fun v sc o ->
        one_world [ dw v; sc.subw ] o (fun () -> make_ref v None sc o))
      (fun v sc ->
        cells ~ws:[ snd v; sc.ssw ] [ sc.scl ] (fun () -> make_sys v None sc))
  in
  let make_held =
    command "make held copy"
      (device ^-> cell ^-> cell ^-> holds ^-> subs ^-> judges unit)
      (fun v a b h sc o ->
        one_world [ dw v; a.w; b.w; h.hw; sc.subw ] o @@ fun () ->
        with_buf a o @@ fun x ->
        with_buf b o @@ fun y ->
        match h.h with
        | Some _ when not h.dropped ->
            make_ref ~hold:h v (Some (prefixes x y)) sc o
        | _ -> ( match o with Error Empty -> () | _ -> fail "an empty hold"))
      (fun v a b h sc ->
        cells
          ~ws:[ snd v; a.sw; b.sw; h.hsw; sc.ssw ]
          [ a.cl; b.cl; h.hcl; sc.scl ]
        @@ fun () -> make_sys ~hold:h v (Some (a, b)) sc)
  in
  let submit =
    command "submit"
      (last @-> cell ^-> cell ^-> subs ^-> judges int)
      (fun l a b sc o ->
        one_world [ a.w; b.w; sc.subw ] o (fun () -> submit_ref l a b sc o))
      (fun l a b sc ->
        cells ~ws:[ a.sw; b.sw; sc.ssw ] [ a.cl; b.cl; sc.scl ] (fun () ->
            submit_sys l a b sc))
  in
  let hold =
    command "hold"
      (cell ^-> cell ^-> holds ^-> judges unit)
      (fun a b h o ->
        one_world [ a.w; b.w; h.hw ] o (fun () -> hold_ref a b h o))
      (fun a b h ->
        cells ~ws:[ a.sw; b.sw; h.hsw ] [ a.cl; b.cl; h.hcl ] (fun () ->
            hold_sys a b h))
  in
  let claim =
    command "claim"
      (cell ^-> judges unit)
      claim_ref
      (fun c -> one c (fun () -> claim_sys c))
  in
  let release =
    command "release"
      (cell ^-> judges unit)
      release_ref
      (fun c -> one c (fun () -> release_sys c))
  in
  let with_ =
    command "with"
      (cell ^-> cell ^-> judges bool)
      (fun a b o -> one_world [ a.w; b.w ] o (fun () -> with_ref a b o))
      (fun a b ->
        cells ~ws:[ a.sw; b.sw ] [ a.cl; b.cl ] (fun () -> with_sys a b))
  in
  let consume =
    command "consume"
      (cell ^-> cell ^-> judges bool)
      (fun a b o -> one_world [ a.w; b.w ] o (fun () -> consume_ref a b o))
      (fun a b ->
        cells ~ws:[ a.sw; b.sw ] [ a.cl; b.cl ] (fun () -> consume_sys a b))
  in
  let wait =
    command "wait"
      (access @-> cell ^-> judges unit)
      wait_ref
      (fun x c -> one c (fun () -> wait_sys x c))
  in
  let wait_device =
    command "wait device"
      (device ^-> judges unit)
      wait_device_ref wait_device_sys
  in
  let facts =
    command "facts"
      (device ^-> judges (triple int int bool))
      facts_ref facts_sys
  in
  let run = command "run" (device ^-> returns unit) unit_ref run_sys in
  let fault =
    command "fault"
      (device ^-> returns unit)
      arm_ref
      (polled_cmd (fun p -> P.fault p "the model's fault"))
  in
  let fail_ =
    command "fail" (device ^-> returns unit) arm_ref (polled_cmd P.fail)
  in
  let budget =
    command "budget" (budgets @-> device ^-> judges unit) budget_ref budget_sys
  in
  let free_cache =
    command "free cache" (device ^-> judges unit) free_cache_ref free_cache_sys
  in
  let barrier =
    command "barrier"
      (last @-> cell ^-> judges unit)
      barrier_ref
      (fun l c -> one c (fun () -> barrier_sys l c))
  in
  let drop =
    command "drop"
      (cell ^-> returns unit)
      (fun c -> c.rb <- None)
      (fun c -> one c (fun () -> Atomic.set c.cell None))
  in
  let drop_sub =
    command "drop submission"
      (subs ^-> returns unit)
      (fun c -> c.sub <- None)
      (fun c -> cells [ c.scl ] (fun () -> Atomic.set c.ss None))
  in
  let drop_hold =
    command "drop hold"
      (holds ^-> returns unit)
      (fun c -> c.dropped <- true)
      (fun c ->
        cells [ c.hcl ] (fun () ->
            Atomic.set c.used true;
            Atomic.set c.sh None))
  in
  let collect =
    command "collect"
      (world ^-> returns unit)
      unit_ref
      (fun _ -> Gc.full_major ())
  in
  (* A command listed [n] times is drawn [n] times as often. *)
  let times n c = List.init n (fun _ -> c) in
  List.concat
    [
      [ start; new_hold ];
      times 2 new_sub;
      times 2 open_;
      times 3 new_cell;
      times 8 create;
      times 3 create_host;
      [ of_bigarray ];
      times 2 file;
      [ of_bigarray ];
      times 3 borrow;
      times 2 view;
      times 3 fill;
      times 4 read;
      [ copy ];
      times 4 copy_prefix;
      times 2 transfer;
      times 3 make_copy;
      times 2 make_own;
      times 2 make_fill;
      (* A held copy takes six values, more than two domains' prefix makes. *)
      (if two then [] else times 2 make_held);
      times 4 submit;
      times 3 hold;
      [ claim; release; with_; consume; wait; wait_device; facts ];
      times 2 run;
      [ fault; fail_; budget; free_cache; barrier ];
      [ drop ];
      [ drop_sub ];
      times 2 drop_hold;
      times 3 collect;
      (if fork then
         times 2
           (command "fork"
              (world ^-> judges forked)
              fork_ref
              (fun w -> cells [] (fun () -> fork_sys w)))
       else []);
    ]
