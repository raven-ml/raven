(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The driver's work runs as programs hand it over, through rig. *)

open Windtrap
module N = Rig_nv
module A = Rig_nv_abi
module B = Rig.Buffer
module S = Rig_nv_support
module H = Rig_gpu_support.Host

let strf = Printf.sprintf
let second = 1_000_000_000
let host = S.host
let address = S.address
let alloc g kind n = require_some (N.alloc g kind n)
let still = Rig_gpu_support.still

(* The [n] unsigned 32-bit words at [a]. *)
let words32 a n = List.init n (fun i -> H.get32 (a + (4 * i)))

(* [f ()], the host word [w] set to [x] after it whether it returns or raises,
   so that work held on [w] ends. *)
let releasing w x f = Fun.protect ~finally:(fun () -> H.set64 (host w) x) f

(* The entry of a segment that writes [x] at [a] once the channel's earlier work
   completed, and of one that waits until the word at [a] is at least [x]. *)
let release l a x = S.segment l (A.Method.release System a x)
let acquire l a x = S.segment l (A.Method.acquire a x)

(* A kernel that copies [n] bytes from the address [src] to [dst], starting [ns]
   nanoseconds late. *)
let copy_after l k ?(ns = 0) ~dst ~src n =
  S.launch l k "copy_after" ~blocks:1 [ ns; dst; src; n ]

(* Driver devices the tests stop themselves *)

(* [g] handed to rig under a name of its own, for the tests of the driver's own
   stop: rig hands it work, and the test stops [g] itself. The device then stays
   open in rig until the process ends, as no driver stops twice. *)
let handed =
  let n = Atomic.make 0 in
  fun g ->
    let name = strf "NV:own-%d" (Atomic.fetch_and_add n 1) in
    require_ok (Rig.open_ (module N) ~name (fun () -> Ok g))

(* Submits [ps] on [d], and is their value. *)
let submit d ps =
  let s = Rig.Submission.make ~reads:0 ~writes:0 d ps in
  let run = Rig.Submission.Run.make () in
  Rig.Point.value (Rig.submit s ~run ~reads:[||] ~writes:[||] ~waits:[||])

(* Paths *)

(* A path whose RM is of release [release] and whose every function fails the
   test: [make] must refuse it before calling any. *)
let path release : unit N.path =
  let called what = fail (what ^ " was called") in
  let rm =
    {
      N.release;
      client = 0;
      alloc = (fun ~parent:_ _ _ -> called "rm.alloc");
      control = (fun _ _ _ -> called "rm.control");
      free = (fun ~parent:_ _ -> called "rm.free");
    }
  in
  {
    N.key = Type.Id.make ();
    index = 0;
    rm;
    device = 0;
    subdevice = 0;
    vaspace = 0;
    gpu =
      {
        channel_class = 0;
        compute_class = 0;
        copy_class = 0;
        sm_version = 0;
        gpcs = 1;
        tpcs_per_gpc = 1;
        sms_per_tpc = 1;
        warps_per_sm = 1;
      };
    budget = 0;
    doorbell = 0;
    alloc = (fun _ _ -> called "alloc");
    map_host = Some (fun _ _ -> called "map_host");
    reaches = (fun _ -> called "reaches");
    map_peer = (fun _ -> called "map_peer");
    free = (fun _ -> called "free");
    register = (fun _ -> called "register");
    unregister = (fun _ -> called "unregister");
    check = (fun () -> called "check");
    hang_ms = None;
    stop = (fun () -> called "stop");
  }

(* A path that refuses its [n]th call (an RM object, a control, memory or a
   registration), [0] for none, over host pages for the memory the host
   addresses: what the device holds of it at any time. *)
module Fake = struct
  type t = {
    refuse : int;
    mutable calls : int;
    mutable objects : (int * int) list; (* object, parent *)
    mutable memory : (int * int option * int) list;
    mutable registered : int list;
    mutable wrong : string list; (* what the device gave back wrongly *)
    mutable next : int;
    mutable at : int option; (* the address of the path's next memory *)
    mutable frees : bool; (* whether the RM frees objects *)
    mutable faults : bool; (* whether RM frees and path memory calls raise *)
    mutable report : string option; (* a fault the path reports *)
    mutable stops : [ `Stopped | `Unknown ]; (* what the path's stop answers *)
    mutable bar_room : bool; (* whether [`Bar] memory is given *)
    mutable kinds : string list; (* the kinds of memory given, newest first *)
    mutable on_free : unit N.memory -> unit; (* runs before each free *)
  }

  (* The path's own objects. *)
  let device = 1
  let subdevice = 2
  let vaspace = 3

  let refused f =
    f.calls <- f.calls + 1;
    f.calls = f.refuse

  let fresh f =
    f.next <- f.next + 1;
    f.next

  let rec under f h parent =
    h = parent
    ||
    match List.assoc_opt h f.objects with
    | Some p -> under f p parent
    | None -> false

  let rm f =
    {
      N.release = 615;
      client = 4;
      alloc =
        (fun ~parent _ _ ->
          if refused f then Error "refused"
          else
            let h = fresh f in
            f.objects <- (h, parent) :: f.objects;
            Ok h);
      control = (fun _ _ _ -> if refused f then Error "refused" else Ok ());
      free =
        (fun ~parent:_ h ->
          if f.faults then raise (N.Fault "the GPU fell off the bus");
          if not f.frees then Error "refused"
          else begin
            if not (List.mem_assoc h f.objects) then
              f.wrong <- Printf.sprintf "freed object %d" h :: f.wrong;
            f.objects <- List.filter (fun (o, _) -> not (under f o h)) f.objects;
            Ok ()
          end);
    }

  let alloc f kind n =
    if f.faults then raise (N.Fault "the GPU fell off the bus");
    if refused f || (kind = `Bar && not f.bar_room) then None
    else
      let () =
        f.kinds <-
          (match kind with
          | `Gpu -> "`Gpu"
          | `Bar -> "`Bar"
          | `System -> "`System")
          :: f.kinds
      in
      let bytes = (n + H.page - 1) / H.page * H.page in
      let host = if kind = `Gpu then None else Some (H.pages bytes) in
      let address =
        match f.at with Some a -> a | None -> fresh f * (1 lsl 24)
      in
      f.memory <- (address, host, bytes) :: f.memory;
      Some { N.address; host; handle = fresh f; data = () }

  let free f (m : unit N.memory) =
    if f.faults then raise (N.Fault "the GPU fell off the bus");
    f.on_free m;
    match List.find_opt (fun (a, _, _) -> a = m.address) f.memory with
    | None -> f.wrong <- Printf.sprintf "freed memory 0x%x" m.address :: f.wrong
    | Some ((_, host, bytes) as x) ->
        f.memory <- List.filter (fun y -> y != x) f.memory;
        Option.iter (fun a -> H.free_pages a bytes) host

  (* One key, so that fake devices map each other's memory. *)
  let key : unit Type.Id.t = Type.Id.make ()

  let path f : unit N.path =
    {
      N.key;
      index = 0;
      rm = rm f;
      device;
      subdevice;
      vaspace;
      gpu =
        {
          channel_class = 0xc86f;
          compute_class = 0xc9c0;
          copy_class = 0xc7b5;
          sm_version = 0x809;
          gpcs = 11;
          tpcs_per_gpc = 6;
          sms_per_tpc = 2;
          warps_per_sm = 48;
        };
      budget = 1 lsl 30;
      doorbell = H.pages H.page;
      alloc = alloc f;
      map_host = Some (fun _ n -> alloc f `System n);
      reaches = (fun _ -> false);
      map_peer = (fun _ -> alloc f `Gpu H.page);
      free = free f;
      register =
        (fun c ->
          if refused f then Error "refused"
          else (
            f.registered <- c :: f.registered;
            Ok ()));
      unregister =
        (fun c ->
          if not (List.mem c f.registered) then
            f.wrong <- Printf.sprintf "unregistered %d" c :: f.wrong;
          f.registered <- List.filter (( <> ) c) f.registered;
          Ok ());
      check = (fun () -> Option.iter (fun r -> raise (N.Fault r)) f.report);
      hang_ms = None;
      stop = (fun () -> f.stops);
    }

  let make refuse =
    {
      refuse;
      calls = 0;
      objects = [];
      memory = [];
      registered = [];
      wrong = [];
      next = 16;
      at = None;
      frees = true;
      faults = false;
      report = None;
      stops = `Unknown;
      bar_room = true;
      kinds = [];
      on_free = ignore;
    }
end

(* Whatever call the path refuses, make gives back all it took; a device made
   and stopped keeps only its timeline word. *)
let refused_makes () =
  let rec each n =
    let f = Fake.make n in
    let at = strf "refusing call %d" n in
    (match N.make (Fake.path f) with
    | Error e when f.calls < n -> failf "make refused nothing: %s" e
    | Error _ ->
        equal (list (pair int int)) ~msg:(at ^ ": RM objects") [] f.objects;
        equal int ~msg:(at ^ ": memory") 0 (List.length f.memory);
        equal (list int) ~msg:(at ^ ": registrations") [] f.registered
    | Ok g ->
        N.stop g ~fault:None;
        equal (list (pair int int)) ~msg:(at ^ ": RM objects") [] f.objects;
        equal (list int) ~msg:(at ^ ": memory")
          [ address (S.word g) ]
          (List.map (fun (a, _, _) -> a) f.memory);
        equal (list int) ~msg:(at ^ ": registrations") [] f.registered);
    equal (list string) ~msg:(at ^ ": given back wrongly") [] f.wrong;
    if f.calls >= n then each (n + 1)
  in
  each 1

(* A cubin is placed over a region the caller allocated: neither [image] nor
   [lay] asks the path for anything. *)
let image_asks_nothing () =
  let f = Fake.make 0 in
  let g = require_ok (N.make (Fake.path f)) in
  let bin = S.fixture "kernels_sm89.cubin" in
  let n, lay =
    match require_ok (N.image g bin) with
    | Place (n, lay) -> (n, lay)
    | Loaded _ -> fail "an image with nothing to place"
  in
  let r = require_some (N.alloc g Device n) in
  let calls = f.calls and memory = List.length f.memory in
  let i, _ = lay r in
  equal int ~msg:"the path's calls" calls f.calls;
  equal int ~msg:"the path's memory" memory (List.length f.memory);
  N.unload g i;
  N.free g r;
  N.stop g ~fault:None

(* A device stopped with an image loaded: stop ends the image, and the code
   region is the caller's to free, after which the device keeps only its
   word. *)
let stop_with_image () =
  let f = Fake.make 0 in
  let g = require_ok (N.make (Fake.path f)) in
  let n, lay =
    match require_ok (N.image g (S.fixture "kernels_sm89.cubin")) with
    | Place (n, lay) -> (n, lay)
    | Loaded _ -> fail "an image with nothing to place"
  in
  let r = require_some (N.alloc g Device n) in
  ignore (lay r);
  N.stop g ~fault:None;
  N.free g r;
  equal (list int) ~msg:"memory"
    [ address (S.word g) ]
    (List.map (fun (a, _, _) -> a) f.memory);
  equal (list string) ~msg:"given back wrongly" [] f.wrong

(* The memory a path answers lies below 2^40, the widest address a channel's
   semaphore takes, up to its last byte; memory past it goes back to the path,
   and the call raises. *)
let address_limit () =
  let f = Fake.make 0 and f' = Fake.make 0 in
  let g = require_ok (N.make (Fake.path f)) in
  let g' = require_ok (N.make (Fake.path f')) in
  let limit = 1 lsl 40 and n = 2 * H.page in
  let r' = require_some (N.alloc g' Device n) in
  let held = List.length f.memory in
  let calls =
    [
      ("Device", "Rig_nv.alloc", fun () -> N.alloc g Device n);
      ("Pinned", "Rig_nv.alloc", fun () -> N.alloc g Pinned n);
      ("Mapped", "Rig_nv.alloc", fun () -> N.alloc g Mapped n);
      ("map_host", "Rig_nv.map_host", fun () -> N.map_host g 0 n);
      ("map_peer", "Rig_nv.map_peer", fun () -> N.map_peer g g' r');
    ]
  in
  let below (what, _, call) =
    f.at <- Some (limit - n);
    let r = require_some ~msg:(what ^ " ending at 2^40") (call ()) in
    equal int ~msg:(what ^ ": its address") (limit - n) (address r);
    N.free g r
  in
  let past (what, fn, call) =
    f.at <- Some (limit - H.page);
    raises_match
      ~msg:(what ^ " ending past 2^40")
      (Exn.invalid_arg ~substring:(fn ^ ":"))
      (fun () -> ignore (call ()));
    equal int ~msg:(what ^ ": the path's memory") held (List.length f.memory)
  in
  List.iter below calls;
  List.iter past calls;
  f.at <- None;
  equal (list string) ~msg:"given back wrongly" [] f.wrong;
  N.free g' r';
  N.stop g ~fault:None;
  N.stop g' ~fault:None

(* A device whose channels the RM stopped on a fault, and kept: nothing of it
   runs, so stop raises the word, though its memory stays the path's. *)
let faulted_stop () =
  let f = Fake.make 0 in
  let g = require_ok (N.make (Fake.path f)) in
  ignore (submit (handed g) [||]);
  let word = address (S.word g) in
  let fault (a, host, bytes) =
    match host with
    | Some h when bytes = H.page && a <> word ->
        for i = 0 to (H.page / 8) - 1 do
          H.set64 (h + (8 * i)) (-1)
        done
    | Some _ | None -> ()
  in
  List.iter fault f.memory;
  f.frees <- false;
  N.stop g ~fault:None;
  equal int ~msg:"the word" 1 (N.signaled g);
  greater int ~msg:"memories the path still gives" ~than:1
    (List.length f.memory)

(* Mapped memory is the BAR's while it has room, then None: no other memory
   is taken. *)
let mapped_full () =
  let f = Fake.make 0 in
  let g = require_ok (N.make (Fake.path f)) in
  let r = require_some ~msg:"with room" (N.alloc g Mapped 64) in
  equal (option string) ~msg:"with room" (Some "`Bar") (List.nth_opt f.kinds 0);
  f.bar_room <- false;
  let held = List.length f.memory in
  is_none ~msg:"without room" (N.alloc g Mapped 64);
  equal int ~msg:"the path's memory" held (List.length f.memory);
  N.free g r;
  N.stop g ~fault:None

(* Compiled code links local memory through the capability, whose [local]
   answers a failure as [Error]. *)
let failing_local () =
  let f = Fake.make 0 in
  let g = require_ok (N.make (Fake.path f)) in
  f.faults <- true;
  ignore (require_error ((S.gpu g).local 1024) : string);
  f.faults <- false;
  N.stop g ~fault:None

(* A path whose frees raise Fault, as one whose GPU is lost: free and stop still
   return, as rig calls them after a loss. *)
let failing_frees () =
  let f = Fake.make 0 in
  let g = require_ok (N.make (Fake.path f)) in
  let r = require_some (N.alloc g Pinned 64) in
  f.faults <- true;
  N.free g r;
  N.stop g ~fault:None

(* A device whose channels the RM keeps at stop: the path's [`Stopped] says none
   of its work runs, so the word reaches the last value and the memory goes
   back; [`Unknown] keeps both, as the work may still run. *)
let path_stops () =
  List.iter
    (fun answer ->
      let f = Fake.make 0 in
      let g = require_ok (N.make (Fake.path f)) in
      ignore (submit (handed g) [||]);
      f.frees <- false;
      f.stops <- answer;
      N.stop g ~fault:None;
      let stopped = answer = `Stopped in
      let what = if stopped then "`Stopped" else "`Unknown" in
      equal int ~msg:(what ^ ": the word")
        (if stopped then 1 else 0)
        (N.signaled g);
      equal bool
        ~msg:(what ^ ": only the word's memory left")
        stopped
        (List.map (fun (a, _, _) -> a) f.memory = [ address (S.word g) ]))
    [ `Stopped; `Unknown ]

(* A fault the path reports surfaces from sleep while a value is outstanding. *)
let path_check () =
  let f = Fake.make 0 in
  let g = require_ok (N.make (Fake.path f)) in
  ignore (submit (handed g) [||]);
  f.report <- Some "the GPU fell off the bus";
  (match N.sleep g ~seen:0 ~still_ms:1 with
  | () -> fail "no Fault for the path's report"
  | exception N.Fault why ->
      contains ~msg:"the report" ~sub:"the GPU fell off the bus" why);
  f.report <- None;
  N.stop g ~fault:None

let paths =
  group ~timeout:10. "paths"
    [
      test "an NVIDIA display controller is a GPU" (fun () ->
          equal (list bool)
            [ true; true; false; false ]
            (List.map
               (fun (vendor, class_) -> N.is_gpu ~vendor ~class_)
               [
                 (0x10de, 0x030000);
                 (0x10de, 0x030200);
                 (0x10de, 0x040300);
                 (0x1002, 0x030000);
               ]));
      test "make refuses an RM of an unknown release" (fun () ->
          let e = require_error (N.make (path 999)) in
          contains ~sub:"999" e);
      test
        "make refused at any call gives back what it took, and a stopped \
         device keeps only its word"
        refused_makes;
      test "image and lay ask the path for nothing" image_asks_nothing;
      test "stop gives back what the path says no work can use" path_stops;
      test "sleep raises the fault the path reports" path_check;
      test "free and stop raise no Fault when the path's frees do" failing_frees;
      test "local memory answers the path's Fault as Error" failing_local;
      test "stop raises the word of channels the RM stopped on a fault"
        faulted_stop;
      test "Mapped memory is None once the BAR is full" mapped_full;
      test
        "a device stopped with an image keeps only its word once its code is \
         freed"
        stop_with_image;
      test "memory a path answers past 2^40 goes back, and the call raises"
        address_limit;
    ]

(* Facts *)

let digits s = String.for_all (function '0' .. '9' -> true | _ -> false) s

let facts () =
  S.with_driver @@ fun g ->
  let f = N.facts g in
  starts_with ~affix:"sm_" f.arch;
  equal bool
    ~msg:(strf "%s ends in digits" f.arch)
    true
    (String.length f.arch > 3
    && digits (String.sub f.arch 3 (String.length f.arch - 3)));
  greater int ~msg:"budget" ~than:0 f.budget;
  equal
    (list (pair string (list string)))
    [ ("COMPUTE:0", [ "Words"; "Launch" ]); ("COPY:0", [ "Words"; "Copy" ]) ]
    (List.map
       (fun (q : Rig_edge.queue) ->
         ( q.name,
           List.map
             (function
               | Rig_edge.Words -> "Words"
               | Fill -> "Fill"
               | Copy -> "Copy"
               | Launch -> "Launch")
             q.runs ))
       f.queues);
  equal bool ~msg:"completion is the store" true (f.completion = Store);
  equal (list bool) ~msg:"waits on stores, hosts, objects"
    [ true; true; false ]
    [ f.waits.stores; f.waits.hosts; f.waits.objects ];
  equal int ~msg:"most waits" 256 f.waits.most;
  equal bool ~msg:"may block" false f.may_block;
  let w = N.locate f.word in
  equal nativeint ~msg:"the word's handle is its address"
    (Nativeint.of_int (Option.get w.address))
    w.handle;
  equal int ~msg:"the word starts at 0" 0 (H.get64 (Option.get w.host));
  equal int ~msg:"signaled reads the word" 0 (N.signaled g)

let capability () =
  S.with_driver @@ fun g ->
  let c = S.gpu g in
  mem int ~msg:"compute class" c.compute_class [ 0xc7c0; 0xc9c0; 0xcec0 ];
  List.iter
    (fun (what, n) -> greater int ~msg:what ~than:0 n)
    [
      ("gpcs", c.gpcs);
      ("tpcs_per_gpc", c.tpcs_per_gpc);
      ("sms_per_tpc", c.sms_per_tpc);
      ("warps_per_sm", c.warps_per_sm);
    ];
  at_least int ~msg:"shared window" ~than:(1 lsl 40) c.shared_window;
  at_least int ~msg:"local window" ~than:(1 lsl 40) c.local_window;
  equal (result unit string) ~msg:"local 0" (Ok ()) (c.local 0)

(* The host addresses every kind but Device memory, and the GPU all of them
   below 2^40. *)
let kinds_of_memory () =
  S.with_driver @@ fun g ->
  List.iter
    (fun (name, kind) ->
      let r = alloc g kind 4096 in
      equal bool
        ~msg:(name ^ ": the host addresses it")
        (kind <> Device)
        (Option.is_some (N.locate r).host);
      less int ~msg:(name ^ ": its address") ~than:(1 lsl 40) (address r);
      N.free g r)
    [ ("Device", Device); ("Pinned", Pinned); ("Mapped", Mapped) ]

let facts =
  group ~timeout:60. "facts"
    [
      test "states a GPU's facts" facts;
      test "declares the GPU to compiled code" capability;
      test "gives memory of each kind where the host and the GPU reach it"
        kinds_of_memory;
    ]

(* Memory *)

(* A copy of more bytes than the copy engine moves at once, out to GPU memory at
   an odd offset and back. *)
let long_copy () =
  S.with_ @@ fun t ->
  let n = A.Method.max_copy + 4099 in
  let h, ha = S.shared t n in
  let d = B.view (B.create t.d (n + 1)) ~first:1 ~length:n in
  S.pattern ha n 3;
  S.run t [| S.copy ~dst:d h |];
  S.pattern ha n 4;
  S.run t [| S.copy ~dst:h d |];
  equal int ~msg:"the first byte that differs" (-1) (S.mismatch ha n 3)

let refusal () =
  S.with_driver @@ fun g ->
  for _ = 1 to 100 do
    is_none ~msg:"twice the budget"
      (N.alloc g Device (2 * (N.facts g).budget))
  done;
  N.free g (require_some ~msg:"64 MiB after" (N.alloc g Device (64 lsl 20)))

(* Host memory no page backs is refused: map_host answers None. *)
let refused_host () =
  S.with_driver @@ fun g ->
  let p = H.pages H.page in
  H.free_pages p H.page;
  equal bool ~msg:"an unmapped page" true (Option.is_none (N.map_host g p 8));
  equal bool ~msg:"address 0" true (Option.is_none (N.map_host g 0 8))

let memory =
  group ~timeout:120. "memory"
    [
      test "a copy longer than the copy engine's is the identity" long_copy;
      test
        "an allocation past the GPU's memory is None, and gives back what it \
         took"
        refusal;
      test "map_host answers None for host memory no page backs" refused_host;
    ]

(* Work *)

let launch () =
  S.with_ @@ fun t ->
  let k = S.kernels t in
  let l = S.launches t.g in
  let n = 1000 in
  let out = alloc t.g Pinned (4 * n) in
  S.run t
    [| S.words (S.launch l k "double_index" ~blocks:4 [ address out; n ]) |];
  equal (list int)
    (List.init n (fun i -> 2 * i))
    (words32 (host out) n);
  S.free_launches l;
  N.free t.g out

let misuse () =
  S.with_driver @@ fun g ->
  let raises name f = raises_match ~msg:name Exn.invalid_arg f in
  raises "alloc of 0 bytes" (fun () -> N.alloc g Device 0);
  raises "map_host of 0 bytes" (fun () -> N.map_host g (host (S.word g)) 0)

(* A wait on another device's word in host memory wherever the process placed
   it, such as above 2^40, the widest address a channel's semaphore names: a
   Polled device's. *)
let high_word () =
  S.with_ @@ fun t ->
  let module P = Rig_support.Polled in
  let pd, p = P.open_ "POLLED:high" in
  Fun.protect ~finally:(fun () -> Rig.close pd) @@ fun () ->
  let word = require_some (P.locate (P.facts p).word).host in
  at_least int ~msg:"the word's host address" ~than:(1 lsl 40) word;
  let empty = Rig.Submission.make ~reads:0 ~writes:0 pd [||] in
  let run = Rig.Submission.Run.make () in
  let point = Rig.submit empty ~run ~reads:[||] ~writes:[||] ~waits:[||] in
  let b = alloc t.g Pinned 16 in
  H.set64 (host b) 0;
  let l = S.launches t.g in
  let s =
    Rig.Submission.make ~reads:0 ~writes:0 t.d
      [| S.words (release l (address b) 9) |]
  in
  S.watchdog "a wait on a high host word" (fun () ->
      let v =
        Rig.Point.value
          (Rig.submit s ~run ~reads:[||] ~writes:[||] ~waits:[| point |])
      in
      still ~msg:"the word" int (v - 1) (fun () -> N.signaled t.g) ~ms:20;
      ignore (P.run p);
      S.wait t v);
  equal int ~msg:"released" 9 (H.get64 (host b));
  S.free_launches l;
  N.free t.g b

(* Parts NV's channels have no room for, refused when submitted: a ring entry
   cut in half, more ring words than a segment holds, and more parts than the
   rings hold. No value is assigned. *)
let refused () =
  S.with_ @@ fun t ->
  let refuses msg ps =
    raises_match ~msg
      (Exn.invalid_arg ~substring:"never fit")
      (fun () -> S.submit t ps)
  in
  refuses "one ring word" [| S.words [| 0 |] |];
  refuses "32,768 ring words" [| S.words (Array.make 32_768 0) |];
  refuses "65,536 parts" (Array.make 65_536 (S.words [||]));
  equal int ~msg:"values assigned" 0 (Rig.submitted t.d)

let work =
  group ~timeout:60. "work"
    [
      test "a kernel scheduled from a ring entry computes" launch;
      test "an allocation or mapping of no bytes raises" misuse;
      test "a wait holds work on another device's host word above 2^40"
        high_word;
      test "a submission the rings have no room for raises" refused;
    ]

(* Launches, through rig. The conformance suite states the laws every driver's
   launches keep; these are NV's. *)

module Sub = Rig.Submission
module Run = Rig.Submission.Run

let launch_image =
  Rig_gpu_support.loader (fun () -> S.fixture "launch_sm89.cubin")

let kernel_image =
  Rig_gpu_support.loader (fun () -> S.fixture "kernels_sm89.cubin")

let refer at slot = { Sub.at; slot }

(* [kernel] of [image] on COMPUTE:0, with [params] bytes of parameters. *)
let launch_part ?(after = [||]) image kernel ~params refs =
  {
    Sub.queue = "COMPUTE:0";
    after;
    work = Launch { image; kernel; params; refs };
  }

let copy_part ?(after = [||]) ~dst src =
  { Sub.queue = "COPY:0"; after; work = Copy { src; dst } }

(* The [n] 32-bit words of [b] from byte [at], copied to the host. *)
let words b ~at n =
  let h = B.create Rig.host (4 * n) in
  B.copy ~src:(B.view b ~first:at ~length:(4 * n)) ~dst:h;
  let a = B.bigarray Bigarray.int32 h in
  Array.init n (fun i -> Int32.to_int a.{i} land 0xffff_ffff)

let submitted s run ~reads ~writes =
  Rig.submit s ~run ~reads ~writes ~waits:[||]

(* Stores [ids]'s parameters into the block [k] of [run]: its words from [out]
   [offset] bytes on hold [a + b * k]. *)
let ids_params run k ~offset ~a ~b =
  Run.int64 run k 0 offset;
  Run.int64 run k 8 a;
  Run.int32 run k 16 b;
  Run.float32 run k 20 0.

(* A launch of [kernel] (fixtures/launch.cu) over [groups] of [threads] with
   [shared] bytes of dynamic shared memory, on [d], into a buffer of its own:
   its submission and run. *)
let launching ?(kernel = "ids") ?(params = 24) d ~groups:(gx, gy, gz)
    ~threads:(tx, ty, tz) ~shared =
  let image = launch_image d in
  let s =
    Sub.make ~reads:0 ~writes:1 d
      [| launch_part image kernel ~params [| refer 0 0 |] |]
  in
  let run = Run.make () in
  let b = Sub.block s 0 in
  Run.groups run b gx gy gz;
  Run.threads run b tx ty tz;
  Run.shared run b shared;
  (s, run)

(* The most dynamic shared memory a block of [rotate] takes on [g]. *)
let dynamic_shared g =
  let c =
    require_ok ~pp:Format.pp_print_string
      (A.Cubin.of_string (S.fixture "launch_sm89.cubin"))
  in
  let k = require_some (A.Cubin.kernel c "rotate") in
  A.Launch.dynamic_shared
    (require_ok ~pp:Format.pp_print_string (A.Launch.make (S.gpu g) k))

(* Launches past NV's limits are refused before any value, and the device
   stays live. The image's load takes values of its own. *)
let refused_launches () =
  S.with_ @@ fun { d; g } ->
  let out = B.create d 4096 in
  ignore (launch_image d);
  let loaded = Rig.submitted d in
  let refuses msg ?kernel ?params ?(groups = (1, 1, 1)) ?(threads = (1, 1, 1))
      ?(shared = 0) () =
    let s, run = launching ?kernel ?params d ~groups ~threads ~shared in
    raises_match ~msg (Exn.invalid_arg ~substring:"never fit") (fun () ->
        submitted s run ~reads:[||] ~writes:[| out |])
  in
  refuses "65536 groups along y" ~groups:(1, 65536, 1) ();
  refuses "65536 groups along z" ~groups:(1, 1, 65536) ();
  refuses "2^31 groups along x" ~groups:(1 lsl 31, 1, 1) ();
  refuses "1025 threads along x" ~threads:(1025, 1, 1) ();
  refuses "65 threads along z" ~threads:(1, 1, 65) ();
  refuses "2048 threads per group" ~threads:(1024, 2, 1) ();
  refuses "a byte of shared memory past the most, rounded to 128"
    ~kernel:"rotate" ~params:12
    ~shared:(dynamic_shared g + 1)
    ();
  equal int ~msg:"values assigned" loaded (Rig.submitted d);
  equal (option string) ~msg:"the device's loss" None (Rig.lost d);
  let s, run =
    launching d ~groups:(1, 1, 65535) ~threads:(1024, 1, 1) ~shared:0
  in
  ids_params run (Sub.block s 0) ~offset:0 ~a:0 ~b:1;
  let out = B.create d (4 * 65535 * 1024) in
  Rig.wait d (Rig.Point.value (submitted s run ~reads:[||] ~writes:[| out |]))

(* A launch takes as much dynamic shared memory as a block may beside its
   own. *)
let most_shared_memory () =
  S.with_ @@ fun { d; g } ->
  let shared = dynamic_shared g in
  let s, run =
    launching ~kernel:"rotate" ~params:12 d ~groups:(2, 1, 1)
      ~threads:(256, 1, 1) ~shared
  in
  let b = Sub.block s 0 in
  Run.int64 run b 0 0;
  Run.int32 run b 8 5;
  let out = B.create d (4 * 512) in
  Rig.wait d (Rig.Point.value (submitted s run ~reads:[||] ~writes:[| out |]));
  let value i g = (5 + (3 * i) + (7 * g)) land 0xffff_ffff in
  equal (array int) ~msg:"the words"
    (Array.init 512 (fun i ->
         let g = i / 256 in
         value ((g * 256) + ((i + 1) mod 256)) g))
    (words out ~at:0 512)

(* A cached launch submitted with a warm run, its block stored anew each time
   through every setter, allocates nothing. *)
let no_allocation () =
  S.with_ @@ fun { d; _ } ->
  let s, run = launching d ~groups:(1, 1, 1) ~threads:(32, 1, 1) ~shared:0 in
  let writes = [| B.create d 4096 |] in
  let b = Sub.block s 0 in
  let once i =
    Run.groups run b 1 1 1;
    Run.threads run b 32 1 1;
    Run.shared run b 0;
    Run.int64 run b 0 0;
    Run.int64 run b 8 i;
    Run.int32 run b 16 1;
    Run.float32 run b 20 1.5;
    Run.float64 run b 16 2.5;
    Run.int32 run b 16 1;
    ignore (Sys.opaque_identity (submitted s run ~reads:[||] ~writes))
  in
  once 0;
  Rig.wait d (Rig.submitted d);
  let before = Gc.minor_words () in
  for i = 1 to 100 do
    once i
  done;
  let words = int_of_float (Gc.minor_words () -. before) / 100 in
  Rig.wait d (Rig.submitted d);
  equal int ~msg:"words per launch" 0 words

(* Launches with a copy between them on the other channel, each waiting for the
   part before it: the second launch is scheduled after the copy, not chained
   to the first. *)
let launch_copy_launch () =
  S.with_ @@ fun { d; _ } ->
  let image = launch_image d in
  let n = 1024 in
  let out = B.create d (4 * n) and mid = B.create d (4 * n) in
  let dst = B.create d (4 * n) in
  let s =
    Sub.make ~reads:0 ~writes:3 d
      [|
        launch_part image "ids" ~params:24 [| refer 0 0 |];
        copy_part ~after:[| 0 |] ~dst:mid out;
        launch_part ~after:[| 1 |] image "twice" ~params:20
          [| refer 0 2; refer 8 1 |];
      |]
  in
  let run = Run.make () in
  let b0 = Sub.block s 0 and b2 = Sub.block s 2 in
  Run.groups run b0 (n / 256) 1 1;
  Run.threads run b0 256 1 1;
  ids_params run b0 ~offset:0 ~a:11 ~b:3;
  Run.groups run b2 (n / 256) 1 1;
  Run.threads run b2 256 1 1;
  Run.int64 run b2 0 0;
  Run.int64 run b2 8 0;
  Run.int32 run b2 16 1;
  Rig.wait d
    (Rig.Point.value (submitted s run ~reads:[||] ~writes:[| out; mid; dst |]));
  equal (array int)
    (Array.init n (fun k -> (2 * (11 + (3 * k))) + 1))
    (words dst ~at:0 n)

(* Launches past the end of the compute channel's launch ring: each writes its
   own word, through a ref whose offset names it. *)
let launches_wrap () =
  S.with_ @@ fun { d; _ } ->
  let n = 4000 in
  let out = B.create d (4 * n) in
  let s, run = launching d ~groups:(1, 1, 1) ~threads:(1, 1, 1) ~shared:0 in
  let b = Sub.block s 0 in
  for i = 0 to n - 1 do
    ids_params run b ~offset:(4 * i) ~a:i ~b:0;
    ignore (submitted s run ~reads:[||] ~writes:[| out |])
  done;
  Rig.wait d (Rig.submitted d);
  equal (array int) (Array.init n Fun.id) (words out ~at:0 n)

(* A function whose threads keep 2 KiB of local memory runs as a launch: its
   entry made the channel's local memory serve it. *)
let local_launch () =
  S.with_ @@ fun { d; _ } ->
  let n = 1024 in
  let out = B.create d (4 * n) in
  let s =
    Sub.make ~reads:0 ~writes:1 d
      [| launch_part (kernel_image d) "stack" ~params:12 [| refer 0 0 |] |]
  in
  let run = Run.make () in
  let b = Sub.block s 0 in
  Run.groups run b (n / 256) 1 1;
  Run.threads run b 256 1 1;
  Run.int64 run b 0 0;
  Run.int32 run b 8 n;
  Rig.wait d (Rig.Point.value (submitted s run ~reads:[||] ~writes:[| out |]));
  equal (array int)
    (Array.init n (fun i -> (512 * i) + 130816))
    (words out ~at:0 n)

let launches =
  group ~timeout:120. "launches"
    [
      test "launches past NV's limits are refused" refused_launches;
      test "a launch takes the most dynamic shared memory" most_shared_memory;
      test "a cached launch allocates nothing" no_allocation;
      test "a launch after a copy after a launch reads the copy"
        launch_copy_launch;
      test "4,000 launches pass the launch ring's end" launches_wrap;
      test "a function with local memory runs as a launch" local_launch;
    ]

(* Room *)

(* 40,000 submissions, empty ones on COMPUTE:0 between one-byte copies on
   COPY:0, wrap both channels' rings and segments; every byte arrives. *)
let wraps () =
  S.with_ @@ fun t ->
  let n = 20_000 in
  let src, sa = S.shared t n in
  let dst, da = S.shared t n in
  S.pattern sa n 5;
  S.pattern da n 6;
  let byte b i = B.view b ~first:i ~length:1 in
  for i = 0 to n - 1 do
    ignore (S.submit t [||]);
    ignore (S.submit t [| S.copy ~dst:(byte dst i) (byte src i) |])
  done;
  Rig.wait t.d (Rig.submitted t.d);
  equal int ~msg:"the word" (2 * n) (N.signaled t.g);
  equal int ~msg:"the first byte that differs" (-1) (S.mismatch da n 5)

(* 40,000 copies of 16 bytes on COPY:0, each waited for, pass the end of the
   copy channel's segment ring at every offset their words take. *)
let sequential () =
  S.with_ @@ fun t ->
  let h, ha = S.shared t 16 in
  let a = B.create t.d 16 and b = B.create t.d 16 in
  S.pattern ha 16 7;
  S.run t [| S.copy ~dst:a h |];
  for i = 1 to 40_000 do
    if i mod 2 = 1 then S.run t [| S.copy ~dst:b a |]
    else S.run t [| S.copy ~dst:a b |]
  done;
  S.pattern ha 16 8;
  S.run t [| S.copy ~dst:h a |];
  equal int ~msg:"the first byte that differs" (-1) (S.mismatch ha 16 7)

(* The tear probe's rounds, and the host's waits before it lets the compute
   release go: [delays] steps of [delay_step] reads of the word. *)
let tear_rounds = 20_000
let delays = 64
let delay_step = 40

(* The copy engine's release of a 64-bit value and the compute engine's,
   unordered on one word across its 32-bit carry: whichever lands last, the word
   holds one of the two values, never a half of each. The compute release waits
   for a host word the host sets after a delay it sweeps, so that the two land
   in both orders and close together. *)
let carry_tear () =
  S.with_ @@ fun t ->
  let l = S.launches t.g in
  let w, wa = S.shared t 8 and gate, ga = S.shared t 8 in
  let below = (1 lsl 32) - 1 and above = 1 lsl 32 in
  let at = B.address w in
  let by_copy =
    {
      (S.words (S.segment l (A.Method.copy_release System at below))) with
      queue = "COPY:0";
    }
  in
  let by_compute =
    S.words
      (S.segment l
         (A.Method.acquire (B.address gate) 1 @ A.Method.release System at above))
  in
  let torn = ref [] and copy_last = ref 0 and compute_last = ref 0 in
  S.watchdog "unordered releases across a carry" (fun () ->
      for i = 1 to tear_rounds do
        H.set64 wa 0;
        H.set64 ga 0;
        let v = S.submit t [| by_copy; by_compute |] in
        for _ = 1 to i mod delays * delay_step do
          ignore (Sys.opaque_identity (H.get64 wa))
        done;
        H.set64 ga 1;
        Rig.wait t.d v;
        let x = H.get64 wa in
        if x = below then incr copy_last
        else if x = above then incr compute_last
        else torn := x :: !torn
      done);
  (* The host reads and writes [w] and [gate] by address: they live until
     here. *)
  ignore (Sys.opaque_identity (w, gate));
  equal (list int) ~msg:"torn values" [] !torn;
  at_least int ~msg:"rounds the copy's release landed last" ~than:1 !copy_last;
  at_least int ~msg:"rounds the compute release landed last" ~than:1
    !compute_last;
  S.free_launches l

(* A copy's join word, which the copy engine writes as two 32-bit words, takes
   tags of the submission's value times 65,536: its high word first moves at
   value 65,536. Work that waits on the copy runs on past it, in order. *)
let join_carry () =
  S.with_ @@ fun t ->
  let l = S.launches t.g in
  let scratch = alloc t.g Pinned 8 in
  let segment = release l (address scratch) 1 in
  let h, ha = S.shared t 16 in
  let a = B.create t.d 16 in
  S.pattern ha 16 7;
  S.watchdog "work across a join's carry" (fun () ->
      for _ = 1 to 66_000 do
        S.run t [| S.copy ~dst:a h; S.words ~after:[| 0 |] segment |]
      done);
  at_least int ~msg:"the values" ~than:65_536 (N.signaled t.g);
  equal int ~msg:"the compute part's release" 1 (H.get64 (host scratch));
  S.free_launches l;
  N.free t.g scratch

let room =
  group ~timeout:300. "room"
    [
      test "40,000 submissions wrap both channels and every copy arrives" wraps;
      test "40,000 copies one after the other pass the segment ring's end"
        sequential;
      test "work waiting on copies runs on past their joins' 32-bit carry"
        join_carry;
      test "two engines' releases across a 32-bit carry never tear" carry_tear;
    ]

(* Local memory *)

(* The 32-bit word i of the stack kernels' output. *)
let stacked i = (512 * i) + 130816

let local () =
  S.with_ @@ fun t ->
  let k = S.kernels t in
  let l = S.launches t.g in
  let n = 1024 in
  let out = alloc t.g Pinned (4 * n) in
  let compute () =
    H.write (host out) (String.make (4 * n) '\000');
    S.run t [| S.words (S.launch l k "stack" ~blocks:4 [ address out; n ]) |];
    S.reset l;
    equal (list int) (List.init n stacked) (words32 (host out) n)
  in
  compute ();
  let { A.Gpu.local; _ } = S.gpu t.g in
  equal (result unit string) ~msg:"a smaller one" (Ok ()) (local 1024);
  compute ();
  S.free_launches l;
  N.free t.g out

(* A kernel holds its local memory while the device grows it and schedules a
   kernel on the new one: both compute right. *)
let growth () =
  S.with_ @@ fun t ->
  let k = S.kernels t in
  let l = S.launches t.g in
  let n = 1024 in
  let flag = alloc t.g Pinned 8 in
  let first = alloc t.g Pinned (4 * n) in
  let next = alloc t.g Pinned (4 * n) in
  H.set64 (host flag) 0;
  releasing flag 1 (fun () ->
      let held =
        S.launch l k "stack_held" ~blocks:4
          [ address flag; 10 * second; address first; n ]
      in
      ignore (S.submit t [| S.words held |]);
      let { A.Gpu.local; _ } = S.gpu t.g in
      equal (result unit string) ~msg:"grown" (Ok ()) (local 4096);
      let v =
        S.submit t
          [| S.words (S.launch l k "stack" ~blocks:4 [ address next; n ]) |]
      in
      still ~msg:"the word while held" int (v - 2)
        (fun () -> N.signaled t.g)
        ~ms:20;
      H.set64 (host flag) 1;
      Rig.wait t.d v);
  let want = List.init n stacked in
  equal (list int) ~msg:"the held kernel" want
    (words32 (host first) n);
  equal (list int) ~msg:"the next kernel" want
    (words32 (host next) n);
  S.free_launches l;
  List.iter (N.free t.g) [ flag; first; next ]

let local =
  group ~timeout:60. "local memory"
    [
      test "local memory serves kernels that keep 2 KiB a thread" local;
      test "a growth leaves a running kernel's local memory intact" growth;
    ]

(* Images *)

let images () =
  S.with_driver @@ fun g ->
  let bin = S.fixture "kernels_sm89.cubin" in
  let size = A.Cubin.size (require_ok (A.Cubin.of_string bin)) in
  let n, lay =
    match require_ok (N.image g bin) with
    | Place (n, lay) -> (n, lay)
    | Loaded _ -> fail "an image with nothing to place"
  in
  equal int ~msg:"the region's bytes" size n;
  let r = alloc g Device n in
  let i, bytes = lay r in
  equal int ~msg:"the image's bytes" size (String.length bytes);
  equal bool ~msg:"double_index" true
    (Option.is_some (N.entry i "double_index"));
  equal (option int) ~msg:"a missing kernel" None
    (Option.map (fun (e : Rig_edge.entry) -> e.code) (N.entry i "missing"));
  is_error ~msg:"not a cubin" (N.image g "not a cubin");
  N.unload g i;
  N.free g r

(* A cubin loads where another one's code ran and was unloaded: its launch runs
   its own code. The second cubin is laid over the first one's region, so that
   its code is where the first one's ran whatever the allocator answers. Each
   image goes to its region by a kernel that copies it from Mapped memory. *)
let reloaded () =
  S.with_ @@ fun t ->
  let k = S.kernels t in
  let l = S.launches t.g in
  let n = 1000 in
  let out = alloc t.g Pinned (4 * n) in
  let copy_in code bytes =
    let staging = alloc t.g Mapped (String.length bytes) in
    H.write (host staging) bytes;
    S.run t
      [|
        S.words
          (copy_after l k ~dst:(address code) ~src:(address staging)
             (String.length bytes));
      |];
    N.free t.g staging
  in
  let compute bin code factor =
    H.write (host out) (String.make (4 * n) '\000');
    S.run t
      [|
        S.words
          (S.launch_at l ~code:(address code) bin "index" ~blocks:4
             [ address out; n ]);
      |];
    S.reset l;
    equal (list int) ~msg:(strf "%d i" factor)
      (List.init n (fun i -> factor * i))
      (words32 (host out) n)
  in
  let twice = S.fixture "twice_sm89.cubin" in
  let i, code, bytes = S.image t.g twice in
  copy_in code bytes;
  compute twice code 2;
  N.unload t.g i;
  let thrice = S.fixture "thrice_sm89.cubin" in
  let i', bytes' =
    match N.image t.g thrice with
    | Ok (Place (size, lay)) ->
        equal int ~msg:"the second's image is the first's size"
          (String.length bytes) size;
        lay code
    | Ok (Loaded _) -> fail "an image with nothing to place"
    | Error e -> failf "loading: %s" e
  in
  copy_in code bytes';
  compute thrice code 3;
  N.unload t.g i';
  N.free t.g code;
  S.free_launches l;
  N.free t.g out

let images =
  group ~timeout:60. "images"
    [
      test "load, find their kernels and unload" images;
      test "code loaded where other code ran runs as loaded" reloaded;
    ]

(* Timeline and loss *)

let stop_idle () =
  S.with_ @@ fun t ->
  let r = alloc t.g Device 64 in
  let a = H.pages H.page in
  let m = require_some (N.map_host t.g a 64) in
  S.run t [||];
  S.close t;
  equal int ~msg:"the word" (Rig.submitted t.d) (Rig.signaled t.d);
  N.free t.g r;
  N.free t.g m;
  H.free_pages a H.page;
  let { A.Gpu.local; _ } = S.gpu t.g in
  is_error ~msg:"local after stop" (local 1024)

(* COMPUTE:0 waits on a host word nobody writes, a release queued behind the
   wait: stop returns with the word at the last value, and the waiting work
   never runs. *)
let stop_waiting () =
  let g = S.driver () in
  let t = { S.d = handed g; g } in
  let l = S.launches t.g in
  let w = alloc t.g Pinned 8 in
  let marked = alloc t.g Pinned 8 in
  H.set64 (host w) 0;
  H.set64 (host marked) 0;
  let ws =
    Array.append (acquire l (address w) 1) (release l (address marked) 7)
  in
  let v = S.submit t [| S.words ws |] in
  still ~msg:"the word" int (v - 1) (fun () -> N.signaled t.g) ~ms:20;
  S.watchdog "stop" (fun () -> S.stop_driver t.g);
  equal int ~msg:"the word" v (N.signaled t.g);
  H.set64 (host w) 1;
  still ~msg:"the waiting work" int 0
    (fun () -> H.get64 (host marked))
    ~ms:200;
  S.free_launches l;
  List.iter (N.free t.g) [ w; marked ];
  S.with_ (fun t' -> S.run t' [||])

(* A kernel runs with a release queued behind it: stop returns with the word at
   the last value, and the queued work never runs. *)
let stop_running () =
  let g = S.driver () in
  let t = { S.d = handed g; g } in
  let k = S.kernels t in
  let l = S.launches t.g in
  let flag = alloc t.g Pinned 8 in
  let marked = alloc t.g Pinned 8 in
  H.set64 (host flag) 0;
  H.set64 (host marked) 0;
  releasing flag 1 (fun () ->
      let spin = S.launch l k "spin" ~blocks:1 [ address flag; 10 * second ] in
      let v1 = S.submit t [| S.words spin |] in
      let v2 = S.submit t [| S.words (release l (address marked) 7) |] in
      still ~msg:"the word" int (v1 - 1) (fun () -> N.signaled t.g) ~ms:20;
      S.watchdog "stop" (fun () -> S.stop_driver t.g);
      equal int ~msg:"the word" v2 (N.signaled t.g));
  still ~msg:"the queued work" int 0 (fun () -> H.get64 (host marked)) ~ms:200;
  S.free_launches l;
  List.iter (N.free t.g) [ flag; marked ]

let timeline =
  group ~timeout:60. "timeline"
    [
      test "a close of an idle device leaves the word at the last value"
        stop_idle;
      test
        "stop of a device waiting on a word that never moves leaves the word \
         at the last value (sampled)"
        stop_waiting;
      test
        "stop of a running device leaves the word at the last value, and its \
         queued work never runs (sampled)"
        stop_running;
    ]

(* The shared device: the stateful tests' programs use one device, opened by the
   first and closed when the run ends. *)

let shared = fixture ~teardown:S.close S.open_

(* map_host: the registry. The path maps whole pages: a range inside the pages
   of a mapped one shares its mapping, a range that shares some of their pages
   is refused. *)

module Registry = struct
  type entry = { lo : int; hi : int; mutable maps : int } (* pages [lo, hi) *)
  type t = { mutable entries : entry list; mutable regions : entry list }

  let arena = 4
  let pages (a, n) = (a / H.page, ((a + n - 1) / H.page) + 1)
  let make () = { entries = []; regions = [] }

  let map m (a, n) =
    let lo, hi = pages (a, n) in
    let inside e = e.lo <= lo && hi <= e.hi in
    match List.find_opt inside m.entries with
    | Some e ->
        cover "a range of a mapped one's pages, elsewhere"
          (lo <> e.lo || hi <> e.hi);
        cover "a range of the same pages as a mapped one"
          (lo = e.lo && hi = e.hi);
        cover "a range under one page that another range maps" (hi - lo = 1);
        e.maps <- e.maps + 1;
        m.regions <- m.regions @ [ e ];
        true
    | None when List.exists (fun e -> lo < e.hi && e.lo < hi) m.entries ->
        cover "a range that shares only some of a mapped one's pages" true;
        false
    | None ->
        let e = { lo; hi; maps = 1 } in
        m.entries <- e :: m.entries;
        m.regions <- m.regions @ [ e ];
        true

  let free m i =
    let e = List.nth m.regions i in
    m.regions <- List.filteri (fun j _ -> j <> i) m.regions;
    e.maps <- e.maps - 1;
    if e.maps = 0 then begin
      cover "the last free of a range" true;
      m.entries <- List.filter (fun e' -> e' != e) m.entries
    end

  (* The system *)

  type sys = {
    t : S.t;
    base : int;
    scratch : N.region;
    kernels : S.kernels;
    launches : S.launches;
    lock : Mutex.t;
    mutable live : (N.region * int * int) list;
  }

  let start () =
    let t = shared () in
    {
      t;
      base = H.pages (arena * H.page);
      scratch = Option.get (N.alloc t.g Pinned (arena * H.page));
      kernels = S.kernels t;
      launches = S.launches t.g;
      lock = Mutex.create ();
      live = [];
    }

  let release s =
    List.iter (fun (r, _, _) -> N.free s.t.g r) s.live;
    N.free s.t.g s.scratch;
    S.free_launches s.launches;
    H.free_pages s.base (arena * H.page)

  let map_sys s (a, n) =
    match N.map_host s.t.g (s.base + a) n with
    | Some r ->
        Mutex.protect s.lock (fun () -> s.live <- s.live @ [ (r, a, n) ]);
        true
    | None -> false

  let free_sys s i =
    let r, _, _ = List.nth s.live i in
    N.free s.t.g r;
    Mutex.protect s.lock (fun () ->
        s.live <- List.filteri (fun j _ -> j <> i) s.live)

  (* Each region shares its bytes both ways: a kernel reads what the host wrote,
     and the host reads what a kernel wrote. *)
  let invariant _ s =
    let across ~dst ~src n =
      S.run s.t [| S.words (copy_after s.launches s.kernels ~dst ~src n) |];
      S.reset s.launches
    in
    List.iteri
      (fun i (r, a, n) ->
        let at = s.base + a in
        S.pattern at n (2 * i);
        across ~dst:(address s.scratch) ~src:(address r) n;
        equal int
          ~msg:(strf "the GPU reads [%d, %d)" a (a + n))
          (-1)
          (S.mismatch (host s.scratch) n (2 * i));
        S.pattern (host s.scratch) n ((2 * i) + 1);
        across ~dst:(address r) ~src:(address s.scratch) n;
        equal int
          ~msg:(strf "the host reads [%d, %d)" a (a + n))
          (-1)
          (S.mismatch at n ((2 * i) + 1)))
      s.live

  let range =
    let point = [ 0; 8; 2048; 4088; 4096; 4104; 8192; 12280 ] in
    let length = [ 8; 64; 2048; 4096; 4104; 8192 ] in
    let scale x = x * H.page / 4096 in
    Gen.with_pp
      (fun ppf (a, n) -> Format.fprintf ppf "(%d, %d)" a n)
      (Gen.frequency
         [
           ( 8,
             Gen.such_that
               (fun (a, n) -> a + n <= arena * H.page)
               (Gen.map
                  (fun (a, n) -> (scale a, scale n))
                  (Gen.pair (Gen.of_list point) (Gen.of_list length))) );
           (* One range drawn often, so that ranges of the same pages are drawn
              too. *)
           (2, Gen.constant (0, H.page));
         ])
end

let registry =
  abstract "r" ~release:Registry.release ~invariant:Registry.invariant

let mapped_regions =
  among int registry (fun m ->
      List.init (List.length m.Registry.regions) Fun.id)

let registry_commands =
  [
    command "start" (Gen.unit @-> makes registry) Registry.make Registry.start;
    command "map_host"
      (registry ^-> Registry.range @-> returns bool)
      Registry.map Registry.map_sys;
    command "free"
      (registry ^-> mapped_regions ^-> returns unit)
      Registry.free Registry.free_sys;
  ]

(* Local memory handed over from two domains: one grows the kernels' local
   memory while the other submits through rig, and the word moves on by small
   steps. A local memory the device replaced goes back to the path only once
   the word reached every value that could run on it: those rig assigned
   before its successor's [local] returned. Each program opens a fake device
   of its own: a run of 30 cases opens at most 15,000, and the opens grow
   with --prop-count, so 150 cases or more can reach the 65,535 devices a
   process may open. *)
module Handover = struct
  type t = {
    g : N.t;
    d : Rig.t;
    f : Fake.t;
    grows : Mutex.t; (* one growth and its record at a time *)
    lock : Mutex.t; (* the word's steps and the memories' last users *)
    mutable newest : int option; (* the newest local memory *)
    users : (int, int) Hashtbl.t; (* a replaced memory's last possible user *)
    mutable early : string option; (* a memory freed before its users ran *)
    mutable returned : int; (* replaced memories gone back *)
    mutable kib : int; (* the local memory a thread has, in KiB *)
  }

  let word h = host (S.word h.g)
  let names = Atomic.make 0

  let start () =
    let f = Fake.make 0 in
    let g = require_ok (N.make (Fake.path f)) in
    let name = strf "NV:handover-%d" (Atomic.fetch_and_add names 1) in
    let d = require_ok (Rig.open_ (module N) ~name (fun () -> Ok g)) in
    let h =
      {
        g;
        d;
        f;
        grows = Mutex.create ();
        lock = Mutex.create ();
        newest = None;
        users = Hashtbl.create 8;
        early = None;
        returned = 0;
        kib = 0;
      }
    in
    let held = List.map (fun (a, _, _) -> a) f.memory in
    f.on_free <-
      (fun m ->
        Mutex.protect h.lock @@ fun () ->
        if not (List.mem m.address held) then
          match Hashtbl.find_opt h.users m.address with
          | Some v when H.get64 (word h) < v ->
              h.early <-
                Some
                  (strf "0x%x freed at word %d, used up to %d" m.address
                     (H.get64 (word h))
                     v)
          | Some _ -> h.returned <- h.returned + 1
          | None -> ());
    h

  (* Hands one empty submission, which takes a pending local memory. *)
  let hand h = ignore (submit h.d [||])

  (* Grows local memory by 1 KiB a thread, which replaces it, then hands a
     submission that takes the new one. A value rig assigned before [local]
     returned is at most [Rig.submitted] after it: no OCaml code runs between
     a value's assignment and its hand-over. *)
  let grow h =
    Fun.protect ~finally:(fun () -> hand h) @@ fun () ->
    Mutex.protect h.grows @@ fun () ->
    h.kib <- h.kib + 1;
    match (S.gpu h.g).local (h.kib * 1024) with
    | Error e -> failf "local: %s" e
    | Ok () -> (
        let last = Rig.submitted h.d in
        let newest =
          List.find_map
            (fun (a, host, _) -> if host = None then Some a else None)
            h.f.memory
        in
        Mutex.protect h.lock @@ fun () ->
        match (h.newest, newest) with
        | Some old, Some m when old <> m ->
            Hashtbl.replace h.users old last;
            h.newest <- Some m
        | None, m -> h.newest <- m
        | Some _, _ -> ())

  (* Hands one submission, then moves the word on by [k] values, up to the
     last handed. *)
  let submit k h =
    hand h;
    Mutex.protect h.lock @@ fun () ->
    let w = H.get64 (word h) in
    H.set64 (word h) (Int.min (Rig.submitted h.d) (w + k))

  (* At the program's end, branches included: a replaced memory went back
     while the program ran. Then the word reaches every value handed, which
     nothing else completes, so the close does not wait for ever, and the
     close's frees are judged too. *)
  let release h =
    cover "a replaced local memory went back"
      (Mutex.protect h.lock (fun () -> h.returned > 0));
    Mutex.protect h.lock (fun () -> H.set64 (word h) (Rig.submitted h.d));
    Rig.close h.d;
    equal (option string) ~msg:"a local memory freed early" None h.early
end

let handover = abstract "h" ~release:Handover.release

let handover_commands =
  [
    command "start" (Gen.unit @-> makes handover) ignore Handover.start;
    command "grow" (handover ^-> returns unit) ignore Handover.grow;
    command "submit"
      (Gen.int_range 0 2 @-> handover ^-> returns unit)
      (fun _ () -> ())
      Handover.submit;
  ]

let stateful =
  group ~timeout:300. "stateful"
    [
      stateful ~count:100 ~steps:20 "map_host shares a host range both ways"
        registry_commands;
      stateful ~count:30 ~steps:10 ~domains:2
        "local memory goes back only once the values that could use it ran"
        handover_commands;
    ]

let () =
  S.hold ();
  exit
    (run "rig_nv"
       [
         paths;
         facts;
         memory;
         work;
         launches;
         room;
         local;
         images;
         timeline;
         stateful;
       ])
