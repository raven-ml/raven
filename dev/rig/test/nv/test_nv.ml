(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The driver's work runs as programs hand it over, through Rig, except for what
   only the C edge reaches: waits on any host word, the 64-bit wrap of a wait's
   value, and room's answers. *)

open Windtrap
module N = Rig_nv
module A = Rig_nv_abi
module C = Rig
module B = Rig.Buffer
module S = Rig_nv_support

let strf = Printf.sprintf
let second = 1_000_000_000
let host = S.host
let address = S.address
let alloc g kind n = require_some (N.alloc g kind n)

(* [f ()], the host word [w] set to [x] after it whether it returns or raises,
   so that work held on [w] ends. *)
let releasing w x f = Fun.protect ~finally:(fun () -> S.set64 (host w) x) f

(* The entry of a segment that writes [x] at [a] once the channel's earlier work
   completed, and of one that waits until the word at [a] is at least [x]. *)
let release l a x = S.segment l (A.Method.release System a x)
let acquire l a x = S.segment l (A.Method.acquire a x)

(* A kernel that copies [n] bytes from the address [src] to [dst], starting [ns]
   nanoseconds late. *)
let copy_after l k ?(ns = 0) ~dst ~src n =
  S.launch l k "copy_after" ~blocks:1 [ ns; dst; src; n ]

(* Parts at the C edge *)

let on_compute ?(after = [||]) ws = { S.queue = 0; after; work = `Words ws }

let on_copy ?(after = [||]) ~dst ~src n =
  { S.queue = 1; after; work = `Copy (dst, src, n) }

(* Hands [ps] to [g] at the C edge as the value after [v], once room fits: while
   room is Later, it waits for [v]. *)
let hand g v ?(waits = [||]) ps =
  let rec fits () =
    match S.edge_room g ps with
    | 0 -> ()
    | 1 ->
        S.wait g !v;
        fits ()
    | r -> failf "room answered %d" r
  in
  fits ();
  incr v;
  S.edge_submit g ~v:!v ~waits ps

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
    mutable hang_ms : int option;
    mutable stops : [ `Stopped | `Unknown ]; (* what the path's stop answers *)
    mutable on_free : unit N.memory -> unit; (* runs before each free *)
    mutable bar_room : bool; (* whether [`Bar] memory is given *)
    mutable kinds : string list; (* the kinds of memory given, newest first *)
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
      let bytes = (n + S.page - 1) / S.page * S.page in
      let host = if kind = `Gpu then None else Some (S.pages bytes) in
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
        Option.iter (fun a -> S.free_pages a bytes) host

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
      doorbell = S.pages S.page;
      alloc = alloc f;
      map_host = Some (fun _ n -> alloc f `System n);
      reaches = (fun _ -> false);
      map_peer = (fun _ -> alloc f `Gpu S.page);
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
      hang_ms = f.hang_ms;
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
      hang_ms = None;
      stops = `Unknown;
      on_free = ignore;
      bar_room = true;
      kinds = [];
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
        N.stop g;
        equal (list (pair int int)) ~msg:(at ^ ": RM objects") [] f.objects;
        equal (list int) ~msg:(at ^ ": memory")
          [ address (N.word g) ]
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
    | `Place (n, lay) -> (n, lay)
    | `Loaded _ -> fail "an image with nothing to place"
  in
  let r = require_some (N.alloc g `Device n) in
  let calls = f.calls and memory = List.length f.memory in
  let i, _ = lay r in
  equal int ~msg:"the path's calls" calls f.calls;
  equal int ~msg:"the path's memory" memory (List.length f.memory);
  N.unload g i;
  N.free g r;
  N.stop g

(* A device stopped with an image loaded: stop ends the image, and the code
   region is the caller's to free, after which the device keeps only its
   word. *)
let stop_with_image () =
  let f = Fake.make 0 in
  let g = require_ok (N.make (Fake.path f)) in
  let n, lay =
    match require_ok (N.image g (S.fixture "kernels_sm89.cubin")) with
    | `Place (n, lay) -> (n, lay)
    | `Loaded _ -> fail "an image with nothing to place"
  in
  let r = require_some (N.alloc g `Device n) in
  ignore (lay r);
  N.stop g;
  N.free g r;
  equal (list int) ~msg:"memory"
    [ address (N.word g) ]
    (List.map (fun (a, _, _) -> a) f.memory);
  equal (list string) ~msg:"given back wrongly" [] f.wrong

(* The memory a path answers lies below 2^40, the widest address a channel's
   semaphore takes, up to its last byte; memory past it goes back to the path,
   and the call raises. *)
let address_limit () =
  let f = Fake.make 0 and f' = Fake.make 0 in
  let g = require_ok (N.make (Fake.path f)) in
  let g' = require_ok (N.make (Fake.path f')) in
  let limit = 1 lsl 40 and n = 2 * S.page in
  let r' = require_some (N.alloc g' `Device n) in
  let held = List.length f.memory in
  let calls =
    [
      ("`Device", "Rig_nv.alloc", fun () -> N.alloc g `Device n);
      ("`Pinned", "Rig_nv.alloc", fun () -> N.alloc g `Pinned n);
      ("`Mapped", "Rig_nv.alloc", fun () -> N.alloc g `Mapped n);
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
    f.at <- Some (limit - S.page);
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
  N.stop g;
  N.stop g'

(* A device whose channels the RM stopped on a fault, and kept: nothing of it
   runs, so stop raises the word, though its memory stays the path's. *)
let faulted_stop () =
  let f = Fake.make 0 in
  let g = require_ok (N.make (Fake.path f)) in
  hand g (ref 0) [||];
  let word = address (N.word g) in
  let fault (a, host, bytes) =
    match host with
    | Some h when bytes = S.page && a <> word ->
        for i = 0 to (S.page / 8) - 1 do
          S.set64 (h + (8 * i)) (-1)
        done
    | Some _ | None -> ()
  in
  List.iter fault f.memory;
  f.frees <- false;
  N.stop g;
  equal int ~msg:"the word" 1 (N.signaled g);
  greater int ~msg:"memories the path still gives" ~than:1
    (List.length f.memory)

(* Mapped memory is the BAR's while it has room, then host memory. *)
let mapped_fallback () =
  let f = Fake.make 0 in
  let g = require_ok (N.make (Fake.path f)) in
  let r = require_some ~msg:"with room" (N.alloc g `Mapped 64) in
  equal (option string) ~msg:"with room" (Some "`Bar") (List.nth_opt f.kinds 0);
  f.bar_room <- false;
  let r' = require_some ~msg:"without room" (N.alloc g `Mapped 64) in
  equal (option string) ~msg:"without room" (Some "`System")
    (List.nth_opt f.kinds 0);
  equal bool ~msg:"host addresses it" true (Option.is_some (N.host r'));
  List.iter (N.free g) [ r; r' ];
  N.stop g

(* Compiled code links local memory through the capability, whose [local]
   answers a failure as [Error]. *)
let failing_local () =
  let f = Fake.make 0 in
  let g = require_ok (N.make (Fake.path f)) in
  f.faults <- true;
  ignore (require_error ((N.capability g).local 1024) : string);
  f.faults <- false;
  N.stop g

(* A path whose frees raise Fault, as one whose GPU is lost: free and stop still
   return, as rig calls them after a loss. *)
let failing_frees () =
  let f = Fake.make 0 in
  let g = require_ok (N.make (Fake.path f)) in
  let r = require_some (N.alloc g `Pinned 64) in
  f.faults <- true;
  N.free g r;
  raises_match ~msg:"a second free" (Exn.invalid_arg ~substring:"was freed")
    (fun () -> N.free g r);
  N.stop g

(* A device whose channels the RM keeps at stop: the path's [`Stopped] says none
   of its work runs, so the word reaches the last value and the memory goes
   back; [`Unknown] keeps both, as the work may still run. *)
let path_stops () =
  List.iter
    (fun answer ->
      let f = Fake.make 0 in
      let g = require_ok (N.make (Fake.path f)) in
      hand g (ref 0) [||];
      f.frees <- false;
      f.stops <- answer;
      N.stop g;
      let stopped = answer = `Stopped in
      let what = if stopped then "`Stopped" else "`Unknown" in
      equal int ~msg:(what ^ ": the word")
        (if stopped then 1 else 0)
        (N.signaled g);
      equal bool
        ~msg:(what ^ ": only the word's memory left")
        stopped
        (List.map (fun (a, _, _) -> a) f.memory = [ address (N.word g) ]))
    [ `Stopped; `Unknown ]

(* A fault the path reports surfaces from sleep while a value is outstanding. *)
let path_check () =
  let f = Fake.make 0 in
  let g = require_ok (N.make (Fake.path f)) in
  hand g (ref 0) [||];
  f.report <- Some "the GPU fell off the bus";
  (match N.sleep g ~seen:0 ~still_ms:1 with
  | () -> fail "no Fault for the path's report"
  | exception N.Fault why ->
      contains ~msg:"the report" ~sub:"the GPU fell off the bus" why);
  f.report <- None;
  N.stop g

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
      test "Mapped memory falls back to host memory once the BAR is full"
        mapped_fallback;
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
  let arch = N.arch g in
  starts_with ~affix:"sm_" arch;
  equal bool
    ~msg:(strf "%s ends in digits" arch)
    true
    (String.length arch > 3
    && digits (String.sub arch 3 (String.length arch - 3)));
  greater int ~msg:"budget" ~than:0 (N.budget g);
  equal (list string) [ "COMPUTE:0"; "COPY:0" ] (N.queues g);
  equal bool ~msg:"completion is the store" true (N.completion g = `Store);
  equal (list bool) [ true; false; true ]
    (List.map (N.waits_on g) [ `Store; `Object; `Host ]);
  equal bool ~msg:"submit returns" true (N.blocks g = `Returns);
  let w = N.word g in
  equal nativeint ~msg:"the word's handle is its address"
    (Nativeint.of_int (address w))
    (N.handle w);
  equal int ~msg:"the word starts at 0" 0 (S.get64 (host w));
  equal int ~msg:"signaled reads the word" 0 (N.signaled g)

let capability () =
  S.with_driver @@ fun g ->
  let c = N.capability g in
  equal bool ~msg:"the key is the ABI's" true
    (Option.is_some (Type.Id.provably_equal N.capability_key A.Gpu.key));
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
        (kind <> `Device)
        (Option.is_some (N.host r));
      less int ~msg:(name ^ ": its address") ~than:(1 lsl 40) (address r);
      N.free g r)
    [ ("Device", `Device); ("Pinned", `Pinned); ("Mapped", `Mapped) ]

let facts =
  group ~timeout:60. "facts"
    [
      test "states a GPU's facts" facts;
      test "declares the GPU to compiled code" capability;
      test "gives memory of each kind where the host and the GPU reach it"
        kinds_of_memory;
    ]

(* Memory *)

let kinds = [ B.Device; B.Pinned; B.Mapped ]

let pp_kind ppf k =
  Format.pp_print_string ppf
    (match k with
    | B.Device -> "Device"
    | Pinned -> "Pinned"
    | Mapped -> "Mapped")

let kind = Gen.of_list ~pp:pp_kind kinds

(* Sizes up to past 8 MiB, where GPU memory takes 2 MiB pages. *)
let size =
  Gen.of_list ~pp:Format.pp_print_int
    [ 1; 7; 4096; (2 lsl 20) + 7; (8 lsl 20) + 3 ]

let offset = Gen.of_list ~pp:Format.pp_print_int [ 0; 1; 4095 ]

(* host -> a -> b -> host through Copy parts. The host's last buffer holds
   another pattern before, so that a copy that did not run shows. *)
let round_trip (ka, kb, n, (oa, ob)) =
  S.with_gpu @@ fun t ->
  let src, sa = S.shared t n in
  let dst, da = S.shared t n in
  let a = B.view (B.create ~memory:ka t.d (n + oa)) ~first:oa ~length:n in
  let b = B.view (B.create ~memory:kb t.d (n + ob)) ~first:ob ~length:n in
  less int ~msg:"address of a" ~than:(1 lsl 40) (B.address a + n);
  let seed = n + oa + ob in
  S.pattern sa n seed;
  S.pattern da n (seed + 1);
  S.run t
    [|
      S.copy ~dst:a src;
      S.copy ~after:[| 0 |] ~dst:b a;
      S.copy ~after:[| 1 |] ~dst b;
    |];
  equal int ~msg:"the first byte that differs" (-1) (S.mismatch da n seed)

(* A copy of more bytes than the copy engine moves at once, out to GPU memory at
   an odd offset and back. *)
let long_copy () =
  S.with_gpu @@ fun t ->
  let n = A.Method.max_copy + 4099 in
  let h, ha = S.shared t n in
  let d = B.view (B.create t.d (n + 1)) ~first:1 ~length:n in
  S.pattern ha n 3;
  S.run t [| S.copy ~dst:d h |];
  S.pattern ha n 4;
  S.run t [| S.copy ~dst:h d |];
  equal int ~msg:"the first byte that differs" (-1) (S.mismatch ha n 3)

(* The host and the GPU see each other's stores to Mapped memory across one
   submission, round after round: a kernel copies between it and Pinned
   memory. *)
let mapped () =
  S.with_gpu @@ fun t ->
  let k = S.kernels t in
  let l = S.launches t.g in
  let n = 4096 in
  let m = alloc t.g `Mapped n in
  let p = alloc t.g `Pinned n in
  let across ~dst ~src = S.run t [| S.words (copy_after l k ~dst ~src n) |] in
  for round = 1 to 100 do
    S.pattern (host m) n (2 * round);
    across ~dst:(address p) ~src:(address m);
    equal int
      ~msg:(strf "round %d: the GPU reads" round)
      (-1)
      (S.mismatch (host p) n (2 * round));
    S.pattern (host p) n ((2 * round) + 1);
    across ~dst:(address m) ~src:(address p);
    equal int
      ~msg:(strf "round %d: the host reads" round)
      (-1)
      (S.mismatch (host m) n ((2 * round) + 1));
    S.reset l
  done;
  S.free_launches l;
  List.iter (N.free t.g) [ m; p ]

let refusal () =
  S.with_driver @@ fun g ->
  for _ = 1 to 100 do
    is_none ~msg:"twice the budget" (N.alloc g `Device (2 * N.budget g))
  done;
  N.free g (require_some ~msg:"64 MiB after" (N.alloc g `Device (64 lsl 20)))

(* Host memory no page backs is refused: map_host answers None. *)
let refused_host () =
  S.with_driver @@ fun g ->
  let p = S.pages S.page in
  S.free_pages p S.page;
  equal bool ~msg:"an unmapped page" true (Option.is_none (N.map_host g p 8));
  equal bool ~msg:"address 0" true (Option.is_none (N.map_host g 0 8))

let memory =
  group ~timeout:120. "memory"
    [
      prop ~count:30 "copies through any two kinds of memory are the identity"
        (Gen.quad kind kind size (Gen.pair offset offset))
        round_trip;
      test "a copy longer than the copy engine's is the identity" long_copy;
      test "host and GPU stores to Mapped memory reach each other" mapped;
      test
        "an allocation past the GPU's memory is None, and gives back what it \
         took"
        refusal;
      test "map_host answers None for host memory no page backs" refused_host;
    ]

(* Work *)

let launch () =
  S.with_gpu @@ fun t ->
  let k = S.kernels t in
  let l = S.launches t.g in
  let n = 1000 in
  let out = alloc t.g `Pinned (4 * n) in
  S.run t
    [| S.words (S.launch l k "double_index" ~blocks:4 [ address out; n ]) |];
  equal (list int)
    (List.init n (fun i -> 2 * i))
    (List.init n (S.get32 (host out)));
  S.free_launches l;
  N.free t.g out

(* A part runs after the parts of the other channel its [after] names: a kernel
   reads what a copy wrote, and a copy reads what a kernel started 100 us late
   wrote. *)
let joins () =
  S.with_gpu @@ fun t ->
  let k = S.kernels t in
  let l = S.launches t.g in
  let n = 4096 in
  let src, sa = S.shared t n in
  let mid = B.create t.d n in
  let dst, da = S.shared t n in
  S.pattern sa n 1;
  S.pattern da n 0;
  S.run t
    [|
      S.copy ~dst:mid src;
      S.words ~after:[| 0 |]
        (copy_after l k ~dst:(B.address dst) ~src:(B.address mid) n);
    |];
  equal int ~msg:"the kernel read the copy" (-1) (S.mismatch da n 1);
  S.pattern sa n 2;
  S.run t
    [|
      S.words
        (copy_after l k ~ns:100_000 ~dst:(B.address mid) ~src:(B.address src) n);
      S.copy ~after:[| 0 |] ~dst mid;
    |];
  equal int ~msg:"the copy read the kernel's" (-1) (S.mismatch da n 2);
  S.free_launches l

(* Two kernels' parts on COMPUTE:0: the second copies what the first, started
   100 us late, wrote. *)
let compute_order () =
  S.with_gpu @@ fun t ->
  let k = S.kernels t in
  let l = S.launches t.g in
  let n = 4096 in
  let src, sa = S.shared t n in
  let mid = B.create t.d n in
  let dst, da = S.shared t n in
  S.pattern sa n 1;
  S.pattern da n 0;
  let late ?ns d s =
    S.words (copy_after l k ?ns ~dst:(B.address d) ~src:(B.address s) n)
  in
  S.run t [| late ~ns:100_000 mid src; late dst mid |];
  equal int ~msg:"the second read the first's" (-1) (S.mismatch da n 1);
  S.free_launches l

let misuse () =
  S.with_driver @@ fun g ->
  let raises name f = raises_match ~msg:name Exn.invalid_arg f in
  let r = alloc g `Device 64 in
  raises "alloc of 0 bytes" (fun () -> N.alloc g `Device 0);
  raises "map_host of 0 bytes" (fun () -> N.map_host g (host (N.word g)) 0);
  raises "map_peer of one device" (fun () -> N.map_peer g g r);
  raises "free of the word" (fun () -> N.free g (N.word g));
  N.free g r;
  raises "free twice" (fun () -> N.free g r);
  let a = S.pages S.page in
  let m = require_some (N.map_host g a 64) in
  N.free g m;
  raises "free of a mapping twice" (fun () -> N.free g m);
  S.free_pages a S.page

(* A device's region, mapping and image, used on the device opened after it
   stopped. *)
let another_device () =
  let a = S.driver () in
  let r = alloc a `Pinned 64 in
  let p = S.pages S.page in
  let m = require_some (N.map_host a p 64) in
  let i, code, _ = S.image a (S.fixture "kernels_sm89.cubin") in
  S.stop a;
  S.with_driver @@ fun b ->
  let raises name f = raises_match ~msg:name Exn.invalid_arg f in
  raises "free of its region" (fun () -> N.free b r);
  raises "free of its mapping" (fun () -> N.free b m);
  raises "unload of its image" (fun () -> N.unload b i);
  List.iter (N.free a) [ r; m; code ];
  S.free_pages p S.page

(* The C edge *)

let fill ?(units = 0) ?(bytes = 0) ?(after = [||]) q =
  { S.queue = q; after; work = `Fill (units, bytes) }

let room_c () =
  S.with_driver @@ fun g ->
  let room ps = S.edge_room g ps in
  let empty = on_compute [||] in
  equal (list int)
    [ 0; 0; 0; 2; 2; 2; 2; 2; 2; 2; 2; 2 ]
    [
      room [| empty |];
      room [| on_compute [| 0; 0 |] |];
      room [| on_copy ~dst:0 ~src:0 1 |];
      room [| fill 0 |];
      room [| fill ~units:1 0 |];
      room [| fill ~bytes:1 0 |];
      room [| on_compute [| 0 |] |];
      room [| { (on_copy ~dst:0 ~src:0 1) with queue = 0 } |];
      room [| { empty with queue = 2 } |];
      room [| on_compute ~after:[| 0 |] [||] |];
      room [| on_compute (Array.make (2 * 16_384) 0) |];
      room (Array.make 65_536 empty);
    ]

(* Work at the C edge held on a host word on COMPUTE:0 alone, COPY:0 alone or
   both, across the wrap of the word's 64 bits: COMPUTE:0 releases a value into
   a host buffer, COPY:0 copies one. *)
let held queues g ~start ~wait ~below ~release:r =
  let l = S.launches g in
  let w = alloc g `Pinned 8 in
  let src = alloc g `Pinned 64 in
  let marked = alloc g `Pinned 8 in
  let copied = alloc g `Pinned 64 in
  S.pattern (host src) 64 1;
  S.set64 (host marked) 0;
  S.write (host copied) (String.make 64 '\000');
  let ps =
    List.map
      (function
        | `Compute -> on_compute (release l (address marked) 7)
        | `Copy -> on_copy ~dst:(address copied) ~src:(address src) 64)
      queues
  in
  let held () =
    S.still ~msg:"the word" int 0 (fun () -> N.signaled g) ~ms:20;
    equal int ~msg:"held on COMPUTE:0" 0 (S.get64 (host marked));
    equal string ~msg:"held on COPY:0" (String.make 64 '\000')
      (S.read (host copied) 64)
  in
  S.set64 (host w) start;
  releasing w r (fun () ->
      let v = ref 0 in
      hand g v ~waits:[| (address w, wait) |] (Array.of_list ps);
      held ();
      S.set64 (host w) below;
      held ();
      S.set64 (host w) r;
      S.wait g 1);
  if List.mem `Compute queues then
    equal int ~msg:"released on COMPUTE:0" 7 (S.get64 (host marked));
  if List.mem `Copy queues then
    equal int ~msg:"copied on COPY:0" (-1) (S.mismatch (host copied) 64 1);
  S.free_launches l;
  List.iter (N.free g) [ w; src; marked; copied ]

let pp_queues ppf qs =
  Format.pp_print_string ppf
    (String.concat " and "
       (List.map (function `Compute -> "COMPUTE:0" | `Copy -> "COPY:0") qs))

(* A wait on a word of host memory wherever the process placed it, such as above
   2^40, the widest address a channel's semaphore names. *)
let high_word () =
  S.with_driver @@ fun g ->
  let p = S.pages S.page in
  at_least int ~msg:"the word's host address" ~than:(1 lsl 40) p;
  S.set64 p 0;
  let w = require_some (N.map_host g p 8) in
  let b = alloc g `Pinned 16 in
  S.set64 (host b) 0;
  let l = S.launches g in
  S.watchdog "a wait on a high host word" (fun () ->
      hand g (ref 0)
        ~waits:[| (address w, 1) |]
        [| on_compute (release l (address b) 9) |];
      S.still ~msg:"the word" int 0 (fun () -> N.signaled g) ~ms:20;
      S.set64 p 1;
      S.wait g 1);
  equal int ~msg:"released" 9 (S.get64 (host b));
  S.free_launches l;
  List.iter (N.free g) [ w; b ];
  S.free_pages p S.page

let waits =
  [
    test "a Word wait holds work on a host word above 2^40" high_word;
    cases
      ~name:
        (Format.asprintf
           "a Word wait holds work on %a across the 64-bit wrap (sampled)"
           pp_queues)
      "foreign waits"
      [ [ `Compute ]; [ `Copy ]; [ `Compute; `Copy ] ]
      (fun qs ->
        S.with_driver (held qs ~start:(-3) ~wait:2 ~below:1 ~release:2));
    cases ~name:(strf "%d satisfied waits complete on both channels")
      "batches" [ 255; 256 ] (fun n ->
        S.with_driver @@ fun g ->
        let w = alloc g `Pinned 8 in
        let b = alloc g `Pinned 16 in
        S.set64 (host w) 1;
        S.set64 (host b) 0;
        let l = S.launches g in
        hand g (ref 0)
          ~waits:(Array.make n (address w, 1))
          [|
            on_copy ~dst:(address b + 8) ~src:(address b) 8;
            on_compute ~after:[| 0 |] (release l (address b) 9);
          |];
        S.wait g 1;
        equal int ~msg:"released" 9 (S.get64 (host b));
        S.free_launches l;
        List.iter (N.free g) [ w; b ]);
  ]

let work =
  group ~timeout:60. "work"
    ([
       test "a kernel scheduled from a ring entry computes" launch;
       test "a part runs after the parts of the other channel it names" joins;
       test "parts on COMPUTE:0 run in array order" compute_order;
       test "misuse raises" misuse;
       test "a region, mapping or image of another device raises" another_device;
       test "the C room refuses what the device does not run" room_c;
     ]
    @ waits)

(* Room *)

(* Work held behind a wait on a host word fills the rings: room answers Later,
   and Fits once the work is reached. *)
let later () =
  S.with_driver @@ fun g ->
  let w = alloc g `Pinned 8 in
  let v = ref 0 in
  S.set64 (host w) 0;
  releasing w 1 (fun () ->
      hand g v ~waits:[| (address w, 1) |] [||];
      let rec fill n =
        if n > 100_000 then fail "room never answered Later";
        match S.edge_room g [||] with
        | 1 -> ()
        | 0 ->
            incr v;
            S.edge_submit g ~v:!v ~waits:[||] [||];
            fill (n + 1)
        | r -> failf "room answered %d" r
      in
      fill 0);
  S.wait g !v;
  equal int ~msg:"room once reached" 0 (S.edge_room g [||]);
  N.free g w

(* 40,000 submissions, empty ones on COMPUTE:0 between one-byte copies on
   COPY:0, wrap both channels' rings and segments; every byte arrives. *)
let wraps () =
  S.with_gpu @@ fun t ->
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
  C.wait t.d (C.submitted t.d);
  equal int ~msg:"the word" (2 * n) (N.signaled t.g);
  equal int ~msg:"the first byte that differs" (-1) (S.mismatch da n 5)

(* 40,000 copies of 16 bytes on COPY:0, each waited for, pass the end of the
   copy channel's segment ring at every offset their words take. *)
let sequential () =
  S.with_gpu @@ fun t ->
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
  S.with_gpu @@ fun t ->
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
        S.set64 wa 0;
        S.set64 ga 0;
        let v = S.submit t [| by_copy; by_compute |] in
        for _ = 1 to i mod delays * delay_step do
          ignore (Sys.opaque_identity (S.get64 wa))
        done;
        S.set64 ga 1;
        Rig.wait t.d v;
        let x = S.get64 wa in
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
  S.with_gpu @@ fun t ->
  let l = S.launches t.g in
  let scratch = alloc t.g `Pinned 8 in
  let segment = release l (address scratch) 1 in
  let h, ha = S.shared t 16 in
  let a = B.create t.d 16 in
  S.pattern ha 16 7;
  S.watchdog "work across a join's carry" (fun () ->
      for _ = 1 to 66_000 do
        S.run t [| S.copy ~dst:a h; S.words ~after:[| 0 |] segment |]
      done);
  at_least int ~msg:"the values" ~than:65_536 (N.signaled t.g);
  equal int ~msg:"the compute part's release" 1 (S.get64 (host scratch));
  S.free_launches l;
  N.free t.g scratch

(* The segment ring of each channel, as the writer fills it: the bytes of the
   words it places around the parts, from the ABI's methods. A model of where
   each channel's words fall, so that the law can tell which submissions put a
   segment's end at the ring's end. *)
module Segments = struct
  let ring = 1 lsl 20
  let bytes p = 4 * A.Packet.size p
  let acquire = bytes (A.Method.acquire 0 0)

  let release q =
    if q = 0 then bytes (A.Method.release System 0 0)
    else bytes (A.Method.copy_release System 0 0)

  let copy = bytes (A.Method.copy ~dst:0 ~src:0 0)
  let idle = bytes A.Method.wait_for_idle

  let setup q (c : A.Gpu.t) =
    if q = 0 then
      bytes
        (A.Method.set_object Compute c.compute_class
        @ A.Method.local_memory_window c.local_window
        @ A.Method.shared_memory_window c.shared_window)
    else bytes (A.Method.set_object Copy 0)

  type channel = {
    mutable written : int;
    mutable released : int;
    mutable setup : bool;
    mutable open_ : bool; (* the segment holds words *)
    mutable at_end : bool; (* the last word ended at the ring's end *)
  }

  type t = { cap : A.Gpu.t; ch : channel array }

  let make cap =
    let ch () =
      { written = 0; released = 0; setup = true; open_ = false; at_end = false }
    in
    { cap; ch = [| ch (); ch () |] }

  let close c =
    cover "a submission's words that end at the ring's end" c.at_end;
    c.open_ <- false;
    c.at_end <- false

  let emit c n =
    cover "words after an open segment that ends at the ring's end"
      (c.at_end && c.open_);
    let at = c.written mod ring in
    cover "words that do not fit before the ring's end" (at + n > ring);
    if at + n > ring then begin
      c.open_ <- false;
      c.written <- c.written + ring - at
    end;
    c.written <- c.written + n;
    c.open_ <- true;
    c.at_end <- c.written mod ring = 0

  (* The words of one submission of value [v]: per channel, its setup once, the
     wait for [v - 1] unless it released it, the waits, the joins, a wait for
     idle before a COMPUTE:0 part after unawaited ones, the copies, then the
     release on the channel of the last part. *)
  let submit m v ~waits (ps : S.part array) =
    let n = Array.length ps in
    let r = if n > 0 then ps.(n - 1).queue else 0 in
    let used = [| false; false |] in
    let last = [| -1; -1 |] in
    Array.iteri (fun i (p : S.part) -> last.(p.queue) <- i) ps;
    let enter q =
      if not used.(q) then begin
        used.(q) <- true;
        let c = m.ch.(q) in
        if c.setup then begin
          emit c (setup q m.cap);
          c.setup <- false
        end;
        if c.released <> v - 1 then emit c acquire;
        for _ = 1 to waits do
          emit c acquire
        done
      end
    in
    let awaited i =
      let rec go k =
        k < n
        && ((ps.(k).queue <> ps.(i).queue && Array.mem i ps.(k).after)
           || go (k + 1))
      in
      go (i + 1)
    in
    let running = ref false in
    Array.iteri
      (fun i (p : S.part) ->
        let q = p.queue and c = m.ch.(p.queue) in
        enter q;
        Array.iter (fun a -> if ps.(a).queue <> q then emit c acquire) p.after;
        if q = 0 && !running then emit c idle;
        if q = 0 then running := true;
        (match p.work with
        | `Words _ -> close c
        | `Copy (_, _, bytes) ->
            for
              _ = 1
              to max 1 ((bytes + A.Method.max_copy - 1) / A.Method.max_copy)
            do
              emit c copy
            done
        | `Fill _ -> ());
        if awaited i || (i = last.(q) && q <> r) then begin
          emit c (release q);
          if q = 0 then running := false
        end)
      ps;
    enter r;
    if used.(1 - r) then emit m.ch.(r) acquire;
    emit m.ch.(r) (release r);
    m.ch.(r).released <- v;
    Array.iteri (fun q u -> if u then close m.ch.(q)) used
end

(* A submission of the law: its number of satisfied waits and its parts. *)
type step = { waits : int; parts : (bool * int * int list) list }
(* each part: on COPY:0, its bytes (copies) or entries (COMPUTE:0), its after *)

let pp_step ppf s =
  Format.fprintf ppf "{waits %d; %a}" s.waits
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
       (fun ppf (copy, n, after) ->
         Format.fprintf ppf "%s %d after [%s]"
           (if copy then "copy" else "entries")
           n
           (String.concat "," (List.map string_of_int after))))
    s.parts

let steps =
  let open Gen in
  let part =
    let+ copy = bool
    and+ bytes = one_of [ int_range 1 64; int_range 1 65536 ]
    and+ entries = int_range 1 3
    and+ after = list ~size:(int_range 0 2) (int_range 0 2) in
    (copy, (if copy then bytes else entries), after)
  in
  let step =
    let+ waits =
      frequency [ (3, constant 0); (3, int_range 1 64); (2, int_range 200 256) ]
    and+ parts = list ~size:(int_range 0 3) part in
    let parts =
      List.mapi
        (fun i (c, n, after) ->
          (c, n, List.sort_uniq compare (List.filter (fun k -> k < i) after)))
        parts
    in
    { waits; parts }
  in
  with_pp
    (fun ppf l ->
      Format.fprintf ppf "%d submissions, first %a" (List.length l)
        (Format.pp_print_option pp_step)
        (List.nth_opt l 0))
    (list ~size:(int_range 1500 2500) step)

(* Submissions of satisfied waits, copies of mixed sizes and ring entries, with
   joins across channels, wrap both segment rings several times: every value
   completes in order, and the copies leave the bytes the model says. *)
let mixed subs =
  S.with_driver @@ fun g ->
  let size = 1 lsl 16 in
  let l = S.launches g in
  let w = alloc g `Pinned 8 in
  let src = alloc g `Pinned size and dst = alloc g `Pinned size in
  let scratch = alloc g `Pinned 8 in
  S.set64 (host w) 1;
  S.pattern (host src) size 1;
  S.pattern (host dst) size 2;
  let copied = Bytes.of_string (S.read (host dst) size) in
  let segment = release l (address scratch) 1 in
  let model = Segments.make (N.capability g) in
  let v = ref 0 in
  let submit ~waits ps =
    hand g v ~waits:(Array.make waits (address w, 1)) ps;
    Segments.submit model !v ~waits ps
  in
  (* On every other pass of COMPUTE:0's segment ring, once its end is near, a
     submission of [p] parts there and satisfied waits whose words end exactly
     at the ring's end: each part after the first adds a wait for idle. *)
  let fit () =
    let c = model.ch.(0) in
    let at = c.written mod Segments.ring in
    let left = Segments.ring - at in
    let entered = if c.released <> !v then Segments.acquire else 0 in
    let waits p =
      let rest =
        left - entered - ((p - 1) * Segments.idle) - Segments.release 0
      in
      if rest >= 0 && rest mod Segments.acquire = 0 then
        let n = rest / Segments.acquire in
        if n <= 256 then Some (p, n) else None
      else None
    in
    if at > 0 && c.written / Segments.ring mod 2 = 0 then
      match List.find_map waits [ 1; 2; 3 ] with
      | None -> ()
      | Some (p, n) ->
          submit ~waits:n (Array.init p (fun _ -> on_compute segment))
  in
  List.iteri
    (fun s { waits; parts } ->
      let at = s * 4099 mod size in
      let ps =
        Array.of_list
          (List.map
             (fun (copy, n, after) ->
               let after = Array.of_list after in
               if copy then begin
                 let at = min at (size - n) in
                 Bytes.blit_string (S.read (host src + at) n) 0 copied at n;
                 on_copy ~after
                   ~dst:(address dst + at)
                   ~src:(address src + at)
                   n
               end
               else
                 on_compute ~after
                   (Array.concat (List.init n (fun _ -> segment))))
             parts)
      in
      let seen = N.signaled g in
      submit ~waits ps;
      at_least int ~msg:"the word" ~than:seen (N.signaled g);
      fit ())
    subs;
  cover "a submission of 200 waits or more"
    (List.exists (fun s -> s.waits >= 200) subs);
  S.wait g !v;
  equal int ~msg:"the word" !v (N.signaled g);
  equal string ~msg:"the copies" (Bytes.to_string copied)
    (S.read (host dst) size);
  S.free_launches l;
  List.iter (N.free g) [ w; src; dst; scratch ]

let room =
  group ~timeout:300. "room"
    [
      test "room is Later while the rings hold unreached work, then Fits" later;
      test "40,000 submissions wrap both channels and every copy arrives" wraps;
      test "40,000 copies one after the other pass the segment ring's end"
        sequential;
      test "work waiting on copies runs on past their joins' 32-bit carry"
        join_carry;
      test "two engines' releases across a 32-bit carry never tear" carry_tear;
      prop ~count:12 "submissions that wrap the segment rings complete in order"
        steps mixed;
    ]

(* Local memory *)

(* The 32-bit word i of the stack kernels' output. *)
let stacked i = (512 * i) + 130816

let local () =
  S.with_gpu @@ fun t ->
  let k = S.kernels t in
  let l = S.launches t.g in
  let n = 1024 in
  let out = alloc t.g `Pinned (4 * n) in
  let compute () =
    S.write (host out) (String.make (4 * n) '\000');
    S.run t [| S.words (S.launch l k "stack" ~blocks:4 [ address out; n ]) |];
    S.reset l;
    equal (list int) (List.init n stacked) (List.init n (S.get32 (host out)))
  in
  compute ();
  let { A.Gpu.local; _ } = N.capability t.g in
  equal (result unit string) ~msg:"a smaller one" (Ok ()) (local 1024);
  compute ();
  S.free_launches l;
  N.free t.g out

(* A kernel holds its local memory while the device grows it and schedules a
   kernel on the new one: both compute right. *)
let growth () =
  S.with_gpu @@ fun t ->
  let k = S.kernels t in
  let l = S.launches t.g in
  let n = 1024 in
  let flag = alloc t.g `Pinned 8 in
  let first = alloc t.g `Pinned (4 * n) in
  let next = alloc t.g `Pinned (4 * n) in
  S.set64 (host flag) 0;
  releasing flag 1 (fun () ->
      let held =
        S.launch l k "stack_held" ~blocks:4
          [ address flag; 10 * second; address first; n ]
      in
      ignore (S.submit t [| S.words held |]);
      let { A.Gpu.local; _ } = N.capability t.g in
      equal (result unit string) ~msg:"grown" (Ok ()) (local 4096);
      let v =
        S.submit t
          [| S.words (S.launch l k "stack" ~blocks:4 [ address next; n ]) |]
      in
      S.still ~msg:"the word while held" int (v - 2)
        (fun () -> N.signaled t.g)
        ~ms:20;
      S.set64 (host flag) 1;
      C.wait t.d v);
  let want = List.init n stacked in
  equal (list int) ~msg:"the held kernel" want
    (List.init n (S.get32 (host first)));
  equal (list int) ~msg:"the next kernel" want
    (List.init n (S.get32 (host next)));
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
    | `Place (n, lay) -> (n, lay)
    | `Loaded _ -> fail "an image with nothing to place"
  in
  equal int ~msg:"the region's bytes" size n;
  let r = alloc g `Device n in
  let i, bytes = lay r in
  equal int ~msg:"the image's bytes" size (String.length bytes);
  equal bool ~msg:"double_index" true
    (Option.is_some (N.entry i "double_index"));
  equal (option int) ~msg:"a missing kernel" None (N.entry i "missing");
  is_error ~msg:"not a cubin" (N.image g "not a cubin");
  N.unload g i;
  raises_match ~msg:"entry after unload" Exn.invalid_arg (fun () ->
      N.entry i "empty");
  raises_match ~msg:"unload twice" Exn.invalid_arg (fun () -> N.unload g i);
  N.free g r

(* A cubin loads where another one's code ran and was unloaded: its launch runs
   its own code. Each image goes to its region by a kernel that copies it from
   Mapped memory. *)
let reloaded () =
  S.with_gpu @@ fun t ->
  let k = S.kernels t in
  let l = S.launches t.g in
  let n = 1000 in
  let out = alloc t.g `Pinned (4 * n) in
  let place file =
    let bin = S.fixture file in
    let i, code, bytes = S.image t.g bin in
    let staging = alloc t.g `Mapped (String.length bytes) in
    S.write (host staging) bytes;
    S.run t
      [|
        S.words
          (copy_after l k ~dst:(address code) ~src:(address staging)
             (String.length bytes));
      |];
    N.free t.g staging;
    (bin, i, code)
  in
  let compute (bin, _, code) factor =
    S.write (host out) (String.make (4 * n) '\000');
    S.run t
      [|
        S.words
          (S.launch_at l ~code:(address code) bin "index" ~blocks:4
             [ address out; n ]);
      |];
    S.reset l;
    equal (list int) ~msg:(strf "%d i" factor)
      (List.init n (fun i -> factor * i))
      (List.init n (S.get32 (host out)))
  in
  let ((_, i, code) as twice) = place "twice_sm89.cubin" in
  compute twice 2;
  let at = address code in
  N.unload t.g i;
  N.free t.g code;
  let ((_, i', code') as thrice) = place "thrice_sm89.cubin" in
  equal int ~msg:"the second loads at the first one's address" at
    (address code');
  compute thrice 3;
  N.unload t.g i';
  N.free t.g code';
  S.free_launches l;
  N.free t.g out

let images =
  group ~timeout:60. "images"
    [
      test "load, find their kernels and unload" images;
      test "code loaded where other code ran runs as loaded" reloaded;
    ]

(* Timeline and loss *)

let long_work () =
  S.with_gpu @@ fun t ->
  let l = S.launches t.g in
  let w = alloc t.g `Pinned 8 in
  S.set64 (host w) 0;
  releasing w 1 (fun () ->
      let v = S.submit t [| S.words (acquire l (address w) 1) |] in
      let t0 = Sys.time () in
      for _ = 1 to 20 do
        N.sleep t.g ~seen:(v - 1) ~still_ms:50;
        equal int ~msg:"the work still waits" (v - 1) (N.signaled t.g)
      done;
      less float_exact ~msg:"CPU seconds over at least 1 s of sleeps" ~than:0.5
        (Sys.time () -. t0);
      S.set64 (host w) 1;
      N.sleep t.g ~seen:(v - 1) ~still_ms:60_000;
      C.wait t.d v;
      N.sleep t.g ~seen:(v - 1) ~still_ms:60_000);
  S.free_launches l;
  N.free t.g w

(* One domain sleeps on the word while another submits: each sleep returns as
   the word moves. *)
let sleep_aside () =
  S.with_gpu @@ fun t ->
  let n = 1000 in
  let sleeper =
    Domain.spawn (fun () ->
        while N.signaled t.g < n do
          N.sleep t.g ~seen:(N.signaled t.g) ~still_ms:1000
        done)
  in
  for _ = 1 to n do
    ignore (S.submit t [||])
  done;
  Domain.join sleeper;
  equal int ~msg:"the word" n (N.signaled t.g)

let stop_idle () =
  let t = S.gpu () in
  let r = alloc t.g `Device 64 in
  let a = S.pages S.page in
  let m = require_some (N.map_host t.g a 64) in
  S.run t [||];
  S.stop t.g;
  equal int ~msg:"the word" (C.submitted t.d) (N.signaled t.g);
  N.free t.g r;
  N.free t.g m;
  S.free_pages a S.page;
  let { A.Gpu.local; _ } = N.capability t.g in
  is_error ~msg:"local after stop" (local 1024)

(* COMPUTE:0 waits on a host word nobody writes, a release queued behind the
   wait: stop returns with the word at the last value, and the waiting work
   never runs. *)
let stop_waiting () =
  let t = S.gpu () in
  let l = S.launches t.g in
  let w = alloc t.g `Pinned 8 in
  let marked = alloc t.g `Pinned 8 in
  S.set64 (host w) 0;
  S.set64 (host marked) 0;
  let ws =
    Array.append (acquire l (address w) 1) (release l (address marked) 7)
  in
  let v = S.submit t [| S.words ws |] in
  S.still ~msg:"the word" int (v - 1) (fun () -> N.signaled t.g) ~ms:20;
  S.watchdog "stop" (fun () -> S.stop t.g);
  equal int ~msg:"the word" v (N.signaled t.g);
  S.set64 (host w) 1;
  S.still ~msg:"the waiting work" int 0
    (fun () -> S.get64 (host marked))
    ~ms:200;
  S.free_launches l;
  List.iter (N.free t.g) [ w; marked ];
  S.with_gpu (fun t' -> S.run t' [||])

(* A kernel runs with a release queued behind it: stop returns with the word at
   the last value, and the queued work never runs. *)
let stop_running () =
  let t = S.gpu () in
  let k = S.kernels t in
  let l = S.launches t.g in
  let flag = alloc t.g `Pinned 8 in
  let marked = alloc t.g `Pinned 8 in
  S.set64 (host flag) 0;
  S.set64 (host marked) 0;
  releasing flag 1 (fun () ->
      let spin = S.launch l k "spin" ~blocks:1 [ address flag; 10 * second ] in
      let v1 = S.submit t [| S.words spin |] in
      let v2 = S.submit t [| S.words (release l (address marked) 7) |] in
      S.still ~msg:"the word" int (v1 - 1) (fun () -> N.signaled t.g) ~ms:20;
      S.watchdog "stop" (fun () -> S.stop t.g);
      equal int ~msg:"the word" v2 (N.signaled t.g));
  S.still ~msg:"the queued work" int 0 (fun () -> S.get64 (host marked)) ~ms:200;
  S.free_launches l;
  List.iter (N.free t.g) [ flag; marked ]

let timeline =
  group ~timeout:60. "timeline"
    [
      test "long work is no fault, and a stale seen returns at once" long_work;
      test "sleep returns as the word moves while another domain submits"
        sleep_aside;
      test "stop of an idle device leaves the word at the last value" stop_idle;
      test
        "stop of a device waiting on a word that never moves leaves the word \
         at the last value (sampled)"
        stop_waiting;
      test
        "stop of a running device leaves the word at the last value, and its \
         queued work never runs (sampled)"
        stop_running;
    ]

(* Two GPUs *)

let two_gpus () =
  if Rig_nv_nvidia.count () < 2 then
    skip ~reason:"the machine has fewer than two NVIDIA GPUs" ();
  let a = S.driver () in
  Fun.protect ~finally:(fun () -> S.stop a) @@ fun () ->
  let b = require_ok (Rig_nv_nvidia.open_ 1) in
  Fun.protect ~finally:(fun () -> N.stop b) @@ fun () ->
  let h = alloc b `Pinned 64 in
  let d = alloc b `Device 64 in
  let ph = require_some ~msg:"host memory maps" (N.map_peer a b h) in
  equal bool ~msg:"Device memory maps iff peer" (N.peer a b)
    (match N.map_peer a b d with
    | Some pd ->
        N.free a pd;
        true
    | None -> false);
  N.free a ph;
  N.free b h;
  N.free b d

let two =
  group ~timeout:60. "two GPUs" [ test "map each other's memory" two_gpus ]

(* The shared device: the stateful tests' programs use one device, opened by the
   first and stopped when the run ends. *)

let shared = fixture ~teardown:S.close S.gpu

(* map_host: the registry. The path maps whole pages: a range inside the pages
   of a mapped one shares its mapping, a range that shares some of their pages
   is refused. *)

module Registry = struct
  type entry = { lo : int; hi : int; mutable maps : int } (* pages [lo, hi) *)
  type t = { mutable entries : entry list; mutable regions : entry list }

  let arena = 4
  let pages (a, n) = (a / S.page, ((a + n - 1) / S.page) + 1)
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
    t : S.dev;
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
      base = S.pages (arena * S.page);
      scratch = Option.get (N.alloc t.g `Pinned (arena * S.page));
      kernels = S.kernels t;
      launches = S.launches t.g;
      lock = Mutex.create ();
      live = [];
    }

  let release s =
    List.iter (fun (r, _, _) -> N.free s.t.g r) s.live;
    N.free s.t.g s.scratch;
    S.free_launches s.launches;
    S.free_pages s.base (arena * S.page)

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
    let scale x = x * S.page / 4096 in
    Gen.with_pp
      (fun ppf (a, n) -> Format.fprintf ppf "(%d, %d)" a n)
      (Gen.frequency
         [
           ( 8,
             Gen.such_that
               (fun (a, n) -> a + n <= arena * S.page)
               (Gen.map
                  (fun (a, n) -> (scale a, scale n))
                  (Gen.pair (Gen.of_list point) (Gen.of_list length))) );
           (* One range drawn often, so that ranges of the same pages are drawn
              too. *)
           (2, Gen.constant (0, S.page));
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

(* Submissions: the order of values *)

module Order = struct
  (* A part copies [n] bytes of buffer [src] at [so] to buffer [dst] at [do_]:
     on COPY:0 iff [copy], else by a kernel on COMPUTE:0 that starts [delay]
     nanoseconds late, so that work run out of order shows in the buffers. It
     runs after the earlier parts [after] lists. *)
  type part = {
    delay : int;
    copy : bool;
    src : int;
    so : int;
    dst : int;
    do_ : int;
    n : int;
    after : int list;
  }

  let buffers = 3
  let size = 65536

  let pp_part ppf p =
    Format.fprintf ppf "{%s%s %d@%d -> %d@%d n=%d after=[%s]}"
      (if p.copy then "COPY" else "COMPUTE")
      (if p.delay > 0 then strf " +%dns" p.delay else "")
      p.src p.so p.dst p.do_ p.n
      (String.concat ";" (List.map string_of_int p.after))

  let pp_one ppf ps =
    Format.fprintf ppf "[%a]"
      (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_part)
      ps

  let pp ppf subs =
    Format.pp_print_list ~pp_sep:Format.pp_print_space pp_one ppf subs

  let initial b =
    Bytes.init size (fun i -> Char.chr (((b * 61) + (i * 7)) land 255))

  type t = { bufs : Bytes.t array; mutable last : bool option }

  let make () = { bufs = Array.init buffers initial; last = None }

  (* Part [i] runs before part [j], [i < j]: both on one queue, or [j] runs
     after a part that runs after [i]. *)
  let rec before ps i j =
    let pi = List.nth ps i and pj = List.nth ps j in
    pi.copy = pj.copy
    || List.exists (fun k -> k = i || (k > i && before ps i k)) pj.after

  let overlap (b, o) (b', o') n n' = b = b' && o < o' + n' && o' < o + n

  let conflict p q =
    overlap (p.dst, p.do_) (q.dst, q.do_) p.n q.n
    || overlap (p.src, p.so) (q.dst, q.do_) p.n q.n
    || overlap (p.dst, p.do_) (q.src, q.so) p.n q.n

  (* The result is one: every pair of parts that conflict is ordered, and no
     part copies over its own source. *)
  let determined_one ps =
    List.for_all
      (fun p -> not (overlap (p.src, p.so) (p.dst, p.do_) p.n p.n))
      ps
    &&
    let n = List.length ps in
    List.for_all
      (fun j ->
        List.for_all
          (fun i ->
            before ps i j || not (conflict (List.nth ps i) (List.nth ps j)))
          (List.init j Fun.id))
      (List.init n Fun.id)

  let determined _ subs = List.for_all determined_one subs

  let run_one m ps =
    let channels = List.sort_uniq compare (List.map (fun p -> p.copy) ps) in
    let releaser =
      match List.rev ps with p :: _ -> Some p.copy | [] -> None
    in
    cover "a submission of no parts after one released on COPY:0"
      (ps = [] && m.last = Some true);
    cover "a submission on the channel that released the last"
      (releaser <> None && m.last = releaser);
    cover "a switch of channel"
      (releaser <> None && m.last <> None && m.last <> releaser);
    cover "two channels in one submission" (List.length channels = 2);
    cover "an after from COPY:0 to COMPUTE:0"
      (List.exists
         (fun p ->
           (not p.copy) && List.exists (fun k -> (List.nth ps k).copy) p.after)
         ps);
    cover "an after from COMPUTE:0 to COPY:0"
      (List.exists
         (fun p ->
           p.copy && List.exists (fun k -> not (List.nth ps k).copy) p.after)
         ps);
    List.iter
      (fun p -> Bytes.blit m.bufs.(p.src) p.so m.bufs.(p.dst) p.do_ p.n)
      ps;
    m.last <- Some (Option.value releaser ~default:false)

  let run m subs =
    cover "several values in flight" (List.length subs > 1);
    List.iter (run_one m) subs
  (* The system *)

  type sys = {
    t : S.dev;
    regions : B.t array;
    staging : B.t;
    staged : int;
    kernels : S.kernels;
    launches : S.launches;
  }

  let start () =
    let t = shared () in
    let staging, staged = S.shared t size in
    let regions =
      Array.init buffers (fun b ->
          let r = B.create t.d size in
          S.write staged (Bytes.to_string (initial b));
          S.run t [| S.copy ~dst:r staging |];
          r)
    in
    {
      t;
      regions;
      staging;
      staged;
      kernels = S.kernels t;
      launches = S.launches t.g;
    }

  let release s = S.free_launches s.launches

  (* Submits [subs] back to back, then reads the word until it holds the last
     value: each read is at least the one before. *)
  let run_sys s subs =
    let part p =
      let after = Array.of_list p.after in
      let dst = s.regions.(p.dst) and src = s.regions.(p.src) in
      if p.copy then
        S.copy ~after
          ~dst:(B.view dst ~first:p.do_ ~length:p.n)
          (B.view src ~first:p.so ~length:p.n)
      else
        S.words ~after
          (copy_after s.launches s.kernels ~ns:p.delay
             ~dst:(B.address dst + p.do_)
             ~src:(B.address src + p.so)
             p.n)
    in
    let first = C.submitted s.t.d + 1 in
    List.iter
      (fun ps -> ignore (S.submit s.t (Array.of_list (List.map part ps))))
      subs;
    let last = C.submitted s.t.d in
    let rec watch seen =
      let w = N.signaled s.t.g in
      at_least int ~msg:"the word" ~than:seen w;
      at_most int ~msg:"the word" ~than:last w;
      if w < last then watch w
    in
    watch (first - 1);
    S.still ~msg:"the word" int last (fun () -> N.signaled s.t.g) ~ms:1;
    S.reset s.launches

  let invariant m s =
    Array.iteri
      (fun b r ->
        S.run s.t [| S.copy ~dst:s.staging r |];
        equal string ~msg:(strf "buffer %d" b)
          (Bytes.to_string m.bufs.(b))
          (S.read s.staged size))
      s.regions

  let parts =
    let open Gen in
    let part =
      let+ delay = frequency [ (2, constant 0); (1, constant 50_000) ]
      and+ copy = bool
      and+ src = int_range 0 (buffers - 1)
      and+ dst = int_range 0 (buffers - 1)
      and+ n = one_of [ int_range 1 64; int_range 1 16384 ]
      and+ so = int_range 0 (size - 16384)
      and+ do_ = int_range 0 (size - 16384)
      and+ after = list ~size:(int_range 0 2) (int_range 0 2) in
      { delay = (if copy then 0 else delay); copy; src; so; dst; do_; n; after }
    in
    let submission =
      let+ ps = list ~size:(int_range 0 3) part in
      List.mapi
        (fun i p ->
          let after = List.filter (fun k -> k < i) p.after in
          { p with after = List.sort_uniq compare after })
        ps
    in
    list ~size:(int_range 1 4) submission
end

let order = abstract "o" ~invariant:Order.invariant ~release:Order.release

let order_commands =
  [
    command "start" (Gen.unit @-> makes order) Order.make Order.start;
    command "submit" ~pre:Order.determined
      (order ^-> Gen.with_pp Order.pp Order.parts @-> returns unit)
      Order.run Order.run_sys;
  ]

(* Ending from two domains: whatever the order, an allocation's first [free], a
   mapping's first [free] and an image's first [unload] return, and every later
   one raises. *)

type ended = { mutable live : bool }

let end_model m =
  if not m.live then invalid_arg "ended";
  m.live <- false

let ends_once ~make ~finish ~release name =
  let v =
    abstract name ~release:(fun x ->
        try release x with Invalid_argument _ -> ())
  in
  [
    command "make" (Gen.unit @-> makes v) (fun () -> { live = true }) make;
    command "end" (v ^-> returns unit) end_model finish;
  ]

(* Each value carries the shared device: the fixture is read on the test's
   domain only. *)
let allocation_commands =
  ends_once "a"
    ~make:(fun () ->
      let g = (shared ()).g in
      (g, Option.get (N.alloc g `Device 64)))
    ~finish:(fun (g, r) -> N.free g r)
    ~release:(fun (g, r) -> N.free g r)

(* A mapping of its own page, freed once the run ends. *)
let mapping_commands =
  ends_once "m"
    ~make:(fun () ->
      let g = (shared ()).g and p = S.pages S.page in
      (g, p, Option.get (N.map_host g p 64)))
    ~finish:(fun (g, _, r) -> N.free g r)
    ~release:(fun (g, p, r) ->
      Fun.protect
        ~finally:(fun () -> S.free_pages p S.page)
        (fun () -> N.free g r))

(* An image over a code region of its own, freed once the run ends. *)
let image_commands =
  ends_once "i"
    ~make:(fun () ->
      let g = (shared ()).g in
      let i, r, _ = S.image g (S.fixture "kernels_sm89.cubin") in
      (g, i, r))
    ~finish:(fun (g, i, _) -> N.unload g i)
    ~release:(fun (g, i, r) ->
      Fun.protect ~finally:(fun () -> N.free g r) (fun () -> N.unload g i))

(* Local memory handed over from two domains: one grows the kernels' local
   memory while the other submits, and the word moves on by small steps. A local
   memory the device replaced goes back to the path only once the word reached
   every value that could run on it: those handed before its successor's [local]
   returned. *)
module Handover = struct
  type t = {
    g : N.t;
    f : Fake.t;
    turn : Mutex.t; (* rig_nv_submit runs one call at a time *)
    grows : Mutex.t; (* one growth and its record at a time *)
    lock : Mutex.t; (* the word's steps and the memories' last users *)
    handed : int Atomic.t;
    mutable newest : int option; (* the newest local memory *)
    users : (int, int) Hashtbl.t; (* a replaced memory's last possible user *)
    mutable early : string option; (* a memory freed before its users ran *)
    mutable returned : int; (* replaced memories gone back *)
    mutable kib : int; (* the local memory a thread has, in KiB *)
  }

  let word h = host (N.word h.g)

  let start () =
    let f = Fake.make 0 in
    let g = require_ok (N.make (Fake.path f)) in
    let h =
      {
        g;
        f;
        turn = Mutex.create ();
        grows = Mutex.create ();
        lock = Mutex.create ();
        handed = Atomic.make 0;
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
          | Some v when S.get64 (word h) < v ->
              h.early <-
                Some
                  (strf "0x%x freed at word %d, used up to %d" m.address
                     (S.get64 (word h))
                     v)
          | Some _ -> h.returned <- h.returned + 1
          | None -> ());
    h

  (* Grows local memory by 1 KiB a thread, which replaces it. *)
  let grow h =
    Mutex.protect h.grows @@ fun () ->
    h.kib <- h.kib + 1;
    match (N.capability h.g).local (h.kib * 1024) with
    | Error e -> failf "local: %s" e
    | Ok () -> (
        let newest =
          List.find_map
            (fun (a, host, _) -> if host = None then Some a else None)
            h.f.memory
        in
        Mutex.protect h.lock @@ fun () ->
        match (h.newest, newest) with
        | Some old, Some m when old <> m ->
            Hashtbl.replace h.users old (Atomic.get h.handed);
            h.newest <- Some m
        | None, m -> h.newest <- m
        | Some _, _ -> ())

  (* Hands one empty submission, which takes a pending local memory, then moves
     the word on by [k] values, up to the last handed. *)
  let submit k h =
    Mutex.protect h.turn (fun () ->
        if S.edge_room h.g [||] = 0 then begin
          let v = Atomic.get h.handed + 1 in
          Atomic.set h.handed v;
          S.edge_submit h.g ~v ~waits:[||] [||]
        end);
    Mutex.protect h.lock @@ fun () ->
    let w = S.get64 (word h) in
    S.set64 (word h) (min (Atomic.get h.handed) (w + k))

  let invariant _ h =
    cover "a replaced local memory went back" (h.returned > 0);
    equal (option string) ~msg:"a local memory freed early" None h.early

  let release h = N.stop h.g
end

let handover =
  abstract "h" ~invariant:Handover.invariant ~release:Handover.release

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
      stateful ~count:100 ~steps:20
        "values complete in order and the word never moves backwards (sampled)"
        order_commands;
      stateful ~count:30 ~domains:2
        "an allocation freed from two domains is freed once" allocation_commands;
      stateful ~count:30 ~domains:2
        "a mapping freed from two domains is freed once" mapping_commands;
      stateful ~count:30 ~domains:2
        "an image unloaded from two domains is unloaded once" image_commands;
      stateful ~count:100 ~steps:40 ~domains:2
        "local memory goes back only once the values that could use it ran"
        handover_commands;
    ]

(* Progress bounds

   A path that bounds work's progress ([hang_ms]) makes [sleep] raise once a
   value is outstanding and the word has not moved for that long. The fake
   path's word moves only when a test stores to it. *)

let bounded hang_ms =
  let f = Fake.make 0 in
  f.hang_ms <- hang_ms;
  require_ok (N.make (Fake.path f))

let hangs () =
  let g = bounded (Some 50) in
  hand g (ref 0) [||];
  let rec sleeps n =
    if n = 0 then fail "no Fault after 10 sleeps of a value making no progress";
    match N.sleep g ~seen:0 ~still_ms:1000 with
    | () -> sleeps (n - 1)
    | exception N.Fault why -> contains ~msg:"the report" ~sub:"50 ms" why
  in
  sleeps 10;
  N.stop g

let idle () =
  let g = bounded (Some 50) in
  N.sleep g ~seen:0 ~still_ms:150;
  hand g (ref 0) [||];
  N.sleep g ~seen:0 ~still_ms:1;
  N.stop g

let moving () =
  let g = bounded (Some 50) in
  let v = ref 0 in
  for _ = 1 to 8 do
    hand g v [||]
  done;
  for v = 1 to 8 do
    N.sleep g ~seen:(v - 1) ~still_ms:20;
    S.set64 (host (N.word g)) v
  done;
  N.stop g

let unbounded () =
  let g = bounded None in
  hand g (ref 0) [||];
  N.sleep g ~seen:0 ~still_ms:150;
  N.sleep g ~seen:0 ~still_ms:150;
  N.stop g

let bounds () =
  List.iter
    (fun n ->
      let f = Fake.make 0 in
      f.hang_ms <- Some n;
      raises_match ~msg:(strf "hang_ms %d" n)
        (Exn.invalid_arg ~substring:"Rig_nv.make") (fun () ->
          N.make (Fake.path f)))
    [ 0; -1; min_int ]

let progress =
  group ~timeout:30. "progress"
    [
      test "a value that makes no progress for hang_ms is a fault" hangs;
      test "an idle device never hangs, nor its next value at once" idle;
      test "values reached more often than hang_ms are no fault" moving;
      test "without hang_ms a value making no progress is no fault" unbounded;
      test "make raises on a hang_ms below 1" bounds;
    ]

let () =
  S.hold_gpu ();
  exit
    (run "rig_nv"
       [
         paths;
         facts;
         memory;
         work;
         room;
         local;
         images;
         timeline;
         progress;
         two;
         stateful;
       ])
