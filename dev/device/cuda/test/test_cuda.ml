(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module C = Device_cuda
module S = Device_cuda_support

let strf = Printf.sprintf

let answer =
  Testable.make
    ~pp:(fun ppf -> function
      | `Ok -> Format.pp_print_string ppf "`Ok"
      | `Failed why -> Format.fprintf ppf "`Failed %S" why)
    ~equal:( = )

let stop_answer =
  Testable.make
    ~pp:(fun ppf -> function
      | `Stopped -> Format.pp_print_string ppf "`Stopped"
      | `Unknown -> Format.pp_print_string ppf "`Unknown")
    ~equal:( = )

let address r = Option.get (C.address r)
let host r = Option.get (C.host r)
let word g = host (C.word g)
let submit g ~v ?(waits = [||]) ps = C.submit g ~v ~waits ~handles:[||] ps

(* Reads [g]'s word for about [ms] milliseconds of CPU time, failing if it is
   ever other than [v]. *)
let still g v ~ms =
  let t0 = Sys.time () in
  while Sys.time () -. t0 < Float.of_int ms /. 1000. do
    equal int ~msg:"the word" v (C.signaled g)
  done

(* Opening *)

let gpu_once () =
  let g = S.gpu () in
  let e = require_error (C.open_ 0) in
  contains ~sub:"open" e;
  equal stop_answer `Stopped (C.stop g);
  let g' = require_ok (C.open_ 0) in
  equal stop_answer `Stopped (C.stop g')

let opening =
  group ~timeout:60. "opening"
    [
      test "count never raises" (fun () -> at_least int ~than:0 (C.count ()));
      test "names GPU 0 CUDA and GPU i CUDA:i" (fun () ->
          equal (list string)
            [ "CUDA"; "CUDA:1"; "CUDA:7" ]
            (List.map C.device_name [ 0; 1; 7 ]));
      test "a negative GPU raises" (fun () ->
          raises_match Exn.invalid_arg (fun () -> C.open_ (-1));
          raises_match Exn.invalid_arg (fun () -> C.device_name (-1)));
      test "a GPU past the count is an error naming the count" (fun () ->
          let n = C.count () in
          let e = require_error (C.open_ n) in
          contains ~sub:(if n = 0 then "CUDA" else strf "%d GPU" n) e);
      test "a GPU has one device until it is stopped" gpu_once;
    ]

(* Facts *)

let facts () =
  S.with_gpu @@ fun g ->
  let arch = C.arch g in
  equal int ~msg:"length of the arch" 5 (String.length arch);
  starts_with ~affix:"sm_" arch;
  greater int ~than:0 (C.budget g);
  equal (option string) None (C.machine g);
  equal (list string) [ "COMPUTE:0"; "COPY:0" ] (C.queues g);
  equal bool ~msg:"completion is the store" true (C.completion g = `Store);
  equal (list bool) [ true; false; true ]
    (List.map (C.waits_on g) [ `Store; `Object; `Host ]);
  equal bool ~msg:"submit may block" true (C.blocks g = `May_block);
  equal nativeint ~msg:"the word's handle is its host address" (word g)
    (C.handle (C.word g));
  equal int ~msg:"the word starts at 0" 0 (C.signaled g)

let symbols () =
  S.with_gpu @@ fun g ->
  let { Device_cuda_abi.symbol } = C.capability g in
  let found n = Option.is_some (symbol n) in
  equal (list bool)
    [ true; true; true; true; false; false ]
    (List.map found
       [
         "cuLaunchKernel";
         "cuMemcpyAsync";
         "cuLaunchHostFunc";
         "cuGraphLaunch";
         "cuNoSuchFunction";
         "cuLaunchKernel\000";
       ]);
  equal bool ~msg:"the key is the ABI's" true
    (Option.is_some
       (Type.Id.provably_equal C.capability_key Device_cuda_abi.key))

let facts =
  group ~timeout:60. "facts"
    [
      test "states a GPU's facts" facts;
      test "finds the functions a fill calls" symbols;
    ]

(* Memory *)

let kinds = [ `Device; `Pinned; `Mapped ]

let pp_kind ppf k =
  Format.pp_print_string ppf
    (match k with
    | `Device -> "`Device"
    | `Pinned -> "`Pinned"
    | `Mapped -> "`Mapped")

let kind = Gen.of_list ~pp:pp_kind kinds
let size = Gen.of_list ~pp:Format.pp_print_int [ 1; 7; 4096; (1 lsl 20) + 7 ]
let offset = Gen.of_list ~pp:Format.pp_print_int [ 0; 1; 4095 ]

let pattern n seed =
  String.init n (fun i -> Char.chr (((i * 7) + seed) land 255))

(* host -> a -> b -> host through Copy parts, on both queues. *)
let round_trip (ka, kb, n, (oa, ob)) =
  S.with_gpu @@ fun g ->
  let src = require_some (C.alloc g `Pinned n) in
  let dst = require_some (C.alloc g `Pinned n) in
  let a = require_some (C.alloc g ka (n + oa)) in
  let b = require_some (C.alloc g kb (n + ob)) in
  equal bool ~msg:"host of a" (ka <> `Device) (Option.is_some (C.host a));
  let data = pattern n (n + oa) in
  S.write (host src) data;
  let copy q after d s = C.part g ~queue:q ~after (`Copy (d, s, n)) in
  let ps =
    [|
      copy "COPY:0" [||] (a, oa) (src, 0);
      copy "COMPUTE:0" [| 0 |] (b, ob) (a, oa);
      copy "COPY:0" [| 1 |] (dst, 0) (b, ob);
    |]
  in
  equal answer `Ok (submit g ~v:1 ps);
  S.wait g 1;
  equal string data (S.read (host dst) n);
  List.iter (C.free g) [ src; dst; a; b ]

let memory =
  group ~timeout:120. "memory"
    [
      prop ~count:30 "copies through any two kinds of memory are the identity"
        (Gen.quad kind kind size (Gen.pair offset offset))
        round_trip;
    ]

(* Work *)

let fills_in_a_fresh_domain () =
  S.with_gpu @@ fun g ->
  let m, kernel = S.kernels g in
  let n = 1000 in
  let out = require_some (C.alloc g `Pinned (4 * n)) in
  let f = S.launch (kernel "double_index") ~grid:4 ~block:256 (address out) n in
  let r, current =
    Domain.join
      (Domain.spawn (fun () ->
           let r = submit g ~v:1 [| S.part g ~queue:"COMPUTE:0" f |] in
           (r, S.current ())))
  in
  equal answer `Ok r;
  equal nativeint ~msg:"the domain's thread has no context after" 0n current;
  not_equal nativeint ~msg:"the fill saw a context" 0n (S.seen f);
  S.wait g 1;
  for i = 0 to n - 1 do
    equal int ~msg:(strf "word %d" i) (2 * i) (S.get32 (host out) i)
  done;
  C.unload g m;
  C.free g out

let failed_fill () =
  let g = S.gpu () in
  let r = submit g ~v:1 [| S.part g ~queue:"COMPUTE:0" (S.failing 1) |] in
  let why = require_match (function `Failed w -> Some w | `Ok -> None) r in
  starts_with ~affix:"CUDA_ERROR_INVALID_VALUE: " why;
  equal answer ~msg:"the next submit" (`Failed why) (submit g ~v:2 [||]);
  equal stop_answer `Stopped (C.stop g);
  equal int ~msg:"the word holds the last value" 2 (C.signaled g)

(* A wait on a host word holds the work until the host writes it. *)
(* A wait on a host word holds a submission's work on both queues until the
   host writes the word: each queue copies GPU memory to a host buffer. *)
let held g ~kind ~start ~wait ~below ~release =
  let w = require_some (C.alloc g `Pinned 8) in
  let src = require_some (C.alloc g `Device 64) in
  let dst = Array.init 2 (fun _ -> require_some (C.alloc g `Pinned 64)) in
  let zeros = String.make 64 '\000' and data = pattern 64 1 in
  S.write_gpu (C.handle src) data;
  Array.iter (fun d -> S.write (host d) zeros) dst;
  let copy q d = C.part g ~queue:q (`Copy ((d, 0), (src, 0), 64)) in
  let held () =
    still g 0 ~ms:20;
    Array.iter
      (fun d -> equal string ~msg:"held" zeros (S.read (host d) 64))
      dst
  in
  S.set64 (host w) start;
  Fun.protect ~finally:(fun () -> S.set64 (host w) release) @@ fun () ->
  let ps = [| copy "COMPUTE:0" dst.(0); copy "COPY:0" dst.(1) |] in
  equal answer `Ok (submit g ~v:1 ~waits:[| (kind, address w, wait) |] ps);
  held ();
  S.set64 (host w) below;
  held ();
  S.set64 (host w) release;
  S.wait g 1;
  Array.iter (fun d -> equal string ~msg:"copied" data (S.read (host d) 64)) dst

let waits =
  [
    test "a Word wait holds the work across the 64-bit wrap (sampled)"
      (fun () ->
        S.with_gpu (held ~kind:`Word ~start:(-3) ~wait:2 ~below:1 ~release:2));
    test "an Equal wait holds the work until the word equals (sampled)"
      (fun () ->
        S.with_gpu (held ~kind:`Equal ~start:4 ~wait:5 ~below:6 ~release:5));
    test "an Object wait raises" (fun () ->
        S.with_gpu @@ fun g ->
        raises_match Exn.invalid_arg (fun () ->
            submit g ~v:1 ~waits:[| (`Object, 0, 1) |] [||]));
    cases ~name:(strf "%d satisfied waits complete on both queues")
      "batches" [ 255; 256 ] (fun n ->
        S.with_gpu @@ fun g ->
        let w = require_some (C.alloc g `Pinned 8) in
        let b = require_some (C.alloc g `Pinned 16) in
        S.set64 (host w) 1;
        let waits = Array.make n (`Word, address w, 1) in
        let copy q after =
          C.part g ~queue:q ~after (`Copy ((b, 8), (b, 0), 8))
        in
        let ps = [| copy "COPY:0" [||]; copy "COMPUTE:0" [| 0 |] |] in
        equal answer `Ok (submit g ~v:1 ~waits ps);
        S.wait g 1);
  ]

let misuse () =
  S.with_gpu @@ fun g ->
  let r = require_some (C.alloc g `Device 64) in
  let fill = (0n, 0n, 0, 0) in
  let part w = ignore (C.part g ~queue:"COMPUTE:0" w) in
  let raises name f = raises_match ~msg:name Exn.invalid_arg f in
  raises "words" (fun () -> part (`Words [| 0 |]));
  raises "ring units" (fun () -> part (`Fill (0n, 0n, 1, 0)));
  raises "segment bytes" (fun () -> part (`Fill (0n, 0n, 0, 1)));
  raises "no such queue" (fun () -> C.part g ~queue:"COPY:1" (`Fill fill));
  raises "a copy past its region" (fun () -> part (`Copy ((r, 0), (r, 40), 32)));
  raises "a negative offset" (fun () -> part (`Copy ((r, -1), (r, 32), 8)));
  raises "a negative after" (fun () ->
      C.part g ~queue:"COMPUTE:0" ~after:[| -1 |] (`Fill fill));
  let p = C.part g ~queue:"COMPUTE:0" ~after:[| 0 |] (`Fill fill) in
  raises "an after at its own index" (fun () -> submit g ~v:1 [| p |]);
  raises "a value other than the next" (fun () -> submit g ~v:2 [||]);
  raises "a negative still_ms" (fun () -> C.sleep g ~seen:1 ~still_ms:(-1));
  raises "alloc of 0 bytes" (fun () -> C.alloc g `Device 0);
  raises "map_host of 0 bytes" (fun () -> C.map_host g (word g) 0);
  raises "map_peer of one device" (fun () -> C.map_peer g g r);
  raises "unmap of an allocation" (fun () -> C.unmap g r);
  raises "free of the word" (fun () -> C.free g (C.word g));
  C.free g r;
  raises "free twice" (fun () -> C.free g r);
  raises "a copy of a freed region" (fun () ->
      part (`Copy ((r, 0), (r, 32), 8)));
  let p = S.pages S.page in
  let m = require_some (C.map_host g p 64) in
  C.unmap g m;
  raises "unmap twice" (fun () -> C.unmap g m);
  S.free_pages p S.page

let another_device () =
  let a = S.gpu () in
  let r = require_some (C.alloc a `Pinned 64) in
  equal stop_answer `Stopped (C.stop a);
  S.with_gpu @@ fun b ->
  raises_match Exn.invalid_arg (fun () ->
      C.part b ~queue:"COPY:0" (`Copy ((r, 0), (r, 32), 8)));
  raises_match Exn.invalid_arg (fun () -> C.free b r);
  C.free a r

let room () =
  S.with_gpu @@ fun g ->
  let room ?(queue = 0) ?(words = false) ?(units = 0) ?(bytes = 0) after =
    S.room g ~queue ~words ~units ~bytes ~after
  in
  equal (list int) [ 0; 0; 2; 2; 2; 2 ]
    [
      room [||];
      room ~queue:1 [||];
      room ~words:true [||];
      room ~units:1 [||];
      room ~bytes:1 [||];
      room ~queue:2 [||];
    ]

let work =
  group ~timeout:60. "work"
    ([
       test "a fill runs with the context current from a fresh domain"
         fills_in_a_fresh_domain;
       test "a failed fill is Failed, and every later submit" failed_fill;
       test "misuse raises" misuse;
       test "a region of another device raises" another_device;
       test "the C room refuses what part refuses" room;
     ]
    @ waits)

(* Images *)

let images () =
  S.with_gpu @@ fun g ->
  let m, upload = require_ok (C.image g (S.fixture "kernels.ptx")) in
  equal bool ~msg:"no upload" true (upload = None);
  equal bool ~msg:"double_index" true
    (Option.is_some (C.entry m "double_index"));
  equal (option int) ~msg:"a missing kernel" None (C.entry m "missing");
  let e = require_error (C.image g "not a module") in
  starts_with ~affix:"CUDA_ERROR_" e;
  (match C.image g (S.fixture "kernels.cubin") with
  | Ok (m', _) ->
      equal string ~msg:"the cubin's GPU" "sm_89" (C.arch g);
      C.unload g m'
  | Error e ->
      not_equal string ~msg:"the cubin's GPU" "sm_89" (C.arch g);
      starts_with ~affix:"CUDA_ERROR_" e);
  let f = S.launch (Option.get (C.entry m "empty")) ~grid:1 ~block:1 0 0 in
  equal answer `Ok (submit g ~v:1 [| S.part g ~queue:"COMPUTE:0" f |]);
  S.wait g 1;
  C.unload g m;
  raises_match Exn.invalid_arg (fun () -> C.entry m "empty");
  raises_match Exn.invalid_arg (fun () -> C.unload g m)

let images =
  group ~timeout:60. "images"
    [ test "load, find their kernels and unload" images ]

(* Timeline and loss *)

let watchdog = 17 (* CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT *)
let second = 1_000_000_000

(* [spin g flag] submits, as value 1, a kernel that runs until [flag]'s first
   word is not 0, for at most 10 seconds. *)
let spin g flag =
  let m, kernel = S.kernels g in
  S.set64 (host flag) 0;
  let f =
    S.launch (kernel "spin") ~grid:1 ~block:1 (address flag) (10 * second)
  in
  equal answer `Ok (submit g ~v:1 [| S.part g ~queue:"COMPUTE:0" f |]);
  m

let long_work () =
  S.with_gpu @@ fun g ->
  if S.attribute watchdog <> 0 then
    skip ~reason:"a display watchdog ends long kernels" ();
  let flag = require_some (C.alloc g `Pinned 8) in
  Fun.protect ~finally:(fun () -> S.set64 (host flag) 1) @@ fun () ->
  let m = spin g flag in
  C.sleep g ~seen:0 ~still_ms:50;
  equal int ~msg:"the work still runs" 0 (C.signaled g);
  S.set64 (host flag) 1;
  S.wait g 1;
  C.sleep g ~seen:0 ~still_ms:60_000;
  C.unload g m

let stop_idle () =
  let g = S.gpu () in
  let p = S.pages S.page in
  let r = require_some (C.map_host g p 64) in
  equal bool ~msg:"locked" true (S.locked p);
  equal answer `Ok (submit g ~v:1 [||]);
  S.wait g 1;
  equal stop_answer `Stopped (C.stop g);
  equal int ~msg:"the word" 1 (C.signaled g);
  C.unmap g r;
  equal bool ~msg:"locked after unmap" false (S.locked p);
  S.free_pages p S.page

let stop_running () =
  let g = S.gpu () in
  if S.attribute watchdog <> 0 then
    skip ~reason:"a display watchdog ends long kernels" ();
  let flag = require_some (C.alloc g `Pinned 8) in
  Fun.protect ~finally:(fun () -> S.set64 (host flag) 1) @@ fun () ->
  ignore (spin g flag);
  equal stop_answer `Unknown (C.stop g);
  S.set64 (host flag) 1;
  S.wait g 1

let registry_is_the_process () =
  let p = S.pages S.page in
  let a = S.gpu () in
  let ra = require_some (C.map_host a p 256) in
  equal stop_answer `Stopped (C.stop a);
  let b = S.gpu () in
  let rb = require_some (C.map_host b (Nativeint.add p 64n) 64) in
  C.unmap a ra;
  equal bool ~msg:"locked after the first unmap" true (S.locked p);
  C.unmap b rb;
  equal bool ~msg:"locked after the last unmap" false (S.locked p);
  equal stop_answer `Stopped (C.stop b);
  S.free_pages p S.page

let timeline =
  group ~timeout:60. "timeline"
    [
      test "long work is no fault, and a stale seen returns at once" long_work;
      test "stop of an idle device leaves the word at the last value" stop_idle;
      test "stop of a running device is Unknown" stop_running;
      test "page-locking is the process's" registry_is_the_process;
    ]

(* Two GPUs *)

let two_gpus () =
  if C.count () < 2 then skip ~reason:"CUDA sees fewer than two GPUs" ();
  let a = S.gpu () in
  Fun.protect ~finally:(fun () -> ignore (C.stop a)) @@ fun () ->
  let b = require_ok (C.open_ 1) in
  Fun.protect ~finally:(fun () -> ignore (C.stop b)) @@ fun () ->
  let h = require_some (C.alloc b `Pinned 64) in
  let d = require_some (C.alloc b `Device 64) in
  let ph = require_some (C.map_peer a b h) in
  equal (option nativeint) ~msg:"host memory maps" (C.host h) (C.host ph);
  (match C.map_peer a b d with
  | Some pd ->
      raises_match Exn.invalid_arg (fun () -> C.free a pd);
      C.unmap a pd
  | None -> ());
  C.unmap a ph;
  C.free b h;
  C.free b d

let two =
  group ~timeout:60. "two GPUs" [ test "map each other's memory" two_gpus ]

(* The shared device: the stateful tests' programs use one device, opened by the
   first and stopped when the run ends. *)

let shared = fixture ~teardown:(fun g -> ignore (C.stop g)) S.gpu

(* map_host: the registry *)

module Registry = struct
  type area = Arena | Foreign | Read_only
  type entry = { start : int; bytes : int; mutable maps : int }
  type region = Counted of entry | Uncounted
  type t = { mutable entries : entry list; mutable regions : region list }

  let arena = 4
  let pages a n = (a / S.page, (a + n - 1) / S.page)

  let shares (a, n) e =
    let lo, hi = pages a n and lo', hi' = pages e.start e.bytes in
    lo <= hi' && lo' <= hi

  let make () = { entries = []; regions = [] }

  let map m (area, a, n) =
    match area with
    | Read_only ->
        cover "read-only memory" true;
        false
    | Foreign ->
        cover "memory CUDA locked for another owner" true;
        m.regions <- m.regions @ [ Uncounted ];
        true
    | Arena -> (
        let inside e = e.start <= a && a + n <= e.start + e.bytes in
        match List.find_opt inside m.entries with
        | Some e ->
            cover "a range inside a registered one, elsewhere"
              (a <> e.start || n <> e.bytes);
            cover "a range equal to a registered one"
              (a = e.start && n = e.bytes);
            e.maps <- e.maps + 1;
            m.regions <- m.regions @ [ Counted e ];
            true
        | None when List.exists (shares (a, n)) m.entries ->
            cover "a range overlapping a registered one"
              (List.exists
                 (fun e -> a < e.start + e.bytes && e.start < a + n)
                 m.entries);
            cover "a range sharing only a page"
              (not
                 (List.exists
                    (fun e -> a < e.start + e.bytes && e.start < a + n)
                    m.entries));
            false
        | None ->
            let e = { start = a; bytes = n; maps = 1 } in
            m.entries <- e :: m.entries;
            m.regions <- m.regions @ [ Counted e ];
            true)

  let unmap m i =
    let r = List.nth m.regions i in
    m.regions <- List.filteri (fun j _ -> j <> i) m.regions;
    match r with
    | Uncounted -> ()
    | Counted e ->
        e.maps <- e.maps - 1;
        if e.maps = 0 then begin
          cover "the last unmap of a registration" true;
          m.entries <- List.filter (fun e' -> e' != e) m.entries
        end

  (* The system *)

  type sys = {
    g : C.t;
    base : nativeint;
    foreign : C.region;
    read_only : nativeint;
    mutable live : C.region list;
  }

  let at s a = Nativeint.add s.base (Nativeint.of_int a)

  let start () =
    let g = shared () in
    let foreign = Option.get (C.alloc g `Pinned (2 * S.page)) in
    {
      g;
      base = S.pages (arena * S.page);
      foreign;
      read_only = S.pages ~read_only:true S.page;
      live = [];
    }

  let release s =
    List.iter (C.unmap s.g) s.live;
    C.free s.g s.foreign;
    S.free_pages s.base (arena * S.page);
    S.free_pages s.read_only S.page

  let map_sys s (area, a, n) =
    let a =
      match area with
      | Arena -> at s a
      | Foreign -> Nativeint.add (host s.foreign) (Nativeint.of_int a)
      | Read_only -> Nativeint.add s.read_only (Nativeint.of_int a)
    in
    match C.map_host s.g a n with
    | Some r ->
        s.live <- s.live @ [ r ];
        true
    | None -> false

  let unmap_sys s i =
    C.unmap s.g (List.nth s.live i);
    s.live <- List.filteri (fun j _ -> j <> i) s.live

  (* CUDA holds every registered range locked, and no arena page that no
     registration shares. *)
  let invariant m s =
    List.iter
      (fun e ->
        equal bool
          ~msg:(strf "first byte of [%d, %d)" e.start (e.start + e.bytes))
          true
          (S.locked (at s e.start));
        equal bool
          ~msg:(strf "last byte of [%d, %d)" e.start (e.start + e.bytes))
          true
          (S.locked (at s (e.start + e.bytes - 1))))
      m.entries;
    for p = 0 to arena - 1 do
      if not (List.exists (shares (p * S.page, S.page)) m.entries) then
        equal bool ~msg:(strf "page %d" p) false (S.locked (at s (p * S.page)))
    done

  let range =
    let point = [ 0; 8; 2048; 4088; 4096; 4104; 8192; 12280 ] in
    let length = [ 8; 64; 2048; 4096; 4104; 8192 ] in
    let pp ppf (area, a, n) =
      Format.fprintf ppf "(%s, %d, %d)"
        (match area with
        | Arena -> "arena"
        | Foreign -> "foreign"
        | Read_only -> "read-only")
        a n
    in
    let scale x = x * S.page / 4096 in
    Gen.with_pp pp
      (Gen.frequency
         [
           ( 8,
             Gen.such_that
               (fun (_, a, n) -> a + n <= arena * S.page)
               (Gen.map
                  (fun (a, n) -> (Arena, scale a, scale n))
                  (Gen.pair (Gen.of_list point) (Gen.of_list length))) );
           ( 1,
             Gen.map
               (fun a -> (Foreign, scale a, 64))
               (Gen.of_list [ 0; 8; 4096 ]) );
           (1, Gen.constant (Read_only, 0, 64));
         ])
end

let registry =
  abstract "r" ~invariant:Registry.invariant ~release:Registry.release

let mapped =
  among int registry (fun m ->
      List.init (List.length m.Registry.regions) Fun.id)

let registry_commands =
  [
    command "start" (Gen.unit @-> makes registry) Registry.make Registry.start;
    command "map_host"
      (registry ^-> Registry.range @-> returns bool)
      Registry.map Registry.map_sys;
    command "unmap"
      (registry ^-> mapped ^-> returns unit)
      Registry.unmap Registry.unmap_sys;
  ]

(* Submissions: the order of values *)

module Order = struct
  (* A part copies [n] bytes of buffer [src] at [so] to buffer [dst] at [do_],
     on COPY:0 iff [copy], after the earlier parts [after] lists. A part with a
     [delay] is a fill that starts its copy that many nanoseconds late, so that
     work run out of order shows in the buffers. *)
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

  (* Part [i] runs before part [j], [i < j]: same queue, or [j] runs after a
     part that runs after [i]. *)
  let rec before ps i j =
    let pj = List.nth ps j in
    (List.nth ps i).copy = pj.copy
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
    let queues = List.sort_uniq compare (List.map (fun p -> p.copy) ps) in
    cover "a submission of no parts after one released on COPY"
      (ps = [] && m.last = Some true);
    cover "a submission on the queue that released the last"
      (queues <> [] && m.last = Some (List.hd (List.rev ps)).copy);
    cover "a switch of queue"
      (queues <> [] && m.last = Some (not (List.hd (List.rev ps)).copy));
    cover "two queues in one submission" (List.length queues = 2);
    cover "an after across queues"
      (List.exists
         (fun p ->
           List.exists (fun k -> (List.nth ps k).copy <> p.copy) p.after)
         ps);
    List.iter
      (fun p -> Bytes.blit m.bufs.(p.src) p.so m.bufs.(p.dst) p.do_ p.n)
      ps;
    m.last <- Some (match List.rev ps with p :: _ -> p.copy | [] -> false)

  let run m subs =
    cover "several values in flight" (List.length subs > 1);
    List.iter (run_one m) subs

  (* The system *)

  type sys = {
    g : C.t;
    regions : C.region array;
    flag : C.region;
    image : C.image;
    spin : int;
  }

  let value = ref 0

  let start () =
    let g = shared () in
    let regions =
      Array.init buffers (fun b ->
          let r = Option.get (C.alloc g `Device size) in
          S.write_gpu (C.handle r) (Bytes.to_string (initial b));
          r)
    in
    let flag = Option.get (C.alloc g `Pinned 8) in
    S.set64 (host flag) 0;
    let image, kernel = S.kernels g in
    { g; regions; flag; image; spin = kernel "spin" }

  let release s =
    Array.iter (C.free s.g) s.regions;
    C.free s.g s.flag;
    C.unload s.g s.image

  (* Submits [subs] back to back, then reads the word until it holds the last
     value: each read is at least the one before. *)
  let run_sys s subs =
    let fills = ref [] in
    let part p =
      let queue = if p.copy then "COPY:0" else "COMPUTE:0" in
      let after = Array.of_list p.after in
      let dst = s.regions.(p.dst) and src = s.regions.(p.src) in
      if p.delay = 0 then
        C.part s.g ~queue ~after (`Copy ((dst, p.do_), (src, p.so), p.n))
      else
        let f =
          S.delayed ~spin:s.spin ~flag:(address s.flag) ~ns:p.delay
            ~dst:(address dst + p.do_)
            ~src:(address src + p.so)
            p.n
        in
        fills := f :: !fills;
        S.part s.g ~queue ~after f
    in
    let first = !value + 1 in
    let hand ps =
      incr value;
      equal answer `Ok (submit s.g ~v:!value (Array.of_list (List.map part ps)))
    in
    List.iter hand subs;
    let rec watch seen =
      let w = C.signaled s.g in
      at_least int ~msg:"the word" ~than:seen w;
      at_most int ~msg:"the word" ~than:!value w;
      if w < !value then watch w
    in
    watch (first - 1);
    S.wait s.g !value;
    still s.g !value ~ms:1;
    (* The fills lived until their submissions returned. *)
    ignore (Sys.opaque_identity !fills)

  let invariant m s =
    Array.iteri
      (fun b r ->
        equal string ~msg:(strf "buffer %d" b)
          (Bytes.to_string m.bufs.(b))
          (S.read_gpu (C.handle r) size))
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
      { delay; copy; src; so; dst; do_; n; after }
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

let stateful =
  group ~timeout:300. "stateful"
    [
      stateful ~count:100 ~steps:20 "map_host shares a host range both ways"
        registry_commands;
      stateful ~count:100 ~steps:20
        "values complete in order and the word never moves backwards (sampled)"
        order_commands;
    ]

let () =
  exit
    (run "device_cuda"
       [ opening; facts; memory; work; images; timeline; two; stateful ])
