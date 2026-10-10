(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module A = Rig_amd
module S = Rig_amd_support
module E = S.Edge
module H = Rig_gpu_support.Host
module Abi = Rig_amd_abi
module Gpu = Abi.Gpu
module Pm4 = Abi.Pm4

type memory = Rig_edge.memory = Device | Pinned | Mapped

let strf = Printf.sprintf
let still = Rig_gpu_support.still
let host r = Option.get (A.locate r).host
let address r = Option.get (A.locate r).address
let word g = (A.facts g).word
let submit g ~v ps = E.submit g ~v ps

let le64 n =
  let b = Bytes.create 8 in
  Bytes.set_int64_le b 0 (Int64.of_int n);
  Bytes.to_string b

let le32s ns =
  let b = Bytes.create (4 * List.length ns) in
  List.iteri (fun i n -> Bytes.set_int32_le b (4 * i) (Int32.of_int n)) ns;
  Bytes.to_string b

let at a off = a + off

let answer =
  Testable.make
    ~pp:(fun ppf -> function
      | `Ok -> Format.pp_print_string ppf "`Ok"
      | `Failed why -> Format.fprintf ppf "`Failed %S" why)
    ~equal:( = )

let pp_room ppf = function
  | `Fits -> Format.pp_print_string ppf "`Fits"
  | `Later -> Format.pp_print_string ppf "`Later"
  | `Never -> Format.pp_print_string ppf "`Never"

let room_answer = Testable.make ~pp:pp_room ~equal:( = )

let gpu =
  Testable.make
    ~pp:(fun ppf g -> Format.pp_print_string ppf (Gpu.processor g))
    ~equal:( = )

(* GPUs *)

let r9700 =
  {
    Gpu.target = (12, 0, 1);
    gc = (12, 0, 1);
    sdma = (7, 0, 1);
    xccs = 1;
    shader_engines = 4;
    compute_units = 64;
    scratch_slots = 32;
  }

let gfx1100 =
  { r9700 with target = (11, 0, 0); gc = (11, 0, 0); sdma = (6, 0, 0) }

(* A GPU of eight dies, whose compute queue reads AQL packets. *)
let mi300 =
  {
    Gpu.target = (9, 4, 2);
    gc = (9, 4, 3);
    sdma = (4, 4, 2);
    xccs = 8;
    shader_engines = 4;
    compute_units = 38;
    scratch_slots = 32;
  }

let gfx90a =
  { mi300 with target = (9, 0, 10); gc = (9, 4, 2); sdma = (4, 4, 0); xccs = 1 }

let gfx1030 =
  { r9700 with target = (10, 3, 0); gc = (10, 3, 0); sdma = (5, 2, 0) }

let gpus =
  group ~timeout:10. "GPUs"
    [
      test "AMD's display controllers and accelerators are GPUs" (fun () ->
          equal bool ~msg:"display" true
            (A.is_gpu ~vendor:0x1002 ~class_:0x030000);
          equal bool ~msg:"accelerator" true
            (A.is_gpu ~vendor:0x1002 ~class_:0x120000);
          equal bool ~msg:"audio" false
            (A.is_gpu ~vendor:0x1002 ~class_:0x040300);
          equal bool ~msg:"bridge" false
            (A.is_gpu ~vendor:0x1002 ~class_:0x060400);
          equal bool ~msg:"another vendor" false
            (A.is_gpu ~vendor:0x10de ~class_:0x030000));
    ]

(* A path of host memory

   The suite's own path: its memory is the C heap, at GPU addresses equal to the
   host's, its queues keep where their rings and write positions are, and its
   HDP register is a word of host memory. Nothing runs the rings: the suite
   completes values by writing the timeline word itself, and reads what the
   device handed its queues. *)

module Host = struct
  type mem = { at : int; bytes : int; view : bool; mutable given : bool }

  type queue = {
    kind : [ `Pm4 | `Aql | `Sdma ];
    ring : int;
    bytes : int;
    write : int;
  }

  (* Any domain may call a path's functions at once: [lock] guards the
     fields. *)
  type t = {
    lock : Mutex.t;
    hdp : int;
    mutable calls : int; (* the allocations and queues asked for *)
    mutable live : int; (* the allocations not given back *)
    mutable allocated : int; (* the allocations made *)
    mutable last_gpu : int; (* the address of the last [`Gpu] allocation *)
    mutable frees : int list; (* the addresses given back, latest first *)
    mutable queue_memory : int list;
        (* what the device allocated before its first queue: rings, positions,
           word, slots and segment, which its queues read *)
    mutable queues : queue list; (* in the order made *)
    mutable refused_queue : bool;
    mutable stops : int;
    mutable stop_fault : string option; (* the fault the stop was given *)
    mutable sleeps : int;
    mutable report : string option; (* the fault its sleep reports *)
    mutable free_fault : string option; (* the fault its free raises *)
    mutable doorbells : int list;
    mutable held : mem list; (* the allocations not given back *)
    mutable closed : bool;
  }

  let key : mem Type.Id.t = Type.Id.make ()

  (* Each path is a GPU of its own. *)
  let gpus = Atomic.make 0

  let memory ~host at bytes ~view =
    {
      A.address = at;
      host = (if host then Some at else None);
      data = { at; bytes; view; given = false };
    }

  let path ?(key = key) ?(reaches = true) ?(gpu = r9700) ?(lds = 65536)
      ?(refuse = fun _ -> false) ?(fault = fun _ -> false)
      ?(on_alloc = fun _ _ -> ()) ?(stop = fun () -> `Stopped) () =
    let h =
      {
        lock = Mutex.create ();
        hdp = H.pages 4;
        calls = 0;
        live = 0;
        allocated = 0;
        last_gpu = 0;
        frees = [];
        queue_memory = [];
        queues = [];
        refused_queue = false;
        stops = 0;
        stop_fault = None;
        sleeps = 0;
        report = None;
        free_fault = None;
        doorbells = [];
        held = [];
        closed = false;
      }
    in
    let refused () =
      let i = h.calls in
      h.calls <- i + 1;
      if fault i then raise (A.Fault (strf "the host failed at call %d" i));
      refuse i
    in
    let alloc kind n =
      on_alloc kind n;
      Mutex.protect h.lock @@ fun () ->
      if refused () then None
      else begin
        let m = memory ~host:(kind <> `Gpu) (H.pages n) n ~view:false in
        h.live <- h.live + 1;
        h.allocated <- h.allocated + 1;
        if kind = `Gpu then h.last_gpu <- m.A.address;
        h.held <- m.data :: h.held;
        if h.queues = [] then h.queue_memory <- m.A.address :: h.queue_memory;
        Some m
      end
    in
    let view (m : mem A.memory) =
      { m with data = { m.data with view = true; given = false } }
    in
    let free (m : mem A.memory) =
      Mutex.protect h.lock @@ fun () ->
      Option.iter (fun why -> raise (A.Fault why)) h.free_fault;
      if not h.closed then begin
        if m.data.given then fail "the device gave the same memory back twice";
        if h.queues <> [] && h.stops = 0 && List.mem m.data.at h.queue_memory
        then
          fail
            "the device gave back memory its queues read before stopping them";
        m.data.given <- true;
        h.frees <- m.data.at :: h.frees;
        if not m.data.view then begin
          h.live <- h.live - 1;
          h.held <- List.filter (fun d -> d != m.data) h.held;
          H.free_pages m.data.at m.data.bytes
        end
      end
    in
    let queue kind ~ring ~bytes ~read:_ ~write =
      Mutex.protect h.lock @@ fun () ->
      if refused () then begin
        h.refused_queue <- true;
        Error "the host makes no queue"
      end
      else begin
        h.queues <- h.queues @ [ { kind; ring; bytes; write } ];
        let doorbell = H.pages 8 in
        h.doorbells <- doorbell :: h.doorbells;
        Ok doorbell
      end
    in
    let sleep ~ms:_ =
      h.sleeps <- h.sleeps + 1;
      Option.iter (fun why -> raise (A.Fault why)) h.report
    in
    let stop ~fault =
      h.stops <- h.stops + 1;
      h.stop_fault <- fault;
      stop ()
    in
    ( h,
      {
        A.key;
        index = Atomic.fetch_and_add gpus 1;
        gpu;
        waves = 32;
        lds;
        clock_hz = 100_000_000;
        mec = 0;
        wgps = Array.make (gpu.shader_engines * gpu.xccs) [| 0xff; 0xff |];
        budget = 1 lsl 34;
        alloc;
        map_host = Some (fun a n -> Some (memory ~host:true a n ~view:true));
        reaches = (fun _ -> reaches);
        map_peer = (fun m -> if reaches then Some (view m) else None);
        free;
        queue;
        hdp = Some h.hdp;
        interrupt = 1;
        hang_ms = None;
        sleep;
        stable_power = (fun () -> Ok ());
        stop;
      } )

  (* Gives back the host memory the path holds, once its device stopped: its
     own, and what the device kept, such as its timeline word. *)
  let close h =
    H.free_pages h.hdp 4;
    List.iter (fun d -> H.free_pages d 8) h.doorbells;
    List.iter (fun d -> H.free_pages d.at d.bytes) h.held;
    h.doorbells <- [];
    h.held <- [];
    h.closed <- true

  let device ?key ?reaches ?gpu ?lds ?on_alloc ?stop () =
    let h, p = path ?key ?reaches ?gpu ?lds ?on_alloc ?stop () in
    match A.make p with Ok g -> (h, g) | Error why -> fail why

  let with_device ?gpu ?lds f =
    let h, g = device ?gpu ?lds () in
    Fun.protect
      ~finally:(fun () ->
        A.stop g ~fault:None;
        close h)
      (fun () -> f h g)

  let compute h = List.nth h.queues 0
  let copy h = List.nth h.queues 1

  (* The position a queue reads its work up to: in words on a PM4 ring, bytes on
     an SDMA ring, packets on an AQL ring. *)
  let position q = H.get64 q.write

  (* Reaches [v]: what the queue that releases [v] does. *)
  let reach g v = H.set64 (host (word g)) v
end

(* Paths *)

let make_refuses_families () =
  List.iter
    (fun g ->
      let _, p = Host.path ~gpu:g () in
      match A.make p with
      | Ok _ -> failf "a device of a %s" (Gpu.processor g)
      | Error why -> contains ~sub:(Gpu.processor g) why)
    [ gfx90a; gfx1030 ]

(* Each allocation or queue a successful [make] asks for, refused in turn. *)
(* Each allocation or queue a successful [make] asks for, refused in turn: the
   path answers its refusal ([None], [Error]) or raises [Fault]. A failed
   [make] stops the queues the path made, then gives back every memory the
   path gave, unless the path cannot say its queues stopped, when a queue may
   still read the memory. *)
let make_gives_back () =
  let calls =
    let h, g = Host.device () in
    A.stop g ~fault:None;
    Host.close h;
    h.calls
  in
  let case how stop i =
    let refuse, fault =
      match how with
      | `Refuse -> (( = ) i, fun _ -> false)
      | `Fault -> ((fun _ -> false), ( = ) i)
    in
    let h, p = Host.path ~refuse ~fault ~stop:(fun () -> stop) () in
    let msg =
      strf "call %d of %d %s, stop %s" i calls
        (match how with `Refuse -> "refused" | `Fault -> "faulted")
        (match stop with `Stopped -> "Stopped" | `Unknown -> "Unknown")
    in
    let answer =
      match A.make p with
      | Ok _ -> `Device
      | Error why -> `Error why
      | exception A.Fault why -> `Fault why
    in
    (match (how, answer) with
    | _, `Device -> failf "%s: a device" msg
    | `Fault, `Error why -> failf "%s: Error %S, expected the Fault" msg why
    | `Refuse, `Error why ->
        if h.refused_queue then contains ~msg ~sub:"the host makes no queue" why
    | `Refuse, `Fault why -> failf "%s: Fault %S" msg why
    | `Fault, `Fault why ->
        equal string ~msg (strf "the host failed at call %d" i) why);
    let queued = h.queues <> [] in
    equal int ~msg:(msg ^ ": stops") (if queued then 1 else 0) h.stops;
    if queued && stop = `Unknown then
      equal int ~msg:(msg ^ ": allocations kept") h.allocated h.live
    else equal int ~msg:(msg ^ ": allocations left") 0 h.live;
    Host.close h
  in
  List.iter
    (fun how ->
      List.iter
        (fun stop ->
          for i = 0 to calls - 1 do
            case how stop i
          done)
        [ `Stopped; `Unknown ])
    [ `Refuse; `Fault ]

(* [stop] gives the device's memory back only once the path stopped its queues
   (the host path refuses a free of memory a queue reads before), and never the
   timeline word, which its free after the stop gives back. *)
let stop_gives_back () =
  let h, g = Host.device () in
  let at = address (word g) in
  equal answer ~msg:"submit" `Ok (submit g ~v:1 [||]);
  A.stop g ~fault:None;
  equal int ~msg:"stops" 1 h.stops;
  equal (option string) ~msg:"the fault the path's stop got" None h.stop_fault;
  equal bool ~msg:"the word given back" false (List.mem at h.frees);
  equal int ~msg:"the memory left: the word" 1 h.live;
  A.free g (word g);
  equal bool ~msg:"the word, freed after the stop" true (List.mem at h.frees);
  Host.close h

let facts () =
  List.iter
    (fun (g, kind, aql) ->
      let msg = Gpu.processor g in
      Host.with_device ~gpu:g @@ fun h d ->
      let f = A.facts d and c = A.capability d in
      let runs =
        List.map (function
          | Rig_edge.Words -> "words"
          | Fill -> "fill"
          | Copy -> "copy"
          | Launch -> "launch")
      in
      equal string ~msg (Gpu.processor g) f.arch;
      equal int ~msg (1 lsl 34) f.budget;
      equal
        (list (pair string (list string)))
        ~msg
        [
          ("COMPUTE:0", [ "words"; "fill"; "launch" ]);
          ("COPY:0", [ "words"; "fill"; "copy" ]);
        ]
        (List.map (fun (q : Rig_edge.queue) -> (q.name, runs q.runs)) f.queues);
      equal bool ~msg:"completion is the word" true (f.completion = Store);
      equal bool ~msg:"may block" false f.may_block;
      equal bool ~msg:"maps host memory" true f.maps_host;
      equal bool ~msg:"waits on objects" false f.waits.objects;
      equal bool ~msg:"waits on words as on the host" f.waits.hosts
        f.waits.stores;
      equal int ~msg:"most waits" 255 f.waits.most;
      equal gpu ~msg c.gpu g;
      equal int ~msg 100_000_000 c.clock_hz;
      equal bool ~msg:"AQL" aql
        (match c.compute with Aql _ -> true | Pm4 -> false);
      equal
        (array (array int))
        ~msg
        (Array.make (g.shader_engines * g.xccs) [| 0xff; 0xff |])
        c.wgps;
      equal (list string) ~msg:"queues the path made" [ kind; "SDMA" ]
        (List.map
           (fun (q : Host.queue) ->
             match q.kind with `Pm4 -> "PM4" | `Aql -> "AQL" | `Sdma -> "SDMA")
           h.queues);
      match f.capability with
      | Capability (k, c') -> (
          match Type.Id.provably_equal k Abi.Capability.key with
          | Some Equal ->
              equal bool ~msg:"the capability's record" true (c' == c)
          | None -> fail "the capability is under another key"))
    [ (r9700, "PM4", false); (mi300, "AQL", true) ]

(* [Mapped] memory needs the HDP register that flushes the host's writes into
   it: a path with none has no [Mapped] memory. *)
let mapped_needs_hdp () =
  let h, p = Host.path () in
  match A.make { p with hdp = None } with
  | Error why -> fail why
  | Ok g ->
      equal bool ~msg:"Mapped is None" true
        (Option.is_none (A.alloc g Mapped 64));
      A.stop g ~fault:None;
      Host.close h

(* [peer g g'] is whether [map_peer g g'] maps [g']'s [`Device] memory. *)
let peers () =
  let other : Host.mem Type.Id.t = Type.Id.make () in
  let case name ?key ~reaches expected =
    let h, g = Host.device ~reaches () and h', g' = Host.device ?key () in
    let r = Option.get (A.alloc g' Device 64) in
    equal bool ~msg:(name ^ ": peer") expected (A.peer g g');
    let view = A.map_peer g g' r in
    equal bool ~msg:(name ^ ": map_peer") expected (Option.is_some view);
    Option.iter (A.free g) view;
    A.free g' r;
    List.iter
      (fun (h, g) ->
        A.stop g ~fault:None;
        Host.close h)
      [ (h, g); (h', g') ]
  in
  case "a path that reaches the GPU" ~reaches:true true;
  case "a path that does not" ~reaches:false false;
  case "two paths" ~key:other ~reaches:true false

let paths =
  group ~timeout:30. "paths"
    [
      test "peer is whether map_peer maps the other's Device memory" peers;
      test "make refuses a GPU of a family it does not drive, naming it"
        make_refuses_families;
      test "make raises on a release interrupt context of 0" (fun () ->
          let h, p = Host.path () in
          raises_match (Exn.invalid_arg ~substring:"Rig_amd.make") (fun () ->
              A.make { p with interrupt = 0 });
          Host.close h);
      test "a failed make gives back what it took" make_gives_back;
      test "stop gives back a device's memory once its queues stopped"
        stop_gives_back;
      test "a device states its path's facts" facts;
      test "a path with no HDP register has no Mapped memory" mapped_needs_hdp;
    ]

(* Code objects *)

let read_fixture name =
  In_channel.with_open_bin ("fixtures/" ^ name) In_channel.input_all

let kernels_bin = lazy (read_fixture "kernels_gfx1201.hsaco")
let work_bin = lazy (read_fixture "work_gfx1201.hsaco")

(* [bin] laid over new [`Device] memory of [g]: the image, its memory and the
   bytes to copy there. *)
let load g bin =
  match A.image g bin with
  | Error why -> fail why
  | Ok (Rig_edge.Loaded _) -> fail "an image the device placed itself"
  | Ok (Rig_edge.Place (n, lay)) ->
      let r = Option.get (A.alloc g Device n) in
      let m, bytes = lay r in
      (m, r, bytes)

(* Misuse *)

let misuse () =
  Host.with_device @@ fun _ g ->
  let raises fn f =
    raises_match ~msg:fn (Exn.invalid_arg ~substring:("Rig_amd." ^ fn ^ ":")) f
  in
  let page = H.pages 4096 in
  raises "alloc" (fun () -> A.alloc g Pinned 0);
  raises "map_host" (fun () -> A.map_host g page 0);
  H.free_pages page 4096

(* What rig_amd_room answers for a part the device does not run. *)
let c_room () =
  let never = room_answer in
  Host.with_device @@ fun _ g ->
  let none = `Words 0 in
  equal never ~msg:"a part of 16 words" `Fits
    (E.room g [| E.raw ~queue:0 ~work:(`Words 16) () |]);
  equal never ~msg:"a copy on the compute queue" `Never
    (E.room g [| E.raw ~queue:0 ~work:(`Copy 64) () |]);
  equal never ~msg:"a part of no kind" `Never (E.room g [| E.raw ~queue:0 () |]);
  equal never ~msg:"after its own index" `Never
    (E.room g [| E.raw ~queue:1 ~work:none ~after:[| 0 |] () |]);
  equal never ~msg:"a queue of no index" `Never
    (E.room g [| E.raw ~queue:2 ~work:none () |]);
  equal never ~msg:"a negative queue" `Never
    (E.room g [| E.raw ~queue:(-1) ~work:none () |]);
  Host.with_device ~gpu:mi300 @@ fun _ g ->
  equal never ~msg:"AQL: a packet" `Fits
    (E.room g [| E.raw ~queue:0 ~work:(`Words 16) () |]);
  equal never ~msg:"AQL: part of a packet" `Never
    (E.room g [| E.raw ~queue:0 ~work:(`Words 15) () |])

let misuse =
  group ~timeout:30. "misuse"
    [
      test "each misuse the interface states raises" misuse;
      test "the C room refuses what part refuses" c_room;
    ]

(* Images *)

let loads () =
  Host.with_device @@ fun _ g ->
  let bin = Lazy.force kernels_bin in
  let co = Result.get_ok (Abi.Code_object.of_string bin) in
  (match A.image g bin with
  | Ok (Rig_edge.Place (n, _)) ->
      equal int ~msg:"the image's size" (Abi.Code_object.size co) n
  | Ok (Rig_edge.Loaded _) | Error _ -> fail "no image to place");
  match load g bin with
  | m, r, bytes ->
      at_most int ~msg:"the image's bytes" ~than:(Abi.Code_object.size co)
        (String.length bytes);
      List.iter
        (fun name ->
          let k = Option.get (Abi.Code_object.kernel co name) in
          equal (option int) ~msg:name
            (Some (address r + k.descriptor))
            (Option.map (fun (e : Rig_edge.entry) -> e.code) (A.entry m name)))
        [ "empty"; "double_index"; "spin"; "wild" ];
      equal (option int) ~msg:"no kernel" None
        (Option.map (fun (e : Rig_edge.entry) -> e.code) (A.entry m "nothing"));
      equal (option int) ~msg:"a symbol of no kernel" None
        (Option.map
           (fun (e : Rig_edge.entry) -> e.code)
           (A.entry m "__clang_ocl_kern_imp_spin"));
      A.unload g m;
      A.free g r

let refusals () =
  let result =
    Testable.make
      ~pp:(fun ppf -> function
        | Ok _ -> Format.pp_print_string ppf "Ok _"
        | Error e -> Format.fprintf ppf "Error %S" e)
      ~equal:(fun a b ->
        match (a, b) with
        | Error a, Error b -> a = b
        | Ok _, Ok _ -> true
        | _ -> false)
  in
  let image ?gpu ?lds bin =
    Host.with_device ?gpu ?lds @@ fun _ g -> Result.map ignore (A.image g bin)
  in
  let work = Lazy.force work_bin in
  equal result ~msg:"another processor"
    (Error "a code object for gfx1201; the GPU is gfx1100")
    (image ~gpu:gfx1100 (Lazy.force kernels_bin));
  equal result ~msg:"not a code object"
    (Result.map ignore (Abi.Code_object.of_string "\127ELF but not one"))
    (image "\127ELF but not one");
  equal result ~msg:"a kernel's local data share, the GPU's" (Ok ())
    (image ~lds:256 work);
  equal result ~msg:"a kernel's local data share, past the GPU's"
    (Error "kernel shared takes 256 bytes of local data share; the GPU has 255")
    (image ~lds:255 work)

let images =
  group ~timeout:30. "images"
    [
      test "an image names its kernels' descriptors in its code" loads;
      test "an image refuses what the GPU cannot run, saying why" refusals;
    ]

(* Launches

   A launch's dispatch is its function's template, made at entry, filled at the
   hand-over: rig_amd_ring.c places its arguments in the segment and its
   dispatch on the compute ring, which the host path's device lets the suite
   read. *)

let launch_bin = lazy (read_fixture "launch_gfx1201.hsaco")
let launch_bin_942 = lazy (read_fixture "launch_gfx942.hsaco")

let launch_kernel ?(bin = launch_bin) name =
  let co = Result.get_ok (Abi.Code_object.of_string (Lazy.force bin)) in
  Option.get (Abi.Code_object.kernel co name)

(* The bytes of [ids]'s parameters: [out]'s offset into its buffer, [a], [b] and
   [f]. *)
let ids_params ~out ~a ~b ~f =
  let p = Bytes.create 24 in
  Bytes.set_int64_le p 0 (Int64.of_int out);
  Bytes.set_int64_le p 8 (Int64.of_int a);
  Bytes.set_int32_le p 16 (Int32.of_int b);
  Bytes.set_int32_le p 20 (Int32.bits_of_float f);
  Bytes.to_string p

(* What [entry] makes of a kernel: a launch, once, and the scratch of its
   private segment; and what it refuses. A GPU of several dies launches a
   kernel that reads its dispatch packet: the packet is the one its queue
   reads. *)
let entries () =
  (Host.with_device ~gpu:mi300 @@ fun h g ->
   let m, r, _ = load g (Lazy.force launch_bin_942) in
   not_equal nativeint ~msg:"an AQL queue's launch" 0n
     (Option.get (A.entry m "ids")).launch;
   not_equal nativeint ~msg:"AQL: a kernel that reads its dispatch packet" 0n
     (Option.get (A.entry m "packet")).launch;
   let before = h.allocated in
   ignore (A.entry m "scratch");
   equal int ~msg:"AQL: the scratch, made at entry" (before + 1) h.allocated;
   A.unload g m;
   A.free g r);
  Host.with_device @@ fun h g ->
  let m, r, _ = load g (Lazy.force launch_bin) in
  let e = Option.get (A.entry m "ids") in
  not_equal nativeint ~msg:"a PM4 queue's launch" 0n e.launch;
  equal nativeint ~msg:"made once" e.launch
    (Option.get (A.entry m "ids")).launch;
  let before = h.allocated in
  raises_match ~msg:"a kernel with scratch that reads its dispatch packet"
    (Exn.invalid_arg ~substring:"dispatch packet") (fun () ->
      A.entry m "packet_scratch");
  equal int ~msg:"no scratch for a kernel entry refuses" before h.allocated;
  ignore (A.entry m "scratch");
  equal int ~msg:"the scratch, made at entry" (before + 1) h.allocated;
  ignore (A.entry m "scratch");
  equal int ~msg:"the scratch, made once" (before + 1) h.allocated;
  raises_match ~msg:"a kernel that reads its dispatch packet"
    (Exn.invalid_arg ~substring:"dispatch packet") (fun () ->
      A.entry m "packet");
  A.unload g m;
  A.free g r;
  let m, r, _ = load g (read_fixture "../abi/fixtures/hidden_gfx1201.hsaco") in
  raises_match ~msg:"a kernel that reads a runtime's service"
    (Exn.invalid_arg ~substring:"hidden_printf_buffer") (fun () ->
      A.entry m "every");
  A.unload g m;
  A.free g r

(* The words of [q]'s ring from position [p] to [p'], a PM4 ring's positions
   being words. *)
let handed (q : Host.queue) p p' =
  let size = q.bytes / 4 in
  List.init (p' - p) (fun i -> H.get32 (at q.ring (4 * ((p + i) mod size))))

(* The [n] words of [ys] from the first index that holds [xs], [n] its
   length, or [ys] if none does: what a test compares with [xs]. *)
let window ys xs =
  let n = List.length xs and a = Array.of_list ys in
  let rec at i =
    if i + n > Array.length a then ys
    else
      let w = Array.to_list (Array.sub a i n) in
      if w = xs then w else at (i + 1)
  in
  at 0

(* A launch of [ids] over 3 x 2 groups of 64 x 2 work-items: its parameters,
   each ref's slot address added, then the implicit arguments its function reads
   go in the segment, and its dispatch, of those arguments, on the compute ring:
   PM4 words on a GPU of one die, an AQL packet on a GPU of several, whose
   ring's positions count packets of 16 words. The arguments lie at the
   segment's start, the third memory the device takes of its path, before its
   queues; on an AQL queue, where the writer's own PM4 words go in the segment
   first, at the address the kernel dispatch packet names, words 10 and 11 of
   the packet of type 2 (hsa.h's HSA_PACKET_TYPE_KERNEL_DISPATCH). *)
let launch_hand_over (gpu, bin) () =
  Host.with_device ~gpu @@ fun h g ->
  let m, r, _ = load g (Lazy.force bin) in
  let k = launch_kernel ~bin "ids" in
  let e = Option.get (A.entry m "ids") in
  let slot = 0x7000_0000 in
  let params = ids_params ~out:0x40 ~a:5 ~b:7 ~f:1.5 in
  let q = Host.compute h in
  let words = if gpu.xccs > 1 then 16 else 1 in
  let p0 = Host.position q * words in
  equal answer `Ok
    (E.submit g ~v:1 ~slots:[| slot |]
       [| E.launch e ~groups:(3, 2, 1) ~threads:(64, 2, 1) params [ (0, 0) ] |]);
  let ring = handed q p0 (Host.position q * words) in
  let segment =
    if gpu.xccs = 1 then List.nth (List.rev h.queue_memory) 2
    else
      let ws = Array.of_list ring in
      let rec packet i =
        if ws.(i) land 0xff = 2 then ws.(i + 10) lor (ws.(i + 11) lsl 32)
        else packet (i + 16)
      in
      packet 0
  in
  let args = H.read segment 96 in
  let u32 at = Int32.to_int (String.get_int32_le args at) land 0xffff_ffff in
  let u16 at = String.get_uint16_le args at in
  equal int ~msg:"out, its slot's address added" (slot + 0x40)
    (Int64.to_int (String.get_int64_le args 0));
  equal string ~msg:"a, b and f" (String.sub params 8 16) (String.sub args 8 16);
  equal (list int) ~msg:"its groups" [ 3; 2; 1 ] [ u32 24; u32 28; u32 32 ];
  equal (list int) ~msg:"its threads per group" [ 64; 2; 1 ]
    [ u16 36; u16 38; u16 40 ];
  equal (list int) ~msg:"no partial group" [ 0; 0; 0 ]
    [ u16 42; u16 44; u16 46 ];
  equal string ~msg:"the first work-item at 0" (String.make 24 '\000')
    (String.sub args 64 24);
  equal int ~msg:"two axes" 2 (u16 88);
  let base = e.code - k.descriptor in
  let dispatch =
    A.dispatch gpu k ~base ~lds:65536
      [| segment; 0; 64; 2; 1; 3; 2; 1 |]
      ~shared:0
  in
  let ws =
    List.init
      (String.length dispatch / 4)
      (fun i ->
        Int32.to_int (String.get_int32_le dispatch (4 * i)) land 0xffff_ffff)
  in
  equal (list int) ~msg:"its dispatch, on the compute ring" ws (window ring ws);
  Host.reach g 1;
  A.unload g m;
  A.free g r

(* What the room check answers for a launch: a launch fits within its function's
   256 work-items per group and the GPU's LDS less the 256 bytes [lds] takes
   itself; it never fits past them, with an empty axis, or with fewer
   parameter bytes than its function reads. *)
let launch_room () =
  Host.with_device @@ fun _ g ->
  let m, r, _ = load g (Lazy.force launch_bin) in
  let ids = Option.get (A.entry m "ids")
  and lds = Option.get (A.entry m "lds") in
  let params = ids_params ~out:0 ~a:0 ~b:0 ~f:0. in
  let room ?(e = ids) ?shared groups threads =
    E.room g [| E.launch e ~groups ~threads ?shared params [ (0, 0) ] |]
  in
  equal room_answer ~msg:"256 work-items" `Fits (room (1, 1, 1) (64, 2, 2));
  equal room_answer ~msg:"16 of the 24 bytes of ids's parameters" `Never
    (E.room g
       [|
         E.launch ids ~groups:(1, 1, 1) ~threads:(64, 1, 1)
           (String.sub params 0 16) [ (0, 0) ];
       |]);
  equal room_answer ~msg:"257 work-items" `Never (room (1, 1, 1) (257, 1, 1));
  equal room_answer ~msg:"an empty grid" `Never (room (1, 0, 1) (64, 1, 1));
  equal room_answer ~msg:"an empty group" `Never (room (1, 1, 1) (64, 1, 0));
  equal room_answer ~msg:"the GPU's LDS" `Fits
    (room ~e:lds ~shared:(65536 - 256) (1, 1, 1) (64, 1, 1));
  equal room_answer ~msg:"past the GPU's LDS" `Never
    (room ~e:lds ~shared:(65536 - 255) (1, 1, 1) (64, 1, 1));
  A.unload g m;
  A.free g r;
  (* An AQL packet's grid counts work-items in 32 bits. *)
  Host.with_device ~gpu:mi300 @@ fun _ g ->
  let m, r, _ = load g (Lazy.force launch_bin_942) in
  let ids = Option.get (A.entry m "ids") in
  let room groups threads =
    E.room g [| E.launch ids ~groups ~threads params [ (0, 0) ] |]
  in
  equal room_answer ~msg:"AQL: 2^32 - 256 work-items along x" `Fits
    (room (0xff_ffff, 1, 1) (256, 1, 1));
  equal room_answer ~msg:"AQL: 2^32 work-items along x" `Never
    (room (0x100_0000, 1, 1) (256, 1, 1));
  equal room_answer ~msg:"AQL: 2^32 work-items along z" `Never
    (room (1, 1, 0x100_0000) (1, 1, 256));
  A.unload g m;
  A.free g r

let launches =
  group ~timeout:30. "launches"
    [
      test
        "entry makes a launch once, with its scratch, and refuses what a \
         launch cannot write"
        entries;
      test "a launch hands over its arguments and its PM4 dispatch"
        (launch_hand_over (r9700, launch_bin));
      test "a launch hands over its arguments and its AQL packet"
        (launch_hand_over (mi300, launch_bin_942));
      test "a launch fits within its function's and the GPU's limits"
        launch_room;
    ]

(* Room

   [room] answers [`Later] only while one of the device's values is unreached:
   once the word holds the last value, it answers what it answers on a device
   that ran nothing. The rings' read positions are never written here, so an
   answer drawn from them would be [`Later] for ever. *)

type work = Words of int | Fill of int * int | Copy of int
type spec = { compute : bool; work : work; after : int list }
type step = Submit of spec list | Burst of int | Reach of int

let pp_spec ppf s =
  Format.fprintf ppf "{%s %s after [%s]}"
    (if s.compute then "COMPUTE" else "COPY")
    (match s.work with
    | Words n -> strf "words %d" n
    | Fill (u, b) -> strf "fill %d units %d bytes" u b
    | Copy n -> strf "copy %d" n)
    (String.concat ";" (List.map string_of_int s.after))

let pp_step ppf = function
  | Submit ss ->
      Format.fprintf ppf "submit [%a]"
        (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_spec)
        ss
  | Burst n -> Format.fprintf ppf "%d empty submissions" n
  | Reach p -> Format.fprintf ppf "reach %d%%" p

let zeros = lazy (Array.make (1 lsl 21) 0)

(* Fills that place nothing and take [bytes] of the segment, made once: every
   device's capability has the same C functions. *)
let fills = Hashtbl.create 8

let fill_taking g bytes =
  match Hashtbl.find_opt fills bytes with
  | Some f -> f
  | None ->
      let f = S.fill (A.capability g) [||] ~bytes in
      Hashtbl.add fills bytes f;
      f

(* [specs] as parts of [g], whose copies run between [src] and [dst]. *)
let parts g (src, dst) specs =
  let part s =
    let after = Array.of_list s.after in
    let queue = if s.compute then "COMPUTE:0" else "COPY:0" in
    match s.work with
    | Words n -> E.words ~queue ~after (Array.sub (Lazy.force zeros) 0 n)
    | Fill (units, bytes) ->
        E.fill ~queue ~after (fill_taking g bytes) ~units ~bytes
    | Copy n -> E.copy ~after ~dst:(address dst) ~src:(address src) n
  in
  Array.of_list (List.map part specs)

let spec =
  let open Gen in
  let* compute = bool in
  let* work =
    if compute then
      frequency
        [
          (* Half the compute ring in one part: programs wrap the ring often
             enough that each run's 40 do, even its first, short ones. *)
          ( 3,
            map
              (fun n -> Words n)
              (of_list [ 1; 64; 1 lsl 16; 1 lsl 20; 1 lsl 21 ]) );
          ( 2,
            let+ u = of_list [ 0; 64; 1 lsl 18 ]
            and+ b = of_list [ 0; 64; 4096; 1 lsl 17 ] in
            Fill (u, b) );
        ]
    else
      frequency
        [
          (4, map (fun n -> Words n) (of_list [ 1; 64; 1 lsl 19; 1 lsl 20 ]));
          (1, map (fun n -> Copy n) (of_list [ 1; 4096 ]));
          ( 1,
            let+ u = of_list [ 0; 64 ] and+ b = of_list [ 0; 4096; 1 lsl 17 ] in
            Fill (u, b) );
        ]
  in
  let compute = compute && match work with Copy _ -> false | _ -> true in
  let+ after = list ~size:(int_range 0 2) (int_range 0 3) in
  { compute; work; after }

let program =
  let open Gen in
  let submission =
    let+ ss = list ~size:(int_range 0 4) spec in
    Submit
      (List.mapi
         (fun i s ->
           {
             s with
             after =
               List.sort_uniq compare (List.filter (fun k -> k < i) s.after);
           })
         ss)
  in
  let step =
    frequency
      [
        (6, submission);
        (1, map (fun n -> Burst n) (int_range 1 6000));
        (2, map (fun p -> Reach p) (of_list [ 0; 50; 100 ]));
      ]
  in
  with_pp
    (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_step)
    (list ~size:(int_range 1 30) step)

let regions g =
  (Option.get (A.alloc g Pinned 4096), Option.get (A.alloc g Pinned 4096))

(* The fresh device: one that ran nothing, which the law compares with. *)
let fresh =
  fixture
    ~teardown:(fun (h, g, _) ->
      A.stop g ~fault:None;
      Host.close h)
    (fun () ->
      let h, g = Host.device () in
      (h, g, regions g))

let room_from_the_word steps =
  let _, f, fr = fresh () in
  Host.with_device @@ fun h g ->
  let rs = regions g in
  let last = ref 0 and word = ref 0 in
  let reach v =
    word := v;
    Host.reach g v
  in
  let one specs =
    let ps = parts g rs specs in
    let reference () = E.room f (parts f fr specs) in
    let a = E.room g ps in
    if !word = !last then
      equal room_answer ~msg:"every value reached" (reference ()) a;
    let a =
      if a <> `Later then a
      else begin
        cover "Later while a value is unreached" true;
        reach !last;
        let a = E.room g ps in
        equal room_answer ~msg:"once the word reached the last value"
          (reference ()) a;
        a
      end
    in
    if a = `Never then cover "Never" true;
    if a = `Fits then begin
      incr last;
      equal answer ~msg:"submit" `Ok (submit g ~v:!last ps)
    end
  in
  List.iter
    (function
      | Submit specs -> one specs
      | Burst n ->
          for _ = 1 to n do
            one []
          done
      | Reach p -> reach (!word + ((!last - !word) * p / 100)))
    steps;
  let wrapped (q : Host.queue) unit = Host.position q * unit > q.bytes in
  cover "the compute ring wrapped" (wrapped (Host.compute h) 4);
  cover "the copy ring wrapped" (wrapped (Host.copy h) 1)

let rank = function `Fits -> 0 | `Later -> 1 | `Never -> 2

(* Fills declaring units or bytes at their extremes, on either queue. *)
let declared =
  let open Gen in
  let amount =
    of_list ~pp:Format.pp_print_int
      [ 0; 1; 1 lsl 20; 1 lsl 22; max_int / 4; max_int / 2; max_int ]
  in
  let+ compute = bool and+ units = amount and+ bytes = amount in
  (compute, units, bytes)

let pp_declared ppf (c, u, b) =
  Format.fprintf ppf "%s %d units %d bytes"
    (if c then "COMPUTE" else "COPY")
    u b

let more =
  Gen.with_pp
    (fun ppf (ds, d) ->
      Format.fprintf ppf "[%a] then %a"
        (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_declared)
        ds pp_declared d)
    Gen.(pair (list ~size:(int_range 0 4) declared) declared)

let grows_with_declarations (ds, d) =
  let _, f, fr = fresh () in
  let room ds =
    E.room f
      (parts f fr
         (List.map
            (fun (compute, u, b) -> { compute; work = Fill (u, b); after = [] })
            ds))
  in
  let before = room ds and alone = room [ d ] and after = room (ds @ [ d ]) in
  at_least int ~msg:"with one part more" ~than:(rank before) (rank after);
  at_least int ~msg:"with parts before it" ~than:(rank alone) (rank after)

let parts_bound () =
  Host.with_device @@ fun _ g ->
  let empty n = Array.init n (fun _ -> E.words ~queue:"COPY:0" [||]) in
  equal room_answer ~msg:"512 parts" `Fits (E.room g (empty 512));
  equal room_answer ~msg:"513 parts" `Never (E.room g (empty 513))

let overflow =
  ( [ (true, max_int, 0); (true, max_int, 0); (true, max_int, 0) ],
    (true, max_int, 0) )

let room =
  group ~timeout:120. "room"
    [
      prop ~count:40 "room is Later only while a value is unreached" program
        room_from_the_word;
      prop ~count:200 ~examples:[ overflow ]
        "room never fits parts that declare more than parts it refused" more
        grows_with_declarations;
      test "room is Never past 512 parts" parts_bound;
    ]

(* The host data path

   Host writes through the BAR reach work submitted after them: [submit] flushes
   the HDP register of every [`Mapped] region its device's work may read, its
   own and those of the views it holds, before it rings a doorbell. Three
   devices of one path, each with its own register. *)

let flushers = 3

(* The regions the devices hold: whose memory each is, and which device holds
   it. *)
type machine = { mutable regions : held list }

and held = {
  machine : machine;
  owner : int;
  holder : int;
  mapped : bool;
  mutable live : bool;
}

type sys_machine = (Host.t * A.t * int ref) array

type sys_held = {
  devices : sys_machine;
  dev : int;
  r : A.region;
  mutable freed : bool;
}

let device_at (ds : sys_machine) d =
  let _, g, _ = ds.(d) in
  g

let give_back s =
  let g = device_at s.devices s.dev in
  s.freed <- true;
  A.free g s.r

let machine =
  abstract "machine" ~release:(fun (ds : sys_machine) ->
      Array.iter
        (fun (h, g, _) ->
          A.stop g ~fault:None;
          Host.close h)
        ds)

let held =
  abstract "r"
    ~pp:(fun ppf r ->
      Format.fprintf ppf "%s memory of %d held by %d%s"
        (if r.mapped then "Mapped" else "Pinned")
        r.owner r.holder
        (if r.live then "" else ", given back"))
    ~release:(fun s -> if not s.freed then give_back s)

let device_index = Gen.int_range 0 (flushers - 1)

let hold m r =
  m.regions <- r :: m.regions;
  r

(* The registers [d]'s submissions flush, in order. *)
let flushes m d =
  List.sort_uniq compare
    (List.filter_map
       (fun r ->
         if r.live && r.mapped && r.holder = d then Some r.owner else None)
       m.regions)

let flushed (ds : sys_machine) d =
  Array.iter (fun ((h : Host.t), _, _) -> H.set32 h.hdp 0xffff_ffff) ds;
  let _, g, last = ds.(d) in
  equal room_answer ~msg:"room" `Fits (E.room g [||]);
  incr last;
  equal answer ~msg:"submit" `Ok (submit g ~v:!last [||]);
  Host.reach g !last;
  List.filter
    (fun i ->
      let (h : Host.t), _, _ = ds.(i) in
      H.get32 h.hdp = 0)
    (List.init flushers Fun.id)

let hdp_commands =
  [
    command "start"
      (Gen.unit @-> makes machine)
      (fun () -> { regions = [] })
      (fun () ->
        Array.init flushers (fun _ ->
            let h, g = Host.device () in
            (h, g, ref 0)));
    command "alloc"
      (machine ^-> device_index @-> Gen.bool @-> makes held)
      (fun m d mapped ->
        hold m { machine = m; owner = d; holder = d; mapped; live = true })
      (fun ds d mapped ->
        let kind = if mapped then Mapped else Pinned in
        let r = Option.get (A.alloc (device_at ds d) kind 64) in
        { devices = ds; dev = d; r; freed = false });
    command "map_peer"
      ~pre:(fun d r -> r.live && d <> r.holder)
      (device_index @-> held ^-> makes held)
      (fun d r -> hold r.machine { r with holder = d; live = true })
      (fun d s ->
        let ds = s.devices in
        match A.map_peer (device_at ds d) (device_at ds s.dev) s.r with
        | Some r -> { s with dev = d; r; freed = false }
        | None -> fail "a view refused");
    command "give back"
      ~pre:(fun r -> r.live)
      (held ^-> returns unit)
      (fun r -> r.live <- false)
      give_back;
    command "submit"
      (machine ^-> device_index @-> returns (list int))
      flushes flushed;
  ]

(* A device of nine GPUs' devices, with Mapped memory of each of the other
   eight. *)
let with_nine f =
  let ds = Array.init 9 (fun _ -> Host.device ()) in
  let peers =
    List.init 8 (fun i ->
        let _, g' = ds.(i + 1) in
        (g', Option.get (A.alloc g' Mapped 64)))
  in
  Fun.protect
    ~finally:(fun () ->
      List.iter (fun (g', r) -> A.free g' r) peers;
      Array.iter
        (fun (h, g) ->
          A.stop g ~fault:None;
          Host.close h)
        ds)
    (fun () -> f ds.(0) peers)

let seven_others () =
  with_nine @@ fun (_, g) peers ->
  let own = Option.get (A.alloc g Mapped 64) in
  let views = List.map (fun (g', r) -> A.map_peer g g' r) peers in
  equal (list bool) ~msg:"views given"
    (List.init 8 (fun i -> i < 7))
    (List.map Option.is_some views);
  List.iter (Option.iter (A.free g)) views;
  A.free g own

(* Whatever views a device holds, the host's writes to its own Mapped memory
   reach its work. *)
let own_after_views () =
  with_nine @@ fun ((h : Host.t), g) peers ->
  let views = List.filter_map (fun (g', r) -> A.map_peer g g' r) peers in
  let own = Option.get (A.alloc g Mapped 64) in
  H.set32 h.hdp 0xffff_ffff;
  equal answer ~msg:"submit" `Ok (submit g ~v:1 [||]);
  equal int ~msg:"its own HDP register" 0 (H.get32 h.hdp);
  List.iter (A.free g) views;
  A.free g own

let hdp =
  group ~timeout:60. "host data path"
    [
      stateful ~count:100 ~steps:20
        "submit flushes the HDP of every Mapped region its work may read"
        hdp_commands;
      test "a device views Mapped memory of seven other GPUs, not an eighth's"
        seven_others;
      test "a device flushes its own Mapped memory whatever views it holds"
        own_after_views;
    ]

(* Failures

   A failed fill hands the queues none of its submission's parts: only the
   release of its value goes on the compute queue. Markers are words no packet
   the device places holds. *)

let marker k = 0x5eed_0000 lor k
let is_marker w = w land 0xffff_0000 = 0x5eed_0000

type failing = {
  before : (bool * int) list; (* parts before it: queue, words *)
  on_compute : bool;
  code : int;
}

let failing =
  Gen.with_pp
    (fun ppf f ->
      Format.fprintf ppf "{before [%s]; fill on %s; code %d}"
        (String.concat "; "
           (List.map
              (fun (c, n) -> strf "%s %d" (if c then "COMPUTE" else "COPY") n)
              f.before))
        (if f.on_compute then "COMPUTE" else "COPY")
        f.code)
    (let open Gen in
     let+ before = list ~size:(int_range 0 3) (pair bool (int_range 1 64))
     and+ on_compute = bool
     and+ code = int_range 1 1000 in
     { before; on_compute; code })

let failure_hands_nothing f =
  Host.with_device @@ fun h g ->
  cover "a fill on COPY:0 fails" (not f.on_compute);
  cover "a fill on COMPUTE:0 fails" f.on_compute;
  cover "parts on both queues before it"
    (List.exists fst f.before && List.exists (fun (c, _) -> not c) f.before);
  let k = ref 0 in
  let markers n =
    Array.init n (fun _ ->
        incr k;
        marker !k)
  in
  let queue c = if c then "COMPUTE:0" else "COPY:0" in
  let before =
    List.map (fun (c, n) -> E.words ~queue:(queue c) (markers n)) f.before
  in
  let ws = markers 4 in
  let fill = S.fill ~code:f.code (A.capability g) ws ~bytes:64 in
  let fails = E.fill ~queue:(queue f.on_compute) fill ~units:4 ~bytes:64 in
  let c0 = Host.position (Host.compute h)
  and s0 = Host.position (Host.copy h) in
  let why = strf "a fill on %s failed with %d" (queue f.on_compute) f.code in
  let ps = Array.of_list (before @ [ fails ]) in
  equal room_answer ~msg:"room" `Fits (E.room g ps);
  equal answer ~msg:"submit" (`Failed why) (submit g ~v:1 ps);
  let c1 = Host.position (Host.compute h)
  and s1 = Host.position (Host.copy h) in
  equal int ~msg:"the copy queue's position" s0 s1;
  greater int ~msg:"the compute queue's position" ~than:c0 c1;
  equal (list int) ~msg:"markers handed over" []
    (List.filter is_marker (handed (Host.compute h) c0 c1));
  equal answer ~msg:"the next submit" (`Failed why) (submit g ~v:2 [||]);
  equal int ~msg:"the copy queue's position, after" s1
    (Host.position (Host.copy h));
  equal int ~msg:"the compute queue's position, after" c1
    (Host.position (Host.compute h))

(* An AQL queue may read a packet past its write position as soon as its header
   is valid. *)
let aql_failure () =
  Host.with_device ~gpu:mi300 @@ fun h g ->
  let packet k =
    Array.init 16 (fun i -> if i = 0 then 2 else marker ((16 * k) + i))
  in
  let first = E.words ~queue:"COMPUTE:0" (packet 0) in
  let ws = Array.append (packet 1) (packet 2) in
  let fill = S.fill ~code:7 (A.capability g) ws ~bytes:0 in
  let fails = E.fill ~queue:"COMPUTE:0" fill ~units:32 ~bytes:0 in
  let q = Host.compute h in
  let slots = q.bytes / 64 in
  let header s = H.get32 (at q.ring (64 * (s mod slots))) land 0xff in
  let ours s =
    List.exists is_marker
      (List.init 15 (fun i ->
           H.get32 (at q.ring ((64 * (s mod slots)) + (4 * (i + 1))))))
  in
  let p0 = Host.position q in
  equal answer ~msg:"submit" (`Failed "a fill on COMPUTE:0 failed with 7")
    (submit g ~v:1 [| first; fails |]);
  let p1 = Host.position q in
  greater int ~msg:"the release handed over" ~than:p0 p1;
  for s = p0 to p1 - 1 do
    equal bool
      ~msg:(strf "packet %d, handed over, is the submission's" s)
      false (ours s)
  done;
  for s = p1 to p1 + 8 do
    if ours s then
      equal int ~msg:(strf "the header type of packet %d" s) 1 (header s)
  done

let sleeps () =
  let h, g = Host.device () in
  A.sleep g ~seen:5 ~still_ms:10_000;
  equal int ~msg:"sleeps with the word past seen" 0 h.sleeps;
  A.sleep g ~seen:0 ~still_ms:1;
  equal int ~msg:"sleeps with the word at seen" 1 h.sleeps;
  h.report <- Some "memory fault at 0x0";
  raises (A.Fault "memory fault at 0x0") (fun () ->
      A.sleep g ~seen:0 ~still_ms:1);
  A.stop g ~fault:None;
  Host.close h

(* [free] never raises the path's faults. *)
let free_through_fault () =
  Host.with_device @@ fun h g ->
  let r = Option.get (A.alloc g Pinned 64) in
  h.free_fault <- Some "the host frees nothing now";
  A.free g r;
  h.free_fault <- None

(* [stop] writes the last value into the word only once the path stopped every
   queue, and never raises; [free] and [signaled] answer after it. *)
let stops () =
  let case name path_stop word =
    let h, g = Host.device ~stop:path_stop () in
    let r = Option.get (A.alloc g Pinned 64) in
    let page = H.pages 64 in
    let m = Option.get (A.map_host g page 64) in
    equal answer ~msg:name `Ok (submit g ~v:1 [||]);
    equal answer ~msg:name `Ok (submit g ~v:2 [||]);
    A.stop g ~fault:None;
    equal int ~msg:(name ^ ": the word") word (A.signaled g);
    equal int ~msg:(name ^ ": stops") 1 h.stops;
    A.free g r;
    A.free g m;
    H.free_pages page 64;
    Host.close h
  in
  case "the path stopped" (fun () -> `Stopped) 2;
  case "the path may run a queue" (fun () -> `Unknown) 0;
  case "the path failed" (fun () -> raise (A.Fault "lost")) 0

(* The path's stop gets the fault the device was lost for. *)
let stop_fault () =
  let case name fault =
    let h, g = Host.device () in
    A.stop g ~fault;
    equal (option string) ~msg:name fault h.stop_fault;
    Host.close h
  in
  case "no fault" None;
  case "the fault it was lost for" (Some "page fault");
  case "a hang" (Some "no progress for 30000 ms")

(* NOPs

   A PM4 NOP one word long (count 0x3fff), the word Linux's amdgpu driver pads
   compute rings with (gfx_v12_0.c). SDMA reads a zero word as a NOP, as the
   writer's own padding does. Fills of 2^i NOPs, one per queue and size up to
   2^20, move a ring along cheaply: every device's capability has the same C
   functions. *)

let pm4_nop = 0xffff1000
let pads = Hashtbl.create 42

let pad g ~compute i =
  match Hashtbl.find_opt pads (compute, i) with
  | Some f -> f
  | None ->
      let w = if compute then pm4_nop else 0 in
      let f = S.fill (A.capability g) (Array.make (1 lsl i) w) ~bytes:0 in
      Hashtbl.add pads (compute, i) f;
      f

(* The largest 2^i at most [n], up to 2^20. *)
let log2 n =
  let i = ref 0 in
  while !i < 20 && 1 lsl (!i + 1) <= n do
    incr i
  done;
  !i

(* A fill on COPY:0 that places exactly the units it declares, in two calls,
   wherever the copy ring's end falls: [room] words from the fill's start to the
   end, then its two calls of [first] and [second] words. SDMA packets never
   wrap, so a call that does not fit before the end goes after it. *)
let fill_at_the_end (room, first, second) =
  cover "the first call does not fit before the end" (first > room);
  cover "the first call ends exactly at the end" (first = room);
  cover "only the second call does not fit"
    (first < room && first + second > room);
  Host.with_device @@ fun h g ->
  let q = Host.copy h in
  let size = q.bytes / 4 in
  let v = ref 0 in
  let go ps =
    equal room_answer ~msg:"room" `Fits (E.room g ps);
    incr v;
    equal answer ~msg:"submit" `Ok (submit g ~v:!v ps);
    Host.reach g !v
  in
  let words n = E.words ~queue:"COPY:0" (Array.sub (Lazy.force zeros) 0 n) in
  let put () = Host.position q / 4 in
  (* A submission of [n] words takes [n] and its release's. *)
  let p0 = put () in
  go [| words 1 |];
  let release = put () - p0 - 1 in
  let rec advance () =
    let n = size - room - put () - release in
    let pad i =
      go
        [|
          E.fill ~queue:"COPY:0" (pad g ~compute:false i) ~units:(1 lsl i)
            ~bytes:0;
        |];
      advance ()
    in
    if n > 4096 then pad (log2 (n - 256))
    else if n >= 1 then go [| words n |]
    else pad 20
  in
  advance ();
  equal int ~msg:"the fill's start" (size - room) (put () mod size);
  let ws = Array.init (first + second) (fun i -> marker (i + 1)) in
  let f = S.fill ~split:first (A.capability g) ws ~bytes:0 in
  let start = put () in
  go [| E.fill ~queue:"COPY:0" f ~units:(first + second) ~bytes:0 |];
  equal (list int) ~msg:"its words, handed over in order" (Array.to_list ws)
    (List.filter is_marker (handed q start (put ())))

(* Room and calls of any size, and a third of them a first call of exactly the
   room, which a draw of three sizes alone meets once in about 40 cases. *)
let at_the_end =
  let open Gen in
  let call = int_range 1 24 in
  let exact =
    let* room = call in
    map (fun second -> (room, room, second)) call
  in
  with_pp
    (fun ppf (r, a, b) ->
      Format.fprintf ppf "%d words left; calls of %d and %d" r a b)
    (frequency [ (2, triple (int_range 1 40) call call); (1, exact) ])

(* An AQL queue's scratch

   On a GPU of several dies a kernel's scratch is the device's, grown through
   the capability from any domain. A scratch a submission took stays the path's
   to free only once the word reaches that submission's value. The interleaving
   the law needs is held: a grow waits inside its allocation while another
   domain's submission takes the scratch published before. *)
let scratch_taken () =
  let hold = Atomic.make false and inside = Atomic.make false in
  let go_on = Atomic.make false in
  let on_alloc kind _ =
    if kind = `Gpu && Atomic.get hold then begin
      Atomic.set inside true;
      while not (Atomic.get go_on) do
        Domain.cpu_relax ()
      done
    end
  in
  let h, g = Host.device ~gpu:mi300 ~on_alloc () in
  Fun.protect ~finally:(fun () ->
      A.stop g ~fault:None;
      Host.close h)
  @@ fun () ->
  let grow =
    match (A.capability g).compute with
    | Aql { scratch } -> scratch
    | Pm4 -> fail "a GPU of eight dies reads PM4"
  in
  let ok =
    Testable.make
      ~pp:(fun ppf -> function
        | Ok () -> Format.pp_print_string ppf "Ok"
        | Error e -> Format.fprintf ppf "Error %S" e)
      ~equal:( = )
  in
  equal ok ~msg:"a first grow" (Ok ()) (grow 64);
  let taken = h.last_gpu in
  Atomic.set hold true;
  let d = Domain.spawn (fun () -> grow 128) in
  while not (Atomic.get inside) do
    Domain.cpu_relax ()
  done;
  (* A kernel's packet: its submission places the scratch it needs. *)
  let kernel = [| E.words ~queue:"COMPUTE:0" (Array.make 16 0) |] in
  equal answer ~msg:"the submission that takes the first scratch" `Ok
    (submit g ~v:1 kernel);
  Atomic.set go_on true;
  equal ok ~msg:"the second grow" (Ok ()) (Domain.join d);
  equal bool ~msg:"the taken scratch freed before its value is reached" false
    (List.mem taken h.frees);
  Host.reach g 1;
  equal answer ~msg:"the submission that takes the second" `Ok
    (submit g ~v:2 kernel);
  Host.reach g 2;
  equal ok ~msg:"a third grow" (Ok ()) (grow 256);
  equal bool ~msg:"the first scratch, replaced and its value reached" true
    (List.mem taken h.frees)

let scratch =
  group ~timeout:30. "scratch"
    [
      test "a scratch a submission took is freed only once its value is reached"
        scratch_taken;
    ]

(* A device waits on other words only where [waits_on] says so, and on at most
   255 per submission: a submission past either fails as a fill's failure
   does. *)
let waits_refused () =
  List.iter
    (fun n ->
      Host.with_device @@ fun _ g ->
      let msg = strf "%d waits" n in
      equal bool ~msg:"waits on words" false (A.facts g).waits.stores;
      let w = host (word g) in
      match E.submit g ~v:1 ~waits:(Array.make n (w, 1)) [||] with
      | `Ok -> failf "%s: handed over where the device cannot wait" msg
      | `Failed why ->
          equal answer
            ~msg:(msg ^ ": the next submit")
            (`Failed why) (submit g ~v:2 [||]))
    [ 1; 255; 256; 300 ]

let failures =
  group ~timeout:60. "failures"
    [
      prop ~count:100 "a failed fill hands over none of its submission" failing
        failure_hands_nothing;
      test
        "on an AQL queue, a failed fill leaves none of its packets valid past \
         the write position"
        aql_failure;
      prop ~count:200
        ~examples:[ (10, 16, 8); (10, 4, 8) ]
        "a fill on COPY:0 placing exactly its units runs wherever the ring's \
         end falls"
        at_the_end fill_at_the_end;
      test "a submission that waits where the device cannot fails" waits_refused;
      test "sleep asks the path only while the word holds seen" sleeps;
      test "free returns when the path fails to free" free_through_fault;
      test "stop writes the last value only once the path stopped its queues"
        stops;
      test "stop gives the path the fault the device was lost for" stop_fault;
    ]

(* [sleep] runs while another domain submits: a device whose submissions
   complete at once, the caller serialising its submits, answers as some order
   of the calls. *)
type sleeper = { g : A.t; h : Host.t; lock : Mutex.t; mutable v : int }

let sleeper =
  abstract "s" ~release:(fun s ->
      A.stop s.g ~fault:None;
      Host.close s.h)

let sleep_commands =
  [
    command "start"
      (Gen.unit @-> makes sleeper)
      (fun () -> ref 0)
      (fun () ->
        let h, g = Host.device () in
        { g; h; lock = Mutex.create (); v = 0 });
    command "submit"
      (sleeper ^-> returns unit)
      (fun m -> incr m)
      (fun s ->
        Mutex.protect s.lock @@ fun () ->
        s.v <- s.v + 1;
        equal answer ~msg:"submit" `Ok (submit s.g ~v:s.v [||]);
        Host.reach s.g s.v);
    command "sleep"
      (sleeper ^-> returns unit)
      ignore
      (fun s -> A.sleep s.g ~seen:(A.signaled s.g) ~still_ms:0);
    command "signaled"
      (sleeper ^-> returns int)
      (fun m -> !m)
      (fun s -> A.signaled s.g);
  ]

let domains =
  group ~timeout:60. "domains"
    [
      stateful ~count:30 ~domains:2 "sleep answers while another domain submits"
        sleep_commands;
    ]

(* On a GPU *)

(* Work through rig, on a device [S.open_] opened *)

type gpu = S.t = { d : Rig.t; g : A.t }

let buffer ?(memory = Rig.Buffer.Device) t n =
  Rig.Buffer.create ~memory t.d n

let addr = Rig.Buffer.address
let view b first length = Rig.Buffer.view b ~first ~length

let host_buffer s =
  let b = Bigarray.(Array1.create char c_layout (String.length s)) in
  String.iteri (fun i c -> b.{i} <- c) s;
  Rig.Buffer.of_bigarray b

(* The host writes [s] at [b]'s start; it reads [b]'s bytes. *)
let put b s =
  Rig.Buffer.copy ~src:(host_buffer s) ~dst:(view b 0 (String.length s))

let get b =
  let n = Rig.Buffer.length b in
  let a = Bigarray.(Array1.create char c_layout n) in
  Rig.Buffer.copy ~src:b ~dst:(Rig.Buffer.of_bigarray a);
  String.init n (fun i -> a.{i})

let copy ?(after = [||]) ~dst src =
  { Rig.Submission.queue = "COPY:0"; after; work = Copy { src; dst } }

(* Buffers only a kernel's arguments name, which rig cannot know work reads:
   kept until the next [run] returns. *)
let live = ref []

(* Submits [ps] and waits for their value. *)
let run t ps =
  Rig.wait t.d (S.submit t ps);
  live := []

let pattern n seed =
  String.init n (fun i -> Char.chr ((seed + (i * 7)) land 0xff))

(* Kernels *)

(* A fixture's code object: its bytes and its description. *)
type code = { binary : string; co : Abi.Code_object.t }

let code_of bin =
  lazy
    (let binary = Lazy.force bin in
     { binary; co = Result.get_ok (Abi.Code_object.of_string binary) })

let kernels = code_of kernels_bin
let other = code_of (lazy (read_fixture "other_gfx1201.hsaco"))
let work_code = code_of work_bin

let words p =
  let s = Abi.Packet.encode Int64.of_int p in
  Array.init
    (String.length s / 4)
    (fun i -> Int32.to_int (String.get_int32_le s (4 * i)) land 0xffff_ffff)

(* The words of a dispatch of [name] of [of_] over [groups] workgroups of 64,
   its arguments at [args], its descriptor at [entry name]. *)
let dispatch ?(of_ = kernels) gpu entry name ~args ~groups =
  let k = Option.get (Abi.Code_object.kernel (Lazy.force of_).co name) in
  let base = entry name - k.descriptor in
  words
    (Pm4.run gpu
       (Pm4.dispatch gpu k ~program:(base + k.entry) ~scratch:0 ~args ~packet:0
          ~threads:(64, 1, 1) ~groups:(groups, 1, 1) ()))

let image ?(of_ = kernels) t =
  match Rig.Image.load t.d (Lazy.force of_).binary with
  | Ok p -> p
  | Error why -> fail why

(* The words of a dispatch of [p]'s [name], its arguments [args] in a buffer of
   their own. *)
let kernel_words ?of_ t p name ~groups args =
  let a = buffer ~memory:Pinned t 4096 in
  put a args;
  live := a :: !live;
  let entry f = Option.get (Rig.Image.entry p f) in
  dispatch ?of_ (A.capability t.g).gpu entry name ~args:(addr a) ~groups

let kernel ?of_ ?after t p name ~groups args =
  S.words_part ~queue:"COMPUTE:0" ?after
    (kernel_words ?of_ t p name ~groups args)

let multiples k n = le32s (List.init n (fun i -> k * i))
let doubled = multiples 2

(* Launches on the GPU *)

module Sub = Rig.Submission

let launch_code = code_of launch_bin

(* Submits one launch of [p]'s [name] over [groups] of [threads], its groups
   taking [shared] bytes, whose block [set] stores, reading [reads] and writing
   [writes], and waits for it. *)
let launch t p name ~params ~refs ~groups:(gx, gy, gz) ~threads:(tx, ty, tz)
    ?(shared = 0) ~set ~reads ~writes () =
  let refs =
    Array.of_list (List.map (fun (at, slot) -> { Sub.at; slot }) refs)
  in
  let part =
    {
      Sub.queue = "COMPUTE:0";
      after = [||];
      work = Launch { image = p; kernel = name; params; refs };
    }
  in
  let s =
    Sub.make ~reads:(Array.length reads) ~writes:(Array.length writes) t.d
      [| part |]
  in
  let run = Sub.Run.make () and b = Sub.block s 0 in
  Sub.Run.groups run b gx gy gz;
  Sub.Run.threads run b tx ty tz;
  Sub.Run.shared run b shared;
  set run b;
  Rig.Point.wait (Rig.submit s ~run ~reads ~writes ~waits:[||])

let scratch_launch () =
  S.with_ @@ fun t ->
  let p = image ~of_:launch_code t in
  let out = buffer ~memory:Pinned t (4 * 64) in
  launch t p "scratch" ~params:12
    ~refs:[ (0, 0) ]
    ~groups:(1, 1, 1) ~threads:(64, 1, 1)
    ~set:(fun run b ->
      Sub.Run.int64 run b 0 0;
      Sub.Run.int32 run b 8 5)
    ~reads:[||] ~writes:[| out |] ();
  equal string ~msg:"out" (multiples 5 64) (get out)

(* [lds] over a group of 64 work-items, its tile of [n] words in the dynamic LDS
   after the 256 bytes it takes itself: the tile's last words, which only a
   group segment grown by the launch's shared memory holds. *)
let lds_launch n () =
  S.with_ @@ fun t ->
  let p = image ~of_:launch_code t in
  let out = buffer ~memory:Pinned t (4 * 64) in
  let own = (launch_kernel "lds").group_segment in
  launch t p "lds" ~params:16
    ~refs:[ (0, 0) ]
    ~groups:(1, 1, 1) ~threads:(64, 1, 1) ~shared:(4 * n)
    ~set:(fun run b ->
      Sub.Run.int64 run b 0 0;
      Sub.Run.int32 run b 8 own;
      Sub.Run.int32 run b 12 n)
    ~reads:[||] ~writes:[| out |] ();
  equal string ~msg:"out" (multiples 1 64) (get out)

(* 257 work-items, one more than [ids] takes: the submit refuses them and the
   device stays live. *)
let past_the_bound () =
  S.with_ @@ fun t ->
  let p = image ~of_:launch_code t in
  let out = buffer t (4 * 257) in
  raises_match ~msg:"257 work-items"
    (function Invalid_argument _ -> true | _ -> false)
    (fun () ->
      launch t p "ids" ~params:24
        ~refs:[ (0, 0) ]
        ~groups:(1, 1, 1) ~threads:(257, 1, 1)
        ~set:(fun run b -> Sub.Run.int64 run b 0 0)
        ~reads:[||] ~writes:[| out |] ());
  equal (option string) ~msg:"the device's loss" None (Rig.lost t.d)

let launching =
  group ~timeout:60. "launching"
    [
      test "a launch of a kernel that takes scratch computes" scratch_launch;
      cases ~name:string_of_int
        "a launch's dynamic LDS follows the kernel's own" [ 64; 4096; 16320 ]
        (fun n -> lds_launch n ());
      test "a launch past its kernel's work-items is refused, the device live"
        past_the_bound;
    ]

let work =
  group ~timeout:60. "work"
    [
      test "an opened GPU's facts are its capability's" (fun () ->
          S.with_ @@ fun t ->
          let c = A.capability t.g in
          let f = A.facts t.g in
          equal string ~msg:"arch" (Gpu.processor c.gpu) f.arch;
          greater int ~msg:"budget" ~than:0 f.budget;
          greater int ~msg:"clock" ~than:0 c.clock_hz;
          equal bool ~msg:"AQL" (c.gpu.xccs > 1)
            (match c.compute with Aql _ -> true | Pm4 -> false));
      test "an allocation past the GPU's memory is None" (fun () ->
          S.with_ @@ fun t ->
          equal bool ~msg:"none" true
            (Option.is_none (A.alloc t.g Device (2 * (A.facts t.g).budget))));
    ]

(* Memory *)

(* The device the property tests share, closed when the run ends. *)
let shared = fixture ~teardown:S.close S.open_
(* Copies of the copy engine's largest packet and its neighbours: the bytes
   before each copy's end arrive, those after stay as they were. Two windows are
   read: the start, and the packet's end. *)
let around_the_packet () =
  let t = shared () in
  let max = Abi.Sdma.max_copy (A.capability t.g).gpu in
  let w = 4096 in
  let n = max + (w / 2) in
  let src = buffer ~memory:Pinned t n and dst = buffer t n in
  let zeros = buffer ~memory:Pinned t w and back = buffer ~memory:Pinned t w in
  put zeros (String.make w '\000');
  let windows = [ 0; max - (w / 2) ] in
  List.iteri (fun i o -> put (view src o w) (pattern w (i + 1))) windows;
  List.iter
    (fun k ->
      run t
        (Array.of_list
           (List.map (fun o -> copy ~dst:(view dst o w) zeros) windows));
      run t [| copy ~dst:(view dst 0 k) (view src 0 k) |];
      List.iteri
        (fun i o ->
          run t [| copy ~dst:back (view dst o w) |];
          let arrived = Int.max 0 (Int.min w (k - o)) in
          let expected =
            String.sub (pattern w (i + 1)) 0 arrived
            ^ String.make (w - arrived) '\000'
          in
          equal string ~msg:(strf "%d bytes, at %d" k o) expected (get back))
        windows)
    [ max - 1; max; max + 1 ]

let memory =
  group ~timeout:120. "memory"
    [
      test "copies around the copy engine's packet size are the identity"
        around_the_packet;
    ]

let spin t p flag n = kernel t p "spin" ~groups:1 (le64 (addr flag) ^ le64 n)

(* Code placed where other code ran runs as placed: each image is unloaded
   once unreachable and its work done, and the next load reuses the memory. *)
let stale_code () =
  S.with_ @@ fun t ->
  let out = buffer ~memory:Pinned t (4 * 64) in
  let load of_ =
    let p = image ~of_ t in
    run t [| kernel ~of_ t p "double_index" ~groups:1 (le64 (addr out)) |];
    let at = Option.get (Rig.Image.entry p "double_index") in
    (at, get out)
  in
  let first, doubled_out = load kernels in
  equal string ~msg:"the first object's" (doubled 64) doubled_out;
  Gc.full_major ();
  let second, tripled_out = load other in
  equal string ~msg:"the second object's" (multiples 3 64) tripled_out;
  let base f m =
    m
    - (Option.get (Abi.Code_object.kernel (Lazy.force f).co "double_index"))
        .descriptor
  in
  (* The case an instruction cache could serve stale: the system's addresses for
     the second object are the first's. *)
  if base kernels first <> base other second then
    skip ~reason:"the second object got other addresses" ()

let code =
  group ~timeout:60. "kernels"
    [
      test "a kernel loaded from its image computes" (fun () ->
          S.with_ @@ fun t ->
          let p = image t in
          let out = buffer ~memory:Pinned t (4 * 256) in
          run t [| kernel t p "double_index" ~groups:4 (le64 (addr out)) |];
          equal string ~msg:"out" (doubled 256) (get out));
      test "code placed where other code ran runs as placed" stale_code;
      test "a fill places its words" (fun () ->
          S.with_ @@ fun t ->
          let word = buffer ~memory:Pinned t 8 in
          put word (le64 0);
          let ws = words (Pm4.write_data (Memory (addr word)) 0xc0ffee) in
          let f = S.fill (A.capability t.g) ws ~bytes:64 in
          run t
            [|
              S.fill_part ~queue:"COMPUTE:0" f ~units:(Array.length ws)
                ~bytes:64;
            |];
          equal string ~msg:"word" (le64 0xc0ffee) (get word));
      test "a release wakes a sleeper well before its bound" (fun () ->
          (* A few milliseconds of work, slept on with a bound of seconds: the
             release's interrupt, not the bound, ends the sleep. *)
          S.with_ @@ fun t ->
          let p = image t in
          let flag = buffer ~memory:Pinned t 8 in
          put flag (le64 0);
          let v = S.submit t [| spin t p flag 2_000 |] in
          let t0 = Rig.Profile.now () in
          while A.signaled t.g < v do
            A.sleep t.g ~seen:(A.signaled t.g) ~still_ms:5_000
          done;
          less int ~msg:"ms asleep" ~than:1_000
            ((Rig.Profile.now () - t0) / 1_000_000));
      test "a loss while work runs stops the work" (fun () ->
          (* The flag is the test's memory, which the stop leaves mapped. *)
          S.with_ @@ fun t ->
          let p = image t in
          let flag = Option.get (A.alloc t.g Pinned 8) in
          H.set64 (host flag) 0;
          let args = le64 (address flag) ^ le64 1_500_000 in
          let v = S.submit t [| kernel t p "spin" ~groups:1 args |] in
          let f = S.fill ~code:5 (A.capability t.g) [||] ~bytes:0 in
          let fails = S.fill_part ~queue:"COMPUTE:0" f ~units:0 ~bytes:0 in
          raises_match
            (function Rig.Lost _ -> true | _ -> false)
            (fun () -> S.submit t [| fails |]);
          S.close t;
          equal int ~msg:"the word" (v + 1) (Rig.signaled t.d);
          still ~msg:"flag (sampled)" int 0
            (fun () -> H.get64 (host flag))
            ~ms:100);
    ]

(* Work in order on one queue

   Parts on one queue run in array order: a compute part after another sees its
   writes, whatever the sizes and whether a copy part sits between them. *)

(* The shared device's image of fixtures/work.cl, loaded once. *)
let work_image = lazy (image ~of_:work_code (shared ()))

let work_words t name ~groups args =
  kernel_words ~of_:work_code t (Lazy.force work_image) name ~groups args

module Chain = struct
  (* A submission of [incs] compute parts, each adding 1 to the words its
     predecessor wrote: by words or by a fill, with a copy part between two of
     them where [via] says, after a submission released on COMPUTE:0 or on
     COPY:0. *)
  type t = { words : int; incs : (bool * bool) list; after_copy : bool }

  let pp ppf c =
    Format.fprintf ppf "{%d words; %s; after a %s release}" c.words
      (String.concat ", "
         (List.map
            (fun (fill, via) ->
              (if fill then "fill" else "words")
              ^ if via then " then copy" else "")
            c.incs))
      (if c.after_copy then "COPY:0" else "COMPUTE:0")

  let most = 1 lsl 20
  let buffers = 8

  let gen =
    let open Gen in
    let+ words = of_list [ 64; 1000; 65536; most ]
    and+ incs = list ~size:(int_range 2 4) (pair bool bool)
    and+ after_copy = bool in
    { words; incs; after_copy }

  type sys = {
    input : Rig.Buffer.t;
    garbage : Rig.Buffer.t;
    out : Rig.Buffer.t;
    bufs : Rig.Buffer.t array;
  }

  let sys =
    lazy
      (let t = shared () in
       let pinned () = buffer ~memory:Pinned t (4 * most) in
       let input = pinned () and garbage = pinned () and out = pinned () in
       put input
         (le32s (List.init most (fun i -> i * 2654435761 land 0xffff_ffff)));
       put garbage (String.make (4 * most) '\xee');
       {
         input;
         garbage;
         out;
         bufs = Array.init buffers (fun _ -> buffer t (4 * most));
       })

  let law c =
    let t = shared () in
    let s = Lazy.force sys in
    let n = c.words and bytes = 4 * c.words in
    let part b = view b 0 bytes in
    let k = List.length c.incs in
    let between = List.filteri (fun i _ -> i < k - 1) c.incs in
    cover "parts back to back" (List.exists (fun (_, via) -> not via) between);
    cover "a copy part between two compute parts" (List.exists snd between);
    cover "a submission whose compute queue released the value before"
      (not c.after_copy);
    cover "a submission whose copy queue released the value before" c.after_copy;
    cover "every compute unit busy" (n >= 65536);
    (* Garbage in every buffer, then the input in the first. *)
    run t
      (Array.map
         (fun b -> copy ~dst:(part b) (part s.garbage))
         (Array.append s.bufs [| s.out |]));
    run t [| copy ~dst:(part s.bufs.(0)) (part s.input) |];
    if c.after_copy then
      run t [| copy ~dst:(view s.out 0 4) (view s.garbage 0 4) |]
    else run t [| S.words_part ~queue:"COMPUTE:0" [| pm4_nop |] |];
    let parts = ref [] in
    let add p =
      parts := p :: !parts;
      List.length !parts - 1
    in
    let src = ref s.bufs.(0) and next = ref 1 and copied = ref None in
    List.iteri
      (fun i (fill, via) ->
        let out = s.bufs.(!next) in
        incr next;
        let after = Option.fold ~none:[||] ~some:(fun j -> [| j |]) !copied in
        let ws =
          work_words t "inc"
            ~groups:((n + 63) / 64)
            (le64 (addr out) ^ le64 (addr !src) ^ le32s [ n ])
        in
        let p =
          if fill then
            let f = S.fill (A.capability t.g) ws ~bytes:0 in
            S.fill_part ~queue:"COMPUTE:0" ~after f ~units:(Array.length ws)
              ~bytes:0
          else S.words_part ~queue:"COMPUTE:0" ~after ws
        in
        let me = add p in
        copied := None;
        src := out;
        if via && i < k - 1 then begin
          let mid = s.bufs.(!next) in
          incr next;
          copied := Some (add (copy ~after:[| me |] ~dst:(part mid) (part out)));
          src := mid
        end)
      c.incs;
    let last = List.length !parts - 1 in
    ignore (add (copy ~after:[| last |] ~dst:(part s.out) (part !src)));
    run t (Array.of_list (List.rev !parts));
    let input = get (part s.input) in
    let expected = Bytes.create bytes in
    for i = 0 to n - 1 do
      Bytes.set_int32_le expected (4 * i)
        (Int32.add (String.get_int32_le input (4 * i)) (Int32.of_int k))
    done;
    equal string ~msg:"the last part's words" (Bytes.to_string expected)
      (get (part s.out))
end

let queue_order =
  group ~timeout:120. "in order"
    [
      prop ~count:24
        "compute parts run in array order, each reading its predecessor's \
         writes"
        (Gen.with_pp Chain.pp Chain.gen)
        Chain.law;
    ]

(* Through the C entries *)

(* A device's values, numbered as the test hands them to its C entries. *)
type run = { g : A.t; mutable v : int }

let device g = { g; v = 0 }

let go r ps =
  r.v <- r.v + 1;
  equal answer ~msg:"submit" `Ok (submit r.g ~v:r.v ps);
  S.reached r.g r.v

let entry m f = (Option.get (A.entry m f)).code

(* [m]'s image laid over new memory [code], and the part that copies it there
   from staging memory the host wrote. *)
let laid ?(of_ = kernels) r =
  let m, code, bytes = load r.g (Lazy.force of_).binary in
  let n = String.length bytes in
  let staging = Option.get (A.alloc r.g Pinned n) in
  H.write (host staging) bytes;
  (m, code, E.copy ~dst:(address code) ~src:(address staging) n)

let arguments r values =
  let args = Option.get (A.alloc r.g Pinned 4096) in
  H.write (host args) (String.concat "" (List.map le64 values));
  args

(* A submit answers RIG_FAILED for a fill's failure, runs none of the parts,
   reaches the value anyway, and answers the same failure after. *)
let failures_hw =
  group ~timeout:60. "C failures"
    [
      test "a fill past its declaration fails, and the word still moves"
        (fun () ->
          S.with_ @@ fun { g; _ } ->
          let word = Option.get (A.alloc g Pinned 8) in
          H.write (host word) (le64 0);
          let ws = words (Pm4.write_data (Memory (address word)) 1) in
          let f = S.fill (A.capability g) ws ~bytes:0 in
          let p =
            E.fill ~queue:"COMPUTE:0" f ~units:(Array.length ws - 1) ~bytes:0
          in
          let why =
            match submit g ~v:1 [| p |] with
            | `Ok -> fail "a fill past its declaration placed"
            | `Failed why ->
                contains ~sub:"a fill on COMPUTE:0 failed with " why;
                why
          in
          S.reached g 1;
          equal string ~msg:"the fill's word" (le64 0) (H.read (host word) 8);
          equal answer ~msg:"the next" (`Failed why) (submit g ~v:2 [||]));
      test "a failed submission runs none of its parts" (fun () ->
          S.with_ @@ fun { g; _ } ->
          let n = 4096 in
          let src = Option.get (A.alloc g Pinned n) in
          let dst = Option.get (A.alloc g Pinned n) in
          H.write (host src) (pattern n 1);
          H.write (host dst) (String.make n '\000');
          let f = S.fill ~code:5 (A.capability g) [||] ~bytes:0 in
          equal answer ~msg:"submit" (`Failed "a fill on COPY:0 failed with 5")
            (submit g ~v:1
               [|
                 E.copy ~dst:(address dst) ~src:(address src) n;
                 E.fill ~queue:"COPY:0" f ~units:0 ~bytes:0;
               |]);
          S.reached g 1;
          equal string ~msg:"dst" (String.make n '\000') (H.read (host dst) n));
      test "a value failed behind running work is reached once it ends"
        (fun () ->
          S.with_ @@ fun ({ g; _ } as t) ->
          let r = device g in
          let m, _, upload = laid r in
          go r [| upload |];
          let flag = Option.get (A.alloc g Pinned 8) in
          H.write (host flag) (le64 0);
          let args = arguments r [ address flag; 150_000 ] in
          let spin =
            E.words ~queue:"COMPUTE:0"
              (dispatch (A.capability g).gpu (entry m) "spin"
                 ~args:(address args) ~groups:1)
          in
          equal answer ~msg:"the work" `Ok (submit g ~v:2 [| spin |]);
          let f = S.fill ~code:5 (A.capability g) [||] ~bytes:0 in
          equal answer ~msg:"the failure"
            (`Failed "a fill on COMPUTE:0 failed with 5")
            (submit g ~v:3 [| E.fill ~queue:"COMPUTE:0" f ~units:0 ~bytes:0 |]);
          S.reached g 3;
          equal string ~msg:"the work's flag" (le32s [ 1 ])
            (H.read (host flag) 4);
          S.close t;
          equal int ~msg:"the word" 3 (A.signaled g));
    ]

(* Rings wrap

   Drawn submissions wrap each ring and the argument segment several times,
   landing a submission exactly at the end, one unit short of it or one unit
   past it (a word of a ring, 64 bytes of the segment). Where each ring stands
   is planned on the host-memory path, whose writer places the same words for
   the same submissions; the plan then runs on the GPU, where every value
   completes in order and writes what it should. *)

module Wrap = struct
  let slots = 4096

  type shape =
    | Pad of bool * int (* 2^i NOPs on COMPUTE:0 (true) or COPY:0, by a fill *)
    | Nops of int (* COMPUTE:0: NOP words, then the value's write *)
    | Fill of int (* COMPUTE:0: a fill writing the value, taking these bytes *)
    | Zeros of int (* COPY:0: zero words, then the value's copy *)
    | Both of int * int (* Nops and Zeros in one submission *)

  type target = Compute | Copy | Segment
  type fit = Exact | Short | Over

  let pp_shape ppf = function
    | Pad (c, i) ->
        Format.fprintf ppf "pad %s %d"
          (if c then "COMPUTE" else "COPY")
          (1 lsl i)
    | Nops n -> Format.fprintf ppf "nops %d" n
    | Fill b -> Format.fprintf ppf "fill %d" b
    | Zeros n -> Format.fprintf ppf "zeros %d" n
    | Both (n, z) -> Format.fprintf ppf "nops %d + zeros %d" n z

  let target_name = function
    | Compute -> "compute ring"
    | Copy -> "copy ring"
    | Segment -> "segment"

  let fit_name = function
    | Exact -> "exactly at"
    | Short -> "short of"
    | Over -> "past"

  let pp_goal ppf (t, f, mix) =
    Format.fprintf ppf "{%s the %s's end, after [%a]}" (fit_name f)
      (target_name t)
      (Format.pp_print_list
         ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ")
         pp_shape)
      mix

  let goals =
    let open Gen in
    let shape =
      one_of
        [
          map (fun n -> Nops n) (of_list [ 0; 1; 7; 4096; 1 lsl 16 ]);
          map (fun b -> Fill b) (of_list [ 64; 4096; 1 lsl 16 ]);
          map (fun n -> Zeros n) (of_list [ 0; 1; 5; 4096; 1 lsl 16 ]);
          map (fun (c, i) -> Pad (c, i)) (pair bool (of_list [ 0; 10; 20 ]));
          map
            (fun (n, z) -> Both (n, z))
            (pair (of_list [ 0; 13 ]) (of_list [ 0; 3 ]));
        ]
    in
    (* Every landing, in a drawn order, each after drawn submissions. *)
    let landings =
      List.concat_map
        (fun t -> List.map (fun f -> (t, f)) [ Exact; Short; Over ])
        [ Compute; Copy; Segment ]
    in
    with_pp
      (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_goal)
      (let* order = permutation landings in
       let+ mixes =
         list ~size:(constant 9) (list ~size:(int_range 0 3) shape)
       in
       List.map2 (fun (t, f) mix -> (t, f, mix)) order mixes)

  (* The memory shapes write: a slot per value for compute's writes, and the
     slots copies move. *)
  type sink = { res : A.region; src : A.region; dst : A.region }

  let sink g =
    let res = Option.get (A.alloc g Pinned (4 * slots)) in
    let src = Option.get (A.alloc g Pinned (8 * slots)) in
    let dst = Option.get (A.alloc g Pinned (8 * slots)) in
    H.write (host res) (String.make (4 * slots) '\000');
    H.write (host src)
      (String.concat "" (List.init slots (fun i -> le64 (i + 1))));
    H.write (host dst) (String.make (8 * slots) '\000');
    { res; src; dst }

  let write k v =
    words (Pm4.write_data (Memory (address k.res + (4 * (v mod slots)))) v)

  (* [shape]'s parts as value [v] of [g], and its fill's argument. *)
  let parts g k v shape =
    let compute n =
      E.words ~queue:"COMPUTE:0"
        (Array.append (Array.make n pm4_nop) (write k v))
    in
    let copy n =
      let o = 8 * (v mod slots) in
      (if n = 0 then [] else [ E.words ~queue:"COPY:0" (Array.make n 0) ])
      @ [ E.copy ~dst:(address k.dst + o) ~src:(address k.src + o) 8 ]
    in
    match shape with
    | Pad (c, i) ->
        let queue = if c then "COMPUTE:0" else "COPY:0" in
        ( [| E.fill ~queue (pad g ~compute:c i) ~units:(1 lsl i) ~bytes:0 |],
          None )
    | Nops n -> ([| compute n |], None)
    | Zeros n -> (Array.of_list (copy n), None)
    | Both (n, z) -> (Array.of_list (compute n :: copy z), None)
    | Fill b ->
        let ws = write k v in
        let f = S.fill (A.capability g) ws ~bytes:b in
        ( [| E.fill ~queue:"COMPUTE:0" f ~units:(Array.length ws) ~bytes:b |],
          Some f )

  let segment = 1 lsl 20
  let quarter = segment / 4
  let round64 n = (n + 63) / 64 * 64

  (* The shapes, in value order, that reach [goals] on a fresh device of [gpu],
     planned on the host-memory path. The writer takes a submission's segment
     bytes in one run of 64 bytes and its fills' bytes rounded to 64, from the
     segment's start when they do not fit before its end. *)
  let plan gpu goals =
    Host.with_device ~gpu @@ fun h g ->
    let k = sink g in
    let shapes = ref [] and v = ref 0 and taken = ref (0, 0) in
    let go shape =
      incr v;
      let ps, arg = parts g k !v shape in
      equal room_answer ~msg:"room" `Fits (E.room g ps);
      equal answer ~msg:"submit" `Ok (submit g ~v:!v ps);
      Host.reach g !v;
      (match (shape, arg) with
      | Fill b, Some f -> taken := (S.fill_address f, b)
      | _ -> ());
      shapes := shape :: !shapes
    in
    let ring = (Host.compute h).bytes / 4 in
    let compute () = Host.position (Host.compute h) in
    let copy () = Host.position (Host.copy h) / 4 in
    go (Fill 64);
    let base = fst !taken in
    let segment_at () =
      let a, b = !taken in
      (a - base + round64 b) mod segment
    in
    let shift = function Exact -> 0 | Short -> -1 | Over -> 1 in
    (* [mk n] places [n >= least] words of its own and [o] more, measured on a
       second one so that its queue released the value before it. Pads of NOPs
       approach the end, each leaving more room than its own release takes. *)
    let land_ring name ~compute:c mk ~least at f =
      go (mk 64);
      let p = at () in
      go (mk 64);
      let o = at () - p - 64 in
      let rec landing () =
        let p = at () in
        let stop = (((p / ring) + 1) * ring) + shift f in
        let n = stop - p - o in
        if n > 4096 then (
          go (Pad (c, log2 (n - 256)));
          landing ())
        else if n < least then (
          go (Pad (c, 20));
          landing ())
        else begin
          go (mk n);
          (* SDMA packets never wrap: past the end, the last one goes after
             it. *)
          if not (name = "copy ring" && f = Over) then
            equal int
              ~msg:(strf "where the %s's landing ends" name)
              stop (at ())
        end
      in
      landing ()
    in
    let land_segment f =
      let rec landing () =
        let at = segment_at () in
        let b = segment - at - 64 + (64 * shift f) in
        if b > quarter || b < 64 then (
          go (Fill quarter);
          landing ())
        else begin
          go (Fill b);
          equal int ~msg:"where the segment's landing takes its bytes"
            (if f = Over then base else base + at)
            (fst !taken)
        end
      in
      landing ()
    in
    let wd = Array.length (write k 0) in
    List.iter
      (fun (t, f, mix) ->
        List.iter go mix;
        match t with
        | Compute ->
            land_ring "compute ring" ~compute:true
              (fun n -> Nops (n - wd))
              ~least:wd compute f
        | Copy ->
            land_ring "copy ring" ~compute:false
              (fun n -> Zeros n)
              ~least:1 copy f
        | Segment -> land_segment f)
      goals;
    cover "the compute ring wrapped twice" (compute () > 2 * ring);
    cover "the copy ring wrapped twice" (copy () > 2 * ring);
    List.rev !shapes

  (* Runs [shapes] as the values from [first] of a device of the GPU: the word
     never moves back, every value is reached, and the slots hold what the last
     value that wrote each wrote. *)
  let run g ~first shapes =
    let k = sink g in
    let rec room ps =
      match E.room g ps with
      | `Fits -> ()
      | `Never -> fail "Never"
      | `Later ->
          S.reached g (A.signaled g + 1);
          room ps
    in
    let seen = ref 0 in
    List.iteri
      (fun i shape ->
        let ps, _ = parts g k (first + i) shape in
        room ps;
        equal answer ~msg:"submit" `Ok (submit g ~v:(first + i) ps);
        let w = A.signaled g in
        at_least int ~msg:"the word" ~than:!seen w;
        seen := w)
      shapes;
    let last = first + List.length shapes - 1 in
    S.reached g last;
    let res = Bytes.make (4 * slots) '\000'
    and dst = Bytes.make (8 * slots) '\000' in
    List.iteri
      (fun i shape ->
        let v = first + i in
        let o = v mod slots in
        let compute () = Bytes.set_int32_le res (4 * o) (Int32.of_int v) in
        let copy () = Bytes.set_int64_le dst (8 * o) (Int64.of_int (o + 1)) in
        match shape with
        | Pad _ -> ()
        | Nops _ | Fill _ -> compute ()
        | Zeros _ -> copy ()
        | Both _ ->
            compute ();
            copy ())
      shapes;
    equal string ~msg:"the compute writes" (Bytes.to_string res)
      (H.read (host k.res) (4 * slots));
    equal string ~msg:"the copies" (Bytes.to_string dst)
      (H.read (host k.dst) (8 * slots))

  let law goals =
    S.with_ @@ fun { g; _ } -> run g ~first:1 (plan (A.capability g).gpu goals)

  (* Values from 2^32 - [k], where a value's release goes on the queue of its
     submission's last part and slot words compare 32 bits: the edge's own
     numbering, which rig's values reach only after 2^32 submits. *)
  let carried =
    let open Gen in
    let shape =
      of_list ~pp:pp_shape [ Nops 0; Zeros 0; Both (0, 0); Both (13, 3) ]
    in
    with_pp
      (fun ppf (k, shapes) ->
        Format.fprintf ppf "from 2^32 - %d: [%a]" k
          (Format.pp_print_list
             ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ")
             pp_shape)
          shapes)
      (pair (int_range 1 4) (list ~size:(int_range 1 8) shape))

  let across_carry (k, shapes) =
    let first = (1 lsl 32) - k in
    let at_carry =
      List.filteri (fun i _ -> (first + i) land 0xffff_ffff = 0) shapes
    in
    cover "the value at 2^32 ends on COPY:0"
      (List.exists (function Zeros _ | Both _ -> true | _ -> false) at_carry);
    cover "the value at 2^32 ends on COMPUTE:0"
      (List.exists (function Nops _ -> true | _ -> false) at_carry);
    S.with_ @@ fun { g; _ } ->
    A.renumber g first;
    run g ~first shapes
end

(* Waits on words

   Where the compute queue compares 64-bit words ([waits_on]), a submission
   holds its work until each word it waits on, as unsigned 64 bits, reaches its
   value; a 32-bit compare would pass a word whose low half alone is above. The
   words here are the host's, in memory the device maps. *)

let waits_on g =
  if not (A.facts g).waits.stores then
    skip ~reason:"the GPU's compute queue does not wait on words" ()

(* [t] at or past 2^32, and a first value of the word below it. *)
let crossing =
  Gen.with_pp
    (fun ppf (w0, t) ->
      Format.fprintf ppf "word 2^32%+d, wait for 2^32%+d"
        (w0 - (1 lsl 32))
        (t - (1 lsl 32)))
    Gen.(
      let+ j = frequency [ (3, constant 0); (3, int_range 1 3) ]
      and+ d = int_range 1 4 in
      let t = (1 lsl 32) + j in
      (t - d, t))

let held (w0, t) =
  S.with_ @@ fun { g; _ } ->
  waits_on g;
  let word = Option.get (A.alloc g Pinned 8) in
  let src = Option.get (A.alloc g Pinned 64)
  and dst = Option.get (A.alloc g Pinned 64) in
  let zeros = String.make 64 '\000' in
  H.write (host src) (String.make 64 'w');
  H.write (host dst) zeros;
  H.write (host word) (le64 w0);
  let below =
    List.sort_uniq compare
      (List.filter
         (fun x -> x >= w0)
         ((t - 1) :: (((t lsr 32) lsl 32) - 1) :: [ w0 ]))
  in
  cover "a value below the wait whose low half is above its"
    (List.mem ((1 lsl 32) - 1) below);
  cover "a wait for a value whose low half is 0" (t land 0xffff_ffff = 0);
  equal answer ~msg:"submit" `Ok
    (E.submit g ~v:1
       ~waits:[| (address word, t) |]
       [| E.copy ~dst:(address dst) ~src:(address src) 64 |]);
  List.iter
    (fun x ->
      H.write (host word) (le64 x);
      still
        ~msg:(strf "the copy, the word at %d (sampled)" x)
        string zeros
        (fun () -> H.read (host dst) 64)
        ~ms:20;
      equal int ~msg:(strf "the timeline, the word at %d" x) 0 (A.signaled g))
    below;
  H.write (host word) (le64 t);
  S.reached g 1;
  equal string ~msg:"the copy, the word reached" (String.make 64 'w')
    (H.read (host dst) 64);
  List.iter (A.free g) [ word; src; dst ]

(* A submission waits on at most 255 words. *)
let wait_bound () =
  S.with_ @@ fun { g; _ } ->
  waits_on g;
  let word = Option.get (A.alloc g Pinned 8) in
  H.write (host word) (le64 1);
  let waits n = Array.make n (address word, 1) in
  equal answer ~msg:"255 waits" `Ok (E.submit g ~v:1 ~waits:(waits 255) [||]);
  S.reached g 1;
  match E.submit g ~v:2 ~waits:(waits 256) [||] with
  | `Ok -> fail "256 waits handed over"
  | `Failed why ->
      S.reached g 2;
      equal answer ~msg:"the next submit" (`Failed why) (submit g ~v:3 [||])

let waits =
  group ~timeout:120. "waits"
    [
      prop ~count:12
        "a wait holds the work until the word reaches it, across 2^32 (sampled)"
        crossing held;
      test "a submission waits on 255 words, and fails on 256" wait_bound;
    ]

(* Slots aged past 2^31

   A slot word holds the low 32 bits of the last value that wrote it, and a wait
   compares 32 bits: a slot left 2^31 values old is rewritten before a wait
   could read it as a value to come. The device starts with every slot 2^31 -
   600 values old; over 1,300 submissions each slot passes 2^31 and is refreshed
   at its next turn, while chains of parts alternating between the queues signal
   and wait on the slots: each compute part writes a token, the copy part after
   it copies the token out. A wait that passed early would copy a cell before
   its token. *)
let aged_slots () =
  S.with_ @@ fun { g; _ } ->
  let values = 1300 and most = 256 in
  let bytes = 4 * values * most in
  let cells = Option.get (A.alloc g Pinned bytes) in
  let out = Option.get (A.alloc g Pinned bytes) in
  H.write (host cells) (String.make bytes '\000');
  H.write (host out) (String.make bytes '\000');
  let token k m = ((k * 1000) + m + 1) land 0xffff_ffff in
  let at r k m = address r + (4 * ((k * most) + m)) in
  let pairs k = if k mod 100 = 0 then most else 8 in
  let base = 1 lsl 31 in
  A.renumber ~age:(base - 600) g base;
  let rec room ps =
    match E.room g ps with
    | `Fits -> ()
    | `Never -> fail "Never"
    | `Later ->
        S.reached g (A.signaled g + 1);
        room ps
  in
  for k = 0 to values - 1 do
    let ps =
      Array.concat
        (List.init (pairs k) (fun m ->
             let after = if m = 0 then [||] else [| (2 * m) - 1 |] in
             [|
               E.words ~queue:"COMPUTE:0" ~after
                 (words (Pm4.write_data (Memory (at cells k m)) (token k m)));
               E.copy ~after:[| 2 * m |] ~dst:(at out k m) ~src:(at cells k m) 4;
             |]))
    in
    room ps;
    equal answer
      ~msg:(strf "value %d" (base + k))
      `Ok
      (submit g ~v:(base + k) ps)
  done;
  S.reached g (base + values - 1);
  let got = H.read (host out) bytes in
  for k = 0 to values - 1 do
    for m = 0 to pairs k - 1 do
      let o = 4 * ((k * most) + m) in
      equal int
        ~msg:(strf "value %d, pair %d" (base + k) m)
        (token k m)
        (Int32.to_int (String.get_int32_le got o) land 0xffff_ffff)
    done
  done;
  A.free g cells;
  A.free g out

let rings =
  group ~timeout:300. "rings"
    [
      test "slots aged past 2^31 are refreshed and their waits hold" aged_slots;
      prop ~count:3
        "submissions landing at, short of and past each ring's end complete in \
         order"
        Wrap.goals Wrap.law;
    ]

let timeline =
  group ~timeout:300. "timeline"
    [
      prop ~count:20
        "values across 2^32 complete in order, ending on either queue"
        Wrap.carried Wrap.across_carry;
    ]

(* Two devices of one GPU *)

(* With no kernel driver, a process takes a GPU once: a second open is
   refused, naming the GPU taken, and the first device goes on. *)
let one_gpu_driverless t =
  (match S.open_gpu () with
  | Ok g' ->
      A.stop g' ~fault:None;
      fail "a second open of a GPU the process took"
  | Error why -> contains ~sub:"is open in this process" why);
  equal int ~msg:"the first device's next value" 1 (S.submit t [||]);
  S.wait t 1

(* Through amdgpu, a process opens a GPU as often as it likes: the devices,
   [g] and one [second] opens, share its memory. *)
let one_gpu_shared ~second g =
  let g' = match second () with Ok g' -> g' | Error why -> fail why in
  Fun.protect ~finally:(fun () -> A.stop g' ~fault:None) @@ fun () ->
  let r = device g and r' = device g' in
  let n = 4096 in
  let p = pattern n 9 in
  let staging' = Option.get (A.alloc g' Pinned n) in
  let theirs = Option.get (A.alloc g' Device n) in
  let copy d s = E.copy ~dst:(address d) ~src:(address s) n in
  H.write (host staging') p;
  go r' [| copy theirs staging' |];
  let back = Option.get (A.alloc g Pinned n) in
  let read_through view =
    H.write (host back) (String.make n '\000');
    go r [| copy back view |];
    H.read (host back) n
  in
  equal bool ~msg:"peers" true (A.peer g g');
  (match A.map_peer g g' theirs with
  | None -> fail "a device refused memory of its own GPU"
  | Some view ->
      equal string ~msg:"read through the view" p (read_through view);
      A.free g view);
  H.write (host staging') (String.make n '\000');
  go r' [| copy staging' theirs |];
  equal string ~msg:"the memory, its view unmapped" p (H.read (host staging') n);
  (* A GPU whose BAR the host does not reach whole has no Mapped memory. *)
  (match A.alloc g' Mapped n with
  | None -> ()
  | Some mapped' -> (
      match A.map_peer g g' mapped' with
      | None -> fail "a device refused Mapped memory of its own GPU"
      | Some view ->
          let p = pattern n 11 in
          H.write (host mapped') p;
          equal string ~msg:"the host's writes, read through a view" p
            (read_through view);
          A.free g view;
          A.free g' mapped'));
  (* A host page maps once per GPU: the second device's region shares the
     first's mapping, which outlives the first region's free. *)
  let page = H.pages n in
  let m = Option.get (A.map_host g page n) in
  let m' =
    require_some ~msg:"a page the other device maps" (A.map_host g' page n)
  in
  let back' = Option.get (A.alloc g' Pinned n) in
  let read_on_second view =
    H.write (host back') (String.make n '\000');
    go r' [| copy back' view |];
    H.read (host back') n
  in
  H.write page (pattern n 13);
  equal string ~msg:"the host's bytes, through the shared mapping"
    (pattern n 13) (read_on_second m');
  A.free g m;
  H.write page (pattern n 15);
  equal string ~msg:"the host's bytes, once the first region is freed"
    (pattern n 15) (read_on_second m');
  A.free g' m';
  (match A.map_host g' page n with
  | None -> fail "a page no device maps refused"
  | Some m' -> A.free g' m');
  H.free_pages page n;
  List.iter (A.free g') [ staging'; theirs; back' ];
  A.free g back

let one_gpu () =
  S.with_ @@ fun t ->
  if S.driverless () then one_gpu_driverless t
  else one_gpu_shared ~second:S.open_gpu t.g

(* Every root shows this machine: a root that links to [/] reaches the same
   GPU, whose address space the process holds once, so devices opened through
   either root share its memory, whichever opens first. The link lives in the
   suite's working directory only while the test runs: one left in a source
   tree, where the suite runs on a GPU host, has dune walk the whole
   filesystem. *)
let mirror = "mirror"

let two_roots () =
  if S.driverless () then skip ~reason:"amdgpu does not hold the GPU" ();
  (match Unix.lstat mirror with
  | _ -> Unix.unlink mirror
  | exception Unix.Unix_error (Unix.ENOENT, _, _) -> ());
  Unix.symlink "/" mirror;
  Fun.protect ~finally:(fun () -> Unix.unlink mirror) @@ fun () ->
  let through_mirror () = Rig_amd_amdgpu.open_ ~root:mirror 0 in
  S.with_ (fun t -> one_gpu_shared ~second:through_mirror t.g);
  let g = match through_mirror () with Ok g -> g | Error why -> fail why in
  Fun.protect ~finally:(fun () -> A.stop g ~fault:None) @@ fun () ->
  one_gpu_shared ~second:S.open_gpu g

(* Host memory of fewer than 64 KiB, which no device maps, copies into Device
   memory through the host's staging memory, which the process keeps: devices
   of one GPU, one after another and two at once, each map it. *)
let staged t seed =
  let n = 16384 in
  let s = pattern n seed in
  let b = buffer t n in
  put b s;
  equal string ~msg:(strf "the copy of seed %d" seed) s (get b)

let staged_devices () =
  if S.driverless () then skip ~reason:"the path maps no host memory" ();
  S.with_ (fun t -> staged t 1);
  S.with_ (fun t -> staged t 2);
  S.with_ @@ fun t ->
  let d =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_ (module A) ~name:"AMD:test-second" S.open_gpu)
  in
  Fun.protect ~finally:(fun () -> Rig.close d) @@ fun () ->
  staged t 3;
  staged { t with d } 4;
  staged t 5

(* A host page's mapping, shared by two devices of one GPU, outlives the
   first device's stop and its region's free after it: the second device's
   work reads the host's bytes through it all along. *)
let shared_after_stop () =
  if S.driverless () then skip ~reason:"the path maps no host memory" ();
  S.with_driver @@ fun g' ->
  let g = match S.open_gpu () with Ok g -> g | Error why -> fail why in
  let n = 4096 in
  let page = H.pages n in
  let first = Option.get (A.map_host g page n) in
  let shared =
    require_some ~msg:"the second device's map" (A.map_host g' page n)
  in
  let back = Option.get (A.alloc g' Pinned n) in
  let r = device g' in
  let read what seed =
    H.write page (pattern n seed);
    H.write (host back) (String.make n '\000');
    go r [| E.copy ~dst:(address back) ~src:(address shared) n |];
    equal string ~msg:what (pattern n seed) (H.read (host back) n)
  in
  read "both devices open" 21;
  A.stop g ~fault:None;
  read "the first device stopped" 22;
  A.free g first;
  read "the first region freed" 23;
  A.free g' shared;
  A.free g' back;
  H.free_pages page n

let two =
  group ~timeout:60. "one GPU"
    [
      test
        "two devices of one GPU share its memory, mapping each page once, \
         where the path allows a second device"
        one_gpu;
      test "devices of one GPU opened through two roots share its memory"
        two_roots;
      test
        "devices of one GPU, one after another and two at once, copy through \
         the host's staging memory"
        staged_devices;
      test
        "a host mapping two devices of one GPU share outlives the first's stop \
         and free"
        shared_after_stop;
    ]

(* Traces *)

(* The level the kernel driver holds GPU 0's clocks at. *)
let level () =
  match Rig_amd_amdgpu.buses () with
  | [] -> None
  | bus :: _ ->
      let file =
        "/sys/bus/pci/devices/" ^ bus ^ "/power_dpm_force_performance_level"
      in
      Option.map String.trim
        (try Some (In_channel.with_open_text file In_channel.input_all)
         with Sys_error _ -> None)

let traces =
  group ~timeout:60. "traces"
    [
      test "a device's trace buffers are made once, the host reading them"
        (fun () ->
          S.with_ @@ fun { g; _ } ->
          let c = A.capability g in
          match (c.trace (), c.trace ()) with
          | Ok t, Ok t' ->
              equal bool ~msg:"the same buffers" true (t = t');
              equal int ~msg:"engines"
                (c.gpu.shader_engines * c.gpu.xccs)
                t.engines;
              equal int ~msg:"window, a multiple of 4096" 0 (t.window mod 4096);
              H.write t.ends_host (String.make (4 * t.slots * t.engines) 'x');
              equal string ~msg:"the ends, host memory" (String.make 4 'x')
                (H.read t.ends_host 4);
              (* The level is amdgpu's file; a driver-less GPU has none. *)
              if not (S.driverless ()) then
                equal (option string) ~msg:"the GPU's clocks"
                  (Some "profile_standard") (level ())
          | Error why, _ | _, Error why -> fail why);
    ]

let () =
  S.hold ();
  exit
    (Windtrap.run "rig_amd"
       [
         gpus;
         paths;
         misuse;
         images;
         launches;
         room;
         hdp;
         failures;
         scratch;
         domains;
         work;
         code;
         launching;
         failures_hw;
         waits;
         two;
         traces;
         rings;
         memory;
         queue_order;
         timeline;
       ])
