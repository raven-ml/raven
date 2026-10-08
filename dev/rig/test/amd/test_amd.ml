(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module A = Rig_amd
module S = Rig_amd_support
module E = S.Edge
module Abi = Rig_amd_abi
module Gpu = Abi.Gpu
module Pm4 = Abi.Pm4

let strf = Printf.sprintf
let host r = Option.get (A.host r)
let address r = Option.get (A.address r)
let submit g ~v ps = E.submit g ~v ps

let le64 n =
  let b = Bytes.create 8 in
  Bytes.set_int64_le b 0 (Int64.of_int n);
  Bytes.to_string b

let le32s ns =
  let b = Bytes.create (4 * List.length ns) in
  List.iteri (fun i n -> Bytes.set_int32_le b (4 * i) (Int32.of_int n)) ns;
  Bytes.to_string b

let get64 a = Int64.to_int (String.get_int64_le (S.read a 8) 0)
let set64 a n = S.write a (le64 n)
let get32 a = Int32.to_int (String.get_int32_le (S.read a 4) 0) land 0xffff_ffff
let set32 a n = S.write a (le32s [ n ])
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
      ?(on_alloc = fun _ _ -> ()) ?(stop = fun () -> `Stopped) ?hang_ms () =
    let h =
      {
        lock = Mutex.create ();
        hdp = S.pages 4;
        calls = 0;
        live = 0;
        allocated = 0;
        last_gpu = 0;
        frees = [];
        queue_memory = [];
        queues = [];
        refused_queue = false;
        stops = 0;
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
        let m = memory ~host:(kind <> `Gpu) (S.pages n) n ~view:false in
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
          S.free_pages m.data.at m.data.bytes
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
        let doorbell = S.pages 8 in
        h.doorbells <- doorbell :: h.doorbells;
        Ok doorbell
      end
    in
    let sleep ~ms:_ =
      h.sleeps <- h.sleeps + 1;
      Option.iter (fun why -> raise (A.Fault why)) h.report
    in
    let stop () =
      h.stops <- h.stops + 1;
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
        hang_ms;
        sleep;
        stable_power = (fun () -> Ok ());
        stop;
      } )

  (* Gives back the host memory the path holds, once its device stopped: its
     own, and what the device kept, such as its timeline word. *)
  let close h =
    S.free_pages h.hdp 4;
    List.iter (fun d -> S.free_pages d 8) h.doorbells;
    List.iter (fun d -> S.free_pages d.at d.bytes) h.held;
    h.doorbells <- [];
    h.held <- [];
    h.closed <- true

  let device ?key ?reaches ?gpu ?lds ?on_alloc ?stop ?hang_ms () =
    let h, p = path ?key ?reaches ?gpu ?lds ?on_alloc ?stop ?hang_ms () in
    match A.make p with Ok g -> (h, g) | Error why -> fail why

  let with_device ?gpu ?lds f =
    let h, g = device ?gpu ?lds () in
    Fun.protect
      ~finally:(fun () ->
        A.stop g;
        close h)
      (fun () -> f h g)

  let compute h = List.nth h.queues 0
  let copy h = List.nth h.queues 1

  (* The position a queue reads its work up to: in words on a PM4 ring, bytes on
     an SDMA ring, packets on an AQL ring. *)
  let position q = get64 q.write

  (* Reaches [v]: what the queue that releases [v] does. *)
  let reach g v = set64 (host (A.word g)) v
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
    A.stop g;
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
   timeline word. *)
let stop_gives_back () =
  let h, g = Host.device () in
  let word = address (A.word g) in
  equal answer ~msg:"submit" `Ok (submit g ~v:1 [||]);
  A.stop g;
  equal int ~msg:"stops" 1 h.stops;
  equal bool ~msg:"the word given back" false (List.mem word h.frees);
  equal int ~msg:"the memory left: the word" 1 h.live;
  Host.close h

let facts () =
  List.iter
    (fun (g, kind, aql) ->
      let msg = Gpu.processor g in
      Host.with_device ~gpu:g @@ fun h d ->
      let c = A.capability d in
      equal string ~msg (Gpu.processor g) (A.arch d);
      equal int ~msg (1 lsl 34) (A.budget d);
      equal (list string) ~msg [ "COMPUTE:0"; "COPY:0" ] (A.queues d);
      equal bool ~msg:"completion is the word" true (A.completion d = `Store);
      equal bool ~msg:"blocks" true (A.blocks d = `Returns);
      equal bool ~msg:"waits on objects" false (A.waits_on d `Object);
      equal bool ~msg:"waits on words as on the host" (A.waits_on d `Host)
        (A.waits_on d `Store);
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
      equal bool ~msg:"the capability's key" true
        (Option.is_some
           (Type.Id.provably_equal A.capability_key Abi.Capability.key)))
    [ (r9700, "PM4", false); (mi300, "AQL", true) ]

(* [peer g g'] is whether [map_peer g g'] maps [g']'s [`Device] memory. *)
let peers () =
  let other : Host.mem Type.Id.t = Type.Id.make () in
  let case name ?key ~reaches expected =
    let h, g = Host.device ~reaches () and h', g' = Host.device ?key () in
    let r = Option.get (A.alloc g' `Device 64) in
    equal bool ~msg:(name ^ ": peer") expected (A.peer g g');
    let view = A.map_peer g g' r in
    equal bool ~msg:(name ^ ": map_peer") expected (Option.is_some view);
    Option.iter (A.free g) view;
    equal bool ~msg:(name ^ ": itself") false (A.peer g g);
    A.free g' r;
    List.iter
      (fun (h, g) ->
        A.stop g;
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
  | Ok (`Loaded _) -> fail "an image the device placed itself"
  | Ok (`Place (n, lay)) ->
      let r = Option.get (A.alloc g `Device n) in
      let m, bytes = lay r in
      (m, r, bytes)

(* Misuse *)

let misuse () =
  Host.with_device @@ fun _ g ->
  Host.with_device @@ fun _ g' ->
  let raises fn f =
    raises_match ~msg:fn (Exn.invalid_arg ~substring:("Rig_amd." ^ fn ^ ":")) f
  in
  let kernels = Lazy.force kernels_bin in
  raises "alloc" (fun () -> A.alloc g `Pinned 0);
  let r = Option.get (A.alloc g `Pinned 4096) in
  let r' = Option.get (A.alloc g' `Pinned 4096) in
  let page = S.pages 4096 in
  let mapped = Option.get (A.map_host g page 4096) in
  raises "free" (fun () -> A.free g r');
  raises "map_peer" (fun () -> A.map_peer g g r);
  raises "map_peer" (fun () -> A.map_peer g g' r);
  raises "map_host" (fun () -> A.map_host g page 0);
  raises "free" (fun () -> A.free g (A.word g));
  let view = Option.get (A.map_peer g g' r') in
  raises "free" (fun () -> A.free g' view);
  A.free g view;
  raises "free" (fun () -> A.free g view);
  let m, code, _ = load g kernels in
  raises "unload" (fun () -> A.unload g' m);
  A.unload g m;
  raises "entry" (fun () -> A.entry m "empty");
  raises "unload" (fun () -> A.unload g m);
  A.free g code;
  A.free g r;
  raises "free" (fun () -> A.free g r);
  A.free g mapped;
  A.free g' r';
  S.free_pages page 4096

(* What rig_amd_room answers for a part the device does not run. *)
let c_room () =
  let never = room_answer in
  Host.with_device @@ fun _ g ->
  equal never ~msg:"a part of 16 words" `Fits
    (E.room g [| E.raw ~queue:0 ~words:16 () |]);
  equal never ~msg:"a copy on the compute queue" `Never
    (E.room g [| E.raw ~queue:0 ~copy:64 () |]);
  equal never ~msg:"words and a fill" `Never
    (E.room g [| E.raw ~queue:0 ~words:4 ~fill:true () |]);
  equal never ~msg:"after its own index" `Never
    (E.room g [| E.raw ~queue:1 ~after:[| 0 |] () |]);
  equal never ~msg:"a queue of no index" `Never
    (E.room g [| E.raw ~queue:2 () |]);
  equal never ~msg:"a negative queue" `Never
    (E.room g [| E.raw ~queue:(-1) () |]);
  Host.with_device ~gpu:mi300 @@ fun _ g ->
  equal never ~msg:"AQL: a packet" `Fits
    (E.room g [| E.raw ~queue:0 ~words:16 () |]);
  equal never ~msg:"AQL: part of a packet" `Never
    (E.room g [| E.raw ~queue:0 ~words:15 () |])

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
  | Ok (`Place (n, _)) ->
      equal int ~msg:"the image's size" (Abi.Code_object.size co) n
  | Ok (`Loaded _) | Error _ -> fail "no image to place");
  match load g bin with
  | m, r, bytes ->
      at_most int ~msg:"the image's bytes" ~than:(Abi.Code_object.size co)
        (String.length bytes);
      List.iter
        (fun name ->
          let k = Option.get (Abi.Code_object.kernel co name) in
          equal (option int) ~msg:name
            (Some (address r + k.descriptor))
            (A.entry m name))
        [ "empty"; "double_index"; "spin"; "wild" ];
      equal (option int) ~msg:"no kernel" None (A.entry m "nothing");
      equal (option int) ~msg:"a symbol of no kernel" None
        (A.entry m "__clang_ocl_kern_imp_spin");
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

let zeros = lazy (Array.make (1 lsl 20) 0)

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
          (3, map (fun n -> Words n) (of_list [ 1; 64; 1 lsl 16; 1 lsl 20 ]));
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
  (Option.get (A.alloc g `Pinned 4096), Option.get (A.alloc g `Pinned 4096))

(* The fresh device: one that ran nothing, which the law compares with. *)
let fresh =
  fixture
    ~teardown:(fun (h, g, _) ->
      A.stop g;
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
type sys_held = { devices : sys_machine; dev : int; r : A.region }

let device_at (ds : sys_machine) d =
  let _, g, _ = ds.(d) in
  g

let give_back s =
  let g = device_at s.devices s.dev in
  A.free g s.r

let machine =
  abstract "machine" ~release:(fun (ds : sys_machine) ->
      Array.iter
        (fun (h, g, _) ->
          A.stop g;
          Host.close h)
        ds)

let held =
  abstract "r"
    ~pp:(fun ppf r ->
      Format.fprintf ppf "%s memory of %d held by %d%s"
        (if r.mapped then "Mapped" else "Pinned")
        r.owner r.holder
        (if r.live then "" else ", given back"))
    ~release:(fun s -> try give_back s with Invalid_argument _ -> ())

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
  Array.iter (fun ((h : Host.t), _, _) -> set32 h.hdp 0xffff_ffff) ds;
  let _, g, last = ds.(d) in
  equal room_answer ~msg:"room" `Fits (E.room g [||]);
  incr last;
  equal answer ~msg:"submit" `Ok (submit g ~v:!last [||]);
  Host.reach g !last;
  List.filter
    (fun i ->
      let (h : Host.t), _, _ = ds.(i) in
      get32 h.hdp = 0)
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
        let kind = if mapped then `Mapped else `Pinned in
        let r = Option.get (A.alloc (device_at ds d) kind 64) in
        { devices = ds; dev = d; r });
    command "map_peer"
      ~pre:(fun d r -> r.live && d <> r.holder)
      (device_index @-> held ^-> makes held)
      (fun d r -> hold r.machine { r with holder = d; live = true })
      (fun d s ->
        let ds = s.devices in
        match A.map_peer (device_at ds d) (device_at ds s.dev) s.r with
        | Some r -> { s with dev = d; r }
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
        (g', Option.get (A.alloc g' `Mapped 64)))
  in
  Fun.protect
    ~finally:(fun () ->
      List.iter (fun (g', r) -> A.free g' r) peers;
      Array.iter
        (fun (h, g) ->
          A.stop g;
          Host.close h)
        ds)
    (fun () -> f ds.(0) peers)

let seven_others () =
  with_nine @@ fun (_, g) peers ->
  let own = Option.get (A.alloc g `Mapped 64) in
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
  let own = Option.get (A.alloc g `Mapped 64) in
  set32 h.hdp 0xffff_ffff;
  equal answer ~msg:"submit" `Ok (submit g ~v:1 [||]);
  equal int ~msg:"its own HDP register" 0 (get32 h.hdp);
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

(* The words of [q]'s ring from position [p] to [p'], a PM4 ring's positions
   being words. *)
let handed (q : Host.queue) p p' =
  let size = q.bytes / 4 in
  List.init (p' - p) (fun i -> get32 (at q.ring (4 * ((p + i) mod size))))

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
  let header s = get32 (at q.ring (64 * (s mod slots))) land 0xff in
  let ours s =
    List.exists is_marker
      (List.init 15 (fun i ->
           get32 (at q.ring ((64 * (s mod slots)) + (4 * (i + 1))))))
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
  A.stop g;
  Host.close h

(* A fault is the device's for good: every sleep after the path reported it
   raises it again, whether the word moved past [seen] or the path would now
   return quietly. *)
let sleeps_after_fault () =
  Host.with_device @@ fun h g ->
  let why = "memory fault at 0x0" in
  h.report <- Some why;
  raises (A.Fault why) (fun () -> A.sleep g ~seen:0 ~still_ms:1);
  h.report <- None;
  raises ~msg:"the word at seen" (A.Fault why) (fun () ->
      A.sleep g ~seen:0 ~still_ms:1);
  raises ~msg:"the word past seen" (A.Fault why) (fun () ->
      A.sleep g ~seen:5 ~still_ms:1)

(* [free] never raises the path's faults: a region freed while the path fails is
   freed, and a second free of it is misuse. *)
let free_through_fault () =
  Host.with_device @@ fun h g ->
  let r = Option.get (A.alloc g `Pinned 64) in
  h.free_fault <- Some "the host frees nothing now";
  A.free g r;
  h.free_fault <- None;
  raises_match ~msg:"a second free" (Exn.invalid_arg ~substring:"Rig_amd.free")
    (fun () -> A.free g r)

(* [stop] writes the last value into the word only once the path stopped every
   queue, and never raises; [free] and [signaled] answer after it. *)
let stops () =
  let case name path_stop word =
    let h, g = Host.device ~stop:path_stop () in
    let r = Option.get (A.alloc g `Pinned 64) in
    let page = S.pages 64 in
    let m = Option.get (A.map_host g page 64) in
    equal answer ~msg:name `Ok (submit g ~v:1 [||]);
    equal answer ~msg:name `Ok (submit g ~v:2 [||]);
    A.stop g;
    equal int ~msg:(name ^ ": the word") word (A.signaled g);
    equal int ~msg:(name ^ ": stops") 1 h.stops;
    A.free g r;
    A.free g m;
    S.free_pages page 64;
    Host.close h
  in
  case "the path stopped" (fun () -> `Stopped) 2;
  case "the path may run a queue" (fun () -> `Unknown) 0;
  case "the path failed" (fun () -> raise (A.Fault "lost")) 0

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

let at_the_end =
  Gen.with_pp
    (fun ppf (r, a, b) ->
      Format.fprintf ppf "%d words left; calls of %d and %d" r a b)
    Gen.(triple (int_range 1 40) (int_range 1 24) (int_range 1 24))

(* Progress bounds

   A path that bounds work's progress ([hang_ms]) makes [sleep] raise once a
   value is outstanding and the word has not moved for that long. The host
   path's [sleep] returns at once, so these loops spin; [ms] of CPU time is at
   most [ms] of the clock. *)

let spin ~ms f =
  let t0 = S.now_ns () in
  while S.now_ns () - t0 < ms * 1_000_000 do
    f ()
  done

let hangs () =
  let h, g = Host.device ~hang_ms:50 () in
  equal answer ~msg:"submit" `Ok (submit g ~v:1 [||]);
  A.sleep g ~seen:0 ~still_ms:1;
  (match spin ~ms:2000 (fun () -> A.sleep g ~seen:0 ~still_ms:1) with
  | () -> fail "no Fault after 2 s of a value making no progress"
  | exception A.Fault why -> contains ~msg:"the report" ~sub:"50 ms" why);
  A.stop g;
  Host.close h

let idle () =
  let h, g = Host.device ~hang_ms:50 () in
  spin ~ms:150 (fun () -> A.sleep g ~seen:0 ~still_ms:1);
  equal answer ~msg:"submit" `Ok (submit g ~v:1 [||]);
  A.sleep g ~seen:0 ~still_ms:1;
  A.stop g;
  Host.close h

let moving () =
  let h, g = Host.device ~hang_ms:50 () in
  for v = 1 to 8 do
    equal answer ~msg:"submit" `Ok (submit g ~v [||])
  done;
  for v = 1 to 8 do
    spin ~ms:20 (fun () -> A.sleep g ~seen:(v - 1) ~still_ms:1);
    Host.reach g v
  done;
  A.stop g;
  Host.close h

let unbounded () =
  let h, g = Host.device () in
  equal answer ~msg:"submit" `Ok (submit g ~v:1 [||]);
  spin ~ms:150 (fun () -> A.sleep g ~seen:0 ~still_ms:1);
  A.stop g;
  Host.close h

let bounds () =
  List.iter
    (fun n ->
      let h, p = Host.path ~hang_ms:n () in
      raises_match ~msg:(strf "hang_ms %d" n)
        (Exn.invalid_arg ~substring:"Rig_amd.make") (fun () -> A.make p);
      Host.close h)
    [ 0; -1; min_int ]

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
      A.stop g;
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

let progress =
  group ~timeout:30. "progress"
    [
      test "a value that makes no progress for hang_ms is a fault" hangs;
      test "an idle device never hangs, nor its next value at once" idle;
      test "values reached more often than hang_ms are no fault" moving;
      test "without hang_ms a value making no progress is no fault" unbounded;
      test "make raises on a hang_ms below 1" bounds;
    ]

(* A device waits on other words only where [waits_on] says so, and on at most
   255 per submission: a submission past either fails as a fill's failure
   does. *)
let waits_refused () =
  List.iter
    (fun n ->
      Host.with_device @@ fun _ g ->
      let msg = strf "%d waits" n in
      equal bool ~msg:"waits on words" false (A.waits_on g `Store);
      let w = host (A.word g) in
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
      test "every sleep after a fault raises it" sleeps_after_fault;
      test "free returns when the path fails to free" free_through_fault;
      test "stop writes the last value only once the path stopped its queues"
        stops;
    ]

(* End once from two domains: whatever the order, the first free, unmap or
   unload returns and every later one raises; the path gets each back once. *)

let shared_host =
  fixture
    ~teardown:(fun (h, g) ->
      A.stop g;
      Host.close h)
    (fun () -> Host.device ())

type ended = { mutable ended : bool }

let end_model m =
  if m.ended then invalid_arg "ended";
  m.ended <- true

let ends_once name ~make ~finish ~release =
  let v =
    abstract name ~release:(fun x ->
        try release x with Invalid_argument _ -> ())
  in
  [
    command "make" (Gen.unit @-> makes v) (fun () -> { ended = false }) make;
    command "end" (v ^-> returns unit) end_model finish;
  ]

(* Each value carries the shared device: the fixture is read on the test's
   domain only. *)
let allocation_commands =
  ends_once "a"
    ~make:(fun () ->
      let g = snd (shared_host ()) in
      (g, Option.get (A.alloc g `Pinned 64)))
    ~finish:(fun (g, r) -> A.free g r)
    ~release:(fun (g, r) -> A.free g r)

(* A mapping of its own page, given back once the program ends. *)
let mapping_commands =
  ends_once "m"
    ~make:(fun () ->
      let g = snd (shared_host ()) and p = S.pages 64 in
      (g, p, Option.get (A.map_host g p 64)))
    ~finish:(fun (g, _, r) -> A.free g r)
    ~release:(fun (g, p, r) ->
      Fun.protect ~finally:(fun () -> S.free_pages p 64) (fun () -> A.free g r))

let image_commands =
  ends_once "i"
    ~make:(fun () ->
      let g = snd (shared_host ()) in
      let m, r, _ = load g (Lazy.force kernels_bin) in
      (g, m, r))
    ~finish:(fun (g, m, _) -> A.unload g m)
    ~release:(fun (g, m, r) ->
      Fun.protect ~finally:(fun () -> A.free g r) (fun () -> A.unload g m))

(* [sleep] runs while another domain submits: a device whose submissions
   complete at once, the caller serialising its submits, answers as some order
   of the calls. *)
type sleeper = { g : A.t; h : Host.t; lock : Mutex.t; mutable v : int }

let sleeper =
  abstract "s" ~release:(fun s ->
      A.stop s.g;
      Host.close s.h)

let sleep_commands =
  [
    command "start"
      (Gen.unit @-> makes sleeper)
      (fun () -> ref 0)
      (fun () ->
        let h, g = Host.device ~hang_ms:60_000 () in
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
      stateful ~count:30 ~domains:2
        "an allocation freed from two domains is freed once" allocation_commands;
      stateful ~count:30 ~domains:2
        "a mapping unmapped from two domains is unmapped once" mapping_commands;
      stateful ~count:30 ~domains:2
        "an image unloaded from two domains is unloaded once" image_commands;
    ]

(* On a GPU *)

(* Work through rig, on a device [S.gpu] opened *)

let buffer ?(memory = Rig.Buffer.Device) g n =
  Rig.Buffer.create ~memory (S.rig g) n

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
let run g ps =
  Rig.wait (S.rig g) (S.submit g ps);
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

let program ?(of_ = kernels) g =
  match Rig.Program.load (S.rig g) (Lazy.force of_).binary with
  | Ok p -> p
  | Error why -> fail why

(* The words of a dispatch of [p]'s [name], its arguments [args] in a buffer of
   their own. *)
let kernel_words ?of_ g p name ~groups args =
  let a = buffer ~memory:Pinned g 4096 in
  put a args;
  live := a :: !live;
  let entry f = Option.get (Rig.Program.entry p f) in
  dispatch ?of_ (A.capability g).gpu entry name ~args:(addr a) ~groups

let kernel ?of_ ?after g p name ~groups args =
  S.words_part ~queue:"COMPUTE:0" ?after
    (kernel_words ?of_ g p name ~groups args)

let multiples k n = le32s (List.init n (fun i -> k * i))
let doubled = multiples 2

let work =
  group ~timeout:60. "work"
    [
      test "an empty submission releases its value" (fun () ->
          S.with_gpu @@ fun g ->
          run g [||];
          equal int ~msg:"the word" 1 (A.signaled g));
      test "an idle device stops with its word at the last value" (fun () ->
          let g = S.gpu () in
          run g [||];
          run g [||];
          S.stop g;
          equal int ~msg:"the word" 2 (A.signaled g));
      test "an opened GPU's facts are its capability's" (fun () ->
          S.with_gpu @@ fun g ->
          let c = A.capability g in
          equal string ~msg:"arch" (Gpu.processor c.gpu) (A.arch g);
          greater int ~msg:"budget" ~than:0 (A.budget g);
          greater int ~msg:"clock" ~than:0 c.clock_hz;
          equal bool ~msg:"AQL" (c.gpu.xccs > 1)
            (match c.compute with Aql _ -> true | Pm4 -> false));
      test "an allocation past the GPU's memory is None" (fun () ->
          S.with_gpu @@ fun g ->
          equal bool ~msg:"none" true
            (Option.is_none (A.alloc g `Device (2 * A.budget g))));
    ]

(* Memory *)

(* The device the property tests share, stopped when the run ends. *)
let shared = fixture ~teardown:S.stop S.gpu
let kinds = Rig.Buffer.[ Device; Pinned; Mapped ]

let pp_kind ppf k =
  Format.pp_print_string ppf
    Rig.Buffer.(
      match k with
      | Device -> "Device"
      | Pinned -> "Pinned"
      | Mapped -> "Mapped")

let kind = Gen.of_list ~pp:pp_kind kinds
let size = Gen.of_list ~pp:Format.pp_print_int [ 1; 7; 4096; (1 lsl 20) + 7 ]
let offset = Gen.of_list ~pp:Format.pp_print_int [ 0; 1; 4095 ]

let round_trip (ka, kb, n, (oa, ob)) =
  let g = shared () in
  let src = buffer ~memory:Pinned g n and dst = buffer ~memory:Pinned g n in
  let a = view (buffer ~memory:ka g (oa + n)) oa n in
  let b = view (buffer ~memory:kb g (ob + n)) ob n in
  let bytes = pattern n (n + oa) in
  put src bytes;
  run g [| copy ~dst:a src |];
  run g [| copy ~dst:b a |];
  run g [| copy ~dst b |];
  equal string ~msg:"the bytes" bytes (get dst)

(* Copies of the copy engine's largest packet and its neighbours: the bytes
   before each copy's end arrive, those after stay as they were. Two windows are
   read: the start, and the packet's end. *)
let around_the_packet () =
  let g = shared () in
  let max = Abi.Sdma.max_copy (A.capability g).gpu in
  let w = 4096 in
  let n = max + (w / 2) in
  let src = buffer ~memory:Pinned g n and dst = buffer g n in
  let zeros = buffer ~memory:Pinned g w and back = buffer ~memory:Pinned g w in
  put zeros (String.make w '\000');
  let windows = [ 0; max - (w / 2) ] in
  List.iteri (fun i o -> put (view src o w) (pattern w (i + 1))) windows;
  List.iter
    (fun k ->
      run g
        (Array.of_list
           (List.map (fun o -> copy ~dst:(view dst o w) zeros) windows));
      run g [| copy ~dst:(view dst 0 k) (view src 0 k) |];
      List.iteri
        (fun i o ->
          run g [| copy ~dst:back (view dst o w) |];
          let arrived = Int.max 0 (Int.min w (k - o)) in
          let expected =
            String.sub (pattern w (i + 1)) 0 arrived
            ^ String.make (w - arrived) '\000'
          in
          equal string ~msg:(strf "%d bytes, at %d" k o) expected (get back))
        windows)
    [ max - 1; max; max + 1 ]

(* Host and GPU writes to the same memory, round after round, are read by the
   next submission and by the host. *)
let rewrites () =
  let g = shared () in
  let n = 4096 in
  let mapped = buffer ~memory:Mapped g n
  and pinned = buffer ~memory:Pinned g n in
  let vram = buffer g n and out = buffer ~memory:Pinned g n in
  for round = 1 to 8 do
    let msg = strf "round %d" round in
    let p = pattern n round in
    put mapped p;
    run g [| copy ~dst:out mapped |];
    equal string ~msg:(msg ^ ": the host's write through the BAR") p (get out);
    let p' = pattern n (round + 100) in
    put pinned p';
    run g [| copy ~dst:vram pinned |];
    run g [| copy ~dst:mapped vram |];
    equal string
      ~msg:(msg ^ ": the GPU's write, read through the BAR")
      p' (get mapped)
  done

let memory =
  group ~timeout:120. "memory"
    [
      prop ~count:30 "copies through any two kinds of memory are the identity"
        Gen.(quad kind kind size (pair offset offset))
        round_trip;
      test "copies around the copy engine's packet size are the identity"
        around_the_packet;
      test "host and GPU writes to one memory are read round after round"
        rewrites;
    ]

let spin g p flag n = kernel g p "spin" ~groups:1 (le64 (addr flag) ^ le64 n)

(* Code placed where other code ran runs as placed: each program is unloaded
   once unreachable and its work done, and the next load reuses the memory. *)
let stale_code () =
  S.with_gpu @@ fun g ->
  let out = buffer ~memory:Pinned g (4 * 64) in
  let load of_ =
    let p = program ~of_ g in
    run g [| kernel ~of_ g p "double_index" ~groups:1 (le64 (addr out)) |];
    let at = Option.get (Rig.Program.entry p "double_index") in
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
          S.with_gpu @@ fun g ->
          let p = program g in
          let out = buffer ~memory:Pinned g (4 * 256) in
          run g [| kernel g p "double_index" ~groups:4 (le64 (addr out)) |];
          equal string ~msg:"out" (doubled 256) (get out));
      test "code placed where other code ran runs as placed" stale_code;
      test "parts on two queues run in their after order" (fun () ->
          S.with_gpu @@ fun g ->
          let p = program g in
          let out = buffer g (4 * 256)
          and back = buffer ~memory:Pinned g (4 * 256) in
          run g
            [|
              kernel g p "double_index" ~groups:4 (le64 (addr out));
              copy ~after:[| 0 |] ~dst:back out;
            |];
          equal string ~msg:"back" (doubled 256) (get back));
      test "a fill places its words" (fun () ->
          S.with_gpu @@ fun g ->
          let word = buffer ~memory:Pinned g 8 in
          put word (le64 0);
          let ws = words (Pm4.write_data (Memory (addr word)) 0xc0ffee) in
          let f = S.fill (A.capability g) ws ~bytes:64 in
          run g
            [|
              S.fill_part ~queue:"COMPUTE:0" f ~units:(Array.length ws)
                ~bytes:64;
            |];
          equal string ~msg:"word" (le64 0xc0ffee) (get word));
      test "long work is no fault" (fun () ->
          S.with_gpu @@ fun g ->
          let p = program g in
          let flag = buffer ~memory:Pinned g 8 in
          put flag (le64 0);
          run g [| spin g p flag 150_000 |];
          equal string ~msg:"flag" (le32s [ 1 ]) (String.sub (get flag) 0 4));
      test "a device stopped while its work runs stops it" (fun () ->
          let g = S.gpu () in
          let p = program g in
          let flag = buffer ~memory:Pinned g 8 in
          put flag (le64 0);
          let v = S.submit g [| spin g p flag 1_500_000 |] in
          S.stop g;
          equal int ~msg:"the word" v (A.signaled g);
          S.still ~msg:"flag (sampled)" string (String.make 8 '\000')
            (fun () -> get flag)
            ~ms:100);
    ]

(* Work in order on one queue, and its writes for every reader

   Parts on one queue run in array order: a compute part after another sees its
   writes, whatever the sizes and whether a copy part sits between them. And
   whatever wrote memory last, a copy, a kernel or the host, the next reader of
   it, the host, a copy or a kernel, reads that write, round after round over
   the same memory. *)

(* The shared device's program of fixtures/work.cl, loaded once. *)
let work_program = lazy (program ~of_:work_code (shared ()))

let work_words g name ~groups args =
  kernel_words ~of_:work_code g (Lazy.force work_program) name ~groups args

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
      (let g = shared () in
       let pinned () = buffer ~memory:Pinned g (4 * most) in
       let input = pinned () and garbage = pinned () and out = pinned () in
       put input
         (le32s (List.init most (fun i -> i * 2654435761 land 0xffff_ffff)));
       put garbage (String.make (4 * most) '\xee');
       {
         input;
         garbage;
         out;
         bufs = Array.init buffers (fun _ -> buffer g (4 * most));
       })

  let law c =
    let g = shared () in
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
    run g
      (Array.map
         (fun b -> copy ~dst:(part b) (part s.garbage))
         (Array.append s.bufs [| s.out |]));
    run g [| copy ~dst:(part s.bufs.(0)) (part s.input) |];
    if c.after_copy then
      run g [| copy ~dst:(view s.out 0 4) (view s.garbage 0 4) |]
    else run g [| S.words_part ~queue:"COMPUTE:0" [| pm4_nop |] |];
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
          work_words g "inc"
            ~groups:((n + 63) / 64)
            (le64 (addr out) ^ le64 (addr !src) ^ le32s [ n ])
        in
        let p =
          if fill then
            let f = S.fill (A.capability g) ws ~bytes:0 in
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
    run g (Array.of_list (List.rev !parts));
    let input = get (part s.input) in
    let expected = Bytes.create bytes in
    for i = 0 to n - 1 do
      Bytes.set_int32_le expected (4 * i)
        (Int32.add (String.get_int32_le input (4 * i)) (Int32.of_int k))
    done;
    equal string ~msg:"the last part's words" (Bytes.to_string expected)
      (get (part s.out))
end

(* Each reader reads the last write, round after round over the same memory: the
   host a kernel's writes to Pinned and Mapped memory, a copy a kernel's writes
   to Device memory, and a kernel a copy's, in one submission and in the
   next. *)
let readers () =
  let g = shared () in
  let n = 4096 in
  let bytes = 4 * n in
  let src = buffer ~memory:Pinned g bytes
  and pinned = buffer ~memory:Pinned g bytes in
  let mapped = buffer ~memory:Mapped g bytes and vram = buffer g bytes in
  let back = buffer ~memory:Pinned g bytes in
  let kernel ?after dst src =
    S.words_part ~queue:"COMPUTE:0" ?after
      (work_words g "copy" ~groups:1
         (le64 (addr dst) ^ le64 (addr src) ^ le32s [ n; 0 ]))
  in
  let fresh round k =
    let p = pattern bytes ((8 * round) + k) in
    put src p;
    p
  in
  for round = 1 to 8 do
    let msg what = strf "round %d: %s" round what in
    let p = fresh round 0 in
    run g [| kernel pinned src; kernel mapped src |];
    equal string
      ~msg:(msg "the host reads a kernel's Pinned writes")
      p (get pinned);
    equal string
      ~msg:(msg "the host reads a kernel's Mapped writes")
      p (get mapped);
    let p = fresh round 1 in
    run g [| kernel vram src; copy ~after:[| 0 |] ~dst:back vram |];
    equal string
      ~msg:(msg "a copy reads a kernel's writes, after it")
      p (get back);
    let p = fresh round 2 in
    run g [| kernel vram src |];
    run g [| copy ~dst:back vram |];
    equal string
      ~msg:(msg "a copy reads a kernel's writes, a value later")
      p (get back);
    let p = fresh round 3 in
    run g [| copy ~dst:vram src; kernel ~after:[| 0 |] back vram |];
    equal string
      ~msg:(msg "a kernel reads a copy's writes, after it")
      p (get back);
    let p = fresh round 4 in
    run g [| copy ~dst:vram src |];
    run g [| kernel back vram |];
    equal string
      ~msg:(msg "a kernel reads a copy's writes, a value later")
      p (get back)
  done

let queue_order =
  group ~timeout:120. "in order"
    [
      prop ~count:24
        "compute parts run in array order, each reading its predecessor's \
         writes"
        (Gen.with_pp Chain.pp Chain.gen)
        Chain.law;
      test "every reader reads the last write, round after round" readers;
    ]

(* Through the C entries, on a device rig does not drive *)

(* A device of GPU 0 opened beside [S.gpu]'s, for work handed to its C entries
   directly, with values the test numbers. *)
let raw =
  fixture ~teardown:A.stop (fun () ->
      if Rig_amd_amdgpu.count () = 0 then
        skip ~reason:"the machine has no AMD GPU" ();
      S.hold_gpu ();
      match Rig_amd_amdgpu.open_ 0 with Ok g -> g | Error why -> failwith why)

(* A device's values, numbered as it submits them. *)
type run = { g : A.t; mutable v : int }

let runs = Hashtbl.create 1

let run_of g =
  match Hashtbl.find_opt runs (A.self g) with
  | Some r -> r
  | None ->
      let r = { g; v = A.signaled g } in
      Hashtbl.add runs (A.self g) r;
      r

let device g = { g; v = 0 }

let go r ps =
  r.v <- r.v + 1;
  equal answer ~msg:"submit" `Ok (submit r.g ~v:r.v ps);
  S.wait r.g r.v

let entry m f = Option.get (A.entry m f)

(* [m]'s image laid over new memory [code], and the part that copies it there
   from staging memory the host wrote. *)
let laid ?(of_ = kernels) r =
  let m, code, bytes = load r.g (Lazy.force of_).binary in
  let n = String.length bytes in
  let staging = Option.get (A.alloc r.g `Pinned n) in
  S.write (host staging) bytes;
  (m, code, E.copy ~dst:(address code) ~src:(address staging) n)

let arguments r values =
  let args = Option.get (A.alloc r.g `Pinned 4096) in
  S.write (host args) (String.concat "" (List.map le64 values));
  args

(* A submit answers RIG_FAILED for a fill's failure, runs none of the parts,
   reaches the value anyway, and answers the same failure after. *)
let failures_hw =
  group ~timeout:60. "C failures"
    [
      test "a fill past its declaration fails, and the word still moves"
        (fun () ->
          S.with_gpu @@ fun g ->
          let word = Option.get (A.alloc g `Pinned 8) in
          S.write (host word) (le64 0);
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
          S.wait g 1;
          equal string ~msg:"the fill's word" (le64 0) (S.read (host word) 8);
          equal answer ~msg:"the next" (`Failed why) (submit g ~v:2 [||]));
      test "a failed submission runs none of its parts" (fun () ->
          S.with_gpu @@ fun g ->
          let n = 4096 in
          let src = Option.get (A.alloc g `Pinned n) in
          let dst = Option.get (A.alloc g `Pinned n) in
          S.write (host src) (pattern n 1);
          S.write (host dst) (String.make n '\000');
          let f = S.fill ~code:5 (A.capability g) [||] ~bytes:0 in
          equal answer ~msg:"submit" (`Failed "a fill on COPY:0 failed with 5")
            (submit g ~v:1
               [|
                 E.copy ~dst:(address dst) ~src:(address src) n;
                 E.fill ~queue:"COPY:0" f ~units:0 ~bytes:0;
               |]);
          S.wait g 1;
          equal string ~msg:"dst" (String.make n '\000') (S.read (host dst) n));
      test "a value failed behind running work is reached once it ends"
        (fun () ->
          let g = S.gpu () in
          let r = device g in
          let m, _, upload = laid r in
          go r [| upload |];
          let flag = Option.get (A.alloc g `Pinned 8) in
          S.write (host flag) (le64 0);
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
          S.wait g 3;
          equal string ~msg:"the work's flag" (le32s [ 1 ])
            (S.read (host flag) 4);
          S.stop g;
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
    let res = Option.get (A.alloc g `Pinned (4 * slots)) in
    let src = Option.get (A.alloc g `Pinned (8 * slots)) in
    let dst = Option.get (A.alloc g `Pinned (8 * slots)) in
    S.write (host res) (String.make (4 * slots) '\000');
    S.write (host src)
      (String.concat "" (List.init slots (fun i -> le64 (i + 1))));
    S.write (host dst) (String.make (8 * slots) '\000');
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

  (* Runs [shapes] on a fresh device of the GPU: the word never moves back,
     every value is reached, and the slots hold what the last value that wrote
     each wrote. *)
  let run g shapes =
    let k = sink g in
    let rec room ps =
      match E.room g ps with
      | `Fits -> ()
      | `Never -> fail "Never"
      | `Later ->
          S.wait g (A.signaled g + 1);
          room ps
    in
    let seen = ref 0 in
    List.iteri
      (fun i shape ->
        let ps, _ = parts g k (i + 1) shape in
        room ps;
        equal answer ~msg:"submit" `Ok (submit g ~v:(i + 1) ps);
        let w = A.signaled g in
        at_least int ~msg:"the word" ~than:!seen w;
        seen := w)
      shapes;
    let last = List.length shapes in
    S.wait g last;
    let res = Bytes.make (4 * slots) '\000'
    and dst = Bytes.make (8 * slots) '\000' in
    List.iteri
      (fun i shape ->
        let v = i + 1 and o = (i + 1) mod slots in
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
      (S.read (host k.res) (4 * slots));
    equal string ~msg:"the copies" (Bytes.to_string dst)
      (S.read (host k.dst) (8 * slots))

  let law goals = S.with_gpu @@ fun g -> run g (plan (A.capability g).gpu goals)
end

(* Values complete in order *)

module Order = struct
  (* A part: a copy on COPY:0, or the kernel copy on COMPUTE:0, which copies
     words after a delay. *)
  type part = {
    copy : bool;
    delay : int;
    src : int;
    so : int;
    dst : int;
    do_ : int;
    n : int; (* bytes; a kernel's are whole words at word offsets *)
    after : int list;
  }

  let buffers = 3
  let size = 65536

  let pp_part ppf p =
    Format.fprintf ppf "{%s%s %d@%d -> %d@%d n=%d after=[%s]}"
      (if p.copy then "COPY" else "COMPUTE")
      (if p.delay > 0 then strf " +%d" p.delay else "")
      p.src p.so p.dst p.do_ p.n
      (String.concat ";" (List.map string_of_int p.after))

  let pp ppf subs =
    Format.pp_print_list ~pp_sep:Format.pp_print_space
      (fun ppf ps ->
        Format.fprintf ppf "[%a]"
          (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_part)
          ps)
      ppf subs

  let initial b = Bytes.of_string (pattern size (b * 61))

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

  (* The result is one: every two parts that conflict are ordered, and no part
     copies over its own source. *)
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
    let last = match List.rev ps with p :: _ -> Some p.copy | [] -> m.last in
    let queues = List.sort_uniq compare (List.map (fun p -> p.copy) ps) in
    cover "a submission of no parts after one released on COPY"
      (ps = [] && m.last = Some true);
    cover "a submission on the queue that released the last"
      (queues <> [] && m.last = last);
    cover "a switch of queue" (queues <> [] && m.last <> None && m.last <> last);
    cover "two queues in one submission" (List.length queues = 2);
    cover "an after across queues"
      (List.exists
         (fun p ->
           List.exists (fun k -> (List.nth ps k).copy <> p.copy) p.after)
         ps);
    cover "a COPY part after a delayed COMPUTE part"
      (List.exists
         (fun p ->
           p.copy && List.exists (fun k -> (List.nth ps k).delay > 0) p.after)
         ps);
    List.iter
      (fun p -> Bytes.blit m.bufs.(p.src) p.so m.bufs.(p.dst) p.do_ p.n)
      ps;
    m.last <-
      (if ps = [] then if m.last = None then Some false else m.last else last)

  let run m subs =
    cover "several values in flight" (List.length subs > 1);
    List.iter (run_one m) subs

  (* The system *)

  type sys = {
    r : run;
    regions : A.region array;
    staging : A.region;
    args : A.region;
    image : A.image;
    code : A.region;
  }

  let copy_all (d, o) (s, o') n =
    E.copy ~dst:(address d + o) ~src:(address s + o') n

  (* Each program's first submission is a multiple of 2^32, where a value's
     release goes on the compute queue and slot words compare 32 bits: the
     values before it are [start]'s four and the invariant's three. *)
  let start () =
    let r = run_of (raw ()) in
    let g = r.g in
    let next = (((r.v + 8) lsr 32) + 1) lsl 32 in
    A.renumber g (next - 7);
    r.v <- next - 8;
    let staging = Option.get (A.alloc g `Pinned size) in
    let regions =
      Array.init buffers (fun b ->
          let reg = Option.get (A.alloc g `Device size) in
          S.write (host staging) (Bytes.to_string (initial b));
          go r [| copy_all (reg, 0) (staging, 0) size |];
          reg)
    in
    let args = Option.get (A.alloc g `Pinned 4096) in
    let image, code, upload = laid ~of_:work_code r in
    go r [| upload |];
    { r; regions; staging; args; image; code }

  let release s =
    Array.iter (A.free s.r.g) s.regions;
    List.iter (A.free s.r.g) [ s.staging; s.args ];
    A.unload s.r.g s.image;
    A.free s.r.g s.code

  (* Submits [subs] back to back, then reads the word until it holds the last
     value: each read is at least the one before and at most the last. *)
  let run_sys s subs =
    let r = s.r and g = s.r.g in
    let block = ref 0 in
    let part p =
      let queue = if p.copy then "COPY:0" else "COMPUTE:0" in
      let after = Array.of_list p.after in
      let dst = s.regions.(p.dst) and src = s.regions.(p.src) in
      if p.copy then
        E.copy ~after ~dst:(address dst + p.do_) ~src:(address src + p.so) p.n
      else begin
        let a = at (host s.args) (32 * !block) in
        incr block;
        S.write a
          (String.concat ""
             [
               le64 (address dst + p.do_);
               le64 (address src + p.so);
               le32s [ p.n / 4; p.delay ];
             ]);
        E.words ~queue ~after
          (dispatch ~of_:work_code (A.capability g).gpu (entry s.image) "copy"
             ~args:a ~groups:1)
      end
    in
    let first = r.v + 1 in
    List.iter
      (fun ps ->
        cover "a value whose low 32 bits are 0 ends with a copy"
          ((r.v + 1) land 0xffff_ffff = 0
          && match List.rev ps with p :: _ -> p.copy | [] -> false);
        let ps = Array.of_list (List.map part ps) in
        equal room_answer ~msg:"room" `Fits (E.room g ps);
        r.v <- r.v + 1;
        equal answer ~msg:"submit" `Ok (submit g ~v:r.v ps))
      subs;
    let rec watch seen =
      let w = A.signaled g in
      at_least int ~msg:"the word" ~than:seen w;
      at_most int ~msg:"the word" ~than:r.v w;
      if w < r.v then watch w
    in
    watch (first - 1);
    S.wait g r.v

  let invariant m s =
    Array.iteri
      (fun b reg ->
        go s.r [| copy_all (s.staging, 0) (reg, 0) size |];
        equal string ~msg:(strf "buffer %d" b)
          (Bytes.to_string m.bufs.(b))
          (S.read (host s.staging) size))
      s.regions

  let parts =
    let open Gen in
    let part =
      let* copy = bool in
      let+ delay =
        if copy then constant 0
        else frequency [ (2, constant 0); (1, constant 20) ]
      and+ src = int_range 0 (buffers - 1)
      and+ dst = int_range 0 (buffers - 1)
      and+ n = one_of [ int_range 1 64; int_range 1 16384 ]
      and+ so = int_range 0 (size - 16384)
      and+ do_ = int_range 0 (size - 16384)
      and+ after = list ~size:(int_range 0 2) (int_range 0 2) in
      let word x = if copy then x else x land lnot 3 in
      let n = if copy then n else Int.max 4 (word n) in
      { copy; delay; src; so = word so; dst; do_ = word do_; n; after }
    in
    let submission =
      let+ ps = list ~size:(int_range 0 3) part in
      List.mapi
        (fun i p ->
          {
            p with
            after =
              List.sort_uniq compare (List.filter (fun k -> k < i) p.after);
          })
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

(* Waits on words

   Where the compute queue compares 64-bit words ([waits_on]), a submission
   holds its work until each word it waits on, as unsigned 64 bits, reaches its
   value; a 32-bit compare would pass a word whose low half alone is above. The
   words here are the host's, in memory the device maps. *)

let waits_on g =
  if not (A.waits_on g `Store) then
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
  S.with_gpu @@ fun g ->
  waits_on g;
  let word = Option.get (A.alloc g `Pinned 8) in
  let src = Option.get (A.alloc g `Pinned 64)
  and dst = Option.get (A.alloc g `Pinned 64) in
  let zeros = String.make 64 '\000' in
  S.write (host src) (String.make 64 'w');
  S.write (host dst) zeros;
  S.write (host word) (le64 w0);
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
      S.write (host word) (le64 x);
      S.still
        ~msg:(strf "the copy, the word at %d (sampled)" x)
        string zeros
        (fun () -> S.read (host dst) 64)
        ~ms:20;
      equal int ~msg:(strf "the timeline, the word at %d" x) 0 (A.signaled g))
    below;
  S.write (host word) (le64 t);
  S.wait g 1;
  equal string ~msg:"the copy, the word reached" (String.make 64 'w')
    (S.read (host dst) 64);
  List.iter (A.free g) [ word; src; dst ]

(* Through rig, between two devices of the GPU: the consumer's submission reads
   memory whose last writer is the producer's value still running, so rig has
   the consumer's queue wait on the producer's word. Its submit returns before
   the producer's value is reached, and its work runs after it. *)
let opens = ref 0

let in_queue () =
  S.with_gpu @@ fun g ->
  waits_on g;
  let c = S.rig g in
  incr opens;
  let made = ref None in
  let pc =
    match
      Rig.open_
        (module A)
        ~name:(strf "AMD:test-producer-%d" !opens)
        (fun () ->
          Result.map
            (fun x ->
              made := Some x;
              x)
            (Rig_amd_amdgpu.open_ 0))
    with
    | Ok pc -> pc
    | Error why -> fail why
  in
  let pg = Option.get !made in
  Fun.protect ~finally:(fun () -> A.stop pg) @@ fun () ->
  let prog =
    match Rig.Program.load pc (Lazy.force kernels).binary with
    | Ok p -> p
    | Error why -> fail why
  in
  let flag = Rig.Buffer.create ~memory:Pinned pc 8 in
  let args = Rig.Buffer.create ~memory:Pinned pc 64 in
  let fresh = Rig.Buffer.create ~memory:Pinned pc 64
  and b = Rig.Buffer.create pc 64 in
  put flag (le64 0);
  put fresh (String.make 64 'n');
  put args (le64 (addr flag) ^ le64 300_000);
  let entry f = Option.get (Rig.Program.entry prog f) in
  let spin =
    S.words_part ~queue:"COMPUTE:0"
      (dispatch (A.capability pg).gpu entry "spin" ~args:(addr args) ~groups:1)
  in
  let s =
    Rig.Submission.make ~reads:0 ~writes:0 pc
      [| spin; copy ~after:[| 0 |] ~dst:b fresh |]
  in
  let vp =
    Rig.Point.value (Rig.submit s ~reads:[||] ~writes:[||] ~waits:[||])
  in
  let out = buffer ~memory:Pinned g 64 in
  put out (String.make 64 '\000');
  let seen = Option.get (Rig.Buffer.borrow c b) in
  let vc = S.submit g [| copy ~dst:out seen |] in
  less int ~msg:"the producer's word, at the consumer's submit" ~than:vp
    (A.signaled pg);
  less int ~msg:"the consumer's word" ~than:vc (A.signaled g);
  (* amdgpu maps host memory at its GPU address *)
  S.still ~msg:"the consumer's copy (sampled)" string (String.make 64 '\000')
    (fun () -> S.read (addr out) 64)
    ~ms:50;
  Rig.wait c vc;
  at_least int ~msg:"the producer's word, once the consumer's is reached"
    ~than:vp (A.signaled pg);
  equal string ~msg:"the consumer's copy" (String.make 64 'n') (get out);
  ignore (Sys.opaque_identity args)

(* A submission waits on at most 255 words. *)
let wait_bound () =
  S.with_gpu @@ fun g ->
  waits_on g;
  let word = Option.get (A.alloc g `Pinned 8) in
  S.write (host word) (le64 1);
  let waits n = Array.make n (address word, 1) in
  equal answer ~msg:"255 waits" `Ok (E.submit g ~v:1 ~waits:(waits 255) [||]);
  S.wait g 1;
  match E.submit g ~v:2 ~waits:(waits 256) [||] with
  | `Ok -> fail "256 waits handed over"
  | `Failed why ->
      S.wait g 2;
      equal answer ~msg:"the next submit" (`Failed why) (submit g ~v:3 [||])

let waits =
  group ~timeout:120. "waits"
    [
      test
        "a submission reading another device's writes waits for them in its \
         queue (sampled)"
        in_queue;
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
  S.with_gpu @@ fun g ->
  let values = 1300 and most = 256 in
  let bytes = 4 * values * most in
  let cells = Option.get (A.alloc g `Pinned bytes) in
  let out = Option.get (A.alloc g `Pinned bytes) in
  S.write (host cells) (String.make bytes '\000');
  S.write (host out) (String.make bytes '\000');
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
        S.wait g (A.signaled g + 1);
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
  S.wait g (base + values - 1);
  let got = S.read (host out) bytes in
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
      stateful ~count:150 ~steps:12
        "values complete in order and the word never moves backwards (sampled)"
        order_commands;
    ]

(* Two devices of one GPU *)

let one_gpu () =
  S.with_gpu @@ fun g ->
  let g' =
    match Rig_amd_amdgpu.open_ 0 with Ok g' -> g' | Error why -> fail why
  in
  Fun.protect ~finally:(fun () -> A.stop g') @@ fun () ->
  let r = device g and r' = device g' in
  let n = 4096 in
  let p = pattern n 9 in
  let staging' = Option.get (A.alloc g' `Pinned n) in
  let theirs = Option.get (A.alloc g' `Device n) in
  let copy d s = E.copy ~dst:(address d) ~src:(address s) n in
  S.write (host staging') p;
  go r' [| copy theirs staging' |];
  let back = Option.get (A.alloc g `Pinned n) in
  let read_through view =
    S.write (host back) (String.make n '\000');
    go r [| copy back view |];
    S.read (host back) n
  in
  equal bool ~msg:"peers" true (A.peer g g');
  (match A.map_peer g g' theirs with
  | None -> fail "a device refused memory of its own GPU"
  | Some view ->
      equal string ~msg:"read through the view" p (read_through view);
      A.free g view);
  S.write (host staging') (String.make n '\000');
  go r' [| copy staging' theirs |];
  equal string ~msg:"the memory, its view unmapped" p (S.read (host staging') n);
  let mapped' = Option.get (A.alloc g' `Mapped n) in
  (match A.map_peer g g' mapped' with
  | None -> fail "a device refused Mapped memory of its own GPU"
  | Some view ->
      let p = pattern n 11 in
      S.write (host mapped') p;
      equal string ~msg:"the host's writes, read through a view" p
        (read_through view);
      A.free g view);
  let page = S.pages n in
  let m = Option.get (A.map_host g page n) in
  equal bool ~msg:"a page the other device maps" true
    (Option.is_none (A.map_host g' page n));
  A.free g m;
  (match A.map_host g' page n with
  | None -> fail "a page no device maps refused"
  | Some m' -> A.free g' m');
  S.free_pages page n;
  List.iter (A.free g') [ staging'; theirs; mapped' ];
  A.free g back

let two =
  group ~timeout:60. "one GPU"
    [
      test "two devices of one GPU share its memory, mapping each page once"
        one_gpu;
    ]

(* Traces *)

(* The level the kernel driver holds GPU 0's clocks at. *)
let level () =
  match Rig_amd_amdgpu.gpus_at "/" with
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
          S.with_gpu @@ fun g ->
          let c = A.capability g in
          match (c.trace (), c.trace ()) with
          | Ok t, Ok t' ->
              equal bool ~msg:"the same buffers" true (t = t');
              equal int ~msg:"engines"
                (c.gpu.shader_engines * c.gpu.xccs)
                t.engines;
              equal int ~msg:"window, a multiple of 4096" 0 (t.window mod 4096);
              S.write t.ends_host (String.make (4 * t.slots * t.engines) 'x');
              equal string ~msg:"the ends, host memory" (String.make 4 'x')
                (S.read t.ends_host 4);
              equal (option string) ~msg:"the GPU's clocks"
                (Some "profile_standard") (level ())
          | Error why, _ | _, Error why -> fail why);
    ]

let () =
  S.hold_gpu ();
  exit
    (Windtrap.run "rig_amd"
       [
         gpus;
         paths;
         misuse;
         images;
         room;
         hdp;
         failures;
         scratch;
         progress;
         domains;
         work;
         code;
         failures_hw;
         waits;
         two;
         traces;
         rings;
         memory;
         queue_order;
         timeline;
       ])
