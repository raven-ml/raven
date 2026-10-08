(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module A = Device_amd
module S = Device_amd_support
module Abi = Device_amd_abi
module Gpu = Abi.Gpu
module Pm4 = Abi.Pm4

let strf = Printf.sprintf
let host r = Option.get (A.host r)
let address r = Option.get (A.address r)
let submit g ~v ps = A.submit g ~v ~waits:[||] ~handles:[||] ps

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
    mutable queues : queue list; (* in the order made *)
    mutable refused_queue : bool;
    mutable stops : int;
    mutable sleeps : int;
    mutable report : string option; (* the fault its sleep reports *)
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
      ?(refuse = fun _ -> false) ?(stop = fun () -> `Stopped) () =
    let h =
      {
        lock = Mutex.create ();
        hdp = S.pages 4;
        calls = 0;
        live = 0;
        queues = [];
        refused_queue = false;
        stops = 0;
        sleeps = 0;
        report = None;
        doorbells = [];
        held = [];
        closed = false;
      }
    in
    let refused () =
      let i = h.calls in
      h.calls <- i + 1;
      refuse i
    in
    let alloc kind n =
      Mutex.protect h.lock @@ fun () ->
      if refused () then None
      else begin
        let m = memory ~host:(kind <> `Gpu) (S.pages n) n ~view:false in
        h.live <- h.live + 1;
        h.held <- m.data :: h.held;
        Some m
      end
    in
    let view (m : mem A.memory) =
      { m with data = { m.data with view = true; given = false } }
    in
    let free (m : mem A.memory) =
      Mutex.protect h.lock @@ fun () ->
      if not h.closed then begin
        if m.data.given then fail "the device gave the same memory back twice";
        m.data.given <- true;
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
        map_host =
          (fun a n -> Some (memory ~host:true a n ~view:true));
        reaches = (fun _ -> reaches);
        map_peer = (fun m -> if reaches then Some (view m) else None);
        free;
        queue;
        hdp = Some h.hdp;
        interrupt = 1;
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

  let device ?key ?reaches ?gpu ?lds ?stop () =
    let h, p = path ?key ?reaches ?gpu ?lds ?stop () in
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

  (* The position a queue reads its work up to: in words on a PM4 ring, bytes
     on an SDMA ring, packets on an AQL ring. *)
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
let make_gives_back () =
  let calls =
    let h, g = Host.device () in
    A.stop g;
    Host.close h;
    h.calls
  in
  for i = 0 to calls - 1 do
    let h, p = Host.path ~refuse:(( = ) i) () in
    let msg = strf "call %d of %d refused" i calls in
    (match A.make p with
    | Ok _ -> failf "%s: a device" msg
    | Error why ->
        if h.refused_queue then contains ~msg ~sub:"the host makes no queue" why);
    equal int ~msg:(msg ^ ": allocations left") 0 h.live;
    equal int ~msg:(msg ^ ": stops") (if h.queues = [] then 0 else 1) h.stops;
    Host.close h
  done

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
      equal (array (array int)) ~msg
        (Array.make (g.shader_engines * g.xccs) [| 0xff; 0xff |])
        c.wgps;
      equal (list string) ~msg:"queues the path made" [ kind; "SDMA" ]
        (List.map
           (fun (q : Host.queue) ->
             match q.kind with `Pm4 -> "PM4" | `Aql -> "AQL" | `Sdma -> "SDMA")
           h.queues);
      equal bool ~msg:"the capability's key" true
        (Option.is_some (Type.Id.provably_equal A.capability_key Abi.Capability.key)))
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
          raises_match (Exn.invalid_arg ~substring:"Device_amd.make") (fun () ->
              A.make { p with interrupt = 0 });
          Host.close h);
      test "a failed make gives back what it took" make_gives_back;
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
    raises_match ~msg:fn (Exn.invalid_arg ~substring:("Device_amd." ^ fn ^ ":")) f
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
  raises "part" (fun () -> A.part g ~queue:"COPY:1" (`Words [||]));
  raises "part" (fun () -> A.part g ~queue:"COPY:0" ~after:[| -1 |] (`Words [||]));
  let f, arg = S.fill (A.capability g) [||] ~bytes:0 in
  raises "part" (fun () -> A.part g ~queue:"COMPUTE:0" (`Fill (f, arg, -1, 0)));
  raises "part" (fun () -> A.part g ~queue:"COMPUTE:0" (`Fill (f, arg, 0, -1)));
  raises "part" (fun () ->
      A.part g ~queue:"COMPUTE:0" (`Copy ((r, 0), (r, 2048), 16)));
  raises "part" (fun () ->
      A.part g ~queue:"COPY:0" (`Copy ((r, 4090), (r, 0), 7)));
  raises "part" (fun () ->
      A.part g ~queue:"COPY:0" (`Copy ((r, -1), (r, 2048), 1)));
  raises "part" (fun () ->
      A.part g ~queue:"COPY:0" (`Copy ((r, 0), (r', 0), 16)));
  raises "submit" (fun () -> submit g ~v:2 [||]);
  raises "submit" (fun () -> submit g ~v:1 [| A.part g' ~queue:"COPY:0" (`Words [||]) |]);
  raises "submit" (fun () ->
      submit g ~v:1 [| A.part g ~queue:"COPY:0" ~after:[| 0 |] (`Words [||]) |]);
  let word = host (A.word g') in
  List.iter
    (fun kind ->
      raises "submit" (fun () ->
          A.submit g ~v:1 ~waits:[| (kind, word, 1) |] ~handles:[||] [||]))
    [ `Word; `Object ];
  equal answer ~msg:"the first value, after the misuse" `Ok (submit g ~v:1 [||]);
  let m, code, _ = load g kernels in
  raises "unload" (fun () -> A.unload g' m);
  A.unload g m;
  raises "entry" (fun () -> A.entry m "empty");
  raises "unload" (fun () -> A.unload g m);
  A.free g code;
  A.free g r;
  raises "free" (fun () -> A.free g r);
  raises "part" (fun () ->
      A.part g ~queue:"COPY:0" (`Copy ((r, 0), (r, 2048), 16)));
  A.free g mapped;
  A.free g' r';
  S.free_pages page 4096

(* What device_amd_room answers for a part that [part] refuses. *)
let c_room () =
  let never = room_answer in
  Host.with_device @@ fun _ g ->
  equal never ~msg:"a part of 16 words" `Fits (S.room g ~queue:0 ~words:16);
  equal never ~msg:"a copy on the compute queue" `Never
    (S.room g ~queue:0 ~copy:64);
  equal never ~msg:"words and a fill" `Never (S.room g ~queue:0 ~words:4 ~fill:true);
  equal never ~msg:"after its own index" `Never (S.room g ~queue:1 ~after:0);
  equal never ~msg:"a queue of no index" `Never (S.room g ~queue:2);
  equal never ~msg:"a negative queue" `Never (S.room g ~queue:(-1));
  Host.with_device ~gpu:mi300 @@ fun _ g ->
  equal never ~msg:"AQL: a packet" `Fits (S.room g ~queue:0 ~words:16);
  equal never ~msg:"AQL: part of a packet" `Never (S.room g ~queue:0 ~words:15)

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
        match (a, b) with Error a, Error b -> a = b | Ok _, Ok _ -> true | _ -> false)
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
    | Words n -> A.part g ~queue ~after (`Words (Array.sub (Lazy.force zeros) 0 n))
    | Fill (units, bytes) ->
        let f, arg = fill_taking g bytes in
        A.part g ~queue ~after (`Fill (f, arg, units, bytes))
    | Copy n -> A.part g ~queue:"COPY:0" ~after (`Copy ((dst, 0), (src, 0), n))
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
  let compute = compute && (match work with Copy _ -> false | _ -> true) in
  let+ after = list ~size:(int_range 0 2) (int_range 0 3) in
  { compute; work; after }

let program =
  let open Gen in
  let submission =
    let+ ss = list ~size:(int_range 0 4) spec in
    Submit
      (List.mapi
         (fun i s -> { s with after = List.sort_uniq compare (List.filter (fun k -> k < i) s.after) })
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

let regions g = (Option.get (A.alloc g `Pinned 4096), Option.get (A.alloc g `Pinned 4096))

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
    let reference () = A.room f (parts f fr specs) in
    let a = A.room g ps in
    if !word = !last then
      equal room_answer ~msg:"every value reached" (reference ()) a;
    let a =
      if a <> `Later then a
      else begin
        cover "Later while a value is unreached" true;
        reach !last;
        let a = A.room g ps in
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
  Format.fprintf ppf "%s %d units %d bytes" (if c then "COMPUTE" else "COPY") u b

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
    A.room f
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
  let empty n = Array.init n (fun _ -> A.part g ~queue:"COPY:0" (`Words [||])) in
  equal room_answer ~msg:"512 parts" `Fits (A.room g (empty 512));
  equal room_answer ~msg:"513 parts" `Never (A.room g (empty 513))

let overflow = ([ (true, max_int, 0); (true, max_int, 0); (true, max_int, 0) ], (true, max_int, 0))

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
  abstract "machine"
    ~release:(fun (ds : sys_machine) ->
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
       (fun r -> if r.live && r.mapped && r.holder = d then Some r.owner else None)
       m.regions)

let flushed (ds : sys_machine) d =
  Array.iter (fun ((h : Host.t), _, _) -> set32 h.hdp 0xffff_ffff) ds;
  let _, g, last = ds.(d) in
  equal room_answer ~msg:"room" `Fits (A.room g [||]);
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
    command "give back" ~pre:(fun r -> r.live)
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
   release of its value goes on the compute queue. Markers are words no
   packet the device places holds. *)

let marker k = 0x5eed_0000 lor k
let is_marker w = w land 0xffff_0000 = 0x5eed_0000

(* The words of [q]'s ring from position [p] to [p'], a PM4 ring's positions
   being words. *)
let handed (q : Host.queue) p p' =
  let size = q.bytes / 4 in
  List.init (p' - p) (fun i ->
      get32 (at q.ring (4 * ((p + i) mod size))))

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
  let markers n = Array.init n (fun _ -> incr k; marker !k) in
  let queue c = if c then "COMPUTE:0" else "COPY:0" in
  let before =
    List.map (fun (c, n) -> A.part g ~queue:(queue c) (`Words (markers n))) f.before
  in
  let ws = markers 4 in
  let fill, arg = S.fill ~code:f.code (A.capability g) ws ~bytes:64 in
  let fails = A.part g ~queue:(queue f.on_compute) (`Fill (fill, arg, 4, 64)) in
  let c0 = Host.position (Host.compute h) and s0 = Host.position (Host.copy h) in
  let why = strf "a fill on %s failed with %d" (queue f.on_compute) f.code in
  let ps = Array.of_list (before @ [ fails ]) in
  equal room_answer ~msg:"room" `Fits (A.room g ps);
  equal answer ~msg:"submit" (`Failed why) (submit g ~v:1 ps);
  let c1 = Host.position (Host.compute h) and s1 = Host.position (Host.copy h) in
  equal int ~msg:"the copy queue's position" s0 s1;
  greater int ~msg:"the compute queue's position" ~than:c0 c1;
  equal (list int) ~msg:"markers handed over" []
    (List.filter is_marker (handed (Host.compute h) c0 c1));
  equal answer ~msg:"the next submit" (`Failed why) (submit g ~v:2 [||]);
  equal int ~msg:"the copy queue's position, after" s1
    (Host.position (Host.copy h));
  equal int ~msg:"the compute queue's position, after" c1
    (Host.position (Host.compute h))

(* An AQL queue may read a packet past its write position as soon as its
   header is valid. *)
let aql_failure () =
  Host.with_device ~gpu:mi300 @@ fun h g ->
  let packet k = Array.init 16 (fun i -> if i = 0 then 2 else marker ((16 * k) + i)) in
  let first = A.part g ~queue:"COMPUTE:0" (`Words (packet 0)) in
  let ws = Array.append (packet 1) (packet 2) in
  let fill, arg = S.fill ~code:7 (A.capability g) ws ~bytes:0 in
  let fails = A.part g ~queue:"COMPUTE:0" (`Fill (fill, arg, 32, 0)) in
  let q = Host.compute h in
  let slots = q.bytes / 64 in
  let header s = get32 (at q.ring (64 * (s mod slots))) land 0xff in
  let ours s =
    List.exists is_marker
      (List.init 15 (fun i -> get32 (at q.ring ((64 * (s mod slots)) + (4 * (i + 1))))))
  in
  let p0 = Host.position q in
  equal answer ~msg:"submit" (`Failed "a fill on COMPUTE:0 failed with 7")
    (submit g ~v:1 [| first; fails |]);
  let p1 = Host.position q in
  greater int ~msg:"the release handed over" ~than:p0 p1;
  for s = p0 to p1 - 1 do
    equal bool ~msg:(strf "packet %d, handed over, is the submission's" s) false (ours s)
  done;
  for s = p1 to p1 + 8 do
    if ours s then equal int ~msg:(strf "the header type of packet %d" s) 1 (header s)
  done

let sleeps () =
  let h, g = Host.device () in
  A.sleep g ~seen:5 ~still_ms:10_000;
  equal int ~msg:"sleeps with the word past seen" 0 h.sleeps;
  A.sleep g ~seen:0 ~still_ms:1;
  equal int ~msg:"sleeps with the word at seen" 1 h.sleeps;
  h.report <- Some "memory fault at 0x0";
  raises (A.Fault "memory fault at 0x0") (fun () -> A.sleep g ~seen:0 ~still_ms:1);
  A.stop g;
  Host.close h

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

let failures =
  group ~timeout:60. "failures"
    [
      prop ~count:100 "a failed fill hands over none of its submission"
        failing failure_hands_nothing;
      test
        "on an AQL queue, a failed fill leaves none of its packets valid past \
         the write position"
        aql_failure;
      test "sleep asks the path only while the word holds seen" sleeps;
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
    abstract name ~release:(fun x -> try release x with Invalid_argument _ -> ())
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

let domains =
  group ~timeout:60. "domains"
    [
      stateful ~count:30 ~domains:2
        "an allocation freed from two domains is freed once" allocation_commands;
      stateful ~count:30 ~domains:2
        "a mapping unmapped from two domains is unmapped once" mapping_commands;
      stateful ~count:30 ~domains:2
        "an image unloaded from two domains is unloaded once" image_commands;
    ]

(* On a GPU *)

let work =
  group ~timeout:60. "work"
    [
      test "an empty submission releases its value" (fun () ->
          S.with_gpu @@ fun g ->
          equal answer ~msg:"submit" `Ok (submit g ~v:1 [||]);
          S.wait g 1;
          equal int ~msg:"the word" 1 (A.signaled g));
      test "an idle device stops with its word at the last value" (fun () ->
          let g = S.gpu () in
          equal answer ~msg:"submit" `Ok (submit g ~v:1 [||]);
          S.wait g 1;
          S.stop g;
          equal int ~msg:"the word" 1 (A.signaled g));
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

let pattern n seed = String.init n (fun i -> Char.chr ((seed + (i * 7)) land 0xff))

let kinds = [ `Device; `Pinned; `Mapped ]

let pp_kind ppf k =
  Format.pp_print_string ppf
    (match k with `Device -> "Device" | `Pinned -> "Pinned" | `Mapped -> "Mapped")

let kind = Gen.of_list ~pp:pp_kind kinds
let size = Gen.of_list ~pp:Format.pp_print_int [ 1; 7; 4096; (1 lsl 20) + 7 ]
let offset = Gen.of_list ~pp:Format.pp_print_int [ 0; 1; 4095 ]

let round_trip (ka, kb, n, (oa, ob)) =
  let r = run_of (shared ()) in
  let g = r.g in
  let src = Option.get (A.alloc g `Pinned n) in
  let a = Option.get (A.alloc g ka (oa + n)) in
  let b = Option.get (A.alloc g kb (ob + n)) in
  let dst = Option.get (A.alloc g `Pinned n) in
  equal bool ~msg:"host addresses a" (ka <> `Device) (Option.is_some (A.host a));
  let bytes = pattern n (n + oa) in
  S.write (host src) bytes;
  let copy (d, o) (s, o') = A.part g ~queue:"COPY:0" (`Copy ((d, o), (s, o'), n)) in
  go r [| copy (a, oa) (src, 0) |];
  go r [| copy (b, ob) (a, oa) |];
  go r [| copy (dst, 0) (b, ob) |];
  equal string ~msg:"the bytes" bytes (S.read (host dst) n);
  List.iter (A.free g) [ src; a; b; dst ]

(* Copies of the copy engine's largest packet and its neighbours: the bytes
   before each copy's end arrive, those after stay as they were. Two windows
   are read: the start, and the packet's end. *)
let around_the_packet () =
  let r = run_of (shared ()) in
  let g = r.g in
  let max = Abi.Sdma.max_copy (A.capability g).gpu in
  let w = 4096 in
  let n = max + (w / 2) in
  let src = Option.get (A.alloc g `Pinned n) in
  let dst = Option.get (A.alloc g `Device n) in
  let back = Option.get (A.alloc g `Pinned (2 * w)) in
  let zeros = Option.get (A.alloc g `Pinned w) in
  S.write (host zeros) (String.make w '\000');
  let windows = [ 0; max - (w / 2) ] in
  List.iteri (fun i o -> S.write (at (host src) o) (pattern w (i + 1))) windows;
  let copy (d, o) (s, o') k = A.part g ~queue:"COPY:0" (`Copy ((d, o), (s, o'), k)) in
  List.iter
    (fun k ->
      go r (Array.of_list (List.map (fun o -> copy (dst, o) (zeros, 0) w) windows));
      go r [| copy (dst, 0) (src, 0) k |];
      go r (Array.of_list (List.mapi (fun i o -> copy (back, i * w) (dst, o) w) windows));
      List.iteri
        (fun i o ->
          let arrived = Int.max 0 (Int.min w (k - o)) in
          let expected =
            String.sub (pattern w (i + 1)) 0 arrived ^ String.make (w - arrived) '\000'
          in
          equal string ~msg:(strf "%d bytes, at %d" k o) expected
            (S.read (at (host back) (i * w)) w))
        windows)
    [ max - 1; max; max + 1 ];
  List.iter (A.free g) [ src; dst; back; zeros ]

(* Host and GPU writes to the same memory, round after round, are read by the
   next submission and by the host. *)
let rewrites () =
  let r = run_of (shared ()) in
  let g = r.g in
  let n = 4096 in
  let mapped = Option.get (A.alloc g `Mapped n) in
  let pinned = Option.get (A.alloc g `Pinned n) in
  let vram = Option.get (A.alloc g `Device n) in
  let out = Option.get (A.alloc g `Pinned n) in
  let copy d s = A.part g ~queue:"COPY:0" (`Copy ((d, 0), (s, 0), n)) in
  for round = 1 to 8 do
    let msg = strf "round %d" round in
    let p = pattern n round in
    S.write (host mapped) p;
    go r [| copy out mapped |];
    equal string ~msg:(msg ^ ": the host's write through the BAR") p
      (S.read (host out) n);
    let p' = pattern n (round + 100) in
    S.write (host pinned) p';
    go r [| copy vram pinned |];
    go r [| copy mapped vram |];
    equal string ~msg:(msg ^ ": the GPU's write, read through the BAR") p'
      (S.read (host mapped) n)
  done;
  List.iter (A.free g) [ mapped; pinned; vram; out ]

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

(* [m]'s image laid over new memory [code], and the part that copies it there
   from staging memory the host wrote. *)
let laid ?(of_ = kernels) r =
  let m, code, bytes = load r.g (Lazy.force of_).binary in
  let n = String.length bytes in
  let staging = Option.get (A.alloc r.g `Pinned n) in
  S.write (host staging) bytes;
  (m, code, A.part r.g ~queue:"COPY:0" (`Copy ((code, 0), (staging, 0), n)))

let image ?of_ r =
  let m, _, upload = laid ?of_ r in
  (m, upload)

(* A dispatch of [name] over [groups] workgroups of 64, its arguments [args]. *)
let dispatch ?(of_ = kernels) r m name ~args ~groups =
  let gpu = (A.capability r.g).gpu in
  let k = Option.get (Abi.Code_object.kernel (Lazy.force of_).co name) in
  let base = Option.get (A.entry m name) - k.descriptor in
  words
    (Pm4.run gpu
       (Pm4.dispatch gpu k ~program:(base + k.entry) ~scratch:0 ~args ~packet:0
          ~threads:(64, 1, 1) ~groups:(groups, 1, 1) ()))

let arguments r values =
  let args = Option.get (A.alloc r.g `Pinned 4096) in
  S.write (host args) (String.concat "" (List.map le64 values));
  args

let multiples k n = le32s (List.init n (fun i -> k * i))
let doubled = multiples 2

let spin r m flag n =
  let args = arguments r [ address flag; n ] in
  A.part r.g ~queue:"COMPUTE:0"
    (`Words (dispatch r m "spin" ~args:(address args) ~groups:1))

let code =
  group ~timeout:60. "kernels"
    [
      test "a kernel loaded from its image computes" (fun () ->
          S.with_gpu @@ fun g ->
          let r = device g in
          let m, upload = image r in
          go r [| upload |];
          let out = Option.get (A.alloc g `Pinned (4 * 256)) in
          let args = arguments r [ address out ] in
          let ws = dispatch r m "double_index" ~args:(address args) ~groups:4 in
          go r [| A.part g ~queue:"COMPUTE:0" (`Words ws) |];
          equal string ~msg:"out" (doubled 256) (S.read (host out) (4 * 256)));
      test "code placed where other code ran runs as placed" (fun () ->
          S.with_gpu @@ fun g ->
          let r = device g in
          let out = Option.get (A.alloc g `Pinned (4 * 64)) in
          let args = arguments r [ address out ] in
          let run ?of_ () =
            let m, code, upload = laid ?of_ r in
            go r [| upload |];
            let ws =
              dispatch ?of_ r m "double_index" ~args:(address args) ~groups:1
            in
            go r [| A.part g ~queue:"COMPUTE:0" (`Words ws) |];
            let at = Option.get (A.entry m "double_index") in
            A.unload g m;
            A.free g code;
            (at, S.read (host out) (4 * 64))
          in
          let first, doubled_out = run () in
          equal string ~msg:"the first object's" (doubled 64) doubled_out;
          let second, tripled_out = run ~of_:other () in
          equal string ~msg:"the second object's" (multiples 3 64) tripled_out;
          let base f m =
            m
            - (Option.get
                 (Abi.Code_object.kernel (Lazy.force f).co "double_index"))
                .descriptor
          in
          (* The case an instruction cache could serve stale: the system's
             addresses for the second object are the first's. *)
          if base kernels first <> base other second then
            skip ~reason:"the second object got other addresses" ());
      test "parts on two queues run in their after order" (fun () ->
          S.with_gpu @@ fun g ->
          let r = device g in
          let m, upload = image r in
          let out = Option.get (A.alloc g `Device (4 * 256)) in
          let back = Option.get (A.alloc g `Pinned (4 * 256)) in
          let args = arguments r [ address out ] in
          let ws = dispatch r m "double_index" ~args:(address args) ~groups:4 in
          go r
            [|
              upload;
              A.part g ~queue:"COMPUTE:0" ~after:[| 0 |] (`Words ws);
              A.part g ~queue:"COPY:0" ~after:[| 1 |]
                (`Copy ((back, 0), (out, 0), 4 * 256));
            |];
          equal string ~msg:"back" (doubled 256) (S.read (host back) (4 * 256)));
      test "a fill places its words" (fun () ->
          S.with_gpu @@ fun g ->
          let r = device g in
          let word = Option.get (A.alloc g `Pinned 8) in
          let ws = words (Pm4.write_data (Memory (address word)) 0xc0ffee) in
          let f, arg = S.fill (A.capability g) ws ~bytes:64 in
          let units = Array.length ws in
          go r [| A.part g ~queue:"COMPUTE:0" (`Fill (f, arg, units, 64)) |];
          equal string ~msg:"word" (le64 0xc0ffee) (S.read (host word) 8));
      test "a fill past its declaration fails, and the word still moves"
        (fun () ->
          S.with_gpu @@ fun g ->
          let word = Option.get (A.alloc g `Pinned 8) in
          let ws = words (Pm4.write_data (Memory (address word)) 1) in
          let f, arg = S.fill (A.capability g) ws ~bytes:0 in
          let p =
            A.part g ~queue:"COMPUTE:0" (`Fill (f, arg, Array.length ws - 1, 0))
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
          let f, arg = S.fill ~code:5 (A.capability g) [||] ~bytes:0 in
          equal answer ~msg:"submit" (`Failed "a fill on COPY:0 failed with 5")
            (submit g ~v:1
               [|
                 A.part g ~queue:"COPY:0" (`Copy ((dst, 0), (src, 0), n));
                 A.part g ~queue:"COPY:0" (`Fill (f, arg, 0, 0));
               |]);
          S.wait g 1;
          equal string ~msg:"dst" (String.make n '\000') (S.read (host dst) n));
      test "a value failed behind running work is reached once it ends"
        (fun () ->
          let g = S.gpu () in
          let r = device g in
          let m, upload = image r in
          go r [| upload |];
          let flag = Option.get (A.alloc g `Pinned 8) in
          S.write (host flag) (le64 0);
          equal answer ~msg:"the work" `Ok (submit g ~v:2 [| spin r m flag 150_000 |]);
          let f, arg = S.fill ~code:5 (A.capability g) [||] ~bytes:0 in
          equal answer ~msg:"the failure"
            (`Failed "a fill on COMPUTE:0 failed with 5")
            (submit g ~v:3 [| A.part g ~queue:"COMPUTE:0" (`Fill (f, arg, 0, 0)) |]);
          S.wait g 3;
          equal string ~msg:"the work's flag" (le32s [ 1 ]) (S.read (host flag) 4);
          S.stop g;
          equal int ~msg:"the word" 3 (A.signaled g));
      test "long work is no fault" (fun () ->
          S.with_gpu @@ fun g ->
          let r = device g in
          let m, upload = image r in
          go r [| upload |];
          let flag = Option.get (A.alloc g `Pinned 8) in
          go r [| spin r m flag 150_000 |];
          equal string ~msg:"flag" (le32s [ 1 ]) (S.read (host flag) 4));
      test "a device stopped while its work runs stops it" (fun () ->
          let g = S.gpu () in
          let r = device g in
          let m, upload = image r in
          go r [| upload |];
          let flag = Option.get (A.alloc g `Pinned 8) in
          equal answer ~msg:"submit" `Ok
            (submit g ~v:2 [| spin r m flag 1_500_000 |]);
          S.stop g;
          equal int ~msg:"the word" 2 (A.signaled g);
          S.still ~msg:"flag (sampled)" string (String.make 4 '\000')
            (fun () -> S.read (host flag) 4)
            ~ms:100);
    ]

(* Rings wrap: streams of submissions longer than each ring complete, the last
   copies having moved their bytes. *)
let wrap () =
  S.with_gpu @@ fun g ->
  let r = device g in
  let n = 350_000 in
  let room ps =
    let rec loop () =
      match A.room g ps with
      | `Fits -> ()
      | `Never -> fail "Never"
      | `Later ->
          S.wait g (A.signaled g + 1);
          loop ()
    in
    loop ()
  in
  let hand ps =
    room ps;
    r.v <- r.v + 1;
    equal answer ~msg:"submit" `Ok (submit g ~v:r.v ps)
  in
  for _ = 1 to n do
    hand [||]
  done;
  S.wait g r.v;
  let slots = 1024 in
  let src = Option.get (A.alloc g `Pinned (8 * slots)) in
  let dst = Option.get (A.alloc g `Pinned (8 * slots)) in
  S.write (host src) (String.concat "" (List.init slots (fun i -> le64 (i + 1))));
  S.write (host dst) (String.make (8 * slots) '\000');
  for k = 1 to n do
    let o = 8 * (k mod slots) in
    hand [| A.part g ~queue:"COPY:0" (`Copy ((dst, o), (src, o), 8)) |]
  done;
  S.wait g r.v;
  equal string ~msg:"the copies" (S.read (host src) (8 * slots))
    (S.read (host dst) (8 * slots))

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
    List.for_all (fun p -> not (overlap (p.src, p.so) (p.dst, p.do_) p.n p.n)) ps
    &&
    let n = List.length ps in
    List.for_all
      (fun j ->
        List.for_all
          (fun i -> before ps i j || not (conflict (List.nth ps i) (List.nth ps j)))
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
         (fun p -> List.exists (fun k -> (List.nth ps k).copy <> p.copy) p.after)
         ps);
    cover "a COPY part after a delayed COMPUTE part"
      (List.exists
         (fun p ->
           p.copy && List.exists (fun k -> (List.nth ps k).delay > 0) p.after)
         ps);
    List.iter (fun p -> Bytes.blit m.bufs.(p.src) p.so m.bufs.(p.dst) p.do_ p.n) ps;
    m.last <- (if ps = [] then (if m.last = None then Some false else m.last) else last)

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
  }

  let copy_all r (d, o) (s, o') n =
    A.part r.g ~queue:"COPY:0" (`Copy ((d, o), (s, o'), n))

  (* Each program's first submission is a multiple of 2^32, where a value's
     release goes on the compute queue and slot words compare 32 bits: the
     values before it are [start]'s four and the invariant's three. *)
  let start () =
    let r = run_of (shared ()) in
    let g = r.g in
    let next = (((r.v + 8) lsr 32) + 1) lsl 32 in
    A.renumber g (next - 7);
    r.v <- next - 8;
    let staging = Option.get (A.alloc g `Pinned size) in
    let regions =
      Array.init buffers (fun b ->
          let reg = Option.get (A.alloc g `Device size) in
          S.write (host staging) (Bytes.to_string (initial b));
          go r [| copy_all r (reg, 0) (staging, 0) size |];
          reg)
    in
    let args = Option.get (A.alloc g `Pinned 4096) in
    let image, upload = image ~of_:work_code r in
    go r [| upload |];
    { r; regions; staging; args; image }

  let release s =
    Array.iter (A.free s.r.g) s.regions;
    List.iter (A.free s.r.g) [ s.staging; s.args ];
    A.unload s.r.g s.image

  (* Submits [subs] back to back, then reads the word until it holds the last
     value: each read is at least the one before and at most the last. *)
  let run_sys s subs =
    let r = s.r and g = s.r.g in
    let block = ref 0 in
    let part p =
      let queue = if p.copy then "COPY:0" else "COMPUTE:0" in
      let after = Array.of_list p.after in
      let dst = s.regions.(p.dst) and src = s.regions.(p.src) in
      if p.copy then A.part g ~queue ~after (`Copy ((dst, p.do_), (src, p.so), p.n))
      else begin
        let a = at (host s.args) (32 * !block) in
        incr block;
        S.write a
          (String.concat ""
             [ le64 (address dst + p.do_); le64 (address src + p.so); le32s [ p.n / 4; p.delay ] ]);
        let ws =
          dispatch ~of_:work_code r s.image "copy"
            ~args:a ~groups:1
        in
        A.part g ~queue ~after (`Words ws)
      end
    in
    let first = r.v + 1 in
    List.iter
      (fun ps ->
        cover "a value whose low 32 bits are 0 ends with a copy"
          ((r.v + 1) land 0xffff_ffff = 0
          && match List.rev ps with p :: _ -> p.copy | [] -> false);
        let ps = Array.of_list (List.map part ps) in
        equal room_answer ~msg:"room" `Fits (A.room g ps);
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
        go s.r [| copy_all s.r (s.staging, 0) (reg, 0) size |];
        equal string ~msg:(strf "buffer %d" b)
          (Bytes.to_string m.bufs.(b))
          (S.read (host s.staging) size))
      s.regions

  let parts =
    let open Gen in
    let part =
      let* copy = bool in
      let+ delay = if copy then constant 0 else frequency [ (2, constant 0); (1, constant 20) ]
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
          { p with after = List.sort_uniq compare (List.filter (fun k -> k < i) p.after) })
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

let rings =
  group ~timeout:120. "rings"
    [ test "streams longer than the rings complete" wrap ]

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
    match Device_amd_amdgpu.open_ 0 with Ok g' -> g' | Error why -> fail why
  in
  Fun.protect ~finally:(fun () -> A.stop g') @@ fun () ->
  let r = device g and r' = device g' in
  let n = 4096 in
  let p = pattern n 9 in
  let staging' = Option.get (A.alloc g' `Pinned n) in
  let theirs = Option.get (A.alloc g' `Device n) in
  S.write (host staging') p;
  go r' [| A.part g' ~queue:"COPY:0" (`Copy ((theirs, 0), (staging', 0), n)) |];
  let back = Option.get (A.alloc g `Pinned n) in
  let read_through view =
    S.write (host back) (String.make n '\000');
    go r [| A.part g ~queue:"COPY:0" (`Copy ((back, 0), (view, 0), n)) |];
    S.read (host back) n
  in
  equal bool ~msg:"peers" true (A.peer g g');
  (match A.map_peer g g' theirs with
  | None -> fail "a device refused memory of its own GPU"
  | Some view ->
      equal string ~msg:"read through the view" p (read_through view);
      A.free g view);
  S.write (host staging') (String.make n '\000');
  go r' [| A.part g' ~queue:"COPY:0" (`Copy ((staging', 0), (theirs, 0), n)) |];
  equal string ~msg:"the memory, its view unmapped" p (S.read (host staging') n);
  let mapped' = Option.get (A.alloc g' `Mapped n) in
  (match A.map_peer g g' mapped' with
  | None -> fail "a device refused Mapped memory of its own GPU"
  | Some view ->
      let p = pattern n 11 in
      S.write (host mapped') p;
      equal string ~msg:"the host's writes, read through a view" p (read_through view);
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
    [ test "two devices of one GPU share its memory, mapping each page once" one_gpu ]

(* Traces *)

(* The level the kernel driver holds GPU 0's clocks at. *)
let level () =
  match Device_amd_amdgpu.gpus_at "/" with
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
              equal (option string) ~msg:"the GPU's clocks" (Some "profile_standard")
                (level ())
          | Error why, _ | _, Error why -> fail why);
    ]

let () =
  S.hold_gpu ();
  exit
    (run "device_amd"
       [
         gpus;
         paths;
         misuse;
         images;
         room;
         hdp;
         failures;
         domains;
         work;
         code;
         two;
         traces;
         rings;
         memory;
         timeline;
       ])
