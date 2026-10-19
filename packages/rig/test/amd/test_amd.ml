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

(* The function an entry answered, which the test needs. *)
let found : Rig_edge.entry_answer -> Rig_edge.entry = function
  | Entry e -> e
  | Absent -> fail "no such function"
  | No_memory n -> failf "no memory for %d bytes" n

(* The code of the function an entry answered, if it answered one. *)
let code_of : Rig_edge.entry_answer -> int option = function
  | Entry e -> Some e.code
  | Absent | No_memory _ -> None

let still = Rig_gpu_support.still
let host r = Option.get (A.locate r).host
let address r = Option.get (A.locate r).address
let submit g ~v ps = E.submit g ~v ps

let le64 n =
  let b = Bytes.create 8 in
  Bytes.set_int64_le b 0 (Int64.of_int n);
  Bytes.to_string b

let le32s ns =
  let b = Bytes.create (4 * List.length ns) in
  List.iteri (fun i n -> Bytes.set_int32_le b (4 * i) (Int32.of_int n)) ns;
  Bytes.to_string b

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
  | `Grow -> Format.pp_print_string ppf "`Grow"

let room_answer = Testable.make ~pp:pp_room ~equal:( = )

(* GPUs *)

(* A path to no GPU, as [gpu] describes it: [make] checks a path's facts before
   it calls any of its functions. *)
let unused () = fail "make used the path"

let path_of gpu : unit A.path =
  {
    key = Type.Id.make ();
    index = 0;
    gpu;
    waves = 32;
    lds = 65536;
    clock_hz = 100_000_000;
    clock = Round_trip;
    mec = 1;
    wgps = [| [| 1 |] |];
    budget = 1 lsl 30;
    alloc = (fun _ _ -> unused ());
    map_host = None;
    reaches = (fun _ -> unused ());
    map_peer = (fun _ -> unused ());
    free = (fun _ -> unused ());
    queue = (fun _ ~ring:_ ~bytes:_ ~read:_ ~write:_ -> unused ());
    retire = (fun _ -> unused ());
    most_ring = (fun _ -> unused ());
    hdp = None;
    interrupt = 1;
    hang_ms = None;
    sleep = (fun ~ms:_ -> unused ());
    stable_power = unused;
    stop = (fun ~fault:_ -> unused ());
  }

let gfx1201 : Gpu.t =
  {
    target = (12, 0, 1);
    gc = (12, 0, 1);
    sdma = (7, 0, 0);
    xccs = 1;
    shader_engines = 4;
    compute_units = 32;
    array_units = 4;
    scratch_slots = 32;
  }

let counts =
  [
    ("xccs", { gfx1201 with xccs = 0 });
    ("shader_engines", { gfx1201 with shader_engines = 0 });
    ("compute_units", { gfx1201 with compute_units = -1 });
    ("scratch_slots", { gfx1201 with scratch_slots = 0 });
  ]

let gpus =
  group ~timeout:10. "GPUs"
    [
      cases
        ~name:(fun (field, _) -> strf "make refuses a GPU of %s below 1" field)
        "counts" counts
        (fun (field, gpu) ->
          raises_match
            (Exn.invalid_arg ~substring:("Rig_amd.make: the GPU's " ^ field))
            (fun () -> A.make ~word:(H.word ()) (path_of gpu)));
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

type gpu = S.t = { d : Rig.t; g : A.t }

let waits_on g =
  if not (A.facts g).waits.stores then
    skip ~reason:"the GPU's compute queue does not wait on words" ()

(* Code objects *)

let read_fixture name =
  In_channel.with_open_bin ("fixtures/" ^ name) In_channel.input_all

let kernels_bin = lazy (read_fixture "kernels_gfx1201.hsaco")
let work_bin = lazy (read_fixture "work_gfx1201.hsaco")
let launch_bin = lazy (read_fixture "launch_gfx1201.hsaco")
let launch_bin_942 = lazy (read_fixture "launch_gfx942.hsaco")

let launch_kernel name =
  let co = Result.get_ok (Abi.Code_object.of_string (Lazy.force launch_bin)) in
  Option.get (Abi.Code_object.kernel co name)

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
  S.with_ @@ fun { g; _ } ->
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
  S.with_ @@ fun { g; _ } ->
  let none = `Words 0 in
  equal never ~msg:"a part of 16 words" `Fits
    (E.room g [| E.raw ~queue:0 ~work:(`Words 16) () |]);
  equal never ~msg:"a copy on the compute queue" `Never
    (E.room g [| E.raw ~queue:0 ~work:(`Copy 64) () |]);
  List.iter
    (fun queue ->
      equal never ~msg:"a ring fill" `Never
        (E.room g [| E.raw ~queue ~work:`Fill () |]))
    [ 0; 1 ];
  equal never ~msg:"a part of no kind" `Never (E.room g [| E.raw ~queue:0 () |]);
  equal never ~msg:"after its own index" `Never
    (E.room g [| E.raw ~queue:1 ~work:none ~after:[| 0 |] () |]);
  equal never ~msg:"a queue of no index" `Never
    (E.room g [| E.raw ~queue:2 ~work:none () |]);
  equal never ~msg:"a negative queue" `Never
    (E.room g [| E.raw ~queue:(-1) ~work:none () |])

let misuse =
  group ~timeout:30. "misuse"
    [
      test "each misuse the interface states raises" misuse;
      test "the C room refuses what part refuses" c_room;
    ]

(* Images *)

let loads () =
  S.with_ @@ fun { g; _ } ->
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
            (code_of (A.entry m name));
          let e = found (A.entry m name) in
          equal
            (pair (option int) (option int))
            ~msg:(name ^ ": params and threads")
            (Some k.explicit_size, Some k.max_threads)
            (e.params, e.threads))
        [ "empty"; "double_index"; "spin"; "wild" ];
      equal (option int) ~msg:"no kernel" None (code_of (A.entry m "nothing"));
      equal (option int) ~msg:"a symbol of no kernel" None
        (code_of (A.entry m "__clang_ocl_kern_imp_spin"));
      A.unload g m;
      A.free g r

(* A code object for another processor, and bytes that are none. *)
let refusals () =
  S.with_ @@ fun { g; _ } ->
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
  let image bin = Result.map ignore (A.image g bin) in
  let arch = (A.facts g).arch in
  let other, processor =
    if arch = "gfx942" then (kernels_bin, "gfx1201")
    else (launch_bin_942, "gfx942")
  in
  equal result ~msg:"another processor"
    (Error (strf "a code object for %s; the GPU is %s" processor arch))
    (image (Lazy.force other));
  equal result ~msg:"not a code object"
    (Result.map ignore (Abi.Code_object.of_string "\127ELF but not one"))
    (image "\127ELF but not one")

let images =
  group ~timeout:30. "images"
    [
      test "an image names its kernels' descriptors in its code" loads;
      test "an image refuses what the GPU cannot run, saying why" refusals;
    ]

(* Entries *)

(* The bytes of [ids]'s parameters: [out]'s offset into its buffer, [a], [b] and
   [f]. *)
let ids_params ~out ~a ~b ~f =
  let p = Bytes.create 24 in
  Bytes.set_int64_le p 0 (Int64.of_int out);
  Bytes.set_int64_le p 8 (Int64.of_int a);
  Bytes.set_int32_le p 16 (Int32.of_int b);
  Bytes.set_int32_le p 20 (Int32.bits_of_float f);
  Bytes.to_string p

(* What [entry] makes of a kernel: a launch, once, with its scratch; and what it
   refuses. *)
let entries () =
  S.with_ @@ fun { g; _ } ->
  let m, r, _ = load g (Lazy.force launch_bin) in
  let e = found (A.entry m "ids") in
  not_equal nativeint ~msg:"a PM4 queue's launch" 0n e.launch;
  equal nativeint ~msg:"made once" e.launch (found (A.entry m "ids")).launch;
  raises_match ~msg:"a kernel with scratch that reads its dispatch packet"
    (Exn.invalid_arg ~substring:"dispatch packet") (fun () ->
      A.entry m "packet_scratch");
  not_equal nativeint ~msg:"a kernel with scratch" 0n
    (found (A.entry m "scratch")).launch;
  raises_match ~msg:"a kernel that reads its dispatch packet"
    (Exn.invalid_arg ~substring:"dispatch packet") (fun () ->
      A.entry m "packet");
  A.unload g m;
  A.free g r;
  let m, r, _ = load g (read_fixture "hidden_gfx1201.hsaco") in
  raises_match ~msg:"a kernel that reads a runtime's service"
    (Exn.invalid_arg ~substring:"hidden_printf_buffer") (fun () ->
      A.entry m "every");
  A.unload g m;
  A.free g r

(* What the room check answers for a launch: a launch fits within its function's
   256 work-items per group and the GPU's 64 KiB of LDS less the 256 bytes
   [lds] takes itself; it never fits past them, or with fewer parameter bytes
   than its function reads. *)
let launch_room () =
  S.with_ @@ fun { g; _ } ->
  let m, r, _ = load g (Lazy.force launch_bin) in
  let ids = found (A.entry m "ids") and lds = found (A.entry m "lds") in
  let params = ids_params ~out:0 ~a:0 ~b:0 ~f:0. in
  let room ?(e = ids) ?shared groups threads =
    E.room g [| E.launch e ~groups ~threads ?shared params [ (0, 0) ] |]
  in
  equal room_answer ~msg:"256 work-items" `Fits (room (1, 1, 1) (64, 2, 2));
  equal room_answer ~msg:"16 of the 24 bytes of ids's parameters" `Never
    (E.room g
       [|
         E.launch ids ~groups:(1, 1, 1) ~threads:(64, 1, 1)
           (String.sub params 0 16)
           [ (0, 0) ];
       |]);
  equal room_answer ~msg:"257 work-items" `Never (room (1, 1, 1) (257, 1, 1));
  equal room_answer ~msg:"the GPU's LDS" `Fits
    (room ~e:lds ~shared:(65536 - 256) (1, 1, 1) (64, 1, 1));
  equal room_answer ~msg:"past the GPU's LDS" `Never
    (room ~e:lds ~shared:(65536 - 255) (1, 1, 1) (64, 1, 1));
  A.unload g m;
  A.free g r

let entries =
  group ~timeout:30. "entries"
    [
      test
        "entry makes a launch once, with its scratch, and refuses what a \
         launch cannot write"
        entries;
      test "a launch fits within its function's and the GPU's limits"
        launch_room;
    ]

(* NOPs

   A PM4 NOP one word long (count 0x3fff), the word Linux's amdgpu driver pads
   compute rings with (gfx_v12_0.c). SDMA reads a zero word as a NOP, as the
   writer's own padding does. Words of 2^i NOPs, one per queue and size up to
   2^20, move a ring along cheaply. *)

let pm4_nop = 0xffff1000
let pads = Hashtbl.create 42

let pad ~compute i =
  match Hashtbl.find_opt pads (compute, i) with
  | Some f -> f
  | None ->
      let w = if compute then pm4_nop else 0 in
      let f =
        E.words
          ~queue:(if compute then "COMPUTE:0" else "COPY:0")
          (Array.make (1 lsl i) w)
      in
      Hashtbl.add pads (compute, i) f;
      f

(* The largest 2^i at most [n], up to 2^20. *)
let log2 n =
  let i = ref 0 in
  while !i < 20 && 1 lsl (!i + 1) <= n do
    incr i
  done;
  !i

(* Room

   [room] answers [`Later] only while one of the device's values is unreached:
   once the word holds the last value, it answers what it answers on a device
   that ran nothing. A wait on a host word holds the values unreached until the
   test runs them. *)

type work = Words of int | Copy of int
type spec = { compute : bool; work : work; after : int list }
type step = Submit of spec list | Burst of int | Run

let nops = lazy (Array.make (1 lsl 21) pm4_nop)
let zeros = lazy (Array.make (1 lsl 21) 0)

(* [specs] as parts of [g], whose copies run between [src] and [dst]. *)
let parts (src, dst) specs =
  let part s =
    let after = Array.of_list s.after in
    match s.work with
    | Words n when s.compute ->
        E.words ~queue:"COMPUTE:0" ~after (Array.sub (Lazy.force nops) 0 n)
    | Words n ->
        E.words ~queue:"COPY:0" ~after (Array.sub (Lazy.force zeros) 0 n)
    | Copy n -> E.copy ~after ~dst:(address dst) ~src:(address src) n
  in
  Array.of_list (List.map part specs)

let spec =
  let open Gen in
  let* compute = bool in
  let* work =
    if compute then
      map (fun n -> Words n) (of_list [ 1; 64; 1 lsl 16; 1 lsl 20; 1 lsl 21 ])
    else
      frequency
        [
          (4, map (fun n -> Words n) (of_list [ 1; 64; 1 lsl 19; 1 lsl 20 ]));
          (1, map (fun n -> Copy n) (of_list [ 1; 4096 ]));
        ]
  in
  let compute = compute && match work with Copy _ -> false | _ -> true in
  let+ after = list ~size:(int_range 0 2) (int_range 0 3) in
  { compute; work; after }

(* Steps that pass each ring's end and meet a full ring while a value is
   unreached. The rings a device opens with hold 2^22 words, of which a
   submission takes fewer than the compute ring's and at most half the copy
   ring's: the third step's 2^21 words do not fit beside the second's, both
   held. *)
let program =
  let c n = { compute = true; work = Words n; after = [] }
  and z n = { compute = false; work = Words n; after = [] }
  and copy n ~after = { compute = false; work = Copy n; after } in
  [
    Submit [ c 1; z 64; copy 4096 ~after:[ 0 ] ];
    Submit [ c (1 lsl 21) ];
    Submit [ c (1 lsl 21) ];
    Run;
    Burst 100;
    Submit [ c (1 lsl 20); copy 1 ~after:[ 0 ]; z (1 lsl 20) ];
    Submit [ z (1 lsl 20) ];
    Run;
    Submit [ z (1 lsl 20) ];
    Submit [ z (1 lsl 20) ];
    Run;
    Submit [ z (1 lsl 20); c (1 lsl 21) ];
    Run;
  ]

(* Another device of the GPU, beside the one [S.open_] gives, which the caller
   stops. A process takes a GPU with no kernel driver once. *)
let second () =
  if not (S.present ()) then skip ~reason:"no AMD GPU" ();
  match S.second ~word:(H.word ()) with
  | None -> skip ~reason:"one device of the GPU at a time" ()
  | Some r -> require_ok r

let regions g =
  (Option.get (A.alloc g Pinned 4096), Option.get (A.alloc g Pinned 4096))

(* The fresh device: a second device of the GPU that runs nothing, which the
   laws compare with. *)
let fresh =
  fixture
    ~teardown:(fun (f, _) -> A.stop f ~fault:None)
    (fun () ->
      let f = second () in
      (f, regions f))

let room_from_the_word () =
  let f, fr = fresh () in
  S.with_ @@ fun { g; _ } ->
  waits_on g;
  let rs = regions g in
  let flag = Option.get (A.alloc g Pinned 8) in
  let last = ref (A.signaled g) and laters = ref 0 in
  let hold () =
    H.set64 (host flag) 0;
    incr last;
    equal answer ~msg:"the hold" `Ok
      (E.submit g ~v:!last ~waits:[| (address flag, 1) |] [||])
  in
  let run () =
    H.set64 (host flag) 1;
    S.reached g !last
  in
  let one specs =
    let ps = parts rs specs in
    let reference () = E.room f (parts fr specs) in
    (* The word read before the room check: the check reads it at least as far
       along. *)
    let idle = A.signaled g = !last in
    let a = E.room g ps in
    if idle then equal room_answer ~msg:"every value reached" (reference ()) a;
    let a =
      if a <> `Later then a
      else begin
        incr laters;
        run ();
        let a = E.room g ps in
        equal room_answer ~msg:"once the word reached the last value"
          (reference ()) a;
        a
      end
    in
    if a = `Fits then begin
      incr last;
      equal answer ~msg:"submit" `Ok (submit g ~v:!last ps)
    end
  in
  hold ();
  List.iter
    (function
      | Submit specs -> one specs
      | Burst n ->
          for _ = 1 to n do
            one []
          done
      | Run ->
          run ();
          hold ())
    program;
  run ();
  at_least int ~msg:"rooms that answered Later" ~than:1 !laters;
  let wrapped q =
    let placed, size = A.ring g q in
    greater int ~msg:(q ^ ": the words placed on its ring") ~than:size placed
  in
  wrapped "COMPUTE:0";
  wrapped "COPY:0"

let rank = function `Fits -> 0 | `Later -> 1 | `Grow -> 2 | `Never -> 3

(* Adding actual parts cannot make work fit that was refused. *)
let more = Gen.pair (Gen.list ~size:(Gen.int_range 0 4) spec) spec

let grows_with_work (ss, s) =
  let f, fr = fresh () in
  let room ss =
    E.room f (parts fr (List.map (fun s -> { s with after = [] }) ss))
  in
  let before = room ss and alone = room [ s ] and after = room (ss @ [ s ]) in
  at_least int ~msg:"with one part more" ~than:(rank before) (rank after);
  at_least int ~msg:"with parts before it" ~than:(rank alone) (rank after)

let room =
  group ~timeout:300. "room"
    [
      test "room is Later only while a value is unreached" room_from_the_word;
      prop ~count:100
        "room never fits work with more parts than work it refused" more
        grows_with_work;
    ]

(* Growth

   A growth makes a larger argument segment, which the next room that needs it
   installs once the device is idle: while a value the device holds is
   unreached, that room answers [`Later], and so does every room after it, also
   for work that fits, until the value is reached and the segment installed. *)

let pp_grown ppf = function
  | Rig_edge.Grown -> Format.pp_print_string ppf "Grown"
  | Short n -> Format.fprintf ppf "Short %d" n

let grown = Testable.make ~pp:pp_grown ~equal:( = )

let held_growth () =
  S.with_ @@ fun { g; _ } ->
  waits_on g;
  let n = 200 in
  let flag = Option.get (A.alloc g Pinned 8) in
  let out = Option.get (A.alloc g Pinned (4 * n)) in
  H.set64 (host flag) 0;
  H.write (host out) (String.make (4 * n) '\000');
  let v = A.signaled g + 1 in
  equal answer ~msg:"the hold" `Ok
    (E.submit g ~v ~waits:[| (address flag, 1) |] [||]);
  let m, r, _ = load g (read_fixture "hidden_gfx1201.hsaco") in
  let wide = found (A.entry m "wide") in
  let params i = le64 (4 * i) ^ le32s [ i + 1; 0 ] in
  let launches =
    Array.init n (fun i ->
        E.launch wide ~groups:(1, 1, 1) ~threads:(1, 1, 1) (params i)
          [ (0, 0) ])
  in
  let fits = [| E.raw ~queue:0 ~work:(`Words 16) () |] in
  equal room_answer ~msg:"work that fits, before the growth" `Fits
    (E.room g fits);
  equal room_answer ~msg:"200 launches of 8 KiB of arguments" `Grow
    (E.room g launches);
  equal grown ~msg:"the growth" Grown (A.grow g);
  equal room_answer ~msg:"the launches, the value held" `Later
    (E.room g launches);
  equal room_answer ~msg:"work that fits, the value held" `Later
    (E.room g fits);
  H.set64 (host flag) 1;
  S.reached g v;
  equal room_answer ~msg:"the launches, the value reached" `Fits
    (E.room g launches);
  equal answer ~msg:"the launches" `Ok
    (E.submit g ~v:(v + 1) ~slots:[| address out |] launches);
  S.reached g (v + 1);
  equal string ~msg:"the stored words"
    (le32s (List.init n (fun i -> i + 1)))
    (H.read (host out) (4 * n));
  A.unload g m;
  List.iter (A.free g) [ r; flag; out ]

let growth =
  group ~timeout:60. "growth"
    [
      test "a growth is installed once the device is idle, every room waiting"
        held_growth;
    ]

(* On a GPU *)

(* Work through rig, on a device [S.open_] opened *)

let buffer ?(memory = Rig.Buffer.Device) t n = Rig.Buffer.create ~memory t.d n
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

let copy ~dst src =
  Rig_gpu_support.work Rig.Submission.(copy ~src:(Buffer src) ~dst:(Buffer dst))

(* Submits [ps] and waits for their point. *)
let run t ps = Rig.Point.wait (S.submit_work t (Array.to_list ps))

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

(* Fixed-address packets are encoded after their submission owns the memory. The
   words' count is independent of the addresses placed in them. *)
let encoded_words ~units ~fixed encode =
  let b = Rig.Buffer.create Rig.host (4 * units) in
  let work = Rig_gpu_support.work (Rig.Submission.words b) in
  {
    work with
    fixed;
    init =
      (fun s _ _ ->
        Rig.Buffer.write b Bigarray.int32 (fun data ->
            Array.iteri (fun i w -> data.{i} <- Int32.of_int w) (encode s)));
  }

let kernel ?of_ t p name ~groups ~fixed args =
  let a = buffer ~memory:Pinned t 4096 in
  let entry f = Rig.Image.entry (Option.get (Rig.Image.resolve p f)) in
  let dispatch args =
    dispatch ?of_ (A.capability t.g).gpu entry name ~args ~groups
  in
  let units = Array.length (dispatch 0) in
  let work =
    encoded_words ~units ~fixed:((a, Rig.Read) :: fixed)
      (fun s ->
        let addr = Rig.Submission.address s in
        let values = args addr in
        Rig.Buffer.blit_from_string values 0 a 0 (String.length values);
        dispatch (addr a))
  in
  { work with keep = Kept p }

let multiples k n = le32s (List.init n (fun i -> k * i))
let doubled = multiples 2

(* Launches on the GPU *)

module Sub = Rig.Submission

let launch_code = code_of launch_bin

(* Submits one launch of [p]'s [name] over [groups] of [threads], its groups
   taking [shared] bytes, whose block [set] stores, using each buffer of [uses]
   with its access, and waits for it. *)
let launch t p name ~params ~refs ~groups:(gx, gy, gz) ~threads:(tx, ty, tz)
    ?(shared = 0) ~set ~uses () =
  let buffers, access = Array.split (Array.of_list uses) in
  let ref_ (at, slot) =
    match access.(slot) with
    | Rig.Read -> Sub.read slot ~at
    | Read_write -> Sub.write slot ~at
  in
  let refs = Array.of_list (List.map ref_ refs) in
  let fn = Option.get (Rig.Image.resolve p name) in
  let s = Sub.make t.d [| Sub.launch ~refs fn ~params |] in
  let run = Sub.Run.make s and b = Sub.block s 0 in
  Sub.Run.groups run b gx gy gz;
  Sub.Run.threads run b tx ty tz;
  Sub.Run.shared run b shared;
  set run b;
  Rig.Point.wait (Rig.submit run ~buffers)

(* What [p]'s [scratch] kernel stores, over a group of 64 work-items each
   storing 5 times its index through its scratch. *)
let scratch_stores t p =
  let out = buffer ~memory:Pinned t (4 * 64) in
  launch t p "scratch" ~params:12
    ~refs:[ (0, 0) ]
    ~groups:(1, 1, 1) ~threads:(64, 1, 1)
    ~set:(fun run b ->
      Sub.Run.int64 run b 0 0L;
      Sub.Run.int32 run b 8 5)
    ~uses:[ (out, Rig.Read_write) ]
    ();
  get out

let scratch_launch () =
  S.with_ @@ fun t ->
  let p = image ~of_:launch_code t in
  equal string ~msg:"out" (multiples 5 64) (scratch_stores t p)

(* [n] words of [t]'s compute queue that do nothing, [n] a multiple of 16:
   PM4 NOPs, or AQL packets of the vendor's format that each run an indirect
   buffer of one PM4 NOP, in a word of [t]'s memory the work holds. *)
let nop_words (t : S.t) n =
  let b = Rig.Buffer.create Rig.host (4 * n) in
  let fill words =
    Rig.Buffer.write b Bigarray.int32 (fun data ->
        for i = 0 to n - 1 do
          data.{i} <- Int32.of_int words.(i mod Array.length words)
        done)
  in
  let work = Rig_gpu_support.work (Sub.words b) in
  match (A.capability t.g).compute with
  | Pm4 ->
      fill [| pm4_nop |];
      work
  | Aql _ ->
      let nop = buffer ~memory:Pinned t 4 in
      put nop (le32s [ pm4_nop ]);
      {
        work with
        fixed = [ (nop, Rig.Read) ];
        init =
          (fun s _ _ ->
            fill
              (words
                 (Abi.Aql.indirect_buffer (Sub.address s nop) ~dwords:1)));
      }

(* An AQL compute ring grows as a new queue, with a descriptor of its own: a
   submission of AQL packets that do nothing ([nop_words]) past the ring's
   room runs whole, and the scratch kernel then runs on the new queue, whose
   descriptor holds the scratch the old one's last value left. A GPU whose
   compute queue reads PM4 packets skips. *)
let aql_growth () =
  S.with_ @@ fun t ->
  (match (A.capability t.g).compute with
  | Pm4 -> skip ~reason:"the compute queue reads PM4 packets" ()
  | Aql _ -> ());
  let p = image ~of_:launch_code t in
  equal string ~msg:"before the growth" (multiples 5 64) (scratch_stores t p);
  let _, size = A.ring t.g "COMPUTE:0" in
  Rig.Point.wait (S.submit_work t [ nop_words t (size + 16) ]);
  let _, grown = A.ring t.g "COMPUTE:0" in
  at_least int ~msg:"the ring's size" ~than:(2 * size) grown;
  equal string ~msg:"after the growth" (multiples 5 64) (scratch_stores t p)

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
      Sub.Run.int64 run b 0 0L;
      Sub.Run.int32 run b 8 own;
      Sub.Run.int32 run b 12 n)
    ~uses:[ (out, Rig.Read_write) ]
    ();
  equal string ~msg:"out" (multiples 1 64) (get out)

(* Many submissions in flight reuse the argument segment while each launch
   writes a distinct output. Every ref must keep the value its submit copied. *)
let launch_segments_hw () =
  S.with_ @@ fun t ->
  let p = image t in
  let count = 64 and rounds = 32 and lanes = 64 in
  let out = buffer ~memory:Pinned t (4 * count * rounds * lanes) in
  put out (String.make (Rig.Buffer.length out) '\255');
  let sizes = [| 4096; 4096; 4096; 8; 63; 64; 65 |] in
  let fn = require_some (Rig.Image.resolve p "double_index") in
  let ps =
    Array.init count (fun i ->
        Sub.launch ~refs:[| Sub.write 0 ~at:0 |] fn
          ~params:sizes.(i mod Array.length sizes))
  in
  let s = Sub.make t.d ps in
  let run = Sub.Run.make s in
  let blocks = Array.init count (Sub.block s) in
  Array.iter
    (fun b ->
      Sub.Run.groups run b 1 1 1;
      Sub.Run.threads run b lanes 1 1)
    blocks;
  Sub.Run.release run;
  let last = ref None in
  for round = 0 to rounds - 1 do
    if not (Sub.Run.take run) then fail "the run is taken";
    Array.iteri
      (fun i b ->
        let at = 4 * lanes * ((round * count) + i) in
        Sub.Run.int64 run b 0 (Int64.of_int at))
      blocks;
    last := Some (Rig.submit run ~buffers:[| out |])
  done;
  Rig.Point.wait (Option.get !last);
  equal string ~msg:"every launch's output"
    (String.concat "" (List.init (count * rounds) (fun _ -> doubled lanes)))
    (get out)

(* A submission of [n] launches of [wide], whose arguments take 8 KiB each, runs
   whole: from 128 on they exceed half the device's first argument segment,
   which grows. A launch takes the room its arguments take. Launch [i] stores
   [i + 1] at word [i] of [out]. *)
let hidden_code = code_of (lazy (read_fixture "hidden_gfx1201.hsaco"))

(* What a submission of [n] launches of [p]'s [wide] on [t] stores into a
   buffer of its own: launch [i] stores [i + 1] at word [i]. *)
let tokens n = le32s (List.init n (fun i -> i + 1))

let stores_wide t p n =
  let out = buffer ~memory:Pinned t (4 * n) in
  let fn = Option.get (Rig.Image.resolve p "wide") in
  let s =
    Sub.make t.d
      (Array.init n (fun _ ->
           Sub.launch ~refs:[| Sub.write 0 ~at:0 |] fn ~params:16))
  in
  let run = Sub.Run.make s in
  for i = 0 to n - 1 do
    let b = Sub.block s i in
    Sub.Run.groups run b 1 1 1;
    Sub.Run.threads run b 1 1 1;
    Sub.Run.int64 run b 0 (Int64.of_int (4 * i));
    Sub.Run.int32 run b 8 (i + 1)
  done;
  Rig.Point.wait (Rig.submit run ~buffers:[| out |]);
  get out

let wide_launches n =
  S.with_ @@ fun t ->
  equal string ~msg:"the stored words" (tokens n)
    (stores_wide t (image ~of_:hidden_code t) n)

(* Two domains submit to one device: launches of 8 KiB of arguments, from one
   to past half of what its segment holds after two growths; words that do
   nothing, past what its compute ring holds, then past what the ring the first
   growth makes holds; and copies of 16 bytes, past what its copy ring holds.
   Each submission runs whole, launches and copies storing their bytes,
   whatever growth the other's needed. *)
type grower = { t : S.t; wide : Rig.Image.t }

(* Whether [g]'s copy ring keeps the 16 MiB it opens with, and room answers
   [`Never] past it: on SDMA 4 over PCI, whose path programs one copy queue
   (rig_amd.mli). *)
let copy_ring_fixed g =
  S.driverless ()
  &&
  let major, _, _ = (A.capability g).gpu.sdma in
  major = 4

let copy_fixed = lazy (S.with_ (fun t -> copy_ring_fixed t.g))

(* A submission of [n] words of [t]'s compute queue that do nothing
   ([nop_words]). *)
let nops_on t n = Rig.Point.wait (S.submit_work t [ nop_words t n ])

(* What a submission of [n] copies of 16 bytes on [t] stores, copy [i] taking
   bytes [16 i] to [16 i + 15] of a pattern. *)
let copies_on t n =
  let src = buffer ~memory:Pinned t (16 * n)
  and dst = buffer ~memory:Pinned t (16 * n) in
  put src (pattern (16 * n) 7);
  let part i = copy ~dst:(view dst (16 * i) 16) (view src (16 * i) 16) in
  Rig.Point.wait (S.submit_work t (List.init n part));
  get dst

(* The reference side holds whether the copy ring keeps its first size. *)
let grower = abstract "g" ~release:(fun x -> S.close x.t)

(* What a copy draw answers where the copy ring keeps its first size and the
   copies exceed it. *)
let never_fits = "never fits"

let launches =
  Gen.with_pp Format.pp_print_int (Gen.of_list [ 1; 100; 200; 600 ])

let nops =
  Gen.with_pp Format.pp_print_int
    (Gen.of_list [ 16; 1 lsl 22; (1 lsl 24) + 16 ])

let copies = Gen.with_pp Format.pp_print_int (Gen.of_list [ 1; 200_000 ])

let growth_commands =
  [
    command "start"
      (Gen.unit @-> makes grower)
      (fun () -> Lazy.force copy_fixed)
      (fun () ->
        let t = S.open_ () in
        { t; wide = image ~of_:hidden_code t });
    command "launches"
      (grower ^-> launches @-> returns string)
      (fun _ n -> tokens n)
      (fun x n -> stores_wide x.t x.wide n);
    command "nops"
      (grower ^-> nops @-> returns unit)
      (fun _ _ -> ())
      (fun x n -> nops_on x.t n);
    command "copies"
      (grower ^-> copies @-> returns string)
      (fun fixed n ->
        if fixed && n > 1 then never_fits else pattern (16 * n) 7)
      (fun x n ->
        match copies_on x.t n with
        | s -> s
        | exception Invalid_argument _ -> never_fits);
  ]

let launching =
  group ~timeout:60. "launching"
    [
      test "launch arguments survive segment reuse with work in flight"
        launch_segments_hw;
      test "a launch of a kernel that takes scratch computes" scratch_launch;
      test "an AQL compute ring grows, its scratch kept" aql_growth;
      cases ~name:string_of_int
        "a submission of launches of 8 KiB of arguments each runs whole"
        [ 1; 128; 300; 1000 ] wide_launches;
      stateful ~count:6 ~steps:8 ~domains:2
        "submissions from two domains run whole while the segment and rings \
         grow"
        growth_commands;
      cases ~name:string_of_int
        "a launch's dynamic LDS follows the kernel's own" [ 64; 4096; 16320 ]
        (fun n -> lds_launch n ());
    ]

(* rig finds the record the device's facts declare, and none on the host. *)
let declared () =
  S.with_ @@ fun t ->
  (match Rig.capability t.d Abi.key with
  | Some c -> equal bool ~msg:"the record" true (c == A.capability t.g)
  | None -> fail "no record under the key");
  is_none ~msg:"the host" (Rig.capability Rig.host Abi.key)

(* [locate] names a region by its GPU address, its handle too, and the host
   addresses every region but Device memory. *)
let locates () =
  S.with_ @@ fun { g; _ } ->
  let regions =
    List.filter_map
      (fun (name, kind, hosted) ->
        Option.map (fun r -> (name, r, hosted)) (A.alloc g kind 4096))
      [
        ("Device", Device, false);
        ("Pinned", Pinned, true);
        ("Mapped", Mapped, true);
      ]
  in
  List.iter
    (fun (name, r, hosted) ->
      let l = A.locate r in
      equal bool ~msg:(name ^ ": host") hosted (Option.is_some l.host);
      equal (option nativeint) ~msg:(name ^ ": handle")
        (Option.map Nativeint.of_int l.address)
        (Some l.handle);
      A.free g r)
    regions

(* [map_host] maps host pages for writing: it refuses read-only pages and, on
   amdgpu, pages of which some but not all lie within a mapping of the GPU. *)
let map_host_refusals () =
  S.with_ @@ fun { g; _ } ->
  let n = 4096 in
  if not (A.facts g).maps_host then
    skip ~reason:"the device maps no host memory" ();
  let read_only = H.pages ~read_only:true n in
  is_none ~msg:"read-only pages" (A.map_host g read_only n);
  H.free_pages read_only n;
  if not (S.driverless ()) then begin
    let pages = H.pages (2 * n) in
    let m = Option.get (A.map_host g pages n) in
    is_none ~msg:"pages partly within a mapping" (A.map_host g pages (2 * n));
    A.free g m;
    H.free_pages pages (2 * n)
  end

(* A device's trace buffers are refused once it stopped. *)
let trace_after_close () =
  let t = S.open_ () in
  let c = A.capability t.g in
  S.close t;
  is_error ~msg:"after the close" (c.trace ())

(* The hooks that read a device's rings refuse once it stopped, whose stop
   freed them. *)
let hooks_after_close () =
  let t = S.open_ () in
  S.close t;
  raises (Invalid_argument "Rig_amd.ring: the device stopped") (fun () ->
      A.ring t.g "COMPUTE:0");
  raises (Invalid_argument "Rig_amd.renumber: the device stopped") (fun () ->
      A.renumber t.g (1 lsl 32))

let work =
  group ~timeout:60. "work"
    [
      test "locate names a region by its GPU address" locates;
      test "map_host refuses pages it cannot map for writing" map_host_refusals;
      test "trace buffers are refused once the device closed" trace_after_close;
      test "the ring hooks refuse once the device closed" hooks_after_close;
      test "an opened GPU's facts are its capability's" (fun () ->
          S.with_ @@ fun t ->
          let c = A.capability t.g in
          let f = A.facts t.g in
          let runs =
            List.map (function
              | Rig_edge.Words -> "words"
              | Fill -> "fill"
              | Copy -> "copy"
              | Launch -> "launch")
          in
          equal string ~msg:"arch" (Gpu.processor c.gpu) f.arch;
          greater int ~msg:"budget" ~than:0 f.budget;
          equal
            (list (pair string (list string)))
            ~msg:"queues"
            [
              ("COMPUTE:0", [ "words"; "launch" ]);
              ("COPY:0", [ "words"; "copy" ]);
            ]
            (List.map
               (fun (q : Rig_edge.queue) -> (q.name, runs q.runs))
               f.queues);
          equal (option int) ~msg:"the copy engine" (Some 1) f.copy_engine;
          equal bool ~msg:"completion is the word" true (f.completion = Store);
          equal bool ~msg:"may block" false f.may_block;
          equal bool ~msg:"host addresses" false f.host_addresses;
          equal int ~msg:"alignment" 4096 f.alignment;
          equal bool ~msg:"waits on words as on the host" f.waits.hosts
            f.waits.stores;
          equal (option int) ~msg:"hang bound" (Some 30_000) f.hang_ms;
          greater int ~msg:"clock" ~than:0 c.clock_hz;
          equal int ~msg:"shader engines"
            (c.gpu.shader_engines * c.gpu.xccs)
            (Array.length c.wgps);
          equal bool ~msg:"AQL" (c.gpu.xccs > 1)
            (match c.compute with Aql _ -> true | Pm4 -> false));
      test "rig finds its record" declared;
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

let spin t p flag n =
  kernel t p "spin" ~groups:1
    ~fixed:[ (flag, Rig.Read_write) ]
    (fun addr -> le64 (addr flag) ^ le64 n)

(* Code placed where other code ran runs as placed: each image is unloaded once
   unreachable and its work done, and the next load reuses the memory. *)
let stale_code () =
  S.with_ @@ fun t ->
  let out = buffer ~memory:Pinned t (4 * 64) in
  let load of_ =
    let p = image ~of_ t in
    run t
      [|
        kernel ~of_ t p "double_index" ~groups:1
          ~fixed:[ (out, Rig.Read_write) ]
          (fun addr -> le64 (addr out));
      |];
    let fn = Option.get (Rig.Image.resolve p "double_index") in
    let at = Rig.Image.entry fn in
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
          run t
            [|
              kernel t p "double_index" ~groups:4
                ~fixed:[ (out, Rig.Read_write) ]
                (fun addr -> le64 (addr out));
            |];
          equal string ~msg:"out" (doubled 256) (get out));
      test "code placed where other code ran runs as placed" stale_code;
      test "Words places its packets" (fun () ->
          S.with_ @@ fun t ->
          let word = buffer ~memory:Pinned t 8 in
          put word (le64 0);
          let encode addr = words (Pm4.write_data (Memory addr) 0xc0ffee) in
          let work =
            encoded_words
              ~units:(Array.length (encode 0))
              ~fixed:[ (word, Rig.Read_write) ]
              (fun s -> encode (Sub.address s word))
          in
          run t [| work |];
          equal string ~msg:"word" (le64 0xc0ffee) (get word));
      test "a release wakes a sleeper well before its bound" (fun () ->
          (* A few milliseconds of work, slept on with a bound of seconds: the
             release's interrupt, not the bound, ends the sleep. *)
          S.with_ @@ fun t ->
          let p = image t in
          let flag = buffer ~memory:Pinned t 8 in
          put flag (le64 0);
          let v = Rig.Point.value (S.submit_work t [ spin t p flag 2_000 ]) in
          let t0 = Rig.Profile.now () in
          while A.signaled t.g < v do
            A.sleep t.g ~seen:(A.signaled t.g) ~still_ms:5_000
          done;
          less int ~msg:"ms asleep" ~than:1_000
            ((Rig.Profile.now () - t0) / 1_000_000));
      test "closing completes submitted work" (fun () ->
          (* Close waits for the kernel. Its mapped output remains readable. *)
          S.with_ @@ fun t ->
          let p = image t in
          let flag = buffer ~memory:Pinned t 8 in
          H.set64 (H.at flag) 0;
          let v = Rig.Point.value (S.submit_work t [ spin t p flag 2_000 ]) in
          S.close t;
          equal int ~msg:"the word" v (Rig.signaled t.d);
          equal int ~msg:"the kernel's flag" 1 (H.get64 (H.at flag));
          equal (option string) ~msg:"the device ended" (Some "closed")
            (Rig.lost t.d));
    ]

(* Work in order on one queue

   Parts on one queue run in array order: a compute part after another sees its
   writes, whatever the sizes and whether a copy part sits between them. *)

(* The shared device's image of fixtures/work.cl, loaded once. *)
let work_image = lazy (image ~of_:work_code (shared ()))

module Chain = struct
  (* A submission of [incs] compute parts, each adding 1 to the words its
     predecessor wrote, with a copy part between two of them where [via] says,
     after a submission released on COMPUTE:0 or on COPY:0. *)
  type t = { words : int; incs : bool list; after_copy : bool }

  let pp ppf c =
    Format.fprintf ppf "{%d words; %s; after a %s release}" c.words
      (String.concat ", "
         (List.map
            (fun via -> if via then "words then copy" else "words")
            c.incs))
      (if c.after_copy then "COPY:0" else "COMPUTE:0")

  let most = 1 lsl 20
  let buffers = 8

  (* Parts back to back and a copy part between two; a value released before
     on each queue; a part of 2^20 words, which keeps every compute unit busy,
     and one of 1000, whose last group is partly empty. *)
  let examples =
    [
      { words = most; incs = [ false; true; false ]; after_copy = false };
      { words = 1000; incs = [ true; false; true; false ]; after_copy = true };
    ]

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
    (* Garbage in every buffer, then the input in the first. *)
    run t
      (Array.map
         (fun b -> copy ~dst:(part b) (part s.garbage))
         (Array.append s.bufs [| s.out |]));
    run t [| copy ~dst:(part s.bufs.(0)) (part s.input) |];
    if c.after_copy then
      run t [| copy ~dst:(view s.out 0 4) (view s.garbage 0 4) |]
    else
      run t
        [|
          Rig_gpu_support.work (S.words_part [| pm4_nop |]);
        |];
    let parts = ref [] in
    let add p =
      parts := p :: !parts;
      List.length !parts - 1
    in
    let src = ref s.bufs.(0) and next = ref 1 and copied = ref None in
    List.iteri
      (fun i via ->
        let out = s.bufs.(!next) in
        incr next;
        let src_buffer = !src in
        let p =
          kernel ~of_:work_code t (Lazy.force work_image) "inc"
            ~groups:((n + 63) / 64)
            ~fixed:[ (out, Rig.Read_write); (src_buffer, Rig.Read) ]
            (fun addr -> le64 (addr out) ^ le64 (addr src_buffer) ^ le32s [ n ])
        in
        ignore (add p);
        copied := None;
        src := out;
        if via && i < k - 1 then begin
          let mid = s.bufs.(!next) in
          incr next;
          copied := Some (add (copy ~dst:(part mid) (part out)));
          src := mid
        end)
      c.incs;
    ignore (add (copy ~dst:(part s.out) (part !src)));
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
      cases
        ~name:(Format.asprintf "%a" Chain.pp)
        "compute parts run in array order, each reading its predecessor's \
         writes"
        Chain.examples Chain.law;
    ]

(* Through the C entries *)

(* A device's values, numbered on from its word's as the test hands them to its
   C entries. *)
type run = { g : A.t; mutable v : int }

let device g = { g; v = A.signaled g }

let go r ps =
  r.v <- r.v + 1;
  equal answer ~msg:"submit" `Ok (submit r.g ~v:r.v ps);
  S.reached r.g r.v

(* Rings wrap

   Submissions wrap each ring three times, landing a submission exactly at the
   end, one word short of it or one word past it, each after submissions of
   every shape. Each is chosen from where its ring stands ([A.ring]), and every
   value completes in order and writes what it should. *)

module Wrap = struct
  let slots = 4096

  type shape =
    | Pad of bool * int (* 2^i NOPs on COMPUTE:0 (true) or COPY:0 *)
    | Nops of int (* COMPUTE:0: NOP words, then the value's write *)
    | Zeros of int (* COPY:0: zero words, then the value's copy *)
    | Both of int * int (* Nops and Zeros in one submission *)

  type target = Compute | Copy
  type fit = Exact | Short | Over

  (* Each ring's three landings, the rings in turn, each after submissions. *)
  let goals =
    [
      (Compute, Exact, [ Nops 7; Zeros 4096 ]);
      (Copy, Short, [ Pad (true, 20); Both (13, 3) ]);
      (Compute, Over, [ Zeros 1; Nops (1 lsl 16) ]);
      (Copy, Exact, [ Pad (false, 10); Nops 0; Both (0, 0) ]);
      (Compute, Short, [ Pad (false, 0); Zeros 0 ]);
      (Copy, Over, [ Nops 4096; Zeros (1 lsl 16); Pad (true, 0) ]);
    ]

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

  (* [shape]'s parts as value [v]. *)
  let parts k v shape =
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
    | Pad (c, i) -> [| pad ~compute:c i |]
    | Nops n -> [| compute n |]
    | Zeros n -> Array.of_list (copy n)
    | Both (n, z) -> Array.of_list (compute n :: copy z)

  (* Submits each shape as the next value from [first], once its parts fit: the
     word never moves back. *)
  let submitter g k ~first =
    let v = ref (first - 1) and seen = ref 0 in
    let rec room ps =
      match E.room g ps with
      | `Fits -> ()
      | `Never -> fail "Never"
      | `Grow -> fail "Grow"
      | `Later ->
          S.reached g (A.signaled g + 1);
          room ps
    in
    fun shape ->
      incr v;
      let ps = parts k !v shape in
      room ps;
      equal answer ~msg:"submit" `Ok (submit g ~v:!v ps);
      let w = A.signaled g in
      at_least int ~msg:"the word" ~than:!seen w;
      seen := w

  (* Once the word reaches the last of [shapes], the values from [first], the
     slots hold what the last value that wrote each wrote. *)
  let check g k ~first shapes =
    S.reached g (first + List.length shapes - 1);
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
        | Nops _ -> compute ()
        | Zeros _ -> copy ()
        | Both _ ->
            compute ();
            copy ())
      shapes;
    equal string ~msg:"the compute writes" (Bytes.to_string res)
      (H.read (host k.res) (4 * slots));
    equal string ~msg:"the copies" (Bytes.to_string dst)
      (H.read (host k.dst) (8 * slots))

  (* Runs [shapes] as the values from [first] of [g]. *)
  let run g ~first shapes =
    let k = sink g in
    List.iter (submitter g k ~first) shapes;
    check g k ~first shapes

  (* Runs shapes that reach [goals] on a fresh device, each chosen from where
     its ring stands. *)
  let law () =
    S.with_ @@ fun { g; _ } ->
    let k = sink g in
    let first = A.signaled g + 1 in
    let submit = submitter g k ~first in
    let shapes = ref [] in
    let go shape =
      submit shape;
      shapes := shape :: !shapes
    in
    let shift = function Exact -> 0 | Short -> -1 | Over -> 1 in
    (* [mk n] places [n >= least] words of its own and [o] more, measured on a
       second one so that its queue released the value before it. Pads of NOPs
       approach the end, each leaving more room than its own release takes. *)
    let land_ring q ~compute:c mk ~least f =
      let at () = fst (A.ring g q) and size = snd (A.ring g q) in
      go (mk 64);
      let p = at () in
      go (mk 64);
      let o = at () - p - 64 in
      let rec landing () =
        let p = at () in
        let stop = (((p / size) + 1) * size) + shift f in
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
          if not (q = "COPY:0" && f = Over) then
            equal int ~msg:(strf "where the %s landing ends" q) stop (at ())
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
            land_ring "COMPUTE:0" ~compute:true
              (fun n -> Nops (n - wd))
              ~least:wd f
        | Copy ->
            land_ring "COPY:0" ~compute:false (fun n -> Zeros n) ~least:1 f)
      goals;
    let thrice q =
      let placed, size = A.ring g q in
      greater int ~msg:(q ^ ": the words placed on its ring") ~than:(2 * size)
        placed
    in
    thrice "COMPUTE:0";
    thrice "COPY:0";
    check g k ~first (List.rev !shapes)

  (* Values from 2^32 - [k], where a value's release goes on the queue of its
     submission's last part and slot words compare 32 bits: the edge's own
     numbering, which rig's values reach only after 2^32 submits. The value at
     2^32 is the shape at [k], which ends on each queue in one case. *)
  let carried =
    [
      ( "COMPUTE:0",
        (2, [ Both (13, 3); Zeros 0; Nops 0; Both (0, 0); Zeros 0 ]) );
      ("COPY:0", (1, [ Nops 0; Both (13, 3); Nops 0; Zeros 0 ]));
    ]

  let across_carry (k, shapes) =
    S.with_ @@ fun { g; _ } ->
    (* The first carry the word can still reach k values ahead of. *)
    let carry = (((A.signaled g + k) lsr 32) + 1) lsl 32 in
    let first = carry - k in
    A.renumber g first;
    run g ~first shapes
end

(* Waits on words

   Where the compute queue compares 64-bit words ([waits_on]), a submission
   holds its work until each word it waits on, as unsigned 64 bits, reaches its
   value; a 32-bit compare would pass a word whose low half alone is above. The
   words here are the host's, in memory the device maps. *)

(* A wait for [t] at or past 2^32 and the word's first value [w0] below it: a
   wait for 2^32, whose low half is 0, and a word at 2^32 - 1, whose low half
   is above the wait's. *)
let crossing = [ ((1 lsl 32) - 2, 1 lsl 32); ((1 lsl 32) - 1, (1 lsl 32) + 1) ]

let crossing_name (w0, t) =
  strf "word 2^32%+d, wait for 2^32%+d" (w0 - (1 lsl 32)) (t - (1 lsl 32))

let held (w0, t) =
  S.with_ @@ fun { g; _ } ->
  waits_on g;
  let base = A.signaled g in
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
  equal answer ~msg:"submit" `Ok
    (E.submit g ~v:(base + 1)
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
      equal int
        ~msg:(strf "the timeline, the word at %d" x)
        base (A.signaled g))
    below;
  H.write (host word) (le64 t);
  S.reached g (base + 1);
  equal string ~msg:"the copy, the word reached" (String.make 64 'w')
    (H.read (host dst) 64);
  List.iter (A.free g) [ word; src; dst ]

(* A submission waits in its compute queue on every producer it follows,
   however many: 300 Polled devices' values, more than a fixed reserve of
   waits held, none of them reached when the submit returns. *)
let many_waits () =
  S.with_ @@ fun { d; g } ->
  waits_on g;
  if not (A.facts g).maps_host then
    skip ~reason:"the GPU maps no host word to wait on" ();
  let module P = Rig_support.Polled in
  let producers = List.init 300 (fun i -> P.open_ (strf "amd:wait-%d" i)) in
  let close () = List.iter (fun (p, _) -> Rig.close p) producers in
  Fun.protect ~finally:close @@ fun () ->
  let point (p, _) =
    let run = Rig.Submission.Run.make (Rig.Submission.make p [||]) in
    Rig.submit run ~buffers:[||]
  in
  let points = Array.of_list (List.map point producers) in
  let s = Rig.Submission.make d [||] in
  let v = Rig.submit (Rig.Submission.Run.make s) ~waits:points ~buffers:[||] in
  let reached = List.filter (fun (_, p) -> P.queued p = 0) producers in
  equal int ~msg:"producers reached on the host" 0 (List.length reached);
  List.iter (fun (_, p) -> ignore (P.run p)) producers;
  Rig.Point.wait v

let waits =
  group ~timeout:120. "waits"
    [
      cases ~name:crossing_name
        "a wait holds the work until the word reaches it, across 2^32 (sampled)"
        crossing held;
      test "a submission waits in its queue on 300 producers" many_waits;
    ]

(* Joins past 512 parts

   A submission of [n] parts alternates between the queues: each compute part
   writes a token, and the copy part after it copies the token out, so every
   part but the first joins the one before it on the other queue. A wait that
   passed before its signal would copy a cell before its token. Half the draws
   take the value 2^32, whose low 32 bits are 0. *)
let joins =
  Gen.(
    with_pp
      (fun ppf (n, carry) ->
        Format.fprintf ppf "%d parts%s" n (if carry then " at 2^32" else ""))
      (pair
         (frequency
            [
              (2, of_list [ 1; 2; 511; 512 ]);
              (3, of_list [ 513; 514; 1024 ]);
              (3, constant 4096);
            ])
         bool))

let joined (n, carry) =
  cover "more than 512 parts" (n > 512);
  cover "4096 parts" (n = 4096);
  S.with_ @@ fun { g; _ } ->
  let cells = (n + 1) / 2 in
  let bytes = 4 * cells in
  let src = Option.get (A.alloc g Pinned bytes) in
  let out = Option.get (A.alloc g Pinned bytes) in
  H.write (host src) (String.make bytes '\000');
  H.write (host out) (String.make bytes '\000');
  let token m = m + 1 in
  let at r m = address r + (4 * m) in
  if carry then A.renumber g (((A.signaled g lsr 32) + 1) lsl 32);
  let v = A.signaled g + 1 in
  let ps =
    Array.init n (fun i ->
        let m = i / 2 and after = if i = 0 then [||] else [| i - 1 |] in
        if i mod 2 = 0 then
          E.words ~queue:"COMPUTE:0" ~after
            (words (Pm4.write_data (Memory (at src m)) (token m)))
        else E.copy ~after ~dst:(at out m) ~src:(at src m) 4)
  in
  equal room_answer ~msg:"room" `Fits (E.room g ps);
  equal answer ~msg:(strf "value %d" v) `Ok (submit g ~v ps);
  S.reached g v;
  let got = H.read (host out) bytes in
  for m = 0 to (n / 2) - 1 do
    equal int ~msg:(strf "cell %d" m) (token m)
      (Int32.to_int (String.get_int32_le got (4 * m)) land 0xffff_ffff)
  done;
  A.free g src;
  A.free g out

(* Rings grow

   Words past a ring's room ask for a growth, which makes a new queue on a
   larger ring, and once the device is idle they run on it, and so does work
   after them. The compute queue's words are PM4: a GPU whose compute queue
   reads AQL packets skips. On SDMA 4 over PCI the copy ring keeps its first
   size, and room answers [`Never] past it ([copy_ring_fixed]). *)

(* [g]'s ring of queue [q], of [size] words, grows for the words [past], which
   then run on the new queue through [r], as does a write or a copy after them.
   *)
let grows_past g q r size past =
  let compute = q = "COMPUTE:0" in
  equal room_answer ~msg:"words past the ring" `Grow (E.room g [| past |]);
  equal grown ~msg:"the growth" Grown (A.grow g);
  equal room_answer ~msg:"once grown, the device idle" `Fits
    (E.room g [| past |]);
  let _, grown_size = A.ring g q in
  at_least int ~msg:"the ring's size" ~than:(2 * size) grown_size;
  go r [| past |];
  let out = Option.get (A.alloc g Pinned 64) in
  let src = Option.get (A.alloc g Pinned 64) in
  H.write (host out) (String.make 64 '\000');
  H.write (host src) (String.make 64 'w');
  if compute then begin
    go r
      [|
        E.words ~queue:q (words (Pm4.write_data (Memory (address out)) 0x2a));
      |];
    equal string ~msg:"the word written after the growth" (le32s [ 0x2a ])
      (H.read (host out) 4)
  end
  else begin
    go r [| E.copy ~dst:(address out) ~src:(address src) 64 |];
    equal string ~msg:"the copy after the growth" (String.make 64 'w')
      (H.read (host out) 64)
  end;
  A.free g out;
  A.free g src

let grown_ring compute () =
  S.with_ @@ fun { g; _ } ->
  (match (A.capability g).compute with
  | Aql _ when compute -> skip ~reason:"the compute queue reads AQL packets" ()
  | Aql _ | Pm4 -> ());
  let q = if compute then "COMPUTE:0" else "COPY:0" in
  let fill n =
    E.words ~queue:q (Array.make n (if compute then pm4_nop else 0))
  in
  let r = device g in
  let _, size = A.ring g q in
  let past = fill (if compute then size else size / 2) in
  if (not compute) && copy_ring_fixed g then
    equal room_answer ~msg:"words past the ring it keeps" `Never
      (E.room g [| past |])
  else grows_past g q r size past

(* A growth that needs more than a queue a growth made and no room installed
   replaces that queue: while a value is held, words [fill first] past the ring
   of queue [q] ask for a growth, then words [fill past] past the handed ring
   ask for another, and once the value is reached the larger queue is
   installed, runs the words, and [after] then runs on it. A growth's ring
   holds twice what it was asked for, so [past] is past the ring the first
   growth handed. Over PCI, which holds two queues of a kind, the second growth
   takes back the first's queue and makes its own in the slot that frees. *)
let regrown_past g q fill ~first ~past ~after =
  waits_on g;
  let flag = Option.get (A.alloc g Pinned 8) in
  H.set64 (host flag) 0;
  let v = A.signaled g + 1 in
  equal answer ~msg:"the hold" `Ok
    (E.submit g ~v ~waits:[| (address flag, 1) |] [||]);
  equal room_answer ~msg:"words past the ring" `Grow
    (E.room g [| fill first |]);
  equal grown ~msg:"the first growth" Grown (A.grow g);
  let n = past in
  let past = fill n in
  equal room_answer ~msg:"words past the handed ring" `Grow
    (E.room g [| past |]);
  equal grown ~msg:"the second growth" Grown (A.grow g);
  equal room_answer ~msg:"the value held" `Later (E.room g [| past |]);
  H.set64 (host flag) 1;
  S.reached g v;
  equal room_answer ~msg:"the value reached" `Fits (E.room g [| past |]);
  let _, size = A.ring g q in
  at_least int ~msg:"the installed ring's size" ~than:(2 * n) size;
  let r = device g in
  go r [| past |];
  after r;
  A.free g flag

let regrown () =
  S.with_ @@ fun { g; _ } ->
  let fill n = E.words ~queue:"COPY:0" (Array.make n 0) in
  let _, size = A.ring g "COPY:0" in
  if copy_ring_fixed g then
    equal room_answer ~msg:"words past the ring it keeps" `Never
      (E.room g [| fill (size / 2) |])
  else
    let after r =
      let out = Option.get (A.alloc g Pinned 64) in
      let src = Option.get (A.alloc g Pinned 64) in
      H.write (host out) (String.make 64 '\000');
      H.write (host src) (String.make 64 'w');
      go r [| E.copy ~dst:(address out) ~src:(address src) 64 |];
      equal string ~msg:"the copy after the growths" (String.make 64 'w')
        (H.read (host out) 64);
      List.iter (A.free g) [ out; src ]
    in
    (* The copy ring's room is half its words. *)
    regrown_past g "COPY:0" fill ~first:(size / 2) ~past:(2 * size) ~after

(* [g]'s compute queue words of the PM4 packets [pm4]: those, or on an AQL
   queue the words of a packet that runs them as an indirect buffer from
   [Pinned] memory, which it returns to be freed. *)
let compute_words g pm4 =
  match (A.capability g).compute with
  | Pm4 -> (pm4, None)
  | Aql _ ->
      let ib = Option.get (A.alloc g Pinned (4 * Array.length pm4)) in
      H.write (host ib) (le32s (Array.to_list pm4));
      ( words
          (Abi.Aql.indirect_buffer (address ib) ~dwords:(Array.length pm4)),
        Some ib )

(* The compute ring's twin of [regrown], with words that do nothing as
   [nop_words] makes them, and a write on the installed queue after them. *)
let regrown_compute () =
  S.with_ @@ fun { g; _ } ->
  let nop, ib = compute_words g [| pm4_nop |] in
  let fill n =
    E.words ~queue:"COMPUTE:0"
      (Array.init n (fun i -> nop.(i mod Array.length nop)))
  in
  let _, size = A.ring g "COMPUTE:0" in
  let after r =
    let out = Option.get (A.alloc g Pinned 64) in
    H.write (host out) (String.make 64 '\000');
    let write, ib =
      compute_words g (words (Pm4.write_data (Memory (address out)) 0x2a))
    in
    go r [| E.words ~queue:"COMPUTE:0" write |];
    equal string ~msg:"the word written after the growths" (le32s [ 0x2a ])
      (H.read (host out) 4);
    List.iter (A.free g) (out :: Option.to_list ib)
  in
  (* The compute ring's room is its words less one, and [size] is a multiple
     of an AQL packet's 16 words. *)
  regrown_past g "COMPUTE:0" fill ~first:size ~past:(4 * size) ~after;
  Option.iter (A.free g) ib

let rings =
  group ~timeout:300. "rings"
    [
      test "a second growth before an install replaces the first's queue"
        regrown;
      test
        "a second compute growth before an install replaces the first's queue"
        regrown_compute;
      test "the compute ring grows as a new queue" (grown_ring true);
      test "the copy ring grows as a new queue" (grown_ring false);
      prop ~count:24 "a submission of parts joined across the queues runs whole"
        joins joined;
      test
        "submissions landing at, short of and past each ring's end complete in \
         order"
        Wrap.law;
    ]

let timeline =
  group ~timeout:300. "timeline"
    [
      cases
        ~name:(fun (q, _) -> "the value at 2^32 ending on " ^ q)
        "values across 2^32 complete in order, ending on either queue"
        Wrap.carried
        (fun (_, c) -> Wrap.across_carry c);
    ]

(* Two devices of one GPU *)

(* With no kernel driver, a process takes a GPU once: a second open is refused,
   naming the GPU taken, and the first device goes on. *)
let one_gpu_driverless t =
  (match S.open_gpu ~word:(Rig_gpu_support.Host.word ()) with
  | Ok g' ->
      A.stop g' ~fault:None;
      fail "a second open of a GPU the process took"
  | Error why -> contains ~sub:"is open in this process" why);
  let p = S.submit t [||] in
  equal int ~msg:"the first device's next value" 1 (Rig.Point.value p);
  Rig.Point.wait p

(* Through amdgpu, a process opens a GPU as often as it likes: the devices, [g]
   and one [second] opens, share its memory. *)
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
  else
    one_gpu_shared
      ~second:(fun () -> S.open_gpu ~word:(Rig_gpu_support.Host.word ()))
      t.g

(* A view of another device's Mapped memory flushes that device's host writes
   until its stop: the viewer's work reads the host's bytes through the view,
   the viewer submits after the stop as before, and a view of the stopped
   device's Mapped memory is None, as nothing would flush the host's writes
   before the viewer's work. *)
let stopped_mapped () =
  if S.driverless () then skip ~reason:"amdgpu does not hold the GPU" ();
  S.with_ @@ fun t ->
  let g = t.g in
  let g' =
    match S.open_gpu ~word:(Rig_gpu_support.Host.word ()) with
    | Ok g' -> g'
    | Error why -> fail why
  in
  let n = 4096 in
  match A.alloc g' Mapped n with
  | None ->
      A.stop g' ~fault:None;
      skip ~reason:"the GPU has no Mapped memory" ()
  | Some m ->
      let view = require_some ~msg:"a view" (A.map_peer g g' m) in
      let r = device g in
      let back = Option.get (A.alloc g Pinned n) in
      let p = pattern n 13 in
      H.write (host m) p;
      go r [| E.copy ~dst:(address back) ~src:(address view) n |];
      equal string ~msg:"the host's bytes, read through the view" p
        (H.read (host back) n);
      A.stop g' ~fault:None;
      go r [||];
      is_none ~msg:"a view of a stopped device's Mapped memory"
        (A.map_peer g g' m);
      A.free g back;
      A.free g view;
      A.free g' m

(* Every root shows this machine: a root that links to [/] reaches the same GPU,
   whose address space the process holds once, so devices opened through either
   root share its memory, whichever opens first. The link lives in the suite's
   working directory only while the test runs: one left in a source tree, where
   the suite runs on a GPU host, has dune walk the whole filesystem. *)
let mirror = "mirror"

let two_roots () =
  if S.driverless () then skip ~reason:"amdgpu does not hold the GPU" ();
  (match Unix.lstat mirror with
  | _ -> Unix.unlink mirror
  | exception Unix.Unix_error (Unix.ENOENT, _, _) -> ());
  Unix.symlink "/" mirror;
  Fun.protect ~finally:(fun () -> Unix.unlink mirror) @@ fun () ->
  let through_mirror () =
    Rig_gpu_support.opened (Rig_amd_amdgpu.gpu_under ~root:mirror 0)
  in
  S.with_ (fun t -> one_gpu_shared ~second:through_mirror t.g);
  let g = match through_mirror () with Ok g -> g | Error why -> fail why in
  Fun.protect ~finally:(fun () -> A.stop g ~fault:None) @@ fun () ->
  one_gpu_shared
    ~second:(fun () -> S.open_gpu ~word:(Rig_gpu_support.Host.word ()))
    g

(* Host memory of fewer than 64 KiB, which no device maps, copies into Device
   memory through the host's staging memory, which the process keeps: devices of
   one GPU, one after another and two at once, each map it. *)
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
      (Rig.open_
         { name = "AMD:test-second"; driver = (module A); make = S.open_gpu })
  in
  Fun.protect ~finally:(fun () -> Rig.close d) @@ fun () ->
  staged t 3;
  staged { t with d } 4;
  staged t 5

(* A host page's mapping, shared by two devices of one GPU, outlives the first
   device's stop and its region's free after it: the second device's work reads
   the host's bytes through it all along. *)
let shared_after_stop () =
  if S.driverless () then skip ~reason:"the path maps no host memory" ();
  S.with_driver @@ fun g' ->
  let g =
    match S.open_gpu ~word:(Rig_gpu_support.Host.word ()) with
    | Ok g -> g
    | Error why -> fail why
  in
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

(* rig's array of timeline words: its address and its bytes. *)
external words : unit -> int * int = "caml_rig_edge_words"

(* Two devices of one GPU, both opened through rig, write their words in rig's
   array, which the GPU maps once for both. The other device's copy waits in
   its queue for a Polled device's point, which the test reaches last: a copy
   on the first device of memory the other wrote, submitted before, reads the
   other's word in the array, or it would copy the memory unwritten. Once the
   other closes, the first still waits through the array. *)
let two_in_array () =
  if S.driverless () then skip ~reason:"one device of the GPU at a time" ();
  S.with_ @@ fun t ->
  waits_on t.g;
  let raw = ref None in
  let make ~word =
    Result.map
      (fun g ->
        raw := Some g;
        g)
      ((S.gpu ()).make ~word)
  in
  let d' =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_ { name = "AMD:second-in-array"; driver = (module A); make })
  in
  let t' = { d = d'; g = Option.get !raw } in
  let start, bytes = words () in
  let in_array g =
    match (A.locate (A.facts g).word).host with
    | Some a -> a >= start && a + 8 <= start + bytes
    | None -> false
  in
  equal (pair bool bool) ~msg:"both words lie in rig's array" (true, true)
    (in_array t.g, in_array t'.g);
  let gate, gp = Rig_support.Polled.open_ "amd:gate" in
  let gated d ws =
    let run, buffers = Rig_gpu_support.prepare d ws in
    let q =
      Rig.submit
        (Rig.Submission.Run.make (Rig.Submission.make gate [||]))
        ~buffers:[||]
    in
    (q, Rig.submit ~waits:[| q |] run ~buffers)
  in
  let n = 1 lsl 20 in
  let p = pattern n 29 in
  let src = buffer ~memory:Pinned t' n and mid = buffer ~memory:Pinned t' n in
  put src p;
  ignore (gated d' [ copy ~dst:mid src ]);
  let dst = buffer t n in
  let borrowed = Option.get (Rig.Buffer.borrow t.d mid) in
  let after = S.submit_work t [ copy ~dst borrowed ] in
  ignore (Rig_support.Polled.run gp);
  Rig.Point.wait after;
  equal string ~msg:"the copy after the other device's" p (get dst);
  Rig.close d';
  let dst' = buffer t n in
  let _, last = gated t.d [ copy ~dst:dst' dst ] in
  ignore (Rig_support.Polled.run gp);
  Rig.Point.wait last;
  equal string ~msg:"a wait through the array once the other closed" p
    (get dst');
  Rig.close gate

let two =
  group ~timeout:60. "one GPU"
    [
      test "two devices of one GPU opened through rig share rig's array"
        two_in_array;
      test
        "two devices of one GPU share its memory, mapping each page once, \
         where the path allows a second device"
        one_gpu;
      test
        "a view of Mapped memory lasts until its owner's stop, after which \
         map_peer of it is None"
        stopped_mapped;
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
  | Error _ | Ok [] -> None
  | Ok (bus :: _) ->
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
  exit
    (Windtrap.run "rig_amd"
       [
         gpus;
         misuse;
         images;
         entries;
         room;
         growth;
         work;
         code;
         launching;
         waits;
         two;
         traces;
         rings;
         memory;
         queue_order;
         timeline;
       ])
