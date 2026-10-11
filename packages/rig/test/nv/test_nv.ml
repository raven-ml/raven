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

(* The code of the function an entry answered, if it answered one. *)
let code_of : Rig_edge.entry_answer -> int option = function
  | Entry e -> Some e.code
  | Absent | No_memory _ -> None

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
    require_ok
      (Rig.open_ { name; driver = (module N); make = (fun () -> Ok g) })

(* GPUs *)

let gpus =
  group ~timeout:10. "GPUs"
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
  equal (option int) ~msg:"the copy engine" (Some 1) f.copy_engine;
  equal bool ~msg:"completion is the store" true (f.completion = Store);
  equal (list bool) ~msg:"waits on stores, hosts, objects" [ true; true; false ]
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
  let c = N.capability g in
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

(* rig finds the record the device's facts declare, and none on the host. *)
let declared () =
  S.with_ @@ fun t ->
  (match Rig.capability t.d A.Gpu.key with
  | Some c -> equal bool ~msg:"the record" true (c == N.capability t.g)
  | None -> fail "no record under the key");
  is_none ~msg:"the host" (Rig.capability Rig.host A.Gpu.key)

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
      less int ~msg:(name ^ ": its last byte") ~than:(1 lsl 40)
        (address r + 4095);
      N.free g r)
    [ ("Device", Device); ("Pinned", Pinned); ("Mapped", Mapped) ];
  if (N.facts g).maps_host then begin
    let p = H.pages H.page in
    let r = require_some ~msg:"map_host" (N.map_host g p H.page) in
    less int ~msg:"map_host: its last byte" ~than:(1 lsl 40)
      (address r + H.page - 1);
    N.free g r;
    H.free_pages p H.page
  end

let facts =
  group ~timeout:60. "facts"
    [
      test "states a GPU's facts" facts;
      test "declares the GPU to compiled code" capability;
      test "rig finds its record" declared;
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
    is_none ~msg:"twice the budget" (N.alloc g Device (2 * (N.facts g).budget))
  done;
  N.free g (require_some ~msg:"64 MiB after" (N.alloc g Device (64 lsl 20)))

(* Host memory no page backs is refused: map_host answers None. *)
let refused_host () =
  S.with_driver @@ fun g ->
  let p = H.pages H.page in
  H.free_pages p H.page;
  equal bool ~msg:"an unmapped page" true (Option.is_none (N.map_host g p 8));
  equal bool ~msg:"address 0" true (Option.is_none (N.map_host g 0 8))

(* Views a GPU's reopened device makes of memory the GPU already maps: each
   region over it keeps it mapped until its own free, in whatever order they
   go. *)
let views_of_mapped () =
  let n = 4096 in
  let a = S.driver () in
  let src = alloc a Pinned n in
  S.pattern (host src) n 5;
  S.stop_driver a;
  S.with_ @@ fun t ->
  let view () =
    require_some ~msg:"a view of memory its GPU maps" (N.map_peer t.g a src)
  in
  let v = view () in
  let v' = view () in
  N.free a src;
  N.free t.g v;
  let k = S.kernels t in
  let l = S.launches t.g in
  let out = alloc t.g Pinned n in
  S.run t [| S.words (copy_after l k ~dst:(address out) ~src:(address v') n) |];
  equal int ~msg:"the first byte that differs" (-1) (S.mismatch (host out) n 5);
  S.free_launches l;
  N.free t.g out;
  N.free t.g v'

(* A view of another device's Mapped memory makes the viewer's submissions
   read a word of the BAR behind it before their doorbells, so that work
   through the view reads what the host wrote there; the viewer holds that
   word until its own stop, past the owner's stop and the memory's free, as a
   read of a word given back would fault. *)
let views_of_bar () =
  let n = 4096 in
  let a = S.driver () in
  let src = alloc a Mapped n in
  S.stop_driver a;
  S.with_ @@ fun t ->
  let v =
    require_some ~msg:"a view of Mapped memory" (N.map_peer t.g a src)
  in
  let k = S.kernels t in
  let l = S.launches t.g in
  let out = alloc t.g Pinned n in
  let copy () =
    S.run t [| S.words (copy_after l k ~dst:(address out) ~src:(address v) n) |]
  in
  for seed = 1 to 3 do
    S.pattern (host src) n seed;
    copy ();
    equal int ~msg:(strf "the host's write %d" seed) (-1)
      (S.mismatch (host out) n seed)
  done;
  N.free a src;
  copy ();
  equal int ~msg:"after the memory's free" (-1) (S.mismatch (host out) n 3);
  S.free_launches l;
  N.free t.g out;
  N.free t.g v

let memory =
  group ~timeout:120. "memory"
    [
      test "a view of memory its GPU maps lasts until its own free"
        views_of_mapped;
      test "a view of Mapped memory reads the host's writes, and its BAR \
            word lasts until the viewer's stop"
        views_of_bar;
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
  equal (list int) (List.init n (fun i -> 2 * i)) (words32 (host out) n);
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
  let empty = Rig.Submission.make pd [||] in
  let run = Rig.Submission.Run.make () in
  let point = Rig.submit empty ~run ~buffers:[||] in
  let b = alloc t.g Pinned 16 in
  H.set64 (host b) 0;
  let l = S.launches t.g in
  let s = Rig.Submission.make t.d [| S.words (release l (address b) 9) |] in
  S.watchdog "a wait on a high host word" (fun () ->
      let v = Rig.submit s ~run ~waits:[| point |] ~buffers:[||] in
      still ~msg:"the word" int
        (Rig.Point.value v - 1)
        (fun () -> N.signaled t.g)
        ~ms:20;
      ignore (P.run p);
      Rig.Point.wait v);
  equal int ~msg:"released" 9 (H.get64 (host b));
  S.free_launches l;
  N.free t.g b

(* Parts NV's channels have no room for, refused when submitted: a ring entry
   cut in half, more ring words than a segment holds, and more parts than the
   rings hold. No value is assigned. *)
let refused () =
  S.with_ @@ fun t ->
  let refuses msg ps =
    raises_match ~msg (Exn.invalid_arg ~substring:"never fit") (fun () ->
        S.submit t ps)
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

let grid_image =
  Rig_gpu_support.loader (fun () -> S.fixture "grid_params_sm89.cubin")

(* A ref at [at] to [slot], which the work writes unless [use] says. *)
let refer ?(use = Sub.write) at slot = use slot ~at

(* [kernel] of [image], with [params] bytes of parameters. *)
let launch_part image kernel ~params refs = Sub.launch ~refs image kernel ~params
let copy_part ~dst src = Sub.copy ~src ~dst

(* The [n] 32-bit words of [b] from byte [at], copied to the host. *)
let words b ~at n =
  let h = B.create Rig.host (4 * n) in
  B.copy ~src:(B.view b ~first:at ~length:(4 * n)) ~dst:h;
  B.read h Bigarray.int32 (fun a ->
      Array.init n (fun i -> Int32.to_int a.{i} land 0xffff_ffff))

let submitted s run ~buffers = Rig.submit s ~run ~buffers

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
    Sub.make d [| launch_part image kernel ~params [| refer 0 0 |] |]
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
    (require_ok ~pp:Format.pp_print_string (A.Launch.make (N.capability g) k))

(* Launches past NV's limits are refused before any value, and the device stays
   live. The image's load takes values of its own. *)
let refused_launches () =
  S.with_ @@ fun { d; g } ->
  let out = B.create d 4096 in
  ignore (launch_image d);
  let loaded = Rig.submitted d in
  let refuses msg ?kernel ?params ?(groups = (1, 1, 1)) ?(threads = (1, 1, 1))
      ?(shared = 0) () =
    let s, run = launching ?kernel ?params d ~groups ~threads ~shared in
    raises_match ~msg (Exn.invalid_arg ~substring:"never fit") (fun () ->
        submitted s run ~buffers:[| out |])
  in
  (* More threads than the function has, which the setter refuses. *)
  List.iter
    (fun threads ->
      raises_match
        (Exn.invalid_arg ~substring:"threads per group")
        (fun () -> launching d ~groups:(1, 1, 1) ~threads ~shared:0))
    [ (1025, 1, 1); (1024, 2, 1) ];
  refuses "65536 groups along y" ~groups:(1, 65536, 1) ();
  refuses "65536 groups along z" ~groups:(1, 1, 65536) ();
  refuses "2^31 groups along x" ~groups:(1 lsl 31, 1, 1) ();
  refuses "65 threads along z" ~threads:(1, 1, 65) ();
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
  Rig.Point.wait (submitted s run ~buffers:[| out |])

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
  Rig.Point.wait (submitted s run ~buffers:[| out |]);
  let value i g = (5 + (3 * i) + (7 * g)) land 0xffff_ffff in
  equal (array int) ~msg:"the words"
    (Array.init 512 (fun i ->
         let g = i / 256 in
         value ((g * 256) + ((i + 1) mod 256)) g))
    (words out ~at:0 512)

(* Functions with static shared memory run with no dynamic shared memory, alone
   and chained, over one group and over several. *)
let static_shared () =
  S.with_ @@ fun { d; _ } ->
  let image = launch_image d in
  let value k i g = (k + (3 * i) + (7 * g)) land 0xffff_ffff in
  let expect ~groups ~threads k =
    Array.init (groups * threads) (fun i ->
        let g = i / threads in
        value k ((g * threads) + ((i + 1) mod threads)) g)
  in
  List.iter
    (fun (groups, threads) ->
      let n = groups * threads in
      let a = B.create d (4 * n) and b = B.create d (4 * n) in
      let turn slot = launch_part image "turn" ~params:12 [| refer 0 slot |] in
      let s =
        Sub.make d
          [| turn 0; turn 1 |]
      in
      let run = Run.make () in
      Array.iteri
        (fun i k ->
          let blk = Sub.block s i in
          Run.groups run blk groups 1 1;
          Run.threads run blk threads 1 1;
          Run.shared run blk 0;
          Run.int64 run blk 0 0;
          Run.int32 run blk 8 k)
        [| 5; 9 |];
      Rig.Point.wait (submitted s run ~buffers:[| a; b |]);
      let msg = strf "%d groups of %d" groups threads in
      equal (array int) ~msg (expect ~groups ~threads 5) (words a ~at:0 n);
      equal (array int) ~msg (expect ~groups ~threads 9) (words b ~at:0 n))
    [ (1, 1); (1, 32); (4, 256) ]

(* A function whose switch is a jump table in its constant bank 2 reads it: pick
   (fixtures/launch.cu) over one group of 32 threads, for each type of element.
   Bank 0 is not the only bank a launch binds. *)
let jump_table () =
  S.with_ @@ fun { d; _ } ->
  let image = launch_image d in
  let n = 32 in
  let data = Bytes.create (8 * n) in
  for i = 0 to n - 1 do
    Bytes.set_int64_le data (8 * i)
      (Int64.of_int (i * 0x0123_4567_89ab lxor (-1 lsl 40)))
  done;
  let floats = Bytes.create (8 * n) and doubles = Bytes.create (8 * n) in
  for i = 0 to n - 1 do
    let x = (float_of_int i *. 1.5) -. 7. in
    Bytes.set_int32_le floats (4 * i) (Int32.bits_of_float x);
    Bytes.set_int64_le doubles (8 * i) (Int64.bits_of_float x)
  done;
  let u32 x = x land 0xffff_ffff in
  let integer b i = function
    | 0 -> Bytes.get_uint8 b i
    | 1 -> Bytes.get_uint16_le b (2 * i)
    | 2 -> Int32.to_int (Bytes.get_int32_le b (4 * i))
    | 3 -> Int64.to_int (Bytes.get_int64_le b (8 * i))
    | 4 -> Bytes.get_int8 b i
    | 5 -> Bytes.get_int16_le b (2 * i)
    | 6 -> Int32.to_int (Bytes.get_int32_le b (4 * i))
    | _ -> Int64.to_int (Bytes.get_int64_le b (8 * i))
  in
  let truncated x = Int64.to_int (Int64.of_float x) in
  List.iter
    (fun t ->
      let src, value =
        match t with
        | 8 ->
            ( floats,
              fun i ->
                truncated
                  (Int32.float_of_bits (Bytes.get_int32_le floats (4 * i))) )
        | 9 ->
            ( doubles,
              fun i ->
                truncated
                  (Int64.float_of_bits (Bytes.get_int64_le doubles (8 * i))) )
        | t -> (data, fun i -> integer data i t)
      in
      let input = B.create d (Bytes.length src) in
      B.copy ~src:(B.of_string (Bytes.to_string src)) ~dst:input;
      let out = B.create d (4 * n) in
      let s =
        Sub.make d
          [|
            launch_part image "pick" ~params:20
              [| refer 0 1; refer ~use:Sub.read 8 0 |];
          |]
      in
      let run = Run.make () in
      let b = Sub.block s 0 in
      Run.groups run b 1 1 1;
      Run.threads run b n 1 1;
      Run.int64 run b 0 0;
      Run.int64 run b 8 0;
      Run.int32 run b 16 t;
      Rig.Point.wait (submitted s run ~buffers:[| input; out |]);
      equal (array int) ~msg:(strf "type %d" t)
        (Array.init n (fun i -> u32 (value i)))
        (words out ~at:0 n))
    (List.init 10 Fun.id)

(* A cached launch submitted with a warm run, its block stored anew each time
   through every setter, allocates nothing. *)
let no_allocation () =
  S.with_ @@ fun { d; _ } ->
  let s, run = launching d ~groups:(1, 1, 1) ~threads:(32, 1, 1) ~shared:0 in
  let buffers = [| B.create d 4096 |] in
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
    Sys.opaque_identity (submitted s run ~buffers)
  in
  Rig.Point.wait (once 0);
  let before = Gc.minor_words () in
  let last = ref (once 1) in
  for i = 2 to 100 do
    last := once i
  done;
  let words = int_of_float (Gc.minor_words () -. before) / 100 in
  Rig.Point.wait !last;
  equal int ~msg:"words per launch" 0 words

(* Public launches and a private copy kernel share the compute channel. The
   edges their memory derives still put the copy between the launches. *)
let launch_copy_launch () =
  S.with_ @@ fun { d; _ } ->
  let image = launch_image d in
  let n = 1024 in
  let out = B.create d (4 * n) and mid = B.create d (4 * n) in
  let dst = B.create d (4 * n) in
  let s =
    Sub.make d
      [|
        launch_part image "ids" ~params:24 [| refer 0 0 |];
        copy_part ~dst:mid out;
        launch_part image "twice" ~params:20
          [| refer 0 2; refer ~use:Sub.read 8 1 |];
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
  Rig.Point.wait (submitted s run ~buffers:[| out; mid; dst |]);
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
  let last = ref None in
  for i = 0 to n - 1 do
    ids_params run b ~offset:(4 * i) ~a:i ~b:0;
    last := Some (submitted s run ~buffers:[| out |])
  done;
  Option.iter Rig.Point.wait !last;
  equal (array int) (Array.init n Fun.id) (words out ~at:0 n)

(* A function whose threads keep 2 KiB of local memory runs as a launch: its
   entry made the channel's local memory serve it. *)
let local_launch () =
  S.with_ @@ fun { d; _ } ->
  let n = 1024 in
  let out = B.create d (4 * n) in
  let s =
    Sub.make d
      [| launch_part (kernel_image d) "stack" ~params:12 [| refer 0 0 |] |]
  in
  let run = Run.make () in
  let b = Sub.block s 0 in
  Run.groups run b (n / 256) 1 1;
  Run.threads run b 256 1 1;
  Run.int64 run b 0 0;
  Run.int32 run b 8 n;
  Rig.Point.wait (submitted s run ~buffers:[| out |]);
  equal (array int)
    (Array.init n (fun i -> (512 * i) + 130816))
    (words out ~at:0 n)

(* Each kernel reads through a generic pointer into its own parameters. Two
   chained launches and fresh values on the next submission distinguish their
   parameter blocks from one another and from an earlier run. *)
let grid_parameters (groups, threads) =
  S.with_ @@ fun { d; _ } ->
  let image = grid_image d in
  let n = groups * threads in
  let buffers = Array.init 2 (fun _ -> B.create d (4 * n)) in
  let s =
    Sub.make d
      (Array.init 2 (fun i ->
           launch_part image "grid_params" ~params:24 [| refer 0 i |]))
  in
  let run = Run.make () in
  let values = [| 0; 0x7fff_ffff; 0x8000_0000; 0xffff_ffff |] in
  let value round part i = values.(i mod 4) lxor ((round * 7) + part) in
  for round = 0 to 2 do
    Array.iteri
      (fun part _ ->
        let b = Sub.block s part in
        Run.groups run b groups 1 1;
        Run.threads run b threads 1 1;
        Run.int64 run b 0 0;
        for i = 0 to 3 do
          Run.int32 run b (8 + (4 * i)) (value round part i)
        done)
      buffers;
    Rig.Point.wait (submitted s run ~buffers);
    Array.iteri
      (fun part out ->
        equal (array int)
          ~msg:(strf "round %d, part %d" round part)
          (Array.init n (value round part))
          (words out ~at:0 n))
      buffers
  done

let launches =
  group ~timeout:120. "launches"
    [
      test "launches past NV's limits are refused" refused_launches;
      test "a launch takes the most dynamic shared memory" most_shared_memory;
      test "functions with static shared memory run alone and chained"
        static_shared;
      test "a function reads the jump table in its constant bank 2" jump_table;
      test "a cached launch allocates nothing" no_allocation;
      test "a launch after a copy after a launch reads the copy"
        launch_copy_launch;
      test "4,000 launches pass the launch ring's end" launches_wrap;
      test "a function with local memory runs as a launch" local_launch;
      cases
        ~name:(fun (g, t) -> strf "%d groups of %d" g t)
        "a kernel addresses its own parameters"
        [ (1, 1); (3, 32); (5, 256) ]
        grid_parameters;
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
  let last = ref None in
  for i = 0 to n - 1 do
    ignore (S.submit t [||]);
    last := Some (S.submit t [| S.copy ~dst:(byte dst i) (byte src i) |])
  done;
  Option.iter Rig.Point.wait !last;
  equal int ~msg:"the word" (2 * n) (N.signaled t.g);
  equal int ~msg:"the first byte that differs" (-1) (S.mismatch da n 5)

(* 40,000 copies of 16 bytes on COPY:0, each waited for, pass the end of the
   selected channel's segment and descriptor rings. *)
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

(* A copy's join word, which the copy engine writes as two 32-bit words, takes
   tags of the submission's value times 65,536: its high word first moves at
   value 65,536. Copies switch channels within their logical queue; alternating
   last channels must release every copy and the dependent words in order. *)
let join_carry () =
  S.with_ @@ fun t ->
  let l = S.launches t.g in
  let scratch = alloc t.g Pinned 8 in
  let segment = release l (address scratch) 1 in
  let bytes = 1 lsl 20 in
  let h, ha = S.shared t bytes in
  let a = B.create t.d bytes and b = B.create t.d bytes in
  let parts length =
    let h = B.view h ~first:0 ~length in
    let a = B.view a ~first:0 ~length in
    let b = B.view b ~first:0 ~length in
    [|
      [|
        S.copy ~dst:a h;
        S.copy ~dst:b a;
        S.words segment;
        S.copy ~dst:h b;
      |];
      [|
        S.copy ~dst:a h;
        S.copy ~dst:b a;
        S.copy ~dst:h b;
        S.words segment;
      |];
    |]
  in
  let small = parts 16 and large = parts bytes in
  (* The words follow the copies by memory they name: their release stands for
     a use of [h] and [b], which the copies write. *)
  let fixed = [ (h, Rig.Read); (b, Rig.Read) ] in
  S.pattern ha bytes 7;
  S.watchdog "work across a join's carry" (fun () ->
      for i = 1 to 66_000 do
        (* Large copies exercise the private kernel around the carry without
           transferring a whole image at every preceding value. *)
        let parts = if i >= 65_520 then large else small in
        Rig.Point.wait
          (Rig.submit (Sub.make ~fixed t.d parts.(i land 1)) ~run:(Run.make ()) ~buffers:[||])
      done);
  at_least int ~msg:"the values" ~than:65_536 (N.signaled t.g);
  equal int ~msg:"the compute part's release" 1 (H.get64 (host scratch));
  equal int ~msg:"the copied bytes" (-1) (S.mismatch ha bytes 7);
  S.free_launches l;
  N.free t.g scratch

(* The tear probe's unordered rounds. *)
let tear_rounds = 20_000

(* The copy engine's release of a 64-bit value and the compute engine's,
   unordered on one word across its 32-bit carry: whichever lands last, the word
   holds one of the two values, never a half of each. rig places no words on the
   copy queue, so the releases go to the driver's C entries as words, one part
   on each queue. Both channels announce they are ready before the host opens
   their gates. Two ordered rounds first establish each complete value;
   unordered rounds may choose either winner. *)
let carry_tear () =
  S.with_driver @@ fun g ->
  let l = S.launches g in
  let w = alloc g Pinned 8 in
  let gates = Array.init 2 (fun _ -> alloc g Pinned 8) in
  let ready = Array.init 2 (fun _ -> alloc g Pinned 8) in
  let below = (1 lsl 32) - 1 and above = 1 lsl 32 in
  let gated i command =
    S.segment l
      (A.Method.release System (address ready.(i)) 1
      @ A.Method.acquire (address gates.(i)) 1
      @ command)
  in
  let parts =
    [|
      (1, gated 0 (A.Method.copy_release System (address w) below));
      (0, gated 1 (A.Method.release System (address w) above));
    |]
  in
  let v = ref 0 in
  let wait_word r x =
    while H.get64 (host r) <> x do
      Domain.cpu_relax ()
    done
  in
  let words = (w :: Array.to_list gates) @ Array.to_list ready in
  let start () =
    List.iter (fun r -> H.set64 (host r) 0) words;
    incr v;
    S.Edge.submit g ~v:!v parts;
    Array.iter (fun r -> wait_word r 1) ready
  in
  let reached () =
    while N.signaled g < !v do
      Domain.cpu_relax ()
    done
  in
  let open_gate i = H.set64 (host gates.(i)) 1 in
  let torn = ref [] in
  S.watchdog "unordered releases across a carry" (fun () ->
      Array.iteri
        (fun first value ->
          start ();
          open_gate first;
          wait_word w value;
          open_gate (1 - first);
          reached ();
          equal int
            ~msg:(strf "engine %d releases last" (1 - first))
            (if first = 0 then above else below)
            (H.get64 (host w)))
        [| below; above |];
      for i = 1 to tear_rounds do
        start ();
        open_gate (i land 1);
        open_gate (1 - (i land 1));
        reached ();
        let x = H.get64 (host w) in
        if x <> below && x <> above then torn := x :: !torn
      done);
  equal (list int) ~msg:"torn values" [] !torn;
  S.free_launches l;
  List.iter (N.free g) words

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
  let { A.Gpu.local; _ } = N.capability t.g in
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
      let { A.Gpu.local; _ } = N.capability t.g in
      equal (result unit string) ~msg:"grown" (Ok ()) (local 4096);
      let v =
        S.submit t
          [| S.words (S.launch l k "stack" ~blocks:4 [ address next; n ]) |]
      in
      still ~msg:"the word while held" int (Rig.Point.value v - 2)
        (fun () -> N.signaled t.g)
        ~ms:20;
      H.set64 (host flag) 1;
      Rig.Point.wait v);
  let want = List.init n stacked in
  equal (list int) ~msg:"the held kernel" want (words32 (host first) n);
  equal (list int) ~msg:"the next kernel" want (words32 (host next) n);
  S.free_launches l;
  List.iter (N.free t.g) [ flag; first; next ]

(* Local memory handed over: each growth replaces it while kernels placed
   before it still keep 2 KiB a thread in the old one, each held until the test
   raises a flag of its own. The test then runs them one at a time and grows
   again after each, so the word moves on by single values and every growth
   frees what the word passed. Every kernel computes right and the device
   lives. An RTX 5000 Ada still reads a memory freed under a held kernel, so
   on it the test does not see when each memory goes back. Each memory takes
   some 600 MiB of the GPU. *)
let handover () =
  S.with_ @@ fun t ->
  let k = S.kernels t and l = S.launches t.g in
  let kernels = 4 and n = 1024 in
  let flags = alloc t.g Pinned (8 * kernels) in
  let out = alloc t.g Pinned (4 * n * kernels) in
  H.write (host flags) (String.make (8 * kernels) '\000');
  let held = Queue.create () and next = ref 0 and bytes = ref 3072 in
  let hand () =
    let i = !next in
    incr next;
    let flag = address flags + (8 * i) and at = address out + (4 * n * i) in
    let launch =
      S.launch l k "stack_held" ~blocks:4 [ flag; 10 * second; at; n ]
    in
    Queue.push (i, S.submit t [| S.words launch |]) held
  in
  let grow () =
    bytes := !bytes + 32;
    equal (result unit string)
      ~msg:(strf "local %d" !bytes)
      (Ok ())
      ((N.capability t.g).local !bytes)
  in
  let run () =
    let i, p = Queue.pop held in
    H.set64 (host flags + (8 * i)) 1;
    Rig.Point.wait p
  in
  let raise_all () = H.write (host flags) (String.make (8 * kernels) '\001') in
  Fun.protect ~finally:raise_all (fun () ->
      for _ = 2 to kernels do
        hand ();
        grow ()
      done;
      hand ();
      while not (Queue.is_empty held) do
        run ();
        grow ()
      done);
  let want = List.init n stacked in
  for i = 0 to kernels - 1 do
    equal (list int) ~msg:(strf "kernel %d" i) want
      (words32 (host out + (4 * n * i)) n)
  done;
  equal (option string) ~msg:"the device's loss" None (Rig.lost t.d);
  S.free_launches l;
  List.iter (N.free t.g) [ flags; out ]

let local =
  group ~timeout:60. "local memory"
    [
      test "local memory serves kernels that keep 2 KiB a thread" local;
      test "a growth leaves a running kernel's local memory intact" growth;
      test "a replaced local memory stays until the kernels on it ran"
        handover;
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
  is_some ~msg:"double_index" (code_of (N.entry i "double_index"));
  equal (option int) ~msg:"a missing kernel" None
    (code_of (N.entry i "missing"));
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
  let { A.Gpu.local; _ } = N.capability t.g in
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
  let v = Rig.Point.value (S.submit t [| S.words ws |]) in
  still ~msg:"the word" int (v - 1) (fun () -> N.signaled t.g) ~ms:20;
  S.watchdog "stop" (fun () -> S.stop_driver t.g);
  equal int ~msg:"the word" v (N.signaled t.g);
  H.set64 (host w) 1;
  still ~msg:"the waiting work" int 0 (fun () -> H.get64 (host marked)) ~ms:200;
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
      let v1 = Rig.Point.value (S.submit t [| S.words spin |]) in
      let v2 =
        Rig.Point.value (S.submit t [| S.words (release l (address marked) 7) |])
      in
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

let stateful =
  group ~timeout:300. "stateful"
    [
      stateful ~count:100 ~steps:20 "map_host shares a host range both ways"
        registry_commands;
    ]

let () =
  exit
    (run "rig_nv"
       [
         gpus;
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
