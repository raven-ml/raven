(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* rig.mli's Host code: images linked from the fixtures' objects, whose sources
   say what they do, and launches of their functions on Rig.host and on a
   software device. *)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module S = Rig_host_support
module R = Rig_support.Reader
module P = Rig_support.Polled

let strf = Printf.sprintf

(* The code other suites also run lives in test/fixtures. *)
let shared_dir = Filename.concat Filename.parent_dir_name "fixtures"

let obj f =
  match f with
  | "affine" | "calling" | "waiting" -> S.fixture ~dir:shared_dir f
  | _ -> S.fixture ~dir:"fixtures" f

let load_on d o = require_ok ~pp:Format.pp_print_string (Rig.Image.load d o)
let load f = load_on Rig.host (obj f)

let entry_of image f =
  Rig.Image.entry (require_some (Rig.Image.resolve image f))

(* A launch's grid: its groups and the threads of each, along x, y and z. *)
type grid = { groups : int * int * int; threads : int * int * int }

let one = { groups = (1, 1, 1); threads = (1, 1, 1) }

(* The values of a launch over [grid], with [shared] bytes of shared memory and
   [params] bytes of parameters, 0 but what [set] writes. *)
let values ?(shared = 0) ?(set = ignore) ?(grid = one) params =
  let b = Bytes.make params '\000' in
  set b;
  { Sub.groups = grid.groups; threads = grid.threads; shared; params = b }

(* The point of one launch of [image]'s [f] on [device] (defaults to the host)
   over [buffers], with [params] bytes of parameters, 0 but what [set]
   writes. *)
let launch ?(device = Rig.host) ?(refs = [||]) ?(params = 0) ?shared ?set ?grid
    image f buffers =
  let fn = require_some (Rig.Image.resolve image f) in
  let s = Sub.make device [| Sub.launch ~refs fn ~params |] in
  Rig.submit s ~buffers ~args:[| values ?shared ?set ?grid params |]

let words = Array.map B.of_bigarray

(* [p]'s work is done: its device [d] signaled at least its value. *)
let done_ ?msg d p =
  at_least ?msg int ~than:(Rig.Point.value p) (Rig.signaled d)

let mapped ?msg a = satisfies ?msg ~claim:"mapped executable" int S.executable a

let unmapped ?msg a =
  satisfies ?msg ~claim:"on no executable page" int
    (fun a -> not (S.executable a))
    a

(* Collects, then drains the releases of the devices [on] (defaults to the host)
   as an allocation does. *)
let collect ?(on = [ Rig.host ]) () =
  for _ = 1 to 3 do
    Gc.full_major ();
    List.iter (fun d -> ignore (B.create d 8)) on
  done

let software = lazy (Result.get_ok (Rig.software "host:software"))
let software_affine = lazy (load_on (Lazy.force software) (obj "affine"))

(* A buffer of [d] holding the words [xs]. *)
let words_on d xs =
  let bytes = Bytes.create (8 * Array.length xs) in
  Array.iteri (fun i x -> Bytes.set_int64_le bytes (8 * i) (Int64.of_int x)) xs;
  let b = B.create d (Bytes.length bytes) in
  B.blit_from_string (Bytes.to_string bytes) 0 b 0 (Bytes.length bytes);
  b

let read_words b =
  let bytes = Bytes.create (B.length b) in
  B.blit_to_bytes b 0 bytes 0 (B.length b);
  Array.init
    (B.length b / 8)
    (fun i -> Int64.to_int (Bytes.get_int64_le bytes (8 * i)))

(* The names of the host's work in [events] on [d]. *)
let work_names d events =
  List.filter_map
    (function
      | Rig.Profile.Work s when Rig.equal s.device d -> Some s.name | _ -> None)
    events

(* Host code: sharing *)

let fresh f = S.fresh (obj f)

(* Their functions have the same entry, and one image runs blocks.c once the
   others are collected and a device of theirs closed. *)
let test_shared () =
  let d = Lazy.force software in
  let e = require_ok ~pp:Format.pp_print_string (Rig.software "host:shared") in
  let o = fresh "blocks" in
  let kept = load_on d o in
  let entry = entry_of kept "blocks" in
  equal ~msg:"the host's entry" int entry
    (entry_of (load_on Rig.host o) "blocks");
  equal ~msg:"another software device's entry" int entry
    (entry_of (load_on e o) "blocks");
  collect ~on:[ Rig.host; d; e ] ();
  Rig.close e;
  let counts = words_on d [| 0; 0; 0 |] and log = words_on d [| 0; 4; 0; 0 |] in
  ignore
    (launch ~device:d kept "blocks"
       ~refs:Sub.[| write 0 ~at:0; write 1 ~at:8 |]
       ~params:16
       ~grid:{ groups = (1, 1, 1); threads = (3, 1, 1) }
       [| counts; log |]);
  equal ~msg:"each thread's runs" (array int) [| 1; 1; 1 |] (read_words counts);
  equal ~msg:"the log" (array int) [| 2; 4; 0; 3 |] (read_words log)

(* The link stays mapped while the image on [keeper] is reachable, after the
   others are collected, and is unmapped once it goes too. *)
let test_last_unmaps (_, keeper) =
  let on = [ Rig.host; Lazy.force software ] in
  let o = fresh "empty" in
  let address =
    (fun () ->
      let kept = load_on (keeper ()) o in
      let address = entry_of kept "empty" in
      List.iter
        (fun d ->
          equal ~msg:(Rig.name d) int address (entry_of (load_on d o) "empty"))
        on;
      collect ~on ();
      mapped ~msg:"while one image is reachable" address;
      equal ~msg:"the kept image's entry" int address (entry_of kept "empty");
      address)
      ()
  in
  collect ~on ();
  unmapped ~msg:"once the last image goes" address

let keepers =
  [
    ("the host", fun () -> Rig.host);
    ("a software device", fun () -> Lazy.force software);
  ]

(* Loads from two domains of three objects, on the host and on a software
   device: each object's images share one link whichever domain links it. *)
type loads = {
  objects : string array;
  device : Rig.t;
  images : (int * Rig.Image.t) list Atomic.t;
}

let shared_fixtures = [| "empty"; "blocks"; "scale" |]
let loads = abstract "l"

let rec push a x =
  let l = Atomic.get a in
  if not (Atomic.compare_and_set a l (x :: l)) then push a x

let load_commands =
  let k = Gen.int_range 0 (Array.length shared_fixtures - 1) in
  let made () =
    {
      objects = Array.map fresh shared_fixtures;
      device = Lazy.force software;
      images = Atomic.make [];
    }
  in
  let load k host l =
    let d = if host then Rig.host else l.device in
    push l.images (k, load_on d l.objects.(k))
  in
  let links k l =
    List.sort_uniq Int.compare
      (List.filter_map
         (fun (k', i) ->
           if k' = k then Some (entry_of i shared_fixtures.(k)) else None)
         (Atomic.get l.images))
  in
  let one_link _ () = function
    | Ok entries ->
        less ~msg:"the object's distinct entries" int (List.length entries)
          ~than:2
    | Error e -> raise e
  in
  [
    command "objects" (Gen.unit @-> makes loads) (fun () -> ()) made;
    command "load"
      (k @-> Gen.bool @-> loads ^-> returns unit)
      (fun _ _ () -> ())
      load;
    command "one link" (k @-> loads ^-> judges (list int)) one_link links;
  ]

(* A load whose exchange another link of its object beats takes that link. The
   hook links the object on the host between a software device's load reading
   the links and its exchange; it records no failure itself, since an exception
   there loses the device. The loser's link is unmapped once collected, and its
   collection leaves the winner's link to later loads. *)
let test_lost_exchange () =
  let d =
    require_ok ~pp:Format.pp_print_string (Rig.software "host:exchange")
  in
  let o = fresh "empty" in
  let lost = ref 0 and inner = ref None in
  Rig.on_host_exchange (fun a ->
      lost := a;
      inner := Some (Rig.Image.load Rig.host o));
  let outer = load_on d o in
  let winner =
    require_ok ~pp:Format.pp_print_string
      (require_some ~msg:"the hook's load" !inner)
  in
  let entry = entry_of winner "empty" in
  equal ~msg:"the outer image's entry" int entry (entry_of outer "empty");
  collect ~on:[ Rig.host; d ] ();
  unmapped ~msg:"the lost link" !lost;
  equal ~msg:"a third load's entry" int entry
    (entry_of (load_on Rig.host o) "empty");
  equal ~msg:"the winner's entry" int entry (entry_of winner "empty");
  Rig.close d

(* Host code: queues *)

let test_host_parts () =
  let b = B.create Rig.host 8 and b' = B.create Rig.host 8 in
  List.iter
    (fun (msg, part) ->
      raises_match ~msg Exn.invalid_arg (fun () -> Sub.make Rig.host [| part |]))
    [
      ("a copy", Sub.(copy ~src:(Buffer b) ~dst:(Buffer b')));
      ("a fill", Sub.fill 0n ~params:0);
      ("words", Sub.words b);
    ]

(* A submission's fill, copy and launch run in order: the copy reads what the
   fill wrote, and the launch what the copy wrote. *)
let test_software_parts () =
  let d = Lazy.force software in
  let affine =
    require_some (Rig.Image.resolve (Lazy.force software_affine) "affine")
  in
  let s =
    Sub.make d
      Sub.
        [|
          fill ~refs:[| write 0 ~at:0 |] Rig_support.bump ~params:8;
          copy ~src:(Slot 0) ~dst:(Slot 1);
          launch ~refs:[| write 2 ~at:0; read 1 ~at:8 |] affine ~params:40;
        |]
  in
  let set b =
    Bytes.set_int64_le b 16 3L;
    Bytes.set_int64_le b 24 10L;
    Bytes.set_int64_le b 32 1L
  in
  let grid = { one with threads = (3, 1, 1) } in
  let args = [| values 8; values ~set ~grid 40 |] in
  let src = words_on d [| 1; 2; 3 |] in
  let tmp = words_on d [| 0; 0; 0 |] and out = words_on d [| 0; 0; 0 |] in
  let p = Rig.submit s ~buffers:[| src; tmp; out |] ~args in
  done_ ~msg:"when submit returns" d p;
  equal (array int) [| 21; 21; 31 |] (read_words out)

(* The launches no host queue runs, as rig.mli's Launches states them. *)
let max_axis = (1 lsl 32) - 1

let grid_refusals =
  [
    ("shared memory", 1, one);
    ("an empty grid", 0, { one with groups = (1, 0, 1) });
    ("an empty group", 0, { one with threads = (1, 1, 0) });
    ("more than 2^62 threads", 0, { one with groups = (max_axis, max_axis, 1) });
  ]

(* A software device runs no words part and refuses the launches the host
   refuses. *)
let test_software_refused () =
  let d = Lazy.force software in
  let empty = load_on d (obj "empty") in
  raises_match ~msg:"words" Exn.invalid_arg (fun () ->
      Sub.make d [| Sub.words (B.create d 8) |]);
  List.iter
    (fun (msg, shared, grid) ->
      raises_match ~msg Exn.invalid_arg (fun () ->
          launch ~device:d ~shared ~grid empty "empty" [||]))
    grid_refusals

(* Host code: code memory *)

(* An object of its own, whose link no image of another test shares. *)
let test_unmapped () =
  let address =
    (fun () -> entry_of (load_on Rig.host (fresh "empty")) "empty") ()
  in
  collect ();
  unmapped address

let test_survives () =
  let image = load "affine" in
  collect ();
  let out = S.words 1 and inb = S.words 1 in
  inb.{0} <- 5L;
  ignore
    (launch image "affine"
       ~refs:Sub.[| write 0 ~at:0; read 1 ~at:8 |]
       ~params:40
       ~set:(fun b ->
         Bytes.set_int64_le b 16 1L;
         Bytes.set_int64_le b 24 3L;
         Bytes.set_int64_le b 32 1L)
       (words [| out; inb |]));
  equal ~msg:"3 * 5 + 1" int64 16L out.{0};
  mapped (entry_of image "affine")

(* waiting.c of [image] over [buffers]: it sets word 1 of slot 0, then spins
   until word 0 is set, at most [rounds] rounds. Every slot is a write ref, slot
   0 at byte 0 and slot [i] at [8 * (i + 1)]. *)
let waiting ?grid image buffers ~rounds =
  let refs =
    Array.mapi
      (fun i _ -> Sub.write i ~at:(if i = 0 then 0 else 8 * (i + 1)))
      buffers
  in
  launch image "waiting" ~refs
    ~params:(8 * (Array.length buffers + 1))
    ~set:(fun b -> Bytes.set_int64_le b 8 (Int64.of_int rounds))
    ?grid buffers

(* Rounds no launch reaches before its test sets word 0. *)
let forever = 20_000_000_000

(* waiting.c, which nothing holds but its running submit, while another domain
   collects. *)
let test_running () =
  let w = S.words 2 in
  let address = Atomic.make 0 and image = load "waiting" in
  let d =
    Domain.spawn (fun () ->
        let waiting = require_some (Rig.Image.resolve image "waiting") in
        Atomic.set address (Rig.Image.entry waiting);
        let s =
          Sub.make Rig.host
            [| Sub.launch ~refs:Sub.[| write 0 ~at:0 |] waiting ~params:16 |]
        in
        let set b = Bytes.set_int64_le b 8 (Int64.of_int forever) in
        ignore
          (Rig.submit s
             ~buffers:[| B.of_bigarray w |]
             ~args:[| values ~set 16 |]))
  in
  Fun.protect
    ~finally:(fun () ->
      w.{0} <- 1L;
      Domain.join d)
    (fun () ->
      Rig_support.await "the launch running" (fun () -> w.{1} = 1L);
      collect ();
      mapped ~msg:"while its launch runs" (Atomic.get address))

(* A submission of calling.c, which calls waiting.c's function through an entry
   ref, with the function's address in [address]. Only the submission keeps
   waiting.c's image. *)
let calls_waiting address =
  let callee = require_some (Rig.Image.resolve (load "waiting") "waiting") in
  address := Rig.Image.entry callee;
  let refs = Sub.[| entry callee ~at:0; write 0 ~at:8 |] in
  let calling = require_some (Rig.Image.resolve (load "calling") "calling") in
  Sub.make Rig.host [| Sub.launch ~refs calling ~params:24 |]

(* The callee lives from make to submit and while the launch that calls it runs
   and another domain collects. Once the submission is unreachable and its work
   done, the callee is unmapped. *)
let test_keeps_callee () =
  let w = S.words 2 in
  let address = ref 0 in
  let s = calls_waiting address in
  collect ();
  mapped ~msg:"while a submission names it" !address;
  let d =
    Domain.spawn (fun () ->
        let set b = Bytes.set_int64_le b 16 (Int64.of_int forever) in
        ignore
          (Rig.submit s
             ~buffers:[| B.of_bigarray w |]
             ~args:[| values ~set 24 |]))
  in
  Fun.protect
    ~finally:(fun () ->
      w.{0} <- 1L;
      Domain.join d)
    (fun () ->
      Rig_support.await "the callee running" (fun () -> w.{1} = 1L);
      collect ();
      mapped ~msg:"while a launch calls it" !address);
  collect ();
  unmapped ~msg:"once its work is done and unreachable" !address

let host_code_tests =
  group "host code"
    [
      test
        "images of one object on the host and two software devices share its \
         link"
        test_shared;
      cases "a link is unmapped once its last image goes, kept on" ~id:fst
        keepers test_last_unmaps;
      stateful ~domains:2 ~count:30
        "loads from two domains share one link per object" load_commands;
      test "a load that loses its exchange takes the winner's link"
        test_lost_exchange;
      test "the host's queue runs only launches" test_host_parts;
      test "a software device runs a fill, a copy and a launch in order"
        test_software_parts;
      test "a software device refuses words and the launches the host refuses"
        test_software_refused;
      test "an unreachable image's code is unmapped" test_unmapped;
      test "a reachable image's code survives collections and runs"
        test_survives;
      test "an image's code survives collections while a launch of it runs"
        test_running;
      test "a callee in another image lives while a submission names it"
        test_keeps_callee;
    ]

(* Launches *)

let affine = lazy (load "affine")

(* affine.c of [image] over [n] elements of [inb] into [out], in groups of [t]
   threads: out[i] = a * inb[i] + c. *)
let affine_on ?device image ~n ~t ~a ~c out inb =
  let set b =
    Bytes.set_int64_le b 16 (Int64.of_int n);
    Bytes.set_int64_le b 24 (Int64.of_int a);
    Bytes.set_int64_le b 32 (Int64.of_int c)
  in
  launch ?device image "affine"
    ~refs:Sub.[| write 0 ~at:0; read 1 ~at:8 |]
    ~params:40 ~set
    ~grid:{ groups = ((n + t - 1) / t, 1, 1); threads = (t, 1, 1) }
    [| out; inb |]

let affine_launch ~n ~t ~a ~c out inb =
  affine_on (Lazy.force affine) ~n ~t ~a ~c (B.of_bigarray out)
    (B.of_bigarray inb)

let affine_case =
  let open Gen in
  let* n = frequency [ (1, constant 1); (4, int_range 1 200) ] in
  let+ t = int_range 1 9
  and+ a = small_int
  and+ c = small_int
  and+ input = array ~size:(constant n) small_int in
  (n, t, a, c, input)

let test_affine (n, t, a, c, input) =
  cover "one element" (n = 1);
  cover "a last group in part past n" (n mod t <> 0);
  let inb = S.words n and out = S.words (n + 1) in
  Array.iteri (fun i x -> inb.{i} <- Int64.of_int x) input;
  out.{n} <- 7L;
  let p = affine_launch ~n ~t ~a ~c out inb in
  done_ ~msg:"when submit returns" Rig.host p;
  let expected = Array.map (fun x -> (a * x) + c) input in
  equal (array int) expected (Array.init n (fun i -> Int64.to_int out.{i}));
  equal ~msg:"the word past the last" int64 7L out.{n}

(* An int64 parameter holds every int64, those whose bits 63 and 62 differ among
   them: affine.c with [a = 0] stores its [c] into [out]. *)
let test_int64_parameter () =
  let values =
    Int64.[ min_int; max_int; add min_int 1L; 0x4000_0000_0000_0000L; -1L; 0L ]
  in
  List.iter
    (fun c ->
      let inb = S.words 1 and out = S.words 1 in
      let set b =
        Bytes.set_int64_le b 16 1L;
        Bytes.set_int64_le b 24 0L;
        Bytes.set_int64_le b 32 c
      in
      ignore
        (launch (Lazy.force affine) "affine"
           ~refs:Sub.[| write 0 ~at:0; read 1 ~at:8 |]
           ~params:40 ~set
           (words [| out; inb |]));
      equal ~msg:(Int64.to_string c) int64 c out.{0})
    values

(* wide.c with 8,192 bytes of parameters, past the 4,096 a launch had: the words
   at bytes 4,096 and 8,188 reach it. *)
let test_wide () =
  let out = S.words 1 in
  ignore
    (launch (load "wide") "wide"
       ~refs:Sub.[| write 0 ~at:0 |]
       ~params:8192
       ~set:(fun b ->
         Bytes.set_int32_le b 4096 41l;
         Bytes.set_int32_le b 8188 43l)
       (words [| out |]));
  equal (pair int int) (41, 43)
    Int64.
      ( to_int (logand out.{0} 0xffff_ffffL),
        to_int (shift_right_logical out.{0} 32) )

(* The host bounds a launch's parameters only by what one block holds, 2^31 - 32
   bytes (rig.mli, Submission.launch): a launch of 2^31 - 32 bytes is made and
   one of 2^31 - 31 refused. *)
let test_run_bound () =
  let wide = require_some (Rig.Image.resolve (load "wide") "wide") in
  let make n = Sub.make Rig.host [| Sub.launch wide ~params:n |] in
  ignore (make ((1 lsl 31) - 32));
  raises_match (Exn.invalid_arg ~sub:"Rig.Submission.launch: ") (fun () ->
      make ((1 lsl 31) - 31))

(* A profile taken around a launch records its work, named after its function:
   [take] reads the host's times before it returns. *)
let test_profiled () =
  let out = S.words 4 and inb = S.words 4 in
  let _, events =
    Rig.Profile.take (fun () -> affine_launch ~n:4 ~t:2 ~a:1 ~c:0 out inb)
  in
  equal (list string) [ "affine" ] (work_names Rig.host events)

(* The host runs work only after the work it follows: a launch that reads memory
   a Polled device's unrun launch writes waits for that work. *)
let test_follows () =
  let d, pd = P.open_ "host:follows" in
  let shared = B.create Rig.host B.paged_from in
  let functions = require_ok (Rig.Image.load d "functions") in
  let main = require_some (Rig.Image.resolve functions "main") in
  let refs = Sub.[| write 0 ~at:0 |] in
  let s = Sub.make d [| Sub.launch ~refs main ~params:8 |] in
  let written =
    Rig.submit s
      ~buffers:[| require_some (B.borrow d shared) |]
      ~args:[| values 8 |]
  in
  equal ~msg:"queued before" int 1 (P.queued pd);
  let out = S.words 1 in
  ignore
    (launch (Lazy.force affine) "affine"
       ~refs:Sub.[| write 0 ~at:0; read 1 ~at:8 |]
       ~params:40
       ~set:(fun b -> Bytes.set_int64_le b 16 1L)
       [| B.of_bigarray out; shared |]);
  done_ ~msg:"the Polled work it follows" d written

(* Splits *)

let pp_grid ppf { groups = gx, gy, gz; threads = tx, ty, tz } =
  Format.fprintf ppf "groups %dx%dx%d, threads %dx%dx%d" gx gy gz tx ty tz

(* A quarter of the grids have one group, and a quarter one thread a group, so
   that each of those covers misses below 1e-9 in 100 cases. *)
let grid_case =
  let open Gen in
  let axis n = frequency [ (2, constant 1); (3, int_range 1 n) ] in
  let ones = constant (1, 1, 1) in
  let+ groups =
    frequency
      [
        (1, ones);
        ( 3,
          triple
            (frequency [ (1, constant 1); (3, int_range 1 200) ])
            (axis 4) (axis 3) );
      ]
  and+ threads =
    frequency [ (1, ones); (3, triple (axis 9) (axis 3) (axis 2)) ]
  in
  { groups; threads }

let grid_case = Gen.with_pp pp_grid grid_case
let size (x, y, z) = x * y * z

(* The records of blocks.c's log: each call's first thread and its end. *)
let records log =
  List.init
    (Int64.to_int log.{0} / 2)
    (fun r -> (Int64.to_int log.{2 + (2 * r)}, Int64.to_int log.{3 + (2 * r)}))

(* blocks.c, loaded once. The mutex lets two domains ask for it at once, which a
   bare lazy refuses. *)
let blocks =
  let image = lazy (load "blocks") and lock = Mutex.create () in
  fun () -> Mutex.protect lock (fun () -> Lazy.force image)

(* blocks.c over [g]: each thread's count and the calls it logged. *)
let run_blocks g =
  let n = size g.groups * size g.threads in
  let counts = S.words n and log = S.words (2 + (2 * size g.groups)) in
  log.{1} <- Int64.of_int (Bigarray.Array1.dim log);
  ignore
    (launch (blocks ()) "blocks"
       ~refs:Sub.[| write 0 ~at:0; write 1 ~at:8 |]
       ~params:16 ~grid:g
       (words [| counts; log |]));
  (Array.init n (fun i -> Int64.to_int counts.{i}), records log)

(* Every thread ran once, and each call's range is whole consecutive groups. *)
let test_split g =
  let groups = size g.groups and t = size g.threads in
  cover "one group" (groups = 1);
  cover "one thread a group" (t = 1);
  cover "more groups than workers" (groups > S.workers ());
  let counts, calls = run_blocks g in
  equal ~msg:"each thread's runs" (array int) (Array.make (groups * t) 1) counts;
  List.iter
    (fun (first, end_) ->
      equal ~msg:"a call's first thread starts a group" int 0 (first mod t);
      equal ~msg:"a call's end ends a group" int 0 (end_ mod t);
      less ~msg:"a call's range" int ~than:end_ first)
    calls;
  at_most ~msg:"calls" int ~than:groups (List.length calls)

let test_refused (_, shared, grid) =
  let empty = load "empty" in
  raises_match Exn.invalid_arg (fun () ->
      launch ~shared ~grid empty "empty" [||])

(* Claims *)

(* A host buffer of [n] zero bytes, within the claims. *)
let zeroed n = B.of_string (String.make n '\000')
let take_none ?msg b = is_none ?msg (Rig_support.donated b)

(* Takes each buffer of [bs], which no claim may hold, and puts the buffer the
   take gives in its place. *)
let retake bs =
  Array.iteri
    (fun i b ->
      bs.(i) <-
        require_some ~msg:(strf "slot %d taken" i) (Rig_support.donated b))
    bs

(* The launch names one buffer in two slots; both claims end together, after the
   linked code returns. *)
let test_claims_while_running () =
  let image = load "waiting" and x = zeroed 16 in
  let at = Rig_support.at x in
  let runner =
    Domain.spawn (fun () -> ignore (waiting image [| x; x |] ~rounds:forever))
  in
  Fun.protect
    ~finally:(fun () ->
      Rig_support.store at 1;
      Domain.join runner)
    (fun () ->
      Rig_support.await "the code running" (fun () ->
          Rig_support.load (at + 8) = 1);
      take_none ~msg:"a take while it runs" x);
  retake [| x |]

(* The claim starts before the host waits for a Polled device's write the launch
   follows: the gate holds that wait. *)
let test_claims_before_waiting () =
  let d, pd = P.open_ "host:claims-wait" in
  let image = load "waiting" and x = zeroed B.paged_from in
  ignore
    (Rig.submit
       (Sub.make d [| Rig_support.touching [| Sub.write |] |])
       ~buffers:[| require_some (B.borrow d x) |]
       ~args:[| values 8 |]);
  P.gate pd;
  let runner =
    Domain.spawn (fun () -> ignore (waiting image [| x |] ~rounds:0))
  in
  Fun.protect
    ~finally:(fun () ->
      P.open_gate pd;
      Domain.join runner)
    (fun () ->
      Rig_support.await "the host launch waiting" (fun () -> P.sleepers pd > 0);
      take_none ~msg:"a take while it waits" x);
  retake [| x |]

(* A later buffer refuses its claim, being dead, after the earlier ones were
   claimed, and the submit ends their claims. *)
let test_claims_refused dead =
  let image = load "waiting" and bs = Array.init 3 (fun _ -> zeroed 16) in
  let live = require_some (Rig_support.donated bs.(dead)) in
  raises_match Exn.invalid_arg (fun () -> waiting image bs ~rounds:0);
  bs.(dead) <- live;
  retake bs;
  ignore (waiting image bs ~rounds:0);
  retake bs

(* A launch the host never runs, of no group, releases its claims. *)
let test_claims_no_group () =
  let image = load "waiting" and bs = [| zeroed 16 |] in
  raises_match Exn.invalid_arg (fun () ->
      waiting ~grid:{ one with groups = (0, 1, 1) } image bs ~rounds:0);
  retake bs;
  ignore (waiting image bs ~rounds:0);
  retake bs

(* A launch keeps its buffers until it returns: waiting.c runs on memory that
   only the submit holds, since its buffers array holds another buffer by then,
   while this domain collects, and a buffer made meanwhile gets other memory. *)
let test_keeps_buffers () =
  let n = 1 lsl 16 and image = load "waiting" in
  let bs = [| zeroed n |] in
  let a = R.host bs.(0) in
  let d = Domain.spawn (fun () -> ignore (waiting image bs ~rounds:forever)) in
  Fun.protect
    ~finally:(fun () ->
      Rig_support.store a 1;
      Domain.join d)
    (fun () ->
      Rig_support.await "the launch running" (fun () ->
          Rig_support.load (a + 8) = 1);
      bs.(0) <- zeroed 16;
      Gc.full_major ();
      Gc.full_major ();
      not_equal ~msg:"a buffer made while the launch runs" int a
        (R.host (B.create Rig.host n)))

(* Domains *)

(* Grids of at most 64 groups of 4 threads: the launches race, so a few groups
   per worker suffice. *)
let small_grid =
  let open Gen in
  let+ gx = int_range 1 64 and+ tx = int_range 1 4 in
  { groups = (gx, 1, 1); threads = (tx, 1, 1) }

let small_grid = Gen.with_pp pp_grid small_grid

(* "launch" makes no value, so with no prefix or suffix every call races the
   other domain's: each program is two branches of launches. *)
let domain_commands =
  [
    command "launch"
      (small_grid @-> returns (array int))
      (fun g -> Array.make (size g.groups * size g.threads) 1)
      (fun g -> fst (run_blocks g));
  ]

(* Sets w[0] once w[1] is set, then whether the launch had returned. *)
let test_released () =
  let w = S.words 2 and image = load "waiting" in
  let returned = Atomic.make 0 in
  let d =
    Domain.spawn (fun () ->
        ignore (waiting image [| B.of_bigarray w |] ~rounds:forever);
        Atomic.incr returned)
  in
  Rig_support.await "the launch running" (fun () -> w.{1} = 1L);
  Gc.full_major ();
  let during = Atomic.get returned in
  w.{0} <- 1L;
  Domain.join d;
  equal ~msg:"launches returned before the collection ended" int 0 during

(* The bytes of host memory a Polled device borrows. *)
let shared_bytes = B.paged_from

(* [f ()] while waiting.c, on its own domain, also reads [m]: the host's submit
   has read [m]'s stamps, and returns once [f] returns or raises. *)
let during_host_read m f =
  let image = load "waiting" and w = S.words 2 in
  let host =
    Domain.spawn (fun () ->
        ignore
          (launch image "waiting"
             ~refs:Sub.[| write 0 ~at:0; read 1 ~at:16 |]
             ~params:24
             ~set:(fun b -> Bytes.set_int64_le b 8 (Int64.of_int forever))
             [| B.of_bigarray w; m |]))
  in
  Fun.protect
    ~finally:(fun () ->
      w.{0} <- 1L;
      Domain.join host)
    (fun () ->
      Rig_support.await "the launch running" (fun () -> w.{1} = 1L);
      f ())

(* The point of work on [d] that reads [m] and does nothing. *)
let device_reading d m =
  Rig.submit
    (Sub.make d [| Rig_support.touching [| Sub.read |] |])
    ~buffers:[| require_some (B.borrow d m) |]
    ~args:[| values 8 |]

(* A device reads memory while a host launch reads it: a wait for a write once
   both submits have returned follows the device's read. *)
let test_beside_host () =
  let d, _ = P.open_ "host:beside" in
  let m = B.create Rig.host shared_bytes in
  let p = during_host_read m (fun () -> device_reading d m) in
  B.wait m Read_write;
  done_ ~msg:"the device's read" d p

(* A host launch on memory whose first use is a closed device's, done, and whose
   second is a live device's read: a wait for a write that starts while the
   launch runs follows the read. *)
let test_host_after_closed () =
  let m = B.create Rig.host shared_bytes in
  let closed, _ = P.open_ "host:closed" in
  Rig.Point.wait (device_reading closed m);
  let d, _ = P.open_ "host:live" in
  let p = device_reading d m in
  Rig.close closed;
  let signaled =
    during_host_read m (fun () ->
        B.wait m Read_write;
        Rig.signaled d)
  in
  at_least ~msg:"the device's read" int ~than:(Rig.Point.value p) signaled

(* [f w] while waiting.c's launch over [w] runs on a domain of its own: [f]'s
   result and the launch's point. *)
let while_held f =
  let w = S.words 2 and image = load "waiting" in
  let holder =
    Domain.spawn (fun () -> waiting image [| B.of_bigarray w |] ~rounds:forever)
  in
  let release () =
    w.{0} <- 1L;
    Domain.join holder
  in
  match
    Rig_support.await "the launch running" (fun () -> w.{1} = 1L);
    f w
  with
  | r -> (r, release ())
  | exception e ->
      ignore (release ());
      raise e

(* While one domain's launch runs, another domain's host submit waits before it
   takes its value. Its work runs after the held launch's: it reads the flag
   that the held launch waited for. *)
let test_held_submit () =
  if not Rig_support.sees_waits then
    skip ~reason:"the system shows no thread's waits" ();
  let out = S.words 1 and thread = Atomic.make None in
  ignore (Lazy.force affine);
  let (during, other), held =
    while_held (fun w ->
        let other =
          Domain.spawn (fun () ->
              Atomic.set thread (Some (Rig_support.self ()));
              affine_launch ~n:1 ~t:1 ~a:2 ~c:1 out w)
        in
        Rig_support.await "the other submit waiting" (fun () ->
            match Atomic.get thread with
            | Some t -> Rig_support.waits_in_condition t
            | None -> false);
        ((Rig.submitted Rig.host, out.{0}), other))
  in
  let p = Domain.join other in
  equal ~msg:"while held: the last value, the other's output" (pair int int64)
    (Rig.Point.value held, 0L)
    during;
  equal ~msg:"the other's value" int
    (Rig.Point.value held + 1)
    (Rig.Point.value p);
  equal ~msg:"the other's output, 2 * flag + 1" int64 3L out.{0}

(* While one domain's launch runs, another domain's waits for the host's earlier
   work return, and so does a submit on another device that waits for it. *)
let test_waits_beside_launch () =
  let d, _ = P.open_ "host:waits-beside" in
  let out = S.words 1 and inb = S.words 1 in
  let first = affine_launch ~n:1 ~t:1 ~a:1 ~c:1 out inb in
  let during, held =
    while_held (fun _ ->
        Rig.Point.wait first;
        let after =
          Rig.submit ~waits:[| first |] (Sub.make d [||]) ~buffers:[||]
            ~args:[||]
        in
        Rig.Point.wait after;
        Rig.signaled Rig.host)
  in
  at_least ~msg:"while held: the first" int ~than:(Rig.Point.value first) during;
  less ~msg:"while held: the held launch's" int ~than:(Rig.Point.value held)
    during

(* Software devices *)

let test_software_affine (n, t, a, c, input) =
  cover "one element" (n = 1);
  cover "a last group in part past n" (n mod t <> 0);
  let d = Lazy.force software in
  let inb = words_on d input and out = words_on d (Array.make (n + 1) 7) in
  let image = Lazy.force software_affine in
  let p = affine_on ~device:d image ~n ~t ~a ~c out inb in
  done_ ~msg:"when submit returns" d p;
  let expected = Array.map (fun x -> (a * x) + c) input in
  equal (array int) (Array.append expected [| 7 |]) (read_words out)

(* A profile records a launch's work on a software device, named after its
   function. *)
let test_software_profiled () =
  let d = Lazy.force software in
  let inb = words_on d [| 1; 2 |] and out = words_on d [| 0; 0 |] in
  let _, events =
    Rig.Profile.take (fun () ->
        affine_on ~device:d
          (Lazy.force software_affine)
          ~n:2 ~t:2 ~a:1 ~c:0 out inb)
  in
  equal (list string) [ "affine" ] (work_names d events)

let launch_tests =
  group "launches"
    [
      prop
        "a launch runs its function over every thread, done when submit returns"
        affine_case test_affine;
      prop
        "a software device's launch runs its function over every thread, done \
         when its submit returns"
        affine_case test_software_affine;
      test "an int64 parameter holds every int64" test_int64_parameter;
      test "a launch reads 8,192 bytes of parameters" test_wide;
      test "a launch's parameters are bounded only by what one block holds"
        test_run_bound;
      prop "a launch runs every thread once, in calls of whole groups" grid_case
        test_split;
      stateful ~domains:2 ~steps:0 ~count:30
        "launches from two domains each run every thread once" domain_commands;
      cases "a launch never runs with"
        ~id:(fun (n, _, _) -> n)
        grid_refusals test_refused;
      test "another domain collects while a launch runs" test_released;
      test
        "a host submit waits for another domain's running launch before its \
         value"
        test_held_submit;
      test "waits on the host go on while a launch runs"
        test_waits_beside_launch;
      test
        "a launch keeps its buffers until it returns, whatever its array holds"
        test_keeps_buffers;
      test "a launch claims its buffers, one in two slots, until it returns"
        test_claims_while_running;
      test "a launch claims its buffers before it waits for a device's write"
        test_claims_before_waiting;
      cases "a submit refused for a dead buffer releases its claims, dead at"
        ~id:(strf "slot %d") [ 0; 1; 2 ] test_claims_refused;
      test "a launch of no group, refused, releases its claims"
        test_claims_no_group;
      test "a launch follows the work on the memory its refs name" test_follows;
      test "a wait for a write follows a device's read beside a host launch"
        test_beside_host;
      test
        "a wait for a write during a host launch follows a device's read after \
         a closed device's"
        test_host_after_closed;
      test "a profile records a launch's work, named after its function"
        test_profiled;
      test "a profile records a software device's launch" test_software_profiled;
    ]

(* Linking *)

let read f = In_channel.with_open_bin f In_channel.input_all
let refused why = Rig.name Rig.host ^ ": " ^ why

let load_error o =
  require_error
    ~pp:(fun ppf _ -> Format.pp_print_string ppf "an image")
    (Rig.Image.load Rig.host o)

let test_functions () =
  let image = load "data_entry" in
  let resolved f = Option.map Rig.Image.name (Rig.Image.resolve image f) in
  equal
    (list (pair string (option string)))
    [ ("code", Some "code"); ("data_entry", None); ("nowhere", None) ]
    (List.map (fun f -> (f, resolved f)) [ "code"; "data_entry"; "nowhere" ])

let test_not_elf () =
  starts_with ~prefix:(refused "") (load_error "not an object")

let other = if S.machine = "x86_64" then "aarch64" else "x86_64"
let host_name = if S.machine = "x86_64" then "x86_64" else "arm64"

let other_machine =
  if other = "x86_64" then "machine 62 (x86_64)" else "machine 183 (arm64)"

(* Each row's object is read as its test runs, which skips where rig links no
   host code ([S.fixture]). *)
let other_refusals =
  [
    ( "an object of another machine",
      (fun () ->
        ignore (obj "affine");
        read (Filename.concat shared_dir (strf "affine_%s.o" other))),
      strf "the object is for %s, expected %s" other_machine host_name );
    ( "writable data",
      (fun () -> obj "writable"),
      "section \".bss\" is writable and not empty; pass variables through the \
       parameters" );
    ( "an undefined symbol",
      (fun () -> obj "undefined"),
      "symbol \"rig_host_test_nowhere\" is defined by no library of the process"
    );
    (* far's jump is R_X86_64_PLT32 at 0x9 or R_AARCH64_JUMP26 at 0x4, to S + A:
       2^61 - 4 on x86_64, whose addend counts from the field's end, and 2^61 on
       arm64. *)
    ( "a jump out of its field's reach",
      (fun () -> obj "far"),
      if S.machine = "x86_64" then
        "relocation at 0x9 reaches 0x1ffffffffffffffc, which does not fit its \
         32-bit field"
      else
        "relocation at 0x4 reaches 0x2000000000000000, which does not fit its \
         28-bit field" );
  ]

(* ELF64's header (the System V gABI, chapter 4): the class at byte 4 (1 for
   32-bit), the data encoding at 5 (2 for big-endian), the type at 16 (2 an
   executable, 3 a shared object); a section header's type at 4, its offset at
   24, its link at 40 and its entry size at 56; a symbol's section index at 6,
   SHN_COMMON 0xfff2 for a common one. *)
let sht_rela = 4
let sht_rel = 9
let sht_symtab = 2
let shn_common = 0xfff2
let set64 b at x = Bytes.set_int64_le b at (Int64.of_int x)
let get64 b at = Int64.to_int (Bytes.get_int64_le b at)

(* The offsets of [b]'s section headers. *)
let headers b =
  let shoff = get64 b 0x28 in
  let entsize = Bytes.get_uint16_le b 0x3a and n = Bytes.get_uint16_le b 0x3c in
  List.init n (fun i -> shoff + (i * entsize))

(* [obj] with the header of its first section of type [kind], at [h], set by
   [set b h]. *)
let patch_section obj ~kind set =
  let b = Bytes.of_string obj in
  let kind_of h = Int32.to_int (Bytes.get_int32_le b (h + 4)) in
  set b (List.find (fun h -> kind_of h = kind) (headers b));
  Bytes.to_string b

(* The string at [at] of the string table whose section header is at [h]. *)
let string_in b h at =
  let at = get64 b (h + 24) + at in
  Bytes.sub_string b at (Bytes.index_from b at '\000' - at)

(* [obj] with the header of its section [name], at [h], set by [set b h]. The
   ELF64 header holds the index of the section of section names, [e_shstrndx],
   at 0x3e. *)
let patch_named obj name set =
  let b = Bytes.of_string obj in
  let hs = headers b in
  let names = List.nth hs (Bytes.get_uint16_le b 0x3e) in
  let name_of h = string_in b names (Int32.to_int (Bytes.get_int32_le b h)) in
  set b (List.find (fun h -> name_of h = name) hs);
  Bytes.to_string b

(* [obj] with the entry of its symbol [name], at [e], set by [set b e]. *)
let patch_symbol obj name set =
  patch_section obj ~kind:sht_symtab (fun b h ->
      let names =
        List.nth (headers b) (Int32.to_int (Bytes.get_int32_le b (h + 40)))
      in
      let first = get64 b (h + 24) and size = get64 b (h + 32) in
      let e =
        List.find
          (fun e ->
            string_in b names (Int32.to_int (Bytes.get_int32_le b e)) = name)
          (List.init (size / 24) (fun k -> first + (24 * k)))
      in
      set b e)

(* The last section of [obj]'s image: its sections lie in the order of their
   headers, and the last that is allocated ([SHF_ALLOC], 2, in the flags at 8)
   and holds bytes ends it. *)
let last_allocated b =
  List.fold_left
    (fun last (i, h) ->
      if get64 b (h + 8) land 2 <> 0 && get64 b (h + 32) > 0 then Some (i, h)
      else last)
    None
    (List.mapi (fun i h -> (i, h)) (headers b))
  |> Option.get

(* [obj name] with its bytes changed by [set]. *)
let patched name set () =
  let b = Bytes.of_string (obj name) in
  set b;
  Bytes.to_string b

(* Objects that break a rule of Linking, each a fixture with one field patched,
   with the reason the load gives. linked's last section is its .rodata.cst32 at
   0x60, of 32 bytes, on x86_64 and arm64. *)
let broken_objects =
  [
    ( "a 32-bit object",
      patched "empty" (fun b -> Bytes.set_uint8 b 4 1),
      "the object is 32-bit, expected 64-bit" );
    ( "a big-endian object",
      patched "empty" (fun b -> Bytes.set_uint8 b 5 2),
      "not a little-endian ELF object" );
    ( "an executable",
      patched "empty" (fun b -> Bytes.set_uint16_le b 16 2),
      "the object is of type 2, expected ET_REL (1)" );
    ( "a shared object",
      patched "empty" (fun b -> Bytes.set_uint16_le b 16 3),
      "the object is of type 3, expected ET_REL (1)" );
    ( "a relocation of a common symbol",
      (fun () ->
        patch_symbol (obj "linked") "twice" (fun b e ->
            Bytes.set_uint16_le b (e + 6) shn_common)),
      "symbol \"twice\" lies outside the image" );
    (* The relocation section keeps its first entry, made to patch the image's
       last section 2 bytes before its end: a 4-byte field that crosses it. A
       section header holds its size at 32 and the section it relocates,
       [sh_info], at 44. *)
    ( "a relocation that patches past the image",
      (fun () ->
        patch_section (obj "linked") ~kind:sht_rela (fun b h ->
            let i, last = last_allocated b in
            Bytes.set_int32_le b (h + 44) (Int32.of_int i);
            set64 b (h + 32) 24;
            set64 b (get64 b (h + 24)) (get64 b (last + 32) - 2))),
      "relocation at 0x7e patches past the image's end" );
  ]

let refusals = other_refusals @ broken_objects
let test_refusal (_, o, msg) = equal string (refused msg) (load_error (o ()))

(* An ELF64 section header holds its alignment, [sh_addralign], at 48. *)
let set_align align b h = set64 b (h + 48) align

(* Alignments at, around and above every host's page and Windows' allocation
   granularity of 64 KiB. *)
let alignments = [ 16; 4096; 16384; 65536; 1 lsl 17; 1 lsl 20 ]

(* aligned.c with its table's section aligned to [align]: the link places the
   table at a multiple of it, and the code reads it there. *)
let test_aligned align =
  let o = patch_named (obj "aligned") ".rodata" (set_align align) in
  let image = load_on Rig.host o in
  let out = S.words 5 in
  ignore
    (launch image "aligned"
       ~refs:Sub.[| write 0 ~at:0 |]
       ~params:8
       ~grid:{ one with threads = (4, 1, 1) }
       (words [| out |]));
  equal ~msg:"the table's address modulo its alignment" int 0
    (Int64.to_int out.{0} land (align - 1));
  equal ~msg:"the table" (array int) [| 3; 1; 4; 1 |]
    (Array.init 4 (fun i -> Int64.to_int out.{1 + i}))

(* An alignment no address space holds: the system refuses the mapping, and the
   error names the alignment. *)
let test_unmappable () =
  let align = 1 lsl 61 in
  let why =
    load_error (patch_named (obj "aligned") ".rodata" (set_align align))
  in
  starts_with ~prefix:(refused "mapping ") why;
  contains ~sub:(strf "at a multiple of %d bytes" align) why

(* linked's first relocation section as an SHT_REL one of its first entry: an
   ELF64 Rel entry is a Rela entry without its addend, 16 bytes. *)
let test_rel () =
  let at = ref 0 in
  let o =
    patch_section (obj "linked") ~kind:sht_rela (fun b h ->
        at := get64 b (get64 b (h + 24));
        Bytes.set_int32_le b (h + 4) (Int32.of_int sht_rel);
        set64 b (h + 32) 16;
        set64 b (h + 56) 16)
  in
  equal string
    (refused
       (strf "relocation at 0x%x has its addend in its field (SHT_REL)" !at))
    (load_error o)

(* got's first relocation reads cbrt's address from a word of a global offset
   table: R_X86_64_REX_GOTPCRELX (42) or R_AARCH64_ADR_GOT_PAGE (311). *)
let test_got () =
  let kind = if S.machine = "x86_64" then 42 else 311 in
  starts_with
    ~prefix:(refused (strf "relocation of type %d at 0x" kind))
    (load_error (obj "got"))

let linking_tests =
  group "linking"
    [
      test "an image's functions are the symbols of its code" test_functions;
      test "a load of what is not ELF is Error naming the host" test_not_elf;
      cases "a load is refused for"
        ~id:(fun (n, _, _) -> n)
        refusals test_refusal;
      test "a load of relocations whose addends are in their fields is refused"
        test_rel;
      test "a load of a relocation through a global offset table is refused"
        test_got;
      cases "a function reads data aligned to" ~id:(strf "%d bytes") alignments
        test_aligned;
      test "a load of an alignment no address space holds is refused"
        test_unmappable;
    ]

(* Relocations *)

let test_linked () =
  let out = Bigarray.(Array1.create float64 c_layout 3)
  and inb = Bigarray.(Array1.of_array float64 c_layout [| 27.; 1.25 |]) in
  ignore
    (launch (load "linked") "linked"
       ~refs:Sub.[| write 0 ~at:0; read 1 ~at:8 |]
       ~params:24
       ~set:(fun b -> Bytes.set_int64_le b 16 2L)
       [| B.of_bigarray out; B.of_bigarray inb |]);
  equal ~msg:"cbrt 27" float_exact (Float.cbrt 27.) out.{0};
  equal ~msg:"table.(2)" float_exact 3.125 out.{1};
  equal ~msg:"twice 1.25" float_exact 2.5 out.{2}

let relocation_tests =
  group "relocations"
    [
      test
        "a function calls the process's cbrt and its own function, and reads \
         its constants"
        test_linked;
    ]

(* Calling conventions *)

(* deep.c over [groups] groups of one thread, each on whichever thread the pool
   gives it, the caller's included: each element of its output. *)
let deep_launch groups =
  let out = S.words groups in
  ignore
    (launch (load "deep") "deep"
       ~refs:Sub.[| write 0 ~at:0 |]
       ~params:16
       ~set:(fun b -> Bytes.set_int64_le b 8 0L)
       ~grid:{ one with groups = (groups, 1, 1) }
       (words [| out |]));
  Array.init groups (fun i -> Int64.to_int out.{i})

let test_deep () =
  List.iter
    (fun groups ->
      equal ~msg:(strf "%d groups" groups) (array int) (Array.make groups 2)
        (deep_launch groups))
    [ 1; 4 * S.workers () ]

let convention_tests =
  group "calling conventions"
    [
      test
        "a function whose stack frame spans 256 KiB, written at its far end \
         first, runs on every thread"
        test_deep;
    ]

(* Platform support *)

(* The host and a software device link where rig's linker, built for the
   process's machine, does: on x86_64 and arm64. *)
let test_links () =
  let isa = Rig.instruction_set Rig.host in
  let expected = isa = "x86_64" || isa = "arm64" in
  equal
    (list (pair string bool))
    [
      (isa, expected);
      ("a software device", expected);
      ("the fixtures' machine is known", expected);
    ]
    [
      (isa, Rig.Image.links Rig.host);
      ("a software device", Rig.Image.links (Lazy.force software));
      ("the fixtures' machine is known", S.machine <> "unknown");
    ]

(* On an instruction set rig links no code for, a load on the host is refused,
   naming the set where rig knows it. *)
let test_unlinked () =
  if S.links then skip ~reason:"rig links host code on this machine" ();
  let set =
    match Rig.instruction_set Rig.host with
    | "" -> "an instruction set it does not know"
    | isa -> isa
  in
  equal string
    (refused ("rig links host code on x86_64 and arm64, not on " ^ set))
    (load_error (read (Filename.concat shared_dir "affine_x86_64.o")))

(* The permissions of the mapping of /proc/self/maps that holds [a]. *)
let permissions a =
  let holds line =
    match String.split_on_char ' ' line with
    | range :: perms :: _ -> (
        match String.split_on_char '-' range with
        | [ lo; hi ] ->
            let lo = int_of_string ("0x" ^ lo)
            and hi = int_of_string ("0x" ^ hi) in
            if lo <= a && a < hi then Some perms else None
        | _ -> None)
    | _ -> None
  in
  In_channel.with_open_text "/proc/self/maps" In_channel.input_lines
  |> List.find_map holds

let test_never_writable () =
  if not (Sys.file_exists "/proc/self/maps") then
    skip ~reason:"needs /proc/self/maps (Linux)" ();
  let image = load "affine" in
  let perms = permissions (entry_of image "affine") in
  ignore (Sys.opaque_identity image);
  equal (option string) (Some "r-xp") perms

let platform_tests =
  group "platform support"
    [
      test "the host and a software device link host code on x86_64 and arm64"
        test_links;
      test "a load on an instruction set rig links no code for is refused"
        test_unlinked;
      test "linked code is executable and read-only once written"
        test_never_writable;
    ]

let () =
  exit
    (run "rig_host"
       [
         host_code_tests;
         launch_tests;
         linking_tests;
         relocation_tests;
         convention_tests;
         platform_tests;
       ])
