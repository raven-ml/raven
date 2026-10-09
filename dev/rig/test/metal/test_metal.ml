(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module S = Rig_metal_support
module H = Rig_gpu_support.Host
module B = Rig.Buffer

let strf = Printf.sprintf

(* The ring

   A model of the ring of a device's command buffers: slots taken in commit
   order, completed in any order, released in commit order. The word is the last
   value released before the first failed slot, and after stop, once no slot is
   taken, the last value taken. Completions go on after stop; commits and waits
   do not. *)

module Ring = struct
  type state = Taken | Done | Failed
  type slot = { index : int; k : int; v : int; mutable state : state }

  type t = {
    n : int;
    mutable taken : slot list;
    mutable tail : int;
    mutable values : int;
    mutable commits : int;
    mutable completed : int list;
    mutable word : int;
    mutable held : bool;
    mutable drain : bool;
    mutable failure : string option;
    mutable over : bool;
  }

  let make n =
    {
      n;
      taken = [];
      tail = 0;
      values = 0;
      commits = 0;
      completed = [];
      word = 0;
      held = false;
      drain = false;
      failure = None;
      over = false;
    }

  let commit m ~last =
    cover "a slot is taken again" (m.tail >= m.n);
    let v = if last then m.values + 1 else 0 in
    let s = { index = m.tail mod m.n; k = m.commits; v; state = Taken } in
    if last then m.values <- v;
    m.taken <- m.taken @ [ s ];
    m.tail <- m.tail + 1;
    m.commits <- m.commits + 1;
    s.index

  let rec release m = function
    | s :: rest when s.state <> Taken ->
        if s.state = Failed then m.held <- true;
        cover "a value completes after a failed slot" (m.held && s.v > 0);
        if s.v > 0 && not m.held then m.word <- s.v;
        release m rest
    | rest -> rest

  let complete m i ~failed =
    let s = List.find (fun s -> s.index = i) m.taken in
    cover "a slot completes before an earlier one" (s != List.hd m.taken);
    s.state <- (if failed then Failed else Done);
    m.completed <- s.k :: m.completed;
    if failed && m.failure = None then
      m.failure <- Some (strf "command buffer %d failed" s.k);
    m.taken <- release m m.taken;
    if m.drain && m.taken = [] then begin
      cover "the last slot completes after stop" true;
      m.word <- m.values
    end

  let stop m =
    m.over <- true;
    if m.taken = [] then m.word <- m.values else m.drain <- true;
    m.taken = []

  let pending m =
    List.filter_map
      (fun s -> if s.state = Taken then Some s.index else None)
      m.taken

  let times m k =
    if List.mem k m.completed then ((10 * k) + 1, (10 * k) + 2) else (0, 0)
end

let ring_invariant (m : Ring.t) r =
  equal int ~msg:"word" m.word (S.word r);
  equal (option string) ~msg:"failure" m.failure (S.failure r);
  for k = 0 to m.commits - 1 do
    equal (pair int int)
      ~msg:(strf "times of commit %d" k)
      (Ring.times m k) (S.times r k)
  done

let ring = abstract "r" ~invariant:ring_invariant
let slot = among int ring Ring.pending
let open_ (m : Ring.t) = not m.over

let ring_commands =
  [
    command "ring" (Gen.int_range 1 6 @-> makes ring) Ring.make S.ring;
    command "commit"
      ~pre:(fun (m : Ring.t) _ -> open_ m && List.length m.taken < m.n)
      (ring ^-> Gen.bool @-> returns int)
      (fun m last -> Ring.commit m ~last)
      (fun r last -> S.commit r ~last);
    command "complete"
      (ring ^-> slot ^-> Gen.bool @-> returns unit)
      (fun m i failed -> Ring.complete m i ~failed)
      (fun r i failed -> S.complete r i ~failed);
    command "sleep" ~pre:open_
      (ring ^-> returns (option string))
      (fun (m : Ring.t) -> m.failure)
      S.sleep;
    command "stop" ~pre:open_ (ring ^-> returns bool) Ring.stop S.stop;
  ]

let ring_tests =
  group ~timeout:60. "ring"
    [
      stateful ~count:500 ~steps:40
        "releases in commit order whatever order slots complete in"
        ring_commands;
    ]

(* Devices

   Work reaches a device through rig, which opens it: [d] is the device there,
   [g] the driver's. One device serves the tests that leave it healthy; a test
   that fails a submission opens its own, which the next shared use replaces.
   The tests of the driver's own stop open a device that rig never takes
   ([S.driver]), and the tests of two devices a second one. *)

type dev = { d : Rig.t; g : Rig_metal.t; fill : Rig_metal.image }

(* The image of the fixture [fill], which Metal places itself. *)
let load g =
  match require_ok (Rig_metal.image g (S.fixture ~dir:"fixtures" "fill")) with
  | `Loaded i -> i
  | `Place (n, _) -> failf "the device asked to place %d bytes of code" n

let dev_of { S.d; g } = { d; g; fill = load g }

(* The shared device, opened again once a test's own replaced it. *)
let dev =
  let lock = Mutex.create () and shared = ref None in
  fun () ->
    Mutex.protect lock @@ fun () ->
    match !shared with
    | Some t when Option.is_none (Rig.lost t.d) -> t
    | _ ->
        let t = dev_of (S.open_ ()) in
        shared := Some t;
        t

(* A second device of the GPU, beside the fixture's: Metal opens a GPU as
   often as asked. The test stops it. *)
let second () =
  if not (S.present ()) then skip ~reason:"the machine has no METAL GPU" ();
  S.hold ();
  require_ok (Rig_metal.open_ 0)

let pipeline t f = require_some (Rig_metal.entry t.fill f)

(* Submits [parts] as [t]'s next value, which it is. *)
let submit_parts t parts =
  let s = Rig.Submission.make ~reads:0 ~writes:0 t.d parts in
  Rig.Point.value (Rig.submit s ~reads:[||] ~writes:[||] ~waits:[||])

let submit t fills = submit_parts t (Array.map S.part fills)
let wait t v = Rig.wait t.d v
let alloc_on g n = require_some (Rig_metal.alloc g `Device n)
let alloc t n = alloc_on t.g n
let host r = require_some (Rig_metal.host r)
let gpu r = require_some (Rig_metal.address r)

(* The arguments of the fill kernels, [{ out; c }], at byte [at] of [args]. *)
let set_args args ~at ~out ~c =
  H.set64 (host args + at) out;
  H.set32 (host args + at + 8) c

let args_bytes = 16

(* A dispatch of [fill] writing [3i + c] at the 32-bit words [0, n) of the GPU
   address [out], its arguments at byte [at] of [args]. *)
let fill_dispatch t ~args ~at ~out ~c n =
  set_args args ~at ~out ~c;
  S.dispatch ~pipeline:(pipeline t "fill") ~offset:at args ~groups:1 ~threads:n

let filled out ?(at = 0) ~c n =
  Array.init n (fun i -> H.get32 (host out + (4 * (at + i))) = (3 * i) + c)
  |> Array.for_all Fun.id

let dispatch ?(offset = 0) ?(groups = (1, 1, 1)) ?(threads = (1, 1, 1)) p =
  { Rig_metal_abi.pipeline = p; offset; groups; threads }

let icb_on g args ds = (Rig_metal.capability g).icb (Rig_metal.handle args) ds
let icb t args ds = icb_on t.g args ds

(* Work *)

let gen_submission = Gen.list ~size:(Gen.int_range 0 3) (Gen.int_range 0 3)

let prefix_completion parts =
  let t = dev () in
  let n = List.length parts in
  let chunk = 64 in
  let out = alloc t (n * 4 * chunk * 4)
  and args = alloc t (n * 4 * args_bytes) in
  let first = Rig.submitted t.d + 1 in
  cover "an empty submission" (List.mem [] parts);
  cover "a submission of several fills"
    (List.exists (fun p -> List.length p > 1) parts);
  cover "a fill that splits" (List.exists (List.exists (fun k -> k > 0)) parts);
  let c s p = (16 * s) + p + 1 in
  let fills s ps =
    List.mapi
      (fun p k ->
        let at = ((4 * s) + p) * args_bytes in
        let out = gpu out + (((4 * s) + p) * chunk * 4) in
        let f = fill_dispatch t ~args ~at ~out ~c:(c s p) chunk in
        S.split f t.g k ~times:0;
        f)
      ps
    |> Array.of_list
  in
  List.iteri (fun s ps -> ignore (submit t (fills s ps))) parts;
  let complete w =
    List.iteri
      (fun s ps ->
        if first + s <= w then
          List.iteri
            (fun p _ ->
              let at = ((4 * s) + p) * chunk in
              if not (filled out ~at ~c:(c s p) chunk) then
                failf "value %d is signaled, part %d of it unwritten"
                  (first + s) p)
            ps)
      parts
  in
  let rec poll seen =
    let w = Rig_metal.signaled t.g in
    at_least int ~than:seen w;
    complete w;
    if w < Rig.submitted t.d then poll w
  in
  poll (first - 1);
  Rig_metal.free t.g out;
  Rig_metal.free t.g args

(* Splits [k] times, recording times; [(before, times, after)]: the host clock
   before the submission and after its value was signaled. *)
let split_times k =
  let t = dev () in
  let out = alloc t 256
  and args = alloc t args_bytes
  and times = alloc t (16 * k) in
  let f = fill_dispatch t ~args ~at:0 ~out:(gpu out) ~c:1 64 in
  S.split f t.g k ~times:(host times);
  let before = S.uptime () in
  let v = submit t [| f |] in
  wait t v;
  let after = S.uptime () in
  let at i = H.get64 (host times + (8 * i)) in
  let r =
    (before, Array.init k (fun i -> (at (2 * i), at ((2 * i) + 1))), after)
  in
  List.iter (Rig_metal.free t.g) [ out; args; times ];
  r

let times_between k =
  let before, times, after = split_times k in
  Array.iteri
    (fun i (start, end_) ->
      let msg = strf "split %d" i in
      at_least int ~msg ~than:before start;
      at_least int ~msg ~than:start end_;
      at_most int ~msg ~than:after end_)
    times

let many_splits () =
  let t = dev () in
  let out = alloc t 256 and args = alloc t args_bytes in
  let f = fill_dispatch t ~args ~at:0 ~out:(gpu out) ~c:5 64 in
  S.split f t.g 1100 ~times:0;
  wait t (submit t [| f |]);
  equal bool true (filled out ~c:5 64);
  List.iter (Rig_metal.free t.g) [ out; args ]

let fresh_allocation () =
  let t = dev () in
  let out = alloc t (1 lsl 20) and args = alloc t args_bytes in
  let f = fill_dispatch t ~args ~at:0 ~out:(gpu out) ~c:9 256 in
  wait t (submit t [| f |]);
  equal bool true (filled out ~c:9 256);
  List.iter (Rig_metal.free t.g) [ out; args ]

let several_fills () =
  let t = dev () in
  let out = alloc t 1024 and args = alloc t (2 * args_bytes) in
  let first = fill_dispatch t ~args ~at:0 ~out:(gpu out) ~c:1 1 in
  let step =
    S.dispatch ~pipeline:(pipeline t "step") ~offset:0 args ~groups:1 ~threads:1
  in
  wait t (submit t [| first; step; step; step |]);
  equal int 4 (H.get32 (host out));
  List.iter (Rig_metal.free t.g) [ out; args ]

(* A submission's fills, after one that writes [3i + c] at each word of a
   region: [`Direct k] bumps every byte of the region and splits [k] times,
   bumping again after each split; [`Icb n] runs an indirect command buffer of
   [n] bumps. Each bump of the region dispatches 1,024 threadgroups. *)
let gen_fills =
  Gen.list ~size:(Gen.int_range 1 6)
    (Gen.one_of
       [
         Gen.map (fun k -> `Direct k) (Gen.int_range 0 2);
         Gen.map (fun n -> `Icb n) (Gen.int_range 1 3);
       ])

let bumps_counted fills =
  let t = dev () in
  let words = 65536 and c = 7 in
  let bytes = 4 * words in
  let out = alloc t bytes and args = alloc t (2 * args_bytes) in
  set_args args ~at:0 ~out:(gpu out) ~c;
  set_args args ~at:args_bytes ~out:(gpu out) ~c:bytes;
  let seed =
    S.dispatch ~pipeline:(pipeline t "fill") args ~groups:(words / 256)
      ~threads:256
  in
  let bump = pipeline t "bump" in
  let bumps k =
    Array.make k
      (dispatch ~offset:args_bytes ~groups:(bytes / 256, 1, 1)
         ~threads:(256, 1, 1) bump)
  in
  let icbs = ref [] in
  let fill = function
    | `Direct k ->
        let f =
          S.dispatch ~pipeline:bump ~offset:args_bytes args
            ~groups:(bytes / 256) ~threads:256
        in
        S.split f t.g k ~times:0;
        f
    | `Icb n ->
        let b = require_ok (icb t args (bumps n)) in
        icbs := b :: !icbs;
        S.execute b
  in
  let rec pairs = function
    | a :: (b :: _ as rest) -> (a, b) :: pairs rest
    | _ -> []
  in
  let follows p q = List.exists (fun (a, b) -> p a && q b) (pairs fills) in
  let direct = function `Direct _ -> true | `Icb _ -> false in
  let icb_fill f = not (direct f) in
  let splits = function `Direct k -> k > 0 | `Icb _ -> false in
  cover "an icb fill after a direct fill" (follows direct icb_fill);
  cover "a direct fill after an icb fill" (follows icb_fill direct);
  cover "a fill after one that splits" (follows splits (fun _ -> true));
  wait t (submit t (Array.of_list (seed :: List.map fill fills)));
  let k =
    List.fold_left
      (fun k -> function `Direct s -> k + s + 1 | `Icb n -> k + n)
      0 fills
  in
  let byte w i = (((w lsr (8 * i)) + k) land 255) lsl (8 * i) in
  for i = 0 to words - 1 do
    let w = (3 * i) + c in
    let want = byte w 0 lor byte w 1 lor byte w 2 lor byte w 3 in
    let got = H.get32 (host out + (4 * i)) in
    if got <> want then
      failf "word %d reads %#x, not %#x after %d bumps" i got want k
  done;
  List.iter (fun (b : Rig_metal_abi.icb) -> b.release ()) !icbs;
  List.iter (Rig_metal.free t.g) [ out; args ]

let empty_submission () =
  let t = dev () in
  let v = submit t [||] in
  wait t v;
  equal int v (Rig_metal.signaled t.g)

(* Metal hands a command buffer to its completion handler before the word moves
   and releases it, with the objects it holds, once the handler returned: no
   signal marks that, so a test waits for the weak reference [w] to empty under
   its group's timeout. *)
let await_release w =
  while S.alive w do
    Domain.cpu_relax ()
  done

let released_buffer () =
  let t = dev () in
  let f, w = S.watching () in
  wait t (submit t [| f |]);
  await_release w

let failing_fill () =
  S.with_ @@ fun s ->
  let t = dev_of s in
  let out = alloc t 256 and args = alloc t args_bytes in
  let ok = fill_dispatch t ~args ~at:0 ~out:(gpu out) ~c:2 64 in
  let reached = submit t [| ok |] in
  let why = "running a fill: it returned 7" in
  raises_match
    (function Rig.Lost (_, w) -> String.equal w why | _ -> false)
    (fun () -> submit t [| ok; S.failing 7 |]);
  raises (Rig_metal.Fault why) (fun () ->
      Rig_metal.sleep t.g ~seen:reached ~still_ms:10);
  S.close s;
  while Rig.signaled t.d < reached + 1 do
    Domain.cpu_relax ()
  done

let work =
  group ~timeout:60. "work"
    [
      prop ~count:50
        "a signaled value's submissions and every earlier one are written"
        (Gen.list ~size:(Gen.int_range 1 50) gen_submission)
        prefix_completion;
      test "an empty submission is signaled" empty_submission;
      test "fills of a submission run in order" several_fills;
      prop ~count:50
        "every bump of every fill is counted, across indirect command buffers \
         and splits"
        gen_fills bumps_counted;
      test "a kernel writes an allocation in the submission after it"
        fresh_allocation;
      cases
        ~name:(strf "a split's times lie in the work's span (%d splits)")
        "times" [ 1; 5; 64 ] times_between;
      test "a fill splitting more often than the queue holds completes"
        many_splits;
      test "a submission's command buffer is released once it completed"
        released_buffer;
      test
        "a failed fill loses the device with its reason, and stop brings the \
         word to its value"
        failing_fill;
    ]

(* Commits *)

(* Three command buffers, each one dispatch of [spin] that runs long enough for
   the work submitted after them to wait in the open command buffer: the regions
   to free once they completed. *)
let busy t =
  let out = alloc t 256 and args = alloc t args_bytes in
  set_args args ~at:0 ~out:(gpu out) ~c:5_000_000;
  let f = S.dispatch ~pipeline:(pipeline t "spin") args ~groups:1 ~threads:1 in
  for _ = 1 to 3 do
    ignore (submit t [| f |])
  done;
  [ out; args ]

(* [n] values behind three running command buffers, each one dispatch of [step]:
   with no commit and no wait, the word reaches the last and [step] counted
   each. *)
let uncommitted n =
  let t = dev () in
  let held = busy t in
  let out = alloc t 256 and args = alloc t args_bytes in
  H.set32 (host out) 0;
  set_args args ~at:0 ~out:(gpu out) ~c:0;
  let f = S.dispatch ~pipeline:(pipeline t "step") args ~groups:1 ~threads:1 in
  let last = ref 0 in
  for _ = 1 to n do
    last := submit t [| f |]
  done;
  while Rig_metal.signaled t.g < !last do
    Domain.cpu_relax ()
  done;
  equal int n (H.get32 (host out));
  List.iter (Rig_metal.free t.g) (out :: args :: held)

(* A region allocated between two submits whose work shares a command buffer is
   written by the second. *)
let allocated_between () =
  let t = dev () in
  let held = busy t in
  let first = alloc t 256 and args = alloc t (2 * args_bytes) in
  let f = fill_dispatch t ~args ~at:0 ~out:(gpu first) ~c:3 64 in
  ignore (submit t [| f |]);
  let second = alloc t (1 lsl 20) in
  let f = fill_dispatch t ~args ~at:args_bytes ~out:(gpu second) ~c:4 64 in
  wait t (submit t [| f |]);
  equal bool ~msg:"the first region" true (filled first ~c:3 64);
  equal bool ~msg:"the second region" true (filled second ~c:4 64);
  List.iter (Rig_metal.free t.g) (first :: second :: args :: held)

(* The bumps of the submissions of two domains, each submission [k] bumps of
   every byte of one region. *)
let gen_bumps =
  let bumps = Gen.list ~size:(Gen.int_range 0 20) (Gen.int_range 1 3) in
  Gen.pair bumps bumps

let bumps_across (mine, theirs) =
  let t = dev () in
  let bytes = 4096 in
  let out = alloc t bytes and args = alloc t args_bytes in
  H.write (host out) (String.make bytes '\000');
  set_args args ~at:0 ~out:(gpu out) ~c:bytes;
  let bump =
    S.dispatch ~pipeline:(pipeline t "bump") args ~groups:(bytes / 256)
      ~threads:256
  in
  let submits = List.iter (fun k -> ignore (submit t (Array.make k bump))) in
  cover "both domains submit" (mine <> [] && theirs <> []);
  let other = Domain.spawn (fun () -> submits theirs) in
  submits mine;
  Domain.join other;
  wait t (Rig.submitted t.d);
  let k = List.fold_left ( + ) 0 (mine @ theirs) in
  for i = 0 to bytes - 1 do
    let got = H.get8 (host out + i) in
    if got <> k then failf "byte %d reads %d, not %d" i got k
  done;
  List.iter (Rig_metal.free t.g) [ out; args ]

(* A loss that finds work waiting in the open command buffer drops it, and the
   stop brings the word to the last value. *)
let lost_open () =
  S.with_ @@ fun s ->
  let t = dev_of s in
  ignore (busy t);
  let out = alloc t 256 and args = alloc t args_bytes in
  let last =
    submit t [| fill_dispatch t ~args ~at:0 ~out:(gpu out) ~c:2 64 |] + 1
  in
  raises_match
    (function Rig.Lost _ -> true | _ -> false)
    (fun () -> submit t [| S.failing 7 |]);
  S.close s;
  while Rig.signaled t.d < last do
    Domain.cpu_relax ()
  done

let commits =
  group ~timeout:60. "commits"
    [
      cases
        ~name:
          (strf
             "%d values behind three running command buffers run with no commit")
        "uncommitted" [ 2; 257 ] uncommitted;
      test
        "a region allocated between two submits of one command buffer is \
         written"
        allocated_between;
      prop ~count:20 "every bump of two domains' submissions is counted"
        gen_bumps bumps_across;
      test
        "a loss drops the open command buffer, and the stop brings the word to \
         the last value"
        lost_open;
    ]

(* Indirect command buffers *)

let run_icb t b = wait t (submit t [| S.execute b |])

let chain n =
  let t = dev () in
  let out = alloc t 256 and args = alloc t args_bytes in
  set_args args ~at:0 ~out:(gpu out) ~c:0;
  let step = pipeline t "step" in
  let b = require_ok (icb t args (Array.make n (dispatch step))) in
  run_icb t b;
  equal int n (H.get32 (host out));
  b.release ();
  List.iter (Rig_metal.free t.g) [ out; args ]

let resized () =
  let t = dev () in
  let out = alloc t 256 and args = alloc t args_bytes in
  set_args args ~at:0 ~out:(gpu out) ~c:4;
  let fill = pipeline t "fill" in
  let b = require_ok (icb t args [| dispatch ~threads:(4, 1, 1) fill |]) in
  run_icb t b;
  equal bool true (filled out ~c:4 4);
  equal int 0 (H.get32 (host out + (4 * 4)));
  S.resize b.commands.(0) ~groups:2 ~threads:4;
  run_icb t b;
  equal bool true (filled out ~c:4 8);
  b.release ();
  List.iter (Rig_metal.free t.g) [ out; args ]

let icb_refusals () =
  let t = dev () in
  let args = alloc t args_bytes in
  let fill = pipeline t "fill" in
  let align = (Rig_metal.capability t.g).align in
  let invalid ds = raises_match Exn.invalid_arg (fun () -> icb t args ds) in
  is_error (icb t args [| dispatch ~threads:(1025, 1, 1) fill |]);
  is_error
    (icb t args [| dispatch ~threads:(1 lsl 21, 1 lsl 21, 1 lsl 21) fill |]);
  invalid [| dispatch ~offset:args_bytes fill |];
  if align > 1 then invalid [| dispatch ~offset:(align / 2) fill |];
  invalid [| dispatch ~groups:(0, 1, 1) fill |];
  invalid [| dispatch ~threads:(1, 1, 0) fill |];
  (require_ok (icb t args [||])).release ();
  Rig_metal.free t.g args

let released_twice () =
  let t = dev () in
  let args = alloc t args_bytes in
  let b = require_ok (icb t args [| dispatch (pipeline t "step") |]) in
  b.release ();
  raises_match Exn.invalid_arg b.release;
  Rig_metal.free t.g args

let released_objects () =
  let t = dev () in
  let args = alloc t args_bytes in
  let step = pipeline t "step" in
  let b = require_ok (icb t args [| dispatch step; dispatch step |]) in
  let weaks = Array.map S.weak (Array.append [| b.handle |] b.commands) in
  b.release ();
  Array.iteri
    (fun k w -> equal bool ~msg:(strf "object %d" k) false (S.alive w))
    weaks;
  Rig_metal.free t.g args

let after_unload () =
  let t = dev () in
  let out = alloc t 256 and args = alloc t args_bytes in
  set_args args ~at:0 ~out:(gpu out) ~c:0;
  let i = load t.g in
  let step = require_some (Rig_metal.entry i "step") in
  let b = require_ok (icb t args [| dispatch step; dispatch step |]) in
  Rig_metal.unload t.g i;
  Gc.full_major ();
  run_icb t b;
  equal int 2 (H.get32 (host out));
  b.release ();
  List.iter (Rig_metal.free t.g) [ out; args ]

let stopped_icb = "the device was stopped"

(* An icb call after a stop retains no pipeline: it answers the stop. *)
let after_stop () =
  let g = S.driver () in
  let args = alloc_on g args_bytes in
  let step = require_some (Rig_metal.entry (load g) "step") in
  S.stop_driver g;
  equal (result pass string) ~msg:"after the stop" (Error stopped_icb)
    (Result.map ignore (icb_on g args [| dispatch step |]));
  Rig_metal.free g args

(* An icb call beside a stop: each answers as if made before the stop or after
   it, never with a pipeline released under it. A device of its own per program,
   as its stop ends it. *)
type stop = { mutable stopped : bool }

let icb_model s = if s.stopped then Error stopped_icb else Ok ()

let icb_sys (g, args, step, _) =
  match icb_on g args (Array.make 64 (dispatch step)) with
  | Ok b ->
      b.release ();
      Ok ()
  | Error e -> Error e

(* The device's one stop: a second call returns once the first did. *)
let stop_once (g, args, _, (lock, stopped)) =
  Mutex.protect lock @@ fun () ->
  if not !stopped then begin
    Rig_metal.stop g;
    Rig_metal.free g args;
    stopped := true
  end

let stop_commands =
  let dev = abstract "d" ~release:stop_once in
  [
    command "open"
      (Gen.unit @-> makes dev)
      (fun () -> { stopped = false })
      (fun () ->
        let g = second () in
        let step = require_some (Rig_metal.entry (load g) "step") in
        (g, alloc_on g args_bytes, step, (Mutex.create (), ref false)));
    command "icb" (dev ^-> returns (result unit string)) icb_model icb_sys;
    command "stop" (dev ^-> returns unit) (fun s -> s.stopped <- true) stop_once;
  ]

let icbs =
  group ~timeout:60. "indirect command buffers"
    [
      cases
        ~name:(strf "a chain of %d dispatches runs each after the one before")
        "chain" [ 0; 1; 2; 17; 64 ] chain;
      test "a command runs with the sizes set before its run" resized;
      test "refuses what it cannot record" icb_refusals;
      test "release frees the buffer and its commands" released_objects;
      test "release raises when called twice" released_twice;
      test "runs after its image is unloaded" after_unload;
      test "an icb call after the stop answers the stop" after_stop;
      stateful ~domains:2 ~count:20
        "an icb call beside a stop answers as before or after it" stop_commands;
    ]

(* Bytes *)

(* Bytes print by their length and digest past a line. *)
let octets =
  let pp ppf s =
    let n = String.length s in
    if n <= 48 then Format.fprintf ppf "%S" s
    else
      Format.fprintf ppf "%d bytes, md5 %s" n (Digest.to_hex (Digest.string s))
  in
  Testable.make ~pp ~equal:String.equal

let random_bytes ~seed n =
  let r = Random.State.make [| seed |] in
  String.init n (fun _ -> Char.unsafe_chr (Random.State.bits r land 255))

let bumped s =
  String.map (fun c -> Char.unsafe_chr ((Char.code c + 1) land 255)) s

let host_buffer s =
  let a =
    Bigarray.Array1.create Bigarray.char Bigarray.c_layout (String.length s)
  in
  String.iteri (Bigarray.Array1.set a) s;
  B.of_bigarray a

(* The bytes of [b], copied to the host. *)
let contents b =
  let h = B.create Rig.host (B.length b) in
  B.copy ~src:b ~dst:h;
  let a = B.bigarray Bigarray.char h in
  String.init (Bigarray.Array1.dim a) (Bigarray.Array1.get a)

(* Memory *)

let page = 16384

let shared_both_ways (offset, pages) =
  let t = dev () in
  let n = (pages * page) - offset in
  let p = H.pages ((pages + 1) * page) + offset in
  for i = 0 to n - 1 do
    H.set8 (p + i) (i mod 251)
  done;
  let r = require_some (Rig_metal.map_host t.g p n) in
  equal int (p land lnot (page - 1)) (host r);
  let args = alloc t args_bytes in
  let into = p - host r in
  set_args args ~at:0 ~out:(gpu r + into) ~c:n;
  let bump = pipeline t "bump" in
  let f =
    S.dispatch ~pipeline:bump args ~groups:((n + 255) / 256) ~threads:256
  in
  wait t (submit t [| f |]);
  for i = 0 to n - 1 do
    if H.get8 (p + i) <> (i mod 251) + 1 then
      failf "byte %d reads %d, not %d" i (H.get8 (p + i)) ((i mod 251) + 1)
  done;
  Rig_metal.free t.g r;
  Rig_metal.free t.g args

let aligned_256 n =
  let t = dev () in
  let r = alloc t n in
  equal int 0 (host r mod 256);
  equal int 0 (gpu r mod 256);
  Rig_metal.free t.g r

let misused_regions () =
  let t = dev () in
  let invalid f = raises_match Exn.invalid_arg f in
  let r = alloc t 64 in
  let m = require_some (Rig_metal.map_host t.g (H.pages page) page) in
  Rig_metal.free t.g r;
  invalid (fun () -> Rig_metal.free t.g r);
  Rig_metal.free t.g m;
  invalid (fun () -> Rig_metal.free t.g m);
  invalid (fun () -> Rig_metal.alloc t.g `Device 0);
  invalid (fun () -> Rig_metal.map_host t.g (H.pages page) 0);
  let other = second () in
  Fun.protect ~finally:(fun () -> Rig_metal.stop other) @@ fun () ->
  let o = alloc_on other 64 in
  invalid (fun () -> Rig_metal.free t.g o);
  equal bool false (Rig_metal.peer t.g other);
  equal (option pass) None (Rig_metal.map_peer t.g other o);
  invalid (fun () -> Rig_metal.map_peer t.g t.g o);
  invalid (fun () -> Rig_metal.map_peer t.g other (alloc t 64));
  Rig_metal.free other o;
  invalid (fun () -> Rig_metal.map_peer t.g other o)

let given_back () =
  let t = dev () in
  let r = alloc t 64 in
  let m = require_some (Rig_metal.map_host t.g (H.pages page) page) in
  let wr = S.weak (Rig_metal.handle r) and wm = S.weak (Rig_metal.handle m) in
  wait t (submit t [||]);
  Rig_metal.free t.g r;
  Rig_metal.free t.g m;
  equal bool ~msg:"allocation" false (S.alive wr);
  equal bool ~msg:"mapping" false (S.alive wm)

(* The host addresses host memory a device borrowed: the copy is its own. *)
let copied_into_borrow () =
  let t = dev () in
  let n = 4 * page in
  let h = B.create Rig.host n in
  let b = require_some (B.borrow t.d h) in
  let s = random_bytes ~seed:19 n in
  B.copy ~src:(host_buffer s) ~dst:b;
  equal octets s (contents h)

let memory =
  group ~timeout:60. "memory"
    [
      cases
        ~name:(fun (o, n) ->
          strf "a host range at offset %d over %d pages is shared both ways" o n)
        "map_host"
        [ (0, 1); (1, 1); (2, 2); (4095, 3); (0, 3) ]
        shared_both_ways;
      prop "an allocation starts at a multiple of 256 bytes"
        ~examples:[ 1; 2; 255; 256; 257; 4095; 4096; page; page + 1 ]
        (Gen.int_range 1 (64 lsl 20))
        aligned_256;
      test
        "free and map_peer refuse another device's region or one \
         given back"
        misused_regions;
      test "free releases an allocation's or a mapping's buffer" given_back;
      test "the host copies into a borrow of host memory" copied_into_borrow;
    ]

(* Files

   Files of the disk, written under this test's directory in _build and removed
   as the tests end. *)

let files = "files"

let clear_files () =
  if not (Sys.file_exists files) then Sys.mkdir files 0o755
  else
    Array.iter
      (fun f -> Sys.remove (Filename.concat files f))
      (Sys.readdir files)

let file_names = Atomic.make 0

(* [with_path f] is [f p], [p] a path under [files] that names nothing, removed
   after [f]. *)
let with_path f =
  let n = Atomic.fetch_and_add file_names 1 in
  let p = Filename.concat files (string_of_int n) in
  let remove () = if Sys.file_exists p then Sys.remove p in
  Fun.protect ~finally:remove (fun () -> f p)

let write_file p s = Out_channel.with_open_bin p (fun oc -> output_string oc s)
let read_file p = In_channel.with_open_bin p In_channel.input_all
let of_file p = require_ok ~pp:Format.pp_print_string (Rig_disk.of_file p)

let create_file p n =
  require_ok ~pp:Format.pp_print_string (Rig_disk.create_file p n)

(* Work on [t]'s device that adds 1 to each byte of [b], a buffer on it that the
   run writes; it returns once the work is done. *)
let bump t b =
  let n = B.length b and args = alloc t args_bytes in
  set_args args ~at:0 ~out:(B.address b) ~c:n;
  let f =
    S.dispatch ~pipeline:(pipeline t "bump") args
      ~groups:((n + 255) / 256)
      ~threads:256
  in
  let s = Rig.Submission.make ~reads:0 ~writes:1 t.d [| S.part f |] in
  ignore (Rig.submit s ~reads:[||] ~writes:[| b |] ~waits:[||]);
  B.wait b Read;
  Rig_metal.free t.g args

let buffer =
  let pp ppf b =
    Format.fprintf ppf "%d bytes on %a" (B.length b) Rig.pp (B.device b)
  in
  Testable.make ~pp ~equal:( == )

(* The bytes of a file from byte [at_file] copy into the device's memory from
   byte [at] and back into a new file. The host addresses the device's memory:
   the copies are the host's, with no value on the device's timeline. *)
let file_round_trip (n, at_file, at) =
  let t = dev () in
  let v = Rig.submitted t.d in
  cover "no bytes" (n = 0);
  cover "one byte" (n = 1);
  cover "a page or more" (n >= page);
  cover "a file range off a page" (at_file mod page <> 0);
  with_path @@ fun src ->
  with_path @@ fun dst ->
  let into s =
    write_file src (String.make at_file 'x' ^ s);
    let file = B.view (of_file src) ~first:at_file ~length:n in
    let mem = B.view (B.create t.d (at + n)) ~first:at ~length:n in
    B.copy ~src:file ~dst:mem;
    mem
  in
  let out mem =
    B.copy ~src:mem ~dst:(create_file dst n);
    read_file dst
  in
  Law.round_trip octets buffer into out (random_bytes ~seed:n n);
  equal int ~msg:"values submitted" v (Rig.submitted t.d)

(* A length is no bytes, one byte, a page's edges, or up to a MiB. Each of the
   first three is drawn in about one case in five, so a property's 100 cases
   cover each whatever its seed (a miss less than once in a billion seeds). *)
let gen_file_range =
  let lengths =
    Gen.frequency
      [
        (1, Gen.of_list [ 0 ]);
        (1, Gen.of_list [ 1 ]);
        (1, Gen.of_list [ page - 1; page; page + 1 ]);
        (2, Gen.int_range 0 (1 lsl 20));
      ]
  in
  Gen.triple lengths (Gen.int_range 0 (2 * page)) (Gen.int_range 0 256)

(* A file [of_file] opened admits only reads: the device borrows its pages, and
   work that writes them is refused, so the file and its copies stay its own
   bytes. *)
let opened_file_borrow () =
  let t = dev () in
  with_path @@ fun path ->
  let n = (1 lsl 20) + 4099 and at = 4 in
  let s = random_bytes ~seed:13 n in
  write_file path s;
  let file = B.view (of_file path) ~first:at ~length:(n - at) in
  let b = require_some (B.borrow t.d file) in
  equal (pair bool string)
    (true, Rig.name t.d)
    (B.is_borrowed b, Rig.name (B.device b));
  raises_match Exn.invalid_arg (fun () -> bump t b);
  let s' = String.sub s at (n - at) in
  equal octets ~msg:"a copy of the file" s' (contents file);
  equal octets ~msg:"the file" s (read_file path)

(* The pages of a file [create_file] made are the file. *)
let created_file_borrow () =
  let t = dev () in
  with_path @@ fun path ->
  let n = (1 lsl 20) + 4099 in
  let s = random_bytes ~seed:17 n in
  let file = create_file path n in
  let b = require_some (B.borrow t.d file) in
  B.copy ~src:(host_buffer s) ~dst:file;
  bump t b;
  equal octets ~msg:"a copy of the file" (bumped s) (contents file);
  equal octets ~msg:"the file" (bumped s) (read_file path)

let file_tests =
  group ~timeout:60. "files"
    [
      prop
        ~examples:[ ((3 lsl 20) + 12345, 5, 16) ]
        "a file's bytes copy into the device's memory and back, with no work \
         on its timeline"
        gen_file_range file_round_trip;
      test
        "a borrow of an opened file is its pages, which the device's work \
         never writes"
        opened_file_borrow;
      test
        "a borrow of a created file is its pages: the device reads a copy's \
         writes, and its own reach the file"
        created_file_borrow;
    ]

(* Images *)

let not_metallib () =
  let t = dev () in
  is_error (Rig_metal.image t.g "not a metallib")

let no_kernel () =
  let t = dev () in
  match Rig_metal.image t.g (S.fixture ~dir:"fixtures" "vertex") with
  | Ok _ -> failf "a vertex function loaded as an image"
  | Error why ->
      equal string "the function \"position\" is no compute kernel" why

(* An image loads whatever its kernels need; the entry of one that needs more
   than the GPU has raises with Metal's reason, each time it is asked, and the
   others' entries work. *)
let beyond_limits () =
  let t = dev () in
  let i =
    match
      require_ok (Rig_metal.image t.g (S.fixture ~dir:"fixtures" "threadgroup"))
    with
    | `Loaded i -> i
    | `Place _ -> failf "Metal asked to place its code"
  in
  ignore (require_some (Rig_metal.entry i "small"));
  let refused () =
    raises_match
      (Exn.invalid_arg
         ~substring:"Rig_metal.entry: Metal makes no pipeline of \"wide\": ")
      (fun () -> Rig_metal.entry i "wide")
  in
  refused ();
  refused ();
  Rig_metal.unload t.g i

let entries () =
  let t = dev () in
  List.iter
    (fun f -> ignore (require_some ~msg:f (Rig_metal.entry t.fill f)))
    [ "fill"; "step"; "spin"; "bump" ];
  equal (option int) None (Rig_metal.entry t.fill "absent")

let unloaded_twice () =
  let t = dev () in
  let i = load t.g in
  let other = second () in
  Fun.protect ~finally:(fun () -> Rig_metal.stop other) @@ fun () ->
  raises_match Exn.invalid_arg (fun () -> Rig_metal.unload other i);
  Rig_metal.unload t.g i;
  raises_match Exn.invalid_arg (fun () -> Rig_metal.unload t.g i);
  raises_match Exn.invalid_arg (fun () -> Rig_metal.entry i "fill")

(* Images entered and unloaded from two domains: whatever the order, an
   image's first [unload] returns and every later one raises, an [entry]
   after the unload raises, and every [entry] of one function answers the
   address the first answered. *)

type loaded = { mutable loaded : bool }

(* An image, with the first address any domain got for each function. *)
type held = { i : Rig_metal.image; first : (string, int) Hashtbl.t; m : Mutex.t }

let unload_model m =
  if not m.loaded then invalid_arg "unloaded";
  m.loaded <- false

let unload_system h = Rig_metal.unload (dev ()).g h.i

let entry_model m _ =
  if not m.loaded then invalid_arg "unloaded";
  true

let entry_system h f =
  match Rig_metal.entry h.i f with
  | None -> false
  | Some p -> (
      Mutex.protect h.m @@ fun () ->
      match Hashtbl.find_opt h.first f with
      | Some q -> p = q
      | None ->
          Hashtbl.add h.first f p;
          true)

let loaded_image =
  abstract "i" ~release:(fun h ->
      try unload_system h with Invalid_argument _ -> ())

let functions =
  Gen.of_list ~pp:Format.pp_print_string [ "fill"; "step"; "spin"; "bump" ]

let unload_commands =
  [
    command "image"
      (Gen.unit @-> makes loaded_image)
      (fun () -> { loaded = true })
      (fun () ->
        { i = load (dev ()).g; first = Hashtbl.create 4; m = Mutex.create () });
    command "entry"
      (loaded_image ^-> functions @-> returns bool)
      entry_model entry_system;
    command "unload" (loaded_image ^-> returns unit) unload_model unload_system;
  ]

let unloaded_releases () =
  let t = dev () in
  let weaks =
    List.init 60 (fun _ ->
        let i = load t.g in
        let w =
          S.weak (Nativeint.of_int (require_some (Rig_metal.entry i "fill")))
        in
        Rig_metal.unload t.g i;
        w)
  in
  Gc.full_major ();
  List.iteri
    (fun k w -> equal bool ~msg:(strf "image %d" k) false (S.alive w))
    weaks;
  several_fills ()

let images =
  group ~timeout:60. "images"
    [
      test "bytes that are no metallib are an error" not_metallib;
      test "a function that is no compute kernel is an error" no_kernel;
      test "an entry beyond the GPU's limits raises, each time" beyond_limits;
      test "each function of the image has an entry" entries;
      test "unload and entry refuse an unloaded image or another device's"
        unloaded_twice;
      stateful ~domains:2 ~count:30
        "an image entered and unloaded from two domains: one pipeline a \
         function, one unload"
        unload_commands;
      test "unloaded images release their pipelines" unloaded_releases;
    ]

(* Timeline and loss *)

let sleep_seen () =
  let t = dev () in
  let v = submit t [||] in
  wait t v;
  Rig_metal.sleep t.g ~seen:(v - 1) ~still_ms:600_000

let stopped_idle () =
  S.with_ @@ fun t ->
  let v = S.submit t [||] in
  S.wait t v;
  S.close t;
  equal int v (Rig.signaled t.d)

(* A loss that finds work running stops the device, the work runs to its end,
   and its indirect command buffer is released after. *)
let stopped_running () =
  S.with_ @@ fun s ->
  let t = dev_of s in
  let out = alloc t 256 and args = alloc t args_bytes in
  set_args args ~at:0 ~out:(gpu out) ~c:5_000_000;
  let spin = pipeline t "spin" in
  let b = require_ok (icb t args [| dispatch spin |]) in
  let w = S.weak b.handle in
  ignore (submit t [| S.execute b |]);
  raises_match
    (function Rig.Lost _ -> true | _ -> false)
    (fun () -> submit t [| S.failing 7 |]);
  S.close s;
  while H.get32 (host out) = 0 do
    Domain.cpu_relax ()
  done;
  b.release ();
  await_release w

(* A stopped device's word frees once, after the stop. *)
let word_after_stop () =
  let g = S.driver () in
  S.stop_driver g;
  Rig_metal.free g (Rig_metal.word g);
  raises_match Exn.invalid_arg (fun () -> Rig_metal.free g (Rig_metal.word g))

(* Images still loaded when the device stops stay loaded: their unload after
   the stop releases them. *)
let unload_after_stop () =
  let g = S.driver () in
  let i = load g in
  let weak f = S.weak (Nativeint.of_int (require_some (Rig_metal.entry i f))) in
  let weaks = [ weak "fill"; weak "step"; weak "bump" ] in
  S.stop_driver g;
  List.iteri
    (fun k w ->
      equal bool ~msg:(strf "pipeline %d after the stop" k) true (S.alive w))
    weaks;
  Rig_metal.unload g i;
  List.iteri
    (fun k w ->
      equal bool ~msg:(strf "pipeline %d after the unload" k) false (S.alive w))
    weaks

let timeline =
  group ~timeout:60. "timeline"
    [
      test "sleep returns at once when the word differs from seen" sleep_seen;
      test "a close of an idle device leaves the word at the last value"
        stopped_idle;
      test
        "a loss while work runs: the work runs to its end, and its indirect \
         command buffer is released after"
        stopped_running;
      test "an image a stop left loaded is released by its unload"
        unload_after_stop;
      test "a stopped device's word frees once" word_after_stop;
    ]

(* Opening and misuse *)

let two_devices () =
  let b = second () in
  Fun.protect ~finally:(fun () -> Rig_metal.stop b) @@ fun () ->
  S.with_ @@ fun a ->
  S.wait a (S.submit a [||]);
  equal int 1 (Rig_metal.signaled a.g);
  equal int 0 (Rig_metal.signaled b)

let apple_align () =
  let t = dev () in
  if not (String.starts_with ~prefix:"Apple" (Rig_metal.arch t.g)) then
    skip ~reason:"the GPU is of a Mac family" ();
  equal int 4 (Rig_metal.capability t.g).align

(* Work the room refuses: words, and a fill that declares room. *)
let refused_work () =
  let t = dev () in
  let refused work =
    let part = { Rig.Submission.queue = "COMPUTE:0"; after = [||]; work } in
    raises_match Exn.invalid_arg (fun () -> submit_parts t [| part |])
  in
  let fill = (S.part (S.failing 0)).work in
  let declaring ~units ~bytes =
    match fill with
    | Fill f ->
        Rig.Submission.Fill { f with ring_units = units; segment_bytes = bytes }
    | w -> w
  in
  refused (Words (Rig.Buffer.create Rig.host 4));
  refused (declaring ~units:1 ~bytes:0);
  refused (declaring ~units:0 ~bytes:64);
  let v = submit t [||] in
  wait t v;
  equal int v (Rig_metal.signaled t.g)

let opening =
  group ~timeout:60. "opening"
    [
      test "a device other than 0 is an error" (fun () ->
          is_error (Rig_metal.open_ 1));
      test "a negative device is misuse" (fun () ->
          raises_match Exn.invalid_arg (fun () -> Rig_metal.open_ (-1));
          raises_match Exn.invalid_arg (fun () -> Rig_metal.device_name (-1)));
      test "device names" (fun () ->
          equal (list string)
            [ "METAL"; "METAL:1"; "METAL:7" ]
            (List.map Rig_metal.device_name [ 0; 1; 7 ]));
      test "two opens are two devices, each with its own word" two_devices;
      test "an Apple GPU aligns arguments to 4 bytes" apple_align;
      test
        "the room refuses words and fills that declare room, and the device \
         runs on"
        refused_work;
      test "off macOS no device opens" (fun () ->
          if S.macos then skip ~reason:"macOS" ();
          equal int 0 (Rig_metal.count ());
          equal (result pass string) (Error "Metal exists on macOS only")
            (Rig_metal.open_ 0));
    ]

let () =
  S.hold ();
  clear_files ();
  exit
    (run "rig_metal"
       [
         ring_tests;
         work;
         commits;
         icbs;
         memory;
         file_tests;
         images;
         timeline;
         opening;
       ])
