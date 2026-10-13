(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module S = Rig_metal_support
module H = Rig_gpu_support.Host
module B = Rig.Buffer

let strf = Printf.sprintf

(* The function an entry answered, which the test needs. *)
let found : Rig_edge.entry_answer -> Rig_edge.entry = function
  | Entry e -> e
  | Absent -> fail "no such function"
  | No_memory n -> failf "no memory for %d bytes" n

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
  | Rig_edge.Loaded i -> i
  | Place (n, _) -> failf "the device asked to place %d bytes of code" n

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

(* A second device of the GPU, beside the fixture's: Metal opens a GPU as often
   as asked. The test stops it. *)
let second () =
  if not (S.present ()) then skip ~reason:"the machine has no METAL GPU rig drives" ();
  require_ok (Rig_gpu_support.opened (Rig_metal.gpu 0))

let pipeline t f = (found (Rig_metal.entry t.fill f)).code

let submit t fills =
  let s, run, buffers =
    Rig_gpu_support.prepare t.d (Array.to_list (Array.map S.part fills))
  in
  Rig.submit s ~run ~buffers

let wait _ p = Rig.Point.wait p
let alloc_on g n = require_some (Rig_metal.alloc g Rig_edge.Device n)
let alloc t n = alloc_on t.g n
let host r = require_some (Rig_metal.locate r).host
let gpu r = require_some (Rig_metal.locate r).address
let handle r = (Rig_metal.locate r).handle

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

let icb_on g args ds = (Rig_metal.capability g).icb (handle args) ds
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
      (dispatch ~offset:args_bytes
         ~groups:(bytes / 256, 1, 1)
         ~threads:(256, 1, 1) bump)
  in
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
  List.iter (Rig_metal.free t.g) [ out; args ]

(* Waits until [f ()] holds, for at most 10 s of the host's monotonic clock,
   then fails with [what ()]. *)
let await what f =
  let t0 = Rig.Profile.now () in
  while not (f ()) do
    if Rig.Profile.now () - t0 > 10_000_000_000 then
      failf "%s after 10 s" (what ());
    Domain.cpu_relax ()
  done

(* Waits for [d]'s word to show [v]. *)
let await_word d v =
  await
    (fun () -> strf "the word shows %d, not %d," (Rig.signaled d) v)
    (fun () -> Rig.signaled d >= v)

(* Metal hands a command buffer to its completion handler before the word moves
   and releases it, with the objects it holds, once the handler returned: no
   signal marks that, so a test waits for the weak reference [w] to empty,
   collecting and draining [d] meanwhile. *)
let await_release d w =
  await
    (fun () -> "the weak reference still holds")
    (fun () ->
      Gc.full_major ();
      Rig.free_cache d;
      not (S.alive w))

let released_buffer () =
  let t = dev () in
  let f, w = S.watching () in
  wait t (submit t [| f |]);
  await_release t.d w

let failing_fill () =
  S.with_ @@ fun s ->
  let t = dev_of s in
  let out = alloc t 256 and args = alloc t args_bytes in
  let ok = fill_dispatch t ~args ~at:0 ~out:(gpu out) ~c:2 64 in
  let reached = Rig.Point.value (submit t [| ok |]) in
  let why = "running a fill: it returned 7" in
  raises_match
    (function Rig.Lost (_, w) -> String.equal w why | _ -> false)
    (fun () -> submit t [| ok; S.failing 7 |]);
  raises (Invalid_argument "Rig_metal.sleep: the device stopped") (fun () ->
      Rig_metal.sleep t.g ~seen:reached ~still_ms:10);
  S.close s;
  await_word t.d (reached + 1)

let work =
  group ~timeout:60. "work"
    [
      prop ~count:50
        "a signaled value's submissions and every earlier one are written"
        (Gen.list ~size:(Gen.int_range 1 50) gen_submission)
        prefix_completion;
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
    last := Rig.Point.value (submit t [| f |])
  done;
  await_word t.d !last;
  equal int n (H.get32 (host out));
  List.iter (Rig_metal.free t.g) (out :: args :: held)

(* The device commits its work on its own: after [n] submissions of one dispatch
   each and nothing more, no wait and no commit through rig, the word reaches
   the last value within 10 s. *)
let progress n =
  let t = dev () in
  let out = alloc t 256 and args = alloc t args_bytes in
  set_args args ~at:0 ~out:(gpu out) ~c:0;
  let f = S.dispatch ~pipeline:(pipeline t "step") args ~groups:1 ~threads:1 in
  let last = ref 0 in
  for _ = 1 to n do
    last := Rig.Point.value (submit t [| f |])
  done;
  await_word t.d !last;
  List.iter (Rig_metal.free t.g) [ out; args ]

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
  let submits = List.map (fun k -> submit t (Array.make k bump)) in
  cover "both domains submit" (mine <> [] && theirs <> []);
  let other = Domain.spawn (fun () -> submits theirs) in
  let points = submits mine in
  List.iter Rig.Point.wait (points @ Domain.join other);
  let k = List.fold_left ( + ) 0 (mine @ theirs) in
  for i = 0 to bytes - 1 do
    let got = H.get8 (host out + i) in
    if got <> k then failf "byte %d reads %d, not %d" i got k
  done;
  List.iter (Rig_metal.free t.g) [ out; args ]

(* Rounds of [fill.metal]'s [gate] far more than a loss, its stop and a close
   take: about 3 s on the M1 Max. The gate opens long before, and its write
   says whether it saw that. *)
let gate_rounds = 1 lsl 24

(* A device of the GPU beside the test's, under a name of its own, whose queue
   runs beside the test device's. *)
let beside () =
  let g = ref None in
  let make ~word =
    let r = (Rig_metal.gpu 0).make ~word in
    Result.iter (fun x -> g := Some x) r;
    r
  in
  let opener = { (Rig_metal.gpu 0) with name = "METAL:beside"; make } in
  let d = require_ok (Rig.open_ opener) in
  dev_of { S.d; g = Option.get !g }

(* A command buffer in flight when a loss stops the device completes and
   writes its memory, and the word reaches the last value: the buffer keeps
   the command queue the stop releases. The buffer runs [gate], which a kernel
   of a device beside opens once the close returned, so it is in flight at the
   stop. *)
let stop_in_flight () =
  S.with_ @@ fun s ->
  let t = dev_of s and o = beside () in
  let page = H.pages H.page in
  H.set32 page 0;
  H.set32 (page + 4) 0;
  let flag = require_some (Rig_metal.map_host t.g page H.page) in
  let flag' = require_some (Rig_metal.map_host o.g page H.page) in
  let args = alloc t args_bytes and args' = alloc o args_bytes in
  set_args args ~at:0 ~out:(gpu flag) ~c:gate_rounds;
  set_args args' ~at:0 ~out:(gpu flag') ~c:1;
  let gate =
    S.dispatch ~pipeline:(pipeline t "gate") args ~groups:1 ~threads:1
  in
  let unlock =
    S.dispatch ~pipeline:(pipeline o "unlock") args' ~groups:1 ~threads:1
  in
  let opened = ref false in
  let open_gate () =
    if not !opened then begin
      opened := true;
      wait o (submit o [| unlock |])
    end
  in
  Fun.protect
    ~finally:(fun () ->
      open_gate ();
      List.iter (Rig_metal.free o.g) [ flag'; args' ];
      Rig.close o.d)
  @@ fun () ->
  let gated = Rig.Point.value (submit t [| gate |]) in
  raises_match
    (function Rig.Lost _ -> true | _ -> false)
    (fun () -> submit t [| S.failing 7 |]);
  S.close s;
  equal ~msg:"the gated work, at the stop" bool false
    (Rig.signaled t.d >= gated);
  open_gate ();
  await_word t.d (Rig.submitted t.d);
  equal ~msg:"the gate's write: it saw the gate open" int 1 (H.get32 page);
  List.iter (Rig_metal.free t.g) [ flag; args ]

(* A loss that finds work waiting in the open command buffer drops it, and the
   stop brings the word to the last value. *)
let lost_open () =
  S.with_ @@ fun s ->
  let t = dev_of s in
  ignore (busy t);
  let out = alloc t 256 and args = alloc t args_bytes in
  let last =
    Rig.Point.value (submit t [| fill_dispatch t ~args ~at:0 ~out:(gpu out) ~c:2 64 |]) + 1
  in
  raises_match
    (function Rig.Lost _ -> true | _ -> false)
    (fun () -> submit t [| S.failing 7 |]);
  S.close s;
  await_word t.d last

let commits =
  group ~timeout:60. "commits"
    [
      cases
        ~name:
          (strf
             "%d values behind three running command buffers run with no commit")
        "uncommitted" [ 2; 257 ] uncommitted;
      cases
        ~name:(strf "%d values reach the word with no later submission")
        "progress"
        [ 1; 2; 3; 4; 255; 256; 257; 1024 ]
        progress;
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
      test
        "a command buffer in flight at a loss's stop completes and writes its \
         memory"
        stop_in_flight;
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
  List.iter (Rig_metal.free t.g) [ out; args ]

let prepared_icb () =
  let t = dev () in
  let out = alloc t 256 and args = alloc t args_bytes in
  set_args args ~at:0 ~out:(gpu out) ~c:0;
  let step = pipeline t "step" in
  let weak = ref 0n in
  let[@inline never] made () =
    let b = require_ok (icb t args [| dispatch step; dispatch step |]) in
    weak := S.weak b.handle;
    Rig_gpu_support.prepare t.d [ S.part (S.execute b) ]
  in
  let s, run, buffers = made () in
  (* A kept value's release and the ICB's finalizer each need a collection. *)
  for round = 1 to 4 do
    Gc.full_major ();
    ignore (Rig.Buffer.create t.d 0);
    equal bool
      ~msg:(strf "the submission keeps its ICB after collection %d" round)
      true (S.alive !weak)
  done;
  Rig.Point.wait (Rig.submit s ~run ~buffers);
  equal int ~msg:"both commands ran" 2 (H.get32 (host out));
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
  invalid [| dispatch ~offset:(-align) fill |];
  if align > 1 then invalid [| dispatch ~offset:(align / 2) fill |];
  invalid [| dispatch ~groups:(0, 1, 1) fill |];
  invalid [| dispatch ~threads:(1, 1, 0) fill |];
  is_ok (icb t args [||]);
  Rig_metal.free t.g args

(* A pipeline that no entry of a loaded image gave: an address of nothing, and
   one its image's unload released. *)
let foreign_pipelines () =
  let t = dev () in
  let args = alloc t args_bytes in
  let i = load t.g in
  let step = (found (Rig_metal.entry i "step")).code in
  Rig_metal.unload t.g i;
  let invalid p =
    raises_match Exn.invalid_arg (fun () -> icb t args [| dispatch p |])
  in
  invalid 1;
  invalid step;
  Rig_metal.free t.g args

(* An argument buffer that is no live allocation or mapping of the device: an
   address of nothing, and one its free released. *)
let foreign_buffers () =
  let t = dev () in
  let step = pipeline t "step" in
  let freed = alloc t args_bytes in
  Rig_metal.free t.g freed;
  let invalid handle =
    raises_match Exn.invalid_arg (fun () ->
        (Rig_metal.capability t.g).icb handle [| dispatch step |])
  in
  invalid 1n;
  invalid (Rig_metal.locate freed).handle

(* The objects of an icb live while the value is reachable, and the driver
   releases them once it is not. *)
let released_objects () =
  let t = dev () in
  let args = alloc t args_bytes in
  let step = pipeline t "step" in
  let[@inline never] made () =
    let b = require_ok (icb t args [| dispatch step; dispatch step |]) in
    (b, Array.map S.weak (Array.append [| b.handle |] b.commands))
  in
  let b, weaks = made () in
  Gc.full_major ();
  Array.iteri
    (fun k w ->
      equal bool ~msg:(strf "object %d, reachable" k) true (S.alive w))
    weaks;
  ignore (Sys.opaque_identity b);
  Gc.full_major ();
  Array.iteri
    (fun k w ->
      equal bool ~msg:(strf "object %d, unreachable" k) false (S.alive w))
    weaks;
  Rig_metal.free t.g args

let released_after_stop () =
  let g = S.driver () in
  let args = alloc_on g args_bytes in
  let i = load g in
  let step = (found (Rig_metal.entry i "step")).code in
  let[@inline never] made () =
    let b = require_ok (icb_on g args [| dispatch step |]) in
    let objects =
      Array.append [| b.handle; Nativeint.of_int step |] b.commands
    in
    (b, Array.map S.weak objects)
  in
  let b, weaks = made () in
  Rig_metal.unload g i;
  S.stop_driver g;
  Gc.full_major ();
  Array.iteri
    (fun k w ->
      equal bool
        ~msg:(strf "object %d, reachable after stop" k)
        true (S.alive w))
    weaks;
  ignore (Sys.opaque_identity b);
  for round = 1 to 4 do
    Gc.full_major ();
    Array.iteri
      (fun k w ->
        equal bool
          ~msg:(strf "object %d, unreachable after collection %d" k round)
          false (S.alive w))
      weaks
  done;
  Rig_metal.free g args

let after_unload () =
  let t = dev () in
  let out = alloc t 256 and args = alloc t args_bytes in
  set_args args ~at:0 ~out:(gpu out) ~c:0;
  let i = load t.g in
  let step = (found (Rig_metal.entry i "step")).code in
  let b = require_ok (icb t args [| dispatch step; dispatch step |]) in
  Rig_metal.unload t.g i;
  Gc.full_major ();
  run_icb t b;
  equal int 2 (H.get32 (host out));
  List.iter (Rig_metal.free t.g) [ out; args ]

let stopped_icb = "the device was stopped"

(* An icb call after a stop retains no pipeline: it answers the stop. *)
let after_stop () =
  let g = S.driver () in
  let args = alloc_on g args_bytes in
  let step = (found (Rig_metal.entry (load g) "step")).code in
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
  | Ok _ -> Ok ()
  | Error e -> Error e

(* The device's one stop: a second call returns once the first did. *)
let stop_once (g, args, _, (lock, stopped)) =
  Mutex.protect lock @@ fun () ->
  if not !stopped then begin
    Rig_metal.stop g ~fault:None;
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
        let step = (found (Rig_metal.entry (load g) "step")).code in
        (g, alloc_on g args_bytes, step, (Mutex.create (), ref false)));
    command "icb" (dev ^-> returns (result unit string)) icb_model icb_sys;
    command "stop" (dev ^-> returns unit) (fun s -> s.stopped <- true) stop_once;
  ]

(* An icb call beside writes to its dispatch array, each slot holding the loaded
   pipeline or one no image gave: the call checks and records the dispatches one
   read gave, so it answers as before or after each write, and never records a
   pipeline it did not check. *)
let slots = 4
let written_icb valid = Array.for_all Fun.id valid

let written_sys (t, args, _, ds) =
  match icb t args ds with
  | Ok _ -> true
  | exception Invalid_argument _ -> false
  | Error why -> failf "the icb call answered %S" why

let write_commands =
  let held =
    abstract "ds" ~release:(fun (t, args, _, _) -> Rig_metal.free t.g args)
  in
  [
    command "dispatches"
      (Gen.unit @-> makes held)
      (fun () -> Array.make slots true)
      (fun () ->
        let t = dev () in
        let step = pipeline t "step" in
        (t, alloc t args_bytes, step, Array.make slots (dispatch step)));
    command "write"
      (held ^-> Gen.int_range 0 (slots - 1) @-> Gen.bool @-> returns unit)
      (fun valid k b -> valid.(k) <- b)
      (fun (_, _, step, ds) k b -> ds.(k) <- dispatch (if b then step else 1));
    command "icb" (held ^-> returns bool) written_icb written_sys;
  ]

let icbs =
  group ~timeout:60. "indirect command buffers"
    [
      cases
        ~name:(strf "a chain of %d dispatches runs each after the one before")
        "chain" [ 0; 1; 2; 17; 64 ] chain;
      test "a command runs with the sizes set before its run" resized;
      test "refuses what it cannot record" icb_refusals;
      test "refuses a pipeline no loaded image gave" foreign_pipelines;
      test "refuses a buffer that is no live region of the device"
        foreign_buffers;
      test "a prepared fill keeps its indirect command buffer" prepared_icb;
      test "an icb's buffer and commands live until it is unreachable"
        released_objects;
      test "a stopped device's icb releases its buffer, commands and pipelines"
        released_after_stop;
      test "runs after its image is unloaded" after_unload;
      test "an icb call after the stop answers the stop" after_stop;
      stateful ~domains:2 ~count:20
        "an icb call beside a stop answers as before or after it" stop_commands;
      stateful ~domains:2 ~count:50
        "an icb call beside writes to its dispatches answers as before or \
         after each"
        write_commands;
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
  B.read h Bigarray.char (fun a ->
      String.init (Bigarray.Array1.dim a) (Bigarray.Array1.get a))

(* Launches

   What Metal adds to the launch laws of the conformance suite: threadgroup
   memory, launches beside fills in one encoder, the most parameter bytes, and
   the limits the room refuses. Kernels of the fixture [launch] run as parts
   through rig, and each case compares what they wrote with the host's own
   computation of the kernel (launch.metal). *)

module Sub = Rig.Submission
module Run = Rig.Submission.Run

let launch_image =
  Rig_gpu_support.loader (fun () -> S.fixture ~dir:"fixtures" "launch")

let launch image kernel ~params refs =
  Sub.launch ~refs:(Array.of_list refs) image kernel ~params

(* The unsigned 32-bit words of [b]. *)
let words b =
  let s = contents b in
  Array.init
    (String.length s / 4)
    (fun i -> Int32.to_int (String.get_int32_le s (4 * i)) land 0xffff_ffff)

(* Submits [s] with [run] over [buffers], and waits for its writes. *)
let run_wait s run buffers =
  ignore (Rig.submit s ~run ~buffers);
  Array.iter (fun b -> B.wait b Read) buffers

let zeroed d n =
  let b = B.create d n in
  B.copy ~src:(B.of_string (String.make n '\000')) ~dst:b;
  b

(* A grid: its groups and the threads of a group, along x, y and z. *)
type grid = { groups : int * int * int; threads : int * int * int }

let pp_grid ppf { groups = gx, gy, gz; threads = tx, ty, tz } =
  Format.fprintf ppf "%dx%dx%d groups of %dx%dx%d threads" gx gy gz tx ty tz

let count (x, y, z) = x * y * z

(* Up to 64 groups of up to 256 threads, the most [neighbours]' threadgroup
   memory holds at 1 KiB. A group of one thread and one of 256 each come in one
   draw in six. *)
let gen_grid =
  let triple n =
    let axis = Gen.int_range 1 n in
    Gen.triple axis axis axis
  in
  let threads =
    Gen.frequency
      [
        (1, Gen.of_list [ (1, 1, 1) ]);
        (1, Gen.of_list [ (256, 1, 1); (16, 16, 1); (4, 8, 8) ]);
        ( 4,
          Gen.map
            (fun (tx, ty, tz) ->
              (tx, ty, Int.max 1 (Int.min tz (256 / (tx * ty)))))
            (triple 16) );
      ]
  in
  Gen.map
    (fun (groups, threads) -> { groups; threads })
    (Gen.pair (triple 4) threads)
  |> Gen.with_pp pp_grid

let set_grid run b { groups = gx, gy, gz; threads = tx, ty, tz } =
  Run.groups run b gx gy gz;
  Run.threads run b tx ty tz

let one = { groups = (1, 1, 1); threads = (1, 1, 1) }

(* [neighbours] reads, through threadgroup memory of one word per thread, the
   index of the thread after each in its group. *)
let shared_law grid =
  let t = dev () in
  let image = launch_image t.d in
  let per = count grid.threads in
  let n = count grid.groups * per in
  cover "one thread a group" (per = 1);
  cover "256 threads a group" (per = 256);
  let out = zeroed t.d (4 * n) in
  let s =
    Sub.make t.d
      [| launch image "neighbours" ~params:16 [ Sub.write 0 ~at:0 ] |]
  in
  let run = Run.make () in
  let k = Sub.block s 0 in
  set_grid run k grid;
  Run.shared run k (4 * per);
  Run.int64 run k 0 0;
  Run.int32 run k 8 7;
  run_wait s run [| out |];
  equal (array int)
    (Array.init n (fun i -> 7 + (((i mod per) + 1) mod per)))
    (words out)

(* A chain of parts that each add 1 to one word: launches of [step], fills that
   dispatch the fixture [fill]'s [step] directly, and fills that run it from an
   indirect command buffer. The count holds only if each part runs after the one
   before, whatever state the part before left in the encoder. *)
let gen_chain =
  Gen.list ~size:(Gen.int_range 1 12) (Gen.of_list [ `Launch; `Direct; `Icb ])

let chained parts =
  let t = dev () in
  let image = launch_image t.d in
  let out = zeroed t.d 4 in
  let args = alloc t args_bytes in
  let step = pipeline t "step" in
  let direct = S.part (S.dispatch ~pipeline:step args ~groups:1 ~threads:1) in
  let b = require_ok (icb t args [| dispatch step |]) in
  let icb_part = S.part (S.execute b) in
  let launch_part =
    {
      Rig_gpu_support.part =
        (fun base -> launch image "step" ~params:8 [ Sub.write base ~at:0 ]);
      buffers = [| out |];
      fixed = [];
      keep = Kept ();
      init = (fun s run i -> set_grid run (Sub.block s i) one);
    }
  in
  let pairs =
    List.combine (List.rev (List.tl (List.rev parts))) (List.tl parts)
  in
  cover "a launch after a fill"
    (List.exists (fun (p, q) -> p <> `Launch && q = `Launch) pairs);
  cover "a fill after a launch"
    (List.exists (fun (p, q) -> p = `Launch && q <> `Launch) pairs);
  let part = function
    | `Launch -> launch_part
    | `Direct -> direct
    | `Icb -> icb_part
  in
  let s, run, buffers =
    Rig_gpu_support.prepare
      ~fixed:[ (out, Rig.Read_write) ]
      t.d (List.map part parts)
  in
  set_args args ~at:0 ~out:(Sub.address s out) ~c:0;
  run_wait s run buffers;
  equal int (List.length parts) (words out).(0);
  Rig_metal.free t.g args

(* A launch of no parameters binds none, and runs. *)
let no_params () =
  let t = dev () in
  let image = launch_image t.d in
  let s = Sub.make t.d [| launch image "empty" ~params:0 [] |] in
  let run = Run.make () in
  set_grid run (Sub.block s 0) one;
  let p = Rig.submit s ~run ~buffers:[||] in
  wait t p;
  equal int (Rig.Point.value p) (Rig_metal.signaled t.g)

(* 4,096 bytes of parameters, the most a launch has: [last] sums the 1,022 words
   before its ref, the last 8 bytes. *)
let widest () =
  let t = dev () in
  let image = launch_image t.d in
  let out = zeroed t.d 4 in
  let s =
    Sub.make t.d
      [| launch image "last" ~params:4096 [ Sub.write 0 ~at:4088 ] |]
  in
  let run = Run.make () in
  let k = Sub.block s 0 in
  set_grid run k one;
  for w = 0 to 1021 do
    Run.int32 run k (4 * w) (w + 1)
  done;
  Run.int64 run k 4088 0;
  run_wait s run [| out |];
  equal int (1022 * 1023 / 2) (words out).(0)

(* Geometry beyond a function's limits is refused, by the setter or at submit,
   before any hand-over, and the device runs on: an empty axis of groups or
   threads, more threads in a threadgroup than its pipeline holds along one
   axis or in all, and threadgroup memory beyond the GPU's. *)
let refused_geometry () =
  let t = dev () in
  let image = launch_image t.d in
  let out = zeroed t.d (4 * 4096) in
  let s =
    Sub.make t.d
      [| launch image "neighbours" ~params:16 [ Sub.write 0 ~at:0 ] |]
  in
  let refused ?(shared = 4) groups threads =
    let run = Run.make () in
    let k = Sub.block s 0 in
    raises_match Exn.invalid_arg (fun () ->
        set_grid run k { groups; threads };
        Run.shared run k shared;
        Run.int64 run k 0 0;
        Rig.submit s ~run ~buffers:[| out |]);
    equal (option string) None (Rig.lost t.d)
  in
  refused (0, 1, 1) (1, 1, 1);
  refused (1, 1, 1) (1, 0, 1);
  refused (1, 1, 1) (2048, 1, 1);
  refused (1, 1, 1) (32, 32, 2);
  refused ~shared:(1 lsl 20) (1, 1, 1) (1, 1, 1);
  let p = submit t [||] in
  wait t p;
  equal int (Rig.Point.value p) (Rig_metal.signaled t.g)

(* Under a profile, each submission's span holds the GPU's times of its work, in
   its own command buffers: named after its launches, within the host's times
   around it, and each after the one before, as the device runs them in
   order. *)
let spans_in_order n =
  let t = dev () in
  let image = launch_image t.d in
  let out = zeroed t.d 4 in
  let s =
    Sub.make t.d
      [|
        launch image "step" ~params:8 [ Sub.write 0 ~at:0 ];
        launch image "step" ~params:8 [ Sub.write 0 ~at:0 ];
      |]
  in
  let run = Run.make () in
  for i = 0 to 1 do
    let k = Sub.block s i in
    set_grid run k one;
    Run.int64 run k 0 0
  done;
  let before = Rig.Profile.now () in
  let (), events =
    Rig.Profile.take (fun () ->
        for _ = 1 to n do
          ignore (Rig.submit s ~run ~buffers:[| out |])
        done;
        B.wait out Read)
  in
  let after = Rig.Profile.now () in
  let spans =
    List.filter_map
      (function
        | Rig.Profile.Span s when Rig.equal s.device t.d ->
            Some (s.lane, s.name, s.start, s.stop)
        | _ -> None)
      events
  in
  equal int ~msg:"spans" n (List.length spans);
  equal int ~msg:"steps" (2 * n) (words out).(0);
  ignore
    (List.fold_left
       (fun last (lane, name, start, stop) ->
         equal (pair string string) ("COMPUTE:0", "step, step") (lane, name);
         at_least ~msg:"a start" int ~than:last start;
         at_least ~msg:"a stop" int ~than:start stop;
         at_most ~msg:"a stop" int ~than:after stop;
         stop)
       before spans)

(* A warm submit of a launch allocates nothing on the OCaml heap, nor does a
   store into its run. *)
let no_allocation () =
  let t = dev () in
  let image = launch_image t.d in
  let out = zeroed t.d 4 in
  let s =
    Sub.make t.d
      [| launch image "step" ~params:8 [ Sub.write 0 ~at:0 ] |]
  in
  let run = Run.make () and buffers = [| out |] in
  let k = Sub.block s 0 in
  let go () =
    set_grid run k one;
    Run.int64 run k 0 0;
    ignore (Rig.submit s ~run ~buffers)
  in
  go ();
  B.wait out Read;
  let before = Gc.minor_words () in
  for _ = 1 to 100 do
    go ()
  done;
  let words = int_of_float (Gc.minor_words () -. before) in
  B.wait out Read;
  equal int 0 words

let launches =
  group ~timeout:60. "launches"
    [
      prop ~count:50 "a launch's threadgroup memory holds a word per thread"
        gen_grid shared_law;
      prop ~count:50
        "launches and fills of one submission each run after the part before"
        gen_chain chained;
      test "a launch of no parameters runs" no_params;
      test "a launch reads 4,096 bytes of parameters, a ref in the last 8"
        widest;
      test
        "geometry beyond a function's limits is refused, and the device runs on"
        refused_geometry;
      test "a warm launch submit allocates nothing" no_allocation;
      cases
        ~name:(strf "a profile holds %d submissions' spans, in order")
        "spans" [ 1; 3; 300 ] spans_in_order;
    ]

(* Memory *)

let page = 16384

let aligned_256 n =
  let t = dev () in
  let r = alloc t n in
  equal int 0 (host r mod 256);
  equal int 0 (gpu r mod 256);
  Rig_metal.free t.g r

let empty_regions () =
  let t = dev () in
  let invalid f = raises_match Exn.invalid_arg f in
  invalid (fun () -> Rig_metal.alloc t.g Rig_edge.Device 0);
  invalid (fun () -> Rig_metal.map_host t.g (H.pages page) 0)

let no_peer () =
  let t = dev () in
  let other = second () in
  Fun.protect ~finally:(fun () -> Rig_metal.stop other ~fault:None) @@ fun () ->
  let o = alloc_on other 64 in
  equal bool false (Rig_metal.peer t.g other);
  equal (option pass) None (Rig_metal.map_peer t.g other o);
  Rig_metal.free other o

let given_back () =
  let t = dev () in
  let r = alloc t 64 in
  let m = require_some (Rig_metal.map_host t.g (H.pages page) page) in
  let wr = S.weak (handle r) and wm = S.weak (handle m) in
  wait t (submit t [||]);
  Rig_metal.free t.g r;
  Rig_metal.free t.g m;
  equal bool ~msg:"allocation" false (S.alive wr);
  equal bool ~msg:"mapping" false (S.alive wm)

(* A mapping wraps the pages that hold its bytes: its host address is the page
   of its first byte. *)
let mapped_pages () =
  let t = dev () in
  let p = H.pages (2 * H.page) in
  let a = p + H.page - 8 in
  let m = require_some (Rig_metal.map_host t.g a 16) in
  equal int ~msg:"the host address" p (host m);
  Rig_metal.free t.g m;
  H.free_pages p (2 * H.page)

let memory =
  group ~timeout:60. "memory"
    [
      prop "an allocation starts at a multiple of 256 bytes"
        ~examples:[ 1; 2; 255; 256; 257; 4095; 4096; page; page + 1 ]
        (Gen.int_range 1 (64 lsl 20))
        aligned_256;
      test "alloc and map_host refuse no bytes" empty_regions;
      test "another device of the GPU maps none of its memory" no_peer;
      test "free releases an allocation's or a mapping's buffer" given_back;
      test "a mapping's host address is its first byte's page" mapped_pages;
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
  let f =
    S.dispatch ~pipeline:(pipeline t "bump") args
      ~groups:((n + 255) / 256)
      ~threads:256
  in
  let s, run, buffers =
    Rig_gpu_support.prepare ~fixed:[ (b, Read_write) ] t.d [ S.part f ]
  in
  set_args args ~at:0 ~out:(Sub.address s b) ~c:n;
  ignore (Rig.submit s ~run ~buffers);
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
    | Rig_edge.Loaded i -> i
    | Place _ -> failf "Metal asked to place its code"
  in
  ignore (found (Rig_metal.entry i "small")).code;
  let refused () =
    raises_match
      (Exn.invalid_arg
         ~substring:"Rig_metal.entry: Metal makes no pipeline of \"wide\": ")
      (fun () -> Rig_metal.entry i "wide")
  in
  refused ();
  refused ();
  Rig_metal.unload t.g i

(* Images entered from two domains: whatever the order, every [entry] of one
   function answers the address the first answered. *)

(* An image, with the first address any domain got for each function. *)
type held = {
  i : Rig_metal.image;
  first : (string, int) Hashtbl.t;
  m : Mutex.t;
}

let entry_system h f =
  match Rig_metal.entry h.i f with
  | Absent | No_memory _ -> false
  | Entry { code = p; _ } -> (
      Mutex.protect h.m @@ fun () ->
      match Hashtbl.find_opt h.first f with
      | Some q -> p = q
      | None ->
          Hashtbl.add h.first f p;
          true)

let loaded_image =
  abstract "i" ~release:(fun h -> Rig_metal.unload (dev ()).g h.i)

let functions =
  Gen.of_list ~pp:Format.pp_print_string [ "fill"; "step"; "spin"; "bump" ]

let entry_commands =
  [
    command "image"
      (Gen.unit @-> makes loaded_image)
      Fun.id
      (fun () ->
        { i = load (dev ()).g; first = Hashtbl.create 4; m = Mutex.create () });
    command "entry"
      (loaded_image ^-> functions @-> returns bool)
      (fun () _ -> true)
      entry_system;
  ]

let unloaded_releases () =
  let t = dev () in
  let weaks =
    List.init 60 (fun _ ->
        let i = load t.g in
        let w =
          S.weak (Nativeint.of_int (found (Rig_metal.entry i "fill")).code)
        in
        Rig_metal.unload t.g i;
        w)
  in
  Gc.full_major ();
  List.iteri
    (fun k w -> equal bool ~msg:(strf "image %d" k) false (S.alive w))
    weaks;
  several_fills ()

(* An entry's parameters are the bytes of the struct its function reads as
   buffer 0, trailing padding included; [Some 0] with no buffer 0, and [None]
   where buffer 0 points to words. *)
let entry_params () =
  let t = dev () in
  let l = launch_image t.d in
  let params f = Rig.Image.params l f in
  equal (option int) ~msg:"twice: two pointers and a word" (Some 24)
    (params "twice");
  equal (option int) ~msg:"neighbours: a pointer and a word" (Some 16)
    (params "neighbours");
  equal (option int) ~msg:"last" (Some 4096) (params "last");
  equal (option int) ~msg:"empty, of no buffer 0" (Some 0) (params "empty");
  let i =
    match
      require_ok (Rig_metal.image t.g (S.fixture ~dir:"fixtures" "threadgroup"))
    with
    | Rig_edge.Loaded i -> i
    | Place _ -> failf "Metal asked to place its code"
  in
  equal (option int) ~msg:"small: words at buffer 0" None
    (found (Rig_metal.entry i "small")).params;
  Rig_metal.unload t.g i

let images =
  group ~timeout:60. "images"
    [
      test "a function that is no compute kernel is an error" no_kernel;
      test "an entry beyond the GPU's limits raises, each time" beyond_limits;
      stateful ~domains:2 ~count:30
        "an image entered from two domains makes one pipeline a function"
        entry_commands;
      test "unloaded images release their pipelines" unloaded_releases;
      test "an entry's parameters are its buffer 0's struct" entry_params;
    ]

(* Timeline and loss *)

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
  (* The submission's hold of the ICB goes at a drain that finds the device
     Stopped, which the word shows once the completion handler has run. *)
  await_word t.d (Rig.submitted t.d);
  await_release t.d w

(* Images still loaded when the device stops stay loaded: their unload after the
   stop releases them. *)
let unload_after_stop () =
  let g = S.driver () in
  let i = load g in
  let weak f = S.weak (Nativeint.of_int (found (Rig_metal.entry i f)).code) in
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
      test
        "a loss while work runs: the work runs to its end, and its indirect \
         command buffer is released after"
        stopped_running;
      test "an image a stop left loaded is released by its unload"
        unload_after_stop;
    ]

(* Opening and misuse *)

let two_devices () =
  let b = second () in
  Fun.protect ~finally:(fun () -> Rig_metal.stop b ~fault:None) @@ fun () ->
  S.with_ @@ fun a ->
  let base = Rig.submitted a.d in
  Rig.Point.wait (S.submit a [||]);
  equal int (base + 1) (Rig_metal.signaled a.g);
  equal int 0 (Rig_metal.signaled b)

(* rig finds the record the device's facts declare, and none on the host. *)
let declared () =
  let t = dev () in
  (match Rig.capability t.d Rig_metal_abi.key with
  | Some c -> equal bool ~msg:"the record" true (c == Rig_metal.capability t.g)
  | None -> fail "no record under the key");
  is_none ~msg:"the host" (Rig.capability Rig.host Rig_metal_abi.key)

let apple_align () =
  let t = dev () in
  if not (String.starts_with ~prefix:"Apple" (Rig_metal.facts t.g).arch) then
    skip ~reason:"the GPU is of a Mac family" ();
  equal int 4 (Rig_metal.capability t.g).align

let opening =
  group ~timeout:60. "opening"
    [
      test "a device other than 0 is an error" (fun () ->
          is_error (Rig_gpu_support.opened (Rig_metal.gpu 1)));
      test "a negative device is misuse" (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Rig_gpu_support.opened (Rig_metal.gpu (-1)));
          raises_match Exn.invalid_arg (fun () -> (Rig_metal.gpu (-1)).name));
      test "device names" (fun () ->
          equal (list string)
            [ "METAL"; "METAL:1"; "METAL:7" ]
            (List.map (fun i -> (Rig_metal.gpu i).name) [ 0; 1; 7 ]));
      test "two opens are two devices, each with its own word" two_devices;
      test "an Apple GPU aligns arguments to 4 bytes" apple_align;
      test "rig finds its record" declared;
      test "off macOS no device opens" (fun () ->
          if S.macos then skip ~reason:"macOS" ();
          equal int 0 (Rig_metal.count ());
          equal (result pass string) (Error "Metal exists on macOS only")
            (Rig_gpu_support.opened (Rig_metal.gpu 0)));
      test "a virtual Mac's GPU opens no device, for want of a residency set"
        (fun () ->
          if not S.virtual_mac then skip ~reason:"no virtual Mac" ();
          equal (result pass string) (Error "Metal made no residency set")
            (Rig_gpu_support.opened (Rig_metal.gpu 0)));
    ]

let () =
  clear_files ();
  exit
    (run "rig_metal"
       [
         ring_tests;
         work;
         commits;
         icbs;
         launches;
         memory;
         file_tests;
         images;
         timeline;
         opening;
       ])
