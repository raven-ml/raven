(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module S = Device_metal_support

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

   One device serves the tests that leave it healthy; a test that fails a
   submission or stops a device opens its own. [v] is the last value
   submitted. *)

type dev = { d : Device_metal.t; mutable v : int; fill : Device_metal.image }

let opened () =
  match Device_metal.open_ 0 with Ok d -> d | Error why -> skip ~reason:why ()

let dev_of d =
  let fill =
    fst (require_ok (Device_metal.image d (S.fixture ~dir:"fixtures" "fill")))
  in
  { d; v = 0; fill }

let shared = lazy (dev_of (opened ()))
let dev () = Lazy.force shared
let pipeline t f = require_some (Device_metal.entry t.fill f)

let submitted =
  Testable.contramap
    (function `Ok -> "`Ok" | `Failed why -> "`Failed " ^ why)
    string

let room =
  Testable.contramap
    (function `Fits -> "`Fits" | `Later -> "`Later" | `Never -> "`Never")
    string

let stopped =
  Testable.contramap
    (function `Stopped -> "`Stopped" | `Unknown -> "`Unknown")
    string

(* Submits [fills] as [t]'s next value; the fills live until it returned. *)
let submit t fills =
  let ps = Array.map (S.part t.d) fills in
  equal room `Fits (Device_metal.room t.d ps);
  t.v <- t.v + 1;
  let r = Device_metal.submit t.d ~v:t.v ~waits:[||] ~handles:[||] ps in
  ignore (Sys.opaque_identity fills);
  r

let submit_ok t fills =
  equal submitted `Ok (submit t fills);
  t.v

let alloc t n = require_some (Device_metal.alloc t.d `Device n)
let host r = require_some (Device_metal.host r)
let gpu r = require_some (Device_metal.address r)

(* The arguments of the fill kernels, [{ out; c }], at byte [at] of [args]. *)
let set_args args ~at ~out ~c =
  S.set64 (host args) (at / 8) (Int64.of_int out);
  S.set32 (host args) ((at / 4) + 2) c

let args_bytes = 16

(* A dispatch of [fill] writing [3i + c] at the 32-bit words [0, n) of the GPU
   address [out], its arguments at byte [at] of [args]. *)
let fill_dispatch t ~args ~at ~out ~c n =
  set_args args ~at ~out ~c;
  S.dispatch ~pipeline:(pipeline t "fill") ~offset:at args ~groups:1 ~threads:n

let filled out ?(at = 0) ~c n =
  Array.init n (fun i -> S.get32 (host out) (at + i) = (3 * i) + c)
  |> Array.for_all Fun.id

(* Work *)

let gen_submission = Gen.list ~size:(Gen.int_range 0 3) (Gen.int_range 0 3)

let prefix_completion parts =
  let t = dev () in
  let n = List.length parts in
  let chunk = 64 in
  let out = alloc t (n * 4 * chunk * 4)
  and args = alloc t (n * 4 * args_bytes) in
  let first = t.v + 1 in
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
        S.split f t.d k ~times:0n;
        f)
      ps
    |> Array.of_list
  in
  List.iteri (fun s ps -> ignore (submit_ok t (fills s ps))) parts;
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
    let w = Device_metal.signaled t.d in
    at_least int ~than:seen w;
    complete w;
    if w < t.v then poll w
  in
  poll (first - 1);
  Device_metal.free t.d out;
  Device_metal.free t.d args

(* Splits [k] times, recording times; [(before, times, after)]: the host clock
   before the submission and after its value was signaled. *)
let split_times k =
  let t = dev () in
  let out = alloc t 256
  and args = alloc t args_bytes
  and times = alloc t (16 * k) in
  let f = fill_dispatch t ~args ~at:0 ~out:(gpu out) ~c:1 64 in
  S.split f t.d k ~times:(host times);
  let before = S.uptime () in
  let v = submit_ok t [| f |] in
  S.wait t.d v;
  let after = S.uptime () in
  let at i = Int64.to_int (S.get64 (host times) i) in
  let r =
    (before, Array.init k (fun i -> (at (2 * i), at ((2 * i) + 1))), after)
  in
  List.iter (Device_metal.free t.d) [ out; args; times ];
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
  S.split f t.d 1100 ~times:0n;
  S.wait t.d (submit_ok t [| f |]);
  equal bool true (filled out ~c:5 64);
  List.iter (Device_metal.free t.d) [ out; args ]

let fresh_allocation () =
  let t = dev () in
  let out = alloc t (1 lsl 20) and args = alloc t args_bytes in
  let f = fill_dispatch t ~args ~at:0 ~out:(gpu out) ~c:9 256 in
  S.wait t.d (submit_ok t [| f |]);
  equal bool true (filled out ~c:9 256);
  List.iter (Device_metal.free t.d) [ out; args ]

let several_fills () =
  let t = dev () in
  let out = alloc t 1024 and args = alloc t (2 * args_bytes) in
  let first = fill_dispatch t ~args ~at:0 ~out:(gpu out) ~c:1 1 in
  let step =
    S.dispatch ~pipeline:(pipeline t "step") ~offset:0 args ~groups:1 ~threads:1
  in
  S.wait t.d (submit_ok t [| first; step; step; step |]);
  equal int 4 (S.get32 (host out) 0);
  List.iter (Device_metal.free t.d) [ out; args ]

let empty_submission () =
  let t = dev () in
  let v = submit_ok t [||] in
  S.wait t.d v;
  equal int v (Device_metal.signaled t.d)

let failing_fill () =
  let t = dev_of (opened ()) in
  let out = alloc t 256 and args = alloc t args_bytes in
  let ok = fill_dispatch t ~args ~at:0 ~out:(gpu out) ~c:2 64 in
  let reached = submit_ok t [| ok |] in
  let failed = submit t [| ok; S.failing 7 |] in
  equal submitted (`Failed "running a fill: it returned 7") failed;
  S.wait t.d reached;
  raises (Device_metal.Fault "running a fill: it returned 7") (fun () ->
      Device_metal.sleep t.d ~seen:reached ~still_ms:10);
  equal int reached (Device_metal.signaled t.d);
  equal submitted (`Failed "running a fill: it returned 7") (submit t [| ok |]);
  equal int reached (Device_metal.signaled t.d)

let work =
  group ~timeout:60. "work"
    [
      prop ~count:50
        "a signaled value's submissions and every earlier one are written"
        (Gen.list ~size:(Gen.int_range 1 50) gen_submission)
        prefix_completion;
      test "an empty submission is signaled" empty_submission;
      test "fills of a submission run in order" several_fills;
      test "a kernel writes an allocation in the submission after it"
        fresh_allocation;
      cases
        ~name:(strf "a split's times lie in the work's span (%d splits)")
        "times" [ 1; 5; 64 ] times_between;
      test "a fill splitting more often than the queue holds completes"
        many_splits;
      test "a failed fill stops the word before its value" failing_fill;
    ]

(* Indirect command buffers *)

let dispatch ?(offset = 0) ?(groups = (1, 1, 1)) ?(threads = (1, 1, 1)) p =
  { Device_metal_abi.pipeline = p; offset; groups; threads }

let icb t args ds =
  (Device_metal.capability t.d).icb (Device_metal.handle args) ds

let run_icb t (b : Device_metal_abi.icb) pipelines =
  S.wait t.d (submit_ok t [| S.execute b ~pipelines |])

let chain n =
  let t = dev () in
  let out = alloc t 256 and args = alloc t args_bytes in
  set_args args ~at:0 ~out:(gpu out) ~c:0;
  let step = pipeline t "step" in
  let b = require_ok (icb t args (Array.make n (dispatch step))) in
  run_icb t b [| step |];
  equal int n (S.get32 (host out) 0);
  b.release ();
  List.iter (Device_metal.free t.d) [ out; args ]

let resized () =
  let t = dev () in
  let out = alloc t 256 and args = alloc t args_bytes in
  set_args args ~at:0 ~out:(gpu out) ~c:4;
  let fill = pipeline t "fill" in
  let b = require_ok (icb t args [| dispatch ~threads:(4, 1, 1) fill |]) in
  run_icb t b [| fill |];
  equal bool true (filled out ~c:4 4);
  equal int 0 (S.get32 (host out) 4);
  S.resize b.commands.(0) ~groups:2 ~threads:4;
  run_icb t b [| fill |];
  equal bool true (filled out ~c:4 8);
  b.release ();
  List.iter (Device_metal.free t.d) [ out; args ]

let icb_refusals () =
  let t = dev () in
  let args = alloc t args_bytes in
  let fill = pipeline t "fill" in
  let align = (Device_metal.capability t.d).align in
  let invalid ds = raises_match Exn.invalid_arg (fun () -> icb t args ds) in
  is_error (icb t args [| dispatch ~threads:(1025, 1, 1) fill |]);
  invalid [| dispatch ~offset:args_bytes fill |];
  if align > 1 then invalid [| dispatch ~offset:(align / 2) fill |];
  invalid [| dispatch ~groups:(0, 1, 1) fill |];
  invalid [| dispatch ~threads:(1, 1, 0) fill |];
  (require_ok (icb t args [||])).release ();
  Device_metal.free t.d args

let released_twice () =
  let t = dev () in
  let args = alloc t args_bytes in
  let b = require_ok (icb t args [| dispatch (pipeline t "step") |]) in
  b.release ();
  raises_match Exn.invalid_arg b.release;
  Device_metal.free t.d args

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
  Device_metal.free t.d args

let after_unload () =
  let t = dev () in
  let out = alloc t 256 and args = alloc t args_bytes in
  set_args args ~at:0 ~out:(gpu out) ~c:0;
  let i, _ =
    require_ok (Device_metal.image t.d (S.fixture ~dir:"fixtures" "fill"))
  in
  let step = require_some (Device_metal.entry i "step") in
  let b = require_ok (icb t args [| dispatch step; dispatch step |]) in
  Device_metal.unload t.d i;
  Gc.full_major ();
  run_icb t b [||];
  equal int 2 (S.get32 (host out) 0);
  b.release ();
  List.iter (Device_metal.free t.d) [ out; args ]

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
    ]

(* Memory *)

let page = 16384

let shared_both_ways (offset, pages) =
  let t = dev () in
  let n = (pages * page) - offset in
  let p =
    Nativeint.add (S.pages ((pages + 1) * page)) (Nativeint.of_int offset)
  in
  for i = 0 to n - 1 do
    S.set8 p i (i mod 251)
  done;
  let r = require_some (Device_metal.map_host t.d p n) in
  equal nativeint
    (Nativeint.logand p (Nativeint.of_int (lnot (page - 1))))
    (host r);
  let args = alloc t args_bytes in
  let into = Nativeint.to_int (Nativeint.sub p (host r)) in
  set_args args ~at:0 ~out:(gpu r + into) ~c:n;
  let bump = pipeline t "bump" in
  let f =
    S.dispatch ~pipeline:bump args ~groups:((n + 255) / 256) ~threads:256
  in
  S.wait t.d (submit_ok t [| f |]);
  for i = 0 to n - 1 do
    if S.get8 p i <> (i mod 251) + 1 then
      failf "byte %d reads %d, not %d" i (S.get8 p i) ((i mod 251) + 1)
  done;
  Device_metal.unmap t.d r;
  Device_metal.free t.d args

let aligned_256 n =
  let t = dev () in
  let r = alloc t n in
  equal int 0 (Nativeint.to_int (host r) mod 256);
  equal int 0 (gpu r mod 256);
  Device_metal.free t.d r

let misused_regions () =
  let t = dev () in
  let invalid f = raises_match Exn.invalid_arg f in
  let r = alloc t 64 in
  let m = require_some (Device_metal.map_host t.d (S.pages page) page) in
  invalid (fun () -> Device_metal.free t.d m);
  invalid (fun () -> Device_metal.free t.d (Device_metal.word t.d));
  invalid (fun () -> Device_metal.unmap t.d r);
  Device_metal.free t.d r;
  invalid (fun () -> Device_metal.free t.d r);
  Device_metal.unmap t.d m;
  invalid (fun () -> Device_metal.unmap t.d m);
  invalid (fun () -> Device_metal.alloc t.d `Device 0);
  invalid (fun () -> Device_metal.map_host t.d (S.pages page) 0);
  let other = opened () in
  let o = require_some (Device_metal.alloc other `Device 64) in
  equal (option pass) None (Device_metal.map_peer t.d other o);
  invalid (fun () -> Device_metal.map_peer t.d t.d o);
  invalid (fun () -> Device_metal.map_peer t.d other (alloc t 64));
  Device_metal.free other o;
  invalid (fun () -> Device_metal.map_peer t.d other o)

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
        "free, unmap and map_peer refuse a region of the wrong kind or given \
         back"
        misused_regions;
    ]

(* Images *)

let not_metallib () =
  let t = dev () in
  is_error (Device_metal.image t.d "not a metallib")

let entries () =
  let t = dev () in
  List.iter
    (fun f -> ignore (require_some ~msg:f (Device_metal.entry t.fill f)))
    [ "fill"; "step"; "spin"; "bump" ];
  equal (option int) None (Device_metal.entry t.fill "absent")

let unloaded_twice () =
  let t = dev () in
  let i, _ =
    require_ok (Device_metal.image t.d (S.fixture ~dir:"fixtures" "fill"))
  in
  let other = opened () in
  raises_match Exn.invalid_arg (fun () -> Device_metal.unload other i);
  Device_metal.unload t.d i;
  raises_match Exn.invalid_arg (fun () -> Device_metal.unload t.d i);
  raises_match Exn.invalid_arg (fun () -> Device_metal.entry i "fill")

(* Images unloaded from two domains: whatever the order, an image's first
   [unload] returns and every later one raises. *)

type loaded = { mutable loaded : bool }

let unload_model m =
  if not m.loaded then invalid_arg "unloaded";
  m.loaded <- false

let unload_system i = Device_metal.unload (dev ()).d i

let loaded_image =
  abstract "i" ~release:(fun i ->
      try unload_system i with Invalid_argument _ -> ())

let unload_commands =
  [
    command "image"
      (Gen.unit @-> makes loaded_image)
      (fun () -> { loaded = true })
      (fun () ->
        let t = dev () in
        fst
          (require_ok
             (Device_metal.image t.d (S.fixture ~dir:"fixtures" "fill"))));
    command "unload" (loaded_image ^-> returns unit) unload_model unload_system;
  ]

let unloaded_releases () =
  let t = dev () in
  let weaks =
    List.init 60 (fun _ ->
        let i, _ =
          require_ok (Device_metal.image t.d (S.fixture ~dir:"fixtures" "fill"))
        in
        let w =
          S.weak (Nativeint.of_int (require_some (Device_metal.entry i "fill")))
        in
        Device_metal.unload t.d i;
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
      test "each function of the image has an entry" entries;
      test "unload and entry refuse an unloaded image or another device's"
        unloaded_twice;
      stateful ~domains:2 ~count:30
        "an image unloaded from two domains is unloaded once" unload_commands;
      test "unloaded images release their pipelines" unloaded_releases;
    ]

(* Timeline and loss *)

let sleep_seen () =
  let t = dev () in
  let v = submit_ok t [||] in
  S.wait t.d v;
  Device_metal.sleep t.d ~seen:(v - 1) ~still_ms:600_000

let stopped_idle () =
  let t = dev_of (opened ()) in
  let v = submit_ok t [||] in
  S.wait t.d v;
  equal stopped `Stopped (Device_metal.stop t.d);
  equal int v (Device_metal.signaled t.d)

let stopped_running () =
  let t = dev_of (opened ()) in
  let out = alloc t 256 and args = alloc t args_bytes in
  set_args args ~at:0 ~out:(gpu out) ~c:5_000_000;
  let spin = pipeline t "spin" in
  let b = require_ok (icb t args [| dispatch spin |]) in
  let w = S.weak b.handle in
  let v = submit_ok t [| S.execute b ~pipelines:[| spin |] |] in
  equal stopped `Unknown (Device_metal.stop t.d);
  S.wait t.d v;
  equal bool true (S.get32 (host out) 0 <> 0);
  b.release ();
  equal bool false (S.alive w)

let timeline =
  group ~timeout:60. "timeline"
    [
      test "sleep returns at once when the word differs from seen" sleep_seen;
      test "stop answers Stopped once the work completed" stopped_idle;
      test
        "stop answers Unknown while work runs, the word reaches the last value \
         once it ends, and its indirect command buffer is released after"
        stopped_running;
    ]

(* Opening and misuse *)

let two_devices () =
  let a = dev_of (opened ()) and b = dev_of (opened ()) in
  S.wait a.d (submit_ok a [||]);
  equal int 1 (Device_metal.signaled a.d);
  equal int 0 (Device_metal.signaled b.d)

let apple_align () =
  let t = dev () in
  if not (String.starts_with ~prefix:"Apple" (Device_metal.arch t.d)) then
    skip ~reason:"the GPU is of a Mac family" ();
  equal int 4 (Device_metal.capability t.d).align

let misused_work () =
  let t = dev () in
  let invalid f = raises_match Exn.invalid_arg f in
  let part w = Device_metal.part t.d ~queue:"COMPUTE:0" w in
  let r = alloc t 64 in
  invalid (fun () -> part (`Words [| 0 |]));
  invalid (fun () -> part (`Copy ((r, 0), (r, 8), 8)));
  invalid (fun () -> part (`Fill (0n, 0n, 1, 0)));
  invalid (fun () -> part (`Fill (0n, 0n, 0, 64)));
  invalid (fun () ->
      Device_metal.part t.d ~queue:"COPY:0" (`Fill (0n, 0n, 0, 0)));
  invalid (fun () ->
      Device_metal.part t.d ~queue:"COMPUTE:0" ~after:[| -1 |]
        (`Fill (0n, 0n, 0, 0)));
  let v = t.v + 1 in
  let submit ?(waits = [||]) ~v ps =
    Device_metal.submit t.d ~v ~waits ~handles:[||] ps
  in
  invalid (fun () -> submit ~v:(v + 1) [||]);
  invalid (fun () -> submit ~v:0 [||]);
  invalid (fun () -> submit ~v ~waits:[| (`Word, 0, 1) |] [||]);
  let forward =
    Device_metal.part t.d ~queue:"COMPUTE:0" ~after:[| 0 |]
      (`Fill (0n, 0n, 0, 0))
  in
  invalid (fun () -> submit ~v [| forward |]);
  let other = opened () in
  let fill =
    Device_metal.part other ~queue:"COMPUTE:0" (`Fill (0n, 0n, 0, 0))
  in
  invalid (fun () -> submit ~v [| fill |]);
  Device_metal.free t.d r

let opening =
  group ~timeout:60. "opening"
    [
      test "a device other than 0 is an error" (fun () ->
          is_error (Device_metal.open_ 1));
      test "a negative device is misuse" (fun () ->
          raises_match Exn.invalid_arg (fun () -> Device_metal.open_ (-1));
          raises_match Exn.invalid_arg (fun () -> Device_metal.device_name (-1)));
      test "device names" (fun () ->
          equal (list string)
            [ "METAL"; "METAL:1"; "METAL:7" ]
            (List.map Device_metal.device_name [ 0; 1; 7 ]));
      test "two opens are two devices, each with its own word" two_devices;
      test "an Apple GPU aligns arguments to 4 bytes" apple_align;
      test "parts and submissions refuse what the device does not run"
        misused_work;
      test "off macOS no device opens" (fun () ->
          if S.macos then skip ~reason:"macOS" ();
          equal int 0 (Device_metal.count ());
          equal (result pass string) (Error "Metal exists on macOS only")
            (Device_metal.open_ 0));
    ]

let () =
  exit
    (run "device_metal"
       [ ring_tests; work; icbs; memory; images; timeline; opening ])
