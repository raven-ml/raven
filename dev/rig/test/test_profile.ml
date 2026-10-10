(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module Prof = Rig.Profile
module P = Rig_support.Polled
module Support = Rig_support

let submit ?(reads = [||]) ?(writes = [||]) ?(waits = [||]) s =
  Rig.submit s ~run:(Sub.Run.make ()) ~reads ~writes ~waits

let timeout = 60.
let memory name = require_ok ~pp:Format.pp_print_string (Rig.memory_device name)
let empty d = Sub.make ~reads:0 ~writes:0 d [||]

(* The spans of [events], as [(lane, name)]. *)
let spans events =
  List.filter_map
    (function Prof.Span s -> Some (s.lane, s.name) | _ -> None)
    events

let named events =
  List.filter_map (function Prof.Span s -> Some s.name | _ -> None) events

(* Taking *)

let test_span () =
  let (), events = Prof.take (fun () -> Prof.span "work" ignore) in
  equal (list (pair string string)) [ ("domain 0", "work") ] (spans events);
  match events with
  | [ Prof.Span s ] ->
      equal bool true (Rig.equal Rig.host s.device);
      at_most int ~than:s.stop s.start
  | _ -> failf "%d events" (List.length events)

(* A span is in the profiles taken when it starts. *)
let test_span_starts () =
  let (), outer =
    Prof.take (fun () ->
        Prof.span "outer" (fun () ->
            let (), inner = Prof.take (fun () -> Prof.span "inner" ignore) in
            equal ~msg:"inner profile" (list string) [ "inner" ] (named inner)))
  in
  equal ~msg:"outer profile" (slist string compare) [ "inner"; "outer" ]
    (named outer)

let test_raises () =
  raises Exit (fun () -> Prof.take (fun () -> raise Exit));
  equal bool false (Prof.enabled ())

(* Two domains: a span the second records while the first's profile is taken is
   in that profile, and only there. *)
let test_two_domains () =
  let lock = Mutex.create () and cond = Condition.create () in
  let stage = ref 0 in
  let at n =
    Mutex.protect lock (fun () ->
        while !stage < n do
          Condition.wait cond lock
        done)
  in
  let next () =
    Mutex.protect lock (fun () ->
        incr stage;
        Condition.broadcast cond)
  in
  let other =
    Domain.spawn (fun () ->
        at 1;
        Prof.span "theirs" ignore;
        next ())
  in
  let (), events =
    Prof.take (fun () ->
        next ();
        at 2;
        Prof.span "mine" ignore)
  in
  Domain.join other;
  equal
    (slist (pair string string) compare)
    [ ("domain 0", "mine"); ("domain 1", "theirs") ]
    (spans events)

(* Profiles taken at once on two domains overlap: each holds the spans recorded
   while it was taken, whichever domain recorded them. *)
let test_overlapping () =
  let lock = Mutex.create () and cond = Condition.create () in
  let stage = ref 0 in
  let at n =
    Mutex.protect lock (fun () ->
        while !stage < n do
          Condition.wait cond lock
        done)
  in
  let next () =
    Mutex.protect lock (fun () ->
        incr stage;
        Condition.broadcast cond)
  in
  let other =
    Domain.spawn (fun () ->
        at 1;
        let (), events = Prof.take (fun () -> Prof.span "theirs" ignore) in
        next ();
        (Printf.sprintf "domain %d" (Domain.self () :> int), events))
  in
  let (), events =
    Prof.take (fun () ->
        next ();
        at 2;
        Prof.span "mine" ignore)
  in
  let lane, theirs = Domain.join other in
  equal ~msg:"the longer profile"
    (slist (pair string string) compare)
    [ ("domain 0", "mine"); (lane, "theirs") ]
    (spans events);
  equal ~msg:"the inner one"
    (slist (pair string string) compare)
    [ (lane, "theirs") ]
    (spans theirs)

let test_counters () =
  raises_match Exn.invalid_arg (fun () ->
      Prof.take ~counters:[ "a"; "a" ] ignore);
  let inner () =
    equal (list string) [ "x"; "y"; "z" ] (Prof.counters ());
    equal bool true (Prof.traced ())
  in
  let (), _ =
    Prof.take ~counters:[ "x"; "y" ] (fun () ->
        ignore (Prof.take ~counters:[ "y"; "z" ] ~trace:true inner);
        equal bool false (Prof.traced ()))
  in
  equal bool false (Prof.enabled ());
  equal (list string) [] (Prof.counters ())

(* Device time *)

let functions d =
  require_ok ~pp:Format.pp_print_string (Rig.Image.load d "functions")

let launch image kernel =
  {
    Sub.queue = "COMPUTE:0";
    after = [||];
    work = Sub.Launch { image; kernel; params = 0; refs = [||] };
  }

let bump arg =
  {
    Sub.queue = "COMPUTE:0";
    after = [||];
    work =
      Fill { fill = Support.bump; arg; ring_units = 0; segment_bytes = 0 };
  }

(* Submits [parts] on [d] with a run whose launches run one group of one
   thread. *)
let submit_parts d parts =
  let s = Sub.make ~reads:0 ~writes:0 d parts in
  let run = Sub.Run.make () in
  Array.iteri
    (fun i (p : Sub.part) ->
      match p.work with
      | Launch _ ->
          let b = Sub.block s i in
          Sub.Run.groups run b 1 1 1;
          Sub.Run.threads run b 1 1 1
      | _ -> ())
    parts;
  Rig.submit s ~run ~reads:[||] ~writes:[||] ~waits:[||]

(* The spans of [d] in [events], as [(lane, name, start, stop)]. *)
let device_spans d events =
  List.filter_map
    (function
      | Prof.Span s when Rig.equal s.device d ->
          Some (s.lane, s.name, s.start, s.stop)
      | _ -> None)
    events

(* A submission's span holds its device's times of all its parts' work, read
   by the wait that reaches it, on its first part's queue, named after its
   launches' functions in the order of its parts. *)
let test_device_span () =
  let d, _ = P.open_ "profile:device" in
  let image = functions d and arg = B.create Rig.host 8 in
  let before = Prof.now () in
  let (), events =
    Prof.take (fun () ->
        Rig.Point.wait
          (submit_parts d
             [| launch image "main"; bump arg; launch image "copy" |]))
  in
  let after = Prof.now () in
  match device_spans d events with
  | [ (lane, name, start, stop) ] ->
      equal (pair string string) ("COMPUTE:0", "main, copy") (lane, name);
      at_least ~msg:"the start" int ~than:before start;
      at_least ~msg:"the stop" int ~than:start stop;
      at_most ~msg:"the stop" int ~than:after stop
  | spans -> failf "%d spans of the device" (List.length spans)

(* A submission with no launch is named after its first part's kind. *)
let test_kind_span () =
  let d, _ = P.open_ "profile:kind" in
  let arg = B.create Rig.host 8 in
  let (), events =
    Prof.take (fun () -> Rig.Point.wait (submit_parts d [| bump arg |]))
  in
  equal
    (list (pair string string))
    [ ("COMPUTE:0", "fills") ]
    (List.map (fun (lane, name, _, _) -> (lane, name)) (device_spans d events))

(* No span: of a submission with no part, of one submitted while no profile
   is taken, and of a device whose driver cannot time its work. *)
let test_no_span () =
  let d, _ = P.open_ "profile:none" and m = memory "profile:none-memory" in
  let arg = B.create Rig.host 8 in
  let earlier = submit_parts d [| bump arg |] in
  let (), events =
    Prof.take (fun () ->
        Rig.Point.wait (submit (empty d));
        Rig.Point.wait earlier;
        Rig.Point.wait (submit_parts m [| bump arg |]))
  in
  equal ~msg:"Polled" int 0 (List.length (device_spans d events));
  equal ~msg:"the memory device" int 0 (List.length (device_spans m events))

(* A submission's span goes to every profile taken when it was submitted. *)
let test_span_profiles () =
  let d, _ = P.open_ "profile:device-profiles" in
  let arg = B.create Rig.host 8 in
  let (), outer =
    Prof.take (fun () ->
        let (), inner =
          Prof.take (fun () ->
              Rig.Point.wait (submit_parts d [| bump arg |]))
        in
        equal ~msg:"inner" int 1 (List.length (device_spans d inner)))
  in
  equal ~msg:"outer" int 1 (List.length (device_spans d outer))

(* A profile leaves out the span of a device lost before it was read. *)
let test_span_lost () =
  let d, p = P.open_ "profile:device-lost" in
  let arg = B.create Rig.host 8 in
  let (), events =
    Prof.take (fun () ->
        ignore (submit_parts d [| bump arg |]);
        P.fail p;
        try ignore (submit (Sub.make ~reads:0 ~writes:0 d [||]))
        with Rig.Lost _ -> ())
  in
  equal (list string) [] (named events)

(* A span records a function that raises. *)
let test_span_raises () =
  let (), events =
    Prof.take (fun () ->
        try Prof.span "raises" (fun () -> raise Exit) with Exit -> ())
  in
  equal (list string) [ "raises" ] (named events)

let test_allocation () =
  let d, _ = P.open_ "profile:allocation" in
  let b, events = Prof.take (fun () -> B.create d 4096) in
  let allocations =
    List.filter_map
      (function
        | Prof.Allocation a when Rig.equal a.device d -> Some a.allocated
        | _ -> None)
      events
  in
  ignore (Sys.opaque_identity b);
  at_least int ~than:4096 (List.fold_left Int.max 0 allocations)

(* Copies *)

let copies events =
  List.filter_map
    (function
      | Prof.Copy c -> Some (Rig.name c.src, Rig.name c.dst, c.bytes)
      | _ -> None)
    events

let test_copy () =
  let d = memory "profile:copy" in
  let src = B.create Rig.host 100 and dst = B.create d 100 in
  let (), events = Prof.take (fun () -> B.copy ~src ~dst) in
  equal
    (list (triple string string int))
    [ ("CPU", "profile:copy", 100) ]
    (copies events)

(* A copy through the staging memory, between devices that map none of each
   other's memory, records its copies into and out of it and no event of its
   own: each transfer that ran is one event. *)
let test_staged_copy () =
  let open_ name = P.open_ ~host_visible:false ~peers:false name in
  let d, _ = open_ "profile:staged-src" and e, _ = open_ "profile:staged-dst" in
  let src = B.create d 64 and dst = B.create e 64 in
  let (), events = Prof.take (fun () -> B.copy ~src ~dst) in
  equal
    (slist (triple string string int) compare)
    [ ("CPU", "profile:staged-dst", 64); ("profile:staged-src", "CPU", 64) ]
    (copies events)

(* A staged copy's leg on a device's queue spans its transfer: the event of a
   device-to-host copy's leg into staging stops once the host saw it done, after
   the device's gate opened. *)
let test_staged_span () =
  let d, pd =
    P.open_ ~host_visible:false ~peers:false "profile:staged-span-src"
  in
  let src = B.create d 64 and dst = B.create Rig.host 64 in
  P.gate pd;
  let opener =
    Domain.spawn (fun () ->
        Support.await "the copy waiting on the device" (fun () ->
            P.sleepers pd = 1);
        let opened = Prof.now () in
        P.open_gate pd;
        opened)
  in
  let (), events = Prof.take (fun () -> B.copy ~src ~dst) in
  let opened = Domain.join opener in
  let leg =
    List.find_map
      (function Prof.Copy c when Rig.equal c.src d -> Some c.stop | _ -> None)
      events
  in
  at_least ~msg:"the leg's stop" int ~than:opened (require_some leg)

(* Cost *)

let test_untaken () =
  let f = Sys.opaque_identity ignore in
  Prof.span "warm" f;
  let before = Gc.minor_words () in
  for _ = 1 to 100 do
    Prof.span "idle" f
  done;
  equal int 0 (int_of_float (Gc.minor_words () -. before))

(* Chrome traces *)

let test_chrome () =
  let d = memory "profile:chrome" in
  let polled, _ = P.open_ "profile:chrome-code" in
  let image =
    require_ok ~pp:Format.pp_print_string (Rig.Image.load polled "code:8")
  in
  let events =
    [
      Prof.Span
        {
          device = Rig.host;
          lane = "domain 1";
          name = "a \"quoted\" \\ name";
          start = 1_500;
          stop = 1_600;
        };
      Prof.Load { image; binary = "code:8"; time = 1_200 };
      Prof.Counters
        {
          device = d;
          name = "kernel";
          start = 2_000;
          stop = 3_500;
          counters = [ ("waves", [| 3; 4 |]) ];
        };
      Prof.Trace
        {
          device = d;
          name = "kernel";
          start = 2_000;
          stop = 3_500;
          part = 1;
          data = "\001\002";
        };
      Prof.Overwritten { device = d; time = 3_600; runs = 2 };
      Prof.Span
        {
          device = Rig.host;
          lane = "domain 0";
          name = "host";
          start = 1_000;
          stop = 4_000;
        };
      Prof.Span
        {
          device = d;
          lane = "COMPUTE:0";
          name = "kernel";
          start = 2_000;
          stop = 3_500;
        };
      Prof.Allocation { device = d; time = 2_500; allocated = 4096 };
      Prof.Copy
        { src = Rig.host; dst = d; bytes = 64; start = 3_000; stop = 3_200 };
    ]
  in
  let file = "chrome.json" in
  Out_channel.with_open_bin file (fun oc -> Prof.output_chrome_trace oc events);
  let text = In_channel.with_open_bin file In_channel.input_all in
  Sys.remove file;
  expect text
  @@ __POS_OF__
       {|
    {"traceEvents":[
    {"ph":"M","pid":1,"tid":0,"name":"process_name","args":{"name":"CPU"}},
    {"ph":"M","pid":1,"tid":1,"name":"thread_name","args":{"name":"domain 0"}},
    {"ph":"X","pid":1,"tid":1,"ts":0.000,"dur":3.000,"name":"host"},
    {"ph":"M","pid":2,"tid":0,"name":"process_name","args":{"name":"profile:chrome-code"}},
    {"ph":"i","pid":2,"tid":0,"ts":0.200,"s":"p","name":"load"},
    {"ph":"M","pid":1,"tid":2,"name":"thread_name","args":{"name":"domain 1"}},
    {"ph":"X","pid":1,"tid":2,"ts":0.500,"dur":0.100,"name":"a \"quoted\" \\ name"},
    {"ph":"M","pid":3,"tid":0,"name":"process_name","args":{"name":"profile:chrome"}},
    {"ph":"M","pid":3,"tid":3,"name":"thread_name","args":{"name":"counters"}},
    {"ph":"X","pid":3,"tid":3,"ts":1.000,"dur":1.500,"name":"kernel","args":{"waves":7}},
    {"ph":"i","pid":3,"tid":0,"ts":1.000,"s":"p","name":"kernel","args":{"part":1,"bytes":2}},
    {"ph":"M","pid":3,"tid":4,"name":"thread_name","args":{"name":"COMPUTE:0"}},
    {"ph":"X","pid":3,"tid":4,"ts":1.000,"dur":1.500,"name":"kernel"},
    {"ph":"C","pid":3,"tid":0,"ts":1.500,"name":"memory","args":{"allocated":4096}},
    {"ph":"M","pid":1,"tid":5,"name":"thread_name","args":{"name":"copy"}},
    {"ph":"X","pid":1,"tid":5,"ts":2.000,"dur":0.200,"name":"copy","args":{"to":"profile:chrome","bytes":64}},
    {"ph":"i","pid":3,"tid":0,"ts":2.600,"s":"p","name":"overwritten","args":{"runs":2}}
    ]}
    |}

let tests =
  [
    group ~timeout "taking"
      [
        test "a profile holds the spans recorded while it is taken" test_span;
        test "a span is in the profiles taken when it starts" test_span_starts;
        test "a profile whose function raises raises again" test_raises;
        test "a profile holds another domain's spans" test_two_domains;
        test "profiles taken on two domains overlap" test_overlapping;
        test "a span records a function that raises" test_span_raises;
        test "an allocation is an event of the profiles taken" test_allocation;
        test "counters and traces are those the profiles taken ask for"
          test_counters;
      ];
    group ~timeout "device time"
      [
        test
          "a submission's span holds its parts' device time, named after its \
           launches"
          test_device_span;
        test "a submission with no launch is named after its first part's kind"
          test_kind_span;
        test
          "no span of an empty submission, an untimed one or an untimeable \
           device"
          test_no_span;
        test "a span goes to the profiles taken at its submit"
          test_span_profiles;
        test "a profile leaves out a lost device's unread span" test_span_lost;
      ];
    group ~timeout "copies"
      [
        test "a copy records the bytes it moved" test_copy;
        test "a staged copy records only its copies into and out of staging"
          test_staged_copy;
        test "a staged copy's queued leg spans its transfer" test_staged_span;
      ];
    group ~timeout "cost"
      [ test "a span while no profile is taken allocates nothing" test_untaken ];
    group ~timeout "chrome" [ test "a trace in Chrome's format" test_chrome ];
  ]

let () = exit (run "rig.profile" tests)
