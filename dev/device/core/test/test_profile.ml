(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module C = Device_core
module B = Device_core.Buffer
module Sub = Device_core.Submission
module Prof = Device_core.Profile
module P = Device_core_support.Polled
module Support = Device_core_support
module S = Device_dtype.Scalar

let timeout = 60.
let memory name = require_ok ~pp:Format.pp_print_string (C.memory_device name)
let empty d = Sub.make ~reads:0 ~writes:0 ~waits:0 d [||]

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
      equal bool true (C.equal C.host s.device);
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

(* Events read after their point *)

(* [after]'s events are read in the first wait that finds the point reached,
   before it returns, and ordered by time, longest first. *)
let test_after () =
  let d, _ = P.open_ "profile:after" in
  let m = B.create d S.UInt8 64 in
  let read = ref false in
  let span name start stop =
    Prof.Span { device = d; lane = "COMPUTE:0"; name; start; stop }
  in
  let (), events =
    Prof.take (fun () ->
        let s = Sub.make ~reads:0 ~writes:1 ~waits:0 d [||] in
        Sub.write s 0 m;
        let p = C.submit s in
        Prof.after p (fun () ->
            read := true;
            [ span "short" 5 6; span "long" 5 9; span "first" 1 2 ]);
        equal ~msg:"before the wait" bool false !read;
        B.wait m B.Read;
        equal ~msg:"once the wait returned" bool true !read)
  in
  equal (list string) [ "first"; "long"; "short" ] (named events)

let test_after_disabled () =
  let d = memory "profile:after-off" in
  let p = C.submit (empty d) in
  Prof.after p (fun () -> failf "read while no profile is taken");
  C.wait d (C.Point.value p)

(* [record] reads the second and fourth words of its stamps. *)
let test_record () =
  let d, _ = P.open_ "profile:record" in
  let stamps = B.create C.host S.UInt64 4 in
  let words = B.bigarray Bigarray.int64 stamps in
  List.iteri (fun i w -> words.{i} <- Int64.of_int w) [ 0; 100; 0; 250 ];
  let (), events =
    Prof.take (fun () ->
        let p = C.submit (empty d) in
        Prof.record p ~lane:"COMPUTE:0" ~name:"kernel" stamps)
  in
  match events with
  | [ Prof.Span s ] ->
      equal (pair int int) (100, 250) (s.start, s.stop);
      equal string "kernel" s.name
  | _ -> failf "%d events" (List.length events)

let refuses_stamps stamps () =
  let d = memory "profile:record-refusals" in
  let p = C.submit (empty d) in
  raises_match Exn.invalid_arg (fun () ->
      Prof.record p ~lane:"l" ~name:"n" stamps)

(* Copies *)

let test_copy () =
  let d = memory "profile:copy" in
  let src = B.create C.host S.UInt8 100 and dst = B.create d S.UInt8 100 in
  let (), events = Prof.take (fun () -> B.copy ~src ~dst) in
  let copies =
    List.filter_map
      (function
        | Prof.Copy c -> Some (C.name c.src, C.name c.dst, c.bytes) | _ -> None)
      events
  in
  equal
    (list (triple string string int))
    [ ("CPU", "profile:copy", 100) ]
    copies

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
  let events =
    [
      Prof.Span
        {
          device = C.host;
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
        { src = C.host; dst = d; bytes = 64; start = 3_000; stop = 3_200 };
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
    {"ph":"M","pid":2,"tid":0,"name":"process_name","args":{"name":"profile:chrome"}},
    {"ph":"M","pid":2,"tid":2,"name":"thread_name","args":{"name":"COMPUTE:0"}},
    {"ph":"X","pid":2,"tid":2,"ts":1.000,"dur":1.500,"name":"kernel"},
    {"ph":"C","pid":2,"tid":0,"ts":1.500,"name":"memory","args":{"allocated":4096}},
    {"ph":"M","pid":1,"tid":3,"name":"thread_name","args":{"name":"copy"}},
    {"ph":"X","pid":1,"tid":3,"ts":2.000,"dur":0.200,"name":"copy","args":{"to":"profile:chrome","bytes":64}}
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
        test "counters and traces are those the profiles taken ask for"
          test_counters;
      ];
    group ~timeout "after"
      [
        test
          "events after a point are read before the wait that reaches it \
           returns"
          test_after;
        test "events after a point are not read while no profile is taken"
          test_after_disabled;
        test "a recorded span reads its stamps' second and fourth words"
          test_record;
        test "a recorded span refuses three words"
          (refuses_stamps (B.create C.host S.UInt64 3));
        test "a recorded span refuses words of another format"
          (refuses_stamps (B.create C.host S.UInt32 8));
      ];
    group ~timeout "copies"
      [ test "a copy records the bytes it moved" test_copy ];
    group ~timeout "cost"
      [ test "a span while no profile is taken allocates nothing" test_untaken ];
    group ~timeout "chrome" [ test "a trace in Chrome's format" test_chrome ];
  ]

let () = exit (run "device_core.profile" tests)
