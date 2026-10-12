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

let submit ?(buffers = [||]) ?waits s =
  Rig.submit ?waits s ~run:(Sub.Run.make ()) ~buffers

let timeout = 60.
let memory name = Rig.memory_device name
let empty d = Sub.make d [||]

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

(* Device time *)

let functions d =
  require_ok ~pp:Format.pp_print_string (Rig.Image.load d "functions")

let launch ?refs ?(params = 0) image kernel = Sub.launch ?refs image kernel ~params
let bump = Sub.fill ~refs:[| Sub.write 0 ~at:0 |] Support.bump ~params:8

(* Submits [parts] on [d] with zeroed parameters and one group of one thread per
   launch. *)
let submit_parts ?(buffers = [||]) d parts =
  let buffers = Array.map (fun b -> require_some (B.borrow d b)) buffers in
  let s = Sub.make d parts in
  let run = Sub.Run.make () in
  (* Parameters a fresh run holds read 0; a fill's grid is never read. *)
  Array.iteri
    (fun i _ ->
      match Sub.block s i with
      | b ->
          Sub.Run.groups run b 1 1 1;
          Sub.Run.threads run b 1 1 1
      | exception Invalid_argument _ -> ())
    parts;
  Rig.submit s ~run ~buffers

(* The spans of [d] in [events], as [(lane, name, start, stop)]. *)
let device_spans d events =
  List.filter_map
    (function
      | Prof.Span s when Rig.equal s.device d ->
          Some (s.lane, s.name, s.start, s.stop)
      | _ -> None)
    events

(* A submission's span holds its device's times of all its parts' work, read by
   the wait that reaches it, on its first part's queue, named after its
   launches' functions in the order of its parts. *)
let test_device_span () =
  let d, _ = P.open_ "profile:device" in
  let image = functions d and arg = B.create Rig.host (1 lsl 16) in
  let copy =
    launch ~refs:[| Sub.write 0 ~at:0; Sub.read 0 ~at:8 |] image "copy" ~params:24
  in
  let before = Prof.now () in
  let (), events =
    Prof.take (fun () ->
        Rig.Point.wait
          (submit_parts ~buffers:[| arg |] d
             [| launch image "main"; bump; copy |]))
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
  let arg = B.create Rig.host (1 lsl 16) in
  let (), events =
    Prof.take (fun () ->
        Rig.Point.wait (submit_parts ~buffers:[| arg |] d [| bump |]))
  in
  equal
    (list (pair string string))
    [ ("COMPUTE:0", "fills") ]
    (List.map (fun (lane, name, _, _) -> (lane, name)) (device_spans d events))

(* Queue names are driver values, including the empty string. Both compiled
   submissions and copies through that queue retain its name in their span. *)
let test_empty_queue_name () =
  let module Driver = struct
    include P

    let facts p =
      let f = P.facts p in
      let q = List.hd f.queues in
      { f with queues = [ { q with name = "" } ]; copy_engine = Some 0 }
  end in
  let d =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_
         {
           name = "profile:empty-queue";
           driver = (module Driver);
           make = (fun () -> Ok (P.make ~host_visible:false ()));
         })
  in
  let image = functions d in
  let src = B.create d 8 and dst = B.create d 8 in
  let (), events =
    Prof.take (fun () ->
        Rig.Point.wait
          (submit_parts d [| launch image "main" |]);
        B.copy ~src ~dst;
        B.wait dst Rig.Read)
  in
  equal
    (list (pair string string))
    [ ("", "main"); ("", "copies") ]
    (List.map (fun (lane, name, _, _) -> (lane, name)) (device_spans d events))

(* No span: of a submission with no part, of one submitted while no profile is
   taken, and of a device whose driver cannot time its work. *)
let test_no_span () =
  let d, _ = P.open_ "profile:none" and m = memory "profile:none-memory" in
  let arg = B.create Rig.host (1 lsl 16) in
  let earlier = submit_parts ~buffers:[| arg |] d [| bump |] in
  let (), events =
    Prof.take (fun () ->
        Rig.Point.wait (submit (empty d));
        Rig.Point.wait earlier;
        Rig.Point.wait (submit_parts ~buffers:[| arg |] m [| bump |]))
  in
  equal ~msg:"Polled" int 0 (List.length (device_spans d events));
  equal ~msg:"the memory device" int 0 (List.length (device_spans m events))

(* A submission's span goes to every profile taken when it was submitted. *)
let test_span_profiles () =
  let d, _ = P.open_ "profile:device-profiles" in
  let arg = B.create Rig.host (1 lsl 16) in
  let (), outer =
    Prof.take (fun () ->
        let (), inner =
          Prof.take (fun () ->
              Rig.Point.wait (submit_parts ~buffers:[| arg |] d [| bump |]))
        in
        equal ~msg:"inner" int 1 (List.length (device_spans d inner)))
  in
  equal ~msg:"outer" int 1 (List.length (device_spans d outer))

(* A profile leaves out the span of a device lost before it was read. *)
let test_span_lost () =
  let d, p = P.open_ "profile:device-lost" in
  let arg = B.create Rig.host (1 lsl 16) in
  let (), events =
    Prof.take (fun () ->
        ignore (submit_parts ~buffers:[| arg |] d [| bump |]);
        P.fail p;
        try ignore (submit (Sub.make d [||])) with Rig.Lost _ -> ())
  in
  equal (list string) [] (named events)

(* A settle that fails loses the device, with the driver's reason, and the
   profile has no span and no copy of the value it settled. *)
let test_settle_fails () =
  let d, p = P.open_ "profile:settle-fails" in
  let host () = B.create Rig.host (1 lsl 16) in
  let copy = Sub.copy ~src:(Slot 1) ~dst:(Slot 2) in
  P.fail_settle p;
  let (), events =
    Prof.take (fun () ->
        Rig.Point.wait
          (submit_parts
             ~buffers:[| host (); host (); host () |]
             d [| bump; copy |]))
  in
  equal (list string) [] (named events);
  equal ~msg:"copies" (list int) []
    (List.filter_map
       (function Prof.Copy c -> Some c.bytes | _ -> None)
       events);
  equal (option string) (Some "the settle failed") (Rig.lost d)

(* A pending timing callback may outlive its failed profile. It must keep no
   discarded image binary, even when loss prevents the callback from running. *)
let test_failed_history failure =
  let interrupted = Atomic.make false in
  let module Driver = struct
    include P

    let image d binary =
      if String.length binary = 0 then Error "empty image"
      else P.image d "functions"

    let sleep d ~seen ~still_ms =
      if Atomic.exchange interrupted false then raise Exit;
      P.sleep d ~seen ~still_ms
  end in
  let p = P.make () in
  let name =
    "profile:discard-"
    ^ match failure with `Body -> "body" | `Lost -> "loss" | `Read -> "read"
  in
  let d =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_ { name; driver = (module Driver); make = (fun () -> Ok p) })
  in
  let arg = B.create Rig.host (1 lsl 16) in
  Rig.Point.wait (submit_parts ~buffers:[| arg |] d [| bump |]);
  let weak = Weak.create 1 and point = ref None in
  raises Exit (fun () ->
      Prof.take (fun () ->
          let binary = String.make (1024 * 1024) 'x' in
          Weak.set weak 0 (Some binary);
          ignore
            (require_ok ~pp:Format.pp_print_string (Rig.Image.load d binary));
          let value = submit_parts ~buffers:[| arg |] d [| bump |] in
          point := Some value;
          match failure with
          | `Body -> raise Exit
          | `Read -> Atomic.set interrupted true
          | `Lost ->
              P.fault p "lost";
              (try Rig.Point.wait value with Rig.Lost _ -> ());
              raise Exit));
  for _ = 1 to 3 do
    Gc.full_major ()
  done;
  equal ~msg:"discarded image binary is released" bool false (Weak.check weak 0);
  equal ~msg:"the failed profile is closed" bool false (Prof.enabled ());
  if failure <> `Lost then Rig.Point.wait (require_some !point)

(* The failed inner profile's callback still records to its outer recipient. *)
let test_failed_recipient () =
  let d, _ = P.open_ "profile:failed-recipient" in
  let arg = B.create Rig.host (1 lsl 16) in
  let (), events =
    Prof.take (fun () ->
        raises Exit (fun () ->
            Prof.take (fun () ->
                ignore (submit_parts ~buffers:[| arg |] d [| bump |]);
                raise Exit)))
  in
  equal (list string) [ "fills" ] (named events);
  let (), events =
    Prof.take (fun () -> ignore (submit_parts ~buffers:[| arg |] d [| bump |]))
  in
  equal ~msg:"later timings still work" (list string) [ "fills" ] (named events)

(* A host span can finish on another domain after one recipient closes. Its
   other recipient survives, while the closed one's earlier payload is freed. *)
let test_late_span () =
  let entered = Atomic.make false and finish = Atomic.make false in
  let other = ref None and weak = Weak.create 1 in
  Fun.protect
    ~finally:(fun () ->
      Atomic.set finish true;
      Option.iter (fun d -> ignore (Domain.join d)) !other)
    (fun () ->
      raises Exit (fun () ->
          Prof.take (fun () ->
              let name = String.make (1024 * 1024) 'x' in
              Weak.set weak 0 (Some name);
              Prof.span name ignore;
              other :=
                Some
                  (Domain.spawn (fun () ->
                       Prof.take (fun () ->
                           Prof.span "late" (fun () ->
                               Atomic.set entered true;
                               Support.await "the failed profile closing"
                                 (fun () -> Atomic.get finish)))));
              Support.await "the other recipient" (fun () -> Atomic.get entered);
              raise Exit));
      for _ = 1 to 3 do
        Gc.full_major ()
      done;
      equal ~msg:"closed history is released before the span finishes" bool
        false (Weak.check weak 0);
      Atomic.set finish true;
      let domain = require_some !other in
      other := None;
      let (), events = Domain.join domain in
      equal (list string) [ "late" ] (named events))

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
   device-to-host copy's leg into staging stops when the device ended it, after
   its gate opened. *)
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

(* [n] bytes of a host buffer borrowed by [d]. *)
let borrowed d n =
  require_some
    (B.borrow d (B.view (B.create Rig.host (1 lsl 16)) ~first:0 ~length:n))

let copy_parts =
  Sub.[| copy ~src:(Slot 0) ~dst:(Slot 1); copy ~src:(Slot 2) ~dst:(Slot 3) |]

(* The times of [events]' copies. *)
let copy_times events =
  List.filter_map
    (function Prof.Copy c -> Some (c.start, c.stop) | _ -> None)
    events

(* Each copy part records a copy between the devices of the allocations its
   bytes leave and reach, a borrow's naming the host, with their count. On a
   device that times no work, it runs from the submit to the wait that saw it
   done. *)
let test_copy_parts () =
  let m = memory "profile:copy-parts" in
  let buffers =
    [| borrowed m 64; B.create m 64; B.create m 32; B.create m 32 |]
  in
  let before = Prof.now () in
  let stop = ref 0 in
  let (), events =
    Prof.take (fun () ->
        Rig.Point.wait (submit ~buffers (Sub.make m copy_parts));
        stop := Prof.now ())
  in
  equal
    (list (triple string string int))
    [
      ("CPU", "profile:copy-parts", 64);
      ("profile:copy-parts", "profile:copy-parts", 32);
    ]
    (copies events);
  List.iter
    (fun (start, stop') ->
      at_least ~msg:"the start" int ~than:before start;
      at_least ~msg:"the stop" int ~than:start stop';
      at_most ~msg:"the stop" int ~than:!stop stop')
    (copy_times events)

(* On a device that times its work, each copy part has its submission's span's
   times. *)
let test_timed_copy_parts () =
  let d, _ = P.open_ "profile:timed-copy-parts" in
  let buffers =
    [| borrowed d 64; B.create d 64; B.create d 32; B.create d 32 |]
  in
  let (), events =
    Prof.take (fun () ->
        Rig.Point.wait (submit ~buffers (Sub.make d copy_parts)))
  in
  equal
    (list (triple string string int))
    [
      ("CPU", "profile:timed-copy-parts", 64);
      ("profile:timed-copy-parts", "profile:timed-copy-parts", 32);
    ]
    (copies events);
  match device_spans d events with
  | [ (_, _, start, stop) ] ->
      equal
        (list (pair int int))
        [ (start, stop); (start, stop) ]
        (copy_times events)
  | spans -> failf "%d spans of the device" (List.length spans)

(* No copy of a submission submitted while no profile is taken. *)
let test_untaken_copy_parts () =
  let m = memory "profile:untaken-copy-parts" in
  let buffers =
    [| borrowed m 64; B.create m 64; B.create m 32; B.create m 32 |]
  in
  let p = submit ~buffers (Sub.make m copy_parts) in
  let (), events = Prof.take (fun () -> Rig.Point.wait p) in
  equal (list (triple string string int)) [] (copies events)

(* Cost *)

let test_untaken () =
  let f = Sys.opaque_identity ignore in
  Prof.span "warm" f;
  let before = Gc.minor_words () in
  for _ = 1 to 100 do
    Prof.span "idle" f
  done;
  equal int 0 (int_of_float (Gc.minor_words () -. before))

(* Recording *)

(* A recorded event goes into each profile being taken: both of two nested
   profiles, the outer one alone before and after the inner, none while no
   profile is taken, and one taken on another domain, each in time order
   among rig's own spans. *)
let test_record () =
  let d = memory "profile:record" in
  let counters start =
    Prof.Counters
      { device = d; name = "kernel"; start; stop = start + 1; counters = [] }
  in
  let starts =
    List.filter_map (function Prof.Counters c -> Some c.start | _ -> None)
  in
  Prof.record (counters 0);
  let (now, ((), inner)), outer =
    Prof.take (fun () ->
        let now = Prof.now () in
        Prof.record (counters (now + 1));
        let r = Prof.take (fun () -> Prof.record (counters (now + 2))) in
        Prof.record (counters (now + 3));
        Domain.join (Domain.spawn (fun () -> Prof.record (counters (now + 4))));
        (now, r))
  in
  equal ~msg:"the inner profile" (list int) [ now + 2 ] (starts inner);
  equal ~msg:"the outer profile" (list int)
    [ now + 1; now + 2; now + 3; now + 4 ]
    (starts outer);
  let (), events =
    Prof.take (fun () ->
        Prof.span "before" ignore;
        Prof.record (counters (Prof.now () + 1_000_000));
        Prof.record (counters (Prof.now () - 1_000_000));
        Prof.span "after" ignore)
  in
  equal ~msg:"in time order among spans" (list string)
    [ "counters"; "before"; "after"; "counters" ]
    (List.map
       (function
         | Prof.Span s -> s.name
         | Prof.Counters _ -> "counters"
         | _ -> "other")
       events)

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
      ];
    group ~timeout "device time"
      [
        test
          "a submission's span holds its parts' device time, named after its \
           launches"
          test_device_span;
        test "a submission with no launch is named after its first part's kind"
          test_kind_span;
        test "submissions and copies record an empty queue name"
          test_empty_queue_name;
        test
          "no span of an empty submission, an untimed one or an untimeable \
           device"
          test_no_span;
        test "a span goes to the profiles taken at its submit"
          test_span_profiles;
        test "a profile leaves out a lost device's unread span" test_span_lost;
        test "a settle that fails loses the device" test_settle_fails;
        cases ~name:fst "failed histories release their image binaries"
          [ ("body", `Body); ("loss", `Lost); ("pending read", `Read) ]
          (fun (_, failure) -> test_failed_history failure);
        test "a failed recipient preserves another profile's device timing"
          test_failed_recipient;
        test "a late span preserves its remaining recipient" test_late_span;
      ];
    group ~timeout "copies"
      [
        test "a copy records the bytes it moved" test_copy;
        test "a staged copy records only its copies into and out of staging"
          test_staged_copy;
        test "a staged copy's queued leg spans its transfer" test_staged_span;
        test "a submission records each copy part, with its bytes"
          test_copy_parts;
        test "a copy part has its submission's span's times"
          test_timed_copy_parts;
        test "no copy part of a submission made before the profile"
          test_untaken_copy_parts;
      ];
    group ~timeout "cost"
      [ test "a span while no profile is taken allocates nothing" test_untaken ];
    group ~timeout "recording"
      [
        test "a recorded event goes into each profile being taken"
          test_record;
      ];
    group ~timeout "chrome" [ test "a trace in Chrome's format" test_chrome ];
  ]

let () = exit (run "rig.profile" tests)
