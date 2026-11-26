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

let submit ?(buffers = [||]) ?waits s = Rig.submit ?waits s ~buffers ~args:[||]

(* Whether an exception is [d]'s loss. *)
let lost_of d = function Rig.Lost (d', _) -> Rig.equal d d' | _ -> false
let software name = Result.get_ok (Rig.software name)
let empty d = Sub.make d [||]
let device = Testable.make ~pp:Rig.pp ~equal:Rig.equal
let point = Testable.make ~pp:Rig.Point.pp ~equal:( = )

(* Whether a profile is being taken, as words. *)
let taking () =
  if Prof.enabled () then "some profile taken" else "no profile taken"

(* The spans of [events], as [(lane, name)]. *)
let spans events =
  List.filter_map
    (function Prof.Span s -> Some (s.lane, s.name) | _ -> None)
    events

(* The names of [events]' spans and works. *)
let named events =
  List.filter_map
    (function
      | Prof.Span s -> Some s.name | Prof.Work w -> Some w.name | _ -> None)
    events

(* Holds a domain at a stage until another domain reaches it. *)
let stages () =
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
  (at, next)

let functions d =
  require_ok ~pp:Format.pp_print_string (Rig.Image.load d "functions")

(* The values a submit here passes each launch and fill: one group of one
   thread, and parameters that read 0. A fill reads only the parameters. *)
let one =
  {
    Sub.groups = (1, 1, 1);
    threads = (1, 1, 1);
    shared = 0;
    params = Bytes.empty;
  }

(* Parts, each with its values, for a launch or a fill. *)
let launch ?refs ?(params = 0) image kernel =
  ( Sub.launch ?refs (require_some (Rig.Image.resolve image kernel)) ~params,
    Some one )

let bump =
  (Sub.fill ~refs:[| Sub.write 0 ~at:0 |] Support.bump ~params:8, Some one)

(* Submits [parts] on [d] with their values. *)
let submit_parts ?(buffers = [||]) d parts =
  let buffers = Array.map (fun b -> require_some (B.borrow d b)) buffers in
  let s = Sub.make d (Array.map fst parts) in
  let args = Array.of_list (List.filter_map snd (Array.to_list parts)) in
  Rig.submit s ~buffers ~args

(* A timed fill of a host buffer on [d], waited for. *)
let fill d =
  Rig.Point.wait
    (submit_parts ~buffers:[| B.create Rig.host B.paged_from |] d [| bump |])

(* The work of [d] in [events], as [(lane, name, start, stop)]. *)
let device_spans d events =
  List.filter_map
    (function
      | Prof.Work w when Rig.equal w.device d ->
          Some (w.lane, w.name, w.start, w.stop)
      | _ -> None)
    events

(* The names of [d]'s work in [events]. *)
let work_names d events =
  List.map (fun (_, name, _, _) -> name) (device_spans d events)

(* The points of [d]'s work in [events]. *)
let work_points d events =
  List.filter_map
    (function
      | Prof.Work w when Rig.equal w.device d -> Some w.point | _ -> None)
    events

let copies events =
  List.filter_map
    (function
      | Prof.Copy c -> Some (Rig.name c.src, Rig.name c.dst, c.bytes)
      | _ -> None)
    events

(* The points of [events]' copies. *)
let copy_points events =
  List.filter_map (function Prof.Copy c -> Some c.point | _ -> None) events

(* The times of [events]' copies. *)
let copy_times events =
  List.filter_map
    (function Prof.Copy c -> Some (c.start, c.stop) | _ -> None)
    events

(* A counters event of [d] named [name], from [start] for [length] ns. *)
let counters d ?(name = "kernel") ?(length = 1) start =
  Prof.Counters
    { device = d; name; start; stop = start + length; counters = [] }

(* What [f ()] ends with: a return, or Ctrl-C's [Sys.Break], which windtrap's
   [raises] would take as the end of the run. *)
let broke f =
  match f () with _ -> "a return" | exception Sys.Break -> "Sys.Break"

(* Profiles *)

(* A span the second domain records while the first's profile is taken is in
   that profile. *)
let test_two_domains () =
  let at, next = stages () in
  let lane () = Printf.sprintf "domain %d" (Domain.self () :> int) in
  let other =
    Domain.spawn (fun () ->
        at 1;
        Prof.span "theirs" ignore;
        next ();
        lane ())
  in
  let (), events =
    Prof.take (fun () ->
        next ();
        at 2;
        Prof.span "mine" ignore)
  in
  let theirs = Domain.join other in
  equal
    (slist (pair string string) compare)
    [ (lane (), "mine"); (theirs, "theirs") ]
    (spans events)

let test_overlapping () =
  let at, next = stages () in
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

let test_submit_profiles () =
  let d, _ = P.open_ "profile:device-profiles" in
  let (), outer =
    Prof.take (fun () ->
        let (), inner = Prof.take (fun () -> fill d) in
        equal ~msg:"inner" (list string) [ "fills" ] (work_names d inner))
  in
  equal ~msg:"outer" (list string) [ "fills" ] (work_names d outer)

(* A host copy that waits for a device's write while another domain starts a
   profile is in the profile around the copy, and not in the later one. *)
let test_copy_asked () =
  let d, p = P.open_ "profile:copy-asked" in
  let n = B.paged_from in
  let src = B.create Rig.host n and dst = B.create Rig.host n in
  P.gate p;
  ignore (submit_parts ~buffers:[| src |] d [| bump |]);
  let copier =
    Domain.spawn (fun () -> Prof.take (fun () -> B.copy ~src ~dst))
  in
  Support.await "the copy's wait" (fun () -> P.sleepers p = 1);
  let ((), around), later =
    Prof.take (fun () ->
        P.open_gate p;
        Domain.join copier)
  in
  equal ~msg:"the profile around it"
    (list (triple string string int))
    [ ("CPU", "CPU", n) ]
    (copies around);
  equal ~msg:"the later profile"
    (list (triple string string int))
    [] (copies later)

(* [n] bytes of a host buffer borrowed by [d]. *)
let borrowed d n =
  require_some
    (B.borrow d (B.view (B.create Rig.host B.paged_from) ~first:0 ~length:n))

let copy_parts =
  Sub.[| copy ~src:(Slot 0) ~dst:(Slot 1); copy ~src:(Slot 2) ~dst:(Slot 3) |]

(* A timed fill and a submission of copy parts, both asked before the profile
   and waited for inside it. *)
let test_asked_before () =
  let d, _ = P.open_ "profile:asked-before" in
  let m = software "profile:asked-before-copies" in
  let filled =
    submit_parts ~buffers:[| B.create Rig.host B.paged_from |] d [| bump |]
  in
  let copied =
    submit
      ~buffers:[| borrowed m 64; B.create m 64; B.create m 32; B.create m 32 |]
      (Sub.make m copy_parts)
  in
  let (), events =
    Prof.take (fun () ->
        Rig.Point.wait filled;
        Rig.Point.wait copied)
  in
  equal ~msg:"work" (list string) [] (work_names d events);
  equal ~msg:"copies" (list (triple string string int)) [] (copies events)

let test_idle_words () =
  let f = Sys.opaque_identity ignore and e = counters Rig.host 0 in
  Prof.span "warm" f;
  Prof.record e;
  equal int 0
    (Support.allocated_bytes (fun () ->
         for _ = 1 to 100 do
           Prof.span "idle" f;
           Prof.record e
         done))

(* Work *)

let test_work () =
  let d, _ = P.open_ "profile:device" in
  let image = functions d and arg = B.create Rig.host B.paged_from in
  let copy =
    launch
      ~refs:[| Sub.write 0 ~at:0; Sub.read 0 ~at:8 |]
      image "copy" ~params:24
  in
  let before = Prof.now () and submitted = ref None in
  let (), events =
    Prof.take (fun () ->
        let p =
          submit_parts ~buffers:[| arg |] d
            [| launch image "main"; bump; copy |]
        in
        submitted := Some p;
        Rig.Point.wait p)
  in
  let after = Prof.now () in
  equal ~msg:"the submit's point" (list point)
    [ require_some !submitted ]
    (work_points d events);
  match device_spans d events with
  | [ (lane, name, start, stop) ] ->
      equal (pair string string) ("COMPUTE:0", "main, copy") (lane, name);
      at_least ~msg:"the start" int ~than:before start;
      at_least ~msg:"the stop" int ~than:start stop;
      at_most ~msg:"the stop" int ~than:after stop
  | spans -> failf "%d spans of the device" (List.length spans)

(* Parts of no launch: a fill, a copy between slots 1 and 2, and words. *)
let kinds =
  [
    ("fills", [ `Fill; `Copy ]);
    ("copies", [ `Copy; `Fill ]);
    ("words", [ `Words; `Fill; `Copy ]);
  ]

let test_kind (kind, parts) =
  let part = function
    | `Fill -> bump
    | `Copy -> (Sub.copy ~src:(Slot 1) ~dst:(Slot 2), None)
    | `Words -> (Sub.words (B.create Rig.host 8), None)
  in
  let parts = Array.of_list (List.map part parts) in
  let d, _ = P.open_ ("profile:kind-" ^ kind) in
  let host () = B.create Rig.host B.paged_from in
  let (), events =
    Prof.take (fun () ->
        Rig.Point.wait
          (submit_parts ~buffers:[| host (); host (); host () |] d parts))
  in
  equal (list string) [ kind ] (work_names d events)

(* Queue names are driver values, the empty string included. *)
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
           make =
             (fun ~word -> Ok (P.bind (P.make ~host_visible:false ()) ~word));
         })
  in
  let image = functions d in
  let src = B.create d 8 and dst = B.create d 8 in
  let (), events =
    Prof.take (fun () ->
        Rig.Point.wait (submit_parts d [| launch image "main" |]);
        B.copy ~src ~dst;
        B.wait dst Rig.Read)
  in
  equal
    (list (pair string string))
    [ ("", "main"); ("", "copies") ]
    (List.map (fun (lane, name, _, _) -> (lane, name)) (device_spans d events))

(* Devices that record no Work, each with the submission it records none of. *)
let unrecorded =
  [
    ( "with no parts",
      fun () ->
        let d, _ = P.open_ "profile:none" in
        (d, fun () -> Rig.Point.wait (submit (empty d))) );
    ( "of a driver that cannot time its work",
      fun () ->
        let d, _ = P.open_ ~timed:false "profile:none-untimed" in
        (d, fun () -> fill d) );
  ]

let test_unrecorded (_, make) =
  let d, run = make () in
  let (), events = Prof.take run in
  equal (list string) [] (work_names d events)

(* The device's Polled driver times its work, on this host's clock. *)
let test_far () =
  let machine = Support.machine "profile:none-far" in
  let d =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_ ~machine (P.opener "profile:none-far-gpu" (P.make ())))
  in
  let src = B.create d 8 and dst = B.create d 8 in
  let copy = Sub.make d [| Sub.copy ~src:(Buffer src) ~dst:(Buffer dst) |] in
  let (), events = Prof.take (fun () -> Rig.Point.wait (submit copy)) in
  equal (list string) [] (work_names d events)

let test_software_work () =
  let d = software "profile:software" in
  let (), events = Prof.take (fun () -> fill d) in
  equal
    (list (pair string string))
    [ ("COMPUTE:0", "fills") ]
    (List.map (fun (lane, name, _, _) -> (lane, name)) (device_spans d events))

(* Span *)

let test_span () =
  let (), events = Prof.take (fun () -> Prof.span "work" ignore) in
  match events with
  | [ Prof.Span s ] ->
      equal
        (triple device string string)
        (Rig.host, "domain 0", "work")
        (s.device, s.lane, s.name);
      at_most int ~than:s.stop s.start
  | _ -> failf "%d events" (List.length events)

(* Allocation *)

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
  equal ~msg:"the last allocation event" (option int)
    (Some (Rig.allocated d))
    (List.fold_left (fun _ a -> Some a) None allocations);
  ignore (Sys.opaque_identity b)

(* Load *)

let image =
  Testable.make ~equal:( == ) ~pp:(fun ppf i ->
      Format.fprintf ppf "an image on %a" Rig.pp (Rig.Image.device i))

let test_load () =
  let d, _ = P.open_ "profile:load" in
  let i, events =
    Prof.take (fun () ->
        require_ok ~pp:Format.pp_print_string (Rig.Image.load d "code:64"))
  in
  let loads =
    List.filter_map
      (function Prof.Load l -> Some (l.image, l.binary) | _ -> None)
      events
  in
  equal (list (pair image string)) [ (i, "code:64") ] loads

(* Copy *)

let test_copy () =
  let d = software "profile:copy" in
  let src = B.create Rig.host 100 and dst = B.create d 100 in
  let (), events = Prof.take (fun () -> B.copy ~src ~dst) in
  equal
    (list (triple string string int))
    [ ("CPU", "profile:copy", 100) ]
    (copies events);
  equal ~msg:"the host ran it"
    (list (option point))
    [ None ] (copy_points events)

let test_io_copy () =
  let io = Support.io "profile:io" in
  let src = Support.io_buffer io 64 and dst = B.create Rig.host 64 in
  let (), events = Prof.take (fun () -> B.copy ~src ~dst) in
  equal
    (list (triple string string int))
    [ ("profile:io", "CPU", 64) ]
    (copies events);
  equal ~msg:"its point" (list (option point)) [ None ] (copy_points events)

(* Three rows of 4 bytes, 8 bytes apart: 12 bytes. *)
let test_box () =
  let src = B.create Rig.host 32 and dst = B.create Rig.host 32 in
  let box =
    {
      B.src_offset = 0;
      dst_offset = 0;
      width = 4;
      rows = 3;
      src_pitch = 8;
      dst_pitch = 8;
    }
  in
  let (), events =
    Prof.take (fun () -> Rig.Point.wait (B.submit_copy ~src ~dst [| box |]))
  in
  equal (list (triple string string int)) [ ("CPU", "CPU", 12) ] (copies events)

(* Between devices that map none of each other's memory: each piece that ran is
   one event. *)
let test_staged_copy () =
  let open_ name = P.open_ ~host_visible:false ~peers:false name in
  let d, _ = open_ "profile:staged-src" and e, _ = open_ "profile:staged-dst" in
  let src = B.create d 64 and dst = B.create e 64 in
  let (), events = Prof.take (fun () -> B.copy ~src ~dst) in
  equal
    (slist (triple string string int) compare)
    [ ("CPU", "profile:staged-dst", 64); ("profile:staged-src", "CPU", 64) ]
    (copies events)

(* On a device that maps no host memory, its staging is Pinned memory, which
   counts in the host's budget where the host does not address the device's
   memory. *)
let test_device_staged_copy () =
  let d, _ =
    P.open_ ~host_visible:false ~maps_host:false "profile:device-staged"
  in
  let src = B.create Rig.host 64 and dst = B.create d 64 in
  let (), events = Prof.take (fun () -> B.copy ~src ~dst) in
  equal
    (slist (triple string string int) compare)
    [ ("CPU", "CPU", 64); ("CPU", "profile:device-staged", 64) ]
    (copies events)

(* A device-to-host copy's leg into staging stops when the device ended it,
   after its gate opened. *)
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

(* A borrow's part names the host, whose memory it borrows. *)
let test_copy_parts () =
  let m, _ = P.open_ ~timed:false "profile:copy-parts" in
  let buffers =
    [| borrowed m 64; B.create m 64; B.create m 32; B.create m 32 |]
  in
  let before = Prof.now () in
  let asked = ref 0 and submitted = ref None in
  let (), events =
    Prof.take (fun () ->
        let p = submit ~buffers (Sub.make m copy_parts) in
        asked := Prof.now ();
        submitted := Some p;
        Rig.Point.wait p)
  in
  let after = Prof.now () in
  equal ~msg:"the submission's point"
    (list (option point))
    [ !submitted; !submitted ] (copy_points events);
  equal
    (list (triple string string int))
    [
      ("CPU", "profile:copy-parts", 64);
      ("profile:copy-parts", "profile:copy-parts", 32);
    ]
    (copies events);
  List.iter
    (fun (start, stop) ->
      at_least ~msg:"the start" int ~than:before start;
      at_most ~msg:"the start" int ~than:!asked start;
      at_least ~msg:"the stop" int ~than:start stop;
      at_most ~msg:"the stop" int ~than:after stop)
    (copy_times events)

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

(* The host does not address the buffer, so the device's copy engine fills
   it. *)
let test_fill () =
  let d, _ = P.open_ ~host_visible:false "profile:fill" in
  let b = B.create d 64 in
  let (), events = Prof.take (fun () -> Rig.Point.wait (B.submit_fill b 'x')) in
  equal ~msg:"copies" (list (triple string string int)) [] (copies events);
  equal ~msg:"work" (list string) [ "copies" ] (work_names d events)

(* take *)

(* Counters events recorded in the drawn order, each [(start, length)] from one
   origin, named by their place in that order. *)
let test_order drawn =
  let origin = Prof.now () in
  let named =
    List.mapi (fun i (start, length) -> (string_of_int i, start, length)) drawn
  in
  let tie (_, s, l) (_, s', l') = s = s' && l <> l'
  and twin (n, s, l) (n', s', l') = n <> n' && s = s' && l = l' in
  let some p = List.exists (fun e -> List.exists (p e) named) named in
  cover "two events at one time of two lengths" (some tie);
  cover "two events at one time of one length" (some twin);
  let (), events =
    Prof.take (fun () ->
        List.iter
          (fun (name, start, length) ->
            Prof.record (counters Rig.host ~name ~length (origin + start)))
          named)
  in
  (* Time order, then longest first, then the order recorded. *)
  let expected =
    List.stable_sort
      (fun (_, s, l) (_, s', l') ->
        match Int.compare s s' with 0 -> Int.compare l' l | c -> c)
      named
  in
  equal (list string)
    (List.map (fun (n, _, _) -> n) expected)
    (List.filter_map
       (function Prof.Counters c -> Some c.name | _ -> None)
       events)

let[@inline never] [@local never] raiser () = raise Exit

(* The backtrace of [f ()]'s exception, as locations. *)
let backtrace f =
  match f () with
  | _ -> failf "no exception"
  | exception Exit ->
      Printexc.get_raw_backtrace ()
      |> Printexc.backtrace_slots |> Option.value ~default:[||] |> Array.to_list
      |> List.filter_map Printexc.Slot.location
      |> List.map (fun (l : Printexc.location) -> (l.filename, l.line_number))

(* Its first location is that of [raiser]'s raise, as when [raiser] raises
   alone. *)
let test_reraise () =
  let recording = Printexc.backtrace_status () in
  Printexc.record_backtrace true;
  Fun.protect
    ~finally:(fun () -> Printexc.record_backtrace recording)
    (fun () ->
      let first f = List.nth_opt (backtrace f) 0 in
      let raised = require_some ~msg:"raiser's location" (first raiser) in
      equal
        (option (pair string int))
        (Some raised)
        (first (fun () -> Prof.take raiser)))

(* A pending read of a profile that raised keeps no binary loaded while it was
   taken, even when a loss stops the read. *)
let test_failed_history (_, failure) =
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
      (Rig.open_
         {
           name;
           driver = (module Driver);
           make = (fun ~word -> Ok (P.bind p ~word));
         })
  in
  let arg = B.create Rig.host B.paged_from in
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
  equal ~msg:"the binary's bytes still reachable" (option int) None
    (Option.map String.length (Weak.get weak 0));
  equal string "no profile taken" (taking ());
  if failure <> `Lost then Rig.Point.wait (require_some !point)

let test_failed_recipient () =
  let d, _ = P.open_ "profile:failed-recipient" in
  let arg = B.create Rig.host B.paged_from in
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
  equal ~msg:"a later profile" (list string) [ "fills" ] (named events)

(* The device's settles wait at a gate, which opens once [take] returned, or
   once the wait returned and a settle waits: a wait that read the work's times
   itself is still held there as [take] returns. *)
let test_reached_elsewhere () =
  let d, p = P.open_ ~may_block:true "profile:reached-elsewhere" in
  let arg = B.create Rig.host B.paged_from in
  P.gate_settle p;
  let waited = Atomic.make false and taken = Atomic.make false in
  let take () =
    let r =
      Prof.take (fun () ->
          let pt = submit_parts ~buffers:[| arg |] d [| bump |] in
          ignore (P.run p);
          let waiter =
            Domain.spawn (fun () ->
                Rig.Point.wait pt;
                Atomic.set waited true)
          in
          Support.await "the wait's return, or its settle" (fun () ->
              Atomic.get waited || P.settlers p > 0);
          waiter)
    in
    Atomic.set taken true;
    r
  in
  let taker = Domain.spawn take in
  Support.await "the take's return, or its settle" (fun () ->
      Atomic.get taken || (Atomic.get waited && P.settlers p > 0));
  P.open_settle p;
  let waiter, events = Domain.join taker in
  Domain.join waiter;
  equal (list string) [ "fills" ] (named events)

(* A submit that another domain makes while [take] waits for its own work adds
   nothing to the profile, and [take] does not wait for it. *)
let test_reads_its_own () =
  let d, p = P.open_ "profile:its-own" in
  let arg = B.create Rig.host B.paged_from in
  P.gate p;
  let taker =
    Domain.spawn (fun () ->
        Prof.take (fun () -> submit_parts ~buffers:[| arg |] d [| bump |]))
  in
  Support.await "take's wait" (fun () -> P.sleepers p = 1);
  let later = submit_parts ~buffers:[| arg |] d [| bump |] in
  P.open_gate p;
  let _, events = Domain.join taker in
  ignore (P.run p);
  Rig.Point.wait later;
  equal (list string) [ "fills" ] (named events)

(* The device is lost by a failed submit, after a timed fill whose events the
   take has yet to read, on the domain [on] runs the two on. *)
let asking =
  [
    ("this domain", fun f -> f ());
    ("another domain", fun f -> Domain.join (Domain.spawn f));
  ]

let test_lost (_, on) =
  let d, p = P.open_ "profile:device-lost" in
  let arg = B.create Rig.host B.paged_from in
  raises_match (lost_of d) (fun () ->
      Prof.take (fun () ->
          on (fun () ->
              ignore (submit_parts ~buffers:[| arg |] d [| bump |]);
              P.fail p;
              try ignore (submit (Sub.make d [||])) with Rig.Lost _ -> ())));
  equal string "no profile taken" (taking ())

let test_settle_fails () =
  let d, p = P.open_ "profile:settle-fails" in
  let host () = B.create Rig.host B.paged_from in
  let copy = (Sub.copy ~src:(Slot 1) ~dst:(Slot 2), None) in
  P.fail_settle p;
  raises_match (lost_of d) (fun () ->
      Prof.take (fun () ->
          Rig.Point.wait
            (submit_parts
               ~buffers:[| host (); host (); host () |]
               d [| bump; copy |])));
  equal (option string) (Some "the settle failed") (Rig.lost d)

(* The second submission is never waited for. *)
let test_closed () =
  let d, _ = P.open_ "profile:device-closed" in
  let arg = B.create Rig.host B.paged_from in
  let (), events =
    Prof.take (fun () ->
        Rig.Point.wait (submit_parts ~buffers:[| arg |] d [| bump |]);
        ignore (submit_parts ~buffers:[| arg |] d [| bump |]);
        Rig.close d)
  in
  equal ~msg:"the closed device's work" (list string) [ "fills"; "fills" ]
    (work_names d events);
  equal ~msg:"its loss" (option string) (Some "closed") (Rig.lost d)

(* A close interrupted in its wait for the first entry it reads drops that
   entry, and the take reads the other; the device stays open. *)
let test_close_interrupted () =
  let d, p = P.open_ "profile:close-interrupted" in
  let arg = B.create Rig.host B.paged_from in
  let (), events =
    Prof.take (fun () ->
        ignore (submit_parts ~buffers:[| arg |] d [| bump |]);
        ignore (submit_parts ~buffers:[| arg |] d [| bump |]);
        P.interrupt p;
        Sys.catch_break true;
        Fun.protect
          ~finally:(fun () -> Sys.catch_break false)
          (fun () ->
            equal ~msg:"the close" string "Sys.Break"
              (broke (fun () -> Rig.close d))))
  in
  equal ~msg:"the device's work" (list string) [ "fills" ] (work_names d events);
  equal ~msg:"its loss" (option string) None (Rig.lost d)

(* The device's next timed submit, once the device ran the dropped work, takes
   no time pair before rig settled the value it timed last. *)
let test_break_in_take () =
  let d, p = P.open_ "profile:break-take" in
  let arg = B.create Rig.host B.paged_from in
  let seen = List.length (P.violations ()) in
  P.interrupt p;
  Sys.catch_break true;
  Fun.protect
    ~finally:(fun () -> Sys.catch_break false)
    (fun () ->
      equal ~msg:"the take" string "Sys.Break"
        (broke (fun () ->
             Prof.take (fun () -> submit_parts ~buffers:[| arg |] d [| bump |]))));
  ignore (P.run p);
  let _, events =
    Prof.take (fun () ->
        let pt = submit_parts ~buffers:[| arg |] d [| bump |] in
        ignore (P.run p);
        pt)
  in
  equal ~msg:"breaches" (list string) [] (List.drop seen (P.violations ()));
  equal (list string) [ "fills" ] (named events)

(* The device may still write the dropped work's time pair: a later timed
   submit's times land elsewhere, and its work lies within its profile. *)
let test_stopped_keeps_pair () =
  let d, p = P.open_ "profile:stopped-pair" in
  let arg = B.create Rig.host B.paged_from in
  let timed () = submit_parts ~buffers:[| arg |] d [| bump |] in
  raises Exit (fun () ->
      Prof.take (fun () ->
          ignore (timed ());
          raise Exit));
  let dropped = P.last_times p in
  let before = Prof.now () in
  let (), events =
    Prof.take (fun () ->
        let pt = timed () in
        not_equal ~msg:"where a later submit's times land" int dropped
          (P.last_times p);
        Rig.Point.wait pt)
  in
  let after = Prof.now () in
  match device_spans d events with
  | [ (_, _, start, stop) ] ->
      at_least ~msg:"the start" int ~than:before start;
      at_least ~msg:"the stop" int ~than:start stop;
      at_most ~msg:"the stop" int ~than:after stop
  | spans -> failf "%d spans of the device" (List.length spans)

(* Once the device ran the dropped work, a later timed submit takes its time
   pair only after the driver settled the work's value, so the driver writes
   that pair for one value at a time. *)
let test_settled_before_reuse () =
  let d, p = P.open_ "profile:settled-reuse" in
  let arg = B.create Rig.host B.paged_from in
  let timed () = submit_parts ~buffers:[| arg |] d [| bump |] in
  let seen = List.length (P.violations ()) in
  raises Exit (fun () ->
      Prof.take (fun () ->
          ignore (timed ());
          raise Exit));
  ignore (P.run p);
  let (), events = Prof.take (fun () -> Rig.Point.wait (timed ())) in
  equal (list string) [ "fills" ] (named events);
  equal ~msg:"breaches" (list string) [] (List.drop seen (P.violations ()))

(* enabled *)

(* Another domain's take stays open until this domain has looked. *)
let test_enabled () =
  let before = taking () in
  let inside = Prof.take taking |> fst in
  let opened = Semaphore.Binary.make false
  and looked = Semaphore.Binary.make false in
  let other =
    Domain.spawn (fun () ->
        Prof.take (fun () ->
            Semaphore.Binary.release opened;
            Semaphore.Binary.acquire looked))
  in
  Semaphore.Binary.acquire opened;
  let theirs = taking () in
  Semaphore.Binary.release looked;
  ignore (Domain.join other);
  raises Exit (fun () -> Prof.take (fun () -> raise Exit));
  equal
    (list (pair string string))
    [
      ("before", "no profile taken");
      ("inside a take", "some profile taken");
      ("inside another domain's", "some profile taken");
      ("after them", "no profile taken");
    ]
    [
      ("before", before);
      ("inside a take", inside);
      ("inside another domain's", theirs);
      ("after them", taking ());
    ]

(* now *)

let test_now () =
  let first = Prof.now () in
  let second = Domain.join (Domain.spawn Prof.now) in
  let third = Prof.now () in
  at_least ~msg:"another domain's, after" int ~than:first second;
  at_least ~msg:"this domain's, after" int ~than:second third

(* record *)

(* Both of two nested profiles, the outer one alone before and after the inner,
   none while no profile is taken, and one taken on another domain. *)
let test_record () =
  let d = software "profile:record" in
  let starts =
    List.filter_map (function Prof.Counters c -> Some c.start | _ -> None)
  in
  Prof.record (counters d 0);
  let (now, ((), inner)), outer =
    Prof.take (fun () ->
        let now = Prof.now () in
        Prof.record (counters d (now + 1));
        let r = Prof.take (fun () -> Prof.record (counters d (now + 2))) in
        Prof.record (counters d (now + 3));
        Domain.join
          (Domain.spawn (fun () -> Prof.record (counters d (now + 4))));
        (now, r))
  in
  equal ~msg:"the inner profile" (list int) [ now + 2 ] (starts inner);
  equal ~msg:"the outer profile" (list int)
    [ now + 1; now + 2; now + 3; now + 4 ]
    (starts outer)

let test_record_order () =
  let d = software "profile:record-order" in
  let (), events =
    Prof.take (fun () ->
        (* Each span lies within [first, last], whatever the load. *)
        let first = Prof.now () in
        Prof.span "before" ignore;
        Prof.span "after" ignore;
        let last = Prof.now () in
        Prof.record (counters d (last + 1));
        Prof.record (counters d (first - 1)))
  in
  equal (list string)
    [ "counters"; "before"; "after"; "counters" ]
    (List.map
       (function
         | Prof.Span s -> s.name | Prof.Counters _ -> "counters" | _ -> "other")
       events)

(* span *)

let test_span_starts () =
  let (), outer =
    Prof.take (fun () ->
        Prof.span "outer" (fun () ->
            let (), inner = Prof.take (fun () -> Prof.span "inner" ignore) in
            equal ~msg:"inner profile" (list string) [ "inner" ] (named inner)))
  in
  equal ~msg:"outer profile" (slist string compare) [ "inner"; "outer" ]
    (named outer)

let test_span_raises () =
  let (), events =
    Prof.take (fun () ->
        raises Exit (fun () -> Prof.span "raises" (fun () -> raise Exit)))
  in
  equal (list string) [ "raises" ] (named events)

(* A span ends on another domain after one of its profiles raised: the other
   profile holds it, and the raised one's events are already collectable. *)
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
      equal ~msg:"the raised profile's span name still reachable" (option int)
        None
        (Option.map String.length (Weak.get weak 0));
      Atomic.set finish true;
      let domain = require_some !other in
      other := None;
      let (), events = Domain.join domain in
      equal (list string) [ "late" ] (named events))

(* output_chrome_trace *)

(* Writes [events] to a fresh file through [write], and is its text. *)
let written write =
  let file = Filename.temp_file "profile" ".json" in
  Fun.protect
    ~finally:(fun () -> Sys.remove file)
    (fun () ->
      Out_channel.with_open_bin file write;
      In_channel.with_open_bin file In_channel.input_all)

let test_chrome () =
  let d = software "profile:chrome" in
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
      Prof.Unmeasured { device = d; time = 3_600; runs = 2 };
      Prof.Span
        {
          device = Rig.host;
          lane = "domain 0";
          name = "host";
          start = 1_000;
          stop = 4_000;
        };
      Prof.Work
        {
          point = submit (empty d);
          device = d;
          lane = "COMPUTE:0";
          name = "kernel";
          start = 2_000;
          stop = 3_500;
        };
      Prof.Allocation { device = d; time = 2_500; allocated = 4096 };
      Prof.Copy
        {
          src = Rig.host;
          dst = d;
          bytes = 64;
          point = None;
          start = 3_000;
          stop = 3_200;
        };
    ]
  in
  expect (written (fun oc -> Prof.output_chrome_trace oc events))
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
    {"ph":"i","pid":3,"tid":0,"ts":2.600,"s":"p","name":"unmeasured","args":{"runs":2}}
    ]}
    |}

(* The channel still takes writes, and the file holds nothing before its close
   flushes it. *)
let test_chrome_channel () =
  let file = Filename.temp_file "profile" ".json" in
  Fun.protect
    ~finally:(fun () -> Sys.remove file)
    (fun () ->
      let unflushed =
        Out_channel.with_open_bin file (fun oc ->
            Prof.output_chrome_trace oc [];
            output_string oc "end";
            In_channel.with_open_bin file In_channel.input_all)
      in
      equal ~msg:"before the close" string "" unflushed;
      ends_with ~msg:"after it" ~suffix:"end"
        (In_channel.with_open_bin file In_channel.input_all))

let tests =
  [
    group "profiles"
      [
        test
          "a profile holds the spans another domain records while it is taken"
          test_two_domains;
        test
          "profiles taken at once on two domains each hold what was recorded \
           while taken"
          test_overlapping;
        test "a submit is in each profile taken when it is asked"
          test_submit_profiles;
        test
          "a copy is in the profiles taken when it is asked, and in none \
           started after"
          test_copy_asked;
        test "work asked before a profile is not in it" test_asked_before;
        test "recording while no profile is taken allocates nothing"
          test_idle_words;
      ];
    group "Work"
      [
        test
          "a submission's Work holds its parts' device time on its first \
           part's lane, named after its launches, with its point"
          test_work;
        cases ~id:fst
          "a submission with no launch is named after its first part's kind"
          kinds test_kind;
        test "a Work's lane is its queue's name, the empty one included"
          test_empty_queue_name;
        cases ~id:fst "no Work is recorded of a submission" unrecorded
          test_unrecorded;
        xfail
          ~reason:
            "rig records a Work of another machine's device whose driver times \
             its work (rig.mli: it records none)"
        @@ test "no Work is recorded of a device of another machine" test_far;
        test "a software device times its work" test_software_work;
      ];
    group "Span"
      [ test "a span is on the calling domain's lane of the host" test_span ];
    group "Allocation"
      [
        test "an allocation records the bytes its device holds from then on"
          test_allocation;
      ];
    group "Load" [ test "a load records its image and binary" test_load ];
    group "Copy"
      [
        test "a host's copy records the bytes it moved, with no point" test_copy;
        test "an io device's copy records its bytes, with no point" test_io_copy;
        test "a box records the bytes of its rows" test_box;
        test "a staged copy records only its pieces into and out of staging"
          test_staged_copy;
        test "a copy through a device's staging names the host as its side"
          test_device_staged_copy;
        test "a staged copy's leg on a device lasts until the device ended it"
          test_staged_span;
        test
          "an untimed device's copy parts record their sides, bytes and point, \
           from the submit until take reads them done"
          test_copy_parts;
        test "a timed device's copy parts have its Work's times"
          test_timed_copy_parts;
        test "a fill records no Copy, and a device's fill is Work named copies"
          test_fill;
      ];
    group "take"
      [
        prop
          "take returns events in time order, longest first, then as recorded"
          Gen.(
            list ~size:(int_range 2 8) (pair (int_range 0 2) (int_range 0 2)))
          test_order;
        test "a take whose function raises raises it again with its backtrace"
          test_reraise;
        cases ~id:fst "a take whose function raises keeps none of its events"
          [
            ("as it returns", `Body);
            ("with its device lost", `Lost);
            ("with a read pending", `Read);
          ]
          test_failed_history;
        test
          "work an inner take that raised left unread is in the outer profile"
          test_failed_recipient;
        test "work whose point another domain's wait reached is in the profile"
          test_reached_elsewhere;
        test "take waits only for the work asked while its function ran"
          test_reads_its_own;
        cases ~id:fst
          "take raises Lost for a device lost before its events are read, work \
           asked on"
          asking test_lost;
        test "a settle that fails loses the device, and take raises its Lost"
          test_settle_fails;
        test
          "a close inside take reads its device's work, and take raises nothing"
          test_closed;
        test "a close interrupted in its reads leaves the rest to take"
          test_close_interrupted;
        test "a Ctrl-C in take's wait gives back the work's time pair"
          test_break_in_take;
        test
          "a profile that raised before its work ran leaves the work its time \
           pair"
          test_stopped_keeps_pair;
        test "a time pair goes to another submit once its value settled"
          test_settled_before_reuse;
      ];
    group "enabled"
      [
        test "enabled holds while a profile is taken on any domain" test_enabled;
      ];
    group "now" [ test "now never decreases, across domains" test_now ];
    group "record"
      [
        test
          "record adds an event to each profile being taken, and to none while \
           none is"
          test_record;
        test "take returns recorded events among rig's own in time order"
          test_record_order;
      ];
    group "span"
      [
        test "a span is in the profiles taken when it starts" test_span_starts;
        test "a span records a function that raises" test_span_raises;
        test
          "a span that ends after one of its profiles raised is in the others"
          test_late_span;
      ];
    group "output_chrome_trace"
      [
        test
          "output_chrome_trace writes a process per device, a thread per lane \
           and an event per record"
          test_chrome;
        test "output_chrome_trace neither flushes nor closes its channel"
          test_chrome_channel;
      ];
  ]

let () = exit (run "rig.profile" tests)
