(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Proxies of another machine, opened through rig over a real link on loopback
   to an agent loop (Proxy_machine). *)

open Windtrap
open Proxy_machine
module Wire = Rig_remote_proxy.Wire
module Link = Rig_remote_proxy.Link
module Proxy = Rig_remote_proxy
module B = Rig.Buffer
module Sub = Rig.Submission

(* The bytes in flight past which a proxy's room check answers later. *)
let in_flight_limit = 64 lsl 20

(* Facts *)

let rec_w =
  Testable.make
    ~pp:(fun ppf -> function
      | Rig_remote_abi.Host h -> Format.fprintf ppf "Host %s" h.machine
      | Rig_remote_abi.Device { id } -> Format.fprintf ppf "Device %d" id)
    ~equal:( == )

let completion_w =
  Testable.structural ~pp:(fun ppf -> function
    | Rig_edge.Host -> Format.pp_print_string ppf "Host"
    | Store -> Format.pp_print_string ppf "Store")

let driver_facts () =
  with_job_link @@ fun far ->
  let a = account 3 ~reaches:[] in
  let c = Rig_remote_abi.Device { id = 3 } in
  let p = Result.get_ok (Rig_gpu_support.opened (Proxy.device far a c)) in
  let f = Proxy.facts p in
  equal (list string) [ "COMPUTE:0"; "COPY:0" ]
    (List.map (fun (q : Rig_edge.queue) -> q.name) f.queues);
  equal ~msg:"a device's queues run copies" bool true
    (List.for_all (fun (q : Rig_edge.queue) -> q.runs = [ Copy ]) f.queues);
  equal (option int) ~msg:"the copy engine" (Some 1) f.copy_engine;
  equal completion_w Host f.completion;
  equal (list bool) [ true; false ] [ f.waits.hosts; f.waits.stores ];
  equal int max_int f.waits.most;
  equal bool true f.may_block;
  equal bool false f.maps_host;
  equal rec_w c (Proxy.capability p);
  equal string "mem3" f.arch;
  equal int ((1 lsl 30) + 3) f.budget;
  equal ~msg:"the account's alignment" int (16 lsl 3) f.alignment

let reaches (forward, backward) =
  with_machine
    ~reaches:
      [ (if forward then [ 2 ] else []); (if backward then [ 1 ] else []) ]
  @@ fun m ->
  let d1 = m.devices.(0) and d2 = m.devices.(1) in
  equal ~msg:"older reaches newer" bool forward (Rig.reaches d1 d2);
  equal ~msg:"newer reaches older" bool backward (Rig.reaches d2 d1)

let two_links () =
  let job = Link.job () in
  Fun.protect
    ~finally:(fun () -> Link.fail job "the test ended")
    (fun () ->
      let link name i =
        let d, a = connected () in
        let l = Link.make job d ~name ~peer:(Wire.Agent i) in
        ignore (Link.make job a ~name:"controller" ~peer:Wire.Controller);
        l
      in
      let l1 = link "one" 1 and l2 = link "two" 2 in
      let device id = Rig_remote_abi.Device { id } in
      let p =
        Result.get_ok
          (Rig_gpu_support.opened
             (Proxy.device l1 (account 1 ~reaches:[]) (device 1)))
      in
      let q =
        Result.get_ok
          ((Proxy.device l2
              (account 2 ~reaches:[ 1 ] ~reached_by:[ 1 ])
              (device 2))
             .make ~word:(Rig_gpu_support.Host.word ()))
      in
      let q' =
        Result.get_ok
          ((Proxy.device l1
              (account 2 ~reaches:[] ~reached_by:[ 1 ])
              (device 2))
             .make ~word:(Rig_gpu_support.Host.word ()))
      in
      equal bool false (Proxy.peer p q);
      equal bool true (Proxy.peer p q');
      let word p = (Proxy.facts p).word in
      let r = word q' in
      equal ~msg:"a word's handle" nativeint 0n (Proxy.locate r).handle;
      equal ~msg:"a word's shadow is this process's" bool true
        (Option.is_some (Proxy.locate r).host);
      equal ~msg:"map_host" bool true
        (Option.is_none (Proxy.map_host p 4096 4096));
      Link.fail job "lost";
      equal ~msg:"once lost" bool true (Proxy.peer p q');
      equal bool true (Option.is_none (Proxy.map_peer p q (word q)));
      match Proxy.map_peer p q' (word q') with
      | Some r -> equal (option int) (Some 2) (Proxy.locate r).address
      | None -> fail "a proxy of the same link maps the word")

let make_misuse () =
  with_job_link @@ fun far ->
  let rig_host =
    Rig_remote_abi.Host
      {
        machine = "m";
        rail = no_rails;
        threads = Proxy.threads;
      }
  in
  raises_match ~msg:"a host's record for device 1" Exn.invalid_arg (fun () ->
      Result.get_ok
        (Rig_gpu_support.opened
           (Proxy.device far (account 1 ~reaches:[]) rig_host)));
  raises_match ~msg:"device 2's record for device 1" Exn.invalid_arg (fun () ->
      Result.get_ok
        ((Proxy.device far (account 1 ~reaches:[])
            (Rig_remote_abi.Device { id = 2 }))
           .make ~word:(Rig_gpu_support.Host.word ())));
  raises_match ~msg:"a device's record for the host" Exn.invalid_arg (fun () ->
      Result.get_ok
        ((Proxy.device far (account 0 ~reaches:[])
            (Rig_remote_abi.Device { id = 1 }))
           .make ~word:(Rig_gpu_support.Host.word ())))

let facts =
  group "facts"
    [
      test
        "a proxy's facts are the agent's account's and the ones its .mli lists"
        driver_facts;
      cases
        ~name:(fun (forward, backward) ->
          Printf.sprintf "older reaches newer: %b; newer reaches older: %b"
            forward backward)
        "directed reachability"
        [ (false, false); (true, false); (false, true); (true, true) ]
        reaches;
      test "proxies of two links never map each other; a word maps at its id"
        two_links;
      test "make raises for a record of another device" make_misuse;
    ]

(* Memory *)

(* Each proxy's buffer is an allocation of the agent on the proxy's own
   device, the agent's 1 and 2 here (0 is its host), of at least the bytes
   asked. *)
let alloc_reaches_agent () =
  with_machine ~reaches:[ []; [] ] @@ fun m ->
  let allocation d =
    let b = B.create d 4096 in
    let id = Nativeint.to_int (B.handle b) in
    require_match
      (List.find_map (function
        | Alloc a when a.id = id -> Some (a.device, a.bytes)
        | _ -> None))
      (events m.ag)
  in
  Array.iteri
    (fun i d ->
      let device, bytes = allocation d in
      equal ~msg:"the allocation's device" int (i + 1) device;
      at_least ~msg:"the allocation's bytes" int ~than:4096 bytes)
    m.devices

let refused_alloc () =
  with_machine ~room:(1 lsl 20) @@ fun m ->
  ignore (B.create m.host 4096);
  raises_match
    (function Rig.Out_of_memory (_, n) -> n = 2 lsl 20 | _ -> false)
    (fun () -> B.create m.host (2 lsl 20))

let dropped () =
  with_machine @@ fun m ->
  let id =
    let b = B.create m.host 4096 in
    Nativeint.to_int (B.handle b)
  in
  Gc.full_major ();
  (* A drain takes the collected memory back into the cache. *)
  ignore (B.create m.host 16);
  Rig.free_cache m.host;
  until ~what:"the drop" (fun () -> List.mem (Drop id) (events m.ag))

(* A borrow maps the memory on the agent; a copy through it moves the memory's
   bytes. *)
let borrowed () =
  with_machine ~reaches:[ [ 2 ]; [] ] @@ fun m ->
  let d1 = m.devices.(0) and d2 = m.devices.(1) in
  let b2 = far_of_string d2 "mapped" in
  let b1 = require_some (B.borrow d1 b2) in
  let region = Nativeint.to_int (B.handle b2) in
  equal bool true
    (List.exists
       (function Map { device = 1; region = r; _ } -> r = region | _ -> false)
       (events m.ag));
  let dst = B.create d1 6 in
  B.copy ~src:b1 ~dst;
  equal string "mapped" (read dst)

let memory =
  group "memory"
    [
      test
        "a buffer on a proxy is an allocation of the agent on its device, \
         named by a handle"
        alloc_reaches_agent;
      test "an allocation the agent refuses raises Out_of_memory" refused_alloc;
      test "memory given back reaches the agent as a drop" dropped;
      test "a device borrows the memory of a device it reaches" borrowed;
    ]

(* Copies *)

(* Two queued batches include empty downloads between nonempty ones. Their
   replies must keep the destinations in submission and part order. *)
let download_batches () =
  with_machine @@ fun m ->
  let far = far_of_string m.host "abcdefgh" in
  let here = host_buffer (String.make 24 '.') in
  let parts first =
    Array.mapi
      (fun i length ->
        let src = B.view far ~first:i ~length in
        let dst = B.view here ~first:(first + (2 * i)) ~length in
        Sub.(copy ~src:(Buffer src) ~dst:(Buffer dst)))
      [| 0; 1; 0; 2; 0 |]
  in
  pause m.ag;
  ignore (submit (Sub.make m.host (parts 0)));
  let last = submit (Sub.make m.host (parts 12)) in
  resume m.ag;
  Rig.Point.wait last;
  equal string "..b...de......b...de...." (read_host here)

let within_machine () =
  with_machine ~reaches:[ [] ] @@ fun m ->
  let a = far_of_string m.host "within the machine" in
  let b = B.create m.host (B.length a) in
  B.copy ~src:a ~dst:b;
  let crossing =
    List.exists
      (function
        | Handover { parts; local; _ } ->
            local = 0
            && List.exists
                 (function
                   | Wire.Copy { src = Wire.Region _; dst = Wire.Region _; _ }
                     ->
                       true
                   | _ -> false)
                 parts
        | _ -> false)
      (events m.ag)
  in
  equal ~msg:"a copy between regions, with no bytes from here" bool true
    crossing;
  equal string "within the machine" (read b)

(* A fill of another machine's memory is one fill part of its view, which
   carries no bytes across the link. *)
let far_fill () =
  with_machine @@ fun m ->
  let far = far_of_string m.host "abcdefgh" in
  B.fill (B.view far ~first:2 ~length:4) 'z';
  equal string "abzzzzgh" (read far);
  let fills =
    List.concat_map
      (function
        | Handover { parts; local; _ } ->
            List.filter_map
              (function
                | Wire.Fill { dst = Wire.Region { offset; _ }; bytes; byte } ->
                    Some (offset, bytes, byte, local)
                | _ -> None)
              parts
        | _ -> [])
      (events m.ag)
  in
  equal
    (list (pair (pair int int) (pair char int)))
    [ ((2, 4), ('z', 0)) ]
    (List.map (fun (o, n, c, l) -> ((o, n), (c, l))) fills)

(* A box from this process's memory whose rows lie apart crosses the link
   packed and lands at its pitch, and a box into this process's memory lands
   its rows at theirs. Rig's interface makes no box yet, so the parts go to
   a proxy opened alone. *)
let boxes_across () =
  with_machine @@ fun m ->
  let module H = Rig_gpu_support.Host in
  let o =
    Proxy.device m.far (account 9 ~reaches:[])
      (Rig_remote_abi.Device { id = 9 })
  in
  let (module D : Rig_edge.Driver with type t = Proxy.t) = o.driver in
  let module E = Rig_gpu_support.Edge (D) in
  let g = ok_or_fail (Rig_gpu_support.opened o) in
  let e = E.make g in
  Fun.protect ~finally:(fun () ->
      E.release e;
      D.stop g ~fault:None)
  @@ fun () ->
  let far = E.alloc e Device 64 in
  let from = H.pages 64 and back = H.pages 64 in
  let s = String.init 64 (fun i -> Char.chr (97 + (i mod 26))) in
  H.write from s;
  let box ~dst ~dst_offset ~src ~src_offset ~src_pitch ~dst_pitch :
      E.part =
    {
      dst;
      dst_offset;
      src = Some src;
      src_offset;
      width = 3;
      rows = 4;
      src_pitch;
      dst_pitch;
      fill = None;
    }
  in
  E.run e
    (box ~dst:(Region far) ~dst_offset:0 ~src:(Host from) ~src_offset:1
       ~src_pitch:5 ~dst_pitch:4);
  E.run e
    (box ~dst:(Host back) ~dst_offset:2 ~src:(Region far) ~src_offset:0
       ~src_pitch:4 ~dst_pitch:7);
  let model = Bytes.make 64 '\000' in
  for r = 0 to 3 do
    Bytes.blit_string s (1 + (5 * r)) model (2 + (7 * r)) 3
  done;
  equal string (Bytes.to_string model) (H.read back 64);
  H.free_pages from 64;
  H.free_pages back 64

let copies =
  group "copy"
    [
      test "queued download batches keep empty copies in order" download_batches;
      test "a copy within the machine moves no byte across the link"
        within_machine;
      test "a fill of the machine's memory is one fill part, carrying no byte"
        far_fill;
      test "boxes into and out of this process's memory land at their pitches"
        boxes_across;
    ]

(* Work *)

let never () =
  with_machine ~reaches:[ [] ] @@ fun m ->
  let far = B.create m.host 16 and far2 = B.create m.host 16 in
  let h = B.create Rig.host 16 in
  let both =
    Sub.make m.host
      [|
        Sub.(copy ~src:(Buffer far) ~dst:(Buffer h));
        Sub.(copy ~src:(Buffer h) ~dst:(Buffer far2));
      |]
  in
  raises_match ~msg:"a copy from here after a copy into here" Exn.invalid_arg
    (fun () -> submit both);
  let d = m.devices.(0) in
  let b = B.create d 16 in
  raises_match ~msg:"a proxy's memory has no address" Exn.invalid_arg (fun () ->
      Sub.address (Sub.make ~fixed:[ (b, Rig.Read) ] d [||]) b);
  let words = host_buffer "\001\000\000\000\002\000\000\000" in
  let on d = Sub.make d [| Sub.words words |] in
  raises_match ~msg:"words on a device other than the host" Exn.invalid_arg
    (fun () -> submit (on d));
  let p = submit (on m.host) in
  Rig.Point.wait p;
  equal bool true
    (List.exists
       (function
         | Handover { device = 0; parts = [ Wire.Words w ]; _ } ->
             w = "\001\000\000\000\002\000\000\000"
         | _ -> false)
       (events m.ag))

(* A proxy's queues run no fill: the submission is refused when it is made. The
   fill is the support's bump, harmless if a hand-over called it. *)
let fill_refused () =
  with_machine @@ fun m ->
  raises_match Exn.invalid_arg @@ fun () ->
  Sub.make m.host
    [|
      Sub.fill Rig_support.bump ~params:8;
    |]

(* Waits between devices of one link travel with the hand-over, as the
   producer's id and value. *)
let waits_travel () =
  with_machine ~reaches:[ []; [] ] @@ fun m ->
  let d1 = m.devices.(0) and d2 = m.devices.(1) in
  let a = B.create d1 64 and b = B.create d1 64 and c = B.create d2 64 in
  let c' = B.create d2 64 in
  pause m.ag;
  let p = submit (copy_submission d1 ~src:a ~dst:b) in
  let q = submit ~waits:[| p |] (copy_submission d2 ~src:c ~dst:c') in
  resume m.ag;
  Rig.Point.wait q;
  let waits =
    require_match
      (List.find_map (function
        | Handover { device = 2; waits; _ } -> Some waits
        | _ -> None))
      (events m.ag)
  in
  equal (list (pair int int)) [ (1, Rig.Point.value p) ] waits

(* With the agent held, copies of 40 MiB within the machine: the third finds
   more than 64 MiB in flight and waits. *)
let room_later () =
  with_machine @@ fun m ->
  let n = in_flight_limit / 8 * 5 in
  let a = B.create m.host n and b = B.create m.host n in
  let s = copy_submission m.host ~src:a ~dst:b in
  pause m.ag;
  ignore (submit s);
  ignore (submit s);
  let finished, third = spawn (fun () -> submit s) in
  Thread.delay 0.3;
  let early = finished () in
  resume m.ag;
  let p = third () in
  Rig.Point.wait p;
  equal ~msg:"the third submit returned while 80 MiB were in flight" bool false
    early

let room_idle () =
  with_machine @@ fun m ->
  let n = 100 lsl 20 in
  let a = B.create m.host n and b = B.create m.host n in
  let p = submit (copy_submission m.host ~src:a ~dst:b) in
  Rig.Point.wait p

(* A Ctrl-C during a wait for a proxy's work loses nothing: the wait raises
   Sys.Break, and the next wait reaches the value. The agent holds the work
   until the wait broke. *)
let break_in_wait () =
  if Sys.win32 then skip ~reason:"Windows has no SIGINT from a thread" ();
  with_machine ~reaches:[ [] ] @@ fun m ->
  let d = m.devices.(0) in
  let a = B.create d 64 and b = B.create d 64 in
  pause m.ag;
  let p = submit (copy_submission d ~src:a ~dst:b) in
  Sys.catch_break true;
  let broke =
    Fun.protect ~finally:(fun () ->
        (try Rig_support.interrupt_join () with Sys.Break -> ());
        Sys.catch_break false)
    @@ fun () ->
    Rig_support.interrupt_after 0.1;
    match Rig.Point.wait p with () -> false | exception Sys.Break -> true
  in
  resume m.ag;
  equal ~msg:"the wait took Sys.Break" bool true broke;
  equal ~msg:"lost" lost_w None (Rig.lost d);
  Rig.Point.wait p

let work =
  group "work"
    [
      test "parts a proxy does not run are refused; words run on the host" never;
      test "a fill is refused" fill_refused;
      test "a wait on another device of the link reaches the agent" waits_travel;
      test "a submission waits while more than 64 MiB are in flight" room_later;
      test "an idle device takes a submission of any size" room_idle;
      test "a Ctrl-C in a wait loses nothing, and the next wait reaches it"
        break_in_wait;
    ]

(* Code *)

let images () =
  with_machine ~reaches:[ [] ] @@ fun m ->
  let i = require_ok (Rig.Image.load m.host "run,step") in
  equal (option int) (Some 1)
    (Option.map Rig.Image.entry (Rig.Image.resolve i "step"));
  is_none (Rig.Image.resolve i "walk");
  let why = require_error (Rig.Image.load m.host "bad code") in
  starts_with ~msg:"the reason starts with the host's name"
    ~affix:(Rig.name m.host) why;
  contains ~msg:"the agent's reason" ~sub:"the agent refuses it" why;
  let why = require_error (Rig.Image.load m.devices.(0) "run") in
  contains ~msg:"says it loads no code" ~sub:"loads no code" why

let code =
  group "code"
    [ test "the machine's host loads code, its devices do not" images ]

(* Failure and close *)

(* A hand-over waiting for its word when the job fails: the wait raises and the
   device stops at its last value. *)
let in_flight_fails () =
  with_machine @@ fun m ->
  let a = B.create m.host 64 and b = B.create m.host 64 in
  pause m.ag;
  let p = submit (copy_submission m.host ~src:a ~dst:b) in
  Link.fail m.ag.job "failed in flight";
  raises_match
    (function Rig.Lost (_, why) -> why = "failed in flight" | _ -> false)
    (fun () -> Rig.Point.wait p);
  (* The close runs the stop, which raises the word. *)
  Rig.close m.host;
  until ~what:"the word at the last value handed over" (fun () ->
      Rig.signaled m.host = Rig.Point.value p)

(* Each counted call of a proxy made directly on the link, with a region and an
   image the agent made before the job failed. *)
let counted =
  [
    ("alloc", fun p _ _ -> ignore (Proxy.alloc p Device 16));
    ("map_peer", fun p r _ -> ignore (Proxy.map_peer p p r));
    ("image", fun p _ _ -> ignore (Proxy.image p "run"));
    ("entry", fun _ _ i -> ignore (Proxy.entry i "run"));
    ("sleep", fun p _ _ -> Proxy.sleep p ~seen:0 ~still_ms:10);
  ]

(* free and unload send nothing once the job failed; signaled reads the shadow
   as before. *)
let quiet_after_failure () =
  with_machine @@ fun m ->
  let p = m.raw in
  let r = require_some (Proxy.alloc p Device 16) in
  let i =
    match Proxy.image p "run" with
    | Ok (Rig_edge.Loaded i) -> i
    | Ok (Place _) -> fail "the host loads its own code"
    | Error why -> fail why
  in
  Link.fail m.ag.job "root";
  Proxy.free p r;
  Proxy.unload p i;
  equal int 0 (Proxy.signaled p)

let counted_after_failure (_, call) =
  with_machine @@ fun m ->
  let p = m.raw in
  let r = require_some (Proxy.alloc p Device 16) in
  let i =
    match Proxy.image p "run" with
    | Ok (Rig_edge.Loaded i) -> i
    | Ok (Place _) -> fail "the host loads its own code"
    | Error why -> fail why
  in
  Link.fail m.ag.job "root";
  raises_match
    (function Proxy.Fault "root" -> true | _ -> false)
    (fun () -> call p r i)

let close_device () =
  with_machine ~reaches:[ [] ] @@ fun m ->
  let d = m.devices.(0) in
  ignore (far_of_string d "closing");
  Rig.close d;
  equal lost_w (Some "closed") (Rig.lost d);
  equal (option string) None (Link.failed m.ag.job);
  equal string "the host goes on"
    (read (far_of_string m.host "the host goes on"))

(* A peer that reads every frame and never closes: the job closes, and stays
   closing, from [Link.close] on. [closing ()] holds once its close came. *)
let silent_peer p =
  let closing = Atomic.make false in
  let b = Bytes.create 4096 in
  let rec drain () =
    let n = try Unix.read p b 0 9 with Unix.Unix_error _ -> 0 in
    if n = 9 then begin
      let len = Int64.to_int (Bytes.get_int64_le b 0) in
      if Char.code (Bytes.get b 8) = 10 then Atomic.set closing true;
      let rec skip k =
        if k > 0 then
          let r =
            try Unix.read p b 0 (min k 4096) with Unix.Unix_error _ -> 0
          in
          if r > 0 then skip (k - r)
      in
      skip len;
      drain ()
    end
  in
  let _, drained = spawn drain in
  ((fun () -> Atomic.get closing), drained)

(* The hand-over refuses a closing job's work; its reason names the close. *)
let closing_handover () =
  with_job @@ fun job ->
  let d, p = connected () in
  let closing, drained = silent_peer p in
  let far = Link.make job d ~name:"far" ~peer:(Wire.Agent 1) in
  let machine = fresh_machine () in
  let host =
    ok_or_fail
      (Rig.open_host ~machine
         (Proxy.device far (account 0 ~reaches:[])
            (Rig_remote_abi.Host
               {
                 machine;
                 rail = no_rails;
                 threads = Proxy.threads;
               })))
  in
  let _, closed = spawn (fun () -> Link.close job) in
  Fun.protect
    ~finally:(fun () ->
      Link.fail job "the test ended";
      (* A failed link reads its peer until the peer's stream ends. *)
      Unix.shutdown p Unix.SHUTDOWN_SEND;
      closed ();
      Unix.close p;
      drained ();
      Rig.close host)
    (fun () ->
      until ~what:"the close" closing;
      let words = host_buffer "\001\000\000\000" in
      let s =
        Sub.make host
          [|
            Sub.words words;
          |]
      in
      match submit s with
      | _ -> fail "a closing job took a hand-over"
      | exception Rig.Lost (_, why) ->
          contains ~msg:"the reason" ~sub:"clos" why)

(* [n] bytes of [p], fewer once its stream ended. *)
let read_n p n =
  let b = Bytes.create n in
  let rec go k =
    if k = n then k
    else
      match Unix.read p b k (n - k) with
      | 0 -> k
      | r -> go (k + r)
      | exception Unix.Unix_error _ -> k
  in
  Bytes.sub_string b 0 (go 0)

(* Answers the request [p] reads next, an allocation, that it succeeded: the
   allocation's id, the u64 after the request's tag. *)
let grant_alloc p =
  let h = read_n p 9 in
  if String.length h < 9 then fail "the link ended before its request";
  let request = read_n p (Int64.to_int (String.get_int64_le h 0)) in
  let answer = "\000\001" in
  let frame = Bytes.create (9 + String.length answer) in
  Bytes.set_int64_le frame 0 (Int64.of_int (String.length answer));
  Bytes.set frame 8 '\002';
  Bytes.blit_string answer 0 frame 9 (String.length answer);
  ignore (Unix.write p frame 0 (Bytes.length frame));
  Int64.to_int (String.get_int64_le request 1)

(* The host device of a job with one link, whose peer end [p] the test
   serves, for [f host p]. *)
let with_raw_host f =
  with_job @@ fun job ->
  let d, p = connected () in
  let far = Link.make job d ~name:"far" ~peer:(Wire.Agent 1) in
  let machine = fresh_machine () in
  let host =
    ok_or_fail
      (Rig.open_host ~machine
         (Proxy.device far (account 0 ~reaches:[])
            (Rig_remote_abi.Host
               {
                 machine;
                 rail = no_rails;
                 threads = Proxy.threads;
               })))
  in
  Fun.protect
    ~finally:(fun () ->
      Link.fail job "the test ended";
      Unix.close p;
      Rig.close host)
    (fun () -> f host p)

(* A hand-over sends its frame itself: to a peer that reads nothing, a copy from
   this process's memory larger than the sockets hold returns only once the peer
   took the frame. *)
let handover_sends () =
  let n = 64 lsl 20 in
  with_raw_host @@ fun host p ->
  let _, region = spawn (fun () -> B.create host n) in
  ignore (grant_alloc p : int);
  let dst = region () in
  let src = B.create Rig.host n in
  let returned, submitted =
    spawn (fun () -> submit (copy_submission host ~src ~dst))
  in
  (match Unix.select [ p ] [] [] patience with
  | [], _, _ -> fail "the hand-over did not start"
  | _ -> ());
  Thread.delay 0.2;
  let early = returned () in
  let h = read_n p 9 in
  let payload = read_n p (Int64.to_int (String.get_int64_le h 0)) in
  ignore (submitted ());
  equal ~msg:"returned while its peer read nothing" bool false early;
  equal ~msg:"the frame's kind" int 3 (Char.code h.[8]);
  less ~msg:"the hand-over's bytes" int ~than:(String.length payload) n

(* A second proxy of one account would take the first's place in the link's
   table, and the first's reports would never reach it. *)
let second_proxy () =
  with_job_link @@ fun far ->
  let a = account 1 ~reaches:[] in
  let first = Proxy.device far a (Rig_remote_abi.Device { id = 1 }) in
  ignore (Result.get_ok (Rig_gpu_support.opened first));
  let second = Proxy.device far a (Rig_remote_abi.Device { id = 1 }) in
  equal string a.name second.name;
  raises_match Exn.invalid_arg (fun () -> Rig_gpu_support.opened second)

let failures =
  group "failure"
    [
      test "a wait for a value handed over when the job fails raises Lost"
        in_flight_fails;
      cases
        ~name:(fun (n, _) ->
          n ^ " raises Fault with the root cause once the job failed")
        "counted" counted counted_after_failure;
      test "free, unload and signaled raise nothing once the job failed"
        quiet_after_failure;
      test "closing a device ends it alone" close_device;
      test "a hand-over on a closing job fails naming the close"
        closing_handover;
      test "a hand-over returns once its frame is sent (sampled)" handover_sends;
      test "make refuses a second proxy of an account on the link" second_proxy;
    ]

let () =
  Watchdog.start ();
  exit
    (run "rig_remote_proxy.proxy"
       [
         group ~timeout:60. "proxy"
           [ facts; memory; copies; work; code; failures ];
       ])
