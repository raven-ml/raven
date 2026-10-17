(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module Probe = Rig_link_probe
module Link = Rig_remote_proxy.Link
module Wire = Rig_remote_proxy.Wire

(* test/fixtures' [ready] on the host. [call_ready image ~out ~counts ~fn v
   c] stores [v] at [out], then calls the function at [fn] with [counts] and
   [c], as compiled code calls a rail's ready function. *)
let ready_image () =
  require_ok ~pp:Format.pp_print_string
    (Rig.Image.load Rig.host
       (Rig_host_support.fixture ~dir:"../../fixtures" "ready"))

let call_ready image ~out ~counts ~fn v c =
  let module Sub = Rig.Submission in
  let refs = Sub.[| write 0 ~at:0; write 1 ~at:8 |] in
  let ready = require_some (Rig.Image.resolve image "ready") in
  let s = Sub.make Rig.host [| Sub.launch ~refs ready ~params:40 |] in
  let run = Sub.Run.make s and k = Sub.block s 0 in
  Sub.Run.groups run k 1 1 1;
  Sub.Run.threads run k 1 1 1;
  Sub.Run.int64 run k 16 (Int64.of_nativeint fn);
  Sub.Run.int64 run k 24 (Int64.of_int v);
  Sub.Run.int64 run k 32 (Int64.of_int c);
  ignore (Rig.submit run ~buffers:[| out; counts |])

let read fd n =
  let b = Bytes.create n in
  let rec loop off =
    if off = n then Bytes.unsafe_to_string b
    else
      match Unix.read fd b off (n - off) with
      | 0 -> failf "peer ended after %d of %d bytes" off n
      | got -> loop (off + got)
  in
  loop 0

let rec frame fd =
  let h = read fd 9 in
  let payload = read fd (Int64.to_int (String.get_int64_le h 0)) in
  if Char.code h.[8] = 8 then frame fd else (Char.code h.[8], payload)

(* [s] as wire.mli lays a string out: its length (u32), then its bytes. *)
let str s =
  let n = Bytes.create 4 in
  Bytes.set_int32_le n 0 (Int32.of_int (String.length s));
  Bytes.to_string n ^ s

let connected () =
  let listener = Unix.socket ~cloexec:true Unix.PF_INET Unix.SOCK_STREAM 0 in
  Unix.bind listener (Unix.ADDR_INET (Unix.inet_addr_loopback, 0));
  Unix.listen listener 1;
  Fun.protect ~finally:(fun () -> Unix.close listener) @@ fun () ->
  let fd = Unix.socket ~cloexec:true Unix.PF_INET Unix.SOCK_STREAM 0 in
  Unix.setsockopt_int fd Unix.SO_SNDBUF 4096;
  Unix.connect fd (Unix.getsockname listener);
  let peer, _ = Unix.accept ~cloexec:true listener in
  Rig_support.recv_timeout peer 5.;
  (fd, peer)

let joined_release () =
  let fd, peer = connected () in
  let job = Link.job () in
  Fun.protect ~finally:(fun () ->
      Probe.resume ();
      Link.fail job "test ended";
      Unix.close peer)
  @@ fun () ->
  let link = Link.make job fd ~name:"peer" ~peer:(Wire.Agent 1) in
  let size = 1 lsl 20 in
  let transfer = { Rig_remote_abi.src = 0; dst = 0; length = size } in
  let e = Link.rail link ~id:4 ~send:[| transfer |] ~receive:[||] in
  Bigarray.Array1.fill e.outbound 'r';
  Probe.arm Sending;
  e.ready 1;
  equal ~msg:"the sender holds the partial transfer" bool true (Probe.held ());
  let returned = Atomic.make 0 in
  let release () =
    Thread.create
      (fun () ->
        Link.release_rail link 4;
        ignore (Atomic.fetch_and_add returned 1))
      ()
  in
  let first = release () and second = release () in
  let entered, waiting = Probe.releases () in
  let early = Atomic.get returned in
  (* The ended callback may store a later count; the handed-over batch is still
     the one whose transfer the sender holds. *)
  call_ready (ready_image ())
    ~out:(Rig.Buffer.create Rig.host 8)
    ~counts:(Rig.Buffer.of_bigarray e.counts)
    ~fn:e.ready_fn 0 2;
  Probe.resume ();
  let dropping = Thread.create (fun () -> Link.drop link 99) () in
  let kind, payload = frame peer in
  let rec after acc =
    match frame peer with 7, _ -> after (acc + 1) | k, p -> (acc, k, p)
  in
  let extra, next, dropped = after 0 in
  Thread.join first;
  Thread.join second;
  Thread.join dropping;
  equal ~msg:"release entrants" int 2 entered;
  equal ~msg:"waiting releases and extra transfers" (pair int int) (2, 0)
    (waiting, extra);
  equal ~msg:"releases returned before the transfer" int 0 early;
  equal ~msg:"release completions" int 2 (Atomic.get returned);
  equal ~msg:"rail frame" int 7 kind;
  equal ~msg:"transfer id" int64 4L (String.get_int64_le payload 0);
  equal ~msg:"transfer count" int64 1L (String.get_int64_le payload 8);
  equal ~msg:"transfer bytes" string (String.make size 'r')
    (String.sub payload 16 size);
  equal ~msg:"ended callback queued no transfer" int 4 next;
  equal ~msg:"drop id" int64 99L (String.get_int64_le dropped 0)

let with_rail ~send ~receive f =
  let fd, peer = connected () in
  let job = Link.job () in
  Fun.protect ~finally:(fun () ->
      Probe.resume ();
      Link.fail job "test ended";
      Unix.close peer)
  @@ fun () ->
  let link = Link.make job fd ~name:"peer" ~peer:(Wire.Agent 1) in
  let e = Link.rail link ~id:4 ~send ~receive in
  Fun.protect
    ~finally:(fun () ->
      Probe.resume ();
      Link.release_rail link 4)
    (fun () -> f job link peer e)

let failed_counts (e : Rig_remote_abi.end_) =
  let counts =
    List.map
      (fun at ->
        String.get_int64_le (String.init 8 (fun i -> e.counts.{at + i})) 0)
      [ 0; 128; 256 ]
  in
  equal ~msg:"ready, sent, arrived" (list int64)
    [ Int64.max_int; Int64.max_int; Int64.max_int ]
    counts

let transfer = { Rig_remote_abi.src = 0; dst = 0; length = 5 }

(* Failure lands between completed I/O and its count publication. [fail]
   leaves the link's abort to the held sender, and the job's other link
   aborts once the job failed. Releasing joins that held use before the
   assertions read the counts. *)
let sent_after_failure () =
  with_rail ~send:[| transfer |] ~receive:[||] @@ fun job link _ e ->
  let fd, peer = connected () in
  Fun.protect ~finally:(fun () -> Unix.close peer) @@ fun () ->
  ignore (Link.make job fd ~name:"other" ~peer:(Wire.Agent 2));
  Probe.arm Sent;
  let sender = Thread.create e.ready 1 in
  Fun.protect ~finally:(fun () ->
      Probe.resume ();
      Thread.join sender)
  @@ fun () ->
  equal ~msg:"sent publication is held" bool true (Probe.held ());
  let failing =
    Thread.create (fun () -> Link.fail job "failed before publishing sent") ()
  in
  equal ~msg:"the other link" int 9 (fst (frame peer));
  Probe.resume ();
  Thread.join failing;
  Link.release_rail link 4;
  failed_counts e

let arrived_after_failure () =
  with_rail ~send:[||] ~receive:[| transfer |] @@ fun job link peer e ->
  let frame = Bytes.make 30 '\000' in
  Bytes.set_int64_le frame 0 21L;
  Bytes.set frame 8 '\007';
  Bytes.set_int64_le frame 9 4L;
  Bytes.set_int64_le frame 17 1L;
  Bytes.blit_string "hello" 0 frame 25 5;
  Probe.arm Arrived;
  let rec write at =
    if at < Bytes.length frame then
      write (at + Unix.write peer frame at (Bytes.length frame - at))
  in
  write 0;
  equal ~msg:"arrived publication is held" bool true (Probe.held ());
  Link.fail job "failed before publishing arrived";
  Probe.resume ();
  Link.release_rail link 4;
  failed_counts e

(* The job fails while the sending thread holds the rest of a transfer its
   ready function began. The rest goes before the abort: the peer reads the
   whole frame, then the abort. The job's other link aborts once the failure
   began, which the test awaits before it lets the transfer go on. *)
let partial_before_abort () =
  let fd, peer = connected () in
  let fd2, peer2 = connected () in
  let job = Link.job () in
  Fun.protect ~finally:(fun () ->
      Probe.resume ();
      Link.fail job "test ended";
      Unix.close peer;
      Unix.close peer2)
  @@ fun () ->
  let link = Link.make job fd ~name:"peer" ~peer:(Wire.Agent 1) in
  ignore (Link.make job fd2 ~name:"other" ~peer:(Wire.Agent 2));
  let size = 1 lsl 20 in
  let transfer = { Rig_remote_abi.src = 0; dst = 0; length = size } in
  let e = Link.rail link ~id:4 ~send:[| transfer |] ~receive:[||] in
  Bigarray.Array1.fill e.outbound 'r';
  Probe.arm Sending;
  e.ready 1;
  equal ~msg:"the sender holds the partial transfer" bool true (Probe.held ());
  let why = "failed mid-transfer" in
  let failing = Thread.create (fun () -> Link.fail job why) () in
  let other = frame peer2 in
  Probe.resume ();
  let rail = frame peer in
  let last = frame peer in
  Thread.join failing;
  let abort = (9, str why) in
  equal ~msg:"the other link" (pair int string) abort other;
  equal ~msg:"rail frame" int 7 (fst rail);
  equal ~msg:"transfer bytes" string (String.make size 'r')
    (String.sub (snd rail) 16 size);
  equal ~msg:"after the frame" (pair int string) abort last

let ready_after_failure () =
  with_rail ~send:[||] ~receive:[| transfer |] @@ fun job _ _ e ->
  Link.fail job "failed before ready";
  e.ready 7;
  failed_counts e

let () =
  exit
    (run "remote rail release"
       [
         test "joining releases drain one held transfer" joined_release;
         test "a sent publication preserves failed counts" sent_after_failure;
         test "an arrived publication preserves failed counts"
           arrived_after_failure;
         test "a ready callback preserves failed counts" ready_after_failure;
         test "a transfer begun before a failure goes whole before the abort"
           partial_before_abort;
       ])
