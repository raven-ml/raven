(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A link's faults at each point link.mli's failure section names: before a
   frame, mid-frame, during a rail transfer and during the abort, each with a
   stalled peer and a severed one. A job holds the faulted link, named "peer",
   and a well one, named "other", whose peers are raw sockets of this process.
   Each test states who learns the root cause, which aborts go out, and that
   every waiter wakes. *)

open Windtrap
module Wire = Rig_remote_proxy.Wire
module Link = Rig_remote_proxy.Link
module Probe = Rig_link_probe

(* wire.mli's frame kinds *)
let k_request = 1
let k_bytes = 6
let k_rail = 7
let k_beat = 8
let k_abort = 9

(* link.mli's bound on a silent peer and on one that reads nothing. *)
let silence = 10.

(* The seconds a read of a peer's socket waits for a byte. *)
let patience = 5.

(* The bytes of a frame or transfer no connection holds whole: its send waits
   for its peer to read. *)
let large = 32 lsl 20

let reset_cause = "peer: closed its connection"
let stalled_cause = "peer: read nothing for 10 s"

(* Sockets *)

(* A connected pair of loopback TCP sockets: the link's end, whose send buffer
   is small, and its peer's, whose reads give up after [patience]. *)
let connected () =
  let l = Unix.socket ~cloexec:true Unix.PF_INET Unix.SOCK_STREAM 0 in
  Unix.bind l (Unix.ADDR_INET (Unix.inet_addr_loopback, 0));
  Unix.listen l 1;
  Fun.protect ~finally:(fun () -> Unix.close l) @@ fun () ->
  let d = Unix.socket ~cloexec:true Unix.PF_INET Unix.SOCK_STREAM 0 in
  Unix.setsockopt_int d Unix.SO_SNDBUF 4096;
  Unix.connect d (Unix.getsockname l);
  let p, _ = Unix.accept ~cloexec:true l in
  Unix.setsockopt_float p Unix.SO_RCVTIMEO patience;
  (d, p)

(* The next [n] bytes of [fd], fewer if its stream ends or no byte comes
   within [patience] first. *)
let read_n fd n =
  let b = Bytes.create n in
  let rec go off =
    if off = n then off
    else
      match Unix.read fd b off (min (n - off) 65536) with
      | 0 -> off
      | k -> go (off + k)
      | exception Unix.Unix_error (Unix.ECONNRESET, _, _) -> off
      | exception Unix.Unix_error ((Unix.EAGAIN | Unix.EWOULDBLOCK), _, _) -> off
  in
  Bytes.sub_string b 0 (go 0)

(* The bytes of [fd] until its stream ends, or no byte comes within
   [patience]. *)
let read_all fd =
  let buf = Buffer.create 65536 in
  let rec go () =
    let s = read_n fd 65536 in
    Buffer.add_string buf s;
    if String.length s = 65536 then go ()
  in
  go ();
  Buffer.contents buf

let write fd s = ignore (Unix.write_substring fd s 0 (String.length s))

(* Writes to the non-blocking [fd] until its connection takes no byte: its
   peer's window is closed once [fd] stays unwritable for half a second, which
   on loopback acknowledgements take microseconds to open. *)
let rec fill fd =
  let rec write_all chunk =
    match write fd chunk with
    | () -> write_all chunk
    | exception Unix.Unix_error ((Unix.EAGAIN | Unix.EWOULDBLOCK), _, _) -> ()
  in
  write_all (String.make 4096 '\000');
  write_all "\000";
  match Unix.select [] [ fd ] [] 0.5 with [], [], [] -> () | _ -> fill fd


(* Frames *)

let str s =
  let n = Bytes.create 4 in
  Bytes.set_int32_le n 0 (Int32.of_int (String.length s));
  Bytes.to_string n ^ s

let beat =
  let b = Bytes.make 9 '\000' in
  Bytes.set b 8 (Char.chr k_beat);
  Bytes.to_string b

(* The kind and payload length of the next frame of [fd] other than a beat, or
   [None] once its stream ends. *)
let rec header fd =
  let h = read_n fd 9 in
  if String.length h < 9 then None
  else
    let n = Int64.to_int (String.get_int64_le h 0) and k = Char.code h.[8] in
    if k = k_beat then header fd else Some (k, n)

(* The frames of [fd] other than beats, until its stream ends. *)
let frames fd =
  let rec go acc =
    match header fd with
    | None -> List.rev acc
    | Some (k, n) ->
        let p = read_n fd n in
        if String.length p < n then failf "a frame of kind %d cut" k;
        go ((k, p) :: acc)
  in
  go []

(* Writes a beat to [fd] each half second until the result is called: a peer
   alive whatever it reads. *)
let beating fd =
  let on = Atomic.make true in
  let t =
    Thread.create
      (fun () ->
        while Atomic.get on do
          (try write fd beat with Unix.Unix_error _ -> ());
          Thread.delay 0.5
        done)
      ()
  in
  fun () ->
    Atomic.set on false;
    Thread.join t

(* Threads *)

(* Runs [f] on a thread: [join] is its result, or the exception it raised. *)
let spawn f =
  let r = ref None in
  let t =
    Thread.create
      (fun () -> r := Some (match f () with v -> Ok v | exception e -> Error e))
      ()
  in
  fun () ->
    Thread.join t;
    match Option.get !r with Ok v -> v | Error e -> raise e

let timed f =
  let t0 = Unix.gettimeofday () in
  let r = f () in
  (r, Unix.gettimeofday () -. t0)

(* Witnesses *)

let state =
  Testable.structural ~pp:(fun ppf -> function
    | Link.Open -> Format.pp_print_string ppf "open"
    | Link.Closed -> Format.pp_print_string ppf "closed"
    | Link.Failed why -> Format.fprintf ppf "failed: %S" why)

let pp_frame ppf (k, p) =
  if String.length p <= 64 then Format.fprintf ppf "kind %d, %S" k p
  else Format.fprintf ppf "kind %d, %d bytes" k (String.length p)

let frame = Testable.structural ~pp:pp_frame

let refused =
  Testable.structural ~pp:(fun ppf -> function
    | `Refused w -> Format.fprintf ppf "refused %S" w
    | `Failed w -> Format.fprintf ppf "failed %S" w)

(* Counts *)

let t5 = { Rig_remote_abi.src = 0; dst = 0; length = 5 }

let failed_counts ~msg (e : Rig_remote_abi.end_) =
  let count at =
    String.get_int64_le (String.init 8 (fun i -> e.counts.{at + i})) 0
  in
  equal ~msg (list int64)
    [ Int64.max_int; Int64.max_int; Int64.max_int ]
    (List.map count [ 0; 128; 256 ])

(* Scenes *)

type scene = {
  job : Link.job;
  l : Link.t;  (** The faulted link, named "peer". *)
  d : Unix.file_descr;  (** [l]'s socket. *)
  p : Unix.file_descr;  (** [l]'s peer. *)
  h : Link.t;  (** The well link, named "other". *)
  q : Unix.file_descr;  (** [h]'s peer, which beats and answers nothing. *)
  mutable severed : bool;  (** [p] is closed. *)
}

(* Ends the faulted link's connection with a reset: the link's next send or
   receive fails. *)
let sever s =
  Unix.setsockopt_optint s.p Unix.SO_LINGER (Some 0);
  Unix.close s.p;
  s.severed <- true


(* Runs [f] on a new job of the two links, [l] the end of a [role] process's
   peer. *)
let with_scene ?(role = Wire.Agent 1) f =
  let job = Link.job () in
  let d, p = connected () and e, q = connected () in
  let stop = beating q in
  let h = Link.make job e ~name:"other" ~peer:(Wire.Agent 2) in
  let l = Link.make job d ~name:"peer" ~peer:role in
  let s = { job; l; d; p; h; q; severed = false } in
  Fun.protect
    ~finally:(fun () ->
      Probe.resume ();
      stop ();
      if Link.wait job ~ms:0 = Link.Open then Link.fail job "the test ended";
      if not s.severed then Unix.close p;
      Unix.close q)
    (fun () -> f s)

(* Starts the waiters of [s]'s failure: a wait on the job, a request to [h]'s
   peer, which reads it and never answers, and a rail of each link. The result
   checks, once given the root cause, that each woke with it, that each rail's
   counts read [Int64.max_int], and that [h]'s peer was sent the abort. *)
let waiters s =
  let el = Link.rail s.l ~id:1 ~send:[| t5 |] ~receive:[| t5 |] in
  let eh = Link.rail s.h ~id:1 ~send:[| t5 |] ~receive:[| t5 |] in
  let waited = spawn (fun () -> Link.wait s.job ~ms:30_000) in
  let requested = spawn (fun () -> Link.request s.h (Wire.Open "MEM")) in
  (match header s.q with
  | Some (k, n) when k = k_request -> ignore (read_n s.q n)
  | _ -> fail "the other link's peer was sent no request");
  fun cause ->
    equal ~msg:"the job" state (Link.Failed cause) (waited ());
    equal ~msg:"the request" (result pass refused)
      (Error (`Failed cause))
      (requested ());
    failed_counts ~msg:"the faulted link's rail" el;
    failed_counts ~msg:"the other link's rail" eh;
    equal ~msg:"what the other link's peer received" (list frame)
      [ (k_abort, str cause) ]
      (frames s.q)

(* Before a frame: an idle link. A severed one fails the job. A stalled peer,
   which sends nothing, fails it after 10 s, and is sent the abort: the link's
   sends did not fail. *)

let severed_idle () =
  with_scene @@ fun s ->
  let woke = waiters s in
  sever s;
  woke reset_cause

let stalled_idle () =
  with_scene @@ fun s ->
  let woke = waiters s in
  let (), took = timed (fun () -> woke "peer: silent for 10 s") in
  at_least float_exact ~than:(silence -. 1.) took;
  equal ~msg:"what the stalled peer received" (list frame)
    [ (k_abort, str "peer: silent for 10 s") ]
    (frames s.p)

(* A failed send cuts its link's stream. The sender is held once it released
   the link, while the stall takes 10 s and each observation at most 5. *)
let held_cut () =
  let rec held n = n > 0 && (Probe.held () || held (n - 1)) in
  equal ~msg:"the failed sender is held" bool true (held 3)

(* The bytes of [s] from its first that is not [c], if any. *)
let stray c s =
  match String.to_seq s |> Seq.find_index (fun x -> x <> c) with
  | None -> None
  | Some i -> Some (String.sub s i (min 32 (String.length s - i)))

(* Mid-frame: an agent's end sends the bytes of a copy too large for the
   connection, and its peer has read the frame's first bytes. A second message
   given meanwhile and the agent's wait for a command return once the job
   failed. *)

let mid_frame s =
  let a = Bigarray.(Array1.create char c_layout large) in
  Bigarray.Array1.fill a 'x';
  let sent = spawn (fun () -> Link.bytes s.l ~device:0 ~value:1 a) in
  (match header s.p with
  | Some (k, n) when k = k_bytes -> equal ~msg:"the frame" int (16 + large) n
  | _ -> fail "the peer was sent no bytes frame");
  ignore (read_n s.p (16 + 4096));
  let next = spawn (fun () -> Link.next s.l) in
  let word = spawn (fun () -> Link.word s.l ~device:0 1) in
  fun cause ->
    sent ();
    word ();
    equal ~msg:"next" (result pass string) (Error cause) (next ())

let severed_mid_frame () =
  with_scene ~role:Wire.Controller @@ fun s ->
  let woke = waiters s in
  let returned = mid_frame s in
  sever s;
  woke reset_cause;
  returned reset_cause

(* The second message waits behind the frame. Once the failed sender released
   the link, and while it is held there, its peer reads the stream to its end:
   the frame's bytes, cut, and no message or abort after them. *)
let stalled_mid_frame () =
  with_scene ~role:Wire.Controller @@ fun s ->
  let stop = beating s.p in
  Fun.protect ~finally:stop @@ fun () ->
  let woke = waiters s in
  let t0 = Unix.gettimeofday () in
  let returned = mid_frame s in
  Probe.arm Cut;
  held_cut ();
  let rest = read_all s.p in
  Probe.resume ();
  equal ~msg:"the bytes after the copy's" (option string) None (stray 'x' rest);
  less ~msg:"the frame's bytes the peer read" int ~than:(large - 4096)
    (String.length rest);
  woke stalled_cause;
  let took = Unix.gettimeofday () -. t0 in
  at_least float_exact ~than:(silence -. 1.) took;
  less float_exact ~than:(silence +. 3.) took;
  returned stalled_cause

(* During a rail transfer: the sending thread sends a transfer too large for
   the connection. *)

let rail s =
  let t = { Rig_remote_abi.src = 0; dst = 0; length = large } in
  let e = Link.rail s.l ~id:2 ~send:[| t |] ~receive:[||] in
  Bigarray.Array1.fill e.outbound 'r';
  e

let severed_rail () =
  with_scene @@ fun s ->
  let woke = waiters s in
  let e = rail s in
  Probe.arm Sending;
  e.ready 1;
  equal ~msg:"the sending thread holds the transfer" bool true (Probe.held ());
  sever s;
  Probe.resume ();
  woke reset_cause;
  failed_counts ~msg:"the transfer's rail" e

(* A drop waits behind the transfer. Once the failed sender released the link,
   and while it is held there, its peer reads the stream to its end: the
   transfer's bytes, cut, and no message or abort after them. *)
let stalled_rail () =
  with_scene @@ fun s ->
  let stop = beating s.p in
  Fun.protect ~finally:stop @@ fun () ->
  let woke = waiters s in
  let e = rail s in
  let t0 = Unix.gettimeofday () in
  e.ready 1;
  let dropped = spawn (fun () -> Link.drop s.l 7) in
  Probe.arm Cut;
  held_cut ();
  let n, rest =
    match header s.p with
    | Some (k, n) when k = k_rail -> (n, read_all s.p)
    | _ -> fail "the peer was sent no rail frame"
  in
  Probe.resume ();
  equal ~msg:"the bytes after the transfer's" (option string) None
    (stray 'r' (String.sub rest 16 (String.length rest - 16)));
  less ~msg:"the frame's bytes the peer read" int ~than:n (String.length rest);
  woke stalled_cause;
  let took = Unix.gettimeofday () -. t0 in
  at_least float_exact ~than:(silence -. 1.) took;
  less float_exact ~than:(silence +. 3.) took;
  dropped ();
  failed_counts ~msg:"the transfer's rail" e

(* During the abort: the job fails, and the faulted link's abort is held as it
   starts. A fault then leaves the root cause as it was, and [fail] returns,
   within 10 s if the peer reads nothing. *)

let why = "the test fails the job"

let severed_abort () =
  with_scene @@ fun s ->
  let woke = waiters s in
  Probe.arm Aborting;
  let failed = spawn (fun () -> Link.fail s.job why) in
  equal ~msg:"the abort is held" bool true (Probe.held ());
  sever s;
  Probe.resume ();
  failed ();
  woke why

(* The peer has read nothing the link sent: while the abort is held, the test
   fills the link's socket, as frames its peer left unread would. *)
let stalled_abort () =
  with_scene @@ fun s ->
  let woke = waiters s in
  Probe.arm Aborting;
  let (), took =
    timed (fun () ->
        let failed = spawn (fun () -> Link.fail s.job why) in
        equal ~msg:"the abort is held" bool true (Probe.held ());
        fill s.d;
        Probe.resume ();
        failed ())
  in
  less float_exact ~than:(silence +. 3.) took;
  woke why

let () =
  Watchdog.start ();
  exit
    (run "rig_remote_proxy.link_fail"
       [
         group ~timeout:30. "before a frame"
           [
             test "a severed link fails the job, and the others abort"
               severed_idle;
             slow "a peer silent for 10 s fails the job, and is sent the abort"
               stalled_idle;
           ];
         group ~timeout:30. "mid-frame"
           [
             test "a severed link fails the job, and every writer returns"
               severed_mid_frame;
             slow "a peer that reads nothing for 10 s fails the job, cut"
               stalled_mid_frame;
           ];
         group ~timeout:30. "rail transfer"
           [
             test "a severed link fails the job, and every waiter wakes"
               severed_rail;
             slow "a peer that reads nothing for 10 s fails the job, cut"
               stalled_rail;
           ];
         group ~timeout:30. "abort"
           [
             test "a severed link keeps the root cause, and fail returns"
               severed_abort;
             slow "a peer that reads nothing keeps the cause; fail returns"
               stalled_abort;
           ];
       ])
