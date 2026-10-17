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

(* The seconds this suite's links bear a silent peer or one that reads nothing,
   shortened from link.mli's 10 ([Probe.bound]). It is below an observation's
   5, so one observation sees a stalled send's end. *)
let bound = 2

(* The seconds this suite's clock may run apart from the link's. *)
let skew = 0.01

(* The seconds a peer waits for a frame's first byte, which a thread of the
   test may send late under load. *)
let patience = 5.

(* The seconds without a byte that end a read of what the link sent, which
   comes at once on loopback. *)
let quiet = 1.

(* The bytes of a frame or transfer no connection holds whole: its send waits
   for its peer to read. *)
let large = 32 lsl 20

let reset_cause = "peer: closed its connection"
let silent_cause = Printf.sprintf "peer: silent for %d s" bound
let stalled_cause = Printf.sprintf "peer: read nothing for %d s" bound
let why = "the test fails the job"

(* Sockets *)

(* A connected pair of loopback TCP sockets: the link's end, whose send buffer
   is small, and its peer's, whose receive buffer is small and whose reads give
   up after [quiet]. The accepted socket takes the listener's receive buffer,
   which Windows' loopback otherwise grows to megabytes its peer has not
   read. *)
let connected () =
  let l = Unix.socket ~cloexec:true Unix.PF_INET Unix.SOCK_STREAM 0 in
  Unix.setsockopt_int l Unix.SO_RCVBUF 4096;
  Unix.bind l (Unix.ADDR_INET (Unix.inet_addr_loopback, 0));
  Unix.listen l 1;
  Fun.protect ~finally:(fun () -> Unix.close l) @@ fun () ->
  let d = Unix.socket ~cloexec:true Unix.PF_INET Unix.SOCK_STREAM 0 in
  Unix.setsockopt_int d Unix.SO_SNDBUF 4096;
  Unix.connect d (Unix.getsockname l);
  let p, _ = Unix.accept ~cloexec:true l in
  Rig_support.recv_timeout p quiet;
  (d, p)

(* The next [n] bytes of [fd], fewer if its stream ends or no byte comes
   within [quiet] first. *)
let read_n fd n =
  let b = Bytes.create n in
  let rec go off =
    if off = n then off
    else
      match Unix.read fd b off (min (n - off) 65536) with
      | 0 -> off
      | k -> go (off + k)
      | exception Unix.Unix_error (Unix.ECONNRESET, _, _) -> off
      | exception
          Unix.Unix_error
            ((Unix.EAGAIN | Unix.EWOULDBLOCK | Unix.ETIMEDOUT), _, _) ->
          off
  in
  Bytes.sub_string b 0 (go 0)

(* The bytes of [fd] until its stream ends, or no byte comes within [quiet]. *)
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

(* Writes [chunk] to the non-blocking [fd] until it takes none: whether it took
   any. *)
let takes fd chunk =
  let rec go took =
    match write fd chunk with
    | () -> go true
    | exception Unix.Unix_error ((Unix.EAGAIN | Unix.EWOULDBLOCK), _, _) -> took
  in
  go false

(* Writes to the non-blocking [fd] until its connection takes no byte for half
   a second: its peer's window is closed, which on loopback acknowledgements
   take microseconds to open. A socket that reports itself unwritable may still
   take bytes, so it writes rather than waits on select. Winsock takes a whole
   write while it holds fewer bytes than the socket's buffer, and Windows'
   loopback goes on taking a few hundred bytes a second from a peer that reads
   nothing: writes of 64 KiB leave held far more than that drains within the
   bound, where writes of 4 KiB left a later small send room. *)
let fill fd =
  let rec go since =
    let took = takes fd (String.make 65536 '\000') || takes fd "\000" in
    let now = Unix.gettimeofday () in
    if took || now -. since < 0.5 then begin
      Thread.delay 0.05;
      go (if took then now else since)
    end
  in
  go (Unix.gettimeofday ())

(* Reads what [fd] holds, waiting for no byte. *)
let drain fd =
  let b = Bytes.create 65536 in
  while
    Unix.select [ fd ] [] [] 0. <> ([], [], []) && Unix.read fd b 0 65536 > 0
  do
    ()
  done

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
   [None] once its stream ends. Its first byte may take [wait] seconds. *)
let rec header ?(wait = patience) fd =
  ignore (Unix.select [ fd ] [] [] wait);
  let h = read_n fd 9 in
  if String.length h < 9 then None
  else
    let n = Int64.to_int (String.get_int64_le h 0) and k = Char.code h.[8] in
    if k = k_beat then header ~wait fd else Some (k, n)

(* The frames of [fd] other than beats, until its stream ends or no byte comes
   within [quiet]. *)
let frames fd =
  let rec go acc =
    match header ~wait:0. fd with
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

(* Checks that the bound passed since [t0]. *)
let after_bound t0 =
  at_least ~msg:"the seconds it took" float_exact
    ~than:(Float.of_int bound -. skew)
    (Unix.gettimeofday () -. t0)

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
  let d, p = connected () and e, q = connected () in
  let stop = beating q in
  let job = Link.job () in
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
   which sends nothing, fails it after the bound, and is sent the abort: the
   link's sends did not fail. *)

let severed_idle () =
  with_scene @@ fun s ->
  let woke = waiters s in
  sever s;
  woke reset_cause

let stalled_idle () =
  let t0 = Unix.gettimeofday () in
  with_scene @@ fun s ->
  let woke = waiters s in
  woke silent_cause;
  after_bound t0;
  equal ~msg:"what the stalled peer received" (list frame)
    [ (k_abort, str silent_cause) ]
    (frames s.p)

(* A next's wait for a command holds no lock. A signal arrives while it
   waits, and the next thread to return to OCaml is the waiting one, as the
   job fails after the bound: its handler runs between the link's locks, and
   a call of rig's from it runs. *)
let handled_in_next () =
  if not Rig_support.sees_waits then
    skip ~reason:"needs the system's view of a thread's wait" ();
  let d, p = connected () in
  let job = Link.job () in
  let l = Link.make job d ~name:"peer" ~peer:Wire.Controller in
  let call = ref "no handler ran" in
  let handle _ =
    call :=
      match Rig.Buffer.create Rig.host 8 with
      | _ -> "ran"
      | exception Invalid_argument why -> why
  in
  let previous = Sys.signal Sys.sigint (Sys.Signal_handle handle) in
  Fun.protect
    ~finally:(fun () ->
      Sys.set_signal Sys.sigint previous;
      Unix.close p)
  @@ fun () ->
  let r, raised = Rig_support.interrupt_in_wait (fun () -> Link.next l) in
  equal ~msg:"the signal raised in the wait" bool true raised;
  (match r with
  | Error why -> equal ~msg:"the next" string silent_cause why
  | Ok _ -> fail "a command from a silent peer");
  equal ~msg:"the handler's call of rig" string "ran" !call

(* A link's thread may see the failure before the thread that began it
   reaches the link. That thread is held once it set the root cause, before
   it sends any abort: the well link's sending thread, which wakes for its
   beat within a second, sees the failure first and sends the abort itself,
   which its peer reads while the failing thread is held. *)
let seen_first () =
  with_scene @@ fun s ->
  Probe.arm Failing;
  let failed = spawn (fun () -> Link.fail s.job why) in
  equal ~msg:"the failing thread is held" bool true (Probe.held ());
  let first =
    match header s.q with
    | Some (k, n) -> Some (k, read_n s.q n)
    | None -> None
  in
  Probe.resume ();
  failed ();
  equal ~msg:"the other link's peer's first frame" (option frame)
    (Some (k_abort, str why))
    first;
  equal ~msg:"what it received after" (list frame) [] (frames s.q)

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
  Probe.arm Cut;
  let returned = mid_frame s in
  equal ~msg:"the failed sender is held" bool true (Probe.held ());
  let rest = read_all s.p in
  Probe.resume ();
  equal ~msg:"the bytes after the copy's" (option string) None (stray 'x' rest);
  less ~msg:"the frame's bytes the peer read" int ~than:(large - 4096)
    (String.length rest);
  woke stalled_cause;
  after_bound t0;
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

(* A drop waits behind the transfer. While the failed sender is held once it
   released the link, its peer reads what the stream holds, and the rest once
   the job has failed: the transfer's bytes, cut, and no message or abort after
   them. *)
let stalled_rail () =
  with_scene @@ fun s ->
  let stop = beating s.p in
  Fun.protect ~finally:stop @@ fun () ->
  let woke = waiters s in
  let e = rail s in
  let t0 = Unix.gettimeofday () in
  Probe.arm Cut;
  e.ready 1;
  let dropped = spawn (fun () -> Link.drop s.l 7) in
  equal ~msg:"the failed sender is held" bool true (Probe.held ());
  let n, held =
    match header s.p with
    | Some (k, n) when k = k_rail -> (n, read_all s.p)
    | _ -> fail "the peer was sent no rail frame"
  in
  Probe.resume ();
  woke stalled_cause;
  after_bound t0;
  let rest = held ^ read_all s.p in
  equal ~msg:"the bytes after the transfer's" (option string) None
    (stray 'r' (String.sub rest 16 (String.length rest - 16)));
  less ~msg:"the frame's bytes the peer read" int ~than:n (String.length rest);
  dropped ();
  failed_counts ~msg:"the transfer's rail" e

(* A release of the cut rail waits for its transfer alone. The thread whose
   send was cut is held as it begins the job's failure, which waits for the
   other links' aborts: two releases of the rail then find no transfer to
   wait for. *)
let stalled_release () =
  with_scene @@ fun s ->
  let stop = beating s.p in
  Fun.protect ~finally:stop @@ fun () ->
  let e = rail s in
  Probe.arm Failing;
  e.ready 1;
  equal ~msg:"the cut sender is held" bool true (Probe.held ());
  let released =
    List.init 2 (fun _ -> spawn (fun () -> Link.release_rail s.l 2))
  in
  equal ~msg:"the releases, and those that waited" (pair int int) (2, 0)
    (Probe.releases ());
  Probe.resume ();
  List.iter (fun join -> join ()) released;
  equal ~msg:"the job" state (Link.Failed stalled_cause)
    (Link.wait s.job ~ms:30_000)

(* During the abort: the job fails, and the faulted link's abort is held as it
   starts. A fault then leaves the root cause as it was, and [fail] returns,
   after the bound if the peer reads nothing. *)

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
   reads what the link sent before, then fills the link's socket, as frames its
   peer left unread would. The abort waits for the bound and gives up: the
   peer receives the test's zeros and none of the abort. *)
let stalled_abort () =
  with_scene @@ fun s ->
  let woke = waiters s in
  Probe.arm Aborting;
  let t0 = Unix.gettimeofday () in
  let failed = spawn (fun () -> Link.fail s.job why) in
  equal ~msg:"the abort is held" bool true (Probe.held ());
  drain s.p;
  fill s.d;
  Probe.resume ();
  failed ();
  after_bound t0;
  equal ~msg:"the bytes after the test's" (option string) None
    (stray '\000' (read_all s.p));
  woke why

let () =
  Watchdog.start ();
  Probe.bound bound;
  exit
    (run "rig_remote_proxy.link_fail"
       [
         group ~timeout:30. "before a frame"
           [
             test "a severed link fails the job, and the others abort"
               severed_idle;
             slow "a silent peer fails the job, and is sent the abort"
               stalled_idle;
             slow "a link's thread that sees the failure first sends its abort"
               seen_first;
             slow
               "a signal handler that runs as a next's wait ends calls rig \
                outside every lock"
               handled_in_next;
           ];
         group ~timeout:30. "mid-frame"
           [
             test "a severed link fails the job, and every writer returns"
               severed_mid_frame;
             slow "a peer that reads nothing fails the job, cut"
               stalled_mid_frame;
           ];
         group ~timeout:30. "rail transfer"
           [
             test "a severed link fails the job, and every waiter wakes"
               severed_rail;
             slow "a peer that reads nothing fails the job, cut" stalled_rail;
             slow "a release of the cut rail waits for its transfer alone"
               stalled_release;
           ];
         group ~timeout:30. "abort"
           [
             test "a severed link keeps the root cause, and fail returns"
               severed_abort;
             slow "a peer that reads nothing keeps the cause; fail returns"
               stalled_abort;
           ];
       ])
