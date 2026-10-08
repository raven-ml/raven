(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Links over loopback TCP. A link's peer is another link of the same job in
   this process, a raw socket that writes and reads the frames wire.mli lays
   out, or support/link_peer.exe, an agent's end in a process of its own. A
   process has one open job at a time, so every test ends its job. *)

open Windtrap
module Wire = Rig_remote_proxy.Wire
module Link = Rig_remote_proxy.Link

(* Frames' kinds, as wire.mli numbers them. *)

(* A request's error: the agent refused it, or the job failed. *)

(* A request's error as one reason, as the model below states it. *)
let reason r = Result.map_error (function `Refused w | `Failed w -> w) r

let refused =
  Testable.structural ~pp:(fun ppf -> function
    | `Refused why -> Format.fprintf ppf "refused: %S" why
    | `Failed why -> Format.fprintf ppf "failed: %S" why)

let k_request = 1
let k_answer = 2
let k_handover = 3
let k_drop = 4
let k_word = 5
let k_bytes = 6
let k_rail = 7
let k_beat = 8
let k_abort = 9
let k_close = 10

(* The time without a byte after which a link fails its job. *)
let silence = 10.

(* The most a raw peer waits for a byte, and for anything the test awaits. *)
let patience = 5.

(* The most a call that answers at once over loopback may take. *)
let prompt = 2.

(* Bytes *)

let u32 n =
  let b = Bytes.create 4 in
  Bytes.set_int32_le b 0 (Int32.of_int n);
  Bytes.to_string b

let u64 n =
  let b = Bytes.create 8 in
  Bytes.set_int64_le b 0 (Int64.of_int n);
  Bytes.to_string b

let str s = u32 (String.length s) ^ s

let frame kind payload =
  u64 (String.length payload) ^ String.make 1 (Char.chr kind) ^ payload

let of_area (a : Rig_remote_abi.area) =
  String.init (Bigarray.Array1.dim a) (fun i -> a.{i})

let pp_bytes ppf s =
  if String.length s <= 24 then Format.fprintf ppf "%S" s
  else Format.fprintf ppf "%d bytes %S..." (String.length s) (String.sub s 0 24)

(* Sockets and threads *)

let loopback () =
  let l = Unix.socket ~cloexec:true Unix.PF_INET Unix.SOCK_STREAM 0 in
  Unix.bind l (Unix.ADDR_INET (Unix.inet_addr_loopback, 0));
  Unix.listen l 1;
  match Unix.getsockname l with
  | Unix.ADDR_INET (_, p) -> (l, p)
  | Unix.ADDR_UNIX _ -> assert false

(* A connected pair of loopback TCP sockets. A read on either gives up after
   [patience]. *)
let connected () =
  let l, port = loopback () in
  Fun.protect
    ~finally:(fun () -> Unix.close l)
    (fun () ->
      let d = Unix.socket ~cloexec:true Unix.PF_INET Unix.SOCK_STREAM 0 in
      Unix.connect d (Unix.ADDR_INET (Unix.inet_addr_loopback, port));
      let a, _ = Unix.accept ~cloexec:true l in
      List.iter
        (fun fd -> Unix.setsockopt_float fd Unix.SO_RCVTIMEO patience)
        [ d; a ];
      (d, a))

(* The next [n] bytes of [fd], fewer if its stream ends first. *)
let read_n fd n =
  let b = Bytes.create n in
  let rec go off =
    if off = n then off
    else
      match Unix.read fd b off (n - off) with
      | 0 -> off
      | k -> go (off + k)
      | exception Unix.Unix_error (Unix.ECONNRESET, _, _) -> off
      | exception Unix.Unix_error ((Unix.EAGAIN | Unix.EWOULDBLOCK), _, _) ->
          failf "no byte within %.0f s" patience
  in
  Bytes.sub_string b 0 (go 0)

let read_all fd =
  let buf = Buffer.create 64 in
  let rec go () =
    let s = read_n fd 4096 in
    Buffer.add_string buf s;
    if String.length s = 4096 then go ()
  in
  go ();
  Buffer.contents buf

let write fd s = ignore (Unix.write_substring fd s 0 (String.length s))

let readable fd s =
  let r, _, _ = Unix.select [ fd ] [] [] s in
  r <> []

let timed f =
  let t0 = Unix.gettimeofday () in
  let r = f () in
  (r, Unix.gettimeofday () -. t0)

(* Runs [f] on a thread: [join] is its result, or the exception it raised. *)
let spawn f =
  let r = ref None in
  let t =
    Thread.create
      (fun () ->
        r := Some (match f () with v -> Ok v | exception e -> Error e))
      ()
  in
  fun () ->
    Thread.join t;
    match Option.get !r with Ok v -> v | Error e -> raise e

(* Waits until [cond ()] holds, for at most [patience]. *)
let until ~what cond =
  let t0 = Unix.gettimeofday () in
  while not (cond ()) do
    if Unix.gettimeofday () -. t0 > patience then
      failf "%s: not within %.0f s" what patience;
    Thread.yield ()
  done

(* Runs [f], failing [j] if it has not returned within [prompt], which ends any
   wait of [f] on [j]: a call that would hang fails the test. *)
let within j ~what f =
  let finished = Atomic.make false in
  let r =
    spawn (fun () ->
        Fun.protect ~finally:(fun () -> Atomic.set finished true) f)
  in
  let t0 = Unix.gettimeofday () in
  while (not (Atomic.get finished)) && Unix.gettimeofday () -. t0 < prompt do
    Thread.delay 0.01
  done;
  let late = not (Atomic.get finished) in
  if late then Link.fail j (what ^ " did not return");
  let v = r () in
  if late then failf "%s did not return within %.0f s" what prompt;
  v

(* Raw frames *)

(* The next frame of [fd], its kind and payload, or [None] once its stream
   ends. *)
let read_frame fd =
  let h = read_n fd 9 in
  if String.length h < 9 then None
  else
    let n = Int64.to_int (String.get_int64_le h 0) in
    if n < 0 || n > 1 lsl 24 then failf "a frame of %d bytes" n;
    let p = read_n fd n in
    if String.length p < n then None else Some (Char.code h.[8], p)

(* The frames of [fd] other than beats, until its stream ends. *)
let frames fd =
  let rec go acc =
    match read_frame fd with
    | None -> List.rev acc
    | Some (k, _) when k = k_beat -> go acc
    | Some f -> go (f :: acc)
  in
  go []

(* The next frame of [fd] other than a beat. *)
let rec next_frame fd =
  match read_frame fd with
  | Some (k, _) when k = k_beat -> next_frame fd
  | f -> f

(* Witnesses *)

let state =
  Testable.structural ~pp:(fun ppf -> function
    | Link.Open -> Format.pp_print_string ppf "open"
    | Link.Closed -> Format.pp_print_string ppf "closed"
    | Link.Failed why -> Format.fprintf ppf "failed: %a" pp_bytes why)

let frame_w =
  Testable.structural ~pp:(fun ppf (k, p) ->
      Format.fprintf ppf "kind %d, %a" k pp_bytes p)

let pp_account ppf (a : Wire.account) =
  Format.fprintf ppf "{id %d; %S; %S; budget %d; reaches [%s]}" a.id a.name
    a.arch a.budget
    (String.concat "; " (List.map string_of_int a.reaches))

let account = Testable.structural ~pp:pp_account

(* Jobs *)

let ended j =
  match Link.wait j ~ms:0 with
  | Link.Open -> Link.fail j "the test ended"
  | _ -> ()

(* A job for [f], failed once [f] returns if it is still open. *)
let with_job f =
  let j = Link.job () in
  Fun.protect ~finally:(fun () -> ended j) (fun () -> f j)

(* A job whose controller's end [c] and agent's end [a] are links of it, to each
   other. *)
let with_pair f =
  with_job @@ fun j ->
  let d, a = connected () in
  let c = Link.make j d ~name:"agent" ~peer:(Wire.Agent 1) in
  let a = Link.make j a ~name:"controller" ~peer:Wire.Controller in
  f j c a

(* A job with one link [l], named "peer", whose peer [p] is a raw socket. The
   link is the controller's end when [peer] is an agent. *)
let with_raw ?(peer = Wire.Agent 1) f =
  with_job @@ fun j ->
  let d, p = connected () in
  let l = Link.make j d ~name:"peer" ~peer in
  Fun.protect ~finally:(fun () -> Unix.close p) (fun () -> f j l p)

(* A process of link_peer *)

let peer_exe =
  Filename.concat (Filename.dirname Sys.executable_name) "support/link_peer.exe"

type helper = { pid : int; out : Unix.file_descr }

(* Starts link_peer in [mode] and makes its link in [j], named [name]. *)
let start j mode ~name =
  let l, port = loopback () in
  Fun.protect
    ~finally:(fun () -> Unix.close l)
    (fun () ->
      let r, w = Unix.pipe ~cloexec:true () in
      let pid =
        Unix.create_process peer_exe
          [| peer_exe; mode; string_of_int port |]
          Unix.stdin w Unix.stderr
      in
      Unix.close w;
      let h = { pid; out = r } in
      if not (readable l patience) then begin
        Unix.kill pid Sys.sigkill;
        failf "link_peer %s did not connect" mode
      end;
      let fd, _ = Unix.accept ~cloexec:true l in
      (h, Link.make j fd ~name ~peer:(Wire.Agent 1)))

(* The lines [h] printed, once it exited with status 0. *)
let finish h =
  let out = read_all h.out in
  Unix.close h.out;
  let _, status = Unix.waitpid [] h.pid in
  (match status with
  | Unix.WEXITED 0 -> ()
  | Unix.WEXITED n -> failf "link_peer exited with %d" n
  | Unix.WSIGNALED n | Unix.WSTOPPED n -> failf "link_peer ended by signal %d" n);
  String.split_on_char '\n' out |> List.filter (( <> ) "")

let kill h =
  (try Unix.kill h.pid Sys.sigkill with Unix.Unix_error _ -> ());
  (try Unix.close h.out with Unix.Unix_error _ -> ());
  try ignore (Unix.waitpid [] h.pid) with Unix.Unix_error _ -> ()

(* Runs [f] with [h], killing [h] if [f] raises. *)
let guard h f =
  match f () with
  | v -> v
  | exception e ->
      kill h;
      raise e

(* link_peer's answers, which this suite states. *)
let echo_open kind =
  Ok [ { Wire.id = 0; name = kind; arch = "echo"; budget = 0; reaches = [] } ]

let echo_entry image name = Ok (if name = "" then None else Some image)
let echo_alloc bytes = Ok (bytes mod 2 = 0)

(* Jobs *)

let one_open_job () =
  let j = Link.job () in
  raises_match Exn.invalid_arg (fun () -> Link.job ());
  Link.fail j "the test ended";
  ended (Link.job ())

let wait_open () =
  with_job @@ fun j ->
  let s, took = timed (fun () -> Link.wait j ~ms:200) in
  equal state Link.Open s;
  at_least float_exact ~than:0.19 took;
  equal (option string) None (Link.failure j)

let first_cause () =
  with_raw @@ fun j _ p ->
  Link.fail j "first";
  Link.fail j "second";
  ignore (frames p);
  equal state (Link.Failed "first") (Link.wait j ~ms:0);
  equal (option string) (Some "first") (Link.failure j)

let empty_close () =
  with_job @@ fun j ->
  let (), took = timed (fun () -> Link.close j) in
  equal state Link.Closed (Link.wait j ~ms:0);
  less float_exact ~than:1. took;
  Link.fail j "too late";
  equal state Link.Closed (Link.wait j ~ms:0)

let job_of_link () =
  with_pair @@ fun j c a ->
  Link.fail (Link.job_of c) "failed through c";
  equal (option string) (Some "failed through c") (Link.failure j);
  equal (option string) (Some "failed through c") (Link.failure (Link.job_of a))

let jobs =
  group "job"
    [
      test "a process has one open job at a time" one_open_job;
      test "wait on an open job is Open after its milliseconds" wait_open;
      test "a job fails once, with its first cause" first_cause;
      test "a job with no link closes at once, and stays closed" empty_close;
      test "job_of is the job a link was made in" job_of_link;
    ]

(* Causes of failure *)

let peer_closes () =
  with_raw @@ fun j _ p ->
  Unix.shutdown p Unix.SHUTDOWN_SEND;
  equal state (Link.Failed "peer: closed its connection") (Link.wait j ~ms:2000)

let peer_resets () =
  with_raw @@ fun j _ p ->
  Unix.setsockopt_optint p Unix.SO_LINGER (Some 0);
  Unix.shutdown p Unix.SHUTDOWN_ALL;
  match Link.wait j ~ms:2000 with
  | Link.Failed why -> starts_with ~affix:"peer: " why
  | s -> failf "the job is %a" (Testable.pp state) s

(* Frames an agent may not send its controller, or that no reader decodes. *)
let malformed_from_agent =
  [
    ("a frame of kind 0", frame 0 "");
    ("a frame of kind 11", frame 11 "");
    ("a frame of kind 255", frame 255 "");
    ("a request", frame k_request "");
    ("a hand-over", frame k_handover "");
    ("a drop", frame k_drop (u64 1));
    ("a beat with a payload", frame k_beat "x");
    ("a close with a payload", frame k_close "x");
    ("a word of 15 bytes", frame k_word (String.make 15 '\000'));
  ]

let malformed_from_controller =
  [
    ("an answer", frame k_answer ("\001" ^ str "no"));
    ("a word", frame k_word (u64 0 ^ u64 1));
    ("a bytes frame", frame k_bytes (u64 0 ^ u64 1 ^ "x"));
    ("a drop of 7 bytes", frame k_drop (String.make 7 '\000'));
    ( "a hand-over copying Local to Local",
      frame k_handover
        (u64 0 ^ u64 1 ^ u32 0 ^ u32 1 ^ "\001" ^ u64 4 ^ "\001" ^ "\001"
       ^ "abcd") );
    ( "a hand-over with a part of kind 2",
      frame k_handover (u64 0 ^ u64 1 ^ u32 0 ^ u32 1 ^ "\002") );
  ]

let malformed_at_controller (_, f) =
  with_raw @@ fun j _ p ->
  write p f;
  equal state (Link.Failed "peer: a malformed frame") (Link.wait j ~ms:2000)

(* An agent's end decodes a command when next reads it. *)
let malformed_at_agent (_, f) =
  with_raw ~peer:Wire.Controller @@ fun j l p ->
  write p f;
  equal (result pass string) (Error "peer: a malformed frame")
    (within j ~what:"next" (fun () -> Link.next l));
  equal state (Link.Failed "peer: a malformed frame") (Link.wait j ~ms:0)

(* A length no process holds, read as a u64: the link must refuse it before it
   allocates. *)
let too_long n =
  with_raw @@ fun j _ p ->
  let b = Bytes.create 9 in
  Bytes.set_int64_le b 0 n;
  Bytes.set b 8 (Char.chr k_beat);
  write p (Bytes.to_string b);
  match Link.wait j ~ms:2000 with
  | Link.Failed _ -> ()
  | s -> failf "the job is %a" (Testable.pp state) s

let no_object (_, f) =
  with_raw @@ fun j _ p ->
  write p f;
  match Link.wait j ~ms:2000 with
  | Link.Failed _ -> ()
  | s -> failf "the job is %a" (Testable.pp state) s

(* A root cause crosses to the peer's process and fails its job there,
   unchanged. *)
let reaches_process ?cut why =
  with_job @@ fun j ->
  let h, _ = start j "idle" ~name:"agent" in
  guard h @@ fun () ->
  Link.fail j why;
  let why = match cut with Some n -> String.sub why 0 n | None -> why in
  equal (list string) [ "failed: " ^ why ] (finish h)

(* Ids are non-negative: a request that names a negative one raises before it
   sends a byte. *)
let negative_id () =
  with_raw @@ fun _ l p ->
  raises_match Exn.invalid_arg (fun () ->
      Link.request l (Wire.Entry { image = -1; name = "f" }));
  raises_match Exn.invalid_arg (fun () -> Link.drop l (-1));
  Link.drop l 7;
  equal (option frame_w) (Some (k_drop, u64 7)) (next_frame p)

let reasons = [ "the agent lost its GPU"; ""; "\xe2\x9c\x93 \xff" ]

(* wire.mli lays a string out as its length (u32) and its bytes. *)
let why_layout () =
  with_raw (fun j _ p ->
      write p (frame k_abort (str "lost"));
      equal ~msg:"an abort read" state (Link.Failed "lost")
        (Link.wait j ~ms:2000));
  with_raw (fun j _ p ->
      Link.fail j "gone";
      equal ~msg:"an abort sent" (list frame_w)
        [ (k_abort, str "gone") ]
        (frames p));
  with_raw (fun _ l p ->
      let peer =
        spawn (fun () ->
            ignore (next_frame p);
            write p (frame k_answer ("\001" ^ str "no such kind")))
      in
      let r = Link.request l (Wire.Open "GPU") in
      peer ();
      equal ~msg:"a refusal read" (result pass refused)
        (Error (`Refused "no such kind"))
        r);
  with_raw (fun j _ p ->
      write p (frame k_abort (u32 100 ^ "why"));
      equal ~msg:"an abort whose reason runs past it" state
        (Link.Failed "peer: a malformed frame") (Link.wait j ~ms:2000))

let causes =
  group "cause"
    [
      test "a peer that ends its stream without a close fails the job"
        peer_closes;
      test "a peer that resets the connection fails the job with the error"
        peer_resets;
      cases
        ~name:(fun (n, _) -> n ^ " from an agent fails the job as malformed")
        "malformed, at the controller" malformed_from_agent
        malformed_at_controller;
      cases
        ~name:(fun (n, _) ->
          n ^ " from the controller fails the job as malformed")
        "malformed, at an agent" malformed_from_controller malformed_at_agent;
      cases
        ~name:(fun n -> Printf.sprintf "a frame of %Lu bytes fails the job" n)
        "too long"
        [ -1L; Int64.min_int; Int64.max_int; 0x10000000000L ]
        too_long;
      cases
        ~name:(fun (n, _) -> n ^ " fails the job")
        "no object"
        [
          ("a word of a device with no proxy", frame k_word (u64 7 ^ u64 1));
          ("a rail frame of no rail", frame k_rail (u64 99 ^ u64 1 ^ "abc"));
        ]
        no_object;
      cases
        ~name:(fun s ->
          Format.asprintf "a root cause reaches the peer's process as %a"
            pp_bytes s)
        "reason" reasons reaches_process;
      test "a root cause with a NUL reaches the peer's process whole" (fun () ->
          reaches_process "nul\000byte");
      test "a root cause of 70000 bytes reaches the peer's process cut to 4096"
        (fun () -> reaches_process ~cut:4096 (String.make 70_000 'w'));
      test "a frame's why is a string, as wire.mli lays strings out" why_layout;
      test "a request naming a negative id raises and sends nothing" negative_id;
    ]

(* After a failure *)

let kinds fd = List.map fst (frames fd)

let abort_reaches_peer () =
  with_raw @@ fun j _ p ->
  Link.fail j "the test fails it";
  equal (list int) [ k_abort ] (kinds p)

let malformed_reaches_peer () =
  with_raw @@ fun j _ p ->
  write p (frame 0 "");
  equal (list int) [ k_abort ] (kinds p);
  equal state (Link.Failed "peer: a malformed frame") (Link.wait j ~ms:0)

(* The frames of the bytes [s], other than beats and rails, up to the last whole
   one. *)
let frames_of s =
  let rec go off acc =
    if off + 9 > String.length s then List.rev acc
    else
      let n = Int64.to_int (String.get_int64_le s off) in
      let k = Char.code s.[off + 8] in
      if off + 9 + n > String.length s then List.rev acc
      else if k = k_beat || k = k_rail then go (off + 9 + n) acc
      else go (off + 9 + n) ((k, String.sub s (off + 9) n) :: acc)
  in
  go 0 []

(* A busy rail fills the connection; once the job fails, the peer keeps the
   connection nearly full for a while and beats, so beats arrive after the
   failed side ended its stream. Its stream still ends with the abort. *)
let abort_behind_rail () =
  with_raw @@ fun j l p ->
  let t = { Rig_remote_abi.src = 0; dst = 0; length = 1 lsl 16 } in
  let e = Link.rail l ~id:1 ~send:[| t |] ~receive:[||] in
  e.ready 1_000_000_000;
  if not (readable p patience) then fail "the rail sent nothing";
  Thread.delay 0.1 (* the rail fills the connection *);
  Link.fail j "the test fails it";
  let beating = Atomic.make true in
  let beats =
    Thread.create
      (fun () ->
        while Atomic.get beating do
          (try write p (frame k_beat "") with Unix.Unix_error _ -> ());
          Thread.delay 0.02
        done)
      ()
  in
  let got = Buffer.create (1 lsl 20) in
  let slow_until = Unix.gettimeofday () +. 1.5 in
  let rec slowly () =
    let s = read_n p 16384 in
    Buffer.add_string got s;
    if String.length s = 16384 && Unix.gettimeofday () < slow_until then begin
      Thread.delay 0.05;
      slowly ()
    end
  in
  slowly ();
  Buffer.add_string got (read_all p);
  Atomic.set beating false;
  Thread.join beats;
  equal (list frame_w)
    [ (k_abort, str "the test fails it") ]
    (frames_of (Buffer.contents got))

(* The job fails on one link; the other link's peer receives an abort. *)
let every_link_aborts () =
  with_job @@ fun j ->
  let d1, p1 = connected () in
  let d2, p2 = connected () in
  Fun.protect
    ~finally:(fun () ->
      Unix.close p1;
      Unix.close p2)
    (fun () ->
      let _ = Link.make j d1 ~name:"one" ~peer:(Wire.Agent 1) in
      let _ = Link.make j d2 ~name:"two" ~peer:(Wire.Agent 2) in
      Unix.shutdown p1 Unix.SHUTDOWN_SEND;
      equal (list int) [ k_abort ] (kinds p2);
      equal state (Link.Failed "one: closed its connection") (Link.wait j ~ms:0))

let after_failure () =
  with_raw @@ fun j l p ->
  Link.fail j "gone";
  equal (result pass refused)
    (Error (`Failed "gone"))
    (Link.request l (Wire.Open "MEM"));
  equal (result pass string) (Error "gone") (Link.next l);
  Link.drop l 3;
  equal string "peer" (Link.name l);
  equal (list int) [ k_abort ] (kinds p)

(* A request once the job's close began answers that the job is closed, as a
   hand-over does. *)
let during_close () =
  with_raw @@ fun j l p ->
  let closed = spawn (fun () -> Link.close j) in
  equal ~msg:"the close sent" (option int) (Some k_close)
    (Option.map fst (next_frame p));
  let r = Link.request l (Wire.Open "MEM") in
  Link.fail j "the test ends";
  ignore (closed ());
  equal (result pass refused) (Error (`Failed "the job is closed")) r

let failed_meanwhile () =
  with_raw @@ fun j l p ->
  let failer =
    spawn (fun () ->
        ignore (next_frame p);
        Link.fail j "failed meanwhile")
  in
  let r = Link.request l (Wire.Open "MEM") in
  failer ();
  equal (result pass refused) (Error (`Failed "failed meanwhile")) r

let make_on_failed () =
  with_job @@ fun j ->
  Link.fail j "gone";
  let d, p = connected () in
  Fun.protect
    ~finally:(fun () -> Unix.close p)
    (fun () ->
      let l = Link.make j d ~name:"late" ~peer:(Wire.Agent 1) in
      equal string "late" (Link.name l);
      equal (result pass refused)
        (Error (`Failed "gone"))
        (Link.request l (Wire.Open "MEM"));
      ignore (frames p))

let make_on_closed () =
  with_job @@ fun j ->
  Link.close j;
  let d, p = connected () in
  Fun.protect
    ~finally:(fun () ->
      Unix.close d;
      Unix.close p)
    (fun () ->
      raises_match Exn.invalid_arg (fun () ->
          Link.make j d ~name:"late" ~peer:(Wire.Agent 1)))

let failures =
  group "failure"
    [
      test "a failed job sends each peer an abort, then ends its stream"
        abort_reaches_peer;
      test "a malformed frame's sender receives the abort"
        malformed_reaches_peer;
      test "a failed job's abort reaches a peer still reading a busy rail"
        abort_behind_rail;
      test "a job that fails on one link aborts every other" every_link_aborts;
      test "a failed job's requests and next answer its root cause"
        after_failure;
      test "a request during the close answers that the job is closed"
        during_close;
      test "a request waiting when the job fails answers its root cause"
        failed_meanwhile;
      test "a link made on a failed job is failed, and its socket ends"
        make_on_failed;
      test "make on a closed job raises" make_on_closed;
    ]

(* Beats and close *)

let beats () =
  with_raw @@ fun _ _ p ->
  let f, took = timed (fun () -> read_frame p) in
  equal (option frame_w) (Some (k_beat, "")) f;
  less float_exact ~than:1.5 took

let close_with_raw ~delay () =
  with_raw @@ fun j _ p ->
  let peer =
    spawn (fun () ->
        let f = next_frame p in
        Thread.delay delay;
        write p (frame k_close "");
        (f, frames p))
  in
  let (), took = timed (fun () -> Link.close j) in
  let f, rest = peer () in
  equal (option frame_w) (Some (k_close, "")) f;
  equal (list frame_w) [] rest;
  equal state Link.Closed (Link.wait j ~ms:0);
  at_least float_exact ~than:delay took

let close_peer_ends () =
  with_raw @@ fun j _ p ->
  let peer =
    spawn (fun () ->
        ignore (next_frame p);
        Unix.shutdown p Unix.SHUTDOWN_SEND)
  in
  Link.close j;
  peer ();
  equal state (Link.Failed "peer: closed its connection") (Link.wait j ~ms:0)

let close_pair () =
  with_pair @@ fun j _ _ ->
  Link.close j;
  equal state Link.Closed (Link.wait j ~ms:0)

(* Frames queued before the close reach the agent before it. *)
let close_helper () =
  with_job @@ fun j ->
  let h, l = start j "serve" ~name:"agent" in
  guard h @@ fun () ->
  Link.drop l 5;
  Link.drop l 6;
  within j ~what:"close" (fun () -> Link.close j);
  equal state Link.Closed (Link.wait j ~ms:0);
  equal (list string) [ "drop 5"; "drop 6"; "close"; "closed" ] (finish h)

let closes =
  group "close"
    [
      test "a link sends a beat after a second without a frame" beats;
      test "close sends a close and returns once the peer's came"
        (close_with_raw ~delay:0.);
      test "close waits for the peer's close" (close_with_raw ~delay:0.5);
      test "close returns Failed when the peer ends without its close"
        close_peer_ends;
      test "a job closed at both ends of its links is closed" close_pair;
      test "an agent's process sees the drops queued before the close, then it"
        close_helper;
    ]

(* Requests

   A request crosses to the agent unchanged, and the agent's answer comes back
   unchanged: a law over every kind of request and answer. *)

type case = Case : 'a Wire.request * ('a, string) result * 'a testable -> case

let same : type a b. a Wire.request -> b Wire.request -> (a, b) Type.eq option =
 fun r r' ->
  let eq c = if c then Some Type.Equal else None in
  match (r, r') with
  | Wire.Join _, Wire.Join _ -> eq (r = r')
  | Wire.Open _, Wire.Open _ -> eq (r = r')
  | Wire.Alloc _, Wire.Alloc _ -> eq (r = r')
  | Wire.Map _, Wire.Map _ -> eq (r = r')
  | Wire.Load _, Wire.Load _ -> eq (r = r')
  | Wire.Entry _, Wire.Entry _ -> eq (r = r')
  | Wire.Rail _, Wire.Rail _ -> eq (r = r')
  | _ -> None

let pp_process ppf = function
  | Wire.Controller -> Format.pp_print_string ppf "controller"
  | Wire.Agent i -> Format.fprintf ppf "agent %d" i

let pp_transfers ppf a =
  Array.iter
    (fun (t : Rig_remote_abi.transfer) ->
      Format.fprintf ppf "(%d->%d,%d)" t.src t.dst t.length)
    a

let pp_request : type a. Format.formatter -> a Wire.request -> unit =
 fun ppf -> function
  | Wire.Join { agents } ->
      Format.fprintf ppf "Join [%s]"
        (String.concat "; "
           (List.map
              (fun (a : Wire.agent) ->
                Printf.sprintf "%S %S:%d" a.name a.host a.port)
              agents))
  | Wire.Open k -> Format.fprintf ppf "Open %a" pp_bytes k
  | Wire.Alloc { id; device; memory; bytes } ->
      Format.fprintf ppf "Alloc {id %d; device %d; %s; bytes %d}" id device
        (match memory with
        | `Device -> "device"
        | `Pinned -> "pinned"
        | `Mapped -> "mapped")
        bytes
  | Wire.Map { id; device; region } ->
      Format.fprintf ppf "Map {id %d; device %d; region %d}" id device region
  | Wire.Load { id; binary } ->
      Format.fprintf ppf "Load {id %d; %a}" id pp_bytes binary
  | Wire.Entry { image; name } ->
      Format.fprintf ppf "Entry {image %d; %a}" image pp_bytes name
  | Wire.Rail { id; peer; send; receive } ->
      Format.fprintf ppf "Rail {id %d; %a; send %a; receive %a}" id pp_process
        peer pp_transfers send pp_transfers receive

let pp_case ppf (Case (r, a, w)) =
  Format.fprintf ppf "%a -> %a" pp_request r (Testable.pp (result w string)) a

let id =
  Gen.frequency
    [
      (4, Gen.int_range 0 1000);
      ( 1,
        Gen.of_list ~pp:Format.pp_print_int
          [ 0; 1; (1 lsl 32) - 1; 1 lsl 32; max_int ] );
    ]

let text =
  Gen.frequency
    [
      (4, Gen.string);
      (1, Gen.of_list ~pp:pp_bytes [ ""; "CUDA"; "nul\000"; "\xff\xfe" ]);
    ]

let small_list g = Gen.list ~size:(Gen.int_range 0 3) g
let small_array g = Gen.array ~size:(Gen.int_range 0 3) g

let transfer =
  Gen.map
    (fun (src, dst, length) -> { Rig_remote_abi.src; dst; length })
    (Gen.triple id id (Gen.int_range 1 1_000_000))

let account_g =
  let open Gen in
  let+ id = id
  and+ name = text
  and+ arch = text
  and+ budget = id
  and+ reaches = small_list id in
  { Wire.id; name; arch; budget; reaches }

(* A request and an answer the agent gives it. *)
let case_g =
  let open Gen in
  let answer ok =
    frequency
      [ (3, map (fun v -> Ok v) ok); (1, map (fun why -> Error why) text) ]
  in
  one_of
    [
      (let+ agents =
         small_list
           (let+ name = text and+ host = text and+ port = int_range 0 65535 in
            { Wire.name; host; port })
       and+ a = answer account_g in
       Case (Wire.Join { agents }, a, account));
      (let+ k = text and+ a = answer (small_list account_g) in
       Case (Wire.Open k, a, Windtrap.list account));
      (let+ id = id
       and+ device = id
       and+ memory =
         of_list
           ~pp:(fun ppf _ -> Format.pp_print_string ppf "memory")
           [ `Device; `Pinned; `Mapped ]
       and+ bytes = id
       and+ a = answer bool in
       Case (Wire.Alloc { id; device; memory; bytes }, a, Windtrap.bool));
      (let+ id = id and+ device = id and+ region = id and+ a = answer bool in
       Case (Wire.Map { id; device; region }, a, Windtrap.bool));
      (let+ id = id and+ binary = string and+ a = answer unit in
       Case (Wire.Load { id; binary }, a, Windtrap.unit));
      (let+ image = id and+ name = text and+ a = answer (option id) in
       Case (Wire.Entry { image; name }, a, Windtrap.(option int)));
      (let+ id = id
       and+ peer =
         one_of
           [
             constant Wire.Controller;
             map (fun i -> Wire.Agent i) (int_range 1 9);
           ]
       and+ send = small_array transfer
       and+ receive = small_array transfer
       and+ a = answer unit in
       Case (Wire.Rail { id; peer; send; receive }, a, Windtrap.unit));
    ]

(* The agent's end answers each request with its case's answer, failing the job
   if the request it reads is not the case's. *)
let agent j a cases =
  List.iter
    (fun (Case (r, ans, _)) ->
      match Link.next a with
      | Ok (Wire.Request r') -> (
          match same r r' with
          | Some Type.Equal -> Link.answer a r' ans
          | None ->
              Link.fail j (Format.asprintf "the agent read %a" pp_request r'))
      | Ok _ -> Link.fail j "the agent read another command"
      | Error _ -> ())
    cases

let requests_law cases =
  with_pair @@ fun j c a ->
  let served = spawn (fun () -> agent j a cases) in
  List.iter
    (fun (Case (r, ans, w)) ->
      cover "a refusal" (Result.is_error ans);
      equal (result w refused)
        (Result.map_error (fun why -> `Refused why) ans)
        (Link.request c r))
    cases;
  served ();
  equal state Link.Open (Link.wait j ~ms:0)

let large_load =
  [
    Case
      ( Wire.Load { id = 1; binary = String.make (1 lsl 20) 'b' },
        Ok (),
        Windtrap.unit );
  ]

let refused_then_answered () =
  with_raw @@ fun j l p ->
  let peer =
    spawn (fun () ->
        let f = next_frame p in
        write p (frame k_answer ("\001" ^ str "no such kind"));
        f)
  in
  is_error (Link.request l (Wire.Open "GPU"));
  equal (option int) (Some k_request) (Option.map fst (peer ()));
  equal state Link.Open (Link.wait j ~ms:0)

let undecodable_answer () =
  with_raw @@ fun j l p ->
  let peer =
    spawn (fun () ->
        ignore (next_frame p);
        write p (frame k_answer "\000"))
  in
  let r = Link.request l (Wire.Open "GPU") in
  peer ();
  let why = require_some (Link.failure j) in
  equal (result pass refused) (Error (`Failed why)) r

(* Frames leave in the order they were queued. *)
let drops_in_order () =
  with_raw @@ fun _ l p ->
  Link.drop l 5;
  Link.drop l max_int;
  let requested = spawn (fun () -> Link.request l (Wire.Open "MEM")) in
  let f1 = next_frame p and f2 = next_frame p and f3 = next_frame p in
  write p (frame k_answer ("\001" ^ str "done"));
  ignore (requested ());
  equal (list frame_w)
    [ (k_drop, u64 5); (k_drop, u64 max_int) ]
    (List.filter_map Fun.id [ f1; f2 ]);
  equal (option int) (Some k_request) (Option.map fst f3)

let requests =
  group "request"
    [
      prop ~count:100 ~examples:[ large_load ]
        "a request and its answer cross unchanged, in order"
        (Gen.with_pp
           (Format.pp_print_list
              ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ")
              pp_case)
           (Gen.list ~size:(Gen.int_range 1 4) case_g))
        requests_law;
      test "an agent's refusal is the request's error, and the job goes on"
        refused_then_answered;
      test "an answer that does not decode fails the job" undecodable_answer;
      test "drops and requests leave in the order they were queued"
        drops_in_order;
    ]

(* The agent's end *)

type side = Region of int * int | Local
type part = Words of string | Copy of side * side * int

let encode_side = function
  | Region (id, offset) -> "\000" ^ u64 id ^ u64 offset
  | Local -> "\001"

let encode_part = function
  | Words w -> "\000" ^ u32 (String.length w / 4) ^ w
  | Copy (src, dst, n) -> "\001" ^ u64 n ^ encode_side src ^ encode_side dst

(* A hand-over and the bytes of its copies from Local, as wire.mli lays them
   out. *)
let encode_handover (device, value, waits, parts, local) =
  u64 device ^ u64 value
  ^ u32 (List.length waits)
  ^ String.concat "" (List.map (fun (d, v) -> u64 d ^ u64 v) waits)
  ^ u32 (List.length parts)
  ^ String.concat "" (List.map encode_part parts)
  ^ String.concat "" local

let expected_handover (device, value, waits, parts, local) =
  let side = function
    | Region (id, offset) -> Wire.Region { id; offset }
    | Local -> Wire.Local
  in
  let part = function
    | Words w -> Wire.Words w
    | Copy (src, dst, bytes) ->
        Wire.Copy { src = side src; dst = side dst; bytes }
  in
  ( {
      Wire.device;
      value;
      waits = Array.of_list waits;
      parts = Array.of_list (List.map part parts);
    },
    local )

let handover_g =
  let open Gen in
  let region = map (fun (i, o) -> Region (i, o)) (pair id id) in
  let side = frequency [ (2, region); (1, constant Local) ] in
  let copy =
    let* src, dst, n = triple side side (int_range 1 64) in
    let dst = if src = Local && dst = Local then Region (0, 0) else dst in
    let+ bytes = string_of ~size:(constant n) char in
    (Copy (src, dst, n), if src = Local then Some bytes else None)
  in
  let words =
    map
      (fun w -> (Words w, None))
      (bind (int_range 0 4) (fun n -> string_of ~size:(constant (4 * n)) char))
  in
  let+ device = id
  and+ value = id
  and+ waits = small_list (pair id id)
  and+ parts =
    list ~size:(int_range 0 4) (frequency [ (1, words); (2, copy) ])
  in
  (device, value, waits, List.map fst parts, List.filter_map snd parts)

let pp_handover ppf (d, v, w, p, l) =
  Format.fprintf ppf "device %d, value %d, %d waits, %d parts, %d local copies"
    d v (List.length w) (List.length p) (List.length l)

let pp_side ppf = function
  | Wire.Region { id; offset } -> Format.fprintf ppf "%d+%d" id offset
  | Wire.Local -> Format.pp_print_string ppf "local"

let pp_part ppf = function
  | Wire.Words w -> Format.fprintf ppf "words %a" pp_bytes w
  | Wire.Copy { src; dst; bytes } ->
      Format.fprintf ppf "copy %d %a->%a" bytes pp_side src pp_side dst

let pp_command ppf = function
  | Wire.Handover (h, areas) ->
      let semi ppf () = Format.fprintf ppf "; " in
      Format.fprintf ppf
        "Handover (device %d, value %d, waits [%a], parts [%a], [%a])" h.device
        h.value
        (Format.pp_print_list ~pp_sep:semi (fun ppf (d, v) ->
             Format.fprintf ppf "%d@%d" d v))
        (Array.to_list h.waits)
        (Format.pp_print_list ~pp_sep:semi pp_part)
        (Array.to_list h.parts)
        (Format.pp_print_list ~pp_sep:semi pp_bytes)
        (List.map of_area (Array.to_list areas))
  | Wire.Drop id -> Format.fprintf ppf "Drop %d" id
  | Wire.Close -> Format.pp_print_string ppf "Close"
  | Wire.Request r -> Format.fprintf ppf "Request %a" pp_request r

(* Areas compare by their bytes. *)
let comparable = function
  | Wire.Handover (h, areas) ->
      `Handover (h, List.map of_area (Array.to_list areas))
  | Wire.Drop id -> `Drop id
  | Wire.Close -> `Close
  | Wire.Request _ -> `Request

let command_w =
  Testable.make ~pp:pp_command ~equal:(fun a b -> comparable a = comparable b)

let area s =
  Bigarray.Array1.init Bigarray.char Bigarray.c_layout (String.length s)
    (String.get s)

let handovers_law hs =
  with_raw ~peer:Wire.Controller @@ fun j l p ->
  List.iter (fun h -> write p (frame k_handover (encode_handover h))) hs;
  List.iter
    (fun h ->
      let e, local = expected_handover h in
      cover "a copy from Local" (local <> []);
      cover "words"
        (Array.exists (function Wire.Words _ -> true | _ -> false) e.parts);
      cover "waits" (e.waits <> [||]);
      equal command_w
        (Wire.Handover (e, Array.of_list (List.map area local)))
        (require_ok (Link.next l)))
    hs;
  equal state Link.Open (Link.wait j ~ms:0)

let commands_in_order () =
  with_raw ~peer:Wire.Controller @@ fun j l p ->
  write p
    (frame k_drop (u64 3) ^ frame k_drop (u64 0) ^ frame k_drop (u64 max_int));
  let next () = within j ~what:"next" (fun () -> Link.next l) in
  equal
    (list (result command_w string))
    [ Ok (Wire.Drop 3); Ok (Wire.Drop 0); Ok (Wire.Drop max_int) ]
    (List.init 3 (fun _ -> next ()))

let close_command () =
  with_raw ~peer:Wire.Controller @@ fun j l p ->
  write p (frame k_drop (u64 3) ^ frame k_close "");
  let next () = within j ~what:"next" (fun () -> Link.next l) in
  equal
    (list (result command_w string))
    [ Ok (Wire.Drop 3); Ok Wire.Close ]
    (List.init 2 (fun _ -> next ()))

let answer_misuse () =
  with_pair @@ fun _ c a ->
  let r1 = spawn (fun () -> Link.request c (Wire.Open "one")) in
  let first = require_ok (Link.next a) in
  let r2 =
    spawn (fun () -> Link.request c (Wire.Entry { image = 1; name = "f" }))
  in
  let second = require_ok (Link.next a) in
  (match (first, second) with
  | Wire.Request (Wire.Open _ as o), Wire.Request (Wire.Entry _ as e) ->
      raises_match ~msg:"not the oldest" Exn.invalid_arg (fun () ->
          Link.answer a e (Ok None));
      raises_match ~msg:"a copy of the oldest" Exn.invalid_arg (fun () ->
          Link.answer a (Wire.Open "one") (Ok []));
      raises_match ~msg:"never given" Exn.invalid_arg (fun () ->
          Link.answer a (Wire.Entry { image = 1; name = "f" }) (Ok None));
      Link.answer a o (Ok []);
      raises_match ~msg:"answered" Exn.invalid_arg (fun () ->
          Link.answer a o (Ok []));
      Link.answer a e (Ok (Some 4))
  | _ -> fail "the agent read other commands");
  equal (result (list account) refused) (Ok []) (r1 ());
  equal (result (option int) refused) (Ok (Some 4)) (r2 ())

(* bytes reads its area in place: it returns only once its peer took the bytes,
   and the frame carries the area. The area outgrows the sockets' buffers. *)
let bytes_in_place () =
  with_raw ~peer:Wire.Controller @@ fun _ l p ->
  let n = (1 lsl 24) - 16 in
  let a =
    Bigarray.Array1.init Bigarray.char Bigarray.c_layout n (fun i ->
        Char.chr (i land 255))
  in
  let returned = Atomic.make false in
  let finished =
    spawn (fun () ->
        Link.bytes l ~device:3 ~value:7 a;
        Atomic.set returned true)
  in
  Thread.delay 0.2;
  let early = Atomic.get returned in
  let f = read_frame p in
  finished ();
  equal ~msg:"returned while its peer read nothing" bool false early;
  equal (option frame_w) (Some (k_bytes, u64 3 ^ u64 7 ^ of_area a)) f

let agents =
  group "agent"
    [
      prop ~count:50
        "a hand-over laid out as wire.mli says reads back as itself"
        (Gen.with_pp
           (Format.pp_print_list
              ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ")
              pp_handover)
           (Gen.list ~size:(Gen.int_range 1 4) handover_g))
        handovers_law;
      test "next gives the controller's commands in the order they came"
        commands_in_order;
      test "next gives the controller's close after the frames before it"
        close_command;
      test "answer raises unless given the oldest request next gave"
        answer_misuse;
      test "bytes returns once its peer took the area's bytes (sampled)"
        bytes_in_place;
    ]

(* Rails *)

external get64 : Rig_remote_abi.area -> int -> int64 = "%caml_bigstring_get64"

(* A full barrier: reads after it see what the stores the count published. *)
let fence = Atomic.make 0
let ready = 0
let sent = 128
let arrived = 256

let count (e : Rig_remote_abi.end_) at =
  let v = Int64.to_int (get64 e.counts at) in
  Atomic.incr fence;
  v

let round256 n = (n + 255) / 256 * 256

let copy_size (ts : Rig_remote_abi.transfer array) at =
  Array.fold_left (fun m t -> max m (round256 (at t))) 0 ts

type plan = {
  send : Rig_remote_abi.transfer array;
  receive : Rig_remote_abi.transfer array;
  runs : int;
  batched : bool; (* the sender stores ready once per run *)
}

(* Transfers whose sources and destinations do not overlap, at offsets with gaps
   drawn between them. *)
let transfers =
  let open Gen in
  let length =
    frequency
      [
        (3, int_range 1 300);
        (1, of_list ~pp:Format.pp_print_int [ 1; 255; 256; 257 ]);
      ]
  in
  let+ l =
    list ~size:(int_range 0 4) (triple length (int_range 0 40) (int_range 0 40))
  in
  let _, _, ts =
    List.fold_left
      (fun (s, d, acc) (length, gs, gd) ->
        let src = s + gs and dst = d + gd in
        (src + length, dst + length, { Rig_remote_abi.src; dst; length } :: acc))
      (0, 0, []) l
  in
  Array.of_list (List.rev ts)

let plan_g =
  let open Gen in
  let+ send, receive =
    such_that
      (fun (s, r) -> Array.length s + Array.length r > 0)
      (pair transfers transfers)
  and+ runs = int_range 1 4
  and+ batched = bool in
  { send; receive; runs; batched }

let pp_plan ppf p =
  Format.fprintf ppf "send %a; receive %a; %d runs%s" pp_transfers p.send
    pp_transfers p.receive p.runs
    (if p.batched then ", batched" else "")

(* The byte [i] of transfer [j] in run [r], distinct between runs. *)
let byte ~dir r j i =
  Char.chr (((dir * 101) + (r * 37) + (j * 11) + i) land 0xff)

(* Runs [p.runs] runs of [ts] from [s] to [d]. *)
let carry ~dir p (ts : Rig_remote_abi.transfer array) (s : Rig_remote_abi.end_)
    (d : Rig_remote_abi.end_) =
  let n = Array.length ts in
  if n > 0 then begin
    let out = Bigarray.Array1.dim s.outbound / 2 in
    let inb = Bigarray.Array1.dim d.inbound / 2 in
    let seen = ref 0 in
    for r = 0 to p.runs - 1 do
      let k = r mod 2 in
      let last = (r + 1) * n in
      Array.iteri
        (fun j (t : Rig_remote_abi.transfer) ->
          for i = 0 to t.length - 1 do
            s.outbound.{(k * out) + t.src + i} <- byte ~dir r j i
          done;
          if not p.batched then s.ready ((r * n) + j + 1))
        ts;
      if p.batched then s.ready last;
      until ~what:"arrived" (fun () ->
          let a = count d arrived in
          if a < !seen then failf "arrived went from %d to %d" !seen a;
          seen := a;
          a >= last);
      equal ~msg:"arrived" int last (count d arrived);
      Array.iteri
        (fun j (t : Rig_remote_abi.transfer) ->
          equal
            ~msg:(Printf.sprintf "run %d, transfer %d" r j)
            string
            (String.init t.length (byte ~dir r j))
            (String.init t.length (fun i -> d.inbound.{(k * inb) + t.dst + i})))
        ts;
      until ~what:"sent" (fun () -> count s sent >= last);
      equal ~msg:"sent" int last (count s sent);
      equal ~msg:"ready" int last (count s ready)
    done;
    (* No byte of the inbound copies outside the transfers' destinations. *)
    let covered i =
      Array.exists
        (fun (t : Rig_remote_abi.transfer) ->
          let i = i mod inb in
          i >= t.dst && i < t.dst + t.length)
        ts
    in
    for i = 0 to (2 * inb) - 1 do
      if (not (covered i)) && d.inbound.{i} <> '\000' then
        failf "inbound byte %d is %C, outside every transfer" i d.inbound.{i}
    done
  end

let check_end ~msg (e : Rig_remote_abi.end_) ~send ~receive =
  equal ~msg:(msg ^ " outbound") int
    (2 * copy_size send (fun t -> t.src + t.length))
    (Bigarray.Array1.dim e.outbound);
  equal ~msg:(msg ^ " inbound") int
    (2 * copy_size receive (fun t -> t.dst + t.length))
    (Bigarray.Array1.dim e.inbound);
  at_least ~msg:(msg ^ " counts") int ~than:264 (Bigarray.Array1.dim e.counts);
  List.iter
    (fun (name, a) ->
      if String.exists (( <> ) '\000') (of_area a) then
        failf "%s %s is not zeroed" msg name)
    [ ("outbound", e.outbound); ("inbound", e.inbound); ("counts", e.counts) ]

let rail_law p =
  cover "a direction with no transfer" (p.send = [||] || p.receive = [||]);
  cover "both directions" (p.send <> [||] && p.receive <> [||]);
  cover "copies used again" (p.runs >= 3);
  cover "ready stored once per run" p.batched;
  cover "ready stored per transfer" (not p.batched);
  with_pair @@ fun j c a ->
  let ec = Link.rail c ~id:1 ~send:p.send ~receive:p.receive in
  let ea = Link.rail a ~id:1 ~send:p.receive ~receive:p.send in
  check_end ~msg:"controller" ec ~send:p.send ~receive:p.receive;
  check_end ~msg:"agent" ea ~send:p.receive ~receive:p.send;
  let there = spawn (fun () -> carry ~dir:0 p p.send ec ea) in
  let back = spawn (fun () -> carry ~dir:1 p p.receive ea ec) in
  there ();
  back ();
  Link.fail j "the test ends";
  List.iter
    (fun (who, e) ->
      List.iter
        (fun (name, at) ->
          equal
            ~msg:(who ^ " " ^ name)
            int64 Int64.max_int
            (get64 e.Rig_remote_abi.counts at))
        [ ("ready", ready); ("sent", sent); ("arrived", arrived) ])
    [ ("controller", ec); ("agent", ea) ]

let t5 = { Rig_remote_abi.src = 0; dst = 0; length = 5 }

let rail_misuse () =
  with_pair @@ fun _ c _ ->
  ignore (Link.rail c ~id:1 ~send:[| t5 |] ~receive:[||]);
  let bad msg send receive =
    raises_match ~msg Exn.invalid_arg (fun () ->
        Link.rail c ~id:2 ~send ~receive)
  in
  raises_match ~msg:"rail 1 again" Exn.invalid_arg (fun () ->
      Link.rail c ~id:1 ~send:[| t5 |] ~receive:[||]);
  bad "no transfer" [||] [||];
  bad "length 0" [| { t5 with length = 0 } |] [||];
  bad "length -1" [||] [| { t5 with length = -1 } |];
  bad "src -1" [| { t5 with src = -1 } |] [||];
  bad "dst -1" [||] [| { t5 with dst = -1 } |]

let rail_from_raw () =
  with_raw @@ fun j l p ->
  let e = Link.rail l ~id:4 ~send:[||] ~receive:[| { t5 with dst = 3 } |] in
  write p (frame k_rail (u64 4 ^ u64 1 ^ "hello"));
  until ~what:"arrived" (fun () -> count e arrived >= 1);
  equal string "hello" (String.init 5 (fun i -> e.inbound.{3 + i}));
  equal state Link.Open (Link.wait j ~ms:0)

let rail_wrong_length () =
  with_raw @@ fun j l p ->
  ignore (Link.rail l ~id:4 ~send:[||] ~receive:[| t5 |]);
  write p (frame k_rail (u64 4 ^ u64 1 ^ "abc"));
  match Link.wait j ~ms:2000 with
  | Link.Failed _ -> ()
  | s -> failf "the job is %a" (Testable.pp state) s

let rail_to_raw () =
  with_raw @@ fun _ l p ->
  let e = Link.rail l ~id:4 ~send:[| { t5 with src = 2 } |] ~receive:[||] in
  String.iteri (fun i ch -> e.outbound.{2 + i} <- ch) "hello";
  e.ready 1;
  equal (option frame_w) (Some (k_rail, u64 4 ^ u64 1 ^ "hello")) (next_frame p);
  until ~what:"sent" (fun () -> count e sent >= 1)

let released () =
  with_pair @@ fun j c a ->
  Link.release_rail c 9;
  let s = Link.rail c ~id:1 ~send:[| t5 |] ~receive:[||] in
  ignore (Link.rail a ~id:1 ~send:[||] ~receive:[| t5 |]);
  Link.release_rail a 1;
  Link.release_rail a 1;
  equal state Link.Open (Link.wait j ~ms:0);
  s.ready 1;
  match Link.wait j ~ms:2000 with
  | Link.Failed _ -> ()
  | st -> failf "the job is %a" (Testable.pp state) st

(* The link and its rail's end are dropped and collected while the sending
   thread has 4 MiB to send from the end's memory. *)
let unreachable_end () =
  with_job @@ fun j ->
  let frames = 64 and n = 1 lsl 16 in
  let p =
    let d, p = connected () in
    let l = Link.make j d ~name:"peer" ~peer:(Wire.Agent 1) in
    let t = { Rig_remote_abi.src = 0; dst = 0; length = n } in
    let e = Link.rail l ~id:1 ~send:[| t |] ~receive:[||] in
    Bigarray.Array1.fill e.outbound 'r';
    e.ready frames;
    p
  in
  Gc.full_major ();
  Gc.full_major ();
  Fun.protect ~finally:(fun () -> Unix.close p) @@ fun () ->
  let others = ref 0 in
  for c = 1 to frames do
    match next_frame p with
    | Some (k, f) when k = k_rail && String.length f = 16 + n ->
        equal ~msg:"its count" string (u64 c) (String.sub f 8 8);
        String.iteri (fun i ch -> if i >= 16 && ch <> 'r' then incr others) f
    | f -> failf "frame %d is %a" c (Testable.pp (option frame_w)) f
  done;
  equal ~msg:"bytes other than the end's" int 0 !others

let rails =
  group "rail"
    [
      prop ~count:60
        "transfers arrive whole, in order, in their copy, run after run"
        (Gen.with_pp pp_plan plan_g)
        rail_law;
      test "rail raises on its id again, no transfer, or a bad transfer"
        rail_misuse;
      test "a rail frame as wire.mli lays it out lands at its destination"
        rail_from_raw;
      test "a rail frame of another length fails the job" rail_wrong_length;
      test "a ready count sends its transfer as wire.mli lays it out"
        rail_to_raw;
      test "a transfer to a released rail fails the job, release twice is one"
        released;
      test "a rail's end lives until its release, reachable or not"
        unreachable_end;
    ]

(* Silence

   A stopped process sends nothing, not even beats. The idle process's link is
   made first, so without beats it would be the first silent. *)

let stopped_peer () =
  with_job @@ fun j ->
  let idle, _ = start j "idle" ~name:"idle" in
  guard idle @@ fun () ->
  Thread.delay 2.;
  let stopped, l = start j "serve" ~name:"stopped" in
  guard stopped @@ fun () ->
  equal
    (result (list account) string)
    (echo_open "MEM")
    (reason (Link.request l (Wire.Open "MEM")));
  Unix.kill stopped.pid Sys.sigstop;
  let s, took = timed (fun () -> Link.wait j ~ms:15_000) in
  kill stopped;
  equal state (Link.Failed "stopped: silent for 10 s") s;
  at_least float_exact ~than:(silence -. 1.) took;
  less float_exact ~than:(silence +. 1.5) took;
  equal (list string) [ "failed: stopped: silent for 10 s" ] (finish idle)

(* A peer that beats but reads nothing: a rail whose ready count runs far ahead
   fills the socket, and once a send has made no progress for 10 s the job fails
   naming the peer. *)
let deaf_peer () =
  with_raw @@ fun j l p ->
  let t = { Rig_remote_abi.src = 0; dst = 0; length = 1 lsl 16 } in
  let e = Link.rail l ~id:1 ~send:[| t |] ~receive:[||] in
  let beating = Atomic.make true in
  let beats =
    Thread.create
      (fun () ->
        while Atomic.get beating do
          (try write p (frame k_beat "") with Unix.Unix_error _ -> ());
          Thread.delay 0.5
        done)
      ()
  in
  e.ready 1_000_000_000;
  let s, took = timed (fun () -> Link.wait j ~ms:15_000) in
  Atomic.set beating false;
  Thread.join beats;
  equal state (Link.Failed "peer: read nothing for 10 s") s;
  at_least float_exact ~than:(silence -. 1.) took

let silences =
  group "silence"
    [
      slow "a stopped peer fails the job after 10 s, and every peer learns it"
        stopped_peer;
      slow "a peer that reads nothing for 10 s fails the job, named" deaf_peer;
    ]

(* Fork *)

let fork_reason = "a child of fork does not use its parent's connections"

(* The child requests and drops through its parent's link; its exit status is
   whether every answer was the fork's refusal. *)
let child j l =
  let ok =
    Link.request l (Wire.Open "child") = Error (`Failed fork_reason)
    && Link.failure j = Some fork_reason
    &&
    (Link.drop l 7;
     Link.next l = Error fork_reason)
  in
  Unix._exit (if ok then 0 else 1)

let forked () =
  with_job @@ fun j ->
  let h, l = start j "serve" ~name:"agent" in
  guard h @@ fun () ->
  equal
    (result (list account) string)
    (echo_open "before")
    (reason (Link.request l (Wire.Open "before")));
  (match Unix.fork () with
  | 0 -> child j l
  | pid -> (
      match Unix.waitpid [] pid with
      | _, Unix.WEXITED 0 -> ()
      | _ -> fail "the child's link did not refuse it"));
  equal (option string) None (Link.failure j);
  equal
    (result (list account) string)
    (echo_open "after")
    (reason (Link.request l (Wire.Open "after")));
  Link.fail j "the test ends";
  equal ~msg:"what the agent received" (list string)
    [ "failed: the test ends" ]
    (finish h)

let forks =
  group "fork"
    [
      test "a child's job is failed in the child alone; the parent's runs on"
        forked;
    ]

(* Two domains

   Each program has its own job, whose agent's end answers on a thread of this
   process as link_peer does, so a request that took another's answer, or a
   frame two domains interleaved, shows. *)

let rec echo a =
  match Link.next a with
  | Ok (Wire.Request (Wire.Open k as r)) ->
      Link.answer a r (echo_open k);
      echo a
  | Ok (Wire.Request (Wire.Entry { image; name } as r)) ->
      Link.answer a r (echo_entry image name);
      echo a
  | Ok (Wire.Request (Wire.Alloc { bytes; _ } as r)) ->
      Link.answer a r (echo_alloc bytes);
      echo a
  | Ok (Wire.Request r) ->
      Link.answer a r (Error "unexpected");
      echo a
  | Ok _ -> echo a
  | Error _ -> ()

type served = { job : Link.job; c : Link.t; served : unit -> unit }

(* The program's job: a program that connects twice shares one. *)
let current = ref None

let connect () =
  match !current with
  | Some s -> s
  | None ->
      let job = Link.job () in
      let d, a = connected () in
      let c = Link.make job d ~name:"agent" ~peer:(Wire.Agent 1) in
      let a = Link.make job a ~name:"controller" ~peer:Wire.Controller in
      let s = { job; c; served = spawn (fun () -> echo a) } in
      current := Some s;
      s

let release s =
  ended s.job;
  s.served ();
  current := None

let link = abstract "l" ~release
let names = result (list string) string

(* A failed job answers every request with its root cause, which no echo answer
   is. *)
let parallel_commands =
  let kind =
    Gen.with_pp pp_bytes
      (Gen.string_of ~size:(Gen.int_range 0 8) (Gen.char_range 'a' 'z'))
  in
  [
    command "connect" (Gen.unit @-> makes link) (fun () -> ()) connect;
    command "open"
      (kind @-> link ^-> returns names)
      (fun k () ->
        Result.map (List.map (fun (a : Wire.account) -> a.name)) (echo_open k))
      (fun k s ->
        Result.map
          (List.map (fun (a : Wire.account) -> a.name))
          (reason (Link.request s.c (Wire.Open k))));
    command "entry"
      (Gen.nat @-> kind @-> link ^-> returns (result (option int) string))
      (fun image name () -> echo_entry image name)
      (fun image name s ->
        reason (Link.request s.c (Wire.Entry { image; name })));
    command "alloc"
      (Gen.nat @-> link ^-> returns (result Windtrap.bool string))
      (fun bytes () -> echo_alloc bytes)
      (fun bytes s ->
        reason
          (Link.request s.c
             (Wire.Alloc { id = 1; device = 0; memory = `Device; bytes })));
  ]

(* Ids taken at once on two domains, and one taken before. *)
let fresh_ids () =
  let take () = List.init 10_000 (fun _ -> Link.fresh ()) in
  let before = Link.fresh () in
  let there = Domain.spawn take in
  let here = take () in
  let all = (before :: here) @ Domain.join there in
  let dups = List.length all - List.length (List.sort_uniq Int.compare all) in
  equal ~msg:"ids given twice" int 0 dups

let domains =
  group "domains"
    [
      test "fresh never gives an id twice, from two domains at once" fresh_ids;
      stateful ~count:20 ~domains:2
        "requests from two domains each get their own answer" parallel_commands;
    ]

(* The fork test runs before the domains are spawned: after them, fork
   raises. *)
let () =
  Watchdog.start ();
  (* A write to a peer that reset its connection fails with EPIPE. *)
  Sys.set_signal Sys.sigpipe Sys.Signal_ignore;
  exit
    (run "rig_remote_proxy.link"
       [
         group ~timeout:60. "link"
           [
             jobs;
             causes;
             failures;
             closes;
             requests;
             agents;
             rails;
             silences;
             forks;
             domains;
           ];
       ])
