(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A link between two machines, timed: what bench_remote.exe times over
   loopback, over a real network. It lives outside the bench suite because a
   suite runs on one machine, and this needs two. lan.sh runs it.

   On one machine, [lan.exe agent PORT] serves a job's link on PORT and the
   floors' connections on PORT + 1. On the other, [lan.exe controller HOST PORT]
   times against it a request's round trip ([Link.request] of an allocation),
   and rail runs of one transfer of 4 KiB or 1 MiB from the controller's end,
   each answered by a transfer of 8 bytes back once the agent's end has it. Each
   beside its floor: the same bytes there and back, on a connection of their
   own. Both processes read the job's key on standard input. The agent ends once
   its job fails, which the controller does when it is done, or when no
   controller comes for [idle] seconds. *)

open Rig_remote_bench
module Wire = Rig_remote_proxy.Wire
module Link = Rig_remote_proxy.Link
module B = Rig.Buffer

let strf = Printf.sprintf
let kib = 1024
let mib = 1024 * kib

let ok what = function
  | Ok v -> v
  | Error why -> failwith (strf "%s: %s" what why)

let reason = function `Refused why | `Failed why -> why
let check what r = if r <> 0 then failwith (strf "%s: %d" what r)
let buffer n = Bigarray.(Array1.create char c_layout n)

(* Seconds an agent waits for a connection before it leaves. *)
let idle = 60.

(* A rail's answer: one transfer of 8 bytes back. *)
let ack : Rig_remote_abi.transfer = { src = 0; dst = 0; length = 8 }

(* The agent *)

let listening port =
  let l = Unix.socket ~cloexec:true Unix.PF_INET Unix.SOCK_STREAM 0 in
  Unix.setsockopt l Unix.SO_REUSEADDR true;
  Unix.bind l (Unix.ADDR_INET (Unix.inet_addr_any, port));
  Unix.listen l 4;
  l

(* The next connection to [l], or [None] after [idle] seconds without one. *)
let accepted l =
  match Unix.select [ l ] [] [] idle with
  | [], _, _ -> None
  | _ ->
      let fd, _ = Unix.accept ~cloexec:true l in
      tune fd;
      Some fd

let get_u64 b =
  let v = ref 0 in
  for i = 7 downto 0 do
    v := (!v lsl 8) lor Char.code (Bytes.get b i)
  done;
  !v

let read_exactly fd n =
  let b = Bytes.create n in
  let rec go at =
    if at < n then begin
      let k = Unix.read fd b at (n - at) in
      if k = 0 then failwith "floor: the stream ended";
      go (at + k)
    end
  in
  go 0;
  b

(* Each floor connection starts with the sizes of its messages there and back
   (u64 each), then echoes until it ends. *)
let floors l =
  let rec serve () =
    match accepted l with
    | None -> ()
    | Some fd ->
        let h = read_exactly fd 16 in
        let there = get_u64 h and back = get_u64 (Bytes.sub h 8 8) in
        ignore (echo fd (buffer (max there back)) there back);
        Unix.close fd;
        serve ()
  in
  serve ()

let serve port key =
  let floor = listening (port + 1) and link = listening port in
  ignore (Thread.create floors floor);
  match accepted link with
  | None ->
      prerr_endline "lan.exe: no controller came";
      exit 1
  | Some fd ->
      let j = Link.job () in
      ignore (ok "accept" (Wire.accept fd ~key ~admit:(fun _ -> Ok ())));
      agent (Link.make j fd ~name:"controller" ~peer:Wire.Controller);
      let why = Option.value (Link.failure j) ~default:"closed" in
      prerr_endline ("lan.exe: the job ended: " ^ why);
      Unix._exit 0

(* The controller *)

(* A connection to [host]'s [port], tried every 100 ms for [idle] seconds: the
   agent may not listen yet. *)
let connect host port =
  let a = (Unix.gethostbyname host).Unix.h_addr_list.(0) in
  let until = Unix.gettimeofday () +. idle in
  let rec go () =
    let fd = Unix.socket ~cloexec:true Unix.PF_INET Unix.SOCK_STREAM 0 in
    match Unix.connect fd (Unix.ADDR_INET (a, port)) with
    | () ->
        tune fd;
        fd
    | exception Unix.Unix_error (Unix.ECONNREFUSED, _, _)
      when Unix.gettimeofday () < until ->
        Unix.close fd;
        Unix.sleepf 0.1;
        go ()
  in
  go ()

(* [timed n f] is the seconds of each of [n] calls of [f], after [n / 10 + 1]
   untimed ones, sorted. *)
let timed n f =
  for _ = 0 to n / 10 do
    f ()
  done;
  let t =
    Array.init n (fun _ ->
        let t0 = Unix.gettimeofday () in
        f ();
        Unix.gettimeofday () -. t0)
  in
  Array.sort Float.compare t;
  t

let us s = strf "%.1f us" (s *. 1e6)
let ms s = if s >= 1e-3 then strf "%.2f ms" (s *. 1e3) else us s

let report name bytes t =
  let n = Array.length t in
  let median = t.(n / 2) in
  let rate =
    if bytes = 0 then "" else strf "  %.1f MB/s" (float bytes /. median /. 1e6)
  in
  Printf.printf "%-16s n=%-4d median %-10s min %-10s p90 %-10s%s\n%!" name n
    (ms median)
    (ms t.(0))
    (ms t.(n * 9 / 10))
    rate

let put_u64 b at v =
  for i = 0 to 7 do
    Bytes.set b (at + i) (Char.chr ((v lsr (8 * i)) land 0xff))
  done

(* A floor connection whose messages are [there] bytes out and [back] in. *)
let floor host port there back =
  let fd = connect host (port + 1) in
  let h = Bytes.create 16 in
  put_u64 h 0 there;
  put_u64 h 8 back;
  ignore (Unix.write fd h 0 16);
  let b = buffer (max there back) in
  (fd, fun () -> check "ask" (ask fd b there back))

(* The sizes of rails' transfers and copies, and the runs timed at each, about 3
   s at 100 Mb/s. *)
let sizes = [ ("4K", 4 * kib, 500); ("1M", mib, 30) ]

let host_account : Wire.account =
  { id = 0; name = "CPU"; arch = "bench"; budget = 1 lsl 30; reaches = [] }

(* The proxy of the agent's machine's host, opened on rig. *)
let proxy l machine =
  let rail _ ~send:_ ~receive:_ = Error "lan.exe makes no rails here" in
  let p =
    Rig_remote_proxy.make l host_account (Rig_remote_abi.Host { machine; rail })
  in
  ok "open"
    (Rig.open_host
       (module Rig_remote_proxy)
       ~machine ~name:"CPU"
       (fun () -> Ok p))

let controller host port key =
  let fd = connect host port in
  ok "dial" (Wire.dial fd ~key ~self:Wire.Controller ~peer:(Wire.Agent 1));
  let j = Link.job () in
  let l = Link.make j fd ~name:host ~peer:(Wire.Agent 1) in
  report "request" 0
    (timed 1000 (fun () ->
         if not (ok "request" (Result.map_error reason (Link.request l alloc)))
         then failwith "request: refused"));
  let f, round = floor host port request_bytes answer_bytes in
  report "request-floor" 0 (timed 1000 round);
  Unix.close f;
  List.iteri
    (fun i (size, n, runs) ->
      let name = "rail-" ^ size in
      let id = i + 1 in
      let t : Rig_remote_abi.transfer = { src = 0; dst = 0; length = n } in
      let e = Link.rail l ~id ~send:[| t |] ~receive:[| ack |] in
      ok "rail"
        (Result.map_error reason
           (Link.request l
              (Wire.Rail
                 {
                   id;
                   peer = Wire.Controller;
                   send = [| ack |];
                   receive = [| t |];
                 })));
      let c = ref 0 in
      report name n
        (timed runs (fun () ->
             incr c;
             check "rail" (rail_run e.counts e.counts !c)));
      let f, round = floor host port n ack.length in
      report (name ^ "-floor") n (timed runs round);
      Unix.close f)
    sizes;
  let h = proxy l host in
  List.iter
    (fun (size, n, runs) ->
      let near = B.create Rig.host n and far = B.create h n in
      report ("copy-to-" ^ size) n
        (timed runs (fun () -> B.copy ~src:near ~dst:far));
      report ("copy-from-" ^ size) n
        (timed runs (fun () -> B.copy ~src:far ~dst:near)))
    sizes;
  Link.fail j "the measurement ended"

let () =
  let key = In_channel.input_all stdin |> String.trim in
  match Array.to_list Sys.argv |> List.tl with
  | [ "agent"; port ] -> serve (int_of_string port) key
  | [ "controller"; host; port ] -> controller host (int_of_string port) key
  | _ ->
      prerr_endline "usage: lan.exe agent PORT | lan.exe controller HOST PORT";
      exit 2
