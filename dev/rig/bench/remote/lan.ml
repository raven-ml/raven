(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A job between two machines, timed: what bench_remote.exe's copy rows time
   over loopback, over a real network. It lives outside the bench suite because
   a suite runs on one machine, and this needs two. lan.sh runs it.

   On one machine, rig agent serves the job at PORT and [lan.exe floors PORT +
   1] echoes the floors' bytes. On the other, [lan.exe controller HOST PORT]
   connects to the agent at HOST and PORT and times an allocation of 4 KiB of
   the agent's host, a round trip, and copies of 4 KiB and 1 MiB between this
   process's memory and the agent's host's, each way. Each beside its floor on
   the echo's connections: the same bytes there and back, the allocation's frame
   and its answer's, a copy's bytes and 8 back. The controller reads the job's
   key on standard input. The echo ends when the controller says it is done, or
   when no connection comes for [idle] seconds. *)

open Rig_remote_bench
module B = Rig.Buffer

let strf = Printf.sprintf
let kib = 1024
let mib = 1024 * kib

let ok what = function
  | Ok v -> v
  | Error why -> failwith (strf "%s: %s" what why)

let check what r = if r <> 0 then failwith (strf "%s: %d" what r)
let buffer n = Bigarray.(Array1.create char c_layout n)

(* Seconds an agent waits for a connection before it leaves. *)
let idle = 60.

(* What a copy's floor answers: 8 bytes. *)
let ack = 8

(* The echo *)

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
   (u64 each), then echoes until it ends; sizes of 0 end the echo. *)
let floors port =
  let l = listening port in
  let rec serve () =
    match accepted l with
    | None -> prerr_endline "lan.exe: no controller came"
    | Some fd ->
        let h = read_exactly fd 16 in
        let there = get_u64 h and back = get_u64 (Bytes.sub h 8 8) in
        if there > 0 then ignore (echo fd (buffer (max there back)) there back);
        Unix.close fd;
        if there > 0 then serve ()
  in
  serve ()

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

(* The sizes of copies, and the runs timed at each, about 3 s at 100 Mb/s. *)
let sizes = [ ("4K", 4 * kib, 500); ("1M", mib, 30) ]

(* [floored name bytes host port there back runs] reports the floor of [name],
   [bytes] its bytes of payload. *)
let floored name bytes host port there back runs =
  let f, round = floor host port there back in
  report (name ^ "-floor") bytes (timed runs round);
  Unix.close f

let controller host port key =
  let j = ok "connect" (Rig_remote.connect ~key [ (host, port) ]) in
  let h = List.hd (Rig_remote.hosts j) in
  report "alloc-4K" 0 (timed 1000 (fun () -> ignore (B.create h (4 * kib))));
  floored "alloc-4K" 0 host port request_bytes answer_bytes 1000;
  List.iter
    (fun (size, n, runs) ->
      let near = B.create Rig.host n and far = B.create h n in
      report ("copy-to-" ^ size) n
        (timed runs (fun () -> B.copy ~src:near ~dst:far));
      report ("copy-from-" ^ size) n
        (timed runs (fun () -> B.copy ~src:far ~dst:near));
      floored ("copy-" ^ size) n host port n ack runs)
    sizes;
  Rig_remote.close j;
  Unix.close (fst (floor host port 0 0))

let () =
  match Array.to_list Sys.argv |> List.tl with
  | [ "floors"; port ] -> floors (int_of_string port)
  | [ "controller"; host; port ] ->
      let key = In_channel.input_all stdin |> String.trim in
      controller host (int_of_string port) key
  | _ ->
      prerr_endline "usage: lan.exe floors PORT | lan.exe controller HOST PORT";
      exit 2
