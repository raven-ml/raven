(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A signal while connect waits for a connection: a process of its own, whose
   only thread that takes the signal is the one in connect. *)

open Windtrap

(* The descriptors this process holds. *)
let descriptors () = Array.length (Sys.readdir "/dev/fd")

(* [thread_self ()] is this thread's id, as the system knows it. *)
external thread_self : unit -> int = "rig_dial_probe_self"

(* [await_select t ms] waits until the thread [t] waits in select, for at most
   [ms] milliseconds: whether it does. *)
external await_select : int -> int -> bool = "rig_dial_probe_await_select"

(* A loopback listener whose backlog is full, with the connections that fill
   it: a connection to it waits. *)
let full () =
  let l = Unix.socket ~cloexec:true Unix.PF_INET Unix.SOCK_STREAM 0 in
  Unix.bind l (Unix.ADDR_INET (Unix.inet_addr_loopback, 0));
  Unix.listen l 0;
  let addr = Unix.getsockname l in
  let rec fill held =
    let c = Unix.socket ~cloexec:true Unix.PF_INET Unix.SOCK_STREAM 0 in
    Unix.set_nonblock c;
    (try Unix.connect c addr
     with Unix.Unix_error ((Unix.EINPROGRESS | Unix.EWOULDBLOCK), _, _) -> ());
    match Unix.select [] [ c ] [] 0.1 with
    | [], [], [] -> c :: held
    | _ -> fill (c :: held)
  in
  let held = fill [] in
  match addr with
  | Unix.ADDR_INET (_, port) -> (l, port, held)
  | Unix.ADDR_UNIX _ -> assert false

(* The signal comes once connect waits in select for its connection, which
   the full listener leaves waiting, and its handler, which runs on the thread
   in connect, closes the listener, so the connection is refused after the
   signal. connect answers the refusal and holds no descriptor after. *)
let interrupted () =
  let before = descriptors () in
  let l, port, held = full () in
  Sys.set_signal Sys.sigusr1
    (Sys.Signal_handle (fun _ -> List.iter Unix.close (l :: held)));
  let dialer = thread_self () in
  let waited = Atomic.make false in
  let interrupter =
    Thread.create
      (fun () ->
        ignore (Thread.sigmask Unix.SIG_BLOCK [ Sys.sigusr1 ]);
        if await_select dialer 5_000 then begin
          Atomic.set waited true;
          Unix.kill (Unix.getpid ()) Sys.sigusr1
        end)
      ()
  in
  let r =
    Rig_remote.connect
      ~key:(Remote_job.as_key Remote_job.key)
      [ ("127.0.0.1", port) ]
  in
  Thread.join interrupter;
  equal ~msg:"connect waited in select" bool true (Atomic.get waited);
  (match r with
  | Ok _ -> fail "connect reached a listener that accepts nothing"
  | Error why ->
      equal string
        (Printf.sprintf "127.0.0.1:%d: %s" port
           (Unix.error_message Unix.ECONNREFUSED))
        why);
  equal ~msg:"descriptors held" int before (descriptors ())

let () =
  Watchdog.start ();
  exit
    (run "rig_remote.dial"
       [
         group ~timeout:60. "rig_remote"
           [ test "a signal while connect dials leaves it to answer" interrupted ];
       ])
