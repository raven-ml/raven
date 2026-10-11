(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A signal while connect waits for a connection: a process of its own, whose
   only thread that takes the signal is the one in connect. *)

open Windtrap

(* The descriptors this process holds. *)
let descriptors () = Array.length (Sys.readdir "/dev/fd")

(* The sockets among them. *)
let sockets () =
  let count n f =
    match Unix.stat (Filename.concat "/dev/fd" f) with
    | { st_kind = S_SOCK; _ } -> n + 1
    | _ | (exception Unix.Unix_error _) -> n
  in
  Array.fold_left count 0 (Sys.readdir "/dev/fd")

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

(* The signal comes once connect's socket exists, and its handler, which runs
   on the thread in connect, closes the listener, so the connection is refused
   after the signal. connect answers the refusal and holds no descriptor after.
   The thread in connect holds the runtime except while it waits in the system:
   a yield once its socket exists hands it the runtime if it wants it, so the
   signal finds it waiting. *)
let interrupted () =
  let before = descriptors () in
  let l, port, held = full () in
  Sys.set_signal Sys.sigusr1
    (Sys.Signal_handle (fun _ -> List.iter Unix.close (l :: held)));
  let listening = sockets () in
  let interrupter =
    Thread.create
      (fun () ->
        ignore (Thread.sigmask Unix.SIG_BLOCK [ Sys.sigusr1 ]);
        while sockets () = listening do
          Thread.yield ()
        done;
        Thread.yield ();
        Unix.kill (Unix.getpid ()) Sys.sigusr1)
      ()
  in
  let r =
    Rig_remote.connect
      ~key:(Remote_job.as_key Remote_job.key)
      [ ("127.0.0.1", port) ]
  in
  Thread.join interrupter;
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
