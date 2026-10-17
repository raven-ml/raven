(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Interrupt walks of the proxies and their links. A call is made once for
   each allocation it makes, with another call from a Gc.Memprof callback at
   that allocation, on the calling thread. What the agents stated of their
   programs is read and changed under no lock, so a load or an unload from
   the callback answers, and through rig the machine's host is not lost. A
   link's next and answer hold a lock across their work, counted among the
   thread's locks: the callback's next or answer raises Invalid_argument,
   having changed nothing, exactly where the thread holds a lock. *)

open Windtrap
open Proxy_machine
module Link = Rig_remote_proxy.Link
module Proxy = Rig_remote_proxy

let strf = Printf.sprintf
let timeout = 120.

(* [walk attempt] runs [attempt n] from [n = 1] until an attempt no longer
   reaches its [n]-th allocation. *)
let walk attempt =
  let rec from n = if attempt n then from (n + 1) in
  from 1

(* The start of the refusal of the link's [fn] inside a lock. *)
let refused fn = "Rig_remote_proxy.Link." ^ fn ^ ": called from"

(* The id the agent loaded the program of the one function [name] as. *)
let program name =
  let named id fs found = if fs = [ name ] then Some id else found in
  match Hashtbl.fold named functions None with
  | Some id -> id
  | None -> failf "the agent loaded no program %s" name

(* What the proxies keep of the program [name], one entry per function. *)
let kept name = Array.length (Proxy.threads (program name))

let loaded p binary =
  match Proxy.image p binary with
  | Ok (Rig_edge.Loaded i) -> i
  | Ok (Place _) -> fail "the host loads its own code"
  | Error why -> fail why

(* Programs *)

(* An unload from a callback in a load: both answer, and the proxies keep
   the loaded program's limits and not the unloaded one's. *)
let unload_in_load () =
  walk (fun n ->
      let reached = ref false in
      with_machine (fun m ->
          let p = m.raw in
          let gone = strf "gone%d" n and first = strf "first%d" n in
          let i = loaded p gone in
          equal ~msg:"the limits of a loaded program" int 1 (kept gone);
          let unloaded = ref false in
          let call () =
            Proxy.unload p i;
            unloaded := true
          in
          reached :=
            Rig_support.interrupted ~landing:(Call call) n (fun () ->
                ignore (loaded p first));
          if not !unloaded then Proxy.unload p i;
          equal ~msg:(strf "from allocation %d, the unloaded program" n) int 0
            (kept gone);
          equal ~msg:(strf "from allocation %d, the loaded program" n) int 1
            (kept first));
      !reached)

(* A load from a callback in a load, through rig: both load, or the
   callback's raises Invalid_argument at a lock of rig's, and the machine's
   host is not lost. *)
let load_in_load_rig () =
  walk (fun n ->
      let reached = ref false in
      with_machine (fun m ->
          let first = strf "rig-first%d" n and second = strf "rig-second%d" n in
          let images = ref [] in
          let call () =
            match Rig.Image.load m.host second with
            | Ok i -> images := i :: !images
            | Error why -> failf "the load in the callback: %s" why
            | exception Invalid_argument _ -> ()
          in
          reached :=
            Rig_support.interrupted ~landing:(Call call) n (fun () ->
                images := require_ok (Rig.Image.load m.host first) :: !images);
          is_none ~msg:(strf "from allocation %d, the loss" n)
            (Rig.lost m.host);
          equal ~msg:(strf "from allocation %d, the first program" n) int 1
            (kept first);
          if List.length !images = 2 then
            equal ~msg:(strf "from allocation %d, the second program" n) int 1
              (kept second);
          ignore (Sys.opaque_identity !images));
      !reached)

(* Links *)

(* A job whose agent's end [a] has the controller's end [c] as its peer. *)
let with_pair f =
  with_job @@ fun j ->
  let d, a = connected () in
  let c = Link.make j d ~name:"agent" ~peer:(Wire.Agent 1) in
  let a = Link.make j a ~name:"controller" ~peer:Wire.Controller in
  f c a

let dropped = function
  | Ok (Wire.Drop id) -> id
  | Ok _ -> fail "a command other than a drop"
  | Error why -> fail why

(* A next from a callback in a next: inside the outer next's lock it raises
   Invalid_argument and takes nothing; outside it takes one of the two
   commands sent. No command is lost. *)
let next_in_next () =
  let seen = Rig_support.seen () in
  walk (fun n ->
      with_pair @@ fun c a ->
      Link.drop c 3;
      Link.drop c 4;
      let outer = ref None and inner = ref None in
      let landing, landed =
        Rig_support.landing seen (refused "next") (fun () ->
            inner := Some (dropped (Link.next a)))
      in
      let reached =
        Rig_support.interrupted ~landing n (fun () ->
            outer := Some (dropped (Link.next a)))
      in
      let what = strf "from allocation %d" n in
      ignore (Rig_support.answered what landed : bool option);
      let inner =
        match !inner with Some id -> id | None -> dropped (Link.next a)
      in
      equal ~msg:(what ^ ", the commands taken") (list int) [ 3; 4 ]
        (List.sort compare [ Option.get !outer; inner ]);
      reached);
  Rig_support.covered seen

let opened = function
  | Ok (Wire.Request (Wire.Open _ as r)) -> r
  | Ok _ -> fail "a command other than an open"
  | Error why -> fail why

(* An answer from a callback in an answer, of a request that is not the
   oldest: inside the outer answer's lock it raises Invalid_argument as it
   starts; outside it raises Invalid_argument since the request is not the
   oldest, which the callback expects. Both requests are answered once the
   outer answer returns. *)
let answer_in_answer () =
  let seen = Rig_support.seen () in
  let not_oldest =
    "Rig_remote_proxy.Link.answer: the request is not the oldest unanswered"
  in
  walk (fun n ->
      with_pair @@ fun c a ->
      let ask () = Link.request c (Wire.Open "MEM") in
      let _, first = spawn ask in
      let r1 = opened (Link.next a) in
      let _, second = spawn ask in
      let r2 = opened (Link.next a) in
      let landing, landed =
        Rig_support.landing seen (refused "answer") (fun () ->
            match Link.answer a r2 (Ok []) with
            | () -> fail "an answer to a request that is not the oldest"
            | exception Invalid_argument why when String.equal why not_oldest
              ->
                ())
      in
      let reached =
        Rig_support.interrupted ~landing n (fun () ->
            Link.answer a r1 (Ok []))
      in
      ignore
        (Rig_support.answered (strf "from allocation %d" n) landed
          : bool option);
      Link.answer a r2 (Ok []);
      let answered = function
        | Ok [] -> ()
        | Ok _ -> fail "accounts in the answer"
        | Error (`Refused why | `Failed why) -> fail why
      in
      answered (first ());
      answered (second ());
      reached);
  Rig_support.covered seen

(* A next, then an answer, each interrupted from its [n]-th allocation by an
   exception at every allocation from that one on: neither leaves a lock held.
   A later next takes a command, the one the interrupted next left or one sent
   after it, and the request the interrupted answer did not answer is
   answered by a later one. *)
let next_and_answer_raise () =
  let walked = ref 0 in
  walk (fun n ->
      with_pair @@ fun c a ->
      Link.drop c 5;
      let reached =
        Rig_support.interrupted n (fun () -> ignore (Link.next a))
      in
      equal ~msg:(strf "from allocation %d, locks held after a next" n) int 0
        (Rig_support.holds ());
      Link.drop c 6;
      (* 5 if the interrupted next did not take it. *)
      let later = dropped (Link.next a) in
      let later = if later = 5 then dropped (Link.next a) else later in
      equal ~msg:(strf "from allocation %d, a later next" n) int 6 later;
      let _, asked = spawn (fun () -> Link.request c (Wire.Open "MEM")) in
      let r = opened (Link.next a) in
      let answered = ref false in
      let reached' =
        Rig_support.interrupted n (fun () ->
            Link.answer a r (Ok []);
            answered := true)
      in
      equal ~msg:(strf "from allocation %d, locks held after an answer" n) int
        0 (Rig_support.holds ());
      if not !answered then Link.answer a r (Ok []);
      (match asked () with
      | Ok [] -> ()
      | Ok _ -> fail "accounts in the answer"
      | Error (`Refused why | `Failed why) -> fail why);
      if reached || reached' then incr walked;
      reached || reached');
  greater ~msg:"allocations interrupted" int ~than:0 !walked

let () =
  exit
    (run "rig_remote_proxy interrupt"
       [
         group ~timeout "a call from a callback at any allocation"
           [
             test "an unload in a load answers" unload_in_load;
             test
               "a next in a next raises Invalid_argument inside a lock and \
                takes a command outside"
               next_in_next;
             test
               "an answer in an answer raises Invalid_argument inside a lock \
                and outside"
               answer_in_answer;
           ];
         group ~timeout "an exception at any allocation"
           [
             test "a next and an answer leave no lock held"
               next_and_answer_raise;
           ];
         group ~timeout
           "a call to rig from a callback at any allocation keeps the host"
           [ test "a load in a load" load_in_load_rig ];
       ])
