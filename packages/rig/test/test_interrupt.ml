(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Interrupt walks. An operation runs once for each allocation it makes, in a
   forked child, with every allocation from that one on raising until the
   operation returns, as an asynchronous exception does, and handlers that
   allocate meet it again. It then runs so again from the first allocation at
   the same call stack. After that, rig is consistent: the device is not lost,
   no profile is left taken, its name answers the same device, and the device
   runs the operation again, closes, and its name opens a device that runs it.
   rig promises no return of memory after an asynchronous exception, so the
   walk checks none. *)

open Windtrap
open Ops
module B = Rig.Buffer
module P = Rig_support.Polled
module Sub = Rig.Submission
module S = Rig_support

(* The limit of each test, and of each child within 1/6 of it: a test run
   with a larger limit, as under a sanitizer, gives each child more time. *)
let timeout =
  match Option.map float_of_string_opt (Sys.getenv_opt "WINDTRAP_TIMEOUT") with
  | Some (Some t) -> t
  | Some None | None -> 60.

let child_limit = timeout /. 6.

(* What an interrupted allocation raises. *)
exception Injected

let outcome f =
  match f () with
  | () -> "done"
  | exception Injected -> "raised Injected"
  | exception e -> Printexc.to_string e

(* The first allocation that raises in an interrupted call: its [n]-th, or its
   first from the call stack [s] within the call, by code addresses. *)
type allocation = Nth of int | Site of Printexc.raw_backtrace_entry array

(* [interrupting a site f] is [f ()] with every allocation from [a] on raising
   [Injected] until [f] returns: a memory profile samples every allocation,
   and its callback raises. [site] is set to the call stack of the first
   allocation that raised. *)
let interrupting a site f =
  let left = ref (match a with Nth n -> n | Site _ -> 0)
  and armed = ref false
  and raising = ref false in
  let outer = Printexc.raw_backtrace_length (Printexc.get_callstack max_int) in
  let alloc (sample : Gc.Memprof.allocation) =
    let stack () =
      let s = Printexc.raw_backtrace_entries sample.callstack in
      Array.sub s 0 (Array.length s - outer)
    in
    if !armed && not !raising then begin
      let here =
        match a with
        | Nth _ ->
            decr left;
            !left = 0
        | Site s -> stack () = s
      in
      if here then begin
        site := Some (stack ());
        raising := true
      end
    end;
    if !armed && !raising then raise Injected;
    None
  in
  let tracker =
    { Gc.Memprof.null_tracker with alloc_minor = alloc; alloc_major = alloc }
  in
  let profile = Gc.Memprof.start ~sampling_rate:1. tracker in
  let stop () =
    armed := false;
    Gc.Memprof.stop ();
    Gc.Memprof.discard profile
  in
  armed := true;
  match f () with
  | v ->
      stop ();
      v
  | exception e ->
      stop ();
      raise e

(* [in_child what f] is [f ()], run in a forked child, which starts from this
   process's state: what one attempt leaves, such as memory an interrupt leaks,
   never changes the next one's allocations. It fails with the child's
   exception, and if the child has not answered within [child_limit] s: a
   lock that an exception left held makes the next call that takes it wait
   for good, in C, where no timeout interrupts it. *)
let in_child what f =
  let r, w = Unix.pipe () in
  match Unix.fork () with
  | 0 ->
      Unix.close r;
      let answer =
        match f () with
        | true -> "raised"
        | false -> "done"
        | exception e -> Printexc.to_string e
      in
      let oc = Unix.out_channel_of_descr w in
      output_string oc answer;
      close_out oc;
      Unix._exit 0
  | pid -> (
      Unix.close w;
      let until = Unix.gettimeofday () +. child_limit in
      let rec wait () =
        match Unix.waitpid [ WNOHANG ] pid with
        | 0, _ when Unix.gettimeofday () > until ->
            Unix.kill pid Sys.sigkill;
            ignore (Unix.waitpid [] pid);
            Unix.close r;
            failf "no return from rig%s after %.0f s" what child_limit
        | 0, _ ->
            Unix.sleepf 0.0005;
            wait ()
        | _, status -> status
      in
      let status = wait () in
      let ic = Unix.in_channel_of_descr r in
      let answer = In_channel.input_all ic in
      close_in ic;
      match (answer, status) with
      | "raised", _ -> true
      | "done", _ -> false
      | "", WEXITED n -> failf "the child exited %d%s" n what
      | "", (WSIGNALED n | WSTOPPED n) ->
          failf "the child got signal %d%s" n what
      | e, _ -> failf "%s%s" e what)

(* The device an interrupted operation runs on: one just opened, or one the
   operation ran on once, which made what the device keeps for good, and whose
   memory from that run the collector took, which the interrupted run drains. *)
type device = Fresh | Used

(* Whether [name] answers [d]: an open of it returns [d] without calling its
   opener. *)
let answers name d =
  let again () = Error "the opener ran again" in
  match Rig.open_ { name; driver = (module P); make = again } with
  | Ok d' -> Rig.equal d d'
  | Error _ -> false

(* The peer memory [borrow_kept] borrowed, kept until the borrowing devices
   have closed. *)
let held = ref []

(* The producers [waiting] opened, with their drivers. *)
let producers = ref []

(* Runs [op] interrupted from its [n]-th allocation, then from the same
   allocation again, and checks that rig is consistent after it. Is whether
   the allocation was reached. *)
let interrupt start op n =
  let at = Printf.sprintf ", interrupted from allocation %d" n in
  in_child at (fun () ->
      let name = fresh op.name in
      let d, p = op.opener name in
      if start = Used then op.prepare d None;
      Gc.full_major ();
      (* Twice, as a resource of two, such as a staging memory's slots, runs
         out only after two runs left one held. *)
      let site = ref None in
      let once () = op.prepare d None in
      let run a =
        let got = outcome (fun () -> interrupting a site once) in
        mem ~msg:"outcome" string got [ "done"; "raised Injected" ];
        got = "raised Injected"
      in
      let raised = run (Nth n) in
      Option.iter (fun s -> ignore (run (Site s))) !site;
      equal ~msg:"lost" (option string) None (Rig.lost d);
      equal ~msg:"profiles" bool false (Rig.Profile.enabled ());
      equal ~msg:"its name answers it" bool true (answers name d);
      op.prepare d None;
      Rig.close d;
      equal ~msg:"closed" (option string) (Some "closed") (Rig.lost d);
      equal ~msg:"stops" int 1 (count "stop" p);
      let d, _ = op.opener name in
      op.prepare d None;
      if !held <> [] then begin
        (* The borrowing devices close before the memory they mapped goes,
           over a few rounds of collections and drains. *)
        Rig.close d;
        for _ = 1 to 4 do
          settle d
        done;
        held := [];
        for _ = 1 to 4 do
          settle d
        done
      end;
      if !producers <> [] then begin
        (* The waiting device and the producers close, each stopped once. *)
        Rig.close d;
        List.iter (fun (p, _) -> Rig.close p) !producers;
        List.iter
          (fun (_, pp) ->
            equal ~msg:"a producer's stops" int 1 (count "stop" pp))
          !producers
      end;
      raised)

let interrupts start op () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  S.needs_fork ();
  let rec from n = if interrupt start op n then from (n + 1) in
  from 1

(* More operations *)

(* Copies by a device that maps no host memory, through its own staging
   memory, beside the stage of a device lost since, which the copy's lookup of
   its stage removes. *)
let unmapped op =
  let open_unmapped = P.open_ ~host_visible:false ~maps_host:false in
  let opener name =
    let other, _ = open_unmapped (name ^ ":lost") in
    B.copy ~src:(B.create Rig.host 8) ~dst:(B.create other 8);
    Rig.close other;
    open_unmapped name
  in
  { op with name = op.name ^ " through staging"; opener }

(* Borrows peer memory that goes back after the borrowing device, its mapper,
   closed: an exception may leave the mapping recorded in the memory alone. *)
let borrow_kept =
  let opener name =
    ignore (peer ());
    P.open_ name
  in
  op ~opener "borrow peer, freed after the mapper" (fun d _ ->
      let b = B.create (peer ()) 64 in
      held := b :: !held;
      match B.borrow d b with
      | Some b' -> equal int 64 (B.length b')
      | None -> fail "the borrow was refused")

(* Submits on a device whose queue waits on another device's word, a new
   producer each time: the wait maps the producer's word. *)
let waiting =
  op ~opener:(P.open_ ~waits_on:[ `Host ]) "submit waiting on a producer"
    (fun d _ ->
      let producer, pp = P.open_ (fresh "producer") in
      producers := (producer, pp) :: !producers;
      let v = once ~run:(Sub.Run.make ()) (Sub.make producer [||]) in
      let s = Sub.make d [||] in
      let run = Sub.Run.make () in
      Rig.Point.wait (Rig.submit s ~run ~buffers:[||] ~waits:[| v |]))

(* Interrupts a close of a machine's host, which holds a device of the
   machine, from its [n]-th allocation, then checks that the machine answers
   as the close left it: an open host opens devices and a closed one opens
   none. A second close ends the machine however the close was interrupted:
   rig has called the stop of each of its devices exactly once. Is whether the
   allocation was reached. *)
let interrupt_close n =
  let at = Printf.sprintf ", interrupted from allocation %d" n in
  in_child at (fun () ->
      let drivers = ref [] in
      let polled () =
        let p = P.make () in
        drivers := p :: !drivers;
        Ok p
      in
      let host =
        Rig.open_host ~machine:(fresh "close")
          { name = "CPU"; driver = (module P); make = polled }
        |> Result.get_ok
      in
      let gpu name =
        let o = { Rig_edge.name; driver = (module P); make = polled } in
        match Rig.open_ ~machine:host o with
        | Ok _ -> "opened"
        | Error e -> e
      in
      equal ~msg:"a device" string "opened" (gpu "gpu");
      Gc.full_major ();
      let got =
        let close () = Rig.close host in
        outcome (fun () -> interrupting (Nth n) (ref None) close)
      in
      mem ~msg:"outcome" string got [ "done"; "raised Injected" ];
      let closed = Printf.sprintf "%s lost: closed" (Rig.name host) in
      (match Rig.lost host with
      | None ->
          equal ~msg:"an open on the machine" string "opened" (gpu "after")
      | Some why ->
          equal ~msg:"the host's loss" string "closed" why;
          equal ~msg:"an open on the machine" string closed (gpu "after"));
      Rig.close host;
      equal ~msg:"an open once closed" string closed (gpu "last");
      equal ~msg:"the host once closed" (option string) (Some "closed")
        (Rig.lost host);
      List.iter
        (fun p -> equal ~msg:"a device's stops" int 1 (P.stops p))
        !drivers;
      got = "raised Injected")

let test_close () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  S.needs_fork ();
  let rec from n = if interrupt_close n then from (n + 1) in
  from 1

(* Opens [names] devices, as many as make the table of names grow, and checks
   that each name answers the device it opened before. *)
let names = 40
let opened = Array.make names Rig.host

let opens =
  op "open" (fun _ _ ->
      for k = 0 to names - 1 do
        let d, _ = P.open_ (Printf.sprintf "interrupt:open:%d" k) in
        if opened.(k) != Rig.host && not (Rig.equal opened.(k) d) then
          failf "interrupt:open:%d opened a second device" k;
        opened.(k) <- d
      done)

(* Drops buffers of [names] sizes and drains them into the device's cache, as
   many as make the cache grow. *)
let caches =
  let drop d =
    for k = 1 to names do
      ignore (B.create d (k * 64))
    done
  in
  op "cache" (fun d _ ->
      drop d;
      collect ();
      ignore (B.create d 0))

let () =
  exit
  @@ run "rig.interrupt"
       [
         group ~timeout "an exception at any allocation leaves rig consistent"
           (List.concat_map
              (fun op ->
                [
                  test op.name (interrupts Fresh op);
                  test (op.name ^ ", on a device it ran on")
                    (interrupts Used op);
                ])
              ((opens :: caches :: borrow_kept :: waiting :: ops)
              @ staged @ List.map unmapped staged)
           @ [ test "close of a machine's host" test_close ]);
       ]
