(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Interrupt walks. An operation runs once for each allocation it makes, in a
   forked child, with every allocation from that one on raising until the
   operation returns, as an asynchronous exception does, and handlers that
   allocate meet it again. It then runs so again from the first allocation at
   the same call stack. After that, rig is consistent: no profile is left
   taken, and the device is either lost, by an exception that escaped a call
   of its driver, with the exception's text and stopped once, or live, its
   name answering it, and it runs the operation again and closes. Either way
   its name then opens a device that runs the operation. rig promises no
   return of memory after an asynchronous exception, so the walk checks
   none. *)

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
let in_child what (f : unit -> 'a) : 'a =
  let r, w = Unix.pipe () in
  match Unix.fork () with
  | 0 ->
      Unix.close r;
      let answer : ('a, string) result =
        match f () with v -> Ok v | exception e -> Error (Printexc.to_string e)
      in
      let oc = Unix.out_channel_of_descr w in
      Marshal.to_channel oc answer [];
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
      | "", WEXITED n -> failf "the child exited %d%s" n what
      | "", (WSIGNALED n | WSTOPPED n) ->
          failf "the child got %s%s" (Sys.signal_to_string n) what
      | answer, _ -> (
          match (Marshal.from_string answer 0 : ('a, string) result) with
          | Ok v -> v
          | Error e -> failf "%s%s" e what))

(* The device an interrupted operation runs on: one just opened, or one the
   operation ran on once, which made what the device keeps for good, and whose
   memory from that run the collector took, which the interrupted run drains. *)
type device = Fresh | Used

(* Whether [name] answers [d]: an open of it returns [d] without calling its
   opener. *)
let answers name d =
  let again ~word:_ = Error "the opener ran again" in
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
      if Rig.lost d = None then
        Option.iter (fun s -> ignore (run (Site s))) !site;
      equal ~msg:"profiles" bool false (Rig.Profile.enabled ());
      (match Rig.lost d with
      | Some why ->
          equal ~msg:"the loss" string (Printexc.to_string Injected) why;
          (* The close runs the stop if the interrupt left it owed. *)
          Rig.close d;
          equal ~msg:"stops" int 1 (P.stops p)
      | None ->
          equal ~msg:"its name answers it" bool true (answers name d);
          op.prepare d None;
          Rig.close d;
          equal ~msg:"closed" (option string) (Some "closed") (Rig.lost d);
          equal ~msg:"stops" int 1 (count "stop" p));
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
          (fun (_, pp) -> equal ~msg:"a producer's stops" int 1 (P.stops pp))
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

(* The run every [kept_run] submits with. *)
let kept = Sub.Run.make ()

(* Submits a copy under a profile with [kept], which serves each call in turn:
   an exception in one submit leaves the run to the next. *)
let kept_run =
  op "profiled submit with a kept run" (fun d _ ->
      let s = Sub.make d [| Sub.copy ~src:(Slot 0) ~dst:(Slot 1) |] in
      let buffers = [| B.create d 8; B.create d 8 |] in
      let submit () = Rig.Point.wait (once ~buffers ~run:kept s) in
      ignore (Rig.Profile.take submit))

(* Interrupts a profiled submit from its [n]-th allocation, then checks that
   the device stays usable and that the next profiled submit, once the
   device ran its earlier work, takes no time pair before rig settled the
   value it timed last. Is whether the allocation was reached. *)
let interrupt_timed n =
  let at = Printf.sprintf ", interrupted from allocation %d" n in
  in_child at (fun () ->
      let d, _ = P.open_ (fresh "timed") in
      let s = Sub.make d [| Sub.copy ~src:(Slot 0) ~dst:(Slot 1) |] in
      let buffers = [| B.create d 8; B.create d 8 |] in
      let run = Sub.Run.make () in
      let wait () = Rig.Point.wait (once ~buffers ~run s) in
      let submit () = ignore (Rig.Profile.take wait) in
      submit ();
      let seen = List.length (P.violations ()) in
      Gc.full_major ();
      let got = outcome (fun () -> interrupting (Nth n) (ref None) submit) in
      mem ~msg:"outcome" string got [ "done"; "raised Injected" ];
      equal ~msg:"lost" (option string) None (Rig.lost d);
      Rig.Point.wait (once ~run:(Sub.Run.make ()) (Sub.make d [||]));
      submit ();
      equal ~msg:"breaches" (list string) [] (List.drop seen (P.violations ()));
      got = "raised Injected")

let test_timed () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  S.needs_fork ();
  let rec from n = if interrupt_timed n then from (n + 1) in
  from 1

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
      let polled ~word =
        let p = P.make () in
        drivers := p :: !drivers;
        Ok (P.bind p ~word)
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

(* Values kept by collected submissions on a device whose work is not done:
   several, so that an interrupt can come between the routing of two of a
   drain's records. *)
let kept_values = 4

let[@inline never] submit_kept d runs =
  for _ = 1 to kept_values do
    let v = ref () in
    Gc.finalise (fun _ -> Atomic.incr runs) v;
    ignore (once ~run:(Sub.Run.make ()) (Sub.make ~keep:v d [||]))
  done

(* Interrupts a drain that finds collected submissions keeping values, from
   its [n]-th allocation: however the drain was interrupted, no value is let
   go before the work of its submission is done. Is whether the allocation
   was reached. *)
let interrupt_kept n =
  let at = Printf.sprintf ", interrupted from allocation %d" n in
  in_child at (fun () ->
      let d, _ = P.open_ (fresh "kept") in
      let runs = Atomic.make 0 in
      submit_kept d runs;
      Gc.full_major ();
      (* An empty buffer calls no driver, so the device stays live. *)
      let drain () = ignore (Sys.opaque_identity (B.create d 0)) in
      let got = outcome (fun () -> interrupting (Nth n) (ref None) drain) in
      mem ~msg:"outcome" string got [ "done"; "raised Injected" ];
      drain ();
      Gc.full_major ();
      equal ~msg:"values let go before their work is done" int 0
        (Atomic.get runs);
      got = "raised Injected")

let test_kept () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  S.needs_fork ();
  let rec from n = if interrupt_kept n then from (n + 1) in
  from 1

(* Interrupts a drain that routes collected buffers into the cache, from its
   [n]-th allocation: the drain leaks no buffer's memory, so once the cache
   is freed the device holds none. Is whether the allocation was reached. *)
let interrupt_cached n =
  let at = Printf.sprintf ", interrupted from allocation %d" n in
  in_child at (fun () ->
      let d, _ = P.open_ (fresh "cached") in
      let[@inline never] drop () =
        for k = 1 to kept_values do
          ignore (Sys.opaque_identity (B.create d (k * 64)))
        done
      in
      drop ();
      Gc.full_major ();
      let drain () = ignore (Sys.opaque_identity (B.create d 0)) in
      let got = outcome (fun () -> interrupting (Nth n) (ref None) drain) in
      mem ~msg:"outcome" string got [ "done"; "raised Injected" ];
      Rig.free_cache d;
      equal ~msg:"bytes held" int 0 (Rig.allocated d);
      got = "raised Injected")

let test_cached () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  S.needs_fork ();
  let rec from n = if interrupt_cached n then from (n + 1) in
  from 1

(* Re-entry: a call that may take a lock, made from a [Gc.Memprof] callback
   inside a lock of its thread's, raises [Invalid_argument] naming it, having
   done nothing; outside every lock it runs as any call does. A call that
   takes no lock runs either way ({!Rig.section-domains}). *)

(* A lock a library counts through the edge refuses rig's calls on its
   thread until it is given back, and changes nothing meanwhile. *)
let test_counted () =
  let d = Rig.memory_device (fresh "counted") in
  let before = S.holds () in
  let held, refused, allocated =
    S.inside_lock @@ fun () ->
    let refused =
      match Rig.Buffer.create d 8 with
      | _ -> "done"
      | exception Invalid_argument why
        when String.starts_with ~prefix:"Rig.Buffer.create: " why ->
          "refused"
    in
    (S.holds (), refused, Rig.allocated d)
  in
  equal ~msg:"the count while held" int (before + 1) held;
  equal ~msg:"a create while held" string "refused" refused;
  equal ~msg:"bytes held after the refusal" int 0 allocated;
  equal ~msg:"the count once given back" int before (S.holds ());
  equal ~msg:"a create once given back" int 8
    (Rig.Buffer.length (Rig.Buffer.create d 8))

(* A call of rig's that a callback makes, labelled [what], with the name rig
   gives it, and whether it may take a lock: rig.mli lists those that take
   none. *)
type probe = { what : string; fn : string; locks : bool; call : unit -> unit }

(* What a probe's call answers: ["refused"] for an [Invalid_argument] that
   names the call. *)
let answer p =
  match p.call () with
  | () -> "done"
  | exception Invalid_argument why
    when String.starts_with ~prefix:("Rig." ^ p.fn ^ ": ") why ->
      "refused"
  | exception e -> Printexc.to_string e

(* The probes, over devices and values made before the walk lands, and the
   state a refused call must leave as it was. *)
let probes () =
  let d = Rig.memory_device (fresh "reenter:probe") in
  let victim = Rig.memory_device (fresh "reenter:victim") in
  let polled_name = fresh "reenter:polled" in
  let polled, pp = P.open_ polled_name in
  let image = Result.get_ok (Rig.Image.load polled "functions") in
  let b = B.create d 64 and b' = B.create d 64 and h = B.create Rig.host 64 in
  let s = Sub.make d [||] and run = Sub.Run.make () in
  let pt = Rig.submit s ~run ~buffers:[||] in
  let set = Rig.Local.new_key (fun _ -> 0) in
  let unset = Rig.Local.new_key (fun _ -> 0) in
  ignore (Rig.Local.get set d : int);
  let ba = Bigarray.Array1.create Bigarray.char Bigarray.c_layout 8 in
  let locks ?(what = "") fn call =
    { what = (if what = "" then fn else what); fn; locks = true; call }
  in
  let free ?(what = "") fn call =
    { what = (if what = "" then fn else what); fn; locks = false; call }
  in
  let state () =
    ( Rig.allocated d,
      Rig.submitted d,
      Rig.lost victim,
      Rig.failure (),
      Rig.Local.get set d )
  in
  let probes =
    [
      locks "memory_device" (fun () ->
          ignore (Rig.memory_device (Rig.name d)));
      locks "open_" (fun () -> ignore (Rig.open_ (P.opener polled_name pp)));
      locks "close" (fun () -> Rig.close victim);
      locks "set_budget" (fun () -> Rig.set_budget d (Rig.budget d));
      locks "free_cache" (fun () -> Rig.free_cache d);
      locks "Point.wait" (fun () -> Rig.Point.wait pt);
      locks "Buffer.create" (fun () -> ignore (B.create d 8));
      locks "Buffer.of_string" (fun () -> ignore (B.of_string "x"));
      locks "Buffer.borrow" (fun () -> ignore (B.borrow d h));
      locks "Buffer.wait" (fun () -> B.wait b Rig.Read);
      locks "Buffer.read" (fun () -> B.read b Bigarray.char ignore);
      locks "Buffer.write" (fun () -> B.write b Bigarray.char ignore);
      locks "Buffer.export" (fun () -> ignore (B.export h Bigarray.char));
      locks "Buffer.copy" (fun () -> B.copy ~src:b ~dst:b');
      locks "Buffer.blit_from_string" (fun () ->
          B.blit_from_string "x" 0 b 0 1);
      locks "Buffer.blit_to_bytes" (fun () ->
          B.blit_to_bytes b 0 (Bytes.create 1) 0 1);
      locks "Buffer.fill" (fun () -> B.fill b 'x');
      locks "Image.load" (fun () -> ignore (Rig.Image.load d "code"));
      locks "Image.mem" (fun () -> ignore (Rig.Image.mem image "fill"));
      locks "Image.params" (fun () -> ignore (Rig.Image.params image "fill"));
      locks "Image.threads" (fun () ->
          ignore (Rig.Image.threads image "fill"));
      locks "Image.entry" (fun () -> ignore (Rig.Image.entry image "fill"));
      locks "Submission.make" (fun () -> ignore (Sub.make d [||]));
      locks "submit" (fun () -> ignore (Rig.submit s ~run ~buffers:[||]));
      locks ~what:"Local.get of an unset value" "Local.get" (fun () ->
          ignore (Rig.Local.get unset d : int));
      locks "Profile.take" (fun () -> ignore (Rig.Profile.take ignore));
      free "name" (fun () -> ignore (Rig.name d));
      free "reaches" (fun () -> ignore (Rig.reaches d Rig.host));
      free "budget" (fun () -> ignore (Rig.budget d));
      free "allocated" (fun () -> ignore (Rig.allocated d));
      free "submitted" (fun () -> ignore (Rig.submitted d));
      free "signaled" (fun () -> ignore (Rig.signaled d));
      free "lost" (fun () -> ignore (Rig.lost victim));
      free "failure" (fun () -> ignore (Rig.failure ()));
      free "Point.reached" (fun () -> ignore (Rig.Point.reached pt));
      free "Buffer.view" (fun () -> ignore (B.view b ~first:0 ~length:1));
      free "Buffer.of_bigarray" (fun () -> ignore (B.of_bigarray ba));
      free "Claim.with_" (fun () ->
          Rig.Claim.with_ ~read:[ b ] ~donate:[] ignore);
      free "Submission.read" (fun () -> ignore (Sub.read 0 ~at:0));
      free "Submission.Run.make" (fun () -> ignore (Sub.Run.make ()));
      free "Profile.enabled" (fun () -> ignore (Rig.Profile.enabled ()));
      free "Local.new_key" (fun () -> ignore (Rig.Local.new_key ignore));
      free ~what:"Local.get of a set value" "Local.get" (fun () ->
          ignore (Rig.Local.get set d : int));
    ]
  in
  (probes, state)

(* Makes every probe's call from a [Gc.Memprof] callback at the [n]-th
   allocation of [f]: each call that may take a lock raises exactly when the
   callback runs inside a lock, and then changes nothing; every other call
   runs. rig stays consistent: [f] runs again once no callback lands. Is
   whether the allocation was reached, and whether the callback ran inside a
   lock. *)
let reenter f n =
  let at = Printf.sprintf ", re-entered at allocation %d" n in
  in_child at (fun () ->
      let probes, state = probes () in
      let held = ref false and answers = ref [] and changed = ref None in
      let call () =
        held := S.holds () > 0;
        let before = state () in
        answers := List.map (fun p -> (p.what, answer p)) probes;
        if !held && state () <> before then changed := Some ()
      in
      let reached = S.interrupted ~landing:(Call call) n f in
      if reached then begin
        let expected p =
          (p.what, if p.locks && !held then "refused" else "done")
        in
        equal ~msg:"the calls' answers"
          (list (pair string string))
          (List.map expected probes) !answers;
        is_none ~msg:"a change by refused calls" !changed
      end;
      f ();
      (reached, !held))

(* An open, whose sections of rig's table of names allocate. *)
let open_named () = ignore (Rig.memory_device (fresh "reenter"))

(* A drain that routes collected buffers into a device's cache, in sections
   of the device's lock that allocate. *)
let drain_cache () =
  let d = Rig.memory_device "reenter:cache" in
  let[@inline never] drop () =
    for k = 1 to 8 do
      ignore (Sys.opaque_identity (B.create d (k * 64)))
    done
  in
  drop ();
  Gc.full_major ();
  ignore (B.create d 0)

let test_reentry f () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  S.needs_fork ();
  let rec from n inside outside =
    match reenter f n with
    | true, true -> from (n + 1) (inside + 1) outside
    | true, false -> from (n + 1) inside (outside + 1)
    | false, _ -> (inside, outside)
  in
  let inside, outside = from 1 0 0 in
  greater ~msg:"landings inside a lock" int ~than:0 inside;
  greater ~msg:"landings outside every lock" int ~than:0 outside

(* Fails the process from a [Gc.Memprof] callback at the [n]-th allocation of
   an open. [fail] never raises: inside a lock it fails the process and leaves
   the owed stops to later calls, and outside every lock it runs them. Is
   whether the allocation was reached, and whether the callback ran inside a
   lock. *)
let fail_in n =
  let at = Printf.sprintf ", failed at allocation %d" n in
  in_child at (fun () ->
      let d, p = P.open_ (fresh "fail-in:polled") in
      let held = ref false and stops = ref (-1) in
      let call () =
        held := S.holds () > 0;
        Rig.fail "failed in a callback";
        stops := P.stops p
      in
      let open_ () =
        try ignore (Rig.memory_device (fresh "fail-in")) with Failure _ -> ()
      in
      let reached = S.interrupted ~landing:(Call call) n open_ in
      if reached then begin
        equal ~msg:"the failure" (option string) (Some "failed in a callback")
          (Rig.failure ());
        equal ~msg:"stops as the fail returned" int
          (if !held then 0 else 1)
          !stops;
        raises_match
          (function Rig.Lost _ -> true | _ -> false)
          (fun () -> B.create d 8);
        equal ~msg:"stops after a later call" int 1 (P.stops p)
      end;
      (reached, !held))

let test_fail_in () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  S.needs_fork ();
  let rec from n inside =
    match fail_in n with
    | true, held -> from (n + 1) (if held then inside + 1 else inside)
    | false, _ -> inside
  in
  greater ~msg:"fails inside a lock" int ~than:0 (from 1 0)

(* Whether [why] refuses an open because its thread is opening the name. *)
let opening_here why =
  String.ends_with ~suffix:" is opening on the calling thread" why

(* Opens a name from a [Gc.Memprof] callback at the [n]-th allocation of that
   name's own open: inside the open, the callback's open raises
   [Invalid_argument], at a lock of rig's or because its thread is the one
   opening the name, and nothing waits for good. Outside it, the callback's
   open answers the name's device. Either way the outer open answers it too.
   Is whether the allocation was reached, and whether the call raised because
   its thread is opening the name. *)
let reenter_own n =
  let at = Printf.sprintf ", re-entered at allocation %d" n in
  in_child at (fun () ->
      let name = fresh "reenter:own" in
      let refused = ref false and inner = ref None and outer = ref None in
      let call () =
        match Rig.memory_device name with
        | d -> inner := Some d
        | exception Invalid_argument why when opening_here why ->
            refused := true
        | exception Invalid_argument _ -> ()
      in
      let open_ () = outer := Some (Rig.memory_device name) in
      let reached = S.interrupted ~landing:(Call call) n open_ in
      let d = Rig.memory_device name in
      List.iter
        (fun (what, opened) ->
          Option.iter
            (fun o -> equal ~msg:what bool true (Rig.equal o d))
            opened)
        [ ("the outer open", !outer); ("the callback's open", !inner) ];
      (reached, !refused))

let test_reentry_own () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  S.needs_fork ();
  let rec from n refused =
    match reenter_own n with
    | true, r -> from (n + 1) (if r then refused + 1 else refused)
    | false, _ -> refused
  in
  greater ~msg:"opens refused as their thread is opening the name" int ~than:0
    (from 1 0)

(* Closes a machine's host from a [Gc.Memprof] callback at the [n]-th
   allocation of an open on that machine. Inside the open the close raises
   [Invalid_argument] and leaves the host open, and nothing waits for good.
   A later close ends the machine, each of its devices stopped once. Is
   whether the allocation was reached, and whether the close raised. *)
let close_in_open n =
  let at = Printf.sprintf ", closed at allocation %d" n in
  in_child at (fun () ->
      let drivers = ref [] in
      let polled ~word =
        let p = P.make () in
        drivers := p :: !drivers;
        Ok (P.bind p ~word)
      in
      let host =
        Rig.open_host ~machine:(fresh "close-in-open")
          { name = "CPU"; driver = (module P); make = polled }
        |> Result.get_ok
      in
      let raised = ref false in
      let call () =
        try Rig.close host with Invalid_argument _ -> raised := true
      in
      let open_ () =
        let o = { Rig_edge.name = "gpu"; driver = (module P); make = polled } in
        ignore (Rig.open_ ~machine:host o)
      in
      let reached = S.interrupted ~landing:(Call call) n open_ in
      if !raised then
        is_none ~msg:"the host after a refused close" (Rig.lost host);
      Rig.close host;
      equal ~msg:"the host once closed" (option string) (Some "closed")
        (Rig.lost host);
      List.iter
        (fun p -> equal ~msg:"a device's stops" int 1 (P.stops p))
        !drivers;
      (reached, !raised))

let test_close_in_open () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  S.needs_fork ();
  let rec from n raised =
    match close_in_open n with
    | true, r -> from (n + 1) (if r then raised + 1 else raised)
    | false, _ -> raised
  in
  greater ~msg:"closes that raised" int ~than:0 (from 1 0)

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
              ((opens :: caches :: borrow_kept :: waiting :: kept_run :: ops)
              @ staged @ List.map unmapped staged)
           @ [
               test "a profiled submit's time pair" test_timed;
               test "close of a machine's host" test_close;
               test "a drain that lets go of kept values" test_kept;
               test "a drain that routes memory into the cache" test_cached;
               test "a callback's calls inside an open"
                 (test_reentry open_named);
               test "a callback's calls inside a drain"
                 (test_reentry drain_cache);
               test "a callback's fail" test_fail_in;
               test "a lock counted through the edge" test_counted;
               test "a callback that opens the name an open is opening"
                 test_reentry_own;
               test
                 "a callback that closes a machine's host inside an open on \
                  it"
                 test_close_in_open;
             ]);
       ]
