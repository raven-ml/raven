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
   of its driver, with the exception's text, its stop not run until its
   close runs it once, or live, its name answering it, and it runs the
   operation again and closes. Either way
   its name then opens a device that runs the operation. rig promises no
   return of memory after an asynchronous exception but what a driver's call
   made, which the walks of records before calls check, so this walk checks
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
          (* The close runs the stop, which the interrupted call left owed. *)
          equal ~msg:"stops before the close" int 0 (P.stops p);
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
      let v = once (Sub.take (Sub.make producer [||])) in
      let s = Sub.make d [||] in
      let run = Sub.take s in
      Rig.Point.wait (Rig.submit run ~buffers:[||] ~waits:[| v |]))

(* Each device's run that [kept_run] submits with: of a copy, and of a fill
   whose block makes a submit take the run. *)
let kept =
  Rig.Local.new_key (fun d ->
      let fill = Sub.fill Rig_support.noop ~params:0 in
      let copy = Sub.copy ~src:(Slot 0) ~dst:(Slot 1) in
      let run = Sub.take (Sub.make d [| copy; fill |]) in
      Sub.Run.release run;
      run)

(* Submits a copy under a profile with its device's [kept] run, which serves
   each call in turn: an exception in one submit leaves the run free for the
   next. *)
let kept_run =
  op "profiled submit with a kept run" (fun d _ ->
      let run = Rig.Local.get kept d in
      let buffers = [| B.create d 8; B.create d 8 |] in
      let submit () = Rig.Point.wait (once ~buffers run) in
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
      let run = Sub.take s in
      let wait () = Rig.Point.wait (once ~buffers run) in
      let submit () = ignore (Rig.Profile.take wait) in
      submit ();
      let seen = List.length (P.violations ()) in
      Gc.full_major ();
      let got = outcome (fun () -> interrupting (Nth n) (ref None) submit) in
      mem ~msg:"outcome" string got [ "done"; "raised Injected" ];
      equal ~msg:"lost" (option string) None (Rig.lost d);
      Rig.Point.wait (once (Sub.take (Sub.make d [||])));
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
    ignore (once (Sub.take (Sub.make ~keep:v d [||])))
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

(* The regions a release gives back to a driver under one mask of the
   calling thread's (rig.mli's Errors section). *)
let regions_per_mask = 64

(* The buffers a release walk frees: enough for two masks. *)
let released_buffers = 100

let blocked () = Thread.sigmask SIG_BLOCK []

(* A SIGINT raised at the [n]-th allocation of [release], which gives back
   the memory of collected buffers to the driver one region at a time. Its
   handler, which returns, runs within 64 regions of the signal, or at the
   walk's end if that comes first, and if the signal came while the thread
   blocked it, once that mask's regions have all gone back: after a multiple
   of 64 regions, or at the walk's end. The thread's mask is as before,
   the device is live and leaks nothing: once the cache is freed the device
   holds no memory, and each region went back once. [~budget], set once the
   buffers are collected, makes the drain free what it would cache. Is
   whether the allocation was reached, whether the signal came while the
   thread blocked it after a first mask's regions, and the regions the walk
   gave back. *)
let release_signaled ?budget ~cache release n =
  let d, p = P.open_ (fresh "released") in
  let regions () = List.length (P.frees p) in
  let[@inline never] drop () =
    for _ = 1 to released_buffers do
      ignore (Sys.opaque_identity (B.create d 64))
    done
  in
  drop ();
  Gc.full_major ();
  Option.iter (Rig.set_budget d) budget;
  if cache then ignore (Sys.opaque_identity (B.create d 0));
  let before = blocked () in
  let landed = ref None and handled = ref None in
  let arrive () =
    landed := Some (regions (), List.mem Sys.sigint (blocked ()));
    S.interrupt ()
  in
  let previous =
    Sys.signal Sys.sigint
      (Sys.Signal_handle (fun _ -> handled := Some (regions ())))
  in
  let reached =
    Fun.protect ~finally:(fun () -> Sys.set_signal Sys.sigint previous)
    @@ fun () -> S.interrupted ~landing:(Call arrive) n (fun () -> release d)
  in
  let walked = regions () in
  equal ~msg:"the thread's mask" (list int) before (blocked ());
  let late =
    match !landed with
    | None -> false
    | Some (at, masked) ->
        let ran = Option.value !handled ~default:(-1) in
        at_least ~msg:"regions when the handler ran" int ~than:at ran;
        at_most ~msg:"regions when the handler ran" int
          ~than:(Int.min walked (at + regions_per_mask))
          ran;
        if masked && ran <> walked then
          equal ~msg:"regions past a mask's end when the handler ran" int 0
            (ran mod regions_per_mask);
        masked && at >= regions_per_mask
  in
  equal ~msg:"the loss" (option string) None (Rig.lost d);
  Rig.free_cache d;
  equal ~msg:"bytes held" int 0 (Rig.allocated d);
  let freed = List.map fst (P.frees p) in
  equal ~msg:"regions freed once" int (List.length freed)
    (List.length (List.sort_uniq Int.compare freed));
  Rig.close d;
  equal ~msg:"stops" int 1 (P.stops p);
  (reached, late, walked)

(* The run past the last allocation is the release whole, which gives back
   regions for more than one mask, and some signal comes while the thread
   blocks it after the first mask's. *)
let test_release_signaled ?budget ~cache release () =
  if Sys.win32 then skip ~reason:"Windows has no signal masks" ();
  let rec from n lates =
    match release_signaled ?budget ~cache release n with
    | true, late, _ -> from (n + 1) (if late then lates + 1 else lates)
    | false, _, walked ->
        greater ~msg:"regions the whole release gave back" int
          ~than:regions_per_mask walked;
        greater ~msg:"signals blocked after a first mask" int ~than:0 lates
  in
  from 1 0

(* Re-entry: a call that may take a lock, made from a [Gc.Memprof] callback
   inside a lock of its thread's, raises [Invalid_argument] naming it, having
   done nothing; outside every lock it runs as any call does. A call that
   takes no lock runs either way ({!Rig.section-domains}). *)

(* A lock a library counts through the edge refuses rig's calls on its
   thread until it is given back, and changes nothing meanwhile. *)
let test_counted () =
  let d = Result.get_ok (Rig.memory_device (fresh "counted")) in
  let before = Rig_edge_primitive.holds () in
  let held, refused, allocated =
    S.inside_lock @@ fun () ->
    let refused =
      match Rig.Buffer.create d 8 with
      | _ -> "done"
      | exception Invalid_argument why
        when String.starts_with ~prefix:"Rig.Buffer.create: " why ->
          "refused"
    in
    (Rig_edge_primitive.holds (), refused, Rig.allocated d)
  in
  equal ~msg:"the count while held" int (before + 1) held;
  equal ~msg:"a create while held" string "refused" refused;
  equal ~msg:"bytes held after the refusal" int 0 allocated;
  equal ~msg:"the count once given back" int before
    (Rig_edge_primitive.holds ());
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
  let d = Result.get_ok (Rig.memory_device (fresh "reenter:probe")) in
  let victim = Result.get_ok (Rig.memory_device (fresh "reenter:victim")) in
  let polled_name = fresh "reenter:polled" in
  let polled, pp = P.open_ polled_name in
  let image = Result.get_ok (Rig.Image.load polled "functions") in
  let fill = Option.get (Rig.Image.resolve image "fill") in
  (* Lost by a wait's fault, which leaves its stop owed. *)
  let own, own_p = P.open_ ~transport:true (fresh "reenter:own-word") in
  let own_pt = Rig.submit (Sub.take (Sub.make own [||])) ~buffers:[||] in
  P.fault own_p "the engine hung";
  (try Rig.Point.wait own_pt with Rig.Lost _ -> ());
  let b = B.create d 64 and b' = B.create d 64 and h = B.create Rig.host 64 in
  let run = Sub.take (Sub.make d [||]) in
  let pt = Rig.submit run ~buffers:[||] in
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
      locks "Image.resolve" (fun () -> ignore (Rig.Image.resolve image "fill"));
      locks "Submission.make" (fun () -> ignore (Sub.make d [||]));
      locks "submit" (fun () -> ignore (Rig.submit run ~buffers:[||]));
      locks ~what:"Local.get of an unset value" "Local.get" (fun () ->
          ignore (Rig.Local.get unset d : int));
      locks "Profile.take" (fun () -> ignore (Rig.Profile.take ignore));
      free "name" (fun () -> ignore (Rig.name d));
      free "reaches" (fun () -> ignore (Rig.reaches d Rig.host));
      free "budget" (fun () -> ignore (Rig.budget d));
      free "allocated" (fun () -> ignore (Rig.allocated d));
      free "submitted" (fun () -> ignore (Rig.submitted d));
      free "signaled" (fun () -> ignore (Rig.signaled d));
      free ~what:"signaled of a lost device's own word" "signaled" (fun () ->
          ignore (Rig.signaled own));
      free "lost" (fun () -> ignore (Rig.lost victim));
      free "failure" (fun () -> ignore (Rig.failure ()));
      free "Point.reached" (fun () -> ignore (Rig.Point.reached pt));
      free "Buffer.view" (fun () -> ignore (B.view b ~first:0 ~length:1));
      free "Buffer.reached" (fun () -> ignore (B.reached b Rig.Read_write));
      free "Buffer.of_bigarray" (fun () -> ignore (B.of_bigarray ba));
      free "Claim.with_" (fun () ->
          Rig.Claim.with_ ~read:[ b ] ~donate:[] ignore);
      free "Claim.release" (fun () -> Rig.Claim.release (Rig.Claim.hold [ h ]));
      free "Image.name" (fun () -> ignore (Rig.Image.name fill));
      free "Image.image" (fun () -> ignore (Rig.Image.image fill));
      free "Image.params" (fun () -> ignore (Rig.Image.params fill));
      free "Image.threads" (fun () -> ignore (Rig.Image.threads fill));
      free "Image.entry" (fun () -> ignore (Rig.Image.entry fill));
      free "Submission.read" (fun () -> ignore (Sub.read 0 ~at:0));
      free "Submission.take" (fun () ->
          Sub.Run.release (Sub.take (Sub.Run.submission run)));
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
        held := Rig_edge_primitive.holds () > 0;
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
  let d = Result.get_ok (Rig.memory_device "reenter:cache") in
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
   an open. [fail] never raises, inside a lock or outside every lock, and runs
   no stop: each device's runs at its close. Is whether the allocation was
   reached, and whether the callback ran inside a lock. *)
let fail_in n =
  let at = Printf.sprintf ", failed at allocation %d" n in
  in_child at (fun () ->
      let d, p = P.open_ (fresh "fail-in:polled") in
      let held = ref false and stops = ref (-1) in
      let call () =
        held := Rig_edge_primitive.holds () > 0;
        Rig.fail "failed in a callback";
        stops := P.stops p
      in
      let open_ () = ignore (Rig.memory_device (fresh "fail-in")) in
      let reached = S.interrupted ~landing:(Call call) n open_ in
      if reached then begin
        equal ~msg:"the failure" (option string) (Some "failed in a callback")
          (Rig.failure ());
        equal ~msg:"stops as the fail returned" int 0 !stops;
        raises_match
          (function Rig.Lost _ -> true | _ -> false)
          (fun () -> B.create d 8);
        equal ~msg:"stops after a later call" int 0 (P.stops p);
        Rig.close d;
        equal ~msg:"stops after the close" int 1 (P.stops p)
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
        | r -> inner := Some (Result.get_ok r)
        | exception Invalid_argument why when opening_here why ->
            refused := true
        | exception Invalid_argument _ -> ()
      in
      let open_ () = outer := Some (Result.get_ok (Rig.memory_device name)) in
      let reached = S.interrupted ~landing:(Call call) n open_ in
      let d = Result.get_ok (Rig.memory_device name) in
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

(* Records before calls. A call that makes memory, a mapping or code through
   Polled runs once with an exception at each of its allocations outside its
   counted calls, where it lands outside the driver's calls, and
   once with a SIGINT raised at each of its allocations, wherever it is, as a
   Ctrl-C comes: one that comes inside a driver's call lands once the call
   returned. rig records what a driver makes before the call, so after either
   no device is lost, each serves the call again, and once the call's values
   are dropped and collected, and the devices drained and their caches
   freed, the driver holds nothing the call made, but a device's mappings of
   words, which go back at its stop; once the devices are closed, it holds
   nothing. *)

(* Whether the allocation [a] is made inside a function whose slot [named]
   holds for. *)
let inside named (a : Gc.Memprof.allocation) =
  let holds s =
    match Printexc.Slot.name s with Some n -> named n | None -> false
  in
  match Printexc.backtrace_slots a.callstack with
  | Some slots -> Array.exists holds slots
  | None -> false

(* Whether the allocation [a] is made inside one of Polled's functions, which
   [Rig_support] defines in its module [Driver], as a give calls them. *)
let in_polled =
  inside (fun n ->
      String.starts_with ~prefix:"Rig_support.Driver." n
      || String.starts_with ~prefix:"Rig_support.Polled." n)

(* Whether the allocation [a] is made inside a driver's call: in a counted
   call, or in Polled. An exception there loses the device and may leave the
   call half done (rig.mli's Errors section). The profile's callback runs at
   the allocation, on its thread. *)
let in_call a = S.counted () > 0 || in_polled a

(* A call that makes something through Polled: [setup ()] opens the devices
   it uses, the first the one it makes on, and is them with the call.
   [words] says whether it maps words, which go back at the device's
   stop. *)
type making = {
  what : string;
  words : bool;
  setup : unit -> (Rig.t * P.t) list * (unit -> unit);
}

let making ?(words = false) what setup = { what; words; setup }

(* What [p] holds: its memory of each kind, its mappings, none counted with
   [but_maps], and its images. *)
let held ?(but_maps = false) p =
  [
    P.allocated p Device;
    P.allocated p Pinned;
    P.allocated p Mapped;
    (if but_maps then 0 else P.mapped p);
    P.images p;
  ]

(* A submit on [d] whose queue waits on a new producer's value. *)
let waits_on_producer d =
  let producer, pp = P.open_ (fresh "making:producer") in
  let v = once (Sub.take (Sub.make producer [||])) in
  let run = Sub.take (Sub.make d [||]) in
  ( [ (producer, pp) ],
    fun () -> Rig.Point.wait (Rig.submit run ~buffers:[||] ~waits:[| v |]) )

let makings =
  [
    making "an allocation of each kind" (fun () ->
        let d, p = P.open_ ~host_visible:false (fresh "making:create") in
        ( [ (d, p) ],
          fun () ->
            List.iter
              (fun memory -> ignore (B.create ~memory d 64 : B.t))
              [ B.Device; Pinned; Mapped ] ));
    making "an allocation from the cache" (fun () ->
        let d, p = P.open_ (fresh "making:cached") in
        ignore (Sys.opaque_identity (B.create d 64));
        Gc.full_major ();
        (* The drain of an empty buffer's creation caches the memory. *)
        ignore (B.create d 0 : B.t);
        ([ (d, p) ], fun () -> ignore (B.create d 64 : B.t)));
    making "a borrow of host memory" (fun () ->
        let d, p = P.open_ (fresh "making:borrow-host") in
        let h = B.create Rig.host page in
        ([ (d, p) ], fun () -> ignore (B.borrow d h : B.t option)));
    making "a borrow of a peer's memory" (fun () ->
        let o, po = P.open_ (fresh "making:owner") in
        let d, p = P.open_ (fresh "making:borrow-peer") in
        let b = B.create o 64 in
        ([ (d, p); (o, po) ], fun () -> ignore (B.borrow d b : B.t option)));
    making "a load of an image its driver places" (fun () ->
        let d, p = P.open_ (fresh "making:functions") in
        ( [ (d, p) ],
          fun () -> ignore (Result.get_ok (Rig.Image.load d "functions")) ));
    (* The load queues its code's copy and returns; work after it, waited
       for, runs the copy, so that its memory can go back. *)
    making "a load of an image into the device's memory" (fun () ->
        let d, p = P.open_ (fresh "making:code") in
        ( [ (d, p) ],
          fun () ->
            let after = Sub.take (Sub.make d [||]) in
            ignore (Result.get_ok (Rig.Image.load d "code:64"));
            Rig.Point.wait (once after) ));
    making ~words:true "a submit that maps rig's array to wait on a producer"
      (fun () ->
        let d, p = P.open_ ~waits_on:[ `Host ] (fresh "making:array") in
        let producers, call = waits_on_producer d in
        ((d, p) :: producers, call));
    making ~words:true "a submit that maps a producer's word to wait on it"
      (fun () ->
        let d, p =
          P.open_ ~maps_host:false ~waits_on:[ `Host ] (fresh "making:word")
        in
        let producers, call = waits_on_producer d in
        ((d, p) :: producers, call));
  ]

(* Runs [m]'s call with [landing] at the [n]-th of its allocations that
   [from] holds for, then checks what rig left: no device lost, the call
   served again, and nothing held by the driver once the call's values are
   collected and the devices closed. Is whether the allocation was
   reached. *)
let made m ~from ~landing n =
  let at = Printf.sprintf ", landing at allocation %d" n in
  in_child at (fun () ->
      let run () =
        let devices, call = m.setup () in
        Gc.full_major ();
        let reached = S.interrupted ~from ~landing n call in
        List.iter
          (fun (d, _) -> equal ~msg:"lost" (option string) None (Rig.lost d))
          devices;
        call ();
        (devices, reached)
      in
      let devices, reached = run () in
      collect ();
      List.iter (fun (d, _) -> settle d) devices;
      List.iter
        (fun (d, p) ->
          equal
            ~msg:(Rig.name d ^ " holds once the values are collected")
            (list int) [ 0; 0; 0; 0; 0 ]
            (held ~but_maps:m.words p))
        devices;
      List.iter (fun (d, _) -> Rig.close d) devices;
      List.iter (fun (d, _) -> settle d) devices;
      List.iter
        (fun (d, p) ->
          equal ~msg:(Rig.name d ^ " holds") (list int) [ 0; 0; 0; 0; 0 ]
            (held p))
        devices;
      reached)

let makes m ~from ~landing () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  S.needs_fork ();
  let rec walk n = if made m ~from ~landing n then walk (n + 1) else n - 1 in
  greater ~msg:"the landings" int ~than:0 (walk 1)

let raise_once = S.Call (fun () -> raise Exit)

(* Polled, whose facts state that the host addresses its memory, which it
   does not. *)
module Overstated = struct
  include P

  let facts d = { (P.facts d) with host_addresses = true }
end

(* Allocates on a device whose driver breaks its facts, with an exception at
   the [n]-th allocation outside the driver's calls, then again: the second
   allocation loses the device if the first did not, and once it is closed
   the driver holds no region. Is whether the allocation was reached. *)
let overstated n =
  let at = Printf.sprintf ", landing at allocation %d" n in
  in_child at (fun () ->
      let p = P.make ~host_visible:false () in
      let name = fresh "making:overstated" in
      let make ~word = Ok (P.bind p ~word) in
      let d =
        Result.get_ok (Rig.open_ { name; driver = (module Overstated); make })
      in
      let create () =
        match B.create d 64 with _ -> () | exception Rig.Lost _ -> ()
      in
      let reached =
        S.interrupted ~from:(fun a -> not (in_call a)) ~landing:raise_once n
          create
      in
      create ();
      is_some ~msg:"the loss" (Rig.lost d);
      Rig.close d;
      settle d;
      equal ~msg:"held" (list int) [ 0; 0; 0; 0; 0 ] (held p);
      reached)

let test_overstated () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  S.needs_fork ();
  let rec walk n = if overstated n then walk (n + 1) else n - 1 in
  greater ~msg:"the landings" int ~than:0 (walk 1)

let making_tests =
  List.concat_map
    (fun m ->
      [
        test (m.what ^ ", an exception outside its driver's calls")
          (makes m ~from:(fun a -> not (in_call a)) ~landing:raise_once);
        test (m.what ^ ", a SIGINT at any allocation")
          (makes m ~from:(fun _ -> true) ~landing:S.Signal);
      ])
    makings
  @ [
      (* A load the driver places stores its answer as it is: rig allocates
         nothing in its counted call. *)
      test
        "a load of an image its driver places, an exception outside Polled's \
         functions"
        (makes
           (List.find
              (fun m -> m.what = "a load of an image its driver places")
              makings)
           ~from:(fun a -> not (in_polled a))
           ~landing:raise_once);
      test
        "an allocation whose region breaks its driver's facts, an exception \
         outside its driver's calls"
        test_overstated;
    ]

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
               test "a Ctrl-C at any allocation of a free_cache"
                 (test_release_signaled ~cache:true Rig.free_cache);
               test "a Ctrl-C at any allocation of a drain that frees"
                 (test_release_signaled ~budget:256 ~cache:false (fun d ->
                      ignore (Sys.opaque_identity (B.create d 0))));
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
         group ~timeout
           "an exception once a driver's call returned leaves what it made \
            to rig"
           making_tests;
       ]
