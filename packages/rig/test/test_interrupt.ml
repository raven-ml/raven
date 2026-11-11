(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Interrupt walks. An operation runs once for each allocation it makes, in a
   forked child, with every allocation from that one on raising until the
   operation returns, as an asynchronous exception does, and handlers that
   allocate meet it again. An operation that repeats one call only to make a
   table grow runs once for the first allocation of each of its call stacks
   instead: its repeats differ by the table's contents alone, and the growth
   allocates from call stacks of its own. Either walk then runs the operation
   so again from the first allocation at the same call stack. After that, rig
   is consistent: no profile is left taken, and the device is either lost, by
   an exception that escaped a call of its driver, with the exception's text,
   its stop not run until its close runs it once, or live, its name answering
   it, and it runs the operation again and closes. Either way its name then
   opens a device that runs the operation. rig promises no return of memory
   after an asynchronous exception but what a driver's call made, which the
   walks of records before calls check, so this walk checks none.

   Poll walks. A call runs once for each poll point it reaches, in native
   code and in bytecode, where every application of an OCaml function and
   every exit from a handler polls too, with a SIGINT whose handler raises
   at that point ([polling]). After it rig is consistent as above, and on a
   device that ran the call before, the device and the host hold what they
   held before and the buffers the caller keeps free: the call leaks
   neither memory that a driver's call made nor a claim. *)

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

(* The call stack of an allocation within an interrupted call, by code
   addresses. *)
type stack = Printexc.raw_backtrace_entry array

(* The first allocation that raises in an interrupted call: its [n]-th, the
   first from a call stack that [landed] does not hold, or the first from the
   call stack [s]. *)
type allocation =
  | Nth of int
  | New_stack of (stack, unit) Hashtbl.t
  | Site of stack

(* The call stack of the allocation [sample] within a call made [outer]
   frames deep. *)
let within outer (sample : Gc.Memprof.allocation) : stack =
  let s = Printexc.raw_backtrace_entries sample.callstack in
  Array.sub s 0 (Array.length s - outer)

(* Whether [slot] lies in the standard library, whose functions every caller
   shares. *)
let in_stdlib slot =
  match Printexc.Slot.name slot with
  | Some n ->
      String.starts_with ~prefix:"Stdlib" n
      || String.starts_with ~prefix:"Camlinternal" n
  | None -> false

(* [slot]'s function and line. *)
let frame slot =
  let name =
    Option.value ~default:"an unnamed function" (Printexc.Slot.name slot)
  in
  match Printexc.Slot.location slot with
  | Some l -> Printf.sprintf "%s (%s:%d)" name l.filename l.line_number
  | None -> name

(* The frames of the call stack [s], by function and line, from the
   innermost out to the first outside the standard library: a stack that
   allocates in a shared function, such as [Printf.sprintf], is named by its
   caller too. *)
let frames (s : stack) =
  let slots =
    List.concat_map
      (fun e ->
        match Printexc.backtrace_slots_of_raw_entry e with
        | Some slots -> Array.to_list slots
        | None -> [])
      (Array.to_list s)
  in
  let rec out = function
    | [] -> []
    | slot :: rest when in_stdlib slot -> frame slot :: out rest
    | slot :: _ -> [ frame slot ]
  in
  match out slots with
  | [] -> "a frame without debugging information"
  | named -> String.concat ", called from " named

(* The pipe on which a child of [in_child] notes where it is; [None] outside
   one. *)
let notes = ref None

(* [note s], in a child of [in_child], tells the parent where the child is:
   the parent adds the last note to any failure of the child it reports, a
   hang or a crash included. *)
let note s =
  match !notes with
  | None -> ()
  | Some fd ->
      let line = s ^ "\n" in
      ignore (Unix.write_substring fd line 0 (String.length line))

(* [interrupting a site f] is [f ()] with every allocation from [a] on raising
   [Injected] until [f] returns: a memory profile samples every allocation,
   and its callback raises. [site] is set to the call stack of the first
   allocation that raised, which the callback notes. *)
let interrupting a site f =
  let left = ref (match a with Nth n -> n | New_stack _ | Site _ -> 0)
  and armed = ref false
  and raising = ref false in
  let outer = Printexc.raw_backtrace_length (Printexc.get_callstack max_int) in
  let alloc (sample : Gc.Memprof.allocation) =
    if !armed && not !raising then begin
      let here =
        match a with
        | Nth _ ->
            decr left;
            !left = 0
        | New_stack landed -> not (Hashtbl.mem landed (within outer sample))
        | Site s -> within outer sample = s
      in
      if here then begin
        let s = within outer sample in
        site := Some s;
        note (", landed at " ^ frames s);
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

(* The signals the calling thread blocks. *)
let blocked () = Thread.sigmask SIG_BLOCK []

let bytecode = Sys.backend_type <> Native

(* [polling n f] is [f ()] with SIGINT's handler run at each poll point of
   the calling thread from [f]'s start on, and raising [Injected] at the
   [n]-th, as a Ctrl-C lands at a poll point. Allocations poll, and in
   bytecode so does every application of an OCaml function and every exit
   from a handler. The handler counts the poll points: it sends the signal
   again each time it runs, which the runtime blocks until the handler
   returns, so the next poll point runs it once more. rig blocks signals
   across its calls to a driver but in the driver's waits, so the walk lands
   only where a Ctrl-C can. Is whether [f] reached the [n]-th poll point.
   The handler is left in place, for a forked child: the signal it sent last
   lands at a later poll point, where it does nothing.

   In native code every poll point from the [n]-th on raises until [f]
   returns, so handlers that poll meet it again, as the allocation walk's
   do: a handler's first step must not poll. In bytecode, where every
   application polls, the walk raises once, as one Ctrl-C does: a handler
   gives back what its call took by applying a function, under a lock that
   only an application takes.

   In bytecode the exit from a handler runs pending handlers again until
   none is pending, so a handler that sends its signal again would run
   there for good. A run at the call stack of the run before it is such a
   repeat, or the next turn of a loop whose body neither applies nor
   allocates. It sends nothing, and the next allocation counts and sends
   again, from a memory profile: the walk lands at the exit once, then at
   the allocation. The poll points between them, applications, go
   unwalked: a poll runs a pending signal's handler, a finaliser or a
   memory profile's callback, and a signal pending at the exit runs there,
   a finaliser after a collection, a callback at an allocation, so none
   runs first at the next application. Native code polls every turn of
   such a loop. *)
let polling n f =
  let me = Thread.id (Thread.self ()) in
  let left = ref n and armed = ref true in
  (* Counts a poll point and sends the signal for the next, or raises at
     the [n]-th. *)
  let reach () =
    if !left > 1 then begin
      decr left;
      S.interrupt ()
    end
    else begin
      left := 0;
      if bytecode then armed := false else S.interrupt ();
      raise Injected
    end
  in
  (* The handler's call stack when it last ran, its own frame first, and
     whether it sent nothing since. *)
  let last = ref [||] and paused = ref false in
  let handle _ =
    if !armed then
      if Thread.id (Thread.self ()) <> me then S.interrupt ()
      else if bytecode then begin
        let here =
          Printexc.raw_backtrace_entries (Printexc.get_callstack max_int)
        in
        if here = !last then paused := true
        else begin
          last := here;
          reach ()
        end
      end
      else reach ()
  in
  (* An allocation where a Ctrl-C could land resumes: neither the handler's
     own, of its call stack, which [last] begins with while paused, nor one
     under rig's mask, as within a call of a driver. *)
  let resume (a : Gc.Memprof.allocation) =
    let s = Printexc.raw_backtrace_entries a.callstack in
    if !paused && !armed
       && Thread.id (Thread.self ()) = me
       && not (Array.length s > 0 && s.(0) = !last.(0))
       && not (List.mem Sys.sigint (blocked ()))
    then begin
      paused := false;
      last := [||];
      reach ()
    end;
    None
  in
  let profile =
    if not bytecode then None
    else
      let tracker =
        {
          Gc.Memprof.null_tracker with
          alloc_minor = resume;
          alloc_major = resume;
        }
      in
      Some (Gc.Memprof.start ~sampling_rate:1. ~callstack_size:1 tracker)
  in
  let stop () =
    Option.iter
      (fun p ->
        Gc.Memprof.stop ();
        Gc.Memprof.discard p)
      profile
  in
  Sys.set_signal Sys.sigint (Sys.Signal_handle handle);
  S.interrupt ();
  (* [armed] is cleared first, in line: in bytecode the application of
     [stop] polls. *)
  match f () with
  | () ->
      armed := false;
      stop ();
      !left = 0
  | exception e ->
      armed := false;
      stop ();
      if e != Injected then raise e;
      true

(* [in_child what f] is [f ()], run in a forked child, which starts from this
   process's state: what one attempt leaves, such as memory an interrupt leaks,
   never changes the next one's allocations. It fails with the child's
   exception, and if the child has not answered within [child_limit] s: a
   lock that an exception left held makes the next call that takes it wait
   for good, in C, where no timeout interrupts it. The child's answer, or
   the end of its pipe as it exits, wakes the wait. A failure adds the
   child's last [note] to [what]. *)
let in_child what (f : unit -> 'a) : 'a =
  let r, w = Unix.pipe () and noted, noting = Unix.pipe () in
  match Unix.fork () with
  | 0 ->
      Unix.close r;
      Unix.close noted;
      notes := Some noting;
      let answer : ('a, string) result =
        match f () with v -> Ok v | exception e -> Error (Printexc.to_string e)
      in
      let oc = Unix.out_channel_of_descr w in
      Marshal.to_channel oc answer [];
      close_out oc;
      Unix._exit 0
  | pid -> (
      Unix.close w;
      Unix.close noting;
      (* Read once the child is gone, so that its notes have ended. *)
      let where () =
        let ic = Unix.in_channel_of_descr noted in
        let lines = In_channel.input_lines ic in
        close_in ic;
        match List.rev lines with last :: _ -> what ^ last | [] -> what
      in
      let until = Unix.gettimeofday () +. child_limit in
      let rec ready () =
        let left = until -. Unix.gettimeofday () in
        left > 0.
        &&
        match Unix.select [ r ] [] [] left with
        | [], _, _ -> ready ()
        | _ -> true
        | exception Unix.Unix_error (EINTR, _, _) -> ready ()
      in
      if not (ready ()) then begin
        Unix.kill pid Sys.sigkill;
        ignore (Unix.waitpid [] pid);
        Unix.close r;
        failf "no return from rig%s after %.0f s" (where ()) child_limit
      end;
      let ic = Unix.in_channel_of_descr r in
      let answer = In_channel.input_all ic in
      close_in ic;
      let _, status = Unix.waitpid [] pid in
      match (answer, status) with
      | "", WEXITED n -> failf "the child exited %d%s" n (where ())
      | "", (WSIGNALED n | WSTOPPED n) ->
          failf "the child got %s%s" (Sys.signal_to_string n) (where ())
      | answer, _ -> (
          match (Marshal.from_string answer 0 : ('a, string) result) with
          | Ok v ->
              Unix.close noted;
              v
          | Error e -> failf "%s%s" e (where ())))

(* The device an interrupted operation runs on: one just opened, or one the
   operation ran on once, which made what the device keeps for good, and whose
   memory from that run the collector took, which the interrupted run drains
   ([Used]) or which went back before it ([Settled]), its caches freed: the
   run then makes its memory anew. *)
type device = Fresh | Used | Settled

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

(* Checks that rig is consistent once a call of [op] on [d], the device of
   [p] opened as [name], was interrupted: no profile is left taken, [d] is
   either lost by an exception that escaped a call of its driver, its stop
   owed until its close runs it once, or live, its name answering it, and it
   runs [op] again and closes; then [name] opens a device that runs [op]. *)
let consistent op name d p =
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
    (* The borrowing devices close before the memory they mapped goes, over
       a few rounds of collections and drains. *)
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
  end

(* What [d] holds once its work is done and what the caller dropped has
   settled: the bytes it and the host hold in their budgets, and the regions
   its driver [p] holds. *)
let census d p =
  Rig.Point.wait (once (Sub.take (Sub.make d [||])));
  settle d;
  (Rig.allocated d, Rig.allocated Rig.host, List.sort compare (P.outstanding p))

(* Runs [op] interrupted from the allocation [a], then from the same
   allocation again, and checks that rig is consistent after it, failing
   with [at] added. Is whether the run raised, and the call stack of the
   allocation if the run reached it. *)
let interrupt start op ~at a =
  in_child at (fun () ->
      let name = fresh op.name in
      let d, p = op.opener name in
      if start <> Fresh then op.prepare d None;
      if start = Settled then ignore (census d p : int * int * int list);
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
      let raised = run a in
      let landed = !site in
      if Rig.lost d = None then
        Option.iter (fun s -> ignore (run (Site s))) landed;
      consistent op name d p;
      (raised, landed))

(* A call that a poll walk interrupts: [made op d] is the buffers that the
   call of [op] on [d] reads and writes, which its caller keeps, and [ready op
   d bs] that call on them, made ready outside the walk: what [op] makes
   first is the walk of another operation. *)
type call = {
  made : op -> Rig.t -> B.t list;
  ready : op -> Rig.t -> B.t list -> unit -> unit;
}

let call =
  {
    made = (fun _ _ -> []);
    ready =
      (fun op d _ ->
        let c = op.prepare d in
        fun () -> c None);
  }

(* The copies and the fill through the staging memory, whose checks read
   their bytes one at a time, each read a poll point: a poll walk interrupts
   the copy or the fill alone, and the check runs after, in [consistent]. *)
let staged_call =
  let made op d =
    let on_d = B.create d staged_bytes in
    match op.name with
    | "copy into" -> [ filled staged_bytes; on_d ]
    | "copy out of" ->
        B.copy ~src:(filled staged_bytes) ~dst:on_d;
        [ on_d; B.create Rig.host staged_bytes ]
    | _ -> [ on_d ]
  in
  let ready _ _ = function
    | [ src; dst ] -> fun () -> B.copy ~src ~dst
    | [ b ] -> fun () -> B.fill b 'f'
    | _ -> invalid_arg "staged_call: one or two buffers"
  in
  { made; ready }

(* [call] of [op] on [d] and [bs] interrupted from its [n]-th poll point:
   what the call made is unreachable once it returns. *)
let[@inline never] polled n call op d bs =
  let f = call.ready op d bs in
  Gc.full_major ();
  polling n f

(* A device opened for [op]'s call [call] as [start] says, its name, its
   driver, the call's buffers, and on a device that ran [op] once what it
   holds before the call, as [census] reads it: read now on a [Settled]
   one, and on a [Used] one, which a census would settle, [used]. *)
let opened_for start op call ~used =
  let name = fresh op.name in
  let d, p = op.opener name in
  let bs = call.made op d in
  if start <> Fresh then op.prepare d None;
  let before =
    match start with
    | Fresh -> None
    | Used -> used
    | Settled -> Some (census d p)
  in
  (name, d, p, bs, before)

(* What a device that ran [op] once holds with the buffers of [call] kept, as
   [census] reads it, from a child of its own: the runs of a poll walk from a
   [Used] device compare with it, and leave their device's memory from that
   run to the interrupted call to drain. *)
let held_once op call =
  in_child ", reading what a device that ran it holds" (fun () ->
      let _, _, _, bs, before = opened_for Settled op call ~used:None in
      ignore (Sys.opaque_identity bs);
      Option.get before)

(* Runs [op]'s call [call] interrupted from its [n]-th poll point, and checks
   that the thread's signal mask is as before and that rig is consistent
   after it. On a device that ran [op] once, which made what it keeps for
   good, it checks too that a live device holds what such a device held
   ([opened_for]), once what the call made has settled: an asynchronous
   exception may leak the memory of the call it interrupts, but none that a
   driver made for it, nor memory taken from the cache, and the buffers the
   caller keeps free. Is whether the run reached that poll point. *)
let interrupt_polls start op call ~used n =
  let at = Printf.sprintf ", interrupted from poll point %d" n in
  in_child at (fun () ->
      let name, d, p, bs, before = opened_for start op call ~used in
      let mask = blocked () in
      let reached = polled n call op d bs in
      equal ~msg:"the thread's mask" (list int) mask (blocked ());
      (match before with
      | Some before when Rig.lost d = None ->
          equal ~msg:"what the device and the host hold"
            (triple int int (list int))
            before (census d p);
          (* No claim of the call outlives it. *)
          List.iter
            (fun b ->
              equal ~msg:"a buffer of the call frees" bool true (B.free b))
            bs
      | _ -> ());
      consistent op name d p;
      reached)

(* Where a walk lands: at each allocation of the operation, or at the first
   allocation of each of its call stacks. A run of a stack walk lands at the
   first stack that no run before it landed at, and the walk ends at the
   first run that meets none: such a run met every stack of the operation,
   so none goes unlanded, and a stack that only the collector's timing makes,
   such as a finaliser's, costs the run that meets it. *)
type walk = Each_allocation | Each_stack

let interrupts ?(walk = Each_allocation) start op () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  S.needs_fork ();
  match walk with
  | Each_allocation ->
      let rec from n =
        let at = Printf.sprintf ", interrupted from allocation %d" n in
        if fst (interrupt start op ~at (Nth n)) then from (n + 1)
      in
      from 1
  | Each_stack ->
      let landed = Hashtbl.create 64 in
      let at = ", interrupted from a call stack no run landed at before" in
      let rec walk () =
        match snd (interrupt start op ~at (New_stack landed)) with
        | Some s ->
            Hashtbl.replace landed s ();
            walk ()
        | None -> ()
      in
      walk ()

(* Walks [op] by its poll points, from the first until a run returns before
   it reaches its own. *)
let polls ?(call = call) start op () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  S.needs_fork ();
  let used = if start = Used then Some (held_once op call) else None in
  let rec from n =
    if interrupt_polls start op call ~used n then from (n + 1)
  in
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
  let d = Result.get_ok (Rig.software (fresh "counted")) in
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
  let d = Result.get_ok (Rig.software (fresh "reenter:probe")) in
  let victim = Result.get_ok (Rig.software (fresh "reenter:victim")) in
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
  (* A buffer for the take alone, which kills it. *)
  let taken = B.create d 64 in
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
      locks "software" (fun () ->
          ignore (Rig.software (Rig.name d)));
      locks "open_" (fun () -> ignore (Rig.open_ (P.opener polled_name pp)));
      locks "close" (fun () -> Rig.close victim);
      locks "set_budget" (fun () -> Rig.set_budget d (Rig.budget d));
      locks "free_cache" (fun () -> Rig.free_cache d);
      locks "Point.wait" (fun () -> Rig.Point.wait pt);
      locks "Point.wait_for" (fun () ->
          ignore (Rig.Point.wait_for ~ns:max_int pt));
      locks ~what:"Point.wait_for with no time" "Point.wait_for" (fun () ->
          ignore (Rig.Point.wait_for ~ns:0 pt));
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
      free "Point.submitted" (fun () -> ignore (Rig.Point.submitted d));
      free "Buffer.view" (fun () -> ignore (B.view b ~first:0 ~length:1));
      free "Buffer.reached" (fun () -> ignore (B.reached b Rig.Read_write));
      free "Buffer.of_bigarray" (fun () -> ignore (B.of_bigarray ba));
      free "Buffer.take" (fun () -> ignore (B.take ~why:"probed" [ taken ]));
      free "Buffer.release" (fun () -> B.release (B.keep [ h ]));
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
let open_named () = ignore (Rig.software (fresh "reenter"))

(* A drain that routes collected buffers into a device's cache, in sections
   of the device's lock that allocate. *)
let drain_cache () =
  let d = Result.get_ok (Rig.software "reenter:cache") in
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
      let open_ () = ignore (Rig.software (fresh "fail-in")) in
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

(* Loses a device from a [Gc.Memprof] callback at each allocation of an open.
   [lose] returns, inside a lock and outside every lock, and runs no stop: the
   device's stop runs once, at its close. *)
let test_lose_in () =
  let seen = S.seen () in
  let rec from n =
    let at = Printf.sprintf "lost at allocation %d" n in
    let d, p = P.open_ (fresh "lose-in:polled") in
    let stops = ref (-1) and opened = ref None in
    let call () =
      Rig.lose d "lost in a callback";
      stops := P.stops p
    in
    let landing, landed = S.landing seen "Rig.lose: " call in
    let reached =
      S.interrupted ~landing n (fun () ->
          opened := Some (Rig.software (fresh "lose-in")))
    in
    (match S.returned at landed with
    | None -> ()
    | Some _ ->
        equal ~msg:at (option string) (Some "lost in a callback") (Rig.lost d);
        equal ~msg:(at ^ ", stops as the lose returned") int 0 !stops);
    Rig.close d;
    equal ~msg:(at ^ ", stops after the close") int 1 (P.stops p);
    Option.iter (fun o -> Rig.close (Result.get_ok o)) !opened;
    if reached then from (n + 1)
  in
  from 1;
  S.covered seen

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
        match Rig.software name with
        | r -> inner := Some (Result.get_ok r)
        | exception Invalid_argument why when opening_here why ->
            refused := true
        | exception Invalid_argument _ -> ()
      in
      let open_ () = outer := Some (Result.get_ok (Rig.software name)) in
      let reached = S.interrupted ~landing:(Call call) n open_ in
      let d = Result.get_ok (Rig.software name) in
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
   many as make the cache grow. "a drain that routes memory into the cache"
   walks every allocation of a drain. *)
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

(* Gives back memory of buffers dropped as the call is made ready: [release]
   of [d] frees it from the cache, or, under a budget of [budget] bytes, as
   a drain judges it. The drops fit the budget an earlier run left. *)
let freeing ?budget name release =
  op name (fun d ->
      Rig.set_budget d max_int;
      let[@inline never] drop () =
        for _ = 1 to 3 do
          ignore (Sys.opaque_identity (B.create d 64))
        done
      in
      drop ();
      Gc.full_major ();
      (match budget with
      | Some n -> Rig.set_budget d n
      | None -> ignore (Sys.opaque_identity (B.create d 0)));
      fun _ -> release d)

(* The host's moves of the bytes of a buffer it addresses, under a read
   claim. *)
let moves =
  let s = String.make page 'p' and bytes = Bytes.create page in
  [
    ("blit from a string", fun b -> B.blit_from_string s 0 b 0 page);
    ("blit to bytes", fun b -> B.blit_to_bytes b 0 bytes 0 page);
    ("fill by the host", fun b -> B.fill b 'h');
    ("read", fun b -> B.read b Bigarray.char ignore);
  ]

let host_moves =
  List.map
    (fun (name, move) ->
      op name (fun d ->
          let b = B.create d page in
          fun _ -> move b))
    moves

(* [host_moves]' calls, on a buffer the caller keeps. *)
let host_call =
  let ready op _ = function
    | [ b ] ->
        let move = List.assoc op.name moves in
        fun () -> move b
    | _ -> invalid_arg "host_call: one buffer"
  in
  { made = (fun _ d -> [ B.create d page ]); ready }

(* Frees [d]'s cache while work it was handed is not done, which defers each
   free until the work is, then waits for the work: the call is the drain
   that runs the frees deferred. *)
let deferred =
  op "a drain that runs deferred frees" (fun d ->
      let[@inline never] drop () =
        for _ = 1 to 3 do
          ignore (Sys.opaque_identity (B.create d 64))
        done
      in
      drop ();
      Gc.full_major ();
      let v = once (Sub.take (Sub.make d [||])) in
      Rig.free_cache d;
      Rig.Point.wait v;
      fun _ -> ignore (Sys.opaque_identity (B.create d 0)))

let freeings =
  [
    freeing "free_cache" Rig.free_cache;
    freeing ~budget:64 "a drain that frees" (fun d ->
        ignore (Sys.opaque_identity (B.create d 0)));
    deferred;
  ]

(* A run of [d] whose copy names two buffers, both of which a submit's frame
   claims. *)
let copy_run d =
  Sub.take (Sub.make d [| Sub.copy ~src:(Slot 0) ~dst:(Slot 1) |])

let submit_two =
  op "submit of two buffers" (fun d ->
      let run = copy_run d and buffers = [| B.create d 8; B.create d 8 |] in
      fun _ -> Rig.Point.wait (once ~buffers run))

(* [submit_two]'s call, on buffers the caller keeps. *)
let two_call =
  let ready _ d = function
    | [ a; b ] ->
        let run = copy_run d and buffers = [| a; b |] in
        fun () -> Rig.Point.wait (once ~buffers run)
    | _ -> invalid_arg "two_call: two buffers"
  in
  { made = (fun _ d -> [ B.create d 8; B.create d 8 ]); ready }

(* Submits a copy of two buffers on [d], which its work runs later, and
   drops the buffers. *)
let[@inline never] leave_busy d =
  ignore (once ~buffers:[| B.create d 8; B.create d 8 |] (copy_run d))

let[@inline never] drop d n = ignore (Sys.opaque_identity (B.create d n))

(* Opens [name] as a device that holds what its stop's drain gives back,
   then loses it to a fault: memory retiring behind work it has not run, a
   free deferred until that work is done, memory in its cache, its mapping
   of a producer's word, its mapping of a peer's memory that a borrow keeps,
   and one whose borrow was collected. Is the device, its driver and the
   borrow it keeps. *)
let lost_holding name =
  let producer, _ = P.open_ (fresh "producer") in
  let d, p = P.open_ ~maps_host:false ~waits_on:[ `Host ] name in
  let v = once (Sub.take (Sub.make producer [||])) in
  Rig.Point.wait
    (Rig.submit (Sub.take (Sub.make d [||])) ~buffers:[||] ~waits:[| v |]);
  let kept = B.borrow d (B.create (peer ()) 64) in
  ignore (Sys.opaque_identity (B.borrow d (B.create (peer ()) 64)));
  leave_busy d;
  drop d 64;
  collect ();
  Rig.free_cache d;
  drop d 256;
  collect ();
  drop d 0;
  P.fault p "lost";
  (match B.create d 1024 with
  | _ -> fail "a create on a faulted device returned"
  | exception Rig.Lost _ -> ());
  (d, p, kept)

(* Runs a close of a lost device that holds memory ([lost_holding]), or an
   open of its name, which runs its stop and its drain first, interrupted
   from its [n]-th poll point. Then the thread's mask is as before, a close
   ends the device, whose stop ran once, and once what it held has settled
   its driver holds nothing: neither its stop nor its drain left the stop
   claimed or leaked what the device held. Is whether the run reached that
   poll point. *)
let interrupt_stop how n =
  let at = Printf.sprintf ", interrupted from poll point %d" n in
  in_child at (fun () ->
      let name = fresh "lost" in
      let d, p, kept = lost_holding name in
      let call () =
        match how with
        | `Close -> Rig.close d
        | `Open -> ignore (P.open_ name : Rig.t * P.t)
      in
      Gc.full_major ();
      let mask = blocked () in
      let reached = polling n call in
      equal ~msg:"the thread's mask" (list int) mask (blocked ());
      Rig.close d;
      equal ~msg:"stops" int 1 (P.stops p);
      settle d;
      equal ~msg:"what its driver holds" (list int) [ 0; 0; 0; 0; 0 ] (held p);
      ignore (Sys.opaque_identity kept);
      reached)

(* A producer opened in the slot of [last], closed first, with its driver,
   and the submit on [d] that waits on its value through [d]'s mapping of
   its word. *)
let producing ?last d =
  Option.iter Rig.close last;
  let producer, pp = P.open_ (fresh "producer") in
  let v = once (Sub.take (Sub.make producer [||])) in
  let run = Sub.take (Sub.make d [||]) in
  ((producer, pp), fun () -> Rig.submit run ~buffers:[||] ~waits:[| v |])

(* Runs a submit on a device whose queue waited on two producers of one
   slot, then a drain, interrupted from its [n]-th poll point. The device's
   last submit before, on the second producer, replaced the slot's pair, and
   retired its mapping of the first producer's word among its releases,
   which that submit's end made due; the call's submit, on a third producer
   of the slot, replaces the pair again, gives that mapping back, and
   retires the second's, which the drain gives back. A mapping that the
   call made may return at the device's stop instead ({!Rig.section-errors}),
   so the checks follow the close of the device and of the producers: the
   thread's mask is as before, and once what they held has settled their
   drivers hold nothing, a mapping of a producer's word included. Is whether
   the run reached that poll point. *)
let interrupt_reopen n =
  let at = Printf.sprintf ", interrupted from poll point %d" n in
  in_child at (fun () ->
      let d, p =
        P.open_ ~maps_host:false ~waits_on:[ `Host ] (fresh "reopen")
      in
      let first, submit = producing d in
      Rig.Point.wait (submit ());
      let second, submit = producing ~last:(fst first) d in
      let point = submit () in
      let third, submit = producing ~last:(fst second) d in
      Rig.Point.wait point;
      let call () =
        Rig.Point.wait (submit ());
        ignore (Sys.opaque_identity (B.create d 0))
      in
      Gc.full_major ();
      let mask = blocked () in
      let reached = polling n call in
      equal ~msg:"the thread's mask" (list int) mask (blocked ());
      Rig.close d;
      Rig.close (fst third);
      settle d;
      List.iter
        (fun (what, driver) ->
          equal ~msg:("what holds " ^ what) (list int) [ 0; 0; 0; 0; 0 ]
            (held driver))
        [
          ("its driver", p);
          ("a producer", snd first);
          ("a producer", snd second);
          ("a producer", snd third);
        ];
      reached)

let reopens () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  S.needs_fork ();
  let rec from n = if interrupt_reopen n then from (n + 1) in
  from 1

(* Runs a borrow of a host buffer on a device whose driver never maps it,
   interrupted from its [n]-th poll point. rig records the mapping in the
   buffer's memory and on the device before the driver's call, and takes
   both back once the driver refused. Then the thread's mask is as before,
   and once the buffer is freed the device and the host hold what they held
   before it: the records kept no memory. Is whether the run reached that
   poll point. *)
let interrupt_refused n =
  let at = Printf.sprintf ", interrupted from poll point %d" n in
  in_child at (fun () ->
      let d, p = P.open_ (fresh "refused") in
      P.never_map p true;
      is_none ~msg:"a refused borrow" (B.borrow d (B.create Rig.host page));
      let before = census d p in
      let h = B.create Rig.host page in
      let call () = ignore (B.borrow d h : B.t option) in
      Gc.full_major ();
      let mask = blocked () in
      let reached = polling n call in
      equal ~msg:"the thread's mask" (list int) mask (blocked ());
      equal ~msg:"the buffer frees" bool true (B.free h);
      equal ~msg:"what the device and the host hold"
        (triple int int (list int))
        before (census d p);
      Rig.close d;
      reached)

let refusals () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  S.needs_fork ();
  let rec from n = if interrupt_refused n then from (n + 1) in
  from 1

(* Runs a submit on a device whose driver never maps a producer's word,
   waiting on a new producer's value, interrupted from its [n]-th poll
   point. rig records the mapping among the device's releases, with one of
   the producer's word unmaps, before the driver's call, and drops both once
   the driver refused: the submit waits on the host. Then the thread's mask
   is as before, and once the device and the producer closed and what they
   held settled, the producer's word went back once: no unmap of it
   outlived them. Is whether the run reached that poll point. *)
let interrupt_unmapped n =
  let at = Printf.sprintf ", interrupted from poll point %d" n in
  in_child at (fun () ->
      let d, p =
        P.open_ ~maps_host:false ~waits_on:[ `Host ] (fresh "unmapped")
      in
      P.never_map p true;
      let producer, pp = P.open_ ~transport:true (fresh "producer") in
      let v = once (Sub.take (Sub.make producer [||])) in
      let run = Sub.take (Sub.make d [||]) in
      let call () =
        Rig.Point.wait (Rig.submit run ~buffers:[||] ~waits:[| v |])
      in
      Gc.full_major ();
      let mask = blocked () in
      let reached = polling n call in
      equal ~msg:"the thread's mask" (list int) mask (blocked ());
      Rig.close d;
      Rig.close producer;
      settle d;
      equal ~msg:"the producer's word frees" int 1 (P.word_frees pp);
      List.iter
        (fun (what, driver) ->
          equal ~msg:("what holds " ^ what) (list int) [ 0; 0; 0; 0; 0 ]
            (held driver))
        [ ("its driver", p); ("the producer", pp) ];
      reached)

let unmapped_words () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  S.needs_fork ();
  let rec from n = if interrupt_unmapped n then from (n + 1) in
  from 1

let stops how () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  S.needs_fork ();
  let rec from n = if interrupt_stop how n then from (n + 1) in
  from 1

(* The poll walks from each of [starts]. Bytecode walks from devices that
   ran the operation alone, whose census shows what the call leaks: a walk
   there polls at every application, and from a fresh device it would walk
   each turn of the loops that make a device's tables. A walk from a [Used]
   device and one from a [Settled] device reach different steps: the first
   finds the earlier run's memory on its way to the cache, the second finds
   none and makes its memory anew. *)
let poll_walks starts =
  let name op = function
    | Fresh -> op.name
    | Used -> op.name ^ ", on a device it ran on"
    | Settled -> op.name ^ ", on a device it ran on, settled"
  in
  group ~timeout "a SIGINT at any poll point leaves rig consistent"
    (List.concat_map
       (fun (call, op) ->
         List.map (fun s -> test (name op s) (polls ~call s op)) starts)
       (List.map (fun op -> (call, op)) (ops @ (wait_for :: freeings))
       @ [ (two_call, submit_two) ]
       @ List.map (fun op -> (staged_call, op)) staged
       @ List.map (fun op -> (host_call, op)) host_moves)
    @ [
        test "a submit waiting on a producer in a reopened slot" reopens;
        test "a borrow whose mapping its driver refuses" refusals;
        test "a submit whose mapping of a producer's word its driver refuses"
          unmapped_words;
        test "close of a lost device that holds memory" (stops `Close);
        test "open of a lost device's name" (stops `Open);
      ])

let () =
  if Sys.backend_type <> Native then
    exit @@ run "rig.interrupt" [ poll_walks [ Used; Settled ] ];
  exit
  @@ run "rig.interrupt"
       [
         poll_walks [ Fresh; Used; Settled ];
         group ~timeout "an exception at any allocation leaves rig consistent"
           (List.concat_map
              (fun (walk, op) ->
                [
                  test op.name (interrupts ~walk Fresh op);
                  test (op.name ^ ", on a device it ran on")
                    (interrupts ~walk Used op);
                ])
              ([ (Each_stack, opens); (Each_stack, caches) ]
              @ List.map
                  (fun op -> (Each_allocation, op))
                  ((borrow_kept :: waiting :: kept_run :: ops)
                  @ staged @ List.map unmapped staged))
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
               test "a callback's lose" test_lose_in;
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
